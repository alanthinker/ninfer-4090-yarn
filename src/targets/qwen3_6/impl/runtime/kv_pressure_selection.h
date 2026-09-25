#pragma once

// Device-to-Host KV pressure selection: which pages of one owner may be moved (or dropped) to
// relieve a capacity deficit, and what that costs. Kept out of program_impl.h so the rule can be
// exercised directly with synthetic store data (tests/test_spill_selection.cpp) instead of only
// through a loaded model.
//
// The includer defines NINFER_QWEN36_RUNTIME_NS (same rule as instantiate.h).

#include "core/host_kv_arena.h"
#include "targets/qwen3_6/impl/runtime/host_kv_extent_store.h"
#include "targets/qwen3_6/impl/runtime/logical_kv_store.h"
#include "targets/qwen3_6/impl/runtime/program.h"

#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS {

inline runtime::ContextTransferRequirement
kv_transfer_requirement(runtime::ContextResourceClass resource,
                        runtime::ContextTransferDirection direction, const HostKVPageLayout& layout,
                        std::uint32_t pages, std::uint32_t contiguous_runs = 1) {
    const TransferWork work = direction == runtime::ContextTransferDirection::DeviceToDevice
                                  ? plan_device_kv_copy_work(layout, pages)
                                  : plan_host_kv_transfer_work(layout, pages, contiguous_runs);
    return runtime::ContextTransferRequirement{
        .resource   = resource,
        .direction  = direction,
        .units      = work.payload_bytes,
        .page_count = pages,
        .work       = work,
    };
}

inline std::uint32_t physical_kv_runs(const KVAddressSpaceStore& addresses,
                               const LogicalKVPageStore& pages, KVAddressSpaceHandle address,
                               std::uint32_t begin, std::uint32_t count) {
    if (count == 0) { return 0; }
    if (begin > addresses.mapped_pages(address) ||
        count > addresses.mapped_pages(address) - begin) {
        throw std::logic_error("physical KV run range is outside its address space");
    }
    std::vector<DeviceKVPageHandle> physical;
    physical.reserve(count);
    for (std::uint32_t offset = 0; offset < count; ++offset) {
        physical.push_back(pages.physical(addresses.logical_page(address, begin + offset)));
    }
    return pages.physical_pool().contiguous_run_count(physical);
}

void append_pressure_transfer(qwen3_6::detail::PressureDecision& option,
                              runtime::ContextTransferRequirement requirement) {
    if (requirement.units != 0) { option.transfer_requirements.push_back(std::move(requirement)); }
}

struct KVPressureSelection {
    std::vector<qwen3_6::detail::PressureKVDecision> actions;
    std::vector<runtime::ContextTransferRequirement> transfer_requirements;
    std::uint32_t removed_device_pages = 0;
    std::size_t removed_host_bytes     = 0;
    std::size_t added_host_bytes       = 0;
    std::size_t host_bytes_remaining   = 0;
};

inline bool logical_page_matches_prefix(const KVAddressSpaceStore& addresses,
                                 std::optional<KVAddressSpaceHandle> prefix,
                                 std::uint32_t prefix_pages, std::uint32_t page_offset,
                                 LogicalKVPageHandle page) {
    if (!prefix || page_offset >= prefix_pages || !addresses.valid(*prefix) ||
        prefix_pages > addresses.mapped_pages(*prefix)) {
        return false;
    }
    return addresses.logical_page(*prefix, page_offset) == page;
}

template <class ProtectedPage>
KVPressureSelection
select_kv_pressure_actions(const KVAddressSpaceStore& addresses, LogicalKVPageStore& pages,
                           HostKVExtentStore* host_extents, bool host_allocation_available,
                           KVAddressSpaceHandle address, std::optional<std::uint32_t> mapped_limit,
                           std::uint32_t requested_device_pages, std::size_t requested_host_bytes,
                           runtime::ContextResourceClass resource,
                           std::span<const qwen3_6::detail::PressureKVDecision> existing_actions,
                           ProtectedPage&& protected_page) {
    KVPressureSelection selection;
    selection.host_bytes_remaining = requested_host_bytes;
    const std::uint32_t mapped     = std::min(addresses.mapped_pages(address),
                                              mapped_limit.value_or(addresses.mapped_pages(address)));
    std::vector<std::uint8_t> selected(mapped, 0);
    const HostKVPageLayout layout = plan_host_kv_page_layout(pages.physical_pool().geometry());

    for (const qwen3_6::detail::PressureKVDecision& action : existing_actions) {
        if (action.kind == qwen3_6::detail::PressureKVDecisionKind::None ||
            action.page_count == 0 || action.begin_page > mapped ||
            action.page_count > mapped - action.begin_page) {
            throw std::logic_error("existing pressure KV action is outside the retained target");
        }
        for (std::uint32_t offset = 0; offset < action.page_count; ++offset) {
            const std::uint32_t page = action.begin_page + offset;
            if (selected[page] != 0) {
                throw std::logic_error("existing pressure KV actions overlap");
            }
            selected[page] = 1;
        }
    }

    const auto mark_action = [&](qwen3_6::detail::PressureKVDecision action) {
        for (std::uint32_t offset = 0; offset < action.page_count; ++offset) {
            const std::uint32_t page = action.begin_page + offset;
            if (page >= selected.size() || selected[page] != 0) {
                throw std::logic_error("pressure KV actions overlap");
            }
            selected[page] = 1;
        }
        selection.actions.push_back(action);
    };

    while (selection.host_bytes_remaining != 0 && host_extents != nullptr) {
        const std::size_t requested_pages =
            1U + (selection.host_bytes_remaining - 1U) / layout.page_stride;
        bool found        = false;
        std::uint32_t end = mapped;
        while (end != 0 && !found) {
            const auto eligible = [&](std::uint32_t page) {
                const LogicalKVPageHandle logical = addresses.logical_page(address, page);
                return selected[page] == 0 && pages.device_resident(logical) &&
                       pages.host_resident(logical) && pages.writer_references(logical) == 0 &&
                       pages.source_pins(logical) == 0 &&
                       !protected_page(page, logical,
                                       qwen3_6::detail::PressureKVDecisionKind::DropHostDuplicate);
            };
            while (end != 0 && !eligible(end - 1U)) { --end; }
            if (end == 0) { break; }
            std::uint32_t begin = end - 1U;
            while (begin != 0 && eligible(begin - 1U)) { --begin; }
            const std::uint32_t count =
                static_cast<std::uint32_t>(std::min<std::size_t>(end - begin, requested_pages));
            const std::uint32_t selected_begin = end - count;
            std::vector<LogicalKVPageHandle> releases;
            releases.reserve(count);
            for (std::uint32_t offset = 0; offset < count; ++offset) {
                releases.push_back(addresses.logical_page(address, selected_begin + offset));
            }
            if (!host_extents->can_release_page_replicas(pages, releases)) {
                end = begin;
                continue;
            }
            mark_action({
                .begin_page = selected_begin,
                .page_count = count,
                .kind       = qwen3_6::detail::PressureKVDecisionKind::DropHostDuplicate,
            });
            const std::size_t bytes = layout.page_stride * static_cast<std::size_t>(count);
            if (bytes > std::numeric_limits<std::size_t>::max() - selection.removed_host_bytes) {
                throw std::overflow_error("pressure Host KV release size overflow");
            }
            selection.removed_host_bytes += bytes;
            selection.host_bytes_remaining = bytes >= selection.host_bytes_remaining
                                                 ? 0
                                                 : selection.host_bytes_remaining - bytes;
            found                          = true;
        }
        if (!found) { break; }
    }

    std::uint32_t device_remaining = requested_device_pages;
    const auto select_device_runs  = [&](bool require_host) {
        std::uint32_t end = mapped;
        while (device_remaining != 0 && end != 0) {
            const auto eligible = [&](std::uint32_t page) {
                if (selected[page] != 0) { return false; }
                const LogicalKVPageHandle logical = addresses.logical_page(address, page);
                const bool replica_safe = require_host ? pages.can_drop_device_replica(logical)
                                                        : !pages.host_resident(logical);
                const qwen3_6::detail::PressureKVDecisionKind action =
                    require_host ? qwen3_6::detail::PressureKVDecisionKind::DropDeviceDuplicate
                                  : qwen3_6::detail::PressureKVDecisionKind::DemoteToHost;
                return pages.device_resident(logical) && pages.writer_references(logical) == 0 &&
                       pages.source_pins(logical) == 0 && replica_safe &&
                       !addresses.has_active_reference(logical) &&
                       !protected_page(page, logical, action);
            };
            while (end != 0 && !eligible(end - 1U)) { --end; }
            if (end == 0) { break; }
            std::uint32_t begin = end - 1U;
            while (begin != 0 && eligible(begin - 1U)) { --begin; }
            const std::uint32_t count = std::min(device_remaining, end - begin);
            qwen3_6::detail::PressureKVDecision action{
                 .begin_page = end - count,
                 .page_count = count,
                 .kind = require_host ? qwen3_6::detail::PressureKVDecisionKind::DropDeviceDuplicate
                                      : qwen3_6::detail::PressureKVDecisionKind::DemoteToHost,
            };
            mark_action(action);
            selection.removed_device_pages += count;
            device_remaining -= count;
            end = action.begin_page;

            if (!require_host) {
                if (count != 0 &&
                    layout.page_stride > std::numeric_limits<std::size_t>::max() / count) {
                    throw std::overflow_error("pressure Host KV extent size overflow");
                }
                const std::size_t bytes = layout.page_stride * static_cast<std::size_t>(count);
                if (bytes > std::numeric_limits<std::size_t>::max() - selection.added_host_bytes) {
                    throw std::overflow_error("pressure KV transfer size overflow");
                }
                selection.added_host_bytes += bytes;
                selection.transfer_requirements.push_back(kv_transfer_requirement(
                    resource, runtime::ContextTransferDirection::DeviceToHost, layout, count,
                    physical_kv_runs(addresses, pages, address, action.begin_page,
                                      action.page_count)));
            }
        }
    };
    select_device_runs(true);
    if (device_remaining != 0 && selection.host_bytes_remaining == 0 && host_allocation_available &&
        host_extents != nullptr) {
        select_device_runs(false);
    }
    if (device_remaining != 0) {
        // Name the blocker when a spill plan comes up short. Production had1812 such plans and
        // none of them could say why (2026-09-25: `status=1` while Host KV had25.6 GiB free), so
        // the request fell back to deleting whole sessions. The tallies separate the policy
        // reasons (protected / active / pinned / already-on-Host) from "the pages were eligible
        // and the selection still did not take them".
        const char* diag = std::getenv("NINFER_REUSE_DIAG");
        if (diag == nullptr || *diag != '0') {
            std::uint32_t taken = 0, not_device = 0, writers = 0, pins = 0, active = 0;
            std::uint32_t protected_pages = 0, host_resident = 0, demote_ok = 0;
            for (std::uint32_t page = 0; page < mapped; ++page) {
                if (selected[page] != 0) { ++taken; continue; }
                const LogicalKVPageHandle logical = addresses.logical_page(address, page);
                if (!pages.device_resident(logical)) { ++not_device; continue; }
                if (pages.writer_references(logical) != 0) { ++writers; continue; }
                if (pages.source_pins(logical) != 0) { ++pins; continue; }
                if (protected_page(page, logical,
                                   qwen3_6::detail::PressureKVDecisionKind::DemoteToHost)) {
                    ++protected_pages;
                    continue;
                }
                if (addresses.has_active_reference(logical)) { ++active; continue; }
                if (pages.host_resident(logical)) { ++host_resident; continue; }
                ++demote_ok;
            }
            std::fprintf(stderr,
                         "[spill] short by %u page(s): mapped=%u taken=%u not_device=%u"
                         " writers=%u pins=%u active=%u protected=%u host_resident=%u"
                         " demote_ok=%u\n",
                         device_remaining, mapped, taken, not_device, writers, pins, active,
                         protected_pages, host_resident, demote_ok);
            std::fflush(stderr);
        }
    }
    return selection;
}

}  // namespace ninfer::targets::qwen3_6::detail::NINFER_QWEN36_RUNTIME_NS
