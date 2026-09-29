// Synthetic-data unit test for the Device-to-Host KV pressure selection rule.
//
// Why this exists: production asked for 2656 Device-KV pages while Host KV had25.6 GiB free and
// the planner still fell back to deleting whole sessions (2026-09-25). The rule that decides
// "which pages may be moved" lived inside program_impl.h, so it could only be reached by loading
// a model (~20 s) or by starting a service (~20 min). It is now in kv_pressure_selection.h and is
// driven here from fake page states: no artifact, no service, milliseconds per case.
//
// Each case states one blocker and asserts how many pages it costs. The engine's own line
//   [spill] short by N page(s): mapped=... taken=... writers=... pins=... active=... protected=...
// is printed for every shortfall, so a failure names itself.
//
// Exit77 = skipped (no usable GPU), matching the other hardware-dependent tests.
#include "targets/qwen3_6_27b/impl/variant.h"

#define NINFER_QWEN36_VARIANT    ::ninfer::targets::qwen3_6_27b::detail::Variant
#define NINFER_QWEN36_RUNTIME_NS qwen3_6_27b_runtime
#include "targets/qwen3_6/impl/runtime/kv_pressure_selection.h"

#include "core/device.h"
#include "core/host_kv_arena.h"
#include "core/paged_kv_cache.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <iostream>
#include <optional>
#include <span>
#include <string>
#include <vector>

namespace {

namespace detail = ninfer::targets::qwen3_6::detail;
namespace runtime_ns = detail::qwen3_6_27b_runtime;
namespace core       = ninfer;

constexpr std::uint32_t kPageSize = static_cast<std::uint32_t>(core::kPagedKVPageSize);

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok) {
        ++failures;
        std::cout << "FAIL: " << what << "\n";
    }
}

// Small fake pools: a handful of Device pages, one address space, a Host arena. Nothing here
// knows about a model.
struct Fixture {
    std::optional<core::DeviceArena> device;  // constructed in main: DeviceArena(0) throws
    core::DeviceKVPagePool* pool        = nullptr;
    core::KVExecutionTablePool* tables  = nullptr;
    detail::LogicalKVPageStore* pages   = nullptr;
    detail::KVAddressSpaceStore* addresses = nullptr;
    core::HostKVArena* host             = nullptr;
    detail::HostKVExtentStore* extents  = nullptr;
    std::optional<detail::KVAddressSpaceHandle> space;

    std::uint32_t mapped = 0;

    ~Fixture() {
        delete extents;
        delete host;
        delete addresses;
        delete pages;
        delete tables;
        delete pool;
    }
};

bool cuda_usable() {
    const cudaError_t err = cudaGetDeviceCount(nullptr);
    return !(err == cudaErrorNoDevice || err == cudaErrorInsufficientDriver);
}

// Fills `target` Device pages into one address space and then freezes it: a retained (idle)
// session is exactly an address space that is no longer active, which is what makes its pages
// spillable at all - an active one carries active references on every page.
// Fills a NEW retained owner into `fix` with as many pages as `target` allows, and returns how
// many it got (0 on failure). The owner is left Catalogued - inactive, committed, writer-clear -
// which is exactly the state a cold conversation the ladder may spill is in.
std::uint32_t fill_owner(Fixture& fix, std::uint32_t target, std::uint32_t row,
                         std::optional<detail::KVAddressSpaceHandle>* out = nullptr) {
    auto handle = fix.addresses->create_active(target, static_cast<std::int32_t>(row));
    if (!handle) {
        std::cout << "FAIL: create_active\n";
        return 0;
    }
    // The token->page mapping is not a whole multiple, so a naive doubling jumps past the
    // entitlement. Grow, stop on the first overshoot, and use whatever page count we actually
    // got as the basis for the cases below.
    std::uint32_t tokens = 1;
    std::uint32_t last   = 0;
    for (int attempt = 0; attempt < 40; ++attempt) {
        if (fix.addresses->mapped_pages(*handle) >= target) { break; }
        try {
            fix.addresses->materialize_to_tokens(*handle, tokens);
        } catch (const std::invalid_argument&) {
            break;  // this token count maps past the entitlement
        } catch (const std::exception& error) {
            std::cout << "FAIL: materialize_to_tokens: " << error.what() << "\n";
            return 0;
        }
        const std::uint32_t now = fix.addresses->mapped_pages(*handle);
        if (now == last) { tokens *= 2; }
        last = now;
    }
    const std::uint32_t mapped = fix.addresses->mapped_pages(*handle);
    if (mapped == 0) {
        std::cout << "FAIL: no page could be materialized\n";
        return 0;
    }
    // A real retained session has a COMMITTED prefix, and that is the only thing a later fork can
    // name as its frontier. Committing before deactivation is also what marks the pages' committed
    // columns, so a fork from this space is otherwise complete rather than being refused early.
    fix.addresses->commit_frontier(*handle, mapped * kPageSize);
    fix.addresses->deactivate(*handle);
    for (std::uint32_t index = 0; index < mapped; ++index) {
        // Freshly materialized pages carry a writer mark; a retained page does not.
        fix.pages->set_writer(fix.addresses->logical_page(*handle, index), false);
    }
    if (out != nullptr) { out->emplace(*handle); }
    return mapped;
}

bool fill_space(Fixture& fix, std::uint32_t target) {
    std::optional<detail::KVAddressSpaceHandle> handle;
    const std::uint32_t mapped = fill_owner(fix, target, 0, &handle);
    if (mapped == 0) { return false; }
    fix.space  = *handle;
    fix.mapped = mapped;
    return true;
}

struct Selection {
    std::uint32_t moved = 0;
    // The pages the plan names. `select_kv_pressure_actions` only PLANS; a caller that wants the
    // move to actually happen has to apply it, which is what `spill_and_apply` below does.
    std::vector<detail::LogicalKVPageHandle> demoted;
};

// Asks the real rule to move `want` Device pages of `space` to Host.
//
// `exclude_shared` is the flag `spill_owner_device_kv_to_host` passes down, and it is the whole
// subject of Case1b: the selection rule itself happily plans a move for a jointly-held page,
// because the page-level gates it checks (`writer_references`, `source_pins`, active references)
// say nothing about OTHER referents. Whoever calls the rule has to say whether joint pages are
// this owner's to relocate.
Selection spill(Fixture& fix, std::uint32_t want,
                const std::vector<std::uint32_t>& protected_offsets,
                std::optional<detail::KVAddressSpaceHandle> space = std::nullopt,
                bool exclude_shared = false) {
    const detail::KVAddressSpaceHandle address = space ? *space : *fix.space;
    const auto protected_page = [&](std::uint32_t offset, detail::LogicalKVPageHandle,
                                    detail::PressureKVDecisionKind) {
        for (const std::uint32_t entry : protected_offsets) {
            if (entry == offset) { return true; }
        }
        return false;
    };
    const runtime_ns::KVPressureSelection selection = runtime_ns::select_kv_pressure_actions(
        *fix.addresses, *fix.pages, fix.extents, true, address, std::nullopt, want,
        /*requested_host_bytes=*/0, ninfer::runtime::ContextResourceClass::MainKV,
        std::span<const detail::PressureKVDecision>{}, protected_page);
    Selection out;
    // `removed_device_pages` is the rule's own verdict and counts BOTH kinds of Device relief
    // (`DemoteToHost` and `DropDeviceDuplicate`), which is what the cases assert on. `demoted`
    // lists only the pages that still need a Host landing - what `spill_and_apply` must execute.
    out.moved = selection.removed_device_pages;
    for (const detail::PressureKVDecision& action : selection.actions) {
        if (action.kind != detail::PressureKVDecisionKind::DemoteToHost) { continue; }
        for (std::uint32_t offset = 0; offset < action.page_count; ++offset) {
            const detail::LogicalKVPageHandle page =
                fix.addresses->logical_page(address, action.begin_page + offset);
            // The `kind_of` gate inside `spill_owner_device_kv_to_host`: a jointly-held page is
            // not this owner's to relocate, so with `exclude_shared` the run stops before it.
            if (exclude_shared && fix.pages->address_references(page) > 1) { continue; }
            out.demoted.push_back(page);
        }
    }
    return out;
}

// The same plan, then executed the way R1 does: land the pages on Host (publish their Host
// replica) and only then drop the Device replica. That is what makes them non-Device-resident
// for anyone who consults them afterwards - precisely the state `prepare_prefix_fork` refuses
// as unstable.
Selection spill_and_apply(Fixture& fix, std::uint32_t want,
                          const std::vector<std::uint32_t>& protected_offsets,
                          std::optional<detail::KVAddressSpaceHandle> space = std::nullopt,
                          bool exclude_shared = false) {
    Selection out = spill(fix, want, protected_offsets, space, exclude_shared);
    if (out.demoted.empty()) { return out; }
    std::optional<detail::HostKVExtentReservation> landing = fix.extents->prepare(
        *fix.pages, std::span<const detail::LogicalKVPageHandle>(out.demoted));
    if (!landing) {
        std::cout << "FAIL: the plan's Host landing could not be prepared\n";
        return out;
    }
    (void)fix.extents->publish(std::move(*landing));
    for (const detail::LogicalKVPageHandle page : out.demoted) {
        if (!fix.pages->drop_device_replica(page)) {
            std::cout << "FAIL: the plan named a page that cannot be demoted\n";
            return out;
        }
    }
    return out;
}

// Puts every page of the fixture's source back on Device. `Case1b` deliberately drains it (that is
// the bug), so the cases after it restore the fixture rather than building a second one.
void restore_device_replicas(Fixture& fix, std::uint32_t pages) {
    for (std::uint32_t index = 0; index < pages; ++index) {
        const detail::LogicalKVPageHandle page = fix.addresses->logical_page(*fix.space, index);
        if (fix.pages->device_resident(page)) { continue; }
        auto reservation = fix.pool->make_empty_reservation();
        fix.pages->reserve_device_pages(reservation, 1);
        (void)fix.pages->reserve_device_replica(page, reservation);
        fix.pages->publish_device_replica(page);
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (!cuda_usable()) {
        std::cout << "SKIP: no usable GPU\n";
        return 77;
    }

    constexpr std::uint32_t kPages = 20;

    Fixture fix;
    try {
        core::KVPageGeometry geometry{.planes = {{core::DType::I8, 8, 2, 256}}};
        core::LayoutBuilder builder;
        const core::DeviceKVPagePoolLayout pages_plan = core::plan_device_kv_page_pool(
            builder, {.page_group_count = kPages, .geometry = geometry});
        const core::KVExecutionTableLayout tables_plan = core::plan_kv_execution_tables(
            builder, {.logical_page_capacity = kPages, .table_rows = 4});
        const std::size_t bytes = builder.finish(256);
        fix.device.emplace(bytes);
        fix.pool                  = new core::DeviceKVPagePool(
            {fix.device->base(), fix.device->capacity()}, pages_plan);
        fix.tables = new core::KVExecutionTablePool(
            {fix.device->base(), fix.device->capacity()}, tables_plan, *fix.pool);
        fix.pages    = new detail::LogicalKVPageStore(*fix.pool, kPages);
        fix.addresses = new detail::KVAddressSpaceStore(*fix.pages, *fix.tables,
                                                       /*address_capacity=*/4,
                                                       /*page_capacity=*/kPages);
        const core::HostKVPageLayout host_layout = core::plan_host_kv_page_layout(fix.pool->geometry());
        const core::HostKVPageLayout layouts[]   = {host_layout};
        fix.host     = new core::HostKVArena(
            host_layout.page_stride * 32, std::span<const core::HostKVPageLayout>(layouts, 1));
        fix.extents  = new detail::HostKVExtentStore(*fix.host, /*descriptor_capacity=*/8);
        if (!fill_space(fix, kPages)) { return 1; }
    } catch (const std::exception& error) {
        std::cout << "FAIL: fixture: " << error.what() << "\n";
        return 1;
    }

    std::cout << "fixture ready: mapped=" << fix.mapped << " pages\n";

    // The token->page mapping decides how many pages we actually got; the cases below are all
    // expressed in terms of that, not of the requested target.
    if (fix.mapped < 6) {
        std::cout << "FAIL: need at least 6 pages, got " << fix.mapped << "\n";
        return 1;
    }
    const std::uint32_t have = fix.mapped;

    // Case1: nothing blocks the move - the rule must take everything it was asked for.
    {
        const Selection got = spill(fix, have, {});
        std::cout << "case clean            moved=" << got.moved << "/" << have << "\n";
        expect(got.moved == have, "clean pool should spill every requested page");
    }

    // Case1b (2026-09-29 production fatal): the Device-KV release hook must NOT be allowed to
    // hollow out the pages the transaction that triggered it is about to fork from.
    //
    // The real production order is:
    //   1. `prepare_kv_restores` reserves Device pages for a RETAINED source's missing replicas.
    //      The pool is full, so it calls the owner's release hook (R1).
    //   2. The hook moves a cold owner's Device KV to Host.
    //   3. The same transaction then calls `prepare_prefix_fork` on that RETAINED source, which
    //      requires EVERY page of its prefix to still be Device-resident - and otherwise throws
    //      "KV retained-prefix source is not stable".
    //
    // That throw reaches `fail_all_locked`, so the engine latches `failed_` and answers 503
    // "inference engine is unavailable" until it is restarted (the 18:24 production stop).
    //
    // The release step already excluded the owner that IS the source (`materialization_pins`), so
    // the case that broke was a page of the source sitting in ANOTHER owner's address space: the
    // page-level gates it consults (`writer_references`, `source_pins`, active references) cannot
    // see that a page is jointly held, and `address_references > 1` reports the sharing without
    // naming the other referent. The fix passes the source prefix down and spares exactly the
    // pages it covers.
    //
    // This case pins both halves of that contract: the covered pages must be refused, and a
    // jointly-held page OUTSIDE the prefix must stay relocatable - refusing every shared page
    // instead starved the reserve and traded a rare latch for a routine queue-timeout 503
    // (rig A/B on one load: `ALL HIT` -> conv4 503).
    {
        // The fork covers only a LEADING part of the source, so the case can also show that a
        // jointly-held page past the prefix stays relocatable. A whole-space frontier would make
        // the prefix the entire address space and the second half of the contract untestable.
        const std::uint32_t prefix_pages = have / 2U;
        const std::uint32_t frontier     = prefix_pages * kPageSize;
        expect(frontier != 0 && frontier <= fix.addresses->committed_frontier(*fix.space),
               "case1b setup: the fork frontier is inside the committed source");
        expect(prefix_pages != 0 && prefix_pages < have,
               "case1b setup: the fork covers a strict leading part of the source");
        // The pool only calls the release hook when the ask does NOT fit, so a fixture with spare
        // pages would prove nothing. Occupy the remainder with a second, genuinely cold owner:
        // that is the production shape (a full Device pool, a new retention wanting room).
        expect(fix.pool->available_pages() != 0,
               "case1b setup: the fixture has the spare pages this case has to consume");
        std::optional<detail::KVAddressSpaceHandle> filler;
        const std::uint32_t spare = fill_owner(fix, fix.pool->available_pages(), 1, &filler);
        if (spare == 0 || !filler) {
            std::cout << "FAIL: case1b could not build the filling owner\n";
            return 1;
        }
        expect(fix.pool->available_pages() == 0, "case1b setup: the Device pool is now full");

        // The production shape: jointly-held pages, both INSIDE the fork prefix and outside it.
        // `retain_reference` is what a `complete_prefix_fork` destination does to the pages it
        // shares with its source, so `address_references` above 1 is exactly the real condition.
        const std::uint32_t shared_outside = std::min<std::uint32_t>(have - prefix_pages, 2);
        expect(shared_outside != 0, "case1b setup: the source has pages past the fork prefix");
        for (std::uint32_t index = 0; index < prefix_pages; ++index) {
            fix.pages->retain_reference(fix.addresses->logical_page(*fix.space, index),
                                        /*writer=*/false);
        }
        for (std::uint32_t index = prefix_pages; index < prefix_pages + shared_outside; ++index) {
            fix.pages->retain_reference(fix.addresses->logical_page(*fix.space, index),
                                        /*writer=*/false);
        }

        const runtime_ns::PendingForkPrefixPages covered(
            *fix.addresses, runtime_ns::ProgramPendingForkPrefix(*fix.space, prefix_pages));
        expect(covered.size() == prefix_pages, "case1b setup: the prefix expands to its pages");
        expect(covered.covers(fix.addresses->logical_page(*fix.space, 0)),
               "case1b: a page inside the fork prefix is covered");
        expect(covered.covers(fix.addresses->logical_page(*fix.space, prefix_pages - 1)),
               "case1b: the last page of the fork prefix is covered");
        expect(!covered.covers(fix.addresses->logical_page(*fix.space, prefix_pages)),
               "case1b: a jointly-held page OUTSIDE the prefix is not covered");

        // The release hook the owner installs. In production it is `release_device_kv_capacity_step`,
        // which walks the cold owners in importance order and moves the first movable one - here it
        // is reduced to "the source owner", which is the worst case for the pending fork.
        const std::uint32_t needed = 2;
        bool hook_ran              = false;
        fix.pages->set_device_kv_release([&]() -> bool {
            // Every reservation runs the step, exactly as the store does; the counter only records
            // that it ran at all. Returning true keeps funding later reservations from the same
            // owner, which is what `reserve_with_release_attempts` relies on while it makes progress.
            hook_ran = true;
            // Free enough that BOTH the reservation that asked and the fork that follows it can
            // be served - in production the same owner is walked repeatedly for exactly this reason.
            Selection out = spill(fix, needed + needed, {}, fix.space);
            // Apply only what the pending fork does not cover - the guard the fix installed.
            std::vector<detail::LogicalKVPageHandle> allowed;
            for (const detail::LogicalKVPageHandle page : out.demoted) {
                if (!covered.covers(page)) { allowed.push_back(page); }
            }
            out.demoted = std::move(allowed);
            if (out.demoted.empty()) { return false; }
            std::optional<detail::HostKVExtentReservation> landing = fix.extents->prepare(
                *fix.pages, std::span<const detail::LogicalKVPageHandle>(out.demoted));
            if (!landing) { return false; }
            (void)fix.extents->publish(std::move(*landing));
            for (const detail::LogicalKVPageHandle page : out.demoted) {
                if (!fix.pages->drop_device_replica(page)) { return false; }
            }
            return true;
        });

        auto reservation   = fix.pool->make_empty_reservation();
        bool reserve_threw = false;
        try {
            fix.pages->reserve_device_pages(reservation, needed);
        } catch (const std::exception& error) {
            reserve_threw = true;
            std::cout << "case1b reserve threw  " << error.what() << "\n";
        }
        std::cout << "case1b reserve        hook_ran=" << (hook_ran ? 1 : 0)
                  << " threw=" << (reserve_threw ? 1 : 0) << "\n";
        expect(hook_ran, "case1b setup: a full pool must run the owner's release step");
        expect(!reserve_threw, "case1b setup: the release must actually fund the reservation");

        // The fork the reservation was funding must still find its source Device-resident.
        bool refused_frontier  = false;
        bool refused_stability = false;
        try {
            const auto destination = fix.addresses->create_inactive();
            if (!destination) {
                std::cout << "FAIL: case1b could not create a destination space\n";
                return 1;
            }
            // `entitlement` is the fork destination's page budget. Page-aligned to the prefix it
            // needs no extra Device pages at all, which keeps this case about the GUARD rather
            // than about the fixture's spare capacity.
            (void)fix.addresses->prepare_prefix_fork(*fix.space, *destination, frontier,
                                                     /*entitlement=*/prefix_pages,
                                                     /*execution_row=*/0);
        } catch (const std::exception& error) {
            const std::string what = error.what();
            refused_frontier       = what.find("frontier is unavailable") != std::string::npos;
            refused_stability      = what.find("source is not stable") != std::string::npos;
            std::cout << "case1b fork           " << what << "\n";
        }
        expect(!refused_frontier, "case1b must reach the stability check, not the frontier precheck");
        expect(!refused_stability,
               "case1b: the guarded release left the fork prefix Device-resident, so the fork "
               "proceeds instead of latching the engine with the production fatal");

        // ...and it did so by sparing only the prefix: the JOINT page past it was still movable.
        // The step must have relocated real pages, and none of them may be a fork-prefix page -
        // that pair is the whole point: the guard is NARROW, not a refusal to move anything.
        std::uint32_t moved_outside = 0;
        for (std::uint32_t index = 0; index < have; ++index) {
            const detail::LogicalKVPageHandle page =
                fix.addresses->logical_page(*fix.space, index);
            const bool moved = fix.pages->host_resident(page);
            if (moved && index >= prefix_pages) { ++moved_outside; }
            if (moved && index < prefix_pages) {
                std::cout << "FAIL: case1b moved a page inside the fork prefix: " << index << "\n";
                ++failures;
            }
        }
        std::cout << "case1b guard          moved_outside=" << moved_outside
                  << " prefix=" << prefix_pages << "\n";
        const bool outside_moved = moved_outside != 0;
        expect(outside_moved,
               "case1b: a jointly-held page outside the prefix must stay relocatable, or the "
               "relief starves the very reservation it is funding (the rig A/B 503)");
    }

    // Case1b's fixture damage is repaired here, before the cases below, which all assert against a
    // fully Device-resident source with every page uniquely owned. Restoring is the mirror of the
    // move (reserve a Device replica, then publish it); the joint references Case1b added are
    // released so `moved` counts stay comparable across cases.
    restore_device_replicas(fix, have);
    for (std::uint32_t index = 0; index < have; ++index) {
        const detail::LogicalKVPageHandle page = fix.addresses->logical_page(*fix.space, index);
        if (fix.pages->address_references(page) > 1) {
            (void)fix.pages->release_reference(page, /*writer=*/false);
        }
    }
    expect(fix.pages->address_references(fix.addresses->logical_page(*fix.space, 0)) == 1,
           "case1b teardown: the source is uniquely owned again");
    expect(fix.pool->available_pages() == 0, "case1b teardown: the source is Device-resident again");

    // Case2: three pages are pinned as a copy source (an in-flight transfer owns them).
    for (std::uint32_t index = 0; index < 3; ++index) {
        fix.pages->pin_source(fix.addresses->logical_page(*fix.space, index));
    }
    {
        const Selection got = spill(fix, have, {});
        std::cout << "case pinned(3)        moved=" << got.moved << "/" << have << "\n";
        expect(got.moved == have - 3, "pinned pages must be refused (and counted as pins)");
    }
    for (std::uint32_t index = 0; index < 3; ++index) {
        fix.pages->unpin_source(fix.addresses->logical_page(*fix.space, index));
    }

    // Case3: three pages are actively referenced by a running address space.
    for (std::uint32_t index = 0; index < 3; ++index) {
        fix.pages->retain_active_reference(fix.addresses->logical_page(*fix.space, index));
    }
    {
        const Selection got = spill(fix, have, {});
        std::cout << "case active(3)        moved=" << got.moved << "/" << have << "\n";
        expect(got.moved == have - 3, "active pages must be refused (and counted as active)");
    }
    for (std::uint32_t index = 0; index < 3; ++index) {
        fix.pages->release_active_reference(fix.addresses->logical_page(*fix.space, index));
    }

    // Case4: three pages belong to the source prefix the request must keep (protection).
    {
        const Selection got = spill(fix, have, {0, 1, 2});
        std::cout << "case protected(3)     moved=" << got.moved << "/" << have << "\n";
        expect(got.moved == have - 3, "protected pages must be refused (counted as protected)");
    }

    // Case5: ask for more than exists - the shortfall has to be reported, not silently accepted.
    {
        const Selection got = spill(fix, have + 5, {});
        std::cout << "case over-ask         moved=" << got.moved << "/" << have + 5 << "\n";
        expect(got.moved == have, "over-ask must move only what exists");
    }

    if (failures != 0) {
        std::cout << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "spill selection ok\n";
    return 0;
}
