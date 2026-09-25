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
bool fill_space(Fixture& fix, std::uint32_t target) {
    auto handle = fix.addresses->create_active(target, 0);
    if (!handle) {
        std::cout << "FAIL: create_active\n";
        return false;
    }
    fix.space = handle;
    // The token->page mapping is not a whole multiple, so a naive doubling jumps past the
    // entitlement. Grow, stop on the first overshoot, and use whatever page count we actually
    // got as the basis for the cases below.
    std::uint32_t tokens  = 1;
    std::uint32_t last    = 0;
    for (int attempt = 0; attempt < 40; ++attempt) {
        if (fix.addresses->mapped_pages(*handle) >= target) { break; }
        try {
            fix.addresses->materialize_to_tokens(*handle, tokens);
        } catch (const std::invalid_argument&) {
            break;  // this token count maps past the entitlement
        } catch (const std::exception& error) {
            std::cout << "FAIL: materialize_to_tokens: " << error.what() << "\n";
            return false;
        }
        const std::uint32_t now = fix.addresses->mapped_pages(*handle);
        if (now == last) { tokens *= 2; }
        last = now;
    }
    if (fix.addresses->mapped_pages(*handle) == 0) {
        std::cout << "FAIL: no page could be materialized\n";
        return false;
    }
    fix.mapped = fix.addresses->mapped_pages(*handle);
    fix.addresses->deactivate(*handle);
    for (std::uint32_t index = 0; index < fix.mapped; ++index) {
        // Freshly materialized pages carry a writer mark; a retained page does not.
        fix.pages->set_writer(fix.addresses->logical_page(*handle, index), false);
    }
    return true;
}

struct Selection {
    std::uint32_t moved = 0;
};

// Asks the real rule to move `want` Device pages to Host.
Selection spill(Fixture& fix, std::uint32_t want,
                const std::vector<std::uint32_t>& protected_offsets) {
    const auto protected_page = [&](std::uint32_t offset, detail::LogicalKVPageHandle,
                                    detail::PressureKVDecisionKind) {
        for (const std::uint32_t entry : protected_offsets) {
            if (entry == offset) { return true; }
        }
        return false;
    };
    const runtime_ns::KVPressureSelection selection = runtime_ns::select_kv_pressure_actions(
        *fix.addresses, *fix.pages, fix.extents, true, *fix.space, std::nullopt, want,
        /*requested_host_bytes=*/0, ninfer::runtime::ContextResourceClass::MainKV,
        std::span<const detail::PressureKVDecision>{}, protected_page);
    return Selection{.moved = selection.removed_device_pages};
}

}  // namespace

int main(int argc, char** argv) {
    if (!cuda_usable()) {
        std::cout << "SKIP: no usable GPU\n";
        return 77;
    }

    constexpr std::uint32_t kPages = 12;

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
