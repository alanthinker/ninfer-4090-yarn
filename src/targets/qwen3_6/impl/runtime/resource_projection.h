#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>

namespace ninfer::targets::qwen3_6::detail {

// Concrete allocator quantities remain inside the Program boundary. These values describe one
// complete physical state or transition; ResourceManager never observes or reconstructs them.
struct PhysicalDeviceResources {
    std::uint32_t active_lanes     = 0;
    std::uint32_t state_slots      = 0;
    std::uint32_t main_kv_pages    = 0;
    std::uint32_t backend_kv_pages = 0;

    [[nodiscard]] friend constexpr bool operator==(PhysicalDeviceResources,
                                                   PhysicalDeviceResources) noexcept = default;
};

struct PhysicalHostResources {
    std::uint32_t state_slots = 0;
    std::size_t kv_bytes      = 0;

    [[nodiscard]] friend constexpr bool operator==(PhysicalHostResources,
                                                   PhysicalHostResources) noexcept = default;
};

struct PhysicalResources {
    PhysicalDeviceResources device;
    PhysicalHostResources host;

    [[nodiscard]] friend constexpr bool operator==(const PhysicalResources&,
                                                   const PhysicalResources&) noexcept = default;
};

struct PhysicalDemand {
    // Capacity owned by the active request through completion. A read-only State/KV source whose
    // lifetime remains with another surviving owner is part of global physical occupancy, not
    // this entitlement; an exclusive optional checkpoint in the active lineage is counted once
    // through that checkpoint ownership instead of through the primary binding.
    PhysicalResources active_entitlement;
    PhysicalResources reservation_added;
    PhysicalResources reservation_credit;
    PhysicalResources physical_peak_additional;
    PhysicalResources final_removed;
    PhysicalResources final_added;

    [[nodiscard]] friend constexpr bool operator==(const PhysicalDemand&,
                                                   const PhysicalDemand&) noexcept = default;
};

struct PhysicalDelta {
    PhysicalResources removed;
    PhysicalResources added;

    [[nodiscard]] friend constexpr bool operator==(const PhysicalDelta&,
                                                   const PhysicalDelta&) noexcept = default;
};

// Exact result of evaluating one complete target against the joint State/KV reference graph.
// It is deliberately not a sum of owner-local deltas: aliases and last-reference releases are
// settled once by unique physical identity before this value is produced.
struct PressureTargetProjection {
    PhysicalDelta unique_object_delta;
    PhysicalDelta ownership_transfer_delta;
    PhysicalDelta active_entitlement_delta;
    PhysicalResources source_optional_resources_added;
    std::optional<bool> source_state_fork_required;
    std::optional<bool> source_text_prefix_fork_required;
    std::optional<bool> source_backend_prefix_fork_required;

    [[nodiscard]] friend constexpr bool
    operator==(const PressureTargetProjection&, const PressureTargetProjection&) noexcept = default;
};

// Release-ladder relief the Program can actually deliver, per dimension. The State dimensions
// always credited theirs (device-state / host-state, via the demote/drop ladder). The KV
// dimensions now credit theirs too: a near-full KV pool is resolved by EVICTING the
// least-important cold owner's KV - R1 demote to Host, cascading into an R2 drop of a cold Host
// replica when Host is itself full - not by a reserved buffer. Crediting it stops the peak gate
// from reading a full pool as "permanently infeasible", which is what made a ~24-page Device
// shortfall (2026-09-28 req#589: endpoint offered 111,901, main_kv add=515 used=9793 cap=10284)
// abandon the tail restore for a shallow long anchor and re-derive ~14.7K tokens.
// Device KV the release ladder can actually move off the pools, per pool. See
// `ProgramImplCore::movable_device_kv_relief` for why the gate needs this bound rather than the
// pool's total occupancy.
struct KvMovableRelief {
    std::uint32_t main_kv_pages    = 0;
    std::uint32_t backend_kv_pages = 0;
};

struct PeakFitRelief {
    std::uint32_t device_state_slots      = 0;
    std::uint32_t device_main_kv_pages    = 0;
    std::uint32_t device_backend_kv_pages = 0;
    std::uint32_t host_state_slots        = 0;
    std::size_t host_kv_bytes             = 0;
};

[[nodiscard]] inline std::uint32_t relieve_u32(std::uint32_t used, std::uint32_t relief) noexcept {
    return relief > used ? 0 : used - relief;
}

[[nodiscard]] inline std::size_t relieve_size(std::size_t used, std::size_t relief) noexcept {
    return relief > used ? 0 : used - relief;
}

[[nodiscard]] inline bool fits_u32(std::uint32_t used, std::uint32_t added,
                                   std::uint32_t capacity) noexcept {
    return added <= capacity && used <= capacity - added;
}

[[nodiscard]] inline bool fits_size(std::size_t used, std::size_t added,
                                    std::size_t capacity) noexcept {
    return added <= capacity && used <= capacity - added;
}

// Pure, allocator-independent admission decision: does `peak` fit given `occupied` usage and
// `limits`, once the release ladder applies `relief`? Kept free of Program/GPU state so the
// Device/Host KV relief (an eviction capability, not a reserved headroom) is unit-testable in
// isolation. `active_lanes` is the request's own entitlement and is never relieved.
[[nodiscard]] inline bool physical_peak_fits_core(const PhysicalResources& occupied,
                                                  const PhysicalResources& limits,
                                                  const PhysicalResources& peak,
                                                  const PeakFitRelief& relief) noexcept {
    return fits_u32(occupied.device.active_lanes, peak.device.active_lanes,
                    limits.device.active_lanes) &&
           fits_u32(relieve_u32(occupied.device.state_slots, relief.device_state_slots),
                    peak.device.state_slots, limits.device.state_slots) &&
           fits_u32(relieve_u32(occupied.device.main_kv_pages, relief.device_main_kv_pages),
                    peak.device.main_kv_pages, limits.device.main_kv_pages) &&
           fits_u32(relieve_u32(occupied.device.backend_kv_pages, relief.device_backend_kv_pages),
                    peak.device.backend_kv_pages, limits.device.backend_kv_pages) &&
           fits_u32(relieve_u32(occupied.host.state_slots, relief.host_state_slots),
                    peak.host.state_slots, limits.host.state_slots) &&
           fits_size(relieve_size(occupied.host.kv_bytes, relief.host_kv_bytes),
                     peak.host.kv_bytes, limits.host.kv_bytes);
}

// How much Device KV the release ladder can demote into the free Host space, per pool. This is the
// pure landing arithmetic behind the Program's `kv_relief()` (whose store/arena inputs are
// GPU-bound and untestable here): the relief is bounded by the Host landing available right now
// (free Host bytes / the page stride), never a reserved buffer. main and backend are symmetric
// Host-landing pools and a request's peak draws from both, so each claims the landing
// independently (neither starves the other). Extracted so the allocation the gate relies on is
// verifiable without a Program, a CUDA context, or a service.
struct KvLandingRelief {
    std::uint32_t main_kv_pages = 0;
    std::uint32_t backend_kv_pages = 0;
};

[[nodiscard]] inline KvLandingRelief kv_landing_relief(std::size_t host_free_bytes,
                                                       std::uint64_t main_page_bytes,
                                                       std::uint64_t backend_page_bytes,
                                                       std::size_t device_main_used,
                                                       std::size_t device_backend_used) noexcept {
    KvLandingRelief out;
    // Each pool independently claims the whole Host landing, bounded by the Device KV it holds.
    // (Claiming the remainder after main would starve backend in the common case where both axes
    // are short, as in req#671; the mild over-credit of both claiming the landing is safe, since a
    // plan it cannot deliver falls back to re-planning from root rather than failing the request.)
    if (main_page_bytes != 0) {
        out.main_kv_pages = static_cast<std::uint32_t>(
            std::min(host_free_bytes / main_page_bytes, device_main_used));
    }
    if (backend_page_bytes != 0) {
        out.backend_kv_pages = static_cast<std::uint32_t>(
            std::min(host_free_bytes / backend_page_bytes, device_backend_used));
    }
    return out;
}

// ③ host water line, as the ABSOLUTE Host KV demand the policy takes: the request's own Host peak,
// plus everything the demote will land here, plus the headroom the arena must still hold free
// afterwards.
//
// This is the function the planner actually calls for `Demand::host_kv` - it is not a re-statement
// of the policy's arithmetic. The policy subtracts the free bytes itself (`cachep::shortfall`,
// "ABSOLUTE peaks, never residuals": handing it a net figure subtracts free twice), so the water
// line has to be expressed on THIS side, by raising the demand by `headroom_bytes`. With
// headroom_bytes = 0 the demand is the pre-③ one (Host ends up pinned full); with the ③ water line
// the policy evicts up to it, so the next demote always has somewhere to land instead of running
// the pool to 100% and deadlocking a restorable tail.
[[nodiscard]] inline std::size_t host_kv_demand(std::size_t base_bytes, std::size_t landing_bytes,
                                                std::size_t headroom_bytes) noexcept {
    const auto add = [](std::size_t lhs, std::size_t rhs) {
        return rhs > std::numeric_limits<std::size_t>::max() - lhs
                   ? std::numeric_limits<std::size_t>::max()
                   : lhs + rhs;
    };
    return add(add(base_bytes, landing_bytes), headroom_bytes);
}

// How many times a Device-KV reservation is retried after the release step freed nothing.
//
// 缓存模块v2.md §三 R1 at reserve time: `fits` is the pool's own capacity test and `release` is
// the owner's ladder step (move the least-important cold owner's Device KV to Host), which reports
// whether it moved anything. A step that reports success without shrinking the pool would spin
// forever, so progress is required: `held` is the pool's allocated+reserved count, and a call that
// leaves it unchanged ends the walk. `holdings` and `release` are asked in that order per attempt,
// and the release never runs once the reservation already fits.
//
// This is the exact policy `LogicalKVPageStore::reserve_device_pages` runs - the loop was extracted
// so its rule is decidable without a Program, a CUDA context, or a Device arena (the pool itself
// needs one, which is why the store-level path is only covered by the rig).
template <class Fits, class Holdings, class Release>
[[nodiscard]] inline std::uint32_t reserve_with_release_attempts(Fits&& fits, Holdings&& holdings,
                                                                Release&& release) {
    std::uint32_t attempts = 0;
    std::uint32_t held     = holdings();
    while (!fits()) {
        if (!release()) { break; }
        const std::uint32_t now = holdings();
        if (now >= held) { break; }
        held = now;
        ++attempts;
    }
    return attempts;
}

} // namespace ninfer::targets::qwen3_6::detail
