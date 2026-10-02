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

// §一: "只要还有一条已经处理完的对话占着缓存，就一定能删". The step that deletes one idle owner so
// its Host side can be reused (`release_idle_owner_host_side`) reaches the strict release only once
// the owner is Device-free, because §四 invariant 1 forbids destroying Device data in place. Two
// retreats in that walk let an idle pool answer "nothing to release" while hundreds of finished
// conversations still held it:
//
//   * `device.state_slots != 0` declined the WHOLE owner ("relocating state is the demote step's
//     job"). The demote step answers a STATE-SLOT deficit and is itself gated on
//     `host.state_slots != 0`, so an owner whose Device KV was already spilled to Host - the normal
//     end state under a full Host KV pool - is refused by both walks and can never be deleted.
//     Measured 2026-10-02 production req#427: `[ladder] degrade declined ... nohoststate=4
//     protected=306 ... noplace=94` plus `[cache] pressure step refused: victim slot=55 cannot be
//     moved to Host (R1) ... (R0)`, then the request was refused outright in 415 ms. slot=55's
//     Device KV had ALREADY moved (`spill continuation slot=55 kv pages=42`); only its state slot
//     was left, and that alone made 319 finished conversations undeletable.
//
// The size of what deletion destroys is what the rule actually turns on, so it is stated as one:
// deleting the owner destroys its Host-resident cache (the bytes the deficit wants) plus whatever
// Device residue is left after the spill. The residue is legitimate to destroy only when it cannot
// serve a reuse on its own - §三 R3 restores a hit from KV, so state with no KV beside it is not a
// usable cache entry.
struct IdleOwnerReleaseShape {
    // Device KV pages this owner still holds; the walk spills these first (#12 先搬后释).
    std::uint32_t device_kv_pages = 0;
    // Device state slots this owner still holds AFTER that spill would run.
    std::uint32_t device_state_slots = 0;
    // Host KV bytes the owner holds, i.e. what deleting it returns to the pool.
    std::size_t host_kv_bytes = 0;
};

// Whether deleting this idle owner is a legal §一 release, and how much Host KV it returns.
// `require_host_state_slot` is invariant 3: a state-slot deficit is not repaid by deleting an owner
// with no Host state slot to give back.
[[nodiscard]] inline bool idle_owner_is_releasable(const IdleOwnerReleaseShape& shape,
                                                   bool require_host_state_slot,
                                                   std::uint32_t host_state_slots) noexcept {
    if (require_host_state_slot && host_state_slots == 0) { return false; }
    // The spill this walk runs first can clear Device KV, so pages alone never block the release.
    // A Device state slot is the one piece the spill does not take - and it is exactly the piece
    // that must already be unusable for the delete to be legal. It is usable only while KV for the
    // same conversation is still resident somewhere this owner can restore from; with nothing on
    // Device and the Host side being deleted in the same step, it is residue.
    return shape.device_state_slots == 0 || shape.device_kv_pages == 0;
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

// The (total, preference) pair `cachep::Demand` takes, computed in ONE place so the two call sites
// that build it cannot diverge - they did, and that divergence was the bug.
//
// `total` is what `Demand::host_kv` gets: the request's own Host peak, plus the landing its spill
// will place, plus ③'s water line. `preference` is what `Demand::host_kv_preference` gets, and it
// is EXACTLY the water line: `cachep::plan` derives the gap as `total - preference`, so whatever is
// declared here stops being a requirement.
//
// INVARIANT: `total - preference == base_bytes + landing_bytes` (saturating). The water line is a
// PREFERENCE that "must not be able to fail the plan" (see `Demand::host_kv_preference`), so the
// bytes above this line are the request's own peak plus its landing - never the headroom. Clamping
// the preference against `base_bytes` instead of against `total` let the headroom leak into the
// requirement whenever `headroom_bytes > base_bytes` (the normal case for a reuse request, whose own
// Host peak is small): production 2026-10-01 req#523, on a Host KV pool at 31.9/32 GiB, turned an
// 111-page Device gap (main) plus a 107-page gap (backend) into a ~2 GiB Host requirement, the
// release loop walked 172 owners and destroyed ~10 GiB of Host cache, the Device gap was still
// open, the plan failed its own simulation and was discarded whole, and the request dropped its
// offered 73,230-token reuse and recomputed from root (`cache 0 (0.0%)`, TTFT 69 s).
struct HostKvDemand {
    std::size_t total      = 0;
    std::size_t preference = 0;
};

// Is the Host pool's FREE SPACE room a landing can use?
//
// §三 R1 keeps the requirement in bytes ("只要空闲字节够,搬动就必须成功") and the landing is split
// into runs, so fragmentation costs segments and never feasibility. The single case where bytes
// stop being room is free space with no run large enough for ONE page: the arena allocates run by
// run, so such bytes are refused by every landing while `free_bytes()` still reports them.
//
// Reported as 0 that space stops hiding the gap: the policy's release loop then has a real Host
// gap to close, so R2 deletes the least-important cached conversation (§一/§三 R2 - on an idle
// engine everything but the working set is deletable) instead of the plan dying as `blocked_host`
// and the request being answered with nothing. Measured 2026-10-01 production req#411-#416:
// `[host-alloc] arena refused ... free=4214784 largest_run=0 pages first_request=1 pages`.
[[nodiscard]] inline std::size_t placeable_host_kv_bytes(std::size_t free_bytes,
                                                         std::uint32_t largest_free_run_pages) noexcept {
    return largest_free_run_pages == 0 ? 0 : free_bytes;
}

[[nodiscard]] inline HostKvDemand
host_kv_demand_with_water_line(std::size_t base_bytes, std::size_t landing_bytes,
                               std::size_t headroom_bytes) noexcept {
    const std::size_t total = host_kv_demand(base_bytes, landing_bytes, headroom_bytes);
    return HostKvDemand{
        .total      = total,
        .preference = headroom_bytes < total ? headroom_bytes : total,
    };
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
// "Memory has no room to receive it" is a state to LEAVE, not a reason to give up.
//
// 缓存模块v2.md §三 R1: "内存没有空间接收时，先执行 R2 腾地方，再执行 R1", and §一 guarantees the
// release keeps working until it succeeds: everything in a pool except the requests being processed
// is deletable, so "deletable but not enough yet" is not a terminal state - only "nothing left to
// release" is. This runs the two halves in that order, repeatedly:
//
//   while the landing fails: release ONE unit (R2) and retry the landing (R1).
//
// Both retreats this replaces were real and both left a residue for a guard to refuse later:
//   * testing "the pool is exactly full" instead of "the landing failed" missed a pool with one
//     free slot and a two-slot need, so the release never ran at all;
//   * releasing exactly once gave up on a need wider than one unit.
// `release` must report whether it freed anything; a call that frees nothing ends the walk, which
// is what keeps the loop bounded by PROGRESS rather than by an arbitrary retry count (§三 R0: no
// "试了 N 次就报错"). Extracted so the rule is decidable without a Program or an arena.
template <class Land, class Release>
[[nodiscard]] inline bool land_releasing_until_it_fits(Land&& land, Release&& release) {
    if (land()) { return true; }
    while (true) {
        if (!release()) { return false; }   // nothing left to free: the caller's honest R0 answer
        if (land()) { return true; }
    }
}

// 缓存模块v2.md §三 R1's second sentence, as a rule: "内存没有空间接收时，先执行 R2 腾地方，
// 再执行 R1". A demote that cannot LAND is not a refused demote - it is a Host pool that has to be
// made to fit first. This runs the two halves in that order for ONE attempted move:
//
//   1. `land()`  - try the move (R1). If it lands, done.
//   2. `release()` - free ONE unit of the least-important cached data (R2) and retry `land()`.
//
// The retry is the whole point. Before this rule existed in both halves of the ladder, a failed
// landing returned false outright, which read as "this owner cannot move" - so the reserve walk
// concluded nothing was releasable and the request parked on a verdict that could never change
// (the engine was idle, so no pool counter moved). Measured 2026-09-30 production: req#629/#650/
// #652/#654 each waited exactly 4m59.9s at `running 0 | waiting 1` and died at HTTP 499 while an
// identical request was served 2.9 s later at 99.2% reuse.
//
// Bounded to ONE retry per call, matching the ladder: the caller (`reserve_with_release_attempts`)
// is itself the loop, so one R2 per attempted move is what keeps each call's cost constant and the
// progress accounting honest. Extracted so the rule is decidable without a Program, a CUDA context,
// or a Device arena.
template <class Land, class Release>
[[nodiscard]] inline bool land_after_releasing_for_room(Land&& land, Release&& release) {
    if (land()) { return true; }
    if (!release()) { return false; }   // nothing left to free: the caller's honest R0 answer
    return land();                      // the freed room is what the landing was missing
}

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
