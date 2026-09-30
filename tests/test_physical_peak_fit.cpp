// Unit tests for the physical peak-fit admission gate (physical_peak_fits_core).
//
// The gate decides whether a reuse source's KV can be placed (Device/Host) given current
// occupancy, the capacity limits, and the release-ladder relief. The State dimensions have
// always credited their relief (device-state / host-state). The KV dimensions now credit theirs
// too: a near-full KV pool is resolved by EVICTING the least-important cold owner's KV - R1
// demote to Host, cascading into an R2 drop of a cold Host replica when Host is itself full -
// not by a reserved buffer. So a full pool must resolve by eviction, never read as
// "permanently infeasible".
//
// The 2026-09-28 req#589 / req#671 production symptom this protects: Device main_kv
// used=9793 cap=10284, tail (endpoint) restore add=515 -> 24 pages short. With no KV relief the
// gate rejected the endpoint (the best reuse) and the engine re-derived from a shallow long
// anchor (req#589: 97,917, ~14.7K tokens, TTFT 18.5s) or from root (req#671: cache 0%).
//
// Pure CPU: the core is a free function of (occupied, limits, peak, relief) - no Program, no
// CUDA, no service. The relief VALUES are computed by the Program (GPU-side kv_relief); the
// decision they feed is what is verified here.

#include "runtime/cache/cache_tier_policy.h"
#include "targets/qwen3_6/impl/runtime/resource_projection.h"

#include <cstdint>
#include <cstdio>
#include <limits>

using namespace ninfer::targets::qwen3_6::detail;

int g_failures = 0;

void check(bool condition, const char* what) {
    if (condition) { return; }
    std::printf("  FAIL: %s\n", what);
    ++g_failures;
}

namespace {

// The 2026-09-28 req#589/#671 snapshot: Device main_kv nearly full, Host KV near its 32 GiB cap.
[[nodiscard]] PhysicalResources occupied_589() {
    PhysicalResources o;
    o.device.active_lanes     = 1;
    o.device.main_kv_pages    = 9793; // "used=9793"
    o.device.backend_kv_pages = 0;
    o.device.state_slots      = 0;
    o.host.kv_bytes           = 34'300'000'000ULL; // Host KV near full
    o.host.state_slots        = 0;
    return o;
}

[[nodiscard]] PhysicalResources limits_589() {
    PhysicalResources l;
    l.device.active_lanes     = 4;
    l.device.main_kv_pages    = 10284; // "cap=10284"
    l.device.backend_kv_pages = 10288;
    l.device.state_slots      = 4;
    l.host.kv_bytes           = 34'359'738'368ULL; // 32 GiB host KV cap
    l.host.state_slots        = 320;
    return l;
}

// The tail (endpoint) restore: place 515 additional Device main-KV pages (the log's "add=515").
[[nodiscard]] PhysicalResources peak_tail() {
    PhysicalResources p;
    p.device.active_lanes     = 0;
    p.device.main_kv_pages    = 515;
    p.device.backend_kv_pages = 0;
    p.device.state_slots      = 0;
    p.host.kv_bytes           = 0;
    p.host.state_slots        = 0;
    return p;
}

} // namespace

void case_full_device_pool_infeasible_without_kv_relief() {
    // The bug: with no KV relief the gate sees 9793 + 515 = 10308 > 10284 and rejects the tail,
    // even though the ladder could evict ~24+ cold Device pages to Host to close the gap.
    PeakFitRelief none;
    check(!physical_peak_fits_core(occupied_589(), limits_589(), peak_tail(), none),
          "full Device pool with no KV relief must be infeasible (reproduces req#589)");
}

void case_kv_eviction_relief_makes_tail_restore_feasible() {
    // The fix: credit the Device KV the ladder can evict (demote cold pages to Host). 300
    // evictable pages exceeds the 24-page gap, so 9793 - 300 + 515 = 10008 <= 10284 and the tail
    // restore is admitted - evict the least-important cold owner instead of re-deriving.
    PeakFitRelief relief;
    relief.device_main_kv_pages = 300;
    check(physical_peak_fits_core(occupied_589(), limits_589(), peak_tail(), relief),
          "evictable Device KV relief must make the tail restore feasible");
}

void case_kv_relief_below_the_gap_stays_infeasible() {
    // 10 evictable pages < the 24-page gap: 9793 - 10 + 515 = 10298 > 10284. The gate must not
    // over-credit: relief smaller than the gap cannot close it.
    PeakFitRelief relief;
    relief.device_main_kv_pages = 10;
    check(!physical_peak_fits_core(occupied_589(), limits_589(), peak_tail(), relief),
          "KV relief below the gap must not close it");
}

void case_kv_relief_exactly_closes_the_gap() {
    // 24 evictable pages == the gap: 9793 - 24 + 515 = 10284 <= 10284 (boundary, feasible).
    PeakFitRelief relief;
    relief.device_main_kv_pages = 24;
    check(physical_peak_fits_core(occupied_589(), limits_589(), peak_tail(), relief),
          "KV relief exactly equal to the gap must close it");
}

void case_host_kv_eviction_relief() {
    // Host KV as the binding axis: a restore that must place Host KV the pool cannot hold is
    // closed by an R2 drop of a cold Host replica (relief.host_kv_bytes), not by a reserved
    // buffer. Device main_kv is left free (peak adds none), so only Host is exercised.
    PhysicalResources peak;
    peak.host.kv_bytes = 100'000'000ULL; // 100 MB must be placed on Host
    check(!physical_peak_fits_core(occupied_589(), limits_589(), peak, PeakFitRelief{}),
          "full Host KV with no relief must be infeasible");
    PeakFitRelief relief;
    relief.host_kv_bytes = 100'000'000ULL; // drop 100 MB of cold Host replica (R2)
    check(physical_peak_fits_core(occupied_589(), limits_589(), peak, relief),
          "R2 Host-KV eviction relief must close the Host gap");
}

void case_state_relief_still_credits() {
    // Regression guard: the pre-existing State relief path is unchanged. A Device-state gap the
    // ladder can free (demote) must still be credited, independent of the new KV relief.
    PhysicalResources o, l, p;
    l.device.main_kv_pages    = 100;
    l.device.active_lanes     = 4;
    l.device.backend_kv_pages = 0;
    l.device.state_slots      = 4;
    l.host.kv_bytes           = 1000;
    l.host.state_slots        = 320;
    o.device.state_slots      = 4; // pool full: all 4 device-state slots used
    p.device.state_slots      = 1; // the restore needs one more device-state slot

    check(!physical_peak_fits_core(o, l, p, PeakFitRelief{}),
          "device-state gap with no relief must be infeasible");
    PeakFitRelief relief;
    relief.device_state_slots = 1; // the ladder frees one slot (demote a state replica)
    check(physical_peak_fits_core(o, l, p, relief),
          "state relief must still close the state gap");
}

void case_relief_never_overflows_past_zero() {
    // A relief larger than the occupancy floors that dimension to zero (no underflow): the
    // dimension then trivially fits. Guards the saturated subtraction in relieve_u32/relieve_size.
    PhysicalResources o, l, p;
    o.device.main_kv_pages = 50;
    l.device.main_kv_pages = 100;
    l.device.active_lanes  = 4;
    l.device.backend_kv_pages = 0;
    l.device.state_slots   = 0;
    l.host.kv_bytes        = 0;
    l.host.state_slots     = 0;
    p.device.main_kv_pages = 100; // need to place 100 pages (more than the 50 in use)
    PeakFitRelief relief;
    relief.device_main_kv_pages = 50; // evict everything in use
    check(physical_peak_fits_core(o, l, p, relief),
          "relief that frees the whole occupancy must leave the dimension at zero (no underflow)");
}

// --- end-to-end ① logic (CPU): pool snapshot -> kv_landing_relief -> gate -> tail feasible ---

void case_589_pipeline_relief_makes_tail_feasible() {
    // Runs the whole ① chain on the req#589 pool snapshot: compute the KV relief the ladder can
    // deliver (kv_landing_relief from the pool state) and feed it to the gate (physical_peak_fits_core).
    // Before the fix the gate rejected the tail (re-derive ~14.7K tokens); with the relief it
    // admits it (demote the least-important cold Device KV to the free Host space).
    const auto occupied = occupied_589(); // main_kv used=9793, host_kv near full (but ~1.5GB free)
    const auto limits   = limits_589();   // main_kv cap=10284
    const auto peak     = peak_tail();    // main_kv add=515 (the tail restore)

    check(!physical_peak_fits_core(occupied, limits, peak, PeakFitRelief{}),
          "589: no KV relief -> tail infeasible (the pre-fix gate)");

    // The ladder's relief from the pool state: Host had ~1.5GB free ([cache] free hkv for #589),
    // enough to demote the ~515 tail pages; main is Device-limited (9793 pages in use).
    const std::size_t host_free = 1'505'828'864ULL; // [cache] free hkv for req#589
    const std::uint64_t stride  = 3072;             // representative main-KV page stride
    const auto landing = kv_landing_relief(host_free, stride, stride,
                                           occupied.device.main_kv_pages, 0);
    PeakFitRelief relief;
    relief.device_main_kv_pages = landing.main_kv_pages;
    check(relief.device_main_kv_pages > 0,
          "589: the ladder has Device KV it can demote to the free Host space");
    check(physical_peak_fits_core(occupied, limits, peak, relief),
          "589: KV eviction relief -> the tail restore is feasible (the fix)");
}

// --- the gate's KV credit must be bounded by what the ladder can MOVE (production req#629) ---

void case_629_credit_is_bounded_by_movable_pages() {
    // 2026-09-29 production req#629: a 184,902-token conversation selected a reuse base at
    // frontier 184,897, so the gate priced a 2,890-page Device-KV peak. Host had 15,417 MiB free,
    // so `kv_landing_relief` credited 3,854 pages and the gate admitted the request. But every cold
    // owner's Device pages had already been spilled to Host (36 of them walked as `stage=noruns`),
    // so the ladder could deliver 145 pages from the one owner that still held Device pages - a 20x
    // shortfall. The pool does not change while the engine is idle, so the re-admission parked on
    // its negative memo and the request waited out its full 300 s deadline at `running 0` (HTTP 499
    // from the client). The same conversation was served 0.5 s later in 2.9 s at 99.2% reuse.
    const auto occupied = occupied_589();   // main_kv used=9793 of cap 10284
    const auto limits   = limits_589();
    const std::uint64_t stride = 4096;      // representative page stride
    const std::size_t host_free = 15'417ULL * 1024 * 1024;  // req#629 [cache] free hkv

    // What the OLD gate credited: bounded only by the pool's occupancy (9,793 pages in use).
    const KvLandingRelief landing =
        kv_landing_relief(host_free, stride, stride, occupied.device.main_kv_pages, 0);
    check(landing.main_kv_pages == occupied.device.main_kv_pages,
          "629: the raw landing credit is bounded by TOTAL occupancy, not by what can move");

    // The request's peak. With the raw credit the gate admits; that admission is what stranded it.
    PhysicalResources peak;
    peak.device.main_kv_pages = static_cast<std::uint32_t>(landing.main_kv_pages);
    PeakFitRelief raw;
    raw.device_main_kv_pages = landing.main_kv_pages;
    check(physical_peak_fits_core(occupied, limits, peak, raw),
          "629: the raw credit admits the peak - this is the over-credit the fix removes");

    // What the ladder could actually deliver: 36 owners with nothing Device-resident (`noruns`)
    // plus the one owner that still had pages. 145 pages, not 9,793.
    const std::uint32_t movable_pages = 145;
    PeakFitRelief bounded;
    bounded.device_main_kv_pages = std::min(landing.main_kv_pages, movable_pages);
    check(bounded.device_main_kv_pages == movable_pages,
          "629: the bounded credit is the movable page count, not the occupancy");
    check(!physical_peak_fits_core(occupied, limits, peak, bounded),
          "629: with the honest credit the gate refuses the peak, so the request goes to R0 "
          "(§三 R0) instead of being admitted into a ladder that cannot fund it");
}

// --- Host pinned full: the gate must credit the R2 relief §三 R1 prescribes ---

void case_host_full_credits_the_releasable_r2_unit() {
    // §三 R1: "内存没有空间接收时，先执行 R2 腾地方，再执行 R1". With the Host pool pinned full
    // the landing term of `kv_landing_relief` is free_bytes/stride = 0, so the gate used to read
    // the Device-KV relief as ZERO and reject a request the ladder could have served by releasing
    // one cold owner's Host side first. No scratch landing exists (the comment claiming one was
    // describing an intention, not code), so the R2 unit is the only relief there is.
    const std::size_t stride = 4096;
    const KvLandingRelief pinned = kv_landing_relief(/*host_free_bytes=*/0, stride, stride,
                                                     /*device_main_used=*/5000, 0);
    check(pinned.main_kv_pages == 0,
          "host-full: with no free Host bytes the landing term is zero (the pinned-full case)");

    // The short axis is HOST KV, which is exactly the axis §三 R1's R2 half answers: `fits_size`
    // needs used <= capacity - added, so 1,000 used against a 1,000 cap with a 100-byte add is
    // 100 bytes short. The ladder closes it by releasing the least valuable cold owner's Host KV
    // (`release_idle_owner_host_side`), which is the unit the gate failed to credit.
    PhysicalResources occupied;
    occupied.device.main_kv_pages = 5000;
    occupied.host.kv_bytes        = 1000;
    PhysicalResources limits;
    limits.device.main_kv_pages = 10284;
    limits.host.kv_bytes        = 1000;
    PhysicalResources peak;
    peak.host.kv_bytes = 100;

    PeakFitRelief none;                // what the gate credited before the fix
    none.device_main_kv_pages = pinned.main_kv_pages;   // 0: Host was pinned full
    check(!physical_peak_fits_core(occupied, limits, peak, none),
          "host-full: without the R2 credit the gate refuses a request the ladder could serve");

    // The R2 unit: the least valuable releasable owner's Host KV. Crediting it admits the request,
    // and the ladder then does exactly that release before retrying its landing.
    PeakFitRelief with_r2;
    with_r2.host_kv_bytes = 100;   // one releasable owner's Host KV
    check(physical_peak_fits_core(occupied, limits, peak, with_r2),
          "host-full: crediting the releasable R2 unit admits a servable request (§三 R1)");
}

// --- kv_landing_relief: the Device->Host demote landing the gate credits (main + backend pools) ---

void case_landing_bounded_by_host_then_device() {
    // 300 pages of Host landing (300 x 1000 B) with 1000 pages of Device KV in each pool: the
    // relief is Host-limited to 300 per pool, not Device-limited.
    const KvLandingRelief r = kv_landing_relief(300 * 1000, 1000, 1000, 1000, 1000);
    check(r.main_kv_pages == 300, "main relief is Host-limited when Host landing < Device in use");
    check(r.backend_kv_pages == 300, "backend relief is Host-limited when Host landing < Device in use");
}

void case_landing_device_bound() {
    // Plenty of Host landing (1000 pages) but only 50 pages of main Device KV in use: main relief
    // is Device-limited to 50; backend (1000 in use) is Host-limited to the 1000-page landing.
    const KvLandingRelief r = kv_landing_relief(1000 * 1000, 1000, 1000, 50, 1000);
    check(r.main_kv_pages == 50, "main relief is Device-limited when Device in use < Host landing");
    check(r.backend_kv_pages == 1000, "backend relief is Host-limited to the landing");
}

void case_landing_671_both_axes_covered() {
    // req#671: BOTH pools short by ~515 pages (main used=9793 cap=10284; backend used=9908
    // cap=10288). The landing must cover BOTH axes, not hand all of it to main and starve backend
    // (the "claim the remainder" flaw). With 2000 pages of Host landing, each pool gets 2000 >= 515.
    const KvLandingRelief r = kv_landing_relief(2000 * 1000, 1000, 1000, 9793, 9908);
    check(r.main_kv_pages >= 515, "req#671: main landing covers its ~515-page gap");
    check(r.backend_kv_pages >= 515, "req#671: backend landing covers its ~515-page gap (not starved)");
}

void case_landing_no_host_free_yields_no_relief() {
    // Both pools full (no Host landing): no KV relief - this is the both-full case the scratch swap
    // pool (a dedicated landing) exists to resolve; the gate alone cannot close it.
    const KvLandingRelief r = kv_landing_relief(0, 1000, 1000, 1000, 1000);
    check(r.main_kv_pages == 0 && r.backend_kv_pages == 0,
          "no Host landing yields no KV relief (both-full case needs the scratch landing)");
}

void case_landing_zero_stride_is_safe() {
    // A zero page stride (pool not configured) must not divide by zero: no relief on that pool.
    const KvLandingRelief r = kv_landing_relief(1000 * 1000, 0, 0, 1000, 1000);
    check(r.main_kv_pages == 0 && r.backend_kv_pages == 0, "zero page stride yields no relief");
}

// --- ③ host water line: the demand the planner builds, and the gap the policy derives from it ---

// These cases call the SAME two functions production does:
//   host_kv_demand(base, landing, headroom)  - what `pressure_planner.h` passes as `Demand::host_kv`
//   cachep::shortfall(demand, free)          - the policy's own gap arithmetic inside `plan()`
// so a green run says the water line holds on the code the service executes, not on a re-statement
// of it.
[[nodiscard]] std::uint64_t host_kv_gap(std::size_t base, std::size_t landing,
                                        std::size_t headroom, std::size_t host_free) {
    return ninfer::runtime::cache::shortfall(
        host_kv_demand(base, landing, headroom), host_free);
}

void case_headroom_keeps_a_water_line_not_a_full_host() {
    // ③: after landing `need`, Host must hold `headroom` free (not run to 100%).
    const std::size_t need     = 1ULL << 30;   // land 1 GiB of demoted KV
    const std::size_t headroom = 2ULL << 30;   // the ③ water line (2 GiB)
    const std::size_t free     = 512ULL << 20; // Host nearly full: 512 MiB free, below the water line
    const std::uint64_t evict  = host_kv_gap(0, need, headroom, free);
    check(evict == need + headroom - free,
          "③: evict enough that, after landing `need`, Host still holds `headroom` free");
    // The pre-③ gate (headroom = 0) evicts only enough to close the gap, leaving Host pinned-full;
    // ③ evicts exactly the headroom deficit (headroom - free) on top of that.
    const std::uint64_t evict_pre3 = host_kv_gap(0, need, 0, free);
    check(evict == evict_pre3 + headroom,
          "③ evicts exactly `headroom` more than pre-③ (lifts Host from full up to the water line)");
    // The request's own Host peak rides in front of the landing, so the demand covers both.
    const std::size_t base = 256ULL << 20;
    check(host_kv_gap(base, need, headroom, free) == base + need + headroom - free,
          "③: the demand is the request's own Host peak + landing + water line");
}

void case_headroom_no_eviction_when_host_has_room() {
    // Host free already covers base + landing + headroom: ③ is a no-op when there is room to spare.
    const std::size_t need     = 1ULL << 30;
    const std::size_t headroom = 2ULL << 30;
    const std::size_t free     = 4ULL << 30; // 4 GiB free >= need + headroom
    check(host_kv_gap(0, need, headroom, free) == 0,
          "③: no eviction when Host already holds need + headroom free");
}

void case_headroom_zero_reproduces_pre3_gap_only() {
    // headroom = 0 reproduces the pre-③ gate: evict exactly the gap, leaving Host pinned-full.
    const std::size_t need = 1ULL << 30;
    const std::size_t free = 512ULL << 20;
    check(host_kv_gap(0, need, 0, free) == need - free,
          "headroom=0 reproduces pre-③: evict just enough to close the gap (Host left full)");
    // Host full (free = 0) + headroom: ③ evicts need + headroom; pre-③ (headroom=0) evicts only need.
    const std::size_t headroom = 2ULL << 30;
    check(host_kv_gap(0, need, headroom, 0) == need + headroom,
          "③ at a full Host: evict need + headroom (pre-③ would evict just need and dead-lock)");
}

void case_headroom_does_not_overflow() {
    // Saturated demand: base + landing + headroom must saturate at max, never wrap - a wrapped
    // demand reads as "Host has room" and the request is admitted onto a pool with none.
    const std::size_t max_sz = std::numeric_limits<std::size_t>::max();
    check(host_kv_demand(max_sz - 2, max_sz - 2, max_sz - 2) == max_sz,
          "③: base + landing + headroom saturates at max instead of wrapping");
    check(host_kv_demand(max_sz - 2, 0, max_sz - 2) == max_sz,
          "③: base + watermark saturates at max instead of wrapping");
    check(ninfer::runtime::cache::shortfall(max_sz, 0) == max_sz,
          "a saturated demand against an empty Host is the full demand, no wrap");
}

// §三 R1 at reserve time: a Device-KV reservation that does not fit must first ask the release
// step to move cold Device KV to Host. This is the loop `LogicalKVPageStore::reserve_device_pages`
// runs, extracted so the rule is decidable without a Device arena.
//
// The bug this pins (2026-09-28 rig: eviction_fix and memory_pressure both HTTP 503): the gate
// credited the KV relief "you can demote cold owners into the free Host", but nothing executed it,
// so `resize_reservation` refused on the real books (alloc=1684 + need=488 > cap=2048), the engine
// re-admitted the request, and an idle engine then waited out the whole queue deadline. The loop
// below is what turns that credit into an actual move - so what it must never do is spin on a
// release step that reports success without freeing anything.
void case_device_kv_reservation_releases_until_it_fits() {
    std::printf("reserve_with_release_attempts: §三 R1 at reserve time\n");
    std::uint32_t held          = 1679;  // the rig's allocation right before the failing reserve
    const std::uint32_t wanted  = 488;   // the activation that request needed
    std::uint32_t release_calls = 0;
    // The pool takes the reservation once the release step has moved enough Device pages to Host.
    // 1679 + 488 > 2048, and ONE 300-page owner already closes it: 1379 + 488 <= 2048.
    const std::uint32_t attempts = reserve_with_release_attempts(
        [&]() { return held + wanted <= 2048; }, [&]() { return held; },
        [&]() {
            ++release_calls;
            if (held < 300) { return false; }  // nothing left that may be moved
            held -= 300;                       // one whole cold owner moves to Host
            return true;
        });
    check(attempts == 1, "one owner's move closed the rig's 124-page shortfall");
    check(release_calls == 1, "and the release step ran exactly that many times");
    check(held + wanted <= 2048, "the reservation fits afterwards");

    // Nothing to move: the walk stops, and the pool's own BAD_ALLOC is what the caller sees.
    std::uint32_t idle_calls = 0;
    check(reserve_with_release_attempts([&]() { return false; }, [&]() { return held; },
                                        [&]() {
                                            ++idle_calls;
                                            return false;
                                        }) == 0,
          "a release step that cannot free anything ends the walk");
    check(idle_calls == 1, "and is asked exactly once, not in a loop");

    // A step that reports success but does not shrink the pool must NOT spin: the reserve path
    // would otherwise hang the engine inside materialization instead of failing honestly.
    std::uint32_t lying_calls = 0;
    check(reserve_with_release_attempts([&]() { return false; }, [&]() { return held; },
                                        [&]() {
                                            ++lying_calls;
                                            return true;  // claims success, frees nothing
                                        }) == 0,
          "a step that frees nothing ends the walk even when it reports success");
    check(lying_calls == 1, "no spinning: one attempt, then the honest failure");

    // Already fits: the release step is never asked and nothing moves, which is the behaviour every
    // ordinary request keeps.
    std::uint32_t untouched = 0;
    check(reserve_with_release_attempts([&]() { return true; }, [&]() { return held; },
                                        [&]() {
                                            ++untouched;
                                            return true;
                                        }) == 0,
          "a reservation that fits performs no release");
    check(untouched == 0, "and never asks the release step");

    // Several owners in a row: the walk keeps going while each step makes progress, so a gap wider
    // than one owner still closes (the rig battery needed 21 such moves). 1000 + 100 > 1024, and
    // each owner hands back only 20 pages, so four of them are needed: 980/960/940/920, the last of
    // which fits.
    std::uint32_t many          = 1000;
    std::uint32_t chain_calls   = 0;
    const std::uint32_t chained = reserve_with_release_attempts(
        [&]() { return many + 100 <= 1024; }, [&]() { return many; },
        [&]() {
            ++chain_calls;
            if (many < 20) { return false; }
            many -= 20;
            return true;
        });
    check(chained == 4, "four owners were moved to close a 76-page gap in a 1024-page pool");
    check(chain_calls == 4, "and each of them was asked once");
    check(many + 100 <= 1024, "and the pool then takes the reservation");
}

int main() {
    std::printf("physical_peak_fits_core: admission-gate Device/Host KV eviction relief\n");
    case_full_device_pool_infeasible_without_kv_relief();
    case_kv_eviction_relief_makes_tail_restore_feasible();
    case_kv_relief_below_the_gap_stays_infeasible();
    case_kv_relief_exactly_closes_the_gap();
    case_host_kv_eviction_relief();
    case_state_relief_still_credits();
    case_relief_never_overflows_past_zero();
    std::printf("589 pipeline: pool snapshot -> kv_landing_relief -> gate -> tail feasible\n");
    case_589_pipeline_relief_makes_tail_feasible();
    std::printf("kv_landing_relief: Device->Host demote landing (main + backend pools)\n");
    case_629_credit_is_bounded_by_movable_pages();
    case_host_full_credits_the_releasable_r2_unit();
    case_landing_bounded_by_host_then_device();
    case_landing_device_bound();
    case_landing_671_both_axes_covered();
    case_landing_no_host_free_yields_no_relief();
    case_landing_zero_stride_is_safe();
    std::printf("③ host_kv_demand + cachep::shortfall: Host water line (keep `headroom` free)\n");
    case_headroom_keeps_a_water_line_not_a_full_host();
    case_headroom_no_eviction_when_host_has_room();
    case_headroom_zero_reproduces_pre3_gap_only();
    case_headroom_does_not_overflow();
    std::printf("reserve_with_release_attempts: Device-KV reservation releases until it fits\n");
    case_device_kv_reservation_releases_until_it_fits();
    if (g_failures != 0) {
        std::printf("  %d check(s) failed\n", g_failures);
        return 1;
    }
    std::printf("  all checks passed\n");
    return 0;
}
