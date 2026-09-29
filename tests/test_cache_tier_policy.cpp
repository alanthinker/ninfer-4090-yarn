// Unit tests for the two-tier cache policy (docs/maintainer/缓存模块v2.md, rules R0-R3).
//
// Pure CPU: no artifact, no CUDA, no service, no inference. Every case is a hand-computed
// snapshot of four occupancy numbers plus one cache pool, so a failure names the exact rule
// that broke.
//
// What is being protected:
//   - Device never takes a drop (deletion exists only on the Host side)
//   - the number of datums touched is the gap, never a batch of it
//   - a plan that cannot close every gap it set out to close is applied as NOTHING and the
//     request enqueues instead (the "deleted a pile of cache that nobody used" failure)
//   - candidates are ranked by one ordering: value → age → id (§2.2's one chain, the same the
//     ladder's retire_preference uses)
//   - the four pools stay separate: a request blocked ONLY on state slots must be visible here,
//     even when every KV axis reports room
//   - one action per conversation: a release answers EVERY axis it holds (catalog row, Host
//     KV/state, Device relief through the move-first teardown) in that single step, so the R1
//     loop never emits a second step for it - HostReleases runs before CopyPreparation, and a
//     released address space would fail any later prepare

#include "runtime/cache/cache_tier_policy.h"

#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

using ninfer::runtime::cache::Action;
using ninfer::runtime::cache::Datum;
using ninfer::runtime::cache::Demand;
using ninfer::runtime::cache::EnqueueReason;
using ninfer::runtime::cache::Plan;
using ninfer::runtime::cache::TierOccupancy;

int g_failures = 0;

void check(bool condition, const char* what) {
    if (condition) { return; }
    std::printf("  FAIL: %s\n", what);
    ++g_failures;
}

// KV-only snapshot: Device and Host state pools start empty with zero capacity, so their gaps
// stay zero unless a case asks for state pressure.
[[nodiscard]] TierOccupancy occupancy(std::uint64_t device_kv_cap, std::uint64_t device_kv_used,
                                      std::uint64_t host_kv_cap, std::uint64_t host_kv_used) {
    return TierOccupancy{
        .device_kv_used        = device_kv_used,
        .device_kv_capacity    = device_kv_cap,
        .device_state_used     = 0,
        .device_state_capacity = 0,
        .host_kv_used          = host_kv_used,
        .host_kv_capacity      = host_kv_cap,
        .host_state_used       = 0,
        .host_state_capacity   = 0,
    };
}

[[nodiscard]] TierOccupancy state_occupancy(std::uint64_t device_state_cap,
                                            std::uint64_t device_state_used,
                                            std::uint64_t host_state_cap,
                                            std::uint64_t host_state_used) {
    return TierOccupancy{
        .device_kv_used        = 0,
        .device_kv_capacity    = 1024,
        .device_state_used     = device_state_used,
        .device_state_capacity = device_state_cap,
        .host_kv_used          = 0,
        .host_kv_capacity      = 0,
        .host_state_used       = host_state_used,
        .host_state_capacity   = host_state_cap,
    };
}

[[nodiscard]] Plan decide(const Demand& need, const TierOccupancy& occupancy,
                          const std::vector<Datum>& pool) {
    return ninfer::runtime::cache::plan(need, occupancy, pool);
}

[[nodiscard]] std::uint64_t action_kv(const Plan& outcome, Action action) {
    std::uint64_t total = 0;
    for (const auto& step : outcome.steps) {
        if (step.action == action) { total += step.kv; }
    }
    return total;
}

[[nodiscard]] std::uint64_t action_state(const Plan& outcome, Action action) {
    std::uint64_t total = 0;
    for (const auto& step : outcome.steps) {
        if (step.action == action) { total += step.state; }
    }
    return total;
}

[[nodiscard]] std::size_t action_count(const Plan& outcome, Action action) {
    std::size_t total = 0;
    for (const auto& step : outcome.steps) {
        if (step.action == action) { ++total; }
    }
    return total;
}

// §7 acceptance #1: Device is full, Host has room -> delete NOTHING and move Device pages to
// Host. "删除数 = 0" is asserted outright: a Device shortage with Host roomy must never cost a
// single cached conversation.
// The two Device KV pools are checked separately by the feasibility gate, so the plan must
// close them separately: a Main-heavy victim cannot answer a Backend gap however large its
// Main relief, and vice versa (battery 2026-09-26: controller targets rejected with
// backend_kv 10288+66/10288 while Main was closed by489 pages).
void case_backend_pool_closes_independently() {
    std::printf("case_backend_pool_closes_independently\n");
    const std::vector<Datum> pool{
        // huge Main, zero Backend - cannot answer the Backend gap at any priority
        {.id = 1, .device_kv = 500, .device_backend_kv = 0, .importance = 1},
        // modest Backend relief - the only one that counts for it
        {.id = 2, .device_kv = 0, .device_backend_kv = 40, .importance = 9},
        {.id = 3, .device_kv = 40, .device_backend_kv = 10, .importance = 50},
    };
    TierOccupancy occ = occupancy(100, 100, 1000, 0);
    occ.device_backend_kv_used     = 100;
    occ.device_backend_kv_capacity = 100;
    const Plan outcome =
        decide(Demand{.device_kv = 30, .device_backend_kv = 30}, occ, pool);

    check(!outcome.enqueue, "both pools close");
    bool took_id1 = false, took_backend_holder = false;
    for (const auto& step : outcome.steps) {
        if (step.action != Action::SpillToHost) { continue; }
        if (step.id == 1) { took_id1 = true; }
        if (step.id == 2 || step.id == 3) { took_backend_holder = true; }
    }
    check(took_backend_holder, "the Backend gap takes a victim that has Backend relief");
    check(took_id1, "the Main gap still takes the Main-heavy victim first");
}

// The soak stall, as a unit test: a Device pool full of fork-shared pages whose owners are
// PURE SHELLS (exclusive=0, no move option, no Host estate) - evicting any one of them frees
// nothing, so the main loop alone says 'device-not-closable' (rig: cand=20 steps=0). The
// shell pass must instead drop the WHOLE referent group: the joint pages return physically
// when the last referent goes, counted once (the max single claim is a sound lower bound for
// one group and never double-counts a page listed by both referents).
void case_shell_group_release_closes_device_gap() {
    std::printf("case_shell_group_release_closes_device_gap\n");
    const std::vector<Datum> pool{
        {.id = 1,
         .device_kv       = 0,
         .evict_device_kv = 0,
         .joint_device_kv = 40,
         .host_kv         = 0,
         .importance      = 10},
        {.id = 2,
         .device_kv       = 0,
         .evict_device_kv = 0,
         .joint_device_kv = 40,
         .host_kv         = 0,
         .importance      = 90},
        {.id = 3, .device_kv = 500, .host_kv = 800, .importance = 50},
    };
    TierOccupancy occ = occupancy(100, 100, 1000, 0); // device full, host roomy
    const Plan outcome = decide(Demand{.device_kv = 40}, occ, pool);

    check(!outcome.enqueue, "the joint pages of the shell group close the Device gap");
    std::size_t drops = 0;
    bool dropped_id3 = false;
    for (const auto& step : outcome.steps) {
        if (step.action != Action::DropFromHost) { continue; }
        ++drops;
        if (step.id == 3) { dropped_id3 = true; }
    }
    check(drops == 2, "BOTH referents of the shared pages are released (last one frees them)");
    check(!dropped_id3, "the ordinary owner with estate is untouched - shells only");
    check(action_count(outcome, Action::SpillToHost) == 0,
          "shells have no move option - there is nothing to spill");
}

// Spill-first survives the shell pass: a joint-capable owner that ALSO has a move option is
// spilled, not released (the shell predicate requires evict=0 AND move=0).
void case_shell_pass_defers_to_move() {
    std::printf("case_shell_pass_defers_to_move\n");
    const std::vector<Datum> pool{
        {.id = 1, .device_kv = 40, .joint_device_kv = 40, .host_kv = 0, .importance = 10},
    };
    TierOccupancy occ = occupancy(100, 100, 1000, 0);
    const Plan outcome = decide(Demand{.device_kv = 40}, occ, pool);

    check(!outcome.enqueue, "gap closes");
    check(action_count(outcome, Action::SpillToHost) == 1,
          "a shell WITH a move option is spilled first, never released");
    check(action_count(outcome, Action::DropFromHost) == 0, "and not dropped");
}

// EXACT soak numbers (from the production gap line): the plan must take at least the
// Host-state step the pool clearly allows - if this yields steps=0 the loop itself is broken.
void case_soak_exact_numbers() {
    std::printf("case_soak_exact_numbers\n");
    TierOccupancy occ;
    occ.device_kv_used = 1024;
    occ.device_kv_capacity = 1024;
    occ.device_state_used = 8;
    occ.device_state_capacity = 8;
    occ.device_backend_kv_used = 0;
    occ.device_backend_kv_capacity = 0;
    occ.host_kv_used = 1354760192;
    occ.host_kv_capacity = 12884901888;
    occ.host_state_used = 48;
    occ.host_state_capacity = 48;
    occ.catalog_rows_vacant = 116;
    const Demand need{.device_kv = 70,
                      .device_backend_kv = 0,
                      .device_state = 1,
                      .host_kv = 293601280,
                      .host_state = 1,
                      .catalog_rows = 0};
    const std::vector<Datum> pool{
        {.id = 0, .device_kv = 134, .evict_device_kv = 11, .host_state = 2, .importance = 10},
        {.id = 1, .device_kv = 49, .evict_device_kv = 11, .host_state = 4, .importance = 20},
        {.id = 2,
         .device_kv = 49,
         .evict_device_kv = 11,
         .evict_device_state = 1, // production's ev_state=8 - a DESTRUCTION footprint, not relief
         .host_state = 9,
         .importance = 30},
        {.id = 3, .device_kv = 11, .evict_device_kv = 11, .host_state = 2, .importance = 40},
    };
    const Plan outcome = decide(need, occ, pool);
    if (outcome.steps.empty() && !outcome.enqueue) {
        std::printf("  (no steps, no enqueue)\n");
    }
    // NOTHING may be released for the Device-state slot: every owner reports `device_state == 0`,
    // i.e. no state of theirs can be MOVED to Host, so a release would destroy the state it holds
    // (§四 invariant 1). The simulation verdict then has to refuse the whole plan - deleting cache
    // for a gap it cannot legally close is the failure this file exists to prevent (§三 R0).
    check(outcome.enqueue, "a Device-state gap nothing can move must wait, not delete");
    check(outcome.reason == EnqueueReason::DeviceNotClosable, "and name the Device axis");
    check(outcome.steps.empty(), "nothing is applied when the plan cannot close every gap");
}

void case_device_short_host_roomy() {
    std::printf("case device_short_host_roomy\n");
    const std::vector<Datum> pool{
        {.id = 1, .device_kv = 30, .importance = 10},
        {.id = 2, .device_kv = 40, .importance = 90},
    };
    // Spill landing folded in by the caller: 50 units of Device KV will end up on Host.
    const Plan outcome =
        decide(Demand{.device_kv = 50, .host_kv = 50}, occupancy(100, 100, 1000, 0), pool);

    check(action_count(outcome, Action::DropFromHost) == 0,
          "device gap must not delete Host cache");
    check(action_kv(outcome, Action::SpillToHost) >= 50, "spill must cover the gap");
    check(!outcome.enqueue, "room exists, nothing to wait for");
    // Whole conversations: the least important one (30) cannot cover a 50-unit gap on its own,
    // so the next one goes too - in full. A conversation is never moved halfway.
    check(outcome.steps.size() == 2, "consume in importance order until the gap closes");
    if (outcome.steps.size() == 2) {
        check(outcome.steps[0].id == 1 && outcome.steps[0].kv == 30,
              "least important conversation goes first, whole");
        check(outcome.steps[1].id == 2 && outcome.steps[1].kv == 40,
              "next conversation also goes whole, overshooting the gap");
    }
}

// R2 accounting: Device is full and Host is nearly full, so Host room must exist BEFORE
// anything lands. One release closes the Host gap and its Device relief counts in the same
// step - one action, no second step for it.
void case_host_room_reserved_for_spill() {
    std::printf("case_host_room_reserved_for_spill\n");
    const std::vector<Datum> pool{
        {.id = 1, .device_kv = 60, .host_kv = 100, .importance = 1},
        {.id = 2, .device_kv = 60, .host_kv = 0, .importance = 9},
    };
    // host gap 40 -> one release; its Device relief (60) also covers the Device gap (50).
    const Plan outcome =
        decide(Demand{.device_kv = 50, .host_kv = 50}, occupancy(100, 100, 100, 95), pool);

    check(!outcome.enqueue, "both gaps close in the release loop");
    check(outcome.steps.size() == 1, "one release answers both gaps - no second action");
    if (!outcome.steps.empty()) {
        check(outcome.steps.front().id == 1 &&
                  outcome.steps.front().action == Action::DropFromHost,
              "the least important conversation is the one released");
    }
    check(action_count(outcome, Action::SpillToHost) == 0,
          "a released conversation never receives a spill step");
}

// R0 dead zone: Device is full but every Device datum is active (nothing movable), while Host
// still holds droppable cache. The answer is "wait" with NO action - dropping Host cache here
// buys nothing and destroys something.
void case_nothing_spillable_enqueues_without_dropping() {
    std::printf("case nothing_spillable_enqueues_without_dropping\n");
    const std::vector<Datum> pool{
        {.id = 1, .device_kv = 100, .importance = 5, .active = true},
        {.id = 2, .device_kv = 0, .host_kv = 50, .importance = 1},
    };
    const Plan outcome =
        decide(Demand{.device_kv = 50, .host_kv = 50}, occupancy(100, 100, 1000, 950), pool);

    check(outcome.enqueue, "Device gap cannot close -> the request waits");
    check(outcome.reason == EnqueueReason::DeviceNotClosable,
          "the log must name WHICH tier could not close");
    check(outcome.steps.empty(), "an unusable plan must not delete anything at all");
    check(action_count(outcome, Action::DropFromHost) == 0, "no pointless Host deletion");
}

// The mirror: Host cannot be cleared enough to receive the spill, so nothing is dropped either.
void case_host_unclosable_enqueues_without_dropping() {
    std::printf("case_host_unclosable_enqueues_without_dropping\n");
    const std::vector<Datum> pool{
        {.id = 1, .device_kv = 60, .host_kv = 20, .importance = 1},
        {.id = 2, .device_kv = 60, .host_kv = 0, .importance = 9},
    };
    // device gap 50 -> needs 50 of Host; Host free is 10 -> must drop 40, but only 20 exists.
    const Plan outcome =
        decide(Demand{.device_kv = 50, .host_kv = 50}, occupancy(100, 100, 100, 90), pool);

    check(outcome.enqueue, "insufficient Host room -> wait");
    check(outcome.reason == EnqueueReason::HostNotClosable,
          "Host shortfall must be distinguishable from a Device shortfall");
    check(outcome.steps.empty(), "a plan that cannot close the gap must not be applied at all");
}

// R3: everything already fits - no steps, no queueing.
void case_no_gap_is_no_action() {
    std::printf("case no_gap_is_no_action\n");
    const std::vector<Datum> pool{{.id = 1, .device_kv = 30, .host_kv = 30, .importance = 1}};
    const Plan outcome =
        decide(Demand{.device_kv = 40, .host_kv = 10}, occupancy(100, 0, 100, 0), pool);
    check(outcome.steps.empty() && !outcome.enqueue, "available room means no work");
}

// Ordering: the least important datum is consumed first, on both sides.
void case_least_important_first_on_both_sides() {
    std::printf("case_least_important_first_on_both_sides\n");
    const std::vector<Datum> pool{
        {.id = 7, .device_kv = 10, .importance = 70},
        {.id = 3, .device_kv = 10, .importance = 30},
        {.id = 9, .device_kv = 10, .importance = 90},
    };
    // device gap 20, Host free 40 -> the reservation fits, no drop needed.
    const Plan outcome =
        decide(Demand{.device_kv = 20, .host_kv = 20}, occupancy(100, 100, 100, 60), pool);

    check(action_count(outcome, Action::DropFromHost) == 0, "Host has room, so nothing deleted");
    check(outcome.steps.size() == 2, "two datums cover a 20-unit gap - not the whole pool");
    if (outcome.steps.size() == 2) {
        check(outcome.steps[0].id == 3, "importance 30 goes before 70");
        check(outcome.steps[1].id == 7, "importance 70 goes before 90");
    }
}

// Active data is a working set, never a candidate - even when it carries the lowest value.
void case_active_data_is_never_a_candidate() {
    std::printf("case_active_data_is_never_a_candidate\n");
    const std::vector<Datum> pool{
        {.id = 1, .device_kv = 100, .host_kv = 100, .importance = 1, .active = true},
        {.id = 2, .device_kv = 30, .host_kv = 30, .importance = 50},
    };
    const Plan outcome =
        decide(Demand{.device_kv = 20, .host_kv = 20}, occupancy(100, 100, 1000, 0), pool);

    check(outcome.steps.size() == 1, "only the cache datum may be touched");
    if (!outcome.steps.empty()) {
        check(outcome.steps.front().id == 2, "the active datum must be left alone");
        check(outcome.steps.front().action == Action::SpillToHost, "and what is touched is moved");
    }
}

// The fifth pool (catalog rows) + Device relief in one action: with no vacant row the plan
// must release a conversation, and that single release also answers the Device gap because
// its relief is counted in the same step (invariant 3: one action, its full footprint).
void case_release_answers_row_and_device_gaps() {
    std::printf("case_release_answers_row_and_device_gaps\n");
    const std::vector<Datum> pool{
        {.id = 1, .device_kv = 30, .host_kv = 40, .catalog_row = 1, .importance = 10},
        {.id = 2, .device_kv = 30, .host_kv = 40, .catalog_row = 1, .importance = 90},
    };
    TierOccupancy occ = occupancy(100, 100, 1000, 0);
    occ.catalog_rows_vacant = 0;
    const Plan outcome = decide(Demand{.device_kv = 30, .catalog_rows = 1}, occ, pool);

    check(!outcome.enqueue, "row + Device gaps close in one release");
    check(action_count(outcome, Action::DropFromHost) == 1, "one row gap, one release");
    check(action_count(outcome, Action::SpillToHost) == 0,
          "the release's Device relief closes the Device gap - no second step");
    if (!outcome.steps.empty()) {
        check(outcome.steps.front().id == 1, "the least important conversation releases first");
    }
}

// A full catalog with nothing releasable: R0 with the reason that names the pool.
void case_catalog_full_without_candidates_enqueues() {
    std::printf("case_catalog_full_without_candidates_enqueues\n");
    const std::vector<Datum> pool; // empty: nothing exists to release
    TierOccupancy occ = occupancy(100, 0, 1000, 0);
    occ.catalog_rows_vacant = 0;
    const Plan outcome = decide(Demand{.catalog_rows = 1}, occ, pool);

    check(outcome.enqueue, "a full catalog with no candidate waits (R0)");
    check(outcome.reason == EnqueueReason::CatalogNotClosable,
          "the log must name WHICH pool could not serve");
    check(outcome.steps.empty(), "nothing exists to release - no half-plan");
}

// One release per gap-set: the least important conversation carries both the Host room the
// plan needs and the Device relief it promises, so the second conversation is never touched.
void case_two_conversations_close_both_gaps() {
    std::printf("case_two_conversations_close_both_gaps\n");
    const std::vector<Datum> pool{
        {.id = 1, .device_kv = 30, .host_kv = 40, .importance = 1},
        {.id = 2, .device_kv = 30, .host_kv = 0, .importance = 9},
    };
    const Plan outcome =
        decide(Demand{.device_kv = 30, .host_kv = 30}, occupancy(100, 100, 100, 95), pool);

    check(!outcome.enqueue, "the release frees Host room and its Device relief counts too");
    check(action_count(outcome, Action::DropFromHost) == 1, "exactly the conversation needed");
    check(action_count(outcome, Action::SpillToHost) == 0, "no second step for it");
    if (!outcome.steps.empty()) {
        check(outcome.steps.front().id == 1, "the least important conversation is released");
    }
    for (const auto& step : outcome.steps) {
        check(step.id != 2, "the second conversation survives untouched");
    }
}

// The four pools stay separate (§R2: each reports its own fullness, one ranking picks). A
// request blocked ONLY on state slots must reach the policy: every KV axis here reports a zero
// gap, and a two-pool implementation printed `gap dev=0 host=0` while Device state was 8/8 and
// Host state 48/48 - the policy went blind exactly when it had to act. One release closes both
// state gaps: the Host slots it frees and the Device state its teardown returns (invariant 3:
// one action, its full footprint).
// A state-only pressure snapshot: Host KV and Device KV are roomy, so the only gaps are the two
// state pools. The Device-state slot a request needs comes from MOVING a state to Host - never from
// releasing a conversation, because a release destroys what it could not move first (§四 invariant
// 1, and the 13 `site=transaction-victim` destructions of 2026-09-27). Here BOTH owners hold a
// moveable Device state, so the plan is: release the least important conversation to free the Host
// slots the move needs (R2 first, in the same least-important-first order), then move the survivor's
// Device state into that room (R1).
void case_state_pressure_is_visible_without_a_kv_gap() {
    std::printf("case_state_pressure_is_visible_without_a_kv_gap\n");
    const std::vector<Datum> pool{
        {.id = 1, .device_state = 4, .host_state = 4, .importance = 10},
        {.id = 2, .device_state = 4, .host_state = 4, .importance = 90},
    };
    const Plan outcome =
        decide(Demand{.device_state = 1, .host_state = 1}, state_occupancy(8, 8, 8, 8), pool);

    check(!outcome.enqueue, "state pressure must be answerable, not reported as nothing to do");
    check(action_count(outcome, Action::DropFromHost) == 1,
          "the full Host state pool frees room by releasing one conversation");
    check(action_kv(outcome, Action::SpillToHost) == 0, "no KV moves at all");
    check(action_state(outcome, Action::DropFromHost) >= 1, "the Host state gap must close");
    check(action_state(outcome, Action::SpillToHost) >= 1,
          "the Device state slot comes from a MOVE - a release may not destroy Device state");
    bool dropped_least_important = false, moved_the_survivor = false;
    for (const auto& step : outcome.steps) {
        if (step.id == 1 && step.action == Action::DropFromHost) { dropped_least_important = true; }
        if (step.id == 2 && step.action == Action::SpillToHost) { moved_the_survivor = true; }
    }
    check(dropped_least_important, "R2 releases the least important conversation");
    check(moved_the_survivor, "R1 then moves the survivor that stays in the pool");
}

// The other half of the same rule: if nothing carries Device state, the Device state gap cannot
// close, so the plan must be refused rather than applied half-way (R0, reason Device).
void case_state_pressure_with_no_state_to_move_enqueues() {
    std::printf("case_state_pressure_with_no_state_to_move_enqueues\n");
    const std::vector<Datum> pool{{.id = 1, .host_state = 4, .importance = 1}};
    const Plan outcome =
        decide(Demand{.device_state = 1, .host_state = 1}, state_occupancy(8, 8, 8, 8), pool);

    check(outcome.enqueue, "nothing carries Device state -> the request waits");
    check(outcome.reason == EnqueueReason::DeviceNotClosable,
          "the Device shortfall must be the named reason");
    check(outcome.steps.empty(),
          "the Host drop must not survive a plan that cannot close Device");
}

// Deleting is only justified by a gap it actually closes. Here the ONLY pressure is state: Host
// KV has room, Device KV has room, and conversation 1 carries a large Host KV footprint but no
// state at all. Taking it anyway is how "deleted a pile of cache that nobody used" reappears
// inside the four-pool rule - and on a saturated pool it compounds, because every request opens
// the same state gap and each one would wipe one more useless conversation.
void case_nothing_is_dropped_for_a_gap_it_cannot_close() {
    std::printf("case_nothing_is_dropped_for_a_gap_it_cannot_close\n");
    const std::vector<Datum> pool{
        {.id = 1, .host_kv = 900, .host_state = 0, .importance = 1},
        {.id = 2, .host_state = 4, .importance = 5},
        {.id = 3, .device_state = 4, .importance = 9},
    };
    TierOccupancy occ{
        .device_kv_used        = 0,
        .device_kv_capacity    = 1024,
        .device_state_used     = 8,
        .device_state_capacity = 8,
        .host_kv_used          = 0,
        .host_kv_capacity      = 1000000,
        .host_state_used       = 8,
        .host_state_capacity   = 8,
    };
    const Plan outcome =
        decide(Demand{.device_state = 1, .host_state = 1}, occ, pool);

    check(!outcome.enqueue, "both state gaps are answerable");
    check(outcome.steps.size() == 2, "exactly one drop and one move - nothing wasted");
    bool touched_one = false;
    for (const auto& step : outcome.steps) {
        if (step.id == 1) { touched_one = true; }
    }
    check(!touched_one,
          "the KV-only conversation answers no open gap and must survive untouched");
    if (outcome.steps.size() == 2) {
        check(outcome.steps[0].action == Action::DropFromHost && outcome.steps[0].id == 2,
              "the Host state gap is closed by the one conversation that carries Host state");
        check(outcome.steps[1].action == Action::SpillToHost && outcome.steps[1].id == 3,
              "the Device state gap is closed by the one conversation that carries Device state");
    }
}

// §7 acceptance #2: when Host is full, delete the least important conversations and STOP at the
// gap. Never a batch: ten candidates, a gap that two of them close - exactly two steps, and the
// two cheapest. This is invariant 3 ("the amount deleted equals the gap") read as "no more
// conversations than the gap needs", because a conversation cannot be deleted halfway.
void case_deletions_stop_at_the_gap() {
    std::printf("case_deletions_stop_at_the_gap\n");
    std::vector<Datum> pool;
    for (std::uint64_t id = 1; id <= 10; ++id) {
        pool.push_back(Datum{.id = id, .host_kv = 100, .importance = id * 10});
    }
    // Host KV 900/1000 -> 100 free; demand 250 -> gap 150 -> two whole conversations.
    const Plan outcome = decide(Demand{.host_kv = 250}, occupancy(1000, 0, 1000, 900), pool);

    check(!outcome.enqueue, "the pool can serve this");
    check(action_count(outcome, Action::DropFromHost) == 2,
          "exactly as many conversations as the gap needs - no batch");
    check(action_count(outcome, Action::SpillToHost) == 0, "Device has room, nothing moves");
    check(outcome.steps.size() == 2, "no extra conversation is touched");
    if (outcome.steps.size() == 2) {
        check(outcome.steps[0].id == 1 && outcome.steps[1].id == 2,
              "and they are the two least important");
    }
}

// Plan and execution must agree on ONE action per conversation: HostReleases runs before
// CopyPreparation, so a released conversation's address space is gone by the time any spill
// step would prepare - a second step would fail prepare and latch the engine. The release
// loop's Device accounting is what makes the second step unnecessary.
void case_release_never_gets_a_second_step() {
    std::printf("case_release_never_gets_a_second_step\n");
    const std::vector<Datum> pool{
        {.id = 1, .device_kv = 50, .host_kv = 50, .importance = 1},
        {.id = 2, .device_kv = 50, .host_kv = 50, .importance = 9},
    };
    const Plan outcome =
        decide(Demand{.device_kv = 50, .host_kv = 100}, occupancy(100, 100, 100, 50), pool);

    check(!outcome.enqueue, "both gaps are answerable");
    check(outcome.steps.size() == 1, "the release alone answers both gaps");
    if (!outcome.steps.empty()) {
        check(outcome.steps.front().id == 1 &&
                  outcome.steps.front().action == Action::DropFromHost,
              "released first, in importance order");
    }
    for (const auto& step : outcome.steps) {
        check(step.id != 2, "the second conversation is never touched for this gap");
    }
}

// §2.2's one chain (`value → age → id`): two equal scores still rank by recency - the oldest
// last-touched goes first - and only then by id. Without the age segment the policy tie-broke on
// catalog id while retire_preference tie-broke on age, so the same snapshot could pick two
// different victims (§2.2).
void case_equal_importance_ranks_by_age_then_id() {
    std::printf("case_equal_importance_ranks_by_age_then_id\n");
    const std::vector<Datum> pool{
        {.id = 5, .device_kv = 30, .importance = 40, .age_key = 900}, // touched recently
        {.id = 2, .device_kv = 30, .importance = 40, .age_key = 100}, // untouched much longer
    };
    const Plan outcome = decide(Demand{.device_kv = 30}, occupancy(100, 100, 1000, 0), pool);
    check(outcome.steps.size() == 1, "one datum covers the gap");
    if (outcome.steps.size() == 1) {
        check(outcome.steps[0].id == 2, "equal scores: the OLDER last-touched goes first");
    }

    const std::vector<Datum> tied{
        {.id = 7, .device_kv = 30, .importance = 40, .age_key = 100},
        {.id = 3, .device_kv = 30, .importance = 40, .age_key = 100},
    };
    const Plan tie_outcome = decide(Demand{.device_kv = 30}, occupancy(100, 100, 1000, 0), tied);
    check(tie_outcome.steps.size() == 1, "one datum covers the gap");
    if (tie_outcome.steps.size() == 1) {
        check(tie_outcome.steps[0].id == 3, "equal score and age tie-break on id (lower first)");
    }
}

// 2026-09-27, the pressure-case finding as a unit test (site=transaction-victim).
//
// `parallel-fresh`, `tiny-overlap`, `decode-vs-plan`, `fill-big`, `twins`, `same-session-race` and
// `spill-race` all run a 4-slot Host state pool. On a saturated Device state pool the policy
// answered the state gap by RELEASING a conversation, crediting the eviction option's Device-state
// footprint (`evict_device_state`) as relief. At execution the move-first prelude could not move
// that state (no Host slot to land on, and the R2 degrade declined), so the teardown destroyed it -
// 13 `[invariant1] strict site=transaction-victim ... state=N slots` lines in one suite run. The rig
// never reproduced it because its Host state pool is 48 slots.
//
// The decision layer must not promise that: 显存里的数据只有两种归宿 - 正在用, 或搬到内存 (§三 R1).
// A Device-state gap may only be closed by a MOVE (`device_state > 0`, which the Program only fills
// in when the demotion can actually land - possibly after an R2 degrade). An owner whose Device
// state cannot move is not releasable, so nothing is dropped and the request waits (R0).
void case_device_state_gap_is_never_closed_by_evicting_a_state_it_cannot_move() {
    std::printf("case_device_state_gap_is_never_closed_by_evicting_a_state_it_cannot_move\n");
    const std::vector<Datum> pool{
        // The eviction option would free 2 Device state slots, but there is no move option: the
        // demotion has nowhere to land (`host_state` is full, `device_state` is 0).
        {.id = 1, .device_state = 0, .evict_device_state = 2, .host_state = 0, .importance = 1},
    };
    const Plan outcome = decide(Demand{.device_state = 1}, state_occupancy(4, 4, 4, 4), pool);

    check(outcome.enqueue, "a Device-state gap nothing can MOVE must wait, not evict");
    check(outcome.reason == EnqueueReason::DeviceNotClosable, "and name the Device axis");
    check(outcome.steps.empty(),
          "no release step: destroying the owner's Device state is what §四 invariant 1 forbids");
}

// The control: with a move option the same gap IS answered - by moving, never by destroying.
void case_device_state_gap_closes_by_moving_a_state_that_can_move() {
    std::printf("case_device_state_gap_closes_by_moving_a_state_that_can_move\n");
    const std::vector<Datum> pool{
        {.id = 1, .device_state = 1, .evict_device_state = 1, .host_state = 1, .importance = 1},
    };
    const Plan outcome =
        decide(Demand{.device_state = 1, .host_state = 1}, state_occupancy(4, 4, 4, 3), pool);

    check(!outcome.enqueue, "a moveable Device state closes the gap");
    check(action_count(outcome, Action::SpillToHost) == 1, "by moving it to Host");
    check(action_count(outcome, Action::DropFromHost) == 0, "and not by releasing the owner");
}

// req-589 (2026-09-28): a returning session keeps its tail KV on Host and its endpoint mirror,
// but to run it must restore that prefix onto a Device pool already full of three concurrent
// full-context sessions. The only way to free Device pages is to demote another session's Device KV
// onto a Host pool that is itself full with nothing droppable - so no legal move closes the gap.
// The policy must WAIT (R0), never apply a partial plan that demotes or deletes cache it cannot
// actually place; the wiring layer then serves the request by re-deriving from the deepest
// Device-restorable long anchor instead of blocking on a restore that will not fit.
//
// Guard: a change that "frees" Device by overwriting a full Host (or drops a Host estate that has
// nowhere to receive the demotion) re-opens "deleted a pile of cache that nobody used" exactly
// here, on the two-tier-saturated shape that produced the 96.8s req-589.
void case_two_tier_saturation_restores_by_r0_not_partial_delete() {
    std::printf("case_two_tier_saturation_restores_by_r0_not_partial_delete\n");
    const std::vector<Datum> pool{
        // the only idle estate: 50 Device pages and NO Host estate, so R2 has nothing of it to
        // drop to make room, and Host is full so a Device demotion has nowhere to land.
        {.id = 1, .device_kv = 50, .host_kv = 0, .importance = 10},
    };
    // Device full (free 0) and Host full (free 0): the restore needs 50 Device pages, and staging
    // the demotion needs 50 Host bytes this pool cannot provide.
    const Plan outcome =
        decide(Demand{.device_kv = 50, .host_kv = 50}, occupancy(100, 100, 1000, 1000), pool);

    check(outcome.enqueue, "neither tier can close the gap -> the request waits (R0)");
    check(outcome.reason == EnqueueReason::HostNotClosable,
          "the binding axis is Host (no room to receive the Device demotion)");
    check(outcome.steps.empty(),
          "no partial plan: Device must not be freed by overwriting a full Host or dropping estate");
}

// Which of ONE victim's own move alternatives the planner picks. The pool reports the CHOSEN
// alternative's own per-pool relief, so an alternative that moves only Main credits Main and
// NOTHING to Backend - however many pages it moves.
//
// The bug this pins (2026-09-28 production: one conversation's request parked to its queue
// deadline four times in a row, cancelled by the client at 5 minutes each time): the alternatives
// were ranked by their SUMMED page count. Main and Backend are mirror pools of equal size per
// owner, so "move all Main" and "move all Backend" tie on pages and the first one always won -
// every victim reported its whole Main estate and zero Backend, the 636-page Backend gap could not
// be closed by anybody, `plan()` enqueued `device-not-closable` forever, and free Device pages
// never moved off 1238 while 88% of the pool sat there as finished conversations' cache. The pool
// sums line showed it only as `mv_main=9046` - exactly the entire Device Main occupancy.
void case_move_alternative_answers_every_short_pool() {
    std::printf("case_move_alternative_answers_every_short_pool\n");
    using ninfer::runtime::cache::MoveRelief;
    using ninfer::runtime::cache::prefer_move_relief;
    const std::uint64_t page_bytes  = 1U << 20;
    const std::uint64_t image_bytes = 1U << 16;
    const auto prefer = [&](const MoveRelief& candidate, const MoveRelief& incumbent,
                            bool main_short, bool backend_short, bool state_short = false) {
        return prefer_move_relief(candidate, incumbent, main_short, backend_short, state_short,
                                  page_bytes, image_bytes);
    };

    // Both pools short, mirror owner (a 242K-token conversation: 3785 pages on each side).
    // "move all Main" (3785) beats "move both, exactly the gap" (640 + 636) on summed pages - and
    // that is precisely the alternative that leaves Backend at zero.
    const MoveRelief full_main{.main_pages = 3785, .backend_pages = 0};
    const MoveRelief both_axes{.main_pages = 640, .backend_pages = 636};
    check(!prefer(full_main, both_axes, true, true),
          "a Main-only alternative must not win while Backend is short too");
    check(prefer(both_axes, full_main, true, true),
          "the alternative that answers BOTH short pools is the victim's move option");
    check(prefer(both_axes, MoveRelief{}, true, true),
          "the first covering alternative becomes the incumbent");

    // Same owner, only Main short: Backend relief is not required, so the largest Main move wins -
    // the per-pool rule must not shrink a move the plan can use, and must not add pages nobody
    // asked for (a Backend-only alternative covers nothing and stays out).
    const MoveRelief full_backend{.main_pages = 0, .backend_pages = 3785};
    check(prefer(full_main, MoveRelief{}, true, false), "Main-only wins when only Main is short");
    check(!prefer(full_backend, full_main, true, false),
          "a Backend-only alternative cannot answer a Main-only gap");
    check(!prefer(MoveRelief{}, full_main, true, false),
          "a no-op alternative never replaces a real move");

    // Only Backend short: the mirror case, and the one the old rule lost.
    check(prefer(full_backend, MoveRelief{}, false, true),
          "Backend-only wins when only Backend is short");
    check(!prefer(full_main, full_backend, false, true),
          "a Main-only alternative cannot answer a Backend-only gap");

    // A victim that can reach only ONE of the short pools has no covering alternative at all: it
    // keeps its largest move and is simply not the victim that closes the other pool.
    const MoveRelief main_only_small{.main_pages = 224, .backend_pages = 0};
    const MoveRelief main_only_large{.main_pages = 2798, .backend_pages = 0};
    check(prefer(main_only_large, main_only_small, true, true),
          "with no covering alternative, the largest move still wins");
    check(!prefer(main_only_small, main_only_large, true, true),
          "and a smaller one does not replace it");

    // State stays a pool of its own with its own weight: a move that hands back state slots and no
    // KV pages is still a move when neither KV pool is short.
    const MoveRelief state_only{.state_slots = 1};
    check(prefer(state_only, MoveRelief{}, false, false),
          "a state-only move is chosen when no KV pool is short");

    // All three Device pools short (the production shape): a KV-only alternative answers two of the
    // three axes and must not win - the state gap would then have no reporter at all, which is the
    // same stall one pool over.
    const MoveRelief kv_both_axes{.main_pages = 640, .backend_pages = 636, .state_slots = 0};
    const MoveRelief residual{.main_pages = 641, .backend_pages = 637, .state_slots = 1};
    check(prefer(residual, kv_both_axes, true, true, true),
          "with Device state short too, the alternative that answers all three axes wins");
    check(!prefer(kv_both_axes, residual, true, true, true),
          "a KV-only alternative must not displace the one that also hands back state");
    check(!prefer(state_only, residual, true, true, true),
          "and a state-only alternative cannot answer the KV pools either");
    // State short on its own: state-only is a covering alternative, so it is chosen on the pool
    // that is actually short rather than on the summed weight of KV pages nobody asked for.
    check(prefer(state_only, kv_both_axes, false, false, true),
          "with only Device state short, the state move covers and the KV alternative does not");
}

// Reproduced: a Host-water-line demand that can never be met must not swallow the plan.
//
// 2026-09-29 production, pool saturated (host KV 31.98/32 GiB, device KV 10236/10284 pages):
// a 4,136-token request sat in the queue for its whole 5-minute deadline and was cancelled with
// `output 0`. The plan line for it reads
//
//   [cache] gap dkv=73 dbkv=73 dstate=1 hkv=2197618688 hstate=0 rows=0
//         | free dkv=50 dbkv=55 dstate=8 hkv=19394560 hstate=0 rows=292
//         | cand=236 steps=62 | gaps dkv=23 dbkv=18 ...
//   [ladder] non-destructive-exhausted drop=0 demote=0 evicthost=0 dropck=0 host_free=2 dev=0/8
//
// ③ raises `Demand::host_kv` by the 2 GiB water line so a later demote always has somewhere to
// land. With Host nearly full that demand becomes a ~2.18 GB gap that NO amount of cache can
// close, and the release loop walks its whole candidate list chasing it: all 62 steps came back
// `removed main=0` (Host-side drops), the Device-KV gap of 23 pages was never answered, and the
// plan then failed its own simulation and was discarded as a whole (§八) - the request waited out
// its deadline against a pool it could have been served from.
//
// The water line is a PREFERENCE, not a requirement: ③'s own comment says it exists "so a demote
// always has room to land", which only matters when the plan actually moves Device data to Host.
// It must never turn an answerable Device gap into an unanswerable plan.
void case_unreachable_host_water_line_does_not_swallow_the_device_gap() {
    std::printf("case_unreachable_host_water_line_does_not_swallow_the_device_gap\n");
    // The production numbers, rounded to what the policy actually takes: bytes for Host, pages
    // for Device.
    TierOccupancy occ = occupancy(/*device_kv_cap=*/10284, /*device_kv_used=*/10261,
                                  /*host_kv_cap=*/32ULL << 30, /*host_kv_used=*/(32ULL << 30) - (19ULL << 20));
    occ.device_backend_kv_capacity = 10284;
    occ.device_backend_kv_used     = 10261;
    occ.device_state_used          = 0;
    occ.device_state_capacity      = 8;   // the `device_state = 1` demand fits: state is NOT the
                                          // binding axis here, the Device KV pages are
    occ.host_state_capacity        = 320;

    // Two idle conversations holding Device pages the plan could move, plus Host-side cache the
    // release loop can drop for the water line.
    const std::vector<Datum> pool{
        {.id = 1, .device_kv = 40, .device_backend_kv = 40, .host_kv = 64ULL << 20,
         .host_state = 2, .catalog_row = 1, .importance = 10},
        {.id = 2, .device_kv = 40, .device_backend_kv = 40, .host_kv = 64ULL << 20,
         .host_state = 2, .catalog_row = 1, .importance = 20},
    };

    // The production decomposition of that request's Host demand: ~41 MB of landing the plan
    // really owes (41 Device pages at the rig's page stride), on top of ③'s 2 GiB water line, with
    // Host KV 19 MB from its cap. The landing fits in what the two idle owners can hand back; the
    // water line can never be reached, and reaching it is not this plan's job.
    // ③ declares its water line through `host_kv_preference`; everything above it is the landing
    // the plan actually owes.
    const Plan outcome = decide(Demand{.device_kv = 73,
                                       .device_backend_kv = 73,
                                       .device_state = 1,
                                       .host_kv = ((41ULL + 8) << 20) + (2ULL << 30),
                                       .host_state = 0,
                                       .catalog_rows = 0,
                                       .host_kv_preference = (2ULL << 30)},
                                occ, pool);

    check(!outcome.enqueue,
          "a 23-page Device gap must be answered even when the Host water line cannot be reached");
    check(action_kv(outcome, Action::SpillToHost) >= 23,
          "the Device gap is closed by moving the idle owners' Device KV to Host");
}

}  // namespace

int main() {
    case_move_alternative_answers_every_short_pool();
    case_unreachable_host_water_line_does_not_swallow_the_device_gap();
    case_backend_pool_closes_independently();
    case_shell_group_release_closes_device_gap();
    case_shell_pass_defers_to_move();
    case_soak_exact_numbers();
    case_device_short_host_roomy();
    case_host_room_reserved_for_spill();
    case_nothing_spillable_enqueues_without_dropping();
    case_host_unclosable_enqueues_without_dropping();
    case_no_gap_is_no_action();
    case_least_important_first_on_both_sides();
    case_active_data_is_never_a_candidate();
    case_release_answers_row_and_device_gaps();
    case_catalog_full_without_candidates_enqueues();
    case_two_conversations_close_both_gaps();
    case_state_pressure_is_visible_without_a_kv_gap();
    case_state_pressure_with_no_state_to_move_enqueues();
    case_device_state_gap_is_never_closed_by_evicting_a_state_it_cannot_move();
    case_device_state_gap_closes_by_moving_a_state_that_can_move();
    case_two_tier_saturation_restores_by_r0_not_partial_delete();
    case_nothing_is_dropped_for_a_gap_it_cannot_close();
    case_deletions_stop_at_the_gap();
    case_release_never_gets_a_second_step();
    case_equal_importance_ranks_by_age_then_id();
    if (g_failures != 0) {
        std::printf("cache tier policy FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("cache tier policy ok\n");
    return 0;
}
