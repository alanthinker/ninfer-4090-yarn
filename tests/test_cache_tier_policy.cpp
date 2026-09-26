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
//   - a drop may fund the spill it used to "eat": releasing a conversation's Host copy and
//     moving its Device copy into that room is ONE net action (§2.1), never a Device destroy

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

// R2 then R1: Device is full and Host is nearly full, so Host room has to exist BEFORE the
// spill lands. Host room is reserved for the incoming spill, then the drop closes exactly that.
void case_host_room_reserved_for_spill() {
    std::printf("case_host_room_reserved_for_spill\n");
    const std::vector<Datum> pool{
        {.id = 1, .device_kv = 60, .host_kv = 100, .importance = 1},
        {.id = 2, .device_kv = 60, .host_kv = 0, .importance = 9},
    };
    // device gap 50 -> the spill will need 50 of Host; Host free is 10 -> drop 40.
    const Plan outcome =
        decide(Demand{.device_kv = 50, .host_kv = 50}, occupancy(100, 100, 100, 90), pool);

    // Whole conversation: it holds 100 host bytes, so dropping it overshoots the 40 needed.
    check(action_kv(outcome, Action::DropFromHost) >= 40, "drop must cover the host gap");
    check(outcome.steps.size() == 2, "one drop (whole conversation) then one spill");
    check(!outcome.steps.empty() && outcome.steps.front().action == Action::DropFromHost,
          "Host must be cleared before the spill lands");
    check(!outcome.enqueue, "both gaps close, no waiting");
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

// §2.1: a drop may fund the very spill it used to "eat". The single candidate holds BOTH the
// Host room to free and the Device room to give back. Releasing its Host copy opens the room its
// own Device copy then lands in — one net action, nothing on Device destroyed — so the plan is
// real instead of a wait.
void case_drop_funds_the_spills_own_landing() {
    std::printf("case_drop_funds_the_spills_own_landing\n");
    const std::vector<Datum> pool{{.id = 1, .device_kv = 30, .host_kv = 40, .importance = 1}};
    const Plan outcome =
        decide(Demand{.device_kv = 30, .host_kv = 30}, occupancy(100, 100, 100, 95), pool);

    check(!outcome.enqueue, "drop + spill on one conversation closes both gaps");
    check(action_count(outcome, Action::DropFromHost) == 1, "the Host copy is released");
    check(action_kv(outcome, Action::SpillToHost) >= 30, "the Device gap must actually close");
    check(outcome.steps.size() == 2, "exactly two steps: release, then move");
    if (outcome.steps.size() == 2) {
        check(outcome.steps[0].id == 1 && outcome.steps[0].action == Action::DropFromHost,
              "the drop runs first - it opens the room");
        check(outcome.steps[1].id == 1 && outcome.steps[1].action == Action::SpillToHost,
              "the same conversation's Device copy lands in that room");
    }
}

// The same shape with a second conversation available: the first one alone closes both gaps, so
// the plan must NOT touch the second — the gap, not the pool, decides how much is touched.
void case_two_conversations_close_both_gaps() {
    std::printf("case_two_conversations_close_both_gaps\n");
    const std::vector<Datum> pool{
        {.id = 1, .device_kv = 30, .host_kv = 40, .importance = 1},
        {.id = 2, .device_kv = 30, .host_kv = 0, .importance = 9},
    };
    const Plan outcome =
        decide(Demand{.device_kv = 30, .host_kv = 30}, occupancy(100, 100, 100, 95), pool);

    check(!outcome.enqueue, "the drop frees Host room and the Device gap closes too");
    check(action_count(outcome, Action::DropFromHost) == 1, "exactly the conversation needed");
    check(action_kv(outcome, Action::SpillToHost) >= 30, "the Device gap must actually close");
    check(outcome.steps.size() == 2, "one drop, one move - no extra conversation");
    if (outcome.steps.size() == 2) {
        check(outcome.steps[0].id == 1, "the least important conversation is the one dropped");
        check(outcome.steps[1].id == 1, "and its own Device copy is what moves (§2.1)");
    }
    for (const auto& step : outcome.steps) {
        check(step.id != 2, "the second conversation survives untouched");
    }
}

// The four pools stay separate (§R2: each reports its own fullness, one ranking picks). A
// request blocked ONLY on state slots must reach the policy: every KV axis here reports a zero
// gap, and a two-pool implementation printed `gap dev=0 host=0` while Device state was 8/8 and
// Host state 48/48 - the policy went blind exactly when it had to act.
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
          "the full Host state pool frees room by dropping one conversation");
    check(action_count(outcome, Action::SpillToHost) == 1,
          "the full Device state pool frees room by moving one conversation");
    check(action_kv(outcome, Action::SpillToHost) == 0,
          "no KV moved at all - this plan is driven purely by state slots");
    check(action_state(outcome, Action::SpillToHost) >= 1, "the Device state gap must close");
    check(action_state(outcome, Action::DropFromHost) >= 1, "the Host state gap must close");
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

// §2.1: drop and spill may land on the SAME conversation — releasing its Host copy and moving
// its Device copy into that room is one net action. What stays structural is different from what
// this case used to assert: no plan step ever destroys Device data (there is no such Action), and
// a second conversation is never touched when the first one closes the gap. The runtime side of
// §7 acceptance #3 ("Device deletion count == 0") is a behavior counter, not a unit test.
void case_drop_and_spill_are_one_net_action() {
    std::printf("case_drop_and_spill_are_one_net_action\n");
    const std::vector<Datum> pool{
        {.id = 1, .device_kv = 50, .host_kv = 50, .importance = 1},
        {.id = 2, .device_kv = 50, .host_kv = 50, .importance = 9},
    };
    // Device gap 50, Host gap 50: conversation 1 answers both.
    const Plan outcome =
        decide(Demand{.device_kv = 50, .host_kv = 100}, occupancy(100, 100, 100, 50), pool);

    check(!outcome.enqueue, "both gaps are answerable");
    check(outcome.steps.size() == 2, "one drop and one move, nothing more");
    if (outcome.steps.size() == 2) {
        check(outcome.steps[0].id == 1 && outcome.steps[0].action == Action::DropFromHost,
              "the least important conversation's Host copy is released first");
        check(outcome.steps[1].id == 1 && outcome.steps[1].action == Action::SpillToHost,
              "and its own Device copy moves into the room that opened - one net action");
    }
    for (const auto& step : outcome.steps) {
        check(step.id != 2, "the second conversation is never touched for this gap");
    }
}

// §2.2's one chain (`value → age → id`): two equal scores still rank by recency — the oldest
// last-touched goes first — and only then by id. Without the age segment the policy tie-broke on
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

}  // namespace

int main() {
    case_device_short_host_roomy();
    case_host_room_reserved_for_spill();
    case_nothing_spillable_enqueues_without_dropping();
    case_host_unclosable_enqueues_without_dropping();
    case_no_gap_is_no_action();
    case_least_important_first_on_both_sides();
    case_active_data_is_never_a_candidate();
    case_drop_funds_the_spills_own_landing();
    case_two_conversations_close_both_gaps();
    case_state_pressure_is_visible_without_a_kv_gap();
    case_state_pressure_with_no_state_to_move_enqueues();
    case_nothing_is_dropped_for_a_gap_it_cannot_close();
    case_deletions_stop_at_the_gap();
    case_drop_and_spill_are_one_net_action();
    case_equal_importance_ranks_by_age_then_id();
    if (g_failures != 0) {
        std::printf("cache tier policy FAILED: %d check(s)\n", g_failures);
        return 1;
    }
    std::printf("cache tier policy ok\n");
    return 0;
}
