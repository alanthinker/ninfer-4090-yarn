// Two-tier cache policy: the ONE place that decides what moves and what is dropped.
//
// The rules (see docs/maintainer/缓存模块v2.md, which is the authority for them):
//
//   R1  Device full  -> move the LEAST important Device datum to Host   (move, never destroy)
//   R2  Host full    -> drop    the LEAST important Host datum          (Host side only)
//   R3  New request  -> hit Device: use it; hit Host only: restore to Device; miss: compute
//   R0  Cannot move and cannot drop -> enqueue; only a queue timeout rejects
//
// Three invariants are structural here rather than advisory:
//
//   - `plan()` never emits a Device-side drop. Device data has exactly two outcomes: it is in
//     use, or it moved to Host.
//   - One ordering. `Datum::importance` is the only notion of value in this file; no caller may
//     re-rank the pool, and no other module names a victim.
//   - The pools are NEVER merged. There are four of them - Device KV, Device state, Host KV,
//     Host state - and each reports its own "am I full" (§R2). What they share is only WHO moves
//     or dies: one conversation-level ranking. Summing a KV pool and a state pool into one budget
//     is how a request gets told it has room while the pool it actually needs is full.
//
// The module is a pure function over an occupancy snapshot plus the cache pool. It owns no
// memory, performs no I/O, and does not know what a KV page or a state image is: every axis is
// counted in its own pool's unit (Device KV in pages, Host KV in bytes, state in slots), because
// every comparison here is between two numbers of the SAME pool.

#pragma once

#include <algorithm>
#include <cstdint>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace ninfer::runtime::cache {

// Capacity and occupancy of all four pools, each in its own unit.
// `*_used <= *_capacity` holds per pool.
struct TierOccupancy {
    std::uint64_t device_kv_used       = 0;
    std::uint64_t device_kv_capacity   = 0;
    std::uint64_t device_state_used    = 0;
    std::uint64_t device_state_capacity = 0;
    std::uint64_t host_kv_used         = 0;
    std::uint64_t host_kv_capacity     = 0;
    std::uint64_t host_state_used      = 0;
    std::uint64_t host_state_capacity  = 0;

    [[nodiscard]] std::uint64_t device_kv_free() const noexcept {
        return device_kv_capacity > device_kv_used ? device_kv_capacity - device_kv_used : 0;
    }
    [[nodiscard]] std::uint64_t device_state_free() const noexcept {
        return device_state_capacity > device_state_used
                   ? device_state_capacity - device_state_used
                   : 0;
    }
    [[nodiscard]] std::uint64_t host_kv_free() const noexcept {
        return host_kv_capacity > host_kv_used ? host_kv_capacity - host_kv_used : 0;
    }
    [[nodiscard]] std::uint64_t host_state_free() const noexcept {
        return host_state_capacity > host_state_used ? host_state_capacity - host_state_used : 0;
    }
};

// What a request (or a restore) needs in order to run, one number per pool, in that pool's unit.
//
// The Host fields must already carry everything a Device spill will consume on Host - both the KV
// and the state images - because the policy performs no conversion between tiers or between
// pools. Only the caller, which owns both layouts, can assemble those numbers.
struct Demand {
    std::uint64_t device_kv    = 0;
    std::uint64_t device_state = 0;
    std::uint64_t host_kv      = 0;
    std::uint64_t host_state   = 0;
};

// One cached CONVERSATION - the unit of retention. `id` is its catalog row.
//
// A conversation occupies all four pools at once, so it carries one number per pool rather than
// one total: a conversation can hold state slots while holding no KV left, and folding the two
// into a single figure per tier would let it be half counted. One conversation is moved or
// dropped whole, and dropping it returns its Host KV AND its Host state together with its
// catalog row - never half of it (§ invariant 6).
//
// `active` data is a working set, not cache: it is never moved and never dropped, so it is
// excluded from every candidate list below.
struct Datum {
    std::uint64_t id           = 0;  // catalog row: the conversation's identity
    std::uint64_t device_kv    = 0;  // its Device KV, in Device KV units
    std::uint64_t device_state = 0;  // its Device state images, in Device state slots
    std::uint64_t host_kv      = 0;  // its Host KV, in bytes
    std::uint64_t host_state   = 0;  // its Host state images, in Host state slots
    std::uint64_t importance   = 0;  // larger == more worth keeping. The ONLY ordering.
    bool          active       = false;
};

enum class Action : std::uint8_t {
    SpillToHost,  // R1: Device -> Host
    DropFromHost, // R2: destroy on the Host side
};

// Why a plan asked the caller to wait. A bare `enqueue = true` is undiagnosable after the fact -
// the two causes need different fixes, so the reason travels with the plan and is asserted in
// tests and printed by the wiring layer.
enum class EnqueueReason : std::uint8_t {
    None,               // the plan is actionable
    DeviceNotClosable,  // Device could not give up enough (what is left is all active)
    HostNotClosable,    // Host could not take the spill or give up enough to receive it
};

// One whole-conversation action, reported per pool rather than in one number: `kv` is the
// conversation's KV (Device units when spilling, Host bytes when dropping) and `state` its state
// slot count, so a log line names the pool that actually moved.
struct Step {
    std::uint64_t id         = 0;
    Action        action     = Action::SpillToHost;
    std::uint64_t kv         = 0;
    std::uint64_t state      = 0;
    std::uint64_t importance = 0;  // carried so a log line needs no second lookup
};

struct Plan {
    std::vector<Step> steps;  // R2 steps first, then R1 steps
    bool enqueue = false;     // R0: not every gap it set out to close could be closed
    EnqueueReason reason = EnqueueReason::None;
};

namespace detail {

// Ascending by importance: the least worth keeping goes first. Ties break on id so the plan is
// reproducible for a given snapshot.
inline void rank_least_important(std::vector<const Datum*>& pool) {
    std::sort(pool.begin(), pool.end(), [](const Datum* left, const Datum* right) {
        if (left->importance != right->importance) {
            return left->importance < right->importance;
        }
        return left->id < right->id;
    });
}

} // namespace detail

// Decide how to make room for `need` given `occupancy` and the current cache pool.
//
// The order is fixed by the rules, not by cost: R2 before R1, because Host has to have somewhere
// to receive what Device is about to hand over. A conversation gets exactly ONE action - one
// dropped for Host room cannot also be the one spilled for Device room - so the Device budget is
// whatever the survivors still carry.
//
// Sufficiency is decided by SIMULATION rather than a pre-check: the plan is built into scratch
// first and committed only when every gap it set out to close is actually closed. Judging
// closability against the whole pool before choosing victims let a plan drop cache and still come
// up short on Device - deletion without delivery - and that is ruled out structurally here.
inline Plan plan(const Demand& need, const TierOccupancy& occupancy,
                 std::span<const Datum> pool) {
    Plan out;

    const auto shortfall = [](std::uint64_t want, std::uint64_t have) -> std::uint64_t {
        return want > have ? want - have : 0;
    };
    // `need.host_*` already carries what a Device spill will consume on Host.
    const std::uint64_t device_kv_gap =
        shortfall(need.device_kv, occupancy.device_kv_free());
    const std::uint64_t device_state_gap =
        shortfall(need.device_state, occupancy.device_state_free());
    const std::uint64_t host_kv_gap    = shortfall(need.host_kv, occupancy.host_kv_free());
    const std::uint64_t host_state_gap = shortfall(need.host_state, occupancy.host_state_free());

    if (device_kv_gap == 0 && device_state_gap == 0 && host_kv_gap == 0 &&
        host_state_gap == 0) {
        return out;  // R3: everything the request needs is already placeable.
    }

    // Rank once. Active data is excluded: it is a working set, not cache.
    std::vector<const Datum*> candidates;
    candidates.reserve(pool.size());
    for (const Datum& datum : pool) {
        if (datum.active) { continue; }
        candidates.push_back(&datum);
    }
    detail::rank_least_important(candidates);

    // R2: delete whole conversations on the Host side, least important first, until BOTH Host
    // pools have their room. A conversation is only taken when it answers a gap that is still
    // open: with `host_kv` already satisfied, deleting a conversation that carries KV but no
    // state frees nothing this plan needs and destroys it for nothing - deletion count must
    // equal gap count (§ invariant 3), not "everything with any footprint". The last one may
    // still free more than the gap, because a conversation cannot be dropped halfway.
    std::vector<std::uint64_t> dropped;
    dropped.reserve(candidates.size());
    std::uint64_t host_kv_freed    = 0;
    std::uint64_t host_state_freed = 0;
    for (const Datum* datum : candidates) {
        if (host_kv_freed >= host_kv_gap && host_state_freed >= host_state_gap) { break; }
        const bool helps_kv =
            host_kv_freed < host_kv_gap && datum->host_kv > 0;
        const bool helps_state =
            host_state_freed < host_state_gap && datum->host_state > 0;
        if (!helps_kv && !helps_state) { continue; }
        dropped.push_back(datum->id);
        host_kv_freed += datum->host_kv;
        host_state_freed += datum->host_state;
    }

    // R1: move whole conversations to Host, least important first, until BOTH Device pools have
    // their room. The same rule applies: only a conversation that answers an open Device gap is
    // moved. Dropped conversations are already gone from both tiers, so they cannot answer it.
    std::vector<Step> spills;
    spills.reserve(candidates.size());
    std::uint64_t device_kv_moved    = 0;
    std::uint64_t device_state_moved = 0;
    for (const Datum* datum : candidates) {
        if (device_kv_moved >= device_kv_gap && device_state_moved >= device_state_gap) { break; }
        if (std::find(dropped.begin(), dropped.end(), datum->id) != dropped.end()) { continue; }
        const bool helps_kv =
            device_kv_moved < device_kv_gap && datum->device_kv > 0;
        const bool helps_state =
            device_state_moved < device_state_gap && datum->device_state > 0;
        if (!helps_kv && !helps_state) { continue; }
        spills.push_back(Step{datum->id, Action::SpillToHost, datum->device_kv,
                              datum->device_state, datum->importance});
        device_kv_moved += datum->device_kv;
        device_state_moved += datum->device_state;
    }

    // Simulation verdict. Anything short means the plan cannot do what it promised, so it is not
    // applied at all - nothing is deleted, nothing is moved, the caller waits instead (R0).
    const bool host_short =
        host_kv_freed < host_kv_gap || host_state_freed < host_state_gap;
    const bool device_short =
        device_kv_moved < device_kv_gap || device_state_moved < device_state_gap;
    if (device_short || host_short) {
        out.enqueue = true;
        out.reason  = device_short ? EnqueueReason::DeviceNotClosable
                                   : EnqueueReason::HostNotClosable;
        return out;
    }

    // R2 steps first, in the same least-important-first order the drop loop walked.
    out.steps.reserve(dropped.size() + spills.size());
    std::size_t position = 0;
    for (const Datum* datum : candidates) {
        if (position >= dropped.size()) { break; }
        if (dropped[position] != datum->id) { continue; }
        out.steps.push_back(Step{datum->id, Action::DropFromHost, datum->host_kv,
                                 datum->host_state, datum->importance});
        ++position;
    }
    for (Step& step : spills) { out.steps.push_back(std::move(step)); }
    return out;
}

// One log line for a whole decision. The policy stays I/O-free; the wiring layer prints this
// under the existing NINFER_REUSE_DIAG switch, so every move and every drop is traceable to the
// gap that caused it without the policy knowing what a log is.
inline std::string describe(const Plan& outcome, const Demand& need,
                            const TierOccupancy& occupancy) {
    const auto reason_name = [](EnqueueReason reason) -> const char* {
        switch (reason) {
            case EnqueueReason::None: return "-";
            case EnqueueReason::DeviceNotClosable: return "device-not-closable";
            case EnqueueReason::HostNotClosable: return "host-not-closable";
        }
        return "?";
    };
    std::string line =
        "[cache] gap dkv=" + std::to_string(need.device_kv) +
        " dstate=" + std::to_string(need.device_state) + " hkv=" + std::to_string(need.host_kv) +
        " hstate=" + std::to_string(need.host_state) + " | free dkv=" +
        std::to_string(occupancy.device_kv_free()) + " dstate=" +
        std::to_string(occupancy.device_state_free()) + " hkv=" +
        std::to_string(occupancy.host_kv_free()) + " hstate=" +
        std::to_string(occupancy.host_state_free()) + " | steps=" +
        std::to_string(outcome.steps.size());
    if (outcome.enqueue) {
        line += " enqueue reason=";
        line += reason_name(outcome.reason);
    }
    for (const Step& step : outcome.steps) {
        line += step.action == Action::SpillToHost ? "\n[cache]   spill  id="
                                                   : "\n[cache]   drop   id=";
        line += std::to_string(step.id) + " kv=" + std::to_string(step.kv) +
                " state=" + std::to_string(step.state) +
                " importance=" + std::to_string(step.importance);
    }
    return line;
}

} // namespace ninfer::runtime::cache
