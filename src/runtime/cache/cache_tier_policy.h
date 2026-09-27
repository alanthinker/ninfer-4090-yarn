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
    std::uint64_t device_kv_used       = 0;   // MAIN KV pool
    std::uint64_t device_kv_capacity   = 0;
    // The speculative BACKEND KV pool is a SEPARATE pool the feasibility gate checks on its own
    // (physical_peak_fits checks backend used+peak against backend capacity). A plan that closes
    // main+backend as one sum while this pool stays pinned is rejected at assessment - battery
    // 2026-09-26: controller target Infeasible with backend_kv 10288+66/10288 while main was
    // fully closed by 489 pages of relief.
    std::uint64_t device_backend_kv_used    = 0;
    std::uint64_t device_backend_kv_capacity = 0;
    std::uint64_t device_state_used    = 0;
    std::uint64_t device_state_capacity = 0;
    std::uint64_t host_kv_used         = 0;
    std::uint64_t host_kv_capacity     = 0;
    std::uint64_t host_state_used      = 0;
    std::uint64_t host_state_capacity  = 0;
    // The fifth pool: catalog rows a NEW conversation may publish into. A row is an index, not
    // cache data - but it is a real admission resource: with none vacant the only way to serve a
    // fresh conversation is to release an existing one (its row comes with it).
    std::uint64_t catalog_rows_vacant  = 0;

    [[nodiscard]] std::uint64_t device_kv_free() const noexcept {
        return device_kv_capacity > device_kv_used ? device_kv_capacity - device_kv_used : 0;
    }
    [[nodiscard]] std::uint64_t device_backend_kv_free() const noexcept {
        return device_backend_kv_capacity > device_backend_kv_used
                   ? device_backend_kv_capacity - device_backend_kv_used
                   : 0;
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
    std::uint64_t device_kv         = 0;  // MAIN pool pages
    std::uint64_t device_backend_kv = 0;  // BACKEND pool pages: never interchangeable with main
    std::uint64_t device_state = 0;
    std::uint64_t host_kv      = 0;
    std::uint64_t host_state   = 0;
    // 1 when the incoming conversation has nowhere to publish itself (no vacant row and no
    // source row it may consume), else 0.
    std::uint64_t catalog_rows  = 0;
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
    std::uint64_t device_kv    = 0;  // MAIN pages a SPILL of this owner takes off Device
    std::uint64_t device_backend_kv = 0;  // BACKEND pages (the separate pool)
    std::uint64_t device_state = 0;  // its Device state images, in Device state slots
    // Device relief a whole-conversation RELEASE delivers (the eviction option's full
    // footprint). Distinct from the three fields above: the release loop credits THESE (a drop
    // evicts everything it holds), the spill loop credits the move attribution above (a spill
    // moves what the option's per-store actions actually touch).
    std::uint64_t evict_device_kv      = 0;
    std::uint64_t evict_device_backend_kv = 0;
    std::uint64_t evict_device_state   = 0;
    // JOINT-FREEABLE Device pages: shared with other owners (fork/copy-on-write,
    // address_references > 1) but EVERY referent is inside the droppable candidate set - so
    // releasing those referents together frees these pages physically, exactly once each. The
    // program fills this (it owns the address spaces); a zero here means the page is shared
    // with someone we may NOT release (active/protected) and cannot be promised at all.
    // owner_exclusive_resources is0 for pure-shared shells - evicting one frees nothing - which
    // is why these pages need their own field and their own pass (rig soak: cand=20 steps=0).
    std::uint64_t joint_device_kv      = 0;
    std::uint64_t joint_device_backend_kv = 0;
    std::uint64_t host_kv      = 0;  // its Host KV, in bytes
    std::uint64_t host_state   = 0;  // its Host state images, in Host state slots
    // The catalog row this conversation occupies (1 for every pooled owner - the pool IS the
    // catalog). Releasing the conversation frees the row with it.
    std::uint64_t catalog_row   = 1;
    std::uint64_t importance   = 0;  // larger == more worth keeping. The ONLY ordering.
    // Last-active age key (ASCENDING: larger == more recent) — the tie-break of §2.2's one chain
    // `value → age → id`, the same key the ladder's retire_preference ranks with (§2.2).
    std::int64_t age_key = 0;
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
    CatalogNotClosable, // no vacant row and no pool owner could be released to make one
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

// §2.2's one chain: `value → age → id`. Ascending importance (least worth keeping first), then
// ascending age (oldest last-touched first), then id so the plan is reproducible for a snapshot —
// the exact chain the ladder's retire_preference ranks with (`cache_owner_rank_less`), so the two
// consumers of `importance` cannot pick different victims on a tie (§2.2).
inline void rank_least_important(std::vector<const Datum*>& pool) {
    std::sort(pool.begin(), pool.end(), [](const Datum* left, const Datum* right) {
        if (left->importance != right->importance) {
            return left->importance < right->importance;
        }
        if (left->age_key != right->age_key) { return left->age_key < right->age_key; }
        return left->id < right->id;
    });
}

} // namespace detail

// Decide how to make room for `need` given `occupancy` and the current cache pool.
//
// The order is fixed by the rules, not by cost: R2 before R1, because Host has to have somewhere
// to receive what Device is about to hand over. A conversation may receive BOTH actions — its
// Host copy is released first and its Device copy lands in the room that release just opened —
// which is one net action (§2.1), not two: Device data is never destroyed here.
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
    const std::uint64_t device_backend_gap =
        shortfall(need.device_backend_kv, occupancy.device_backend_kv_free());
    const std::uint64_t device_state_gap =
        shortfall(need.device_state, occupancy.device_state_free());
    const std::uint64_t host_kv_gap    = shortfall(need.host_kv, occupancy.host_kv_free());
    const std::uint64_t host_state_gap = shortfall(need.host_state, occupancy.host_state_free());
    const std::uint64_t rows_gap = shortfall(need.catalog_rows, occupancy.catalog_rows_vacant);

    if (device_kv_gap == 0 && device_backend_gap == 0 && device_state_gap == 0 &&
        host_kv_gap == 0 && host_state_gap == 0 && rows_gap == 0) {
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

    // R2 + the catalog-row gap: release whole conversations, least important first, until BOTH
    // Host pools have their room AND the row demand is met. One released conversation answers
    // every axis at once: its row, its Host KV/state - and, at execution, its Device data too
    // (the victim prelude moves Device pages to Host before the teardown, §三 R1/R2). Its
    // Device relief is counted HERE so the R1 loop below never needs a second step for the
    // same conversation: HostReleases runs before CopyPreparation, so a spilled address space
    // that was already released would fail prepare and latch the engine. A conversation is
    // only taken when it answers a gap that is still open (§ invariant 3); the last one may
    // still free more than the gap, because a conversation cannot be dropped halfway.
    // Spill first, release only when the spill route does not exist: a Device gap with ANY
    // move relief in the pool must be answered by moving (never by deleting Host cache - the
    // contract case_device_short_host_roomy asserts exactly that), but a Device pool whose
    // pages NO option may touch (orphan tails beyond every retained prefix) has no move relief
    // at all - only then does a whole release shed Device pages (rig req34: Device free=0,
    // Host12.7 GB free, zero move options, steps=0, device-not-closable forever).
    // Per axis, not all-or-nothing: a pool that can MOVE main KV may still have nobody able
    // to MOVE Device state - gating every axis on one flag blocked the only remaining route
    // (soak: mv_state=0, ev_state=8, dstate gap1 -> nothing could ever close it and the whole
    // plan enqueued). Each axis falls back to release exactly when THAT axis has no mover.
    const bool spill_capable_kv =
        std::any_of(candidates.begin(), candidates.end(), [](const Datum* datum) {
            return datum->device_kv > 0 || datum->device_backend_kv > 0;
        });
    const bool spill_capable_state =
        std::any_of(candidates.begin(), candidates.end(),
                    [](const Datum* datum) { return datum->device_state > 0; });
    std::vector<std::uint64_t> dropped;
    dropped.reserve(candidates.size());
    std::uint64_t host_kv_freed        = 0;
    std::uint64_t host_state_freed     = 0;
    std::uint64_t rows_freed             = 0;
    std::uint64_t released_device_kv      = 0;
    std::uint64_t released_device_backend = 0;
    for (const Datum* datum : candidates) {
        // Device STATE is deliberately absent here: a release cannot deliver it (see below), so the
        // loop must not keep walking for it. The R1 loop and the simulation verdict own that axis.
        if (host_kv_freed >= host_kv_gap && host_state_freed >= host_state_gap &&
            rows_freed >= rows_gap &&
            released_device_kv >= device_kv_gap &&
            released_device_backend >= device_backend_gap) {
            break;
        }
        const bool helps_kv =
            host_kv_freed < host_kv_gap && datum->host_kv > 0;
        const bool helps_state =
            host_state_freed < host_state_gap && datum->host_state > 0;
        const bool helps_rows = rows_freed < rows_gap && datum->catalog_row > 0;
        // A PURE Device KV gap also enters here: when the Device pool is full of pages no spill
        // may touch (orphan tails beyond every retained prefix) while Host has room, only a
        // whole-conversation release sheds Device pages - the spill loop below cannot. The
        // prelude moves what is moveable, the tail goes with its owner (rig req34: Device
        // free=0 with12.7 GB Host free, steps=0, device-not-closable).
        //
        // A Device STATE gap is different and is NOT answered here: `evict_device_state` is the
        // footprint a whole-conversation teardown would DESTROY, and that teardown only runs after
        // the move-first prelude has taken the owner's Device data to Host - a DeviceOnly state with
        // no Host slot to land on cannot be moved at all (缓存模块v2.md §三 R1: 显存里的数据只有两种
        // 归宿, 正在用, 或搬到内存). Crediting it answered a Device-state gap by evicting a
        // conversation whose state had nowhere to go, and 13 `[invariant1] strict
        // site=transaction-victim ... state=N slots` destructions followed in one pressure-suite run
        // (2026-09-27, 4-slot Host state pools). The R1 loop below answers the gap through
        // `datum.device_state`, which the Program fills only when the demotion can actually land
        // (possibly after an R2 degrade); when nothing can move, the plan enqueues (R0).
        const bool helps_device_kv =
            !spill_capable_kv && released_device_kv < device_kv_gap &&
            (datum->evict_device_kv > 0 || datum->device_kv > 0);
        const bool helps_device_backend =
            !spill_capable_kv && released_device_backend < device_backend_gap &&
            (datum->evict_device_backend_kv > 0 || datum->device_backend_kv > 0);
        if (!helps_kv && !helps_state && !helps_rows && !helps_device_kv &&
            !helps_device_backend) {
            continue;
        }
        dropped.push_back(datum->id);
        host_kv_freed += datum->host_kv;
        host_state_freed += datum->host_state;
        rows_freed += datum->catalog_row;
        // The RELEASE delivers the eviction option's full footprint; fall back to the move
        // attribution when no eviction footprint was recorded (the drop subsumes the move
        // anyway - the prelude moves its pages first, then the release frees them all).
        released_device_kv += datum->evict_device_kv > datum->device_kv
                                  ? datum->evict_device_kv
                                  : datum->device_kv;
        released_device_backend += datum->evict_device_backend_kv > datum->device_backend_kv
                                       ? datum->evict_device_backend_kv
                                       : datum->device_backend_kv;
    }

    // R1: move Device-resident SURVIVORS to Host, least important first, until the Device gaps
    // are closed - after subtracting the Device relief the release loop above already
    // delivers. A released conversation never receives a second step: its address space is
    // gone by the time the spill phase would run, so a spill step for it would fail prepare
    // and latch the engine (plan and execution must agree on one action per conversation).
    // Shell pass: owners whose ONLY estate is joint-freeable shared pages (exclusive=0, no
    // move, no Host) would never be picked by the main loop - 'evicting it frees nothing' -
    // even though releasing their WHOLE referent group frees the shared physical pages. Drop
    // every such shell still standing (their joint pages are by construction fully inside the
    // candidate set, and a shared page only returns when its LAST referent goes), then credit
    // the pages ONCE - the maximum single owner's joint claim is a sound lower bound for one
    // group and never double-counts a page listed by both referents.
    if (device_kv_gap > released_device_kv || device_backend_gap > released_device_backend) {
        std::uint64_t joint_main_max = 0, joint_bkv_max = 0;
        std::size_t shells = 0;
        for (const Datum* datum : candidates) {
            if (std::find(dropped.begin(), dropped.end(), datum->id) != dropped.end()) {
                continue;
            }
            // A TRUE shell: no exclusive estate, no move option (spill-first still wins when
            // a move exists), no Host release - its pages exist only jointly.
            const bool shell =
                datum->evict_device_kv == 0 && datum->evict_device_state == 0 &&
                datum->device_kv == 0 && datum->device_state == 0 && datum->host_kv == 0 &&
                (datum->joint_device_kv > 0 || datum->joint_device_backend_kv > 0);
            if (!shell) { continue; }
            if (device_kv_gap <= released_device_kv &&
                device_backend_gap <= released_device_backend) {
                break;
            }
            dropped.push_back(datum->id);
            ++shells;
            host_kv_freed += datum->host_kv;
            host_state_freed += datum->host_state;
            rows_freed += datum->catalog_row;
            joint_main_max = std::max(joint_main_max, datum->joint_device_kv);
            joint_bkv_max = std::max(joint_bkv_max, datum->joint_device_backend_kv);
        }
        if (shells > 0) {
            released_device_kv += std::min(joint_main_max, device_kv_gap - std::min(released_device_kv, device_kv_gap));
            released_device_backend += std::min(joint_bkv_max, device_backend_gap - std::min(released_device_backend, device_backend_gap));
        }
    }

    const std::uint64_t device_kv_needed =
        device_kv_gap > released_device_kv ? device_kv_gap - released_device_kv : 0;
    const std::uint64_t device_backend_needed =
        device_backend_gap > released_device_backend
            ? device_backend_gap - released_device_backend
            : 0;
    // Device STATE is answered by the R1 loop alone: a release never delivers it, so the whole gap
    // stands until a state is moved (or the simulation below enqueues the request).
    const std::uint64_t device_state_needed = device_state_gap;
    std::vector<Step> spills;
    spills.reserve(candidates.size());
    std::uint64_t device_kv_moved      = 0;
    std::uint64_t device_backend_moved = 0;
    std::uint64_t device_state_moved   = 0;
    for (const Datum* datum : candidates) {
        if (std::find(dropped.begin(), dropped.end(), datum->id) != dropped.end()) { continue; }
        if (device_kv_moved >= device_kv_needed &&
            device_backend_moved >= device_backend_needed &&
            device_state_moved >= device_state_needed) {
            break;
        }
        // Per-pool: a victim counts only toward the pool it actually relieves. Main-heavy owners
        // cannot close a Backend gap however large their main relief is - and the assessment
        // checks the pools separately, so the plan must too.
        const bool helps_kv =
            device_kv_moved < device_kv_needed && datum->device_kv > 0;
        const bool helps_backend = device_backend_moved < device_backend_needed &&
                                   datum->device_backend_kv > 0;
        const bool helps_state =
            device_state_moved < device_state_needed && datum->device_state > 0;
        if (!helps_kv && !helps_backend && !helps_state) { continue; }
        spills.push_back(Step{datum->id, Action::SpillToHost, datum->device_kv,
                              datum->device_state, datum->importance});
        device_kv_moved += datum->device_kv;
        device_backend_moved += datum->device_backend_kv;
        device_state_moved += datum->device_state;
    }

    // Simulation verdict. Anything short means the plan cannot do what it promised, so it is not
    // applied at all - nothing is deleted, nothing is moved, the caller waits instead (R0).
    const bool host_short =
        host_kv_freed < host_kv_gap || host_state_freed < host_state_gap;
    const bool device_short = device_kv_moved < device_kv_needed ||
                              device_backend_moved < device_backend_needed ||
                              device_state_moved < device_state_needed;
    const bool rows_short = rows_freed < rows_gap;
    if (device_short || host_short || rows_short) {
        out.enqueue = true;
        out.reason  = device_short   ? EnqueueReason::DeviceNotClosable
                     : host_short    ? EnqueueReason::HostNotClosable
                                     : EnqueueReason::CatalogNotClosable;
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
                            const TierOccupancy& occupancy,
                            std::size_t candidates = 0) {
    const auto reason_name = [](EnqueueReason reason) -> const char* {
        switch (reason) {
            case EnqueueReason::None: return "-";
            case EnqueueReason::DeviceNotClosable: return "device-not-closable";
            case EnqueueReason::HostNotClosable: return "host-not-closable";
            case EnqueueReason::CatalogNotClosable: return "catalog-not-closable";
        }
        return "?";
    };
    std::string line =
        "[cache] gap dkv=" + std::to_string(need.device_kv) +
        " dbkv=" + std::to_string(need.device_backend_kv) +
        " dstate=" + std::to_string(need.device_state) + " hkv=" + std::to_string(need.host_kv) +
        " hstate=" + std::to_string(need.host_state) + " rows=" +
        std::to_string(need.catalog_rows) + " | free dkv=" +
        std::to_string(occupancy.device_kv_free()) + " dbkv=" +
        std::to_string(occupancy.device_backend_kv_free()) + " dstate=" +
        std::to_string(occupancy.device_state_free()) + " hkv=" +
        std::to_string(occupancy.host_kv_free()) + " hstate=" +
        std::to_string(occupancy.host_state_free()) + " rows=" +
        std::to_string(occupancy.catalog_rows_vacant) + " | cand=" +
        std::to_string(candidates) + " steps=" +
        std::to_string(outcome.steps.size()) +
        " | gaps dkv=" +
        std::to_string(need.device_kv > occupancy.device_kv_free()
                           ? need.device_kv - occupancy.device_kv_free()
                           : 0) +
        " dstate=" +
        std::to_string(need.device_state > occupancy.device_state_free()
                           ? need.device_state - occupancy.device_state_free()
                           : 0) +
        " hkv=" +
        std::to_string(need.host_kv > occupancy.host_kv_free()
                           ? need.host_kv - occupancy.host_kv_free()
                           : 0) +
        " hstate=" +
        std::to_string(need.host_state > occupancy.host_state_free()
                           ? need.host_state - occupancy.host_state_free()
                           : 0);
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
