// Which state images the cache ladder may reclaim, as a standalone decision.
//
// 缓存模块v2.md §2.1 / §三 R0: 正在执行的对话是工作集,其余任何时刻都可以删,唯一的要求是挑出最没
// 价值的那个. In code that means a state image is a LIVE BINDING - untouchable by the release ladder -
// only while the conversation that references it is BEING EXECUTED, or while this Program's in-flight
// reservation is about to write it. Every other reference, including the turn closure of a FINISHED
// conversation, is cache: the value ordering decides that it is sacrificed last (it is that
// conversation's dearest component), never that it can never be sacrificed.
//
// Why this is a separate header rather than a predicate buried in ProgramImplCore (2026-09-27): the
// rule is a decision table, and the bug it fixes was invisible until it was measured - an IDLE Engine
// parked for its whole admission deadline because the Host state pool was filled with the turn
// closures of FINISHED conversations, the R2 component step accepts only HostOnly replicas, and the
// lossless EvictHostReplica step refused every closure because the predicate bound the closure of
// every non-Free row, executed or not. Here the table is driven directly by
// tests/test_resource_manager.cpp `test_finished_conversations_images_are_reclaimable` in
// milliseconds, instead of waiting on a 9-minute rig run or a 30-minute production battery.

#pragma once

#include <cstdint>

namespace ninfer::targets::qwen3_6::detail::state_reclaim {

enum class StateOwnerUse : std::uint8_t {
    InflightDestination,  // this Program's in-flight reservation will write it
    ExecutingActiveState, // read/write state of the conversation being executed
    ExecutingClosure,     // turn closure of the conversation being executed
    ExecutingAnchor,      // fork point of the conversation being executed
    IdleClosure,          // turn closure of a FINISHED conversation: cache
    IdleActiveState,      // read/write state of a FINISHED conversation: cache
    IdleAnchor,           // fork point of a FINISHED conversation: cache
};

[[nodiscard]] constexpr bool state_is_live_binding(StateOwnerUse use) noexcept {
    switch (use) {
    case StateOwnerUse::InflightDestination:
    case StateOwnerUse::ExecutingActiveState:
    case StateOwnerUse::ExecutingClosure:
    case StateOwnerUse::ExecutingAnchor:
        return true;
    case StateOwnerUse::IdleClosure:
    case StateOwnerUse::IdleActiveState:
    case StateOwnerUse::IdleAnchor:
        return false;
    }
    return true;
}

// 缓存模块v2.md §四 invariant 1: **显存侧不存在删除**。A whole-owner release destroys every component
// the owner still holds, so it may only run where the Device footprint is already zero - every
// cache-policy release must have moved that data to Host first (§三 R1, 先搬后释). The exceptions are
// the two releases that are NOT cache decisions: the client consuming its handle (the conversation
// ends by definition, §六.5 ownership return) and the process-exit teardown.
//
// This is ENFORCED, not measured. The `[invariant1]` probe made a violation visible only after it had
// already destroyed data, and only in the configurations a battery happened to run: the 2026-09-27
// pressure cases destroyed 13 Device state images through `site=transaction-victim` in a 4-slot Host
// state pool while the rig's 48-slot pool never reproduced it. Now the release asks this table first
// and refuses instead of destroying, so the invariant holds for every input, not for the inputs that
// were tested. The table is driven by tests/test_resource_manager.cpp in milliseconds.
enum class ReleaseIntent : std::uint8_t {
    PolicyMaterializationVictim,   // a pressure plan evicted a private owner to make room
    PolicyCaptureVictim,           // a capture's pressure plan evicted a private owner
    PolicySharedPressureVictim,    // a pressure plan evicted a catalogued shared prefix
    PolicyCaptureSharedPressureVictim,
    PolicyCaptureReplacement,      // a capture published over a catalogued shared prefix
    PolicyLadderHostOnly,          // the capacity ladder's Host-side steps
    OwnershipHandleRelease,        // the client consumed the handle: the conversation is over
    ShutdownTeardown,              // the Program is going away
};

// The log spelling of each intent (kept identical to the names the batteries grep for).
[[nodiscard]] constexpr const char* release_intent_name(ReleaseIntent intent) noexcept {
    switch (intent) {
    case ReleaseIntent::PolicyMaterializationVictim: return "transaction-victim";
    case ReleaseIntent::PolicyCaptureVictim: return "capture-victim";
    case ReleaseIntent::PolicySharedPressureVictim: return "shared-pressure-victim";
    case ReleaseIntent::PolicyCaptureSharedPressureVictim: return "capture-shared-pressure-victim";
    case ReleaseIntent::PolicyCaptureReplacement: return "capture-replacement";
    case ReleaseIntent::PolicyLadderHostOnly: return "ladder-host-only";
    case ReleaseIntent::OwnershipHandleRelease: return "handle-release";
    case ReleaseIntent::ShutdownTeardown: return "shutdown";
    }
    return "?";
}

// A cache-policy release may never destroy Device data; the two non-cache releases may.
[[nodiscard]] constexpr bool release_may_destroy_device(ReleaseIntent intent) noexcept {
    switch (intent) {
    case ReleaseIntent::PolicyMaterializationVictim:
    case ReleaseIntent::PolicyCaptureVictim:
    case ReleaseIntent::PolicySharedPressureVictim:
    case ReleaseIntent::PolicyCaptureSharedPressureVictim:
    case ReleaseIntent::PolicyCaptureReplacement:
    case ReleaseIntent::PolicyLadderHostOnly:
        return false;
    case ReleaseIntent::OwnershipHandleRelease:
    case ReleaseIntent::ShutdownTeardown:
        return true;
    }
    return false;
}

// The one question every strict release asks before it destroys anything.
[[nodiscard]] constexpr bool release_admits_device_destruction(ReleaseIntent intent,
                                                               bool holds_device_data) noexcept {
    return !holds_device_data || release_may_destroy_device(intent);
}

} // namespace ninfer::targets::qwen3_6::detail::state_reclaim
