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

} // namespace ninfer::targets::qwen3_6::detail::state_reclaim
