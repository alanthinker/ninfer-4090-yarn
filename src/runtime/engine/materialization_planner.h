#pragma once

#include "runtime/engine/context_cost.h"
#include "runtime/engine/context_portfolio_value.h"
#include "runtime/engine/resource_search.h"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <tuple>
#include <utility>
#include <vector>

namespace ninfer::runtime {

struct MaterializationCheckpointPolicy {
    PlanningOwnerId owner;
    CheckpointRef checkpoint;
    RetentionClass retention_class     = RetentionClass::RecentPrivate;
    std::uint64_t selected_hit_count   = 0;
    std::uint64_t last_hit_epoch       = 0;
    std::uint32_t demand_mask          = 0;
    std::uint64_t rebuild_ns           = 0;
    std::uint64_t baseline_recovery_ns = 0;
};

struct MaterializationOwnerPolicy {
    PlanningOwnerId owner;
    RetentionClass retention_class         = RetentionClass::RecentPrivate;
    std::uint64_t selected_hit_count       = 0;
    std::uint64_t last_hit_epoch           = 0;
    std::uint32_t private_retention_weight = 0;
    bool explicit_shared_credit            = false;
    // Reuse evidence for this owner in Q16: how often this conversation has been reused as a base,
    // decayed by how long ago that last happened, with a floor so a session that has not been
    // re-read yet is cheap rather than worthless. Owner-scoped on purpose: a growing conversation
    // replaces its own checkpoint every turn, so a per-checkpoint count cannot represent "this
    // conversation has been reused fifty times".
    std::uint64_t reuse_evidence_q16 = 1U << 16U;
};

// Reuse evidence in Q16: a floored, saturating hyperbola over the age-decayed lifetime reuse
// count. It answers "how likely is this owner to be read again" and multiplies the value at risk
// in the victim score. The floor keeps a session that has not been re-read yet cheap but never
// free - without it "never reused" sorts as the cheapest owner, so a freshly built deep
// conversation outranks every session a test loop has been hammering, no matter how much prefill
// it holds. The decay is wall-clock, so evidence fades for a session nobody touches any more.
[[nodiscard]] inline std::uint64_t reuse_evidence_q16(
    std::uint64_t hits, std::chrono::steady_clock::time_point last,
    std::chrono::steady_clock::time_point now) noexcept {
    constexpr std::uint64_t kOne    = 1U << 16U;
    constexpr std::uint64_t kFloor  = kOne / 8U; // 0.125 for a never-reused owner
    constexpr std::uint64_t kHalf   = 8U * kOne; // 8 decayed reuses reach half saturation
    constexpr std::uint64_t kHitCap = 4096U;     // bounds the fixed-point arithmetic
    constexpr double kTauSeconds    = 1800.0;    // 30 minutes
    std::uint64_t decayed_q16       = 0;
    if (hits != 0 && last.time_since_epoch().count() > 0) {
        const double age   = std::max(0.0, std::chrono::duration<double>(now - last).count());
        const double decay = std::exp2(-age / kTauSeconds);
        const std::uint64_t decay_q16 =
            static_cast<std::uint64_t>(decay * static_cast<double>(kOne));
        const std::uint64_t capped = std::min(hits, kHitCap);
        decayed_q16                = ((capped << 16U) * decay_q16) >> 16U;
    }
    const std::uint64_t ratio_q16 = (decayed_q16 << 16U) / (decayed_q16 + kHalf);
    return kFloor + (((kOne - kFloor) * ratio_q16) >> 16U);
}

// One owner's victim cost, kept in parts because the diagnostic prints them: `value_ns` is the
// value at risk before the reuse-evidence multiplier, and `priced_checkpoints` is how many of the
// owner's checkpoints could be priced at all. An owner whose Program handle is already gone prices
// nothing, so its value is zero - it is not "cheap", it simply has nothing left to give, and its
// rank must not be read as a decision the planner could have made differently.
struct MaterializationVictimCost {
    std::uint64_t value_ns           = 0;
    std::uint64_t score              = 0;
    std::size_t priced_checkpoints   = 0;
};

// The two factors of the victim score, split out so the Program's last-resort release step can be
// ordered by the very same arithmetic when the common layer prices the owners for it (it owns the
// cost model and the observation history; the ladder does not).
[[nodiscard]] inline std::uint64_t victim_value_ns(std::uint32_t retention_weight,
                                                   std::uint64_t private_saving,
                                                   std::span<const std::uint64_t> demand_values,
                                                   bool explicit_shared_credit) noexcept {
    const auto saturating_add = [](std::uint64_t& value, std::uint64_t add) {
        value = value > std::numeric_limits<std::uint64_t>::max() - add
                    ? std::numeric_limits<std::uint64_t>::max()
                    : value + add;
    };
    const auto saturating_mul = [](std::uint64_t value, std::uint64_t factor) {
        if (value == 0 || factor == 0) { return std::uint64_t{0}; }
        return value > std::numeric_limits<std::uint64_t>::max() / factor
                   ? std::numeric_limits<std::uint64_t>::max()
                   : value * factor;
    };
    std::uint64_t value_ns = saturating_mul(private_saving, retention_weight);
    for (const std::uint64_t demand_value : demand_values) { saturating_add(value_ns, demand_value); }
    if (explicit_shared_credit) { saturating_add(value_ns, private_saving); }
    return value_ns;
}

[[nodiscard]] inline std::uint64_t victim_score_ns(std::uint64_t value_ns,
                                                   std::uint64_t reuse_evidence_q16) noexcept {
    if (value_ns == 0 || reuse_evidence_q16 == 0) { return 0; }
    return value_ns > std::numeric_limits<std::uint64_t>::max() / reuse_evidence_q16
               ? std::numeric_limits<std::uint64_t>::max()
               : (value_ns * reuse_evidence_q16) >> 16U;
}

// Victim ordering is one combined score in the planner's own currency, not a lexicographic field
// chain. A field-first chain lets a single dimension decide alone - when the reuse count differs,
// the retention weight and the recency are never read at all - which is how a deep endpoint that
// had not been re-read yet lost to shallow probe sessions a test loop had just hammered, even
// though the economic model already priced that endpoint at tens of seconds of prefill. The score
// multiplies the value at risk (the largest rebuild saving the owner would lose, scaled by its
// retention weight, plus the demand-observed public value) by the reuse evidence, so depth,
// retention class, demand, recency, and reuse history contribute to one comparable number.
[[nodiscard]] inline MaterializationVictimCost materialization_victim_cost(
    std::span<const MaterializationOwnerPolicy> owners,
    std::span<const MaterializationCheckpointPolicy> checkpoints, PlanningOwnerId owner) noexcept {
    const MaterializationOwnerPolicy* policy = nullptr;
    for (const MaterializationOwnerPolicy& candidate : owners) {
        if (candidate.owner == owner) {
            policy = &candidate;
            break;
        }
    }
    if (policy == nullptr) { return {}; }

    std::uint64_t private_saving = 0;
    std::size_t priced           = 0;
    std::array<std::uint64_t, 32> demand_best{};
    for (const MaterializationCheckpointPolicy& checkpoint : checkpoints) {
        if (checkpoint.owner != owner) { continue; }
        ++priced;
        const std::uint64_t saving = checkpoint.rebuild_ns > checkpoint.baseline_recovery_ns
                                         ? checkpoint.rebuild_ns - checkpoint.baseline_recovery_ns
                                         : 0;
        private_saving = std::max(private_saving, saving);
        for (std::uint32_t bit = 0; bit < demand_best.size(); ++bit) {
            if ((checkpoint.demand_mask & (1U << bit)) == 0) { continue; }
            demand_best[bit] = std::max(demand_best[bit], saving);
        }
    }

    std::uint64_t value_ns =
        victim_value_ns(policy->private_retention_weight, private_saving, demand_best,
                       policy->explicit_shared_credit);
    return MaterializationVictimCost{
        .value_ns         = value_ns,
        .score            = victim_score_ns(value_ns, policy->reuse_evidence_q16),
        .priced_checkpoints = priced,
    };
}

[[nodiscard]] inline std::uint64_t materialization_victim_score(
    std::span<const MaterializationOwnerPolicy> owners,
    std::span<const MaterializationCheckpointPolicy> checkpoints, PlanningOwnerId owner) noexcept {
    return materialization_victim_cost(owners, checkpoints, owner).score;
}

template <class Package>
class MaterializationPlanner {
public:
    using Program                = typename Package::Program;
    using PreparedPrompt         = typename Package::PreparedPrompt;
    using AdmissionCandidate     = typename Package::AdmissionCandidate;
    using ResourcePlan           = typename Package::ResourcePlan;
    using ContinuationHandle     = typename Package::ContinuationHandle;
    using SharedPrefixHandle     = typename Package::SharedPrefixHandle;
    using PressureTargetHandle   = typename Package::PressureTargetHandle;
    using AssessedPressureTarget = typename Package::AssessedPressureTarget;
    using Clock                  = std::chrono::steady_clock;

    struct CandidateInput {
        AdmissionCandidate* candidate = nullptr;
        PlanningCandidateId id;
        std::uint32_t stable_ordinal = 0;
        bool current_session_binding = false;
    };

    /**
     * The most reuse any candidate offered to the planner. Candidates are built upstream,
     * before planning, so this is what reuse was ON THE TABLE - independent of what the
     * search then did with it.
     *
     * The first version of this hooked assess_target instead and always read 0: those
     * targets are eviction alternatives, not reuse candidates, and targets_evaluated
     * already starts at candidates.size() precisely because the candidates were scored
     * elsewhere. A positive control caught it - a request that demonstrably reused 40,051
     * tokens still reported 0.
     */
    [[nodiscard]] static const char*
    materialization_rejection_name(runtime::MaterializationRejection reason) noexcept {
        switch (reason) {
        case runtime::MaterializationRejection::None: return "";
        case runtime::MaterializationRejection::ContextBusy: return "context busy";
        case runtime::MaterializationRejection::HostAllocationBlocked: return "host allocation";
        case runtime::MaterializationRejection::PhysicalPeak: return "physical peak";
        case runtime::MaterializationRejection::SourceUnavailable: return "source unavailable";
        case runtime::MaterializationRejection::DestinationStale: return "destination stale";
        case runtime::MaterializationRejection::VictimStale: return "victim stale";
        case runtime::MaterializationRejection::InvariantFailure: return "invariant";
        }
        return "";
    }

    [[nodiscard]] static std::uint32_t
    best_offered_reuse(std::span<const CandidateInput> candidates) noexcept {
        std::uint32_t best = 0;
        for (const CandidateInput& input : candidates) {
            if (input.candidate == nullptr) { continue; }
            best = std::max(best, input.candidate->summary().reusable_prompt_tokens);
        }
        return best;
    }

    struct LogicalGoal {
        std::uint32_t publication_slot = std::numeric_limits<std::uint32_t>::max();
    };

    struct PressureInputs {
        std::span<const ContinuationHandle* const> private_owners;
        std::span<const PlanningOwnerId> private_owner_ids;
        std::span<const SharedPrefixHandle* const> shared_owners;
        std::span<const PlanningOwnerId> shared_owner_ids;
        std::span<const MaterializationOwnerPolicy> owner_policy;
        std::span<const MaterializationCheckpointPolicy> checkpoint_policy;
    };

    struct Result {
        std::optional<ResourcePlan> plan;
        PlanningCandidateId candidate;
        std::uint32_t publication_slot = std::numeric_limits<std::uint32_t>::max();
        PrivateSourceMode source_mode  = PrivateSourceMode::ConsumeToActive;
        std::vector<PressureOwnerOutcome> owner_outcomes;
        std::vector<PressureCheckpointOutcome> checkpoint_outcomes;
        MaterializationDiagnostics diagnostics;
    };

    MaterializationPlanner() : target_ledger_(kTargetBudget + 17U) {
        queue_.reserve(kTargetBudget);
        pending_.reserve(kTargetBudget);
        guided_.reserve(kTargetBudget);
        identity_costs_.reserve(16);
        candidate_guided_steps_.reserve(16);
        candidate_seed_complete_.reserve(16);
        impact_scratch_.reserve(32);
        portfolio_owner_scratch_.reserve(32);
        portfolio_checkpoint_scratch_.reserve(64);
    }

    // `negative_sound` (nullable) reports, on a nullopt result, whether the failure PROVED the
    // request infeasible (the full victim domain has no plan) rather than the search giving up
    // early. Only a sound verdict may be memoized by a caller: it is stable while the pool
    // state it was planned against holds.
    template <class PressureInputsFn, class LogicalGoalFn, class FinalScheduleFn>
    [[nodiscard]] std::optional<Result>
    plan(Program& program, const PreparedPrompt& prompt,
         const ContextMachineCostModel& machine_cost, std::span<const CandidateInput> candidates,
         std::uint32_t root_candidate_index, PressureInputsFn&& pressure_inputs,
         LogicalGoalFn&& logical_goal, FinalScheduleFn&& final_schedule,
         std::uint32_t prompt_tokens, Clock::time_point planning_started,
         bool* negative_sound = nullptr) {
        if (candidates.empty() || root_candidate_index >= candidates.size()) {
            throw std::invalid_argument("materialization planning problem has no root candidate");
        }
        for (std::size_t index = 0; index < candidates.size(); ++index) {
            if (candidates[index].candidate == nullptr ||
                std::find_if(candidates.begin(), candidates.begin() + index,
                             [&](const CandidateInput& prior) {
                                 return prior.id == candidates[index].id;
                             }) != candidates.begin() + index) {
                throw std::invalid_argument("materialization candidate IDs are invalid");
            }
        }
        queue_.clear();
        pending_.clear();
        guided_.clear();
        const std::size_t frontier_capacity = candidates.size() + 1U + kTargetBudget;
        queue_.reserve(frontier_capacity);
        pending_.reserve(frontier_capacity);
        guided_.reserve(frontier_capacity);
        identity_costs_.clear();
        candidate_guided_steps_.assign(candidates.size(), 0);
        candidate_seed_complete_.assign(candidates.size(), false);
        target_ledger_.reset(candidates.size() + 1U + kTargetBudget);

        // Whether the most reusable candidate is feasible at all is the difference between "the
        // planner priced reuse worse" and "reuse was never physically available"; the two need
        // different fixes, and the plan alone cannot tell them apart.
        bool best_reuse_feasible = false;
        MaterializationPhysicalStatus best_reuse_physical_status =
            MaterializationPhysicalStatus::StructuralInvalid;
        runtime::MaterializationRejection best_reuse_rejection = runtime::MaterializationRejection::None;
        std::string best_reuse_rejection_detail;
        std::optional<Incumbent> identity_best;
        std::vector<IdentityRoot> roots;
        roots.reserve(candidates.size());
        std::uint64_t projection_work = 0;
        for (std::size_t index = 0; index < candidates.size(); ++index) {
            const CandidateInput& input = candidates[index];
            const IdentityMaterializationAssessment& identity =
                input.candidate->identity_assessment();
            planning_saturating_add(projection_work, identity.projection_work);
            const FoldedCost cost = fold_identity(input, identity, machine_cost);
            identity_costs_.push_back(cost);
            std::optional<LogicalGoal> goal;
            if (identity.physical_status == MaterializationPhysicalStatus::Feasible) {
                goal = logical_goal(input.id, identity.source_mode,
                                    std::span<const PressureOwnerOutcome>{});
            }
            if (input.candidate != nullptr &&
                input.candidate->summary().reusable_prompt_tokens != 0 &&
                input.candidate->summary().reusable_prompt_tokens ==
                    best_offered_reuse(candidates)) {
                best_reuse_physical_status = identity.physical_status;
                best_reuse_feasible =
                    identity.physical_status == MaterializationPhysicalStatus::Feasible;
                best_reuse_rejection = input.candidate->identity_rejection();
                best_reuse_rejection_detail = input.candidate->identity_rejection_detail();
            }
            if (goal && (!identity_best || cost.less(identity_best->cost))) {
                identity_best = Incumbent{
                    .candidate_index  = static_cast<std::uint32_t>(index),
                    .publication_slot = goal->publication_slot,
                    .source_mode      = identity.source_mode,
                    .cost             = cost,
                };
            }
            if (goal) { candidate_seed_complete_[index] = true; }
            const bool needs_pressure =
                !goal.has_value() &&
                (identity.physical_status == MaterializationPhysicalStatus::Feasible ||
                 identity.expandable);
            const bool pressure_can_improve =
                goal.has_value() &&
                identity.physical_status == MaterializationPhysicalStatus::Feasible &&
                identity.pressure_may_change_machine_work;
            roots.push_back(IdentityRoot{
                .candidate_index = static_cast<std::uint32_t>(index),
                .lower_bound_ns  = cost.lower_bound_ns,
                .expandable      = needs_pressure || pressure_can_improve,
            });
        }

        if (identity_best) {
            const bool needs_optional_search =
                std::any_of(roots.begin(), roots.end(),
                            [](const IdentityRoot& root) { return root.expandable; });
            if (!needs_optional_search) {
                const CandidateInput& selected = candidates[identity_best->candidate_index];
                const auto price_split         = [&](std::span<const std::uint32_t> frontiers) {
                    const std::uint64_t baseline =
                        machine_cost.prefill_ns(selected.candidate->identity_assessment()
                                                            .machine_work.remaining_prefill_work);
                    const std::uint64_t target =
                        machine_cost.prefill_ns(program.shared_capture_split_prefill_work(
                            *selected.candidate, prompt, frontiers));
                    return target > baseline ? target - baseline : 0;
                };
                std::vector<std::uint32_t> shared_frontiers =
                    final_schedule(selected.id, selected.candidate->summary(), price_split);
                std::optional<ResourcePlan> sealed = program.seal_identity(
                    *selected.candidate, prompt,
                    FinalScheduleIntent{.shared_capture_frontiers = shared_frontiers});
                if (!sealed) {
                    if (negative_sound) { *negative_sound = false; }
                    return std::nullopt;
                }
                MaterializationDiagnostics diagnostics = complete_diagnostics(
                    identity_best->cost, static_cast<std::uint32_t>(candidates.size()),
                    projection_work, planning_started, MaterializationStopReason::NoPressure,
                    false);
                diagnostics.candidates = static_cast<std::uint32_t>(candidates.size());
                diagnostics.best_reuse_prompt_tokens = best_offered_reuse(candidates);
        diagnostics.best_reuse_feasible       = best_reuse_feasible;
        diagnostics.best_reuse_rejection =
            std::string(materialization_rejection_name(best_reuse_rejection)) + " status=" +
            std::to_string(static_cast<int>(best_reuse_physical_status)) + " rej=" +
            std::to_string(static_cast<int>(best_reuse_rejection)) + " " +
            best_reuse_rejection_detail;
                diagnostics.best_reuse_feasible       = best_reuse_feasible;
                diagnostics.best_reuse_rejection =
                    std::string(materialization_rejection_name(best_reuse_rejection)) +
                    " status=" +
                    std::to_string(static_cast<int>(best_reuse_physical_status)) + " rej=" +
                    std::to_string(static_cast<int>(best_reuse_rejection)) + " " +
                    best_reuse_rejection_detail;
                Result result;
                result.plan             = std::move(*sealed);
                result.candidate        = candidates[identity_best->candidate_index].id;
                result.publication_slot = identity_best->publication_slot;
                result.source_mode      = identity_best->source_mode;
                result.diagnostics      = diagnostics;
                return result;
            }
        }

        std::vector<const AdmissionCandidate*> candidate_handles;
        std::vector<PlanningCandidateId> candidate_ids;
        candidate_handles.reserve(candidates.size());
        candidate_ids.reserve(candidates.size());
        for (const CandidateInput& input : candidates) {
            candidate_handles.push_back(input.candidate);
            candidate_ids.push_back(input.id);
        }
        const PressureInputs pressure = pressure_inputs();
        if (pressure.private_owners.size() != pressure.private_owner_ids.size() ||
            pressure.shared_owners.size() != pressure.shared_owner_ids.size()) {
            throw std::logic_error("materialization pressure owner arrays are not aligned");
        }
        auto session = program.begin_pressure_planning(
            candidate_handles, candidate_ids, pressure.private_owners, pressure.private_owner_ids,
            pressure.shared_owners, pressure.shared_owner_ids);
        const auto candidate_index_for = [&](PlanningCandidateId id) -> std::uint32_t {
            const auto found =
                std::find_if(candidates.begin(), candidates.end(),
                             [&](const CandidateInput& input) { return input.id == id; });
            if (found == candidates.end()) {
                throw std::logic_error("pressure target references an unknown candidate ID");
            }
            return static_cast<std::uint32_t>(found - candidates.begin());
        };

        Incumbent incumbent;
        std::uint32_t targets_evaluated = static_cast<std::uint32_t>(candidates.size());
        if (identity_best) {
            incumbent        = std::move(*identity_best);
            incumbent.target = session.identity_target(candidates[incumbent.candidate_index].id);
        } else {
            PressureTargetHandle root_maximal =
                session.root_maximal_target(candidates[root_candidate_index].id);
            AssessedPressureTarget assessed            = session.assess(root_maximal);
            const PressureTargetAssessment& assessment = assessed.assessment();
            if (assessment.candidate != candidates[root_candidate_index].id) {
                throw std::logic_error("maximal pressure target changed admission candidate");
            }
            ++targets_evaluated;
            planning_saturating_add(projection_work, assessment.projection_work);
            std::optional<LogicalGoal> goal;
            if (assessment.physical_status == MaterializationPhysicalStatus::Feasible) {
                goal = logical_goal(assessment.candidate, assessment.source_mode,
                                    assessment.owner_outcomes);
            }
            if (!goal) {
                // The root-maximal target is infeasible or cannot publish: neither any owner,
                // checkpoint, nor capture can free the capacity, so the complete victim domain
                // has no plan. Sound to memoize while the pool state is unchanged.
                if (negative_sound) { *negative_sound = true; }
                return std::nullopt;
            }
            const FoldedCost cost =
                fold_assessment(candidates[root_candidate_index], assessment, pressure.owner_policy,
                                pressure.checkpoint_policy, machine_cost);
            incumbent = make_incumbent(root_maximal, root_candidate_index, assessment,
                                       std::move(assessed), cost, *goal);
            mark_target(assessment.stable_target_ordinal, kTargetDiscovered | kTargetAssessed);
        }

        const Clock::time_point search_started = Clock::now();
        // The budget bounds admission-time planning work. The relative term binds ordinary
        // plans; the cap only has to leave room for a plan whose reuse is worth minutes of
        // prefill. The floor and the per-candidate floor guarantee that every candidate's
        // retention closure is seeded even on machines whose exact target assessment is slow
        // relative to the prefill cost model: the pre-scale budget (5 ms, later
        // incumbent-cost/20) starved the 2026-09-11 579k-token production case of every
        // assessment, and on 2026-09-12 a 201k-token prompt with 60+ indexed candidates
        // stopped on TimeBudget after a fraction of them because the budget scaled with
        // incumbent prefill cost only, which this machine prices far below its actual
        // assessment time. Planning time is charged against the prefill it is trying to
        // avoid, so a budget worth a fraction of that recompute is a favorable trade; the
        // cap bounds the worst case at two minutes of planning latency.
        // All budget terms are nanoseconds (the elapsed comparison below and the diagnostics
        // display both assume ns). An earlier revision wrote the per-candidate/floor/cap
        // constants in microseconds, silently capping the effective budget at 90 ms and
        // reproducing the 2026-09-12 time_budget failures on 100k+ prompts (observed
        // 2026-09-13: 115 candidates, spurious fair-share release, 0% reuse).
        const std::uint64_t per_candidate_assessment_ns = 500'000'000ULL;
        const std::uint64_t candidate_floor_ns = candidates.size() * per_candidate_assessment_ns;
        // The planning floor scales with this request's prompt size instead of the historical
        // flat 15 s. That flat floor exists so a large prompt (100k+ tokens, many indexed
        // candidates) gets enough window to seed every candidate's retention closure - the
        // 2026-09-12 regression was a budget that starved those assessments. But the same flat
        // 15 s forced a 14-token cold prompt to spend 15-28 s of host-CPU planning to avoid a
        // prefill that costs a fraction of a second: once the trivially-eligible candidates were
        // pruned there was nothing left for the search to find. So keep the full 15 s floor for
        // prompts at/above kPlanningFloorFullTokens (bit-identical to the previous behavior for
        // the cases that actually need it) and taper it toward a small constant for tiny
        // prompts, whose prefill a 15 s search could never be worth avoiding. Larger prompts
        // keep their window through the incumbent-cost/20 term, which grows with prompt size.
        // The root-maximal fallback and any feasible reuse incumbent are seeded before the
        // budgeted search, so a smaller floor only trims eviction refinement - it can never turn
        // a reuse hit into a miss or a runnable request into a blocked one.
        constexpr std::uint64_t kPlanningFloorMinNs      = 250'000'000ULL;    // 0.25 s
        constexpr std::uint64_t kPlanningFloorMaxNs      = 15'000'000'000ULL; // 15 s
        constexpr std::uint64_t kPlanningFloorFullTokens = 8192U;
        // 10-bit fixed-point taper: proportional below the full floor, saturating at 15 s.
        const std::uint64_t size_floor_ns =
            kPlanningFloorMinNs +
            (kPlanningFloorMaxNs - kPlanningFloorMinNs) *
                (std::min<std::uint64_t>(prompt_tokens, kPlanningFloorFullTokens) *
                 1024U /
                kPlanningFloorFullTokens) /
                1024U;
        // Cap the planning budget by the cost of the root prefill it is trying to avoid.
        // A 45-token prompt costs ~150 ms to prefill; spending 12.5 s of host-CPU
        // searching for a reuse hit that saves 45 tokens is never worth it. For large
        // prompts (100k+ tokens) the root cost is hundreds of seconds, so the cap is
        // far above the 90 s hard limit and has no effect.
        const std::uint64_t root_prefill_ns =
            machine_cost.prefill_ns(PrefillWork{.chunks = 1, .tokens = prompt_tokens});
        const std::uint64_t root_cost_cap_ns =
            root_prefill_ns > UINT64_MAX / 10U ? UINT64_MAX : root_prefill_ns * 10U;
        const std::uint64_t floors_max = std::max<std::uint64_t>(
            size_floor_ns, std::max(incumbent.cost.total_ns / 20U, candidate_floor_ns));
        const std::uint64_t capped_by_root = std::min(root_cost_cap_ns, floors_max);
        const std::uint64_t search_budget_ns =
            std::min<std::uint64_t>(90'000'000'000ULL, capped_by_root);
        // The directed pass seeds each candidate's own retention closure, one candidate per
        // iteration, so its window scales per candidate at the same rate; after it spends the
        // window, the best-first phase evaluates whatever remains within the full budget.
        // Capping it at the search budget keeps the pass from turning into a breadth-first
        // walk of every alternative on the way (a 7-owner pressure case once assessed more
        // than twice the targets its closure needed).
        const std::uint64_t guided_watchdog_ns =
            std::min(search_budget_ns,
                     std::max<std::uint64_t>(5'000'000ULL, candidate_floor_ns));
        // Assessing one pressure target costs on the order of a millisecond of host CPU (measured:
        // 3,968 targets in 5.36 s on 2026-09-21), so the arena is also bounded by what the time at
        // stake can pay for at a 20x leverage: refining a plan whose saving is a fraction of a
        // second is not worth thousands of exact assessments. kTargetBudget stays the hard ceiling.
        constexpr std::uint64_t kTargetAssessmentCostNs = 1'000'000ULL;  // ~1 ms
        // The floor has to cover each candidate's guided closure plus the bounded seed probe
        // (kSeedProbeSteps) that turns a seed into an early acceptance; starving that phase made
        // the search stop on the target budget before the probe could accept (2026-09-21 unit test).
        constexpr std::uint32_t kMinimumTargetBudget    = 256U;
        const std::uint64_t at_stake_ns =
            incumbent.cost.total_ns > incumbent.cost.lower_bound_ns
                ? incumbent.cost.total_ns - incumbent.cost.lower_bound_ns
                : 0;
        const std::uint64_t affordable_targets =
            at_stake_ns / (20ULL * kTargetAssessmentCostNs);
        const std::uint32_t effective_target_budget = static_cast<std::uint32_t>(
            std::min<std::uint64_t>(kTargetBudget,
                                    std::max<std::uint64_t>(kMinimumTargetBudget,
                                                            affordable_targets)));
        std::uint64_t maximum_step_ns          = 0;
        std::uint32_t optional_targets         = 0;
        std::uint32_t guided_assessments       = 0;
        std::uint32_t expansions               = 0;
        std::uint32_t fanned_out_children_max  = 0;
        std::uint32_t guided_closures_ok       = 0;
        std::uint32_t guided_closures_failed   = 0;
        MaterializationStopReason stop_reason  = MaterializationStopReason::QueueExhausted;
        bool budget_exhausted                  = false;

        for (const IdentityRoot& root : roots) {
            if (!root.expandable) { continue; }
            QueueEntry entry;
            entry.target            = session.identity_target(candidates[root.candidate_index].id);
            entry.candidate_index   = root.candidate_index;
            entry.lower_bound_ns    = root.lower_bound_ns;
            entry.remaining_prefill = identity_costs_[root.candidate_index].remaining_text_prefill;
            entry.remaining_vision_prefill =
                identity_costs_[root.candidate_index].remaining_vision_prefill;
            entry.reused_prompt_tokens = identity_costs_[root.candidate_index].reused_prompt_tokens;
            entry.current_session_binding =
                identity_costs_[root.candidate_index].current_session_binding;
            entry.candidate_ordinal     = identity_costs_[root.candidate_index].candidate_ordinal;
            entry.stable_target_ordinal = root.candidate_index;
            mark_target(entry.stable_target_ordinal, kTargetDiscovered | kTargetAssessed);
            queue_push(entry);
            const PressureTargetGuidance guidance = session.guidance(entry.target);
            if (guidance.candidate != candidates[root.candidate_index].id) {
                throw std::logic_error("pressure guidance changed admission candidate");
            }
            guided_insert(GuidedEntry{
                .target          = entry.target,
                .candidate_index = root.candidate_index,
                .lower_bound_ns  = root.lower_bound_ns,
                .guidance        = fold_guidance(candidates[root.candidate_index], guidance,
                                                 pressure.owner_policy, machine_cost),
                .already_assessed_expandable = true,
            });
        }

        const auto make_queue_entry = [](PressureTargetHandle target, std::uint32_t candidate_index,
                                         const PressureTargetAssessment& assessment,
                                         const FoldedCost& cost) {
            return QueueEntry{
                .target                    = target,
                .candidate_index           = candidate_index,
                .lower_bound_ns            = cost.lower_bound_ns,
                .affected_selected_hits    = cost.affected_selected_hits,
                .newest_affected_hit_epoch = cost.newest_affected_hit_epoch,
                .owner_evictions           = cost.owner_evictions,
                .checkpoint_drops          = cost.checkpoint_drops,
                .copy_operations           = cost.copy_operations,
                .transferred_bytes         = cost.transferred_bytes,
                .remaining_prefill         = cost.remaining_text_prefill,
                .remaining_vision_prefill  = cost.remaining_vision_prefill,
                .reused_prompt_tokens      = cost.reused_prompt_tokens,
                .current_session_binding   = cost.current_session_binding,
                .candidate_ordinal         = cost.candidate_ordinal,
                .stable_target_ordinal     = assessment.stable_target_ordinal,
            };
        };

        const auto assess_target =
            [&](PressureTargetHandle target, std::uint32_t expected_candidate,
                std::uint32_t expected_ordinal,
                bool from_guided_closure = false) -> std::optional<QueueEntry> {
            AssessedPressureTarget assessed            = session.assess(target);
            const PressureTargetAssessment& assessment = assessed.assessment();
            if (assessment.candidate != candidates[expected_candidate].id ||
                candidate_index_for(assessment.candidate) != expected_candidate ||
                assessment.stable_target_ordinal != expected_ordinal) {
                throw std::logic_error("pressure target changed admission candidate");
            }
            mark_target(assessment.stable_target_ordinal, kTargetDiscovered | kTargetAssessed);
            ++targets_evaluated;
            planning_saturating_add(projection_work, assessment.projection_work);
            const FoldedCost cost =
                fold_assessment(candidates[expected_candidate], assessment, pressure.owner_policy,
                                pressure.checkpoint_policy, machine_cost);
            std::optional<LogicalGoal> goal;
            if (assessment.physical_status == MaterializationPhysicalStatus::Feasible) {
                goal = logical_goal(assessment.candidate, assessment.source_mode,
                                    assessment.owner_outcomes);
            }
            if (goal && !assessment.root_maximal) {
                candidate_seed_complete_[expected_candidate] = true;
            }
            if (goal && cost.less(incumbent.cost)) {
                incumbent = make_incumbent(target, expected_candidate, assessment,
                                           std::move(assessed), cost, *goal);
                incumbent.guided_closure = from_guided_closure;
            }
            if (!assessment.expandable) { return std::nullopt; }
            QueueEntry entry = make_queue_entry(target, expected_candidate, assessment, cost);
            queue_push(entry);
            return entry;
        };

        const auto expand_target = [&](const QueueEntry& parent) {
            if (target_marked(parent.stable_target_ordinal, kTargetExpanded)) { return true; }
            // The first expansion is always allowed. It is the minimum unit of search, and a
            // request whose guided closure failed must still be able to look at one neighbourhood;
            // otherwise it would fall back to the maximal (release-everything) plan, which destroys
            // other sessions' cache -- the failure mode the 2026-09-21 retention collapse showed.
            // One expansion costs ~300 assessments (~0.3 s), bounded and rare.
            if (expansions != 0 && optional_targets >= effective_target_budget) { return false; }
            auto prepared = session.prepare_expansion(parent.target);
            if (expansions != 0 &&
                prepared.new_canonical_count() > effective_target_budget - optional_targets) {
                session.discard_expansion(std::move(prepared));
                return false;
            }
            const auto children = session.commit_expansion(std::move(prepared));
            optional_targets += children.new_canonical_count;
            ++expansions;
            fanned_out_children_max = std::max(fanned_out_children_max, children.new_canonical_count);
            mark_target(parent.stable_target_ordinal, kTargetExpanded);
            for (const PressureTargetHandle child : children.children) {
                const PressureTargetGuidance guidance = session.guidance(child);
                if (guidance.candidate != candidates[parent.candidate_index].id) {
                    throw std::logic_error("pressure guidance changed admission candidate");
                }
                const std::uint32_t candidate_index = candidate_index_for(guidance.candidate);
                if (target_marked(guidance.stable_target_ordinal, kTargetDiscovered)) { continue; }
                mark_target(guidance.stable_target_ordinal, kTargetDiscovered);
                const std::uint64_t lower_bound_ns = std::max(
                    identity_costs_[candidate_index].lower_bound_ns, parent.lower_bound_ns);
                const GuidanceCost cost = fold_guidance(candidates[candidate_index], guidance,
                                                        pressure.owner_policy, machine_cost);
                PendingEntry pending{
                    .target          = child,
                    .candidate_index = candidate_index,
                    .lower_bound_ns  = lower_bound_ns,
                    .guidance        = cost,
                };
                pending_push(pending);
                if (!candidate_seed_complete_[candidate_index]) {
                    guided_insert(GuidedEntry{
                        .target          = child,
                        .candidate_index = candidate_index,
                        .lower_bound_ns  = lower_bound_ns,
                        .guidance        = cost,
                    });
                }
            }
            return true;
        };

        const auto candidate_needs_seed = [&](std::uint32_t candidate_index) {
            return candidate_index < roots.size() && roots[candidate_index].expandable &&
                   !candidate_seed_complete_[candidate_index];
        };
        const auto has_open_seed = [&] {
            return std::any_of(roots.begin(), roots.end(), [&](const IdentityRoot& root) {
                return candidate_needs_seed(root.candidate_index);
            });
        };

        // The Program walks this list and sacrifices the first owner that can pay for the
        // deficit, so the order is the victim decision. Rank by the combined score, cheapest
        // first, and keep the owner id only as a deterministic tie-break.
        struct PreferredOwner {
            const MaterializationOwnerPolicy* policy = nullptr;
            std::uint64_t victim_cost                = 0;
        };
        std::vector<PreferredOwner> preferred;
        preferred.reserve(pressure.owner_policy.size());
        for (const MaterializationOwnerPolicy& policy : pressure.owner_policy) {
            preferred.push_back(PreferredOwner{
                .policy = &policy,
                .victim_cost =
                    materialization_victim_score(pressure.owner_policy, pressure.checkpoint_policy,
                                                 policy.owner),
            });
        }
        // Value first; among owners the score cannot separate, the least recently reused one goes
        // first (never-hit owners carry epoch 0). Replacing the old lexicographic order dropped
        // recency from this decision entirely, which made a freshly published conversation as
        // attractive a victim as an hour-old one of the same value.
        std::sort(preferred.begin(), preferred.end(),
                  [](const PreferredOwner& left, const PreferredOwner& right) {
                      if (left.victim_cost != right.victim_cost) {
                          return left.victim_cost < right.victim_cost;
                      }
                      if (left.policy->last_hit_epoch != right.policy->last_hit_epoch) {
                          return left.policy->last_hit_epoch < right.policy->last_hit_epoch;
                      }
                      return left.policy->owner.value < right.policy->owner.value;
                  });
        std::vector<PlanningOwnerId> preferred_owner_ids;
        preferred_owner_ids.reserve(preferred.size());
        for (const PreferredOwner& entry : preferred) {
            preferred_owner_ids.push_back(entry.policy->owner);
        }

        std::vector<IdentityRoot> closure_order;
        closure_order.reserve(roots.size());
        for (const IdentityRoot& root : roots) {
            if (candidate_needs_seed(root.candidate_index)) { closure_order.push_back(root); }
        }
        std::sort(closure_order.begin(), closure_order.end(),
                  [](const IdentityRoot& left, const IdentityRoot& right) {
                      return std::tuple{left.lower_bound_ns, left.candidate_index} <
                             std::tuple{right.lower_bound_ns, right.candidate_index};
                  });
        for (const IdentityRoot& root : closure_order) {
            if (!candidate_needs_seed(root.candidate_index) ||
                elapsed_ns(search_started, Clock::now()) >= guided_watchdog_ns ||
                optional_targets >= effective_target_budget) {
                continue;
            }
            const Clock::time_point step_started              = Clock::now();
            const std::optional<PressureTargetHandle> closure = session.guided_closure_target(
                candidates[root.candidate_index].id, preferred_owner_ids);
            maximum_step_ns = std::max(maximum_step_ns, elapsed_ns(step_started, Clock::now()));
            if (!closure) {
                // A failed closure leaves this candidate unseeded for the whole search: it is
                // never retried, so its only remaining route is a full expansion of its identity
                // root. Count it instead of losing the fact.
                ++guided_closures_failed;
                continue;
            }
            const PressureTargetGuidance closure_guidance = session.guidance(*closure);
            if (closure_guidance.candidate != candidates[root.candidate_index].id) {
                throw std::logic_error("guided closure changed admission candidate");
            }
            if (target_marked(closure_guidance.stable_target_ordinal, kTargetAssessed)) {
                continue;
            }
            if (!target_marked(closure_guidance.stable_target_ordinal, kTargetDiscovered)) {
                mark_target(closure_guidance.stable_target_ordinal, kTargetDiscovered);
                ++optional_targets;
            }
            const Clock::time_point assessment_started = Clock::now();
            (void)assess_target(*closure, root.candidate_index,
                                closure_guidance.stable_target_ordinal,
                                /*from_guided_closure=*/true);
            ++guided_assessments;
            ++guided_closures_ok;
            maximum_step_ns =
                std::max(maximum_step_ns, elapsed_ns(assessment_started, Clock::now()));
        }

        // Build one ordinary feasible seed per expandable candidate. Estimated machine cost orders
        // independent beams but never excludes a candidate or certifies an incumbent.
        while (has_open_seed() && !guided_.empty() &&
               guided_assessments < kGuidedAssessmentBudget) {
            if (elapsed_ns(search_started, Clock::now()) >= guided_watchdog_ns) { break; }
            const GuidedEntry next = guided_pop();
            if (!candidate_needs_seed(next.candidate_index) ||
                target_marked(next.guidance.stable_target_ordinal, kTargetExpanded)) {
                continue;
            }
            std::optional<QueueEntry> exact;
            if (next.already_assessed_expandable) {
                exact = QueueEntry{
                    .target          = next.target,
                    .candidate_index = next.candidate_index,
                    .lower_bound_ns  = next.lower_bound_ns,
                    .remaining_prefill =
                        identity_costs_[next.candidate_index].remaining_text_prefill,
                    .remaining_vision_prefill =
                        identity_costs_[next.candidate_index].remaining_vision_prefill,
                    .reused_prompt_tokens =
                        identity_costs_[next.candidate_index].reused_prompt_tokens,
                    .current_session_binding =
                        identity_costs_[next.candidate_index].current_session_binding,
                    .candidate_ordinal = identity_costs_[next.candidate_index].candidate_ordinal,
                    .stable_target_ordinal = next.guidance.stable_target_ordinal,
                };
            } else if (!target_marked(next.guidance.stable_target_ordinal, kTargetAssessed)) {
                const Clock::time_point step_started = Clock::now();
                exact = assess_target(next.target, next.candidate_index,
                                      next.guidance.stable_target_ordinal);
                ++guided_assessments;
                maximum_step_ns = std::max(maximum_step_ns, elapsed_ns(step_started, Clock::now()));
            }
            if (!exact || !candidate_needs_seed(next.candidate_index)) { continue; }
            if (elapsed_ns(search_started, Clock::now()) >= guided_watchdog_ns) { break; }
            const Clock::time_point step_started = Clock::now();
            if (!expand_target(*exact)) { break; }
            maximum_step_ns = std::max(maximum_step_ns, elapsed_ns(step_started, Clock::now()));
        }

        // Value-of-computation acceptance.  The seeded incumbent is the result of the scoring
        // pass (guidance ordering: coldest, least reused, lowest retention weight first)
        // verified by one exact assessment.  On a large candidate frontier the budgeted
        // refinement is what makes admission cost multi-second host time while the best plan
        // almost never moves off the seed.  The loop therefore spends a small bounded probe
        // of refinements first: if the seeded plan is feasible, preserves every owner (no
        // eviction), costs less than a quarter of the search budget, and none of the probe's
        // evaluations beats it, it is accepted.  Any improvement disables the acceptance and
        // the search continues under its normal stopping rules; on a small frontier the queue
        // simply exhausts before the probe completes and the behavior is unchanged.  Plans
        // that evict any owner never take this path: eviction is the destructive decision
        // class and must keep the guarantee that every preserving alternative is explored.
        constexpr std::uint64_t kSeedProbeSteps = 64U;
        bool seed_acceptable                    = false;
        if (!incumbent.root_maximal && incumbent.assessed) {
            const bool seed_has_no_eviction = std::none_of(
                incumbent.owner_outcomes.begin(), incumbent.owner_outcomes.end(),
                [](const PressureOwnerOutcome& outcome) {
                    return outcome.disposition == VictimDisposition::Evicted;
                });
            // A guided-closure seed already applied victims in the planner's value order (coldest,
            // least reused, lowest retention weight first), so its release set is the greedy
            // minimal one. Accept it after the bounded probe instead of refining for seconds: the
            // 2026-09-21 production case spent 5.36 s of a 6.66 s TTFT evaluating 3,968 targets
            // and settled on a three-unit degradation anyway, while a request that had already hit
            // 89% of its prefix waited in the queue.
            constexpr std::uint32_t kGuidedSeedMaxEvictions = 8U;
            if (seed_has_no_eviction) {
                seed_acceptable = incumbent.cost.total_ns < search_budget_ns / 4U;
            } else if (incumbent.guided_closure &&
                       incumbent.cost.owner_evictions <= kGuidedSeedMaxEvictions) {
                seed_acceptable = incumbent.cost.total_ns < search_budget_ns;
            } else if (session.retention_infeasible(candidates[incumbent.candidate_index].id)) {
                // Certified: even maximal retention pressure cannot close the device
                // shortfall, so an evicting seed is a safe acceptance candidate after the
                // probe (value of computation: the refinement gain is bounded by the seed's
                // own total cost, which must not pay for the whole remaining budget).
                seed_acceptable = incumbent.cost.total_ns < search_budget_ns;
            }
        }
        const std::uint64_t seed_total_ns       = seed_acceptable ? incumbent.cost.total_ns : 0;
        const std::uint32_t seed_candidate      = seed_acceptable
                                                      ? incumbent.candidate_index
                                                      : static_cast<std::uint32_t>(-1);
        std::uint64_t seed_probe_steps          = 0;

        for (;;) {
            while (!queue_.empty() &&
                   target_marked(queue_.front().stable_target_ordinal, kTargetExpanded)) {
                (void)queue_pop();
            }
            while (
                !pending_.empty() &&
                target_marked(pending_.front().guidance.stable_target_ordinal, kTargetAssessed)) {
                (void)pending_pop();
            }
            if (queue_.empty() && pending_.empty()) {
                stop_reason = MaterializationStopReason::QueueExhausted;
                break;
            }
            const std::uint64_t queue_bound   = queue_.empty()
                                                    ? std::numeric_limits<std::uint64_t>::max()
                                                    : queue_.front().lower_bound_ns;
            const std::uint64_t pending_bound = pending_.empty()
                                                    ? std::numeric_limits<std::uint64_t>::max()
                                                    : pending_.front().lower_bound_ns;
            const std::uint64_t next_bound    = std::min(queue_bound, pending_bound);
            const std::uint64_t elapsed       = elapsed_ns(search_started, Clock::now());
            if (elapsed >= search_budget_ns) {
                stop_reason      = MaterializationStopReason::TimeBudget;
                budget_exhausted = true;
                break;
            }
            // A* dominance: if the cheapest remaining node's lower bound is >= the incumbent's
            // total cost, no remaining node can improve the solution. Stop immediately.
            if (next_bound >= incumbent.cost.total_ns) {
                stop_reason = MaterializationStopReason::QueueExhausted;
                break;
            }
            const std::uint64_t possible_improvement =
                incumbent.cost.total_ns > next_bound ? incumbent.cost.total_ns - next_bound : 0;
            if (possible_improvement != 0 && maximum_step_ns != 0 &&
                maximum_step_ns >= possible_improvement) {
                stop_reason = MaterializationStopReason::ValueOfNextExpansion;
                break;
            }

            const bool assess_pending =
                !pending_.empty() && (queue_.empty() || pending_bound <= queue_bound);
            const Clock::time_point step_started = Clock::now();
            if (assess_pending) {
                const PendingEntry next = pending_pop();
                if (!target_marked(next.guidance.stable_target_ordinal, kTargetAssessed)) {
                    (void)assess_target(next.target, next.candidate_index,
                                        next.guidance.stable_target_ordinal);
                }
            } else {
                if (optional_targets >= effective_target_budget) {
                    stop_reason      = MaterializationStopReason::TargetBudget;
                    budget_exhausted = true;
                    break;
                }
                const QueueEntry parent = queue_pop();
                if (!expand_target(parent)) {
                    stop_reason      = MaterializationStopReason::ExpansionCapacity;
                    budget_exhausted = true;
                    break;
                }
            }
            maximum_step_ns = std::max(maximum_step_ns, elapsed_ns(step_started, Clock::now()));
            if (seed_acceptable) {
                if (incumbent.cost.total_ns < seed_total_ns ||
                    incumbent.candidate_index != seed_candidate) {
                    // The probe beat the seed: run the ordinary search to its normal stop.
                    seed_acceptable = false;
                } else if (++seed_probe_steps >= kSeedProbeSteps) {
                    // Heuristic stop like ValueOfNextExpansion: the budget was not exhausted.
                    stop_reason = MaterializationStopReason::SeedAccepted;
                    break;
                }
            }
        }

        const std::uint64_t search_elapsed_ns = elapsed_ns(search_started, Clock::now());
        if (!incumbent.assessed) {
            AssessedPressureTarget assessed            = session.assess(incumbent.target);
            const PressureTargetAssessment& assessment = assessed.assessment();
            if (assessment.candidate != candidates[incumbent.candidate_index].id ||
                assessment.physical_status != MaterializationPhysicalStatus::Feasible) {
                throw std::logic_error("selected identity target lost exact feasibility");
            }
            incumbent.assessed.emplace(std::move(assessed));
        }
        const CandidateInput& selected = candidates[incumbent.candidate_index];
        const auto price_split         = [&](std::span<const std::uint32_t> frontiers) {
            const std::uint64_t baseline = machine_cost.prefill_ns(
                incumbent.assessed->assessment().machine_work.remaining_prefill_work);
            const std::uint64_t target = machine_cost.prefill_ns(
                session.shared_capture_split_prefill_work(*incumbent.assessed, prompt, frontiers));
            return target > baseline ? target - baseline : 0;
        };
        std::vector<std::uint32_t> shared_frontiers =
            final_schedule(selected.id, selected.candidate->summary(), price_split);
        std::optional<ResourcePlan> sealed =
            session.seal(std::move(*incumbent.assessed), prompt,
                         FinalScheduleIntent{.shared_capture_frontiers = shared_frontiers});
        if (!sealed) { throw std::logic_error("selected pressure target could not be sealed"); }

        MaterializationDiagnostics diagnostics = make_diagnostics(
            incumbent.cost, targets_evaluated, projection_work, planning_started, search_elapsed_ns,
            search_budget_ns, stop_reason, budget_exhausted, incumbent.degradation_units,
            incumbent.root_maximal);
        diagnostics.candidates               = static_cast<std::uint32_t>(candidates.size());
        diagnostics.optional_targets         = optional_targets;
        diagnostics.expansions               = expansions;
        diagnostics.fanned_out_children_max  = fanned_out_children_max;
        diagnostics.guided_closures_succeeded = guided_closures_ok;
        diagnostics.guided_closures_failed    = guided_closures_failed;
        diagnostics.best_reuse_prompt_tokens = best_offered_reuse(candidates);
        diagnostics.best_reuse_feasible      = best_reuse_feasible;
        diagnostics.best_reuse_rejection =
            std::string(materialization_rejection_name(best_reuse_rejection)) + " status=" +
            std::to_string(static_cast<int>(best_reuse_physical_status)) + " rej=" +
            std::to_string(static_cast<int>(best_reuse_rejection)) + " " +
            best_reuse_rejection_detail;

        Result result;
        result.plan                = std::move(*sealed);
        result.candidate           = candidates[incumbent.candidate_index].id;
        result.publication_slot    = incumbent.publication_slot;
        result.source_mode         = incumbent.source_mode;
        result.owner_outcomes      = std::move(incumbent.owner_outcomes);
        result.checkpoint_outcomes = std::move(incumbent.checkpoint_outcomes);
        result.diagnostics         = diagnostics;
        return result;
    }

    template <class PressureInputsFn, class LogicalGoalFn>
    [[nodiscard]] std::optional<Result>
    plan(Program& program, const PreparedPrompt& prompt,
         const ContextMachineCostModel& machine_cost, std::span<const CandidateInput> candidates,
         std::uint32_t root_candidate_index, PressureInputsFn&& pressure_inputs,
         LogicalGoalFn&& logical_goal, std::uint32_t prompt_tokens,
          Clock::time_point planning_started) {
        const auto no_optional_schedule = [](PlanningCandidateId, const RequestPlanSummary&,
                                             const auto&) { return std::vector<std::uint32_t>{}; };
        return plan(program, prompt, machine_cost, candidates, root_candidate_index,
                    std::forward<PressureInputsFn>(pressure_inputs),
                    std::forward<LogicalGoalFn>(logical_goal), no_optional_schedule,
                    prompt_tokens, planning_started);
    }

private:
    static constexpr std::uint32_t kTargetBudget           = 4096;
    static constexpr std::uint32_t kGuidedBeamWidth        = 16;
    static constexpr std::uint32_t kGuidedAssessmentBudget = 32;

    struct FoldedCost {
        std::uint64_t now_ns                    = 0;
        std::uint64_t future_loss_ns            = 0;
        std::uint64_t total_ns                  = 0;
        std::uint64_t lower_bound_ns            = 0;
        std::uint64_t affected_selected_hits    = 0;
        std::uint64_t newest_affected_hit_epoch = 0;
        std::uint32_t owner_evictions           = 0;
        std::uint32_t checkpoint_drops          = 0;
        std::uint32_t copy_operations           = 0;
        std::uint64_t transferred_bytes         = 0;
        std::uint64_t remaining_text_prefill    = 0;
        std::uint64_t remaining_vision_prefill  = 0;
        std::uint32_t reused_prompt_tokens      = 0;
        bool current_session_binding            = false;
        std::uint32_t candidate_ordinal         = 0;
        std::uint32_t target_ordinal            = 0;

        [[nodiscard]] auto key() const noexcept {
            return std::tuple{
                total_ns,
                affected_selected_hits,
                newest_affected_hit_epoch,
                owner_evictions,
                checkpoint_drops,
                copy_operations,
                transferred_bytes,
                remaining_text_prefill,
                remaining_vision_prefill,
                std::numeric_limits<std::uint32_t>::max() - reused_prompt_tokens,
                current_session_binding ? 0U : 1U,
                candidate_ordinal,
                target_ordinal,
            };
        }

        [[nodiscard]] bool less(const FoldedCost& other) const noexcept {
            return key() < other.key();
        }
    };

    struct Incumbent {
        PressureTargetHandle target{};
        std::optional<AssessedPressureTarget> assessed;
        std::uint32_t candidate_index  = 0;
        std::uint32_t publication_slot = std::numeric_limits<std::uint32_t>::max();
        PrivateSourceMode source_mode  = PrivateSourceMode::ConsumeToActive;
        FoldedCost cost;
        std::vector<PressureOwnerOutcome> owner_outcomes;
        std::vector<PressureCheckpointOutcome> checkpoint_outcomes;
        std::uint32_t degradation_units = 0;
        bool root_maximal               = false;
        // True when the guided closure produced this incumbent. The closure applies victims in the
        // planner's own value order, so its release set is the greedy minimal one.
        bool guided_closure = false;
    };

    struct IdentityRoot {
        std::uint32_t candidate_index = 0;
        std::uint64_t lower_bound_ns  = 0;
        bool expandable               = false;
    };

    struct QueueEntry {
        PressureTargetHandle target{};
        std::uint32_t candidate_index           = 0;
        std::uint64_t lower_bound_ns            = 0;
        std::uint64_t affected_selected_hits    = 0;
        std::uint64_t newest_affected_hit_epoch = 0;
        std::uint32_t owner_evictions           = 0;
        std::uint32_t checkpoint_drops          = 0;
        std::uint32_t copy_operations           = 0;
        std::uint64_t transferred_bytes         = 0;
        std::uint64_t remaining_prefill         = 0;
        std::uint64_t remaining_vision_prefill  = 0;
        std::uint32_t reused_prompt_tokens      = 0;
        bool current_session_binding            = false;
        std::uint32_t candidate_ordinal         = 0;
        std::uint32_t stable_target_ordinal     = 0;
    };

    struct GuidanceCost {
        std::uint32_t estimated_remaining_steps = 0;
        std::uint32_t unsatisfied_constraints   = 0;
        std::uint64_t normalized_residual_q20   = 0;
        std::uint64_t affected_selected_hits    = 0;
        std::uint64_t newest_affected_hit_epoch = 0;
        std::uint64_t retention_weight          = 0;
        std::uint32_t explicit_shared_losses    = 0;
        std::uint32_t owner_evictions           = 0;
        std::uint32_t checkpoint_drops          = 0;
        std::uint32_t degradation_units         = 0;
        std::uint64_t estimated_immediate_ns    = 0;
        std::uint32_t copy_operations           = 0;
        std::uint64_t transferred_bytes         = 0;
        std::uint64_t remaining_prefill         = 0;
        std::uint64_t remaining_vision_prefill  = 0;
        std::uint32_t reused_prompt_tokens      = 0;
        bool current_session_binding            = false;
        std::uint32_t candidate_ordinal         = 0;
        std::uint32_t stable_target_ordinal     = 0;

        [[nodiscard]] auto key() const noexcept {
            return std::tuple{
                affected_selected_hits,
                explicit_shared_losses,
                owner_evictions,
                checkpoint_drops,
                newest_affected_hit_epoch,
                estimated_remaining_steps,
                unsatisfied_constraints,
                normalized_residual_q20,
                retention_weight,
                degradation_units,
                estimated_immediate_ns,
                copy_operations,
                transferred_bytes,
                remaining_prefill,
                remaining_vision_prefill,
                std::numeric_limits<std::uint32_t>::max() - reused_prompt_tokens,
                current_session_binding ? 0U : 1U,
                candidate_ordinal,
                stable_target_ordinal,
            };
        }
    };

    struct PendingEntry {
        PressureTargetHandle target{};
        std::uint32_t candidate_index = 0;
        std::uint64_t lower_bound_ns  = 0;
        GuidanceCost guidance;
    };

    struct GuidedEntry {
        PressureTargetHandle target{};
        std::uint32_t candidate_index = 0;
        std::uint64_t lower_bound_ns  = 0;
        GuidanceCost guidance;
        bool already_assessed_expandable = false;
    };

    struct CombinedImpact {
        PlanningOwnerId owner;
        CheckpointRef checkpoint;
        std::uint64_t target_ns = 0;
    };

    [[nodiscard]] static std::uint64_t elapsed_ns(Clock::time_point begin,
                                                  Clock::time_point end) noexcept {
        const auto count =
            std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
        return count > 0 ? static_cast<std::uint64_t>(count) : 0;
    }

    static void planning_saturating_add(std::uint64_t& value, std::uint64_t add) noexcept {
        value = add > std::numeric_limits<std::uint64_t>::max() - value
                    ? std::numeric_limits<std::uint64_t>::max()
                    : value + add;
    }

    [[nodiscard]] static const MaterializationOwnerPolicy*
    owner_policy_for(std::span<const MaterializationOwnerPolicy> policies,
                     PlanningOwnerId owner) noexcept {
        const auto found = std::find_if(policies.begin(), policies.end(),
                                        [&](const auto& policy) { return policy.owner == owner; });
        return found == policies.end() ? nullptr : &*found;
    }

    // Indexed owner lookup for the fold's checkpoint loop. The linear `owner_policy_for` made one
    // assessment cost owners x checkpoints, i.e. catalog-proportional planning cost (measured
    // ~1.5 ms per assessment at ~3,800 checkpoint rows on 2026-09-21, 5.7 s for 3,700 targets).
    // The index is rebuilt only when the policy span itself changes, so repeated folds over the
    // same candidate stay O(checkpoints x log owners).
    [[nodiscard]] const MaterializationOwnerPolicy*
    owner_policy_indexed(std::span<const MaterializationOwnerPolicy> policies,
                         PlanningOwnerId owner) const {
        if (owner_policy_index_source_ != policies.data() ||
            owner_policy_index_size_ != policies.size()) {
            owner_policy_index_.clear();
            owner_policy_index_.reserve(policies.size());
            for (const MaterializationOwnerPolicy& policy : policies) {
                owner_policy_index_.push_back(&policy);
            }
            std::sort(owner_policy_index_.begin(), owner_policy_index_.end(),
                      [](const MaterializationOwnerPolicy* left,
                         const MaterializationOwnerPolicy* right) {
                          return left->owner.value < right->owner.value;
                      });
            owner_policy_index_source_ = policies.data();
            owner_policy_index_size_   = policies.size();
        }
        const auto found = std::lower_bound(
            owner_policy_index_.begin(), owner_policy_index_.end(), owner.value,
            [](const MaterializationOwnerPolicy* entry, std::uint32_t value) {
                return entry->owner.value < value;
            });
        if (found == owner_policy_index_.end() || (*found)->owner != owner) { return nullptr; }
        return *found;
    }

    [[nodiscard]] static const MaterializationCheckpointPolicy*
    checkpoint_policy_for(std::span<const MaterializationCheckpointPolicy> policies,
                          PlanningOwnerId owner, CheckpointRef checkpoint) noexcept {
        const auto found = std::find_if(policies.begin(), policies.end(), [&](const auto& policy) {
            return policy.owner == owner && policy.checkpoint == checkpoint;
        });
        return found == policies.end() ? nullptr : &*found;
    }

    [[nodiscard]] FoldedCost
    fold_identity(const CandidateInput& candidate,
                  const IdentityMaterializationAssessment& assessment,
                  const ContextMachineCostModel& machine_cost) const noexcept {
        const PricedMaterializationMachineWork priced =
            price_materialization_machine_work(machine_cost, assessment.machine_work);
        FoldedCost cost;
        cost.now_ns                 = priced.immediate_ns;
        cost.total_ns               = cost.now_ns;
        cost.lower_bound_ns         = priced.optimistic_request_ns;
        cost.copy_operations        = priced.copy_operations;
        cost.transferred_bytes      = priced.transferred_bytes;
        cost.remaining_text_prefill = assessment.machine_work.remaining_prefill_work.tokens;
        cost.remaining_vision_prefill =
            assessment.machine_work.remaining_prefill_work.vision_patches;
        cost.reused_prompt_tokens    = assessment.machine_work.reused_prompt_tokens;
        cost.current_session_binding = candidate.current_session_binding;
        cost.candidate_ordinal       = candidate.stable_ordinal;
        cost.target_ordinal          = candidate.stable_ordinal;
        return cost;
    }

    [[nodiscard]] GuidanceCost
    fold_guidance(const CandidateInput& candidate, const PressureTargetGuidance& guidance,
                  std::span<const MaterializationOwnerPolicy> owner_policies,
                  const ContextMachineCostModel& machine_cost) const {
        const PricedMaterializationMachineWork priced =
            price_materialization_machine_work(machine_cost, guidance.estimated_machine_work);
        GuidanceCost cost;
        cost.estimated_remaining_steps = guidance.physical.estimated_remaining_steps;
        cost.unsatisfied_constraints   = guidance.physical.unsatisfied_constraints;
        cost.normalized_residual_q20   = guidance.physical.normalized_residual_q20;
        cost.checkpoint_drops          = guidance.dropped_checkpoints;
        cost.degradation_units         = guidance.degradation_units;
        cost.estimated_immediate_ns    = priced.immediate_ns;
        cost.copy_operations           = priced.copy_operations;
        cost.transferred_bytes         = priced.transferred_bytes;
        cost.remaining_prefill = guidance.estimated_machine_work.remaining_prefill_work.tokens;
        cost.remaining_vision_prefill =
            guidance.estimated_machine_work.remaining_prefill_work.vision_patches;
        cost.reused_prompt_tokens    = guidance.estimated_machine_work.reused_prompt_tokens;
        cost.current_session_binding = candidate.current_session_binding;
        cost.candidate_ordinal       = candidate.stable_ordinal;
        cost.stable_target_ordinal   = guidance.stable_target_ordinal;

        for (const PressureOwnerOutcome& outcome : guidance.owner_outcomes) {
            const MaterializationOwnerPolicy* policy =
                owner_policy_indexed(owner_policies, outcome.owner);
            if (policy == nullptr) {
                throw std::logic_error("pressure guidance references an unknown logical owner");
            }
            if (outcome.disposition == VictimDisposition::Evicted) { ++cost.owner_evictions; }
            const bool may_reduce_recovery_value =
                outcome.disposition == VictimDisposition::Evicted ||
                outcome.dropped_checkpoints != 0 || outcome.degradation_units != 0;
            if (!may_reduce_recovery_value) { continue; }
            planning_saturating_add(cost.affected_selected_hits, policy->selected_hit_count);
            cost.newest_affected_hit_epoch =
                std::max(cost.newest_affected_hit_epoch, policy->last_hit_epoch);
            planning_saturating_add(cost.retention_weight, policy->private_retention_weight);
            if (policy->explicit_shared_credit) { ++cost.explicit_shared_losses; }
        }
        return cost;
    }

    [[nodiscard]] FoldedCost
    fold_assessment(const CandidateInput& candidate, const PressureTargetAssessment& assessment,
                    std::span<const MaterializationOwnerPolicy> owner_policies,
                    std::span<const MaterializationCheckpointPolicy> checkpoint_policies,
                    const ContextMachineCostModel& machine_cost) {
        const PricedMaterializationMachineWork priced =
            price_materialization_machine_work(machine_cost, assessment.machine_work);
        FoldedCost cost;
        cost.now_ns                 = priced.immediate_ns;
        cost.copy_operations        = priced.copy_operations;
        cost.transferred_bytes      = priced.transferred_bytes;
        cost.remaining_text_prefill = assessment.machine_work.remaining_prefill_work.tokens;
        cost.remaining_vision_prefill =
            assessment.machine_work.remaining_prefill_work.vision_patches;
        cost.reused_prompt_tokens    = assessment.machine_work.reused_prompt_tokens;
        cost.current_session_binding = candidate.current_session_binding;
        cost.candidate_ordinal       = candidate.stable_ordinal;
        cost.target_ordinal          = assessment.stable_target_ordinal;
        cost.checkpoint_drops        = assessment.dropped_checkpoints;

        for (const PressureOwnerOutcome& outcome : assessment.owner_outcomes) {
            const MaterializationOwnerPolicy* policy =
                owner_policy_indexed(owner_policies, outcome.owner);
            if (policy == nullptr) {
                throw std::logic_error("pressure target references an unknown logical owner");
            }
            if (outcome.disposition == VictimDisposition::Evicted) { ++cost.owner_evictions; }
        }

        impact_scratch_.clear();
        for (const PressureCheckpointRecoveryImpact& impact : assessment.checkpoint_impacts) {
            const auto found = std::find_if(
                impact_scratch_.begin(), impact_scratch_.end(), [&](const CombinedImpact& item) {
                    return item.owner == impact.owner && item.checkpoint == impact.checkpoint;
                });
            if (found == impact_scratch_.end()) {
                if (impact.target_recovery_work.empty()) {
                    throw std::logic_error("pressure recovery impact has no supported recipe");
                }
                impact_scratch_.push_back(CombinedImpact{
                    .owner      = impact.owner,
                    .checkpoint = impact.checkpoint,
                    .target_ns =
                        price_checkpoint_recovery_work(machine_cost, impact.target_recovery_work),
                });
            } else {
                throw std::logic_error("pressure recovery impact is duplicated");
            }
        }
        // Keep the impact rows ordered so the checkpoint loop above/below can binary-search them
        // instead of scanning once per row.
        std::sort(impact_scratch_.begin(), impact_scratch_.end(),
                  [](const CombinedImpact& left, const CombinedImpact& right) {
                      return std::tuple{left.owner.value,
                                        static_cast<std::uint8_t>(left.checkpoint.kind),
                                        left.checkpoint.frontier, left.checkpoint.ordinal} <
                             std::tuple{right.owner.value,
                                        static_cast<std::uint8_t>(right.checkpoint.kind),
                                        right.checkpoint.frontier, right.checkpoint.ordinal};
                  });
        portfolio_owner_scratch_.clear();
        for (const MaterializationOwnerPolicy& policy : owner_policies) {
            portfolio_owner_scratch_.push_back(ContextPortfolioOwnerPolicy{
                .owner                    = policy.owner,
                .private_retention_weight = policy.private_retention_weight,
                .explicit_shared_credit   = policy.explicit_shared_credit,
            });
        }
        portfolio_checkpoint_scratch_.clear();
        bool portfolio_degraded = false;
        for (const MaterializationCheckpointPolicy& policy : checkpoint_policies) {
            const MaterializationOwnerPolicy* owner =
                owner_policy_indexed(owner_policies, policy.owner);
            if (owner == nullptr) {
                throw std::logic_error("checkpoint policy has no portfolio owner");
            }
            const auto lookup_key = std::tuple{
                policy.owner.value, static_cast<std::uint8_t>(policy.checkpoint.kind),
                policy.checkpoint.frontier, policy.checkpoint.ordinal};
            const auto impact = std::lower_bound(
                impact_scratch_.begin(), impact_scratch_.end(), lookup_key,
                [](const CombinedImpact& value, const auto& key) {
                    return std::tuple{value.owner.value,
                                      static_cast<std::uint8_t>(value.checkpoint.kind),
                                      value.checkpoint.frontier,
                                      value.checkpoint.ordinal} < key;
                });
            const std::uint64_t target_recovery =
                impact == impact_scratch_.end() || impact->owner != policy.owner ||
                        impact->checkpoint != policy.checkpoint
                    ? policy.baseline_recovery_ns
                    : impact->target_ns;
            portfolio_checkpoint_scratch_.push_back(ContextPortfolioCheckpointValue{
                .owner                = policy.owner,
                .demand_mask          = policy.demand_mask,
                .rebuild_ns           = policy.rebuild_ns,
                .baseline_recovery_ns = policy.baseline_recovery_ns,
                .target_recovery_ns   = target_recovery,
            });
            if (target_recovery > policy.baseline_recovery_ns) {
                portfolio_degraded = true;
                planning_saturating_add(cost.affected_selected_hits, policy.selected_hit_count);
                cost.newest_affected_hit_epoch =
                    std::max(cost.newest_affected_hit_epoch, policy.last_hit_epoch);
            }
        }
        const ContextPortfolioValueResult portfolio =
            portfolio_value_.fold(portfolio_owner_scratch_, portfolio_checkpoint_scratch_);
        if (portfolio.saturated && portfolio_degraded) {
            cost.future_loss_ns = std::numeric_limits<std::uint64_t>::max();
        } else {
            cost.future_loss_ns =
                portfolio.baseline_public_value > portfolio.target_public_value
                    ? portfolio.baseline_public_value - portfolio.target_public_value
                    : 0;
            planning_saturating_add(cost.future_loss_ns, portfolio.private_transition_loss);
        }
        cost.total_ns = cost.now_ns;
        planning_saturating_add(cost.total_ns, cost.future_loss_ns);
        cost.lower_bound_ns = priced.optimistic_request_ns;
        planning_saturating_add(cost.lower_bound_ns, cost.future_loss_ns);
        return cost;
    }

    [[nodiscard]] static Incumbent make_incumbent(PressureTargetHandle target,
                                                  std::uint32_t candidate_index,
                                                  const PressureTargetAssessment& assessment,
                                                  AssessedPressureTarget&& assessed,
                                                  FoldedCost cost, LogicalGoal goal) {
        return Incumbent{
            .target           = target,
            .assessed         = std::move(assessed),
            .candidate_index  = candidate_index,
            .publication_slot = goal.publication_slot,
            .source_mode      = assessment.source_mode,
            .cost             = std::move(cost),
            .owner_outcomes   = std::vector<PressureOwnerOutcome>(assessment.owner_outcomes.begin(),
                                                                  assessment.owner_outcomes.end()),
            .checkpoint_outcomes =
                [&] {
                    std::vector<PressureCheckpointOutcome> outcomes;
                    outcomes.reserve(assessment.checkpoint_impacts.size());
                    for (const PressureCheckpointRecoveryImpact& impact :
                         assessment.checkpoint_impacts) {
                        outcomes.push_back(PressureCheckpointOutcome{
                            .owner      = impact.owner,
                            .checkpoint = impact.checkpoint,
                            .survives   = impact.survives,
                        });
                    }
                    return outcomes;
                }(),
            .degradation_units = assessment.degradation_units,
            .root_maximal      = assessment.root_maximal,
        };
    }

    [[nodiscard]] static auto queue_key(const QueueEntry& entry) noexcept {
        return std::tuple{
            entry.lower_bound_ns,
            entry.affected_selected_hits,
            entry.newest_affected_hit_epoch,
            entry.owner_evictions,
            entry.checkpoint_drops,
            entry.copy_operations,
            entry.transferred_bytes,
            entry.remaining_prefill,
            entry.remaining_vision_prefill,
            std::numeric_limits<std::uint32_t>::max() - entry.reused_prompt_tokens,
            entry.current_session_binding ? 0U : 1U,
            entry.candidate_ordinal,
            entry.stable_target_ordinal,
        };
    }

    void queue_push(QueueEntry entry) {
        queue_.push_back(std::move(entry));
        std::push_heap(queue_.begin(), queue_.end(), [](const auto& left, const auto& right) {
            return queue_key(right) < queue_key(left);
        });
    }

    [[nodiscard]] QueueEntry queue_pop() {
        std::pop_heap(queue_.begin(), queue_.end(), [](const auto& left, const auto& right) {
            return queue_key(right) < queue_key(left);
        });
        QueueEntry result = std::move(queue_.back());
        queue_.pop_back();
        return result;
    }

    [[nodiscard]] static auto pending_key(const PendingEntry& entry) noexcept {
        return std::tuple{entry.lower_bound_ns, entry.guidance.key()};
    }

    void pending_push(PendingEntry entry) {
        pending_.push_back(std::move(entry));
        std::push_heap(pending_.begin(), pending_.end(), [](const auto& left, const auto& right) {
            return pending_key(right) < pending_key(left);
        });
    }

    [[nodiscard]] PendingEntry pending_pop() {
        std::pop_heap(pending_.begin(), pending_.end(), [](const auto& left, const auto& right) {
            return pending_key(right) < pending_key(left);
        });
        PendingEntry result = std::move(pending_.back());
        pending_.pop_back();
        return result;
    }

    [[nodiscard]] static bool guidance_dominates(const GuidanceCost& left,
                                                 const GuidanceCost& right) noexcept {
        const std::array<std::uint64_t, 13> left_dimensions{
            left.estimated_remaining_steps, left.unsatisfied_constraints,
            left.normalized_residual_q20,   left.affected_selected_hits,
            left.newest_affected_hit_epoch, left.explicit_shared_losses,
            left.retention_weight,          left.owner_evictions,
            left.checkpoint_drops,          left.estimated_immediate_ns,
            left.degradation_units,         left.copy_operations,
            left.transferred_bytes,
        };
        const std::array<std::uint64_t, 13> right_dimensions{
            right.estimated_remaining_steps, right.unsatisfied_constraints,
            right.normalized_residual_q20,   right.affected_selected_hits,
            right.newest_affected_hit_epoch, right.explicit_shared_losses,
            right.retention_weight,          right.owner_evictions,
            right.checkpoint_drops,          right.estimated_immediate_ns,
            right.degradation_units,         right.copy_operations,
            right.transferred_bytes,
        };
        bool strict = false;
        for (std::size_t index = 0; index < left_dimensions.size(); ++index) {
            if (left_dimensions[index] > right_dimensions[index]) { return false; }
            strict = strict || left_dimensions[index] < right_dimensions[index];
        }
        return strict;
    }

    void guided_insert(GuidedEntry entry) {
        if (std::any_of(guided_.begin(), guided_.end(), [&](const GuidedEntry& existing) {
                return existing.candidate_index == entry.candidate_index &&
                       existing.lower_bound_ns <= entry.lower_bound_ns &&
                       guidance_dominates(existing.guidance, entry.guidance);
            })) {
            return;
        }
        std::erase_if(guided_, [&](const GuidedEntry& existing) {
            return existing.candidate_index == entry.candidate_index &&
                   entry.lower_bound_ns <= existing.lower_bound_ns &&
                   guidance_dominates(entry.guidance, existing.guidance);
        });
        guided_.push_back(std::move(entry));
        const std::uint32_t candidate_index = guided_.back().candidate_index;
        const std::size_t candidate_size    = static_cast<std::size_t>(
            std::count_if(guided_.begin(), guided_.end(), [&](const GuidedEntry& item) {
                return item.candidate_index == candidate_index;
            }));
        if (candidate_size <= kGuidedBeamWidth) { return; }
        auto worst = guided_.end();
        for (auto item = guided_.begin(); item != guided_.end(); ++item) {
            if (item->candidate_index != candidate_index) { continue; }
            if (worst == guided_.end() ||
                std::tuple{worst->lower_bound_ns, worst->guidance.key()} <
                    std::tuple{item->lower_bound_ns, item->guidance.key()}) {
                worst = item;
            }
        }
        if (worst == guided_.end()) {
            throw std::logic_error("candidate guided beam accounting is inconsistent");
        }
        guided_.erase(worst);
    }

    [[nodiscard]] GuidedEntry guided_pop() {
        const auto key = [&](const GuidedEntry& entry) {
            if (entry.candidate_index >= candidate_guided_steps_.size()) {
                throw std::logic_error("guided target candidate index is invalid");
            }
            return std::tuple{
                candidate_guided_steps_[entry.candidate_index] == 0 ? 0U : 1U,
                entry.lower_bound_ns,
                entry.guidance.key(),
            };
        };
        const auto best    = std::min_element(guided_.begin(), guided_.end(),
                                              [&](const GuidedEntry& left, const GuidedEntry& right) {
                                               return key(left) < key(right);
                                           });
        GuidedEntry result = std::move(*best);
        guided_.erase(best);
        ++candidate_guided_steps_[result.candidate_index];
        return result;
    }

    static constexpr std::uint8_t kTargetDiscovered = BoundedTargetLedger::Discovered;
    static constexpr std::uint8_t kTargetAssessed   = BoundedTargetLedger::Assessed;
    static constexpr std::uint8_t kTargetExpanded   = BoundedTargetLedger::Expanded;

    [[nodiscard]] bool target_marked(std::uint32_t ordinal, std::uint8_t mark) const {
        return target_ledger_.contains(ordinal, mark);
    }

    void mark_target(std::uint32_t ordinal, std::uint8_t mark) {
        target_ledger_.mark(ordinal, mark);
    }

    [[nodiscard]] static MaterializationDiagnostics
    complete_diagnostics(const FoldedCost& cost, std::uint32_t targets_evaluated,
                         std::uint64_t projection_work, Clock::time_point planning_started,
                         MaterializationStopReason reason, bool maximal_fallback) noexcept {
        return make_diagnostics(cost, targets_evaluated, projection_work, planning_started, 0,
                                0, reason, false, 0, maximal_fallback);
    }

    [[nodiscard]] static MaterializationDiagnostics
    make_diagnostics(const FoldedCost& cost, std::uint32_t targets_evaluated,
                     std::uint64_t projection_work, Clock::time_point planning_started,
                     std::uint64_t search_elapsed_ns, std::uint64_t search_budget_ns,
                     MaterializationStopReason reason, bool budget_exhausted,
                     std::uint32_t degradation_units, bool maximal_fallback) noexcept {
        return MaterializationDiagnostics{
            .predicted_now_ns           = cost.now_ns,
            .predicted_future_loss_ns   = cost.future_loss_ns,
            .predicted_total_ns         = cost.total_ns,
            .targets_evaluated          = targets_evaluated,
            .projection_work            = projection_work,
            .planning_elapsed_ns        = elapsed_ns(planning_started, Clock::now()),
            .search_elapsed_ns          = search_elapsed_ns,
            .search_budget_ns           = search_budget_ns,
            .stop_reason                = reason,
            .budget_exhausted           = budget_exhausted,
            .selected_degradation_units = degradation_units,
            .selected_maximal_fallback  = maximal_fallback,
        };
    }

    std::vector<QueueEntry> queue_;
    std::vector<PendingEntry> pending_;
    std::vector<GuidedEntry> guided_;
    std::vector<FoldedCost> identity_costs_;
    std::vector<std::uint32_t> candidate_guided_steps_;
    std::vector<std::uint8_t> candidate_seed_complete_;
    BoundedTargetLedger target_ledger_;
    std::vector<CombinedImpact> impact_scratch_;
    // Lookup cache, not planning state: a const planner may refresh it for a new policy span.
    mutable std::vector<const MaterializationOwnerPolicy*> owner_policy_index_;
    mutable const void* owner_policy_index_source_ = nullptr;
    mutable std::size_t owner_policy_index_size_   = 0;
    ContextPortfolioValue portfolio_value_;
    std::vector<ContextPortfolioOwnerPolicy> portfolio_owner_scratch_;
    std::vector<ContextPortfolioCheckpointValue> portfolio_checkpoint_scratch_;
};

} // namespace ninfer::runtime
