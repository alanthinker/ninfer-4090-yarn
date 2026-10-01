#include "core/site_bad_alloc.h"
#include "targets/qwen3_6/impl/runtime/state_reclaim_policy.h"
#include "runtime/cache/cache_tier_policy.h"
#include "runtime/engine/resource_manager.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string_view>
#include <thread>
#include <utility>
#include <variant>
#include <vector>

namespace {

using ninfer::PrefixReusePath;
using ninfer::RuntimeStats;
namespace state_reclaim = ninfer::targets::qwen3_6::detail::state_reclaim;
using ninfer::runtime::CancellationFlagView;
using ninfer::runtime::CheckpointKind;
using ninfer::runtime::CheckpointRecoveryAlternativeWork;
using ninfer::runtime::CheckpointRef;
using ninfer::runtime::CheckpointScope;
using ninfer::runtime::CommitDisposition;
using ninfer::runtime::ConsumeStatus;
using ninfer::runtime::ContextOperationCounts;
using ninfer::runtime::ContextTransactionInProgress;
using ninfer::runtime::ContextTransactionReserveStatus;
using ninfer::runtime::ContextTransactionStatus;
using ninfer::runtime::ContextTransferObservation;
using ninfer::runtime::ContextTransferRequirement;
using ninfer::runtime::FinishDisposition;
using ninfer::runtime::LaneId;
using ninfer::runtime::MaterializationCheckpointPolicy;
using ninfer::runtime::MaterializationMachineWork;
using ninfer::runtime::MaterializationOwnerPolicy;
using ninfer::runtime::OwnerImportance;
using ninfer::runtime::PrefillWork;
using ninfer::runtime::PlanningCandidateId;
using ninfer::runtime::PlanningOwnerId;
using ninfer::runtime::PrivateSourceMode;
using ninfer::runtime::ProgramResourceRevision;
using ninfer::runtime::Readiness;
using ninfer::runtime::RequestPlanSummary;
using ninfer::runtime::RetentionClass;
using ninfer::runtime::live_multiplier_q16;
using ninfer::runtime::OwnerScore;
using ninfer::runtime::owner_score_ns;
using ninfer::runtime::reuse_evidence_q16;
using ninfer::runtime::RetirePreferenceEntry;
using ninfer::runtime::victim_score_ns;
using ninfer::runtime::VictimDisposition;

int failures = 0;

void require(bool condition, const char* message) {
    if (!condition) { throw std::runtime_error(message); }
}

template <class Test>
void run_test(const char* name, Test&& test) {
    try {
        test();
    } catch (const std::exception& error) {
        ++failures;
        std::cerr << "FAIL " << name << ": " << error.what() << '\n';
    }
}

ninfer::runtime::ContextMachineCostModel test_cost_model() {
    ninfer::runtime::ContextMachineCostModel model;
    for (auto& transfer : model.transfer) {
        transfer.batch_ns        = 0;
        transfer.operation_ns    = 0;
        transfer.ns_per_byte_q32 = ninfer::runtime::kContextCostQ32One;
    }
    model.prefill.token_ns_q32          = 100ULL * ninfer::runtime::kContextCostQ32One;
    model.prefill.attention_pair_ns_q32 = ninfer::runtime::kContextCostQ32One;
    model.prefill.vision_item_ns        = 1;
    model.prefill.vision_patch_ns_q32   = ninfer::runtime::kContextCostQ32One;
    return model;
}

void set_fake_machine_costs(MaterializationMachineWork& work, std::uint64_t optimistic_ns,
                            std::uint64_t immediate_ns) {
    work.optimistic_candidate_transfers                  = {};
    work.candidate_transfers                             = {};
    work.pressure_transfers                              = {};
    work.remaining_prefill_work                          = {};
    work.optimistic_candidate_transfers[2].payload_bytes = optimistic_ns;
    work.candidate_transfers[2].payload_bytes            = immediate_ns;
}

CheckpointRecoveryAlternativeWork fake_recovery_work(std::uint64_t ns) {
    CheckpointRecoveryAlternativeWork work;
    work.prefill.attention_pairs = ns;
    return work;
}

struct FakePreparedPrompt {
    std::uint32_t content_key = 0;
};

struct FakeCacheSessionKey {
    std::uint32_t value = 0;

    [[nodiscard]] std::string_view view() const noexcept {
        return {reinterpret_cast<const char*>(&value), sizeof(value)};
    }

    friend bool operator==(FakeCacheSessionKey, FakeCacheSessionKey) = default;
};

struct FakeShortlistKey {
    std::uint32_t digest   = 0;
    std::uint32_t frontier = 0;

    friend bool operator==(FakeShortlistKey, FakeShortlistKey) = default;
};

struct FakeRequiredKV {
    std::uint32_t main_pages    = 1;
    std::uint32_t backend_pages = 0;

    friend bool operator==(FakeRequiredKV, FakeRequiredKV) = default;
};

struct FakeCheckpointSummary {
    CheckpointRef ref;
    CheckpointScope scope = CheckpointScope::Private;
    FakeShortlistKey shortlist_key;
    std::optional<std::array<std::uint8_t, 32>> echo_key;
    ninfer::runtime::ReplicaResidency state_residency =
        ninfer::runtime::ReplicaResidency::DeviceOnly;
    FakeRequiredKV required_kv;
    PrefillWork rebuild_work;

    friend bool operator==(const FakeCheckpointSummary&, const FakeCheckpointSummary&) = default;
};

struct FakeContinuationSummary {
    std::optional<FakeCheckpointSummary> endpoint;
    std::optional<FakeCheckpointSummary> rewrite;
    std::vector<FakeCheckpointSummary> long_anchors;
    std::uint32_t active_references = 0;
};

struct FakeSharedPrefixSummary {
    FakeCheckpointSummary checkpoint;
    std::uint32_t active_references = 0;

    friend bool operator==(const FakeSharedPrefixSummary&,
                           const FakeSharedPrefixSummary&) = default;
};

FakeCheckpointSummary endpoint(std::uint32_t digest, std::uint32_t frontier) {
    return FakeCheckpointSummary{
        .ref           = CheckpointRef{.kind     = CheckpointKind::SessionEndpoint,
                                       .frontier = frontier,
                                       .ordinal  = 0},
        .scope         = CheckpointScope::Private,
        .shortlist_key = FakeShortlistKey{.digest = digest, .frontier = frontier},
        .required_kv   = FakeRequiredKV{.main_pages = 1, .backend_pages = 0},
        .rebuild_work  = PrefillWork{.tokens = frontier},
    };
}

FakeCheckpointSummary rewrite_checkpoint(std::uint32_t digest, std::uint32_t frontier) {
    return FakeCheckpointSummary{
        .ref =
            CheckpointRef{.kind = CheckpointKind::TurnClosure, .frontier = frontier, .ordinal = 0},
        .scope         = CheckpointScope::Private,
        .shortlist_key = FakeShortlistKey{.digest = digest, .frontier = frontier},
        .required_kv   = FakeRequiredKV{.main_pages = 1, .backend_pages = 0},
        .rebuild_work  = PrefillWork{.tokens = frontier},
    };
}

FakeCheckpointSummary long_anchor(std::uint32_t digest, std::uint32_t frontier,
                                  std::uint32_t ordinal) {
    return FakeCheckpointSummary{
        .ref           = CheckpointRef{.kind     = CheckpointKind::LongAnchor,
                                       .frontier = frontier,
                                       .ordinal  = ordinal},
        .scope         = CheckpointScope::Private,
        .shortlist_key = FakeShortlistKey{.digest = digest, .frontier = frontier},
        .required_kv   = FakeRequiredKV{.main_pages = 1, .backend_pages = 0},
        .rebuild_work  = PrefillWork{.tokens = frontier},
    };
}

FakeCheckpointSummary shared_checkpoint(std::uint32_t digest, std::uint32_t frontier) {
    return FakeCheckpointSummary{
        .ref           = CheckpointRef{.kind     = CheckpointKind::SharedStablePrefix,
                                       .frontier = frontier,
                                       .ordinal  = 0},
        .scope         = CheckpointScope::Shared,
        .shortlist_key = FakeShortlistKey{.digest = digest, .frontier = frontier},
        .required_kv   = FakeRequiredKV{.main_pages = 1, .backend_pages = 0},
        .rebuild_work  = PrefillWork{.tokens = frontier},
    };
}

struct FakeContextCache {
    struct Opportunity {
        ninfer::PromptCacheMarkerKind kind = ninfer::PromptCacheMarkerKind::SharedStablePrefix;
        ninfer::SharedCandidateEvidence evidence =
            ninfer::SharedCandidateEvidence::ExplicitBoundary;
        std::uint32_t frontier = 0;
    };

    std::optional<FakeCacheSessionKey> session_key;
    RetentionClass retention  = RetentionClass::RecentPrivate;
    bool update_session_index = true;
    std::vector<Opportunity> opportunities;
};

struct FakeRequestBasePlan {
    RequestPlanSummary value;
    FakeContextCache cache;
    std::uint32_t shortlist_digest = 0;
    bool allow_shortlist           = true;
    bool isolated_feasible         = true;
    std::array<std::uint8_t, 32> echo_key{};
    bool allow_echo = false;

    [[nodiscard]] const RequestPlanSummary& summary() const noexcept { return value; }

    [[nodiscard]] const FakeContextCache& context_cache() const noexcept { return cache; }

    [[nodiscard]] std::optional<std::array<std::uint8_t, 32>> echo_prefix_key() const {
        if (!allow_echo) { return std::nullopt; }
        return echo_key;
    }

    [[nodiscard]] std::optional<FakeShortlistKey>
    prefix_shortlist_key(std::uint32_t frontier) const noexcept {
        if (!allow_shortlist || frontier == 0) { return std::nullopt; }
        return FakeShortlistKey{.digest = shortlist_digest, .frontier = frontier};
    }

    [[nodiscard]] std::optional<PrefillWork>
    shared_candidate_rebuild_work(std::uint32_t frontier) const noexcept {
        const auto found =
            std::find_if(cache.opportunities.begin(), cache.opportunities.end(),
                         [&](const auto& opportunity) { return opportunity.frontier == frontier; });
        return found == cache.opportunities.end()
                   ? std::nullopt
                   : std::optional<PrefillWork>(PrefillWork{.tokens = frontier});
    }
};

FakeRequestBasePlan make_base(std::uint32_t digest,
                              std::optional<FakeCacheSessionKey> session = std::nullopt,
                              RetentionClass retention  = RetentionClass::RecentPrivate,
                              bool update_session_index = true) {
    FakeRequestBasePlan out;
    out.value.prompt_tokens           = 64;
    out.value.requested_output_tokens = 8;
    out.value.effective_output_tokens = 8;
    out.value.service_work_quanta     = 64;
    out.value.publish_continuation    = true;
    out.cache.session_key             = session;
    out.cache.retention               = retention;
    out.cache.update_session_index    = update_session_index;
    out.shortlist_digest              = digest;
    return out;
}

struct FakeContinuationHandle {
    std::uint32_t id          = 0;
    std::uint32_t content_key = 0;

    FakeContinuationHandle() = default;

    FakeContinuationHandle(std::uint32_t id_value, std::uint32_t key_value)
        : id(id_value), content_key(key_value) {}

    FakeContinuationHandle(FakeContinuationHandle&& other) noexcept
        : id(std::exchange(other.id, 0)), content_key(other.content_key) {}

    FakeContinuationHandle& operator=(FakeContinuationHandle&& other) noexcept {
        id          = std::exchange(other.id, 0);
        content_key = other.content_key;
        return *this;
    }

    FakeContinuationHandle(const FakeContinuationHandle&)            = delete;
    FakeContinuationHandle& operator=(const FakeContinuationHandle&) = delete;
};

struct FakeSharedPrefixHandle {
    std::uint32_t id          = 0;
    std::uint32_t content_key = 0;

    FakeSharedPrefixHandle()                                             = default;
    FakeSharedPrefixHandle(FakeSharedPrefixHandle&&) noexcept            = default;
    FakeSharedPrefixHandle& operator=(FakeSharedPrefixHandle&&) noexcept = default;
    FakeSharedPrefixHandle(const FakeSharedPrefixHandle&)                = delete;
    FakeSharedPrefixHandle& operator=(const FakeSharedPrefixHandle&)     = delete;
};

struct FakeSequenceHandle {
    std::uint32_t id = 0;

    friend bool operator==(FakeSequenceHandle, FakeSequenceHandle) = default;
};

struct FakeCaptureOffer {
    std::uint32_t id = 0;
};

struct FakeAdmissionCandidate {
    RequestPlanSummary value;
    PrefillWork remaining;
    std::vector<ContextTransferRequirement> transfers;
    ninfer::runtime::IdentityMaterializationAssessment identity;
    PrivateSourceMode source_mode           = PrivateSourceMode::ConsumeToActive;
    std::uint32_t private_source_id         = 0;
    std::uint32_t shared_source_id          = 0;
    std::uint32_t shared_source_content_key = 0;
    std::uint32_t shared_source_frontier    = 0;
    ninfer::runtime::MaterializationRejection rejection =
        ninfer::runtime::MaterializationRejection::None;
    std::string rejection_detail;

    [[nodiscard]] const RequestPlanSummary& summary() const noexcept { return value; }

    [[nodiscard]] const ninfer::runtime::IdentityMaterializationAssessment&
    identity_assessment() const noexcept {
        return identity;
    }

    // The planner reports the best-reuse candidate's own identity verdict in its diagnostics, so a
    // package admitted to that interface must be able to name the verdict and its detail.
    [[nodiscard]] ninfer::runtime::MaterializationRejection identity_rejection() const noexcept {
        return rejection;
    }

    [[nodiscard]] const std::string& identity_rejection_detail() const noexcept {
        return rejection_detail;
    }
};

struct FakeTargetDecision {
    std::uint64_t id                  = 0;
    std::uint64_t immediate_ns        = 100'000'000;
    std::uint32_t degradation_units   = 1;
    std::uint32_t dropped_checkpoints = 0;
    bool evicts_continuation          = false;
    bool shared_owner                 = false;

    friend bool operator==(const FakeTargetDecision&, const FakeTargetDecision&) = default;
};

struct FakePressureTargetHandle {
    std::uint32_t generation = 0;
    std::uint32_t index      = 0;

    friend bool operator==(FakePressureTargetHandle, FakePressureTargetHandle) = default;
};

struct FakePreparedPressureExpansion {
    std::uint32_t generation         = 0;
    std::uint32_t scratch_generation = 0;
    std::uint32_t new_count          = 0;

    FakePreparedPressureExpansion(std::uint32_t generation_value,
                                  std::uint32_t scratch_generation_value,
                                  std::uint32_t new_count_value) noexcept
        : generation(generation_value), scratch_generation(scratch_generation_value),
          new_count(new_count_value) {}

    FakePreparedPressureExpansion(FakePreparedPressureExpansion&&) noexcept            = default;
    FakePreparedPressureExpansion& operator=(FakePreparedPressureExpansion&&) noexcept = default;
    FakePreparedPressureExpansion(const FakePreparedPressureExpansion&)                = delete;
    FakePreparedPressureExpansion& operator=(const FakePreparedPressureExpansion&)     = delete;

    [[nodiscard]] std::uint32_t new_canonical_count() const noexcept { return new_count; }
};

struct FakePressureExpansionView {
    std::span<const FakePressureTargetHandle> children;
    std::uint32_t new_canonical_count = 0;
};

class FakeAssessedPressureTarget {
public:
    FakeAssessedPressureTarget(FakeAssessedPressureTarget&&) noexcept            = default;
    FakeAssessedPressureTarget& operator=(FakeAssessedPressureTarget&&) noexcept = default;
    FakeAssessedPressureTarget(const FakeAssessedPressureTarget&)                = delete;
    FakeAssessedPressureTarget& operator=(const FakeAssessedPressureTarget&)     = delete;

    [[nodiscard]] const ninfer::runtime::PressureTargetAssessment& assessment() const noexcept {
        return assessment_;
    }

private:
    FakeAssessedPressureTarget(FakePressureTargetHandle target,
                               ninfer::runtime::PressureTargetAssessment assessment) noexcept
        : target_(target), assessment_(assessment) {}

    FakePressureTargetHandle target_;
    ninfer::runtime::PressureTargetAssessment assessment_;

    friend class FakePressurePlanningSession;
};

struct FakeResourcePlan {
    FakeAdmissionCandidate admission;
    ProgramResourceRevision revision;
    std::vector<FakeTargetDecision> private_actions;
    std::vector<FakeTargetDecision> shared_actions;
    std::vector<std::uint32_t> private_owner_ids;
    std::vector<std::uint32_t> shared_owner_ids;
    std::vector<PlanningOwnerId> private_planning_ids;
    std::vector<PlanningOwnerId> shared_planning_ids;

    FakeResourcePlan() = default;

    FakeResourcePlan(FakeAdmissionCandidate admission_value, ProgramResourceRevision revision_value)
        : admission(std::move(admission_value)), revision(revision_value) {}

    FakeResourcePlan(FakeResourcePlan&&) noexcept            = default;
    FakeResourcePlan& operator=(FakeResourcePlan&&) noexcept = default;
    FakeResourcePlan(const FakeResourcePlan&)                = delete;
    FakeResourcePlan& operator=(const FakeResourcePlan&)     = delete;

    [[nodiscard]] const RequestPlanSummary& summary() const noexcept { return admission.summary(); }

    [[nodiscard]] bool needs_transfer() const noexcept { return !admission.transfers.empty(); }

    [[nodiscard]] ProgramResourceRevision resource_revision() const noexcept { return revision; }
};

struct FakePersistentBackfillProof {
    ProgramResourceRevision revision;

    [[nodiscard]] ProgramResourceRevision resource_revision() const noexcept { return revision; }
};

struct FakeStartResult {
    FakeSequenceHandle sequence;
};

struct FakeMaterializationVictimResult {
    PlanningOwnerId owner;
    VictimDisposition disposition = VictimDisposition::Retained;
    bool pressure_committed       = false;
    std::optional<FakeContinuationSummary> final_summary;
};

struct FakeMaterializationSharedVictimResult {
    PlanningOwnerId owner;
    VictimDisposition disposition = VictimDisposition::Retained;
    bool pressure_committed       = false;
    std::optional<FakeSharedPrefixSummary> final_summary;
};

struct FakeMaterializationSourceResult {
    PrivateSourceMode mode = PrivateSourceMode::Retain;
    std::optional<FakeContinuationSummary> final_summary;
};

struct FakeMaterializationSharedSourceResult {
    std::optional<FakeSharedPrefixSummary> final_summary;
};

struct FakeMaterializationResult {
    ContextTransactionStatus status = ContextTransactionStatus::Aborted;
    std::optional<FakeStartResult> published;
    std::optional<FakeMaterializationSourceResult> source;
    std::optional<FakeMaterializationSharedSourceResult> shared_source;
    std::vector<FakeMaterializationVictimResult> victims;
    std::vector<FakeMaterializationSharedVictimResult> shared_victims;
    std::vector<ContextTransferObservation> transfer_observations;
    ContextOperationCounts operations;
};

struct FakeSharedPrefixPublication {
    FakeSharedPrefixHandle handle;
    FakeSharedPrefixSummary summary;
};

struct FakeActiveCaptureResult {
    ContextTransactionStatus status     = ContextTransactionStatus::Aborted;
    bool capacity_preparation_committed = false;
    FakeContinuationSummary active_summary;
    std::optional<FakeSharedPrefixPublication> shared;
    std::vector<FakeMaterializationVictimResult> victims;
    std::vector<FakeMaterializationSharedVictimResult> shared_victims;
    std::vector<ContextTransferObservation> transfer_observations;
    ContextOperationCounts operations;
};

using FakeContextTransactionProgress =
    std::variant<ContextTransactionInProgress, FakeMaterializationResult, FakeActiveCaptureResult>;

struct FakeCaptureAssessment {
    FakeShortlistKey shortlist_key;
    ninfer::SharedCandidateEvidence shared_evidence = ninfer::SharedCandidateEvidence::None;
    PrefillWork protected_rebuild_work;
    std::vector<ContextTransferRequirement> transfer_requirements;
    std::vector<CheckpointRecoveryAlternativeWork> projected_recovery_work{fake_recovery_work(0)};
    std::vector<CheckpointRef> private_replacement_candidates;
    std::uint32_t frontier = 0;
    bool publishes_private   = false;
    bool publishes_shared    = false;
    bool needs_transfer      = false;
    bool physically_feasible = true;
};

struct FakeTimings {
    std::uint64_t value = 0;
};

struct FakeSpeculativeStats {
    std::uint64_t value = 0;
};

struct FakeFinishResult {
    ConsumeStatus status          = ConsumeStatus::InvariantMismatch;
    FinishDisposition disposition = FinishDisposition::Released;
    FakeTimings timings;
    FakeSpeculativeStats speculative;
    FakeContinuationSummary summary;
    std::optional<FakeContinuationHandle> continuation;
    // Why a cancelled publication declined, when it did.
    ninfer::runtime::AbandonedPrefixOutcome abandon_outcome =
        ninfer::runtime::AbandonedPrefixOutcome::Retained;
    std::string abandon_detail;
};

struct FakeAbortResult {
    ConsumeStatus status = ConsumeStatus::InvariantMismatch;
    FakeTimings timings;
    FakeSpeculativeStats speculative;
};

struct FakeReleaseResult {
    ConsumeStatus status = ConsumeStatus::InvariantMismatch;
};

struct FakeCommitRowResult {
    CommitDisposition disposition = CommitDisposition::Active;
};

struct FakeCommitResult {
    std::array<FakeCommitRowResult, ninfer::kMaximumConcurrency> rows{};
    std::size_t row_count = 0;
};

struct FakeDiscardResult {
    ConsumeStatus status  = ConsumeStatus::InvariantMismatch;
    std::size_t row_count = 0;
};

struct FakePhysicalUsage {
    std::uint32_t device_state_slots      = 0;
    std::uint32_t host_state_slots        = 0;
    std::uint32_t device_main_kv_pages    = 0;
    std::uint32_t device_backend_kv_pages = 0;
    std::size_t host_kv_bytes             = 0;
};

class FakeProgram;

class FakePressurePlanningSession {
public:
    FakePressurePlanningSession(FakeProgram& program,
                                std::span<const FakeAdmissionCandidate* const> candidates,
                                std::span<const PlanningCandidateId> candidate_ids,
                                std::span<const FakeContinuationHandle* const> private_owners,
                                std::span<const PlanningOwnerId> private_owner_ids,
                                std::span<const FakeSharedPrefixHandle* const> shared_owners,
                                std::span<const PlanningOwnerId> shared_owner_ids);

    FakePressurePlanningSession(FakePressurePlanningSession&&) noexcept            = default;
    FakePressurePlanningSession& operator=(FakePressurePlanningSession&&) noexcept = default;
    FakePressurePlanningSession(const FakePressurePlanningSession&)                = delete;
    FakePressurePlanningSession& operator=(const FakePressurePlanningSession&)     = delete;

    [[nodiscard]] FakePressureTargetHandle identity_target(PlanningCandidateId candidate) const;
    [[nodiscard]] FakePressureTargetHandle identity_target() const;

    [[nodiscard]] static constexpr PlanningCandidateId candidate_id() noexcept {
        return PlanningCandidateId{.value = 0};
    }

    [[nodiscard]] std::optional<FakePressureTargetHandle> tier_policy_target(
        PlanningCandidateId candidate, std::span<const OwnerImportance> owner_values);
    [[nodiscard]] ninfer::runtime::PressureTargetGuidance guidance(FakePressureTargetHandle target);
    [[nodiscard]] FakeAssessedPressureTarget assess(FakePressureTargetHandle target);
    [[nodiscard]] FakePreparedPressureExpansion prepare_expansion(FakePressureTargetHandle parent);
    [[nodiscard]] FakePressureExpansionView
    commit_expansion(FakePreparedPressureExpansion&& prepared);
    void discard_expansion(FakePreparedPressureExpansion&& prepared) noexcept;
    [[nodiscard]] PrefillWork
    shared_capture_split_prefill_work(const FakeAssessedPressureTarget&, const FakePreparedPrompt&,
                                      std::span<const std::uint32_t> frontiers) const;
    [[nodiscard]] std::optional<FakeResourcePlan> seal(FakeAssessedPressureTarget&& assessed,
                                                       const FakePreparedPrompt& prompt,
                                                       ninfer::runtime::FinalScheduleIntent intent);
    [[nodiscard]] std::optional<FakeResourcePlan> seal(FakeAssessedPressureTarget&& assessed);
    [[nodiscard]] std::optional<FakeResourcePlan>
    seal_capture(FakeAssessedPressureTarget&& assessed);

private:
    struct Owner {
        const FakeContinuationHandle* private_handle = nullptr;
        const FakeSharedPrefixHandle* shared_handle  = nullptr;
        PlanningOwnerId id;
        bool shared = false;
    };

    struct Target {
        std::uint32_t candidate_index = 0;
        std::vector<std::uint16_t> choices;
        std::uint32_t stable_ordinal = 0;
        bool root_capped            = false;
    };

    [[nodiscard]] bool valid(FakePressureTargetHandle target) const noexcept;
    [[nodiscard]] std::uint32_t candidate_index(PlanningCandidateId candidate) const;
    void populate_options(std::uint32_t candidate_index);
    [[nodiscard]] bool same_target(const Target& left, const Target& right) const noexcept;
    [[nodiscard]] std::vector<FakeTargetDecision> decisions_for(std::uint32_t candidate_index,
                                                                std::size_t owner_index) const;

    FakeProgram* program_ = nullptr;
    ProgramResourceRevision revision_;
    std::uint32_t generation_         = 0;
    std::uint32_t scratch_generation_ = 0;
    bool scratch_live_                = false;
    std::vector<const FakeAdmissionCandidate*> candidates_;
    std::vector<PlanningCandidateId> candidate_ids_;
    std::vector<Owner> owners_;
    std::vector<std::vector<std::vector<FakeTargetDecision>>> options_;
    std::vector<std::uint8_t> options_populated_;
    std::vector<Target> targets_;
    std::vector<Target> expansion_scratch_;
    std::vector<FakePressureTargetHandle> committed_children_;
    std::vector<ninfer::runtime::PressureOwnerOutcome> guidance_outcomes_;
    std::vector<ninfer::runtime::PressureOwnerOutcome> assessment_outcomes_;
    std::vector<ninfer::runtime::PressureCheckpointRecoveryImpact> assessment_impacts_;
    std::vector<CheckpointRecoveryAlternativeWork> assessment_recovery_work_;
};

class FakeProgram {
public:
    friend class FakePressurePlanningSession;

    enum class TransactionKind : std::uint8_t {
        None,
        Materialization,
        Capture,
    };

    [[nodiscard]] bool isolated_request_feasible(const FakeRequestBasePlan& base) const noexcept {
        return base.isolated_feasible;
    }

    struct FakeAdoptedSource {
        FakeContinuationHandle handle;
        std::uint32_t frontier = 0;
    };
    [[nodiscard]] std::optional<FakeAdoptedSource>
    try_adopt_from_index(const FakePreparedPrompt&, const FakeRequestBasePlan&) {
        return std::nullopt;
    }

    // Retired owners are reported by the Program so the catalog can repair itself; a fake that
    // never retires keeps every catalogued handle live.
    [[nodiscard]] bool continuation_is_live(const FakeContinuationHandle& handle) const noexcept {
        return std::none_of(retired_continuation_ids.begin(), retired_continuation_ids.end(),
                            [&](std::uint32_t id) { return id == handle.id; });
    }
    [[nodiscard]] bool shared_prefix_is_live(const FakeSharedPrefixHandle& handle) const noexcept {
        return std::none_of(retired_shared_ids.begin(), retired_shared_ids.end(),
                            [&](std::uint32_t id) { return id == handle.id; });
    }

    // The manager hands its victim score order over before any call that may reach the ladder.
    // `exclude_slots` are the admission's own sources - off limits to every R2 walk while this
    // order stands (the Program keeps them for both the score walk and the oldest-touched
    // fallback; the fake keeps them so a test can read the hand-off back).
    void set_retire_preference(std::span<const ninfer::runtime::RetirePreferenceEntry> order,
                               std::span<const std::uint32_t> exclude_slots = {}) {
        retire_preference.assign(order.begin(), order.end());
        retire_excluded.assign(exclude_slots.begin(), exclude_slots.end());
    }
    std::vector<ninfer::runtime::RetirePreferenceEntry> retire_preference;
    std::vector<std::uint32_t> retire_excluded;

    [[nodiscard]] std::optional<FakeAdmissionCandidate>
    inspect_admission(const FakePreparedPrompt& prompt, const FakeRequestBasePlan& base, LaneId,
                      const FakeContinuationHandle* source,
                      const FakeSharedPrefixHandle* shared_source,
                      std::optional<CheckpointRef> checkpoint, bool must_retain_source) {
        ++admission_inspections;
        if (source != nullptr) {
            inspected_private_sources.push_back(source->id);
            if (source->content_key != prompt.content_key || !checkpoint) { return std::nullopt; }
        }
        if (shared_source != nullptr) {
            inspected_shared_sources.push_back(shared_source->id);
            if (shared_source->content_key != prompt.content_key || !checkpoint) {
                return std::nullopt;
            }
        }

        FakeAdmissionCandidate plan;
        plan.value = base.summary();
        if (checkpoint) {
            plan.value.reusable_prompt_tokens = checkpoint->frontier;
            switch (checkpoint->kind) {
            case CheckpointKind::SessionEndpoint:
                plan.value.prefix_reuse_path = PrefixReusePath::PrivateEndpoint;
                break;
            case CheckpointKind::TurnClosure:
                plan.value.prefix_reuse_path = PrefixReusePath::PrivateTurnClosure;
                break;
            case CheckpointKind::ResponseReplay:
                plan.value.prefix_reuse_path = PrefixReusePath::PrivateResponseReplay;
                break;
            case CheckpointKind::LongAnchor:
                plan.value.prefix_reuse_path = PrefixReusePath::PrivateLongAnchor;
                break;
            case CheckpointKind::SharedStablePrefix:
                plan.value.prefix_reuse_path = PrefixReusePath::SharedStablePrefix;
                break;
            }
        } else {
            plan.value.reusable_prompt_tokens = 0;
            plan.value.prefix_reuse_path      = PrefixReusePath::Root;
        }
        plan.remaining.tokens = plan.value.prompt_tokens > plan.value.reusable_prompt_tokens
                                    ? plan.value.prompt_tokens - plan.value.reusable_prompt_tokens
                                    : 0;
        if (source != nullptr) {
            plan.private_source_id = source->id;
            plan.source_mode =
                must_retain_source ? PrivateSourceMode::Retain : PrivateSourceMode::ConsumeToActive;
        } else if (shared_source != nullptr) {
            plan.shared_source_id          = shared_source->id;
            plan.shared_source_content_key = shared_source->content_key;
            plan.shared_source_frontier    = checkpoint->frontier;
            plan.source_mode               = PrivateSourceMode::Retain;
        }
        plan.identity.machine_work.remaining_prefill_work = plan.remaining;
        plan.identity.machine_work.reused_prompt_tokens   = plan.value.reusable_prompt_tokens;
        plan.identity.physical_status =
            target_feasible(std::span<const FakeTargetDecision>{})
                ? ninfer::runtime::MaterializationPhysicalStatus::Feasible
                : ninfer::runtime::MaterializationPhysicalStatus::Infeasible;
        plan.identity.source_mode = plan.source_mode;
        plan.identity.expandable  = plan.identity.physical_status !=
                                   ninfer::runtime::MaterializationPhysicalStatus::Feasible;
        plan.identity.projection_work = 1;
        plan.identity.assessment_digest =
            (static_cast<std::uint64_t>(plan.value.reusable_prompt_tokens) << 32U) ^
            revision_.value;
        return plan;
    }

    [[nodiscard]] std::optional<FakeResourcePlan>
    seal_identity(const FakeAdmissionCandidate& admission, const FakePreparedPrompt&,
                  ninfer::runtime::FinalScheduleIntent intent) {
        if (admission.identity.physical_status !=
            ninfer::runtime::MaterializationPhysicalStatus::Feasible) {
            return std::nullopt;
        }
        selected_shared_capture_frontiers.assign(intent.shared_capture_frontiers.begin(),
                                                 intent.shared_capture_frontiers.end());
        seal_attempts.emplace_back();
        return FakeResourcePlan(admission, revision_);
    }

    [[nodiscard]] FakePressurePlanningSession
    begin_pressure_planning(std::span<const FakeAdmissionCandidate* const> candidates,
                            std::span<const PlanningCandidateId> candidate_ids,
                            std::span<const FakeContinuationHandle* const> private_owners,
                            std::span<const PlanningOwnerId> private_owner_ids,
                            std::span<const FakeSharedPrefixHandle* const> shared_owners,
                            std::span<const PlanningOwnerId> shared_owner_ids);

    [[nodiscard]] PrefillWork
    shared_capture_split_prefill_work(const FakeAdmissionCandidate& candidate,
                                      const FakePreparedPrompt&,
                                      std::span<const std::uint32_t> frontiers) const noexcept {
        PrefillWork work = candidate.identity.machine_work.remaining_prefill_work;
        work.attention_pairs += static_cast<std::uint64_t>(frontiers.size()) * 100U;
        return work;
    }

    [[nodiscard]] bool
    target_feasible(std::span<const FakeTargetDecision> decisions) const noexcept {
        if (pressure_units(decisions) < required_pressure_actions) { return false; }
        const bool has_eviction =
            std::any_of(decisions.begin(), decisions.end(),
                        [](const auto& decision) { return decision.evicts_continuation; });
        if (required_action_id && !has_eviction &&
            std::none_of(decisions.begin(), decisions.end(), [&](const auto& decision) {
                return decision.id == *required_action_id;
            })) {
            return false;
        }
        if (require_evictions &&
            std::any_of(decisions.begin(), decisions.end(),
                        [](const auto& decision) { return !decision.evicts_continuation; })) {
            return false;
        }
        return true;
    }

    [[nodiscard]] std::size_t
    pressure_units(std::span<const FakeTargetDecision> decisions) const noexcept {
        std::size_t units = 0;
        for (const FakeTargetDecision& decision : decisions) {
            units += decision.evicts_continuation ? eviction_pressure_action_units : 1U;
        }
        return units;
    }

    [[nodiscard]] ContextTransactionReserveStatus
    start_resource_transaction(FakeResourcePlan&& plan, FakePreparedPrompt&& prompt,
                               CancellationFlagView cancellation) {
        ++start_calls;
        started_source_id   = plan.admission.private_source_id;
        started_source_mode = plan.admission.source_mode;
        started_action_ids.clear();
        for (const auto& action : plan.private_actions) { started_action_ids.push_back(action.id); }
        for (const auto& action : plan.shared_actions) { started_action_ids.push_back(action.id); }
        if (capacity_miss_on_start) {
            // The Program releases its staging before rethrowing, so the resource manager may treat
            // this as a retryable capacity miss rather than a fatal engine error.
            throw ninfer::core::SiteBadAlloc("fake: materialization state restore after LRU evict");
        }
        if (cancellation.requested() || abort_start || plan.revision != revision_) {
            return ContextTransactionReserveStatus::Aborted;
        }
        pending_prompt_ = prompt;
        pending_plan_.emplace(std::move(plan));
        transaction_kind_ = TransactionKind::Materialization;
        advance_revision();
        return ContextTransactionReserveStatus::Reserved;
    }

    [[nodiscard]] std::optional<FakePersistentBackfillProof>
    prove_persistent_backfill(const FakeRequestBasePlan&, const FakeResourcePlan& candidate,
                              std::span<const FakeSequenceHandle>) const {
        if (candidate.resource_revision() != revision_) { return std::nullopt; }
        return FakePersistentBackfillProof{.revision = revision_};
    }

    [[nodiscard]] FakeContextTransactionProgress
    progress_context_transaction(CancellationFlagView cancellation) {
        require(transaction_kind_ != TransactionKind::None,
                "fake Program has no context transaction");
        if (progress_in_progress_once) {
            progress_in_progress_once = false;
            return ContextTransactionInProgress{};
        }
        if (transaction_kind_ == TransactionKind::Capture) {
            FakeActiveCaptureResult result;
            result.status =
                cancellation.requested() ? ContextTransactionStatus::Aborted : capture_status;
            if (pending_plan_) {
                for (std::size_t index = 0; index < pending_plan_->private_actions.size();
                     ++index) {
                    const FakeTargetDecision& action = pending_plan_->private_actions[index];
                    FakeMaterializationVictimResult victim{
                        .owner       = pending_plan_->private_planning_ids[index],
                        .disposition = action.evicts_continuation ? VictimDisposition::Evicted
                                                                  : VictimDisposition::Retained,
                        .pressure_committed = action.evicts_continuation ||
                                              result.status == ContextTransactionStatus::Published,
                    };
                    if (!action.evicts_continuation) {
                        const std::uint32_t owner = pending_plan_->private_owner_ids[index];
                        victim.final_summary.emplace();
                        const std::uint32_t content = sequence_content_keys_.at(owner);
                        if (action.dropped_checkpoints == 0 ||
                            malform_private_checkpoint_identity) {
                            victim.final_summary->endpoint = endpoint(content, finish_frontier);
                        }
                        if (finish_with_rewrite && (action.dropped_checkpoints == 0 ||
                                                    !malform_private_checkpoint_identity)) {
                            victim.final_summary->rewrite =
                                rewrite_checkpoint(content, finish_frontier - 1U);
                        }
                    }
                    result.victims.push_back(std::move(victim));
                }
                for (std::size_t index = 0; index < pending_plan_->shared_actions.size(); ++index) {
                    const FakeTargetDecision& action = pending_plan_->shared_actions[index];
                    FakeMaterializationSharedVictimResult victim{
                        .owner       = pending_plan_->shared_planning_ids[index],
                        .disposition = action.evicts_continuation ? VictimDisposition::Evicted
                                                                  : VictimDisposition::Retained,
                        .pressure_committed = action.evicts_continuation ||
                                              result.status == ContextTransactionStatus::Published,
                    };
                    if (!action.evicts_continuation) {
                        const std::uint32_t owner = pending_plan_->shared_owner_ids[index];
                        victim.final_summary      = FakeSharedPrefixSummary{
                                 .checkpoint = shared_checkpoint(owner, finish_frontier),
                        };
                    }
                    result.shared_victims.push_back(std::move(victim));
                }
            }
            if (malform_last_capture_private_victim && !result.victims.empty()) {
                FakeMaterializationVictimResult& victim = result.victims.back();
                victim.disposition                      = VictimDisposition::Evicted;
                victim.pressure_committed               = false;
                victim.final_summary.reset();
            }
            if (reverse_pressure_results) {
                std::reverse(result.victims.begin(), result.victims.end());
                std::reverse(result.shared_victims.begin(), result.shared_victims.end());
            }
            if (result.status == ContextTransactionStatus::Published) {
                result.active_summary = capture_summary;
                if (pending_capture_publish_shared_) {
                    FakeSharedPrefixHandle handle;
                    handle.id          = next_shared_id_++;
                    handle.content_key = capture_assessment.shortlist_key.digest;
                    result.shared      = FakeSharedPrefixPublication{
                             .handle = std::move(handle),
                             .summary =
                            FakeSharedPrefixSummary{
                                     .checkpoint =
                                    shared_checkpoint(capture_assessment.shortlist_key.digest,
                                                           capture_assessment.shortlist_key.frontier),
                                     .active_references = 1,
                            },
                    };
                }
            }
            return result;
        }

        require(pending_plan_.has_value(), "fake materialization plan disappeared");
        const FakeResourcePlan& plan = *pending_plan_;
        FakeMaterializationResult result;
        result.status = (abort_progress || cancellation.requested())
                            ? ContextTransactionStatus::Aborted
                            : ContextTransactionStatus::Published;
        for (std::size_t index = 0; index < plan.private_actions.size(); ++index) {
            const FakeTargetDecision& action = plan.private_actions[index];
            const bool evicted               = action.evicts_continuation;
            FakeMaterializationVictimResult victim{
                .owner       = plan.private_planning_ids[index],
                .disposition = evicted ? VictimDisposition::Evicted : VictimDisposition::Retained,
                .pressure_committed =
                    evicted || result.status == ContextTransactionStatus::Published,
            };
            if (!evicted) {
                const std::uint32_t owner_id = plan.private_owner_ids.at(index);
                victim.final_summary.emplace();
                const std::uint32_t content = sequence_content_keys_.at(owner_id);
                if (action.dropped_checkpoints == 0 || malform_private_checkpoint_identity) {
                    victim.final_summary->endpoint = endpoint(content, finish_frontier);
                }
                if (finish_with_rewrite &&
                    (action.dropped_checkpoints == 0 || !malform_private_checkpoint_identity)) {
                    victim.final_summary->rewrite =
                        rewrite_checkpoint(content, finish_frontier - 1U);
                }
            }
            result.victims.push_back(std::move(victim));
        }
        for (std::size_t index = 0; index < plan.shared_actions.size(); ++index) {
            const FakeTargetDecision& action = plan.shared_actions[index];
            result.shared_victims.push_back(FakeMaterializationSharedVictimResult{
                .owner              = plan.shared_planning_ids[index],
                .disposition        = action.evicts_continuation ? VictimDisposition::Evicted
                                                                 : VictimDisposition::Retained,
                .pressure_committed = action.evicts_continuation ||
                                      result.status == ContextTransactionStatus::Published,
            });
        }
        if (malform_last_private_victim && !result.victims.empty()) {
            FakeMaterializationVictimResult& victim = result.victims.back();
            victim.disposition                      = VictimDisposition::Evicted;
            victim.pressure_committed               = false;
            victim.final_summary.reset();
        }
        if (reverse_pressure_results) {
            std::reverse(result.victims.begin(), result.victims.end());
            std::reverse(result.shared_victims.begin(), result.shared_victims.end());
        }
        if (plan.admission.private_source_id != 0) {
            result.source = FakeMaterializationSourceResult{
                .mode = result.status == ContextTransactionStatus::Aborted
                            ? PrivateSourceMode::Retain
                            : plan.admission.source_mode};
        }
        if (plan.admission.shared_source_id != 0) {
            result.shared_source = FakeMaterializationSharedSourceResult{};
            if (report_shared_source_summary) {
                const std::uint32_t references =
                    result.status == ContextTransactionStatus::Published
                        ? ++reported_shared_active_references
                        : reported_shared_active_references;
                FakeCheckpointSummary checkpoint =
                    shared_checkpoint(plan.admission.shared_source_content_key,
                                      plan.admission.shared_source_frontier);
                if (change_shared_source_residency_on_second_report && references >= 2) {
                    checkpoint.state_residency = ninfer::runtime::ReplicaResidency::Both;
                }
                result.shared_source->final_summary = FakeSharedPrefixSummary{
                    .checkpoint        = std::move(checkpoint),
                    .active_references = references,
                };
            }
        }
        if (result.status == ContextTransactionStatus::Published) {
            const std::uint32_t sequence_id        = next_sequence_id_++;
            sequence_content_keys_.at(sequence_id) = pending_prompt_.content_key;
            result.published = FakeStartResult{.sequence = FakeSequenceHandle{sequence_id}};
        }
        return result;
    }

    void finalize_context_transaction() noexcept {
        transaction_kind_ = TransactionKind::None;
        pending_plan_.reset();
        pending_capture_publish_shared_ = false;
    }

    [[nodiscard]] bool has_context_transaction() const noexcept {
        return transaction_kind_ != TransactionKind::None;
    }

    [[nodiscard]] FakeCaptureAssessment inspect_capture(const FakeCaptureOffer&,
                                                        const FakeSharedPrefixHandle*,
                                                        const FakeSharedPrefixHandle* replacement,
                                                        std::optional<CheckpointRef>,
                                                        bool permit_shared_publication) const {
        if (replacement != nullptr) { ++shared_replacement_inspections; }
        FakeCaptureAssessment assessment = capture_assessment;
        if (!permit_shared_publication) { assessment.publishes_shared = false; }
        return assessment;
    }

    [[nodiscard]] std::vector<CheckpointRecoveryAlternativeWork>
    checkpoint_recovery_work(const FakeContinuationHandle&, CheckpointRef) const {
        return {fake_recovery_work(0)};
    }

    [[nodiscard]] std::vector<CheckpointRecoveryAlternativeWork>
    checkpoint_recovery_work(const FakeSharedPrefixHandle&, CheckpointRef) const {
        return {fake_recovery_work(0)};
    }

    [[nodiscard]] FakePressurePlanningSession
    begin_capture_pressure_planning(const FakeCaptureAssessment& assessment,
                                    std::span<const FakeContinuationHandle* const> private_owners,
                                    std::span<const PlanningOwnerId> private_owner_ids,
                                    std::span<const FakeSharedPrefixHandle* const> shared_owners,
                                    std::span<const PlanningOwnerId> shared_owner_ids) {
        capture_pressure_candidate_       = std::make_unique<FakeAdmissionCandidate>();
        FakeAdmissionCandidate& candidate = *capture_pressure_candidate_;
        capture_private_owner_ids.clear();
        capture_shared_owner_ids.clear();
        for (const FakeContinuationHandle* owner : private_owners) {
            capture_private_owner_ids.push_back(owner->id);
        }
        for (const FakeSharedPrefixHandle* owner : shared_owners) {
            capture_shared_owner_ids.push_back(owner->id);
        }
        candidate.value.prompt_tokens     = assessment.shortlist_key.frontier;
        for (const ContextTransferRequirement& transfer : assessment.transfer_requirements) {
            const std::size_t direction = static_cast<std::size_t>(transfer.direction);
            candidate.identity.machine_work.candidate_transfers[direction].payload_bytes +=
                transfer.work.payload_bytes;
            candidate.identity.machine_work.candidate_transfers[direction].copy_operations +=
                transfer.work.copy_operations;
        }
        candidate.identity.machine_work.optimistic_candidate_transfers =
            candidate.identity.machine_work.candidate_transfers;
        candidate.identity.physical_status =
            target_feasible(std::span<const FakeTargetDecision>{})
                ? ninfer::runtime::MaterializationPhysicalStatus::Feasible
                : ninfer::runtime::MaterializationPhysicalStatus::Infeasible;
        candidate.identity.expandable = candidate.identity.physical_status !=
                                        ninfer::runtime::MaterializationPhysicalStatus::Feasible;
        candidate.identity.projection_work = 1;
        candidate.identity.assessment_digest =
            (static_cast<std::uint64_t>(assessment.shortlist_key.frontier) << 32U) ^
            revision_.value;
        const FakeAdmissionCandidate* candidate_handle = &candidate;
        const std::array candidate_ids{FakePressurePlanningSession::candidate_id()};
        return begin_pressure_planning(
            std::span<const FakeAdmissionCandidate* const>(&candidate_handle, 1), candidate_ids,
            private_owners, private_owner_ids, shared_owners, shared_owner_ids);
    }

    [[nodiscard]] bool shared_capture_matches(const FakeCaptureOffer&,
                                              const FakeSharedPrefixHandle&) const {
        return false;
    }

    void skip_capture(FakeCaptureOffer&&) { ++skipped_captures; }

    [[nodiscard]] ContextTransactionReserveStatus
    reserve_active_capture(FakeCaptureOffer&&, const FakeSharedPrefixHandle*,
                           const FakeSharedPrefixHandle*, std::optional<CheckpointRef>, bool,
                           CancellationFlagView cancellation) {
        if (cancellation.requested() || abort_capture_start) {
            return ContextTransactionReserveStatus::Aborted;
        }
        pending_plan_.reset();
        pending_capture_publish_shared_ = false;
        transaction_kind_               = TransactionKind::Capture;
        advance_revision();
        return ContextTransactionReserveStatus::Reserved;
    }

    [[nodiscard]] ContextTransactionReserveStatus reserve_active_capture_with_pressure(
        FakeCaptureOffer&&, const FakeSharedPrefixHandle*, const FakeSharedPrefixHandle*,
        std::optional<CheckpointRef>, bool publish_shared, FakeResourcePlan&& pressure,
        CancellationFlagView cancellation) {
        if (cancellation.requested() || abort_capture_start || pressure.revision != revision_) {
            return ContextTransactionReserveStatus::Aborted;
        }
        started_action_ids.clear();
        for (const auto& action : pressure.private_actions) {
            started_action_ids.push_back(action.id);
        }
        for (const auto& action : pressure.shared_actions) {
            started_action_ids.push_back(action.id);
        }
        pending_plan_.emplace(std::move(pressure));
        pending_capture_publish_shared_ = publish_shared;
        transaction_kind_               = TransactionKind::Capture;
        advance_revision();
        return ContextTransactionReserveStatus::Reserved;
    }

    [[nodiscard]] FakeFinishResult finish(FakeSequenceHandle sequence) noexcept {
        ++finish_calls;
        if (finish_fail_next) {
            finish_fail_next = false;
            return {};
        }
        advance_revision();
        FakeFinishResult result;
        result.status = ConsumeStatus::Consumed;
        if (finish_release) {
            result.disposition = FinishDisposition::Released;
            return result;
        }
        result.disposition      = FinishDisposition::Catalogued;
        const std::uint32_t key = sequence_content_keys_[sequence.id];
        result.summary.endpoint = endpoint(key, finish_frontier);
        if (finish_with_rewrite) {
            result.summary.rewrite = rewrite_checkpoint(key, finish_frontier - 1U);
        }
        result.continuation.emplace(sequence.id, key);
        return result;
    }

    [[nodiscard]] FakeAbortResult abort(FakeSequenceHandle) noexcept {
        ++abort_calls;
        advance_revision();
        return FakeAbortResult{.status      = ConsumeStatus::Consumed,
                               .timings     = FakeTimings{.value = 7},
                               .speculative = FakeSpeculativeStats{.value = 9}};
    }

    // Cancelled-request publication: the manager calls abandon_prefill for a cancelled prefill and
    // publish_cancelled for a cancellation after the prefill completed. The fake models both on the
    // same outcome knob, since the manager only distinguishes them by which call it makes.
    [[nodiscard]] FakeFinishResult abandon_prefill(FakeSequenceHandle sequence) noexcept {
        ++abandon_prefill_calls;
        FakeFinishResult result = finish(sequence);
        result.abandon_outcome  = result.disposition == FinishDisposition::Catalogued
                                      ? ninfer::runtime::AbandonedPrefixOutcome::Retained
                                      : ninfer::runtime::AbandonedPrefixOutcome::PublicationDeclined;
        return result;
    }

    [[nodiscard]] FakeFinishResult publish_cancelled(FakeSequenceHandle sequence) noexcept {
        ++publish_cancelled_calls;
        FakeFinishResult result = finish(sequence);
        result.abandon_outcome  = result.disposition == FinishDisposition::Catalogued
                                      ? ninfer::runtime::AbandonedPrefixOutcome::Retained
                                      : ninfer::runtime::AbandonedPrefixOutcome::PublicationDeclined;
        return result;
    }

    [[nodiscard]] FakeReleaseResult
    release_continuation(FakeContinuationHandle&& continuation) noexcept {
        released_continuations.push_back(continuation.id);
        advance_revision();
        return FakeReleaseResult{.status = ConsumeStatus::Consumed};
    }

    [[nodiscard]] ProgramResourceRevision resource_revision() const noexcept { return revision_; }

    [[nodiscard]] FakePhysicalUsage physical_usage() const noexcept { return usage; }

    // One unit of the least valuable cached data is gone: the deficit it was holding down is now
    // smaller (the fake lowers the requirement a target has to satisfy) and the pool revision moves,
    // so the caller must re-plan instead of trusting its previous verdict.
    [[nodiscard]] bool release_one_cached_unit() {
        ++cached_relief_requests;
        // §三 R1's two halves, modelled separately because they answer DIFFERENT deficits: a state
        // slot does not move a Device KV page, and vice versa. Whichever the request is short of is
        // the one that has to be available for the relief to be real - the fake refuses to report
        // progress for a unit the deficit cannot use, exactly as the Program must not.
        const bool usable = cached_relief_available &&
                            (relief_axis == CachedReliefAxis::StateSlot
                                 ? deficit_is_state_slot
                                 : deficit_is_device_kv);
        if (!usable) { return false; }
        cached_relief_available   = false;
        required_pressure_actions = 0;
        advance_revision();
        return true;
    }

    void invalidate_resources() noexcept { advance_revision(); }

    // Cached-relief model for the R0/R2 contract: `release_one_cached_unit` stands for the
    // Program's release ladder - it deletes one unit of the least valuable CACHED data and reports
    // whether the pool moved.
    bool cached_relief_available       = false;
    std::size_t cached_relief_requests = 0;
    // Which axis the one releasable unit lives on, and which axis this request is short of.
    // Production 2026-09-30: the relief was state-slot-only (`release_one_device_state_slot`) while
    // req#629/#650/#652/#654 were short of Device KV PAGES, so it freed a unit the deficit could
    // not use, reported "nothing releasable", and the request parked on a negative memo for its
    // full 300 s deadline at `running 0` (HTTP 499) - the pool could not change because the engine
    // was idle. The KV half (`spill_one_movable_owner_device_kv`) is what makes relief real here.
    enum class CachedReliefAxis { StateSlot, DeviceKv };
    CachedReliefAxis relief_axis     = CachedReliefAxis::StateSlot;
    bool deficit_is_state_slot       = true;
    bool deficit_is_device_kv        = false;
    // Shared-replacement model for the §三 R1 invariant: `shared_victim_device_data` says the
    // catalogued shared owner still holds Device replicas, `shared_victim_move_possible` whether its
    // data can be moved to Host, `shared_replacement_moves` counts the move attempts and
    // `shared_replacement_inspections` how often the manager went on to OFFER that owner as a
    // replacement victim (the moment that leads to
    // `[invariant1] strict-shared site=capture-replacement`).
    bool shared_victim_device_data                    = false;
    bool shared_victim_move_possible                  = true;
    mutable std::size_t shared_replacement_moves      = 0;
    mutable std::size_t shared_replacement_inspections = 0;
    // §三 R1: move first. Nothing on Device is ready as is; Device data that CAN be moved is moved
    // here (and is then Host-only); Device data that cannot be moved makes this victim unusable as a
    // replacement, because destroying it is exactly what R1 forbids.
    [[nodiscard]] bool prepare_shared_replacement(const FakeSharedPrefixHandle&) {
        ++shared_replacement_moves;
        if (!shared_victim_device_data) { return true; }
        if (!shared_victim_move_possible) { return false; }
        shared_victim_device_data = false;
        return true;
    }
    std::size_t required_pressure_actions       = 0;
    std::size_t eviction_pressure_action_units  = 1;
    std::uint32_t private_pressure_alternatives = 1;
    std::optional<std::size_t> pressure_optional_target_capacity;
    std::uint64_t pressure_action_immediate_ns      = 100'000'000;
    std::uint32_t pressure_action_degradation_units = 1;
    bool include_cumulative_private_target          = false;
    bool combined_target_cancels_pressure_copy      = false;
    std::optional<std::uint64_t> pressure_target_immediate_ns_override;
    std::optional<std::uint64_t> required_action_id;
    std::uint32_t pressure_assessment_delay_us           = 0;
    std::uint64_t pressure_checkpoint_recovery_ns        = 100;
    bool require_evictions                               = false;
    bool abort_start                                     = false;
    // Capacity miss raised from start_resource_transaction (the release ladder could not produce
    // the state or KV capacity the sealed plan needs).
    bool capacity_miss_on_start                          = false;
    bool abort_progress                                  = false;
    bool malform_last_private_victim                     = false;
    bool malform_last_capture_private_victim             = false;
    bool malform_private_checkpoint_identity             = false;
    bool reverse_pressure_results                        = false;
    bool progress_in_progress_once                       = false;
    bool finish_fail_next                                = false;
    bool finish_release                                  = false;
    bool finish_with_rewrite                             = false;
    bool abort_capture_start                             = false;
    bool report_shared_source_summary                    = false;
    bool change_shared_source_residency_on_second_report = false;
    std::uint32_t reported_shared_active_references      = 0;
    ContextTransactionStatus capture_status              = ContextTransactionStatus::Published;
    FakeCaptureAssessment capture_assessment;
    FakeContinuationSummary capture_summary;
    FakePhysicalUsage usage;

    std::uint64_t admission_inspections       = 0;
    std::uint64_t pressure_planning_sessions  = 0;
    std::uint64_t pressure_target_assessments = 0;
    std::uint64_t start_calls                 = 0;
    std::uint64_t finish_calls                = 0;
    std::uint64_t abandon_prefill_calls       = 0;
    std::uint64_t publish_cancelled_calls     = 0;
    std::uint64_t abort_calls                 = 0;
    std::uint64_t skipped_captures            = 0;
    std::size_t pressure_target_count_peak    = 0;
    std::uint32_t finish_frontier             = 16;
    std::uint32_t started_source_id           = 0;
    PrivateSourceMode started_source_mode     = PrivateSourceMode::ConsumeToActive;
    std::vector<std::uint32_t> inspected_private_sources;
    // Owners the Program retired behind the catalog's back (capacity reclamation).
    std::vector<std::uint32_t> retired_continuation_ids;
    std::vector<std::uint32_t> retired_shared_ids;
    // Owners handed to the capture planner as victims in the last capture planning pass.
    std::vector<std::uint32_t> capture_private_owner_ids;
    std::vector<std::uint32_t> capture_shared_owner_ids;
    std::vector<std::uint32_t> inspected_shared_sources;
    std::vector<std::vector<std::uint64_t>> seal_attempts;
    std::vector<std::uint64_t> started_action_ids;
    std::vector<std::uint32_t> selected_shared_capture_frontiers;
    std::vector<std::uint32_t> released_continuations;

private:
    void advance_revision() noexcept {
        if (++revision_.value == 0) { ++revision_.value; }
    }

    ProgramResourceRevision revision_{.value = 1};
    std::uint32_t planning_generation_ = 0;
    std::uint32_t next_sequence_id_    = 1;
    std::uint32_t next_shared_id_      = 1;
    std::array<std::uint32_t, 256> sequence_content_keys_{};
    TransactionKind transaction_kind_ = TransactionKind::None;
    FakePreparedPrompt pending_prompt_;
    std::optional<FakeResourcePlan> pending_plan_;
    std::unique_ptr<FakeAdmissionCandidate> capture_pressure_candidate_;
    bool pending_capture_publish_shared_ = false;
};

FakePressurePlanningSession::FakePressurePlanningSession(
    FakeProgram& program, std::span<const FakeAdmissionCandidate* const> candidates,
    std::span<const PlanningCandidateId> candidate_ids,
    std::span<const FakeContinuationHandle* const> private_owners,
    std::span<const PlanningOwnerId> private_owner_ids,
    std::span<const FakeSharedPrefixHandle* const> shared_owners,
    std::span<const PlanningOwnerId> shared_owner_ids)
    : program_(&program), revision_(program.resource_revision()) {
    require(!candidates.empty() && candidates.size() == candidate_ids.size(),
            "fake pressure session has no candidate identity");
    require(private_owners.size() == private_owner_ids.size() &&
                shared_owners.size() == shared_owner_ids.size(),
            "fake pressure owner arrays are not aligned");
    candidates_.assign(candidates.begin(), candidates.end());
    candidate_ids_.assign(candidate_ids.begin(), candidate_ids.end());
    for (std::size_t index = 0; index < private_owners.size(); ++index) {
        owners_.push_back(Owner{.private_handle = private_owners[index],
                                .id             = private_owner_ids[index],
                                .shared         = false});
    }
    for (std::size_t index = 0; index < shared_owners.size(); ++index) {
        owners_.push_back(Owner{
            .shared_handle = shared_owners[index], .id = shared_owner_ids[index], .shared = true});
    }
    std::sort(owners_.begin(), owners_.end(),
              [](const Owner& left, const Owner& right) { return left.id.value < right.id.value; });
    options_.resize(candidates_.size());
    options_populated_.resize(candidates_.size());
    for (std::size_t index = 0; index < candidates_.size(); ++index) {
        require(candidates_[index] != nullptr, "fake pressure candidate is null");
        targets_.push_back(Target{
            .candidate_index = static_cast<std::uint32_t>(index),
            .choices         = std::vector<std::uint16_t>(owners_.size(), 0),
            .stable_ordinal  = static_cast<std::uint32_t>(index),
        });
    }
    program.pressure_target_count_peak =
        std::max(program.pressure_target_count_peak, targets_.size());
    if (++program.planning_generation_ == 0) { ++program.planning_generation_; }
    ++program.pressure_planning_sessions;
    generation_ = program.planning_generation_;
}

bool FakePressurePlanningSession::valid(FakePressureTargetHandle target) const noexcept {
    return program_ != nullptr && target.generation == generation_ &&
           target.index < targets_.size() && program_->resource_revision() == revision_;
}

std::uint32_t FakePressurePlanningSession::candidate_index(PlanningCandidateId candidate) const {
    const auto found = std::find(candidate_ids_.begin(), candidate_ids_.end(), candidate);
    require(found != candidate_ids_.end(), "fake pressure candidate is foreign");
    return static_cast<std::uint32_t>(found - candidate_ids_.begin());
}

bool FakePressurePlanningSession::same_target(const Target& left,
                                              const Target& right) const noexcept {
    return left.candidate_index == right.candidate_index && left.choices == right.choices;
}

std::vector<FakeTargetDecision>
FakePressurePlanningSession::decisions_for(std::uint32_t selected_candidate,
                                           std::size_t owner_index) const {
    const FakeAdmissionCandidate& candidate = *candidates_.at(selected_candidate);
    const Owner& owner                      = owners_.at(owner_index);
    if ((!owner.shared && candidate.private_source_id != 0 &&
         candidate.private_source_id == owner.private_handle->id) ||
        (owner.shared && candidate.shared_source_id != 0 &&
         candidate.shared_source_id == owner.shared_handle->id)) {
        return {};
    }

    std::vector<FakeTargetDecision> decisions;
    if (!owner.shared) {
        for (std::uint32_t index = 0; index < program_->private_pressure_alternatives; ++index) {
            decisions.push_back(FakeTargetDecision{
                .id = 1000U + owner.private_handle->id + 10000U * static_cast<std::uint64_t>(index),
                .immediate_ns      = program_->pressure_action_immediate_ns,
                .degradation_units = program_->pressure_action_degradation_units,
            });
        }
        if (program_->include_cumulative_private_target) {
            decisions.push_back(FakeTargetDecision{
                .id                  = 5000U + owner.private_handle->id,
                .degradation_units   = 2,
                .dropped_checkpoints = 1,
            });
        }
        // An eviction removes EVERY checkpoint the owner holds. Claiming one drop while the owner
        // also holds a rewrite checkpoint made `selected_checkpoint_drops` reject the target as
        // incomplete (observed=2, dropped=2, claimed=1).
        const std::uint32_t eviction_drops =
            1U + (program_->finish_with_rewrite ? 1U : 0U);
        decisions.push_back(FakeTargetDecision{
            .id                  = 2000U + owner.private_handle->id,
            .degradation_units   = 4,
            .dropped_checkpoints = eviction_drops,
            .evicts_continuation = true,
        });
    } else {
        decisions.push_back(FakeTargetDecision{
            .id           = 3000U + owner.shared_handle->id,
            .shared_owner = true,
        });
        decisions.push_back(FakeTargetDecision{
            .id                  = 4000U + owner.shared_handle->id,
            .degradation_units   = 4,
            .dropped_checkpoints = 1,
            .evicts_continuation = true,
            .shared_owner        = true,
        });
    }
    return decisions;
}

void FakePressurePlanningSession::populate_options(std::uint32_t selected_candidate) {
    require(selected_candidate < candidates_.size(), "fake pressure candidate index is invalid");
    if (options_populated_[selected_candidate] != 0) { return; }
    options_[selected_candidate].resize(owners_.size());
    for (std::size_t index = 0; index < owners_.size(); ++index) {
        options_[selected_candidate][index] = decisions_for(selected_candidate, index);
    }
    options_populated_[selected_candidate] = 1;
}

FakePressureTargetHandle
FakePressurePlanningSession::identity_target(PlanningCandidateId candidate) const {
    return FakePressureTargetHandle{.generation = generation_, .index = candidate_index(candidate)};
}

FakePressureTargetHandle FakePressurePlanningSession::identity_target() const {
    require(candidates_.size() == 1, "fake capture pressure domain has multiple candidates");
    return FakePressureTargetHandle{.generation = generation_, .index = 0};
}

std::optional<FakePressureTargetHandle>
FakePressurePlanningSession::tier_policy_target(
    PlanningCandidateId candidate, std::span<const OwnerImportance> owner_values) {
    namespace cachep = ::ninfer::runtime::cache;
    if (scratch_live_) { return std::nullopt; }
    const std::uint32_t selected = candidate_index(candidate);
    populate_options(selected);
    if (owners_.empty()) { return std::nullopt; }

    // §6.2: the planner reports a VALUE per owner, never an ordering, so it does not name a
    // victim. The policy ranks these itself - which is the whole point of handing over worth
    // instead of a list. An owner the planner did not price is worth the maximum, i.e. taken
    // last, matching what the old "priced first, unpriced appended after" order did.
    std::vector<cachep::Datum> pool;
    pool.reserve(owners_.size());
    for (std::size_t index = 0; index < owners_.size(); ++index) {
        const auto found = std::find_if(
            owner_values.begin(), owner_values.end(),
            [&](const OwnerImportance& entry) { return entry.owner == owners_[index].id; });
        pool.push_back(cachep::Datum{
            .id           = index,
            .device_kv    = 1,
            .device_state = 0,
            .host_kv      = 1,
            .host_state   = 0,
            .importance   = found != owner_values.end()
                                ? found->value
                                : std::numeric_limits<std::uint64_t>::max(),
            .active       = false});
    }

    // One unit of Device KV per conversation with Device full - the only condition in which this
    // controller is reached, because the ordinary search has already failed. Host is given
    // exactly the room the spill will consume, so R1 (move, never destroy) is the rule that
    // fires here; the fake prices no state pool, so both state axes stay at zero capacity with
    // zero demand and cannot open a gap. Host-side deletion has its own cases in
    // tests/test_cache_tier_policy_test, which owns the rules. The demand is the scenario's own
    // "how much room must be freed".
    const auto units = static_cast<std::uint64_t>(program_->required_pressure_actions);
    const cachep::TierOccupancy occupancy{
        .device_kv_used        = owners_.size(),
        .device_kv_capacity    = owners_.size(),
        .device_state_used     = 0,
        .device_state_capacity = 0,
        .host_kv_used          = owners_.size(),
        .host_kv_capacity      = owners_.size() + units,
        .host_state_used       = 0,
        .host_state_capacity   = 0,
    };
    const cachep::Demand demand{.device_kv = units, .host_kv = units};
    const cachep::Plan outcome = cachep::plan(demand, occupancy, pool);
    if (outcome.enqueue || outcome.steps.empty()) { return std::nullopt; }

    Target target{
        .candidate_index = selected,
        .choices         = std::vector<std::uint16_t>(owners_.size(), 0),
    };
    const auto selected_decisions = [&] {
        std::vector<FakeTargetDecision> decisions;
        for (std::size_t index = 0; index < owners_.size(); ++index) {
            const std::uint16_t choice = target.choices[index];
            if (choice != 0) { decisions.push_back(options_[selected][index][choice - 1U]); }
        }
        return decisions;
    };
    // The policy decides WHICH conversations act and HOW MANY. The polarity it asks for is a
    // preference; the scenario's own feasibility predicate (`required_action_id`,
    // `require_evictions`) has the last word, so a second pass flips every step to the opposite
    // polarity before the controller gives up.
    for (int pass = 0; pass < 2; ++pass) {
        target.choices.assign(owners_.size(), 0);
        bool assigned = true;
        for (const cachep::Step& step : outcome.steps) {
            const auto& alternatives = options_[selected][static_cast<std::size_t>(step.id)];
            const bool want_evict = (step.action == cachep::Action::DropFromHost) != (pass != 0);
            std::size_t picked = alternatives.size();
            for (std::size_t alt = 0; alt < alternatives.size(); ++alt) {
                if (alternatives[alt].evicts_continuation != want_evict) { continue; }
                picked = alt;
                if (program_->required_action_id &&
                    alternatives[alt].id == *program_->required_action_id) {
                    break;
                }
            }
            if (picked == alternatives.size()) { assigned = false; break; }
            target.choices[static_cast<std::size_t>(step.id)] =
                static_cast<std::uint16_t>(picked + 1U);
        }
        if (!assigned || !program_->target_feasible(selected_decisions())) { continue; }
        auto existing = std::find_if(
            targets_.begin(), targets_.end(),
            [&](const Target& prior) { return same_target(prior, target); });
        if (existing != targets_.end()) {
            return FakePressureTargetHandle{
                .generation = generation_,
                .index      = static_cast<std::uint32_t>(existing - targets_.begin()),
            };
        }
        target.stable_ordinal = static_cast<std::uint32_t>(targets_.size());
        targets_.push_back(std::move(target));
        program_->pressure_target_count_peak =
            std::max(program_->pressure_target_count_peak, targets_.size());
        return FakePressureTargetHandle{
            .generation = generation_,
            .index      = static_cast<std::uint32_t>(targets_.size() - 1U),
        };
    }
    return std::nullopt;
}

ninfer::runtime::PressureTargetGuidance
FakePressurePlanningSession::guidance(FakePressureTargetHandle handle) {
    require(valid(handle) && !scratch_live_, "fake pressure guidance is stale");
    const Target& target = targets_[handle.index];
    populate_options(target.candidate_index);
    const FakeAdmissionCandidate& candidate = *candidates_[target.candidate_index];
    guidance_outcomes_.clear();
    std::vector<FakeTargetDecision> selected;
    std::uint32_t degradation_units = 0;
    std::uint32_t dropped           = 0;
    for (std::size_t index = 0; index < owners_.size(); ++index) {
        const std::uint16_t choice = target.choices[index];
        if (choice == 0) { continue; }
        const auto& alternatives = options_[target.candidate_index][index];
        require(choice <= alternatives.size(), "fake pressure guidance choice is invalid");
        const FakeTargetDecision& decision = alternatives[choice - 1U];
        selected.push_back(decision);
        degradation_units += decision.degradation_units;
        dropped += decision.dropped_checkpoints;
        guidance_outcomes_.push_back(ninfer::runtime::PressureOwnerOutcome{
            .owner               = owners_[index].id,
            .disposition         = decision.evicts_continuation ? VictimDisposition::Evicted
                                                                : VictimDisposition::Retained,
            .degradation_units   = decision.degradation_units,
            .dropped_checkpoints = decision.dropped_checkpoints,
        });
    }
    MaterializationMachineWork machine = candidate.identity.machine_work;
    const bool combined_copy_cancelled =
        program_->combined_target_cancels_pressure_copy && selected.size() > 1U &&
        std::none_of(selected.begin(), selected.end(),
                     [](const auto& decision) { return decision.evicts_continuation; });
    if (!selected.empty() && program_->pressure_target_immediate_ns_override) {
        const auto optimistic = machine.optimistic_candidate_transfers;
        set_fake_machine_costs(machine, 0, *program_->pressure_target_immediate_ns_override);
        machine.optimistic_candidate_transfers = optimistic;
    } else if (combined_copy_cancelled) {
        ++machine.pressure_transfers[2].payload_bytes;
    } else {
        for (const FakeTargetDecision& decision : selected) {
            machine.pressure_transfers[2].payload_bytes += decision.immediate_ns;
            ++machine.pressure_transfers[2].copy_operations;
        }
    }
    const std::size_t selected_units = program_->pressure_units(selected);
    const std::size_t remaining      = selected_units >= program_->required_pressure_actions
                                           ? 0
                                           : program_->required_pressure_actions - selected_units;
    return ninfer::runtime::PressureTargetGuidance{
        .physical =
            {
                .unsatisfied_constraints   = remaining == 0 ? 0U : 1U,
                .estimated_remaining_steps = static_cast<std::uint32_t>(remaining),
                .normalized_residual_q20   = static_cast<std::uint64_t>(remaining) << 20U,
            },
        .estimated_machine_work = machine,
        .owner_outcomes         = guidance_outcomes_,
        .candidate              = candidate_ids_[target.candidate_index],
        .stable_target_ordinal  = target.stable_ordinal,
        .degradation_units      = degradation_units,
        .dropped_checkpoints    = dropped,
    };
}

FakeAssessedPressureTarget FakePressurePlanningSession::assess(FakePressureTargetHandle handle) {
    require(valid(handle) && !scratch_live_, "fake pressure assessment is stale");
    if (program_->pressure_assessment_delay_us != 0) {
        std::this_thread::sleep_for(
            std::chrono::microseconds(program_->pressure_assessment_delay_us));
    }
    ++program_->pressure_target_assessments;
    const Target& target = targets_[handle.index];
    populate_options(target.candidate_index);
    const FakeAdmissionCandidate& candidate = *candidates_[target.candidate_index];
    assessment_outcomes_.clear();
    assessment_impacts_.clear();
    assessment_recovery_work_.clear();
    assessment_recovery_work_.reserve(2U * owners_.size());
    std::vector<FakeTargetDecision> selected;
    std::uint32_t degradation_units = 0;
    std::uint32_t dropped           = 0;
    bool expandable                 = false;
    for (std::size_t index = 0; index < owners_.size(); ++index) {
        const std::uint16_t choice = target.choices[index];
        const auto& alternatives   = options_[target.candidate_index][index];
        if (choice == 0) {
            expandable = expandable || !alternatives.empty();
            continue;
        }
        require(choice <= alternatives.size(), "fake pressure target choice is invalid");
        const FakeTargetDecision& decision = alternatives[choice - 1U];
        selected.push_back(decision);
        degradation_units += decision.degradation_units;
        dropped += decision.dropped_checkpoints;
        assessment_outcomes_.push_back(ninfer::runtime::PressureOwnerOutcome{
            .owner               = owners_[index].id,
            .disposition         = decision.evicts_continuation ? VictimDisposition::Evicted
                                                                : VictimDisposition::Retained,
            .degradation_units   = decision.degradation_units,
            .dropped_checkpoints = decision.dropped_checkpoints,
        });
        const auto append_checkpoint_outcome = [&](CheckpointRef checkpoint, bool survives) {
            assessment_recovery_work_.push_back(
                fake_recovery_work(survives ? 0 : program_->pressure_checkpoint_recovery_ns));
            assessment_impacts_.push_back(ninfer::runtime::PressureCheckpointRecoveryImpact{
                .owner                = owners_[index].id,
                .checkpoint           = checkpoint,
                .target_recovery_work = std::span<const CheckpointRecoveryAlternativeWork>(
                    &assessment_recovery_work_.back(), 1),
                .survives = survives,
            });
        };
        const bool endpoint_survives =
            !decision.evicts_continuation && decision.dropped_checkpoints == 0;
        append_checkpoint_outcome(CheckpointRef{.kind     = owners_[index].shared
                                                                ? CheckpointKind::SharedStablePrefix
                                                                : CheckpointKind::SessionEndpoint,
                                                .frontier = program_->finish_frontier,
                                                .ordinal  = 0},
                                  endpoint_survives);
        if (!owners_[index].shared && program_->finish_with_rewrite) {
            append_checkpoint_outcome(CheckpointRef{.kind     = CheckpointKind::TurnClosure,
                                                    .frontier = program_->finish_frontier - 1U,
                                                    .ordinal  = 0},
                                      !decision.evicts_continuation);
        }
        if (!decision.evicts_continuation) { expandable = true; }
    }

    MaterializationMachineWork machine = candidate.identity.machine_work;
    const bool combined_copy_cancelled =
        program_->combined_target_cancels_pressure_copy && selected.size() > 1U &&
        std::none_of(selected.begin(), selected.end(),
                     [](const auto& decision) { return decision.evicts_continuation; });
    if (!selected.empty() && program_->pressure_target_immediate_ns_override) {
        const auto optimistic = machine.optimistic_candidate_transfers;
        set_fake_machine_costs(machine, 0, *program_->pressure_target_immediate_ns_override);
        machine.optimistic_candidate_transfers = optimistic;
    } else if (combined_copy_cancelled) {
        // The complete target removes a transfer required by each partial target.  This models
        // source Move replacing Fork, a later eviction cancelling an earlier D2H, or two physical
        // actions coalescing into one direct stage.  Exact target cost is therefore intentionally
        // non-monotonic along the search edge.
        ++machine.pressure_transfers[2].payload_bytes;
    } else {
        for (const FakeTargetDecision& decision : selected) {
            machine.pressure_transfers[2].payload_bytes += decision.immediate_ns;
            ++machine.pressure_transfers[2].copy_operations;
        }
    }
    const bool identity  = selected.empty();
    std::uint64_t digest = candidate.identity.assessment_digest;
    if (!identity) {
        digest = 1469598103934665603ULL;
        for (const std::uint16_t choice : target.choices) {
            digest ^= choice;
            digest *= 1099511628211ULL;
        }
        digest ^= target.candidate_index;
    }
    ninfer::runtime::PressureTargetAssessment assessment{
        .physical_status       = program_->target_feasible(selected)
                                     ? ninfer::runtime::MaterializationPhysicalStatus::Feasible
                                     : ninfer::runtime::MaterializationPhysicalStatus::Infeasible,
        .source_mode           = candidate.source_mode,
        .machine_work          = machine,
        .owner_outcomes        = assessment_outcomes_,
        .checkpoint_impacts    = assessment_impacts_,
        .candidate             = candidate_ids_[target.candidate_index],
        .stable_target_ordinal = target.stable_ordinal,
        .degradation_units     = degradation_units,
        .dropped_checkpoints   = dropped,
        .projection_work       = 1U + assessment_outcomes_.size(),
        .assessment_digest     = digest,
        .expandable            = expandable,
        .root_capped          = target.root_capped,
    };
    return FakeAssessedPressureTarget(handle, assessment);
}

FakePreparedPressureExpansion
FakePressurePlanningSession::prepare_expansion(FakePressureTargetHandle handle) {
    require(valid(handle) && !scratch_live_, "fake pressure expansion is stale");
    const Target& parent = targets_[handle.index];
    populate_options(parent.candidate_index);
    expansion_scratch_.clear();
    for (std::size_t owner = 0; owner < owners_.size(); ++owner) {
        const std::uint16_t current = parent.choices[owner];
        const auto& alternatives    = options_[parent.candidate_index][owner];
        if (current == 0) {
            for (std::size_t choice = 1; choice <= alternatives.size(); ++choice) {
                Target child         = parent;
                child.choices[owner] = static_cast<std::uint16_t>(choice);
                child.root_capped   = false;
                if (std::none_of(expansion_scratch_.begin(), expansion_scratch_.end(),
                                 [&](const Target& prior) { return same_target(prior, child); })) {
                    expansion_scratch_.push_back(std::move(child));
                }
            }
        } else if (current <= alternatives.size() &&
                   !alternatives[current - 1U].evicts_continuation) {
            Target child         = parent;
            child.choices[owner] = static_cast<std::uint16_t>(alternatives.size());
            child.root_capped   = false;
            if (std::none_of(expansion_scratch_.begin(), expansion_scratch_.end(),
                             [&](const Target& prior) { return same_target(prior, child); })) {
                expansion_scratch_.push_back(std::move(child));
            }
        }
    }
    std::uint32_t new_count = 0;
    for (const Target& child : expansion_scratch_) {
        if (std::none_of(targets_.begin(), targets_.end(),
                         [&](const Target& prior) { return same_target(prior, child); })) {
            ++new_count;
        }
    }
    if (++scratch_generation_ == 0) { ++scratch_generation_; }
    scratch_live_ = true;
    return FakePreparedPressureExpansion(generation_, scratch_generation_, new_count);
}

FakePressureExpansionView
FakePressurePlanningSession::commit_expansion(FakePreparedPressureExpansion&& prepared) {
    require(scratch_live_ && prepared.generation == generation_ &&
                prepared.scratch_generation == scratch_generation_,
            "fake prepared expansion is stale");
    if (program_->pressure_optional_target_capacity) {
        const std::size_t maximum =
            candidates_.size() + 1U + *program_->pressure_optional_target_capacity;
        if (prepared.new_count > maximum - std::min(maximum, targets_.size())) {
            throw std::length_error("prepared pressure expansion exceeds the target arena");
        }
    }
    committed_children_.clear();
    std::uint32_t new_count = 0;
    for (Target& child : expansion_scratch_) {
        auto found          = std::find_if(targets_.begin(), targets_.end(),
                                           [&](const Target& prior) { return same_target(prior, child); });
        std::uint32_t index = 0;
        if (found == targets_.end()) {
            child.stable_ordinal = static_cast<std::uint32_t>(targets_.size());
            targets_.push_back(std::move(child));
            index = static_cast<std::uint32_t>(targets_.size() - 1U);
            ++new_count;
        } else {
            index = static_cast<std::uint32_t>(found - targets_.begin());
        }
        committed_children_.push_back(
            FakePressureTargetHandle{.generation = generation_, .index = index});
    }
    program_->pressure_target_count_peak =
        std::max(program_->pressure_target_count_peak, targets_.size());
    require(new_count == prepared.new_count, "fake expansion count changed before commit");
    expansion_scratch_.clear();
    scratch_live_ = false;
    return FakePressureExpansionView{
        .children            = committed_children_,
        .new_canonical_count = new_count,
    };
}

void FakePressurePlanningSession::discard_expansion(
    FakePreparedPressureExpansion&& prepared) noexcept {
    if (scratch_live_ && prepared.generation == generation_ &&
        prepared.scratch_generation == scratch_generation_) {
        expansion_scratch_.clear();
        scratch_live_ = false;
    }
}

PrefillWork FakePressurePlanningSession::shared_capture_split_prefill_work(
    const FakeAssessedPressureTarget&, const FakePreparedPrompt&,
    std::span<const std::uint32_t> frontiers) const {
    return PrefillWork{.attention_pairs = static_cast<std::uint64_t>(frontiers.size()) * 100U};
}

std::optional<FakeResourcePlan>
FakePressurePlanningSession::seal(FakeAssessedPressureTarget&& assessed, const FakePreparedPrompt&,
                                  ninfer::runtime::FinalScheduleIntent intent) {
    const FakePressureTargetHandle handle = assessed.target_;
    require(valid(handle) && !scratch_live_, "fake pressure seal is stale");
    const Target& target   = targets_[handle.index];
    const auto& assessment = assessed.assessment_;
    if (assessment.physical_status != ninfer::runtime::MaterializationPhysicalStatus::Feasible) {
        return std::nullopt;
    }
    assessed.target_.generation = 0;
    FakeResourcePlan plan(*candidates_[target.candidate_index], revision_);
    std::vector<std::uint64_t> action_ids;
    for (std::size_t index = 0; index < owners_.size(); ++index) {
        const std::uint16_t choice = target.choices[index];
        if (choice == 0) { continue; }
        const FakeTargetDecision& decision = options_[target.candidate_index][index][choice - 1U];
        action_ids.push_back(decision.id);
        if (owners_[index].shared) {
            plan.shared_actions.push_back(decision);
            plan.shared_owner_ids.push_back(owners_[index].shared_handle->id);
            plan.shared_planning_ids.push_back(owners_[index].id);
        } else {
            plan.private_actions.push_back(decision);
            plan.private_owner_ids.push_back(owners_[index].private_handle->id);
            plan.private_planning_ids.push_back(owners_[index].id);
        }
    }
    program_->selected_shared_capture_frontiers.assign(intent.shared_capture_frontiers.begin(),
                                                       intent.shared_capture_frontiers.end());
    program_->seal_attempts.push_back(std::move(action_ids));
    return plan;
}

std::optional<FakeResourcePlan>
FakePressurePlanningSession::seal_capture(FakeAssessedPressureTarget&& assessed) {
    return seal(std::move(assessed), FakePreparedPrompt{}, {});
}

std::optional<FakeResourcePlan>
FakePressurePlanningSession::seal(FakeAssessedPressureTarget&& assessed) {
    return seal_capture(std::move(assessed));
}

FakePressurePlanningSession
FakeProgram::begin_pressure_planning(std::span<const FakeAdmissionCandidate* const> candidates,
                                     std::span<const PlanningCandidateId> candidate_ids,
                                     std::span<const FakeContinuationHandle* const> private_owners,
                                     std::span<const PlanningOwnerId> private_owner_ids,
                                     std::span<const FakeSharedPrefixHandle* const> shared_owners,
                                     std::span<const PlanningOwnerId> shared_owner_ids) {
    return FakePressurePlanningSession(*this, candidates, candidate_ids, private_owners,
                                       private_owner_ids, shared_owners, shared_owner_ids);
}

struct FakePackage {
    using Program                    = FakeProgram;
    using PreparedPrompt             = FakePreparedPrompt;
    using RequestBasePlan            = FakeRequestBasePlan;
    using AdmissionCandidate         = FakeAdmissionCandidate;
    using ResourcePlan               = FakeResourcePlan;
    using PersistentBackfillProof    = FakePersistentBackfillProof;
    using SequenceHandle             = FakeSequenceHandle;
    using ContinuationHandle         = FakeContinuationHandle;
    using SharedPrefixHandle         = FakeSharedPrefixHandle;
    using CaptureOffer               = FakeCaptureOffer;
    using ContinuationSummary        = FakeContinuationSummary;
    using SharedPrefixSummary        = FakeSharedPrefixSummary;
    using CaptureAssessment          = FakeCaptureAssessment;
    using CapturePressurePlan        = FakeResourcePlan;
    using ActiveCaptureResult        = FakeActiveCaptureResult;
    using ContextTransactionProgress = FakeContextTransactionProgress;
    using MaterializationResult      = FakeMaterializationResult;
    using StartResult                = FakeStartResult;
    using FinishResult               = FakeFinishResult;
    using AbortResult                = FakeAbortResult;
    using PressureTargetHandle       = FakePressureTargetHandle;
    using AssessedPressureTarget     = FakeAssessedPressureTarget;
    using CommitResult               = FakeCommitResult;
    using DiscardResult              = FakeDiscardResult;
    using CacheSessionKey            = FakeCacheSessionKey;
};

using FakeManager = ninfer::runtime::ResourceManager<FakePackage>;

FakeManager make_manager(std::uint32_t lanes = 1, std::uint32_t private_capacity = 4,
                         std::uint32_t shared_capacity = 0, bool cache_enabled = true,
                         std::uint32_t fair_share_buckets = 8) {
    return FakeManager(lanes, private_capacity, shared_capacity, cache_enabled, 2,
                       fair_share_buckets, test_cost_model());
}

struct ActiveRequest {
    LaneId lane;
    FakeSequenceHandle sequence;
};

ActiveRequest start_active(FakeManager& manager, FakeProgram& program, std::uint32_t content_key,
                           const FakeRequestBasePlan& base, std::uint64_t publication_order) {
    auto inspection =
        manager.inspect(program, FakePreparedPrompt{content_key}, base, publication_order);
    require(inspection.choice.has_value(), "request did not produce an admission choice");
    const LaneId lane   = inspection.choice->destination();
    const auto reserved = manager.reserve_materialization(program, std::move(*inspection.choice),
                                                          FakePreparedPrompt{content_key}, {});
    require(reserved == FakeManager::MaterializationReserveResult::Reserved,
            "request materialization was not reserved");
    auto outcome = [&]() -> FakeManager::MaterializationOutcome {
        auto progress = manager.progress_context_transaction(program, {});
        if (!std::holds_alternative<ContextTransactionInProgress>(progress)) {
            return std::get<FakeManager::MaterializationOutcome>(std::move(progress));
        }
        auto completed = manager.progress_context_transaction(program, {});
        return std::get<FakeManager::MaterializationOutcome>(std::move(completed));
    }();
    require(outcome.status == ContextTransactionStatus::Published && outcome.activation,
            "request materialization did not publish");
    auto activation                   = std::move(*outcome.activation);
    const FakeSequenceHandle sequence = activation.sequence();
    manager.adopt(program, std::move(activation));
    require(manager.lane_state(lane) == ninfer::runtime::LogicalLaneState::Active,
            "published lane was not adopted as active");
    return ActiveRequest{.lane = lane, .sequence = sequence};
}

void test_private_portfolio_loss_keeps_checkpoint_identity_fixed() {
    using ninfer::runtime::ContextPortfolioCheckpointValue;
    using ninfer::runtime::ContextPortfolioOwnerPolicy;
    using ninfer::runtime::ContextPortfolioValue;

    const std::array owners{
        ContextPortfolioOwnerPolicy{.owner                    = PlanningOwnerId{.value = 0},
                                    .private_retention_weight = 4},
    };
    const std::array checkpoints{
        ContextPortfolioCheckpointValue{
            .owner                = PlanningOwnerId{.value = 0},
            .rebuild_ns           = 1000,
            .baseline_recovery_ns = 100,
            .target_recovery_ns   = 100,
        },
        ContextPortfolioCheckpointValue{
            .owner                = PlanningOwnerId{.value = 0},
            .rebuild_ns           = 800,
            .baseline_recovery_ns = 100,
            .target_recovery_ns   = 800,
        },
    };
    ContextPortfolioValue value;
    const auto result = value.fold(owners, checkpoints);
    require(result.baseline_public_value == 0 && result.target_public_value == 0 &&
                result.private_transition_loss == 2800 && !result.saturated,
            "a surviving endpoint masked loss of an earlier private checkpoint");
}

void test_portfolio_demand_and_owner_aggregation() {
    using ninfer::runtime::ContextPortfolioCheckpointValue;
    using ninfer::runtime::ContextPortfolioOwnerPolicy;
    using ninfer::runtime::ContextPortfolioValue;

    {
        const std::array owners{ContextPortfolioOwnerPolicy{.owner = PlanningOwnerId{.value = 0}}};
        const std::array checkpoints{
            ContextPortfolioCheckpointValue{
                .owner                = PlanningOwnerId{.value = 0},
                .demand_mask          = 1,
                .rebuild_ns           = 1000,
                .baseline_recovery_ns = 200,
                .target_recovery_ns   = 500,
            },
            ContextPortfolioCheckpointValue{
                .owner                = PlanningOwnerId{.value = 0},
                .demand_mask          = 1,
                .rebuild_ns           = 800,
                .baseline_recovery_ns = 200,
                .target_recovery_ns   = 400,
            },
        };
        ContextPortfolioValue value;
        const auto result = value.fold(owners, checkpoints);
        require(result.baseline_public_value == 800 && result.target_public_value == 500 &&
                    result.private_transition_loss == 0,
                "nested checkpoints counted one empirical demand more than once");
    }

    {
        const std::array owners{
            ContextPortfolioOwnerPolicy{.owner                    = PlanningOwnerId{.value = 0},
                                        .private_retention_weight = 1},
            ContextPortfolioOwnerPolicy{.owner                    = PlanningOwnerId{.value = 1},
                                        .private_retention_weight = 4},
        };
        const std::array checkpoints{
            ContextPortfolioCheckpointValue{
                .owner                = PlanningOwnerId{.value = 0},
                .rebuild_ns           = 1000,
                .baseline_recovery_ns = 100,
                .target_recovery_ns   = 400,
            },
            ContextPortfolioCheckpointValue{
                .owner                = PlanningOwnerId{.value = 1},
                .rebuild_ns           = 1000,
                .baseline_recovery_ns = 100,
                .target_recovery_ns   = 200,
            },
        };
        ContextPortfolioValue value;
        const auto result = value.fold(owners, checkpoints);
        require(result.private_transition_loss == 700,
                "private checkpoint transition losses were not summed across owners");
    }
}

void test_shared_capture_subtracts_private_transition_loss() {
    using Planner = ninfer::runtime::SharedCapturePlanner<FakePackage>;

    FakeProgram program;
    program.required_pressure_actions             = 1;
    program.require_evictions                     = true;
    program.pressure_target_immediate_ns_override = 0;
    FakeCaptureAssessment capture{
        .shared_evidence     = ninfer::SharedCandidateEvidence::ExplicitBoundary,
        .publishes_shared    = true,
        .physically_feasible = false,
    };
    FakeContinuationHandle owner{7, 0};
    const std::array<const FakeContinuationHandle*, 1> private_owners{&owner};
    const std::array<PlanningOwnerId, 1> private_owner_ids{PlanningOwnerId{.value = 0}};
    const std::array<Planner::OwnerPolicy, 1> owner_policies{
        Planner::OwnerPolicy{.owner = PlanningOwnerId{.value = 0}, .private_retention_weight = 4},
    };
    const std::array<Planner::CheckpointPolicy, 1> checkpoint_policies{
        Planner::CheckpointPolicy{
            .owner                = PlanningOwnerId{.value = 0},
            .checkpoint           = CheckpointRef{.kind     = CheckpointKind::SessionEndpoint,
                                                  .frontier = 16,
                                                  .ordinal  = 0},
            .rebuild_ns           = 1000,
            .baseline_recovery_ns = 0,
        },
    };

    Planner planner;
    const auto result = planner.plan(program, test_cost_model(),
                                     Planner::Input{
                                         .capture              = &capture,
                                         .private_owners       = private_owners,
                                         .private_owner_ids    = private_owner_ids,
                                         .shared_owners        = {},
                                         .shared_owner_ids     = {},
                                         .owner_policies       = owner_policies,
                                         .checkpoint_policies  = checkpoint_policies,
                                         .candidate_rebuild_ns = 1000,
                                     });
    require(result && result->baseline_value == 0 && result->target_value == 1000 &&
                result->immediate_ns == 0 && result->net_gain == 600,
            "shared capture gain did not subtract the private capability transition loss");
}

void test_shared_capture_budget_bounds_committed_canonical_targets() {
    using Planner = ninfer::runtime::SharedCapturePlanner<FakePackage>;

    constexpr std::size_t private_owner_count = 16;
    constexpr std::size_t shared_owner_count  = 4;
    constexpr std::uint32_t target_budget     = 64;

    FakeProgram program;
    program.required_pressure_actions         = private_owner_count + shared_owner_count + 1U;
    program.pressure_optional_target_capacity = target_budget;

    std::array<FakeContinuationHandle, private_owner_count> private_handles;
    std::array<const FakeContinuationHandle*, private_owner_count> private_owners;
    std::array<PlanningOwnerId, private_owner_count> private_owner_ids;
    for (std::size_t index = 0; index < private_owner_count; ++index) {
        private_handles[index] = FakeContinuationHandle(static_cast<std::uint32_t>(index + 1U), 0);
        private_owners[index]  = &private_handles[index];
        private_owner_ids[index] = PlanningOwnerId{.value = static_cast<std::uint32_t>(index)};
    }

    std::array<FakeSharedPrefixHandle, shared_owner_count> shared_handles;
    std::array<const FakeSharedPrefixHandle*, shared_owner_count> shared_owners;
    std::array<PlanningOwnerId, shared_owner_count> shared_owner_ids;
    for (std::size_t index = 0; index < shared_owner_count; ++index) {
        shared_handles[index].id = static_cast<std::uint32_t>(index + 1U);
        shared_owners[index]     = &shared_handles[index];
        shared_owner_ids[index] =
            PlanningOwnerId{.value = static_cast<std::uint32_t>(private_owner_count + index)};
    }

    const FakeCaptureAssessment capture{
        .shortlist_key       = FakeShortlistKey{.digest = 91, .frontier = 64},
        .publishes_shared    = true,
        .physically_feasible = false,
    };
    Planner planner;
    const auto result = planner.plan(program, test_cost_model(),
                                     Planner::Input{
                                         .capture           = &capture,
                                         .private_owners    = private_owners,
                                         .private_owner_ids = private_owner_ids,
                                         .shared_owners     = shared_owners,
                                         .shared_owner_ids  = shared_owner_ids,
                                         .target_budget     = target_budget,
                                     });
    require(!result, "bounded infeasible shared-capture search unexpectedly found a plan");
    require(program.pressure_target_count_peak <= target_budget,
            "shared-capture search committed more canonical targets than its budget");
}

void test_equal_lower_bound_does_not_short_circuit_tie_break() {
    using Planner = ninfer::runtime::MaterializationPlanner<FakePackage>;

    FakeProgram program;
    program.required_pressure_actions         = 1;
    program.pressure_action_immediate_ns      = 0;
    program.pressure_action_degradation_units = 0;

    FakeAdmissionCandidate incumbent;
    set_fake_machine_costs(incumbent.identity.machine_work, 1'000'000'000, 1'000'000'000);
    incumbent.identity.machine_work.candidate_transfers[2].copy_operations = 1;
    incumbent.identity.physical_status   = ninfer::runtime::MaterializationPhysicalStatus::Feasible;
    incumbent.identity.source_mode       = PrivateSourceMode::ConsumeToActive;
    incumbent.identity.assessment_digest = 11;

    FakeAdmissionCandidate tied_pressure;
    set_fake_machine_costs(tied_pressure.identity.machine_work, 1'000'000'000, 1'000'000'000);
    tied_pressure.identity.physical_status =
        ninfer::runtime::MaterializationPhysicalStatus::Infeasible;
    tied_pressure.identity.source_mode       = PrivateSourceMode::ConsumeToActive;
    tied_pressure.identity.expandable        = true;
    tied_pressure.identity.assessment_digest = 22;

    std::array<Planner::CandidateInput, 2> candidates{
        Planner::CandidateInput{.candidate               = &incumbent,
                                .id                      = PlanningCandidateId{.value = 0},
                                .stable_ordinal          = 0,
                                .current_session_binding = false},
        Planner::CandidateInput{.candidate               = &tied_pressure,
                                .id                      = PlanningCandidateId{.value = 1},
                                .stable_ordinal          = 1,
                                .current_session_binding = true},
    };
    FakeContinuationHandle owner{7, 0};
    const std::array<const FakeContinuationHandle*, 1> private_owners{&owner};
    const std::array<PlanningOwnerId, 1> private_owner_ids{PlanningOwnerId{.value = 0}};
    const std::array<ninfer::runtime::MaterializationOwnerPolicy, 1> owner_policy{
        ninfer::runtime::MaterializationOwnerPolicy{.owner = PlanningOwnerId{.value = 0}},
    };
    const std::array<ninfer::runtime::MaterializationCheckpointPolicy, 1> checkpoint_policy{
        ninfer::runtime::MaterializationCheckpointPolicy{
            .owner      = PlanningOwnerId{.value = 0},
            .checkpoint = CheckpointRef{.kind     = CheckpointKind::SessionEndpoint,
                                        .frontier = 16,
                                        .ordinal  = 0},
            .rebuild_ns = 100,
        },
    };

    Planner planner;
    const auto logical_goal = [](PlanningCandidateId, PrivateSourceMode,
                                 std::span<const ninfer::runtime::PressureOwnerOutcome>)
        -> std::optional<Planner::LogicalGoal> {
        return Planner::LogicalGoal{.publication_slot = 0};
    };
    const auto pressure_inputs = [&]() -> Planner::PressureInputs {
        return Planner::PressureInputs{
            .private_owners    = private_owners,
            .private_owner_ids = private_owner_ids,
            .shared_owners     = {},
            .shared_owner_ids  = {},
            .owner_policy      = owner_policy,
            .checkpoint_policy = checkpoint_policy,
        };
    };
    auto result = planner.plan(program, FakePreparedPrompt{}, test_cost_model(), candidates, 0,
                               pressure_inputs, logical_goal, 1'000'000U, Planner::Clock::now());

    require(result && result->candidate == PlanningCandidateId{.value = 1} &&
                program.pressure_planning_sessions == 1,
            "equal lower bound bypassed the pressure target that wins the stable tie-break");
}

void test_machine_cost_changes_selection_without_changing_physical_assessment() {
    using Planner = ninfer::runtime::MaterializationPlanner<FakePackage>;

    FakeProgram program;
    FakeAdmissionCandidate prefill_candidate;
    prefill_candidate.identity.machine_work.remaining_prefill_work.tokens = 10;
    prefill_candidate.identity.physical_status =
        ninfer::runtime::MaterializationPhysicalStatus::Feasible;
    prefill_candidate.identity.assessment_digest = 31;

    FakeAdmissionCandidate transfer_candidate;
    transfer_candidate.identity.machine_work.candidate_transfers[1] = {.payload_bytes   = 100,
                                                                       .copy_operations = 1};
    transfer_candidate.identity.machine_work.optimistic_candidate_transfers[1] = {
        .payload_bytes = 100, .copy_operations = 1};
    transfer_candidate.identity.physical_status =
        ninfer::runtime::MaterializationPhysicalStatus::Feasible;
    transfer_candidate.identity.assessment_digest = 32;

    const std::array<Planner::CandidateInput, 2> candidates{
        Planner::CandidateInput{.candidate      = &prefill_candidate,
                                .id             = PlanningCandidateId{.value = 0},
                                .stable_ordinal = 0},
        Planner::CandidateInput{.candidate      = &transfer_candidate,
                                .id             = PlanningCandidateId{.value = 1},
                                .stable_ordinal = 1},
    };
    const auto pressure_inputs = [] { return Planner::PressureInputs{}; };
    const auto logical_goal    = [](PlanningCandidateId, PrivateSourceMode,
                                 std::span<const ninfer::runtime::PressureOwnerOutcome>)
        -> std::optional<Planner::LogicalGoal> {
        return Planner::LogicalGoal{.publication_slot = 0};
    };

    auto prefill_expensive                 = test_cost_model();
    prefill_expensive.prefill.token_ns_q32 = 100U * ninfer::runtime::kContextCostQ32One;
    Planner first_planner;
    const auto first =
        first_planner.plan(program, FakePreparedPrompt{}, prefill_expensive, candidates, 0,
                           pressure_inputs, logical_goal, 1'000'000U, Planner::Clock::now());

    auto transfer_expensive                 = test_cost_model();
    transfer_expensive.prefill.token_ns_q32 = ninfer::runtime::kContextCostQ32One;
    for (auto& direction : transfer_expensive.transfer) {
        direction.ns_per_byte_q32 = 100U * ninfer::runtime::kContextCostQ32One;
    }
    Planner second_planner;
    const auto second =
        second_planner.plan(program, FakePreparedPrompt{}, transfer_expensive, candidates, 0,
                            pressure_inputs, logical_goal, 1'000'000U, Planner::Clock::now());

    require(first && first->candidate == PlanningCandidateId{.value = 1} && second &&
                second->candidate == PlanningCandidateId{.value = 0} &&
                prefill_candidate.identity.physical_status ==
                    ninfer::runtime::MaterializationPhysicalStatus::Feasible &&
                transfer_candidate.identity.physical_status ==
                    ninfer::runtime::MaterializationPhysicalStatus::Feasible,
            "machine cost policy changed physical assessment or failed to change selection");
}

void test_candidate_search_prefers_deep_reuse_without_eviction() {
    using Planner = ninfer::runtime::MaterializationPlanner<FakePackage>;

    FakeProgram program;
    program.required_pressure_actions         = 2;
    program.eviction_pressure_action_units    = 2;
    program.pressure_action_immediate_ns      = 1'000'000;
    program.pressure_action_degradation_units = 1;
    program.pressure_assessment_delay_us      = 1'000;
    program.pressure_checkpoint_recovery_ns   = 8'000'000'000ULL;

    FakeAdmissionCandidate root;
    set_fake_machine_costs(root.identity.machine_work, 8'000'000'000ULL, 8'000'000'000ULL);
    root.identity.physical_status   = ninfer::runtime::MaterializationPhysicalStatus::Infeasible;
    root.identity.source_mode       = PrivateSourceMode::ConsumeToActive;
    root.identity.expandable        = true;
    root.identity.assessment_digest = 101;

    FakeAdmissionCandidate reuse;
    reuse.value.reusable_prompt_tokens = 55'048;
    reuse.private_source_id            = 1;
    set_fake_machine_costs(reuse.identity.machine_work, 100'000'000, 100'000'000);
    reuse.identity.machine_work.reused_prompt_tokens = 55'048;
    reuse.identity.physical_status   = ninfer::runtime::MaterializationPhysicalStatus::Infeasible;
    reuse.identity.source_mode       = PrivateSourceMode::ConsumeToActive;
    reuse.identity.expandable        = true;
    reuse.identity.assessment_digest = 202;

    std::array<Planner::CandidateInput, 2> candidates{
        Planner::CandidateInput{.candidate               = &root,
                                .id                      = PlanningCandidateId{.value = 0},
                                .stable_ordinal          = 0,
                                .current_session_binding = false},
        Planner::CandidateInput{.candidate               = &reuse,
                                .id                      = PlanningCandidateId{.value = 1},
                                .stable_ordinal          = 1,
                                .current_session_binding = true},
    };
    std::array<FakeContinuationHandle, 3> owner_handles{
        FakeContinuationHandle{1, 0},
        FakeContinuationHandle{2, 0},
        FakeContinuationHandle{3, 0},
    };
    const std::array<const FakeContinuationHandle*, 3> private_owners{
        &owner_handles[0], &owner_handles[1], &owner_handles[2]};
    const std::array<PlanningOwnerId, 3> private_owner_ids{
        PlanningOwnerId{.value = 0}, PlanningOwnerId{.value = 1}, PlanningOwnerId{.value = 2}};
    const std::array<ninfer::runtime::MaterializationOwnerPolicy, 3> owner_policy{
        ninfer::runtime::MaterializationOwnerPolicy{.owner           = PlanningOwnerId{.value = 0},
                                                    .retention_class = RetentionClass::LiveSession,
                                                    .private_retention_weight = 16},
        ninfer::runtime::MaterializationOwnerPolicy{.owner           = PlanningOwnerId{.value = 1},
                                                    .retention_class = RetentionClass::LiveSession,
                                                    .private_retention_weight = 16},
        ninfer::runtime::MaterializationOwnerPolicy{.owner           = PlanningOwnerId{.value = 2},
                                                    .retention_class = RetentionClass::LiveSession,
                                                    .private_retention_weight = 16},
    };
    const std::array<ninfer::runtime::MaterializationCheckpointPolicy, 3> checkpoint_policy{
        ninfer::runtime::MaterializationCheckpointPolicy{
            .owner      = PlanningOwnerId{.value = 0},
            .checkpoint = CheckpointRef{.kind     = CheckpointKind::SessionEndpoint,
                                        .frontier = 16,
                                        .ordinal  = 0},
            .rebuild_ns = 8'000'000'000ULL},
        ninfer::runtime::MaterializationCheckpointPolicy{
            .owner      = PlanningOwnerId{.value = 1},
            .checkpoint = CheckpointRef{.kind     = CheckpointKind::SessionEndpoint,
                                        .frontier = 16,
                                        .ordinal  = 0},
            .rebuild_ns = 8'000'000'000ULL},
        ninfer::runtime::MaterializationCheckpointPolicy{
            .owner      = PlanningOwnerId{.value = 2},
            .checkpoint = CheckpointRef{.kind     = CheckpointKind::SessionEndpoint,
                                        .frontier = 16,
                                        .ordinal  = 0},
            .rebuild_ns = 8'000'000'000ULL},
    };

    Planner planner;
    const auto pressure_inputs = [&]() -> Planner::PressureInputs {
        return Planner::PressureInputs{
            .private_owners    = private_owners,
            .private_owner_ids = private_owner_ids,
            .shared_owners     = {},
            .shared_owner_ids  = {},
            .owner_policy      = owner_policy,
            .checkpoint_policy = checkpoint_policy,
        };
    };
    const auto logical_goal = [](PlanningCandidateId, PrivateSourceMode,
                                 std::span<const ninfer::runtime::PressureOwnerOutcome>)
        -> std::optional<Planner::LogicalGoal> {
        return Planner::LogicalGoal{.publication_slot = 0};
    };
    auto result = planner.plan(program, FakePreparedPrompt{}, test_cost_model(), candidates, 0,
                               pressure_inputs, logical_goal, 1'000'000U, Planner::Clock::now());

    require(result && result->plan && result->candidate == PlanningCandidateId{.value = 1},
            "shallow Root pressure path starved the cheaper reuse candidate");
    require(result->plan->private_actions.size() == 2 &&
                std::none_of(
                    result->plan->private_actions.begin(), result->plan->private_actions.end(),
                    [](const FakeTargetDecision& action) { return action.evicts_continuation; }),
            "one-step eviction outranked the multi-step preserving reuse closure");
}

void test_feasible_identity_expands_when_pressure_can_remove_copy() {
    using Planner = ninfer::runtime::MaterializationPlanner<FakePackage>;

    FakeProgram program;
    program.pressure_action_immediate_ns          = 0;
    program.pressure_action_degradation_units     = 1;
    program.pressure_target_immediate_ns_override = 100'000'000;

    FakeAdmissionCandidate candidate;
    set_fake_machine_costs(candidate.identity.machine_work, 100'000'000, 1'000'000'000);
    candidate.identity.machine_work.candidate_transfers[2].copy_operations = 1;
    candidate.identity.pressure_may_change_machine_work                    = true;
    candidate.identity.physical_status   = ninfer::runtime::MaterializationPhysicalStatus::Feasible;
    candidate.identity.source_mode       = PrivateSourceMode::ConsumeToActive;
    candidate.identity.assessment_digest = 17;

    const std::array<Planner::CandidateInput, 1> candidates{
        Planner::CandidateInput{.candidate               = &candidate,
                                .id                      = PlanningCandidateId{.value = 0},
                                .stable_ordinal          = 0,
                                .current_session_binding = false},
    };
    FakeContinuationHandle owner{9, 0};
    const std::array<const FakeContinuationHandle*, 1> private_owners{&owner};
    const std::array<PlanningOwnerId, 1> private_owner_ids{PlanningOwnerId{.value = 0}};
    const std::array<ninfer::runtime::MaterializationOwnerPolicy, 1> owner_policy{
        ninfer::runtime::MaterializationOwnerPolicy{.owner = PlanningOwnerId{.value = 0}},
    };
    const std::array<ninfer::runtime::MaterializationCheckpointPolicy, 1> checkpoint_policy{
        ninfer::runtime::MaterializationCheckpointPolicy{
            .owner      = PlanningOwnerId{.value = 0},
            .checkpoint = CheckpointRef{.kind     = CheckpointKind::SessionEndpoint,
                                        .frontier = 16,
                                        .ordinal  = 0},
            .rebuild_ns = 100,
        },
    };

    Planner planner;
    const auto pressure_inputs = [&]() -> Planner::PressureInputs {
        return Planner::PressureInputs{
            .private_owners    = private_owners,
            .private_owner_ids = private_owner_ids,
            .shared_owners     = {},
            .shared_owner_ids  = {},
            .owner_policy      = owner_policy,
            .checkpoint_policy = checkpoint_policy,
        };
    };
    const auto logical_goal = [](PlanningCandidateId, PrivateSourceMode,
                                 std::span<const ninfer::runtime::PressureOwnerOutcome>)
        -> std::optional<Planner::LogicalGoal> {
        return Planner::LogicalGoal{.publication_slot = 0};
    };
    auto result = planner.plan(program, FakePreparedPrompt{}, test_cost_model(), candidates, 0,
                               pressure_inputs, logical_goal, 1'000'000U, Planner::Clock::now());

    require(result && result->diagnostics.predicted_now_ns == 100'000'000 &&
                result->diagnostics.selected_degradation_units == 1 &&
                program.pressure_planning_sessions == 1 && !program.seal_attempts.empty() &&
                program.seal_attempts.back() == std::vector<std::uint64_t>{1009},
            "feasible identity suppressed a cheaper complete pressure target");
}

void test_dominating_identity_does_not_build_pressure_graph() {
    using Planner = ninfer::runtime::MaterializationPlanner<FakePackage>;

    FakeProgram program;
    FakeAdmissionCandidate candidate;
    set_fake_machine_costs(candidate.identity.machine_work, 100'000'000, 100'000'000);
    candidate.identity.physical_status   = ninfer::runtime::MaterializationPhysicalStatus::Feasible;
    candidate.identity.source_mode       = PrivateSourceMode::ConsumeToActive;
    candidate.identity.assessment_digest = 23;
    const std::array<Planner::CandidateInput, 1> candidates{
        Planner::CandidateInput{.candidate               = &candidate,
                                .id                      = PlanningCandidateId{.value = 0},
                                .stable_ordinal          = 0,
                                .current_session_binding = false},
    };

    bool pressure_inputs_built = false;
    const auto pressure_inputs = [&]() -> Planner::PressureInputs {
        pressure_inputs_built = true;
        return {};
    };
    const auto logical_goal = [](PlanningCandidateId, PrivateSourceMode,
                                 std::span<const ninfer::runtime::PressureOwnerOutcome>)
        -> std::optional<Planner::LogicalGoal> {
        return Planner::LogicalGoal{.publication_slot = 0};
    };

    Planner planner;
    auto result = planner.plan(program, FakePreparedPrompt{}, test_cost_model(), candidates, 0,
                               pressure_inputs, logical_goal, 1'000'000U, Planner::Clock::now());
    require(result &&
                result->diagnostics.stop_reason == ninfer::MaterializationStopReason::NoPressure &&
                !pressure_inputs_built && program.pressure_planning_sessions == 0,
            "dominating identity eagerly constructed the pressure graph");
}

FakeFinishResult finish_active(FakeManager& manager, FakeProgram& program, ActiveRequest request,
                               std::uint32_t frontier = 16) {
    program.finish_frontier = frontier;
    manager.mark_terminal_pending(request.lane);
    return manager.finish(program, request.lane, request.sequence);
}

void test_root_lifecycle_and_prefix_reuse() {
    FakeManager manager = make_manager(1, 2);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 7, make_base(7), 1);
    require(program.pressure_planning_sessions == 0,
            "no-pressure root admission created a pressure planning session");
    const FakeFinishResult finish = finish_active(manager, program, first);
    require(finish.status == ConsumeStatus::Consumed &&
                finish.disposition == FinishDisposition::Catalogued,
            "terminal continuation was not catalogued");
    require(manager.lane_state(first.lane) == ninfer::runtime::LogicalLaneState::Free,
            "terminal lane did not return to Free");

    auto reuse = manager.inspect(program, FakePreparedPrompt{7}, make_base(7), 2);
    require(reuse.readiness == Readiness::Ready && reuse.choice,
            "catalogued endpoint was not reusable");
    require(reuse.choice->summary().reusable_prompt_tokens == 16,
            "endpoint reuse frontier was not selected");
    program.abort_start = true;
    const auto status   = manager.reserve_materialization(program, std::move(*reuse.choice),
                                                          FakePreparedPrompt{7}, {});
    require(status == FakeManager::MaterializationReserveResult::Stale &&
                program.started_source_id == first.sequence.id,
            "selected endpoint did not reach the sealed Program plan");
    require(manager.catalog_state(0) == FakeManager::CatalogState::Catalogued,
            "failed start did not roll back its logical source claim");
}

void test_capacity_miss_is_retryable() {
    FakeManager manager = make_manager(1, 2);
    FakeProgram program;
    const ActiveRequest seed = start_active(manager, program, 31, make_base(31), 1);
    (void)finish_active(manager, program, seed);

    auto inspection = manager.inspect(program, FakePreparedPrompt{31}, make_base(31), 2);
    require(inspection.choice.has_value(), "reuse choice was not produced");

    // Saturated pools: the Program cannot place the state this plan restores even after its release
    // ladder. Before 2026-09-22 that exception escaped to the engine worker, which wiped every
    // session and left the service answering 503 until restart.
    program.capacity_miss_on_start = true;
    const auto status = manager.reserve_materialization(program, std::move(*inspection.choice),
                                                        FakePreparedPrompt{31}, {});
    require(status == FakeManager::MaterializationReserveResult::Stale,
            "capacity miss was not reported as retryable work");
    require(manager.catalog_state(0) == FakeManager::CatalogState::Catalogued,
            "capacity miss did not roll back its logical reservation");

    // The retry re-plans against the pool the ladder produced and succeeds.
    program.capacity_miss_on_start = false;
    auto retry = manager.inspect(program, FakePreparedPrompt{31}, make_base(31), 3);
    require(retry.choice.has_value(), "retry produced no plan");
    const auto retried = manager.reserve_materialization(program, std::move(*retry.choice),
                                                         FakePreparedPrompt{31}, {});
    require(retried == FakeManager::MaterializationReserveResult::Reserved,
            "capacity miss left the manager unable to reserve after re-planning");
    auto outcome = [&]() -> FakeManager::MaterializationOutcome {
        auto progress = manager.progress_context_transaction(program, {});
        if (std::holds_alternative<ContextTransactionInProgress>(progress)) {
            auto completed = manager.progress_context_transaction(program, {});
            return std::get<FakeManager::MaterializationOutcome>(std::move(completed));
        }
        return std::get<FakeManager::MaterializationOutcome>(std::move(progress));
    }();
    require(outcome.status == ContextTransactionStatus::Published && outcome.activation,
            "re-planned materialization did not publish");
    auto activation = std::move(*outcome.activation);
    manager.adopt(program, std::move(activation));
}

void test_stale_revision_is_retryable() {
    FakeManager manager = make_manager();
    FakeProgram program;
    auto inspection = manager.inspect(program, FakePreparedPrompt{1}, make_base(1), 1);
    require(inspection.choice.has_value(), "root choice was not produced");
    const std::uint64_t start_calls = program.start_calls;
    program.invalidate_resources();
    const auto status = manager.reserve_materialization(program, std::move(*inspection.choice),
                                                        FakePreparedPrompt{1}, {});
    require(status == FakeManager::MaterializationReserveResult::Stale,
            "revision mismatch was not reported as retryable stale work");
    require(program.start_calls == start_calls,
            "known-stale plan was incorrectly passed into Program start");
    require(manager.lane_state(LaneId{0}) == ninfer::runtime::LogicalLaneState::Free,
            "known-stale plan changed logical lane state");
}

void test_retired_owner_is_not_offered_as_reuse_source() {
    FakeManager manager = make_manager(1, 2);
    FakeProgram program;
    const ActiveRequest seed      = start_active(manager, program, 7, make_base(7), 1);
    const FakeFinishResult finish = finish_active(manager, program, seed);
    require(finish.disposition == FinishDisposition::Catalogued,
            "seed continuation was not catalogued");
    require(manager.catalog_state(0) == FakeManager::CatalogState::Catalogued,
            "seed owner is not resident before retirement");
    // The Program reclaims capacity at reservation time for the newest request's own capture and may
    // retire an idle owner this catalog still lists. Offering that owner as a reuse source used to
    // abort the whole request ('admission source continuation is stale', 2026-09-22 harness); the
    // planning pass must repair the catalog instead.
    program.retired_continuation_ids.push_back(seed.sequence.id);
    const auto inspection = manager.inspect(program, FakePreparedPrompt{7}, make_base(7), 2);
    require(inspection.readiness == Readiness::Ready && inspection.choice,
            "retired owner made the request unplannable");
    require(manager.catalog_state(0) == FakeManager::CatalogState::Vacant,
            "retired owner was not cleared from the catalog");
    require(std::none_of(program.inspected_private_sources.begin(),
                         program.inspected_private_sources.end(),
                         [&](std::uint32_t id) { return id == seed.sequence.id; }),
            "retired owner was offered to the Program as a reuse source");
}

void test_materialization_abort_preserves_source() {
    FakeManager manager = make_manager(1, 2);
    FakeProgram program;
    const ActiveRequest seed = start_active(manager, program, 5, make_base(5), 1);
    (void)finish_active(manager, program, seed);

    auto inspection = manager.inspect(program, FakePreparedPrompt{5}, make_base(5), 2);
    require(inspection.choice && inspection.choice->summary().reusable_prompt_tokens == 16,
            "abort test did not select its private source");
    program.abort_progress = true;
    require(manager.reserve_materialization(program, std::move(*inspection.choice),
                                            FakePreparedPrompt{5}, {}) ==
                FakeManager::MaterializationReserveResult::Reserved,
            "abort test could not reserve materialization");
    auto progress = manager.progress_context_transaction(program, {});
    auto outcome  = std::get<FakeManager::MaterializationOutcome>(std::move(progress));
    require(outcome.status == ContextTransactionStatus::Aborted && !outcome.activation,
            "cancelled materialization did not abort before publication");
    require(manager.catalog_state(0) == FakeManager::CatalogState::Catalogued,
            "aborted Move did not restore its source visibility");

    program.abort_progress = false;
    program.abort_start    = true;
    auto retry             = manager.inspect(program, FakePreparedPrompt{5}, make_base(5), 3);
    require(retry.choice && retry.choice->summary().reusable_prompt_tokens == 16,
            "restored source was not reusable after abort");
    (void)manager.reserve_materialization(program, std::move(*retry.choice), FakePreparedPrompt{5},
                                          {});
    require(program.started_source_id == seed.sequence.id,
            "abort restored the wrong source capability");
}

void test_committed_victim_survives_transaction_abort() {
    FakeManager manager = make_manager(1, 2);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 10, make_base(10), 1);
    (void)finish_active(manager, program, first);
    const ActiveRequest second = start_active(manager, program, 20, make_base(20), 2);
    (void)finish_active(manager, program, second);

    auto inspection = manager.inspect(program, FakePreparedPrompt{30}, make_base(30), 3);
    require(inspection.choice.has_value(), "full catalog did not produce an eviction closure");
    program.abort_progress = true;
    require(manager.reserve_materialization(program, std::move(*inspection.choice),
                                            FakePreparedPrompt{30}, {}) ==
                FakeManager::MaterializationReserveResult::Reserved,
            "evicting materialization was not reserved");
    auto progress = manager.progress_context_transaction(program, {});
    auto outcome  = std::get<FakeManager::MaterializationOutcome>(std::move(progress));
    require(outcome.status == ContextTransactionStatus::Aborted,
            "pressure transaction did not take the abort path");
    const std::uint32_t catalogued =
        (manager.catalog_state(0) == FakeManager::CatalogState::Catalogued ? 1U : 0U) +
        (manager.catalog_state(1) == FakeManager::CatalogState::Catalogued ? 1U : 0U);
    require(catalogued == 1,
            "committed victim eviction was incorrectly rolled back with request-local abort");
}

void test_uncommitted_pressure_acknowledgement_is_not_degradation() {
    FakeManager manager = make_manager(1, 2);
    FakeProgram program;
    const ActiveRequest seed = start_active(manager, program, 40, make_base(40), 1);
    (void)finish_active(manager, program, seed);

    program.required_pressure_actions = 1;
    auto inspection = manager.inspect(program, FakePreparedPrompt{41}, make_base(41), 2);
    require(inspection.choice.has_value(),
            "pressure-abort test did not select a preserving pressure target");
    program.abort_progress = true;
    require(manager.reserve_materialization(program, std::move(*inspection.choice),
                                            FakePreparedPrompt{41}, {}) ==
                FakeManager::MaterializationReserveResult::Reserved,
            "pressure-abort test could not reserve materialization");
    require(program.started_action_ids.size() == 1 &&
                program.started_action_ids.front() == 1000U + seed.sequence.id,
            "pressure-abort test selected an eviction instead of preserving pressure");

    auto progress = manager.progress_context_transaction(program, {});
    auto outcome  = std::get<FakeManager::MaterializationOutcome>(std::move(progress));
    RuntimeStats stats;
    manager.populate_runtime_stats(program, stats);
    require(outcome.status == ContextTransactionStatus::Aborted &&
                manager.catalog_state(0) == FakeManager::CatalogState::Catalogued,
            "uncommitted pressure acknowledgement changed owner availability");
    require(stats.pressure_private_owners_degraded == 0 &&
                stats.pressure_private_owners_evicted == 0 &&
                stats.pressure_checkpoints_dropped == 0,
            "uncommitted pressure acknowledgement was counted as a degradation");
    require(stats.pressure_searches == 1,
            "accepted pressure plan was hidden when its request later aborted");
}

void test_aborted_source_selection_does_not_create_hit_history() {
    FakeManager manager = make_manager(1, 3);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 61, make_base(61), 1);
    (void)finish_active(manager, program, first);
    const ActiveRequest second = start_active(manager, program, 62, make_base(62), 2);
    (void)finish_active(manager, program, second);

    auto reuse = manager.inspect(program, FakePreparedPrompt{61}, make_base(61), 3);
    require(reuse.choice && reuse.choice->summary().reusable_prompt_tokens == 16,
            "cancelled-hit test did not select its exact source");
    program.abort_progress = true;
    require(manager.reserve_materialization(program, std::move(*reuse.choice),
                                            FakePreparedPrompt{61}, {}) ==
                FakeManager::MaterializationReserveResult::Reserved,
            "cancelled-hit test could not reserve exact reuse");
    auto progress = manager.progress_context_transaction(program, {});
    auto outcome  = std::get<FakeManager::MaterializationOutcome>(std::move(progress));
    require(outcome.status == ContextTransactionStatus::Aborted,
            "cancelled-hit test unexpectedly published its request");

    program.abort_progress            = false;
    program.required_pressure_actions = 1;
    program.require_evictions         = true;
    auto pressure = manager.inspect(program, FakePreparedPrompt{63}, make_base(63), 4);
    require(pressure.choice.has_value(), "cancelled-hit test could not plan pressure");
    program.abort_start = true;
    (void)manager.reserve_materialization(program, std::move(*pressure.choice),
                                          FakePreparedPrompt{63}, {});
    require(program.started_action_ids.size() == 1 &&
                program.started_action_ids.front() == 2000U + first.sequence.id,
            "aborted source selection incorrectly biased later retention policy");
}

void test_retained_source_is_protected_until_terminal() {
    FakeManager manager = make_manager(2, 3);
    FakeProgram program;
    const FakeCacheSessionKey first_session{1};
    const FakeCacheSessionKey second_session{2};
    const ActiveRequest seed = start_active(
        manager, program, 9, make_base(9, first_session, RetentionClass::LiveSession), 1);
    (void)finish_active(manager, program, seed);

    const ActiveRequest fork = start_active(
        manager, program, 9, make_base(9, second_session, RetentionClass::LiveSession), 2);
    require(program.started_source_mode == PrivateSourceMode::Retain,
            "different-session source was destructively moved");

    program.required_pressure_actions = 1;
    auto blocked = manager.inspect(program, FakePreparedPrompt{77}, make_base(77), 3);
    require(blocked.readiness == Readiness::TemporarilyBlocked && !blocked.choice,
            "active retained source was exposed as a pressure victim");

    (void)manager.abort(program, fork.lane, fork.sequence);
    program.abort_start = true;
    auto available      = manager.inspect(program, FakePreparedPrompt{77}, make_base(77), 4);
    require(available.choice.has_value(),
            "terminal release did not return retained source to pressure policy");
    (void)manager.reserve_materialization(program, std::move(*available.choice),
                                          FakePreparedPrompt{77}, {});
    require(!program.started_action_ids.empty(),
            "released source did not participate in the sealed pressure plan");
}

void test_session_publication_order_controls_tied_source() {
    FakeManager manager = make_manager(2, 3);
    FakeProgram program;
    const FakeCacheSessionKey session{42};
    const FakeRequestBasePlan base = make_base(42, session, RetentionClass::LiveSession, true);

    const ActiveRequest older = start_active(manager, program, 42, base, 10);
    const ActiveRequest newer = start_active(manager, program, 42, base, 20);
    (void)finish_active(manager, program, newer, 16);
    (void)finish_active(manager, program, older, 16);

    auto next = manager.inspect(program, FakePreparedPrompt{42}, base, 30);
    require(next.choice && next.choice->summary().reusable_prompt_tokens == 16,
            "same-session endpoint candidates were not reusable");
    program.abort_start = true;
    (void)manager.reserve_materialization(program, std::move(*next.choice), FakePreparedPrompt{42},
                                          {});
    require(program.started_source_id == newer.sequence.id,
            "older out-of-order finish displaced the current session binding on a tied cost");

    program.abort_start             = false;
    const ActiveRequest replacement = start_active(
        manager, program, 99, make_base(99, session, RetentionClass::LiveSession, true), 40);
    (void)finish_active(manager, program, replacement, 16);
    require(program.released_continuations.empty(),
            "session publication synchronously released an old physical continuation");
    require(manager.catalog_state(0) == FakeManager::CatalogState::Catalogued &&
                manager.catalog_state(1) == FakeManager::CatalogState::Catalogued &&
                manager.catalog_state(2) == FakeManager::CatalogState::Catalogued,
            "session replacement did not retain its old binding as anonymous cache");
}

void test_canonical_pressure_starts_with_disposable_owner() {
    FakeManager manager = make_manager(1, 3);
    FakeProgram program;
    const ActiveRequest disposable = start_active(
        manager, program, 1, make_base(1, std::nullopt, RetentionClass::Disposable), 1);
    (void)finish_active(manager, program, disposable);
    const ActiveRequest live = start_active(
        manager, program, 2, make_base(2, FakeCacheSessionKey{2}, RetentionClass::LiveSession), 2);
    (void)finish_active(manager, program, live);

    program.required_pressure_actions = 1;
    auto inspection = manager.inspect(program, FakePreparedPrompt{3}, make_base(3), 3);
    require(inspection.choice.has_value(), "canonical pressure did not find a feasible prefix");
    program.abort_start = true;
    (void)manager.reserve_materialization(program, std::move(*inspection.choice),
                                          FakePreparedPrompt{3}, {});
    require(program.started_action_ids.size() == 1 &&
                program.started_action_ids.front() == 1000U + disposable.sequence.id,
            "canonical pressure did not degrade Disposable before LiveSession");
}

void test_pressure_tries_every_preserving_alternative_before_eviction() {
    FakeManager manager = make_manager(1, 2);
    FakeProgram program;
    const ActiveRequest seed = start_active(manager, program, 15, make_base(15), 1);
    (void)finish_active(manager, program, seed);

    program.required_pressure_actions     = 1;
    program.private_pressure_alternatives = 2;
    program.required_action_id            = 11000U + seed.sequence.id;
    auto inspection = manager.inspect(program, FakePreparedPrompt{25}, make_base(25), 2);
    require(inspection.choice.has_value(), "second preserving pressure alternative was skipped");

    program.abort_start = true;
    (void)manager.reserve_materialization(program, std::move(*inspection.choice),
                                          FakePreparedPrompt{25}, {});
    require(program.started_action_ids.size() == 1 &&
                program.started_action_ids.front() == *program.required_action_id,
            "pressure escalated before trying the feasible preserving alternative");
}

void test_cumulative_owner_target_closes_pressure_without_eviction() {
    FakeManager manager = make_manager(1, 2);
    FakeProgram program;
    program.finish_with_rewrite = true;
    const ActiveRequest seed    = start_active(manager, program, 31, make_base(31), 1);
    (void)finish_active(manager, program, seed);

    program.required_pressure_actions         = 1;
    program.include_cumulative_private_target = true;
    program.required_action_id                = 5000U + seed.sequence.id;
    auto inspection = manager.inspect(program, FakePreparedPrompt{32}, make_base(32), 2);
    require(inspection.choice.has_value(),
            "cumulative checkpoint-drop and spill owner target was unreachable");

    program.abort_start = true;
    (void)manager.reserve_materialization(program, std::move(*inspection.choice),
                                          FakePreparedPrompt{32}, {});
    require(program.started_action_ids.size() == 1 &&
                program.started_action_ids.front() == *program.required_action_id,
            "planner replaced a feasible cumulative owner target with eviction");
}

void test_two_owners_jointly_close_pressure() {
    FakeManager manager = make_manager(1, 3);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 41, make_base(41), 1);
    (void)finish_active(manager, program, first);
    const ActiveRequest second = start_active(manager, program, 42, make_base(42), 2);
    (void)finish_active(manager, program, second);

    program.required_pressure_actions = 2;
    auto inspection = manager.inspect(program, FakePreparedPrompt{43}, make_base(43), 3);
    require(inspection.choice.has_value(), "joint two-owner pressure target was unreachable");

    program.abort_start = true;
    (void)manager.reserve_materialization(program, std::move(*inspection.choice),
                                          FakePreparedPrompt{43}, {});
    std::sort(program.started_action_ids.begin(), program.started_action_ids.end());
    const std::vector<std::uint64_t> expected{
        1000U + first.sequence.id,
        1000U + second.sequence.id,
    };
    require(program.started_action_ids == expected,
            "planner did not combine preserving targets from two owners");
}

void test_materialization_result_is_validated_before_any_adoption() {
    FakeManager manager = make_manager(1, 3);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 141, make_base(141), 1);
    (void)finish_active(manager, program, first);
    const ActiveRequest second = start_active(manager, program, 142, make_base(142), 2);
    (void)finish_active(manager, program, second);

    program.required_pressure_actions   = 2;
    program.malform_last_private_victim = true;
    auto inspection = manager.inspect(program, FakePreparedPrompt{143}, make_base(143), 3);
    require(inspection.choice.has_value(), "malformed-result fixture found no pressure plan");
    require(manager.reserve_materialization(program, std::move(*inspection.choice),
                                            FakePreparedPrompt{143}, {}) ==
                FakeManager::MaterializationReserveResult::Reserved,
            "malformed-result fixture could not reserve materialization");
    require(manager.catalog_state(0) == FakeManager::CatalogState::Claimed &&
                manager.catalog_state(1) == FakeManager::CatalogState::Claimed,
            "malformed-result fixture did not claim both owners");

    bool rejected = false;
    try {
        (void)manager.progress_context_transaction(program, {});
    } catch (const std::logic_error&) { rejected = true; }
    require(rejected, "malformed materialization result was accepted");
    require(manager.catalog_state(0) == FakeManager::CatalogState::Claimed &&
                manager.catalog_state(1) == FakeManager::CatalogState::Claimed,
            "malformed materialization result partially adopted an earlier victim");
}

void test_materialization_result_binds_exact_checkpoint_identity() {
    FakeManager manager = make_manager(1, 2);
    FakeProgram program;
    program.finish_with_rewrite = true;
    const ActiveRequest seed    = start_active(manager, program, 145, make_base(145), 1);
    (void)finish_active(manager, program, seed);

    program.required_pressure_actions           = 1;
    program.include_cumulative_private_target   = true;
    program.required_action_id                  = 5000U + seed.sequence.id;
    program.malform_private_checkpoint_identity = true;
    auto inspection = manager.inspect(program, FakePreparedPrompt{146}, make_base(146), 2);
    require(inspection.choice.has_value(), "checkpoint-identity fixture found no pressure plan");
    require(manager.reserve_materialization(program, std::move(*inspection.choice),
                                            FakePreparedPrompt{146}, {}) ==
                FakeManager::MaterializationReserveResult::Reserved,
            "checkpoint-identity fixture could not reserve materialization");

    bool rejected = false;
    try {
        (void)manager.progress_context_transaction(program, {});
    } catch (const std::logic_error&) { rejected = true; }
    require(rejected, "same-count checkpoint substitution was accepted");
    require(manager.catalog_state(0) == FakeManager::CatalogState::Claimed,
            "checkpoint substitution partially mutated its logical owner");
}

void test_materialization_result_is_adopted_by_owner_identity() {
    FakeManager manager = make_manager(1, 3);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 151, make_base(151), 1);
    (void)finish_active(manager, program, first);
    const ActiveRequest second = start_active(manager, program, 152, make_base(152), 2);
    (void)finish_active(manager, program, second);

    program.required_pressure_actions = 2;
    program.reverse_pressure_results  = true;
    const ActiveRequest published     = start_active(manager, program, 153, make_base(153), 3);
    require(manager.catalog_state(0) == FakeManager::CatalogState::Catalogued &&
                manager.catalog_state(1) == FakeManager::CatalogState::Catalogued,
            "reordered materialization results changed victim availability");
    (void)finish_active(manager, program, published);

    program.required_pressure_actions = 0;
    {
        auto reuse = manager.inspect(program, FakePreparedPrompt{151}, make_base(151), 4);
        require(reuse.choice && reuse.choice->summary().reusable_prompt_tokens == 16,
                "reordered materialization result attached the first summary to another owner");
    }
    {
        auto reuse = manager.inspect(program, FakePreparedPrompt{152}, make_base(152), 5);
        require(reuse.choice && reuse.choice->summary().reusable_prompt_tokens == 16,
                "reordered materialization result attached the second summary to another owner");
    }
}

void test_guided_pressure_reaches_deep_retention_before_capped_fallback() {
    constexpr std::size_t owner_count = 7;
    FakeManager manager               = make_manager(1, owner_count + 1U);
    FakeProgram program;
    std::array<std::uint32_t, owner_count> owner_ids{};
    for (std::size_t index = 0; index < owner_count; ++index) {
        const std::uint32_t content = static_cast<std::uint32_t>(70U + index);
        const ActiveRequest active =
            start_active(manager, program, content, make_base(content), index + 1U);
        owner_ids[index] = active.sequence.id;
        (void)finish_active(manager, program, active);
    }

    program.required_pressure_actions     = 3;
    program.private_pressure_alternatives = 4;
    program.pressure_assessment_delay_us  = 2'000;
    auto inspection = manager.inspect(program, FakePreparedPrompt{90}, make_base(90), 20);
    require(inspection.choice.has_value(), "guided pressure search found no admission plan");

    program.abort_start = true;
    (void)manager.reserve_materialization(program, std::move(*inspection.choice),
                                          FakePreparedPrompt{90}, {});
    require(program.started_action_ids.size() == program.required_pressure_actions,
            "guided pressure search selected maximal release instead of a retention closure");
    for (const std::uint32_t owner_id : owner_ids) {
        require(std::find(program.started_action_ids.begin(), program.started_action_ids.end(),
                          2000U + owner_id) == program.started_action_ids.end(),
                "guided pressure search evicted a parked owner");
    }
    // Planning effort is bounded by the search budget (15 s floor / 90 s cap) and, far before
    // that wall, by the finite target set: the fake charges 2 ms per assessment, so the budget
    // is non-binding here and the planner explores the reachable closure to queue exhaustion.
    // What this case must never do is walk every parked owner into a maximal drop (covered
    // above); the loose bound below only catches a runaway re-assessment loop.
    require(program.pressure_target_assessments <= 512,
            "guided pressure search assessed far beyond the reachable target set");
}

void test_combined_target_reprices_cancelled_pressure_copy() {
    FakeManager manager = make_manager(1, 3);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 51, make_base(51), 1);
    (void)finish_active(manager, program, first);
    const ActiveRequest second = start_active(manager, program, 52, make_base(52), 2);
    (void)finish_active(manager, program, second);

    program.required_pressure_actions             = 2;
    program.combined_target_cancels_pressure_copy = true;
    auto inspection = manager.inspect(program, FakePreparedPrompt{53}, make_base(53), 3);
    require(inspection.choice.has_value(),
            "non-monotonic complete target cost made the feasible combination unreachable");

    program.abort_start = true;
    (void)manager.reserve_materialization(program, std::move(*inspection.choice),
                                          FakePreparedPrompt{53}, {});
    std::sort(program.started_action_ids.begin(), program.started_action_ids.end());
    const std::vector<std::uint64_t> expected{
        1000U + first.sequence.id,
        1000U + second.sequence.id,
    };
    require(program.started_action_ids == expected,
            "planner accumulated parent transfer cost instead of repricing the complete target");
}

void test_in_progress_adoption_and_private_capture() {
    FakeManager manager = make_manager(1, 2);
    FakeProgram program;
    program.progress_in_progress_once = true;
    const ActiveRequest active        = start_active(manager, program, 12, make_base(12), 1);

    program.capture_assessment = FakeCaptureAssessment{
        .shortlist_key     = FakeShortlistKey{.digest = 12, .frontier = 24},
        .publishes_private = true,
    };
    program.capture_summary.endpoint = endpoint(12, 24);
    program.capture_summary.long_anchors.push_back(long_anchor(12, 16, 1));
    const auto reserved =
        manager.reserve_active_capture(program, active.lane, FakeCaptureOffer{.id = 1}, true, {});
    require(reserved == FakeManager::ActiveCaptureReserveResult::Reserved,
            "private capture was not reserved");
    auto progress = manager.progress_context_transaction(program, {});
    auto outcome  = std::get<FakeManager::ActiveCaptureOutcome>(std::move(progress));
    require(outcome.status == ContextTransactionStatus::Published &&
                !manager.context_transaction_kind(),
            "private capture was not adopted to a stable logical state");
    require(manager.lane_state(active.lane) == ninfer::runtime::LogicalLaneState::Active,
            "private capture disturbed active lane ownership");
    (void)finish_active(manager, program, active, 24);
}

void test_capture_planning_repairs_retired_owner() {
    FakeManager manager = make_manager(1, 2);
    FakeProgram program;
    // A catalogued owner that the Program retired while reclaiming capacity for an earlier capture:
    // the catalog still lists it, and before 2026-09-22 capture planning threw
    // "capture owner has no planning ID" for exactly this disagreement. That throw reached the
    // engine's fatal path, wiped every session and left the service answering 503 until restart.
    const ActiveRequest seed = start_active(manager, program, 21, make_base(21), 1);
    (void)finish_active(manager, program, seed);
    require(manager.catalog_state(0) == FakeManager::CatalogState::Catalogued,
            "seed owner is not catalogued before retirement");
    program.retired_continuation_ids.push_back(seed.sequence.id);

    const ActiveRequest active = start_active(manager, program, 22, make_base(22), 2);
    program.capture_assessment = FakeCaptureAssessment{
        .shortlist_key     = FakeShortlistKey{.digest = 22, .frontier = 24},
        .publishes_private = true,
    };
    program.capture_summary.endpoint = endpoint(22, 24);
    const auto reserved =
        manager.reserve_active_capture(program, active.lane, FakeCaptureOffer{.id = 5}, true, {});
    require(reserved == FakeManager::ActiveCaptureReserveResult::Reserved,
            "a retired owner made the capture unreservable");
    require(std::none_of(program.capture_private_owner_ids.begin(),
                         program.capture_private_owner_ids.end(),
                         [&](std::uint32_t id) { return id == seed.sequence.id; }),
            "capture planning offered a retired owner as a victim");
    auto outcome = [&]() -> FakeManager::ActiveCaptureOutcome {
        auto progress = manager.progress_context_transaction(program, {});
        if (std::holds_alternative<ContextTransactionInProgress>(progress)) {
            auto completed = manager.progress_context_transaction(program, {});
            return std::get<FakeManager::ActiveCaptureOutcome>(std::move(completed));
        }
        return std::get<FakeManager::ActiveCaptureOutcome>(std::move(progress));
    }();
    require(outcome.status == ContextTransactionStatus::Published,
            "capture did not publish after the retired owner was repaired");
    (void)finish_active(manager, program, active, 24);
}

void test_projected_nested_shared_candidates_use_marginal_value() {
    FakeManager manager = make_manager(1, 2, 2);
    FakeProgram program;
    FakeRequestBasePlan base = make_base(61);
    base.cache.opportunities = {
        FakeContextCache::Opportunity{
            .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence = ninfer::SharedCandidateEvidence::EngineStructural,
            .frontier = 32,
        },
        FakeContextCache::Opportunity{
            .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence = ninfer::SharedCandidateEvidence::EngineStructural,
            .frontier = 64,
        },
    };
    const ActiveRequest active = start_active(manager, program, 61, base, 1);
    require(program.selected_shared_capture_frontiers == std::vector<std::uint32_t>{64},
            "nested shared candidates were selected independently instead of by marginal value");
    (void)finish_active(manager, program, active);
}

void test_observed_shared_candidate_requires_independent_domains() {
    const auto observed_base = [](std::optional<FakeCacheSessionKey> session) {
        FakeRequestBasePlan base = make_base(71, session);
        base.cache.opportunities.push_back(FakeContextCache::Opportunity{
            .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence = ninfer::SharedCandidateEvidence::EngineObserved,
            .frontier = 64,
        });
        return base;
    };

    {
        FakeManager manager = make_manager(1, 3, 1);
        FakeProgram program;
        const ActiveRequest first = start_active(manager, program, 71, observed_base({}), 1);
        require(program.selected_shared_capture_frontiers.empty(),
                "first stateless observation was treated as independent reuse");
        (void)finish_active(manager, program, first);
        const ActiveRequest second = start_active(manager, program, 71, observed_base({}), 2);
        require(program.selected_shared_capture_frontiers == std::vector<std::uint32_t>{64},
                "two stateless observations did not establish independent reuse demand");
        (void)finish_active(manager, program, second);
    }
    {
        FakeManager manager = make_manager(1, 3, 1);
        FakeProgram program;
        const FakeCacheSessionKey session{.value = 9};
        const ActiveRequest first = start_active(manager, program, 71, observed_base(session), 1);
        (void)finish_active(manager, program, first);
        const ActiveRequest second = start_active(manager, program, 71, observed_base(session), 2);
        require(program.selected_shared_capture_frontiers.empty(),
                "same-session replay was misclassified as shared fanout demand");
        (void)finish_active(manager, program, second);
    }
}

void test_repeated_private_reuse_selects_zero_prefill_shared_promotion() {
    const auto observed_base = [] {
        FakeRequestBasePlan base = make_base(81);
        base.cache.opportunities.push_back(FakeContextCache::Opportunity{
            .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence = ninfer::SharedCandidateEvidence::EngineObserved,
            .frontier = 16,
        });
        return base;
    };

    FakeManager manager = make_manager(1, 3, 1);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 81, observed_base(), 1);
    require(program.selected_shared_capture_frontiers.empty(),
            "first observation selected an unsupported shared promotion");
    (void)finish_active(manager, program, first, 16);

    const ActiveRequest second = start_active(manager, program, 81, observed_base(), 2);
    require(program.selected_shared_capture_frontiers == std::vector<std::uint32_t>{16},
            "repeated private base was not selected for zero-prefill shared promotion");
    (void)finish_active(manager, program, second, 16);
}

void test_shared_fanout_keeps_owner_edges_live_across_summary_refresh() {
    FakeManager manager = make_manager(2, 3, 1);
    FakeProgram program;

    FakeRequestBasePlan seed_base = make_base(91);
    seed_base.cache.opportunities.push_back(FakeContextCache::Opportunity{
        .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
        .evidence = ninfer::SharedCandidateEvidence::ExplicitBoundary,
        .frontier = 64,
    });
    const ActiveRequest seed   = start_active(manager, program, 91, seed_base, 1);
    program.capture_assessment = FakeCaptureAssessment{
        .shortlist_key          = FakeShortlistKey{.digest = 91, .frontier = 64},
        .shared_evidence        = ninfer::SharedCandidateEvidence::ExplicitBoundary,
        .protected_rebuild_work = PrefillWork{.tokens = 64},
        .publishes_shared       = true,
        .physically_feasible    = true,
    };
    require(manager.reserve_active_capture(program, seed.lane, FakeCaptureOffer{.id = 31}, 0, {}) ==
                FakeManager::ActiveCaptureReserveResult::Reserved,
            "shared-fanout fixture could not publish its source");
    auto capture_progress = manager.progress_context_transaction(program, {});
    const auto capture = std::get<FakeManager::ActiveCaptureOutcome>(std::move(capture_progress));
    require(capture.status == ContextTransactionStatus::Published,
            "shared-fanout fixture did not publish its source");
    (void)finish_active(manager, program, seed);

    program.report_shared_source_summary                    = true;
    program.change_shared_source_residency_on_second_report = true;
    program.reported_shared_active_references               = 0;
    const ActiveRequest first  = start_active(manager, program, 91, make_base(91), 2);
    const ActiveRequest second = start_active(manager, program, 91, make_base(91), 3);

    RuntimeStats stats;
    manager.populate_runtime_stats(program, stats);
    require(stats.shared_active_references == 2,
            "two shared branches did not retain two logical owner edges");
    (void)finish_active(manager, program, first);
    manager.populate_runtime_stats(program, stats);
    require(stats.shared_active_references == 1,
            "first shared branch release invalidated the surviving owner edge");
    (void)finish_active(manager, program, second);
    manager.populate_runtime_stats(program, stats);
    require(stats.shared_active_references == 0,
            "shared fanout did not release both logical owner edges");
}

void test_shared_capture_combines_two_pressure_owners() {
    // Capture pressure against private owners: fair-share protection off, the bucket behavior
    // is covered by test_fair_share_capture_pressure_cannot_touch_protected_sessions.
    FakeManager manager = make_manager(1, 4, 1, true, 0);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 41, make_base(41), 1);
    (void)finish_active(manager, program, first);
    const ActiveRequest second = start_active(manager, program, 42, make_base(42), 2);
    (void)finish_active(manager, program, second);

    FakeRequestBasePlan shared_request = make_base(43);
    shared_request.cache.opportunities.push_back(FakeContextCache::Opportunity{
        .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
        .evidence = ninfer::SharedCandidateEvidence::ExplicitBoundary,
        .frontier = 64,
    });
    const ActiveRequest active           = start_active(manager, program, 43, shared_request, 3);
    program.required_pressure_actions    = 2;
    program.pressure_action_immediate_ns = 0;
    program.capture_assessment           = FakeCaptureAssessment{
                  .shortlist_key          = FakeShortlistKey{.digest = 43, .frontier = 64},
                  .shared_evidence        = ninfer::SharedCandidateEvidence::ExplicitBoundary,
                  .protected_rebuild_work = PrefillWork{.tokens = 64},
                  .publishes_shared       = true,
                  .physically_feasible    = false,
    };

    const auto reserved =
        manager.reserve_active_capture(program, active.lane, FakeCaptureOffer{.id = 7}, 0, {});
    require(reserved == FakeManager::ActiveCaptureReserveResult::Reserved,
            "shared capture did not reserve a multi-owner pressure target");
    auto progress      = manager.progress_context_transaction(program, {});
    const auto outcome = std::get<FakeManager::ActiveCaptureOutcome>(std::move(progress));
    require(outcome.status == ContextTransactionStatus::Published &&
                program.started_action_ids.size() == 2,
            "shared capture did not publish the selected two-owner target");
    RuntimeStats stats;
    manager.populate_runtime_stats(program, stats);
    require(stats.shared_active_references == 1,
            "shared capture publication did not retain the active owner reference");
    program.required_pressure_actions = 0;
    (void)finish_active(manager, program, active);
}

void test_aborted_shared_capture_start_rolls_back_logical_claims() {
    // Fair-share protection off: this fixture verifies claim rollback, not bucket behavior.
    FakeManager manager = make_manager(1, 4, 1, true, 0);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 141, make_base(141), 1);
    (void)finish_active(manager, program, first);
    const ActiveRequest second = start_active(manager, program, 142, make_base(142), 2);
    (void)finish_active(manager, program, second);

    FakeRequestBasePlan request = make_base(143);
    request.cache.opportunities.push_back(FakeContextCache::Opportunity{
        .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
        .evidence = ninfer::SharedCandidateEvidence::ExplicitBoundary,
        .frontier = 64,
    });
    const ActiveRequest active           = start_active(manager, program, 143, request, 3);
    program.required_pressure_actions    = 2;
    program.pressure_action_immediate_ns = 0;
    program.capture_assessment           = FakeCaptureAssessment{
                  .shortlist_key          = FakeShortlistKey{.digest = 143, .frontier = 64},
                  .shared_evidence        = ninfer::SharedCandidateEvidence::ExplicitBoundary,
                  .protected_rebuild_work = PrefillWork{.tokens = 64},
                  .publishes_shared       = true,
                  .physically_feasible    = false,
    };
    program.abort_capture_start = true;

    const auto reserved =
        manager.reserve_active_capture(program, active.lane, FakeCaptureOffer{.id = 13}, 0, {});
    require(reserved == FakeManager::ActiveCaptureReserveResult::Skipped &&
                !manager.context_transaction_kind() && !program.has_context_transaction(),
            "aborted shared capture start retained transaction ownership");
    require(manager.catalog_state(0) == FakeManager::CatalogState::Catalogued &&
                manager.catalog_state(1) == FakeManager::CatalogState::Catalogued,
            "aborted shared capture start leaked logical victim claims");

    program.abort_capture_start = false;
    const auto retried =
        manager.reserve_active_capture(program, active.lane, FakeCaptureOffer{.id = 14}, 0, {});
    require(retried == FakeManager::ActiveCaptureReserveResult::Reserved,
            "aborted shared capture start leaked the shared publication slot");
    auto progress      = manager.progress_context_transaction(program, {});
    const auto outcome = std::get<FakeManager::ActiveCaptureOutcome>(std::move(progress));
    require(outcome.status == ContextTransactionStatus::Published,
            "shared capture retry did not publish after rollback");

    program.required_pressure_actions = 0;
    (void)finish_active(manager, program, active);
}

void test_capture_result_is_validated_before_any_adoption() {
    // Fair-share protection off: this fixture verifies result validation, not bucket behavior.
    FakeManager manager = make_manager(1, 4, 1, true, 0);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 241, make_base(241), 1);
    (void)finish_active(manager, program, first);
    const ActiveRequest second = start_active(manager, program, 242, make_base(242), 2);
    (void)finish_active(manager, program, second);

    FakeRequestBasePlan request = make_base(243);
    request.cache.opportunities.push_back(FakeContextCache::Opportunity{
        .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
        .evidence = ninfer::SharedCandidateEvidence::ExplicitBoundary,
        .frontier = 64,
    });
    const ActiveRequest active           = start_active(manager, program, 243, request, 3);
    program.required_pressure_actions    = 2;
    program.pressure_action_immediate_ns = 0;
    program.capture_assessment           = FakeCaptureAssessment{
                  .shortlist_key          = FakeShortlistKey{.digest = 243, .frontier = 64},
                  .shared_evidence        = ninfer::SharedCandidateEvidence::ExplicitBoundary,
                  .protected_rebuild_work = PrefillWork{.tokens = 64},
                  .publishes_shared       = true,
                  .physically_feasible    = false,
    };
    program.malform_last_capture_private_victim = true;

    require(manager.reserve_active_capture(program, active.lane, FakeCaptureOffer{.id = 17}, 0,
                                           {}) == FakeManager::ActiveCaptureReserveResult::Reserved,
            "malformed capture fixture could not reserve pressure");
    require(manager.catalog_state(0) == FakeManager::CatalogState::Claimed &&
                manager.catalog_state(1) == FakeManager::CatalogState::Claimed,
            "malformed capture fixture did not claim both owners");

    bool rejected = false;
    try {
        (void)manager.progress_context_transaction(program, {});
    } catch (const std::logic_error&) { rejected = true; }
    require(rejected, "malformed capture result was accepted");
    require(manager.catalog_state(0) == FakeManager::CatalogState::Claimed &&
                manager.catalog_state(1) == FakeManager::CatalogState::Claimed,
            "malformed capture result partially adopted an earlier victim");
}

void test_capture_result_is_adopted_by_owner_identity() {
    // Fair-share protection off: this fixture verifies adoption identity, not bucket behavior.
    FakeManager manager = make_manager(1, 4, 1, true, 0);
    FakeProgram program;
    const ActiveRequest first = start_active(manager, program, 251, make_base(251), 1);
    (void)finish_active(manager, program, first);
    const ActiveRequest second = start_active(manager, program, 252, make_base(252), 2);
    (void)finish_active(manager, program, second);

    FakeRequestBasePlan request = make_base(253);
    request.cache.opportunities.push_back(FakeContextCache::Opportunity{
        .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
        .evidence = ninfer::SharedCandidateEvidence::ExplicitBoundary,
        .frontier = 64,
    });
    const ActiveRequest active           = start_active(manager, program, 253, request, 3);
    program.required_pressure_actions    = 2;
    program.pressure_action_immediate_ns = 0;
    program.reverse_pressure_results     = true;
    program.capture_assessment           = FakeCaptureAssessment{
                  .shortlist_key          = FakeShortlistKey{.digest = 253, .frontier = 64},
                  .shared_evidence        = ninfer::SharedCandidateEvidence::ExplicitBoundary,
                  .protected_rebuild_work = PrefillWork{.tokens = 64},
                  .publishes_shared       = true,
                  .physically_feasible    = false,
    };

    require(manager.reserve_active_capture(program, active.lane, FakeCaptureOffer{.id = 27}, 0,
                                           {}) == FakeManager::ActiveCaptureReserveResult::Reserved,
            "reordered capture fixture could not reserve pressure");
    auto progress      = manager.progress_context_transaction(program, {});
    const auto outcome = std::get<FakeManager::ActiveCaptureOutcome>(std::move(progress));
    require(outcome.status == ContextTransactionStatus::Published &&
                manager.catalog_state(0) == FakeManager::CatalogState::Catalogued &&
                manager.catalog_state(1) == FakeManager::CatalogState::Catalogued,
            "reordered capture results changed victim availability");
    program.required_pressure_actions = 0;
    (void)finish_active(manager, program, active);

    {
        auto reuse = manager.inspect(program, FakePreparedPrompt{251}, make_base(251), 4);
        require(reuse.choice && reuse.choice->summary().reusable_prompt_tokens == 16,
                "reordered capture result attached the first summary to another owner");
    }
    {
        auto reuse = manager.inspect(program, FakePreparedPrompt{252}, make_base(252), 5);
        require(reuse.choice && reuse.choice->summary().reusable_prompt_tokens == 16,
                "reordered capture result attached the second summary to another owner");
    }
}

void test_terminal_fallback_releases_failed_retention() {
    FakeManager manager = make_manager(1, 1);
    FakeProgram program;
    const ActiveRequest active = start_active(manager, program, 4, make_base(4), 1);
    manager.mark_terminal_pending(active.lane);
    program.finish_fail_next      = true;
    const FakeFinishResult result = manager.finish(program, active.lane, active.sequence);
    require(result.status == ConsumeStatus::Consumed &&
                result.disposition == FinishDisposition::Released,
            "failed retention did not converge to a released terminal result");
    require(result.timings.value == 7 && result.speculative.value == 9,
            "terminal fallback lost abort accounting");
    require(program.abort_calls == 1 &&
                manager.lane_state(active.lane) == ninfer::runtime::LogicalLaneState::Free &&
                manager.catalog_state(0) == FakeManager::CatalogState::Vacant,
            "terminal fallback did not free every logical owner");
}

void test_terminal_settlement_waits_for_open_resource_transaction() {
    FakeManager manager = make_manager(2, 3);
    FakeProgram program;
    const ActiveRequest active = start_active(manager, program, 31, make_base(31), 1);

    auto inspection = manager.inspect(program, FakePreparedPrompt{32}, make_base(32), 2);
    require(inspection.choice.has_value(), "concurrent materialization choice was not produced");
    require(manager.reserve_materialization(program, std::move(*inspection.choice),
                                            FakePreparedPrompt{32}, {}) ==
                FakeManager::MaterializationReserveResult::Reserved,
            "concurrent materialization was not reserved");

    const auto capture =
        manager.reserve_active_capture(program, active.lane, FakeCaptureOffer{.id = 9}, true, {});
    require(capture == FakeManager::ActiveCaptureReserveResult::Skipped &&
                program.skipped_captures == 1 &&
                manager.context_transaction_kind() ==
                    ninfer::runtime::ContextTransactionKind::Materialization,
            "optional capture was not skipped behind the open materialization");

    manager.mark_terminal_pending(active.lane);
    bool rejected = false;
    try {
        (void)manager.finish(program, active.lane, active.sequence);
    } catch (const std::logic_error&) { rejected = true; }
    require(rejected && manager.lane_state(active.lane) ==
                            ninfer::runtime::LogicalLaneState::TerminalPending,
            "terminal settlement changed topology during an open resource transaction");
}

void test_commit_and_discard_terminal_states() {
    FakeManager manager = make_manager(2, 2);
    FakeProgram program;
    const ActiveRequest first  = start_active(manager, program, 1, make_base(1), 1);
    const ActiveRequest second = start_active(manager, program, 2, make_base(2), 2);
    const std::array<LaneId, 2> lanes{first.lane, second.lane};
    FakeCommitResult commit;
    commit.row_count           = 2;
    commit.rows[0].disposition = CommitDisposition::Active;
    commit.rows[1].disposition = CommitDisposition::Finishable;
    manager.apply_commit(lanes, commit);
    require(manager.lane_state(first.lane) == ninfer::runtime::LogicalLaneState::Active &&
                manager.lane_state(second.lane) ==
                    ninfer::runtime::LogicalLaneState::TerminalPending,
            "row-aligned commit did not establish terminal pending state");
    (void)manager.finish(program, second.lane, second.sequence);

    FakeDiscardResult discard{.status = ConsumeStatus::Consumed, .row_count = 1};
    const std::array<LaneId, 1> remaining{first.lane};
    manager.apply_discard(remaining, discard);
    require(manager.lane_state(first.lane) == ninfer::runtime::LogicalLaneState::Free,
            "discard did not release cancelled active membership");
}

void test_backfill_proof_and_stats_follow_program_revision() {
    FakeManager manager = make_manager();
    FakeProgram program;
    auto inspection = manager.inspect(program, FakePreparedPrompt{8}, make_base(8), 1);
    require(inspection.choice.has_value(), "backfill test did not produce a candidate plan");
    const std::array<FakeSequenceHandle, 0> borrowers{};
    auto proof =
        manager.prove_persistent_backfill(program, make_base(99), *inspection.choice, borrowers);
    require(proof && proof->resource_revision() == program.resource_revision(),
            "Program did not seal a current-revision persistent proof");
    program.invalidate_resources();
    proof =
        manager.prove_persistent_backfill(program, make_base(99), *inspection.choice, borrowers);
    require(!proof, "resource revision change did not invalidate persistent proof");

    program.usage = FakePhysicalUsage{
        .device_state_slots      = 3,
        .host_state_slots        = 2,
        .device_main_kv_pages    = 11,
        .device_backend_kv_pages = 5,
        .host_kv_bytes           = 4096,
    };
    RuntimeStats stats;
    manager.populate_runtime_stats(program, stats);
    require(stats.device_state_occupied_slots == 3 && stats.host_state_occupied_slots == 2 &&
                stats.device_main_kv_occupied_pages == 11 &&
                stats.device_backend_kv_occupied_pages == 5 && stats.host_kv_occupied_bytes == 4096,
            "runtime physical gauges did not come directly from Program");
}

void test_shortlist_collision_requires_program_exact_verification() {
    FakeManager manager = make_manager(1, 2);
    FakeProgram program;
    const ActiveRequest seed = start_active(manager, program, 55, make_base(123), 1);
    (void)finish_active(manager, program, seed);

    auto collision = manager.inspect(program, FakePreparedPrompt{99}, make_base(55), 2);
    require(collision.choice && collision.choice->summary().reusable_prompt_tokens == 0,
            "shortlist collision bypassed Program exact identity verification");
}

// 2026-09-12 17:09 incident class: an idle deep session lost its endpoint to an active
// neighbor's growth/compaction churn and paid a full re-prefill on return. With the default
// fair-share buckets both idle sessions here are protected, i.e. ranked above every unprotected
// record; between two equally valuable protected sessions the one chain falls through to age, so
// the OLDEST is sacrificed and the MRU session's endpoint survives the churn (缓存模块v2.md
// §2.2: protection is a value - the old exclusion + bucket-release path is gone).
void test_fair_share_sacrifices_the_oldest_protected_session_last() {
    FakeManager manager = make_manager(1, 4); // default fair_share_buckets = 8
    FakeProgram program;
    const ActiveRequest older = start_active(
        manager, program, 401,
        make_base(401, FakeCacheSessionKey{401}, RetentionClass::LiveSession), 1);
    (void)finish_active(manager, program, older);
    const ActiveRequest newer = start_active(
        manager, program, 402,
        make_base(402, FakeCacheSessionKey{402}, RetentionClass::LiveSession), 2);
    (void)finish_active(manager, program, newer);

    // One pressure action is required and only an owner release can provide it: both sessions are
    // protected, so the plan must take the one the chain ranks cheapest - the older of two equal
    // scores - and never the MRU session.
    program.required_pressure_actions = 1;
    auto inspection = manager.inspect(program, FakePreparedPrompt{403}, make_base(403), 3);
    require(inspection.choice.has_value(),
            "a protected-session release must still be planned when nothing else can answer");

    program.abort_start = true;
    (void)manager.reserve_materialization(program, std::move(*inspection.choice),
                                          FakePreparedPrompt{403}, {});
    require(program.started_action_ids.size() == 1 &&
                program.started_action_ids.front() == 1000U + older.sequence.id,
            "bucket release sacrificed a session other than the oldest protected one");
    require(manager.catalog_state(1) == FakeManager::CatalogState::Catalogued,
            "MRU bucket lost its endpoint to a neighbor's pressure");
}

// The manager owns the victim value model; the Program's last-resort release step runs where that
// model is unavailable, so the order has to cross the boundary before any call that can reach the
// ladder. Without this push the fallback silently reverts to oldest-touched, which is how a
// 237k-token conversation died while freshly created test sessions stayed (2026-09-23).
void test_manager_hands_the_retire_preference_to_the_program() {
    FakeManager manager = make_manager(1, 4);
    FakeProgram program;
    const ActiveRequest first = start_active(
        manager, program, 501,
        make_base(501, FakeCacheSessionKey{501}, RetentionClass::LiveSession), 1);
    (void)finish_active(manager, program, first);
    const ActiveRequest second = start_active(
        manager, program, 502,
        make_base(502, FakeCacheSessionKey{502}, RetentionClass::LiveSession), 2);
    (void)finish_active(manager, program, second);

    program.retire_preference.clear();
    (void)manager.inspect(program, FakePreparedPrompt{503}, make_base(503), 3);
    require(!program.retire_preference.empty(),
            "planning did not hand a retire preference to the Program");
    require(std::is_sorted(program.retire_preference.begin(), program.retire_preference.end(),
                           [](const RetirePreferenceEntry& left,
                              const RetirePreferenceEntry& right) {
                               return left.score_ns < right.score_ns;
                           }),
            "the retire preference is not ordered by score");
    std::vector<std::uint32_t> private_slots;
    for (const RetirePreferenceEntry& entry : program.retire_preference) {
        if (!entry.shared_prefix) { private_slots.push_back(entry.slot); }
    }
    require(private_slots.size() >= 2,
            "the retire preference does not cover the catalogued sessions");
    std::sort(private_slots.begin(), private_slots.end());
    require(std::adjacent_find(private_slots.begin(), private_slots.end()) == private_slots.end(),
            "the retire preference names a slot twice");

    program.retire_preference.clear();
    const ActiveRequest third = start_active(
        manager, program, 504,
        make_base(504, FakeCacheSessionKey{504}, RetentionClass::LiveSession), 4);
    (void)manager.reserve_active_capture(program, third.lane, FakeCaptureOffer{.id = 9}, true, {});
    require(!program.retire_preference.empty(),
            "capture reservation did not hand a retire preference to the Program");
}

// §2.1 / §三 R0: what an admission works on is working set, not cache. The order handed to the
// ladder at RESERVE time must therefore leave the choice's own private source out of the victim
// set - the reservation is about to restore FROM that owner, so a ladder that picked it would
// delete the very checkpoint the plan priced. Measured on the rig (fork_hit 2026-09-28): the
// conversation's own grow ran `[cache] R0 relief` -> `[ladder] degrade anchor slot=33 frontier=733`,
// and the fork three requests later had nothing below its fork point (`fork@4 cached=0`, root
// re-prefill). Excluding the ONE source is not a way to empty the domain: every other owner stays
// in the order (protection is ordering; only the in-flight source is excluded).
void test_reserve_order_excludes_its_own_private_source() {
    FakeManager manager = make_manager(1, 4);
    FakeProgram program;
    const ActiveRequest first = start_active(
        manager, program, 601,
        make_base(601, FakeCacheSessionKey{601}, RetentionClass::LiveSession), 1);
    (void)finish_active(manager, program, first);
    require(manager.catalog_state(0) == FakeManager::CatalogState::Catalogued,
            "the first session did not take catalog slot 0");
    const ActiveRequest other = start_active(
        manager, program, 602,
        make_base(602, FakeCacheSessionKey{602}, RetentionClass::LiveSession), 2);
    (void)finish_active(manager, program, other);

    auto reuse = manager.inspect(program, FakePreparedPrompt{601}, make_base(601), 3);
    require(reuse.readiness == Readiness::Ready && reuse.choice,
            "the catalogued session was not reusable");
    require(reuse.choice->summary().reusable_prompt_tokens > 0,
            "the reuse choice carries no reusable prefix");

    program.retire_preference.clear();
    (void)manager.reserve_materialization(program, std::move(*reuse.choice),
                                          FakePreparedPrompt{601}, {});
    require(!program.retire_preference.empty(),
            "reserve did not hand a retire preference to the Program");
    require(std::find(program.retire_excluded.begin(), program.retire_excluded.end(), 0u) !=
                program.retire_excluded.end(),
            "reserve did not declare its own source as excluded from the R2 walks");
    bool other_owner_kept = false;
    for (const RetirePreferenceEntry& entry : program.retire_preference) {
        if (entry.shared_prefix) { continue; }
        require(entry.slot != 0,
                ("the reserve handed its own reuse source (slot 0) to the R2 walk (slot=" +
                 std::to_string(entry.slot) + ")").c_str());
        if (entry.slot == 1) { other_owner_kept = true; }
    }
    require(other_owner_kept,
            "the reserve emptied the victim domain instead of excluding its one source");
}

// Victim ordering must be one combined score, not a lexicographic field chain. The chain's first
// field was the reuse count, so an owner that had not been re-read yet sorted as the cheapest
// victim however deep it was: on 2026-09-23 that evicted a 62k-token conversation which had been
// idle since its last turn while shallow probe sessions a test loop had just hammered stayed
// resident. The score multiplies the value at risk by reuse evidence, so depth, retention class,
// and recency all contribute to the same number.
void test_victim_score_ranks_by_value_not_by_one_field() {
    const auto now = std::chrono::steady_clock::now();
    constexpr std::uint64_t kSecond = 1000000000ULL;
    const std::array owners{
        // Deep conversation that nobody has re-read yet: 60 s of rebuild work at RecentPrivate.
        MaterializationOwnerPolicy{
            .owner                    = PlanningOwnerId{.value = 0},
            .retention_class          = RetentionClass::RecentPrivate,
            .private_retention_weight = 4,
            .reuse_evidence_q16       = reuse_evidence_q16(0, {}, now),
        },
        // Shallow probe session a test loop has just hammered twenty times: 4 s of rebuild work.
        MaterializationOwnerPolicy{
            .owner                    = PlanningOwnerId{.value = 1},
            .retention_class          = RetentionClass::RecentPrivate,
            .selected_hit_count       = 20,
            .private_retention_weight = 4,
            .reuse_evidence_q16       = reuse_evidence_q16(20, now, now),
        },
        // Same value at risk as the deep session, but a LiveSession retention class.
        MaterializationOwnerPolicy{
            .owner                    = PlanningOwnerId{.value = 2},
            .retention_class          = RetentionClass::LiveSession,
            .private_retention_weight = 16,
            .reuse_evidence_q16       = reuse_evidence_q16(0, {}, now),
        },
    };
    const std::array checkpoints{
        MaterializationCheckpointPolicy{
            .owner = PlanningOwnerId{.value = 0}, .rebuild_ns = 60U * kSecond},
        MaterializationCheckpointPolicy{
            .owner = PlanningOwnerId{.value = 1}, .rebuild_ns = 4U * kSecond},
        MaterializationCheckpointPolicy{
            .owner = PlanningOwnerId{.value = 2}, .rebuild_ns = 60U * kSecond},
    };
    const auto score = [&](std::uint32_t owner) {
        return materialization_victim_score(owners, checkpoints, PlanningOwnerId{.value = owner});
    };

    require(score(0) > score(1),
            "a deep session that has not been re-read yet must not be the cheapest victim");
    require(score(0) == 4U * 60U * kSecond / 8U,
            "an unread owner keeps only the reuse-evidence floor of its value at risk");
    require(score(2) == score(0) * 4U,
            "the retention weight must scale the value at risk instead of only breaking ties");

    require(reuse_evidence_q16(0, {}, now) == (1U << 16U) / 8U,
            "a never-reused owner must keep a nonzero evidence floor");
    require(reuse_evidence_q16(20, now - std::chrono::minutes(180), now) <
                reuse_evidence_q16(20, now, now),
            "reuse evidence must decay with the age of the last reuse");
    require(reuse_evidence_q16(20, now, now) > reuse_evidence_q16(5, now, now),
            "more reuses must mean more protection");
}

// The two orderings the battery's saturation steps actually exercise, at the decision level. Both
// were red on the rig for a whole round and both were the same defect: the score could not tell
// "just published / in use" from "idle for an hour". A session published two seconds ago was priced
// like one nobody had touched since the fill, so the pool evicted what it had just built (the reuse
// suites lost their switch-back hits); a 20 179-token conversation reused on every turn lost its
// anchors to a flood of 1 873-token disposables (important_session). Value now carries live-ness in
// two places - `reuse_evidence_q16` prices it, and the ordering prices a recent owner at the
// LiveSession weight - and these two cases pin them.
void test_victim_score_prices_live_and_idle_owners_differently() {
    const auto now = std::chrono::steady_clock::now();
    constexpr std::uint64_t kSecond     = 1000000000ULL;
    constexpr std::uint64_t kLiveWeight = 16U;
    constexpr std::uint64_t kRecentWeight = 4U;

    // (A) rejust published, never re-read, idle two seconds: must outrank a session ten minutes
    //     idle however much deeper that stale one is (the reuse-suite shape).
    {
        const std::array owners{
            MaterializationOwnerPolicy{
                .owner                    = PlanningOwnerId{.value = 0},
                .retention_class          = RetentionClass::LiveSession,
                .private_retention_weight = kLiveWeight,
                .reuse_evidence_q16 = reuse_evidence_q16(0, now - std::chrono::seconds(2), now),
            },
            MaterializationOwnerPolicy{
                .owner                    = PlanningOwnerId{.value = 1},
                .retention_class          = RetentionClass::RecentPrivate,
                .private_retention_weight = kRecentWeight,
                .reuse_evidence_q16 = reuse_evidence_q16(1, now - std::chrono::minutes(10), now),
            },
        };
        const std::array checkpoints{
            MaterializationCheckpointPolicy{.owner = PlanningOwnerId{.value = 0},
                                            .rebuild_ns = 2U * kSecond},
            MaterializationCheckpointPolicy{.owner = PlanningOwnerId{.value = 1},
                                            .rebuild_ns = 31U * kSecond},
        };
        require(materialization_victim_score(owners, checkpoints, PlanningOwnerId{.value = 0}) >
                    materialization_victim_score(owners, checkpoints,
                                                 PlanningOwnerId{.value = 1}),
                "a conversation published seconds ago must outrank a stale, deeper one");
    }

    // (B) A deep, repeatedly reused conversation idle for half a minute must outrank a shallow
    //     disposable flood (the important-session shape: 8 x 2.5K turns against 40 x 1.9K
    //     one-shot sessions).
    {
        const std::array owners{
            MaterializationOwnerPolicy{
                .owner                    = PlanningOwnerId{.value = 0},
                .retention_class          = RetentionClass::RecentPrivate,
                .selected_hit_count       = 8,
                .private_retention_weight = kRecentWeight,
                .reuse_evidence_q16 = reuse_evidence_q16(8, now - std::chrono::seconds(30), now),
            },
            MaterializationOwnerPolicy{
                .owner                    = PlanningOwnerId{.value = 1},
                .retention_class          = RetentionClass::LiveSession,
                .private_retention_weight = kLiveWeight,
                .reuse_evidence_q16       = reuse_evidence_q16(0, now, now),
            },
        };
        const std::array checkpoints{
            MaterializationCheckpointPolicy{.owner = PlanningOwnerId{.value = 0},
                                            .rebuild_ns = 20U * kSecond},
            MaterializationCheckpointPolicy{.owner = PlanningOwnerId{.value = 1},
                                            .rebuild_ns = 2U * kSecond},
        };
        require(materialization_victim_score(owners, checkpoints, PlanningOwnerId{.value = 0}) >
                    materialization_victim_score(owners, checkpoints,
                                                 PlanningOwnerId{.value = 1}),
                "a deep, repeatedly reused conversation must outrank a shallow disposable flood");
    }
}

// EVERY consumer of the value order must price freshness, and they must do it through the same
// function. The shape below is the production failure of 2026-09-27, at the decision level: a pool
// full of deep-but-stale fill sessions, and one shallow conversation a client has just been reading.
//
//   * fresh probe:  7 s of rebuild work, re-read a second ago  -> live 8x, evidence ~0.5
//   * stale filler: 31 s of rebuild work, never re-read         -> live 1x, evidence floor 1/8
//
// With freshness priced the probe is worth ~4x the filler, so the ladder may not degrade it; without
// the multiplier the probe scored BELOW the filler and the pool destroyed what it had just built
// (`fork@4 cached=0`, `continue cached=5611/7066 = 79%`). Fast tier for a rule whose slow tier is a
// 20-45 minute battery.
void test_owner_score_prices_freshness_for_every_consumer() {
    const auto now              = std::chrono::steady_clock::now();
    constexpr std::uint64_t kSecond = 1000000000ULL;
    constexpr std::uint64_t kOne    = 1U << 16U;
    constexpr std::uint64_t kWeight = 16U; // LiveSession

    const auto probe_value  = 7U * kSecond * kWeight;
    const auto filler_value = 31U * kSecond * kWeight;
    const auto probe_live   = live_multiplier_q16(now - std::chrono::seconds(1), now);
    const auto filler_live  = live_multiplier_q16({}, now); // no recorded use: 1x, never more
    const auto probe_evidence  = reuse_evidence_q16(1, now - std::chrono::seconds(1), now);
    const auto filler_evidence = reuse_evidence_q16(0, {}, now);

    require(filler_live == kOne,
            "an owner with no recorded reuse must not receive a freshness boost");
    require(probe_live > 6U * kOne,
            "a conversation read a second ago must be near the 8x freshness peak");
    require(filler_value > probe_value,
            "the stale filler must hold the deeper value this case is about");

    const OwnerScore probe  = owner_score_ns(probe_value, probe_live, probe_evidence);
    const OwnerScore filler = owner_score_ns(filler_value, filler_live, filler_evidence);
    require(probe.score_ns > filler.score_ns,
            "a just-read conversation must outrank a deeper stale one for every consumer: this is "
            "what keeps the pool from degrading the conversation a client is currently using");
    // And it must be the LIVE multiplier that flips it: with freshness removed the deeper stale
    // owner wins, which is exactly the ladder bug this test exists for.
    require(victim_score_ns(probe_value, probe_evidence) <
                victim_score_ns(filler_value, filler_evidence),
            "the case must be one where freshness decides - depth alone must favour the filler");
    require(probe.boosted_value_ns ==
                static_cast<std::uint64_t>(probe_value * probe_live / kOne),
            "the diagnostic value must be the value at risk after freshness, before evidence");

    // Saturation instead of wraparound, and zero stays zero.
    const OwnerScore huge =
        owner_score_ns(std::numeric_limits<std::uint64_t>::max(), 8U * kOne, kOne);
    require(huge.score_ns == std::numeric_limits<std::uint64_t>::max(),
            "a saturated owner score must clamp instead of wrapping");
    require(owner_score_ns(0, kOne, kOne).score_ns == 0 &&
                owner_score_ns(kOne, 0, kOne).score_ns == 0,
            "a zero value or a zero multiplier must score zero");
}

// The LADDER's own call site, not just the shared arithmetic: the order the Program's last-resort
// release step walks must price freshness. This is the production failure of 2026-09-27 reduced to
// two owners: a deep 31K conversation nobody has re-read since it was built, and a 7K conversation
// the client re-read just now. The cheap owner must be the STALE DEEP one (its never-read floor is
// weight x 31000 tokens x 100 ns / 8 = 6.2 ms here); before `live(x)` reached this path the pool
// made the just-read 7K conversation the cheapest thing it had and degraded that conversation's own
// endpoint and anchors (`fork@4 cached=0`, `continue 79%` on a full production pool).
void test_retire_order_prices_a_just_read_owner_above_a_stale_deeper_one() {
    const auto cheapest_private = [](const FakeProgram& program) -> std::optional<std::uint64_t> {
        std::optional<std::uint64_t> best;
        for (const RetirePreferenceEntry& entry : program.retire_preference) {
            if (entry.shared_prefix) { continue; }
            best = best ? std::min(*best, entry.score_ns) : entry.score_ns;
        }
        return best;
    };

    FakeManager manager = make_manager(4, 8, 1);
    FakeProgram program;
    const auto deep_key = FakeCacheSessionKey{701};
    const auto read_key = FakeCacheSessionKey{702};

    const ActiveRequest deep = start_active(
        manager, program, 701, make_base(701, deep_key, RetentionClass::LiveSession), 1);
    const FakeFinishResult deep_finish = finish_active(manager, program, deep, 31000);
    require(deep_finish.status == ConsumeStatus::Consumed &&
                deep_finish.disposition == FinishDisposition::Catalogued,
            "the deep session was not catalogued");
    const ActiveRequest shallow = start_active(
        manager, program, 702, make_base(702, read_key, RetentionClass::LiveSession), 2);
    (void)finish_active(manager, program, shallow, 7000);

    // Nobody has re-read the deep conversation: the shallow one is the cheapest owner.
    program.retire_preference.clear();
    (void)manager.inspect(program, FakePreparedPrompt{703}, make_base(703), 3);
    const std::optional<std::uint64_t> before = cheapest_private(program);
    require(before.has_value(), "the retire order named no private owner");

    // Now the client re-reads the shallow conversation: one full reuse cycle over its checkpoint
    // digest and session key, which is what records the reuse evidence the score reads.
    const ActiveRequest follow_up = start_active(
        manager, program, 702, make_base(702, read_key, RetentionClass::LiveSession), 4);
    const FakeFinishResult follow_up_finish = finish_active(manager, program, follow_up, 7000);
    require(follow_up_finish.status == ConsumeStatus::Consumed,
            "the re-read of the shallow conversation did not complete");

    program.retire_preference.clear();
    (void)manager.inspect(program, FakePreparedPrompt{705}, make_base(705), 5);
    const std::optional<std::uint64_t> after = cheapest_private(program);
    require(after.has_value(), "the retire order named no private owner after the re-read");
    require(*after > *before,
            ("re-reading a conversation must raise its value: the cheapest owner must not stay the "
             "one the client is using (before=" + std::to_string(*before) + " after=" +
             std::to_string(*after) + ")").c_str());
    require(*after >= 31000ULL * 100ULL * 16ULL / 8ULL,
            ("after the re-read the cheapest owner must be the stale 31K conversation priced at its "
             "never-read floor (freshness, not depth, decides) after=" +
             std::to_string(*after)).c_str());
}

// §三 R1 / §四 invariant 1: replacing a shared owner DESTROYS its replicas, and R2 may only delete
// Host-side data. A shared owner that still holds Device data may therefore not be offered as a
// replacement victim at all - the replacement path cannot move the data first without contradicting
// the capacity equation it already committed to. The promotion is optional (§7.2: only spare
// capacity that does not degrade an existing owner), so skipping such a victim costs the contract
// nothing: the request still gets its own private checkpoint.
//
// Measured 2026-09-27: this is the only POLICY path that destroyed Device data in either tier
// (`[invariant1] strict-shared site=capture-replacement`); every other occurrence was the shutdown
// teardown, which is tagged `site=shutdown` and is correct there.
void test_shared_replacement_is_not_offered_for_a_device_holding_victim() {
    // Fixture: publish one shared owner, then offer a second capture with shared evidence strong
    // enough for the manager to consider replacing it.
    const auto publish_seed = [](FakeManager& manager, FakeProgram& program) {
        FakeRequestBasePlan seed_base = make_base(91);
        seed_base.cache.opportunities.push_back(FakeContextCache::Opportunity{
            .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence = ninfer::SharedCandidateEvidence::ExplicitBoundary,
            .frontier = 64,
        });
        const ActiveRequest seed = start_active(manager, program, 91, seed_base, 1);
        program.capture_assessment = FakeCaptureAssessment{
            .shortlist_key          = FakeShortlistKey{.digest = 91, .frontier = 64},
            .shared_evidence        = ninfer::SharedCandidateEvidence::ExplicitBoundary,
            .protected_rebuild_work = PrefillWork{.tokens = 64},
            .publishes_shared       = true,
            .physically_feasible    = true,
        };
        require(manager.reserve_active_capture(program, seed.lane, FakeCaptureOffer{.id = 31}, 0,
                                               {}) == FakeManager::ActiveCaptureReserveResult::Reserved,
                "shared fixture could not reserve its source capture");
        auto progress = manager.progress_context_transaction(program, {});
        const auto capture = std::get<FakeManager::ActiveCaptureOutcome>(std::move(progress));
        require(capture.status == ContextTransactionStatus::Published,
                "shared fixture did not publish its source");
        (void)finish_active(manager, program, seed);
    };

    const auto offer_replacement = [](FakeManager& manager, FakeProgram& program,
                                      std::uint32_t id) {
        FakeRequestBasePlan base = make_base(id);
        base.cache.opportunities.push_back(FakeContextCache::Opportunity{
            .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
            .evidence = ninfer::SharedCandidateEvidence::ExplicitBoundary,
            .frontier = 32,
        });
        const ActiveRequest request = start_active(manager, program, id, base, 1);
        (void)manager.reserve_active_capture(program, request.lane, FakeCaptureOffer{.id = id}, 0,
                                             {});
    };

    // (A) The victim holds Device data that CAN be moved: it is moved first (§三 R1: 显存里的数据只有
    //     两种归宿, 正在用, 或搬到内存), and only then may the replacement go ahead.
    {
        FakeManager manager = make_manager(2, 3, 1);
        FakeProgram program;
        publish_seed(manager, program);
        program.shared_victim_device_data = true;
        program.shared_victim_move_possible = true;
        offer_replacement(manager, program, 93);
        require(program.shared_replacement_moves == 1,
                "the replacement path must MOVE the victim's Device data to Host before destroying it");
        require(!program.shared_victim_device_data,
                "the move must leave the victim Host-only");
        require(program.shared_replacement_inspections == 1,
                "a victim that could be moved is still a replacement candidate");
    }
    // (B) The victim's Device data CANNOT be moved: it must not be replaced at all - destroying it is
    //     what R1 forbids, and the promotion is optional (§7.2).
    {
        FakeManager manager = make_manager(2, 3, 1);
        FakeProgram program;
        publish_seed(manager, program);
        program.shared_victim_device_data  = true;
        program.shared_victim_move_possible = false;
        offer_replacement(manager, program, 95);
        require(program.shared_replacement_moves == 1,
                "the path must TRY to move before declining");
        require(program.shared_replacement_inspections == 0,
                "a shared owner whose Device data cannot be moved was offered as a replacement victim "
                "(releasing it would destroy Device data; R2 may only delete Host)");
    }
    // (C) Control: nothing on Device is ready as is, so the path is not disabled.
    {
        FakeManager manager = make_manager(2, 3, 1);
        FakeProgram program;
        publish_seed(manager, program);
        program.shared_victim_device_data = false;
        offer_replacement(manager, program, 97);
        require(program.shared_replacement_inspections == 1,
                "a host-only shared owner must still be considered as a replacement victim");
    }
}

// 缓存模块v2.md §2.1 / §三 R0: the ladder may only refuse to reclaim what is being EXECUTED (or what
// an in-flight reservation is about to write). A FINISHED conversation's images - its turn closure
// included, dearest of them - are cache and must be reclaimable; the value order decides that they go
// last, not that they can never go.
//
// Measured 2026-09-27 (`parallel-fresh`, real model, and the doc's §十.1): an IDLE Engine waited its
// entire admission deadline (`blocked-in-idle … active=0`) because the Host state pool held the turn
// closures of FINISHED conversations, the R2 component step accepts only HostOnly replicas, and the
// lossless EvictHostReplica step refused every closure - the predicate bound the closure of every
// non-Free row, whether or not that conversation was being executed.
void test_finished_conversations_images_are_reclaimable() {
    // Executed work is the working set: never reclaimable.
    require(state_reclaim::state_is_live_binding(state_reclaim::StateOwnerUse::InflightDestination),
            "an in-flight reservation's destination must stay bound");
    require(state_reclaim::state_is_live_binding(state_reclaim::StateOwnerUse::ExecutingActiveState),
            "the executed conversation's live state must stay bound");
    require(state_reclaim::state_is_live_binding(state_reclaim::StateOwnerUse::ExecutingClosure),
            "the executed conversation's turn closure must stay bound");
    require(state_reclaim::state_is_live_binding(state_reclaim::StateOwnerUse::ExecutingAnchor),
            "the executed conversation's fork points must stay bound");

    // A finished conversation is cache, whatever the image is.
    require(!state_reclaim::state_is_live_binding(state_reclaim::StateOwnerUse::IdleClosure),
            "a FINISHED conversation's turn closure is cache: the pool must be able to sacrifice it "
            "(value-ordered last) instead of waiting out the queue deadline");
    require(!state_reclaim::state_is_live_binding(state_reclaim::StateOwnerUse::IdleActiveState),
            "a FINISHED conversation's retained state is cache");
    require(!state_reclaim::state_is_live_binding(state_reclaim::StateOwnerUse::IdleAnchor),
            "a FINISHED conversation's fork point is cache");
}

// §2.1: the protection an in-flight reservation places on its own source is an OWNERSHIP fact, not
// a retention one. A state allocation is aliased by private and shared checkpoints, so testing the
// handle alone also protects every owner that merely CACHED that allocation as one of its long
// anchors - and those owners are pure cache, which §三 R2 must remain free to delete.
//
// Reproduced from production 2026-10-01 req#407-#412 (an IDLE engine, 256 finished conversations,
// Host KV at 31.9 GiB): `[ladder] degrade declined ... protected=295 ... noplace=91`, then
// `[cache] pressure step refused: victim slot=54 cannot be moved to Host (R1), rolling the step
// back instead of destroying Device data (R0)`, and the request was answered with an HTTP 200
// carrying `finish_reason: "stop"` and zero content. The pool could neither degrade nor spill
// anything, so §一's guarantee ("只要还有一条已经处理完的对话占着缓存,就一定能删") was violated by a
// predicate that described RETENTION while the ladder needed OWNERSHIP.
void test_a_cached_alias_does_not_make_an_owner_unreleasable() {
    using state_reclaim::state_is_live_owner_of;
    using Handle                     = std::uint64_t;
    const Handle* const no_handle    = nullptr;
    const std::uint64_t protected_allocation = 7;
    const std::uint64_t other                = 9;

    // The reservation's own source owner: the handle IS its live binding.
    require(state_is_live_owner_of(protected_allocation, other, no_handle, no_handle,
                                   protected_allocation),
            "the owner whose read state is the protected handle IS protected (§2.1)");
    require(state_is_live_owner_of(other, protected_allocation, no_handle, no_handle,
                                   protected_allocation),
            "an owner that WRITES the protected handle IS protected");
    const std::uint64_t reserved = protected_allocation;
    const std::uint64_t rewrite  = protected_allocation;
    require(state_is_live_owner_of(other, other, &reserved, no_handle, protected_allocation),
            "the in-flight reservation's destination IS protected while it is about to be written");
    require(state_is_live_owner_of(other, other, no_handle, &rewrite, protected_allocation),
            "an owner's turn closure IS protected while it is being rewritten");

    // Every OTHER owner: unrelated live state. These are the ones the over-broad predicate pinned.
    require(!state_is_live_owner_of(other, other, no_handle, no_handle, protected_allocation),
            "an owner that does not hold the handle live must stay releasable (it is cache, §三 R2)");
    require(!state_is_live_owner_of(protected_allocation + 1, other, no_handle, no_handle,
                                    protected_allocation),
            "a DIFFERENT live state must not match the protected handle");
    // The alias case itself: caching the allocation as a long anchor is retention, not ownership -
    // `holds_any` (retention, still used for the fatal active-capture latch) is deliberately NOT
    // what the release ladder asks, and this pins that the ownership test cannot see the anchor.
    require(!state_is_live_owner_of(other, other, no_handle, no_handle, protected_allocation),
            "a long-anchor alias of the protected allocation does not protect its holder");
}

// Reproduced: a Device-slot gap must be answerable by EVERY checkpoint that still holds a Device
// replica, whichever of the two move steps its replica layout selects.
//
// 2026-09-28 14:4x production, on an IDLE engine: `[cache] gap ... dstate=1 | ... enqueue
// reason=device-not-closable` next to `pool sums ... mv_state=0`, i.e. the ladder found NO victim
// that could move a state, while 8 Device slots were held by finished conversations' checkpoints.
// The two steps partitioned the candidates the wrong way round - `CopyToHost` accepted only
// DeviceOnly, `DropDeviceReplica` ran under a different veto - so a Both-resident image that the
// drop step vetoed was invisible to both, and the request waited out its queue deadline. §一 says
// waiting on a finished conversation's cache is a defect, not R0.
void test_a_device_slot_gap_is_answerable_by_every_checkpoint_holding_a_replica() {
    using state_reclaim::StateReplicaLayout;

    // Both layouts free a Device slot; the layout only decides which step runs.
    require(state_reclaim::state_can_answer_device_slot_gap(StateReplicaLayout::DeviceOnly, false),
            "a DeviceOnly checkpoint answers a Device-slot gap by demoting");
    require(state_reclaim::state_can_answer_device_slot_gap(StateReplicaLayout::Both, false),
            "a Both-resident checkpoint answers it by dropping the Device replica - this is the "
            "candidate the ladder used to lose, and losing it parked the request (mv_state=0)");

    // An image with no Device replica cannot free a Device slot, and a HostOnly one cannot either.
    require(!state_reclaim::state_can_answer_device_slot_gap(StateReplicaLayout::None, false),
            "an image with no replica cannot answer a Device-slot gap");
    require(!state_reclaim::state_can_answer_device_slot_gap(StateReplicaLayout::HostOnly, false),
            "a HostOnly image holds no Device slot to give back");

    // The caller's ownership veto still refuses a victim, whatever its layout: the fix widens which
    // LAYOUT is a candidate, never which OWNER may be reclaimed.
    require(!state_reclaim::state_can_answer_device_slot_gap(StateReplicaLayout::Both, true),
            "a vetoed image is never a victim, even when it holds both replicas");
    require(!state_reclaim::state_can_answer_device_slot_gap(StateReplicaLayout::DeviceOnly, true),
            "and neither is a vetoed DeviceOnly one");

    // Host-slot gaps are the mirror image: only a redundant Host replica may be given up, because
    // dropping a HostOnly checkpoint's only replica would leave it unpublished (§2.1).
    require(state_reclaim::state_can_answer_host_slot_gap(StateReplicaLayout::Both, false),
            "a Both-resident checkpoint answers a Host-slot gap by dropping the Host replica");
    require(!state_reclaim::state_can_answer_host_slot_gap(StateReplicaLayout::HostOnly, false),
            "a HostOnly checkpoint's only replica is not evictable on its own");
    require(!state_reclaim::state_can_answer_host_slot_gap(StateReplicaLayout::DeviceOnly, false),
            "a DeviceOnly image holds no Host slot to give back");
}

// §四 invariant 1 as the CODE-LEVEL prohibition, not a measurement: the strict release asks this
// table before it destroys anything, and every cache-policy intent must refuse an owner that still
// holds Device data. The `[invariant1]` probe only made a violation visible after it had happened,
// and only in the pool geometries a battery happened to run: the 2026-09-27 pressure cases destroyed
// 13 Device state images through `site=transaction-victim` in 4-slot Host state pools while the rig's
// 48-slot pool never reproduced it. With the guard in the release primitive, "显存里的数据只有两种
// 归宿 - 正在用, 或搬到内存" holds for EVERY input: a victim that cannot be moved first is not
// releasable (the step rolls back and the request waits, R0).
void test_only_an_ownership_return_may_destroy_device_data() {
    using state_reclaim::ReleaseIntent;
    const ReleaseIntent policy_intents[] = {
        ReleaseIntent::PolicyMaterializationVictim,
        ReleaseIntent::PolicyCaptureVictim,
        ReleaseIntent::PolicySharedPressureVictim,
        ReleaseIntent::PolicyCaptureSharedPressureVictim,
        ReleaseIntent::PolicyCaptureReplacement,
        ReleaseIntent::PolicyLadderHostOnly,
    };
    for (const ReleaseIntent intent : policy_intents) {
        require(!state_reclaim::release_may_destroy_device(intent),
                "a cache-policy release may never destroy Device data (R1 says move it first)");
        require(state_reclaim::release_admits_device_destruction(intent, false),
                "a Device-free owner is always releasable on a policy path");
        require(!state_reclaim::release_admits_device_destruction(intent, true),
                "a policy release holding Device data must be REFUSED, not executed");
    }
    // The two releases that are not cache decisions: the client consumed the handle (the
    // conversation ends by definition), and the process-exit teardown releases every replica.
    require(state_reclaim::release_may_destroy_device(ReleaseIntent::OwnershipHandleRelease),
            "a consumed handle ends that conversation - its Device data goes with it");
    require(state_reclaim::release_may_destroy_device(ReleaseIntent::ShutdownTeardown),
            "the shutdown teardown releases everything because the Program is going away");
    require(state_reclaim::release_admits_device_destruction(
                ReleaseIntent::OwnershipHandleRelease, true),
            "the ownership return is admitted with Device data");
    // Every intent has a name (the batteries grep these strings).
    for (const ReleaseIntent intent : policy_intents) {
        const std::string name = state_reclaim::release_intent_name(intent);
        require(!name.empty() && name != "?", "every release intent must have a log name");
    }
}

// §2.2 in the form the acceptance cases depend on: `score = value x live x evidence`, where a
// NEVER-REUSED owner has live = 1 and evidence = 1/8, so its score is its SIZE and nothing else,
// while an owner with reuse evidence is worth several times more per token. This is why an idle
// conversation may legitimately be sacrificed (its record is the cheapest) and why the fix for such
// a case is to give its conversation EVIDENCE - one extra read - rather than to enlarge it: size
// spends the pool, evidence does not (measured 2026-09-27: the switch-back case's conversation was
// deleted at 3 011 tokens, at 16 525 and still at 20 553, and the 20.5K version cost 82K of the rig's
// 131K-token KV pool and starved the next step into an HTTP 503).
void test_reuse_evidence_outranks_size_among_never_reused_owners() {
    const auto private_min_score = [](const FakeProgram& program) -> std::uint64_t {
        std::uint64_t best = std::numeric_limits<std::uint64_t>::max();
        for (const RetirePreferenceEntry& entry : program.retire_preference) {
            if (!entry.shared_prefix) { best = std::min(best, entry.score_ns); }
        }
        return best;
    };

    FakeManager manager = make_manager(4, 8, 1);
    FakeProgram program;
    const auto never_key  = FakeCacheSessionKey{810};
    const auto reused_key = FakeCacheSessionKey{811};

    // Two conversations of the SAME size: one only ever created, one read once more.
    const ActiveRequest never = start_active(
        manager, program, 810, make_base(810, never_key, RetentionClass::RecentPrivate), 1);
    (void)finish_active(manager, program, never, 4096);
    const ActiveRequest reused = start_active(
        manager, program, 811, make_base(811, reused_key, RetentionClass::RecentPrivate), 2);
    (void)finish_active(manager, program, reused, 4096);
    const ActiveRequest read_again = start_active(
        manager, program, 811, make_base(811, reused_key, RetentionClass::RecentPrivate), 3);
    const FakeFinishResult read_finish = finish_active(manager, program, read_again, 4096);
    require(read_finish.status == ConsumeStatus::Consumed,
            "the evidence read of the second conversation did not complete");

    program.retire_preference.clear();
    (void)manager.inspect(program, FakePreparedPrompt{812}, make_base(812), 4);
    const std::uint64_t cheapest = private_min_score(program);
    require(cheapest < std::numeric_limits<std::uint64_t>::max(),
            "the retire order named no private owner");
    // The never-reused owner's score is value x (1/8): with the fake cost model (100 ns/token) and the
    // RecentPrivate weight (4) that is 4096 x 100 x 4 / 8 exactly, i.e. independent of ANY clock.
    require(cheapest == 4096ULL * 100ULL * 4ULL / 8ULL,
            ("the cheapest record must be the never-reused conversation sized 4096 (got " +
             std::to_string(cheapest) + "): reuse evidence is what protects an idle conversation, so a "
             "case that wants its conversation to survive must give it evidence, not size").c_str());
}

// R0 must be the LAST resort, not the first answer. When no plan exists because the pool is full,
// the engine owes the request one unit of R2 relief first - the current request is the highest-value
// thing in the pool (缓存模块v2.md §2.1: 正在执行的对话价值最高), so the least valuable CACHED entry
// gives way and the request is re-inspected against the pool the ladder just produced. Only when the
// ladder has nothing left to release may the verdict stand and the request park to its deadline.
//
// Measured 2026-09-27 in production: `[search] target infeasible: choices=321 | device.main_kv
// used=10282 peak=81 cap=10284 | host.state used=320 peak=0 cap=320` followed by
// `[cache] policy target rejected reason=assessment candidate=root` and
// `[engine] blocked-in-idle: head=714 waits for its queue deadline (R0)` - the head was served
// seven minutes later only because an unrelated cancelled request released capacity. The release
// set was one unit short and the ladder could have paid it immediately (it was degrading other
// owners' checkpoints in the same window).
void test_unplannable_request_takes_cached_relief_before_parking() {
    FakeManager manager = make_manager(4, 8, 1);
    FakeProgram program;
    const ActiveRequest seed = start_active(
        manager, program, 801, make_base(801, FakeCacheSessionKey{801}, RetentionClass::LiveSession), 1);
    (void)finish_active(manager, program, seed, 4096);

    // The pool is full enough that no target can pay for this request's claim...
    program.required_pressure_actions = 4;
    // ...and the ladder CAN release one unit of cached data. The fake models that release the way a
    // dropped checkpoint lowers the deficit: the requirement the plan must satisfy gets smaller.
    program.cached_relief_available = true;

    const FakeManager::Inspection relieved =
        manager.inspect(program, FakePreparedPrompt{802}, make_base(802), 2);
    require(program.cached_relief_requests == 1,
            "an unplannable request must ask the ladder for one unit of cached relief before parking");
    require(relieved.readiness == Readiness::TemporarilyBlocked && relieved.relief_taken,
            "taking relief must be reported: the engine has to re-inspect now instead of waiting "
            "for an event that may never come");

    // ...and the re-inspection is served, which is the whole point of taking the relief.
    const FakeManager::Inspection served =
        manager.inspect(program, FakePreparedPrompt{802}, make_base(802), 3);
    require(served.readiness == Readiness::Ready || served.readiness == Readiness::NeedsTransfer,
            "after the ladder released one unit the request must be planned, not parked again");
    require(program.cached_relief_requests == 1,
            "a served request must not keep taking relief");

    // With nothing left to release the request parks (R0), and it does not spin: the ladder is asked
    // once more and the verdict stands.
    program.cached_relief_available  = false;
    program.required_pressure_actions = 4;
    const FakeManager::Inspection parked = manager.inspect(program, FakePreparedPrompt{803}, make_base(803), 4);
    require(parked.readiness == Readiness::TemporarilyBlocked && !parked.relief_taken,
            "with nothing releasable the head parks (R0) and must not report progress");
    require(program.cached_relief_requests == 2,
            "the second request must have asked the ladder too (once per unplannable plan)");
}

// The R0 relief must serve the axis the request is actually short of. A state-slot unit cannot
// pay a Device-KV deficit, so a relief that only frees state slots reports "nothing releasable" for
// a request short of KV pages - and then the negative memo parks it for its whole deadline while the
// engine sits idle, which is the wait §三 R0 forbids (it is not "waiting for the working set to
// finish", nothing is running at all).
//
// Measured 2026-09-30 production: req#629/#650/#652/#654, each a ~186K-token conversation whose
// reuse base needed 2,890 Device pages, sat at `running 0 | waiting 1` for exactly 4m59.9s and died
// at HTTP 499 after 36 `spill declined ... stage=noruns`; the identical conversation was served
// 2.9 s later at 99.2% reuse. The relief existed (`release_one_device_state_slot`) but released a
// unit the deficit could not use.
void test_r0_relief_must_serve_the_axis_the_deficit_is_on() {
    FakeManager manager = make_manager(4, 8, 1);
    FakeProgram program;
    const ActiveRequest seed = start_active(
        manager, program, 811, make_base(811, FakeCacheSessionKey{811}, RetentionClass::LiveSession), 1);
    (void)finish_active(manager, program, seed, 4096);

    program.required_pressure_actions = 4;
    // The one releasable unit is a STATE slot, but this request is short of Device KV pages.
    program.cached_relief_available = true;
    program.relief_axis             = FakeProgram::CachedReliefAxis::StateSlot;
    program.deficit_is_state_slot   = false;
    program.deficit_is_device_kv    = true;

    const FakeManager::Inspection mismatched =
        manager.inspect(program, FakePreparedPrompt{812}, make_base(812), 2);
    require(mismatched.readiness == Readiness::TemporarilyBlocked && !mismatched.relief_taken,
            "a state-slot relief must not be reported as progress for a Device-KV deficit");

    // With the KV half available - what `spill_one_movable_owner_device_kv` provides - the same
    // request is relieved and re-planned instead of parking.
    program.cached_relief_available = true;
    program.relief_axis             = FakeProgram::CachedReliefAxis::DeviceKv;
    const FakeManager::Inspection relieved =
        manager.inspect(program, FakePreparedPrompt{812}, make_base(812), 3);
    require(relieved.readiness == Readiness::TemporarilyBlocked && relieved.relief_taken,
            "the KV half of the relief must count as progress for a Device-KV deficit (§三 R1)");

    const FakeManager::Inspection served =
        manager.inspect(program, FakePreparedPrompt{812}, make_base(812), 4);
    require(served.readiness == Readiness::Ready || served.readiness == Readiness::NeedsTransfer,
            "after the KV relief the request must be planned, not parked again");
}

// Fair-share protection is a VALUE, not an exclusion (缓存模块v2.md §六.4): a shared-capture
// offer whose only feasible target needs pressure no longer refuses the bucket outright - it
// RESERVES, and the one importance chain prices the protected session (K + age ranks it
// last-but-eligible, so it is only taken when nothing cheaper exists). The catalog entry must
// still be untouched at reserve time: claims land when the pressure target seals, not here.
void test_fair_share_capture_prices_protected_sessions_instead_of_refusing() {
    FakeManager manager = make_manager(1, 4, 1); // default fair_share_buckets = 8
    FakeProgram program;
    const ActiveRequest idle = start_active(
        manager, program, 411,
        make_base(411, FakeCacheSessionKey{411}, RetentionClass::LiveSession), 1);
    (void)finish_active(manager, program, idle);

    FakeRequestBasePlan request = make_base(412);
    request.cache.opportunities.push_back(FakeContextCache::Opportunity{
        .kind     = ninfer::PromptCacheMarkerKind::SharedStablePrefix,
        .evidence = ninfer::SharedCandidateEvidence::ExplicitBoundary,
        .frontier = 64,
    });
    const ActiveRequest active           = start_active(manager, program, 412, request, 2);
    program.required_pressure_actions    = 1;
    program.pressure_action_immediate_ns = 0;
    program.capture_assessment           = FakeCaptureAssessment{
        .shortlist_key          = FakeShortlistKey{.digest = 412, .frontier = 64},
        .shared_evidence        = ninfer::SharedCandidateEvidence::ExplicitBoundary,
        .protected_rebuild_work = PrefillWork{.tokens = 64},
        .publishes_shared       = true,
        .physically_feasible    = false,
    };

    const auto reserved =
        manager.reserve_active_capture(program, active.lane, FakeCaptureOffer{.id = 11}, 0, {});
    require(reserved == FakeManager::ActiveCaptureReserveResult::Reserved,
            "fair-share must be priced by importance, not refused (section6.4: value, "
            "not exclusion)");
    require(manager.catalog_state(0) == FakeManager::CatalogState::Claimed,
            "reserve prices its victim and claims it atomically - fair-share is a value the "
            "chain pays for, not a veto the path refuses with");
    // The reservation opened a capture transaction: drive it home the way every other reserved
    // capture does, then the lane must be finishable again.
    auto progress = manager.progress_context_transaction(program, {});
    auto outcome  = std::get<FakeManager::ActiveCaptureOutcome>(std::move(progress));
    require(!manager.context_transaction_kind(),
            "reserved capture transaction must close on progress");
    (void)outcome;
    (void)finish_active(manager, program, active);
}

// The long-anchor budget is full: the released anchor must be the one whose removal keeps
// the retained set closest to uniform spacing, never the pinned head anchor, and a redundant
// offered anchor is skipped instead of forcing a release.
void test_uniform_private_anchor_replacement() {
    const auto assess = [](std::initializer_list<std::uint32_t> frontiers, std::uint32_t offer) {
        FakeCaptureAssessment assessment;
        assessment.frontier = offer;
        std::uint32_t ordinal = 1;
        for (const std::uint32_t value : frontiers) {
            assessment.private_replacement_candidates.push_back(
                CheckpointRef{.kind = CheckpointKind::LongAnchor, .frontier = value,
                              .ordinal = ordinal++});
        }
        return assessment;
    };
    // A redundant tail offer is skipped: the retained set is already uniformly spaced.
    const auto redundant =
        ninfer::runtime::select_uniform_private_anchor_replacement(assess({100, 200, 300}, 400));
    require(!redundant.has_value(), "a redundant offered anchor forced a release of a uniform set");
    // A dense tail cluster is thinned from the middle, never from the pinned head.
    const auto victim = ninfer::runtime::select_uniform_private_anchor_replacement(
        assess({100, 200, 300, 310, 320}, 400));
    require(victim && victim->frontier == 310,
            "the uniform-spacing policy did not thin the densest retained cluster");
    // The head anchor is pinned even when its own gap is the densest one.
    const auto pinned =
        ninfer::runtime::select_uniform_private_anchor_replacement(assess({100, 110, 500}, 600));
    require(pinned && pinned->frontier == 110, "the pinned head anchor was released");
    // A single-slot budget always follows the conversation.
    const auto single =
        ninfer::runtime::select_uniform_private_anchor_replacement(assess({100}, 200));
    require(single && single->frontier == 100,
            "a single-slot budget did not track the offered anchor");
}

} // namespace


// The conversation being EXECUTED is not "ranked last", it is absent from the victim order: ranking
// can only say who is least important, and a session that is mid-flight cannot be re-derived at all.
// This is the head of the `prefix_switch` shape in the fake harness: one conversation active, one
// idle, a capture reservation asking for a Host state slot.
void test_executing_conversation_is_absent_from_the_retire_order() {
    FakeManager manager = make_manager(4, 8, 1);
    FakeProgram program;
    const ActiveRequest running = start_active(
        manager, program, 601,
        make_base(601, FakeCacheSessionKey{601}, RetentionClass::LiveSession), 1);
    const ActiveRequest idle = start_active(
        manager, program, 602,
        make_base(602, FakeCacheSessionKey{602}, RetentionClass::RecentPrivate), 1);
    (void)finish_active(manager, program, idle);

    program.retire_preference.clear();
    (void)manager.reserve_active_capture(program, running.lane, FakeCaptureOffer{.id = 17}, true, {});
    require(!program.retire_preference.empty(),
            "capture reservation did not hand a retire preference to the Program");
    // Two conversations exist - one executed, one finished - so the order must name exactly ONE
    // private owner: the executed one is absent by construction, not ranked last.
    std::size_t private_entries = 0;
    for (const ninfer::runtime::RetirePreferenceEntry& entry : program.retire_preference) {
        if (!entry.shared_prefix) { ++private_entries; }
    }
    require(private_entries == 1,
            "the conversation being executed was offered as a victim (only the idle one may be)");
    (void)running;
}

int main() {
    run_test("private checkpoint identity loss",
             test_private_portfolio_loss_keeps_checkpoint_identity_fixed);
    run_test("portfolio demand and owner aggregation", test_portfolio_demand_and_owner_aggregation);
    run_test("shared capture private transition loss",
             test_shared_capture_subtracts_private_transition_loss);
    run_test("shared capture committed target budget",
             test_shared_capture_budget_bounds_committed_canonical_targets);
    run_test("equal lower-bound tie-break",
             test_equal_lower_bound_does_not_short_circuit_tie_break);
    run_test("machine cost is selection-only",
             test_machine_cost_changes_selection_without_changing_physical_assessment);
    run_test("candidate-stratified reuse closure",
             test_candidate_search_prefers_deep_reuse_without_eviction);
    run_test("feasible identity pressure improvement",
             test_feasible_identity_expands_when_pressure_can_remove_copy);
    run_test("dominating identity fast path",
             test_dominating_identity_does_not_build_pressure_graph);
    run_test("root lifecycle and prefix reuse", test_root_lifecycle_and_prefix_reuse);
    run_test("retired owner catalog repair", test_retired_owner_is_not_offered_as_reuse_source);
    run_test("capacity miss is retryable", test_capacity_miss_is_retryable);
    run_test("stale revision is retryable", test_stale_revision_is_retryable);
    run_test("materialization abort preserves source", test_materialization_abort_preserves_source);
    run_test("committed victim survives abort", test_committed_victim_survives_transaction_abort);
    run_test("uncommitted pressure acknowledgement",
             test_uncommitted_pressure_acknowledgement_is_not_degradation);
    run_test("aborted source is not a hit",
             test_aborted_source_selection_does_not_create_hit_history);
    run_test("retained source protection", test_retained_source_is_protected_until_terminal);
    run_test("session publication order", test_session_publication_order_controls_tied_source);
    run_test("canonical pressure", test_canonical_pressure_starts_with_disposable_owner);
    run_test("all preserving pressure alternatives",
             test_pressure_tries_every_preserving_alternative_before_eviction);
    run_test("cumulative owner target",
             test_cumulative_owner_target_closes_pressure_without_eviction);
    run_test("joint two-owner pressure", test_two_owners_jointly_close_pressure);
    run_test("validate complete materialization result before adoption",
             test_materialization_result_is_validated_before_any_adoption);
    run_test("materialization result checkpoint identity",
             test_materialization_result_binds_exact_checkpoint_identity);
    run_test("materialization result owner identity",
             test_materialization_result_is_adopted_by_owner_identity);
    run_test("guided deep retention",
             test_guided_pressure_reaches_deep_retention_before_capped_fallback);
    run_test("combined target exact repricing",
             test_combined_target_reprices_cancelled_pressure_copy);
    run_test("in-progress and capture", test_in_progress_adoption_and_private_capture);
    run_test("capture planning repairs retired owners",
             test_capture_planning_repairs_retired_owner);
    run_test("projected shared marginal value",
             test_projected_nested_shared_candidates_use_marginal_value);
    run_test("observed shared independent domains",
             test_observed_shared_candidate_requires_independent_domains);
    run_test("zero-prefill private promotion",
             test_repeated_private_reuse_selects_zero_prefill_shared_promotion);
    run_test("shared fanout owner edges",
             test_shared_fanout_keeps_owner_edges_live_across_summary_refresh);
    run_test("shared capture multi-owner pressure",
             test_shared_capture_combines_two_pressure_owners);
    run_test("aborted shared capture logical rollback",
             test_aborted_shared_capture_start_rolls_back_logical_claims);
    run_test("validate complete capture result before adoption",
             test_capture_result_is_validated_before_any_adoption);
    run_test("capture result owner identity", test_capture_result_is_adopted_by_owner_identity);
    run_test("uniform private anchor replacement", test_uniform_private_anchor_replacement);
    run_test("terminal fallback", test_terminal_fallback_releases_failed_retention);
    run_test("terminal waits for resource transaction",
             test_terminal_settlement_waits_for_open_resource_transaction);
    run_test("commit and discard", test_commit_and_discard_terminal_states);
    run_test("backfill proof and stats", test_backfill_proof_and_stats_follow_program_revision);
    run_test("shortlist exact verification",
             test_shortlist_collision_requires_program_exact_verification);
    run_test("fair-share sacrifices the oldest protected session last",
             test_fair_share_sacrifices_the_oldest_protected_session_last);
    run_test("fair-share capture pricing",
             test_fair_share_capture_prices_protected_sessions_instead_of_refusing);
    run_test("manager hands over the retire preference",
             test_manager_hands_the_retire_preference_to_the_program);
    run_test("reserve order excludes its own private source",
             test_reserve_order_excludes_its_own_private_source);
    run_test("victim score combines value and reuse evidence",
             test_victim_score_ranks_by_value_not_by_one_field);
    run_test("victim score prices live against idle owners",
             test_victim_score_prices_live_and_idle_owners_differently);
    run_test("owner score prices freshness for every consumer",
             test_owner_score_prices_freshness_for_every_consumer);
    run_test("reuse evidence outranks size among never-reused owners",
             test_reuse_evidence_outranks_size_among_never_reused_owners);
    run_test("retire order prices a just-read owner above a stale deeper one",
             test_retire_order_prices_a_just_read_owner_above_a_stale_deeper_one);
    run_test("unplannable request takes cached relief before parking",
             test_unplannable_request_takes_cached_relief_before_parking);
    run_test("r0 relief serves the axis the deficit is on",
             test_r0_relief_must_serve_the_axis_the_deficit_is_on);
    run_test("finished conversations' images are reclaimable",
             test_finished_conversations_images_are_reclaimable);
    run_test("a cached alias does not make an owner unreleasable",
             test_a_cached_alias_does_not_make_an_owner_unreleasable);
    test_only_an_ownership_return_may_destroy_device_data();
    run_test("a device-slot gap is answerable by every checkpoint holding a replica",
             test_a_device_slot_gap_is_answerable_by_every_checkpoint_holding_a_replica);
    run_test("shared replacement skips a device-holding victim",
             test_shared_replacement_is_not_offered_for_a_device_holding_victim);
    run_test("executing conversation is absent from the retire order",
             test_executing_conversation_is_absent_from_the_retire_order);
    if (failures != 0) { return 1; }
    std::cout << "ok\n";
    return 0;
}
