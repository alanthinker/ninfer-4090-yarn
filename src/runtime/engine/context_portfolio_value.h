#pragma once

#include "runtime/contract/types.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <span>
#include <stdexcept>
#include <vector>

namespace ninfer::runtime {

struct ContextPortfolioOwnerPolicy {
    PlanningOwnerId owner;
    std::uint32_t private_retention_weight = 0;
    bool explicit_shared_credit            = false;
};

struct ContextPortfolioCheckpointValue {
    PlanningOwnerId owner;
    std::uint32_t demand_mask          = 0;
    std::uint64_t rebuild_ns           = 0;
    std::uint64_t baseline_recovery_ns = 0;
    std::uint64_t target_recovery_ns   = 0;
};

struct ContextPortfolioValueResult {
    std::uint64_t baseline_public_value   = 0;
    std::uint64_t target_public_value     = 0;
    std::uint64_t private_transition_loss = 0;
    bool saturated                        = false;
};

// Folds one complete inactive checkpoint portfolio. Empirical demands and explicit shared credit
// form public portfolio values. A private owner contributes the largest loss of one concrete
// checkpoint capability, so a surviving later checkpoint cannot mask damage to an earlier one.
class ContextPortfolioValue {
public:
    ContextPortfolioValue() {
        owner_scratch_.reserve(32U);
        owner_key_.reserve(32U);
    }

    [[nodiscard]] ContextPortfolioValueResult
    fold(std::span<const ContextPortfolioOwnerPolicy> owners,
         std::span<const ContextPortfolioCheckpointValue> checkpoints) {
        std::array<std::uint64_t, 32> baseline_demand{};
        std::array<std::uint64_t, 32> target_demand{};
        owner_scratch_.clear();
        for (const ContextPortfolioOwnerPolicy& policy : owners) {
            owner_scratch_.push_back(
                OwnerValue{.owner                    = policy.owner,
                           .private_retention_weight = policy.private_retention_weight,
                           .explicit_shared_credit   = policy.explicit_shared_credit});
        }
        // Order-preserving lookup index over owner_scratch_: each checkpoint resolves its
        // owner in O(log n) instead of a linear scan (O(n) per checkpoint, which is the
        // dominant term when both owners and checkpoints number in the thousands).  The
        // sorted key run also carries the duplicate-ID check the linear scan used to do.
        owner_key_.resize(owner_scratch_.size());
        for (std::size_t index = 0; index < owner_scratch_.size(); ++index) {
            owner_key_[index] = {.value = owner_scratch_[index].owner.value, .position = index};
        }
        std::sort(owner_key_.begin(), owner_key_.end(),
                  [](const OwnerKey& left, const OwnerKey& right) { return left.value < right.value; });
        for (std::size_t index = 1; index < owner_key_.size(); ++index) {
            if (owner_key_[index].value == owner_key_[index - 1].value) {
                throw std::logic_error("portfolio owner policy ID is duplicated");
            }
        }

        for (const ContextPortfolioCheckpointValue& checkpoint : checkpoints) {
            const auto key = std::lower_bound(
                owner_key_.begin(), owner_key_.end(), checkpoint.owner.value,
                [](const OwnerKey& entry, std::uint32_t value) { return entry.value < value; });
            if (key == owner_key_.end() || key->value != checkpoint.owner.value) {
                throw std::logic_error("portfolio checkpoint has no owner policy");
            }
            OwnerValue& owner = owner_scratch_[key->position];
            const std::uint64_t baseline_saving =
                checkpoint.rebuild_ns > checkpoint.baseline_recovery_ns
                    ? checkpoint.rebuild_ns - checkpoint.baseline_recovery_ns
                    : 0;
            const std::uint64_t target_saving =
                checkpoint.rebuild_ns > checkpoint.target_recovery_ns
                    ? checkpoint.rebuild_ns - checkpoint.target_recovery_ns
                    : 0;
            owner.baseline_best = std::max(owner.baseline_best, baseline_saving);
            owner.target_best   = std::max(owner.target_best, target_saving);
            owner.private_transition_loss =
                std::max(owner.private_transition_loss,
                         baseline_saving > target_saving ? baseline_saving - target_saving : 0);
            for (std::uint32_t bit = 0; bit < 32U; ++bit) {
                if ((checkpoint.demand_mask & (1U << bit)) == 0) { continue; }
                baseline_demand[bit] = std::max(baseline_demand[bit], baseline_saving);
                target_demand[bit]   = std::max(target_demand[bit], target_saving);
            }
        }

        ContextPortfolioValueResult result;
        for (std::size_t bit = 0; bit < baseline_demand.size(); ++bit) {
            add(result.baseline_public_value, baseline_demand[bit], result.saturated);
            add(result.target_public_value, target_demand[bit], result.saturated);
        }
        for (const OwnerValue& owner : owner_scratch_) {
            if (owner.private_retention_weight != 0) {
                add_weighted(result.private_transition_loss, owner.private_transition_loss,
                             owner.private_retention_weight, result.saturated);
            }
            if (owner.explicit_shared_credit) {
                add(result.baseline_public_value, owner.baseline_best, result.saturated);
                add(result.target_public_value, owner.target_best, result.saturated);
            }
        }
        return result;
    }

private:
    struct OwnerValue {
        PlanningOwnerId owner;
        std::uint32_t private_retention_weight = 0;
        bool explicit_shared_credit            = false;
        std::uint64_t baseline_best            = 0;
        std::uint64_t target_best              = 0;
        std::uint64_t private_transition_loss  = 0;
    };

    static void add(std::uint64_t& value, std::uint64_t increment, bool& saturated) noexcept {
        if (increment > std::numeric_limits<std::uint64_t>::max() - value) {
            value     = std::numeric_limits<std::uint64_t>::max();
            saturated = true;
        } else {
            value += increment;
        }
    }

    static void add_weighted(std::uint64_t& value, std::uint64_t increment, std::uint32_t weight,
                             bool& saturated) noexcept {
        if (weight != 0 && increment > std::numeric_limits<std::uint64_t>::max() / weight) {
            add(value, std::numeric_limits<std::uint64_t>::max(), saturated);
            saturated = true;
            return;
        }
        add(value, increment * weight, saturated);
    }

    struct OwnerKey {
        std::uint32_t value    = 0;
        std::size_t position   = 0;
    };

    std::vector<OwnerValue> owner_scratch_;
    std::vector<OwnerKey> owner_key_;
};

} // namespace ninfer::runtime
