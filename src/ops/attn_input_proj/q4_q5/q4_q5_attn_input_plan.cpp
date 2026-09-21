#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_plan.h"

#include "ops/attn_input_proj/q4_q5/q4_q5_attn_input_kernels.h"

#include "core/layout.h"

#include <array>
#include <limits>
#include <stdexcept>

namespace ninfer::ops::detail {
namespace {

constexpr std::int32_t kAnyCols = std::numeric_limits<std::int32_t>::max();

struct ColsSet {
    std::int32_t first;
    std::int32_t last;

    constexpr bool contains(std::int32_t cols) const noexcept {
        return cols >= first && cols <= last;
    }
};

struct RouteSpec {
    ColsSet cols;
    Q4Q5AttnInputScheduleId schedule;
};

constexpr std::array<RouteSpec, 3> kA16Routes{{
    {{1, 16}, Q4Q5AttnInputScheduleId::ParentSplitFixed},
    {{17, 20}, Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR16C64S3},
    {{21, kAnyCols}, Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR32C64S4},
}};

// AllowA8 replaces the widest interval with the INT8 prefill route. The BF16
// grouped route reaches only 85.8 TFLOP/s here - 51.9% of the BF16 tensor-core
// rate - so this is the least well served of the dense Ops; the INT8 jobs run at
// 234.5/212.5 TFLOP/s on the 6144-row halves. Group-64 activation quantization
// is a declared semantic boundary and is reachable only through AllowA8. See
// docs/maintainer/op-development.md section 2.1.
// A8 is a single route over every T. A token's group-64 activation scale
// depends only on that token, and the schedule is selected by output rows
// alone, so a token's outputs do not depend on the width of the call that
// produced it. Prefix reuse therefore replays a cached prefix exactly. Splitting
// the interval and leaving small T on a BF16 route would break that invariance:
// a reused prefix computed at T=4 would disagree with the same tokens computed
// inside a full-prompt chunk.
constexpr std::array<RouteSpec, 1> kA8Routes{{
    {{1, kAnyCols}, Q4Q5AttnInputScheduleId::Int8Pairs},
}};

template <std::size_t N>
constexpr bool catalog_is_closed(const std::array<RouteSpec, N>& routes) noexcept {
    std::int64_t expected = 1;
    for (const RouteSpec& route : routes) {
        if (route.cols.first != expected || route.cols.last < route.cols.first) { return false; }
        expected = static_cast<std::int64_t>(route.cols.last) + 1;
    }
    return routes.back().cols.last == kAnyCols &&
           expected == static_cast<std::int64_t>(kAnyCols) + 1;
}

static_assert(catalog_is_closed(kA16Routes) && catalog_is_closed(kA8Routes),
              "attention input routes must be exact and closed");

template <class Visit>
auto visit_routes(LinearPolicy policy, Visit&& visit) {
    if (policy == LinearPolicy::AllowA8) { return visit(kA8Routes); }
    return visit(kA16Routes);
}

template <class Allocator>
Int8ProjWorkspace allocate_int8_workspace(Allocator& allocator, std::int32_t k, std::int32_t cols) {
    const std::int32_t padded_k = int8_proj_padded_k(k);
    const std::int32_t groups   = padded_k / 64;
    const std::int32_t tile     = int8_proj_token_tile(cols);
    Tensor codes                = allocator.alloc(DType::I8, {padded_k, tile});
    Tensor scales               = allocator.alloc(DType::FP32, {groups, tile});
    return {static_cast<std::int8_t*>(codes.data), static_cast<float*>(scales.data)};
}

std::size_t int8_workspace_bytes(std::int32_t k, std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_int8_workspace(layout, k, cols);
    return layout.peak_bytes(1);
}

bool supported_shape(const Q4Q5AttnInputProblem& problem) noexcept {
    return problem.input_rows == 5120 && problem.query_rows == 6144 && problem.kv_rows == 1024 &&
           problem.padded_k == 5120;
}

} // namespace

const char* q4_q5_attn_input_schedule_name(Q4Q5AttnInputScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4Q5AttnInputScheduleId::ParentSplitFixed:
        return "attn_input_proj.q4_q5.parent_split_fixed";
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR16C64S3:
        return "attn_input_proj.q4_q5.grouped_homogeneous_pair.mma.r16.c64.s3";
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR32C64S4:
        return "attn_input_proj.q4_q5.grouped_homogeneous_pair.mma.r32.c64.s4";
    case Q4Q5AttnInputScheduleId::Int8Pairs:
        return "attn_input_proj.q4_q5.int8.pairs";
    }
    return "attn_input_proj.q4_q5.unknown";
}

bool q4_q5_attn_input_admits(const Q4Q5AttnInputProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q4Q5AttnInputPlan q4_q5_attn_input_resolve_plan(const Q4Q5AttnInputProblem& problem,
                                                LinearPolicy policy) {
    if (!q4_q5_attn_input_admits(problem)) {
        throw std::invalid_argument(
            "Q4/Q5 attention input: exact problem or column count is not admitted");
    }

    return visit_routes(policy, [&](const auto& routes) -> Q4Q5AttnInputPlan {
        for (const RouteSpec& route : routes) {
            if (!route.cols.contains(problem.cols)) { continue; }
            if (route.schedule == Q4Q5AttnInputScheduleId::Int8Pairs) {
                return {route.schedule, int8_workspace_bytes(problem.input_rows, problem.cols)};
            }
            return {route.schedule, 0};
        }
        throw std::logic_error("Q4/Q5 attention input: admitted problem has no covering route");
    });
}

std::size_t q4_q5_attn_input_capacity_workspace_bytes(std::int32_t min_tokens,
                                                      std::int32_t max_tokens,
                                                      LinearPolicy policy) {
    if (min_tokens <= 0 || max_tokens < min_tokens) {
        throw std::invalid_argument("Q4/Q5 attention input: invalid token interval");
    }
    const Q4Q5AttnInputProblem lo{5120, 6144, 1024, 5120, min_tokens};
    const Q4Q5AttnInputProblem hi{5120, 6144, 1024, 5120, max_tokens};
    (void)q4_q5_attn_input_resolve_plan(lo, policy);
    // The INT8 staging grows with the token count up to the tile cap, so the
    // interval's high-water mark is at max_tokens; every other route is
    // workspace-free.
    return q4_q5_attn_input_resolve_plan(hi, policy).workspace_bytes;
}

void q4_q5_attn_input_execute_plan(const Q4Q5AttnInputPlan& plan, const Tensor& x,
                                   const Weight& query_key_weight, const Weight& gate_value_weight,
                                   Tensor& q, Tensor& gate, Tensor& k, Tensor& v,
                                   WorkspaceArena* ws, LinearPolicy policy, cudaStream_t stream) {
    const Q4Q5AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], query_key_weight.padded_shape[1],
                                       x.ne[1]};
    const Q4Q5AttnInputPlan resolved = q4_q5_attn_input_resolve_plan(problem, policy);
    if (resolved.schedule != plan.schedule || resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("Q4/Q5 attention input: plan does not match exact problem");
    }

    switch (plan.schedule) {
    case Q4Q5AttnInputScheduleId::Int8Pairs: {
        if (ws == nullptr) {
            throw std::invalid_argument("Q4/Q5 attention input: INT8 route requires a workspace");
        }
        auto scratch_scope             = ws->scope();
        const Int8ProjWorkspace scratch = allocate_int8_workspace(*ws, problem.input_rows,
                                                                  problem.cols);
        q4_q5_attn_input_int8_launch(x, query_key_weight, gate_value_weight, q, gate, k, v, scratch,
                                     stream);
        return;
    }
    case Q4Q5AttnInputScheduleId::ParentSplitFixed:
        q4_q5_attn_input_small_t_launch(x, query_key_weight, gate_value_weight, q, gate, k, v,
                                        stream);
        return;
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR16C64S3:
        q4_q5_attn_input_grouped_mma_r16_c64_s3_launch(x, query_key_weight, gate_value_weight, q,
                                                       gate, k, v, stream);
        return;
    case Q4Q5AttnInputScheduleId::GroupedHomogeneousPairMmaR32C64S4:
        q4_q5_attn_input_grouped_mma_r32_c64_s4_launch(x, query_key_weight, gate_value_weight, q,
                                                       gate, k, v, stream);
        return;
    }
    throw std::logic_error("Q4/Q5 attention input: unknown schedule");
}

void q4_q5_attn_input_dispatch(const Tensor& x, const Weight& query_key_weight,
                               const Weight& gate_value_weight, Tensor& q, Tensor& gate, Tensor& k,
                               Tensor& v, WorkspaceArena* ws, LinearPolicy policy,
                               cudaStream_t stream) {
    const Q4Q5AttnInputProblem problem{x.ne[0], q.ne[0], k.ne[0], query_key_weight.padded_shape[1],
                                       x.ne[1]};
    const Q4Q5AttnInputPlan plan = q4_q5_attn_input_resolve_plan(problem, policy);
    q4_q5_attn_input_execute_plan(plan, x, query_key_weight, gate_value_weight, q, gate, k, v, ws,
                                  policy, stream);
}

} // namespace ninfer::ops::detail
