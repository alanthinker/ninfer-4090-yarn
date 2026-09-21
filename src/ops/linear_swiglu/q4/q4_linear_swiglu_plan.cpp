#include "ops/linear_swiglu/q4/q4_linear_swiglu_plan.h"

#include "ninfer/ops/linear.h"
#include "ninfer/ops/silu_mul.h"
#include "core/layout.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"

#include <algorithm>
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
    Q4LinearSwiGluScheduleId schedule;
};

constexpr Q4LinearSwiGluProblem kShape{34816, 17408, 5120, 5120, 1};

constexpr std::array<RouteSpec, 10> kA16Routes{{
    {{1, 1}, Q4LinearSwiGluScheduleId::GemvPair},
    {{2, 32}, Q4LinearSwiGluScheduleId::SmallTExact},
    {{33, 40}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C40},
    {{41, 48}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C48},
    {{49, 128}, Q4LinearSwiGluScheduleId::Materialized},
    {{129, 256}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128},
    {{257, 384}, Q4LinearSwiGluScheduleId::Materialized},
    {{385, 512}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128},
    {{513, 640}, Q4LinearSwiGluScheduleId::Materialized},
    {{641, kAnyCols}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128},
}};

// AllowA8 replaces every route above the crossover with the INT8 prefill route.
// Measured on RTX 4090 / sm_89 against the BF16 fused route, activation
// quantization included: 1.59x at T=256, 1.68x at T=512, 1.78x at T=1024, 1.87x
// at T=2048. INT8 loses below the crossover (0.94x at T=128), so the decode and
// small-T intervals are shared with A16 and stay bit-identical. See
// docs/maintainer/linear-benchmark.md section 10.
//
// Group-64 activation quantization is a semantic boundary, not a private
// implementation choice: it costs roughly 8x the BF16 route's error against the
// FP64 oracle on dense activations. It is therefore reachable only through
// AllowA8, exactly as NVFP4's W4A4 routes are reachable only through AllowA4.
constexpr std::array<RouteSpec, 7> kA8Routes{{
    {{1, 1}, Q4LinearSwiGluScheduleId::GemvPair},
    {{2, 32}, Q4LinearSwiGluScheduleId::SmallTExact},
    {{33, 40}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C40},
    {{41, 48}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C48},
    {{49, 128}, Q4LinearSwiGluScheduleId::Materialized},
    {{129, 256}, Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128},
    {{257, kAnyCols}, Q4LinearSwiGluScheduleId::Int8FoldedR64C256},
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

static_assert(catalog_is_closed(kA16Routes),
              "Q4 LinearSwiGLU A16 routes must be exact, contiguous, and closed");
static_assert(catalog_is_closed(kA8Routes),
              "Q4 LinearSwiGLU A8 routes must be exact, contiguous, and closed");

// Both catalogs are iterated through this, so a policy never sees a partial view.
// The two arrays differ in size, so the branches cannot be a conditional expression.
template <class Visit>
auto visit_routes(LinearPolicy policy, Visit&& visit) {
    if (policy == LinearPolicy::AllowA8) { return visit(kA8Routes); }
    return visit(kA16Routes);
}

bool supported_shape(const Q4LinearSwiGluProblem& problem) noexcept {
    return problem.gate_up_rows == kShape.gate_up_rows &&
           problem.output_rows == kShape.output_rows && problem.k == kShape.k &&
           problem.padded_k == kShape.padded_k;
}

template <class Allocator>
Tensor allocate_materialized_workspace(Allocator& allocator, std::int32_t rows, std::int32_t cols) {
    return allocator.alloc(DType::BF16, {rows, cols});
}

std::size_t materialized_workspace_bytes(std::int32_t rows, std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_materialized_workspace(layout, rows, cols);
    return layout.peak_bytes(1);
}

// Group-64 quantized activation codes plus their FP32 group scales. The GEMM
// contracts whole 256-element weight super-groups, so K is padded to that
// boundary and the tail groups are zero-filled by the quantizer.
template <class Allocator>
Q4LinearSwiGluInt8Workspace allocate_int8_workspace(Allocator& allocator, std::int32_t k,
                                                    std::int32_t cols) {
    const std::int32_t padded_k = q4_linear_swiglu_int8_padded_k(k);
    const std::int32_t groups   = padded_k / 64;
    const std::int32_t tile     = q4_linear_swiglu_int8_token_tile(cols);
    Tensor codes                = allocator.alloc(DType::I8, {padded_k, tile});
    Tensor scales               = allocator.alloc(DType::FP32, {groups, tile});
    return {static_cast<std::int8_t*>(codes.data), static_cast<float*>(scales.data)};
}

std::size_t int8_workspace_bytes(std::int32_t k, std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_int8_workspace(layout, k, cols);
    return layout.peak_bytes(1);
}

} // namespace

const char* q4_linear_swiglu_schedule_name(Q4LinearSwiGluScheduleId schedule) noexcept {
    switch (schedule) {
    case Q4LinearSwiGluScheduleId::GemvPair:
        return "linear_swiglu.q4.gemv.paired_rows";
    case Q4LinearSwiGluScheduleId::SmallTExact:
        return "linear_swiglu.q4.mma.small_t.exact";
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C40:
        return "linear_swiglu.q4.mma.split_half_pair.r32.c40";
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C48:
        return "linear_swiglu.q4.mma.split_half_pair.r32.c48";
    case Q4LinearSwiGluScheduleId::Materialized:
        return "linear_swiglu.q4.materialized";
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128:
        return "linear_swiglu.q4.mma.split_half_pair.r32.c128";
    case Q4LinearSwiGluScheduleId::Int8FoldedR64C256:
        return "linear_swiglu.q4.int8.folded.r64.c256";
    }
    return "linear_swiglu.q4.unknown";
}

bool q4_linear_swiglu_admits(const Q4LinearSwiGluProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q4LinearSwiGluPlan q4_linear_swiglu_resolve_plan(const Q4LinearSwiGluProblem& problem,
                                                 LinearPolicy policy) {
    if (!q4_linear_swiglu_admits(problem)) {
        throw std::invalid_argument(
            "q4 linear_swiglu: exact problem or column count is not admitted");
    }

    return visit_routes(policy, [&](const auto& routes) {
        for (const RouteSpec& route : routes) {
            if (!route.cols.contains(problem.cols)) { continue; }
            Q4LinearSwiGluPlan plan{
                route.schedule,
                0,
            };
            switch (route.schedule) {
            case Q4LinearSwiGluScheduleId::GemvPair:
            case Q4LinearSwiGluScheduleId::SmallTExact:
            case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C40:
            case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C48:
                return plan;
            case Q4LinearSwiGluScheduleId::Materialized:
                plan.workspace_bytes =
                    materialized_workspace_bytes(problem.gate_up_rows, problem.cols);
                return plan;
            case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128:
                return plan;
            case Q4LinearSwiGluScheduleId::Int8FoldedR64C256:
                plan.workspace_bytes = int8_workspace_bytes(problem.k, problem.cols);
                return plan;
            }
        }
        throw std::logic_error("q4 linear_swiglu: admitted problem has no covering route");
    });
}

std::size_t q4_linear_swiglu_capacity_workspace_bytes(std::int32_t gate_up_rows,
                                                      std::int32_t output_rows, std::int32_t k,
                                                      std::int32_t padded_k, LinearPolicy policy,
                                                      std::int32_t min_cols,
                                                      std::int32_t max_cols) {
    if (min_cols <= 0 || max_cols < min_cols) {
        throw std::invalid_argument("q4 linear_swiglu: invalid column interval");
    }
    (void)q4_linear_swiglu_resolve_plan({gate_up_rows, output_rows, k, padded_k, min_cols}, policy);
    (void)q4_linear_swiglu_resolve_plan({gate_up_rows, output_rows, k, padded_k, max_cols}, policy);

    return visit_routes(policy, [&](const auto& routes) {
        std::size_t maximum = 0;
        for (const RouteSpec& route : routes) {
            if (route.cols.last < min_cols || route.cols.first > max_cols) { continue; }
            const std::int32_t endpoint = std::min(route.cols.last, max_cols);
            maximum = std::max(maximum, q4_linear_swiglu_resolve_plan(
                                            {gate_up_rows, output_rows, k, padded_k, endpoint},
                                            policy)
                                            .workspace_bytes);
        }
        return maximum;
    });
}

void q4_linear_swiglu_execute_plan(const Q4LinearSwiGluPlan& plan, const Tensor& x, const Weight& w,
                                   Tensor& out, LinearPolicy policy, WorkspaceArena& ws,
                                   cudaStream_t stream) {
    const Q4LinearSwiGluProblem problem{w.n, out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q4LinearSwiGluPlan resolved = q4_linear_swiglu_resolve_plan(problem, policy);
    if (resolved.schedule != plan.schedule || resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("q4 linear_swiglu: plan does not match the exact problem");
    }

    switch (plan.schedule) {
    case Q4LinearSwiGluScheduleId::GemvPair:
        q4_linear_swiglu_gemv_pair_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::SmallTExact:
        q4_linear_swiglu_small_t_exact_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C40:
        q4_linear_swiglu_mma_split_half_pair_r32_c40_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C48:
        q4_linear_swiglu_mma_split_half_pair_r32_c48_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::Materialized: {
        auto scratch_scope = ws.scope();
        Tensor gate_up = allocate_materialized_workspace(ws, problem.gate_up_rows, problem.cols);
        linear(x, w, gate_up, stream);
        silu_mul(gate_up.slice(0, 0, problem.output_rows),
                 gate_up.slice(0, problem.output_rows, problem.output_rows), out, stream);
        return;
    }
    case Q4LinearSwiGluScheduleId::MmaSplitHalfPairR32C128:
        q4_linear_swiglu_mma_split_half_pair_r32_c128_launch(x, w, out, stream);
        return;
    case Q4LinearSwiGluScheduleId::Int8FoldedR64C256: {
        auto scratch_scope = ws.scope();
        const Q4LinearSwiGluInt8Workspace scratch =
            allocate_int8_workspace(ws, problem.k, problem.cols);
        q4_linear_swiglu_int8_folded_launch(x, w, out, scratch, stream);
        return;
    }
    }
    throw std::logic_error("q4 linear_swiglu: unknown schedule");
}

void q4_linear_swiglu_dispatch(const Tensor& x, const Weight& w, Tensor& out, LinearPolicy policy,
                               WorkspaceArena& ws, cudaStream_t stream) {
    const Q4LinearSwiGluProblem problem{w.n, out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q4LinearSwiGluPlan plan = q4_linear_swiglu_resolve_plan(problem, policy);
    q4_linear_swiglu_execute_plan(plan, x, w, out, policy, ws, stream);
}

} // namespace ninfer::ops::detail
