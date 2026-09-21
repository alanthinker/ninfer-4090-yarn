#include "ops/linear_add/q5/q5_linear_add_plan.h"

#include "ops/linear_add/q5/q5_linear_add_kernels.h"

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

struct SupportSpec {
    std::int32_t rows;
    std::int32_t k;
    std::int32_t padded_k;
};

struct RouteSpec {
    ColsSet cols;
    Q5LinearAddScheduleId schedule;
};

constexpr std::array<SupportSpec, 2> kSupports{{
    {5120, 6144, 6144},
    {5120, 17408, 17408},
}};

constexpr std::array<RouteSpec, 6> kK6144Routes{{
    {{1, 1}, Q5LinearAddScheduleId::GemvResidual},
    {{2, 13}, Q5LinearAddScheduleId::Split2ExactResidual},
    {{14, 32}, Q5LinearAddScheduleId::MmaResidualR64C16},
    {{33, 48}, Q5LinearAddScheduleId::MmaResidualR64C24},
    {{49, 128}, Q5LinearAddScheduleId::MmaResidualR64C64},
    {{129, kAnyCols}, Q5LinearAddScheduleId::MmaResidualR64C128},
}};

constexpr std::array<RouteSpec, 6> kK17408Routes{{
    {{1, 1}, Q5LinearAddScheduleId::GemvResidual},
    {{2, 16}, Q5LinearAddScheduleId::Split2ExactResidual},
    {{17, 32}, Q5LinearAddScheduleId::MmaResidualR64C16},
    {{33, 48}, Q5LinearAddScheduleId::MmaResidualR64C24},
    {{49, 128}, Q5LinearAddScheduleId::MmaResidualR64C64},
    {{129, kAnyCols}, Q5LinearAddScheduleId::MmaResidualR64C128},
}};

// AllowA8 replaces the widest interval - the one the BF16 catalog already gives
// to MmaResidualR64C128 - with the INT8 prefill route. Measured on
// RTX 4090 / sm_89 against that route, activation quantization included:
// 1.50x at T=256 and 1.62x/1.49x at T=1024 for k=6144/17408. Every narrower
// interval is shared with A16 and stays bit-identical.
//
// Group-64 activation quantization is a semantic boundary, not a private
// implementation choice: it costs roughly 5x the BF16 route's error against the
// FP64 oracle on dense activations. It is therefore reachable only through
// AllowA8, exactly as NVFP4's W4A4 routes are reachable only through AllowA4.
// See docs/maintainer/op-development.md section 2.1.
constexpr std::array<RouteSpec, 6> kK6144A8Routes{{
    {{1, 1}, Q5LinearAddScheduleId::GemvResidual},
    {{2, 13}, Q5LinearAddScheduleId::Split2ExactResidual},
    {{14, 32}, Q5LinearAddScheduleId::MmaResidualR64C16},
    {{33, 48}, Q5LinearAddScheduleId::MmaResidualR64C24},
    {{49, 128}, Q5LinearAddScheduleId::MmaResidualR64C64},
    {{129, kAnyCols}, Q5LinearAddScheduleId::Int8ResidualR64C128},
}};

constexpr std::array<RouteSpec, 6> kK17408A8Routes{{
    {{1, 1}, Q5LinearAddScheduleId::GemvResidual},
    {{2, 16}, Q5LinearAddScheduleId::Split2ExactResidual},
    {{17, 32}, Q5LinearAddScheduleId::MmaResidualR64C16},
    {{33, 48}, Q5LinearAddScheduleId::MmaResidualR64C24},
    {{49, 128}, Q5LinearAddScheduleId::MmaResidualR64C64},
    {{129, kAnyCols}, Q5LinearAddScheduleId::Int8ResidualR64C128},
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

static_assert(catalog_is_closed(kK6144Routes) && catalog_is_closed(kK17408Routes) &&
                  catalog_is_closed(kK6144A8Routes) && catalog_is_closed(kK17408A8Routes),
              "Q5 LinearAdd routes must be exact, contiguous, and closed");

// Both catalogs for a given K are iterated through this, so a policy never sees
// a partial view.
template <class Visit>
auto visit_routes(std::int32_t k, LinearPolicy policy, Visit&& visit) {
    if (policy == LinearPolicy::AllowA8) {
        return k == 6144 ? visit(kK6144A8Routes) : visit(kK17408A8Routes);
    }
    return k == 6144 ? visit(kK6144Routes) : visit(kK17408Routes);
}

// Group-64 quantized activation codes plus their FP32 group scales. The GEMM
// contracts whole 256-element weight super-groups, so K is padded to that
// boundary and the tail groups are zero-filled by the quantizer.
template <class Allocator>
Q5LinearAddInt8Workspace allocate_int8_workspace(Allocator& allocator, std::int32_t k,
                                                 std::int32_t cols) {
    const std::int32_t padded_k = q5_linear_add_int8_padded_k(k);
    const std::int32_t groups   = padded_k / 64;
    const std::int32_t tile     = q5_linear_add_int8_token_tile(cols);
    Tensor codes                = allocator.alloc(DType::I8, {padded_k, tile});
    Tensor scales               = allocator.alloc(DType::FP32, {groups, tile});
    return {static_cast<std::int8_t*>(codes.data), static_cast<float*>(scales.data)};
}

std::size_t int8_workspace_bytes(std::int32_t k, std::int32_t cols) {
    WorkspaceLayoutBuilder layout;
    (void)allocate_int8_workspace(layout, k, cols);
    return layout.peak_bytes(1);
}

bool supported_shape(const Q5LinearAddProblem& problem) noexcept {
    for (const SupportSpec& support : kSupports) {
        if (problem.rows == support.rows && problem.k == support.k &&
            problem.padded_k == support.padded_k) {
            return true;
        }
    }
    return false;
}

} // namespace

const char* q5_linear_add_schedule_name(Q5LinearAddScheduleId schedule) noexcept {
    switch (schedule) {
    case Q5LinearAddScheduleId::GemvResidual:
        return "linear_add.q5.gemv.residual";
    case Q5LinearAddScheduleId::Split2ExactResidual:
        return "linear_add.q5.simt.split2.exact.residual";
    case Q5LinearAddScheduleId::MmaResidualR64C16:
        return "linear_add.q5.mma.r64.c16.cta_collective_residual";
    case Q5LinearAddScheduleId::MmaResidualR64C24:
        return "linear_add.q5.mma.r64.c24.cta_collective_residual";
    case Q5LinearAddScheduleId::MmaResidualR64C64:
        return "linear_add.q5.mma.r64.c64.cta_collective_residual";
    case Q5LinearAddScheduleId::MmaResidualR64C128:
        return "linear_add.q5.mma.r64.c128.cta_collective_residual";
    case Q5LinearAddScheduleId::Int8ResidualR64C128:
        return "linear_add.q5.int8.r64.c128.residual";
    }
    return "linear_add.q5.unknown";
}

bool q5_linear_add_admits(const Q5LinearAddProblem& problem) noexcept {
    return supported_shape(problem) && problem.cols >= 1;
}

Q5LinearAddPlan q5_linear_add_resolve_plan(const Q5LinearAddProblem& problem, LinearPolicy policy) {
    if (!q5_linear_add_admits(problem)) {
        throw std::invalid_argument("q5 linear_add: exact problem or column count is not admitted");
    }

    return visit_routes(problem.k, policy, [&](const auto& routes) -> Q5LinearAddPlan {
        for (const RouteSpec& route : routes) {
            if (!route.cols.contains(problem.cols)) { continue; }
            if (route.schedule == Q5LinearAddScheduleId::Int8ResidualR64C128) {
                return {route.schedule, int8_workspace_bytes(problem.k, problem.cols)};
            }
            return {route.schedule, 0};
        }
        throw std::logic_error("q5 linear_add: admitted problem has no covering route");
    });
}

std::size_t q5_linear_add_capacity_workspace_bytes(std::int32_t rows, std::int32_t k,
                                                   std::int32_t padded_k, std::int32_t min_cols,
                                                   std::int32_t max_cols, LinearPolicy policy) {
    if (min_cols <= 0 || max_cols < min_cols) {
        throw std::invalid_argument("q5 linear_add: invalid column interval");
    }
    (void)q5_linear_add_resolve_plan({rows, k, padded_k, min_cols}, policy);

    // The INT8 staging grows with the token count up to the tile cap, so the
    // interval's high-water mark is at max_cols; every other route is
    // workspace-free.
    return q5_linear_add_resolve_plan({rows, k, padded_k, max_cols}, policy).workspace_bytes;
}

void q5_linear_add_execute_plan(const Q5LinearAddPlan& plan, const Tensor& x, const Weight& w,
                                Tensor& residual_out, WorkspaceArena& ws, LinearPolicy policy,
                                cudaStream_t stream) {
    const Q5LinearAddProblem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q5LinearAddPlan resolved = q5_linear_add_resolve_plan(problem, policy);
    if (resolved.schedule != plan.schedule || resolved.workspace_bytes != plan.workspace_bytes) {
        throw std::invalid_argument("q5 linear_add: plan does not match the exact problem");
    }

    switch (plan.schedule) {
    case Q5LinearAddScheduleId::Int8ResidualR64C128: {
        auto scratch_scope                     = ws.scope();
        const Q5LinearAddInt8Workspace scratch = allocate_int8_workspace(ws, problem.k,
                                                                        problem.cols);
        q5_linear_add_int8_residual_launch(x, w, residual_out, scratch, stream);
        return;
    }
    case Q5LinearAddScheduleId::GemvResidual:
        q5_linear_add_gemv_residual_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::Split2ExactResidual:
        q5_linear_add_split2_exact_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::MmaResidualR64C16:
        q5_linear_add_mma_r64_c16_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::MmaResidualR64C24:
        q5_linear_add_mma_r64_c24_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::MmaResidualR64C64:
        q5_linear_add_mma_r64_c64_launch(x, w, residual_out, stream);
        return;
    case Q5LinearAddScheduleId::MmaResidualR64C128:
        q5_linear_add_mma_r64_c128_launch(x, w, residual_out, stream);
        return;
    }
    throw std::logic_error("q5 linear_add: unknown schedule");
}

void q5_linear_add_dispatch(const Tensor& x, const Weight& w, Tensor& residual_out,
                            WorkspaceArena& ws, LinearPolicy policy, cudaStream_t stream) {
    const Q5LinearAddProblem problem{residual_out.ne[0], x.ne[0], w.padded_shape[1], x.ne[1]};
    const Q5LinearAddPlan plan = q5_linear_add_resolve_plan(problem, policy);
    q5_linear_add_execute_plan(plan, x, w, residual_out, ws, policy, stream);
}

} // namespace ninfer::ops::detail
