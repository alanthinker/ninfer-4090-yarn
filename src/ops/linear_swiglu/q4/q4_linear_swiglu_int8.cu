#include "ops/linear_swiglu/q4/q4_linear_swiglu_kernels.h"

#include "ops/common/act_quant_g64.h"
#include "ops/linear_swiglu/q4/q4_linear_swiglu_int8_gemm.cuh"

#include "core/device.h"
#include "ops/common/math.h"

#include <algorithm>
#include <cstdint>

namespace ninfer::ops::detail {
namespace {

// Tuned on RTX 4090 / sm_89 against the registered BF16 fused route. The fold
// requires a warp to own both a gate tile and its matching up tile, so the warp
// tile carries 2 * kMmaPairRows accumulator rows; WPR=16 with WN=128 is the
// combination that keeps that at 128 FP32 accumulator registers while giving the
// widest N reuse per decoded weight fragment.
//
//   T:      256    512    1024   2048
//   ratio:  1.59x  1.68x  1.78x  1.87x   (activation quantization included)
using Int8FoldedCfg = Q4Int8SwiGluSchedule<64, 256, 16, 128, 3, 1>;

template <class Cfg, bool Full>
void launch_folded_int8(const std::int8_t* xq, const float* xs, const Weight& weight, Tensor& out,
                        std::int32_t tokens, std::int32_t padded_k, cudaStream_t stream) {
    static const bool configured = [] {
        CUDA_CHECK(cudaFuncSetAttribute(q4_linear_swiglu_int8_gemm_kernel<Cfg, Full>,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize,
                                        Cfg::kSharedBytes));
        return true;
    }();
    (void)configured;

    const dim3 grid(static_cast<unsigned>(div_up(out.ne[0], Cfg::kPairRows)),
                    static_cast<unsigned>(div_up(tokens, Cfg::kBlockCols)));
    q4_linear_swiglu_int8_gemm_kernel<Cfg, Full>
        <<<grid, Cfg::kThreads, Cfg::kSharedBytes, stream>>>(
            xq, xs, static_cast<const std::uint8_t*>(weight.qdata),
            static_cast<const std::uint8_t*>(weight.scales), static_cast<__nv_bfloat16*>(out.data),
            out.ne[0], tokens, padded_k);
    CUDA_CHECK(cudaGetLastError());
}

} // namespace

void q4_linear_swiglu_int8_folded_launch(const Tensor& x, const Weight& weight, Tensor& out,
                                         const Q4LinearSwiGluInt8Workspace& scratch,
                                         cudaStream_t stream) {
    using Cfg                  = Int8FoldedCfg;
    const std::int32_t k       = x.ne[0];
    const std::int32_t padded_k = q4_linear_swiglu_int8_padded_k(k);

    // The token loop is bounded by the workspace tile, not by the CUDA grid
    // limit: the quantized activation staging is O(tokens) and must not scale
    // with the context length. Each tile is quantized and contracted before the
    // next reuses the same staging buffers.
    const std::int32_t tile = q4_linear_swiglu_int8_token_tile(x.ne[1]);
    for (std::int32_t offset = 0; offset < x.ne[1]; offset += tile) {
        const std::int32_t count = std::min(tile, x.ne[1] - offset);
        const Tensor x_slice     = x.slice(1, offset, count);
        Tensor out_slice         = out.slice(1, offset, count);
        act_quant_g64_launch(static_cast<const __nv_bfloat16*>(x_slice.data), scratch.codes,
                             scratch.scales, k, count, padded_k, stream);
        const bool full = (count % Cfg::kBlockCols) == 0 && (out.ne[0] % Cfg::kPairRows) == 0;
        if (full) {
            launch_folded_int8<Cfg, true>(scratch.codes, scratch.scales, weight, out_slice, count,
                                          padded_k, stream);
        } else {
            launch_folded_int8<Cfg, false>(scratch.codes, scratch.scales, weight, out_slice, count,
                                           padded_k, stream);
        }
    }
}

std::int32_t q4_linear_swiglu_int8_block_cols() noexcept { return Int8FoldedCfg::kBlockCols; }

} // namespace ninfer::ops::detail
