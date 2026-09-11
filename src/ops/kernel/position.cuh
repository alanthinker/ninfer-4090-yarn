#pragma once

#include <cstdint>

namespace ninfer::ops {

__global__ void fill_i32_positions_kernel(std::int32_t* positions, std::int32_t count,
                                          std::int32_t start) {
    const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) { positions[i] = start + i; }
}

__global__ void offset_i32_positions_kernel(const std::int32_t* source, const std::int32_t* delta,
                                            std::int32_t* destination, std::int32_t count) {
    const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) { destination[i] = source[i] + delta[0]; }
}

__global__ void scale_positions_yarn_kernel(std::int32_t* positions, std::int32_t count,
                                            std::int32_t original_context, float factor) {
    const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) {
        const std::int32_t p = positions[i];
        if (p > original_context) {
            // Two things matter here and both are deliberate.
            //
            // The rounding term is evaluated at small magnitude and original_context is added
            // afterwards: folding original_context into a single-precision sum first loses the
            // +0.5 to the float spacing at that magnitude (~0.0156 near 310k), which biases the
            // result and would disagree with the ramp used on the host.
            //
            // The quotient is computed in double. The host-side ramp for the decode path uses the
            // same arithmetic so that one token position maps to one RoPE position regardless of
            // whether it arrives through prefill or decode. A single-precision quotient sits close
            // enough to the .5 boundary that the two paths would otherwise disagree for a
            // measurable fraction of positions.
            const double delta = static_cast<double>(p - original_context);
            positions[i] =
                original_context + static_cast<std::int32_t>(delta / static_cast<double>(factor) +
                                                             0.5);
        }
    }
}

} // namespace ninfer::ops
