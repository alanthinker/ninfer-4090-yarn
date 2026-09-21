#pragma once

// INT8 Tensor Core primitives for large-token (prefill) quantized GEMM on sm_89.
//
// Ada runs mma.sync.m16n8k32.s32.s8.s8.s32 at 660.6 TOPS, four times the
// 165.2 TFLOPS BF16-with-FP32-accumulate rate, and covers twice the K depth per
// instruction. Q4G64 codes live in [-8, 7] and Q5G64 codes in [-16, 15], so both
// fit INT8 exactly: the weight side of an INT8 route carries no additional
// quantization error and needs no floating-point decode at all.
//
// The helpers below convert four consecutive stored codes into the four signed
// bytes of one MMA A-fragment register using integer logic only. Both forms were
// verified exhaustively against Q4SimtDecodeAtom and Q5ScalarDecodeAtom
// (all 65536 Q4 code pairs; all Q5 code-pair x high-byte x nibble-offset cases).

#include "ops/common/mma.cuh"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

namespace ninfer::ops::detail {

// Activation quantization group. Matches Q4G64/Q5G64 so one activation scale
// spans exactly the K extent of one weight group.
inline constexpr int kInt8ActGroupK = 64;

// Symmetric INT8 range. -128 is excluded so negation is exact and the decode
// stays symmetric with the weight codes.
inline constexpr float kInt8ActMax = 127.0f;

// Spread four consecutive stored codes into one nibble per byte lane, in K order.
//
// Storage packs code 2j in the low nibble of byte j and code 2j+1 in the high
// nibble, so two consecutive bytes carry exactly the four codes an A-fragment
// register needs. __byte_perm replicates the pair to {b0, b0, b1, b1}; the two
// masked selections then place n0..n3 in byte lanes 0..3.
__device__ __forceinline__ std::uint32_t int8_spread_code_nibbles(std::uint32_t packed_pair) {
    const std::uint32_t d = __byte_perm(packed_pair, 0u, 0x1100u);
    return (d & 0x000F000Fu) | ((d >> 4) & 0x0F000F00u);
}

// Sign-extend a 4-bit code in each byte lane to int8.
// (x & 0x08) * 0x1E == 0xF0 exactly when the code is negative, and 0x08 * 0x1E
// fits one byte, so the multiply never carries between lanes.
__device__ __forceinline__ std::uint32_t int8_sign_extend_4(std::uint32_t x) {
    return x | ((x & 0x08080808u) * 0x1Eu);
}

// Sign-extend a 5-bit code in each byte lane to int8. 0x10 * 0x0E == 0xE0.
__device__ __forceinline__ std::uint32_t int8_sign_extend_5(std::uint32_t x) {
    return x | ((x & 0x10101010u) * 0x0Eu);
}

// Place four consecutive Q5 high bits at bit 4 of byte lanes 0..3.
//
// The high plane is a lane-major bitstream: the high bit of code c is
// high[c >> 3] bit (c & 7). Four consecutive codes therefore occupy four
// consecutive bits of one byte, starting at bit 0 or bit 4. The multiply moves
// bit i to bit 8i + 4; the 16 partial products land on 16 distinct bit
// positions, so nothing carries and the mask keeps only the four wanted bits.
__device__ __forceinline__ std::uint32_t int8_spread_high_bits(std::uint32_t high_byte,
                                                               int bit_offset) {
    const std::uint32_t h = (high_byte >> bit_offset) & 0xFu;
    return (h * 0x02040810u) & 0x10101010u;
}

// Four consecutive Q4G64 codes -> one MMA A-fragment register (4 signed bytes).
__device__ __forceinline__ std::uint32_t q4_codes_to_s8x4(std::uint32_t packed_pair) {
    return int8_sign_extend_4(int8_spread_code_nibbles(packed_pair));
}

// Four consecutive Q5G64 codes -> one MMA A-fragment register (4 signed bytes).
// `bit_offset` is (first_code_index & 7); it is 0 or 4 for every A-fragment.
__device__ __forceinline__ std::uint32_t q5_codes_to_s8x4(std::uint32_t packed_pair,
                                                          std::uint32_t high_byte, int bit_offset) {
    const std::uint32_t nibbles = int8_spread_code_nibbles(packed_pair);
    return int8_sign_extend_5(nibbles | int8_spread_high_bits(high_byte, bit_offset));
}

// Round-to-nearest-even float -> int8 with saturation, matching the quantizer
// oracle. cvt.rni.s8.f32 saturates, so no explicit clamp is needed.
__device__ __forceinline__ int int8_quantize_rn(float v) {
    int r;
    asm volatile("cvt.rni.sat.s8.f32 %0, %1;" : "=r"(r) : "f"(v));
    return r;
}

// Pack four signed values into the four byte lanes of one word.
__device__ __forceinline__ std::uint32_t int8_pack_s8x4(int a, int b, int c, int d) {
    const std::uint32_t lo = __byte_perm(static_cast<std::uint32_t>(a),
                                         static_cast<std::uint32_t>(b), 0x0040u);
    const std::uint32_t hi = __byte_perm(static_cast<std::uint32_t>(c),
                                         static_cast<std::uint32_t>(d), 0x0040u);
    return __byte_perm(lo, hi, 0x5410u);
}

} // namespace ninfer::ops::detail
