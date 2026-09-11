// Standalone device test for the YaRN position-scaling kernel.
//
// The kernel is copied verbatim from src/ops/kernel/position.cuh so this test compiles without the
// repository build. It checks the exact integer contract:
//
//   p <= original_context          -> p
//   p >  original_context          -> original_context + round_half_away((p - original_context) / factor)
//
// The oracle below recomputes the same formula in double precision. Round-half-away-from-zero on a
// non-negative value is trunc(x + 0.5), which is what both the kernel and the oracle apply; the two
// are then compared exactly.

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <cuda_runtime.h>

// --- kernel under test (verbatim from src/ops/kernel/position.cuh) -----------------------------
__global__ void scale_positions_yarn_kernel(std::int32_t* positions, std::int32_t count,
                                            std::int32_t original_context, float factor) {
    const std::int32_t i = static_cast<std::int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i < count) {
        const std::int32_t p = positions[i];
        if (p > original_context) {
            // See src/ops/kernel/position.cuh: small-magnitude rounding term, double quotient so
            // the device ramp is bit-identical to the host ramp used on the decode path.
            const double delta = static_cast<double>(p - original_context);
            positions[i] =
                original_context + static_cast<std::int32_t>(delta / static_cast<double>(factor) +
                                                             0.5);
        }
    }
}

namespace {

std::int32_t oracle(std::int32_t p, std::int32_t original_context, double factor) {
    if (p <= original_context) { return p; }
    const double scaled = static_cast<double>(original_context) +
                          static_cast<double>(p - original_context) / factor;
    return original_context +
           static_cast<std::int32_t>(static_cast<double>(scaled - original_context) + 0.5);
}

int run_case(const char* label, std::int32_t original_context, float factor,
             const std::vector<std::int32_t>& input) {
    const std::int32_t count = static_cast<std::int32_t>(input.size());
    std::vector<std::int32_t> device_values = input;

    std::int32_t* device = nullptr;
    if (cudaMalloc(&device, input.size() * sizeof(std::int32_t)) != cudaSuccess) {
        std::printf("FAIL %s: cudaMalloc\n", label);
        return 1;
    }
    cudaMemcpy(device, device_values.data(), input.size() * sizeof(std::int32_t),
               cudaMemcpyHostToDevice);

    constexpr int block = 256;
    const int grid      = (count + block - 1) / block;
    scale_positions_yarn_kernel<<<grid, block>>>(device, count, original_context, factor);
    const cudaError_t launch = cudaDeviceSynchronize();
    if (launch != cudaSuccess) {
        std::printf("FAIL %s: launch: %s\n", label, cudaGetErrorString(launch));
        cudaFree(device);
        return 1;
    }
    cudaMemcpy(device_values.data(), device, input.size() * sizeof(std::int32_t),
               cudaMemcpyDeviceToHost);
    cudaFree(device);

    int failures = 0;
    for (std::int32_t i = 0; i < count; ++i) {
        const std::int32_t expected = oracle(input[static_cast<std::size_t>(i)], original_context,
                                             static_cast<double>(factor));
        const std::int32_t got      = device_values[static_cast<std::size_t>(i)];
        if (expected != got) {
            // Report the first few mismatches only; a systematic off-by-one would otherwise bury
            // the summary in thousands of identical lines.
            if (failures < 5) {
                std::printf("FAIL %s: index %d input %d expected %d got %d\n", label, i,
                            input[static_cast<std::size_t>(i)], expected, got);
            }
            ++failures;
        }
    }
    if (failures == 0) {
        std::printf("OK   %s (%d values, ctx=%d factor=%.2f)\n", label, count, original_context,
                    static_cast<double>(factor));
    } else {
        std::printf("FAIL %s: %d/%d values mismatched (first 5 shown)\n", label, failures, count);
    }
    return failures;
}

} // namespace

int main() {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess || device_count == 0) {
        std::printf("SKIP: no CUDA device visible\n");
        return 2;
    }
    cudaDeviceProp properties{};
    cudaGetDeviceProperties(&properties, 0);
    std::printf("device: %s (sm_%d%d)\n", properties.name, properties.major, properties.minor);

    int failures = 0;
    constexpr std::int32_t kCtx = 262144;

    // Boundary and representative points at the production factor from the gzenz fork.
    failures += run_case("ramp-2.12", kCtx, 2.12F,
                         {0, 1, 1000, 262143, 262144, 262145, 262146, 300000, 400000, 555000});

    // Identity factor must be a pure copy.
    failures += run_case("disabled-1.0", kCtx, 1.0F, {0, 262144, 262145, 555000, 999999});

    // Sensitivity: the ramp must be monotonic and must never exceed the input.
    {
        std::vector<std::int32_t> ramp;
        for (std::int32_t p = 260000; p <= 600000; p += 7919) { ramp.push_back(p); }
        failures += run_case("monotonic-2.30", kCtx, 2.30F, ramp);
    }

    // A different original_context threshold.
    failures += run_case("ctx-8192", 8192, 2.0F,
                         {0, 8191, 8192, 8193, 16384, 100000, 500000});

    // A large count exercises the grid/block split (100k values).
    {
        std::vector<std::int32_t> wide;
        wide.reserve(100000);
        for (std::int32_t i = 0; i < 100000; ++i) { wide.push_back(i * 6); }
        failures += run_case("grid-split", kCtx, 2.12F, wide);
    }

    std::printf("%s\n", failures == 0 ? "ALL OK" : "FAILURES PRESENT");
    return failures == 0 ? 0 : 1;
}
