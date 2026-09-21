// State-image D2H/H2D copy microbenchmark.
//
// Reproduces the StateImageDevicePool::copy_to_host / copy_from_host issue
// pattern (src/targets/qwen3_6/impl/state/state_image.cpp): one state image
// (147 MiB) is moved as many sequential cudaMemcpyAsync calls on a single
// stream (per-layer conv + recurrent slices, then the continuation-hidden
// tensor), and the capacity-release ladder then does a full
// cudaStreamSynchronize before the source slot is reused.
//
// Build: g++ -O2 state_copy_bench.c -I$CUDA/include -L$CUDA/lib64 -lcudart -o state_copy_bench

#include <cuda_runtime.h>

#include <chrono>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define CHECK(x)                                                                                               \
    do {                                                                                                       \
        cudaError_t err__ = (x);                                                                               \
        if (err__ != cudaSuccess) {                                                                            \
            fprintf(stderr, "CUDA error %s at line %d: %s\n", #x, __LINE__, cudaGetErrorString(err__));        \
            exit(1);                                                                                           \
        }                                                                                                      \
    } while (0)

typedef void (*CopyKind)(void*, const void*, size_t, enum cudaMemcpyKind, cudaStream_t);

static void run_case(const char* name, CopyKind copy, cudaStream_t stream,
                     unsigned char* device, unsigned char* host, size_t chunk, int nchunks,
                     int with_sync) {
    const size_t total = chunk * (size_t)nchunks;
    cudaEvent_t e0, e1;
    CHECK(cudaEventCreate(&e0));
    CHECK(cudaEventCreate(&e1));

    // warmup (two passes: first also triggers any lazy engine work)
    for (int w = 0; w < 2; ++w) {
        for (int i = 0; i < nchunks; ++i) {
            if (name[0] == 'D') {
                CHECK(cudaMemcpyAsync(host + i * chunk, device + i * chunk, chunk,
                                      cudaMemcpyDeviceToHost, stream));
            } else {
                CHECK(cudaMemcpyAsync(device + i * chunk, host + i * chunk, chunk,
                                      cudaMemcpyHostToDevice, stream));
            }
        }
        CHECK(cudaStreamSynchronize(stream));
    }

    double best = 1e30;
    for (int rep = 0; rep < 7; ++rep) {
        double ms;
        if (with_sync) {
            // CPU-timed: enqueue the whole image, then block on the stream —
            // exactly what the capacity-release ladder does per demotion.
            auto t0 = std::chrono::steady_clock::now();
            for (int i = 0; i < nchunks; ++i) {
                if (name[0] == 'D') {
                    CHECK(cudaMemcpyAsync(host + i * chunk, device + i * chunk, chunk,
                                          cudaMemcpyDeviceToHost, stream));
                } else {
                    CHECK(cudaMemcpyAsync(device + i * chunk, host + i * chunk, chunk,
                                          cudaMemcpyHostToDevice, stream));
                }
            }
            CHECK(cudaStreamSynchronize(stream));
            auto t1 = std::chrono::steady_clock::now();
            ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        } else {
            CHECK(cudaEventRecord(e0, stream));
            for (int i = 0; i < nchunks; ++i) {
                if (name[0] == 'D') {
                    CHECK(cudaMemcpyAsync(host + i * chunk, device + i * chunk, chunk,
                                          cudaMemcpyDeviceToHost, stream));
                } else {
                    CHECK(cudaMemcpyAsync(device + i * chunk, host + i * chunk, chunk,
                                          cudaMemcpyHostToDevice, stream));
                }
            }
            CHECK(cudaEventRecord(e1, stream));
            CHECK(cudaEventSynchronize(e1));
            float elapsed = 0;
            CHECK(cudaEventElapsedTime(&elapsed, e0, e1));
            ms = elapsed;
        }
        if (ms < best) { best = ms; }
    }
    const double mibs = total / 1048576.0;
    printf("  %-34s %3d x %7.2f MiB%s : %8.2f ms   %8.1f MiB/s\n", name, nchunks,
           chunk / 1048576.0, with_sync ? " +sync" : "", best, mibs / (best / 1000.0));
    CHECK(cudaEventDestroy(e0));
    CHECK(cudaEventDestroy(e1));
}

int main(void) {
    CHECK(cudaSetDevice(0));
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s | %zu MiB HBM\n", prop.name, prop.totalGlobalMem / 1048576);

    const size_t image = 147u * 1024 * 1024; // StateImage slot size (320 slots x 147 MiB = 45.9 GiB)
    unsigned char *device = nullptr, *host = nullptr;
    CHECK(cudaMalloc(&device, image));
    CHECK(cudaHostAlloc(&host, image, cudaHostAllocDefault));
    cudaStream_t stream;
    CHECK(cudaStreamCreate(&stream));

    printf("Device -> Host (pinned, single dedicated stream, per-copy issue pattern of the engine):\n");
    run_case("D2H single 147 MiB memcpy", nullptr, stream, device, host, image, 1, 0);
    run_case("D2H 73 x 2 MiB (36 layers x2 + hidden)", nullptr, stream, device, host, 2u * 1024 * 1024, 73, 0);
    run_case("D2H 73 x 1 MiB", nullptr, stream, device, host, 1u * 1024 * 1024, 73, 0);
    run_case("D2H 294 x 512 KiB", nullptr, stream, device, host, 512u * 1024, 294, 0);
    run_case("D2H 73 x 2 MiB + stream sync (ladder)", nullptr, stream, device, host, 2u * 1024 * 1024, 73, 1);
    run_case("D2H single 147 MiB + stream sync (ladder)", nullptr, stream, device, host, image, 1, 1);

    printf("Host -> Device (same patterns):\n");
    run_case("H2D single 147 MiB memcpy", nullptr, stream, device, host, image, 1, 0);
    run_case("H2D 73 x 2 MiB (36 layers x2 + hidden)", nullptr, stream, device, host, 2u * 1024 * 1024, 73, 0);
    run_case("H2D 73 x 1 MiB", nullptr, stream, device, host, 1u * 1024 * 1024, 73, 0);
    run_case("H2D 294 x 512 KiB", nullptr, stream, device, host, 512u * 1024, 294, 0);
    run_case("H2D 73 x 2 MiB + stream sync", nullptr, stream, device, host, 2u * 1024 * 1024, 73, 1);

    // Five sequential ladder steps, as release_state_capacity_step would issue them.
    printf("Ladder simulation: 5 sequential state images (73 x 2 MiB each, per-image sync):\n");
    for (int rep = 0; rep < 3; ++rep) {
        auto t0 = std::chrono::steady_clock::now();
        for (int img = 0; img < 5; ++img) {
            for (int i = 0; i < 73; ++i) {
                CHECK(cudaMemcpyAsync(host + i * (2u * 1024 * 1024), device + i * (2u * 1024 * 1024),
                                      2u * 1024 * 1024, cudaMemcpyDeviceToHost, stream));
            }
            CHECK(cudaStreamSynchronize(stream));
        }
        auto t1 = std::chrono::steady_clock::now();
        const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
        printf("  [pass %d] 5 x 147 MiB ladder steps : %8.2f ms   (%.2f ms per image)\n", rep + 1, ms,
               ms / 5.0);
    }

    CHECK(cudaFreeHost(host));
    CHECK(cudaFree(device));
    CHECK(cudaStreamDestroy(stream));
    return 0;
}
