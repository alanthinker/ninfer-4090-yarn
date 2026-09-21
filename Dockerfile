# syntax=docker/dockerfile:1

# Merged sm_89 build: alanthinker-yarn base + re-implemented optimizations
# (SM-count adaptive grids, INT8 tensor-core prefill routes, T=1 Ada MMA).
# CUDA 13.2 matches the host stack proven on this RTX 4090 D machine.

FROM nvidia/cuda:13.2.0-cudnn-devel-ubuntu24.04 AS build

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        cmake \
        curl \
        libavcodec-dev \
        libavformat-dev \
        libavutil-dev \
        libcurl4-openssl-dev \
        libswscale-dev \
        ninja-build \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /src
COPY . .

# Cache mount keeps /build across image rebuilds: ninja only recompiles the
# translation units that changed between builds.
RUN --mount=type=cache,id=ninfer-merged-sm89-v2,target=/build,sharing=locked \
    cmake -S . -B /build -G Ninja \
        -DCMAKE_BUILD_TYPE=Release \
        -DNINFER_BUILD_APPS=ON \
        -DBUILD_TESTING=OFF \
        -DNINFER_BUILD_BENCHMARKS=OFF \
    && cmake --build /build --parallel --target ninfer ninfer-serve \
    && install -D /build/apps/ninfer /opt/ninfer/bin/ninfer \
    && install -D /build/apps/ninfer-serve /opt/ninfer/bin/ninfer-serve

FROM nvidia/cuda:13.2.0-cudnn-runtime-ubuntu24.04

ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install --yes --no-install-recommends \
        ca-certificates \
        curl \
        libavcodec60 \
        libavformat60 \
        libavutil58 \
        libcurl4t64 \
        libswscale7 \
    && rm -rf /var/lib/apt/lists/*

# The CUDA runtime image ships forward-compatibility libraries in
# /usr/local/cuda*/compat. On GeForce cards the loader picks them up and every
# CUDA call fails with cudaErrorCompatNotSupportedOnDevice. Remove them so the
# container uses the host driver through ordinary minor-version compatibility.
RUN rm -rf /usr/local/cuda-13.2/compat /usr/local/cuda-13/compat /usr/local/cuda/compat

# Model is NOT baked into the image. Mount the host-side .ninfer file at runtime:
#   -v /path/to/model-dir:/models:ro
RUN mkdir -p /models

COPY --from=build /opt/ninfer/bin/ninfer /usr/local/bin/ninfer
COPY --from=build /opt/ninfer/bin/ninfer-serve /usr/local/bin/ninfer-serve

WORKDIR /workspace
EXPOSE 18080
STOPSIGNAL SIGTERM
VOLUME ["/var/cache/ninfer"]

# Placeholder CMD: always start with an explicit `ninfer-serve <path> ...` command.
CMD ["/bin/bash", "-lc", "echo 'No default model. Mount a .ninfer file at /models and run ninfer-serve manually.'; sleep infinity"]