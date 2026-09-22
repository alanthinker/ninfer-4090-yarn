#!/bin/bash
# Wait until the GPU is genuinely free before starting an NInfer instance.
#
# Stopping an instance (or the agent harness) releases its ~31 GiB of VRAM
# asynchronously, so an immediate `ninfer_service.sh start` can fail with
# "minimum Engine runtime reservation requires ... but only N bytes are
# available after weights" even though the port checks passed (2026-09-22:
# production start failed, the fill then hit connection-refused for three
# rounds). Poll the free-memory counter instead of the port.
#
# Usage: ./agent_wait_gpu.sh [minimum_free_mib] [timeout_seconds]
set -euo pipefail

MIN_FREE_MIB="${1:-31000}"
TIMEOUT_S="${2:-180}"

for _ in $(seq 1 "$TIMEOUT_S"); do
    used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -1 | tr -d ' ')
    free=$((32760 - used))
    if [ "$free" -ge "$MIN_FREE_MIB" ]; then
        echo "gpu free: ${free} MiB (used ${used} MiB)"
        exit 0
    fi
    sleep 1
done

echo "gpu still busy after ${TIMEOUT_S}s: $(nvidia-smi --query-gpu=memory.used --format=csv,noheader)"
nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv | head -5
exit 1
