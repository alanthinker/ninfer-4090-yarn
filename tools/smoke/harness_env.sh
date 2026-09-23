#!/usr/bin/env bash
# Shared environment for the small-parameter Engine launcher in this directory
# (ninfer_service_agent.sh) and for sweep_concurrency.sh.
#
# Sourced, never executed. It resolves the three things every launcher needs and keeps the source
# tree clean while doing it:
#
#   NINFER_BIN           Engine binary          (default <repo>/build/apps/ninfer-serve)
#   NINFER_WEIGHTS       .ninfer artifact       (default: first existing candidate, see below)
#   NINFER_HARNESS_DIR   logs, pid files, request log, and reqdump/ written by the Engine
#                        (default <repo>/build/harness — inside the untracked build directory)
#
# The launchers `cd` into the harness directory before starting the Engine so its relative
# `reqdump/` output lands there instead of in the source tree.
set -euo pipefail

HARNESS_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
NINFER_BIN="${NINFER_BIN:-$HARNESS_REPO_ROOT/build/apps/ninfer-serve}"
NINFER_HARNESS_DIR="${NINFER_HARNESS_DIR:-$HARNESS_REPO_ROOT/build/harness}"

if [[ -z "${NINFER_WEIGHTS:-}" ]]; then
    for candidate in \
        "$HARNESS_REPO_ROOT/out/qwen3_8_27b.ninfer" \
        "$HARNESS_REPO_ROOT/../ninfer-4090/models/qwen3_8_27b.ninfer" \
        "$HARNESS_REPO_ROOT/models/qwen3_8_27b.ninfer"; do
        if [[ -f "$candidate" ]]; then
            NINFER_WEIGHTS="$candidate"
            break
        fi
    done
fi

harness_require() {
    if [[ ! -x "$NINFER_BIN" ]]; then
        echo "engine binary not found: $NINFER_BIN (build it: cmake --build build -j)" >&2
        exit 1
    fi
    if [[ -z "${NINFER_WEIGHTS:-}" || ! -f "$NINFER_WEIGHTS" ]]; then
        echo "artifact not found; set NINFER_WEIGHTS to a .ninfer file" >&2
        echo "  tried: out/qwen3_8_27b.ninfer, ../ninfer-4090/models/qwen3_8_27b.ninfer," >&2
        echo "         models/qwen3_8_27b.ninfer (relative to $HARNESS_REPO_ROOT)" >&2
        exit 1
    fi
    mkdir -p "$NINFER_HARNESS_DIR"
}
