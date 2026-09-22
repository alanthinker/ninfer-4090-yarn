#!/usr/bin/env bash
# Full-pool soak: drive the cache-pressure sequences that failed in production and report each
# window from the Engine's own logs.
#
# Usage: tools/smoke/pool_soak.sh [PORT] [DUMP_DIR]
#
#   PORT       service port (default $NINFER_PORT or 30000)
#   DUMP_DIR   directory of recorded request dumps to replay (default /tmp/crash243, the dumps of
#              the 2026-09-22 incident; the newest dumps the Engine wrote itself live in
#              `$NINFER_REQDUMP_DIR`)
#
# Run it against a *full* host-state pool: the whole point is that a saturated pool must still
# publish the newest request's own capture, and that concurrency must not turn a capture decision
# into a 500 or a fatal. Confirm the pool first with `fill_anchors.py` (it prints
# `host_state_slots`), then run this and read the windows: every window must show 500/503/fatal 0.
#
# Environment: NINFER_HARNESS_DIR / NINFER_REQDUMP_DIR / NINFER_REQUEST_LOG / NINFER_SERVICE_LOG
# select the service's runtime files, exactly as the other harnesses in this directory.
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${1:-${NINFER_PORT:-30000}}"
DUMP_DIR="${2:-/tmp/crash243}"
cd "$HERE/../.." || exit 1

# The soak drives a *running* service, so it resolves only where that service writes; it does not
# need the engine binary or the artifact (that is what harness_env.sh is for).
HARNESS_DIR="${NINFER_HARNESS_DIR:-$PWD/build/harness}"
SERVICE_LOG="${NINFER_SERVICE_LOG:-$HARNESS_DIR/ninfer_serve.log}"
REQUEST_LOG="${NINFER_REQUEST_LOG:-$HARNESS_DIR/request_log.jsonl}"
export NINFER_HARNESS_DIR="$HARNESS_DIR" NINFER_SERVICE_LOG="$SERVICE_LOG" NINFER_REQUEST_LOG="$REQUEST_LOG"

DUMPS=("$DUMP_DIR"/*.json)
if [ ! -e "${DUMPS[0]}" ]; then
    echo "pool_soak: no request dumps in $DUMP_DIR" >&2
    exit 1
fi

window() {  # window <label> <command...> -- one labelled sequence, reported from both logs
    local label="$1" serve_offset request_offset
    serve_offset=$(stat -c%s "$SERVICE_LOG")
    request_offset=$(stat -c%s "$REQUEST_LOG")
    shift
    "$@" >/dev/null 2>&1
    python3 "$HERE/report_window.py" "$label" "$serve_offset" "$request_offset"
}

echo "== A. recorded crash-fork sequence, sequential, 4 rounds =="
for round in 1 2 3 4; do
    window "fork sequence round $round" timeout 1800 python3 "$HERE/replay_dump.py" "$PORT" "${DUMPS[@]}"
done

echo "== B. anchor semantics probe (edit / fork / continue, 15 requests) =="
window "anchor probe" timeout 1800 python3 "$HERE/anchor_semantics_probe.py" "$PORT" "$SERVICE_LOG"

echo "== C. cold request + sibling messages (the live client shape) =="
window "cold+siblings" env AGENT_PORT="$PORT" SIB_GAP=0.3 SIB_TAG=soak \
    timeout 900 python3 "$HERE/agent_capture_repro.py" siblings 4

echo "== D. concurrent bursts (capture and reclaim under contention) =="
for round in 1 2; do
    window "parallel burst $round" timeout 1800 python3 "$HERE/replay_dump.py" "$PORT" "${DUMPS[@]}" --parallel 4
done
