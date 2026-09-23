#!/bin/bash
# Agent-owned NInfer test launcher: a small-parameter instance used to reproduce context-cache
# capture/planning defects fast, without touching the production service (ninfer_service.sh).
# This is the only small-pool launcher; the retired pressure script's sizing is reproduced with
#   AGENT_PRIVATE=4 AGENT_FIRST_SPACING=100 AGENT_MAX_ANCHORS=32 ./ninfer_service_agent.sh start
# Port: always :30000 - one GPU cannot host two services, so no harness uses another port.
#
# Two-stage rule: every engine change is validated here first (a full pool is one or two filler
# requests away, so a pair costs seconds), and only a change that passes here is taken to the
# production configuration, where filling 320 host slots costs about eleven minutes per cycle.
#
# Freeing the GPU takes a moment: an instance releases ~31 GiB asynchronously, so starting another
# one right after `stop` can fail with "minimum Engine runtime reservation requires ... but only N
# bytes are available after weights" even though the port and the script's own memory pre-check
# looked fine. Run ./agent_wait_gpu.sh before starting anything after a stop.
#
# Usage: ./ninfer_service_agent.sh {start|stop|status}
#
# Logs, pid file, request log and reqdump/ are written to $NINFER_HARNESS_DIR
# (default <repo>/build/harness); NINFER_BIN / NINFER_WEIGHTS select what to run.
# Every parameter is env-overridable so one experiment can vary a single dimension, e.g.
#   AGENT_HOST_SLOTS=0 ./ninfer_service_agent.sh start     # no host state pool at all
#   REUSE_DIAG=1 ./ninfer_service_agent.sh start           # prefix-candidate diagnostics
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=tools/smoke/harness_env.sh
source "$SCRIPT_DIR/harness_env.sh"
harness_require
BIN="$NINFER_BIN"
MODEL="$NINFER_WEIGHTS"
PORT="${AGENT_PORT:-30000}"                 # one GPU, one service: every harness uses :30000
LOG="$NINFER_HARNESS_DIR/ninfer_serve_agent.log"
PIDFILE="$NINFER_HARNESS_DIR/ninfer_serve_agent.pid"

# Sizing rule: the rig holds the WORKING SET (4 concurrent sessions - the production
# concurrency) comfortably, and only the cache beyond it is small, so boundaries are
# reachable in seconds without measuring a starved working set:
#   device state objects = concurrency 4 + device-state-slots 4 = 8 (production count)
#   host state slots 48  = 4 sessions x ~8 retained checkpoints + headroom (production 320);
#                           session #5+ hits the eviction/retire ladder, session <=4 must not
#   kv-capacity 65536     = 2 x max-context: one full-context request is always feasible,
#                           4 x full-context crosses the shared-capacity boundary
HOST_SLOTS="${AGENT_HOST_SLOTS:-48}"       # working set fits; the 5th session fills the cache
DEV_SLOTS="${AGENT_DEV_SLOTS:-4}"
HOST_KV_MIB="${AGENT_HOST_KV_MIB:-512}"
MAX_CTX="${AGENT_MAX_CTX:-32768}"           # battery prompts ~8.5K; 40K oversized still overflows
KV_CAPACITY="${AGENT_KV_CAPACITY:-65536}"
PRIVATE="${AGENT_PRIVATE:-16}"
CONCURRENCY="${AGENT_CONCURRENCY:-4}"       # production concurrency: 4-lane admission boundary
FIRST_SPACING="${AGENT_FIRST_SPACING:-4096}"   # production value: first spread anchor at tools end
ANCHOR_SPACING="${AGENT_ANCHOR_SPACING:-256}"  # dense auto anchors: mid-history reuse in short runs
MAX_ANCHORS="${AGENT_MAX_ANCHORS:-16}"      # production cap: anchor replacement under dense fill
SHARED_PREFIXES="${AGENT_SHARED_PREFIXES:-4}"   # small shared catalog fills fast (replacement path)
AUTO_ANCHORS="${AGENT_AUTO_ANCHORS:-5}"     # production value: last-N user transitions
FAIR_BUCKETS="${AGENT_FAIR_BUCKETS:-4}"     # protected-vs-evictable boundary visible in small runs
MAX_PENDING="${AGENT_MAX_PENDING:-16}"      # production value: bounded FIFO ingress
PENDING_TIMEOUT_MS="${AGENT_PENDING_TIMEOUT_MS:-600000}"  # production value; lower per-test for
                                                          # the queue-timeout boundary
REUSE_DIAG="${REUSE_DIAG:-0}"
REQUEST_LOG="${AGENT_REQUEST_LOG:-$NINFER_HARNESS_DIR/request_log_agent.jsonl}"

PARAMS=(
    "$MODEL"
    --port "$PORT"
    --model-id myai
    --vision
    --device-state-slots "$DEV_SLOTS"
    --host-state-slots "$HOST_SLOTS"
    --host-kv-mib "$HOST_KV_MIB"
    --max-context "$MAX_CTX"
    --kv-capacity "$KV_CAPACITY"
    --max-private-continuations "$PRIVATE"
    --max-concurrency "$CONCURRENCY"
    --first-anchor-spacing "$FIRST_SPACING"
    --auto-anchor-spacing "$ANCHOR_SPACING"
    --max-long-anchors-per-continuation "$MAX_ANCHORS"
    --max-shared-prefixes "$SHARED_PREFIXES"
    --auto-long-anchors "$AUTO_ANCHORS"
    --fair-share-buckets "$FAIR_BUCKETS"
    --max-pending-requests "$MAX_PENDING"
    --pending-timeout-ms "$PENDING_TIMEOUT_MS"
    --request-log-jsonl "$REQUEST_LOG"
)

start() {
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "Already running (PID $(cat "$PIDFILE"))."
        return
    fi
    echo "Starting agent test instance on port $PORT"
    echo "  device/host state slots: $DEV_SLOTS/$HOST_SLOTS | host KV: ${HOST_KV_MIB} MiB"
    echo "  kv-capacity: $KV_CAPACITY | max-context: $MAX_CTX | private: $PRIVATE | conc: $CONCURRENCY"
    echo "  first/auto anchor spacing: $FIRST_SPACING/$ANCHOR_SPACING | max anchors: $MAX_ANCHORS auto: $AUTO_ANCHORS"
    echo "  reuse diag: $REUSE_DIAG | log: $LOG"
    if [[ "$REUSE_DIAG" != "0" && -n "$REUSE_DIAG" ]]; then
        export NINFER_REUSE_DIAG=1
    else
        unset NINFER_REUSE_DIAG
    fi
    # Request dumps: the binary reads NINFER_DUMP_REQUESTS from the environment (not the command
    # line), and tools/smoke/harness_paths.py looks for them in the same directory, so a harness run
    # can replay the exact client shape it just served without copying files around.
    export NINFER_DUMP_REQUESTS="${NINFER_DUMP_REQUESTS:-$NINFER_HARNESS_DIR/reqdump}"
    export NINFER_DUMP_REQUESTS_LIMIT="${NINFER_DUMP_REQUESTS_LIMIT:-200}"
    mkdir -p "$NINFER_DUMP_REQUESTS"
    # One GPU, one service: refuse when the port is already served (e.g. the production
    # instance on :30000). Without this the health probe below would hit THAT service and
    # report Ready while our own engine died on bind. SO_REUSEADDR lets a just-stopped
    # instance's TIME_WAIT through; a live LISTENER still fails the bind.
    python3 -c "import socket; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('127.0.0.1', $PORT)); s.close()" 2>/dev/null \
        || { echo "错误: 端口 $PORT 已被服务监听 - 单卡单服务, 请先 stop 正在运行的服务"; exit 1; }
    cd "$NINFER_HARNESS_DIR"
    setsid nohup "$BIN" "${PARAMS[@]}" > "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    for _ in $(seq 1 120); do
        # Ready = healthy AND OUR pid still alive: a healthy foreign answer alone must not pass.
        if curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q ok \
            && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            echo "Ready after $((_ * 2))s (PID $(cat "$PIDFILE"))."
            return
        fi
        sleep 2
    done
    echo "FAILED to become ready. Check $LOG"
    tail -20 "$LOG"
    exit 1
}

stop() {
    if [[ -f "$PIDFILE" ]]; then
        PID="$(cat "$PIDFILE")"
        if kill -0 "$PID" 2>/dev/null; then
            kill "$PID" 2>/dev/null || true
            for _ in $(seq 1 30); do
                kill -0 "$PID" 2>/dev/null || break
                sleep 1
            done
            kill -9 "$PID" 2>/dev/null || true
        fi
        rm -f "$PIDFILE"
        echo "Stopped."
    else
        echo "Not running."
    fi
}

status() {
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "running pid $(cat "$PIDFILE") port $PORT"
        curl -s -m 2 "http://127.0.0.1:$PORT/health" || true
        echo
    else
        echo "not running"
    fi
}

case "${1:-}" in
    start) start ;;
    stop) stop ;;
    status) status ;;
    *) echo "usage: $0 {start|stop|status}"; exit 2 ;;
esac
