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
# Resolve to an ABSOLUTE path NOW: start() cds into the harness directory before spawning, so
# a relative NINFER_BIN used to exec-fail silently there (empty log, "failed to become ready"
# with no reason - 2026-09-26). A missing or unresolvable binary must stop the launcher with
# the reason, not stall the health probe.
BIN="$(realpath -e "$NINFER_BIN" 2>/dev/null)" || {
    echo "engine binary not found (cwd=$(pwd), NINFER_BIN=$NINFER_BIN): build it with: cmake --build build -j" >&2
    exit 1
}
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
#   kv-capacity auto      = production's mode. Measured production banner (2026-09-23):
#                           kv_capacity_tokens=658176 kv_page_groups=10284. The old default
#                           65536 tokens (1024 pages) was 10x tighter than production and
#                           saturation on that axis - not host state - drove whole-session
#                           evictions the battery could not attribute: a no-pressure plan was
#                           judged infeasible at device.main_kv used=1024 peak=1 cap=1024, which
#                           production never reaches (peak observed 6423 of 10284).
#   host-kv-mib 4096      = production is 32768 MiB over 320 host slots (~102 MiB/slot); 48
#                           slots x that ratio = ~4.9 GB. The old 512 MiB made every ~1.1 GB
#                           host-KV allocation report blocked_host and never fit (166 infeasible
#                           targets in one battery run), which production cannot reproduce.
HOST_SLOTS="${AGENT_HOST_SLOTS:-48}"       # working set fits; the 5th session fills the cache
DEV_SLOTS="${AGENT_DEV_SLOTS:-4}"
HOST_KV_MIB="${AGENT_HOST_KV_MIB:-4096}"   # ~102 MiB/host slot, production's ratio
MAX_CTX="${AGENT_MAX_CTX:-32768}"           # battery prompts ~8.5K; 40K oversized still overflows
KV_CAPACITY="${AGENT_KV_CAPACITY:-auto}"   # same mode as production (see banner above)
PRIVATE="${AGENT_PRIVATE:-128}"            # above the 48 host slots; production is 512 over 320
CONCURRENCY="${AGENT_CONCURRENCY:-4}"       # production concurrency: 4-lane admission boundary
FIRST_SPACING="${AGENT_FIRST_SPACING:-4096}"   # production value: first spread anchor at tools end
ANCHOR_SPACING="${AGENT_ANCHOR_SPACING:-256}"  # dense auto anchors: mid-history reuse in short runs
MAX_ANCHORS="${AGENT_MAX_ANCHORS:-16}"      # production cap: anchor replacement under dense fill
SHARED_PREFIXES="${AGENT_SHARED_PREFIXES:-4}"   # small shared catalog fills fast (replacement path)
AUTO_ANCHORS="${AGENT_AUTO_ANCHORS:-5}"     # production value: last-N user transitions
FAIR_BUCKETS="${AGENT_FAIR_BUCKETS:-4}"     # protected-vs-evictable boundary visible in small runs
MAX_PENDING="${AGENT_MAX_PENDING:-16}"      # production value: bounded FIFO ingress
PENDING_TIMEOUT_MS="${AGENT_PENDING_TIMEOUT_MS:-60000}"  # tests fail fast: an R0-parked request waits60s, not the production
                                                          # the queue-timeout boundary
# Diagnostics ON by default: a TEST rig must emit the evidence its suites assert on
# (fair_share_sim greps `reuse-diag: ... fair=N ... CANDIDATE`, gated by NINFER_REUSE_DIAG).
REUSE_DIAG="${REUSE_DIAG:-1}"
# NINFER_REQUEST_LOG is the path every reader uses (harness_paths.py, pool_soak.sh, the battery's
# occupancy gate), so the instance must write there or the readers inspect a file this run never
# touched. It used to answer only to AGENT_REQUEST_LOG and default to request_log_agent.jsonl,
# which no caller passes: full_battery.sh --rig then read a stale build/harness file and its
# saturation gate failed on peak=-1 - no records inside the battery window (2026-09-30).
REQUEST_LOG="${NINFER_REQUEST_LOG:-${AGENT_REQUEST_LOG:-$NINFER_HARNESS_DIR/request_log_agent.jsonl}}"
export NINFER_REQUEST_LOG="$REQUEST_LOG"

# Parameter gate: refuse a rig that cannot bind the axes this harness exists to test.
#
# The rig's job is to reproduce PRODUCTION's binding axes (host state 48 ~ 320, device state 8) at
# a size where a fill costs seconds. The device-KV axis is deliberately harness-supplied, and an
# explicit `--kv-capacity` below max-concurrency x max-context puts it far under production: the
# 2026-09-23 calibration measured the old 65536 default (1024 pages) as 10x tighter than production
# there, producing whole-owner evictions and R0 queue-timeout 503s that the battery cannot attribute
# to the policy it is testing (2026-09-27: the first cold 2 566-token request after the fill parked
# its whole 60 s deadline at `device.main_kv used=1016 cap=1024`, while the same build passes every
# step at the capacity the device can actually hold - see
# docs/maintainer/上下文缓存物理存储与恢复.md, "rig 容量校准"). The engine considers such a pool
# legal (--kv-capacity is the SHARED pool, see docs/serving.md), so the mistake is only visible
# here, at the harness: fail before the model loads instead of 40 minutes later.
if [ "$KV_CAPACITY" != "auto" ] && [ "${AGENT_ALLOW_UNDERSIZED_KV:-0}" != "1" ]; then
    if [ "$KV_CAPACITY" -lt $((CONCURRENCY * MAX_CTX)) ] 2>/dev/null; then
        echo "refusing this rig: AGENT_KV_CAPACITY=$KV_CAPACITY tokens < AGENT_CONCURRENCY($CONCURRENCY)" \
             "x AGENT_MAX_CTX($MAX_CTX) = $((CONCURRENCY * MAX_CTX)) tokens" >&2
        echo "  the device-KV axis would be far under production and dominate every pressure test;" >&2
        echo "  use AGENT_KV_CAPACITY=auto (the default), or set AGENT_ALLOW_UNDERSIZED_KV=1 for a" >&2
        echo "  deliberate tight-pool experiment." >&2
        exit 2
    fi
fi

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
    # One GPU, one service: a live service on the port is caught by the health probe above (it
    # answers /health), and the readiness check below additionally requires OUR pid to be alive.
    #
    # The bind probe that follows is about TIME_WAIT, not about a live listener: a retired engine
    # leaves server-side TIME_WAIT sockets on the port, and this kernel refuses an exact-address
    # bind over them even with SO_REUSEADDR, so a restart within ~60 s of a stop used to die with a
    # misleading "端口已被服务监听". Wait the socket state out instead of refusing - and wait BEFORE
    # spawning, because the engine's own bind would hit the same window.
    local bind_ready=0
    for attempt in $(seq 1 45); do
        if python3 -c "import socket; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('0.0.0.0', $PORT)); s.close()" 2>/dev/null; then
            bind_ready=1
            break
        fi
        sleep 2
    done
    if [ "$bind_ready" != "1" ]; then
        echo "错误: 端口 $PORT 在 90s 内仍不可绑定 - 请确认没有其他进程占用该端口"; exit 1
    fi
    cd "$NINFER_HARNESS_DIR"
    setsid nohup "$BIN" "${PARAMS[@]}" > "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    for attempt in $(seq 1 120); do
        # Ready = healthy AND OUR pid still alive: a healthy foreign answer alone must not pass.
        if curl -s -m 2 "http://127.0.0.1:$PORT/health" | grep -q ok \
            && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
            echo "Ready after $((attempt * 2))s (PID $(cat "$PIDFILE"))."
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
