#!/bin/bash
# NInfer pressure test service: small memory pools to easily trigger eviction/pressure paths.
# Usage: ./ninfer_service_pressure.sh {start|stop|status}
#
# Logs, pid file and reqdump/ live in $NINFER_HARNESS_DIR (default <repo>/build/harness).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=tools/smoke/harness_env.sh
source "$SCRIPT_DIR/harness_env.sh"
harness_require
BIN="$NINFER_BIN"
MODEL="$NINFER_WEIGHTS"
PORT="${PRESSURE_PORT:-30001}"
LOG="$NINFER_HARNESS_DIR/ninfer_serve_pressure.log"
PIDFILE="$NINFER_HARNESS_DIR/ninfer_serve_pressure.pid"

# Small parameters to trigger memory pressure easily:
# - Only 2 device state slots (beyond active lanes)
# - Only 8 host state slots (vs 320 in production)
# - Only 512 MiB host KV (vs 32 GiB)
# - Small KV capacity (32K tokens ≈ 128 pages)
# - 锚点间距压到 100 token, 且放开单会话锚点数(32): 前端只在消息边界放锚点, 所以必须配多轮
#   对话才能产生密集锚点; 间距小 + 上限高 才能快速把 host state 池(8 槽)顶满, 观察降级/驱逐/丢最老
# - Only 4 private continuation slots (vs 16)
# - Only 2 concurrent requests
PARAMS=(
    "$MODEL"
    --port "$PORT"
    --model-id myai
    --device-state-slots 2
    --host-state-slots 8
    --host-kv-mib 512
    --max-context 16384
    --kv-capacity 32768
    --max-private-continuations 4
    --max-concurrency 2
    --first-anchor-spacing 100
    --auto-anchor-spacing 100
    --max-long-anchors-per-continuation 32
    --fair-share-buckets 2
)

start() {
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "Already running (PID $(cat "$PIDFILE"))."
        return
    fi
    echo "Starting NInfer pressure test service on port $PORT..."
    echo "Device state slots: 2 | Host state slots: 8 | Host KV: 512 MiB | KV capacity: 32768"
    echo "Private continuations: 4 | Max concurrency: 2"
    cd "$NINFER_HARNESS_DIR"
    setsid nohup "$BIN" "${PARAMS[@]}" > "$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    sleep 2
    if kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "Started (PID $(cat "$PIDFILE")). Log: $LOG"
    else
        echo "FAILED to start. Check $LOG"
        tail -20 "$LOG"
        exit 1
    fi
}

stop() {
    if [[ -f "$PIDFILE" ]]; then
        local pid
        pid=$(cat "$PIDFILE")
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid"
            sleep 2
            kill -9 "$pid" 2>/dev/null || true
        fi
        rm -f "$PIDFILE"
        echo "Stopped."
    else
        echo "Not running."
    fi
}

status() {
    if [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
        echo "Running (PID $(cat "$PIDFILE")) on port $PORT"
    else
        echo "Not running."
    fi
}

case "${1:-}" in
    start)   start ;;
    stop)    stop ;;
    status)  status ;;
    restart) stop; sleep 2; start ;;
    *)       echo "Usage: $0 {start|stop|status|restart}"; exit 1 ;;
esac
