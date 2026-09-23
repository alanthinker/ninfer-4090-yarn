#!/usr/bin/env bash
# 并发度 vs 池子容量 扫描。
#
# 目的: 回答"允许 N 路并发时, 总池子和单会话 context 会怎样变"。
#
# 引擎的约束(见 src/targets/qwen3_6/impl/runtime/layouts_impl.h):
#   logical_pages  = page_count(max_context)
#   minimum_pages  = max(logical_pages, max_concurrency)
#   maximum_pages  = max_concurrency * logical_pages
#   auto 在 [minimum_pages, maximum_pages] 内选显存允许的最大值
# 关键: minimum_pages 只跟 max_context 走, 与并发无关; maximum_pages 随并发线性放大。
# 所以并发越高, auto 的容量上限越高(但仍受显存封顶); 而 max_context 是每会话逻辑天花板。
#
# 用法: bash sweep_concurrency.sh
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
# shellcheck source=tools/smoke/harness_env.sh
source "$HERE/harness_env.sh"
harness_require
BIN="$NINFER_BIN"
MODEL="$NINFER_WEIGHTS"
PORT=30000          # 与其他 harness 同端口; 本脚本自起实例, 必须先停正在跑的服务
# 单 GPU 单服务: 真有服务 LISTEN 才拒绝; TIME_WAIT 不算占用 (否则刚 stop 完会误报)。
python3 -c "import socket; s=socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); s.bind(('127.0.0.1',$PORT)); s.close()" 2>/dev/null \
    || { echo "错误: 端口 $PORT 正被服务监听 - 请先 stop 正在运行的服务再跑本脚本"; exit 1; }
LOG="$NINFER_HARNESS_DIR/sweep.log"

probe() {   # probe <concurrency> <max_context>
  local n="$1" mc="$2"
  : > "$LOG"
  # setsid: the probe engine gets its own process group, so our kill sequence can never
  # reach this script (observed: the script died silently right after kill -9 without it).
  setsid "$BIN" "$MODEL" --host 127.0.0.1 --port "$PORT" --model-id probe \
    --max-context "$mc" --kv-capacity auto \
    --max-concurrency "$n" --max-pending-requests 4 \
    --prefill-chunk 1024 --kv-dtype rk4v4-e8 \
    --spec mtp --draft-tokens 3 --lm-head-draft \
    --vision --host-kv-mib 4096 --host-state-slots 4 \
    >"$LOG" 2>&1 < /dev/null &
  local pid=$!
  local line="" ok=0
  for _ in $(seq 1 40); do
    sleep 1
    if grep -q 'engine capacity' "$LOG" 2>/dev/null; then
      line=$(grep 'engine capacity' "$LOG" | tail -1); ok=1; break
    fi
    if ! kill -0 "$pid" 2>/dev/null; then break; fi
  done
  if [ "$ok" = 1 ]; then
    local cap slack dslots
    cap=$(echo "$line"   | grep -oE 'kv_capacity_tokens=[0-9]+'        | cut -d= -f2)
    slack=$(echo "$line" | grep -oE 'planned_slack_bytes=[0-9]+'       | cut -d= -f2)
    dslots=$(grep -oE 'device_state_slots=[0-9]+' "$LOG" | tail -1 | cut -d= -f2)
    printf 'OK   N=%-2s max_ctx=%-8s pool=%-9s slack=%-11s dev_state_slots=%s\n' \
      "$n" "$mc" "$cap" "${slack:-?}" "${dslots:-?}"
  else
    local err
    err=$(grep -iE 'error|fatal|exceed|insufficient|at least' "$LOG" | tail -1 | cut -c1-160 || true)
    printf 'FAIL N=%-2s max_ctx=%-8s %s\n' "$n" "$mc" "${err:-<no capacity line>}"
  fi
  kill -TERM "$pid" 2>/dev/null || true
  for _ in $(seq 1 15); do kill -0 "$pid" 2>/dev/null || break; sleep 1; done
  kill -KILL "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  # VRAM is released asynchronously (~tens of GiB): the next probe must not start until the
  # previous instance has actually given its memory back, or its reservation fails.
  "$HERE/agent_wait_gpu.sh" 31000 180 >/dev/null 2>&1 || true
  sleep 2
}

echo "=== 阶段 A: 固定 max_context=300000, 看总池子随并发如何变 (上限 = 显存) ==="
for n in 1 2 3 4 8; do probe "$n" 300000; done

echo
echo "=== 阶段 B: 每个并发度下能支持的最大单会话 context ==="
for n in 1 3 4 8; do
  for mc in 724000 700000 660000 620000 560000; do
    probe "$n" "$mc"
  done
done
