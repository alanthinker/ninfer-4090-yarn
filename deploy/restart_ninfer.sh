#!/usr/bin/env bash
# 重启 NInfer (停 + 起, 配置沿用 start_ninfer.sh; 日志追加不丢历史)
# 注意: 重启会清空引擎保留的会话/前缀缓存, 之后第一个请求要重新 prefill 整段上下文。
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
echo "[$(date '+%F %T')] 重启 NInfer ..."
bash "$HERE/stop_ninfer.sh" >/dev/null 2>&1
sleep 3
if bash "$HERE/start_ninfer.sh" >>"$HERE/restart.log" 2>&1; then
  echo "[$(date '+%F %T')] RESTART_OK" | tee -a "$HERE/restart.log"
else
  echo "[$(date '+%F %T')] RESTART_FAILED — 看 $HERE/restart.log" | tee -a "$HERE/restart.log"
  exit 1
fi
