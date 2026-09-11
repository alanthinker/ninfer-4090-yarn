#!/usr/bin/env bash
# 停止 NInfer 服务, 释放显存
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
PIDF="$HERE/ninfer_serve.pid"
if [ -f "$PIDF" ]; then
  PID=$(cat "$PIDF")
  kill "$PID" 2>/dev/null || true
  for _ in $(seq 1 20); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
  kill -9 "$PID" 2>/dev/null || true
  rm -f "$PIDF"
fi
pkill -9 -f "[n]infer-serve" 2>/dev/null || true
sleep 3
echo "显存状态:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
echo
echo "NInfer 已停止。恢复 AI 后端 (vLLM myai, :30000) 执行:"
echo "  /root/ai/large_models/qwen_3.8_27b/vllm_service.sh start"
