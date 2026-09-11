#!/usr/bin/env bash
# 停止 NInfer-YaRN 服务, 释放显存。
# 只处理本目录 (deploy-yarn) 的 PID, 不会误杀 deploy/ 下原版实例。
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
echo "显存状态:"
nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
echo
echo "NInfer-YaRN 已停止。恢复 AI 后端 (vLLM myai, :30000) 执行:"
echo "  /root/ai/large_models/qwen_3.8_27b/vllm_service.sh start"
