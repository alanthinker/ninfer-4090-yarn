#!/usr/bin/env bash
# NInfer-YaRN 黑盒性能 bench (复用 ninfer-4090d-bench 仓库)
#
# 用法:
#   ./run_bench.sh smoke        # 快速冒烟(少量 fixture)
#   ./run_bench.sh niah         # 长文检索 (8k/64k/128k/256k, 超窗自动跳过)
#   ./run_bench.sh scenario     # 代码/故事/翻译/结构化
#   ./run_bench.sh decode       # 数学长解码 (--maxnew=4096, 较慢)
#
# 环境变量:
#   ADX_BASE     服务地址 (默认 http://127.0.0.1:30000/v1/chat/completions)
#   ADX_MODEL    模型名 (默认 myai)
#   ADX_MAX_CTX  客户端跳窗阈值 (默认 555000, 与 YaRN 扩展后的逻辑上限一致;
#                想只跑原生档就设 262144)
# 结果输出: ninfer-4090d-bench/profiles/bench/<时间戳>/results.json + skipped.json
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BENCH_DIR="$HERE/../ninfer-4090d-bench"
MODE="${1:-niah}"
cd "$BENCH_DIR" || exit 1

export ADX_BASE="${ADX_BASE:-http://127.0.0.1:30000/v1/chat/completions}"
export ADX_MODEL="${ADX_MODEL:-myai}"
export ADX_MAX_CTX="${ADX_MAX_CTX:-724000}"
# The 256K NIAH fixture prefills 262144 tokens; at ~420-700 tok/s that is 6-10 minutes, so the
# upstream 300 s default times the request out before it can answer. Allow a long single request.
export ADX_REQ_TIMEOUT="${ADX_REQ_TIMEOUT:-1800}"
echo "bench 模式: $MODE   base: $ADX_BASE   model: $ADX_MODEL   max_ctx: $ADX_MAX_CTX   req_timeout: ${ADX_REQ_TIMEOUT}s"

case "$MODE" in
  decode) exec env ADX_REQ_TIMEOUT="${ADX_REQ_TIMEOUT}" python3 bench/adx_blackbox_serve.py decode --maxnew=4096 ;;
  niah|scenario|smoke) exec python3 bench/adx_blackbox_serve.py "$MODE" ;;
  *) echo "用法: $0 {smoke|niah|scenario|decode}"; exit 1 ;;
esac
