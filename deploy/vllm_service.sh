#!/usr/bin/env bash
# vLLM 生产服务管理（引擎: vLLM 0.28.0 @ cu130）—— 2026-09-04
#   当前部署文档: vLLM部署交接文档_历史存档_Qwen3.8-27B-AWQ.md
#   ./vllm_service.sh start | stop | status
#   2026-09-06 实测(见 sglang并发与上下文内存逻辑说明.md §vllm): 采用
#     max-model-len 220000 / gpu-util 0.97 / max-num-seqs 8 / MTP=3
#   - vLLM 的 KV 池 = 剩余显存, 与并发( max-num-seqs )无关;
#     降并发不能增大 context(与 sglang 的 mamba 槽不同); 提并发也不缩小 context。
#     实测 max-num-seqs 2 / 4 / 8 下 KV 池均 232,941, 并发同跑无 preempt。
#   - context 上限= max-model-len 能 boot 的最大值; 220k boot(KV 232,941, conv 1.06×);
#     240k 报 "KV cache needs 8.56GiB > available 8.42GiB" 拒绝。
#   - MTP=qwen3_5_mtp(3步) 实测 acceptance ~3.2; 2 并发正常, 无 preempt。
#   - 模型 Qwen3_5ForConditionalGeneration 为视觉/多模态架构(vision_config 存在);
#     vLLM 对 VLM 模型自动加载 vision tower, 无需额外多模态参数;
#     当前参数已支持多模态(2026-09-06 实测可用, 见 引擎选型对比与决策文档 §1.1)。
#   - KV offload: --kv-offloading-size 32 (GiB) 把 KV 块按内容哈希落到 CPU 内存(LRU 淘汰),
#     跨请求/会话复用,省 GPU 显存。注意:该 offload 缓冲是 mmap 到 /dev/shm(跨进程共享 + page-locked),
#     所以 /dev/shm 必须 ≥ offload 大小; 当前 /dev/shm remount 为 40G, 且进程被强杀时可能残留
#     /dev/shm/vllm_offload_*.mmap(需 rm 清掉, 否则占满 /dev/shm 会导致启动报 Insufficient space)。
#     本机 91G 内存主要跑大模型故保留 32G 较大档位。
set -uo pipefail
VLLM_VENV=/root/.vllm-venv
MODEL_DIR=/root/ai/large_models/qwen_3.8_27b/models/Qwen3.8-27B-AWQ-INT4
PORT=30000
LOGF=/root/ai/large_models/qwen_3.8_27b/.vllm_service.log
ARGS=(
  --model "$MODEL_DIR"
  --served-model-name myai
  --host 0.0.0.0 --port "$PORT"
  --max-model-len 220000
  --gpu-memory-utilization 0.97
  --max-num-seqs 8
  --kv-cache-dtype fp8_e4m3
  --enforce-eager
  --reasoning-parser qwen3
  --enable-auto-tool-choice
  --tool-call-parser qwen3_coder
  --enable-prefix-caching
  # 在响应 usage 里返回 prompt_tokens_details.cached_tokens(默认关闭, 客户端才能算缓存命中)
  --enable-prompt-tokens-details
  # MTP=3 (qwen3_5_mtp 模型内投机, 3 步); 实测 acceptance ~3.2@220k
  --spec-method qwen3_5_mtp
  --speculative-config '{"num_speculative_tokens": 3}'
  # KV 块 offload 到 CPU 内存(32 GiB, 内容哈希/LRU, 跨会话复用)
  --kv-offloading-size 32
)
case "${1:-status}" in
  start)
    pgrep -f "vllm[.]entrypoints" >/dev/null && { echo "已在运行 (pid $(pgrep -f 'vllm[.]entrypoints'|head -1))"; exit 0; }
    source "$VLLM_VENV/bin/activate"
    cd /root || exit 1
    nohup python -m vllm.entrypoints.openai.api_server "${ARGS[@]}" >"$LOGF" 2>&1 &
    echo "启动中 pid=$! ... 日志: $LOGF"
    for _ in $(seq 1 40); do curl -s -m 4 -o /dev/null "http://localhost:$PORT/health" && { echo "OK: ready on :$PORT"; exit 0; }; sleep 5; done
    echo "未就绪, 看日志: tail -60 $LOGF"; exit 1
    ;;
  stop)
    pkill -9 -f "vllm[.]entrypoints" 2>/dev/null; pkill -9 -f "[E]ngineCore" 2>/dev/null; sleep 2
    rm -f /dev/shm/vllm_offload_*.mmap   # 强杀可能残留 KV offload 映射, 不清会占满 /dev/shm 导致下次启动 Insufficient space
    pgrep -f "vllm[.]entrypoints" >/dev/null && echo "仍有残留" || echo "已停止"
    ;;
  status)
    pid=$(pgrep -f "vllm[.]entrypoints" | head -1)
    echo "pid: ${pid:-无}"
    curl -s -m 5 -o /dev/null -w "health: HTTP %{http_code}\n" "http://localhost:$PORT/health"
    curl -s -m 5 "http://localhost:$PORT/v1/models" | python3 -c "import sys,json
try:
    d=json.load(sys.stdin); print('model:', d['data'][0]['id'], 'max_len:', d['data'][0]['max_model_len'])
except Exception: pass"
    ;;
  *) echo "用法: $0 {start|stop|status}"; exit 1;;
esac
