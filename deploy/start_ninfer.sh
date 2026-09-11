#!/usr/bin/env bash
# 启动 NInfer (Qwen3.8-27B, E8 4-bit KV, 全 262K 原生上下文, MTP3)
# 前置: 必须先停掉 vLLM (myai, :30000) —— 它是本会话 AI 的模型后端。
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NINFER_DIR="$HERE/../ninfer-4090"
BIN="$NINFER_DIR/build/apps/ninfer-serve"
MODEL="$NINFER_DIR/models/qwen3_8_27b.ninfer"
PORT="${NINFER_PORT:-30000}"   # 与 vLLM 同端口, 客户端配置不用改
LOG="$HERE/ninfer_serve.log"
PIDF="$HERE/ninfer_serve.pid"

echo "== 前置检查 =="
[ -x "$BIN" ] || { echo "错误: 二进制不存在: $BIN"; exit 1; }
[ -f "$MODEL" ] || { echo "错误: 模型文件不存在: $MODEL (下载未完成?)"; exit 1; }
SZ=$(stat -c%s "$MODEL")
EXPECT=18210531328
if [ "$SZ" != "$EXPECT" ]; then
  echo "警告: 模型大小 $SZ != $EXPECT (官方 16.96 GiB / rev 3526913004)。请用下面命令校验 SHA-256:"
  echo "  cd $NINFER_DIR/models && echo 'eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e  qwen3_8_27b.ninfer' | sha256sum --check"
fi
if ss -tln | grep -q ":$PORT "; then echo "错误: 端口 $PORT 被占用"; exit 1; fi
if pgrep -f "vllm[.]entrypoints" >/dev/null; then
  echo "错误: vLLM (myai, :30000) 还在运行。先执行:"
  echo "  /root/ai/large_models/qwen_3.8_27b/vllm_service.sh stop"
  exit 1
fi
# 显存检查。nvidia-smi 失败通常不是缺驱动, 而是 NVIDIA 用户态库被 apt 升级后与仍在内存里的
# 旧内核模块版本不匹配(常见于 unattended-upgrade 之后): 已在运行的服务不受影响, 但任何新进程
# 都无法初始化 CUDA。这里给出可执行的提示, 而不是让脚本以一句空值的数值比较失败。
# 注意: nvidia-smi 在 NVML 失败时把错误文本写到 stdout 且退出码非 0, 管道还会把退出码换成
# head 的 0 —— 所以必须额外校验取到的是数字, 不能只看退出码或空值。
if ! FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1) \
   || ! [[ "${FREE:-}" =~ ^[0-9]+$ ]]; then
  echo "错误: 无法通过 nvidia-smi 读取显存。"
  if nvidia-smi 2>&1 | grep -qi 'version mismatch'; then
    echo "  原因: NVIDIA 用户态库与内核模块版本不匹配。"
    echo "    当前内核模块: $(cat /sys/module/nvidia/version 2>/dev/null || echo 未知)"
    echo "    磁盘上模块:   $(modinfo nvidia 2>/dev/null | awk '/^version/{print $2}' || echo 未知)"
    echo "  已在运行的服务不受影响; 但本脚本要启动的新服务无法初始化 CUDA。"
    echo "  修复: 重启机器(会加载与库匹配的新模块), 重启后再执行本脚本。"
  else
    nvidia-smi 2>&1 | head -3 | sed 's/^/  /'
  fi
  exit 1
fi
echo "显存空闲: ${FREE} MiB"

# 待重启标记: 内核/驱动包已更新但尚未重启时, 已加载的内核模块仍是旧的。此时若 NVIDIA 用户态库
# 已随包升级, 新进程会以 "Driver/library version mismatch" 失败。这个信号系统自己就会给,
# 直接提示出来, 免得再次出现"服务重启后起不来"。
if [ -f /var/run/reboot-required ]; then
  echo "警告: /var/run/reboot-required 存在 —— 有更新(通常含内核/NVIDIA 驱动)尚未通过重启生效。"
  [ -f /var/run/reboot-required.pkgs ] && sed 's/^/    /' /var/run/reboot-required.pkgs | head -5
  echo "  内核模块: $(cat /sys/module/nvidia/version 2>/dev/null || echo 未知)" \
       "/ 磁盘上模块: $(modinfo nvidia 2>/dev/null | awk '/^version/{print $2}')"
  echo "  若两者不一致, 本脚本启动的服务将无法初始化 CUDA —— 请先重启机器。"
fi

if [ "$FREE" -lt 27000 ]; then echo "错误: 空闲显存不足 27000 MiB (需要 ~23 GiB)"; exit 1; fi

echo "== 启动服务 (E8 4-bit KV / 262144 ctx / MTP3 / Vision / 4 路) =="
# 32 GB 卡预算: 固定 17.55 GiB(权重+MTP+Graph) + vision 0.26 tower + 0.33 scratchpad KV
#              + 每路 lane 0.38 GiB; --kv-capacity auto 让引擎按实际可用显存算满池子
#              (留 1024 MiB headroom), 实际值看启动日志
# 4 路 = 最多 4 个会话按需共享池子(不均分), 实测池子 658,176 token:
#        单个会话活跃时能用满 658,176; 4 路同时活跃则各约 164,544。
#        --max-context 262144 是每会话的逻辑天花板(原生长度, 未启用 YaRN),
#        4 × 262144 = 1,048,576 > 658,176, 所以 4 路各要满 262K 时会有会话排队/被驱逐。
# 不需要多模态可删掉 --vision
# 日志用追加模式, 重启不丢历史请求记录
# setsid: 让服务脱离启动者的会话/进程组。否则启动它的 shell 收到 SIGINT/SIGTERM 时
#         (例如在终端里按 Ctrl+C), 整组连同 nohup 的子进程会一起被杀。
echo "===== $(date '+%F %T') 启动 (kv=${NINFER_KV_DTYPE:-rk4v4-e8}) =====" >> "$LOG"
setsid nohup "$BIN" "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --model-id myai \
  --max-context 262144 --kv-capacity auto \
  --max-concurrency 4 --max-pending-requests 16 \
  --pending-timeout-ms 600000 \
  --prefill-chunk 1024 --kv-dtype "${NINFER_KV_DTYPE:-rk4v4-e8}" \
  --default-max-tokens 32768 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  --vision \
  --host-kv-mib 32768 \
  --host-state-slots 16 \
  --max-private-continuations 8 \
  --max-long-anchors-per-continuation 4 \
  --auto-long-anchors 4 \
  --preserve-thinking >>"$LOG" 2>&1 &
# 历史会话重用 (对标 vLLM --kv-offloading-size 32):
#   --default-max-tokens 32768  客户端不传 max_tokens 时的输出预算 (引擎默认 8192 偏小:
#                           思考与回答共享预算, 且 NInfer 不支持 vLLM 的 thinking_token_budget,
#                           8192 容易被思考吃光导致没有答案/没有工具调用)。
#                           优先级: 请求里的 max_tokens > 本默认值; 硬天花板是 prompt+output ≤ max-context。
#   --host-kv-mib 32768      32 GiB pinned 主机内存保留非活跃会话的 KV/前缀 (默认 8 GiB)
#   --host-state-slots 16    混合模型的循环状态(48 层 GDN)也保留在主机内存 (默认 8)
#   --auto-long-anchors 4    在每个 prompt 的最后 4 个消息边界打锚点, 客户端改写近期历史时
#                            从锚点恢复而不是重新 prefill (配合上面的 anchor 上限与 host slots)
# 需要在重启后仍能恢复旧会话, 再追加(需先建目录, 客户端调 /slots/{id}?action=save|restore):
#   --slot-save-path /root/ai/large_models/_ninfer_repos/deploy/slots --auto-save-evicted
echo $! > "$PIDF"
echo "pid=$(cat "$PIDF")  日志: $LOG"

echo "== 等待就绪 (加载 17 GiB 权重 + CUDA Graph 捕获, 最长 3 分钟) =="
ready=0
for _ in $(seq 1 36); do
  if curl -s -m 3 -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then ready=1; break; fi
  if ! kill -0 "$(cat "$PIDF")" 2>/dev/null; then
    echo "进程已退出, 最后 40 行日志:"; tail -40 "$LOG"; exit 1
  fi
  sleep 5
done
[ "$ready" = 1 ] || { echo "3 分钟未就绪, 查看日志: $LOG"; exit 1; }

echo "== 服务信息 =="
curl -s "http://127.0.0.1:$PORT/v1/models" | python3 -m json.tool

echo "== 冒烟测试 =="
curl -s -m 120 "http://127.0.0.1:$PORT/v1/chat/completions" \
  -H 'content-type: application/json' \
  -d '{"model":"myai","messages":[{"role":"user","content":"用一句话介绍你自己。"}],"max_tokens":64}' \
  | python3 -m json.tool
echo "完成。API: http://127.0.0.1:$PORT/v1  模型名: myai  (与 vLLM 一致, 客户端不用改)"
