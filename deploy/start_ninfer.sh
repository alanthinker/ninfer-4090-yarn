#!/usr/bin/env bash
# 启动 NInfer-YaRN (Qwen3.8-27B, E8 4-bit KV, YaRN 线性位置缩放, MTP3, Vision)
#
# 这是 ninfer-4090-yarn 这份独立构建的启动脚本, 与 deploy/ 下的原版互不影响:
# 二进制来自 ninfer-4090-yarn/build, 日志/PID 也落在本目录。
#
# 前置: 必须先停掉 vLLM (myai, :30000) —— 它是本会话 AI 的模型后端。
#
# 环境变量 (全部可选):
#   NINFER_PORT          服务端口, 默认 30000 (与 vLLM 一致, 客户端不用改)
#   NINFER_MAX_CTX       逻辑上下文上限, 默认 658176 (4 路并发下显存允许的最大值)
#   NINFER_YARN_FACTOR   YaRN 缩放因子, 默认 2.51 (= 658176/262144); 设 1.0 关闭扩展回到原生
#   NINFER_YARN_ORIG     YaRN 起点阈值, 默认 262144 (模型原生训练长度)
#   NINFER_CONCURRENCY   并发路数, 默认 4
#   NINFER_KV_DTYPE      KV 量化, 默认 rk4v4-e8
#   NINFER_NO_VISION     设 1 则不加 --vision, 省约 0.6 GiB
#   NINFER_MODEL         模型路径, 默认复用 ninfer-4090/models 下已校验的那份
#   NINFER_AUTO_LONG_ANCHORS / NINFER_MAX_LONG_ANCHORS
#                        长锚点窗口与保留上限, 默认 32。跨会话前缀缓存就靠它: 窗口是从末尾往前数
#                        的 N 个消息边界, 必须 N >= 消息数-1 才能包住"系统提示词末尾"这条所有会话
#                        都相同的边界。见 README 前缀缓存一节。
#   NINFER_MAX_SHARED_PREFIXES  共享前缀目录容量, 默认 4 (实测调大不解决长会话问题)
#   NINFER_DUMP_REQUESTS 若设为目录, 服务会把每个推理请求的原始 JSON body 落盘到该目录,
#                        用于对比"为什么两个会话没共享前缀缓存"(请求体不落盘时无法从服务端诊断)
#                        用法: NINFER_DUMP_REQUESTS=$PWD/reqdump ./start_ninfer.sh
#                        自动清理(默认开启, 每次写入时清理一次, 不会无限增长):
#                          NINFER_DUMP_REQUESTS_LIMIT         最多保留多少个 body, 默认 100 (0=不限)
#                          NINFER_DUMP_REQUESTS_MAX_AGE_HOURS 超过多少小时删除, 默认 24 (0=不限)
#                        只清理本脚本自己写的 req-*.json, 目录里其他文件不动。
#
# 并发与容量的关系(实测, 见 README): 池子按需共享、不均分。4 路时池子 658,176,
# 单个会话活跃时能用满, 4 路同时活跃则各分到约 164,544。
# 想最大化单会话上下文改用 NINFER_CONCURRENCY=1 NINFER_MAX_CTX=724000 NINFER_YARN_FACTOR=2.76。
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NINFER_DIR="$HERE/../ninfer-4090-yarn"
BIN="$NINFER_DIR/build/apps/ninfer-serve"
MODEL="${NINFER_MODEL:-$HERE/../ninfer-4090/models/qwen3_8_27b.ninfer}"
PORT="${NINFER_PORT:-30000}"
LOG="$HERE/ninfer_serve.log"
PIDF="$HERE/ninfer_serve.pid"

MAX_CTX="${NINFER_MAX_CTX:-658176}"
YARN_FACTOR="${NINFER_YARN_FACTOR:-2.51}"
YARN_ORIG="${NINFER_YARN_ORIG:-262144}"
CONCURRENCY="${NINFER_CONCURRENCY:-4}"
KV_DTYPE="${NINFER_KV_DTYPE:-rk4v4-e8}"
# 缺省推理档位 (--default-reasoning-effort)。引擎把档位**渲染进 prompt 开头**的一段指导语
# (medium 不渲染, xhigh 渲染 38 个 token 的 "Reasoning effort is set to xhigh..."), 而这段
# 前缀在消息之前 —— 档位一变, 后面所有 token 的位置整体平移, 前缀缓存从头失效。客户端(DSH)
# 的普通对话显式发 reasoning_effort=medium, 而它的压缩摘要请求不带这个字段, 于是引擎落到
# xhigh 缺省 → 摘要请求每次都是整段冷 prefill (实测 18,279 token 全冷; 277k 会话那次 278,038
# token 全冷、撞 5 分钟流超时)。把缺省钉在客户端实际发送的档位上, 两边 prompt 就逐字节一致。
# 根治在客户端 (compaction-basic 应当把会话档位带上), 这里是不改客户端的服务端兜底。
DEFAULT_EFFORT="${NINFER_DEFAULT_EFFORT:-medium}"
# 跨会话前缀缓存的关键: 自动长锚点窗口 + 保留上限。
# 引擎在每个 prompt 的「最后 N 个消息边界」上放置私有长锚点 (N=--auto-long-anchors),
# 而每条 continuation 最多保留 --max-long-anchors-per-continuation 个, 满了就替换最浅的那个。
# 窗口是从结尾往前数的, 所以 N 太小 + 历史消息多时, 唯一对所有会话都相同的那条边界
# (developer/system 之后, 即整个系统提示词的末尾) 会落在窗口之外 —— 于是每个新会话都 0 命中。
# 实测 (本机, 同一份 DSH 历史会话 payload): N=4 时 7/21 条消息的会话全部 0 命中 (7.6-8.5s);
# N=8 时 7 条消息能命中 (1.4s) 但 10 条消息仍 0 命中; N=32 时 7/10/21 条消息都能命中 8,669
# (87-90%, TTFT 1.4-2.6s)。显存/内存占用与 N 无关 (锚点复用已预留的 snapshot arena, 实测
# RSS 36.57 GiB、显存 30,966 MiB 在 N=4/8/32/64 下完全相同), 所以默认给足 32。
AUTO_ANCHORS="${NINFER_AUTO_LONG_ANCHORS:-32}"
MAX_ANCHORS="${NINFER_MAX_LONG_ANCHORS:-64}"
# 共享前缀目录容量。引擎自己对每个 prompt 提出三个候选: 「全部 tools 之后」「连续 leading
# System/Developer 之后」「full prompt」——第二个就是所有会话都相同的系统提示词末尾。
# 但这些是 EngineStructural 证据, 按设计 (docs/maintainer/resource-scheduling-and-context-cache.md
# §7.2) 只能用"不降低现有 owner 的空余终态", 默认容量只有 max(并发,4)=4, 容易被占满而发布不出去。
SHARED_PREFIXES="${NINFER_MAX_SHARED_PREFIXES:-4}"
# Host StateImage 槽位数。每个 checkpoint(会话端点 1 个 + 每个长锚点 1 个)都要独占一张
# GDN 递归状态快照,147 MiB,而且它一旦没有 device/host 副本,该 checkpoint 就按设计不可用
# —— 即使它的 KV 页还在显存里。所以槽位数不足时,症状是"深度端点反复消失、命中退化到浅的
# 共享前缀、每轮重算几万 token",而不是报错。
#   Host 槽位需求 ≈ 并发保留的会话数 x (2 + AUTO_ANCHORS) + 活跃 lane 的余量
#   160 槽 ≈ 23.5 GiB 常驻(在 32 GiB host KV 之外),够 4~5 个长会话用满 32 个锚点
#   内存不够就减小 AUTO_ANCHORS(每减 8 个锚点省 ~1.2 GiB/会话)
# 注意 device_state_slots 只有 4(加 4 个活跃 lane);深度端点绝大多数时候待在 host 槽里,
# 所以这个数字才是长会话能不能"睡下去再醒来"的关键。
HOST_STATE_SLOTS="${NINFER_HOST_STATE_SLOTS:-160}"
# 自动长锚点的"铺开"间隔(token)。除了上面"最后 N 个消息边界"的窗口(服务于最近编辑),
# 再在整个 prompt 上每隔 N 个 token 放一个锚点,对齐到其后的第一个消息边界。
# 为什么需要:会话压缩(compaction)会保留末尾一小段逐字内容、丢掉中间,并把摘要指令接在切割点
# 上 —— 于是上一次会话的所有检查点(端点、尾部锚点)都在切割点**之后**,一个都用不上,压缩只能
# 从 token 0 重新 prefill(~55 万 token,8-10 分钟,容易被客户端超时打断)。铺开后,切割点下方
# 最近的锚点就能被复用,压缩只需 prefill ≤ N 个 token。
#   0 = 关闭(只保留尾部窗口;上游行为)
#   32768 时,658k 上下文约产生 20 个铺开锚点 + 32 个尾部锚点 = 52 张状态镜像 ≈ 7.6 GiB/会话,
#   所以 --host-state-slots 要按 会话数 x (2 + 尾部锚点 + 铺开锚点) 配足(160 够 3 个会话)。
ANCHOR_SPACING="${NINFER_ANCHOR_SPACING:-32768}"

echo "== 前置检查 =="
[ -x "$BIN" ] || { echo "错误: 二进制不存在: $BIN"; echo "  先编译: cd $NINFER_DIR && cmake --build build -j"; exit 1; }
[ -f "$MODEL" ] || { echo "错误: 模型文件不存在: $MODEL"; exit 1; }
SZ=$(stat -c%s "$MODEL")
EXPECT=18210531328
if [ "$SZ" != "$EXPECT" ]; then
  echo "警告: 模型大小 $SZ != $EXPECT. 校验命令:"
  echo "  cd $(dirname "$MODEL") && echo 'eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e  $(basename "$MODEL")' | sha256sum --check"
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

# YaRN 参数: factor<=1.0 时不传, 保持原生行为
YARN_ARGS=()
YARN_DESC="关闭 (原生 262144)"
if awk "BEGIN{exit !($YARN_FACTOR > 1.0)}"; then
  YARN_ARGS=(--rope-scaling-factor "$YARN_FACTOR" --rope-scaling-original-context "$YARN_ORIG")
  YARN_DESC="开启 factor=$YARN_FACTOR 起点=$YARN_ORIG → 逻辑上限 $MAX_CTX"
fi
VISION_ARGS=(--vision)
[ "${NINFER_NO_VISION:-0}" = "1" ] && VISION_ARGS=()

echo "== 启动服务 (kv=$KV_DTYPE / ctx=$MAX_CTX / YaRN: $YARN_DESC / 并发 $CONCURRENCY / 长锚点 auto=$AUTO_ANCHORS max=$MAX_ANCHORS / 共享前缀 $SHARED_PREFIXES) =="
if [ -n "${NINFER_DUMP_REQUESTS:-}" ]; then
  mkdir -p "$NINFER_DUMP_REQUESTS"
  echo "请求体落盘: $NINFER_DUMP_REQUESTS"
  echo "  保留策略: 最多 ${NINFER_DUMP_REQUESTS_LIMIT:-100} 个 / ${NINFER_DUMP_REQUESTS_MAX_AGE_HOURS:-24} 小时 (0=不限); 含完整对话内容, 不需要时清空该目录即可"
fi
echo "===== $(date '+%F %T') 启动 (yarn factor=$YARN_FACTOR ctx=$MAX_CTX) =====" >> "$LOG"
# setsid: 让服务脱离启动者的会话/进程组。否则启动它的 shell 收到 SIGTERM 时, 整组连同
# nohup 的子进程一起被杀 —— 服务启动脚本本身不应该有这个脆弱性。
setsid nohup "$BIN" "$MODEL" \
  --host 0.0.0.0 --port "$PORT" \
  --model-id myai \
  --max-context "$MAX_CTX" --kv-capacity auto \
  --max-concurrency "$CONCURRENCY" --max-pending-requests 16 \
  --pending-timeout-ms 600000 \
  --prefill-chunk 1024 --kv-dtype "$KV_DTYPE" \
  --default-max-tokens 32768 \
  --spec mtp --draft-tokens 3 --lm-head-draft \
  "${VISION_ARGS[@]}" \
  "${YARN_ARGS[@]}" \
  --host-kv-mib 32768 \
  --host-state-slots "$HOST_STATE_SLOTS" \
  --max-private-continuations 8 \
  --max-shared-prefixes "$SHARED_PREFIXES" \
  --max-long-anchors-per-continuation "$MAX_ANCHORS" \
  --auto-long-anchors "$AUTO_ANCHORS" \
  --auto-anchor-spacing "$ANCHOR_SPACING" \
  --default-reasoning-effort "$DEFAULT_EFFORT" \
  --preserve-thinking >>"$LOG" 2>&1 &
echo $! > "$PIDF"
echo "pid=$(cat "$PIDF")  日志: $LOG"

echo "== 等待就绪 (最长 3 分钟) =="
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
echo
echo "完成。API: http://127.0.0.1:$PORT/v1  模型名: myai"
echo "本次生效的 YaRN: $YARN_DESC"
