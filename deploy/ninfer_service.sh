#!/usr/bin/env bash
# NInfer-YaRN 服务管理 (Qwen3.8-27B, E8 4-bit KV, YaRN 线性位置缩放, MTP3, Vision)
#   本仓库 (ninfer-4090-yarn) 的部署脚本: 二进制来自本仓库 build/, 日志/PID 落在本目录 (deploy/)。
#   ./ninfer_service.sh start | stop | status   (缺省 status)
#
# 前置: 必须先停掉 vLLM (myai, :30000) —— 它是本会话 AI 的模型后端:
#   /root/ai/large_models/qwen_3.8_27b/vllm_service.sh stop
#
# 环境变量 (全部可选, 只对 start 生效):
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
#   NINFER_FAIR_SHARE_BUCKETS  公平份额桶数: 最近活跃的 N 个空闲会话被驱逐硬保护, 默认 8, 0 关闭
#   NINFER_DUMP_REQUESTS  请求体落盘目录, 默认 $HERE/reqdump (每次 start 自动开启)。服务把每个推理
#                        请求的原始 JSON body 写进该目录, 是服务端排查"为什么前缀缓存没命中"的唯一
#                        原始证据(相邻两次请求的 body 可逐字节 diff)。设为 0 或空字符串关闭。
#                        自动清理(默认开启, 每次写入时清理一次, 不会无限增长):
#                          NINFER_DUMP_REQUESTS_LIMIT         最多保留多少个 body, 本脚本默认 2000 (二进制内置 100 太小:
#                                                             本服务 ~75 req/h, 100 个只有 ~1.5 小时; 0=不限)
#                          NINFER_DUMP_REQUESTS_MAX_AGE_HOURS 超过多少小时删除, 默认 24 (0=不限)
#                        只清理服务自己写的 req-*.json, 目录里其他文件不动。
#   NINFER_REUSE_DIAG    前缀复用诊断, 默认 1 (每次 start 自动开启)。每个请求把前缀索引里
#                        每一个 checkpoint 及拒绝它的具体门槛(index-invalid / 状态 / 内容不匹配
#                        等)逐条打到 stderr(即服务日志)。只读不改行为, 用于区分 0% 命中到底是
#                        "请求字节变了"还是"规划器没来得及评估候选"(长上下文的 time_budget 超时)。
#                        日志量随 checkpoint 数线性增长, 不需要时设 0 或空字符串关闭。
#   NINFER_LOG_LEVEL     日志级别, 默认 debug。相比 info 多出的只有: 启动时一条内存台账(权重后/
#                        启动后/余量)、每个 prompt 一条上下文成本(传输/预填/画像)、warmup 两条
#                        —— 共个位数行/请求, 对性能无可测影响。(trace 与 debug 输出完全相同:
#                        源码里没有任何 trace 级别日志点。)
#   NINFER_LOG_MAX_BYTES 服务日志滚动阈值, 默认 32MiB。服务日志只追加不滚动(无 logrotate);
#                        每次启动前若 ninfer_serve.log 超过阈值, 归档为 ninfer_serve.log.1
#                        (更旧的 .1→.2 依次后移), 最多保留 NINFER_LOG_KEEP 份归档(默认 2)。
#   NINFER_LOG_KEEP      归档份数, 默认 2 (即 log + log.1 + log.2 三个文件)。
#   NINFER_SKIP_MEM_CHECK 设 1 跳过启动前的内存门禁(默认开启)。
#
# 本脚本不会自动停 vLLM: 启动前检查 vLLM (python -m vllm.entrypoints 父进程 / VLLM:: 子进程)
# 是否还在运行, 在则报错退出, 需要先手工执行:
#   /root/ai/large_models/qwen_3.8_27b/vllm_service.sh stop
# 注意 vLLM 被停掉/强杀后 /dev/shm/vllm_offload_*.mmap 的 offload 残留不会自己 unlink, 残留的
# 文件页会一直占着 RAM, 可能让下面的内存门禁拒绝启动。
# 内存门禁按 host state 槽位 x 147 MiB + host KV 32 GiB + 4 GiB 余量估算需求, 不够就明确
# 拒绝启动: 否则不会有任何报错, 只会在 "pinning host KV" 之后被内核 OOM 杀掉。
#
# 并发与容量的关系(实测, 见 README): 池子按需共享、不均分。4 路时池子 658,176,
# 单个会话活跃时能用满, 4 路同时活跃则各分到约 164,544。
# 想最大化单会话上下文改用 NINFER_CONCURRENCY=1 NINFER_MAX_CTX=724000 NINFER_YARN_FACTOR=2.76。
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
NINFER_DIR="$HERE/.."
BIN="$NINFER_DIR/build/apps/ninfer-serve"
MODEL="${NINFER_MODEL:-$HERE/../../ninfer-4090/models/qwen3_8_27b.ninfer}"
PORT="${NINFER_PORT:-30000}"
LOG="$HERE/ninfer_serve.log"
PIDF="$HERE/ninfer_serve.pid"
# 该模型对应的进程特征 (pgrep -f / pkill -f): 二进制路径 + 模型路径都在命令行里,
# 用它可以在 27B / 35B 两个服务共用同一份二进制时精确区分、避免误杀。
PROC_RE="$BIN $MODEL"

case "${1:-status}" in
start)
  if pgrep -f "$PROC_RE" >/dev/null; then
    echo "已在运行 (pid $(pgrep -f "$PROC_RE" | head -1))"
    exit 0
  fi

  # 启动前滚动服务日志: 超阈值则归档旧文件(.1→.2 后移), 删除超出的最老归档。
  LOG_MAX_BYTES="${NINFER_LOG_MAX_BYTES:-$((32 * 1024 * 1024))}"
  LOG_KEEP="${NINFER_LOG_KEEP:-2}"
  if [ -f "$LOG" ] && [ "$(stat -c%s "$LOG")" -gt "$LOG_MAX_BYTES" ]; then
    rm -f "$LOG.$LOG_KEEP"
    for ((i = LOG_KEEP - 1; i >= 1; i--)); do
      [ -f "$LOG.$i" ] && mv -f "$LOG.$i" "$LOG.$((i + 1))"
    done
    mv -f "$LOG" "$LOG.1"
    echo "日志已滚动: 旧日志归档为 $LOG.1 (保留 ${LOG_KEEP} 份归档)"
  fi

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
  # 同时保留的续算条目数(每个会话链一份端点/rewrite)。8 时,5~6 条会话每次发布都挤同一个
  # 上限,容易把别的会话的深度端点顶掉;16 给多会话留出余量。真正的上限是 Host KV 池
  # (32 GiB ≈ 188 万 token),条目再多也超不过它。
  PRIVATE_CONTINUATIONS="${NINFER_MAX_PRIVATE_CONTINUATIONS:-16}"
  # 共享前缀目录容量。引擎自己对每个 prompt 提出三个候选: 「全部 tools 之后」「连续 leading
  # System/Developer 之后」「full prompt」——第二个就是所有会话都相同的系统提示词末尾。
  # 但这些是 EngineStructural 证据, 按设计 (docs/maintainer/resource-scheduling-and-context-cache.md
  # §7.2) 只能用"不降低现有 owner 的空余终态", 默认容量只有 max(并发,4)=4, 容易被占满而发布不出去。
  SHARED_PREFIXES="${NINFER_MAX_SHARED_PREFIXES:-8}"
  # Host StateImage 槽位数。每个 checkpoint(会话端点 1 个 + 每个长锚点 1 个)都要独占一张
  # GDN 递归状态快照,147 MiB,而且它一旦没有 device/host 副本,该 checkpoint 就按设计不可用
  # —— 即使它的 KV 页还在显存里。所以槽位数不足时,症状是"深度端点反复消失、命中退化到浅的
  # 共享前缀、每轮重算几万 token",而不是报错。
  #   Host 槽位需求 ≈ 并发保留的会话数 x (2 + 尾部锚点 + 铺开锚点) + 活跃 lane 的余量
  #   实测 2026-09-12:一条 278k 会话在 07:58-08:03 全热(100%/99.9%),08:25 再发时目录里只剩
  #   26 个 checkpoint、它自己的深度工件一个都不剩,只能从 token 0 重算 —— 同一时刻按
  #   "5 个会话 x (2 + 32 尾部 + 8~10 铺开) ≈ 165" 已经越过当时 160 槽的上限,引擎按策略
  #   把 checkpoint 释放掉了。所以这里按"这台机器专职跑这个"给足:
  #   320 槽 ≈ 46 GiB 常驻(在 32 GiB host KV 之外),够 6~7 个长会话同时保住全套锚点。
  #   内存账(本机 91.9 GiB):host state 46 + host KV 32 + 引擎自身 ≈ 80 GiB,余 ~11 GiB;
  #   跑别的重活(第二个引擎实例、大编译)之前先把这个数降下来,或用
  #   NINFER_HOST_STATE_SLOTS=256 临时跑。
  # device_state_slots 保持 4(引擎缺省):那是"活跃 lane 之外还能在显存里留住几个 checkpoint"的额度,
  # 深度端点绝大多数时候待在 host 槽里,所以 host 槽位数才是长会话能不能"睡下去再醒来"的关键。
  # 实测 2026-09-12:改成 8 时启动直接被拒 ——
  #   "minimum Engine runtime reservation requires 14869674752 B + 1 GiB headroom,
  #    but only 15329323520 B are available after weights"
  # 差的 0.57 GiB 正好是 4 x 147 MiB。weights 之后的可用显存只有 ~15.3 GB,没有余量再加槽位。
  HOST_STATE_SLOTS="${NINFER_HOST_STATE_SLOTS:-320}"
  DEVICE_STATE_SLOTS="${NINFER_DEVICE_STATE_SLOTS:-4}"
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
  # 公平份额桶 (2026-09-12 17:09 事故修复)。最近活跃的 N 个空闲会话的整套 checkpoint
  # (状态镜像 + KV 页, 随 owner 一体) 被硬保护: 别的会话的压力(增长/压缩 churn)只能驱逐
  # 共享池里的 owner, 动不到它们的深度端点。没有这个, 空闲会话会被"价值模型"当成最划算的
  # 释放对象 —— 实测 155k 空闲会话在邻居 76 分钟 churn 中端点被整段驱逐, 回来 0% 命中、
  # 全额重算 3m40s。只有当共享池耗尽、连最大释放计划都放不下新请求时, 引擎才按 MRU 顺序
  # 释放最老的桶 (请求照样能跑, 不会被误判为 blocked), 然后重试。0 = 关闭回到纯价值模型。
  FAIR_SHARE_BUCKETS="${NINFER_FAIR_SHARE_BUCKETS:-8}"
  # 日志级别。二进制默认 info; 抬到 debug 只多 4 个日志点(启动内存台账、每 prompt 一条上下文成本、
  # warmup 两条), 是排查显存/Host 槽位/预填成本问题时的有用增量, 代价可忽略。
  LOG_LEVEL="${NINFER_LOG_LEVEL:-debug}"

  echo "== 前置检查 =="
  [ -x "$BIN" ] || { echo "错误: 二进制不存在: $BIN"; echo "  先编译: cd $NINFER_DIR && cmake --build build -j"; exit 1; }
  [ -f "$MODEL" ] || { echo "错误: 模型文件不存在: $MODEL"; exit 1; }
  SZ=$(stat -c%s "$MODEL")
  EXPECT=18210531328
  if [ "$SZ" != "$EXPECT" ]; then
    echo "警告: 模型大小 $SZ != $EXPECT. 校验命令:"
    echo "  cd $(dirname "$MODEL") && echo 'eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e  $(basename "$MODEL")' | sha256sum --check"
  fi
  # vLLM (myai, :30000) 是本会话 AI 的模型后端, 与本服务互斥: 共用 :30000 和同一块 GPU,
  # 它的 KV offload 段还要占 30+ GiB RAM。本脚本不自动停它: 还在运行就报错退出,
  # 必须先手工执行 /root/ai/large_models/qwen_3.8_27b/vllm_service.sh stop。
  if pgrep -f '^[^ ]*python[0-9.]* -m vllm[.]entrypoints' >/dev/null \
     || pgrep -f '^VLLM::' >/dev/null; then
    echo "错误: vLLM (myai, :30000) 还在运行。先执行:"
    echo "  /root/ai/large_models/qwen_3.8_27b/vllm_service.sh stop"
    exit 1
  fi

  # 端口检查(vLLM 还在跑时上面已经退出; 这里确认 :30000 真的空出来了,
  # 若被非 vLLM 的东西占着, 由这里报错。)
  if ss -tln | grep -q ":$PORT "; then echo "错误: 端口 $PORT 被占用"; exit 1; fi

  # 内存门禁。本服务启动时要把 host state (槽位 x 147 MiB) 和 host KV (--host-kv-mib)
  # 整块 pin 在 RAM 里, 再加引擎自身/视觉缓存约 4 GiB。不够的话不会报错, 而是走到
  # "pinning host KV" 那一行之后被内核 OOM 直接杀掉 —— 日志里没有任何错误信息, 只能靠
  # dmesg/journalctl 才发现, 极难定位。所以这里先把账算出来, 不够就明确拒绝启动。
  # 实测 2026-09-13: 32 GiB offload 残留 + swap 用满, 三次启动(ninfer_serve.log 与
  # journalctl -k)全部死在 pinning host KV, 没有任何一行报错。
  HOST_KV_MIB=32768   # 与下面启动命令里的 --host-kv-mib 保持一致
  NEED_MIB=$(( HOST_STATE_SLOTS * 147 + HOST_KV_MIB + 4096 ))
  AVAIL_MIB=$(awk '/^MemAvailable:/{printf "%d", $2/1024}' /proc/meminfo)
  SWAP_FREE_MIB=$(awk '/^SwapFree:/{printf "%d", $2/1024}' /proc/meminfo)
  echo "内存检查: MemAvailable ${AVAIL_MIB} MiB / 需要 ≈ ${NEED_MIB} MiB" \
       "(host state ${HOST_STATE_SLOTS}x147 + host KV ${HOST_KV_MIB} + 引擎与余量 4096); swap 可用 ${SWAP_FREE_MIB} MiB"
  if [ "${AVAIL_MIB:-0}" -lt "$NEED_MIB" ] && [ "${NINFER_SKIP_MEM_CHECK:-0}" != "1" ]; then
    echo "错误: 可用内存不足 —— 继续启动会在 pinning host KV 阶段被 OOM 杀掉(日志里不会有错误行)。"
    echo "  排查: free -g; ls -lh /dev/shm/; pgrep -af 'vllm|ninfer'; journalctl -k | grep -i 'killed process'"
    echo "  处理: 清掉 /dev/shm 残留、停掉其它大内存服务; 或用 NINFER_HOST_STATE_SLOTS=256 降低需求"
    echo "        (每减 64 槽省 9.2 GiB); 重启机器可一并清空 swap。确认余量足够时可 NINFER_SKIP_MEM_CHECK=1 跳过。"
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

  echo "== 启动服务 (kv=$KV_DTYPE / ctx=$MAX_CTX / YaRN: $YARN_DESC / 并发 $CONCURRENCY / 长锚点 auto=$AUTO_ANCHORS max=$MAX_ANCHORS / 共享前缀 $SHARED_PREFIXES / 公平份额桶 $FAIR_SHARE_BUCKETS) =="
  # 请求体落盘(默认开启)。二进制是直接 getenv("NINFER_DUMP_REQUESTS") 读的(非命令行参数),
  # 所以这里必须 export, 保证 setsid nohup 起的服务进程一定继承到该目录。
  DUMP_REQUESTS="${NINFER_DUMP_REQUESTS:-$HERE/reqdump}"
  if [ -n "$DUMP_REQUESTS" ] && [ "$DUMP_REQUESTS" != "0" ]; then
    export NINFER_DUMP_REQUESTS="$DUMP_REQUESTS"
    # 二进制内置缺省只留 100 个(全局, 不分会话): 本服务实测 ~75 req/h, 100 个仅覆盖 ~1.5 小时,
    # 单个长会话就能超掉, 事故证据来不及看就被顶走。默认抬到 2000 (≈ 一整天的量),
    # 磁盘占用仍受 24 小时龄期上限兜底 (实测节奏下约 1~2 GB)。
    export NINFER_DUMP_REQUESTS_LIMIT="${NINFER_DUMP_REQUESTS_LIMIT:-2000}"
    mkdir -p "$DUMP_REQUESTS"
    echo "请求体落盘: $DUMP_REQUESTS"
    echo "  保留策略: 最多 ${NINFER_DUMP_REQUESTS_LIMIT} 个 / ${NINFER_DUMP_REQUESTS_MAX_AGE_HOURS:-24} 小时 (0=不限); 含完整对话内容, 不需要时清空该目录即可"
  else
    unset NINFER_DUMP_REQUESTS
    echo "请求体落盘: 已关闭 (设 NINFER_DUMP_REQUESTS=目录 可开启)"
  fi
  # 前缀复用诊断(默认开启)。同样是 getenv 读取(非命令行参数), 必须 export。
  # 每个请求逐条打印前缀索引里每个 checkpoint 的拒绝门槛: 排查长上下文 0% 命中时,
  # 用它区分"请求字节变了"(内容不匹配)与"规划器 time_budget 超时放弃"(候选没被评估)。
  REUSE_DIAG="${NINFER_REUSE_DIAG:-1}"
  if [ -n "$REUSE_DIAG" ] && [ "$REUSE_DIAG" != "0" ]; then
    export NINFER_REUSE_DIAG=1
    echo "前缀复用诊断: 开 (每个请求在日志里列出各 checkpoint 的拒绝原因; NINFER_REUSE_DIAG=0 关闭)"
  else
    unset NINFER_REUSE_DIAG
    echo "前缀复用诊断: 已关闭"
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
    --device-state-slots "$DEVICE_STATE_SLOTS" \
    --max-private-continuations "$PRIVATE_CONTINUATIONS" \
    --max-shared-prefixes "$SHARED_PREFIXES" \
    --max-long-anchors-per-continuation "$MAX_ANCHORS" \
    --auto-long-anchors "$AUTO_ANCHORS" \
    --auto-anchor-spacing "$ANCHOR_SPACING" \
    --fair-share-buckets "$FAIR_SHARE_BUCKETS" \
    --default-reasoning-effort "$DEFAULT_EFFORT" \
    --log-level "$LOG_LEVEL" \
    --preserve-thinking >>"$LOG" 2>&1 &
  echo $! > "$PIDF"
  echo "pid=$(cat "$PIDF")  日志: $LOG"

  echo "== 等待就绪 (最长 3 分钟) =="
  ready=0
  for _ in $(seq 1 36); do
    if curl -s -m 3 -o /dev/null "http://127.0.0.1:$PORT/v1/models"; then ready=1; break; fi
    if ! kill -0 "$(cat "$PIDF")" 2>/dev/null; then
      echo "进程已退出, 最后 40 行日志:"; tail -40 "$LOG"; rm -f "$PIDF"; exit 1
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
  ;;
stop)
  # 只匹配 "二进制路径 + 本模型路径", 不会误杀 35B 服务 (不同二进制+模型)。
  # 注意: deploy-yarn/ninfer_service.sh 用的是同一份二进制+同一模型, 命令行特征相同, 两个实例互可见。
  if [ -f "$PIDF" ]; then
    PID=$(cat "$PIDF")
    kill "$PID" 2>/dev/null || true
    for _ in $(seq 1 20); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
    kill -9 "$PID" 2>/dev/null || true
    rm -f "$PIDF"
  fi
  # 兜底: PID 文件丢失/陈旧时, 按命令行特征清残留
  if pgrep -f "$PROC_RE" >/dev/null; then
    echo "发现残留进程, 强杀:"
    pgrep -af "$PROC_RE" | sed 's/^/  /'
    pkill -9 -f "$PROC_RE" 2>/dev/null || true
  fi
  echo "显存状态:"
  nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
  echo
  echo "NInfer-YaRN (Qwen3.8-27B) 已停止。恢复 AI 后端 (vLLM myai, :30000) 执行:"
  echo "  /root/ai/large_models/qwen_3.8_27b/vllm_service.sh start"
  ;;
status)
  pid=""
  [ -f "$PIDF" ] && pid=$(cat "$PIDF")
  { [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null; } || pid=$(pgrep -f "$PROC_RE" | head -1)
  if [ -n "$pid" ]; then
    echo "pid: $pid"
    curl -s -m 5 "http://127.0.0.1:$PORT/v1/models" | python3 -c "import sys,json
try:
    d=json.load(sys.stdin); m=d['data'][0]
    print('model:', m['id'], 'max_len:', m.get('max_model_len'))
except Exception: print('model: (API 未响应)')"
    nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader
    echo "日志: $LOG"
  else
    rm -f "$PIDF"
    echo "未运行。启动: $0 start"
    echo "恢复 vLLM 后端: /root/ai/large_models/qwen_3.8_27b/vllm_service.sh start"
  fi
  ;;
*) echo "用法: $0 {start|stop|status}"; exit 1;;
esac
