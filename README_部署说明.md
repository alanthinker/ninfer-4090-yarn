# NInfer (Qwen3.8-27B) 部署说明 — RTX 4080S 32 GB

仓库与产物（已就绪，AI 侧全部完成并校验）:

| 项 | 位置 |
|---|---|
| 引擎仓库 (sm_89 fork) | `ai/large_models/_ninfer_repos/ninfer-4090/` |
| 编译产物 | `ninfer-4090/build/apps/{ninfer,ninfer-serve,ninfer-perplexity}` |
| 模型 (19.03 GiB, **SHA-256 已校验通过**) | `ninfer-4090/models/qwen3_8_27b.ninfer` |
| 黑盒 bench 仓库 | `ninfer-4090d-bench/` (manifest 已就位, 脚本已参数化) |
| 部署脚本 | 本目录 `deploy/` |

构建环境: Ubuntu 26.04, **CUDA 13.2.86** (13.1 与 glibc 2.43 头文件冲突, 已装 13.2 组件),
gcc-13, CMake 4.2.3, `CMAKE_CUDA_ARCHITECTURES=89`。

## 执行步骤（你来操作，因为第 1 步会停掉本会话 AI 的后端）

```bash
cd /root/ai/large_models/_ninfer_repos/deploy

# 1) 停掉 vLLM (myai, :30000 = 本 AI 的模型后端) —— 执行后本会话将中断
/root/ai/large_models/qwen_3.8_27b/vllm_service.sh stop

# 2) 启动 NInfer (E8 4-bit KV, 全 262K 原生上下文, MTP3, Vision)
#    自检: 二进制/模型/端口/vLLM 是否已停/显存 > 27 GiB, 然后等服务就绪 + 冒烟测试
bash start_ninfer.sh

# 3) 性能 bench (服务常驻, 逐个跑)
bash run_bench.sh smoke        # 最快, 先确认通路
bash run_bench.sh niah         # 长文检索 8k/64k (128k/256k 默认超窗跳过)
bash run_bench.sh scenario     # 代码/故事/翻译/结构化
bash run_bench.sh decode       # 数学长解码, 较慢
# 想连 256K NIAH 也跑: ADX_MAX_CTX=262144 bash run_bench.sh niah

# 4) 测完恢复 AI 后端
bash stop_ninfer.sh
/root/ai/large_models/qwen_3.8_27b/vllm_service.sh start
```

## 配置说明（start_ninfer.sh 内可改）

- `--kv-dtype rk4v4-e8`: E8 4-bit KV，262,144 全原生上下文，代价 ~5.7% decode。
  32 GB 卡也可用 `--kv-dtype int8` 跑满 262K（精度最高，KV ~9.7 GiB，总 ~27.7 GiB）。
- **脚本默认：`--max-concurrency 4 --kv-capacity auto`**。`auto` = 引擎按实际可用显存
  算满 KV 池（留 1024 MiB headroom），实际容量看启动日志；池子按需共享、不均分，
  4 路是上限而非均分。**实测该配置下池子 = 658,176 token**：单个会话活跃时能用满
  658,176；4 路同时活跃则各约 164,544。注意 4 × 262144 = 1,048,576 > 658,176，
  所以 4 路各要满 262K 时会有会话排队或被驱逐。
  每多一路的显存代价实测约 0.38 GiB（每路一张 device state image），
  所以 3 路的池子是 680,256、4 路是 658,176、8 路是 569,728。
  想省显存给别的用途就把 3 改小；想锁死单路官方口径可改回 `--max-concurrency 1
  --kv-capacity 262144`。
- **显存预算（本机实测可用 32,230 MiB = 31.47 GiB，E8 4-bit KV，含 vision）**：
  固定开销 17.55 GiB（权重+MTP+CUDA Graph）+ vision 0.26 GiB（tower）+
  scratchpad KV 0.33 GiB（8192 tok），每多一路 lane +0.39 GiB。
  主 KV 池 = 31.47 − 固定 − 1.0(运行时 headroom)：

  | 配置 | 主 KV 池上限 | 全 262K 路数 |
  |---|---:|---:|
  | 纯文本 1 路 | ~666K tok | 1 |
  | 纯文本 2 路 | ~646K tok | **2** |
  | +vision 1 路 | ~638K tok | 1 |
  | **+vision 2 路** | **~615K tok** | **2**（余 ~90K 给第 3 路） |
  | +vision 3 路 | ~596K tok | 2 + 第 3 路 ~70K |

- **全 262K 深度最多 2 路并发**（32 GB 卡独享，4090 24 GB 做不到）：2 路满 262K
  的 E8 主 KV = 10.16 GiB，总占用 ~28.7 GiB，还剩 ~2.8 GiB。第 3 路只能吃
  ~90K tokens 的剩余池子（再往上就得等空闲会话被驱逐或排队）。
- `--vision`: 多模态（图/视频）。不需要就删掉，省 ~0.6 GiB。
- 服务启动前先做显存校验，配置放不下会快速失败，不会挂到一半。

## 预期

- 4080S 比 4090 慢约 25-35%（README 数字为 4090 口径）。
  4090 参考: code-gen decode MTP3 148.6 tok/s @ 81% 接受率 → 你的卡约 100-115 tok/s。
- bench 结果: `ninfer-4090d-bench/profiles/bench/<时间戳>/results.json`。

## 实测（2026-09-10 预检，旧 artifact rev 3526913004）

SHA-256 校验通过（`eec39564...14bf3e`，18,210,531,328 字节）。binder 接受，
引擎 **14.0 秒** ready（权重 16.9 GiB），启动日志实测：

```text
host state pinned | 2.29 GiB | 431 ms
host KV pinned    | 32.0 GiB | 8.2s
CUDA graphs ready | 1.2s
engine capacity  kv_capacity_mode=auto kv_capacity_tokens=680256
                 kv_page_groups=10629 / kv_max_page_groups=12288
                 runtime_reservation_bytes=14253122816  (13.3 GiB)
                 available_after_weights_bytes=15327756288 (14.3 GiB)
                 planned_slack_bytes=1074633472 (auto 的 1 GiB headroom)
engine context_cache  active_lanes=3 device_state_slots=3 host_state_slots=16
                      host_kv_bytes=34359738368 private_continuations=8
                      shared_prefixes=4 long_anchors_per_continuation=4
engine state_pools    text_kv_bytes=11841896448 (11.0 GiB) mtp_kv_bytes=740327424
                      gdn_state_bytes=923664384 (0.86 GiB) workspace=433324032
hardware_class="nvidia-geforce-rtx-4080-super-sm89"
```

即 **3 路 + vision + 32 GiB 主机 KV 下，共享池实测 680,256 tokens**
（你原 vLLM 池是 232,941 → 约 2.9x），`auto` 只留了 1 GiB slack，没有浪费。

## 本地补丁：timings 增加 `ttft_ms`

fork 的 README 宣称响应带 `ttft_ms`，但源码里**从未发出该字段**（`grep ttft_ms src/` 为空；
`ttft_seconds` 只存在于内部 metrics/日志）。客户端因此显示"首 token 时间不可用"。

已在 `src/serve/openai_chat_response.cpp` 补上（1 处字段 + 1 处 JSON 键 + 2 处取值）：
- 非流式 / 流式最终 chunk：`ttft_ms = outcome.metrics.ttft_seconds * 1000`
  （引擎真实口径 = prepare + 首 token 时间，含排队与 prefill）
- 流式中间 chunk：`ttft_ms = prompt_elapsed_ms`（近似值，该路径无首 token 观测）

重新编译：`cd ninfer-4090 && cmake --build build -j`（增量约 10 秒），然后重启服务。

## 实测性能（2026-09-10，RTX 4080S 32GB，E8 4-bit KV，3 路，MTP3，vision，冷 prompt）

| prompt_n | E8 prefill tok/s | FP8 prefill tok/s | E8 decode tok/s | FP8 decode tok/s |
|---:|---:|---:|---:|---:|
| 1,054 | 1,142 | 1,143 | 99.7 | 95.3 |
| 4,011 | 1,286 | 1,289 | 93.2 | 89.2 |
| 15,888 | 1,307 | 1,320 | 95.3 | 101.0 |
| 39,631 | 1,223 | 1,243 | 85.8 | 97.0 |
| 79,524 | 1,096 | 1,122 | 87.8 | 85.1 |

**结论：prefill 差异 ≤2%（噪声内），fp8 不提升速度，却把池子从 680,256 砍到 358,592
tokens——所以保持 E8。** MTP 接受率 65-85%，decode 约 85-100 tok/s。

注意：客户端看到的 `prompt_per_second` 在**前缀缓存命中**时会很低（`prompt_n` 只算未命中
token，分母含固定开销），不代表真实 prefill 能力。真实值看上面的冷 prompt 数据，或用
`timings.prompt_n / timings.prompt_ms`。

## 备注

- 模型 SHA-256: `0634abb07024221de141456cf04a42ab74b18bc38e1b781c6eb2e062a467eec3`
  （官方 model card；ninfer-4090 README 写的 16.96 GiB 是过期数字，实际 19.03 GiB）
- 系统改动（为修 glibc 2.43 链接问题）: 装了 CUDA 13.2 组件包到
  `/usr/local/cuda-13.2`（未动驱动 595.84，`/usr/local/cuda` 软链现指向 13.2，
  llama-server 的 cublas rpath 仍钉在 13.1，不受影响）；
  补了 `/usr/lib/x86_64-linux-gnu/libstdc++.so -> libstdc++.so.6` 软链。
