# NInfer-YaRN — 长上下文扩展部署说明（RTX 4080S 32 GB）

本仓库是**一份从 `ninfer-4090/` 复制出来、加装了 YaRN 线性位置缩放的独立 NInfer 构建**：
部署脚本、日志、PID 全部落在本仓库 `deploy/` 下，二进制在本仓库 `build/` 编译，
`ninfer-4090/` 原版未被改动。

| 项 | 位置 |
|---|---|
| 引擎仓库（YaRN 版） | 本仓库（`ninfer-4090-yarn/`） |
| 编译产物 | `build/apps/{ninfer,ninfer-serve,ninfer-perplexity}` |
| 模型（与原版共用，未复制） | `../ninfer-4090/models/qwen3_8_27b.ninfer` |
| kernel 数值测试 | `tests/ops/test_scale_positions_yarn.cu` |
| 部署脚本 | `deploy/ninfer_service.sh`（start/stop/status）/ `deploy/run_bench.sh` / `deploy/test_yarn_niah.py` |

## 当前生效配置（推荐，已实测）

```bash
bash deploy/ninfer_service.sh start    # 脚本默认即此配置, 无需环境变量
```

| 参数 | 值 | 说明 |
|---|---|---|
| `--max-context` | 658176 | 每会话逻辑上下文上限（4 路并发下显存允许的最大值） |
| `--rope-scaling-factor` | 2.51 | YaRN 线性缩放因子（= 658176/262144） |
| `--rope-scaling-original-context` | 262144 | 原生训练长度，从此之上开始压缩 |
| `--kv-capacity` | auto | 引擎按可用显存算满池子 |
| `--max-concurrency` | 4 | 4 路并发；池子按需共享、不均分 |
| `--kv-dtype` | rk4v4-e8 | E8 4-bit KV |
| `--vision` | 开 | 多模态 |

启动日志实测：

```text
kv_capacity_tokens=658176   kv_page_groups=10284/41136
runtime_reservation_bytes=14253857536   (13.27 GiB)
available_after_weights_bytes=15327756288 (14.27 GiB)
planned_slack_bytes=1073898752          (1.07 GiB ≈ 引擎自身的 1 GiB headroom)
active_lanes=4   device_state_slots=4
```

`kv_max_page_groups=41136 = 4 × 10284` 正好印证了 `maximum_pages = max_concurrency × logical_pages`。

`nvidia-smi`：**30,966 / 32,760 MiB 已用**，free 1,266 MiB。

### 单会话 vs 多并发：两个可选档

| 档位 | 启动命令 | 池子 | 含义 |
|---|---|---|---|
| **4 路并发（默认）** | `bash deploy/ninfer_service.sh start` | 658,176 | 单会话活跃时能用满 658,176；4 路同时活跃各约 164,544 |
| 最大化单会话 | `NINFER_CONCURRENCY=1 NINFER_MAX_CTX=724000 NINFER_YARN_FACTOR=2.76 bash deploy/ninfer_service.sh start` | 724,032 | 只有一个会话，可用上下文最大 |

档位的选择逻辑见下文「并发度 vs 池子容量」：并发本身不显著缩小池子（每路只多占约 0.38 GiB 的
device state image），但 `max_context` 的可设上限等于池子大小，所以 4 路的上限比单路低 9%。

### 为什么 `max_context` 决定了池子

引擎的 `auto` 容量策略里 `minimum_pages = max(logical_pages, max_concurrency)`、
`maximum_pages = max_concurrency × logical_pages`。单路时两者相等，**池子被锁死为 `max_context`**；
多路时 `maximum_pages` 放大，池子转为由显存封顶——此时 `max_context` 是每会话的逻辑天花板，
不再决定池子大小（但它的可设上限仍等于显存允许的池子）。

按启动数据反推的容量模型（用于选值）：

```text
每 token 可变开销 = 18,496 B   (text_kv 17,408 + mtp_kv 1,088)
固定开销          = 853,288,192 B (workspace + gdn_state + replay + graphs)
每多一路           ≈ 22,100 token ≈ 0.38 GiB (每路一张 device state image)
```

### 因子选择规律

对比 gzenz fork 的取值可以看出一条简单规律——**factor ≈ 目标上下文 / 262144**：

| 目标 | 其 factor | 262144 × factor | 
|---|---|---|
| 500k | 1.91 | 500,695 |
| 555k | 2.12 | 555,745 |
| 600k | 2.30 | 602,931 |
| **658k（本机默认）** | **2.51** | **658,281** |
| 724k（单路档） | 2.76 | 723,517 |

## 实测

### YaRN 超原生上下文 needle 检索（自建，4/4 通过）

内置 bench 的 NIAH fixture 最大只到 256k（= 原生 262144 边界），**无法验证扩展本身**。
`test_yarn_niah.py` 把 fixture 的 haystack 继续重复扩展（与原 fixture 的构造方式一致：同一语料
重复），在其中植入一个**全新唯一**的 needle，然后只问这个 needle——所以答对必然意味着模型真的
检索到了那个深度，不可能靠先验猜中。

```
     目标    深度    实际prompt      TTFT   prefill  decode    MTP  结果
------------------------------------------------------------------------------
   300K   10%     296,305      426s      696     61    93%  ✓
   300K   50%     296,305      426s      696     61    93%  ✓
   500K   50%     493,783      946s      522     48    93%  ✓
   700K   10%     691,317     1653s      418     40    93%  ✓
------------------------------------------------------------------------------
  通过 4/4
```

四个用例的回答全部是精确的 `VERMILION=918273; COLOR=MAGENTA`。

关键点：700K 用例的 needle 埋在约 10% 深度（≈6.9 万 token 处），距问题约 62 万 token 远，
仍然被精确取回。这是 YaRN 位置缩放在真实硬件上生效的直接证据。

### 当前默认档（N=4 / 658,176 / factor 2.51）的复验

改档位后因子随之从 2.76 调到 2.51，因此重新验证了一遍（YaRN 的因子不能照搬，见下文规律）：

```
     目标    深度    实际prompt      TTFT   prefill  decode    MTP  结果
   500K   10%     493,781      943s      524     48    93%  ✓
   620K   50%     612,285     1348s      454     43    93%  ✓
```

两例同样精确命中 `VERMILION=918273; COLOR=MAGENTA`。620K 已接近 658,176 的逻辑上限。

### 并发实测：1 / 2 / 4 路下的 decode 吞吐

**先明确一个容易误读的点**：上表里 `ctx658176 / N4` 那两行的 decode `48`/`43` 是**单流**值
（当时只发了一个请求）。`N4` 是服务端配置的并发**上限**，不是当时跑了 4 路。下面才是真正的
并发对照。

测法：每路要求长输出（2048 token）以进入稳态；**主口径取服务端 5 秒吞吐日志中
`batch` 恰为整数 N 且 `running == N` 的窗口**——此时全部 N 路都在解码、无 prefill 混入。
客户端各路速率相加作为交叉验证（偏差 <8%）。

| 上下文 | 并发 1 | 并发 2 | 并发 4 |
|---|---:|---:|---:|
| **聚合** @256 token | 74.2 | **101.7**（1.37×） | **127.2**（1.71×） |
| **聚合** @8K token | 75.5 | **107.7**（1.37×） | **133.7**（1.69×） |
| **每路** @256 token | 74.2 | 50.8 | 31.8 |
| **每路** @8K token | 75.5 | 53.9 | 33.4 |

（单位 tok/s；每格取样 3–11 个稳态窗口，4 路档 11 个窗口的 min/max 是 119–133 与 128–140）

结论：

- **批处理有增益但远非线性**：2 路约 **1.37×**，4 路约 **1.70×**，不是 2×/4×。
- **缩放比几乎与上下文长度无关**（256 与 8K 都是 1.37× / 1.70×）——说明瓶颈不在 KV 读取，
  而在**权重读取 + 批处理开销**。
- **4 路聚合 127–134 tok/s，达不到 160+。**
- 每路速率随并发上升而下降：74 → 51–54 → 32–33 tok/s。所以「开 4 路」不等于「单路快 4 倍」，
  也不等于「每路还是单路那么快」。
- 值得注意：**单流始终是最快单流**（约 74–79 tok/s），且 256 与 8K 差别很小——这与「decode
  对上下文长度不敏感」的既有结论一致（64 层里 48 层是 GDN 线性注意力，decode 时 O(1)）。

**踩过的坑**（记录以免重犯）：

1. 第一版测试只让模型输出 79–206 token，太短进不了稳态，5 秒窗口混进 prefill 与收尾，
   算出来的「decode 速率」没有意义（甚至出现 8.4 tok/s 这种被拖尾拉低的数）。
2. 客户端按墙钟跨度算 `总token / span` 会包含 prefill，4 路 @8K 因此被低估到 101.8（真实 133.7）。
   **服务端口径才是权威。**
3. decode 速率还受 MTP 接受率影响（速率 = 前向次数/s × 每轮 token 数）。本次各轮接受率
   41–52%，属可比范围；若两轮接受率差异大，直接比 tok/s 会失真。

原始数据：`concurrency_decode_results.json`；逐路明细：`conc_decode2.log`（1/4 路）
与 `conc_decode_2way.log`（2 路）。

### 原始证据与配置归属

每次 `test_yarn_niah.py` 的结果都会追加到 `yarn_niah_results.jsonl`，**每条记录带 `config` 字段**
（`context_window` / `max_concurrency` / `yarn_factor` / `yarn_original_context` / `kv_dtype`，
由脚本从运行中的进程命令行与 `/v1/models` 现场抓取）。运行日志分别是
`yarn_niah_run.log`（724K 档 4 例）与 `yarn_niah_658k.log`（658K 档 2 例）。

> 教训：最初的结果记录**不含配置**，而脚本的结果文件名是固定的，所以换档后两轮数据混进了同一个
> 文件、无法区分归属。现已修复（脚本现场抓配置），历史 6 条也补上了 `config` 字段，并标注
> `config_source: "reconstructed-from-run-log"` 以区分「事后重建」与「运行时采集」。原始文件
> 保留为 `yarn_niah_results.jsonl.pre-config`。

汇总（全部 6 例，均为精确命中 `VERMILION=918273; COLOR=MAGENTA`）：

> ⚠️ **这 6 例的实际并发都是 1。** 表里的 `N1`/`N4` 是**服务端配置的并发上限**
> （`--max-concurrency`），不是测试时的并发路数——每轮都是**一次只发一个请求**，串行跑完再跑下一个。
> 因此 `decode` 列是**单流 decode 速率**，不是「4 路各自 48」也不是「聚合 4×48」。
> 4 路并发时的聚合速率是另一个量，实测见下节。

| 配置（服务端上限） | 实际并发 | 目标 | 深度 | 实际 prompt | TTFT | prefill | decode | MTP |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ctx724000 / f2.76 / N1 | 1 | 300K | 10% | 296,305 | 426 s | 696 | 61 | 93% |
| ctx724000 / f2.76 / N1 | 1 | 300K | 50% | 296,305 | 426 s | 696 | 61 | 93% |
| ctx724000 / f2.76 / N1 | 1 | 500K | 50% | 493,783 | 946 s | 522 | 48 | 93% |
| ctx724000 / f2.76 / N1 | 1 | 700K | 10% | 691,317 | 1653 s | 418 | 40 | 93% |
| ctx658176 / f2.51 / N4 | 1 | 500K | 10% | 493,781 | 943 s | 524 | 48 | 93% |
| ctx658176 / f2.51 / N4 | 1 | 620K | 50% | 612,285 | 1348 s | 454 | 43 | 93% |

单流 decode 随上下文长度单调下降（同一份配置下）：

```
  8K   -> 123 tok/s   (bench, 单流)
260K   -> 102 tok/s
296K   ->  61 tok/s      <- 注意 260K->296K 之间有一个明显台阶
494K   ->  48 tok/s
612K   ->  43 tok/s
691K   ->  40 tok/s
```

原因：模型 64 层里 48 层是 GDN 线性注意力（decode 时 O(1)），只有 16 层全注意力付 O(n) 代价，
所以衰减不是线性的；但 KV 读取量随上下文增长，深度越大越明显。

### 性能随上下文的衰减（单路，E8 4-bit KV，MTP3，vision）

| prompt | TTFT | prefill | decode |
|---:|---:|---:|---:|
| 296,305 | 426 s | 696 tok/s | 61 tok/s |
| 493,783 | 946 s | 522 tok/s | 48 tok/s |
| 691,317 | 1,653 s | 418 tok/s | 40 tok/s |

prefill 从 696 降到 418 tok/s（−40%），decode 从 61 降到 40 tok/s（−34%）。TTFT 近似线性：
约 0.42 s / 1k token（296K 档）到 2.4 s / 1k token（691K 档）——注意 TTFT 随长度增长快于线性，
因为 chunked prefill 在更深上下文下注意力开销更大。

MTP 接受率在全部档位稳定在 **93%**，与上下文长度无关。

> **MTP 接受率本身就是修复 #3 的证据。** 若 MTP 的 AR 位置未缩放而 target 位置已缩放，
> 69 万 token 下 draft 与 verifier 会相差数万位置、接受率必然崩到接近 0。实测 93%
> （且 8K 原生档为 100%）说明两条路径的位置是一致的。

### 内置 NIAH（8k/64k/128k/256k，冷启动档）

| fixture | prompt | TTFT | prefill | decode | MTP |
|---:|---:|---:|---:|---:|---:|
| 8k | 7,680 | 5.8 s | 1327 tok/s | 123 tok/s | 100% |
| 64k | 64,512 | 56.3 s | 1145 tok/s | 109 tok/s | 100% |
| 128k | 130,048 | 134.3 s | 968 tok/s | 97 tok/s | 100% |
| 256k | 260,096 | 351.1 s | 741 tok/s | 80 tok/s | 100% |

内置 bench 只测性能、**不校验答案**，所以 256K 档的答案单独验证过：

```
prompt 260,096 → 回答 'ORCHID=493817; COLOR=COBALT'   精确命中
```

### 内置 bench 汇总（结果在 `ninfer-4090d-bench/profiles/bench/`）

| suite | decode tok/s | MTP 接受 | tok/round |
|---|---:|---:|---:|
| niah | 102.3 ± 16.5 | 100% | 2.0 |
| scenario / code | 103.6 ± 3.7 | 77.1% | 1.8 |
| scenario / structured | 114.3 ± 7.2 | 88.4% | 1.9 |
| scenario / translation | 101.5 ± 7.0 | 73.9% | 1.7 |
| scenario / story | 64.7 ± 5.2 | 35.5% | 1.4 |
| long_decode / reasoning | 79.8 ± 14.1 | 51.5% | 1.5 |

story 与 reasoning 的接受率明显偏低（35.5% / 51.5%）——创意文本与长链推理的下一 token 本就
难预测，这是 MTP 的固有特性，与 YaRN 无关（原生上下文下同样如此）。

### 多模态 + YaRN（修复 #6 的回归测试）

`test_yarn_vision.py`，2/2 通过：

| 用例 | prompt | TTFT | 回答 |
|---|---:|---:|---|
| image_chart | 428 | 0.60 s | `NIFER VISION 731；3；左侧` |
| image_natural | 417 | 0.60 s | `…邮箱上的数字是 24，太阳位于画面的右侧` |

修复前这两个请求都会以 `invalid_argument` 直接失败（`[len,3]` MRoPE 矩阵不满足 Op 的
`ne[1]==1` 校验）。注意端点只接受 `image_url`（data: URI 或 http(s)），不接受本地路径；
fixture 里的内部格式需要转换。

### 长上下文前缀复用（多轮一致性）

对 26 万 token 的文档发起追问：

```
prompt=260,143  cached=260,112  TTFT=0.6s  (冷启动同长度需 350s)
回答: '493817'   正确
```

这是 prefill 与 decode 位置一致性的最强证据：prefill 阶段用 device 内核写入的 26 万 token KV
被完整复用，后续 decode 走 host 侧缩放，答案仍然正确。两条路径若不一致，复用的 KV 会不自洽。

## 移植内容与修复的缺陷

从 gzenz/ninfer（5090, sm_120a）移植到 sm_89。改动 18 个源文件 / 约 260 行。

**新增 Op**：`scale_positions_yarn`（`include/ninfer/ops/position.h`、`src/ops/kernel/position.cuh`、
`src/ops/launcher/position.{h,cu}`、`src/ops/wrapper/position.cpp`）。

**配置链路**：CLI → `ServeOptions` → `EngineOptions` → `SequencePlanImpl` → `ProgramImplCore`
→ `ExecutionCore` → `TextContext`。

**接入点**：prefill 走 device kernel（`text_context_impl.h`），decode 的四条路径
（ordinary / MTP AR / MTP target / DFlash）走 host 侧 `scale_rope_pos`。

移植过程中发现并修复了 7 个缺陷，其中 4 个是移植引入的、3 个源自上游：

| # | 缺陷 | 后果 | 修复 |
|---|---|---|---|
| 1 | 纯文本 prefill 时 `rope_positions` 与 cache positions 是**同一个 tensor**，原地缩放会连 KV 寻址位置一起改 | 缓存错乱 | 缩放开启时强制分配独立 buffer，用零 delta 的 offset 做拷贝 |
| 2 | 上游 kernel 把 `original_context` 折进 float 加法再 `+0.5`，该量级下 float 间距 0.015625 会吃掉 0.5 并进位 | 10 万采样中 380 个位置差 1 | 舍入项在小量级计算、商用 double；host 侧同改，保证 device/host 逐位一致 |
| 3 | MTP 的 AR 位置由 `mtp_prepare_next_round_kernel` 从**未缩放**的 frontier 推导，而 target 位置是缩放后的 | 超 262144 后 draft 与 verifier 位置差数万，MTP 接受率崩塌 | 在 round kernel 后对 AR rope frame 整块缩放 |
| 4 | DFlash 的 `proposal_positions` **同时**用作 RoPE 位置和 cache positions，无法只缩放其一 | — | 启动守卫：YaRN + DFlash 组合直接拒绝（该组合对本 27B 目标本就不支持） |
| 5 | `ExecutionCore` 有 **7 处**嵌入聚合构造点，字段带默认值 → 遗漏时静默取 `factor=1.0` | **prefill 缩放静默失效** | 改为构造函数初始化（删掉默认值），遗漏变成编译错误 |
| 6 | multimodal 时 `rope_positions` 是 `[len,3]` MRoPE 矩阵，而 Op 的校验要求 `ne[1]==1` | **带图/视频的请求抛异常** | 调用点用 `view()` 展平（两种布局都必然连续） |
| 7 | DFlash representative 的 `target_rope_positions` 未包装缩放（另两处 representative 包了） | 不可达（被 #4 守卫拦截），仅一致性问题 | 统一包装 |

缺陷 5 和 6 由子代理 review 发现（其会话记录在 `~/.dsh/sessions/` 下，zstd 压缩 JSONL）；
其余由内核单元测试、启动日志与自查发现。

验证手段：

- `tests/ops/test_scale_positions_yarn.cu` — 真卡数值测试（含 10 万值网格切分、边界值、
  `factor=1.0` 退化、不同 `original_context`），全部通过
- CLI 参数校验：`0.5` / `33` / 缺值 / 非数字均被拒绝并给出具体错误
- 上述 4 个 needle 用例

## 并发度 vs 池子容量（实测）

用 `sweep_concurrency.sh` 实测（换端口 30001，不打扰生产）。引擎的容量规则见
`src/targets/qwen3_6/impl/runtime/layouts_impl.h`：

```text
logical_pages = page_count(max_context)
minimum_pages = max(logical_pages, max_concurrency)
maximum_pages = max_concurrency * logical_pages
auto 在 [minimum, maximum] 内选显存允许的最大值
```

两个要点：**`minimum_pages` 只跟 `max_context` 走、与并发无关**；`maximum_pages` 随并发线性放大。
所以并发越高 auto 的容量上限越高（仍被显存封顶），而 `max_context` 是每会话的逻辑天花板。

固定 `max_context=300000` 扫描：

| N | 池子 | 说明 |
|---:|---:|---|
| 1 | 300,032 | 被 max_context 锁死（min==max） |
| 2 | 600,064 | = N × max_context，尚未触到显存上限 |
| 3 | 680,256 | 显存封顶 |
| 4 | 658,176 | 显存封顶 |
| 8 | 569,728 | 显存封顶 |

每多一路的代价：N=3→4 差 22,080 token；N=4→8 每路 22,112 token。按 18,496 B/token 即
**每路约 0.38 GiB**——正是每路一张 device state image（`device_state_slots` 恰好等于 N）。
这与原版 README 记的「每多一路 +0.39 GiB」吻合；N=3 的 680,256 也与原版 README 实测值
**逐位一致**，互为交叉验证。

各并发度下的上水：

| N | 最大池子 | 最大单会话 context | 保证 2 路并发 | 3 路 | N 路 |
|---:|---:|---:|---:|---:|---:|
| 1 | 724,032 | 724,032 | — | — | — |
| 2 | ~702,000 | ~702,000 | ~351,000 | — | — |
| 3 | 680,256 | 680,256 | 340,128 | 226,752 | 226,752 |
| 4 | 658,176 | 658,176 | 329,088 | 219,392 | 164,544 |
| 8 | 569,728 | 569,728 | 284,864 | 189,909 | 71,216 |

结论：

- **池子确实随并发略降**，但原因不是并发本身，而是每路要占一张 device state image（~0.38 GiB）。
  1→3 路只降 6%（724,032 → 680,256）。
- **单会话 `--max-context` 与并发无关**，由你自己设；但它的**可设上限**等于显存允许的池子大小，
  因此随并发下降。设超了启动直接失败，报
  `minimum Engine runtime reservation requires N bytes in addition to 1 GiB`。
- **池子按需共享、不均分**：即使 N=8，只有一个会话活跃时它仍能用满 569,728。多路同时活跃才互相挤压。
- 想保证 **N 路各自都有 C 的上下文**，需 `N × C ≤ pool(N)`，上表最后几列即该配额。
- 两个硬性限制：`max_concurrency` 上限 8（代码里的 `kMaximumConcurrency`）；
  `max_context` 不得超过显存允许值，否则启动失败。

本部署默认 **N=4 / 658,176**（4 路并发档）。若要 3 路各保证 22 万 token：
`NINFER_CONCURRENCY=3 NINFER_MAX_CTX=226000 NINFER_YARN_FACTOR=1.86 bash deploy/ninfer_service.sh start`。
要最大化单会话上下文则用单路档（见上表）。

## 缺陷归属：移植引入 vs 上游既有

对照 gzenz 仓库的对应源码逐一定性：

| # | 缺陷 | 归属 | 依据 |
|---|---|---|---|
| 1 | prefill 时 rope/cache 位置别名 | **移植遗漏** | gzenz 的 `else if` 条件写作 `rope_delta_ != 0 \|\| rope_scaling_factor_ != 1.0F`，我最初漏了后半句；最终修复与他们的代码逐字一致 |
| 2 | float 精度 off-by-one | **上游既有** | gzenz 的 `position.cuh` 就是那行 float 公式，逐字相同 |
| 3 | MTP AR 位置未缩放 | **移植遗漏** | gzenz 的 `mtp_impl.h` 有逐步 `slice + scale_positions_yarn` 块，我漏了；我的整帧 `view` 做法与它功能等价 |
| 4 | DFlash + YaRN 未处理 | **上游未实现**（我加了守卫） | gzenz 的 `program_impl.h` 中没有任何 DFlash rope 缩放代码 |
| 5 | `ExecutionCore` 字段漏传 7 处 | **上游既有** | gzenz 有**完全相同的 7 处** raw 构造点（以 `proposal_head}` 结尾）+ 相同的默认值 `= 1.0F` |
| 6 | MRoPE `[len,3]` 触发校验异常 | **上游既有** | 相同的 `text_prefill_roots` 形状（`matrix(tokens, rope_axes)`）+ 相同的 `require_i32_vector(ne[1]==1)` + 相同的无条件调用 |
| 7 | DFlash representative 未包装 | 移植引入（一致性） | 不可达，仅可读性 |

即 **#1、#3 是移植遗漏**（gzenz 的代码是对的，我没抄全）；**#2、#5、#6 是上游既有缺陷**，
移植继承了它们并已修掉；#4 是上游未实现的组合。

### 关于 #5 的一个要紧推论

gzenz 的 `configure_text_card` 会无条件调 `card.set_rope_scaling(execution.rope_scaling_factor,
...)`，但 7 处 `ExecutionCore` 构造点都不传该字段，全部落到默认值 `1.0F`；而 prefill 的缩放由
`TextContext::rope_scaling_factor_` 把关（`if (rope_scaling_factor_ != 1.0F)`）。因此：

> **gzenz 仓库里 prefill 的 device 侧缩放从未真正执行过**——只有 decode/host 侧的位置被缩放。
> 他们的 555K/600K 质量数据是在「prefill 不缩放 + decode 缩放」这种不自洽状态下取得的。

本实现把该字段变成构造函数必填参数（漏传即编译失败），因此 prefill 与 decode **都**缩放且一致。
所有长上下文测试（含 26 万 token 前缀复用）都是在「两侧都缩放」下通过的。

## 用法

```bash
cd /root/ai/large_models/_ninfer_repos/ninfer-4090-yarn

# 启动（默认即推荐配置：658K / factor 2.51 / 4 路 / vision）
bash deploy/ninfer_service.sh start

# 最大化单会话上下文（放弃并发）
NINFER_CONCURRENCY=1 NINFER_MAX_CTX=724000 NINFER_YARN_FACTOR=2.76 bash deploy/ninfer_service.sh start

# 想回到原生 262144 行为（用于回归对比）
NINFER_MAX_CTX=262144 NINFER_YARN_FACTOR=1.0 bash deploy/ninfer_service.sh start

# 不需要多模态可省约 0.6 GiB，还能再抬高 max_context（单路档）
NINFER_CONCURRENCY=1 NINFER_NO_VISION=1 NINFER_MAX_CTX=780000 NINFER_YARN_FACTOR=2.98 bash deploy/ninfer_service.sh start

# 超原生上下文检索验证（这是 YaRN 的核心验收项）
python3 deploy/test_yarn_niah.py 300000 50
python3 deploy/test_yarn_niah.py 700000 10

# 多模态回归（验证 MRoPE 的 [len,3] 路径没被破坏）
python3 deploy/test_yarn_vision.py

# 并发度 ↔ 池子容量扫描（换端口 30001，不打扰生产）
bash deploy/sweep_concurrency.sh

# 性能 bench（已内置 1800s 单请求超时，256K fixture 需要 6-10 分钟 prefill）
bash deploy/run_bench.sh niah
bash deploy/run_bench.sh scenario
bash deploy/run_bench.sh decode

# 停止
bash deploy/ninfer_service.sh stop
```

环境变量一览（全部可选）：`NINFER_PORT`、`NINFER_MAX_CTX`、`NINFER_YARN_FACTOR`、
`NINFER_YARN_ORIG`、`NINFER_CONCURRENCY`、`NINFER_KV_DTYPE`、`NINFER_NO_VISION`、`NINFER_MODEL`、
`NINFER_DUMP_REQUESTS`（请求体落盘目录，默认 `./reqdump` 每次启动自动开启，设为 0 或空关闭；诊断前缀缓存用）、
`NINFER_DUMP_REQUESTS_LIMIT`（最多保留几个 body，启动脚本默认 2000（二进制内置 100），0=不限）、
`NINFER_DUMP_REQUESTS_MAX_AGE_HOURS`（超过多少小时删除，默认 24，0=不限）、
`NINFER_REUSE_DIAG`（前缀复用诊断，启动脚本默认 1 自动开启，设为 0 或空关闭；每个请求把前缀索引里每个 checkpoint 的拒绝原因逐条打到日志，只读不改行为，用于区分 0% 命中是"字节变了"还是"长上下文规划器超时放弃"）、
`NINFER_AUTO_LONG_ANCHORS` / `NINFER_MAX_LONG_ANCHORS`（长锚点窗口与保留上限，默认 32，见前缀缓存一节）、
`NINFER_MAX_SHARED_PREFIXES`（共享前缀目录容量，默认 4）、
`NINFER_LOG_LEVEL`（日志级别，启动脚本默认 debug，比 info 只多启动内存台账/每 prompt 上下文成本等 4 个日志点）。

## 前缀缓存：跨会话复用与三次连续会话交替 miss

**现象**：DSH 每个新会话要发约 9.6K 的 system prompt + 工具定义，新会话首轮 `cache 0 (0.0%)`、
TTFT ≈ 8s；而同会话追问 `cache 99.9%`、TTFT ≈ 0.15s。

**已确认**：**引擎侧支持跨会话复用**，只要前缀逐字节一致，且服务自上次启动后见过该前缀。
**已查清**：远程客户端那三次「交替 miss」与请求体无关，是引擎侧状态造成的——见下文。

### 引擎侧没问题：跨会话复用确实可用

用两段**全新会话**（system+tools 逐字节相同、只有用户问题不同）实测：

| 请求 | prompt | cache | TTFT |
|---|---:|---:|---:|
| 新会话 A | 12,202 | 0 (0.0%) | 9.31 s |
| 新会话 B（同前缀） | 12,202 | **12,187 (99.9%)** | **0.15 s** |

换成 DSH 真实的 system+tools（26 个工具、8.1K token）复现，同样成立：新会话 B/C 命中
**99.8% / 99.9%**，TTFT 降到 0.08–0.15 s。

### 一个已实测现象：换模型名会让**新**前缀 miss（但不是"差异必然致命"的证明）

扫描本机 30 个 DSH 会话的 `request/header`（其中含完整 system 与 tools）：

```text
tools 版本数        : 1     ← 26 个工具，30 个会话逐字节相同
system prompt 版本数: 5     ← 系统提示词有 5 个不同版本
```

5 个版本**只在同一处不同**——那句模型自我描述：

```text
基准 (19 会话): "You are a coding agent powered by the myai model. …"
变体 ( 6 会话): "…powered by the deepseek-v4-flash-vision-exp model…"
变体 ( 2 会话): "…powered by the qwen3.8-flash model…"
变体 ( 1 会话): "…powered by the qwen38-27b-wq model…"
```

当时的对比（system 只改这一句，其余完全相同）：

| system 版本 | prompt | cache | TTFT |
|---|---:|---:|---:|
| `myai` | 8,122 | 8,115 (99.9%) | 0.10 s |
| `myai`（另一新会话） | 8,122 | 8,115 (99.9%) | 0.08 s |
| `deepseek-v4-flash-vision-exp` | 8,128 | **0 (0.0%)** | **6.15 s** |
| 换回 `myai` | 8,122 | 8,115 (99.9%) | 0.08 s |
| `qwen3.8-flash` | 8,126 | **0 (0.0%)** | **6.18 s** |

**这张表不能证明"6 个 token 的差异必然导致 0 命中"。**两种解释在数据上不可分：(a) 差异本身
致命；(b) 这两个变体前缀是**自上次服务启动以来第一次出现**，任何首次出现的前缀都必然 miss。
要区分只能把同一个变体**连续发两次**，看第二次是否命中——当时没有做这个对照。

能把结论钉死的是另一条更硬的事实：**每次重启服务都会清空内存里的 checkpoint 缓存**（`deploy/ninfer_serve.log` 里 7 次 YaRN 重启都带着新的启动行）。所以「A 会话没命中、B 会话命中」如果
跨越了一次重启，就与提示词差异无关。

### 已查清：历史会话（回一句 OK）为什么一条都没命中

现象：在 4–5 个历史会话里各回一句 `ok`，prompt 9,639–9,705，**全部** `cache 0`、TTFT 7.6–7.8 s。
按理说这些会话的主要内容都是同一份系统提示词，应该能复用。

**请求体比对（逐字节）**：4 个会话的 `developer` 消息（7,023 字符、sha256 `5e2cc767…`）和 27 个
tool 定义**完全相同**，只有 message 1 的尾号不同（`这又是个测试会话5/6/8/9`），message 2/3/4 也相同。
所以唯一对所有会话都一致的前缀就是「系统提示词结束处」。

**真正的原因：我自己把长锚点窗口设小了。** 引擎对每个 prompt 自动放置私有长锚点，位置是
**从末尾往前数的最后 N 个消息边界**（`--auto-long-anchors N`，我原先设 4），并且每条 continuation
最多保留 `--max-long-anchors-per-continuation` 个。窗口是从结尾往前数的，所以当历史消息较多时，
唯一对所有会话都相同的边界——`developer` 之后（= 8,669 token，即系统提示词末尾）——落在了窗口
**外面**，一条检查点都没建，于是每个会话都只能从头 prefill。

引擎检查点表（`/slots`）直接印证了这点：

```text
N=4, 7 条消息:  最浅检查点 = 9539                  ← developer 边界(8669) 不在
N=8, 10 条消息: 最浅检查点 = 8681                  ← 仍不含 8669
N=32, 21 条消息:最浅检查点 = 8669                  ← 含
```

**实测覆盖规律：`N ≥ 消息数 − 1` 才能包住 developer 边界。**

| `--auto-long-anchors` | 5 条消息 | 7 条消息 | 10 条消息 | 21 条消息 |
|---:|---|---|---|---|
| 4（原值） | ✅ 命中 8,669 | ❌ 0 / 8.4 s | ❌ 0 | ❌ 0 |
| 8 | ✅ | ✅ 1.4 s | ❌ 0 | ❌ 0 |
| 32（现值） | ✅ | ✅ 1.4 s | ✅ 1.7 s | ✅ 2.6 s |

**已改的默认值**：`--auto-long-anchors 32`、`--max-long-anchors-per-continuation 32`
（`deploy/ninfer_service.sh`），可用 `NINFER_AUTO_LONG_ANCHORS` / `NINFER_MAX_LONG_ANCHORS` 调。

**代价实测为 0**：锚点复用已预留的 snapshot arena，进程 RSS 36.57 GiB、显存 30,966 MiB 在
N=4 / 8 / 32 / 64 下**完全相同**（同一 21 消息负载下逐项对比），所以默认给足更划算。

**复刻你的场景验证**（空引擎、5 个只差 msg1 尾号的 7 消息会话）：

```text
会话31 cache 0     TTFT 7.76s   ← 引擎刚重启, 缓存为空
会话32 cache 8,669 (89.9%)  TTFT 1.38s
会话33 cache 8,669 (89.9%)  TTFT 1.33s
会话34 cache 8,669 (89.9%)  TTFT 1.31s
会话35 cache 8,669 (89.9%)  TTFT 1.31s
```

**仍然存在的限制**：
- 每次重启引擎后第一个会话必然冷启（缓存为空），之后才命中。
- 超过 ~33 条消息的会话**自己**建不出 developer 边界锚点；但只要期间有任何一个较短会话跑过，
  锚点就存在，而复用是跨会话的，长会话同样能命中它。
- 引擎自带的「连续 leading System/Developer 之后」共享前缀候选本该与消息数无关地解决这个问题
  （`SharedCandidateEvidence::EngineStructural`），但它没有发布成功：把 `--max-shared-prefixes`
  从 4 提到 16 后 10 消息会话仍然 0 命中，说明卡在别处（`resource_manager.h` 里该类候选
  `pressure_capable = false`，只允许用空余终态）。**这一条未解决，仅记录。**

### 待查：远程客户端连续 3 个新会话为何交替 miss

现象（09:04，引擎 09:01 启动）：三次会话参数看起来一致（5 条消息、`thinking medium`、27 个工具、
prompt 9,542），却出现

```text
req#7  cache 0     (0.0%)  TTFT 8.2s
req#9  cache 8,669 (90.9%, long anchor)  TTFT 1.4s
req#11 cache 0     (0.0%)  TTFT 8.2s
```

加请求体落盘后（`NINFER_DUMP_REQUESTS=<目录> ./deploy/ninfer_service.sh start`，文件名
`req-<毫秒>-<序号>-<chat|responses|messages>.json`，两次请求的 body 可逐字节 diff；**默认自动清理：
最多保留 100 个、超过 24 小时删除**，可用 `NINFER_DUMP_REQUESTS_LIMIT` /
`NINFER_DUMP_REQUESTS_MAX_AGE_HOURS` 调整，0 表示不限；只删本服务写的 `req-*.json`，目录里其他文件
和子目录不动），重跑同一测试：

**1. 客户端每次「会话」其实发两个请求**，相隔约 10 ms：

```text
req-...157623-1-chat.json      673 B   ← 标题生成 (2 条消息, max 64)
req-...157624-2-chat.json   40,838 B   ← 真正的会话 (5 条消息, 27 tools)
```

所以日志里 `req#6/#7`、`#8/#9`、`#10/#11` 是**两条并行的流**（标题流 + 会话流），不是一条流里的
连续请求。

**2. 三个会话的 body 只差 message 1 的最后一个字符**：`这又是个测试会话1/2/3`。
developer 块（7,023 字符）、message 2/3/4、27 个 tool 定义、以及全部标量字段逐字节相同。
**请求体里不存在任何能让第 3 次单独 miss 的东西。**

**3. 关键反证**：09:04 那次，**标题流和会话流同时**呈现 0/hit/0——

```text
标题流: cache 0 → 128 (hit) → cache 0
会话流: cache 0 → 8,669 (hit) → cache 0
```

两条 prompt 长度相差 60 倍、内容毫无关系，不可能被同一个请求体差异同时影响。所以那是**引擎状态
事件**，不是 payload 差异。

**4. 空缓存重跑（09:11 启动引擎）：只有第一个会话冷启，之后连续命中**

```text
req#3  cache 0     (0.0%)              TTFT 8.2s   ← 引擎刚启动, 缓存为空
req#5  cache 8,669 (90.8%, long anchor) TTFT 1.4s
req#7  cache 8,669 (90.8%)             TTFT 1.4s
req#9  cache 8,669 (90.8%)             TTFT 1.5s
req#11 cache 8,669 (90.8%)             TTFT 1.4s
req#13 cache 8,669 (90.8%)             TTFT 1.4s
req#20 cache 8,669 (90.8%)             TTFT 1.4s
```

标题流同样：第一次 miss，之后连续 8 次命中 128（TTFT ≈0.17 s）。

**最可能的机制**（未证实）：跨会话命中的 8,669 前沿走的是 `PrefixReusePath::PrivateLongAnchor`
——**属于某条 private continuation 的「长锚点」**，不是 shared prefix catalog。它受
`--max-long-anchors-per-continuation 4` 约束（"a full set replaces its shallowest anchor"），
所在 continuation 表也是有界的（`--max-private-continuations 8`、device/host state slots）。
09:04 那台引擎在用户测试之前已经服务过本机几轮测试请求（含 12K 前缀的跨会话复用测试、8,552 的
DSH 形状请求），表更满；第 3 个新会话 admission 时锚点已不在可选中 → 冷 prefill → `cache 0`，
而这次冷 prefill 又重建了锚点，于是下一次再命中——正好是 0/hit/0。09:11 那台引擎在用户测试前
只服务过一条 57 token 的启动冒烟请求，有富余，所以每次都命中。

**判别实验**（会破坏当前缓存，因此没在用户测试期间做）：先用若干互不相同的新会话把 continuation
表填满，再连发 3 个只差一个字符的会话——若 0/hit/0 复现，机制即确认。

**结论性做法**：重启服务必然让下一次会话冷启（8.2 s / 0 命中），这只是 checkpoint 身份验证的正常
代价；**不要让其他客户端制造大量互不相同的会话**，也不要在用户使用期间重启。

### 怎么办

1. **用本地引擎时保持模型 id 固定**（都用 `myai`）。已确认 30 个会话中 19 个共享同一哈希。
2. **切模型（本地 ↔ 云）后第一次用新前缀会慢一轮**，这是 checkpoint 身份验证的正常代价。
   DSH 侧 `PromptContext` 才是 cache-safe 的对应物（动态内容物化成 user-role snapshot、放在
   retained history **之后**），模型名却留在了静态 system 中间；把它挪进 `PromptContext`
   可让 system+tools 前缀与模型无关。**这是一处值得改的 DSH 源码**（未改动，仅记录）。
3. 长文档场景仍建议把内容放在稳定的对话前缀里，让前缀缓存命中。
4. **不要频繁重启服务**——每次重启都会丢掉已有的跨会话前缀缓存。

复现脚本：`test_prefix_reuse.py`（跨会话复用验证）。

## 已知限制

### ⚠️ 已知环境事件：NVIDIA 用户态库被自动升级后与内核模块不匹配 → **重启即解**

**记录（2026-09-11 06:59 实际发生过一次）**

`unattended-upgrade` 在 **06:59:47** 把 `nvidia-driver-595-open` 升到 **595.91.07**，而内核里
加载的仍是 **595.84**（Linux 内核模块只有重启才会换成新版）：

```text
内核模块（已加载）: 595.84     ← /sys/module/nvidia/version
磁盘上的模块:       595.91.07  ← modinfo nvidia（包已装好，只是没生效）
用户态库:           595.91.07  ← dpkg
$ nvidia-smi
Failed to initialize NVML: Driver/library version mismatch   (exit 18)
```

**表现**：

- **已在运行的服务不受影响**——它的 GPU 上下文在升级前就建立了，映射的是旧库，实测推理正常。
- **任何新进程都无法初始化 CUDA**（`nvidia-smi`、新起的引擎进程）。所以**服务一旦重启就会失败**，
  且换 vLLM/llama.cpp/sglang 也一样——这是宿主驱动层的问题，与应用无关。

**解决办法：重启机器。** 磁盘上的模块已经是与库匹配的 595.91.07，重启就会加载它，之后
正常执行 `deploy/ninfer_service.sh` 即可。**不需要改任何系统配置，也不需要重装驱动。**

系统自己会提示：出事后 `/var/run/reboot-required` 及 `/var/run/reboot-required.pkgs` 存在。
（本次是先手动重载模块恢复的，见下；重启效果等价。）

<details>
<summary>附：不想重启时的等价操作（本次实际用了这条）</summary>

前提是磁盘上已有匹配的新模块。会短暂中断服务：

```bash
bash deploy/ninfer_service.sh stop                              # 必须释放 /dev/nvidia*（本机只有它在用）
rmmod nvidia_uvm nvidia_drm nvidia_modeset nvidia            # 卸载旧模块
modprobe nvidia nvidia_uvm nvidia_modeset nvidia_drm         # 加载新模块
nvidia-smi                                                   # 验证三层版本一致
bash deploy/ninfer_service.sh start
```

风险：重载失败则 GPU 到重启前都不可用。所以只在「服务反正已经停了 / 方便重启」时才值得做。
</details>

**启动脚本已加固**：遇到该情况会打印诊断（两个版本号 + 修复指引），而不是以一句空值的数值比较
失败。同时会检查 `/var/run/reboot-required` 并提示。**注意 `nvidia-smi` 在 NVML 失败时把错误文本
写到 stdout、退出码 18，而管道会把退出码替换成 `head` 的 0**，所以脚本校验的是「取到的是否为数字」
而非退出码或空值。

**如果想从源头避免**（本机当前**未**启用，仅作备选记录）：根因是 `unattended-upgrades` 开着，
且 `/etc/apt/apt.conf.d/50unattended-upgrades` 的 `Allowed-Origins` 里
`"${distro_id}:${distro_codename}"`（常规更新仓）未注释——不只是安全更新，NVIDIA 驱动会作为
常规更新被静默安装；而 `Automatic-Reboot` 默认为 false，于是只装包不重启。三种可选策略：

1. **冻结 GPU 关键栈**（保住其它安全更新）——在 `50unattended-upgrades` 的
   `Package-Blacklist { ... }` 里加入：
   `"nvidia-"; "libnvidia-"; "linux-modules-nvidia-"; "linux-signatures-nvidia-";
   "linux-image-"; "linux-modules-"; "linux-headers-"; "linux-generic"; "linux-firmware";`
   （内核必须一起冻：否则自动升级内核后新内核拿不到被冻的 nvidia 模块包，重启反而 GPU 不可用。
   代价：内核与驱动的安全更新转为手动，维护窗口里 `apt upgrade` 后重启。）
2. 整个停掉 `unattended-upgrades`，全手动。
3. 保留自动更新但设 `Unattended-Upgrade::Automatic-Reboot "true"` +
   `Automatic-Reboot-Time "04:00"`，让它自己在维护窗口重启。代价是会打断服务，且本服务目前
   没有开机自启（重启后需手动 `deploy/ninfer_service.sh start`）。

**为什么"复制一份 / 自己编译一份"解决不了这个问题**：本仓库确实已经是独立副本
（`ninfer-4090-yarn/` + 独立编译的二进制），但故障点在**宿主 NVIDIA 驱动**这一层——用户态
`libcuda.so` 必须与内核 `nvidia.ko` 版本匹配，这是全机共享的，vLLM/llama.cpp/sglang/NInfer
以及 `nvidia-smi` 会同时撞上，且无法私有化（Docker 也不隔离：nvidia-container-toolkit 注入的
正是宿主驱动库）。独立副本能保住的是**应用层**（改动隔离、可回退、可对比），保不住驱动层。
唯一值得单独固定的是 **CUDA toolkit**（`/usr/local/cuda-13.2`，构建时已显式指定），而它来自
NVIDIA 仓库、本就不在 `Allowed-Origins` 里，不会被自动更新。

### 其他限制

- **`--max-concurrency 1`**。单路时池子上限锁死为 `max_context` 才能吃满显存；多路共享池虽然
  仍允许单个会话用满池子，但 `auto` 的 `maximum_pages` 会放大，实际容量需要重新标定。
- **DFlash + YaRN 不支持**（见缺陷 4）。对本 27B 目标 DFlash 本身也不可用。
- **TTFT 很长**：69 万 token 要 27 分钟。这是 prefill 算力限制，不是配置问题。
- gzenz 文档里 555K/600K 的数字是 **5090 + NVFP4 KV** 口径，与本机 4080S + E8 KV 不可直接
  对比；上表数字均为本机实测。

## 备注

- 模型 SHA-256: `0634abb07024221de141456cf04a42ab74b18bc38e1b781c6eb2e062a467eec3`
- `kNativeContext = 1048576`（本机 fork 的值）只用作 `--max-context` 的硬上限，
  与 YaRN 阈值无关；本地这个 1M 上限反而比 gzenz 的 262144 更宽松，无需修改。
- 内置 bench 脚本 `ninfer-4090d-bench/bench/adx_blackbox_serve.py` 有一处路径 bug：
  `REPO = parents[2]` 指向了 `_ninfer_repos/`，导致 manifest 找不到、bench 完全无法运行。
  已修正为 `parents[1]`（该行同时决定 manifest 与结果输出目录，两者都要求它等于 bench 仓库根）。
