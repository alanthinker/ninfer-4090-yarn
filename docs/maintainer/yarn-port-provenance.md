# YaRN port provenance: defect attribution and consistency record

> **English summary.** When this repository merged the YaRN rope-scaling work from
> [gzenz/ninfer](https://github.com/gzenz/ninfer) (one of the two merge parents), seven defects
> surfaced. Each was verified line-by-line against the gzenz sources to establish whether the
> defect was introduced by the merge or inherited from upstream. Result: #1 and #3 are merge
> omissions (the gzenz code was correct); #2, #5 and #6 are pre-existing upstream defects that
> this repository inherited and fixed; #4 is a combination upstream never implemented, now
> fail-fast guarded here; #7 is a merge-introduced, unreachable, readability-only artifact.
> The analysis also established that in the gzenz tree the prefill-side device ramp never
> executed (its `set_rope_scaling` call fed a field its construction sites never populated), so
> its published 555K/600K long-context quality data was produced in an inconsistent state:
> prefill unscaled, decode scaled. This repository scales both paths, with the policy made a
> mandatory constructor input so the two cannot silently diverge; the 260K-token prefix-reuse
> test is the direct consistency evidence. The record below is in Chinese.

## 方法

对 gzenz/ninfer（本仓库两个 merge parent 之一）的对应源码逐个拉下来、逐行对照，判定每个缺陷
是移植产生的，还是两个仓库本来就有。

## 缺陷归属

| # | 缺陷 | 归属 | 依据 |
|---|---|---|---|
| 1 | prefill 时 rope/cache 位置别名 | 移植遗漏 | gzenz 的条件写作 `rope_delta_ != 0 \|\| rope_scaling_factor_ != 1.0F`，移植时漏了后半句。scaling 激活而 rope delta 为 0 时，rope 位置 buffer 与 cache 位置 buffer 被别名到同一块内存，ramp 写坏 cache 位置 |
| 2 | float 精度 off-by-one | 上游既有 | gzenz 的 `position.cuh` 就是那行 float 公式，逐字相同 |
| 3 | MTP AR 未缩放 | 移植遗漏 | gzenz 的 `mtp_impl.h` 有逐步 slice + scale 块，移植时漏了 |
| 4 | DFlash + YaRN | 上游未实现 | gzenz 的 `program_impl.h` 里没有任何 DFlash rope 缩放 |
| 5 | ExecutionCore 漏传 7 处 | 上游既有 | gzenz 有完全相同的 7 处 raw 构造点 + 相同默认值 |
| 6 | MRoPE [len,3] 抛异常 | 上游既有 | 相同形状 + 相同校验 + 相同无条件调用 |
| 7 | DFlash representative | 移植引入 | 不可达，仅可读性 |

所以：#1、#3 是移植时没抄全（gzenz 的代码是对的）；#2、#5、#6 是上游既有缺陷，本仓库的移植
继承了它们并修掉了；#4 是上游没实现的组合，本仓库加了 fail-fast 守卫。

## 当前代码状态（与本仓库源码的对应关系）

| # | 位置 | 现状 |
|---|---|---|
| 1 | `src/targets/qwen3_6/impl/runtime/text_context_impl.h` | `rope_axes` 的条件为 `rope_delta_ != 0 \|\| rope_scaling_factor_ > 1.0F`（factor 经 `scale_positions_yarn` 入口校验 ≥ 1.0，与 gzenz 的 `!= 1.0F` 等价）；scaling 激活时 rope 位置使用独立 buffer，不再与 cache 位置别名（rope delta 为 0 时填充退化为拷贝） |
| 2 | `src/ops/kernel/position.cuh`（`scale_positions_yarn_kernel`） | ramp 的舍入项在小量级求值、`original_context` 后加——先折进单精度和会丢失 +0.5（float 在 ~310k 量级间距约 0.0156），结果产生偏差并与 host 侧 ramp 不一致；商改用 double 计算，与 decode/host 路径同一算术，保证同一 token 位置无论经 prefill 还是 decode 到达都映射到同一 RoPE 位置 |
| 3 | `src/targets/qwen3_6/impl/runtime/mtp_impl.h` | 每轮对 AR rope 帧整帧 `scale_positions_yarn`。整帧缩放的原因：kernel 写入行距为整帧 batch 的矩阵，本轮未写的槽位是陈旧值，由 `ar_valid_columns` 门控，缩放它们是惰性的。不缩的后果是 draft 与 verifier 在原生上下文之外位置不一致，MTP 接受率在扩展区塌陷 |
| 4 | `src/targets/qwen3_6/impl/runtime/layouts_impl.h` | `rope_scaling_factor > 1.0` 且 DFlash 时 fail-fast 抛错。原因：DFlash 提议路径的 RoPE 位置与 KV cache 位置来自同一 tensor（`proposal_positions` 同时喂给 `ops::rope` 和 `cache_positions`），无法只缩一边 |
| 5 | `src/targets/qwen3_6/impl/runtime/schedule.h`（`ExecutionCore`） | `rope_scaling_factor` / `rope_scaling_original_context` 是构造函数必填参数而非聚合初始化字段：漏传会静默回落到"不缩放"，使 prefill 与 decode 位置不一致，因此"漏传必须响亮地失败" |
| 6 | `src/targets/qwen3_6/impl/runtime/text_context_impl.h` | 多模态 chunk 的 rope 位置是 `[len,3]` MRoPE 矩阵（而非文本的 `[len,1]` 布局），ramp 对整块 buffer 逐元素作用，故先 flatten 再缩放；两种布局按构造都是连续内存（`matrix()` 用密集 stride） |
| 7 | `src/targets/qwen3_6/impl/runtime/program_impl.h`（`prepare_representative`） | CUDA Graph representative 路径；该处移植引入的差异不可达，仅可读性问题 |

## 推论：gzenz 树中 prefill 的 device 侧缩放从未真正执行

查 #5 时顺带确认：

- gzenz 的 `configure_text_card` 会无条件调 `set_rope_scaling(execution.rope_scaling_factor, ...)`；
- 但那 7 处（#5）构造点都不传该字段 → 全部落到默认值 `1.0F`；
- 而 prefill 的缩放由 `TextContext::rope_scaling_factor_` 把关（`if (... != 1.0F)`）。

因此：**gzenz 仓库里 prefill 的 device 侧缩放从未真正执行过——只有 decode/host 侧被缩放**。
他们公布的 555K/600K 质量数据，是在「prefill 不缩放 + decode 缩放」这种不自洽状态下取得的。

## 本仓库的行为差异与一致性证据

- 本实现把该字段变成 `ExecutionCore` 构造函数必填参数（见 #5），prefill 和 decode 都缩放、
  两者一致。
- 所有长上下文测试（包括 26 万 token 前缀复用 0.6s 命中、答案正确）都是在"两侧都缩放"下
  通过的。
- 那份前缀复用测试恰是一致性的直接证据：prefill 写入的 KV 被完整复用后仍能答对，说明两条
  路径（prefill 写入 / decode 读取）的位置自洽。
