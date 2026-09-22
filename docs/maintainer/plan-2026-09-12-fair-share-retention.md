# 计划(2026-09-12):检查点保留公平份额 + 复用规划器预算缩放

> 本文件是跨会话压缩的持久工作计划。背景事故、设计规格、改动点、验证方案俱全,
> 压缩后可直接据此继续实现。部署侧事实见 deploy-yarn 日志与 reqdump(路径
> `/root/ai/large_models/_ninfer_repos/deploy-yarn/`)。

## 一、背景:三起"0% 命中"事故(2026-09-12,实例 13:57:10)

| 时间 | 请求 | 现象 | 根因(已用 reuse-diag + reqdump diff 定性) |
|---|---|---|---|
| 15:12 | req#11, 74,119 tok | 0%, TTFT 1m24s | 客户端 pnpm store 路径进静态 developer 头,客户端重启后哈希变 → 头字节漂移。**已修**(启动改 npx 锁版本,清 pnpm 残留) |
| 17:09 | req#217, 155,618 tok | 0%, TTFT 3m40s | **驱逐**:纯追加请求(body diff 证明 201 条共享消息逐字节同),但本会话 155k 深度端点在 76 分钟空窗内被活跃邻居(Windows 大会话 177k→238k churn + 自身压缩)从前缀索引中释放 |
| 18:00 | req#241/242, 201,650 tok | 0%, TTFT 3m40s+4m16s | **DSH 客户端工具结果修剪**:6 条旧 tool 消息被原地改写(中间替换为 `[... tool result middle pruned ...]`),第一处改写 @msg#21(约 2 万 token 处)→ 其后全部检查点定义上作废;235,230→201,650 的缩减全来自修剪(非新压缩,两个压缩 checkpoint 在事故前后位置不变)。叠加次要因素:规划器在 200k+ prompt 上仅评估 `targets 19`(共 62+ 候选)即撞时间预算 |

关键结论:
- 17:09 型 = **引擎保留策略**问题(活跃邻居挤掉空闲会话端点)→ 公平份额方案(本文档 2.2);
- 18:00 型 = **DSH 客户端**在热缓存期间零敲碎打修剪旧工具消息(第一处改写很深,引擎侧无解,
  理论上最多复用 ~9.6k 共享前缀)→ 客户端修:修剪只动尾部窗口内消息,或并入压缩时机;
  这是 DSH 上游/本地补丁事项,**不在本引擎提交范围**;
- 规划器预算不随 prompt 长度缩放(18:00 的次要因素,且会放大任何"本可匹配"的场景)
  → 2.3 一并修。
**引擎侧两个改动(2.2 + 2.3)同一次提交,用户只重启一次服务**。

## 二、设计规格(用户已确认方向:"一半保底均分 8 会话,一半共享给最近活跃")

### 2.1 资源模型(现状)
- Host 状态镜像槽:`--host-state-slots 320` × 147 MiB ≈ 46 GiB。每个检查点
  (端点/rewrite/锚点)可用需**同时**持有:状态镜像 + 对应 KV 页。
- Host KV 页池:32 GiB,约 18.3 B/token(238k 会话 ≈ 4.4 GB)。
- 私有续算链目录:16 条链(会话"户口"),非瓶颈,不动。

### 2.2 保留策略:保底桶 + 共享池
1. **切分**:总量各 50%。保底区 = 8 个桶(桶数新 flag `--fair-share-buckets`,默认 8),
   每桶 = 1/8 保底区 ≈ 20 状态槽 + 2 GB KV。共享区供活跃会话按需使用(现有行为)。
2. **桶内容(可恢复工作集)**:每桶锁住该会话的 端点(1) + 最近尾部锚点(~4)
   + 最近铺开锚点(2~3) + 其 KV 页。≈ 8 张镜像(1.2 GB) + KV,远小于桶容量,
   桶余量归还共享区。
3. **桶分配**:按会话 MRU 轮换;第 9 个活跃会话接管最久空闲会话的桶。
4. **驱逐顺序**(压力到来时):共享区(先砍最浅铺开锚点,再砍浅尾部锚点)
   → 共享区耗尽后才按 MRU 顺序牺牲最老保底桶。**保底桶有内容时,永远不优先于
   它的"别的会话端点"**——直接修复 17:09 型事故。
5. **KV 与镜像捆绑**:桶锁定必须同时覆盖两类资源,否则"保了镜像丢了页"等于没保。

### 2.3 规划器预算缩放
- 现状:固定墙钟预算;实测 155k prompt 评 62 候选、201k 只评 19 个即撞顶。
- 改法:预算随 `prompt_tokens` 与候选数缩放(具体公式读代码后定,原则:
  **常规负载下必须能评完全部候选**;超长 prompt 下宁可花更久也不能漏报可用端点)。
- 同时:识别压缩请求(客户端压缩摘要请求特征)时临时钉住该会话锚点,
  消除"压缩切割点附近锚点恰被挤掉"的档位 2 退化(可选,视实现复杂度取舍)。

### 2.4 已知代价(已与用户对齐)
- 单活跃超大会话的全套锚点保留能力减半(共享区 160 槽/16 GB);砍的是低价值浅锚点,
  端点/尾部锚点不受影响,下一轮复用不变;
- 压缩重算:常见 ≤32k(30-50s)不变;共享池高压下 64k~128k(1.5-3min);
  极端(所有铺开锚点被挤)退到共享前缀边界(罕见,≥3 重会话同活跃)。

## 三、改动点清单(待读码确认)
1. `src/runtime/engine/resource_manager.h`(3634 行):保留/驱逐策略核心——
   `catalog_`/`shared_catalog_`/`prefix_index_`、状态镜像槽与 KV 页的按会话记账、
   victim 选择。新增:per-continuation 的桶标签(保底/共享)、桶表、MRU 记账、
   新驱逐顺序;规划器时间预算的取值处(找到常量/flag 定义)。
2. `src/serve/serve_options.{h,cpp}`:新 flag `--fair-share-buckets`(默认 8)、
   预算缩放开关(如有必要)。
3. `src/serve/operational_log.cpp`:done 行的 reuse 诊断补充规划器预算/评估数
   (已有 `targets N`;补上 budget 与是否撞顶)。
4. `docs/maintainer/resource-scheduling-and-context-cache.md`:策略变更必须同步
   这份权威文档(AGENTS.md 要求)。
5. 部署侧 `deploy-yarn/start_ninfer.sh`:新 flag 的默认值接线 + 注释。

## 四、验证(全部通过才通知用户重启;用户重启有全量重算代价,只允许一次)
1. `cmake --build build -j` 通过(不加 -jN 上限,用满并行度)。
2. **17:09 复现回归**:两会话——A 热身至 ~150k 后空闲,B 做增长+压缩 churn;
   断言 A 的深度端点在 B 全程 churn 后仍在前缀索引且可复用(A 回来 99%+ 热)。
3. **18:00 复现回归**:单长会话 ~200k 触发压缩;压缩后首请求应复用切割点附近锚点
   (重算 ≤ 32k),且日志 `targets == 索引内候选总数`(不再中途撞预算)。
4. **单会话不回归**:单活跃会话行为与现状一致(桶机制无副作用,池子仍可吃满)。
5. 显存/内存账:启动后核对 host state / KV 实际占用与 50/50 切分一致;
   8 桶 × (端点+7 锚点) 工作集在 320 槽内。

## 五、执行顺序(已完成 2026-09-12 18:50)
读权威文档 → 通读 resource_manager.h / materialization_planner.h → 实现 → 文档 → 构建 → 测试 → 通知用户重启。

### 实现落点(与设计的对应)
1. **公平份额**(`resource_manager.h`,纯逻辑层,未动 Program):
   - `ContextCacheOptions.fair_share_buckets`(默认 8,0 禁用,cache 禁用时强制 0;
     新 serve flag `--fair-share-buckets`,启动日志 `fair-share N`);
   - `CatalogEntry.activity_epoch`:publication / reuse hit / restore 时推进,做 MRU 排序;
   - `fair_share_protected_slots()`:无 active edge 且持有 checkpoint 集合的空闲会话按 MRU
     取前 N 个 = 保底桶;
   - 保护方式 = **victim domain 缺席**(owner 整体不进 materialization 压力域、shared-capture
     压力域与 capture portfolio)——镜像与 KV 随 owner 一体,结构上无法被任何压力目标驱逐;
   - `plan_materialization` 重试环:首次规划全桶保护;`nullopt`(连 root maximal 都不可行 =
     共享池耗尽)时释放最老桶重规划,直至可行或桶全释放;末次尝试运行于完整 victim 域,
     root maximal 即无界 correctness fallback —— 保底桶永远不制造 blocked;
   - 诊断:`reuse-diag` 头行 `fair=N`、逐条 `fair=1`;done 行 `budget Xs` /
     `fair-share released N`(`MaterializationDiagnostics` 新增 `search_budget_ns`、
     `fair_share_released_buckets`)。
2. **规划器预算**(`materialization_planner.h`):
   `search_budget = min(90s, max(15s, 成本/20, 候选数×500ms))`;
   `guided window = min(budget, max(5ms, 候选数×500ms))` —— 每个候选的 retention closure
   都能被 seed,慢机器上不再漏评。
3. 测试:`test_resource_manager.cpp` 新增 2 个回归(最老桶释放顺序 = 17:09 类;
   capture 压力不碰保护桶),4 个 capture 机制测试显式 `fair_share_buckets=0`
   (机制验证与桶行为分离);全部通过。

### 验证状态(用户重启前)
- `cmake --build build -j` 全绿(368 targets);
- `ninfer_resource_manager_test` 全过(含新回归);serve/cli/public-api/context-cost/
  admission/kv-capacity/request-log/serve-metrics/openai/anthropic 各逻辑测试全过;
- `build/apps/ninfer-serve --help` 确认新 flag 已入二进制;
- `deploy-yarn/start_ninfer.sh` 已接线 `--fair-share-buckets`(env `NINFER_FAIR_SHARE_BUCKETS`,
  默认 8),语法检查通过;
- GPU 实测(验证 2/3/4 的活体重放)**无法在不重启的前提下进行**——GPU 被运行中的实例
  占满;由用户重启后按第一/二节的事故形态自然回归:
  - 17:09 类:邻居活跃 churn 期间,空闲会话回来应 ≥99% 热;日志 `reuse-diag` 头行
    `fair=N`,其 checkpoint 带 `fair=1`;
  - 规划器:done 行出现 `budget Xs`,大候选集请求 `targets` 应覆盖全部候选
    (stop_reason 不再提前 `time_budget`);
  - 极端:`fair-share released N>0` 只在共享池真耗尽时出现。
- 重启后验收(2026-09-12 晚,新实例 19:21:02):reuse-diag 头行 fair=N 随会话数
  增长、检查点行带 fair=1、真实 DSH 会话连续 99%+ 端点命中,均通过;启动行
  `fair_share_buckets=8` 因死代码缺陷未出现在运行实例(仅日志行,功能逻辑不受影响),
  磁盘二进制已修复,下次重启后在 banner 行可见。多客户端活体重放脚本:
  `tools/smoke/test_fair_share_sim.py`(A 60k 热身空闲 / B 240k churn+压缩改写 /
  A 返回,断言端点 fair=1 CANDIDATE、命中 ≥95%、无桶释放)。
- 已知未修(客户端侧,超出本次引擎提交):DSH 工具结果原地修剪会作废深度检查点,
  18:00 型事故引擎侧无解,需要 DSH 侧改修剪策略。
