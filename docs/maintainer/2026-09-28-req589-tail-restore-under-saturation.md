# req-589 尾部复用在近饱和下退化为重算 —— 诊断 + 修复笔记

> 状态：**① 与 ③ 已落地并端到端验证**（小参数 rig 48 槽 + 生产 320 槽的 `full_battery` 全绿，
> 见第 7 节）。本文**不加入文档索引**，不是实施方案，也不替代 `缓存模块v2.md`、`paged-kv-cache.md`、
> `resource-scheduling-and-context-cache.md` 等权威。
>
> 仍未完成的是**③ 交换池**（Device 与 Host 都满时的 demote 落点，见第 6 节），它没有被实现，
> 所以本文暂不删除；③ 落地后把稳定结论回写到 active 文档并删除本文。

## 1. 现象

`2026-09-28 13:33:36` 的 req-589（会话 digest `a00ec560e2648bc3`）：prompt 112,623 tok，
prefix 命中 97,917（86.9%，`long anchor`），重算 14,706 tok，TTFT 18.5s（≈prefill），总 96.8s。
它命中了一个**中段** long anchor（≈97,917），而不是尾部的 endpoint/closure，因此把上一轮生成
的内容重新 prefill。

> 更正（后续日志核实）：req-589 其实**几乎独占**在途——前序请求 588 于 `13:33:36.162` 结束，
> 0.5s 后 589 于 `13:33:36.678` 开始。Device KV 之所以 `10284` 页近满，是**前序会话（580–588）
> 留下的空闲缓存 KV**，不是 4 路活跃上下文。这是"容量看似饱和"的表象来源，也说明问题不在
> "并发太多"。

## 2. 根因：准入闸门 `physical_peak_fits` 少算了 KV 的驱逐余量（本笔记早期判错，已更正）

> 早期把根因判为"两级容量饱和、非保留顺序 bug"，据此只给了"降并发 / 留 Host 余量 / 有界 R0 /
> 加大容量"这类杠杆。后经代码 + 日志更正：**这是 CPU 可见、可修的准入决策缺口。**

真正的卡点是**准入决策读错了池子状态**：

- 尾部 endpoint（frontier `111,901`）内容匹配、状态在、**KV 保留在 Host（可恢复）**，既没丢
  也不是 digest-mismatch（满屏 `digest-mismatch` 是**别的**会话的 slot，正常被拒）。
- 恢复这份 Host 尾部只需 `add=515` 页 Device KV。但 `physical_peak_fits`（`program_impl.h`）
  对 **KV 维**用**裸占用** `used + add <= cap` 判定，对 **State 维**却**计入** release-ladder 的
  驱逐余量（`state_slot_relief` / `host_slot_relief`）。
- 于是 Device KV `used=9793, add=515, cap=10284` → `9793+515 > 10284` 被读成"**永久不可行**"
  （`rej=3`），闸门拒绝恢复尾部 → 引擎退回"显存里放得下的最深 long anchor（97,917）"重算。
  **其实只要把最不重要的空闲 Device KV demote 到那 ~1.5GB 空闲 Host，就能腾出 515 页。**
- req-671/677 同一症状（`main_kv` 与 `backend_kv` **双双** `used+add>cap`，`budget 15.0s`，
  cache `0.0%`，TTFT 2m11.5s / 3m3.7s）。

链条：**闸门少算 KV 驱逐余量 → 把"可恢复的尾部"误判为不可行 → 放弃 Host 恢复、退回浅 anchor
重算**。这与 2026-09-22 在 State 维修过的同类 bug 同族——KV 维一直没补上。

## 3. 为什么"改保留顺序"不是解法（保留顺序本就正确）

用户原则「KV 与镜像端点都优先保尾部」在**保留**这一半**已经成立**，无需改：

- KV 侧：`cache_tier_policy.h` 的 `plan()` 已按**单一 importance**（冷会话先、整条会话）驱逐，
  R2 先腾 Host 房间再让 R1 demote（`case_host_room_reserved_for_spill`）。
- 镜像侧：589 的 slot 259 里 endpoint 本来就比 long anchor 活得久。

589 疼在**恢复准入**（把可恢复的尾部判成不可行），不在排序——所以修复点是**闸门**，不是保留顺序。

## 4. 修复

1. **① 闸门补 KV 驱逐余量**（CPU 已验证）：`physical_peak_fits` 现在对 KV 维也计入 `kv_relief()`
   ——把最不重要的空闲 Device KV demote 到**当前**空闲 Host 空间能腾出的页数。**是驱逐能力，
   不是固定余量**（固定余量会被下一次复用吃光，正是被否掉的做法）。纯逻辑抽到
   `resource_projection.h::kv_landing_relief`（CPU 可测）；main/backend 各自**独立**认领落点
   （避免 main 吃光落点把 backend 饿死——671 双轴症状）。
2. **② 恢复源保护**（已有）：`protected_materialization_page` 保证"正在恢复的那份缓存"不会被
   R1/R2 选中当受害者（不会"把正要恢复的那份驱逐掉"）。

待办 **③ 交换池（scratch）**：Device 与 Host **都满**时，demote 无处可落。需要一个小的专用交换
落点（~2–4 GiB，启动可配），让"显存 →（Host 满则 scratch）→ 显存"的恢复永不因"两边都满"
而死锁、也不会误驱逐恢复源。设计已定，**待确认** + GPU 验证（本沙箱无 nvcc/GPU：只能 host
编译，端到端要在你的机器上跑）。

## 5. CPU 可验证的验证（已跑，全绿）

- `tests/test_physical_peak_fit.cpp`（新增，已注册进 `tests/CMakeLists.txt`）：
  - 复现 589：Device 满 + 无 KV 余量 → 尾部**不可行**（修复前闸门）；补上 `kv_relief` → **可行**（修复）。
  - 671 双轴：main + backend 各 ~515 页缺口，落点**两轴都覆盖**（不被 main 饿死）。
  - 589 全链路：池快照 → `kv_landing_relief` → `physical_peak_fits_core` → 尾部可行
    （用真实 `used=9793 / cap=10284 / free-hkv≈1.5GB`）。
  - 边界：余量 < 缺口仍不可行、= 缺口恰好可行、Host KV（R2）余量、State 余量回归、不过减。
- `ninfer_cache_tier_policy_test`、`ninfer_resource_manager_test` 回归通过。
- `libninfer_engine.a` 重新编译 + 链接通过（改动进了引擎库，无新增告警）。

## 6. 待办（需要人 + GPU 环境）

- **③ 交换池**：确认后落 GPU 层（host 可编译，端到端需你的机器）。
- **端到端 GPU soak**：4 会话把 Device+Host 灌到 589 那种形状，确认尾部被**恢复**（重算从
  ~14.7K 掉到尾部增量）而不是退回 anchor/root。本沙箱无 GPU，`spill_selection` /
  `pressure_prepare_cases` 只能 skip。
- 确认后把稳定结论回写 `缓存模块v2.md` / `resource-scheduling-and-context-cache.md`，删除本文。

## 7. 验证结果（2026-09-28 晚，本笔记的收尾依据）

- **单元测试**：`ctest` 115/115。
- **小参数 rig**（`tools/smoke/fair_share_small.sh` + `tools/smoke/full_battery.sh --rig`，
  48 host state 槽 / 32K 上下文）：phase 3/4/5 全部 rc=0，窗口内错误行 0，显存数据销毁 0，
  守卫拒绝 0；fair-share 段 `allocate FAILED -> 0`。
- **生产**（`tools/smoke/full_battery.sh`，320 槽 / 658,176 token / RTX 4080 SUPER）：
  灌池到 host 320/320 后，phase 3/4 全部 rc=0；HTTP 503 = 0、HTTP 500 = 0、engine fatal = 0、
  不变量违反 = 0、`device-not-closable` = 0；**400/400 请求全部完成**。
- **① 的信用在执行侧兑现**：新增 `release_device_kv_capacity_step`（§三 R1 于预留时刻），
  生产窗口内执行 209 次 Device→Host 搬运，0 次 `Device-KV capacity exhausted`。A/B 复现：
  修复前 1 次 `resize_reservation` 失败 + 1 次 re-admit；修复后 21 次搬运、0 次失败。
- **两条被证伪的推断**（记录以免重蹈）：设备 state 并非"被锁住"（锁有 commit/abort 配平与
  传输析构自动 abort，不会漏锁）；Host 满也并非"无合法出路"——§三 R1 明确要求先 R2 腾地方再 R1，
  实测问题是阶梯每次只腾 1 个 Host 槽而事务需要 2 个，已由"按事务总需求一次腾够"修复。
- **仍未闭环**：生产 14:4x 那次 `mv_state=0` 的现场日志已被删除，无法确认当时是哪一步消耗了缺口；
  现有证据只排除了"阶梯选不到受害者"和"锁漏放"两种解释。
