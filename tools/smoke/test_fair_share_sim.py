#!/usr/bin/env python3
"""多客户端公平份额保留回归测试(17:09 事故形态活体重放)。

形态:
  A: 三个纯追加请求热身到 ~60k(端点+尾部锚点+32768 铺开锚点),然后空闲;
  B: 从零增长 12 轮(每轮 +20k,到 ~250k),第 6 轮后做一次"压缩"改写
     (前史替换为摘要 → B 的旧 checkpoint digest 全部作废,模拟 DSH 压缩 churn);
  A2: A 纯追加 +5k 返回。

断言(从 ninfer_serve.log 的 reuse-diag / done 行解析):
  1. A2 的 reuse-diag 中 A 的深度端点仍是 CANDIDATE 且 fair=1;
  2. A2 的 done 行 cache 命中率 >= 95%(端点复用,TTFT 小);
  3. 整个测试窗口内不出现 "fair-share released N>0"(共享池未真耗尽)。

用法: python3 test_fair_share_sim.py [--A-depth 60000] [--B-rounds 12] [--step 20000]
不影响真实客户端;测试期间其他活跃会话的请求会在 FIFO 队列中等待。
"""
import argparse
import json
import os
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:30000/v1/chat/completions"
MODEL = "myai"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness_paths import HARNESS_DIR  # noqa: E402

LOG = os.environ.get("NINFER_SERVE_LOG") or os.path.join(HARNESS_DIR, "ninfer_serve.log")

# 约 1024 字符 ≈ 256 token 的确定性填充段(英文,tokenizer 行为稳定)
FILLER_UNIT = (
    "The inference engine retains a durable checkpoint for every published "
    "continuation boundary so that a later request sharing the same prefix can "
    "resume from the cached boundary instead of recomputing the prefix. "
) * 4  # ≈ 1.2 KB


def filler_to_tokens(chars: int) -> str:
    reps = max(1, chars // len(FILLER_UNIT))
    return (FILLER_UNIT * reps)[:chars]


def est_tokens(text: str) -> int:
    return int(len(text) / 4.2)  # 英文近似;实际以 usage 为准


def post(session_id: str, messages, max_tokens: int = 16, timeout: int = 900):
    body = json.dumps({
        "model": MODEL,
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
        "temperature": 0.0,
    }).encode()
    req = urllib.request.Request(BASE, data=body, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = json.loads(r.read())
    dt = time.time() - t0
    usage = data.get("usage", {})
    reply = ""
    try:
        reply = data["choices"][0]["message"]["content"] or ""
    except Exception:
        pass
    return {
        "prompt_tokens": usage.get("prompt_tokens"),
        "completion_tokens": usage.get("completion_tokens"),
        "elapsed": dt,
        "reply": reply,
    }


RUN_ID = int(time.time())  # 每次运行唯一 → digest 全新,重跑干净

class Session:
    def __init__(self, tag: str):
        # 唯一的 system 文本(RUN_ID 隔离历次运行)保证 digest 与任何其他会话不同
        self.system = (
            f"[SIM-{tag}-{RUN_ID}] fair-share regression probe. "
            + filler_to_tokens(800)
        )
        self.history = []  # user/assistant 消息
        self.last_prompt = 0
        self.tag = tag
        self.chars_per_token = 4.0  # 首轮后用实测 usage 校准

    def grow_to(self, target_prompt: int):
        """追加 user 块使 prompt 达到 target_prompt,发送并自适应校准 token 估算。"""
        base = self.last_prompt if self.last_prompt else int(len(self.system) / self.chars_per_token)
        need = max(1024, target_prompt - base)
        chars = int(need * self.chars_per_token)
        msg = {"role": "user", "content": f"[{self.tag}+{len(self.history)}] " + filler_to_tokens(chars)}
        self.history.append(msg)
        messages = [{"role": "system", "content": self.system}] + self.history
        resp = post(self.tag, messages)
        pt = resp["prompt_tokens"]
        if pt:
            self.last_prompt = pt
            # 校准:本次新增 token ≈ pt - base,对应 chars(不含消息骨架开销)
            measured = (pt - base) / max(1, chars)
            if measured > 0.05:
                self.chars_per_token = 1.0 / measured
        return resp

    def compact_to_summary(self):
        """把前半历史替换为一条摘要(模拟 DSH 压缩):旧 checkpoint digest 全部作废。"""
        keep_head = self.history[:1]
        self.history = keep_head + [
            {"role": "user", "content": f"[{self.tag} COMPACTION] summary of prior context. "
                                        + filler_to_tokens(2000)}
        ]


def log_offset():
    return os.path.getsize(LOG) if os.path.exists(LOG) else 0


def log_since(offset: int):
    if not os.path.exists(LOG):
        return ""
    with open(LOG, "r", errors="replace") as f:
        f.seek(offset)
        return f.read()


def summarize(tag: str, chunk: str):
    """从一段日志里抽取 reuse-diag 头行/done 行,输出紧凑报告。"""
    out = []
    for line in chunk.splitlines():
        if "reuse-diag:" in line and ("prompt=" in line or "endpoint" in line or "candidates=" in line):
            out.append(line.strip())
        elif "req#" in line and " done " in line:
            out.append(line.strip()[:300])
        elif "fair-share released" in line:
            out.append("!! " + line.strip())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--A-depth", type=int, default=60000)
    ap.add_argument("--B-rounds", type=int, default=12)
    ap.add_argument("--step", type=int, default=20000)
    ap.add_argument("--compact-round", type=int, default=6)
    args = ap.parse_args()

    A = Session("A"); A.tag = "A"
    B = Session("B"); B.tag = "B"

    print(f"== phase A warm: 3 轮纯追加 → {args.A_depth} tok", flush=True)
    o = log_offset()
    steps = 3
    for i in range(1, steps + 1):
        target = args.A_depth // steps * i
        resp = A.grow_to(target)
        print(f"  A r{i}: prompt={resp['prompt_tokens']} elapsed={resp['elapsed']:.1f}s", flush=True)
        time.sleep(1.0)
    a_warm = log_since(o)
    for line in summarize("A", a_warm):
        print("   | " + line)

    print(f"== phase B churn: {args.B_rounds} 轮,每轮 +{args.step} tok,"
          f" 第 {args.compact_round} 轮后压缩改写", flush=True)
    o = log_offset()
    cur = 0
    for i in range(1, args.B_rounds + 1):
        if i == args.compact_round + 1 and B.last_prompt:
            B.compact_to_summary()
            cur = est_tokens(B.system) + 400
            print(f"  B: COMPACTION (前史→摘要, 旧 checkpoint 作废)", flush=True)
        cur += args.step
        resp = B.grow_to(cur)
        print(f"  B r{i}: prompt={resp['prompt_tokens']} elapsed={resp['elapsed']:.1f}s", flush=True)
        time.sleep(0.5)
    b_churn = log_since(o)
    rel = [l for l in b_churn.splitlines() if "fair-share released" in l]
    print("  B churn 期间 'fair-share released' 出现次数:", len(rel), flush=True)

    print("== phase A2: A 空闲后返回,+5000 tok 纯追加", flush=True)
    o = log_offset()
    a_final = A.last_prompt  # A 返回前的最终深度(断言基准,须在 A2 请求前取值)
    target = a_final + 5000
    resp = A.grow_to(target)
    print(f"  A2: prompt={resp['prompt_tokens']} elapsed={resp['elapsed']:.1f}s", flush=True)
    a2 = log_since(o)
    print("  --- A2 日志 ---")
    for line in summarize("A", a2):
        print("   | " + line)
    rel2 = [l for l in a2.splitlines() if "fair-share released" in l]
    print("  A2 期间 'fair-share released' 出现次数:", len(rel2), flush=True)

    # ---- 断言 ----
    # 17:09 形态的验收不是"绝对命中率",而是:
    #   a) A 的深度检查点(rewrite/anchor, fair=1)在 B churn 后仍为 CANDIDATE;
    #   b) 复用深度 ≥ A 最终深度 - 2048(除新追加内容与 ~20 token 开场区外全部复用);
    #   c) TTFT 小(走复用而非冷重算);
    #   d) 全程无 'fair-share released'(共享池未真耗尽,不应释放保底桶)。
    import re
    ok = True
    if "fair=1" not in a2 or "CANDIDATE" not in a2:
        print("FAIL: A2 日志中没有 fair=1 的 CANDIDATE 检查点"); ok = False
    done = [l for l in a2.splitlines() if " done " in l]
    if not done:
        print("FAIL: A2 没有 done 行"); ok = False
    else:
        l = done[-1]
        m = re.search(r"cache (\d[\d,]*) \((\d+\.\d)%", l)
        mt = re.search(r"TTFT ([\d.]+)s", l)
        if not m:
            print("FAIL: 无法解析 A2 done 行 cache"); ok = False
        else:
            reused = int(m.group(1).replace(",", ""))
            pct = float(m.group(2))
            theoretical_max = a_final / resp["prompt_tokens"] * 100
            print(f"  A2 cache 命中率: {pct}% (复用 {reused} tok,"
                  f" 理论上限 {a_final}/{resp['prompt_tokens']} ≈ {theoretical_max:.1f}%)")
            if reused < a_final - 2048:
                print(f"FAIL: 复用深度 {reused} < A 最终深度 {a_final} - 2048 (深度检查点疑似被驱逐)"); ok = False
            if mt and float(mt.group(1)) > 15:
                print(f"FAIL: TTFT {mt.group(1)}s > 15s (走了冷重算路径?)"); ok = False
    if rel or rel2:
        print("FAIL: 出现 'fair-share released'(共享池耗尽,释放了保底桶)"); ok = False

    print("== 结果:", "PASS" if ok else "FAIL")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
