#!/usr/bin/env python3
"""验证"锚点满 → 驱逐旧锚点 → 复用仍命中"的修复(在打满的池子上运行)。

前提:fill_anchors.py 已把 host state 池打到 320/320。
做法:对一条填充会话发一个 follow-up(完整旧历史 + 一条新 user 消息)。
  旧二进制:device 8/8 + host 320/320 → "reuse infeasible (physical peak)" → cache 0,冷 prefill。
  新二进制:阶梯 ②b 驱逐最老锚点的 host 副本 → 降级腾出 device 槽 → H2D 恢复 → 高命中率。

判定:cache% > 80 且 日志窗口内无 "reuse infeasible" 即 PASS;
      日志窗口内出现 "anchor host evict" 行是新机制生效的直接证据(非必需)。
"""
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from harness_paths import HARNESS_DIR, cached_prompt_tokens  # noqa: E402
from fill_anchors import build_session, occupancy, load_corpus  # noqa: E402

# NINFER_SERVICE_LOG is the documented harness variable (the agent rig exports it and writes
# ninfer_serve_agent.log); NINFER_SERVE_LOG is kept as the historical alias.
SERVE_LOG = Path(os.environ.get("NINFER_SERVICE_LOG")
                 or os.environ.get("NINFER_SERVE_LOG")
                 or HARNESS_DIR / "ninfer_serve.log")


def ask(base: str, msgs: list, timeout: int = 600) -> dict:
    payload = json.dumps({"model": "myai", "messages": msgs, "stream": False,
                          "max_completion_tokens": 1, "temperature": 0}).encode()
    req = urllib.request.Request(base + "/v1/chat/completions", data=payload,
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        out = json.loads(r.read())
    u = out.get("usage", {})
    p = u.get("prompt_tokens") or 0
    c = cached_prompt_tokens(u)
    return {"prompt": p, "cached": c, "cache_pct": round(100.0 * c / p, 1) if p else 0,
            "elapsed": round(time.time() - t0, 1)}


def main() -> None:
    port = sys.argv[1] if len(sys.argv) > 1 else "30000"
    base = f"http://127.0.0.1:{port}"
    corpus = load_corpus()
    print(f"occupancy before: {occupancy()}")

    log_before = SERVE_LOG.read_text(errors="replace") if SERVE_LOG.exists() else ""
    n_infeasible_before = len(re.findall(r"reuse infeasible", log_before))
    n_evict_before = log_before.count("anchor host evict")

    # 对三条不同新旧程度的填充会话发 follow-up(旧历史 + 一条新消息)
    results = []
    for s in (4, 9, 14):
        msgs = build_session(s, corpus)
        msgs.append({"role": "user", "content": f"Reply with exactly: VERIFY-{s:02d} ACK."})
        r = ask(base, msgs)
        results.append((s, r))
        print(f"session {s:02d}: prompt={r['prompt']:>6} cached={r['cached']:>6} "
              f"cache={r['cache_pct']:>5}%  {r['elapsed']:>6.1f}s", flush=True)

    log_after = SERVE_LOG.read_text(errors="replace")
    window = log_after[len(log_before):]
    n_infeasible = len(re.findall(r"reuse infeasible", window))
    n_evict = window.count("anchor host evict")
    evict_lines = [l for l in window.splitlines() if "anchor host evict" in l][:5]

    ok = all(r["cache_pct"] > 80 for _, r in results) and n_infeasible == 0
    print(f"\nwindow: reuse_infeasible={n_infeasible} (before={n_infeasible_before}) "
          f"anchor_host_evict={n_evict}")
    for l in evict_lines:
        print("  |", l[:160])
    print("== PASS: 池满下复用仍命中, 无 physical-peak 拒绝 ==" if ok
          else "== FAIL: 存在 cache 未命中或 physical-peak 拒绝 ==")


if __name__ == "__main__":
    main()
