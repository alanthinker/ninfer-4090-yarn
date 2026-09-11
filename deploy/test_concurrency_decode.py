#!/usr/bin/env python3
"""并发批处理对 decode 吞吐的影响：单流 vs N 路并发。

存在的理由：引擎 `--max-concurrency N` 只是能力上限；单流 decode 速率（如 48 tok/s）和
「N 路并发时的聚合速率」是两个完全不同的量，很容易被误读成后者。

机理假设：decode 是权重读取受限（每轮前向要把权重过一遍），所以批处理 N 路时权重读取被
共享，聚合吞吐理论上可接近 N 倍。本脚本测量实际能达到多少。

用法:
  python3 test_concurrency_decode.py 4 8192       # 4 路, 每路 ~8K 上下文
  python3 test_concurrency_decode.py 4 150000     # 4 路, 每路 ~150K 上下文 (需 4×150K ≤ 池子)
"""
import json
import sys
import threading
import time
import urllib.request
from pathlib import Path

BENCH_CLI = Path("/root/ai/large_models/_ninfer_repos/ninfer-4090d-bench/examples/cli")
BASE = "http://127.0.0.1:30000/v1/chat/completions"
MODEL = "myai"
CHARS_PER_TOKEN = 32338 / 7680

# 要求长输出: 短输出(几十~一两百 token)根本进不了稳态, 5 秒吞吐窗口会混进 prefill 与收尾,
# 测出来的"decode 速率"没有意义。这里要求穷举式长回答, 让 decode 阶段足够长且 4 路充分重叠。
PROMPT_SUFFIX = (
    "\n\nTask: produce a long, detailed analysis of the document above. "
    "Go through it section by section (at least 12 sections) and for each one write a paragraph "
    "of 8-10 sentences describing what it says, quoting distinctive wording from that section. "
    "Do not stop early and do not summarize the whole thing at the end; keep going until every "
    "section has been covered in full."
)


def load_haystack() -> str:
    msgs = json.loads((BENCH_CLI / "messages" / "long_niah_256k.json").read_text(encoding="utf-8"))
    body = msgs[1]["content"]
    return body[: body.index("</document>")]


def make_prompt(haystack: str, tokens: int, lane: int) -> str:
    """给每一路不同起点的文档，避免它们共享前缀（否则测的不是 N 个独立会话）。"""
    chars = int(tokens * CHARS_PER_TOKEN)
    reps = chars // len(haystack) + 2
    big = haystack * reps
    offset = lane * 4096                      # 错开起点 -> 内容不同
    doc = big[offset: offset + chars]
    return (
        "Read the document and answer the question after it.\n\n<document>\n"
        + doc + "\n</document>" + PROMPT_SUFFIX
    )


def run_one(idx: int, prompt: str, max_tokens: int, out: dict, lock) -> None:
    payload = json.dumps({
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "enable_thinking": False,
        "max_completion_tokens": max_tokens,
        "stream": False,
    }).encode()
    req = urllib.request.Request(BASE, data=payload, headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            d = json.loads(r.read())
        tm = d.get("timings", {})
        with lock:
            out[idx] = dict(
                ok=True, t_start=t0, t_end=time.time(),
                pt=d["usage"]["prompt_tokens"], ct=d["usage"]["completion_tokens"],
                ttft_ms=tm.get("ttft_ms", 0),
                prefill_tps=tm.get("prompt_per_second", 0),
                decode_tps=tm.get("predicted_per_second", 0),
                pred_ms=tm.get("predicted_ms", 0), pred_n=tm.get("predicted_n", 0),
                draft_n=tm.get("draft_n", 0), draft_acc=tm.get("draft_n_accepted", 0),
            )
    except Exception as exc:  # noqa: BLE001
        with lock:
            out[idx] = dict(ok=False, t_start=t0, t_end=time.time(),
                            err=f"{type(exc).__name__}: {exc}")


def summarise(label: str, out: dict, n: int) -> dict:
    ok = [r for r in out.values() if r.get("ok")]
    print(f"\n{'=' * 78}\n{label}\n{'=' * 78}")
    if not ok:
        for i, r in out.items():
            print(f"  lane {i}: 失败 {r.get('err')}")
        return {"aggregate": 0, "ok": 0}
    span_start = min(r["t_start"] for r in ok)
    span_end = max(r["t_end"] for r in ok)
    span = span_end - span_start
    total_tokens = sum(r["pred_n"] for r in ok)          # 用引擎口径的 predicted_n
    print(f"{'lane':>4} {'prompt':>8} {'输出':>6} {'TTFT':>8} {'prefill':>8} {'decode':>8} {'MTP':>7}")
    for i in sorted(out):
        r = out[i]
        if not r.get("ok"):
            print(f"{i:>4} {'失败':>8}  {r.get('err')}")
            continue
        mtp = f"{r['draft_acc']}/{r['draft_n']}" if r["draft_n"] else "-"
        print(f"{i:>4} {r['pt']:>8,} {r['ct']:>6} {r['ttft_ms']/1000:>7.1f}s "
              f"{r['prefill_tps']:>8.0f} {r['decode_tps']:>8.1f} {mtp:>7}")
    # 聚合口径: 所有 lane 的 decode token 总和 / 从首请求开始到末请求结束的墙钟
    agg_span = total_tokens / span if span > 0 else 0
    sum_lane = sum(r["decode_tps"] for r in ok)          # 各路速率直接相加(需真并发才有意义)
    print(f"\n  成功 {len(ok)}/{n} 路")
    print(f"  decode token 合计 : {total_tokens}")
    print(f"  并发墙钟跨度      : {span:.1f}s")
    print(f"  >>> 聚合 decode   : {agg_span:.1f} tok/s  (总 token / 墙钟跨度)")
    print(f"      (各路速率相加) : {sum_lane:.1f} tok/s")
    return {"aggregate": agg_span, "sum_lane": sum_lane, "ok": len(ok), "lanes": ok}


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    n = int(sys.argv[1])
    tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 8192
    max_tokens = int(sys.argv[3]) if len(sys.argv) > 3 else 2048

    haystack = load_haystack()
    prompts = [make_prompt(haystack, tokens, i) for i in range(n)]
    print(f"并发 {n} 路, 每路目标 ~{tokens} token 上下文, 每路输出上限 {max_tokens} token")
    print(f"预估总 prompt ≈ {n * tokens:,} token (池子 658,176, 4 路各约 {(tokens // 1000)}K)")

    # 基线: 先单流跑一路（无竞争）
    base_out, lock = {}, threading.Lock()
    run_one(0, prompts[0], max_tokens, base_out, lock)
    base = summarise(f"基线: 单流 1 路 @ ~{tokens} token", base_out, 1)

    # 并发: n 路同时
    conc_out = {}
    threads = [threading.Thread(target=run_one, args=(i, prompts[i], max_tokens, conc_out, lock))
               for i in range(n)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0
    conc = summarise(f"并发: {n} 路同时 @ ~{tokens} token  (总墙钟 {wall:.1f}s)", conc_out, n)

    print(f"\n{'=' * 78}\n对照\n{'=' * 78}")
    if base["ok"] and conc["ok"]:
        b = base["lanes"][0]["decode_tps"]
        a = conc["aggregate"]
        per = conc["lanes"][0]["decode_tps"] if conc["lanes"] else 0
        print(f"  单流 decode           : {b:.1f} tok/s")
        print(f"  {n} 路聚合 decode      : {a:.1f} tok/s")
        print(f"  {n} 路每路 decode      : {per:.1f} tok/s")
        print(f"  聚合 / 单流           : {a / b:.2f}x      (1.0x = 批处理无增益)")
        print(f"  每路 / 单流           : {per / b:.2f}x      (1.00x = 各路互不影响)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
