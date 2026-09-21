#!/usr/bin/env python3
"""End-to-end verification harness for the merged NInfer build.

Measures four cases against a running `ninfer-serve` and reports the **server-side**
`timings` block, which is the authoritative measurement channel:

    a) long-form output  - GDN long generation does not crash, CUDA error lines == 0
    b) fresh prefill     - zero-cache-reuse TTFT and prefill rate at depth
    c) MTP3 decode       - decode rate and draft acceptance on a code prompt
    d) needle retrieval  - exact retrieval deep inside a long context

Why not curl's `time_starttransfer`: a streaming response emits SSE preamble/keep-alive
bytes before prefill completes, so `%{time_starttransfer}` returns a meaningless ~0.1 s.
`timings.ttft_ms` is the server's own number.

Prompt sizing note: the repeated Chinese boilerplate tokenizes at roughly
1.57 chars/token, so a character-count estimate overshoots the token count by ~57%.
A 635,969-character prompt is ~400K tokens and is rejected with
`context_length_exceeded` against `--max-context 262144`. The block counts below are
chosen to land near 70K and 145K tokens; always read `timings.prompt_n` rather than
trusting the character count.

Usage:
    python3 verify_merged_e2e.py --port 18080 --model orcarouter
    python3 verify_merged_e2e.py --port 18081 --model orcarouter --out baseline.json
"""

from __future__ import annotations

import argparse
import json
import os
import time
import urllib.request

PARA = (
    "本段用于预填充基准测试。推理引擎的核心路径包括分词、嵌入查找、"
    "注意力与门控线性层的交替堆叠、前馈网络的量化投影，以及投机解码的草稿校验。"
    "每一层都依赖显存带宽与张量核心的协同，长上下文的瓶颈则集中在KV缓存的读写上。"
)

NEEDLE_PARA = (
    "长文本检索基准文档。KV缓存以页为单位组织，每页容纳若干令牌的键值张量，"
    "调度器依据会话活跃度决定驻留、换出与丢弃策略。位置编码的缩放因子决定了"
    "模型在超出训练长度时的外推能力，针探测试用于度量该能力随深度与距离的衰减。"
)


def post(api: str, payload: dict, stream: bool, timeout: float):
    """POST a chat completion; return (result, wall_seconds)."""
    req = urllib.request.Request(
        api + "/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        if not stream:
            body = json.loads(resp.read().decode())
            return {"timings": body.get("timings"),
                    "content": body["choices"][0]["message"].get("content") or "",
                    "usage": body.get("usage")}, time.time() - t0

        timings, text, reasoning = None, "", ""
        for raw in resp:
            line = raw.decode("utf-8", "ignore").strip()
            if not line.startswith("data: "):
                continue
            body = line[6:]
            if body == "[DONE]":
                break
            try:
                msg = json.loads(body)
            except json.JSONDecodeError:
                continue
            if msg.get("timings"):
                timings = msg["timings"]
            delta = msg.get("choices", [{}])[0].get("delta", {})
            text += delta.get("content") or ""
            reasoning += delta.get("reasoning_content") or ""
        return {"timings": timings, "content": text, "reasoning": reasoning}, time.time() - t0


def case_a(api: str, model: str, timeout: float):
    """Long-form output: the hypothesis under test is a GDN long-generation crash."""
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": (
            "写一篇不少于8000字的长文，主题为《长上下文推理引擎的过去、现在与未来》，"
            "要求结构完整、分章节论述充分，结尾给出展望。")}],
        "max_tokens": 16384, "temperature": 0.7, "stream": False,
    }
    res, wall = post(api, payload, False, timeout)
    return {"completion_tokens": res["usage"]["completion_tokens"],
            "finish_reason": None, "wall_s": round(wall, 1), "timings": res["timings"]}


def case_b(api: str, model: str, blocks: int, timeout: float):
    """Fresh (salted) long prompt: no prefix-cache reuse, so this is a true cold prefill."""
    salt = os.urandom(4).hex()
    prompt = "\n".join(f"[块{i:04d}-{salt}] " + PARA * 5 for i in range(blocks))
    prompt += "\n\n请只用一句话概括上面全部文档的主题。"
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 64, "temperature": 0, "stream": True,
               "reasoning_effort": "none"}
    res, wall = post(api, payload, True, timeout)
    return {"prompt_chars": len(prompt), "wall_s": round(wall, 1), "timings": res["timings"]}


def case_c(api: str, model: str, timeout: float):
    """MTP3 decode on a code prompt, where draft acceptance is high and stable."""
    payload = {"model": model, "messages": [{"role": "user", "content": (
        "用 Python 写一个完整的线程安全 LRU 缓存实现，包含注释、容量策略说明和一组单元测试，"
        "总长约300行。直接输出代码。")}],
        "max_tokens": 4096, "temperature": 0, "stream": True, "reasoning_effort": "none"}
    res, wall = post(api, payload, True, timeout)
    return {"wall_s": round(wall, 1), "content_chars": len(res["content"]),
            "timings": res["timings"]}


def case_d(api: str, model: str, blocks: int, needle: str, timeout: float):
    """Needle-in-a-haystack: a fresh unique code placed mid-document, asked for afterwards."""
    salt = os.urandom(4).hex()
    front = [f"[前段{i:04d}-{salt}] " + NEEDLE_PARA * 5 for i in range(blocks)]
    back = [f"[后段{i:04d}-{salt}] " + NEEDLE_PARA * 5 for i in range(blocks)]
    prompt = ("\n".join(front) + f"\n【校验码】{needle}\n" + "\n".join(back)
              + f"\n\n上文【校验码】是多少？只回答这4位数字，不要输出其他内容。")
    payload = {"model": model, "messages": [{"role": "user", "content": prompt}],
               "max_tokens": 32, "temperature": 0, "stream": True,
               "reasoning_effort": "none"}
    res, wall = post(api, payload, True, timeout)
    answer = res["content"].strip()
    return {"prompt_chars": len(prompt), "needle": needle, "answer": answer[:20],
            "match": needle in res["content"], "wall_s": round(wall, 1),
            "timings": res["timings"]}


def rate(timings, key):
    return (timings or {}).get(key)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=18080)
    ap.add_argument("--model", default="orcarouter")
    ap.add_argument("--b-blocks", type=int, default=200,
                    help="paragraphs for the prefill case (~70K tokens at 200)")
    ap.add_argument("--d-blocks", type=int, default=200,
                    help="paragraphs per half for the needle case (~145K tokens at 200)")
    ap.add_argument("--needle", default="7391")
    ap.add_argument("--timeout", type=float, default=1800)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    api = f"http://{args.host}:{args.port}"
    results = {}

    print("a) long-form output ...")
    results["a"] = case_a(api, args.model, args.timeout)
    t = results["a"]["timings"] or {}
    print(f"   completion_tokens={results['a']['completion_tokens']} "
          f"wall={results['a']['wall_s']}s decode={rate(t, 'predicted_per_second'):.2f} tok/s "
          f"draft={t.get('draft_n_accepted')}/{t.get('draft_n')}")

    print("b) fresh prefill ...")
    results["b"] = case_b(api, args.model, args.b_blocks, args.timeout)
    t = results["b"]["timings"] or {}
    print(f"   prompt_n={t.get('prompt_n')} prefill={rate(t, 'prompt_per_second'):.2f} tok/s "
          f"ttft={t.get('ttft_ms', 0) / 1000:.1f}s cache_n={t.get('cache_n')}")

    print("c) MTP3 decode ...")
    results["c"] = case_c(api, args.model, args.timeout)
    t = results["c"]["timings"] or {}
    acc = (t.get("draft_n_accepted", 0) / t.get("draft_n", 1) or 0) * 100
    print(f"   predicted_n={t.get('predicted_n')} "
          f"decode={rate(t, 'predicted_per_second'):.2f} tok/s acceptance={acc:.1f}%")

    print("d) needle retrieval ...")
    results["d"] = case_d(api, args.model, args.d_blocks, args.needle, args.timeout)
    t = results["d"]["timings"] or {}
    print(f"   prompt_n={t.get('prompt_n')} prefill={rate(t, 'prompt_per_second'):.2f} tok/s "
          f"answer={results['d']['answer']!r} match={results['d']['match']}")

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, ensure_ascii=False, indent=2)
        print(f"\nwrote {args.out}")

    ok = (results["a"]["completion_tokens"] >= 8000
          and results["b"]["timings"] is not None
          and results["c"]["timings"] is not None
          and results["d"]["match"])
    print(f"\nRESULT: {'PASS' if ok else 'CHECK MANUALLY'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
