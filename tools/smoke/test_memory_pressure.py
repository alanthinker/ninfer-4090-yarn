#!/usr/bin/env python3
"""Memory pressure test: exhaust device state slots to trigger eviction paths.

Strategy:
  Phase 1: Create N concurrent conversations (N > device_slots) to fill all device slots.
  Phase 2: While all slots are occupied, send follow-ups to force H2D (needs free device slot
           → must D2H-demote a victim).
  Phase 3: Keep adding conversations to pressure the host slot pool.

Usage:
    python3 tools/smoke/test_memory_pressure.py --port 30000 --conversations 12
"""

import argparse
import json
import random
import string
import sys
import time
import urllib.request


def random_text(n_words):
    words = []
    for _ in range(n_words):
        words.append("".join(random.choices(string.ascii_lowercase, k=random.randint(3, 10))))
    return " ".join(words)


def send_request(base_url, messages, max_tokens=30, timeout=300):
    data = json.dumps({
        "model": "myai",
        "messages": messages,
        "max_tokens": max_tokens,
        "stream": False,
    }).encode()
    req = urllib.request.Request(
        f"{base_url}/chat/completions", data=data,
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read().decode())
            choices = result.get("choices", [])
            content = ""
            if choices:
                msg = choices[0].get("message", {})
                content = msg.get("content", "") or ""
            usage = result.get("usage", {})
            return {
                "content": content,
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "cached_tokens": (usage.get("prompt_tokens_details", {}) or {}).get("cached_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }
    except Exception as e:
        return {"error": str(e), "content": "", "prompt_tokens": 0, "cached_tokens": 0}


def main():
    parser = argparse.ArgumentParser(description="Memory pressure test")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--conversations", type=int, default=12,
                        help="Number of concurrent conversations (should be > device_slots)")
    parser.add_argument("--words", type=int, default=4000,
                        help="Words per prompt (larger = more KV pages)")
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    base_url = args.base_url or f"http://127.0.0.1:{args.port}/v1"
    print(f"=== Memory Pressure Test ===")
    print(f"Base URL: {base_url}")
    print(f"Conversations: {args.conversations}")
    print(f"Words/prompt: ~{args.words}")
    print(f"Device state slots: 4 (8 with both replicas)")
    print(f"Host state slots: 320")
    print(f"KV page pool: 10,284 pages (256 tokens/page)")
    print()

    # Phase 1: Create conversations to fill device slots
    print("--- Phase 1: Create conversations (fill device slots) ---")
    conversations = []
    for i in range(args.conversations):
        text = random_text(args.words)
        messages = [{"role": "user", "content": text}]
        start = time.time()
        result = send_request(base_url, messages, timeout=args.timeout)
        elapsed = time.time() - start

        if "error" in result:
            print(f"  conv {i:2d}: ERROR: {result['error']}")
            print("\nFATAL: Service error during creation. Check logs.")
            sys.exit(1)

        cached_pct = (result["cached_tokens"] / max(result["prompt_tokens"], 1)) * 100
        conversations.append({"messages": messages + [{"role": "assistant", "content": result["content"]}],
                              "initial_time": elapsed,
                              "prompt_tokens": result["prompt_tokens"]})
        print(f"  conv {i:2d}: {elapsed:.2f}s  prompt={result['prompt_tokens']}  cached={cached_pct:.0f}%")

    print(f"\n  All {args.conversations} conversations created.")
    print(f"  Device slots should now be full (first 4-8 on device, rest on host).")
    print()

    # Phase 2: Force H2D by sending follow-ups (needs free device slot → triggers D2H eviction)
    print("--- Phase 2: Follow-ups (force H2D, trigger D2H eviction of pinned states) ---")
    hits = 0
    total = 0
    for i, conv in enumerate(conversations):
        followup = conv["messages"] + [{"role": "user", "content": "Continue."}]
        start = time.time()
        result = send_request(base_url, followup, timeout=args.timeout)
        elapsed = time.time() - start

        if "error" in result:
            print(f"  conv {i:2d}: ERROR: {result['error']}")
            sys.exit(1)

        total += 1
        cached_pct = (result["cached_tokens"] / max(result["prompt_tokens"], 1)) * 100
        is_hit = result["cached_tokens"] > result["prompt_tokens"] * 0.5
        if is_hit:
            hits += 1
        status = "HIT " if is_hit else "MISS"
        print(f"  conv {i:2d}: {elapsed:.2f}s ({status}) cached={cached_pct:.0f}% "
              f"[initial was {conv['initial_time']:.2f}s]")

    print(f"\n  Follow-up results: {hits}/{total} cache hits")
    print()

    # Phase 3: Add more conversations to pressure the host pool
    print("--- Phase 3: Additional conversations (pressure host slot pool) ---")
    extra = max(0, args.conversations + 8)
    for i in range(args.conversations, extra):
        text = random_text(args.words // 2)
        messages = [{"role": "user", "content": text}]
        start = time.time()
        result = send_request(base_url, messages, timeout=args.timeout)
        elapsed = time.time() - start

        if "error" in result:
            print(f"  conv {i:2d}: ERROR: {result['error']}")
            sys.exit(1)

        cached_pct = (result["cached_tokens"] / max(result["prompt_tokens"], 1)) * 100
        print(f"  conv {i:2d}: {elapsed:.2f}s  prompt={result['prompt_tokens']}  cached={cached_pct:.0f}%")

    print()
    if hits == total:
        print("=== RESULT: ALL HIT — memory pressure handled correctly ===")
    else:
        print(f"=== RESULT: {hits}/{total} HIT — some misses under pressure ===")
        print("  Check service log for [FATAL] or [can_release_strict] messages.")


if __name__ == "__main__":
    main()
