#!/usr/bin/env python3
"""Context-cache pressure test: multi-turn conversations, eviction, and restore.

Why multi-turn: the engine places automatic long anchors only on *message boundaries* and only
when a request carries more than one message (see frontend.cpp, `message_count > 1`). A batch of
single-message prompts therefore produces no interior anchors at all, no matter how small the
anchor spacing is, so it cannot exercise incremental capture or anchor reuse. This script builds
conversations with alternating user/assistant turns so that:

  * each conversation accumulates several long anchors (`--first-anchor-spacing` controls how far
    apart they may be),
  * a returning conversation can reuse an anchor in the middle of its prompt instead of the
    endpoint only,
  * eviction has real checkpoints to demote, retain, or drop.

Three phases:

  1. Create N multi-turn conversations (each round is a user message plus a canned assistant reply).
  2. Send M unrelated single-turn conversations to push the first ones out of the catalog.
  3. Re-query the first K conversations with one more turn and report the server-side cache hit.

Acceptance is the server's own `cached_tokens` and the TTFT, not a ratio against a baseline that
may already be warm.

Usage:
    python3 tools/smoke/test_context_cache_pressure.py --base-url http://127.0.0.1:30000/v1
    python3 tools/smoke/test_context_cache_pressure.py --base-url http://127.0.0.1:30001/v1 \
        --conversations 20 --rounds 6 --words 300 --pressure 20 --verify 3

To relate the run to what the engine did, point --log at the service log and the script reports
the eviction count, the peak host/device slot occupancy, the host KV usage, and any exhaustion
drops recorded while it ran.
"""

import argparse
import json
import random
import re
import string
import sys
import time
import urllib.request


def words(count: int, seed: int) -> str:
    rng = random.Random(seed)
    return " ".join("".join(rng.choices(string.ascii_lowercase, k=5)) for _ in range(count))


def chat(base_url: str, messages, max_tokens: int = 8, timeout: float = 300.0):
    body = json.dumps(
        {
            "model": "myai",
            "messages": messages,
            "max_tokens": max_tokens,
            "stream": False,
            "chat_template_kwargs": {"enable_thinking": False},
        }
    ).encode()
    request = urllib.request.Request(
        base_url.rstrip("/") + "/chat/completions", body, {"Content-Type": "application/json"}
    )
    started = time.monotonic()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    usage = payload.get("usage", {})
    details = usage.get("prompt_tokens_details", {}) or {}
    return (
        time.monotonic() - started,
        details.get("cached_tokens", 0),
        usage.get("prompt_tokens", 0),
    )


def conversation(conv_id: int, rounds: int, words_per_round: int):
    """A conversation with `rounds` message boundaries, so anchors have positions to land on."""
    messages = [{"role": "user", "content": f"conv{conv_id} start. " + words(words_per_round, conv_id * 1000)}]
    for round_index in range(1, rounds):
        messages.append(
            {"role": "assistant", "content": f"ack {conv_id}.{round_index} " + words(15, conv_id * 1000 + round_index)}
        )
        messages.append(
            {
                "role": "user",
                "content": f"turn {round_index}. " + words(words_per_round, conv_id * 1000 + round_index + 500),
            }
        )
    return messages


def log_mark(path: str) -> int:
    """Line count of the service log now, so a run can summarize only its own window."""
    try:
        with open(path, "r", errors="replace") as handle:
            return sum(1 for _ in handle)
    except OSError:
        return -1


def summarize_log(path: str, mark: int):
    """Report what the engine did during this run, from the log mark taken at start."""
    if mark < 0:
        print("  (no readable service log; pass --log to summarize engine activity)")
        return
    try:
        with open(path, "r", errors="replace") as handle:
            lines = handle.readlines()[mark:]
    except OSError as error:
        print(f"  (log not readable: {error})")
        return
    evictions = [line for line in lines if "[evict] slot=" in line]
    exhaust = [line for line in lines if line.startswith("[exhaust]")]
    errors = [line for line in lines if "HTTP 500" in line or "HTTP 503" in line]

    def peak(pattern: str):
        values = [int(match.group(1)) for line in evictions if (match := re.search(pattern, line))]
        return max(values) if values else 0

    print("  --- engine activity (this run) ---")
    print(f"  evictions            : {len(evictions)}")
    print(f"  peak host state slots: {peak(r'host_state=(\d+)/')}")
    device = [match.group(1) for line in evictions if (match := re.search(r'device_state=\d+/(\d+)', line))]
    print(f"  peak device state    : {peak(r'device_state=(\d+)/')}/{max((int(v) for v in device), default=0)}")
    print(f"  peak host KV         : {peak(r'host_kv=(\d+)MiB')} MiB")
    print(f"  peak anchors held    : {peak(r'anchors=(\d+)')}")
    print(f"  exhaustion drops     : {len(exhaust)}")
    print(f"  HTTP 500/503 lines   : {len(errors)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Context-cache pressure test (multi-turn)")
    parser.add_argument("--base-url", default="http://127.0.0.1:30000/v1")
    parser.add_argument("--conversations", type=int, default=20, help="multi-turn conversations")
    parser.add_argument("--rounds", type=int, default=6, help="message rounds per conversation")
    parser.add_argument("--words", type=int, default=400, help="words per user turn")
    parser.add_argument("--pressure", type=int, default=20,
                        help="unrelated single-turn conversations sent between phase 1 and 3")
    parser.add_argument("--verify", type=int, default=5, help="how many early conversations to requery")
    parser.add_argument("--log", default="", help="service log to summarize after the run")
    args = parser.parse_args()

    print("=== Context-cache pressure test ===")
    print(f"  service       : {args.base_url}")
    print(f"  conversations : {args.conversations} x {args.rounds} rounds x {args.words} words")
    print(f"  pressure      : {args.pressure} unrelated conversations before requerying")
    print()

    mark = log_mark(args.log) if args.log else -1
    prompts = [conversation(i, args.rounds, args.words) for i in range(args.conversations)]
    print("--- Phase 1: multi-turn conversations (anchor capture) ---")
    baseline = []
    for index, messages in enumerate(prompts):
        ttft, cached, prompt = chat(args.base_url, messages)
        baseline.append((ttft, prompt))
        print(f"  conv {index:2d}: {ttft:6.2f}s  prompt {prompt:6d}  cached {cached:6d}")
    if args.pressure:
        print()
        print(f"--- Phase 2: {args.pressure} unrelated conversations (force eviction) ---")
        started = time.monotonic()
        for index in range(args.pressure):
            chat(
                args.base_url,
                [{"role": "user", "content": f"filler{index}. " + words(800, 90000 + index)}],
            )
        print(f"  done in {time.monotonic() - started:.0f}s")

    print()
    print("--- Phase 3: requery the first conversations (restore path) ---")
    hits = 0
    for index in range(min(args.verify, args.conversations)):
        messages = prompts[index] + [{"role": "user", "content": "repeat the last turn number"}]
        ttft, cached, prompt = chat(args.base_url, messages)
        ratio = 100.0 * cached / max(prompt, 1)
        hit = ratio >= 50.0
        hits += 1 if hit else 0
        print(f"  conv {index:2d}: {ttft:6.2f}s  cache {cached:6d}/{prompt:6d} ({ratio:5.1f}%)  "
              f"{'HIT ' if hit else 'MISS'}")
    print()
    print(f"=== RESULT: {hits}/{min(args.verify, args.conversations)} rescued from cache ===")
    if args.log:
        print()
        summarize_log(args.log, mark)
    return 0


if __name__ == "__main__":
    sys.exit(main())
