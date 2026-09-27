#!/usr/bin/env python3
"""Prefix cache reuse test: verify that idle conversations can be quickly restored
after an active conversation has been running and potentially evicted their device
state replicas.

Usage:
    python3 tools/smoke/test_prefix_reuse_switch.py --port 30000

Configuration (via flags or env):
    --port          Service port (default: 30000)
    --conversations Number of conversations to create (default: 9)
    --random-words  Words per conversation's initial message (default: 800, ~8K tokens)
    --active-rounds Rounds for the last conversation to be "active" (default: 12)
    --timeout       Per-request timeout in seconds (default: 120)

What it tests:
    1. Create N conversations, each with ~8K random tokens (unique per conversation)
    2. Make conversation N very active (many rounds of back-and-forth)
    3. Switch back to conversation 1
    4. Verify: conversation 1 should get a cache hit (its state survived in host RAM)

    If the LRU eviction + fair-share mechanism is working:
      - Conversation N's activity may evict other conversations' DEVICE replicas
      - But their HOST replicas are protected (fair-share)
      - Switching back to conversation 1: LRU frees a device slot, H2D load from host
      - Result: high cache hit rate, sub-second TTFT

    If broken (e.g., host replica evicted, or no LRU fallback):
      - Conversation 1's state is lost entirely
      - Full prefill required (seconds to minutes depending on prompt size)
      - Cache hit = 0%

Exit code: 0 = PASS (cache hit on switch-back), 1 = FAIL (full prefill)
"""

import argparse
import json
import random
import string
import sys
import time
import os
import urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness_paths import cached_prompt_tokens  # noqa: E402  (tools/smoke/harness_paths.py)


def random_text(n_words):
    """Generate random text with the given number of words."""
    words = []
    for _ in range(n_words):
        words.append("".join(random.choices(string.ascii_lowercase, k=random.randint(3, 10))))
    return " ".join(words)


def send_request(base_url, messages, max_tokens=50, timeout=120):
    """Send a chat completion request and return (cached_tokens, prompt_tokens, elapsed)."""
    data = json.dumps({
        "model": "myai",
        "messages": messages,
        "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(
        base_url, data=data, headers={"Content-Type": "application/json"}
    )
    start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            result = json.loads(resp.read())
        elapsed = time.time() - start
        usage = result.get("usage", {})
        cached = cached_prompt_tokens(usage)
        prompt = usage.get("prompt_tokens", 0)
        return cached, prompt, elapsed
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ERROR: {e}", file=sys.stderr)
        return 0, 0, elapsed


def main():
    parser = argparse.ArgumentParser(description="Prefix cache reuse switch-back test")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--conversations", type=int, default=9)
    parser.add_argument("--random-words", type=int, default=800)
    parser.add_argument("--active-rounds", type=int, default=12)
    parser.add_argument("--warmup-turns", type=int, default=3,
                        help="turns each conversation runs on ITSELF before the busy phase: self-reuse "
                             "is what raises its value (evidence 1/8 -> ~0.5, live(x) 1 -> up to 8) "
                             "without spending any pool space")
    parser.add_argument("--timeout", type=int, default=120)
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    n_conv = args.conversations
    active_conv = n_conv  # The last conversation is the "active" one

    print(f"=== Prefix Cache Reuse Test ===")
    print(f"  Conversations: {n_conv}")
    print(f"  Random words per conversation: {args.random_words}")
    print(f"  Active rounds for conv {active_conv}: {args.active_rounds}")
    print(f"  Service: http://127.0.0.1:{args.port}")
    print()

    # Phase 1: Create all conversations, then READ each one once more.
    #
    # The second read is what makes each conversation "reused", and that is the whole difference in the
    # engine's value formula (缓存模块v2.md §2.2):
    #
    #   score = rebuild_value(tokens) x retention_weight x live(x) x reuse_evidence
    #   从未被复用  : live = 1, evidence = 1/8      -> score ∝ tokens, nothing else
    #   被复用过的  : live 可达 8, evidence ≈ 0.5   -> 每 token 贵 4~24 倍
    #
    # A conversation that was only ever CREATED is a never-reused record, so the pool's cheapest thing
    # is simply the smallest such record - and the value order correctly deletes it when the pool is
    # full (measured 2026-09-27: this case's conversation was deleted at 3 011 tokens, at 16 525, and
    # still at 20 553, because the fill's own never-reused sessions are the same size and marginally
    # larger). Enlarging the conversation is the WRONG lever: it spends the pool (4 x 20.5K = 82K of
    # the rig's 131K-token KV pool) and the next step then has nowhere to go (important_session got a
    # 503 queue timeout in the same run). The right lever is evidence: one extra read per conversation
    # costs one round trip and no pool space, and makes every conversation in this case more valuable
    # per token than anything the pool may sacrifice.
    print(f"--- Phase 1: Create {n_conv} conversations, then read each one once ---")
    conversations = {}
    for i in range(1, n_conv + 1):
        content = f"CONV{i}: " + random_text(args.random_words)
        messages = [{"role": "user", "content": content}]
        cached, prompt, elapsed = send_request(base_url, messages, timeout=args.timeout)
        conversations[i] = messages + [{"role": "assistant", "content": f"CONV{i} ack."}]
        status = "✓" if cached > 0 else " "
        print(f"  Conv{i}: prompt={prompt} cached={cached} time={elapsed:.1f}s {status}")
        # Evidence turns: the conversation reuses ITSELF a few times. That is what raises its value,
        # and it is the cheapest possible way to do so in the engine's formula (§2.2):
        #   evidence: 1/8 (never reused, no timestamp) -> ~0.5 once it has a last-use time
        #   live(x):  1  (never reused)               -> up to 8 while fresh
        # i.e. ~32x more value per token, for a few round trips and NO extra pool space. Enlarging the
        # conversation instead spends the pool (measured: 4 x 20.5K = 82K of the rig's 131K-token KV
        # pool, which starved the next battery step into an HTTP 503).
        for turn in range(1, args.warmup_turns + 1):
            follow_up = conversations[i] + [
                {"role": "user", "content": f"CONV{i} warm-up turn {turn}"}]
            cached2, prompt2, elapsed2 = send_request(base_url, follow_up, max_tokens=8,
                                                      timeout=args.timeout)
            conversations[i] = follow_up + [{"role": "assistant", "content": f"CONV{i} ack {turn}."}]
            pct2 = cached2 * 100 // max(prompt2, 1)
            print(f"          evidence turn {turn}: prompt={prompt2} cached={cached2} "
                  f"({pct2}%) time={elapsed2:.1f}s")

    # Phase 2: Make the active conversation very busy
    print(f"\n--- Phase 2: Conversation {active_conv} active ({args.active_rounds} rounds) ---")
    msgs = list(conversations[active_conv])
    for round_num in range(2, args.active_rounds + 2):
        msgs.append({"role": "user", "content": f"CONV{active_conv} r{round_num}: " + random_text(100)})
        msgs.append({"role": "assistant", "content": f"CONV{active_conv} response r{round_num}."})
        cached, prompt, elapsed = send_request(base_url, msgs[:-1], max_tokens=20, timeout=args.timeout)
        pct = cached * 100 // max(prompt, 1)
        status = "✓" if pct > 50 else " "
        print(f"  Round{round_num:2d}: prompt={prompt:6d} cached={cached:6d} ({pct:3d}%) time={elapsed:.1f}s {status}")

    # Phase 3: Switch back to conversation 1
    print(f"\n--- Phase 3: Switch back to Conversation 1 ---")
    test_msgs = conversations[1] + [
        {"role": "user", "content": f"CONV1: follow-up after {args.active_rounds} rounds of other activity"}
    ]
    cached, prompt, elapsed = send_request(base_url, test_msgs, max_tokens=20, timeout=args.timeout)
    pct = cached * 100 // max(prompt, 1) if prompt else 0

    print(f"\n=== RESULT ===")
    print(f"  Prompt:    {prompt} tokens")
    print(f"  Cached:    {cached} tokens ({pct}%)")
    print(f"  TTFT:      {elapsed:.2f}s")
    print()

    if cached > 0:
        print("  PASS: Cache hit on switch-back. LRU eviction + fair-share working correctly.")
        print("        The idle conversation's state survived in host RAM despite the active")
        print("        conversation's device pressure, and was restored quickly.")
        return 0
    else:
        print("  FAIL: No cache hit. Full prefill was required.")
        print("        Possible causes:")
        print("          - Host state replica was evicted (fair-share not protecting this session)")
        print("          - No free device slot available and LRU eviction failed")
        print("          - Digest mismatch (separate bug: tokenization/prompt construction)")
        print("          - State was DeviceOnly (no host replica) and device slot was lost")
        return 1


if __name__ == "__main__":
    sys.exit(main())
