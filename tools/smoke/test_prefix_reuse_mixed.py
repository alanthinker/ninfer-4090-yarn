#!/usr/bin/env python3
"""Mixed-workload prefix cache reuse test: varying conversation lengths,
round counts, and repeated switching patterns.

Validates that the state management (LRU device eviction + fair-share host
protection + H2D/D2H transfer) works correctly under realistic mixed loads:

  - Conversations of very different lengths (short/medium/long/very-long)
  - Different activity levels per conversation
  - Repeated non-sequential switching (A→C→B→D→A→C)
  - Memory pressure: multiple large conversations competing for device slots
  - H2D/D2H transfer: states demoted to host must be reloadable on demand

Usage:
    python3 tools/smoke/test_prefix_reuse_mixed.py --port 30000

Exit code: 0 = all switches hit cache, 1 = at least one switch missed.
"""

import argparse
import json
import random
import string
import sys
import time
import urllib.request


def random_text(n_words):
    """Generate random text with the given number of words."""
    words = []
    for _ in range(n_words):
        words.append("".join(random.choices(string.ascii_lowercase, k=random.randint(3, 10))))
    return " ".join(words)


def send_request(base_url, messages, max_tokens=30, timeout=300):
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
        cached = usage.get("prompt_tokens_details", {}).get("cached_tokens", 0)
        prompt = usage.get("prompt_tokens", 0)
        return cached, prompt, elapsed
    except Exception as e:
        elapsed = time.time() - start
        print(f"  ERROR: {e}", file=sys.stderr)
        return 0, 0, elapsed


class Conversation:
    """Track a conversation's message history."""

    def __init__(self, name, initial_words):
        self.name = name
        self.initial_words = initial_words
        self.messages = []
        # Initial user message
        content = f"{name}: " + random_text(initial_words)
        self.messages.append({"role": "user", "content": content})
        self.rounds = 0

    def add_round(self, round_num, extra_words=50):
        """Add a user message and a canned assistant response (simulating history)."""
        user_content = f"{self.name} r{round_num}: " + random_text(extra_words)
        self.messages.append({"role": "user", "content": user_content})
        assistant_content = f"{self.name} response r{round_num}."
        self.messages.append({"role": "assistant", "content": assistant_content})
        self.rounds = round_num

    def next_request_messages(self, extra_user=None):
        """Build the message list for the next request."""
        msgs = list(self.messages)
        if extra_user:
            msgs.append({"role": "user", "content": extra_user})
        return msgs


def run_conversation_rounds(base_url, conv, n_rounds, label=""):
    """Send sequential rounds for a conversation, expecting cache hits on rounds 2+."""
    print(f"  {label} {conv.name} ({conv.initial_words}w, {n_rounds} rounds):")
    all_hit = True
    for r in range(1, n_rounds + 1):
        conv.add_round(r)
        # Remove the assistant we just added (engine will generate its own)
        conv.messages.pop()  # remove assistant
        msgs = list(conv.messages)
        cached, prompt, elapsed = send_request(base_url, msgs, max_tokens=20)
        pct = cached * 100 // max(prompt, 1) if prompt else 0
        # Put the assistant back for history tracking
        conv.messages.append({"role": "assistant", "content": f"{conv.name} response r{r}."})
        if r == 1:
            status = "init"
        elif pct > 50:
            status = "HIT "
        else:
            status = "MISS"
            all_hit = False
        print(f"    r{r:2d}: prompt={prompt:6d} cached={cached:6d} ({pct:3d}%) {elapsed:.2f}s {status}")
    return all_hit


def switch_to(base_url, conv, extra_user=None, timeout=300):
    """Switch to a conversation (send its full history + optional new message)."""
    if extra_user:
        conv.messages.append({"role": "user", "content": extra_user})
        conv.rounds += 1
    msgs = list(conv.messages)
    cached, prompt, elapsed = send_request(base_url, msgs, max_tokens=20, timeout=timeout)
    pct = cached * 100 // max(prompt, 1) if prompt else 0
    return cached, prompt, elapsed, pct


def main():
    parser = argparse.ArgumentParser(description="Mixed-workload prefix cache reuse test")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--timeout", type=int, default=300)
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}/v1/chat/completions"
    print("=== Mixed-Workload Prefix Cache Reuse Test ===")
    print(f"  Service: http://127.0.0.1:{args.port}")
    print()

    # Define conversations with very different sizes and activity levels
    # SHORT: ~200 words (~800 tokens), 2 rounds
    # MEDIUM: ~600 words (~2400 tokens), 5 rounds
    # LONG: ~1200 words (~4800 tokens), 8 rounds
    # VERY-LONG: ~2000 words (~8000 tokens), 3 rounds (large but few rounds)
    convs = {
        "S": Conversation("SHORT", 200),
        "M": Conversation("MEDIUM", 600),
        "L": Conversation("LONG", 1200),
        "XL": Conversation("VERYLONG", 2000),
    }

    print("--- Phase 1: Create conversations (varying sizes) ---")
    for key, conv in convs.items():
        cached, prompt, elapsed = send_request(base_url, conv.messages, max_tokens=20,
                                                timeout=args.timeout)
        print(f"  {key:3s} ({conv.initial_words:4d}w): prompt={prompt:5d} time={elapsed:.1f}s")
        conv.messages.append({"role": "assistant", "content": f"{conv.name} ack."})
        conv.rounds = 0

    print()
    print("--- Phase 2: Activity bursts (different rounds per conversation) ---")
    all_rounds_hit = True
    # Very-long gets the most device pressure (large prompts, multiple rounds)
    all_rounds_hit &= run_conversation_rounds(base_url, convs["XL"], 3, label="XL")
    all_rounds_hit &= run_conversation_rounds(base_url, convs["L"], 5, label="L")
    all_rounds_hit &= run_conversation_rounds(base_url, convs["M"], 3, label="M")
    all_rounds_hit &= run_conversation_rounds(base_url, convs["S"], 2, label="S")

    print()
    print("--- Phase 3: Repeated non-sequential switching ---")
    # Switch pattern: L → S → XL → M → L → S → XL
    # Each switch should hit the cache (state preserved in host or device)
    switch_sequence = [
        ("L", "LONG: what did we discuss earlier?"),
        ("S", "SHORT: back again, any updates?"),
        ("XL", "VERYLONG: continuing our long discussion"),
        ("M", "MEDIUM: checking in on this thread"),
        ("L", "LONG: another follow-up question"),
        ("S", "SHORT: quick final check"),
        ("XL", "VERYLONG: wrapping up this conversation"),
    ]

    all_switches_hit = True
    for i, (key, extra_msg) in enumerate(switch_sequence):
        conv = convs[key]
        cached, prompt, elapsed, pct = switch_to(base_url, conv, extra_user=extra_msg,
                                                  timeout=args.timeout)
        status = "HIT " if pct > 50 else "MISS"
        if pct <= 50:
            all_switches_hit = False
        print(f"  Switch {i+1} → {key:3s}: prompt={prompt:5d} cached={cached:5d} "
              f"({pct:3d}%) {elapsed:.2f}s {status}")

    # Phase 4: Go back to the FIRST switch target to test repeated return
    print()
    print("--- Phase 4: Return to previously-visited conversations ---")
    for key in ["L", "S", "XL"]:
        conv = convs[key]
        extra = f"{conv.name}: returning after multiple other conversations"
        cached, prompt, elapsed, pct = switch_to(base_url, conv, extra_user=extra,
                                                  timeout=args.timeout)
        status = "HIT " if pct > 50 else "MISS"
        if pct <= 50:
            all_switches_hit = False
        print(f"  Return → {key:3s}: prompt={prompt:5d} cached={cached:5d} "
              f"({pct:3d}%) {elapsed:.2f}s {status}")

    # Summary
    print()
    print("=" * 60)
    print("=== SUMMARY ===")
    print("=" * 60)
    results = [
        ("Phase 2: Sequential rounds", all_rounds_hit),
        ("Phase 3+4: Switch-back hits", all_switches_hit),
    ]
    all_pass = all(ok for _, ok in results)
    for name, ok in results:
        print(f"  {'PASS' if ok else 'FAIL'}: {name}")

    if all_pass:
        print("\n  ALL TESTS PASSED")
        print("  Mixed-size conversations, repeated switching, and H2D/D2H")
        print("  state transfers are all working correctly.")
        return 0
    else:
        print("\n  SOME TESTS FAILED")
        print("  Check service log for reuse-diag output to identify which")
        print("  states were evicted or failed to match.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
