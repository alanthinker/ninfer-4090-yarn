#!/usr/bin/env python3
"""Test prefix reuse: N short conversations, then re-query the first few.

Each conversation is a unique prompt with one request. After all of them complete we wait (the user
steps away) and re-query the first few: those should come back as cache hits from the retained
checkpoints.

Verdict under a SATURATED pool (2026-09-23): an all-hit expectation is not always satisfiable by
design. With the pool at its ceiling, the engine may legitimately (a) skip a capture it cannot place
(`capture: skip frontier=... reason=static-infeasible|capacity-*`) or (b) evict the oldest idle
conversation to make room (`[evict] ... session=<digest>`). Both are designed degradation, but a
state that disappears with NO such line in the serve log is a real retention bug. So when
NINFER_SERVICE_LOG/NINFER_SERVE_LOG is set, each miss must be explained by one of those two lines
(same conversation digest, or a captured frontier within a few tokens of that conversation's prompt
length); unexplained misses, or more than half the verifications missing, still fail.

Usage:
    python3 test_state_index_short.py --base-url http://127.0.0.1:8123/v1
    python3 test_state_index_short.py --base-url http://127.0.0.1:8123/v1 --words 4000
"""

import argparse
import json
import os
import random
import re
import string
import sys
import time
import urllib.request

SERVE_LOG = os.environ.get("NINFER_SERVICE_LOG") or os.environ.get("NINFER_SERVE_LOG") or ""


def cached_prompt_tokens(response: dict) -> int:
    """The engine's own reuse accounting: how much of this prompt came from retained state."""
    usage = response.get("usage") or {}
    details = usage.get("prompt_tokens_details") or {}
    return details.get("cached_tokens") or 0


def read_serve_log() -> str:
    if not SERVE_LOG:
        return ""
    try:
        with open(SERVE_LOG, errors="ignore") as handle:
            return handle.read()
    except OSError:
        return ""


def explain_miss(window: str, prompt_tokens: int, digest: str) -> list:
    """Design reasons the engine may legitimately have no state for this conversation."""
    reasons = []
    frontier = max(0, prompt_tokens - 5)   # the capture lands a few tokens before the prompt end
    for line in window.splitlines():
        skipped = re.search(r"capture: skip frontier=(\d+) reason=([\w-]+)", line)
        if skipped and abs(int(skipped.group(1)) - frontier) <= 16:
            reasons.append(f"capture skipped ({skipped.group(2)}) at frontier {skipped.group(1)}")
        if digest and line.startswith("[evict]") and f"session={digest}" in line:
            reasons.append("session evicted for capacity ([evict])")
    return sorted(set(reasons))


def make_prompt(conv_id: int, words: int) -> str:
    """Generate a unique ~`words` word prompt for conversation conv_id."""
    rng = random.Random(conv_id * 7919)  # deterministic per conversation
    parts = [
        f"You are assistant #{conv_id}. This is conversation {conv_id} "
        f"of a batch test. Remember this number: {conv_id * 1000 + 7}. "
        f"The magic phrase is 'SITH{conv_id:04d}END'. "
        f"Today's date is 2026-09-20. "
        f"Answer the question at the end with a single word.\n"
    ]
    # Fill with unique filler text to reach ~`words` words
    word_count = len(" ".join(parts).split())
    while word_count < words:
        line_words = 30 + rng.randint(0, 20)
        line = " ".join(
            "".join(rng.choices(string.ascii_lowercase, k=4 + rng.randint(0, 6)))
            for _ in range(line_words)
        )
        parts.append(f"[C{conv_id} L{len(parts)}] {line}")
        word_count += line_words + 3
    parts.append(
        f"\nQuestion: What was the magic phrase I gave you? "
        f"(Answer with just the phrase.)"
    )
    return "\n".join(parts)


def make_followup(conv_id: int) -> str:
    """A short follow-up that references the original conversation."""
    return (
        f"I'm still in conversation {conv_id}. "
        f"What was the magic phrase? "
        f"(You should remember it from earlier in this conversation.)"
    )


def chat(base_url: str, messages: list, model: str = "myai") -> dict:
    """Send a chat completion request and return the response."""
    url = f"{base_url}/chat/completions"
    body = json.dumps({
        "model": model,
        "messages": messages,
        "max_tokens": 50,
        "temperature": 0,
    }).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    start = time.monotonic()
    with urllib.request.urlopen(req, timeout=300) as resp:
        data = json.loads(resp.read())
    elapsed = time.monotonic() - start
    data["_elapsed_s"] = elapsed
    return data


def main():
    parser = argparse.ArgumentParser(description="StateIndex short conversation test")
    parser.add_argument("--base-url", default="http://127.0.0.1:8123/v1")
    parser.add_argument("--words", type=int, default=4000,
                        help="Target word count per prompt (~8K tokens at ~2 words/token)")
    parser.add_argument("--conversations", type=int, default=20,
                        help="Number of distinct short conversations")
    parser.add_argument("--verify", type=int, default=5,
                        help="How many of the first conversations to re-query")
    args = parser.parse_args()

    print(f"=== StateIndex Short Conversation Test ===")
    print(f"Base URL:    {args.base_url}")
    print(f"Words/prompt: ~{args.words}")
    print(f"Conversations: {args.conversations}")
    print(f"Verify first: {args.verify}")
    print()

    # Phase 1: Send one request to each of the N conversations
    log_before = read_serve_log()
    print("--- Phase 1: Initial requests (one per conversation) ---")
    prompts = []
    results_initial = []
    results_initial_cached = []
    ident = []           # (prompt_tokens, session_digest) per conversation, for the miss verdict
    for i in range(args.conversations):
        prompt = make_prompt(i, args.words)
        prompts.append(prompt)
        messages = [{"role": "user", "content": prompt}]
        t0 = time.monotonic()
        resp = chat(args.base_url, messages)
        ttft = resp["_elapsed_s"]
        results_initial.append(ttft)
        ident.append(((resp.get("usage") or {}).get("prompt_tokens") or 0,
                      resp.get("session_digest") or ""))
        results_initial_cached.append(cached_prompt_tokens(resp))
        print(f"  conv {i:2d}: {ttft:8.2f}s  ({len(prompt.split())} words)")
        time.sleep(0.1)  # small gap to let the server settle

    print()
    print(f"  Initial avg TTFT: {sum(results_initial)/len(results_initial):.2f}s")
    print(f"  Initial max TTFT: {max(results_initial):.2f}s")
    print()

    # Phase 2: Wait a bit (simulate user stepping away)
    gap_seconds = 5
    print(f"--- Waiting {gap_seconds}s before follow-up phase ---")
    time.sleep(gap_seconds)
    print()

    # Phase 3: Re-query the first N conversations (follow-up, most recent first
    # to avoid earlier test requests evicting later ones from the cache)
    print(f"--- Phase 2: Follow-up requests (first {args.verify} conversations, reversed) ---")
    results_followup = []
    for i in range(args.verify - 1, -1, -1):
        messages = [
            {"role": "user", "content": prompts[i]},
            {"role": "assistant", "content": f"SITH{i:04d}END"},
            {"role": "user", "content": make_followup(i)},
        ]
        resp = chat(args.base_url, messages)
        ttft = resp["_elapsed_s"]
        cached = cached_prompt_tokens(resp)
        prompt = (resp.get("usage") or {}).get("prompt_tokens") or 0
        results_followup.append(cached >= 0.5 * prompt if prompt else False)
        verdict = "HIT " if results_followup[-1] else "MISS"
        print(f"  conv {i:2d}: {ttft:8.2f}s ({verdict})  cached {cached:>6}/{prompt:<6} "
              f"({100.0 * cached / prompt if prompt else 0:5.1f}%)  "
              f"[initial {results_initial[i]:.2f}s/{results_initial_cached[i]} cached]")
        time.sleep(0.1)

    print()
    order = list(range(args.verify - 1, -1, -1))
    misses = [order[index] for index in range(args.verify) if not results_followup[index]]
    hits = args.verify - len(misses)
    print(f"=== RESULT: {hits}/{args.verify} cache hits ===")
    if hits == args.verify:
        print("All verified conversations got cache hits. Context cache working.")
        return 0

    window = read_serve_log()[len(log_before):] if log_before is not None else ""
    unexplained = []
    if not SERVE_LOG:
        print(f"Partial: {hits}/{args.verify} hits, and no serve log was given to explain the "
              f"misses (set NINFER_SERVICE_LOG) - treating them as unexplained.")
        unexplained = list(misses)
    else:
        for i in misses:
            reasons = explain_miss(window, ident[i][0], ident[i][1])
            if reasons:
                print(f"  conv {i:2d} missed, explained: {'; '.join(reasons)}")
            else:
                print(f"  conv {i:2d} missed with NO capacity reason in the serve log")
                unexplained.append(i)
    allowed = max(1, args.verify // 2)
    if unexplained:
        print(f"FAIL: {len(unexplained)} state(s) disappeared without a capacity reason "
              f"{unexplained} - that is a retention bug, not designed degradation.")
        return 1
    if len(misses) > allowed:
        print(f"FAIL: {len(misses)}/{args.verify} states missing (more than half) - even with "
              f"capacity reasons this is too much loss to call retention healthy.")
        return 1
    print(f"PASS (degraded): {len(misses)}/{args.verify} states were dropped for capacity, each "
          f"with a reason in the serve log, and {hits}/{args.verify} still hit.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
