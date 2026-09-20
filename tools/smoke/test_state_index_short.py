#!/usr/bin/env python3
"""Test StateIndex: 20 different 8K short conversations, then re-query the first 5.

Each conversation is a unique 8K-token prompt with one request.
After all 20 complete, we send a second request to conversations 0-4.
If the StateIndex (or 8K first anchor + catalog) works, those should get
significant cache hits (TTFT much lower than the initial request).

Usage:
    python3 test_state_index_short.py --base-url http://127.0.0.1:8123/v1
    python3 test_state_index_short.py --base-url http://127.0.0.1:8123/v1 --words 4000
"""

import argparse
import json
import random
import string
import sys
import time
import urllib.request


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


def chat(base_url: str, messages: list, model: str = "default") -> dict:
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
    print("--- Phase 1: Initial requests (one per conversation) ---")
    prompts = []
    results_initial = []
    for i in range(args.conversations):
        prompt = make_prompt(i, args.words)
        prompts.append(prompt)
        messages = [{"role": "user", "content": prompt}]
        t0 = time.monotonic()
        resp = chat(args.base_url, messages)
        ttft = resp["_elapsed_s"]
        results_initial.append(ttft)
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

    # Phase 3: Re-query the first N conversations (follow-up)
    print(f"--- Phase 2: Follow-up requests (first {args.verify} conversations) ---")
    results_followup = []
    for i in range(args.verify):
        messages = [
            {"role": "user", "content": prompts[i]},
            {"role": "assistant", "content": f"SITH{i:04d}END"},
            {"role": "user", "content": make_followup(i)},
        ]
        resp = chat(args.base_url, messages)
        ttft = resp["_elapsed_s"]
        results_followup.append(ttft)
        hit = "HIT " if ttft < results_initial[i] * 0.5 else "MISS"
        print(f"  conv {i:2d}: {ttft:8.2f}s ({hit})  [initial was {results_initial[i]:.2f}s]")
        time.sleep(0.1)

    print()
    hits = sum(1 for i in range(args.verify) if results_followup[i] < results_initial[i] * 0.5)
    print(f"=== RESULT: {hits}/{args.verify} cache hits ===")
    if hits == args.verify:
        print("All verified conversations got cache hits. StateIndex working.")
    elif hits > 0:
        print(f"Partial: {hits}/{args.verify} hits. Some states were not found.")
    else:
        print("No cache hits. States were not preserved (expected without StateIndex/8K anchor).")

    return 0 if hits == args.verify else 1


if __name__ == "__main__":
    sys.exit(main())
