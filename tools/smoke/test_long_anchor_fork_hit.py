#!/usr/bin/env python3
"""Long-anchor fork-hit test: a fork from a mid-history user turn must hit an anchor.

Reproduces the production failure shape: a conversation keeps growing (its retained long
anchors used to slide into the newest turns), and a later fork re-sends the history truncated
at an OLDER user turn. With the shallowest-anchor retention policy every retained anchor sat
past the fork point, so the fork re-prefilled its entire prompt (0% cache, ~2 minutes on a
120k prompt). With user-turn anchor candidates and uniform-spacing retention the fork must
restore at the anchor below its fork point instead:

  1. Create a long conversation (R user/assistant rounds, each round unique text).
  2. Grow it with two extra rounds so the anchor set has a reason to spread.
  3. Fork: resend the history truncated at round F (< R) plus one new user message.
     Acceptance: the server's own `cached_tokens` must cover a clear majority of the fork
     prompt (it restores at the round-F anchor and re-prefills only the new message),
     not 0 (which means the fork fell back to a root prefill).
  4. Control: continue the full conversation with one more turn; the endpoint covers it, so
     this must stay a near-total hit.

Usage:
    python3 tools/smoke/test_long_anchor_fork_hit.py --base-url http://127.0.0.1:30000/v1
    python3 tools/smoke/test_long_anchor_fork_hit.py --base-url http://127.0.0.1:30000/v1 \
        --rounds 10 --fork-round 5 --words 300 --log /path/to/ninfer_serve.log
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


def words(count: int, seed: int) -> str:
    rng = random.Random(seed)
    return " ".join("".join(rng.choices(string.ascii_lowercase, k=5)) for _ in range(count))


def chat(base_url: str, messages, max_tokens: int = 8, timeout: float = 600.0):
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
    return (
        time.monotonic() - started,
        cached_prompt_tokens(usage),
        usage.get("prompt_tokens", 0),
    )


def rounds(conv_id: int, count: int, words_per_round: int):
    """User/assistant message pairs with unique text per round (round 0 = first user turn)."""
    messages = []
    for round_index in range(count):
        messages.append(
            {
                "role": "user",
                "content": f"forkprobe {conv_id} round {round_index}. "
                + words(words_per_round, conv_id * 10000 + round_index),
            }
        )
        messages.append(
            {
                "role": "assistant",
                "content": f"ack {conv_id}.{round_index} "
                + words(20, conv_id * 10000 + round_index + 777),
            }
        )
    return messages


def ratio(cached: int, total: int) -> float:
    return cached / total if total else 0.0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--rounds", type=int, default=8, help="conversation rounds to build")
    parser.add_argument(
        "--fork-round",
        type=int,
        default=4,
        help="user-turn index to fork from (must be < --rounds)",
    )
    parser.add_argument("--words", type=int, default=250, help="unique words per user round")
    parser.add_argument("--log", default=None, help="service log path (for the run window note)")
    args = parser.parse_args()
    if not (1 <= args.fork_round < args.rounds):
        parser.error("--fork-round must satisfy 1 <= fork-round < rounds")

    base_url = args.base_url
    seed = int(time.time()) % 100000
    history = rounds(seed, args.rounds, args.words)

    print(f"building {args.rounds}-round conversation ({args.words} unique words/round) ...")
    seconds, cached, prompt = chat(base_url, history)
    print(f"  build: {seconds:6.2f}s  prompt={prompt:6d}  cached={cached:6d}")

    # Grow the conversation past the fork point so the retained anchor set has room to spread.
    for extra in range(2):
        tail = history[-1:]
        tail[0] = {
            "role": "user",
            "content": f"forkprobe {seed} extra {extra}. "
            + words(args.words, seed * 10000 + 500 + extra),
        }
        grown = history[:-1] + tail
        seconds, cached, prompt = chat(base_url, grown)
        print(f"  grow {extra + 1}/2: {seconds:6.2f}s  prompt={prompt:6d}  cached={cached:6d}")

    # Fork from the mid-history user turn: resend the history through the fork point plus one
    # new user message the engine has never seen.
    fork_prompt = history[: 2 * args.fork_round + 1] + [
        {
            "role": "user",
            "content": f"forkprobe {seed} fork. " + words(args.words, seed * 10000 + 900),
        }
    ]
    seconds, cached, prompt = chat(base_url, fork_prompt)
    hit = ratio(cached, prompt)
    print(
        f"  fork@{args.fork_round}: {seconds:6.2f}s  prompt={prompt:6d}  "
        f"cached={cached:6d}  ({hit:.0%})"
    )
    if args.log:
        print(f"  (engine window: {args.log} from this run's start; check reuse-diag verdicts)")

    # Control: continue the grown conversation from its endpoint.
    tail = history[-1:]
    tail[0] = {
        "role": "user",
        "content": f"forkprobe {seed} continue. " + words(args.words, seed * 10000 + 600),
    }
    seconds, cached, prompt = chat(base_url, history[:-1] + tail)
    endpoint_hit = ratio(cached, prompt)
    print(f"  continue:    {seconds:6.2f}s  prompt={prompt:6d}  cached={cached:6d}  ({endpoint_hit:.0%})")

    failures = 0
    # The fork must restore at the anchor below its fork point: a clear majority of its prompt
    # is cached, and the re-prefilled suffix is the new message only (well under two rounds).
    if hit < 0.5:
        print(f"FAIL fork hit {hit:.0%} < 50%: the fork re-prefilled its history (root fallback)")
        failures += 1
    if cached <= 0:
        print("FAIL fork cached_tokens is 0: no anchor served the fork at all")
        failures += 1
    # The uncached suffix of the control is the new user message itself (the endpoint covers
    # everything before it); with short prompts that message is a large share of the total, so
    # 80% - not 90% - is the floor that still separates "endpoint works" from "root re-prefill".
    if endpoint_hit < 0.8:
        print(f"FAIL control hit {endpoint_hit:.0%} < 80%: the endpoint no longer covers the tail")
        failures += 1
    if failures:
        return 1
    print("ok: mid-history fork hit its user-turn anchor; the endpoint still covers the tail")
    return 0


if __name__ == "__main__":
    sys.exit(main())
