#!/usr/bin/env python3
"""Does a valuable conversation survive a flood of disposable ones?

Builds ONE important conversation - deep (many accumulated turns) and repeatedly reused (every turn
is a follow-up, so its own state is re-read) - then floods the pool with uniform, freshly created
sessions until the capacity ladder starts retiring owners, and checks that the important session is
still there and still reusable.

This is the user-facing property the victim score exists for: on 2026-09-23 a 237k-token
conversation was retired while freshly created test sessions stayed, because the last-resort step
picked the longest-untouched owner. Value must dominate; among equally valuable owners the least
recently active one goes first, so a just-inserted cache is not the first to die either.

PASS requires all of:
  * the flood actually caused retirements (otherwise the run proves nothing);
  * none of them named the important session (its digest in no `[evict]` line, and it is still
    listed by /slots with its checkpoints);
  * a final follow-up on the important conversation still reuses > 80% of its prompt.

Usage: test_important_session_survival.py --port 30000 [--turns 8] [--turn-tokens 2500]
                                          [--flood 40] [--keep 0]
"""

import argparse
import json
import os
import re
import sys
import time
import urllib.error
import urllib.request

SERVE_LOG = (os.environ.get("NINFER_SERVICE_LOG")
             or os.environ.get("NINFER_SERVE_LOG")
             or "")
REQUEST_LOG = os.environ.get("NINFER_REQUEST_LOG") or ""


def post(base: str, messages: list, max_tokens: int = 8, timeout: int = 600) -> dict:
    payload = json.dumps({"model": "myai", "messages": messages, "stream": False,
                          "max_completion_tokens": max_tokens, "temperature": 0}).encode()
    request = urllib.request.Request(base + "/v1/chat/completions", data=payload,
                                     headers={"Content-Type": "application/json"})
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    body["_elapsed"] = time.time() - started
    return body


def usage_of(response: dict) -> tuple:
    usage = response.get("usage") or {}
    prompt = usage.get("prompt_tokens") or 0
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
    return prompt, cached


def read_log() -> str:
    if not SERVE_LOG:
        return ""
    try:
        with open(SERVE_LOG, errors="ignore") as handle:
            return handle.read()
    except OSError:
        return ""


def occupancy() -> int:
    """Latest host_state_slots from the request log, or -1 when unavailable."""
    if not REQUEST_LOG:
        return -1
    best = -1
    try:
        with open(REQUEST_LOG, errors="ignore") as handle:
            for line in handle:
                if '"occupancy"' not in line:
                    continue
                try:
                    slot = json.loads(line)["context_cache"]["occupancy"]["host_state_slots"]
                except Exception:
                    continue
                if isinstance(slot, int):
                    best = slot
    except OSError:
        return -1
    return best


def slots_view(port: int) -> list:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/slots", timeout=30) as response:
        return json.loads(response.read())


def filler(tokens: int, seed: str) -> str:
    return f"{seed} " + "token " * max(0, tokens)


def main() -> int:
    parser = argparse.ArgumentParser(description="Important-session survival under pool pressure")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--turns", type=int, default=8, help="turns that make the context deep")
    parser.add_argument("--turn-tokens", type=int, default=2500, help="filler tokens per turn")
    parser.add_argument("--flood", type=int, default=40, help="disposable sessions to send")
    parser.add_argument("--keep", type=int, default=0,
                        help="leave N disposable sessions after the check (0 = none)")
    args = parser.parse_args()

    base = f"http://127.0.0.1:{args.port}"
    print("=== Important session survival ===")
    print(f"turns={args.turns} x ~{args.turn_tokens} tokens, flood={args.flood} sessions")
    print(f"occupancy before: host_state_slots={occupancy()}")

    log_before = read_log()

    # --- 1. the valuable conversation: deep context, reused every turn -------------------------
    messages = [{"role": "system", "content": "You are myai, a coding assistant."}]
    digest = ""
    print("\n--- building the important conversation ---")
    for turn in range(args.turns):
        messages.append({"role": "user", "content": filler(args.turn_tokens, f"important turn {turn}")})
        response = post(base, messages)
        digest = response.get("session_digest") or digest
        prompt, cached = usage_of(response)
        print(f"  turn {turn}: prompt={prompt:>6} cached={cached:>6} "
              f"({100.0 * cached / prompt if prompt else 0:5.1f}%) {response['_elapsed']:5.2f}s")
        messages.append({"role": "assistant", "content": "ack"})
    if not digest:
        print("FAIL: the service returned no session digest for the important conversation")
        return 1
    print(f"  digest={digest}")

    # --- 2. flood with disposable sessions until the ladder retires owners ---------------------
    print(f"\n--- flooding with {args.flood} disposable sessions ---")
    retirements = 0
    for index in range(args.flood):
        disposable = [{"role": "user", "content": filler(1200, f"disposable {index}")},
                      {"role": "assistant", "content": "ack"},
                      {"role": "user", "content": filler(600, f"disposable {index} more")}]
        try:
            post(base, disposable, max_tokens=1)
        except urllib.error.HTTPError as error:
            print(f"FAIL: disposable session {index} answered HTTP {error.code}: "
                  f"{error.read().decode(errors='replace')[:160]}")
            return 1
        window = read_log()[len(log_before):]
        retirements = len(re.findall(r"^\[evict\] slot=", window, re.M))
        if retirements >= 3 and index >= 12:
            print(f"  stopped after {index + 1} sessions: {retirements} retirements")
            break
    window = read_log()[len(log_before):]
    evictions = [line for line in window.splitlines() if line.startswith("[evict] slot=")]
    fallbacks = [line for line in window.splitlines() if line.startswith("[exhaust]")]
    print(f"  retirements during the flood: {len(evictions)}")
    for line in evictions[-4:]:
        print("   ", line[:140])
    for line in fallbacks[-4:]:
        print("   ", line[:140])

    # --- 3. verdict ----------------------------------------------------------------------------
    print("\n--- verdict ---")
    survivors = {entry["session_digest"]: entry for entry in slots_view(args.port)
                 if entry.get("session_digest")}
    named = [line for line in evictions if digest in line]
    still_there = survivors.get(digest)
    if len(evictions) == 0:
        print("INCONCLUSIVE: the flood caused no retirement, so nothing was tested "
              "(raise --flood or check that the pool is small enough to saturate)")
        return 1
    if named:
        print(f"FAIL: the important session was retired:\n  {named[0][:160]}")
        return 1
    if still_there is None:
        print("FAIL: the important session is no longer listed by /slots")
        return 1
    print(f"  important session still resident: slot={still_there['id']} "
          f"depth={still_there['n_prompt_tokens']} checkpoints={len(still_there['checkpoints'])}")
    unused_order = [line for line in fallbacks if "source=oldest" in line]
    print(f"  fallback retirements: {len(fallbacks)} "
          f"(from the score order: {len(fallbacks) - len(unused_order)}, oldest-touched: "
          f"{len(unused_order)})")

    messages.append({"role": "user", "content": "Reply with exactly: IMPORTANT STILL HERE"})
    response = post(base, messages, max_tokens=16)
    prompt, cached = usage_of(response)
    ratio = 100.0 * cached / prompt if prompt else 0.0
    print(f"  follow-up after the flood: prompt={prompt} cached={cached} ({ratio:.1f}%) "
          f"{response['_elapsed']:.2f}s")
    if ratio <= 80.0:
        print(f"FAIL: the important conversation no longer reuses its context ({ratio:.1f}% <= 80%)")
        return 1
    print("\nPASS: the valuable conversation survived the flood and still reuses its context.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
