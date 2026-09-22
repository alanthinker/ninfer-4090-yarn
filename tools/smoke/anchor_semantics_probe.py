#!/usr/bin/env python3
"""Measure which prefix a branch at each message boundary can actually reuse.

Answers a question about anchor placement with the server's own reuse decision instead of the
anchor rule's prose: build an alternating conversation, then send variants that change exactly one
message (or append to the end) and read the frontier/path the server reports.

Each variant is sent twice: the first send is the cold-ish measurement, the second shows the
steady state after the variant's own checkpoints exist.

Usage: anchor_semantics_probe.py [port] [log-path]
"""
import json
import os
import re
import sys
import time
import urllib.request

PORT = sys.argv[1] if len(sys.argv) > 1 else "30000"
LOG = sys.argv[2] if len(sys.argv) > 2 else os.environ.get(
    "NINFER_SERVE_LOG", "/root/ai/large_models/_ninfer_repos/deploy-yarn/ninfer_serve.log")
BASE = f"http://127.0.0.1:{PORT}/v1/chat/completions"

# ~1.2K tokens per message, distinctive per (turn, role): a changed message is a real divergence.
BODY = ("alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu " * 40)[:900]


def message(turn: int, role: str, salt: str = "") -> dict:
    text = f"[{salt}{role.upper()}-TURN-{turn}] " + BODY
    return {"role": role, "content": text}


def conversation(edits: dict, append: bool = False) -> list:
    """u1 a1 u2 a2 u3 a3 u4, with `edits` replacing the message at that 1-based index."""
    roles = ["user", "assistant", "user", "assistant", "user", "assistant", "user"]
    msgs = []
    for index, role in enumerate(roles, start=1):
        if index in edits:
            msgs.append(message(index, role, salt=f"{edits[index]}-"))
        else:
            msgs.append(message(index, role))
    if append:
        msgs.append(message(8, "assistant"))
        msgs.append(message(9, "user", salt="NEW-"))
    return msgs


def log_size() -> int:
    try:
        return os.path.getsize(LOG)
    except OSError:
        return 0


def server_decision(offset: int) -> str:
    try:
        with open(LOG, "r", errors="replace") as handle:
            handle.seek(offset)
            tail = handle.read()
    except OSError:
        return ""
    lines = [l for l in tail.splitlines() if "done | openai-chat" in l]
    if not lines:
        return "(no server line)"
    match = re.search(r"prompt ([\d,]+) \|.*?(cache [^|]+)\| TTFT ([0-9.]+ ?m?s)", lines[-1])
    return f"prompt {match.group(1):>7} | {match.group(2).strip():34} | TTFT {match.group(3)}" \
        if match else lines[-1][-110:]


def post(messages: list) -> None:
    payload = json.dumps({"model": "myai", "messages": messages, "stream": False,
                          "max_completion_tokens": 1, "temperature": 0}).encode()
    request = urllib.request.Request(BASE, data=payload,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=1800) as response:
        response.read()


def main() -> None:
    # Establish the baseline conversation, then let it settle.
    baseline = conversation({})
    post(baseline)
    time.sleep(0.5)

    # (label, messages) — the label says which message differs from the baseline conversation.
    variants = [
        ("identical", conversation({})),
        ("edit last user (u4)", conversation({7: "EDITED"})),
        ("edit u3", conversation({5: "EDITED"})),
        ("edit a2", conversation({4: "EDITED"})),
        ("edit u2", conversation({3: "EDITED"})),
        ("edit u1", conversation({1: "EDITED"})),
        ("append a8+u9 (continuation)", conversation({}, append=True)),
    ]
    for label, messages in variants:
        for round_index in (1, 2):
            offset = log_size()
            started = time.time()
            post(messages)
            print(f"{label:30} #{round_index}  {time.time() - started:5.2f}s  "
                  f"{server_decision(offset)}", flush=True)


if __name__ == "__main__":
    main()
