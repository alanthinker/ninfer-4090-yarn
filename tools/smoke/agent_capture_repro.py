#!/usr/bin/env python3
"""Reproduce and measure the context-cache capture decision for the one-shot client shape.

The production incident (2026-09-21): a client sends one-shot chat requests whose prompt is
developer+tools (~8.1K tokens) + its own user message + constant reminder/context blocks. On a
lightly loaded engine the first request publishes an anchor at the tools boundary, so later sibling
requests reuse ~8.45K tokens. Once the host state pool was pinned full, three consecutive requests
missed, including siblings that should have hit the anchor published by the first.

This script drives the same shape against a test instance and prints, per request, the server's
cache hit and (from the instance log, grepped by the caller) capture decisions.

Modes:
  fill N     send N filler requests whose dense anchors fill the small state pools
  cold TAG   send one cold request of the client shape (user message = TAG)
  sibling TAG send a sibling of the previous cold request (same dev+tools, new user message)
  pair       cold then sibling (the two-message acceptance sequence)
  run        [fill F] cold then sibling, printing a verdict

Templates come from reqdump: the newest 4-message dump with the 6812-char developer and 26 tools.
"""
import json
import glob
import os
import sys
import time
import urllib.request

BASE = "http://127.0.0.1:30002"
PORT = os.environ.get("AGENT_PORT", "30002")
BASE = f"http://127.0.0.1:{PORT}"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness_paths import REQDUMP_DIR  # noqa: E402  (tools/smoke/harness_paths.py)


def synthetic_template():
    """The one-shot client shape, synthesized so the harness never depends on reqdump.

    Measured against the real client: developer 6,812 characters, 26 tool schemas, and the
    developer+tools boundary at about 8.1K tokens (so `--first-anchor-spacing 4096` puts the first
    spread anchor exactly there), followed by a constant reminder and runtime-context block.
    """
    developer = (
        "You are an AI agent powered by DeepSeek Harness. The DeepSeek Harness implementation "
        "checkout is at /root/.npm/_npx/2f3a729d991ac520/. The checkout location and current "
        "working directory are separate values and may differ; never infer the working directory "
        "from this path. Use pwd to determine the current working directory. Use the read tool "
        "rather than shell commands like cat to inspect text files. Use glob to find files by path "
        "pattern. Use grep to search file contents. Check the exit code marker on every bash "
        "result. "
    )
    developer = (developer * ((6812 // len(developer)) + 1))[:6812]
    tools = []
    for index in range(26):
        name = f"tool_{index:02d}"
        tools.append(
            {
                "type": "function",
                "function": {
                    "name": name,
                    "description": (
                        f"Tool {index}: operate on the workspace. " + ("Describe behaviour. " * 45)
                    )[:1000],
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "path": {"type": "string", "description": "Path to operate on."},
                            "mode": {"type": "string", "enum": ["read", "write", "append"]},
                        },
                        "required": ["path"],
                    },
                },
            }
        )
    reminder = (
        "<system-reminder> The following workspace instructions may be relevant: "
        + ("Repository rules apply to the whole tree. " * 20)
    )[:911]
    context = ("Current runtime context. This snapshot supersedes earlier snapshots. "
               + ("Filesystem policy and approval state. " * 10))[:446]
    return {
        "messages": [
            {"role": "developer", "content": developer},
            {"role": "user", "content": "placeholder"},
            {"role": "user", "content": reminder},
            {"role": "user", "content": context},
        ],
        "tools": tools,
    }


def newest_template():
    best = None
    best_mtime = 0.0
    for path in glob.glob(os.path.join(str(REQDUMP_DIR), "*.json")):
        try:
            data = json.load(open(path))
        except Exception:
            continue
        msgs = data.get("messages") or []
        if len(msgs) not in (4, 5) or msgs[0].get("role") != "developer":
            continue
        if not data.get("tools"):
            continue
        if any(m.get("role") != "user" for m in msgs[1:]):
            continue
        mtime = os.path.getmtime(path)
        if mtime > best_mtime:
            best, best_mtime = data, mtime
    # A recorded request is preferred (byte-faithful), but the harness must not depend on the
    # request-dump retention window.
    return best if best is not None else synthetic_template()


def post(messages, tools, timeout=600, max_tokens=1):
    payload = json.dumps(
        {
            "model": "myai",
            "messages": messages,
            "tools": tools,
            "stream": False,
            "max_completion_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
    request = urllib.request.Request(
        f"{BASE}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        out = json.loads(response.read())
    usage = out["usage"]
    prompt = usage["prompt_tokens"]
    cached = usage.get("cached_tokens", 0)
    elapsed = time.time() - started
    print(
        f"  prompt={prompt} cached={cached} ({100.0 * cached / max(prompt, 1):.1f}%) "
        f"elapsed={elapsed:.2f}s",
        flush=True,
    )
    return prompt, cached, elapsed


def client_messages(template, user_text, big_tokens=0, salt=None):
    # Faithful to the live client: every constant block first, the variable text LAST. A harness
    # that put the variable text in the middle measured a shape no request produces, and made the
    # deepest reusable prefix look one block shallower than it is (2026-09-22).
    dev = template["messages"][0]["content"]
    if salt is not None:
        # A distinct developer prefix keeps filler anchors from matching the measured shape: the
        # filler must occupy the pools without becoming a reuse candidate for the pair.
        dev = dev[:-16] + f"[filler-{salt}]".ljust(16, ".")
    constants = [{"role": "developer", "content": dev}]
    for message in template["messages"][1:]:
        constants.append({"role": message["role"], "content": message["content"]})
    if big_tokens:
        # Dense filler prompt: long enough to produce many anchors (auto spacing 100) and
        # therefore to occupy the small state pools.
        words = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod " * 40)[
            : big_tokens * 4
        ]
        return constants + [{"role": "user", "content": words}]
    return constants + [{"role": "user", "content": user_text}]


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "run"
    argument = sys.argv[2] if len(sys.argv) > 2 else "tag"
    template = newest_template()
    tools = template["tools"]
    # PAIR_SALT gives the measured pair its own developer prefix, so the first request of a pair is
    # genuinely cold instead of reusing an anchor a previous pair published.
    pair_salt = os.environ.get("PAIR_SALT") or None

    if mode == "fill":
        count = int(argument)
        for index in range(count):
            print(f"fill {index + 1}/{count}", flush=True)
            post(client_messages(template, "", big_tokens=3000, salt=index), tools)
        return

    if mode == "siblings":
        # Sequential sibling sends: each request carries the identical constant blocks and a fresh
        # final user message, which is exactly the live client's shape. Sequential (not
        # concurrent) so every `capture:`/`capture-plan:`/`reuse-diag:` line belongs to one request.
        count = int(argument)
        tag = os.environ.get("SIB_TAG", str(int(time.time())))
        for index in range(count):
            print(f"sibling {index + 1}/{count}", flush=True)
            post(client_messages(template, f"sib-{tag}-{index}"), tools)
            time.sleep(float(os.environ.get("SIB_GAP", "1.0")))
        return

    if mode == "cold":
        print(f"cold {argument}", flush=True)
        post(client_messages(template, argument), tools)
        return

    if mode == "sibling":
        print(f"sibling {argument}", flush=True)
        post(client_messages(template, argument), tools)
        return

    if mode == "pair":
        print(f"cold (salt={pair_salt})", flush=True)
        post(client_messages(template, argument, salt=pair_salt), tools)
        time.sleep(2)
        print("sibling", flush=True)
        prompt, cached, _ = post(client_messages(template, argument + "-sib", salt=pair_salt), tools)
        print("VERDICT hit" if cached > 0 else "VERDICT miss")
        return

    if mode == "run":
        filler = int(os.environ.get("FILL", "0"))
        for index in range(filler):
            print(f"fill {index + 1}/{filler}", flush=True)
            post(client_messages(template, "", big_tokens=3000, salt=index), tools)
        print("cold", flush=True)
        post(client_messages(template, f"cold-{int(time.time())}"), tools)
        time.sleep(2)
        print("sibling", flush=True)
        prompt, cached, _ = post(client_messages(template, f"sib-{int(time.time())}"), tools)
        print("VERDICT hit" if cached > 0 else "VERDICT miss")
        return

    raise SystemExit(f"unknown mode {mode!r}")


if __name__ == "__main__":
    main()
