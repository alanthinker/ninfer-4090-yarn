#!/usr/bin/env python3
"""Replay a recorded request dump against a running service.

Usage: replay_dump.py <port> <dump.json> [<dump.json> ...] [--max-tokens N] [--parallel N]

--parallel N sends the dumps N at a time (threads), which is how a saturated pool is stressed the
way concurrent client requests stress it.

Prints the server-side result of each request (prompt tokens, elapsed, and the completion text
prefix). Used to reproduce a crash from the request dumps the Engine writes itself, so the exact
client shape that failed can be re-sent instead of approximated.
"""
import json
import sys
import time
import urllib.request

port = sys.argv[1]
files = [a for a in sys.argv[2:] if not a.startswith("--")]
max_tokens = 1
if "--max-tokens" in sys.argv:
    max_tokens = int(sys.argv[sys.argv.index("--max-tokens") + 1])
    files = [f for f in files if f != str(max_tokens)]

def send(path: str) -> None:
    payload = json.load(open(path, encoding="utf-8"))
    payload["stream"] = False
    payload["max_completion_tokens"] = max_tokens
    payload.pop("stream_options", None)
    body = json.dumps(payload).encode()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions",
        data=body,
        headers={"Content-Type": "application/json"},
    )
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=1800) as response:
            out = json.loads(response.read())
        usage = out.get("usage", {})
        text = (out["choices"][0]["message"].get("content") or "")[:40].replace("\n", " ")
        print(
            f"{path.split('/')[-1][:34]:36} prompt={usage.get('prompt_tokens')} "
            f"cached={usage.get('cached_tokens')} {time.time() - started:6.2f}s | {text}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001 - the harness reports whatever the service did
        print(f"{path.split('/')[-1][:34]:36} FAILED {type(exc).__name__}: {exc}", flush=True)


parallel = 1
if "--parallel" in sys.argv:
    parallel = max(1, int(sys.argv[sys.argv.index("--parallel") + 1]))
    files = [f for f in files if f != str(parallel)]

if parallel == 1:
    for path in files:
        send(path)
else:
    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=parallel) as pool:
        list(pool.map(send, files))
