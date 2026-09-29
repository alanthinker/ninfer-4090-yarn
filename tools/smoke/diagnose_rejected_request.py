#!/usr/bin/env python3
"""Find the requests the service REJECTED and hand back their bodies, unchanged.

Why this exists (2026-09-29): two chat requests were rejected with
    invalid_prompt: "failed to normalize UTF-8 text as NFC: Invalid UTF-8 string"
and the first instinct was to scan the dump bodies for invalid UTF-8. That scan found nothing -
because the bodies were perfectly valid. `reqdump` writes the body BYTE FOR BYTE BEFORE the
request is handled, so a failing request is always on disk and its encoding is always clean; the
invalid bytes were produced inside the server while it split the prompt at byte boundaries that
fell inside multi-byte characters. The lesson is: read the dumped request, do not just scan it.

How this differs from `replay_dump.py`, so the two do not grow into each other:
  * `replay_dump.py` takes dump paths you already know, REWRITES each payload (stream=false,
    one completion token) and can fire several in parallel - it exists to stress a pool and
    reproduce a crash.
  * this tool starts from the `request_rejected` records in the runtime request log, so it
    answers "WHICH request failed, with what shape", then re-sends the body BYTE FOR BYTE, which
    is what a "is this still broken?" question needs: a rewritten body would not have failed in
    the same way.
When you already know the file and want it re-sent as-is, `replay_dump.py <port> <dump.json>`
with `--max-tokens` set to the original value is the shorter path.

Usage:
    python3 tools/smoke/diagnose_rejected_request.py                 # list rejected requests
    python3 tools/smoke/diagnose_rejected_request.py --replay 551    # re-send that body unchanged
    python3 tools/smoke/diagnose_rejected_request.py --replay 551 --port 30000
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
import time
import urllib.request
from pathlib import Path

RUNTIME_DIR = Path("/root/ai/large_models/_ninfer_repos/deploy-yarn")
REQUEST_LOG = RUNTIME_DIR / "request_log.jsonl"
DUMP_DIR = Path(os.environ.get("NINFER_REQDUMP_DIR") or RUNTIME_DIR / "reqdump")

ROUTE_BY_PROTOCOL = {
    "openai_chat_completions": "chat",
    "openai_responses": "responses",
    "anthropic_messages": "messages",
}


def rejected_requests() -> list[dict]:
    """Every `request_rejected` record in the runtime request log, oldest first."""
    rows: list[dict] = []
    if not REQUEST_LOG.exists():
        return rows
    with REQUEST_LOG.open(errors="replace") as handle:
        for line in handle:
            if "request_rejected" not in line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def dump_for(request_id: int, protocol: str) -> Path | None:
    """The dumped body for one request id, whatever millisecond it was captured in."""
    route = ROUTE_BY_PROTOCOL.get(protocol, "chat")
    matches = sorted(glob.glob(str(DUMP_DIR / f"*-req-{request_id}-{route}.json")))
    # The pending name appears as "...-req-<stamp>-s<id>-<route>.json" while the request runs.
    matches += sorted(glob.glob(str(DUMP_DIR / f"*-req-*-s{request_id}-{route}.json")))
    return Path(matches[-1]) if matches else None


def describe(path: Path) -> None:
    raw = path.read_bytes()
    print(f"  body      : {path.name}  ({len(raw):,} bytes)")
    try:
        raw.decode("utf-8")
        print("  encoding  : valid UTF-8 (the failure was produced INSIDE the server, not by "
              "the client)")
    except UnicodeDecodeError as error:
        start = max(0, error.start - 40)
        print(f"  encoding  : INVALID UTF-8 at byte {error.start} ({error.reason})")
        print(f"  context   : {raw[start:error.start + 40]!r}")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception as error:  # noqa: BLE001 - diagnostic path, any failure is the finding
        print(f"  json      : unparsable ({error})")
        return
    messages = payload.get("messages") or []
    blocks = [block for message in messages
              for block in (message.get("content") if isinstance(message.get("content"), list)
                            else [])
              if isinstance(block, dict)]
    media = [block for block in blocks if block.get("type") not in (None, "text")]
    print(f"  shape     : {len(messages)} messages, {len(media)} media blocks, "
          f"{len(payload.get('tools') or [])} tools")
    non_ascii = sum(1 for message in messages if isinstance(message.get("content"), str)
                    and any(ord(ch) > 127 for ch in message["content"]))
    print(f"  non-ASCII : {non_ascii} message(s) carry multi-byte text "
          f"(the byte-boundary case needs exactly that)")


def replay(path: Path, port: int, timeout: int) -> int:
    body = path.read_bytes()
    request = urllib.request.Request(
        f"http://127.0.0.1:{port}/v1/chat/completions", data=body,
        headers={"Content-Type": "application/json"})
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            status, text = response.status, response.read(600).decode("utf-8", "replace")
    except urllib.error.HTTPError as error:
        status, text = error.code, error.read(600).decode("utf-8", "replace")
    except Exception as error:  # noqa: BLE001
        print(f"  replay    : transport failure after {time.time() - started:.1f}s: {error}")
        return 1
    print(f"  replay    : HTTP {status} after {time.time() - started:.1f}s")
    if status != 200:
        print(f"  response  : {text.strip()[:400]}")
        return 1
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--replay", type=int, metavar="REQUEST_ID",
                        help="replay the dumped body of this request id against the service")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--timeout", type=int, default=600)
    args = parser.parse_args()

    rows = rejected_requests()
    print(f"rejected requests in {REQUEST_LOG}: {len(rows)}")
    for row in rows[-8:]:
        request = row.get("request") or {}
        stamp = time.strftime("%m-%d %H:%M:%S",
                              time.localtime(row.get("timestamp_unix_ms", 0) / 1000))
        print(f"  req#{request.get('request_id')}  {stamp}  "
              f"{(row.get('error') or {}).get('code')}: "
              f"{(row.get('error') or {}).get('message', '')[:70]}")
        print(f"    shape: {request.get('message_count')} msgs, "
              f"{request.get('media_item_count')} media, {request.get('tool_count')} tools")

    if args.replay is None:
        return 0

    row = next((item for item in rows
                if (item.get("request") or {}).get("request_id") == args.replay), None)
    if row is None:
        print(f"\nno rejected request recorded with id {args.replay}")
        return 1
    protocol = (row.get("request") or {}).get("protocol", "openai_chat_completions")
    path = dump_for(args.replay, protocol)
    print(f"\nreq#{args.replay}:")
    if path is None:
        print("  body      : NOT FOUND - it aged out of the dump directory "
              f"({DUMP_DIR}); raise NINFER_DUMP_REQUESTS_LIMIT / "
              "NINFER_DUMP_REQUESTS_MAX_AGE_HOURS to keep more")
        return 1
    describe(path)
    return replay(path, args.port, args.timeout)


if __name__ == "__main__":
    sys.exit(main())
