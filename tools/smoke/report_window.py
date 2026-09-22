#!/usr/bin/env python3
"""Summarize one measurement window of a running service, from the Engine's own two logs.

Usage: report_window.py <label> <serve-log-bytes> <request-log-bytes>

Both arguments are byte offsets taken with `stat -c%s` on the respective file *before* the window
starts. They are two different files and must not be swapped or shared: `nfinfer_serve.log` carries
the HTTP status and fatal lines, while `request_log.jsonl` carries the per-request record
(prompt/hit/reuse path/TTFT). Reading one file at the other's offset reports an empty window and
makes a clean run look like "no requests happened".

Per request the line is::

    req <id> prompt=<n> hit=<n> (<pct>%) path=<reuse path> ttft=<s> prefill=<s> plan=<ms> ...

`path` is the reuse path the Engine selected (`root` means no reuse). A sibling request that hits
its own anchor from the previous request in the same conversation shows
`path=private_long_anchor` (or `private_endpoint` / `private_response_replay`) with a high `pct`.
"""
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from harness_paths import REQUEST_LOG, SERVICE_LOG  # noqa: E402  (tools/smoke/harness_paths.py)


def main() -> None:
    if len(sys.argv) != 4:
        raise SystemExit(__doc__)
    label = sys.argv[1]
    serve_offset, request_offset = int(sys.argv[2]), int(sys.argv[3])

    with open(REQUEST_LOG, errors="replace") as fh:
        fh.seek(request_offset)
        text = fh.read()
    with open(SERVICE_LOG, errors="replace") as fh:
        fh.seek(serve_offset)
        serve_text = fh.read()

    requests, captures, pressure = [], Counter(), Counter()
    for line in text.splitlines():
        line = line.strip()
        if not line.startswith("{"):
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        event = record.get("event")
        if event == "request_done":
            result = record.get("result") or {}
            materialization = record.get("materialization") or {}
            timings = record.get("timings_seconds") or {}
            prompt = result.get("prompt_tokens") or 0
            hit = result.get("prefix_cache_hit_tokens") or 0
            requests.append(
                {
                    "id": (record.get("request") or {}).get("request_id"),
                    "prompt": prompt,
                    "hit": hit,
                    "pct": 100.0 * hit / prompt if prompt else 0.0,
                    "path": result.get("prefix_reuse_path"),
                    "ttft": timings.get("ttft") or 0.0,
                    "prefill": timings.get("prefill") or 0.0,
                    "plan_ms": (materialization.get("planning_elapsed_ns") or 0) / 1e6,
                    "search_ms": (materialization.get("search_elapsed_ns") or 0) / 1e6,
                    "targets": materialization.get("targets_evaluated"),
                    "stop": materialization.get("stop_reason"),
                    "best": materialization.get("best_reuse_prompt_tokens"),
                    "queue": (record.get("engine_timing") or {}).get("queue_wait_seconds"),
                }
            )
        elif event in ("request_start", "throughput"):
            cache = record.get("context_cache") or {}
            capture = cache.get("captures") or {}
            captures["completed"] += capture.get("completed", 0)
            captures["aborted"] += capture.get("aborted", 0)
            for key, value in (cache.get("pressure") or {}).items():
                if value:
                    pressure[key] += value

    print(f"=== {label} ===")
    for q in requests:
        print(
            f"  req {str(q['id']):>5} prompt={q['prompt']:>7} hit={q['hit']:>7} ({q['pct']:5.1f}%) "
            f"path={str(q['path']):<24} ttft={q['ttft']:6.2f}s prefill={q['prefill']:6.2f}s "
            f"plan={q['plan_ms']:7.1f}ms search={q['search_ms']:7.1f}ms targets={q['targets']} "
            f"stop={q['stop']} best={q['best']}"
        )
    if requests:
        ttfts = [q["ttft"] for q in requests]
        pcts = [q["pct"] for q in requests]
        print(
            f"  --> requests={len(requests)} "
            f"ttft min/med/max={min(ttfts):.2f}/{sorted(ttfts)[len(ttfts) // 2]:.2f}/{max(ttfts):.2f}s "
            f"cache min/med={min(pcts):.1f}/{sorted(pcts)[len(pcts) // 2]:.1f}% "
            f"paths={dict(Counter(q['path'] for q in requests))}"
        )
    print(f"  --> captures={dict(captures)}")
    if pressure:
        print(f"  --> pressure={dict(pressure)}")
    # An error only counts when it is inside the window: the serve log is append-only across
    # restarts, so absolute counts would attribute an older incident to this run.
    errors = {key: serve_text.count(key) for key in
              ("HTTP 500", "HTTP 503", "[engine] fatal", "gave up after", "capacity miss site=",
               "catalog: clear retired")}
    print(f"  --> serve log {errors}")
    failed = [key for key, count in errors.items() if count and key in ("HTTP 500", "HTTP 503", "[engine] fatal")]
    if failed:
        print(f"  --> FAIL window contains {', '.join(failed)}")


if __name__ == "__main__":
    main()
