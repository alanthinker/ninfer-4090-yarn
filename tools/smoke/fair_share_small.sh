#!/usr/bin/env bash
# Small-parameter A/B for the fair-share escalation path, on a deliberately tight instance.
#
# The pools are small enough that a couple of short conversations fill them (dense anchors, so a
# short conversation already publishes several checkpoints) but real enough to discriminate: the
# reuse sources are genuine checkpoints and the workload mixes device-resident and host-only
# owners. The point is the state production hits at host 320/320 - a reuse is offered, the pools
# have nowhere to put the restore, and the only owners able to give capacity are the ones the
# fair-share buckets protect.
#
# Fail fast: any HTTP error or timeout aborts with the message instead of retrying.
#
# Usage: tools/smoke/fair_share_small.sh <label> <NINFER_BIN>
set -euo pipefail
LABEL="${1:?label}"
BIN="${2:?binary}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.." || exit 1

export NINFER_BIN="$BIN"
export NINFER_HARNESS_DIR="/tmp/fs-small-$LABEL"
rm -rf "$NINFER_HARNESS_DIR"
mkdir -p "$NINFER_HARNESS_DIR"
export NINFER_REQDUMP_DIR="$NINFER_HARNESS_DIR/reqdump"
export NINFER_REQUEST_LOG="$NINFER_HARNESS_DIR/request_log.jsonl"
export NINFER_SERVICE_LOG="$NINFER_HARNESS_DIR/ninfer_serve_agent.log"
export AGENT_PORT="${AGENT_PORT:-30005}"
export AGENT_HOST_SLOTS="${AGENT_HOST_SLOTS:-3}" AGENT_DEV_SLOTS="${AGENT_DEV_SLOTS:-1}"
export AGENT_HOST_KV_MIB="${AGENT_HOST_KV_MIB:-48}"
export AGENT_MAX_CTX="${AGENT_MAX_CTX:-16384}" AGENT_KV_CAPACITY="${AGENT_KV_CAPACITY:-16384}"
export AGENT_PRIVATE="${AGENT_PRIVATE:-6}" AGENT_CONCURRENCY="${AGENT_CONCURRENCY:-2}"
export AGENT_FAIR_BUCKETS="${AGENT_FAIR_BUCKETS:-2}"
export AGENT_FIRST_SPACING="${AGENT_FIRST_SPACING:-256}" AGENT_ANCHOR_SPACING="${AGENT_ANCHOR_SPACING:-256}"
export AGENT_AUTO_ANCHORS="${AGENT_AUTO_ANCHORS:-8}" AGENT_MAX_ANCHORS="${AGENT_MAX_ANCHORS:-8}"
export AGENT_SHARED_PREFIXES="${AGENT_SHARED_PREFIXES:-4}"
export FSL_LABEL="$LABEL" FSL_TIMEOUT="${FSL_TIMEOUT:-90}"

cleanup() { "$HERE/ninfer_service_agent.sh" stop >/dev/null 2>&1 || true; }
trap cleanup EXIT

"$HERE/ninfer_service_agent.sh" stop >/dev/null 2>&1 || true
"$HERE/agent_wait_gpu.sh" >/dev/null 2>&1 || true
"$HERE/ninfer_service_agent.sh" start >/dev/null
for _ in $(seq 1 30); do
    curl -sf -m 2 "http://127.0.0.1:$AGENT_PORT/health" >/dev/null && break
    sleep 1
done
curl -sf -m 3 "http://127.0.0.1:$AGENT_PORT/health" >/dev/null || {
    echo "$LABEL: instance did not become healthy"; exit 1; }
echo "== $LABEL: instance up (host slots $AGENT_HOST_SLOTS, device slots $AGENT_DEV_SLOTS, fair-share $AGENT_FAIR_BUCKETS) =="

python3 - <<'PY'
import json
import os
import time
import urllib.error
import urllib.request

label   = os.environ["FSL_LABEL"]
base    = f"http://127.0.0.1:{os.environ['AGENT_PORT']}"
timeout = float(os.environ["FSL_TIMEOUT"])

def post(messages, tag):
    payload = json.dumps({"model": "myai", "messages": messages, "stream": False,
                          "max_completion_tokens": 1, "temperature": 0}).encode()
    request = urllib.request.Request(base + "/v1/chat/completions", data=payload,
                                     headers={"Content-Type": "application/json"})
    started = time.time()
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            out = json.loads(response.read())
    except urllib.error.HTTPError as error:
        body = error.read().decode(errors="replace")[:200]
        raise SystemExit(f"{label}: request {tag} failed with HTTP {error.code}: {body}")
    except Exception as error:
        raise SystemExit(f"{label}: request {tag} failed: {type(error).__name__}: {error}")
    usage  = out.get("usage", {})
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    prompt = usage.get("prompt_tokens", 0) or 1
    print(f"  {tag:<14} prompt={prompt:>6} cache={cached:>6} ({100.0 * cached / prompt:5.1f}%) "
          f"{time.time() - started:5.2f}s", flush=True)
    return cached, prompt

# One long constant block shared by every conversation: the reusable prefix production refuses.
FSL_STABLE = "myai coding agent. " + ("stable instruction and tool schema token " * 900)

def conversation(session, turns, words=600):
    # The live client's shape: one long constant block that every conversation shares (its
    # developer prompt and tools), then per-session turns. The reusable prefix is therefore a
    # SHARED stable identity - the checkpoint that survives dense-anchor churn - which is what
    # production refuses to restore when the pools have nowhere to put it.
    out = [{"role": "system", "content": FSL_STABLE}]
    for turn in range(turns):
        out.append({"role": "user", "content": f"session{session} turn{turn} " + ("token " * words)})
        out.append({"role": "assistant", "content": f"ack {session}/{turn}"})
    return out

results = []
for session in range(1, 4):
    for turn in range(1, 5):
        results.append(post(conversation(session, 4)[: 1 + 2 * turn], f"s{session}.t{turn}"))
# Siblings: the same history with a new final user message - exactly what the anchors are for.
for round_index in range(3):
    for session in range(1, 4):
        messages = conversation(session, 4)[:9]
        messages.append({"role": "user",
                         "content": f"session{session} sibling r{round_index} " + ("token " * 600)})
        results.append(post(messages, f"s{session}.sib{round_index}"))

text   = open(os.environ["NINFER_SERVICE_LOG"], errors="ignore").read()
shares = sorted(100.0 * cached / prompt for cached, prompt in results)
print(f"  请求 {len(results)} 条 | 0% 命中 {sum(1 for c, _ in results if c == 0)} 条"
      f" | 命中率中位 {shares[len(shares) // 2]:.1f}%")
print("  fair-share released:", text.count("fair-share released"),
      "| maximal fallback:", text.count("maximal fallback"),
      "| offered-but-root:", text.count("but planned from root"),
      "| 500/503/fatal:", text.count("HTTP 500"), text.count("HTTP 503"),
      text.count("[engine] fatal"))
PY
