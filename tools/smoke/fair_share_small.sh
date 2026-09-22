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
# Sizing rule: the small rig must still hold the WORKING SET - four concurrent sessions, the
# production concurrency - comfortably. Only the cache beyond that is small, so the rig fills in
# seconds while the behaviour it exercises stays production-like (a starved working set measures a
# different thing entirely: it cannot even serve four sessions).
#   device state slots = concurrency + device-state-slots (production: 4 + 4 = 8 objects)
#   host state slots  = 4 sessions x ~8 retained checkpoints + headroom (production: 320)
#   host KV / kv capacity: 4 sessions x ~9K tokens with room to restore them (production: 32 GiB)
export AGENT_CONCURRENCY="${AGENT_CONCURRENCY:-4}"
export AGENT_DEV_SLOTS="${AGENT_DEV_SLOTS:-4}"
export AGENT_HOST_SLOTS="${AGENT_HOST_SLOTS:-48}"
export AGENT_HOST_KV_MIB="${AGENT_HOST_KV_MIB:-512}"
export AGENT_MAX_CTX="${AGENT_MAX_CTX:-32768}" AGENT_KV_CAPACITY="${AGENT_KV_CAPACITY:-65536}"
export AGENT_PRIVATE="${AGENT_PRIVATE:-16}" AGENT_FAIR_BUCKETS="${AGENT_FAIR_BUCKETS:-4}"
export AGENT_FIRST_SPACING="${AGENT_FIRST_SPACING:-256}" AGENT_ANCHOR_SPACING="${AGENT_ANCHOR_SPACING:-256}"
export AGENT_AUTO_ANCHORS="${AGENT_AUTO_ANCHORS:-8}" AGENT_MAX_ANCHORS="${AGENT_MAX_ANCHORS:-8}"
export AGENT_SHARED_PREFIXES="${AGENT_SHARED_PREFIXES:-4}"
export FSL_LABEL="$LABEL" FSL_TIMEOUT="${FSL_TIMEOUT:-90}"

cleanup() {
    if [ "${FSL_KEEP:-0}" = "1" ]; then
        echo "== $LABEL: instance left running on port $AGENT_PORT (FSL_KEEP=1) =="
        return 0
    fi
    "$HERE/ninfer_service_agent.sh" stop >/dev/null 2>&1 || true
}
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

STRESS = os.environ.get("FSL_STRESS", "0") == "1"
results = []

def parallel(posts, tag):
    """Four concurrent submissions: the production concurrency, so captures, reclaims and restores
    contend exactly as they do live."""
    import threading
    out = [None] * len(posts)
    failures = []

    def run(index, messages, name):
        try:
            out[index] = post(messages, name)
        except SystemExit as error:      # post() aborts the run on HTTP errors
            failures.append(str(error))

    threads = [threading.Thread(target=run, args=(i, m, f"{tag}.{i}"))
               for i, m in enumerate(posts)]
    for thread in threads: thread.start()
    for thread in threads: thread.join()
    if failures:
        raise SystemExit(f"{label}: concurrent request failed: {failures[0]}")
    return [value for value in out if value is not None]

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

# The agent launcher writes its own JSONL name inside the harness dir.
log_path = os.environ.get("FSL_REQUEST_LOG") or os.path.join(
    os.environ["NINFER_HARNESS_DIR"], "request_log_agent.jsonl")
occupancy = {}
for line in reversed(open(log_path, errors="ignore").read().splitlines()):
    if '"occupancy"' in line:
        occupancy = json.loads(line)["context_cache"]["occupancy"]
        break

if STRESS:
    # 0. Fill the cache beyond the working set: distinct sessions, so the pools saturate and every
    #    later request has to make room. Four sessions are the working set this rig must always be
    #    able to serve; everything past that is cache and must be evictable on demand.
    print("  -- fill past the working set", flush=True)
    for session in range(10, 24):
        for turn in (4,):
            results.append(post(conversation(session, turn, words=500), f"fill.s{session}"))
    # Boundary battery, all at the production concurrency and on the same saturated rig.
    # 1. four concurrent siblings of four different sessions (captures + reclaims at once).
    print("  -- concurrent siblings x4", flush=True)
    for round_index in range(2):
        batches = []
        for session in range(1, 5):
            messages = conversation(session, 4)[:9]
            messages.append({"role": "user",
                             "content": f"session{session} concurrent r{round_index} " + ("token " * 600)})
            batches.append(messages)
        results.extend(parallel(batches, f"conc{round_index}"))
    # 2. edits and forks of a middle turn: the anchors a client rewrites history with.
    print("  -- edit/forks", flush=True)
    for session in (1, 2):
        base_history = conversation(session, 4)
        for fork_turn in (2, 3):
            edited = base_history[: 1 + 2 * fork_turn]
            edited[-1] = {"role": "user",
                          "content": f"session{session} edited turn{fork_turn} " + ("token " * 600)}
            results.append(post(edited, f"s{session}.edit{fork_turn}"))
    # 3. a prompt past the configured context: a clean client error, never a 500.
    print("  -- oversized prompt", flush=True)
    huge = [{"role": "user", "content": "big " + ("token " * 40000)}]
    try:
        post(huge, "oversized")
        raise SystemExit(f"{label}: oversized prompt was accepted")
    except SystemExit as error:
        if "HTTP 400" not in str(error):
            raise
        print("    oversized rejected with HTTP 400 as expected", flush=True)
    if False:
        pass
    # 4. long conversation: deep history, many checkpoints, repeated reuse.
    print("  -- long conversation", flush=True)
    for turn in range(1, 9):
        results.append(post(conversation(5, turn, words=400), f"long.t{turn}"))

text   = open(os.environ["NINFER_SERVICE_LOG"], errors="ignore").read()
shares = sorted(100.0 * cached / prompt for cached, prompt in results)
print(f"  请求 {len(results)} 条 | 0% 命中 {sum(1 for c, _ in results if c == 0)} 条"
      f" | 命中率中位 {shares[len(shares) // 2]:.1f}%")
print("  pool: host_state_slots={}/{} device_state_slots={} host_kv={:.1f}GB".format(
    occupancy.get("host_state_slots"), os.environ["AGENT_HOST_SLOTS"],
    occupancy.get("device_state_slots"), (occupancy.get("host_kv_bytes") or 0) / 1e9))
print("  capacity re-plans:", text.count("re-add after") + text.count("capacity re-plans"),
      "| degrade lines:", text.count("re-admit after progress capacity miss"))
cannot_delete = text.count("retire site=") - text.count("retire site=") # placeholder
retire_failed = len([line for line in text.splitlines()
                     if "retire site=" in line and "ok=0" in line])
demote_failed = text.count("demote begin failed")
allocate_failed = text.count("allocate FAILED")
print(f"  deletions refused: retire ok=0 -> {retire_failed} | demote failed -> {demote_failed} "
      f"| allocate FAILED -> {allocate_failed}")
print("  fair-share released:", text.count("fair-share released"),
      "| maximal fallback:", text.count("maximal fallback"),
      "| offered-but-root:", text.count("but planned from root"),
      "| 500/503/fatal:", text.count("HTTP 500"), text.count("HTTP 503"),
      text.count("[engine] fatal"))
PY
