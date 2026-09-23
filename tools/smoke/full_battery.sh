#!/usr/bin/env bash
# Full-flow stability battery - SATURATE FIRST, and stop pressing the moment it WAS full.
#
# Saturation semantics (09-23): the gate is a HISTORICAL PEAK - host_state_slots >= FULL_AT in
# the occupancy records of THIS battery's time window at any instant counts as full, even if the
# pool has been evicted back down afterwards. Two mechanics make a naive "read after fill" wrong:
#   - occupancy records are written every ~5s, so a mid-session peak can come and go between
#     readings (observed:48/48 held for15s, dropped to17 before the next session-end read);
#   - fill_anchors' big31K/16-anchor sessions are bounded by reserve(kept) + batch(17) <= total,
#     so their session-end readings cycle {17,34} at total=48 and can never show >=46; small
#     sessions (batch<=4) climb linearly and are what actually reach the top.
# The flow: base fill (big sessions, early-stops on peak) -> small-session top-up (stops at the
# first record proving FULL) -> peak gate (fail-closed) -> pressure tests -> correctness tests.
# After saturation the churn itself is the test material: phase5 counts D2H/H2D transfers, and a
# healthy run keeps producing them as sessions evict and restore.
#
# Client-only: the target service must already own :30000 (one GPU, one service). The C++
# suite (ci_smoke/ctest) is excluded on purpose: it needs the VRAM the service holds; run it
# separately with the service stopped.
#
# Usage:
#   tools/smoke/full_battery.sh            # production-scale (pool total 320, deploy-yarn logs)
#   tools/smoke/full_battery.sh --rig      # small rig (pool total 48, agent harness logs)
#   FILL_SESSION_BASE=8 ...                # shift fill sessions so they are not cache replays
#
# --rig sizing facts (diagnosed09-23, five single-variable experiments + code reading):
#   1. --kv-capacity is validated to [max_context, max_concurrency x max_context]; retained
#      checkpoint KV lives in the device pool, so ~3.2 sessions of31K fit at the legal max.
#   2. The state-image object table = device_slots + host_slots (NOT private-driven).
#   3. private/fair/dev/kv knobs do NOT move the fill plateau - the plateau is the batch math
#      above plus eviction churn, which is expected behaviour, not a defect.
set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$HERE/../.." || exit 1

MODE="${1:-prod}"
PORT=30000
if [ "$MODE" = "--rig" ]; then
    TOTAL=48
    export NINFER_SERVICE_LOG="${NINFER_SERVICE_LOG:-$PWD/build/harness/ninfer_serve_agent.log}"
    export NINFER_REQUEST_LOG="${NINFER_REQUEST_LOG:-$PWD/build/harness/request_log_agent.jsonl}"
    # Saturation: fill at MSG780 reaches the ever-full peak (>=46) within cap6 sessions
    # (probe-measured: 5,19,33,43,45,46); the small-session top-up below is the fallback.
    export NINFER_FILL_MSG_TOKENS="${NINFER_FILL_MSG_TOKENS:-780}"
    FILL_CAP="${NINFER_FILL_CAP:-6}"
    PREFIX_TARGET=4096                       # reuse only needs a cacheable prefix, not 9.6K
    # Scale test DATA with test params: trigger conditions are ratios (pool full / cache hit /
    # depth %), not absolute token counts - so small params need only small prompts. Every knob
    # below keeps its assertion semantics: fill keeps its message count (state semantics intact),
    # NIAH keeps its depth percentage, reuse tests keep cache>0 and switch-back hits.
    CONC_SMALL=(4 1024 128)                  # queueing happens at admission, prompt size is irrelevant
    CONC_BIG=(6 1024 128)
    NIAH_TOKENS=4000                         # depth-percent test scales proportionally
    # conversations must fit the fair-share protection (4 buckets): with more, the first-built
    # conversation (the switch-back target) sits outside the protected set and a full pool
    # legitimately evicts it -> "No cache hit". fork_hit keeps its DEFAULTS: at --words100 the
    # fork-point coverage lands at 49.8% against a 50% threshold - scaling saved 5s on a10s
    # test and cost all of the margin.
    SWITCH_ARGS=(--random-words 150 --active-rounds 6 --conversations 4)
    MIXED_ARGS=(--max-rounds 4 --scale 2.0)
    FORK_ARGS=()
else
    TOTAL=320
    FILL_CAP=80
    export NINFER_SERVICE_LOG="${NINFER_SERVICE_LOG:-/root/ai/large_models/_ninfer_repos/deploy-yarn/ninfer_serve.log}"
    export NINFER_REQUEST_LOG="${NINFER_REQUEST_LOG:-/root/ai/large_models/_ninfer_repos/deploy-yarn/request_log.jsonl}"
    # production defaults (A60K/step20K/B12) prefill ~2M tokens - tens of minutes; this keeps
    # the churn/compaction cycle while staying a battery instead of a soak.
    PREFIX_TARGET=9600
    CONC_SMALL=(4 4096 512)
    CONC_BIG=(6 4096 512)
    NIAH_TOKENS=8000
    # production fair buckets =8: the default9 conversations would push the switch-back target
    # (the oldest) outside the protected set on a full pool - same math as the rig fix.
    SWITCH_ARGS=(--conversations 8)
    MIXED_ARGS=()
    FORK_ARGS=()
fi
export NINFER_REQDUMP_DIR="${NINFER_REQDUMP_DIR:-$(dirname "$NINFER_SERVICE_LOG")/reqdump}"
S="$NINFER_SERVICE_LOG"
FULL_AT=$((TOTAL - 2))
ERR_PAT='HTTP 500|HTTP 503|\[engine\] fatal|Segmentation|terminate called'
FAILED_STEPS=0

# Window peak: max host_state_slots over occupancy records newer than this battery's start.
# Implemented in python so a missing/short log or a non-numeric value fails CLOSED (exit 1),
# never passes: the same code guards phase1b and phase2.
window_peak_check() {
    python3 - "$NINFER_REQUEST_LOG" "$FULL_AT" "$BATTERY_START_MS" <<'PY'
import json, sys
log, full_at, since = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
best = -1
try:
    with open(log, errors="ignore") as fh:
        for line in fh:
            if '"occupancy"' not in line:
                continue
            try:
                d = json.loads(line)
                if (d.get("timestamp_unix_ms") or 0) < since:
                    continue
                h = (d.get("context_cache", {}).get("occupancy") or {}).get("host_state_slots")
                if isinstance(h, int) and h > best:
                    best = h
            except Exception:
                continue
except FileNotFoundError:
    pass
print(f"window peak host_state_slots={best} (full_at={full_at})")
sys.exit(0 if best >= full_at else 1)
PY
}

step() {
    local name="$1" to="$2"
    shift 2
    local before rc errs health start end
    before=$(wc -l < "$S" 2>/dev/null || echo 0)
    start=$(date +%s)
    timeout "$to" "$@" > "/tmp/fb_$name.out" 2>&1
    rc=$?
    end=$(date +%s)
    errs=$(tail -n +"$((before + 1))" "$S" | grep -cE "$ERR_PAT" || true)
    health=$(curl -s -m 3 "http://127.0.0.1:$PORT/health" 2>/dev/null || echo DOWN)
    printf '%-24s rc=%-4s %4ss new_err=%-3s health=%s\n' \
        "$name" "$rc" "$((end - start))" "$errs" "$health"
    if [ "${errs:-0}" != "0" ]; then
        tail -n +"$((before + 1))" "$S" | grep -E "$ERR_PAT" | head -3 | sed 's/^/    /'
    fi
    # A suite's own verdict must reach this battery's exit status: a step that printed FAIL and
    # returned 0 (verify_eviction_fix did exactly that until 2026-09-23) showed rc=0 here and the
    # battery still ended green. Count a non-zero return, a new error line, or a dead service.
    if [ "$rc" != "0" ] || [ "${errs:-0}" != "0" ] || [ "$health" = "DOWN" ]; then
        FAILED_STEPS=$((FAILED_STEPS + 1))
        echo "    ^ recorded as FAILED (rc=$rc new_err=${errs:-?} health=$health)"
    fi
    return 0
}

occ_host() {  # last occupancy record's host_state_slots from the (env-selected) request log
    grep -o '"occupancy":{[^}]*}' "$NINFER_REQUEST_LOG" 2>/dev/null | tail -1 |
        grep -o '"host_state_slots":[0-9]*' | tail -1 | sed 's/^.*://'
}

echo "== phase 0: service gate (:$PORT, mode=$MODE, pool total=$TOTAL) =="
curl -sf -m 5 "http://127.0.0.1:$PORT/health" | grep -q ok ||
    { echo "FAIL: no service on :$PORT (start it first - single GPU rule)"; exit 1; }
# Mode-consistency gate: --rig only makes sense against the32768-context rig, and the
# production-scale mode only against a big-context service. Without this, a rig-mode battery
# pointed at production fills the WRONG pool while reading occupancy from the rig's stale log
# (observed: bash-20 accidentally ran the whole rig suite against production :30000).
model_len=$(curl -sf -m 5 "http://127.0.0.1:$PORT/v1/models" |
    python3 -c 'import json,sys; print(json.load(sys.stdin)["data"][0].get("max_model_len",""))' 2>/dev/null || echo "")
case "$MODE" in
    --rig) [ "$model_len" = "32768" ] ||
        { echo "FAIL: --rig mode but service max_model_len=$model_len (not the32768 rig) - wrong instance"; exit 1; } ;;
    *)      [ "$model_len" != "32768" ] ||
        { echo "FAIL: prod mode but service max_model_len=32768 (that is the rig) - use --rig"; exit 1; } ;;
esac
echo "(service max_model_len=$model_len matches mode $MODE)"
# This battery's own window: occupancy peaks and serve-log counters are scoped to it, so a
# FULL reached by an earlier run can never satisfy today's gate, and churn counts are fresh.
BATTERY_START_MS=$(( $(python3 -c 'import time; print(int(time.time() * 1000))') - 2000 ))
BAT_START_LINE=$(wc -l < "$S" 2>/dev/null || echo 0)

echo "== phase 1a: base fill (big sessions; fill_anchors early-stops if the window peak hits FULL) =="
timeout 2400 python3 tools/smoke/fill_anchors.py "$PORT" "$FILL_CAP" "$TOTAL" |
    tee /tmp/fb_fill.out
echo "(base fill ended rc=$?; top-up below carries the gate)"

echo "== phase 1b: top-up small sessions - stop at the FIRST record proving FULL =="
{
timeout 900 python3 - "$PORT" "$FULL_AT" "$BATTERY_START_MS" <<'PY'
import json, os, sys, time, urllib.request
port, full_at, since = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
log = os.environ["NINFER_REQUEST_LOG"]

def window_peak():
    best = -1
    try:
        with open(log, errors="ignore") as fh:
            for line in fh:
                if '"occupancy"' not in line:
                    continue
                try:
                    d = json.loads(line)
                    if (d.get("timestamp_unix_ms") or 0) < since:
                        continue
                    h = (d.get("context_cache", {}).get("occupancy") or {}).get("host_state_slots")
                    if isinstance(h, int) and h > best:
                        best = h
                except Exception:
                    continue
    except FileNotFoundError:
        pass
    return best

def send(i):
    # A top-up session MUST be multi-message: the frontend only places long anchors when
    # message_count > 1 (single-message sessions came out as "anchors=0 endpoint=1", and a
    # catalog swap of anchor-less sessions has net-zero states, freezing the peak at44).
    # Three messages at auto-spacing256 yield ~3-4 anchors + endpoint per session, so even
    # when catalog=16/16 forces oldest-out swaps, new(4 states) > evicted(1) and the peak climbs.
    shared = "retention filler token "
    messages = [
        {"role": "user", "content": f"topup-{int(time.time())}-{i} " + shared * 300},
        {"role": "assistant", "content": "ack"},
        {"role": "user", "content": f"turn two {i} " + shared * 150},
    ]
    payload = json.dumps({"model": "myai", "messages": messages,
                          "max_completion_tokens": 1, "temperature": 0}).encode()
    req = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions",
                                 data=payload, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        json.loads(resp.read())

for i in range(80):
    peak = window_peak()
    print(f"  top-up {i:02d}: window peak={peak} (>= {full_at} = stop pressing)", flush=True)
    if peak >= full_at:
        raise SystemExit(0)
    send(i)
    time.sleep(1.0)
raise SystemExit(f"TOP-UP EXHAUSTED: peak={window_peak()} < {full_at}")
PY
top_rc=$?
if [ "$top_rc" -ne 0 ]; then
    echo "SATURATION GATE FAILED (phase 1b top-up, rc=$top_rc)"
    exit 1
fi
}   # phase 1b top-up (fallback when fill alone did not reach the ever-full peak)

echo "== phase 2: saturation gate (window peak, fail-closed re-check) =="
window_peak_check
gate_rc=$?
if [ "$gate_rc" -ne 0 ]; then
    echo "SATURATION GATE FAILED (phase 2 peak check)"
    exit 1
fi

echo "== phase 3: pressure/eviction-sensitive tests (pool is FULL) =="
# state_index FIRST: its invariant is idle-retention (build, wait5s, switch back hits), and its
# conversations must PUBLISH checkpoints to exist at all. At gate time the catalog still holds
# only the fill's sessions (room to publish; the pool is full either way); run it after the other
# suites and the catalog is already16/16 with [exhaust] drop churn, so fresh checkpoints miss
# publication and the all-hit expectation measures someone else's catalog pressure instead.
# conversations=verify=4 keeps every conversation inside the4 fair-share buckets (and avoids
# the prompts[verify>conversations] IndexError the old defaults hit).
step state_index      300 python3 tools/smoke/test_state_index_short.py \
    --base-url "http://127.0.0.1:$PORT/v1" --conversations 4 --verify 4 --words 600
step pool_soak        600 tools/smoke/pool_soak.sh "$PORT" /tmp/crash243
step eviction_fix     300 python3 tools/smoke/verify_eviction_fix.py "$PORT"
step memory_pressure  300 python3 tools/smoke/test_memory_pressure.py \
    --port "$PORT" --conversations 6 --words 800
step cache_pressure   600 python3 tools/smoke/test_context_cache_pressure.py \
    --base-url "http://127.0.0.1:$PORT/v1" --conversations 12 --rounds 4 --words 200 \
    --pressure 8 --verify 4 --log "$S"
step prefix_switch    300 python3 tools/smoke/test_prefix_reuse_switch.py --port "$PORT" \
    "${SWITCH_ARGS[@]}"
step prefix_mixed     300 python3 tools/smoke/test_prefix_reuse_mixed.py --port "$PORT" \
    "${MIXED_ARGS[@]}"
step fork_hit         300 python3 tools/smoke/test_long_anchor_fork_hit.py \
    --base-url "http://127.0.0.1:$PORT/v1" "${FORK_ARGS[@]}"
# fair_share_sim retired from the battery (09-23): the branch it wants to
# observe (fair-share bucket release) is covered at the decision level by
#   ./build/tests/ninfer_resource_manager_test
#   ::test_fair_share_releases_oldest_bucket_only_when_shared_pool_exhausted (green)
# and its remaining A/B/compaction choreography can be run manually:
#   REUSE_DIAG=1 rig + python3 tools/smoke/test_fair_share_sim.py --A-depth 16000 ...

echo "== phase 4: correctness/contract tests =="
step serve_contract   600 python3 tools/smoke/serve_contract.py \
    --base-url "http://127.0.0.1:$PORT" --model myai
step yarn_vision      300 python3 tools/smoke/test_yarn_vision.py
step yarn_niah_sm     300 python3 tools/smoke/test_yarn_niah.py "$NIAH_TOKENS" 10
step anchor_probe     300 python3 tools/smoke/anchor_semantics_probe.py "$PORT" "$S"
step prefix_reuse     300 python3 tools/smoke/test_prefix_reuse.py "$PREFIX_TARGET"
step capture_repro    300 python3 tools/smoke/agent_capture_repro.py
step conc_at_limit    300 python3 tools/smoke/test_concurrency_decode.py "${CONC_SMALL[@]}"
step conc_overload    300 python3 tools/smoke/test_concurrency_decode.py "${CONC_BIG[@]}"
step replay_dumps     300 python3 tools/smoke/replay_dump.py "$PORT" \
    $(ls "$NINFER_REQDUMP_DIR"/*.json | head -3)

echo "== phase 5: final occupancy, transfer churn, error tally =="
host_end=$(occ_host)
window_peak_check || true          # informational: peak already gated above
echo "host_state_slots last=${host_end:-?} of $TOTAL"
# Transfer churn is the test material after saturation: evictions/restores keep the D2H/H2D
# lanes moving while every test runs. Fork restores print no H2D line, so the capture-direction
# counters (host-snapshot=D2H side, device-fork=H2D side) are reported alongside.
scope() { tail -n +"$((BAT_START_LINE + 1))" "$S" | grep -c "$1" || true; }
echo "transfer churn (this battery): D2H copies=$(scope 'state-store: D2H copy handle=')" \
     "H2D loads=$(scope 'state-store: H2D load handle=')" \
     "capture host-snapshot=$(scope 'reason=host-snapshot')" \
     "device-fork=$(scope 'reason=device-fork')"
# Window-scoped: the prod serve log persists across restarts, so a whole-file count mixes in
# historical shutdown fatals (observed:12 old "[engine] fatal: shutting down" from this
# morning's stops while every step slice of this battery was0). Count only this battery's tail.
echo -n "matching error lines in $S (this battery's window): "
tail -n +"$((BAT_START_LINE + 1))" "$S" | grep -cE "$ERR_PAT" || true
if [ "$FAILED_STEPS" != "0" ]; then
    echo "BATTERY FAILED: $FAILED_STEPS step(s) reported a failure"
    exit 1
fi
echo "== battery done =="
