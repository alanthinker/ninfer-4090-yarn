#!/usr/bin/env bash
# Reproduce the structural-closure -> evict-everything mass-eviction accident with a
# DELTA-based verdict (counts snapshotted before stage 1, so the whole run - multi-turn
# fill included - is the experiment).
#
#   --rig   small-parameter preset (kv cap1024 pages / host48 / ~6K-token trigger):
#           fast fix-iteration loop. Log paths default to the rig harness dir
#           (NINFER_SERVICE_LOG / NINFER_REQUEST_LOG override).
#   default production preset (kv10284 / host320 / ~122K-token trigger).
#
# Trigger trio (catalog-independent, works at catalog=512):
#   1. multi-turn private sessions sharing ONE stable system prefix (page/state overlap),
#   2. device KV AND host state near cap,
#   3. one large reuse-less root prompt.
#
# PRE-FIX  signature: (structural_delta>=1 or maxfb_delta>=1) [+ victims_delta>=10 = mass]
# POST-FIX signature: structural_delta=0 and maxfb_delta=0
#
# Requires the target service on :30000 with NINFER_REUSE_DIAG=1.
set -uo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1

MODE=prod
if [ "${1:-}" = "--rig" ]; then MODE=rig; fi

export MFR_REQLOG="${NINFER_REQUEST_LOG:-$([ "$MODE" = rig ] && echo /tmp/fs-ab/request_log_agent.jsonl || echo /root/ai/large_models/_ninfer_repos/deploy-yarn/request_log.jsonl)}"
export MFR_SERVELOG="${NINFER_SERVICE_LOG:-$([ "$MODE" = rig ] && echo /tmp/fs-ab/ninfer_serve_agent.log || echo /root/ai/large_models/_ninfer_repos/deploy-yarn/ninfer_serve.log)}"
if [ "$MODE" = rig ]; then
    export MFR_KV_CAP="${KV_CAP:-1024}" MFR_HOST_CAP="${HOST_CAP:-48}" \
           MFR_FILL_CAP="${FILL_CAP:-30}" MFR_BIG_WORDS="${BIG_WORDS:-3000}"
else
    export MFR_KV_CAP="${KV_CAP:-10284}" MFR_HOST_CAP="${HOST_CAP:-320}" \
           MFR_FILL_CAP="${FILL_CAP:-60}" MFR_BIG_WORDS="${BIG_WORDS:-61400}"
fi

curl -sf -m 3 "http://127.0.0.1:30000/health" >/dev/null \
  || { echo "FAIL: no service on :30000"; exit1; }
tr '\0' '\n' < "/proc/$(ss -ltnp | grep ':30000' | grep -o 'pid=[0-9]*' | cut -d= -f2 | head -1)/environ" 2>/dev/null \
  | grep -q '^NINFER_REUSE_DIAG=1' || echo "WARN: NINFER_REUSE_DIAG missing (site tags unavailable)"

echo "== mode=$MODE kv_cap=$MFR_KV_CAP host_cap=$MFR_HOST_CAP fill_cap=$MFR_FILL_CAP big_words=$MFR_BIG_WORDS"
python3 - <<'PY'
import json, os, time, urllib.request, sys, datetime

port = "30000"
reqlog = os.environ["MFR_REQLOG"]
servelog = os.environ["MFR_SERVELOG"]
SHAPE = os.environ.get("MFR_SHAPE", "single")
KV_CAP = int(os.environ["MFR_KV_CAP"]); HOST_CAP = int(os.environ["MFR_HOST_CAP"])
FILL_CAP = int(os.environ["MFR_FILL_CAP"]); BIG_WORDS = int(os.environ["MFR_BIG_WORDS"])
shared = ("myai coding agent. shared stable system prefix token " * 1600)

def occ():
    try: lines = open(reqlog, errors="ignore").read().splitlines()
    except FileNotFoundError: return None
    for line in reversed(lines):
        if '"occupancy"' not in line: continue
        try:
            o = json.loads(line)["context_cache"]["occupancy"]
            return {"host": o.get("host_state_slots", 0),
                    "kv": o.get("device_main_kv_pages", 0)}
        except Exception: continue
    return None

def log_counts():
    struct = tags = victims = 0
    try:
        for line in open(servelog, errors="ignore"):
            if "(no residual: structural)" in line: struct += 1
            elif "[pressure] target rejected site=" in line: tags += 1
            elif "site=materialization-victim" in line: victims += 1
    except FileNotFoundError: pass
    maxfb = 0
    try:
        for line in open(reqlog, errors="ignore"):
            if '"selected_capped_fallback": true' in line and 'request_done' in line:
                maxfb += 1
    except FileNotFoundError: pass
    return struct, tags, victims, maxfb

def last_request():
    last = None
    try:
        for line in open(reqlog, errors="ignore"):
            if '"materialization"' not in line or 'request_done' not in line: continue
            try: d = json.loads(line)
            except Exception: continue
            ts = d.get("timestamp_unix_ms", 0)
            if last is None or ts > last[0]: last = (ts, d.get("materialization", {}))
    except FileNotFoundError: pass
    return last

def post(messages, tag, max_tokens=1, timeout=600):
    payload = json.dumps({"model": "myai", "messages": messages, "stream": False,
                          "max_completion_tokens": max_tokens, "temperature": 0}).encode()
    r = urllib.request.Request(f"http://127.0.0.1:{port}/v1/chat/completions", data=payload,
                               headers={"Content-Type": "application/json"})
    t0 = time.time()
    try:
        with urllib.request.urlopen(r, timeout=timeout) as resp:
            out = json.loads(resp.read())
    except Exception as e:
        print(f"  {tag}: FAIL {type(e).__name__}: {e}", flush=True)
        return None
    u = out.get("usage", {}); p = u.get("prompt_tokens", 0) or 1
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    print(f"  {tag:<16} prompt={p:>7} cache={cached:>7} ({100.0*cached/p:5.1f}%) "
          f"{time.time()-t0:6.1f}s occ={occ()}", flush=True)
    return out

before = log_counts()
print(f"== snapshot before: struct={before[0]} tags={before[1]} victims={before[2]} maxfb={before[3]}", flush=True)

print(f"== [1/2] multi-turn shared-prefix fill (host>={int(HOST_CAP*0.97)} kv>={int(KV_CAP*0.97)}) ==", flush=True)
for s in range(1, FILL_CAP + 1):
    o = occ() or {"host": 0, "kv": 0}
    if o["host"] >= int(HOST_CAP * 0.97) and o["kv"] >= int(KV_CAP * 0.97):
        print(f"  FILLED after {s-1} sessions: {o}", flush=True)
        break
    if SHAPE == "single":
        # v1 proven shape: ONE long system (shared head + long tail) + tiny user turn.
        # Auto-long-anchors land inside the shared head - the collision surface the
        # multi-turn shape never produces (its per-turn span stays <4096).
        post([{"role": "system", "content": shared + f" unique session{s} " + ("fill token " * 8000)},
              {"role": "user", "content": f"fill turn {s}"}], f"fill.s{s}")
    else:
        msgs = [{"role": "system", "content": shared + f" unique session{s} header."}]
        for t in range(8):
            msgs.append({"role": "user",
                         "content": f"session{s} turn{t} " + ("payload token " * 500)})
            msgs.append({"role": "assistant", "content": f"ack {s}/{t} " + ("reply token " * 150)})
        post(msgs, f"fill.s{s}")
else:
    print(f"  fill cap hit: {occ()}", flush=True)

o = occ() or {}
if o.get("kv", 0) < int(KV_CAP * 0.90):
    print(f"ABORT: device KV only {o.get('kv')}/{KV_CAP} (<90%); residual would not force pressure", flush=True)
    sys.exit(2)

print("== [2/2] big unique root request ==", flush=True)
big = "unique root request payload " + ("divergent token " * BIG_WORDS)
post([{"role": "system", "content": "standalone brief system. " + big},
      {"role": "user", "content": "begin"}], "BIG-ROOT", max_tokens=64, timeout=900)

after = log_counts()
d_struct = after[0] - before[0]
d_tags   = after[1] - before[1]
d_vict   = after[2] - before[2]
d_maxfb  = after[3] - before[3]
print()
print(f"== DELTA: structural={d_struct} reason_tags={d_tags} victims={d_vict} maxfb={d_maxfb}")
last = last_request()
if last:
    t = datetime.datetime.fromtimestamp(last[0]/1000)
    m = last[1]
    print(f"   latest request_done {t:%H:%M:%S}: maxfb={m.get('selected_capped_fallback')} "
          f"closures={m.get('guided_closures_succeeded')}/{m.get('guided_closures_failed')} "
          f"degr={m.get('selected_degradation_units')} stop={m.get('stop_reason')}")
trigger = (d_struct >= 1 or d_maxfb >= 1)
mass = d_vict >= 10
clean = (d_struct == 0 and d_maxfb == 0)
print(f"   trigger(path fired)  : {'YES' if trigger else 'no'}" + (f"  MASS={d_vict} victims" if mass else ""))
print(f"   clean(post-fix)     : {'YES' if clean else 'no'}")
print("   PRE-FIX  expectation: trigger=YES (+mass on fat workloads)")
print("   POST-FIX expectation: clean=YES")
PY
RC=$?
echo "repro_rc=$RC"
