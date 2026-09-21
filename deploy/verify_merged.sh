#!/usr/bin/env bash
# Merged-build end-to-end / A-B verification driver.
#
#   ./verify_merged.sh merged     # run the harness against ninfer-4090-merged:sm89
#   ./verify_merged.sh baseline   # same harness against ninfer-4090-merged:baseline
#   ./verify_merged.sh ab         # baseline first, then merged, then print the delta table
#
# The harness (verify_merged_e2e.py) reports the server-side `timings` block, which is the
# authoritative measurement channel; see that file for why curl's time_starttransfer is not used.
#
# PREREQUISITE: the GPU must be free. One merged service needs ~24.8 GB on a 49 GB card and
# does not coexist with a second int8-KV service of ~30 GB.
#
# Environment:
#   NINFER_MODEL_DIR   host directory holding the .ninfer artifact (required)
#   NINFER_MODEL       artifact file name (default qwen3_8_27b_orcarouter.ninfer)
#   NINFER_MODEL_ID    served model id (default orcarouter)
#   NINFER_IMAGE       image for the merged run (default ninfer-4090-merged:sm89)
#   NINFER_BASELINE    image for the baseline run (default ninfer-4090-merged:baseline)
#   NINFER_PORT        merged port (default 18080)
#   NINFER_PORT_BASE   baseline port (default 18081)
#   NINFER_CACHE_VOL   docker volume for the continuation cache (default ninfer-merged-cache)
#   NINFER_TIMEOUT     per-request timeout seconds (default 1800)
#   NINFER_READY_S     max seconds to wait for /health (default 1500)
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"

MODE="${1:-merged}"
MODEL_DIR="${NINFER_MODEL_DIR:?set NINFER_MODEL_DIR to the directory holding the .ninfer artifact}"
MODEL="${NINFER_MODEL:-qwen3_8_27b_orcarouter.ninfer}"
MODEL_ID="${NINFER_MODEL_ID:-orcarouter}"
IMAGE="${NINFER_IMAGE:-ninfer-4090-merged:sm89}"
IMAGE_BASE="${NINFER_BASELINE:-ninfer-4090-merged:baseline}"
PORT="${NINFER_PORT:-18080}"
PORT_BASE="${NINFER_PORT_BASE:-18081}"
CACHE_VOL="${NINFER_CACHE_VOL:-ninfer-merged-cache}"
TIMEOUT="${NINFER_TIMEOUT:-1800}"
READY_S="${NINFER_READY_S:-1500}"

OUTDIR="${NINFER_OUTDIR:-$HERE/../.verify-merged}"
mkdir -p "$OUTDIR"

# docker is commonly aliased/hijacked in this environment; call the binary directly.
DOCKER="${NINFER_DOCKER:-docker}"

svc_args=(
  ninfer-serve "/models/$MODEL" --model-id "$MODEL_ID"
  --host 0.0.0.0 --port 8080
  --max-context 262144 --kv-capacity 262144 --kv-dtype rk4v4-e8
  --max-concurrency 4 --max-pending-requests 16 --pending-timeout-ms 600000
  --prefill-chunk 1024 --spec mtp --draft-tokens 3 --lm-head-draft
  --preserve-thinking --vision --vision-max-tokens 32768
)

start_one() { # $1=name $2=port $3=image
  local name="$1" port="$2" image="$3"
  $DOCKER rm -f "$name" >/dev/null 2>&1
  $DOCKER volume create "$CACHE_VOL" >/dev/null 2>&1
  if [ "$image" = "$IMAGE" ]; then
    # The merged build accepts the Tensorninja-spelled flags too; this reproduces the
    # production command line so the compatibility layer is exercised as well.
    $DOCKER run -d --name "$name" --gpus all -p "$port:8080" \
      -v "$MODEL_DIR:/models:ro" -v "$CACHE_VOL:/var/cache/ninfer" "$image" \
      "${svc_args[@]}" \
      --prefix-checkpoint-policy rolling-tool \
      --continuation-cache l1-l2-l3 \
      --continuation-cache-dir /var/cache/ninfer \
      --continuation-cache-namespace local \
      --continuation-cache-l1-mib 8192 \
      --continuation-cache-l2-mib 16384 \
      --turn-checkpoints 32 \
      >/dev/null || { echo "FAIL: docker run ($name)"; return 1; }
  else
    # The pristine baseline has no compatibility aliases: base-spelled flags only.
    $DOCKER run -d --name "$name" --gpus all -p "$port:8080" \
      -v "$MODEL_DIR:/models:ro" -v "$CACHE_VOL:/var/cache/ninfer" "$image" \
      "${svc_args[@]}" >/dev/null || { echo "FAIL: docker run ($name)"; return 1; }
  fi
  echo "started $name ($image) on :$port"
}

wait_ready() { # $1=name $2=port
  local name="$1" port="$2" i=0
  while [ $i -lt "$READY_S" ]; do
    if curl -sf -o /dev/null "http://127.0.0.1:$port/health"; then
      echo "$name healthy after $((i * 5))s"; return 0
    fi
    if ! $DOCKER ps --format '{{.Names}}' | grep -qx "$name"; then
      echo "FAIL: $name exited"; $DOCKER logs --tail 40 "$name"; return 1
    fi
    i=$((i + 1)); sleep 5
  done
  echo "FAIL: $name not healthy within ${READY_S}s"; $DOCKER logs --tail 40 "$name"; return 1
}

user_vram() { $DOCKER run --rm --gpus all --entrypoint nvidia-smi "$IMAGE" \
  --query-gpu=memory.used,memory.free --format=csv,noheader 2>/dev/null || echo "n/a"; }

run_harness() { # $1=port $2=outfile
  python3 "$HERE/verify_merged_e2e.py" \
    --port "$1" --model "$MODEL_ID" --timeout "$TIMEOUT" --out "$2"
}

report_cuda_errors() { # $1=name
  local n
  n=$($DOCKER logs "$1" 2>&1 \
      | grep -ciE 'cooperativelaunch|cuda ?error|unhandled exception|illegal memory' || true)
  echo "$1: cuda_error_lines=$n"
}

run_side() { # $1=label $2=name $3=port $4=image
  start_one "$2" "$3" "$4" || return 1
  wait_ready "$2" "$3" || return 1
  echo "=== $1 GPU: $(user_vram | tail -1) ==="
  run_harness "$3" "$OUTDIR/$1.json"
  echo "rc=$?"
  report_cuda_errors "$2"
}

if [ "$MODE" = "ab" ]; then
  run_side baseline ninfer-baseline-18081 "$PORT_BASE" "$IMAGE_BASE"
  $DOCKER rm -f ninfer-baseline-18081 >/dev/null 2>&1
  sleep 5
  run_side merged ninfer-merged-18080 "$PORT" "$IMAGE"
  echo
  python3 - "$OUTDIR/baseline.json" "$OUTDIR/merged.json" <<'EOF'
import json, sys
def load(p):
    return json.load(open(p))
try:
    b, m = load(sys.argv[1]), load(sys.argv[2])
except Exception as exc:
    print(f"delta table skipped: {exc}"); raise SystemExit(0)

def t(res, case, field):
    return (res.get(case, {}).get("timings") or {}).get(field)

print(f"{'metric':<34}{'baseline':>12}{'merged':>12}{'delta':>10}")
for label, case, field in (
    ("prefill tok/s (case b)", "b", "prompt_per_second"),
    ("prefill prompt_n (case b)", "b", "prompt_n"),
    ("prefill tok/s (case d)", "d", "prompt_per_second"),
    ("decode tok/s (case c)", "c", "predicted_per_second"),
    ("long-output tokens (case a)", "a", "predicted_n"),
):
    vb, vm = t(b, case, field), t(m, case, field)
    if isinstance(vb, (int, float)) and isinstance(vm, (int, float)) and vb:
        print(f"{label:<34}{vb:>12.2f}{vm:>12.2f}{(vm / vb - 1) * 100:>9.1f}%")
    else:
        print(f"{label:<34}{str(vb):>12}{str(vm):>12}{'-':>10}")
EOF
else
  if [ "$MODE" = "baseline" ]; then
    run_side baseline ninfer-baseline-18081 "$PORT_BASE" "$IMAGE_BASE"
  else
    run_side merged ninfer-merged-18080 "$PORT" "$IMAGE"
  fi
fi

echo
echo "artifacts in $OUTDIR"
echo "stop test containers:  $DOCKER rm -f ninfer-merged-18080 ninfer-baseline-18081"
