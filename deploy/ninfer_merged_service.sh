#!/usr/bin/env bash
# Service manager for the merged NInfer build (sm_89 + INT8 tensor-core prefill).
#
#   ./ninfer_merged_service.sh start|stop|restart|status|logs|health
#
# Unlike deploy/ninfer_service.sh (which runs a native build/ binary on the RTX 4080S host),
# this one runs the containerised merged build and mounts the model read-only, so the image
# carries no weights.
#
# Environment:
#   NINFER_IMAGE        image (default ninfer-4090-merged:sm89)
#   NINFER_NAME         container name (default ninfer-merged)
#   NINFER_PORT         host port (default 8080)
#   NINFER_MODEL_DIR    host directory holding the .ninfer artifact (required for start)
#   NINFER_MODEL        artifact file name (default qwen3_8_27b_orcarouter.ninfer)
#   NINFER_MODEL_ID     served model id (default orcarouter)
#   NINFER_CACHE_VOL    docker volume for the continuation cache (default ninfer-merged-cache)
#   NINFER_KV_DTYPE     KV dtype: rk4v4-e8 (default, 4.25 GiB at 262K) or int8 (8.25 GiB at 262K)
#   NINFER_MAX_CTX      max context (default 262144)
#   NINFER_CONCURRENCY  max concurrency (default 4)
#   NINFER_COMPAT_FLAGS set to 0 to launch without the Tensorninja-spelled flags
set -uo pipefail

IMAGE="${NINFER_IMAGE:-ninfer-4090-merged:sm89}"
NAME="${NINFER_NAME:-ninfer-merged}"
PORT="${NINFER_PORT:-8080}"
MODEL_DIR="${NINFER_MODEL_DIR:-}"
MODEL="${NINFER_MODEL:-qwen3_8_27b_orcarouter.ninfer}"
MODEL_ID="${NINFER_MODEL_ID:-orcarouter}"
CACHE_VOL="${NINFER_CACHE_VOL:-ninfer-merged-cache}"
KV_DTYPE="${NINFER_KV_DTYPE:-rk4v4-e8}"
MAX_CTX="${NINFER_MAX_CTX:-262144}"
CONCURRENCY="${NINFER_CONCURRENCY:-4}"
COMPAT_FLAGS="${NINFER_COMPAT_FLAGS:-1}"
DOCKER="${NINFER_DOCKER:-docker}"

compat_args=()
if [ "$COMPAT_FLAGS" != "0" ]; then
  # Tensorninja spellings; the merged build maps them onto this build's mechanisms and
  # prints a `note:` per mapping. Remove this block to launch with base spellings only.
  compat_args=(
    --prefix-checkpoint-policy rolling-tool
    --continuation-cache l1-l2-l3
    --continuation-cache-dir /var/cache/ninfer
    --continuation-cache-namespace local
    --continuation-cache-l1-mib 8192
    --continuation-cache-l2-mib 16384
    --continuation-cache-l3-mib 49152
    --continuation-cache-filesystem-reserve-mib 4096
    --turn-checkpoints 32
    --slot-save-path /var/cache/ninfer/slots
    --auto-save-evicted
    --request-log-jsonl /var/cache/ninfer/requests.jsonl
  )
fi

case "${1:-}" in
  start)
    [ -n "$MODEL_DIR" ] || { echo "set NINFER_MODEL_DIR to the directory holding $MODEL" >&2; exit 1; }
    $DOCKER rm -f "$NAME" >/dev/null 2>&1
    $DOCKER volume create "$CACHE_VOL" >/dev/null 2>&1 || true
    $DOCKER run -d --name "$NAME" --gpus all -p "$PORT:8080" \
      -v "$MODEL_DIR:/models:ro" -v "$CACHE_VOL:/var/cache/ninfer" \
      "$IMAGE" \
      ninfer-serve "/models/$MODEL" --model-id "$MODEL_ID" \
      --host 0.0.0.0 --port 8080 \
      --max-context "$MAX_CTX" --kv-capacity "$MAX_CTX" --kv-dtype "$KV_DTYPE" \
      --max-concurrency "$CONCURRENCY" --max-pending-requests 16 --pending-timeout-ms 600000 \
      --prefill-chunk 1024 --spec mtp --draft-tokens 3 --lm-head-draft \
      "${compat_args[@]}" \
      --preserve-thinking --vision --vision-max-tokens 32768 \
      || { echo "start failed" >&2; exit 1; }
    echo "started $NAME on :$PORT (kv=$KV_DTYPE ctx=$MAX_CTX concurrency=$CONCURRENCY)"
    echo "note: a merged service needs ~24.8 GB at rk4v4-e8; stop any other GPU service first"
    ;;
  stop)    $DOCKER rm -f "$NAME" && echo "stopped $NAME" ;;
  restart) "$0" stop; sleep 3; "$0" start ;;
  status)
    $DOCKER ps --filter "name=^${NAME}$" --format '{{.Names}}  {{.Status}}  {{.Ports}}'
    $DOCKER ps --filter "name=^${NAME}$" -q | grep -q . || echo "$NAME is not running"
    ;;
  logs)    $DOCKER logs --tail "${2:-80}" "$NAME" ;;
  health)
    curl -sf "http://127.0.0.1:$PORT/health" && echo || { echo "health check failed" >&2; exit 1; }
    ;;
  *)
    echo "usage: $0 {start|stop|restart|status|logs [N]|health}" >&2
    exit 2
    ;;
esac
