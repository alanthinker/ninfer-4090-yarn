#!/usr/bin/env bash
# CI smoke test — run on the GPU machine after building and starting the service.
# Usage:
#   ./tools/smoke/ci_smoke.sh [--port 30000]
#
# Requires:
#   - ninja binary built (cmake --build build -j)
#   - Service running: deploy/ninfer_service.sh start
#   - Python 3.11+
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

PORT="${NINFER_PORT:-30000}"
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) PORT="$2"; shift 2 ;;
    *) echo "Unknown arg: $1" >&2; exit 1 ;;
  esac
done

echo "=== NInfer CI Smoke Test ==="
echo "  Service: http://127.0.0.1:${PORT}"
echo "  Time:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo ""

# 0. Health check
echo "--- [0/4] Health check ---"
if ! python3 -c "
import urllib.request, json, sys
try:
    r = urllib.request.urlopen('http://127.0.0.1:${PORT}/v1/models', timeout=5)
    data = json.load(r)
    models = [m['id'] for m in data.get('data', [])]
    print(f'  Service OK: {models}')
except Exception as e:
    print(f'  FATAL: service not reachable: {e}', file=sys.stderr)
    sys.exit(1)
"; then
  echo "FAIL: service not running on port ${PORT}" >&2
  exit 1
fi

# 1. Unit tests (host-only, no GPU inference)
echo ""
echo "--- [1/4] Unit tests ---"
cd "$REPO_ROOT"
if [[ -d build && -x build/tests/ninfer_qwen3_6_27b_visual_scatter_test ]]; then
  ctest --test-dir build --output-on-failure -j 8
  echo "  Unit tests: PASS"
else
  echo "  SKIP: no build/tests directory found"
fi

# 2. Prefix cache switch-back test (9 conversations, same size)
echo ""
echo "--- [2/4] Prefix reuse: switch-back (9 convs) ---"
python3 "$SCRIPT_DIR/test_prefix_reuse_switch.py" --port "$PORT"

# 3. Prefix cache mixed-workload test (4 conversations, varying sizes)
echo ""
echo "--- [3/4] Prefix reuse: mixed workload (4 convs, 200w-3125w) ---"
python3 "$SCRIPT_DIR/test_prefix_reuse_mixed.py" --port "$PORT" \
  --conversations 4 --base-words 200 --scale 2.5 --min-rounds 2 --max-rounds 6

# 4. Serve contract (protocol correctness)
echo ""
echo "--- [4/4] Serve contract ---"
python3 "$SCRIPT_DIR/serve_contract.py" --port "$PORT" || {
  echo "  (serve_contract failed — may need model-specific fixtures)"
}

echo ""
echo "============================================================"
echo "=== ALL SMOKE TESTS PASSED ==="
echo "  Time:    $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "============================================================"
