#!/usr/bin/env bash
# Check the Service end to end through a port-forward: health, readiness, metrics, one inspection.
set -euo pipefail
# shellcheck source=scripts/k8s/lib.sh
source "$(dirname "$0")/lib.sh"
cd "$REPO_ROOT"
need curl "curl is required"

PORT="${PORT:-18080}"
kubectl -n "$NS" port-forward svc/inspectedge-api "$PORT:80" >/dev/null 2>&1 &
PF=$!
trap 'kill $PF 2>/dev/null || true' EXIT
wait_until "curl -fsS http://127.0.0.1:$PORT/health/live >/dev/null 2>&1" 30 >/dev/null \
  || die "port-forward did not come up"
URL="http://127.0.0.1:$PORT"

log "GET /health/live";  curl -fsS "$URL/health/live"; echo
log "GET /health/ready"; curl -fsS "$URL/health/ready"; echo
log "GET /health";       curl -fsS "$URL/health"; echo

SAMPLE="$REPO_ROOT/reports/k8s/sample.png"
[ -f "$SAMPLE" ] || py tests/load/make_sample_image.py --out "$SAMPLE"
log "POST /inspect"
curl -fsS -F "file=@$SAMPLE;type=image/png" "$URL/inspect" | py -c \
  'import json,sys; r=json.load(sys.stdin); print({k: r[k] for k in ("decision","anomaly_score","model_version","latency_ms")})'

log "GET /metrics (InspectEdge series)"
curl -fsS "$URL/metrics" | grep -E '^inspectedge_(http_requests_total|decisions_total|model_loaded|inference_duration_seconds_count)' | head -20

log "Service endpoints (pod IP, ready):"
endpoints
log "smoke test passed"
