# Shared helpers for scripts/k8s/*.sh. Works with macOS bash 3.2 and Linux bash.
# shellcheck shell=bash disable=SC2034  # variables are used by the sourcing scripts

NS="${NS:-inspectedge}"
APP_LABEL="app.kubernetes.io/name=inspectedge-api"
DEPLOY="deployment/inspectedge-api"
BASE_IMAGE="${BASE_IMAGE:-inspectedge-ai:0.2.0}"
MODEL_IMAGE="${MODEL_IMAGE:-inspectedge-ai-model:0.2.0}"
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

log() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "'$1' not found. $2"; }
now() { date +%s; }

# Python from the project venv if present (has opencv for the sample image).
py() {
  if [ -x "$REPO_ROOT/.venv/bin/python" ]; then "$REPO_ROOT/.venv/bin/python" "$@"
  else python3 "$@"; fi
}

results_dir() {
  local d
  d="$REPO_ROOT/reports/k8s/$(date +%Y%m%d-%H%M%S)-$1"
  mkdir -p "$d"
  echo "$d"
}

ready_pods() {
  kubectl -n "$NS" get deploy inspectedge-api -o jsonpath='{.status.readyReplicas}' 2>/dev/null \
    | sed 's/^$/0/'
}

desired_pods() {
  kubectl -n "$NS" get deploy inspectedge-api -o jsonpath='{.spec.replicas}'
}

# "<podIP> <ready>" for every endpoint behind the Service.
endpoints() {
  kubectl -n "$NS" get endpointslices -l kubernetes.io/service-name=inspectedge-api \
    -o jsonpath='{range .items[*].endpoints[*]}{.addresses[0]} {.conditions.ready}{"\n"}{end}'
}

# Wait until $1 (a shell condition string) is true, polling every second; prints seconds taken.
wait_until() {
  local cond="$1" timeout="${2:-300}" start
  start=$(now)
  while ! eval "$cond"; do
    if [ $(( $(now) - start )) -ge "$timeout" ]; then echo "timeout"; return 1; fi
    sleep 1
  done
  echo $(( $(now) - start ))
}

# Prepare the k6 ConfigMap (script + sample image) and start a k6 Job.
#   start_k6 <profile> [MAX_VUS] [RATE] [DURATION]
start_k6() {
  local profile="$1" max_vus="${2:-12}" rate="${3:-2}" duration="${4:-4m}" sample
  sample="$REPO_ROOT/reports/k8s/sample.png"
  [ -f "$sample" ] || py "$REPO_ROOT/tests/load/make_sample_image.py" --out "$sample"
  kubectl -n "$NS" create configmap k6-load \
    --from-file=k6.js="$REPO_ROOT/tests/load/k6.js" --from-file=sample.png="$sample" \
    --dry-run=client -o yaml | kubectl apply -f - >/dev/null
  kubectl -n "$NS" delete job "k6-$profile" --ignore-not-found --wait=true >/dev/null
  sed -e "s/\${PROFILE}/$profile/g" -e "s/\${MAX_VUS}/$max_vus/g" \
      -e "s/\${RATE}/$rate/g" -e "s/\${DURATION}/$duration/g" \
      "$REPO_ROOT/tests/load/k6-job.yaml" | kubectl apply -f - >/dev/null
  log "k6 job k6-$profile started (profile=$profile max_vus=$max_vus rate=$rate duration=$duration)"
}

# Wait for a k6 Job to finish and save its log; prints the summary block.
finish_k6() {
  local profile="$1" out="$2"
  kubectl -n "$NS" wait --for=condition=complete "job/k6-$profile" --timeout=20m >/dev/null 2>&1 \
    || kubectl -n "$NS" wait --for=condition=failed "job/k6-$profile" --timeout=10s >/dev/null 2>&1 \
    || true
  kubectl -n "$NS" logs "job/k6-$profile" > "$out" 2>&1 || true
  sed -n '/=== InspectEdge load test/,$p' "$out"
}
