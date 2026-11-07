#!/usr/bin/env bash
# Failure drills, run while k6 sends a steady 2 req/s through the Service:
#   A. delete a pod             -> time until the Deployment is back to full readiness
#   B. fail readiness (drain)   -> pod leaves Service endpoints without a restart, then returns
#   C. slow startup (60 s)      -> rollout waits on the startup probe; old pods keep serving
# Optional (adds ~5 min):  WITH_FAILING_STARTUP=1  -> startup exceeds the probe budget, the
#   kubelet restarts the container, the rollout stalls, traffic stays on the old pods.
#
# Results: reports/k8s/<timestamp>-recovery/{recovery.txt,k6.log,events.txt}
set -euo pipefail
# shellcheck source=scripts/k8s/lib.sh
source "$(dirname "$0")/lib.sh"
cd "$REPO_ROOT"

OUT="$(results_dir recovery)"
R="$OUT/recovery.txt"
DRAIN_FILE="/tmp/inspectedge.drain"
note() { log "$*"; echo "$*" >> "$R"; }
restarts() {
  kubectl -n "$NS" get pod "$1" -o jsonpath='{.status.containerStatuses[0].restartCount}'
}
pod_ip() { kubectl -n "$NS" get pod "$1" -o jsonpath='{.status.podIP}'; }
first_pod() {
  kubectl -n "$NS" get pods -l "$APP_LABEL" --field-selector=status.phase=Running \
    -o jsonpath='{.items[0].metadata.name}'
}
cleanup() {
  kubectl -n "$NS" set env "$DEPLOY" INSPECTEDGE_STARTUP_DELAY_SECONDS- >/dev/null 2>&1 || true
}
trap cleanup EXIT

DESIRED=$(desired_pods)
[ "$(ready_pods)" = "$DESIRED" ] || die "deployment not fully ready ($(ready_pods)/$DESIRED)"
[ "${WITH_FAILING_STARTUP:-0}" = "1" ] && DURATION_S=900 || DURATION_S=480
start_k6 steady 0 2 "${DURATION_S}s"
sleep 20

# --- A. pod deletion ---------------------------------------------------------------------
POD=$(first_pod)
# shellcheck disable=SC2034  # used inside the wait_until condition string
OLD_PODS=$(kubectl -n "$NS" get pods -l "$APP_LABEL" -o jsonpath='{range .items[*]}{.metadata.name}{"\n"}{end}')
pod_readiness() {
  kubectl -n "$NS" get pods -l "$APP_LABEL" -o \
    jsonpath='{range .items[*]}{.metadata.name} {.status.conditions[?(@.type=="Ready")].status}{"\n"}{end}'
}
note "A. deleting pod $POD (ready $(ready_pods)/$DESIRED)"
kubectl -n "$NS" delete pod "$POD" --wait=false >/dev/null
T=$(wait_until "pod_readiness | grep -v -F \"\$OLD_PODS\" | grep -q ' True\$'" 300) || true
note "A. replacement pod created automatically and Ready after ${T}s"
wait_until "[ \"\$(ready_pods)\" = \"$DESIRED\" ]" 120 >/dev/null || true
note "A. deployment ready $(ready_pods)/$DESIRED"
sleep 20

# --- B. readiness failure (drain) --------------------------------------------------------
POD=$(first_pod); IP=$(pod_ip "$POD"); RC0=$(restarts "$POD")
note "B. failing readiness on $POD ($IP) by creating $DRAIN_FILE"
kubectl -n "$NS" exec "$POD" -- touch "$DRAIN_FILE"
T=$(wait_until "endpoints | grep -q \"^$IP false\"" 60) || true
note "B. marked not-ready in Service endpoints after ${T}s"
sleep 20
note "B. restarts while unready: $(( $(restarts "$POD") - RC0 )) (expected 0: liveness unaffected)"
kubectl -n "$NS" exec "$POD" -- rm -f "$DRAIN_FILE"
T=$(wait_until "endpoints | grep -q \"^$IP true\"" 60) || true
note "B. back in Service endpoints after ${T}s"
sleep 20

# --- C. slow startup within the probe budget ---------------------------------------------
note "C. rolling out with INSPECTEDGE_STARTUP_DELAY_SECONDS=60 (startup budget 180 s)"
T0=$(now)
kubectl -n "$NS" set env "$DEPLOY" INSPECTEDGE_STARTUP_DELAY_SECONDS=60 >/dev/null
kubectl -n "$NS" rollout status "$DEPLOY" --timeout=600s >/dev/null || true
note "C. rollout with slow startup completed after $(( $(now) - T0 ))s; ready $(ready_pods)/$(desired_pods)"

if [ "${WITH_FAILING_STARTUP:-0}" = "1" ]; then
  note "C2. rolling out with INSPECTEDGE_STARTUP_DELAY_SECONDS=240 (exceeds 180 s budget)"
  kubectl -n "$NS" set env "$DEPLOY" INSPECTEDGE_STARTUP_DELAY_SECONDS=240 >/dev/null
  sleep 210
  NEW=$(kubectl -n "$NS" get pods -l "$APP_LABEL" --sort-by=.metadata.creationTimestamp \
        -o jsonpath='{.items[-1:].metadata.name}')
  note "C2. newest pod $NEW restarts: $(restarts "$NEW"); ready $(ready_pods)/$(desired_pods) (old pods still serving)"
  kubectl -n "$NS" get events --field-selector "involvedObject.name=$NEW" \
    | grep -iE 'startup probe|killing|back-off' | tail -5 | tee -a "$R" || true
fi

note "restoring normal startup"
kubectl -n "$NS" set env "$DEPLOY" INSPECTEDGE_STARTUP_DELAY_SECONDS- >/dev/null
kubectl -n "$NS" rollout status "$DEPLOY" --timeout=600s >/dev/null || true

log "stopping background traffic"
kubectl -n "$NS" get events --sort-by=.lastTimestamp > "$OUT/events.txt" 2>&1 || true
finish_k6 steady "$OUT/k6.log" | tee -a "$R"
log "results saved in $OUT"
