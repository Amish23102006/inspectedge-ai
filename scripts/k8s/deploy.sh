#!/usr/bin/env bash
# Apply the manifests on a (clean) cluster and wait until pods are Ready and the HPA sees CPU.
set -euo pipefail
# shellcheck source=scripts/k8s/lib.sh
source "$(dirname "$0")/lib.sh"
cd "$REPO_ROOT"
need kubectl "Install with: brew install kubectl"

log "kubectl apply -k k8s/"
kubectl apply -k k8s/

log "waiting for rollout (first start loads torch + the model)"
if ! kubectl -n "$NS" rollout status "$DEPLOY" --timeout=300s; then
  kubectl -n "$NS" get pods -o wide
  kubectl -n "$NS" describe pods -l "$APP_LABEL" | tail -40
  die "rollout did not complete. 'ErrImageNeverPull' means run scripts/k8s/build-images.sh first."
fi

log "waiting for the HPA to report CPU (needs Metrics Server)"
wait_until "kubectl -n $NS get hpa inspectedge-api -o jsonpath='{.status.currentMetrics[0].resource.current.averageUtilization}' | grep -q '[0-9]'" 180 >/dev/null \
  || log "HPA has no CPU metrics yet; check 'kubectl get apiservices | grep metrics'"

kubectl -n "$NS" get deploy,pods,svc,hpa -o wide
log "deployed. Next: scripts/k8s/smoke-test.sh"
