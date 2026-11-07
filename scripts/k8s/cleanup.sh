#!/usr/bin/env bash
# Remove everything this project created in the cluster. DELETE_CLUSTER=1 also deletes minikube.
set -euo pipefail
# shellcheck source=scripts/k8s/lib.sh
source "$(dirname "$0")/lib.sh"
cd "$REPO_ROOT"

log "deleting load-test jobs and config"
kubectl -n "$NS" delete job -l app.kubernetes.io/name=k6 --ignore-not-found
kubectl -n "$NS" delete configmap k6-load --ignore-not-found
log "deleting app resources (namespace last)"
kubectl delete -k k8s/ --ignore-not-found --wait=true
if [ "${DELETE_CLUSTER:-0}" = "1" ]; then
  log "deleting the minikube cluster"
  minikube delete
else
  log "cluster kept. Stop it with 'minikube stop' or delete it with 'minikube delete'."
fi
