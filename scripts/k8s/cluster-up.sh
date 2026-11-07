#!/usr/bin/env bash
# Start a local minikube cluster with Metrics Server (needed by `kubectl top` and the HPA).
# Override sizing with CPUS / MEMORY, e.g.  CPUS=6 MEMORY=8g scripts/k8s/cluster-up.sh
set -euo pipefail
# shellcheck source=scripts/k8s/lib.sh
source "$(dirname "$0")/lib.sh"

need minikube "Install with: brew install minikube"
need kubectl "Install with: brew install kubectl"
need docker "Install and start Docker Desktop (minikube's docker driver uses it)."
docker info >/dev/null 2>&1 || die "Docker is not running. Start Docker Desktop first."

CPUS="${CPUS:-4}"
MEMORY="${MEMORY:-6g}"

if minikube status >/dev/null 2>&1; then
  log "minikube already running"
else
  log "starting minikube (cpus=$CPUS memory=$MEMORY)"
  minikube start --driver=docker --cpus="$CPUS" --memory="$MEMORY"
fi

log "enabling metrics-server addon"
minikube addons enable metrics-server >/dev/null
kubectl -n kube-system rollout status deployment/metrics-server --timeout=180s

log "waiting for resource metrics API (can take ~60 s on first start)"
wait_until "kubectl top nodes >/dev/null 2>&1" 180 >/dev/null || die "metrics API not ready"
kubectl top nodes
log "cluster ready: $(kubectl config current-context)"
