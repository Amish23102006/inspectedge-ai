#!/usr/bin/env bash
# Build the base image and the model image with your local Docker (Docker Desktop), then load
# the model image into minikube so the cluster can use it with imagePullPolicy: Never.
# Nothing is pushed to any registry.
#
#   scripts/k8s/build-images.sh [ARTEFACT_DIR]
# ARTEFACT_DIR defaults to the newest directory under artefacts/ (or artefacts/models/) that
# contains model_version.json (created by `make eval`).
# SKIP_MINIKUBE_LOAD=1 builds only (e.g. for kind: `kind load docker-image ...`).
set -euo pipefail
# shellcheck source=scripts/k8s/lib.sh
source "$(dirname "$0")/lib.sh"
cd "$REPO_ROOT"
need docker "Install and start Docker Desktop."

ART="${1:-${INSPECTEDGE_ARTEFACT_DIR:-}}"
if [ -z "$ART" ]; then
  # shellcheck disable=SC2012
  ART="$(ls -1dt artefacts/models/*/ artefacts/*/ 2>/dev/null | while read -r d; do
           [ -f "$d/model_version.json" ] && { echo "${d%/}"; break; }; done || true)"
fi
[ -n "$ART" ] && [ -f "$ART/model_version.json" ] || die "no trained model artefact found.
Run 'make prepare && make eval' first (see README), or pass the artefact dir as an argument."
log "model artefact: $ART"

log "building $BASE_IMAGE"
docker build -f docker/Dockerfile -t "$BASE_IMAGE" .

log "building $MODEL_IMAGE"
docker build -f docker/Dockerfile.model --build-arg BASE_IMAGE="$BASE_IMAGE" \
  -t "$MODEL_IMAGE" "$ART"

if [ "${SKIP_MINIKUBE_LOAD:-0}" != "1" ]; then
  need minikube "Install minikube first (see docs/kubernetes.md)."
  log "loading $MODEL_IMAGE into minikube (a few minutes, no progress bar)"
  minikube image load "$MODEL_IMAGE"
  minikube image ls | grep -q "inspectedge-ai-model:${MODEL_IMAGE##*:}" \
    || die "image not visible in minikube after load"
fi

docker image ls --format '{{.Repository}}:{{.Tag}}  {{.Size}}' | grep -E '^inspectedge-ai' || true
log "done. Next: scripts/k8s/deploy.sh"
