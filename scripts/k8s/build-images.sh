#!/usr/bin/env bash
# Build the base image and the model image *inside* minikube's Docker, so the cluster can use
# them with imagePullPolicy: Never (nothing is pushed to any registry).
#
#   scripts/k8s/build-images.sh [ARTEFACT_DIR]
# ARTEFACT_DIR defaults to the newest artefacts/*/ that contains model_version.json
# (created by `make eval`).
set -euo pipefail
# shellcheck source=scripts/k8s/lib.sh
source "$(dirname "$0")/lib.sh"
cd "$REPO_ROOT"

ART="${1:-${INSPECTEDGE_ARTEFACT_DIR:-}}"
if [ -z "$ART" ]; then
  # shellcheck disable=SC2012
  ART="$(ls -1dt artefacts/*/ 2>/dev/null | while read -r d; do
           [ -f "$d/model_version.json" ] && { echo "${d%/}"; break; }; done || true)"
fi
[ -n "$ART" ] && [ -f "$ART/model_version.json" ] || die "no trained model artefact found.
Run 'make prepare && make eval' first (see README), or pass the artefact dir as an argument."
log "model artefact: $ART"

if [ "${USE_HOST_DOCKER:-0}" != "1" ]; then
  need minikube "Install with: brew install minikube"
  # shellcheck disable=SC2046
  eval $(minikube docker-env)
  log "building inside minikube's Docker daemon"
fi

log "building $BASE_IMAGE"
docker build -f docker/Dockerfile -t "$BASE_IMAGE" .

log "building $MODEL_IMAGE"
docker build -f docker/Dockerfile.model --build-arg BASE_IMAGE="$BASE_IMAGE" \
  -t "$MODEL_IMAGE" "$ART"

docker image ls --format '{{.Repository}}:{{.Tag}}  {{.Size}}' | grep -E '^inspectedge-ai' || true
log "done. Next: scripts/k8s/deploy.sh"
