#!/usr/bin/env bash
# Ramp load with k6 (in-cluster), record latency/throughput/errors plus HPA replicas, CPU and
# memory every 10 s, then keep watching until the HPA scales back down to its minimum.
#
#   MAX_VUS=12 scripts/k8s/load-test.sh
# Results: reports/k8s/<timestamp>-load/{k6.log,timeline.tsv,summary.txt}
set -euo pipefail
# shellcheck source=scripts/k8s/lib.sh
source "$(dirname "$0")/lib.sh"
cd "$REPO_ROOT"

MAX_VUS="${MAX_VUS:-12}"
SCALE_DOWN_WAIT="${SCALE_DOWN_WAIT:-600}"   # seconds to wait for scale-down after the test
OUT="$(results_dir load)"
TL="$OUT/timeline.tsv"
MIN="$(kubectl -n "$NS" get hpa inspectedge-api -o jsonpath='{.spec.minReplicas}')"
T0=$(now)

sample() {
  local t cpu cur des top
  t=$(( $(now) - T0 ))
  cpu=$(kubectl -n "$NS" get hpa inspectedge-api \
        -o jsonpath='{.status.currentMetrics[0].resource.current.averageUtilization}' 2>/dev/null)
  cur=$(kubectl -n "$NS" get hpa inspectedge-api -o jsonpath='{.status.currentReplicas}')
  des=$(kubectl -n "$NS" get hpa inspectedge-api -o jsonpath='{.status.desiredReplicas}')
  top=$(kubectl -n "$NS" top pods -l "$APP_LABEL" --no-headers 2>/dev/null \
        | awk '{printf "%s=%s/%s ", $1, $2, $3}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$t" "${cpu:-?}" "$cur" "$des" "$(ready_pods)" "$top" >> "$TL"
}

printf 't_s\thpa_cpu_pct\tcurrent\tdesired\tready\tpod=cpu/mem\n' > "$TL"
kubectl -n "$NS" get hpa inspectedge-api
start_k6 ramp "$MAX_VUS"

log "load running (~7.5 min). Live view in another terminal:"
log "  kubectl -n $NS get hpa -w      |   kubectl -n $NS top pods"
while ! kubectl -n "$NS" get job k6-ramp \
        -o jsonpath='{.status.conditions[*].type}' 2>/dev/null | grep -qE 'Complete|Failed'; do
  sample; tail -1 "$TL"; sleep 10
done
T_LOAD_END=$(( $(now) - T0 ))
finish_k6 ramp "$OUT/k6.log" | tee "$OUT/summary.txt"

log "load finished; waiting up to ${SCALE_DOWN_WAIT}s for scale-down to $MIN replicas"
while :; do
  sample
  cur=$(kubectl -n "$NS" get hpa inspectedge-api -o jsonpath='{.status.currentReplicas}')
  [ "$cur" -le "$MIN" ] && break
  [ $(( $(now) - T0 - T_LOAD_END )) -ge "$SCALE_DOWN_WAIT" ] && { log "scale-down wait timed out"; break; }
  sleep 15
done

PEAK=$(awk -F'\t' 'NR>1 && $3>m {m=$3} END {print m+0}' "$TL")
FIRST_UP=$(awk -F'\t' -v min="$MIN" 'NR>1 && $3>min {print $1; exit}' "$TL")
BACK_DOWN=$(awk -F'\t' -v min="$MIN" -v e="$T_LOAD_END" 'NR>1 && $1>e && $3<=min {print $1-e; exit}' "$TL")
{
  echo
  echo "=== HPA behaviour ==="
  echo "min replicas:              $MIN"
  echo "peak replicas:             $PEAK"
  echo "first scale-up at:         ${FIRST_UP:-never} s after start"
  echo "back to min after load:    ${BACK_DOWN:-not within wait} s"
  echo "timeline:                  $TL"
} | tee -a "$OUT/summary.txt"
log "results saved in $OUT  (copy the numbers into docs/results.md)"
