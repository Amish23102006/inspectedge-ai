# Monitoring

Every API pod exposes Prometheus metrics at `GET /metrics` (port 8000) and carries the usual
`prometheus.io/scrape|port|path` annotations.

## Metrics

| Metric | Type | Labels | Use |
|---|---|---|---|
| `inspectedge_http_requests_total` | counter | method, route, status | request rate, status mix |
| `inspectedge_http_errors_total` | counter | method, route | 5xx responses and unhandled exceptions |
| `inspectedge_http_request_duration_seconds` | histogram | method, route | latency percentiles |
| `inspectedge_http_requests_in_flight` | gauge | – | concurrency per pod |
| `inspectedge_inference_duration_seconds` | histogram | – | time inside the model pipeline only |
| `inspectedge_decisions_total` | counter | decision | PASS / MANUAL_REVIEW / LIKELY_ANOMALY / RECAPTURE mix |
| `inspectedge_model_loaded` | gauge | – | 0 means the pod can't serve `/inspect` |
| `inspectedge_model_info` | gauge | model_version, category | which model each pod serves |

`route` is the route template (`/inspections/{inspection_id}`), never a raw ID, so the number
of series stays small.

## Look at it without Prometheus

```bash
kubectl -n inspectedge port-forward svc/inspectedge-api 18080:80
curl -s localhost:18080/metrics | grep ^inspectedge_
```

## Prometheus + Grafana (after the basic deployment works)

```bash
helm repo add prometheus-community https://prometheus-community.github.io/helm-charts
helm install monitoring prometheus-community/kube-prometheus-stack \
  -n monitoring --create-namespace --set grafana.adminPassword="$(openssl rand -hex 12)"
kubectl apply -f monitoring/servicemonitor.yaml
kubectl -n monitoring port-forward svc/monitoring-grafana 3000:80    # http://localhost:3000
kubectl -n monitoring get secret monitoring-grafana -o jsonpath='{.data.admin-password}' | base64 -d
```

The chart needs about 2 GB of memory, so run it with `MEMORY=8g scripts/k8s/cluster-up.sh`.

## Useful queries

```promql
# Request rate and error ratio for /inspect
sum(rate(inspectedge_http_requests_total{route="/inspect"}[1m]))
sum(rate(inspectedge_http_errors_total{route="/inspect"}[5m]))
  / sum(rate(inspectedge_http_requests_total{route="/inspect"}[5m]))

# p50 / p95 latency for /inspect
histogram_quantile(0.50, sum by (le) (rate(inspectedge_http_request_duration_seconds_bucket{route="/inspect"}[1m])))
histogram_quantile(0.95, sum by (le) (rate(inspectedge_http_request_duration_seconds_bucket{route="/inspect"}[1m])))

# Pods currently able to serve, and in-flight requests per pod
sum(inspectedge_model_loaded)
inspectedge_http_requests_in_flight
```
