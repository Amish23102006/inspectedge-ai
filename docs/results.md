# Kubernetes results

**No results yet.** Fill this in only from a run whose raw files are committed under
`reports/k8s/` (link them in the last column). Do not put a number on a CV that is not in this
table.

## Environment

| Item | Value |
|---|---|
| Machine | e.g. MacBook Air M2, 16 GB |
| Cluster | minikube `<version>`, docker driver, 4 CPU / 6 GB |
| Kubernetes | `kubectl version` server version |
| Image | `inspectedge-ai-model:0.2.0`, model_id `<from /health>` |
| Commit | `<git sha>` |

## Load test (`scripts/k8s/load-test.sh`)

| Run | Profile | Requests | Throughput (req/s) | p50 (ms) | p95 (ms) | Error rate | Peak replicas | First scale-up (s) | Back to 2 replicas after load (s) | Max pod CPU / memory | Raw |
|---|---|---|---|---|---|---|---|---|---|---|---|
| – | ramp, 12 VUs | – | – | – | – | – | – | – | – | – | – |

## Recovery test (`scripts/k8s/recovery-test.sh`)

| Drill | Result | Raw |
|---|---|---|
| A. Pod deleted → replacement Ready | – s | – |
| B. Readiness failed → removed from endpoints / restarts / back in endpoints | – s / – / – s | – |
| C. Rollout with 60 s slow startup | – s, errors during rollout: – | – |
| C2. Startup over budget (optional) | restarts: –, old pods kept serving: – | – |
| Background traffic error rate during drills | – % | – |

## Resource tuning decisions

| Date | Change | Evidence |
|---|---|---|
| – | Started at requests 250m/512Mi, limits 1 CPU/1Gi | plan default |

## Screenshots

Add to `docs/screenshots/`: `kubectl get pods,hpa` during scale-up, `kubectl top pods`, and the
endpoints view during drill B.
