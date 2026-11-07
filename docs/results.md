# Kubernetes results

Measured on a local minikube cluster. Raw evidence for every number is committed under
`reports/k8s/` (linked in each table). These numbers describe this laptop and this model,
not production capacity.

## Environment

| Item | Value |
|---|---|
| Machine | MacBook Air, Apple Silicon (arm64), macOS 14.6 |
| Container runtime | Docker Desktop 4.98.0 (VM: 4 CPUs, 8 GB) |
| Cluster | minikube v1.39.0, docker driver, containerd, 1 node with 4 CPUs / 6 GB |
| Kubernetes | v1.37.0, Metrics Server addon |
| Image | `inspectedge-ai-model:0.2.0` (python 3.11-slim, torch 2.7.1+cpu, linux/arm64) |
| Model | `knn_patch-bottle-6016e191570e`: kNN patch features (ResNet-18 layer2+3), MVTec AD *bottle* |
| Model quality (same run, `reports/runs/`) | image AUROC 1.000, image AUPR 1.000, pixel AUROC 0.983 (83 test images: 20 normal, 63 defective) |
| Pod resources | requests 250m CPU / 512Mi, limits 1 CPU / 1Gi, 1 uvicorn worker, max 2 concurrent inferences |
| HPA | autoscaling/v2, min 2, max 5, target 65% of CPU request |

## Load test (`scripts/k8s/load-test.sh`)

In-cluster k6 Job, `POST /inspect` with a defect-free MVTec bottle image (256×256),
ramp 4 → 12 virtual users over 4 min, hold 2 min, ramp down 1 min.

| Requests | Throughput | p50 | p90 | p95 | p99 | max | Error rate |
|---|---|---|---|---|---|---|---|
| 1,212 | 2.88 req/s | 2,349 ms | 4,325 ms | 5,316 ms | 6,335 ms | 6,634 ms | **0.00 %** |

| HPA behaviour | Value |
|---|---|
| Replicas | 2 → **5** (max) |
| First scale-up | 114–124 s after start (CPU ≈ 232% of request when the HPA acted) |
| Peak CPU (HPA view) | 399% of request (~1 core per busy pod, i.e. at the 1-CPU limit) |
| Back to 2 replicas after load stopped | 292 s (120 s stabilisation window + 1 pod / 30 s policy + metrics lag) |
| Pod memory | 375 MiB idle, **max 439 MiB** under load (request 512Mi, limit 1Gi): no OOM |
| Restarts during test | 0 |

Raw: [`reports/k8s/20261108-002127-load/`](../reports/k8s/20261108-002127-load/)
(`k6.log`, `timeline.tsv`, `summary.txt`).

**Interpretation**

- Single-request latency is ~0.75 s (smoke test). The higher p50/p95 under load is queueing:
  at 12 users every busy pod sat at its 1-CPU limit, and 5 pods × 1 CPU exceeds the node's
  4 CPUs, so the cluster was saturated. Capacity on this laptop is ≈ 3 inspections/s.
- One of the three pods added by the HPA stayed almost idle (2m CPU). k6 reuses keep-alive
  connections and kube-proxy balances per connection, so new pods only receive traffic from
  new connections. Documented in `docs/kubernetes.md` → Known limitations.
- An earlier attempt was discarded because two load tests were started at the same time.

## Recovery test (`scripts/k8s/recovery-test.sh`)

Background traffic: k6 at a constant 2 req/s for 8 min while the drills ran.

| Drill | Result |
|---|---|
| A. Pod deleted | Replacement created automatically and **Ready after 7 s**; deployment back to 2/2 |
| B. Readiness failed (`/tmp/inspectedge.drain`) | Removed from Service endpoints after **10 s**, **0 restarts** (liveness unaffected), back in endpoints **4 s** after the file was removed |
| C. Rollout with 60 s slow startup | Completed in **266 s** (5 pods rolled one at a time, `maxUnavailable: 0`); old pods kept serving |
| Background traffic during all drills | **960 requests, 0.00 % errors**, p50 901 ms, p95 5,296 ms |

Raw: [`reports/k8s/20261108-003329-recovery/`](../reports/k8s/20261108-003329-recovery/)
(`recovery.txt`, `k6.log`, `events.txt`).

The HPA had scaled to 5 replicas under the 2 req/s background load by the time drill C ran,
which is why C rolled 5 pods. Drill C2 (startup over budget) was not run.

## Resource tuning decisions

| Date | Change | Evidence |
|---|---|---|
| 2026-11-08 | Kept requests 250m / 512Mi and limits 1 CPU / 1Gi. Max memory 439 MiB stays under the request; CPU is the bottleneck, so more replicas (or a larger node) help more than a bigger limit. | load test above |

## Screenshots

`docs/screenshots/` (add): HPA + pods during scale-up, `kubectl top pods` at peak.
