# InspectEdge AI

**Explainable industrial anomaly detection with human review, reproducible evaluation and (planned) edge benchmarking.**

**▶ Live demo: [amish-inspectedge-ai.streamlit.app](https://amish-inspectedge-ai.streamlit.app)**: pick a sample bottle image (or upload one) and get a decision, heatmap and audit trail. The first visit after a restart trains the model (~8 min); later visits are instant.

> ⚠️ Research / portfolio prototype trained on public data. Not validated for production rejection or safety-critical decisions. A human reviewer always makes the final disposition.

## Problem

Factories have many images of normal parts but few, unrepresentative defect examples, and new defect types appear over time. InspectEdge therefore learns **only from defect-free images**, flags deviations with an anomaly score and a localisation heatmap, separates bad imaging from bad parts, and routes uncertain cases to a human.

**Intended users:** inspectors (upload → decision → review), quality engineers (false alarms / misses / thresholds), ML engineers (reproduce, compare, benchmark), auditors (trace every decision).

**Non-goals:** live production rejection, autonomous rejection without human review, use of any proprietary plant data, "accuracy" as a headline metric, assuming one category's results transfer to another factory.

## Workflow

```
upload → validation (type, magic bytes, size, pixel limit)
       → quality gate (blur / brightness / contrast) ──fail──► RECAPTURE (model not called)
       → scope gate (does it look like this product's setup?) ──no──► OUT_OF_SCOPE (not scored)
       → anomaly model → score + heatmap
       → calibrated thresholds → PASS | MANUAL_REVIEW | LIKELY_ANOMALY
       → human disposition + reason code → SQLite audit trail
```

**Why a scope gate?** A model trained only on normal bottles can answer "how different is this from the bottles I know?" but not "is this a bottle at all?", so without the gate any unrelated photo comes back as LIKELY_ANOMALY. The gate compares the image's global embedding with the training images and refuses to score anything outside the calibrated range. Its threshold comes from held-out normal images plus synthetic defects made from them (cut-paste patches, brightness changes, scratches), never from the test set.

The score is the maximum of the smoothed heatmap, so the score and the highlighted region always agree. Heatmap colours use a fixed scale (0 → likely-anomaly threshold), not per-image normalisation, so normal images do not look "hot".

## Quick start in VS Code

1. **Open the folder** in VS Code and accept the recommended extensions prompt (Python, Pylance, Ruff, Docker, Jupyter…).
2. **Create the environment** — `Ctrl+Shift+P` → **Tasks: Run Task** and run, in order:
   - `Setup: create .venv` (needs Python 3.10–3.12; 3.11 recommended)
   - `Setup: install core + api + dashboard + dev`
   - `Setup: install PyTorch (CPU)`
   Then `Ctrl+Shift+P` → **Python: Select Interpreter** → choose `.venv`.
3. **Run the tests** — `Ctrl+Shift+P` → **Tasks: Run Test Task**, or the Testing (flask) icon in the sidebar. Tests need neither PyTorch nor the dataset.
4. **Get the data** — follow [`data/README.md`](data/README.md) to download MVTec AD into `data/raw/mvtec_ad/`.
5. **Run & Debug panel** (`Ctrl+Shift+D`), in order:
   - `1. Prepare data` — indexes the category, writes a manifest, checks for missing masks and train/test leakage.
   - `2. Train + evaluate baseline` — trains Baseline A, calibrates thresholds on held-out normals, writes `reports/runs/<model_id>/metrics.json` + an error gallery, and prints the artefact path.
   - Copy `.env.example` to `.env` and set `INSPECTEDGE_ARTEFACT_DIR` to the printed artefact path.
   - `API + Dashboard` (compound) — API on <http://localhost:8000/docs>, dashboard on <http://localhost:8501>.

## Run on Kubernetes (minikube)

With a trained artefact from step 5, deploy the API to a local cluster with probes, resource limits, autoscaling (2–5 pods), Prometheus metrics, an in-cluster k6 load test and failure drills:

```bash
brew install minikube kubectl            # plus Docker Desktop
scripts/k8s/cluster-up.sh && scripts/k8s/build-images.sh && scripts/k8s/deploy.sh
scripts/k8s/smoke-test.sh
scripts/k8s/load-test.sh                 # results -> reports/k8s/<timestamp>-load/
scripts/k8s/recovery-test.sh             # results -> reports/k8s/<timestamp>-recovery/
scripts/k8s/cleanup.sh
```

Full guide, design notes and limitations: [`docs/kubernetes.md`](docs/kubernetes.md). Measured results: [`docs/results.md`](docs/results.md).

Prefer a container? **Dev Containers: Reopen in Container** uses `.devcontainer/` and installs everything. macOS/Linux users can also use `make install`, `make test`, `make eval`, etc.

## Repository layout

```
configs/default.yaml        experiment config (hash stored with every result)
data/README.md              dataset download, licence and layout
docs/                       architecture + architecture decision records
src/inspectedge/
  config.py, schemas.py     typed config and Pydantic schemas
  data/                     MVTec indexing, deterministic split, leakage check, prepare CLI
  quality/                  input-quality gate
  policy/                   deterministic decision policy + threshold calibration
  vision/                   Baseline A (kNN patch features), Baseline B (autoencoder)
  evaluation/               metrics + run_eval CLI
  inference/                upload validation, inspection service, heatmap rendering
  audit/                    SQLite audit store
  observability.py          Prometheus metrics + readiness state
api/                        FastAPI (/inspect, /reviews, /inspections/{id}, /heatmaps/{id},
                            /health, /health/live, /health/ready, /metrics)
dashboard/                  Streamlit reviewer UI
tests/                      pytest suite (runs without torch or data)
docker/, .github/workflows/ container build (pinned, non-root) and CI (tests + kind smoke test)
requirements/               hashed lock file for the Docker image
k8s/                        namespace, ConfigMaps, Deployment, Service, HPA (kubectl apply -k k8s/)
scripts/k8s/                cluster up, build, deploy, smoke, load, recovery, cleanup
tests/load/                 k6 load test (runs as an in-cluster Job)
monitoring/                 metrics reference, PromQL, optional Prometheus/Grafana setup
reports/                    model card; run reports are generated (git-ignored)
```

## Methods

| Level | Method | Status |
|---|---|---|
| Baseline A | ImageNet ResNet-18 layer2+layer3 patch features, nearest-neighbour distance to a normal memory bank | Implemented |
| Baseline B | Convolutional autoencoder, per-pixel reconstruction error | Implemented |
| Baseline C | Stronger patch-based industrial method (e.g. coreset memory bank) | Planned |

**Threshold selection:** the defect-free training images are split (seeded) into train and validation. Thresholds are quantiles of validation normal scores (`policy.pass_quantile`, `policy.likely_anomaly_quantile`). The test set is never used to choose thresholds. With small validation sets the high quantile approaches the maximum normal score; this is reported in the calibration string.

## Results

Every number below comes from a recorded run with its dataset, split, seed, config hash, command and environment. Reproduce it with the manual **Evaluate (MVTec AD)** workflow (`.github/workflows/eval.yml`) or `make eval` locally.

| Category | Method | Image AUROC (95% CI) | Image AUPR | Pixel AUROC | False alarms / 100 @ pass thr. | Missed anomalies routed to PASS | Report |
|---|---|---|---|---|---|---|---|
| bottle | Baseline A: kNN patch features (ResNet-18 layer2+3) | 1.000 (1.000–1.000) | 1.000 | 0.983 | 6.0 | 0 of 63 | [report](reports/evaluations/knn_patch-bottle-1239f723b8d1.json) · [run](https://github.com/Amish23102006/inspectedge-ai/actions/runs/42527445412) |

**Scope gate** (same run; threshold 0.031 cosine distance):

| Images | Accepted for inspection | Rejected as OUT_OF_SCOPE |
|---|---|---|
| bottle test, normal (20) | 20 (100%) | 0 |
| bottle test, defective (63) | 63 (100%) | 0 |
| Other MVTec AD objects, test split (694: cable, screw, hazelnut, metal nut, toothbrush, carpet) | 0 | 694 (100%); closest at 0.066, about 2× the threshold |

How this was reached, for transparency: the first version calibrated the gate on normal bottles alone. Normal bottles are nearly identical, so that threshold (0.0017) also turned away 54 of the 63 defective test bottles ([run](https://github.com/Amish23102006/inspectedge-ai/actions/runs/42526900884)). Calibration was then changed to normals + synthetic defects; the threshold itself still never sees test data, but the design change was prompted by that test result. "Other objects" are other MVTec categories photographed in a similar studio setup; everyday photos are usually much further away.

**Reading these numbers honestly:** the test set is small (83 images: 20 normal, 63 defective) and *bottle* is one of the easier MVTec AD categories, so image AUROC saturates at 1.0 and the CI is degenerate. The operationally useful figures are the threshold ones: at the pass threshold no defective bottle is passed, while 5 of 20 normal bottles are not auto-passed (3 sent to manual review, 2 flagged as likely anomalies; 6.0 false alarms per 100 images). Decision thresholds were calibrated only on 42 held-out normal images; the test set was never used to choose them. Results on one category say nothing about other categories or real factory images.

## Project status

| Phase | Deliverable | Status |
|---|---|---|
| 0 Design | README, data licence, architecture, ADR | ✅ |
| 1 Vision MVP | One category, two baselines, heatmap, metrics, report | ✅ Baseline A measured on *bottle* / ⏳ Baseline B run |
| 2 Workflow | API, dashboard, three-way decision, review, audit | ✅ MVP |
| 3 Robustness | Lighting / blur / contrast / shift tests | ⬜ not started |
| 4 Edge | ONNX export, parity, INT8 quantisation, CPU benchmark | ⬜ not started |
| 5 Sensors | NASA CMAPSS RUL baselines | ⬜ not started |
| 6 Portfolio quality | Docker, CI, model card, security checks | 🟡 partial |
| 6b Kubernetes | Probes, HPA, metrics, load + recovery tests on minikube | ✅ measured (docs/results.md) |
| 7 Research | VisA external validation, ablations | ⬜ not started |

## Security, privacy and limitations

- Uploads: allow-listed MIME types, magic-byte check, byte and pixel (decompression) limits; heatmaps stored under random UUID names; UUID-typed routes block path traversal.
- Decisions come only from deterministic model scores and configured thresholds. No generative model decides anything.
- **No authentication yet**: `reviewer_id` is self-declared. Do not expose the API publicly. Role-based access is on the roadmap.
- No rate limiting or malware scanning yet; SQLite is single-node only (on Kubernetes each pod has its own audit DB; see `docs/kubernetes.md`).
- Models are trained per category; results on one MVTec category say nothing about another factory or product.
- A heatmap is a model explanation, not a verified physical root cause.

## Data licence

MVTec AD is **CC BY-NC-SA 4.0 (non-commercial)**. It is not redistributed in this repository. See [`data/README.md`](data/README.md). Code in this repository is MIT-licensed.
