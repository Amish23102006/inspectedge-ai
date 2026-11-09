# Architecture (MVP)

```mermaid
flowchart LR
    U[Inspector] --> D[Streamlit dashboard]
    D -->|REST| A[FastAPI]
    A --> V[Upload validation]
    V --> Q[Quality gate]
    Q -->|fail| R[RECAPTURE]
    Q -->|pass| G[Scope gate<br/>global embedding vs training set]
    G -->|unlike training setup| O[OUT_OF_SCOPE]
    G -->|in scope| M[Anomaly model<br/>kNN patch / autoencoder]
    M --> P[Decision policy<br/>calibrated thresholds]
    P --> H[Heatmap store]
    P --> S[(SQLite audit:<br/>inspections, reviews, events)]
    D -->|review + reason code| A
    E[run_eval CLI] --> MA[(Model artefact +<br/>model_version.json)]
    E --> RP[(Report: metrics.json,<br/>error gallery, provenance)]
    MA --> M
```

## Core records

| Entity | Key fields | Where |
|---|---|---|
| Inspection | inspection_id, created_at, input_sha256, category, model_version, preprocessing_version | `schemas.InspectionResponse` |
| Prediction | anomaly_score, thresholds, decision, scope (in_scope, distance, threshold), heatmap_uri, latency_ms | same |
| HumanReview | reviewer_id, disposition, reason_code, comment, reviewed_at | `schemas.HumanReview` |
| ModelVersion | model_id, method, dataset, category, config_hash, seed, metrics_uri, artefact_uri, status | `schemas.ModelVersion` |
| AuditEvent | actor, action, object_id, request_id, result, timestamp | `schemas.AuditEvent` |

Planned: `SensorWindow` (Phase 5), monitoring/drift store (Phase 6), ONNX artefacts with hardware profile (Phase 4).
