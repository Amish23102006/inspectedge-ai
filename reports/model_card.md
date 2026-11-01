# Model card — InspectEdge AI (template, fill only from a saved run report)

| Field | Value |
|---|---|
| Model ID | _from reports/runs/<model_id>/metrics.json_ |
| Method | |
| Training data | MVTec AD, category: ___, defect-free train images only, n = ___ |
| Threshold calibration | held-out normal validation, n = ___, quantiles ___ |
| Evaluation split | test normal n = ___, anomalous n = ___ |
| Seed / config hash | |
| Hardware / software | _from provenance.environment_ |

## Intended use
Research and portfolio demonstration of normal-only anomaly detection with human review.

## Out of scope
Production rejection, safety-critical decisions, other product categories or factories without re-evaluation.

## Metrics
_Copy from the run report only. No estimates._

## Known failure modes
_From the error gallery: highest-scoring normals and lowest-scoring anomalies._

## Licence and ethics
Trained on CC BY-NC-SA 4.0 data: non-commercial use only. No personal or proprietary data.
