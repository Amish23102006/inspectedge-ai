# ADR 0001: Normal-only anomaly detection with a deterministic three-way policy

**Status:** accepted · **Date:** project start

## Context
Industrial inspection has abundant defect-free images and scarce, unrepresentative defect images; new defect types appear. A supervised good/bad classifier would need defect labels it cannot have and would not generalise to unseen defect types.

## Decision
1. Train only on defect-free images (MVTec AD `train/good`), one category at a time.
2. Start with two simple, explainable baselines under one shared data pipeline and evaluation script: (A) pretrained patch features + nearest-neighbour distance; (B) convolutional autoencoder reconstruction error.
3. Image score = max of the smoothed anomaly heatmap (score and localisation are consistent by construction).
4. Calibrate thresholds from held-out normal validation scores only. Never tune on test.
5. A deterministic policy maps score + input quality to PASS / MANUAL_REVIEW / LIKELY_ANOMALY / RECAPTURE. Human review is always required for disposition.
6. A quality gate runs first; failing images are never scored.

## Consequences
- Metrics are AUROC / AUPR / pixel AUROC / operational rates, not accuracy.
- Thresholds from a small validation set are noisy; report n and bootstrap CIs.
- Each category needs its own model; cross-category transfer must be evaluated, not assumed.
- Heavier infrastructure (Kubernetes, queues, agents) is out of scope until a measured need exists.
