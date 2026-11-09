"""Image-level, pixel-level and operational metrics. Accuracy is deliberately not reported."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from inspectedge.schemas import Decision


def _both_classes(labels: np.ndarray) -> bool:
    return bool(labels.min() != labels.max())


def image_level_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    labels, scores = np.asarray(labels), np.asarray(scores)
    if not _both_classes(labels):
        return {"image_auroc": float("nan"), "image_aupr": float("nan")}
    return {
        "image_auroc": float(roc_auc_score(labels, scores)),
        "image_aupr": float(average_precision_score(labels, scores)),
    }


def bootstrap_auroc_ci(
    labels: np.ndarray, scores: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float]:
    """Percentile bootstrap CI for image AUROC (resamples images with replacement)."""
    labels, scores = np.asarray(labels), np.asarray(scores)
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(labels), len(labels))
        if _both_classes(labels[idx]):
            vals.append(roc_auc_score(labels[idx], scores[idx]))
    if not vals:
        return float("nan"), float("nan")
    lo, hi = np.quantile(vals, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def threshold_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    labels = np.asarray(labels).astype(bool)
    pred = np.asarray(scores) >= threshold
    tp = int(np.sum(pred & labels))
    fp = int(np.sum(pred & ~labels))
    fn = int(np.sum(~pred & labels))
    tn = int(np.sum(~pred & ~labels))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "false_alarms_per_100_images": 100.0 * fp / len(labels),
    }


def decision_metrics(labels: np.ndarray, decisions: Sequence[Decision]) -> dict[str, float]:
    labels = np.asarray(labels).astype(bool)
    d = np.array([x.value for x in decisions])
    n = len(d)
    return {
        "n_images": n,
        "pass_rate": float(np.mean(d == "PASS")),
        "manual_review_rate": float(np.mean(d == "MANUAL_REVIEW")),
        "likely_anomaly_rate": float(np.mean(d == "LIKELY_ANOMALY")),
        "recapture_rate": float(np.mean(d == "RECAPTURE")),
        "out_of_scope_rate": float(np.mean(d == "OUT_OF_SCOPE")),
        # Worst operational error: an anomalous part auto-routed to PASS.
        "missed_anomalies_passed": int(np.sum(labels & (d == "PASS"))),
        "normals_flagged_likely_anomaly": int(np.sum(~labels & (d == "LIKELY_ANOMALY"))),
        "anomalies_routed_out_of_scope": int(np.sum(labels & (d == "OUT_OF_SCOPE"))),
    }


def pixel_auroc(masks: np.ndarray, heatmaps: np.ndarray) -> float:
    y = np.asarray(masks).ravel().astype(np.uint8)
    if y.min() == y.max():
        return float("nan")
    return float(roc_auc_score(y, np.asarray(heatmaps).ravel()))
