import math

import numpy as np
import pytest

from inspectedge.evaluation.metrics import (
    bootstrap_auroc_ci,
    decision_metrics,
    image_level_metrics,
    pixel_auroc,
    threshold_metrics,
)
from inspectedge.schemas import Decision as D  # noqa: N817


def test_perfect_separation():
    m = image_level_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9]))
    assert m == {"image_auroc": 1.0, "image_aupr": 1.0}


def test_single_class_gives_nan():
    assert math.isnan(image_level_metrics(np.array([0, 0]), np.array([1.0, 2.0]))["image_auroc"])


def test_threshold_metrics_counts():
    m = threshold_metrics(np.array([0, 0, 1, 1]), np.array([0.1, 0.9, 0.2, 0.8]), 0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == m["recall"] == m["f1"] == 0.5
    assert m["false_alarms_per_100_images"] == 25.0


def test_decision_metrics_counts_missed_anomalies():
    m = decision_metrics(
        np.array([1, 0, 1, 0]), [D.PASS, D.LIKELY_ANOMALY, D.MANUAL_REVIEW, D.RECAPTURE]
    )
    assert m["missed_anomalies_passed"] == 1
    assert m["normals_flagged_likely_anomaly"] == 1
    assert m["manual_review_rate"] == m["recapture_rate"] == 0.25


def test_pixel_auroc():
    masks = np.zeros((1, 4, 4), bool)
    masks[0, :2] = True
    maps = masks.astype(float)
    assert pixel_auroc(masks, maps) == 1.0
    assert math.isnan(pixel_auroc(np.zeros((1, 4, 4), bool), maps))


def test_bootstrap_ci_contains_point_estimate_and_is_seeded():
    rng = np.random.default_rng(0)
    y = np.r_[np.zeros(50), np.ones(50)].astype(int)
    s = np.r_[rng.normal(0, 1, 50), rng.normal(1.5, 1, 50)]
    point = image_level_metrics(y, s)["image_auroc"]
    lo, hi = bootstrap_auroc_ci(y, s, n_boot=300, seed=1)
    assert lo <= point <= hi
    assert (lo, hi) == pytest.approx(bootstrap_auroc_ci(y, s, n_boot=300, seed=1))
