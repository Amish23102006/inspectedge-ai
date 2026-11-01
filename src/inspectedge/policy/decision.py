"""Deterministic three-way decision policy (+ RECAPTURE for poor input quality).

The model produces a score; this module alone turns it into a decision. No generative model
is involved. Human review remains required for the final disposition.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np

from inspectedge.schemas import Decision, Thresholds


def decide(score: float | None, quality_passed: bool, thresholds: Thresholds) -> Decision:
    if not quality_passed:
        return Decision.RECAPTURE
    if score is None or not math.isfinite(score):
        raise ValueError(f"a finite anomaly score is required when quality passed, got {score!r}")
    if score < thresholds.pass_threshold:
        return Decision.PASS
    if score < thresholds.likely_anomaly_threshold:
        return Decision.MANUAL_REVIEW
    return Decision.LIKELY_ANOMALY


def calibrate_thresholds(
    normal_val_scores: Sequence[float] | np.ndarray,
    pass_quantile: float,
    likely_anomaly_quantile: float,
) -> Thresholds:
    """Calibrate thresholds from scores of held-out DEFECT-FREE images only.

    pass_threshold = q(pass_quantile): roughly (1 - pass_quantile) of normal parts go to review.
    likely_anomaly_threshold = q(likely_anomaly_quantile).
    With small validation sets, high quantiles approach the max score; report this limitation.
    """
    scores = np.asarray(normal_val_scores, dtype=np.float64)
    if scores.size < 2 or not np.all(np.isfinite(scores)):
        raise ValueError("need >= 2 finite normal validation scores to calibrate")
    return Thresholds(
        pass_threshold=float(np.quantile(scores, pass_quantile)),
        likely_anomaly_threshold=float(np.quantile(scores, likely_anomaly_quantile)),
        calibration=(
            f"normal-val quantiles p={pass_quantile}, l={likely_anomaly_quantile}, n={scores.size}"
        ),
    )
