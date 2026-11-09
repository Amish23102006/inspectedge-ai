import math

import pytest

from inspectedge.policy.decision import calibrate_thresholds, decide
from inspectedge.schemas import Decision, Thresholds


@pytest.mark.parametrize(
    ("score", "expected"),
    [
        (0.5, Decision.PASS),
        (1.0, Decision.MANUAL_REVIEW),  # boundary: equal to pass threshold -> review
        (1.5, Decision.MANUAL_REVIEW),
        (2.0, Decision.LIKELY_ANOMALY),  # boundary: equal to likely threshold
        (9.0, Decision.LIKELY_ANOMALY),
    ],
)
def test_three_way_decision(score, expected, thresholds):
    assert decide(score, True, thresholds) is expected


def test_quality_failure_overrides_any_score(thresholds):
    assert decide(9.0, False, thresholds) is Decision.RECAPTURE
    assert decide(None, False, thresholds) is Decision.RECAPTURE
    assert decide(None, False, thresholds, in_scope=False) is Decision.RECAPTURE


def test_out_of_scope_is_not_scored(thresholds):
    assert decide(None, True, thresholds, in_scope=False) is Decision.OUT_OF_SCOPE
    assert decide(0.1, True, thresholds, in_scope=False) is Decision.OUT_OF_SCOPE


@pytest.mark.parametrize("bad", [None, math.nan, math.inf])
def test_missing_or_nonfinite_score_raises(bad, thresholds):
    with pytest.raises(ValueError):
        decide(bad, True, thresholds)


def test_calibration_is_deterministic_and_ordered():
    scores = [float(i) for i in range(100)]
    a = calibrate_thresholds(scores, 0.95, 0.995)
    b = calibrate_thresholds(scores, 0.95, 0.995)
    assert a == b
    assert a.pass_threshold == pytest.approx(94.05)
    assert a.pass_threshold <= a.likely_anomaly_threshold
    assert "n=100" in a.calibration


def test_calibration_rejects_bad_input():
    with pytest.raises(ValueError):
        calibrate_thresholds([1.0], 0.9, 0.99)
    with pytest.raises(ValueError):
        calibrate_thresholds([1.0, math.nan], 0.9, 0.99)


def test_thresholds_must_be_ordered():
    with pytest.raises(ValueError):
        Thresholds(pass_threshold=2.0, likely_anomaly_threshold=1.0)
