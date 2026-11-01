import pytest
from pydantic import ValidationError

from inspectedge.config import AppConfig
from inspectedge.schemas import (
    Decision,
    HumanReviewCreate,
    InspectionResponse,
    QualityReport,
    utcnow,
)

GOOD_Q = QualityReport(passed=True, blur_variance=100, brightness=120, contrast=40)
BAD_Q = QualityReport(passed=False, blur_variance=1, brightness=120, contrast=2, warnings=["BLUR"])


def _resp(thresholds, **kw):
    base = dict(
        inspection_id="x",
        created_at=utcnow(),
        category="bottle",
        model_version="m",
        preprocessing_version="p",
        input_sha256="a" * 64,
        anomaly_score=0.5,
        thresholds=thresholds,
        decision=Decision.PASS,
        quality=GOOD_Q,
        heatmap_uri="/heatmaps/x",
        latency_ms=1.0,
    )
    base.update(kw)
    return InspectionResponse(**base)


def test_response_json_round_trip(thresholds):
    r = _resp(thresholds)
    again = InspectionResponse.model_validate_json(r.model_dump_json())
    assert again == r
    assert again.human_review_required is True


def test_recapture_allows_missing_score(thresholds):
    r = _resp(
        thresholds, decision=Decision.RECAPTURE, anomaly_score=None, quality=BAD_Q, heatmap_uri=None
    )
    assert r.anomaly_score is None


@pytest.mark.parametrize(
    "kw",
    [
        {"anomaly_score": None},  # non-recapture without a score
        {"anomaly_score": float("inf")},
        {"decision": Decision.RECAPTURE},  # recapture although quality passed
        {"input_sha256": "not-a-hash"},
        {"latency_ms": -1},
    ],
)
def test_invalid_responses_rejected(thresholds, kw):
    with pytest.raises(ValidationError):
        _resp(thresholds, **kw)


def test_review_validation():
    ok = HumanReviewCreate(
        inspection_id="i",
        reviewer_id="amish_01",
        disposition="FALSE_ALARM",
        reason_code="REFLECTION",
    )
    assert ok.disposition.value == "FALSE_ALARM"
    for bad in [{"reason_code": "lowercase"}, {"reviewer_id": "../etc"}, {"comment": "x" * 1001}]:
        payload = {
            "inspection_id": "i",
            "reviewer_id": "r",
            "disposition": "ACCEPT",
            "reason_code": "OK_PART",
            **bad,
        }
        with pytest.raises(ValidationError):
            HumanReviewCreate(**payload)


def test_config_rejects_unknown_category_and_keys():
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"data": {"category": "tyre"}})
    with pytest.raises(ValidationError):
        AppConfig.model_validate({"data": {"typo_key": 1}})


def test_default_yaml_loads():
    from inspectedge.config import load_config

    cfg = load_config("configs/default.yaml")
    assert cfg.data.category == "bottle"
    assert cfg.preprocessing_version == "rgb-resize256-v1"
