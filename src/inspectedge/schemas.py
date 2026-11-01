"""Pydantic schemas shared by the evaluation code, API, audit store and dashboard."""

from __future__ import annotations

import math
from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

DISCLAIMER = (
    "Research/portfolio prototype trained on public data. Not validated for production "
    "rejection or safety-critical decisions. A human reviewer makes the final disposition."
)


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class _Base(BaseModel):
    model_config = ConfigDict(protected_namespaces=())


class Decision(str, Enum):
    PASS = "PASS"
    MANUAL_REVIEW = "MANUAL_REVIEW"
    LIKELY_ANOMALY = "LIKELY_ANOMALY"
    RECAPTURE = "RECAPTURE"


class Disposition(str, Enum):
    ACCEPT = "ACCEPT"
    CONFIRMED_DEFECT = "CONFIRMED_DEFECT"
    FALSE_ALARM = "FALSE_ALARM"
    RECAPTURE_REQUESTED = "RECAPTURE_REQUESTED"
    ESCALATE = "ESCALATE"


class Thresholds(_Base):
    model_config = ConfigDict(frozen=True, protected_namespaces=())

    pass_threshold: float
    likely_anomaly_threshold: float
    calibration: str = "unspecified"

    @model_validator(mode="after")
    def _ordered(self) -> Thresholds:
        if not (
            math.isfinite(self.pass_threshold) and math.isfinite(self.likely_anomaly_threshold)
        ):
            raise ValueError("thresholds must be finite")
        if self.pass_threshold > self.likely_anomaly_threshold:
            raise ValueError("pass_threshold must be <= likely_anomaly_threshold")
        return self


class QualityReport(_Base):
    passed: bool
    blur_variance: float
    brightness: float
    contrast: float
    warnings: list[str] = Field(default_factory=list)


class InspectionResponse(_Base):
    inspection_id: str
    created_at: datetime
    category: str
    model_version: str
    preprocessing_version: str
    input_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    anomaly_score: float | None
    thresholds: Thresholds
    decision: Decision
    quality: QualityReport
    heatmap_uri: str | None
    latency_ms: float = Field(ge=0)
    human_review_required: Literal[True] = True
    disclaimer: str = DISCLAIMER

    @model_validator(mode="after")
    def _consistent(self) -> InspectionResponse:
        if self.anomaly_score is not None and not math.isfinite(self.anomaly_score):
            raise ValueError("anomaly_score must be finite")
        if self.decision is Decision.RECAPTURE:
            if self.quality.passed:
                raise ValueError("RECAPTURE is only valid when the input-quality gate failed")
        elif self.anomaly_score is None:
            raise ValueError("anomaly_score is required unless decision is RECAPTURE")
        return self


class HumanReviewCreate(_Base):
    inspection_id: str
    reviewer_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    disposition: Disposition
    reason_code: str = Field(pattern=r"^[A-Z][A-Z0-9_]{1,31}$")
    comment: str = Field("", max_length=1000)


class HumanReview(HumanReviewCreate):
    reviewed_at: datetime


class ModelVersion(_Base):
    model_id: str
    method: str
    dataset: str
    category: str
    config_hash: str
    seed: int
    metrics_uri: str
    artefact_uri: str
    status: Literal["candidate", "registered", "retired"] = "candidate"
    created_at: datetime = Field(default_factory=utcnow)


class AuditEvent(_Base):
    actor: str
    action: str
    object_id: str
    request_id: str
    result: str
    timestamp: datetime = Field(default_factory=utcnow)


class InspectionRecord(_Base):
    inspection: InspectionResponse
    reviews: list[HumanReview]
    audit: list[AuditEvent]
