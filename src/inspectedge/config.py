"""Typed experiment / application configuration loaded from YAML."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

MVTEC_CATEGORIES: tuple[str, ...] = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", protected_namespaces=())


class ProjectConfig(_Strict):
    seed: int = 42


class DataConfig(_Strict):
    root: Path = Path("data/raw/mvtec_ad")
    category: str = "bottle"
    image_size: int = Field(256, ge=32, le=1024)
    val_fraction: float = Field(0.2, gt=0.0, lt=0.5)

    @field_validator("category")
    @classmethod
    def _known_category(cls, v: str) -> str:
        if v not in MVTEC_CATEGORIES:
            raise ValueError(f"unknown MVTec AD category {v!r}; expected one of {MVTEC_CATEGORIES}")
        return v


class ModelConfig(_Strict):
    method: Literal["knn_patch", "autoencoder"] = "knn_patch"
    backbone: str = "resnet18"
    layers: list[str] = Field(default_factory=lambda: ["layer2", "layer3"])
    k: int = Field(1, ge=1)
    memory_bank_max: int = Field(20000, ge=100)
    ae_epochs: int = Field(50, ge=1)
    ae_batch_size: int = Field(16, ge=1)
    ae_lr: float = Field(1e-3, gt=0)
    device: str = "cpu"


class PolicyConfig(_Strict):
    pass_quantile: float = Field(0.95, gt=0, lt=1)
    likely_anomaly_quantile: float = Field(0.995, gt=0, le=1)

    @model_validator(mode="after")
    def _ordered(self) -> PolicyConfig:
        if self.pass_quantile >= self.likely_anomaly_quantile:
            raise ValueError("pass_quantile must be < likely_anomaly_quantile")
        return self


class QualityConfig(_Strict):
    min_blur_variance: float = 20.0
    min_brightness: float = 25.0
    max_brightness: float = 230.0
    min_contrast: float = 8.0


class ScopeConfig(_Strict):
    """Out-of-scope gate: refuse to score images unlike the training setup.

    threshold = margin * quantile(held-out normal distances); test data is never used.
    """

    enabled: bool = True
    quantile: float = Field(1.0, gt=0, le=1)
    margin: float = Field(1.5, ge=1.0)


class UploadConfig(_Strict):
    max_bytes: int = Field(10 * 1024 * 1024, gt=0)
    max_pixels: int = Field(40_000_000, gt=0)
    allowed_content_types: list[str] = Field(default_factory=lambda: ["image/png", "image/jpeg"])


class PathsConfig(_Strict):
    artefacts: Path = Path("artefacts")
    reports: Path = Path("reports/runs")
    db: Path = Path("artefacts/inspectedge.db")


class AppConfig(_Strict):
    project: ProjectConfig = Field(default_factory=ProjectConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    model: ModelConfig = Field(default_factory=ModelConfig)
    policy: PolicyConfig = Field(default_factory=PolicyConfig)
    quality: QualityConfig = Field(default_factory=QualityConfig)
    scope: ScopeConfig = Field(default_factory=ScopeConfig)
    upload: UploadConfig = Field(default_factory=UploadConfig)
    paths: PathsConfig = Field(default_factory=PathsConfig)

    @property
    def preprocessing_version(self) -> str:
        return f"rgb-resize{self.data.image_size}-v1"


def load_config(path: str | Path) -> AppConfig:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    return AppConfig.model_validate(raw)


def config_hash(cfg: AppConfig) -> str:
    """Short, stable hash of the full config, stored with every result."""
    payload = json.dumps(cfg.model_dump(mode="json"), sort_keys=True).encode()
    return hashlib.sha256(payload).hexdigest()[:12]
