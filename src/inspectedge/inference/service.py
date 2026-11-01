"""Inspection service: validate -> quality gate -> model -> deterministic decision -> heatmap."""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

import cv2

from inspectedge.config import AppConfig
from inspectedge.inference.uploads import decode_upload
from inspectedge.inference.visualize import overlay, save_rgb_png
from inspectedge.policy.decision import decide
from inspectedge.quality.gate import assess_quality
from inspectedge.repro import sha256_bytes
from inspectedge.schemas import InspectionResponse, ModelVersion, Thresholds, utcnow
from inspectedge.vision.base import AnomalyModel


@dataclass
class InspectionService:
    model: AnomalyModel
    thresholds: Thresholds
    model_version: str
    category: str
    cfg: AppConfig
    heatmap_dir: Path

    def __post_init__(self) -> None:
        self.heatmap_dir = Path(self.heatmap_dir)
        self.heatmap_dir.mkdir(parents=True, exist_ok=True)

    def heatmap_path(self, inspection_id: str) -> Path:
        return self.heatmap_dir / f"{uuid.UUID(inspection_id)}.png"  # UUID check blocks traversal

    def inspect(self, data: bytes, content_type: str | None) -> InspectionResponse:
        t0 = time.perf_counter()
        image = decode_upload(data, content_type, self.cfg.upload)
        size = self.cfg.data.image_size
        resized = cv2.resize(image, (size, size), interpolation=cv2.INTER_AREA)
        quality = assess_quality(resized, self.cfg.quality)
        inspection_id = str(uuid.uuid4())

        score: float | None = None
        heatmap_uri: str | None = None
        if quality.passed:  # never score an image the gate rejected
            scores, maps = self.model.predict(resized[None])
            score = float(scores[0])
            vis = overlay(resized, maps[0], 0.0, self.thresholds.likely_anomaly_threshold)
            save_rgb_png(self.heatmap_path(inspection_id), vis)
            heatmap_uri = f"/heatmaps/{inspection_id}"

        decision = decide(score, quality.passed, self.thresholds)
        return InspectionResponse(
            inspection_id=inspection_id,
            created_at=utcnow(),
            category=self.category,
            model_version=self.model_version,
            preprocessing_version=self.cfg.preprocessing_version,
            input_sha256=sha256_bytes(data),
            anomaly_score=score,
            thresholds=self.thresholds,
            decision=decision,
            quality=quality,
            heatmap_uri=heatmap_uri,
            latency_ms=round((time.perf_counter() - t0) * 1000, 3),
        )


def load_service(artefact_dir: str | Path, cfg: AppConfig) -> InspectionService:
    """Load a model saved by run_eval. Requires the [ml] extra."""
    d = Path(artefact_dir)
    meta = json.loads((d / "model_version.json").read_text())
    mv = ModelVersion.model_validate(meta["model_version"])
    thresholds = Thresholds.model_validate(meta["thresholds"])
    if meta["preprocessing_version"] != cfg.preprocessing_version:
        raise ValueError(
            f"artefact preprocessing {meta['preprocessing_version']} != "
            f"config {cfg.preprocessing_version}"
        )
    model: AnomalyModel
    if mv.method == "knn_patch":
        from inspectedge.vision.features import TorchvisionFeatureExtractor
        from inspectedge.vision.knn_baseline import KnnPatchModel

        ext = TorchvisionFeatureExtractor(cfg.model.backbone, cfg.model.layers, cfg.model.device)
        model = KnnPatchModel.load(d, ext)
    elif mv.method == "autoencoder":
        from inspectedge.vision.autoencoder import AutoencoderModel

        model = AutoencoderModel.load(d, cfg.model.device)
    else:
        raise ValueError(f"unknown method {mv.method}")
    return InspectionService(
        model, thresholds, mv.model_id, mv.category, cfg, Path(cfg.paths.artefacts) / "heatmaps"
    )
