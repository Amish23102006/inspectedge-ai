"""Out-of-scope gate: does this image look like the inspection setup the model was trained on?

An anomaly model trained on one product can only say "how different is this from the normal
parts I saw". It cannot tell "a defective bottle" from "not a bottle at all", so without this gate
any unrelated photo is reported as LIKELY_ANOMALY. The gate summarises each image as one global
embedding (spatial mean of the same pretrained patch features, L2-normalised) and measures the
cosine distance to the nearest training image. The threshold is calibrated on held-out NORMAL
images only (``margin * quantile``); the test set is never used to choose it.

Defects are local, so a defective part usually stays close to the normal parts in this global
view; the evaluation reports how many test anomalies the gate would wrongly turn away.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np

from inspectedge.schemas import ScopeReport


class FeatureExtractor(Protocol):
    def __call__(self, images: np.ndarray) -> np.ndarray:
        """(N,H,W,3) uint8 -> (N,C,h,w) float32 patch features."""
        ...


class ScopeGate:
    def __init__(self, extractor: FeatureExtractor, batch_size: int = 8) -> None:
        self.extractor = extractor
        self.batch_size = batch_size
        self.reference: np.ndarray | None = None
        self.threshold: float | None = None
        self.calibration = "uncalibrated"

    def embed(self, images: np.ndarray) -> np.ndarray:
        """(N,H,W,3) uint8 -> (N,C) L2-normalised global embeddings."""
        out = []
        for i in range(0, len(images), self.batch_size):
            feats = self.extractor(images[i : i + self.batch_size])
            out.append(feats.mean(axis=(2, 3)))
        emb = np.concatenate(out).astype(np.float32)
        return emb / np.maximum(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12)

    def distances(self, images: np.ndarray) -> np.ndarray:
        if self.reference is None:
            raise RuntimeError("scope gate is not fitted")
        sims = self.embed(images) @ self.reference.T
        return np.clip(1.0 - sims.max(axis=1), 0.0, 2.0).astype(np.float64)

    def fit(
        self, train_images: np.ndarray, val_images: np.ndarray, quantile: float, margin: float
    ) -> ScopeGate:
        """Reference = training images; threshold from held-out normal validation images only."""
        if len(train_images) < 1 or len(val_images) < 2:
            raise ValueError("need >= 1 training and >= 2 validation images to fit the scope gate")
        self.reference = self.embed(train_images)
        val_d = self.distances(val_images)
        self.threshold = float(margin * np.quantile(val_d, quantile))
        self.calibration = (
            f"{margin} x q{quantile} of held-out normal distances "
            f"(n={len(val_d)}, max={val_d.max():.4f})"
        )
        return self

    def assess(self, images: np.ndarray) -> list[ScopeReport]:
        if self.threshold is None:
            raise RuntimeError("scope gate is not fitted")
        return [
            ScopeReport(
                in_scope=bool(d <= self.threshold), distance=float(d), threshold=self.threshold
            )
            for d in self.distances(images)
        ]

    def save(self, directory: str | Path) -> None:
        if self.reference is None or self.threshold is None:
            raise RuntimeError("scope gate is not fitted")
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "scope_reference.npy", self.reference)
        meta = {"threshold": self.threshold, "calibration": self.calibration}
        (d / "scope_gate.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: str | Path, extractor: FeatureExtractor) -> ScopeGate:
        d = Path(directory)
        meta = json.loads((d / "scope_gate.json").read_text())
        gate = cls(extractor)
        gate.reference = np.load(d / "scope_reference.npy")
        gate.threshold = float(meta["threshold"])
        gate.calibration = meta["calibration"]
        return gate

    @staticmethod
    def exists(directory: str | Path) -> bool:
        return (Path(directory) / "scope_gate.json").exists()
