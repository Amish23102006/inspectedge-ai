"""Out-of-scope gate: does this image look like the inspection setup the model was trained on?

An anomaly model trained on one product can only say "how different is this from the normal
parts I saw". It cannot tell "a defective bottle" from "not a bottle at all", so without this gate
any unrelated photo is reported as LIKELY_ANOMALY. The gate summarises each image as one global
embedding (spatial mean of the same pretrained patch features, L2-normalised) and measures the
cosine distance to the nearest training image.

Calibration never touches the test set. Normal parts of one product are nearly identical, so a
threshold from normal images alone would also turn away genuinely DEFECTIVE parts (measured:
54 of 63 test defects). The threshold is therefore ``margin * quantile`` of the distances of the
held-out normal images AND synthetic defects made from them (cut-paste patches, brightness
changes, scratches), i.e. "how far can a damaged part of this product drift and still be this
product". The evaluation reports acceptance of real test defects and rejection of other objects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from inspectedge.schemas import ScopeReport


class FeatureExtractor(Protocol):
    def __call__(self, images: np.ndarray) -> np.ndarray:
        """(N,H,W,3) uint8 -> (N,C,h,w) float32 patch features."""
        ...


def synthetic_defects(images: np.ndarray, per_image: int, seed: int) -> np.ndarray:
    """Local corruptions of normal images (cut-paste, brightness, scratches) for calibration."""
    rng = np.random.default_rng(seed)
    n, h, w, _ = images.shape
    out = []
    for img in images:
        for _ in range(per_image):
            x = img.copy()
            s = max(2, int(rng.uniform(0.1, 0.3) * min(h, w)))
            sy, sx, ty, tx = (int(v) for v in rng.integers(0, [h - s, w - s, h - s, w - s]))
            patch = np.rot90(img[sy : sy + s, sx : sx + s], k=int(rng.integers(0, 4)))
            shift = int(rng.integers(-60, 61))
            x[ty : ty + s, tx : tx + s] = np.clip(patch.astype(np.int16) + shift, 0, 255)
            if rng.random() < 0.5:
                p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
                p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
                colour = tuple(int(c) for c in rng.integers(0, 256, 3))
                cv2.line(x, p1, p2, colour, int(rng.integers(1, max(2, h // 40))))
            out.append(x)
    return np.stack(out)


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
        self,
        train_images: np.ndarray,
        val_images: np.ndarray,
        quantile: float,
        margin: float,
        synthetic_per_image: int = 4,
        seed: int = 42,
    ) -> ScopeGate:
        """Reference = training images. Threshold from held-out normals + synthetic defects."""
        if len(train_images) < 1 or len(val_images) < 2:
            raise ValueError("need >= 1 training and >= 2 validation images to fit the scope gate")
        self.reference = self.embed(train_images)
        calib = val_images
        if synthetic_per_image > 0:
            calib = np.concatenate(
                [val_images, synthetic_defects(val_images, synthetic_per_image, seed)]
            )
        d = self.distances(calib)
        self.threshold = float(margin * np.quantile(d, quantile))
        self.calibration = (
            f"{margin} x q{quantile} of distances of {len(val_images)} held-out normals + "
            f"{len(calib) - len(val_images)} synthetic defects (max={d.max():.4f})"
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
