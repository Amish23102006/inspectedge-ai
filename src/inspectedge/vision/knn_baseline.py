"""Baseline A: pretrained patch features + nearest-neighbour distance to a normal memory bank.

Trained on defect-free images only. Each test patch is scored by its distance to the closest
normal patch; the heatmap is that distance map, upsampled and smoothed.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol

import numpy as np
from sklearn.neighbors import NearestNeighbors

from inspectedge.vision.base import score_from_heatmaps, smooth_and_resize


class FeatureExtractor(Protocol):
    def __call__(self, images: np.ndarray) -> np.ndarray:
        """(N,H,W,3) uint8 -> (N,C,h,w) float32 patch features."""
        ...


class KnnPatchModel:
    method = "knn_patch"

    def __init__(
        self,
        extractor: FeatureExtractor,
        k: int = 1,
        memory_bank_max: int = 20000,
        seed: int = 42,
        batch_size: int = 8,
        smoothing_sigma: float = 4.0,
    ) -> None:
        self.extractor = extractor
        self.k = k
        self.memory_bank_max = memory_bank_max
        self.seed = seed
        self.batch_size = batch_size
        self.smoothing_sigma = smoothing_sigma
        self.memory_bank: np.ndarray | None = None
        self._nn: NearestNeighbors | None = None

    def _batches(self, images: np.ndarray):
        for i in range(0, len(images), self.batch_size):
            yield self.extractor(images[i : i + self.batch_size])

    def fit(self, images: np.ndarray) -> KnnPatchModel:
        rng = np.random.default_rng(self.seed)
        per_image = max(1, int(np.ceil(self.memory_bank_max / len(images))))
        chunks = []
        for feats in self._batches(images):
            n, c, h, w = feats.shape
            patches = feats.transpose(0, 2, 3, 1).reshape(n, h * w, c)
            for p in patches:  # random per-image subsample keeps memory bounded
                take = min(per_image, len(p))
                chunks.append(p[rng.choice(len(p), size=take, replace=False)])
        bank = np.concatenate(chunks).astype(np.float32)
        if len(bank) > self.memory_bank_max:
            bank = bank[rng.choice(len(bank), size=self.memory_bank_max, replace=False)]
        self._set_bank(bank)
        return self

    def _set_bank(self, bank: np.ndarray) -> None:
        self.memory_bank = bank
        self._nn = NearestNeighbors(n_neighbors=min(self.k, len(bank))).fit(bank)

    def predict(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if self._nn is None:
            raise RuntimeError("model is not fitted")
        out_hw = images.shape[1:3]
        maps = []
        for feats in self._batches(images):
            n, c, h, w = feats.shape
            q = feats.transpose(0, 2, 3, 1).reshape(-1, c)
            dist, _ = self._nn.kneighbors(q)
            patch = dist.mean(axis=1).reshape(n, h, w)
            maps.extend(smooth_and_resize(m, out_hw, self.smoothing_sigma) for m in patch)
        heatmaps = np.stack(maps).astype(np.float32)
        return score_from_heatmaps(heatmaps), heatmaps

    def save(self, directory: str | Path) -> None:
        if self.memory_bank is None:
            raise RuntimeError("model is not fitted")
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "memory_bank.npy", self.memory_bank)
        meta = {
            "method": self.method,
            "k": self.k,
            "memory_bank_max": self.memory_bank_max,
            "seed": self.seed,
            "smoothing_sigma": self.smoothing_sigma,
        }
        (d / "knn_meta.json").write_text(json.dumps(meta, indent=2))

    @classmethod
    def load(cls, directory: str | Path, extractor: FeatureExtractor) -> KnnPatchModel:
        d = Path(directory)
        meta = json.loads((d / "knn_meta.json").read_text())
        model = cls(
            extractor,
            k=meta["k"],
            memory_bank_max=meta["memory_bank_max"],
            seed=meta["seed"],
            smoothing_sigma=meta["smoothing_sigma"],
        )
        model._set_bank(np.load(d / "memory_bank.npy"))
        return model
