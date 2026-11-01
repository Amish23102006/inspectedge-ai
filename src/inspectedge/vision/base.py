"""Common model interface and heatmap post-processing."""

from __future__ import annotations

from typing import Protocol

import cv2
import numpy as np


class AnomalyModel(Protocol):
    method: str

    def predict(self, images: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """images: (N,H,W,3) uint8 RGB -> (scores (N,), heatmaps (N,H,W) float32)."""
        ...


def smooth_and_resize(patch_map: np.ndarray, out_hw: tuple[int, int], sigma: float) -> np.ndarray:
    h, w = out_hw
    up = cv2.resize(patch_map.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    if sigma > 0:
        up = cv2.GaussianBlur(up, (0, 0), sigmaX=sigma)
    return up


def score_from_heatmaps(heatmaps: np.ndarray) -> np.ndarray:
    """Image score = max of the (smoothed) heatmap, so the score and heatmap always agree."""
    return heatmaps.reshape(len(heatmaps), -1).max(axis=1).astype(np.float64)
