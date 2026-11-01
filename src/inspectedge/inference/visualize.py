"""Heatmap rendering. Colours use FIXED threshold-based scaling, not per-image normalisation,
so a normal image does not look 'hot' just because its own max is stretched to red."""

from __future__ import annotations

import cv2
import numpy as np


def heatmap_to_rgb(heatmap: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    span = max(vmax - vmin, 1e-12)
    norm = np.clip((heatmap - vmin) / span, 0.0, 1.0)
    colored = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    return cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)


def overlay(
    image_rgb: np.ndarray, heatmap: np.ndarray, vmin: float, vmax: float, alpha: float = 0.45
) -> np.ndarray:
    colored = heatmap_to_rgb(heatmap, vmin, vmax)
    return cv2.addWeighted(image_rgb, 1 - alpha, colored, alpha, 0)


def save_rgb_png(path, image_rgb: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise ValueError("PNG encoding failed")
    buf.tofile(str(path))
