"""Input-quality gate: separates poor imaging (blur, exposure, contrast) from product anomalies."""

from __future__ import annotations

import cv2
import numpy as np

from inspectedge.config import QualityConfig
from inspectedge.schemas import QualityReport


def assess_quality(image: np.ndarray, cfg: QualityConfig) -> QualityReport:
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    elif image.ndim == 2:
        gray = image
    else:
        raise ValueError(f"expected HxW or HxWx3 image, got shape {image.shape}")

    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    brightness = float(gray.mean())
    contrast = float(gray.std())

    warnings: list[str] = []
    if blur < cfg.min_blur_variance:
        warnings.append(f"BLUR: Laplacian variance {blur:.1f} < {cfg.min_blur_variance}")
    if brightness < cfg.min_brightness:
        warnings.append(f"TOO_DARK: mean {brightness:.1f} < {cfg.min_brightness}")
    if brightness > cfg.max_brightness:
        warnings.append(f"TOO_BRIGHT: mean {brightness:.1f} > {cfg.max_brightness}")
    if contrast < cfg.min_contrast:
        warnings.append(f"LOW_CONTRAST: std {contrast:.1f} < {cfg.min_contrast}")

    return QualityReport(
        passed=not warnings,
        blur_variance=round(blur, 3),
        brightness=round(brightness, 3),
        contrast=round(contrast, 3),
        warnings=warnings,
    )
