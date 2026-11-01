import cv2
import numpy as np

from inspectedge.config import QualityConfig
from inspectedge.quality.gate import assess_quality
from tests.conftest import flat_image, noise_image

Q = QualityConfig()


def test_sharp_textured_image_passes():
    assert assess_quality(noise_image(), Q).passed


def test_flat_image_fails_blur_and_contrast():
    r = assess_quality(flat_image(), Q)
    assert not r.passed
    assert any(w.startswith("BLUR") for w in r.warnings)
    assert any(w.startswith("LOW_CONTRAST") for w in r.warnings)


def test_dark_and_bright_images_flagged():
    dark = (noise_image() * 0.1).astype(np.uint8)
    bright = np.clip(noise_image().astype(int) + 200, 0, 255).astype(np.uint8)
    assert any(w.startswith("TOO_DARK") for w in assess_quality(dark, Q).warnings)
    assert any(w.startswith("TOO_BRIGHT") for w in assess_quality(bright, Q).warnings)


def test_heavy_blur_detected():
    blurred = cv2.GaussianBlur(noise_image(), (0, 0), sigmaX=8)
    r = assess_quality(blurred, Q)
    assert any(w.startswith("BLUR") for w in r.warnings)


def test_grayscale_input_supported():
    assert assess_quality(noise_image()[..., 0], Q).passed
