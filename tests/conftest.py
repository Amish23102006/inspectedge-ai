from pathlib import Path

import cv2
import numpy as np
import pytest

from inspectedge.config import AppConfig
from inspectedge.schemas import Thresholds


def png_bytes(img_rgb: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))
    assert ok
    return buf.tobytes()


def noise_image(size: int = 256, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(40, 215, (size, size, 3), dtype=np.uint8)


def flat_image(size: int = 256, value: int = 128) -> np.ndarray:
    return np.full((size, size, 3), value, dtype=np.uint8)


class FakeModel:
    """Deterministic stand-in for a trained model (no torch needed)."""

    method = "fake"

    def __init__(self, score: float) -> None:
        self.score = score
        self.calls = 0

    def predict(self, images):
        self.calls += 1
        n, h, w = images.shape[:3]
        return np.full(n, self.score), np.zeros((n, h, w), dtype=np.float32)


@pytest.fixture
def cfg(tmp_path: Path) -> AppConfig:
    c = AppConfig()
    c.paths.artefacts = tmp_path / "artefacts"
    c.paths.db = tmp_path / "artefacts" / "test.db"
    return c


@pytest.fixture
def thresholds() -> Thresholds:
    return Thresholds(pass_threshold=1.0, likely_anomaly_threshold=2.0, calibration="test")


def _write(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), img)


@pytest.fixture
def mvtec_tree(tmp_path: Path) -> Path:
    """Tiny synthetic MVTec-style tree: 10 train, 3 test-good, 2 defects (1 missing mask)."""
    root = tmp_path / "mvtec"
    cat = root / "bottle"
    for i in range(10):
        _write(cat / "train/good" / f"{i:03d}.png", noise_image(32, seed=i))
    for i in range(3):
        _write(cat / "test/good" / f"{i:03d}.png", noise_image(32, seed=100 + i))
    for i in range(2):
        _write(cat / "test/broken_large" / f"{i:03d}.png", noise_image(32, seed=200 + i))
    mask = np.zeros((32, 32), np.uint8)
    mask[8:16, 8:16] = 255
    _write(cat / "ground_truth/broken_large/000_mask.png", mask)  # 001 deliberately missing
    return root
