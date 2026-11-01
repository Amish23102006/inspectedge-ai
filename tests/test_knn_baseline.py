import numpy as np
import pytest

from inspectedge.vision.knn_baseline import KnnPatchModel


def block_mean_extractor(images: np.ndarray) -> np.ndarray:
    """Toy extractor: 8x8 block means of grayscale -> (N,1,H/8,W/8). No torch required."""
    g = images.mean(axis=3)
    n, h, w = g.shape
    return g.reshape(n, h // 8, 8, w // 8, 8).mean(axis=(2, 4))[:, None].astype(np.float32)


def normals(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(100 + rng.normal(0, 3, (n, 64, 64, 3)), 0, 255).astype(np.uint8)


def test_defect_scores_higher_and_heatmap_localises():
    model = KnnPatchModel(block_mean_extractor, memory_bank_max=5000).fit(normals(10))
    test = normals(2, seed=1)
    test[1, 40:56, 8:24] = 230  # synthetic bright defect
    scores, maps = model.predict(test)
    assert maps.shape == (2, 64, 64) and np.all(np.isfinite(scores))
    assert scores[1] > 5 * scores[0]
    r, c = np.unravel_index(np.argmax(maps[1]), maps[1].shape)
    assert 36 <= r < 60 and 4 <= c < 28
    assert scores[1] == pytest.approx(maps[1].max())  # score is the heatmap max


def test_memory_bank_respects_budget_and_seed():
    a = KnnPatchModel(block_mean_extractor, memory_bank_max=100, seed=3).fit(normals(10))
    b = KnnPatchModel(block_mean_extractor, memory_bank_max=100, seed=3).fit(normals(10))
    assert len(a.memory_bank) <= 100
    np.testing.assert_array_equal(a.memory_bank, b.memory_bank)


def test_save_load_round_trip(tmp_path):
    model = KnnPatchModel(block_mean_extractor).fit(normals(5))
    model.save(tmp_path)
    loaded = KnnPatchModel.load(tmp_path, block_mean_extractor)
    x = normals(2, seed=9)
    np.testing.assert_allclose(model.predict(x)[0], loaded.predict(x)[0])


def test_predict_before_fit_raises():
    with pytest.raises(RuntimeError):
        KnnPatchModel(block_mean_extractor).predict(normals(1))
