import numpy as np
import pytest

from inspectedge.quality.scope import ScopeGate


def rgb_mean_extractor(images: np.ndarray) -> np.ndarray:
    """Toy extractor: per-channel means -> (N,3,2,2). Colour direction stands in for 'setup'."""
    m = images.astype(np.float32).mean(axis=(1, 2))
    return np.repeat(np.repeat(m[:, :, None, None], 2, axis=2), 2, axis=3)


def grey_parts(n: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return np.clip(120 + rng.normal(0, 4, (n, 32, 32, 3)), 0, 255).astype(np.uint8)


@pytest.fixture
def gate() -> ScopeGate:
    return ScopeGate(rgb_mean_extractor).fit(grey_parts(10), grey_parts(5, seed=1), 1.0, 1.5)


def test_same_setup_in_scope_even_with_local_defect(gate):
    defect = grey_parts(1, seed=2)
    defect[0, 4:10, 4:10] = 250  # local bright defect barely moves the global embedding
    reports = gate.assess(np.concatenate([grey_parts(3, seed=3), defect]))
    assert all(r.in_scope for r in reports)


def test_unrelated_image_is_out_of_scope(gate):
    red = np.zeros((1, 32, 32, 3), np.uint8)
    red[..., 0] = 200
    (r,) = gate.assess(red)
    assert not r.in_scope and r.distance > r.threshold


def test_threshold_from_validation_only_and_round_trip(gate, tmp_path):
    assert gate.threshold is not None and gate.threshold > 0
    assert "n=5" in gate.calibration
    gate.save(tmp_path)
    assert ScopeGate.exists(tmp_path)
    loaded = ScopeGate.load(tmp_path, rgb_mean_extractor)
    probe = grey_parts(2, seed=9)
    np.testing.assert_allclose(loaded.distances(probe), gate.distances(probe))
    assert loaded.threshold == gate.threshold


def test_unfitted_gate_raises():
    g = ScopeGate(rgb_mean_extractor)
    with pytest.raises(RuntimeError):
        g.assess(grey_parts(1))
    with pytest.raises(ValueError):
        g.fit(grey_parts(2), grey_parts(1), 1.0, 1.5)
