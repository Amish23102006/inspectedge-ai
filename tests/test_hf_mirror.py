from pathlib import Path

from inspectedge.data.hf_mirror import _jobs


def _sample(path, category, split, defect, mask=None):
    s = {
        "filepath": path,
        "split": split,
        "category": {"label": category},
        "defect": {"label": defect},
    }
    if mask:
        s["defect_mask"] = {"mask_path": mask}
    return s


def test_jobs_rebuild_mvtec_layout():
    root = Path("root/bottle")
    samples = [
        _sample("data/a/001-1.png", "bottle", "train", "good"),
        _sample("data/b/002-9.png", "bottle", "test", "broken_large", "fields/m/002_mask-3.png"),
        _sample("data/c/003-4.png", "cable", "test", "good"),
    ]
    jobs = _jobs(samples, "bottle", root)
    assert jobs == [
        ("data/a/001-1.png", root / "train" / "good" / "001-1.png"),
        ("data/b/002-9.png", root / "test" / "broken_large" / "002-9.png"),
        ("fields/m/002_mask-3.png", root / "ground_truth" / "broken_large" / "002-9_mask.png"),
    ]
