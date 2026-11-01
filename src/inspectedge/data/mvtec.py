"""MVTec AD indexing, deterministic splitting, integrity and leakage checks.

Expected layout (as distributed by MVTec):
    <root>/<category>/train/good/*.png
    <root>/<category>/test/<defect_type or good>/*.png
    <root>/<category>/ground_truth/<defect_type>/<stem>_mask.png
"""

from __future__ import annotations

import csv
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from inspectedge.repro import sha256_file

IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp"}


@dataclass(frozen=True)
class Sample:
    path: Path
    split: str  # "train" | "val" | "test"
    label: int  # 0 = normal, 1 = anomalous
    defect_type: str
    mask_path: Path | None


def _list_images(d: Path) -> list[Path]:
    if not d.is_dir():
        return []
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS)


def index_category(root: str | Path, category: str, val_fraction: float, seed: int) -> list[Sample]:
    cat_dir = Path(root) / category
    if not cat_dir.is_dir():
        raise FileNotFoundError(
            f"category folder not found: {cat_dir}. See data/README.md for download steps."
        )
    train_good = _list_images(cat_dir / "train" / "good")
    if len(train_good) < 2:
        raise ValueError(f"need >= 2 defect-free training images in {cat_dir / 'train' / 'good'}")

    # Deterministic split: sorted file list + seeded permutation.
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(train_good))
    n_val = min(len(train_good) - 1, max(1, round(len(train_good) * val_fraction)))
    val_idx = set(order[:n_val].tolist())

    samples = [
        Sample(p, "val" if i in val_idx else "train", 0, "good", None)
        for i, p in enumerate(train_good)
    ]

    test_dir = cat_dir / "test"
    if not test_dir.is_dir():
        raise FileNotFoundError(f"test folder not found: {test_dir}")
    for defect_dir in sorted(d for d in test_dir.iterdir() if d.is_dir()):
        defect = defect_dir.name
        for p in _list_images(defect_dir):
            if defect == "good":
                samples.append(Sample(p, "test", 0, "good", None))
            else:
                m = cat_dir / "ground_truth" / defect / f"{p.stem}_mask.png"
                samples.append(Sample(p, "test", 1, defect, m if m.exists() else None))
    return samples


def integrity_report(samples: list[Sample]) -> dict:
    counts = Counter(f"{s.split}/{'anomalous' if s.label else 'normal'}" for s in samples)
    missing = [str(s.path) for s in samples if s.label == 1 and s.mask_path is None]
    defects = sorted({s.defect_type for s in samples if s.label == 1})
    return {
        "counts": dict(sorted(counts.items())),
        "defect_types": defects,
        "missing_masks": missing,
        "n_total": len(samples),
    }


def find_leakage(samples: list[Sample]) -> list[dict]:
    """Return byte-identical files that appear in more than one split."""
    by_hash: dict[str, list[Sample]] = defaultdict(list)
    for s in samples:
        by_hash[sha256_file(s.path)].append(s)
    leaks = []
    for h, group in by_hash.items():
        splits = {s.split for s in group}
        if len(splits) > 1:
            leaks.append(
                {"sha256": h, "splits": sorted(splits), "files": [str(s.path) for s in group]}
            )
    return leaks


def write_manifest(samples: list[Sample], out_csv: str | Path) -> None:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "split", "label", "defect_type", "mask_path"])
        for s in samples:
            w.writerow(
                [
                    s.path.as_posix(),
                    s.split,
                    s.label,
                    s.defect_type,
                    s.mask_path.as_posix() if s.mask_path else "",
                ]
            )


def _imread(path: Path, flags: int) -> np.ndarray:
    # np.fromfile + imdecode works with non-ASCII Windows paths, unlike cv2.imread.
    data = np.fromfile(str(path), dtype=np.uint8)
    img = cv2.imdecode(data, flags)
    if img is None:
        raise ValueError(f"could not decode image: {path}")
    return img


def load_image(path: str | Path, size: int) -> np.ndarray:
    """Load as RGB uint8, resized to (size, size)."""
    img = cv2.cvtColor(_imread(Path(path), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)


def load_mask(path: str | Path | None, size: int) -> np.ndarray:
    if path is None:
        return np.zeros((size, size), dtype=bool)
    m = _imread(Path(path), cv2.IMREAD_GRAYSCALE)
    return cv2.resize(m, (size, size), interpolation=cv2.INTER_NEAREST) > 127


def load_split(
    samples: list[Sample], split: str, size: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[Sample]]:
    chosen = [s for s in samples if s.split == split]
    if not chosen:
        raise ValueError(f"no samples in split {split!r}")
    images = np.stack([load_image(s.path, size) for s in chosen])
    labels = np.array([s.label for s in chosen], dtype=np.int64)
    masks = np.stack([load_mask(s.mask_path, size) for s in chosen])
    return images, labels, masks, chosen
