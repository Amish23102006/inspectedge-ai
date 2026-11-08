"""Fetch one MVTec AD category from the public Voxel51 mirror on Hugging Face.

Used by the hosted demo (``streamlit_app.py``) so the cloud app can train on first start.
Rebuilds the official MVTec layout that :mod:`inspectedge.data.mvtec` expects:

    <root>/<category>/train/good/*.png
    <root>/<category>/test/<defect_type or good>/*.png
    <root>/<category>/ground_truth/<defect_type>/<stem>_mask.png

MVTec AD is licensed CC BY-NC-SA 4.0 (non-commercial). Cite Bergmann et al., CVPR 2019 / IJCV 2021.
"""

from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

REPO_ID = "Voxel51/mvtec-ad"


def _jobs(samples: list[dict], category: str, cat_dir: Path) -> list[tuple[str, Path]]:
    """Map mirror samples of ``category`` to (remote path, local destination) pairs."""
    jobs: list[tuple[str, Path]] = []
    for s in samples:
        if s.get("category", {}).get("label") != category:
            continue
        split = s["split"]
        defect = s["defect"]["label"]
        stem = Path(s["filepath"]).stem
        jobs.append((s["filepath"], cat_dir / split / defect / f"{stem}.png"))
        mask = (s.get("defect_mask") or {}).get("mask_path")
        if defect != "good" and mask:
            jobs.append((mask, cat_dir / "ground_truth" / defect / f"{stem}_mask.png"))
    return jobs


def fetch_category(category: str, root: str | Path, workers: int = 16) -> Path:
    """Download ``category`` into ``root`` (no-op if it is already there). Returns its folder."""
    from huggingface_hub import hf_hub_download

    cat_dir = Path(root) / category
    if any((cat_dir / "train" / "good").glob("*.png")):
        return cat_dir

    meta = Path(hf_hub_download(REPO_ID, "samples.json", repo_type="dataset"))
    jobs = _jobs(json.loads(meta.read_text(encoding="utf-8"))["samples"], category, cat_dir)
    if not jobs:
        raise ValueError(f"category {category!r} not found in {REPO_ID}")

    def fetch(job: tuple[str, Path]) -> None:
        remote, dst = job
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(hf_hub_download(REPO_ID, remote, repo_type="dataset"), dst)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, jobs))
    return cat_dir
