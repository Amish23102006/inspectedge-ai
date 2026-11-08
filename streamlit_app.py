"""Hosted demo entry point (Streamlit Community Cloud).

One process runs both halves of InspectEdge: the FastAPI backend in a background thread and the
Streamlit reviewer dashboard in front of it. On first start it downloads MVTec AD 'bottle',
trains Baseline A (kNN patch features) and calibrates thresholds on held-out normal images,
exactly as `make eval` does locally. Later visits reuse the trained model.

Run locally: streamlit run streamlit_app.py
"""

from __future__ import annotations

import os
import runpy
import socket
import threading
import time
from pathlib import Path

import streamlit as st

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
CONFIG = "configs/default.yaml"
API_HOST, API_PORT = "127.0.0.1", 8000
SAMPLES_DIR = ROOT / "artefacts" / "samples"

st.set_page_config(page_title="InspectEdge AI", page_icon="🔍", layout="wide")


def _train_if_needed() -> Path:
    from inspectedge.config import load_config
    from inspectedge.data.hf_mirror import fetch_category
    from inspectedge.evaluation import run_eval

    cfg = load_config(CONFIG)
    models = Path(cfg.paths.artefacts) / "models"
    found = sorted(models.glob(f"{cfg.model.method}-{cfg.data.category}-*/model_version.json"))
    if not found:
        fetch_category(cfg.data.category, cfg.data.root)
        if run_eval.main(["--config", CONFIG]) != 0:
            raise RuntimeError("training failed; see logs")
        found = sorted(models.glob(f"{cfg.model.method}-{cfg.data.category}-*/model_version.json"))
    _write_samples(Path(cfg.data.root) / cfg.data.category)
    return found[-1].parent


def _write_samples(cat_dir: Path) -> None:
    """Copy a few held-out test images (plus one blurred copy) for visitors to try."""
    import cv2

    if any(SAMPLES_DIR.glob("*.png")):
        return
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
    for defect_dir in sorted(p for p in (cat_dir / "test").iterdir() if p.is_dir()):
        for i, img in enumerate(sorted(defect_dir.glob("*.png"))[:2]):
            (SAMPLES_DIR / f"{defect_dir.name}_{i + 1}.png").write_bytes(img.read_bytes())
    good = sorted((cat_dir / "test" / "good").glob("*.png"))
    if good:
        blurred = cv2.GaussianBlur(cv2.imread(str(good[-1])), (0, 0), 12)
        cv2.imwrite(str(SAMPLES_DIR / "zz_blurry_photo.png"), blurred)


def _api_up() -> bool:
    try:
        with socket.create_connection((API_HOST, API_PORT), timeout=1):
            return True
    except OSError:
        return False


@st.cache_resource(
    show_spinner="First start: downloading MVTec AD 'bottle' and training the model. "
    "This takes a few minutes once; later visits are instant."
)
def start_backend() -> str:
    artefact = _train_if_needed()
    os.environ["INSPECTEDGE_ARTEFACT_DIR"] = str(artefact)
    if not _api_up():
        import uvicorn

        from api.main import create_app

        server = uvicorn.Server(
            uvicorn.Config(create_app(), host=API_HOST, port=API_PORT, log_level="warning")
        )
        threading.Thread(target=server.run, daemon=True, name="inspectedge-api").start()
        for _ in range(120):
            if _api_up():
                break
            time.sleep(0.5)
    return f"http://{API_HOST}:{API_PORT}"


os.environ["INSPECTEDGE_API_URL"] = start_backend()
os.environ["INSPECTEDGE_SAMPLES_DIR"] = str(SAMPLES_DIR)
runpy.run_path(str(ROOT / "dashboard" / "app.py"), run_name="__main__")

st.divider()
st.caption(
    "Demo images and model training data: MVTec AD 'bottle' (Bergmann et al., CVPR 2019 / "
    "IJCV 2021), CC BY-NC-SA 4.0, non-commercial use. Source: "
    "[github.com/Amish23102006/inspectedge-ai](https://github.com/Amish23102006/inspectedge-ai)"
)
