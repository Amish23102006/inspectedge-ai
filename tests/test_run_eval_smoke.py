"""End-to-end smoke test of run_eval on a synthetic tree, with a toy extractor (no torch)."""

import json

import yaml

from inspectedge.evaluation import run_eval
from inspectedge.vision.knn_baseline import KnnPatchModel
from tests.test_knn_baseline import block_mean_extractor


def test_run_eval_writes_report_and_artefact(mvtec_tree, tmp_path, monkeypatch):
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "data": {"root": str(mvtec_tree), "category": "bottle", "image_size": 64},
                "paths": {
                    "artefacts": str(tmp_path / "art"),
                    "reports": str(tmp_path / "rep"),
                    "db": str(tmp_path / "art/db.sqlite"),
                },
            }
        )
    )
    monkeypatch.setattr(
        run_eval, "build_model", lambda cfg: KnnPatchModel(block_mean_extractor, seed=42)
    )
    assert run_eval.main(["--config", str(cfg_path)]) == 0

    (report_dir,) = (tmp_path / "rep").iterdir()
    report = json.loads((report_dir / "metrics.json").read_text())
    prov = report["provenance"]
    assert prov["split"]["train"] == 8 and prov["split"]["test_anomalous"] == 2
    assert prov["anomalous_masks_missing"] == 1
    assert {"image_auroc", "pixel_auroc", "decisions", "scope_gate"} <= report["metrics"].keys()
    assert report["metrics"]["scope_gate"]["enabled"] is True
    assert list((report_dir / "gallery").glob("*.png"))

    (art_dir,) = (tmp_path / "art/models").iterdir()
    meta = json.loads((art_dir / "model_version.json").read_text())
    assert meta["model_version"]["dataset"] == "MVTec AD"
    assert meta["thresholds"]["pass_threshold"] <= meta["thresholds"]["likely_anomaly_threshold"]
    assert (art_dir / "scope_gate.json").exists() and (art_dir / "scope_reference.npy").exists()
