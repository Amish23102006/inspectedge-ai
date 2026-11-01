"""CLI: train one baseline on defect-free images, calibrate thresholds on held-out normals,
evaluate on the test split, and save a model artefact plus a provenance-stamped report.

Usage:
    python -m inspectedge.evaluation.run_eval --config configs/default.yaml [--method autoencoder]
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np

from inspectedge.config import AppConfig, config_hash, load_config
from inspectedge.data.mvtec import Sample, find_leakage, index_category, load_split
from inspectedge.evaluation.metrics import (
    bootstrap_auroc_ci,
    decision_metrics,
    image_level_metrics,
    pixel_auroc,
    threshold_metrics,
)
from inspectedge.inference.visualize import overlay, save_rgb_png
from inspectedge.policy.decision import calibrate_thresholds, decide
from inspectedge.quality.gate import assess_quality
from inspectedge.repro import environment_info, set_seed
from inspectedge.schemas import ModelVersion, Thresholds


def build_model(cfg: AppConfig):
    m = cfg.model
    if m.method == "knn_patch":
        from inspectedge.vision.features import TorchvisionFeatureExtractor
        from inspectedge.vision.knn_baseline import KnnPatchModel

        ext = TorchvisionFeatureExtractor(m.backbone, m.layers, m.device)
        return KnnPatchModel(ext, k=m.k, memory_bank_max=m.memory_bank_max, seed=cfg.project.seed)
    from inspectedge.vision.autoencoder import AutoencoderModel

    return AutoencoderModel(m.ae_epochs, m.ae_batch_size, m.ae_lr, cfg.project.seed, m.device)


def _gallery(
    out_dir: Path,
    kind: str,
    idx: np.ndarray,
    images,
    maps,
    samples: list[Sample],
    scores,
    thr: Thresholds,
) -> list[dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for rank, i in enumerate(idx):
        vis = np.concatenate(
            [images[i], overlay(images[i], maps[i], 0.0, thr.likely_anomaly_threshold)], axis=1
        )
        name = f"{kind}_{rank:02d}_{samples[i].defect_type}_{samples[i].path.stem}.png"
        save_rgb_png(out_dir / name, vis)
        rows.append({"file": name, "source": samples[i].path.as_posix(), "score": float(scores[i])})
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--method", choices=["knn_patch", "autoencoder"], default=None)
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    if args.method:
        cfg.model.method = args.method
    set_seed(cfg.project.seed)
    chash = config_hash(cfg)
    model_id = f"{cfg.model.method}-{cfg.data.category}-{chash}"
    size = cfg.data.image_size

    samples = index_category(
        cfg.data.root, cfg.data.category, cfg.data.val_fraction, cfg.project.seed
    )
    leaks = find_leakage(samples)
    if leaks:
        print(f"ERROR: {len(leaks)} files leak across splits; aborting.", file=sys.stderr)
        return 1

    x_train, _, _, _ = load_split(samples, "train", size)
    x_val, _, _, _ = load_split(samples, "val", size)
    x_test, y_test, m_test, s_test = load_split(samples, "test", size)
    print(f"[{model_id}] train={len(x_train)} val={len(x_val)} test={len(x_test)}")

    model = build_model(cfg)
    t0 = time.perf_counter()
    model.fit(x_train)
    fit_s = time.perf_counter() - t0

    val_scores, _ = model.predict(x_val)
    thr = calibrate_thresholds(
        val_scores, cfg.policy.pass_quantile, cfg.policy.likely_anomaly_quantile
    )

    t0 = time.perf_counter()
    test_scores, test_maps = model.predict(x_test)
    per_image_ms = (time.perf_counter() - t0) * 1000 / len(x_test)

    quality = [assess_quality(img, cfg.quality) for img in x_test]
    decisions = [decide(float(s), q.passed, thr) for s, q in zip(test_scores, quality, strict=True)]
    anomalous_masks_missing = int(sum(1 for s in s_test if s.label == 1 and s.mask_path is None))

    metrics = {
        **image_level_metrics(y_test, test_scores),
        "image_auroc_95ci_bootstrap": bootstrap_auroc_ci(
            y_test, test_scores, seed=cfg.project.seed
        ),
        "pixel_auroc": pixel_auroc(m_test, test_maps),
        "at_pass_threshold": threshold_metrics(y_test, test_scores, thr.pass_threshold),
        "at_likely_anomaly_threshold": threshold_metrics(
            y_test, test_scores, thr.likely_anomaly_threshold
        ),
        "decisions": decision_metrics(y_test, decisions),
        "timing": {
            "fit_seconds": fit_s,
            "batch_predict_ms_per_image": per_image_ms,
            "note": "batch timing on eval machine; NOT an edge latency benchmark",
        },
    }

    report_dir = Path(cfg.paths.reports) / model_id
    art_dir = Path(cfg.paths.artefacts) / "models" / model_id
    normal_idx = np.where(y_test == 0)[0]
    anom_idx = np.where(y_test == 1)[0]
    fp_idx = normal_idx[np.argsort(-test_scores[normal_idx])][:6]  # most suspicious normals
    fn_idx = anom_idx[np.argsort(test_scores[anom_idx])][:6]  # least suspicious anomalies
    gallery = {
        "highest_scoring_normals": _gallery(
            report_dir / "gallery", "fp", fp_idx, x_test, test_maps, s_test, test_scores, thr
        ),
        "lowest_scoring_anomalies": _gallery(
            report_dir / "gallery", "fn", fn_idx, x_test, test_maps, s_test, test_scores, thr
        ),
    }

    mv = ModelVersion(
        model_id=model_id,
        method=cfg.model.method,
        dataset="MVTec AD",
        category=cfg.data.category,
        config_hash=chash,
        seed=cfg.project.seed,
        metrics_uri=(report_dir / "metrics.json").as_posix(),
        artefact_uri=art_dir.as_posix(),
    )
    model.save(art_dir)
    (art_dir / "model_version.json").write_text(
        json.dumps(
            {
                "model_version": mv.model_dump(mode="json"),
                "thresholds": thr.model_dump(mode="json"),
                "preprocessing_version": cfg.preprocessing_version,
            },
            indent=2,
        )
    )
    shutil.copyfile(args.config, art_dir / "config.yaml")

    provenance = {
        "model_id": model_id,
        "dataset": "MVTec AD (CC BY-NC-SA 4.0)",
        "category": cfg.data.category,
        "data_root": str(cfg.data.root),
        "split": {
            "train": len(x_train),
            "val_normal_for_calibration": len(x_val),
            "test_normal": int((y_test == 0).sum()),
            "test_anomalous": int(y_test.sum()),
            "val_fraction": cfg.data.val_fraction,
        },
        "anomalous_masks_missing": anomalous_masks_missing,
        "seed": cfg.project.seed,
        "config_hash": chash,
        "config": cfg.model_dump(mode="json"),
        "command": " ".join(["python -m inspectedge.evaluation.run_eval", *(argv or sys.argv[1:])]),
        "environment": environment_info(),
        "thresholds": thr.model_dump(mode="json"),
    }
    report = {"provenance": provenance, "metrics": metrics, "error_gallery": gallery}
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "metrics.json").write_text(json.dumps(report, indent=2, default=str))

    print(
        json.dumps({k: metrics[k] for k in ("image_auroc", "image_aupr", "pixel_auroc")}, indent=2)
    )
    print(f"Report:   {report_dir}")
    print(f"Artefact: {art_dir}   (set INSPECTEDGE_ARTEFACT_DIR to this path to serve it)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
