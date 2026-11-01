"""CLI: index one MVTec AD category, check integrity and leakage, write a manifest.

This script never downloads data: MVTec AD requires accepting its licence on the official site.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from inspectedge.config import config_hash, load_config
from inspectedge.data.mvtec import find_leakage, index_category, integrity_report, write_manifest


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    samples = index_category(
        cfg.data.root, cfg.data.category, cfg.data.val_fraction, cfg.project.seed
    )
    report = integrity_report(samples)
    leaks = find_leakage(samples)
    report.update(
        {
            "category": cfg.data.category,
            "seed": cfg.project.seed,
            "val_fraction": cfg.data.val_fraction,
            "config_hash": config_hash(cfg),
            "leakage": leaks,
        }
    )

    out_dir = Path(cfg.paths.artefacts) / "manifests"
    write_manifest(samples, out_dir / f"{cfg.data.category}_manifest.csv")
    (out_dir / f"{cfg.data.category}_summary.json").write_text(json.dumps(report, indent=2))

    print(json.dumps({k: report[k] for k in ("category", "counts", "defect_types")}, indent=2))
    if report["missing_masks"]:
        print(f"WARNING: {len(report['missing_masks'])} anomalous images have no mask")
    if leaks:
        print(f"ERROR: {len(leaks)} identical files appear in multiple splits", file=sys.stderr)
        return 1
    print(f"OK: manifest written to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
