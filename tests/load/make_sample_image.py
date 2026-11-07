"""Write the image used by the k6 load test.

Prefers a real defect-free MVTec test image (resized to the model's input size so it fits in a
ConfigMap); falls back to a synthetic textured image that passes the quality gate.

    python tests/load/make_sample_image.py [--from PATH] [--out PATH] [--size 256]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

DEFAULT_SOURCE = Path("data/raw/mvtec_ad/bottle/test/good/000.png")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="src", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--out", type=Path, default=Path("reports/k8s/sample.png"))
    ap.add_argument("--size", type=int, default=256)
    args = ap.parse_args(argv)

    if args.src.is_file():
        img = cv2.imread(str(args.src), cv2.IMREAD_COLOR)
        if img is None:
            raise SystemExit(f"could not read {args.src}")
        img = cv2.resize(img, (args.size, args.size), interpolation=cv2.INTER_AREA)
        origin = str(args.src)
    else:
        rng = np.random.default_rng(0)
        img = rng.integers(40, 215, (args.size, args.size, 3), dtype=np.uint8)
        origin = "synthetic noise (MVTec image not found)"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.out), img):
        raise SystemExit(f"could not write {args.out}")
    print(f"sample image: {args.out} ({args.out.stat().st_size} bytes) from {origin}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
