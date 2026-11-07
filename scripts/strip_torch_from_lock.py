"""Copy a uv-generated hashed lock without torch/torchvision (installed from the CPU index)."""

import re
import sys

HEADER = """\
# Pinned, hashed runtime dependencies for docker/Dockerfile (Python 3.11, Linux, CPU).
# torch / torchvision are NOT in this file: they come from the PyTorch CPU index, pinned by
# TORCH_VERSION / TORCHVISION_VERSION in docker/Dockerfile (PyPI's x86 wheels pull in CUDA).
# Regenerate with:  make lock
"""


def main(src: str, dst: str) -> None:
    text = open(src, encoding="utf-8").read()
    blocks = re.split(r"\n(?=[a-z0-9])", text)
    keep = [b for b in blocks if not re.match(r"(torch|torchvision)==", b)]
    with open(dst, "w", encoding="utf-8") as f:
        f.write(HEADER + "\n".join(keep).rstrip() + "\n")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
