"""Upload validation: allow-listed types, magic bytes, size and decompression (pixel) limits."""

from __future__ import annotations

import io

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError

from inspectedge.config import UploadConfig

MAGIC = {
    "image/png": (b"\x89PNG\r\n\x1a\n",),
    "image/jpeg": (b"\xff\xd8\xff",),
}


class UploadError(ValueError):
    def __init__(self, message: str, status_code: int = 400) -> None:
        super().__init__(message)
        self.status_code = status_code


def decode_upload(data: bytes, content_type: str | None, cfg: UploadConfig) -> np.ndarray:
    """Validate and decode an uploaded image to RGB uint8. Raises UploadError."""
    if len(data) == 0:
        raise UploadError("empty upload", 400)
    if len(data) > cfg.max_bytes:
        raise UploadError(f"upload exceeds {cfg.max_bytes} bytes", 413)
    if content_type not in cfg.allowed_content_types:
        raise UploadError(f"content type {content_type!r} not allowed", 415)
    if not any(data.startswith(m) for m in MAGIC.get(content_type, ())):
        raise UploadError("file content does not match declared content type", 415)

    # Read dimensions from the header BEFORE full decode (decompression-bomb guard).
    try:
        with Image.open(io.BytesIO(data)) as im:
            w, h = im.size
    except (UnidentifiedImageError, OSError) as e:
        raise UploadError("unreadable image header", 400) from e
    if w * h > cfg.max_pixels:
        raise UploadError(f"image has {w * h} pixels; limit is {cfg.max_pixels}", 413)

    img = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise UploadError("corrupted or undecodable image", 400)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
