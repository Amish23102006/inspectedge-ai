import pytest

from inspectedge.config import UploadConfig
from inspectedge.inference.uploads import UploadError, decode_upload
from tests.conftest import noise_image, png_bytes


def test_valid_png_decodes_to_rgb():
    img = decode_upload(png_bytes(noise_image(64)), "image/png", UploadConfig())
    assert img.shape == (64, 64, 3)


@pytest.mark.parametrize(
    ("data", "ctype", "cfg", "status"),
    [
        (b"", "image/png", UploadConfig(), 400),
        (b"GIF89a...", "image/gif", UploadConfig(), 415),
        (b"hello, not a png", "image/png", UploadConfig(), 415),  # magic-byte mismatch
        (b"\x89PNG\r\n\x1a\n" + b"\x00" * 50, "image/png", UploadConfig(), 400),  # corrupted
        (png_bytes(noise_image(64)), "image/png", UploadConfig(max_bytes=100), 413),
        (png_bytes(noise_image(64)), "image/png", UploadConfig(max_pixels=1000), 413),
    ],
)
def test_rejections(data, ctype, cfg, status):
    with pytest.raises(UploadError) as e:
        decode_upload(data, ctype, cfg)
    assert e.value.status_code == status
