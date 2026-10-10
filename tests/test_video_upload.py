from io import BytesIO
from pathlib import Path

import pytest

from core.video_upload import MAX_UPLOAD_BYTES, VideoUploadError, save_video_upload


def _multipart(filename: str, payload: bytes, boundary: str = "demo") -> bytes:
    return (
        f"--{boundary}\r\n"
        f"Content-Disposition: form-data; name=\"video\"; filename=\"{filename}\"\r\n"
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + payload + f"\r\n--{boundary}--\r\n".encode()


def test_binary_avi_upload_preserves_payload(tmp_path: Path):
    payload = b"AVI\x00\r\nframe-bytes\x00\xff"
    body = _multipart("clip.avi", payload)
    path = save_video_upload(
        BytesIO(body), "multipart/form-data; boundary=demo", len(body), tmp_path
    )
    assert path.suffix == ".avi"
    assert path.read_bytes() == payload


def test_upload_rejects_unsupported_extension(tmp_path: Path):
    body = _multipart("clip.exe", b"bad")
    with pytest.raises(VideoUploadError) as error:
        save_video_upload(BytesIO(body), "multipart/form-data; boundary=demo", len(body), tmp_path)
    assert error.value.status_code == 400


def test_upload_rejects_missing_multipart_boundary(tmp_path: Path):
    body = b"not multipart"
    with pytest.raises(VideoUploadError):
        save_video_upload(BytesIO(body), "multipart/form-data", len(body), tmp_path)


def test_upload_rejects_oversized_content_length(tmp_path: Path):
    with pytest.raises(VideoUploadError) as error:
        save_video_upload(BytesIO(), "multipart/form-data; boundary=demo",
                          MAX_UPLOAD_BYTES + 1, tmp_path)
    assert error.value.status_code == 413
