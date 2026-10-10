"""Bounded standard-library multipart video upload handling."""

from __future__ import annotations

from email import policy
from email.parser import BytesParser
from io import BytesIO
from pathlib import Path
from tempfile import SpooledTemporaryFile
from typing import BinaryIO
from uuid import uuid4

MAX_UPLOAD_BYTES = 256 * 1024 * 1024
VIDEO_EXTENSIONS = frozenset({
    ".avi", ".m4v", ".mkv", ".mov", ".mp4", ".mpeg", ".mpg", ".webm", ".wmv",
})


class VideoUploadError(Exception):
    """An HTTP-safe validation error for the local demo upload endpoint."""

    def __init__(self, message: str, status_code: int = 400):
        super().__init__(message)
        self.status_code = status_code


def save_video_upload(
    stream: BinaryIO,
    content_type: str,
    content_length: int,
    upload_dir: Path,
) -> Path:
    """Parse one multipart ``video`` field and save its bytes safely."""
    if content_length < 0:
        raise VideoUploadError("missing or invalid Content-Length")
    if content_length > MAX_UPLOAD_BYTES:
        raise VideoUploadError("upload exceeds the 256 MiB limit", 413)
    if not content_type.lower().startswith("multipart/form-data"):
        raise VideoUploadError("expected multipart/form-data")

    body = SpooledTemporaryFile(max_size=8 * 1024 * 1024, mode="w+b")
    try:
        header = (
            f"Content-Type: {content_type}\r\n"
            "MIME-Version: 1.0\r\n\r\n"
        ).encode("ascii", "strict")
        body.write(header)
        remaining = content_length
        while remaining:
            chunk = stream.read(min(1024 * 1024, remaining))
            if not chunk:
                raise VideoUploadError("upload body ended before Content-Length")
            body.write(chunk)
            remaining -= len(chunk)
        body.seek(len(header))
        raw_body = body.read()
        header_message = BytesParser(policy=policy.compat32).parse(BytesIO(header))
    except UnicodeError as exc:
        raise VideoUploadError("invalid multipart Content-Type") from exc
    finally:
        body.close()

    boundary = header_message.get_boundary()
    if not boundary:
        raise VideoUploadError("multipart boundary is missing")
    parts = []
    delimiter = b"--" + boundary.encode("utf-8")
    for raw_part in raw_body.split(delimiter)[1:]:
        if raw_part.startswith(b"--"):
            break
        if raw_part.startswith(b"\r\n"):
            raw_part = raw_part[2:]
        if raw_part.endswith(b"\r\n"):
            raw_part = raw_part[:-2]
        header_end = raw_part.find(b"\r\n\r\n")
        if header_end < 0:
            continue
        part_headers = BytesParser(policy=policy.compat32).parse(
            BytesIO(raw_part[:header_end])
        )
        if (part_headers.get("content-disposition", "").lower().startswith("form-data")
                and part_headers.get_param("name", header="content-disposition") == "video"):
            parts.append((part_headers, raw_part[header_end + 4:]))
    if len(parts) != 1:
        raise VideoUploadError("multipart request must contain exactly one video field")

    part_headers, payload = parts[0]
    filename = Path(part_headers.get_param("filename", header="content-disposition") or "")
    suffix = filename.suffix.lower()
    if not suffix or suffix not in VIDEO_EXTENSIONS:
        raise VideoUploadError(
            "unsupported video extension; use MP4, AVI, MOV, MKV, MPEG, MPG, WebM, or WMV"
        )
    if len(payload) > MAX_UPLOAD_BYTES:
        raise VideoUploadError("upload exceeds the 256 MiB limit", 413)

    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / f"{uuid4().hex}{suffix}"
    try:
        destination.write_bytes(payload)
    except OSError as exc:
        raise VideoUploadError(f"could not save upload: {exc}", 500) from exc
    return destination
