"""Multimodal attachments: images, audio, and (opt-in) video.

Media is treated like every other input the agent touches: it must live inside
the workspace, it is size-capped per kind, its type is decided by the file's
actual bytes rather than its name, and it only reaches a provider that has
declared it can handle that kind.

The agent does not "see" a file merely because a path was mentioned. Something
has to load it — the user attaching it, or the `view_media` tool — and that load
goes through :class:`agent.security.paths.PathPolicy` first.
"""

from __future__ import annotations

import base64
from enum import StrEnum
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from .errors import ErrorCategory, LimitExceededError, ToolError


class MediaKind(StrEnum):
    """What sort of thing an attachment is."""

    IMAGE = "image"
    AUDIO = "audio"
    #: Defined so the boundary is explicit. Refused unless `enable_video` is set,
    #: and then still only by a provider that declares video support.
    VIDEO = "video"


#: MIME types accepted per kind. Anything else is refused rather than guessed at.
ALLOWED_MIME_TYPES: dict[MediaKind, frozenset[str]] = {
    MediaKind.IMAGE: frozenset(
        {"image/png", "image/jpeg", "image/gif", "image/webp", "image/heic"}
    ),
    MediaKind.AUDIO: frozenset(
        {"audio/wav", "audio/mpeg", "audio/aiff", "audio/aac", "audio/ogg", "audio/flac"}
    ),
    MediaKind.VIDEO: frozenset({"video/mp4", "video/quicktime", "video/webm", "video/x-matroska"}),
}

#: Magic-number signatures: `(signature, offset, mime)`.
_SIGNATURES: tuple[tuple[bytes, int, str], ...] = (
    (b"\x89PNG\r\n\x1a\n", 0, "image/png"),
    (b"\xff\xd8\xff", 0, "image/jpeg"),
    (b"GIF87a", 0, "image/gif"),
    (b"GIF89a", 0, "image/gif"),
    (b"ID3", 0, "audio/mpeg"),
    (b"\xff\xfb", 0, "audio/mpeg"),
    (b"\xff\xf3", 0, "audio/mpeg"),
    (b"\xff\xf2", 0, "audio/mpeg"),
    (b"\xff\xf1", 0, "audio/aac"),
    (b"\xff\xf9", 0, "audio/aac"),
    (b"OggS", 0, "audio/ogg"),
    (b"fLaC", 0, "audio/flac"),
    (b"\x1a\x45\xdf\xa3", 0, "video/x-matroska"),
)

#: Bytes read when sniffing a type.
SNIFF_BYTES = 32

#: Returned when the bytes match nothing known.
UNKNOWN_MIME_TYPE = "application/octet-stream"


def detect_mime_type(data: bytes, filename: str = "") -> str:
    """Determine a MIME type from file *content*.

    Returns :data:`UNKNOWN_MIME_TYPE` when the bytes match no known signature.
    The filename is accepted for error messages only and deliberately does not
    influence the result — trusting an extension is exactly what lets a
    mislabelled file reach a model as something it is not.
    """
    head = data[:SNIFF_BYTES]

    # Container formats need a second field to disambiguate.
    if head.startswith(b"RIFF") and len(head) >= 12:
        container = head[8:12]
        if container == b"WEBP":
            return "image/webp"
        if container == b"WAVE":
            return "audio/wav"
        return UNKNOWN_MIME_TYPE
    if head.startswith(b"FORM") and len(head) >= 12 and head[8:12] in (b"AIFF", b"AIFC"):
        return "audio/aiff"
    if len(head) >= 12 and head[4:8] == b"ftyp":
        brand = bytes(head[8:12]).lower()
        if brand.startswith((b"heic", b"heix", b"hevc", b"heim", b"heis", b"mif1")):
            return "image/heic"
        if brand.startswith(b"qt"):
            return "video/quicktime"
        return "video/mp4"
    if head.startswith(b"\x1a\x45\xdf\xa3"):
        # Matroska and WebM share a header; WebM declares itself in the doctype.
        return "video/webm" if b"webm" in data[:64] else "video/x-matroska"

    for signature, offset, mime in _SIGNATURES:
        if head[offset : offset + len(signature)] == signature:
            return mime

    return UNKNOWN_MIME_TYPE


def kind_for_mime_type(mime_type: str) -> MediaKind | None:
    """The media kind a MIME type belongs to, or None when unsupported."""
    for kind, allowed in ALLOWED_MIME_TYPES.items():
        if mime_type in allowed:
            return kind
    return None


class Attachment(BaseModel):
    """One piece of media travelling with a message.

    Bytes are carried in memory rather than re-read at send time, so what a
    provider receives is exactly what was validated.
    """

    kind: MediaKind
    mime_type: str
    #: Workspace-relative path, for display and provenance. Never absolute.
    path: str = ""
    size_bytes: int = 0
    #: Raw bytes, base64-encoded so the model is JSON-serialisable.
    data_base64: str = Field(default="", repr=False)
    #: A short, model-visible note about where this came from.
    description: str = ""

    @property
    def data(self) -> bytes:
        return base64.b64decode(self.data_base64) if self.data_base64 else b""

    @classmethod
    def from_bytes(
        cls, data: bytes, *, mime_type: str = "", path: str = "", description: str = ""
    ) -> Attachment:
        """Build an attachment, deriving and verifying the type from the bytes.

        Raises:
            ToolError: When the content matches no supported format, or when a
                declared `mime_type` disagrees with what the bytes actually are.
        """
        detected = detect_mime_type(data, path)
        if detected == UNKNOWN_MIME_TYPE:
            raise ToolError(
                f"{path or 'the file'} does not match any supported media format. "
                "Its contents were checked directly, so renaming it will not help.",
                category=ErrorCategory.INVALID_ARGUMENTS,
            )
        # The bytes are authoritative. A declared type that disagrees means the
        # file is not what it claims to be — precisely the case worth refusing.
        if mime_type and mime_type != detected:
            raise ToolError(
                f"{path or 'the file'} is declared as {mime_type} but its contents are "
                f"{detected}; refusing to send mislabelled media to a model",
                category=ErrorCategory.INVALID_ARGUMENTS,
            )
        kind = kind_for_mime_type(detected)
        if kind is None:
            supported = ", ".join(sorted(t for types in ALLOWED_MIME_TYPES.values() for t in types))
            raise ToolError(
                f"{detected} is not a supported media type. Supported: {supported}",
                category=ErrorCategory.INVALID_ARGUMENTS,
            )
        return cls(
            kind=kind,
            mime_type=detected,
            path=path,
            size_bytes=len(data),
            data_base64=base64.b64encode(data).decode("ascii"),
            description=description,
        )

    def summary(self) -> str:
        """A short, model-safe description. Never includes the bytes."""
        where = f" from {self.path}" if self.path else ""
        return f"[{self.kind.value} {self.mime_type}, {self.size_bytes} bytes{where}]"

    def without_data(self) -> Attachment:
        """A copy with the bytes stripped, for logging and persistence."""
        return self.model_copy(update={"data_base64": ""})


class MediaLimits(BaseModel):
    """Per-kind size caps and the video opt-in."""

    max_image_bytes: int = Field(default=8_000_000, ge=1)
    max_audio_bytes: int = Field(default=25_000_000, ge=1)
    max_video_bytes: int = Field(default=50_000_000, ge=1)
    #: Maximum attachments the runtime carries on one message.
    max_attachments_per_message: int = Field(default=4, ge=1, le=32)
    #: Video is off by default: it is large, costly, and unsupported by most
    #: local models. Enabling it does not make an incapable provider accept it.
    enable_video: bool = False

    def limit_for(self, kind: MediaKind) -> int:
        return {
            MediaKind.IMAGE: self.max_image_bytes,
            MediaKind.AUDIO: self.max_audio_bytes,
            MediaKind.VIDEO: self.max_video_bytes,
        }[kind]

    def check(self, attachment: Attachment) -> None:
        """Raise when an attachment is of a disabled kind or is too large."""
        if attachment.kind is MediaKind.VIDEO and not self.enable_video:
            raise ToolError(
                "video input is disabled. Set `media.enable_video: true` and use a "
                "provider that supports video; most local models do not.",
                category=ErrorCategory.CAPABILITY_UNAVAILABLE,
            )
        limit = self.limit_for(attachment.kind)
        if attachment.size_bytes > limit:
            raise LimitExceededError(
                f"{attachment.path or 'the file'} is {attachment.size_bytes} bytes, above "
                f"the {limit} byte limit for {attachment.kind.value}",
                details={"size": attachment.size_bytes, "limit": limit},
            )


def load_attachment(
    path: Path, *, limits: MediaLimits, display_path: str = "", description: str = ""
) -> Attachment:
    """Read a media file from an already-validated path.

    The caller must have resolved `path` through `PathPolicy` first — this
    function does no containment checking of its own, deliberately, so that the
    workspace boundary stays enforced in exactly one place.
    """
    name = display_path or path.name
    size = path.stat().st_size

    # Sniff before reading, so an oversized file is never loaded into memory.
    with path.open("rb") as handle:
        head = handle.read(SNIFF_BYTES)
    detected = detect_mime_type(head, name)
    kind = kind_for_mime_type(detected)
    if kind is None:
        raise ToolError(
            f"{name} is not a supported media type (its contents look like {detected})",
            category=ErrorCategory.INVALID_ARGUMENTS,
        )
    limit = limits.limit_for(kind)
    if size > limit:
        raise LimitExceededError(
            f"{name} is {size} bytes, above the {limit} byte limit for {kind.value}",
            details={"size": size, "limit": limit},
        )

    attachment = Attachment.from_bytes(path.read_bytes(), path=name, description=description)
    limits.check(attachment)
    return attachment


def describe_support(capabilities: Any) -> str:
    """Render which media kinds a provider accepts, for errors and `doctor`."""
    supported = [
        kind.value
        for kind, flag in (
            (MediaKind.IMAGE, getattr(capabilities, "vision", False)),
            (MediaKind.AUDIO, getattr(capabilities, "audio", False)),
            (MediaKind.VIDEO, getattr(capabilities, "video", False)),
        )
        if flag
    ]
    return ", ".join(supported) or "text only"
