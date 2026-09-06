"""Media detection, limits, and the `view_media` tool."""

from __future__ import annotations

from pathlib import Path

import pytest

from agent.errors import ErrorCategory, LimitExceededError, PathEscapeError, ToolError
from agent.media import (
    UNKNOWN_MIME_TYPE,
    Attachment,
    MediaKind,
    MediaLimits,
    describe_support,
    detect_mime_type,
    kind_for_mime_type,
    load_attachment,
)
from agent.messages import ProviderCapabilities
from agent.tools.base import ToolContext
from agent.tools.media_tools import ViewMediaTool

PNG = b"\x89PNG\r\n\x1a\n" + b"\x00" * 64
JPEG = b"\xff\xd8\xff\xe0" + b"\x00" * 64
WAV = b"RIFF" + b"\x00" * 4 + b"WAVE" + b"\x00" * 64
WEBP = b"RIFF" + b"\x00" * 4 + b"WEBP" + b"\x00" * 64
MP4 = b"\x00\x00\x00\x18ftypisom" + b"\x00" * 64
ZIP = b"PK\x03\x04" + b"\x00" * 64


# -- detection --------------------------------------------------------------
@pytest.mark.parametrize(
    ("data", "expected"),
    [
        (PNG, "image/png"),
        (JPEG, "image/jpeg"),
        (b"GIF89a" + b"\x00" * 32, "image/gif"),
        (WEBP, "image/webp"),
        (WAV, "audio/wav"),
        (b"ID3" + b"\x00" * 32, "audio/mpeg"),
        (b"\xff\xfb" + b"\x00" * 32, "audio/mpeg"),
        (b"\xff\xf1" + b"\x00" * 32, "audio/aac"),
        (b"OggS" + b"\x00" * 32, "audio/ogg"),
        (b"fLaC" + b"\x00" * 32, "audio/flac"),
        (b"FORM\x00\x00\x00\x00AIFF" + b"\x00" * 32, "audio/aiff"),
        (MP4, "video/mp4"),
    ],
)
def test_types_are_detected_from_content(data: bytes, expected: str) -> None:
    assert detect_mime_type(data) == expected


def test_an_unknown_format_is_reported_as_unknown() -> None:
    assert detect_mime_type(ZIP) == UNKNOWN_MIME_TYPE
    assert detect_mime_type(b"just some text") == UNKNOWN_MIME_TYPE


def test_the_filename_never_decides_the_type() -> None:
    """The whole point: a zip renamed to .png is still a zip."""
    assert detect_mime_type(ZIP, "totally-an-image.png") == UNKNOWN_MIME_TYPE


def test_kind_for_mime_type() -> None:
    assert kind_for_mime_type("image/png") is MediaKind.IMAGE
    assert kind_for_mime_type("audio/wav") is MediaKind.AUDIO
    assert kind_for_mime_type("video/mp4") is MediaKind.VIDEO
    assert kind_for_mime_type("application/pdf") is None


# -- attachments ------------------------------------------------------------
def test_attachment_round_trips_its_bytes() -> None:
    attachment = Attachment.from_bytes(PNG, path="files/x.png")
    assert attachment.kind is MediaKind.IMAGE
    assert attachment.data == PNG
    assert attachment.size_bytes == len(PNG)


def test_a_mislabelled_attachment_is_refused() -> None:
    with pytest.raises(ToolError, match="mislabelled"):
        Attachment.from_bytes(PNG, mime_type="audio/wav", path="x.png")


def test_an_unrecognised_attachment_is_refused() -> None:
    with pytest.raises(ToolError, match="does not match any supported media format"):
        Attachment.from_bytes(ZIP, path="x.png")


def test_summary_never_contains_the_bytes() -> None:
    attachment = Attachment.from_bytes(PNG, path="files/x.png")
    summary = attachment.summary()
    assert "files/x.png" in summary and "image/png" in summary
    assert attachment.data_base64 not in summary


def test_without_data_strips_the_payload() -> None:
    stripped = Attachment.from_bytes(PNG, path="x.png").without_data()
    assert stripped.data_base64 == ""
    assert stripped.size_bytes == len(PNG)


# -- limits -----------------------------------------------------------------
def test_size_limits_are_enforced_per_kind() -> None:
    limits = MediaLimits(max_image_bytes=10)
    with pytest.raises(LimitExceededError, match="limit for image"):
        limits.check(Attachment.from_bytes(PNG, path="x.png"))


def test_video_is_disabled_by_default() -> None:
    limits = MediaLimits()
    assert limits.enable_video is False
    with pytest.raises(ToolError, match="video input is disabled"):
        limits.check(Attachment.from_bytes(MP4, path="x.mp4"))


def test_video_can_be_enabled_explicitly() -> None:
    limits = MediaLimits(enable_video=True)
    limits.check(Attachment.from_bytes(MP4, path="x.mp4"))


def test_describe_support() -> None:
    assert describe_support(ProviderCapabilities(vision=True, audio=True)) == "image, audio"
    assert describe_support(ProviderCapabilities()) == "text only"


# -- loading from disk ------------------------------------------------------
def test_load_attachment(tmp_path: Path) -> None:
    target = tmp_path / "a.png"
    target.write_bytes(PNG)
    attachment = load_attachment(target, limits=MediaLimits(), display_path="files/a.png")
    assert attachment.path == "files/a.png"
    assert attachment.kind is MediaKind.IMAGE


def test_load_attachment_refuses_an_oversized_file(tmp_path: Path) -> None:
    target = tmp_path / "a.png"
    target.write_bytes(PNG)
    with pytest.raises(LimitExceededError):
        load_attachment(target, limits=MediaLimits(max_image_bytes=4))


def test_load_attachment_refuses_an_unsupported_file(tmp_path: Path) -> None:
    target = tmp_path / "a.zip"
    target.write_bytes(ZIP)
    with pytest.raises(ToolError, match="not a supported media type"):
        load_attachment(target, limits=MediaLimits())


# -- the tool ---------------------------------------------------------------
async def test_view_media_attaches_an_image(context: ToolContext, workspace: Path) -> None:
    (workspace / "files" / "chart.png").write_bytes(PNG)
    context.provider_capabilities = ProviderCapabilities(vision=True, audio=True)
    output = await ViewMediaTool().run({"path": "files/chart.png"}, context)
    assert output["kind"] == "image"
    assert output["attached"] is True
    assert output["_attachment"].data == PNG


async def test_view_media_attaches_audio(context: ToolContext, workspace: Path) -> None:
    (workspace / "files" / "clip.wav").write_bytes(WAV)
    context.provider_capabilities = ProviderCapabilities(vision=True, audio=True)
    output = await ViewMediaTool().run({"path": "files/clip.wav"}, context)
    assert output["kind"] == "audio"


async def test_view_media_refuses_what_the_model_cannot_perceive(
    context: ToolContext, workspace: Path
) -> None:
    """Loading media a text-only model would silently drop is worse than refusing."""
    (workspace / "files" / "chart.png").write_bytes(PNG)
    context.provider_capabilities = ProviderCapabilities(vision=False, audio=False)
    with pytest.raises(ToolError) as exc_info:
        await ViewMediaTool().run({"path": "files/chart.png"}, context)
    assert exc_info.value.category is ErrorCategory.CAPABILITY_UNAVAILABLE
    assert "text only" in exc_info.value.message


async def test_view_media_cannot_escape_the_workspace(context: ToolContext) -> None:
    with pytest.raises(PathEscapeError):
        await ViewMediaTool().run({"path": "../../secret.png"}, context)


async def test_view_media_refuses_a_non_media_file(context: ToolContext, workspace: Path) -> None:
    (workspace / "files" / "notes.txt").write_text("just text")
    with pytest.raises(ToolError):
        await ViewMediaTool().run({"path": "files/notes.txt"}, context)


async def test_view_media_refuses_video_while_disabled(
    context: ToolContext, workspace: Path
) -> None:
    (workspace / "files" / "clip.mp4").write_bytes(MP4)
    context.provider_capabilities = ProviderCapabilities(vision=True, audio=True, video=True)
    with pytest.raises(ToolError, match="video input is disabled"):
        await ViewMediaTool().run({"path": "files/clip.mp4"}, context)


def test_view_media_is_read_only_and_needs_no_approval() -> None:
    definition = ViewMediaTool().definition()
    assert definition.read_only is True
    assert definition.requires_approval is False
