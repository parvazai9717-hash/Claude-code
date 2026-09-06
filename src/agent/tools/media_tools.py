"""`view_media` — load a workspace image or audio file for the model to perceive.

A path being mentioned in conversation does not let the model see a file. This
tool is the only way media enters the context, and it goes through the same path
policy, size limits and content verification as everything else.

The loaded bytes ride back on the :class:`~agent.messages.ToolResult` as an
attachment; the runtime places them on the answering message, and each provider
sends them only if it has declared support for that media kind.
"""

from __future__ import annotations

from typing import Any

from ..errors import ErrorCategory, ToolError
from ..media import MediaKind, load_attachment
from ..messages import RiskCategory, RiskLevel
from .base import Tool, ToolContext


class ViewMediaTool(Tool):
    name = "view_media"
    description = (
        "Load an image or audio file from the workspace so you can actually look at or "
        "listen to it. Use this before describing any media — you cannot perceive a file "
        "just because its path was mentioned. Returns the file's type and size, and "
        "attaches its contents for you to examine."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Workspace-relative path to an image or audio file.",
            },
            "purpose": {
                "type": "string",
                "description": "What you are trying to determine from it, in one short line.",
                "maxLength": 300,
            },
        },
        "required": ["path"],
        "additionalProperties": False,
    }
    # Reading media changes nothing, so it needs no approval — but it does consume
    # context and cost, which the per-run tool-call budget already bounds.
    risk = RiskLevel.READ_ONLY
    risk_category = RiskCategory.READ
    read_only = True
    requires_approval = False

    async def run(self, arguments: dict[str, Any], context: ToolContext) -> dict[str, Any]:
        policy = context.paths
        target = policy.resolve(arguments["path"], must_exist=True)
        if target.is_dir():
            raise ToolError(
                f"{policy.relative(target)} is a directory, not a media file",
                category=ErrorCategory.INVALID_ARGUMENTS,
            )

        limits = context.config.media
        attachment = load_attachment(
            target,
            limits=limits,
            display_path=policy.relative(target),
            description=arguments.get("purpose", ""),
        )

        capabilities = context.provider_capabilities
        if capabilities is not None and not capabilities.accepts(attachment.kind.value):
            from ..media import describe_support

            raise ToolError(
                f"the active model cannot accept {attachment.kind.value} input "
                f"(it supports: {describe_support(capabilities)}). "
                "Switch to a multimodal model, or describe the file another way.",
                category=ErrorCategory.CAPABILITY_UNAVAILABLE,
            )

        return {
            "path": attachment.path,
            "kind": attachment.kind.value,
            "mime_type": attachment.mime_type,
            "size_bytes": attachment.size_bytes,
            "attached": True,
            "note": (
                "The file is attached to this result. Describe only what you can actually "
                f"perceive in it. Video is {'enabled' if limits.enable_video else 'disabled'}."
                if attachment.kind is not MediaKind.VIDEO
                else "The video is attached to this result."
            ),
            # Consumed by the runtime, not shown to the model as text.
            "_attachment": attachment,
        }
