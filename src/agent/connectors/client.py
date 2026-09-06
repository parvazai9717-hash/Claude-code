"""MCP client: connect to a server, discover its tools, call them.

All MCP-protocol knowledge lives here, in the same way all Gemini knowledge lives
in the Gemini adapter. The rest of the runtime sees ordinary
:class:`~agent.tools.base.Tool` objects and ordinary
:class:`~agent.messages.ToolResult` values.

A connection is opened lazily on first use and held for the run. The session is
injectable so the whole layer is testable with no server, no subprocess and no
network.
"""

from __future__ import annotations

import asyncio
import os
import shutil
from contextlib import AsyncExitStack
from typing import Any, Protocol

from ..errors import AgentError, ErrorCategory
from ..media import Attachment
from .config import ConnectorConfig, ConnectorKind

#: Environment variables always passed to a stdio server, so it can find its own
#: runtime. Everything else must be named explicitly in `env`.
_BASE_ENV_KEYS = ("PATH", "HOME", "LANG", "LC_ALL", "TERM", "TZ", "SystemRoot", "APPDATA")


def build_server_env(connector: ConnectorConfig, environ: dict[str, str]) -> dict[str, str]:
    """The environment a stdio connector's subprocess receives.

    A minimal base so the server can find its own runtime, plus exactly the
    variables the connector named. Everything else is dropped, so a connector
    cannot inherit the credentials the agent holds for other purposes.
    """
    env = {key: environ[key] for key in _BASE_ENV_KEYS if key in environ}
    env.update(connector.resolve_env(environ))
    return env


class RemoteTool(Protocol):
    """The shape of a tool as MCP reports it."""

    name: str
    description: str | None


class ConnectorError(AgentError):
    """A connector could not be reached, or refused a call."""

    category = ErrorCategory.CAPABILITY_UNAVAILABLE


class MCPToolSpec:
    """A remote tool, normalized away from the SDK's types."""

    def __init__(self, name: str, description: str, input_schema: dict[str, Any]) -> None:
        self.name = name
        self.description = description
        self.input_schema = input_schema or {"type": "object", "properties": {}}

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"MCPToolSpec({self.name!r})"


class MCPCallResult:
    """A remote tool's answer, normalized."""

    def __init__(
        self,
        *,
        text: str = "",
        is_error: bool = False,
        structured: dict[str, Any] | None = None,
        attachments: list[Attachment] | None = None,
    ) -> None:
        self.text = text
        self.is_error = is_error
        self.structured = structured or {}
        self.attachments = attachments or []


class MCPConnection:
    """One live connection to an MCP server."""

    def __init__(
        self,
        connector: ConnectorConfig,
        *,
        environ: dict[str, str] | None = None,
        session_factory: Any = None,
    ) -> None:
        self.connector = connector
        self.environ = dict(os.environ if environ is None else environ)
        #: Injectable for tests: an async callable returning an open session.
        self._session_factory = session_factory
        self._session: Any = None
        self._stack: AsyncExitStack | None = None
        self._lock = asyncio.Lock()

    @property
    def connected(self) -> bool:
        return self._session is not None

    # -- lifecycle ----------------------------------------------------------
    async def connect(self) -> None:
        """Open the connection, if it is not already open."""
        async with self._lock:
            if self._session is not None:
                return
            if self._session_factory is not None:
                self._session = await self._session_factory(self.connector, self.environ)
                return
            self._session = await self._open_real_session()

    async def _open_real_session(self) -> Any:
        try:
            from mcp import ClientSession, StdioServerParameters
            from mcp.client.stdio import stdio_client
            from mcp.client.streamable_http import streamable_http_client
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise ConnectorError(
                'the MCP SDK is not installed. Install it with: pip install "local-agent[mcp]"'
            ) from exc

        stack = AsyncExitStack()
        self._stack = stack
        connector = self.connector
        try:
            if connector.kind is ConnectorKind.MCP_STDIO:
                program = shutil.which(connector.command)
                if program is None:
                    raise ConnectorError(
                        f"connector {connector.name!r} needs {connector.command!r}, "
                        "which is not installed on this system"
                    )
                parameters = StdioServerParameters(
                    command=program,
                    args=list(connector.args),
                    env=build_server_env(connector, self.environ),
                )
                streams = await stack.enter_async_context(stdio_client(parameters))
            else:
                # The SDK's own factory, so the client is the httpx flavour and
                # configuration (redirects, SSE timeouts) the transport expects.
                from mcp.client.streamable_http import create_mcp_http_client

                http_client = await stack.enter_async_context(
                    create_mcp_http_client(headers=connector.resolve_headers(self.environ))
                )
                streams = await stack.enter_async_context(
                    streamable_http_client(connector.url, http_client=http_client)
                )
            read_stream, write_stream = streams[0], streams[1]
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await asyncio.wait_for(session.initialize(), timeout=connector.timeout_seconds)
            return session
        except ConnectorError:
            await stack.aclose()
            self._stack = None
            raise
        except (TimeoutError, Exception) as exc:
            await stack.aclose()
            self._stack = None
            raise self._map_error(exc, "connect") from exc

    async def aclose(self) -> None:
        """Close the connection and release the subprocess or HTTP client."""
        async with self._lock:
            self._session = None
            if self._stack is not None:
                try:
                    await self._stack.aclose()
                finally:
                    self._stack = None

    # -- protocol -----------------------------------------------------------
    async def list_tools(self) -> list[MCPToolSpec]:
        """Discover the server's tools, filtered by the connector's allowlist."""
        await self.connect()
        try:
            listed = await asyncio.wait_for(
                self._session.list_tools(), timeout=self.connector.timeout_seconds
            )
        except Exception as exc:
            raise self._map_error(exc, "list_tools") from exc

        specs: list[MCPToolSpec] = []
        for tool in getattr(listed, "tools", []) or []:
            name = str(getattr(tool, "name", "") or "")
            if not name or not self.connector.permits(name):
                continue
            schema = getattr(tool, "input_schema", None) or getattr(tool, "inputSchema", None)
            specs.append(
                MCPToolSpec(
                    name=name,
                    description=str(getattr(tool, "description", "") or ""),
                    input_schema=dict(schema) if isinstance(schema, dict) else {},
                )
            )
        return specs

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> MCPCallResult:
        """Invoke a remote tool and normalize its reply."""
        await self.connect()
        if not self.connector.permits(name):
            raise ConnectorError(
                f"{name!r} is not on the allowlist for connector {self.connector.name!r}",
                category=ErrorCategory.PERMISSION_DENIED,
            )
        try:
            raw = await asyncio.wait_for(
                self._session.call_tool(name, arguments),
                timeout=self.connector.timeout_seconds,
            )
        except TimeoutError as exc:
            raise ConnectorError(
                f"{self.connector.name}/{name} did not respond within "
                f"{self.connector.timeout_seconds:g}s",
                category=ErrorCategory.TOOL_TIMEOUT,
            ) from exc
        except Exception as exc:
            raise self._map_error(exc, f"call_tool {name}") from exc
        return self._normalize_result(raw)

    @staticmethod
    def _normalize_result(raw: Any) -> MCPCallResult:
        """Turn an SDK `CallToolResult` into our own shape.

        Text is concatenated; image and audio blocks become attachments so a
        multimodal model can actually perceive them; anything else is described
        rather than dropped silently.
        """
        texts: list[str] = []
        attachments: list[Attachment] = []
        for block in getattr(raw, "content", None) or []:
            block_type = str(getattr(block, "type", "") or "")
            if block_type == "text":
                texts.append(str(getattr(block, "text", "") or ""))
            elif block_type in {"image", "audio"}:
                data = getattr(block, "data", None)
                mime = str(getattr(block, "mimeType", "") or getattr(block, "mime_type", "") or "")
                if isinstance(data, str) and data:
                    import base64

                    try:
                        decoded = base64.b64decode(data, validate=True)
                        attachments.append(
                            Attachment.from_bytes(
                                decoded, path=f"(from connector, {mime or 'unknown'})"
                            )
                        )
                        continue
                    except Exception:
                        pass
                texts.append(f"[{block_type} content that could not be decoded]")
            else:
                texts.append(f"[unsupported content block: {block_type or 'unknown'}]")

        structured = getattr(raw, "structured_content", None) or getattr(
            raw, "structuredContent", None
        )
        return MCPCallResult(
            text="\n".join(t for t in texts if t),
            is_error=bool(getattr(raw, "is_error", False) or getattr(raw, "isError", False)),
            structured=structured if isinstance(structured, dict) else {},
            attachments=attachments,
        )

    def _map_error(self, exc: Exception, operation: str) -> ConnectorError:
        """Map any transport or protocol failure onto the shared taxonomy."""
        if isinstance(exc, ConnectorError):
            return exc
        name = self.connector.name
        text = str(exc).lower()
        if isinstance(exc, TimeoutError) or "timeout" in text or "timed out" in text:
            return ConnectorError(
                f"connector {name!r} timed out during {operation}",
                category=ErrorCategory.TOOL_TIMEOUT,
            )
        if "connect" in text or "refused" in text or "unreachable" in text:
            target = self.connector.url or self.connector.command
            return ConnectorError(
                f"cannot reach connector {name!r} at {target}: the server is not responding",
                category=ErrorCategory.PROVIDER_UNAVAILABLE,
            )
        if "401" in text or "403" in text or "unauthor" in text or "forbidden" in text:
            missing = self.connector.missing_credentials(self.environ)
            hint = f"; unset variables: {', '.join(missing)}" if missing else ""
            return ConnectorError(
                f"connector {name!r} rejected the credentials{hint}",
                category=ErrorCategory.PROVIDER_AUTH,
            )
        return ConnectorError(
            f"connector {name!r} failed during {operation}: {type(exc).__name__}",
            category=ErrorCategory.TOOL_FAILED,
        )
