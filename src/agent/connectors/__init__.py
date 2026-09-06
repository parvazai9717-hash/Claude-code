"""Connectors: extra tools from external servers, on the same security terms.

A connector is a named, configured source of tools — today, an MCP server over
stdio or streamable HTTP. Its tools are namespaced, approval-gated, and run
through the identical registry as everything built in, because a second, weaker
path into the machine is the thing worth not building.
"""

from .client import ConnectorError, MCPCallResult, MCPConnection, MCPToolSpec
from .config import (
    NAME_PATTERN,
    TOOL_NAME_TEMPLATE,
    ConnectorCollection,
    ConnectorConfig,
    ConnectorKind,
)
from .manager import ConnectorManager, ConnectorStatus
from .store import CONNECTORS_FILENAME, ConnectorStore
from .tools import MCPTool, normalize_schema, sanitize_description

__all__ = [
    "CONNECTORS_FILENAME",
    "NAME_PATTERN",
    "TOOL_NAME_TEMPLATE",
    "ConnectorCollection",
    "ConnectorConfig",
    "ConnectorError",
    "ConnectorKind",
    "ConnectorManager",
    "ConnectorStatus",
    "ConnectorStore",
    "MCPCallResult",
    "MCPConnection",
    "MCPTool",
    "MCPToolSpec",
    "normalize_schema",
    "sanitize_description",
]
