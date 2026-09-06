"""The connector manager: discovery, health, and tool production.

This is what the runtime and the UI both talk to. It owns the lifecycle of every
enabled connector: connect, discover tools, wrap them, and shut them down again.

A failing connector never stops a run. If a server is unreachable or misbehaving
its tools are simply absent, and the reason is reported — an agent that silently
loses a capability is worse than one that says which capability it lost.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from typing import Any

from ..errors import AgentError
from ..events import EventBus, EventType
from ..tools.base import Tool
from .client import MCPConnection
from .config import ConnectorCollection, ConnectorConfig
from .tools import MCPTool


@dataclass
class ConnectorStatus:
    """What happened when we tried to use a connector."""

    name: str
    enabled: bool
    ok: bool
    tool_count: int = 0
    tools: list[str] = field(default_factory=list)
    detail: str = ""
    missing_credentials: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.enabled,
            "ok": self.ok,
            "tool_count": self.tool_count,
            "tools": self.tools,
            "detail": self.detail,
            "missing_credentials": self.missing_credentials,
        }


class ConnectorManager:
    """Connects to enabled connectors and turns their tools into local ones."""

    def __init__(
        self,
        collection: ConnectorCollection,
        *,
        environ: dict[str, str] | None = None,
        events: EventBus | None = None,
        session_factory: Any = None,
    ) -> None:
        self.collection = collection
        self.environ = environ
        self.events = events or EventBus()
        #: Injected by tests to stand in for a real server.
        self._session_factory = session_factory
        self._connections: dict[str, MCPConnection] = {}
        self._statuses: dict[str, ConnectorStatus] = {}

    # -- lifecycle ----------------------------------------------------------
    def _connection(self, connector: ConnectorConfig) -> MCPConnection:
        existing = self._connections.get(connector.name)
        if existing is None:
            existing = MCPConnection(
                connector,
                environ=self.environ,
                session_factory=self._session_factory,
            )
            self._connections[connector.name] = existing
        return existing

    async def load_tools(self) -> list[Tool]:
        """Connect to every enabled connector and return all their tools.

        Connectors are contacted concurrently, because one slow server should not
        delay the rest, and each failure is isolated to its own connector.
        """
        enabled = self.collection.enabled()
        if not enabled:
            return []
        results = await asyncio.gather(
            *(self._load_one(connector) for connector in enabled), return_exceptions=False
        )
        return [tool for group in results for tool in group]

    async def _load_one(self, connector: ConnectorConfig) -> list[Tool]:
        try:
            connection = self._connection(connector)
            specs = await connection.list_tools()
        except AgentError as exc:
            self._statuses[connector.name] = ConnectorStatus(
                name=connector.name,
                enabled=True,
                ok=False,
                detail=exc.message,
                missing_credentials=connector.missing_credentials(
                    self.environ if self.environ is not None else _environ()
                ),
            )
            self.events.emit_event(
                EventType.ERROR,
                message=f"connector {connector.name!r} is unavailable: {exc.message}",
                data={"connector": connector.name, "category": exc.category.value},
            )
            return []
        except Exception as exc:
            self._statuses[connector.name] = ConnectorStatus(
                name=connector.name,
                enabled=True,
                ok=False,
                detail=f"unexpected failure: {type(exc).__name__}",
            )
            return []

        tools: list[Tool] = [
            MCPTool(connector, spec, self._connection(connector)) for spec in specs
        ]
        self._statuses[connector.name] = ConnectorStatus(
            name=connector.name,
            enabled=True,
            ok=True,
            tool_count=len(tools),
            tools=[t.name for t in tools],
            detail=f"{len(tools)} tool(s) available",
        )
        self.events.emit_event(
            EventType.MESSAGE,
            message=f"connector {connector.name!r} provided {len(tools)} tool(s)",
            data={"connector": connector.name, "tools": [t.name for t in tools]},
        )
        return tools

    async def health_check(self, name: str | None = None) -> list[ConnectorStatus]:
        """Test connectors without registering their tools.

        Used by `doctor` and by a UI's "Test connection" button. A disabled
        connector is reported as disabled rather than silently skipped.
        """
        targets = (
            [c for c in self.collection.connectors if c.name == name]
            if name
            else self.collection.connectors
        )
        statuses: list[ConnectorStatus] = []
        environ = self.environ if self.environ is not None else _environ()
        for connector in targets:
            missing = connector.missing_credentials(environ)
            if not connector.enabled:
                statuses.append(
                    ConnectorStatus(
                        name=connector.name,
                        enabled=False,
                        ok=False,
                        detail="disabled; enable it to use its tools",
                        missing_credentials=missing,
                    )
                )
                continue
            try:
                specs = await self._connection(connector).list_tools()
            except AgentError as exc:
                statuses.append(
                    ConnectorStatus(
                        name=connector.name,
                        enabled=True,
                        ok=False,
                        detail=exc.message,
                        missing_credentials=missing,
                    )
                )
                continue
            except Exception as exc:
                statuses.append(
                    ConnectorStatus(
                        name=connector.name,
                        enabled=True,
                        ok=False,
                        detail=f"unexpected failure: {type(exc).__name__}",
                        missing_credentials=missing,
                    )
                )
                continue
            statuses.append(
                ConnectorStatus(
                    name=connector.name,
                    enabled=True,
                    ok=True,
                    tool_count=len(specs),
                    tools=[connector.tool_name(s.name) for s in specs],
                    detail=f"reachable; {len(specs)} tool(s)",
                    missing_credentials=missing,
                )
            )
        self._statuses.update({s.name: s for s in statuses})
        return statuses

    def statuses(self) -> list[ConnectorStatus]:
        """The last known state of every configured connector."""
        known = dict(self._statuses)
        for connector in self.collection.connectors:
            known.setdefault(
                connector.name,
                ConnectorStatus(
                    name=connector.name,
                    enabled=connector.enabled,
                    ok=False,
                    detail="not contacted yet",
                ),
            )
        return [known[c.name] for c in self.collection.connectors]

    async def aclose(self) -> None:
        """Shut every connection down, releasing subprocesses and HTTP clients."""
        connections = list(self._connections.values())
        self._connections.clear()
        for connection in connections:
            try:
                await connection.aclose()
            except Exception:
                continue


def _environ() -> dict[str, str]:
    import os

    return dict(os.environ)
