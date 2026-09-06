"""Connector definitions.

A **connector** is a named, configured source of extra tools. The first kind is
an **MCP server**, reached over stdio or streamable HTTP.

The security stance is deliberately more conservative than for built-in tools,
because a connector is third-party code the runtime did not write and cannot
audit:

- Connectors are **disabled by default**; each is enabled explicitly.
- Their tools are **namespaced**, so a connector cannot shadow a built-in tool.
- Their tools are treated as **side-effecting and approval-gated by default** —
  the runtime cannot know whether a remote `search` writes to a database.
- Their tool **descriptions are untrusted input**: a hostile server can put
  instructions in them, so they are labelled and never given prompt authority.
- Only **named environment variables** are forwarded to a stdio server, so a
  connector cannot inherit every credential in the environment.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field, field_validator, model_validator

from ..errors import ConfigurationError

#: Connector names become part of a tool name, so they must stay identifier-safe.
NAME_PATTERN = re.compile(r"^[a-z][a-z0-9_]{1,31}$")

#: How a connector's tools are exposed to the model. The prefix makes the origin
#: visible to the user in every approval prompt and log line.
TOOL_NAME_TEMPLATE = "mcp__{connector}__{tool}"


class ConnectorKind(StrEnum):
    """How to reach a connector."""

    #: An MCP server launched as a local subprocess, speaking over stdio.
    MCP_STDIO = "mcp_stdio"
    #: An MCP server reached over streamable HTTP.
    MCP_HTTP = "mcp_http"


class ConnectorConfig(BaseModel):
    """One configured connector.

    Credentials are never stored here. `env` names the environment variables to
    forward; `header_env` maps a header name to the variable holding its value.
    The values are read at connection time and never written to disk or logs.
    """

    name: str
    kind: ConnectorKind
    enabled: bool = False
    description: str = ""

    # -- stdio transport ----------------------------------------------------
    command: str = ""
    args: list[str] = Field(default_factory=list)
    #: Names of environment variables to forward. Values are never stored.
    env: list[str] = Field(default_factory=list)

    # -- http transport -----------------------------------------------------
    url: str = ""
    #: `{"Authorization": "MY_TOKEN_VAR"}` — the value comes from the environment.
    header_env: dict[str, str] = Field(default_factory=dict)

    # -- policy -------------------------------------------------------------
    #: When set, only these remote tool names may be used. Empty means all of them.
    tool_allowlist: list[str] = Field(default_factory=list)
    #: Remote tools the operator has judged read-only. They then run without
    #: approval, like a built-in read tool. Everything else stays approval-gated.
    read_only_tools: list[str] = Field(default_factory=list)
    #: Remote tools pre-approved in `automatic` mode. Requires deliberate opt-in.
    auto_approve_tools: list[str] = Field(default_factory=list)
    timeout_seconds: float = Field(default=30.0, gt=0, le=600)

    @field_validator("name")
    @classmethod
    def _check_name(cls, value: str) -> str:
        if not NAME_PATTERN.match(value):
            raise ValueError(
                f"connector name must match {NAME_PATTERN.pattern} "
                "(lowercase letters, digits and underscores)"
            )
        return value

    @field_validator("url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        if value and not value.startswith(("http://", "https://")):
            raise ValueError("connector url must start with http:// or https://")
        return value

    @field_validator("env")
    @classmethod
    def _check_env_names(cls, value: list[str]) -> list[str]:
        for name in value:
            if not re.match(r"^[A-Za-z_][A-Za-z0-9_]*$", name):
                raise ValueError(f"{name!r} is not a valid environment variable name")
        return value

    @model_validator(mode="after")
    def _check_transport(self) -> ConnectorConfig:
        if self.kind is ConnectorKind.MCP_STDIO:
            if not self.command:
                raise ValueError("a stdio connector needs a `command`")
            if "/" in self.command or "\\" in self.command:
                # Mirrors the shell policy: a bare program name, resolved on PATH,
                # so a connector cannot point at an arbitrary binary by path.
                raise ValueError("connector `command` must be a bare program name, not a path")
        elif self.kind is ConnectorKind.MCP_HTTP and not self.url:
            raise ValueError("an http connector needs a `url`")
        return self

    def tool_name(self, remote_tool: str) -> str:
        """The namespaced name a remote tool is exposed under."""
        safe = re.sub(r"[^a-z0-9_]", "_", remote_tool.lower())
        return TOOL_NAME_TEMPLATE.format(connector=self.name, tool=safe)

    def permits(self, remote_tool: str) -> bool:
        """Whether this remote tool may be exposed at all."""
        return not self.tool_allowlist or remote_tool in self.tool_allowlist

    def is_read_only(self, remote_tool: str) -> bool:
        """Whether the operator has declared this remote tool read-only."""
        return remote_tool in self.read_only_tools

    def resolve_env(self, environ: dict[str, str]) -> dict[str, str]:
        """Collect the forwarded environment. Missing names are simply absent."""
        return {name: environ[name] for name in self.env if name in environ}

    def resolve_headers(self, environ: dict[str, str]) -> dict[str, str]:
        """Build request headers from the named environment variables."""
        headers: dict[str, str] = {}
        for header, variable in self.header_env.items():
            value = environ.get(variable, "").strip()
            if value:
                headers[header] = value
        return headers

    def missing_credentials(self, environ: dict[str, str]) -> list[str]:
        """Environment variables this connector expects but that are not set."""
        expected = [*self.env, *self.header_env.values()]
        return [name for name in expected if not environ.get(name, "").strip()]

    def safe_dump(self) -> dict[str, Any]:
        """A display-safe view: variable *names* only, never their values."""
        data = self.model_dump(mode="json")
        data["credentials"] = {
            "env_vars": sorted({*self.env, *self.header_env.values()}),
            "note": "values are read from the environment and never stored",
        }
        return data


class ConnectorCollection(BaseModel):
    """The set of configured connectors, as persisted for a UI to edit."""

    version: int = 1
    connectors: list[ConnectorConfig] = Field(default_factory=list)

    def get(self, name: str) -> ConnectorConfig | None:
        return next((c for c in self.connectors if c.name == name), None)

    def add(self, connector: ConnectorConfig, *, replace: bool = False) -> None:
        existing = self.get(connector.name)
        if existing is not None and not replace:
            raise ConfigurationError(
                f"a connector named {connector.name!r} already exists; "
                "remove it first or pass replace=True"
            )
        if existing is not None:
            self.connectors.remove(existing)
        self.connectors.append(connector)
        self.connectors.sort(key=lambda c: c.name)

    def remove(self, name: str) -> bool:
        existing = self.get(name)
        if existing is None:
            return False
        self.connectors.remove(existing)
        return True

    def enabled(self) -> list[ConnectorConfig]:
        return [c for c in self.connectors if c.enabled]
