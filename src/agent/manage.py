"""The management API: one surface for the CLI, a UI, or an HTTP server.

Everything an operator can do to configure and inspect the agent lives here as
plain Python returning plain dictionaries. The CLI is a thin renderer over it,
and a future UI is another renderer over the same calls — so there is no
privileged path, and no second implementation to drift.

Two rules shape it:

- **Nothing here executes agent work.** It configures and inspects. Running a
  task still goes through :class:`~agent.runtime.AgentRunner`, with its approvals.
- **Nothing here returns a credential.** Connectors name environment variables;
  this API reports whether they are set, never what they contain.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .config import Config
from .connectors.config import ConnectorConfig, ConnectorKind
from .connectors.manager import ConnectorManager
from .connectors.store import ConnectorStore
from .errors import ConfigurationError
from .media import ALLOWED_MIME_TYPES, MediaKind
from .memory.conversations import ConversationStore
from .memory.database import Database
from .memory.facts import FactStore
from .memory.tasks import TaskStore
from .providers.factory import KNOWN_PROVIDERS, create_provider
from .security.permissions import PermissionChecker
from .skills.registry import SkillRegistry
from .task_state import TaskStatus
from .tools import build_default_tools
from .tools.registry import ToolRegistry


@dataclass
class AgentManager:
    """Configuration and inspection for one configured agent."""

    config: Config
    database: Database

    # -- connectors ---------------------------------------------------------
    @property
    def connector_store(self) -> ConnectorStore:
        return ConnectorStore.for_data_dir(self.config.data_dir)

    def list_connectors(self) -> list[dict[str, Any]]:
        """Every configured connector, with credential *presence* only."""
        import os

        environ = dict(os.environ)
        collection = self.connector_store.load()
        return [
            {
                **connector.safe_dump(),
                "missing_credentials": connector.missing_credentials(environ),
                "tool_prefix": connector.tool_name(""),
            }
            for connector in collection.connectors
        ]

    def add_connector(
        self,
        *,
        name: str,
        kind: str,
        command: str = "",
        args: list[str] | None = None,
        url: str = "",
        env: list[str] | None = None,
        header_env: dict[str, str] | None = None,
        description: str = "",
        enabled: bool = False,
        tool_allowlist: list[str] | None = None,
        read_only_tools: list[str] | None = None,
        timeout_seconds: float = 30.0,
        replace: bool = False,
    ) -> dict[str, Any]:
        """Define a connector. It starts **disabled** unless explicitly enabled.

        Raises:
            ConfigurationError: On an invalid definition, with the reason named.
        """
        try:
            connector = ConnectorConfig(
                name=name,
                kind=ConnectorKind(kind),
                command=command,
                args=args or [],
                url=url,
                env=env or [],
                header_env=header_env or {},
                description=description,
                enabled=enabled,
                tool_allowlist=tool_allowlist or [],
                read_only_tools=read_only_tools or [],
                timeout_seconds=timeout_seconds,
            )
        except ValueError as exc:
            raise ConfigurationError(f"invalid connector definition: {exc}") from exc
        self.connector_store.add(connector, replace=replace)
        return connector.safe_dump()

    def remove_connector(self, name: str) -> bool:
        return self.connector_store.remove(name)

    def set_connector_enabled(self, name: str, enabled: bool) -> dict[str, Any] | None:
        connector = self.connector_store.set_enabled(name, enabled)
        return connector.safe_dump() if connector else None

    def connector_manager(self, **kwargs: Any) -> ConnectorManager:
        return ConnectorManager(self.connector_store.load(), **kwargs)

    async def test_connectors(self, name: str | None = None, **kwargs: Any) -> list[dict[str, Any]]:
        """Contact connectors and report what each one offers, or why it failed."""
        manager = self.connector_manager(**kwargs)
        try:
            statuses = await manager.health_check(name)
            return [status.to_dict() for status in statuses]
        finally:
            await manager.aclose()

    # -- tools --------------------------------------------------------------
    def list_tools(self) -> list[dict[str, Any]]:
        """The built-in tool set, with its risk and approval classification.

        Connector tools are not listed here because they exist only while a
        connection is open; use :meth:`test_connectors` to enumerate those.
        """
        registry = ToolRegistry(permissions=PermissionChecker(self.config))
        registry.register_all(build_default_tools(include_shell=self.config.shell_enabled))
        return [
            {
                "name": definition.name,
                "description": definition.description,
                "risk": definition.risk.value,
                "risk_category": definition.risk_category.value,
                "read_only": definition.read_only,
                "requires_approval": definition.requires_approval,
                "requires_verification": definition.requires_verification,
                "reversible": definition.reversible,
                "schema": definition.json_schema(),
                "source": "built-in",
            }
            for definition in registry.definitions()
        ]

    # -- providers and capabilities ----------------------------------------
    def list_providers(self) -> list[dict[str, Any]]:
        """Selectable providers and what each can perceive."""
        entries: list[dict[str, Any]] = []
        for name in KNOWN_PROVIDERS:
            candidate = self.config.model_copy(update={"provider": name})
            entry: dict[str, Any] = {
                "name": name,
                "active": name == self.config.provider,
                "model": candidate.active_model,
                "credential_env_var": candidate.credential_env_var(),
                "credential_present": candidate.has_credential(),
            }
            try:
                provider = create_provider(candidate)
                capabilities = provider.capabilities()
                entry["capabilities"] = capabilities.model_dump()
            except Exception as exc:
                entry["capabilities"] = None
                entry["detail"] = str(exc)[:200]
            entries.append(entry)
        return entries

    async def check_provider(self) -> dict[str, Any]:
        """Health-check the active provider."""
        provider = create_provider(self.config)
        try:
            status = await provider.health_check()
            return status.model_dump(mode="json")
        finally:
            close = getattr(provider, "aclose", None)
            if close is not None:
                await close()

    def media_support(self) -> dict[str, Any]:
        """What media the active provider accepts, and the limits in force."""
        try:
            capabilities = create_provider(self.config).capabilities()
            accepted = {
                MediaKind.IMAGE.value: capabilities.vision,
                MediaKind.AUDIO.value: capabilities.audio,
                MediaKind.VIDEO.value: capabilities.video and self.config.media.enable_video,
            }
        except Exception:
            accepted = {kind.value: False for kind in MediaKind}
        return {
            "accepted": accepted,
            "video_enabled": self.config.media.enable_video,
            "limits": self.config.media.model_dump(),
            "supported_types": {
                kind.value: sorted(types) for kind, types in ALLOWED_MIME_TYPES.items()
            },
        }

    # -- skills -------------------------------------------------------------
    def list_skills(self) -> list[dict[str, Any]]:
        registry = SkillRegistry.from_directory(self.config.skills_dir)
        available_tools = [t.name for t in build_default_tools()]
        entries = [
            {
                "name": name,
                "description": (
                    check := registry.check(
                        name, available_tools=available_tools, config=self.config
                    )
                ).skill.description,
                "available": check.available,
                "reason": check.reason,
                "required_tools": check.skill.required_tools,
                "missing_tools": list(check.missing_tools),
            }
            for name in registry.names()
        ]
        entries.extend(
            {
                "name": name,
                "description": "",
                "available": False,
                "reason": reason,
                "required_tools": [],
                "missing_tools": [],
            }
            for name, reason in registry.problems
        )
        return entries

    # -- history ------------------------------------------------------------
    def list_tasks(self, *, status: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        store = TaskStore(self.database)
        return store.list_tasks(status=TaskStatus(status) if status else None, limit=limit)

    def get_task(self, task_id: str) -> dict[str, Any] | None:
        task = TaskStore(self.database).find(task_id)
        return task.model_dump(mode="json") if task else None

    def list_sessions(self, limit: int = 50) -> list[dict[str, Any]]:
        return ConversationStore(self.database).list_sessions(limit=limit)

    def list_facts(self, *, approved_only: bool = True) -> list[dict[str, Any]]:
        return FactStore(self.database).list_facts(approved_only=approved_only)

    def stats(self) -> dict[str, Any]:
        """A dashboard summary: what is configured and what is stored."""
        connectors = self.connector_store.load()
        return {
            "provider": self.config.provider,
            "model": self.config.active_model,
            "approval_mode": self.config.approval_mode.value,
            "workspace": str(self.config.workspace.expanduser()),
            "shell_enabled": self.config.shell_enabled,
            "connectors_enabled": self.config.connectors_enabled,
            "connectors": {
                "configured": len(connectors.connectors),
                "enabled": len(connectors.enabled()),
            },
            "database": self.database.stats(),
        }

    def safe_config(self) -> dict[str, Any]:
        """The effective configuration, with credentials reduced to presence flags."""
        return self.config.safe_dump()
