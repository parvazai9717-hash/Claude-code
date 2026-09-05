"""Configuration loading and validation.

Precedence, highest first:

1. Explicit keyword overrides (CLI flags).
2. Environment variables — ``LOCAL_AGENT_*`` for settings, provider-native names
   (``GEMINI_API_KEY``) for credentials.
3. A YAML or TOML configuration file.
4. Built-in defaults.

Credentials are never stored in the config file and never rendered by
``config show``; only their presence is reported.
"""

from __future__ import annotations

import os
import tomllib
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, Field, ValidationError, field_validator, model_validator

from .errors import ConfigurationError

ENV_PREFIX = "LOCAL_AGENT_"

#: Config filenames searched, in order, when no path is given.
CONFIG_FILENAMES = ("config.yaml", "config.yml", "config.toml")

#: Environment variables that hold credentials. Their *values* never leave this module.
CREDENTIAL_ENV_VARS = {
    "gemini": "GEMINI_API_KEY",
}

ProviderName = Literal["gemini", "ollama", "mock"]


class ApprovalMode(StrEnum):
    """How much the human is asked to confirm."""

    #: Approve every action that is not read-only.
    ALWAYS = "always"
    #: Approve writes, shell, network, deletion, installs, auth and external effects.
    RISKY = "risky"
    #: Run only explicitly allowlisted tools unattended; everything else is denied.
    AUTOMATIC = "automatic"


#: Conservative deny-by-default shell allowlist. Read-only inspection commands only.
DEFAULT_SHELL_ALLOWED = [
    "ls",
    "cat",
    "head",
    "tail",
    "wc",
    "echo",
    "pwd",
    "grep",
    "find",
    "python3",
    "pytest",
    "git",
]

#: Never runnable, whatever the allowlist says. Belt and braces around the allowlist.
SHELL_FORBIDDEN = frozenset(
    {
        "sudo",
        "su",
        "doas",
        "rm",
        "rmdir",
        "mkfs",
        "dd",
        "shutdown",
        "reboot",
        "halt",
        "poweroff",
        "chown",
        "chmod",
        "mount",
        "umount",
        "useradd",
        "userdel",
        "passwd",
        "curl",
        "wget",
        "nc",
        "ncat",
        "netcat",
        "ssh",
        "scp",
        "sftp",
        "rsync",
        "pip",
        "pip3",
        "npm",
        "yarn",
        "pnpm",
        "apt",
        "apt-get",
        "brew",
        "docker",
        "kubectl",
        "systemctl",
        "crontab",
        "sh",
        "bash",
        "zsh",
        "fish",
        "eval",
        "exec",
    }
)


class Limits(BaseModel):
    """Every bound the runtime enforces. All are configurable."""

    max_steps: int = Field(default=20, ge=1, le=500)
    max_tool_calls_per_step: int = Field(default=4, ge=1, le=50)
    max_total_tool_calls: int = Field(default=40, ge=1, le=1000)
    max_retries_per_action: int = Field(default=2, ge=0, le=10)
    max_replans: int = Field(default=5, ge=0, le=50)

    provider_timeout_seconds: float = Field(default=120.0, gt=0, le=3600)
    tool_timeout_seconds: float = Field(default=30.0, gt=0, le=3600)

    max_file_bytes: int = Field(default=1_000_000, ge=1, le=100_000_000)
    max_tool_output_chars: int = Field(default=12_000, ge=100, le=1_000_000)
    max_model_output_tokens: int = Field(default=4096, ge=1, le=1_000_000)
    max_conversation_messages: int = Field(default=200, ge=4, le=10_000)
    max_persisted_facts: int = Field(default=500, ge=1, le=100_000)
    max_search_results: int = Field(default=100, ge=1, le=10_000)
    max_list_entries: int = Field(default=500, ge=1, le=100_000)
    max_directory_depth: int = Field(default=6, ge=1, le=64)


class Config(BaseModel):
    """Validated runtime configuration."""

    provider: ProviderName = "ollama"
    model: str | None = None

    workspace: Path = Path("./workspace")
    data_dir: Path = Path("~/.local-agent")

    gemini_model: str = "gemini-2.5-flash"
    gemini_timeout_seconds: float = 120.0

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.1"
    ollama_timeout_seconds: float = 120.0

    approval_mode: ApprovalMode = ApprovalMode.RISKY
    auto_approve_tools: list[str] = Field(default_factory=list)
    #: Testing-only. UNSAFE: skips every approval prompt. Never a default.
    unsafe_disable_approvals: bool = False

    shell_allowed_commands: list[str] = Field(default_factory=lambda: list(DEFAULT_SHELL_ALLOWED))
    shell_enabled: bool = True

    skills_dir: Path = Path("./skills")

    limits: Limits = Field(default_factory=Limits)

    log_level: str = "INFO"
    verbose_tool_logging: bool = False

    # -- validation ---------------------------------------------------------
    @field_validator("workspace", "data_dir", "skills_dir", mode="before")
    @classmethod
    def _expand(cls, value: Any) -> Any:
        if value is None:
            return value
        return Path(str(value)).expanduser()

    @field_validator("log_level")
    @classmethod
    def _check_log_level(cls, value: str) -> str:
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        upper = value.upper()
        if upper not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return upper

    @field_validator("ollama_base_url")
    @classmethod
    def _check_url(cls, value: str) -> str:
        if not value.startswith(("http://", "https://")):
            raise ValueError("ollama_base_url must start with http:// or https://")
        return value.rstrip("/")

    @field_validator("shell_allowed_commands")
    @classmethod
    def _check_shell_allowlist(cls, value: list[str]) -> list[str]:
        forbidden = sorted(set(value) & SHELL_FORBIDDEN)
        if forbidden:
            raise ValueError("these commands can never be allowlisted: " + ", ".join(forbidden))
        return value

    @model_validator(mode="after")
    def _check_consistency(self) -> Config:
        if self.approval_mode is not ApprovalMode.AUTOMATIC and self.auto_approve_tools:
            # Harmless, but the user probably expected it to take effect.
            pass
        return self

    # -- derived ------------------------------------------------------------
    @property
    def active_model(self) -> str:
        """The model actually used, resolving the generic `model` override."""
        if self.model:
            return self.model
        if self.provider == "gemini":
            return self.gemini_model
        if self.provider == "ollama":
            return self.ollama_model
        return "mock"

    @property
    def provider_timeout(self) -> float:
        """Provider-specific timeout, falling back to the shared limit."""
        if self.provider == "gemini":
            return self.gemini_timeout_seconds
        if self.provider == "ollama":
            return self.ollama_timeout_seconds
        return self.limits.provider_timeout_seconds

    @property
    def database_path(self) -> Path:
        return self.data_dir / "agent.sqlite3"

    @property
    def log_path(self) -> Path:
        return self.data_dir / "events.jsonl"

    def credential_env_var(self) -> str | None:
        """The env var holding this provider's credential, if it needs one."""
        return CREDENTIAL_ENV_VARS.get(self.provider)

    def has_credential(self) -> bool:
        """Whether the credential is present. Never returns the value itself."""
        var = self.credential_env_var()
        if var is None:
            return True
        return bool(os.environ.get(var, "").strip())

    def safe_dump(self) -> dict[str, Any]:
        """A display-safe view: paths resolved, secrets reduced to presence flags."""
        data = self.model_dump(mode="json")
        data["workspace"] = str(self.workspace.expanduser())
        data["data_dir"] = str(self.data_dir.expanduser())
        data["skills_dir"] = str(self.skills_dir.expanduser())
        data["active_model"] = self.active_model
        var = self.credential_env_var()
        data["credential"] = {
            "env_var": var or "(none required)",
            "present": self.has_credential(),
        }
        return data

    def ensure_directories(self) -> None:
        """Create the workspace, data directory and workspace subdirectories."""
        from .tools.filesystem import WORKSPACE_SUBDIRS

        workspace = self.workspace.expanduser()
        workspace.mkdir(parents=True, exist_ok=True)
        for sub in WORKSPACE_SUBDIRS:
            (workspace / sub).mkdir(parents=True, exist_ok=True)
        self.data_dir.expanduser().mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def _coerce_env_value(raw: str) -> Any:
    """Turn an environment string into a bool/int/float/list where sensible."""
    lowered = raw.strip().lower()
    if lowered in {"true", "yes", "on"}:
        return True
    if lowered in {"false", "no", "off"}:
        return False
    if lowered in {"null", "none", ""}:
        return None
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        pass
    if "," in raw:
        return [part.strip() for part in raw.split(",") if part.strip()]
    return raw


def _read_config_file(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigurationError(f"cannot read config file {path}: {exc.strerror}") from exc
    try:
        if path.suffix == ".toml":
            loaded: Any = tomllib.loads(text)
        else:
            loaded = yaml.safe_load(text)
    except (yaml.YAMLError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"config file {path} is not valid: {exc}") from exc
    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigurationError(f"config file {path} must contain a mapping at the top level")
    return loaded


def find_config_file(start: Path | None = None) -> Path | None:
    """Locate a config file in `start` (default: the current directory)."""
    base = (start or Path.cwd()).expanduser()
    for name in CONFIG_FILENAMES:
        candidate = base / name
        if candidate.is_file():
            return candidate
    return None


def _env_overrides(environ: dict[str, str]) -> dict[str, Any]:
    """Collect `LOCAL_AGENT_*` variables, mapping limit keys into `limits`."""
    limit_fields = set(Limits.model_fields)
    config_fields = set(Config.model_fields)
    overrides: dict[str, Any] = {}
    limits: dict[str, Any] = {}
    for key, raw in environ.items():
        if not key.startswith(ENV_PREFIX):
            continue
        field = key[len(ENV_PREFIX) :].lower()
        value = _coerce_env_value(raw)
        if field in limit_fields:
            limits[field] = value
        elif field in config_fields:
            overrides[field] = value
        # Unknown LOCAL_AGENT_* names are ignored rather than failing startup:
        # the user may be setting variables for a future version or a wrapper.
    if limits:
        overrides["limits"] = limits
    return overrides


def _merge(base: dict[str, Any], overlay: dict[str, Any]) -> dict[str, Any]:
    """Shallow merge, with a nested merge for the `limits` mapping."""
    merged = dict(base)
    for key, value in overlay.items():
        if key == "limits" and isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        elif value is not None or key not in merged:
            merged[key] = value
    return merged


def load_config(
    config_path: Path | None = None,
    *,
    environ: dict[str, str] | None = None,
    load_dotenv_file: bool = True,
    **overrides: Any,
) -> Config:
    """Build a validated :class:`Config` from file, environment and overrides.

    Args:
        config_path: Explicit config file. When omitted, the current directory is
            searched for `config.yaml`, `config.yml`, then `config.toml`.
        environ: Environment mapping to read (defaults to `os.environ`). Injectable
            so tests never depend on the real environment.
        load_dotenv_file: Load a `.env` file into the process environment first.
        **overrides: Highest-precedence values, typically from CLI flags. `None`
            values are ignored so an unset flag does not clobber configuration.

    Raises:
        ConfigurationError: With an actionable message when validation fails.
    """
    if load_dotenv_file and environ is None:
        try:
            from dotenv import load_dotenv

            load_dotenv(override=False)
        except ImportError:  # pragma: no cover - python-dotenv is a hard dependency
            pass

    env = dict(os.environ if environ is None else environ)

    file_values: dict[str, Any] = {}
    resolved_path = config_path or find_config_file()
    if config_path is not None and not config_path.is_file():
        raise ConfigurationError(f"config file not found: {config_path}")
    if resolved_path is not None:
        file_values = _read_config_file(resolved_path)
        # Flat limit keys in the file are folded into `limits` for convenience.
        limit_fields = set(Limits.model_fields)
        flat_limits = {k: file_values.pop(k) for k in list(file_values) if k in limit_fields}
        if flat_limits:
            existing = file_values.get("limits")
            file_values["limits"] = {**(existing or {}), **flat_limits}

    values = _merge(file_values, _env_overrides(env))
    values = _merge(values, {k: v for k, v in overrides.items() if v is not None})

    try:
        config = Config(**values)
    except ValidationError as exc:
        details = "; ".join(
            f"{'.'.join(str(p) for p in err['loc'])}: {err['msg']}" for err in exc.errors()
        )
        source = str(resolved_path) if resolved_path else "defaults/environment"
        raise ConfigurationError(
            f"invalid configuration ({source}): {details}",
            details={"errors": details},
        ) from exc
    return config
