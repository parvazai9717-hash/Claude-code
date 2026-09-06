"""Configuration loading, precedence and validation."""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from agent.config import (
    DEFAULT_SHELL_ALLOWED,
    ApprovalMode,
    Config,
    Limits,
    find_config_file,
    load_config,
)
from agent.errors import ConfigurationError


def test_defaults_are_safe() -> None:
    config = Config()
    assert config.approval_mode is ApprovalMode.RISKY
    assert config.unsafe_disable_approvals is False
    assert config.auto_approve_tools == []
    assert config.shell_allowed_commands == DEFAULT_SHELL_ALLOWED


def test_file_values_are_loaded(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("provider: gemini\ngemini_model: test-model\nmax_steps: 7\n")
    config = load_config(path, environ={}, load_dotenv_file=False)
    assert config.provider == "gemini"
    assert config.gemini_model == "test-model"
    # Flat limit keys in the file are folded into `limits`.
    assert config.limits.max_steps == 7


def test_environment_overrides_file(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("provider: ollama\nlog_level: INFO\n")
    config = load_config(
        path,
        environ={"LOCAL_AGENT_PROVIDER": "gemini", "LOCAL_AGENT_LOG_LEVEL": "DEBUG"},
        load_dotenv_file=False,
    )
    assert config.provider == "gemini"
    assert config.log_level == "DEBUG"


def test_explicit_override_beats_environment(tmp_path: Path) -> None:
    config = load_config(
        None,
        environ={"LOCAL_AGENT_PROVIDER": "gemini"},
        load_dotenv_file=False,
        provider="mock",
    )
    assert config.provider == "mock"


def test_none_overrides_do_not_clobber(tmp_path: Path) -> None:
    """An unset CLI flag must not override a configured value."""
    path = tmp_path / "config.yaml"
    path.write_text("provider: gemini\n")
    config = load_config(path, environ={}, load_dotenv_file=False, provider=None, model=None)
    assert config.provider == "gemini"


def test_environment_limits_are_folded() -> None:
    config = load_config(
        None,
        environ={"LOCAL_AGENT_MAX_STEPS": "3", "LOCAL_AGENT_TOOL_TIMEOUT_SECONDS": "1.5"},
        load_dotenv_file=False,
    )
    assert config.limits.max_steps == 3
    assert config.limits.tool_timeout_seconds == 1.5


def test_environment_booleans_and_lists() -> None:
    config = load_config(
        None,
        environ={
            "LOCAL_AGENT_VERBOSE_TOOL_LOGGING": "true",
            "LOCAL_AGENT_AUTO_APPROVE_TOOLS": "read_file,list_files",
        },
        load_dotenv_file=False,
    )
    assert config.verbose_tool_logging is True
    assert config.auto_approve_tools == ["read_file", "list_files"]


def test_unknown_env_vars_are_ignored() -> None:
    config = load_config(
        None, environ={"LOCAL_AGENT_NOT_A_REAL_SETTING": "x"}, load_dotenv_file=False
    )
    assert config.provider == "ollama"


def test_missing_config_file_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="not found"):
        load_config(tmp_path / "nope.yaml", environ={}, load_dotenv_file=False)


def test_malformed_yaml_is_reported(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("provider: [unclosed\n")
    with pytest.raises(ConfigurationError, match="not valid"):
        load_config(path, environ={}, load_dotenv_file=False)


def test_invalid_value_gives_an_actionable_error(tmp_path: Path) -> None:
    path = tmp_path / "config.yaml"
    path.write_text("log_level: LOUD\n")
    with pytest.raises(ConfigurationError, match="log_level"):
        load_config(path, environ={}, load_dotenv_file=False)


def test_forbidden_shell_command_cannot_be_allowlisted() -> None:
    with pytest.raises(ValueError, match="never be allowlisted"):
        Config(shell_allowed_commands=["ls", "sudo"])


def test_bad_ollama_url_is_rejected() -> None:
    with pytest.raises(ValueError, match="http"):
        Config(ollama_base_url="localhost:11434")


def test_active_model_resolves_per_provider() -> None:
    assert Config(provider="gemini", gemini_model="g").active_model == "g"
    assert Config(provider="ollama", ollama_model="o").active_model == "o"
    assert Config(provider="gemini", model="override").active_model == "override"


def test_safe_dump_never_contains_a_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "sk-super-secret-value")
    data = Config(provider="gemini").safe_dump()
    assert "sk-super-secret-value" not in str(data)
    assert data["credential"]["present"] is True
    assert data["credential"]["env_var"] == "GEMINI_API_KEY"


def test_has_credential_is_false_when_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert Config(provider="gemini").has_credential() is False
    assert Config(provider="ollama").has_credential() is True


def test_limits_reject_out_of_range() -> None:
    with pytest.raises(ValueError):
        Limits(max_steps=0)
    with pytest.raises(ValueError):
        Limits(tool_timeout_seconds=-1)


def test_find_config_file_prefers_yaml(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text("provider='mock'\n")
    (tmp_path / "config.yaml").write_text("provider: mock\n")
    assert find_config_file(tmp_path) == tmp_path / "config.yaml"


def test_toml_config_is_supported(tmp_path: Path) -> None:
    path = tmp_path / "config.toml"
    path.write_text('provider = "mock"\nmax_steps = 5\n')
    config = load_config(path, environ={}, load_dotenv_file=False)
    assert config.provider == "mock"
    assert config.limits.max_steps == 5


# -- .env discovery ---------------------------------------------------------
def test_env_file_is_found_in_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A user expects the `.env` beside them, not one beside the installed package."""
    from agent.config import find_environment_file

    monkeypatch.chdir(tmp_path)
    assert find_environment_file() is None
    (tmp_path / ".env").write_text("GEMINI_API_KEY=x\n")
    assert find_environment_file() == tmp_path / ".env"


def test_env_file_is_found_in_a_parent_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("GEMINI_API_KEY=x\n")
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    monkeypatch.chdir(nested)
    from agent.config import find_environment_file

    assert find_environment_file() == tmp_path / ".env"


def test_loading_env_does_not_clobber_the_real_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An exported variable is a deliberate override and must win over a file."""
    from agent.config import load_environment_file

    monkeypatch.chdir(tmp_path)
    (tmp_path / ".env").write_text("LOCAL_AGENT_TEST_VALUE=from-file\n")
    monkeypatch.setenv("LOCAL_AGENT_TEST_VALUE", "from-shell")
    load_environment_file()
    assert os.environ["LOCAL_AGENT_TEST_VALUE"] == "from-shell"


def test_loading_env_sets_an_unset_variable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent.config import load_environment_file

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("LOCAL_AGENT_TEST_VALUE", raising=False)
    (tmp_path / ".env").write_text("LOCAL_AGENT_TEST_VALUE=from-file\n")
    assert load_environment_file() == tmp_path / ".env"
    assert os.environ["LOCAL_AGENT_TEST_VALUE"] == "from-file"


def test_a_missing_env_file_is_not_an_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from agent.config import load_environment_file

    monkeypatch.chdir(tmp_path)
    assert load_environment_file() is None


# -- Windows paths in YAML --------------------------------------------------
def test_a_windows_path_loads_from_yaml(tmp_path: Path) -> None:
    """Forward slashes and single quotes both work; double quotes do not.

    A double-quoted YAML scalar processes escape sequences, so `"D:\\Agent"`
    fails to parse at all. That is a confusing way to lose an afternoon, so the
    forms are pinned here and the example file warns about it.
    """
    for text in (
        "data_dir: D:/Agent/data",
        r"data_dir: 'D:\Agent\data'",
        r"data_dir: D:\Agent\data",
    ):
        path = tmp_path / "config.yaml"
        path.write_text(text + "\n")
        config = load_config(path, environ={}, load_dotenv_file=False)
        assert "Agent" in str(config.data_dir)


def test_a_double_quoted_windows_path_is_reported_not_swallowed(
    tmp_path: Path,
) -> None:
    path = tmp_path / "config.yaml"
    path.write_text('data_dir: "D:\\Agent\\data"\n')
    with pytest.raises(ConfigurationError, match="not valid"):
        load_config(path, environ={}, load_dotenv_file=False)


def test_the_shipped_example_still_parses() -> None:
    """The example is documentation people copy; it must always load."""
    config = load_config(Path("config.example.yaml"), environ={}, load_dotenv_file=False)
    assert config.provider in {"gemini", "ollama", "mock"}
