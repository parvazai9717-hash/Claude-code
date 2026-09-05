"""Provider construction.

The factory is the only place that maps a configuration name onto a concrete
adapter, so adding a provider means adding one file and one entry here.
"""

from __future__ import annotations

from typing import Any

from ..config import Config
from ..errors import ConfigurationError
from .base import ModelProvider
from .mock import MockProvider

#: Provider names the factory understands.
KNOWN_PROVIDERS = ("gemini", "ollama", "mock")


def create_provider(config: Config, **overrides: Any) -> ModelProvider:
    """Build the provider named by `config.provider`.

    Args:
        config: Validated configuration.
        **overrides: Passed through to the adapter — used by tests to inject a
            fake HTTP client or a scripted response list.

    Raises:
        ConfigurationError: For an unknown provider or a missing credential.
    """
    name = config.provider
    model = config.active_model

    if name == "mock":
        return MockProvider(model=model, **overrides)

    if name == "gemini":
        from .gemini import GeminiProvider

        return GeminiProvider(model=model, timeout=config.gemini_timeout_seconds, **overrides)

    if name == "ollama":
        from .ollama import OllamaProvider

        return OllamaProvider(
            model=model,
            base_url=config.ollama_base_url,
            timeout=config.ollama_timeout_seconds,
            **overrides,
        )

    raise ConfigurationError(
        f"unknown provider {name!r}; choose one of: {', '.join(KNOWN_PROVIDERS)}"
    )
