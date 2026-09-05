"""Structured error taxonomy.

Every failure the runtime can encounter is mapped onto an :class:`ErrorCategory`
so that the runtime, the logs and the model context all describe failures in the
same vocabulary. Error *messages* that cross the model boundary are redacted by
:mod:`agent.security.redaction`; the categories themselves never carry secrets.
"""

from __future__ import annotations

from enum import StrEnum


class ErrorCategory(StrEnum):
    """Coarse, stable failure classes used for logging, replanning and tests."""

    # --- provider failures -------------------------------------------------
    PROVIDER_AUTH = "provider_auth"
    PROVIDER_UNAVAILABLE = "provider_unavailable"
    PROVIDER_TIMEOUT = "provider_timeout"
    PROVIDER_RATE_LIMIT = "provider_rate_limit"
    PROVIDER_INVALID_REQUEST = "provider_invalid_request"
    PROVIDER_MALFORMED_RESPONSE = "provider_malformed_response"
    PROVIDER_UNSUPPORTED = "provider_unsupported"
    PROVIDER_UNKNOWN = "provider_unknown"

    # --- tool failures -----------------------------------------------------
    UNKNOWN_TOOL = "unknown_tool"
    INVALID_ARGUMENTS = "invalid_arguments"
    PERMISSION_DENIED = "permission_denied"
    APPROVAL_DENIED = "approval_denied"
    APPROVAL_REQUIRED = "approval_required"
    PATH_ESCAPE = "path_escape"
    SECRET_PROTECTED = "secret_protected"
    NOT_FOUND = "not_found"
    TOOL_FAILED = "tool_failed"
    TOOL_TIMEOUT = "tool_timeout"
    LIMIT_EXCEEDED = "limit_exceeded"

    # --- runtime / lifecycle ----------------------------------------------
    VERIFICATION_FAILED = "verification_failed"
    VERIFICATION_UNAVAILABLE = "verification_unavailable"
    CONTEXT_OVERFLOW = "context_overflow"
    CANCELLED = "cancelled"
    PAUSED = "paused"
    CONFIGURATION = "configuration"
    CAPABILITY_UNAVAILABLE = "capability_unavailable"
    SKILL_INVALID = "skill_invalid"
    STORAGE = "storage"
    UNKNOWN = "unknown"


class AgentError(Exception):
    """Base class for every error raised inside the runtime.

    Attributes:
        category: The stable failure class.
        message: A human-readable, already-safe description.
        details: Optional structured, non-secret context for logs and tests.
        retryable: Whether re-attempting the same action could plausibly succeed.
    """

    category: ErrorCategory = ErrorCategory.UNKNOWN

    def __init__(
        self,
        message: str,
        *,
        category: ErrorCategory | None = None,
        details: dict[str, object] | None = None,
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        if category is not None:
            self.category = category
        self.details: dict[str, object] = details or {}
        self.retryable = retryable

    def to_dict(self) -> dict[str, object]:
        """Return a JSON-safe representation for logs and tool results."""
        return {
            "category": self.category.value,
            "message": self.message,
            "details": self.details,
            "retryable": self.retryable,
        }

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"{type(self).__name__}({self.category.value}: {self.message!r})"


# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------
class ConfigurationError(AgentError):
    """Configuration is missing, malformed or internally inconsistent."""

    category = ErrorCategory.CONFIGURATION


# --------------------------------------------------------------------------
# Providers
# --------------------------------------------------------------------------
class ProviderError(AgentError):
    """Base class for model-provider failures."""

    category = ErrorCategory.PROVIDER_UNKNOWN


class ProviderAuthError(ProviderError):
    category = ErrorCategory.PROVIDER_AUTH


class ProviderUnavailableError(ProviderError):
    category = ErrorCategory.PROVIDER_UNAVAILABLE


class ProviderTimeoutError(ProviderError):
    category = ErrorCategory.PROVIDER_TIMEOUT


class ProviderRateLimitError(ProviderError):
    category = ErrorCategory.PROVIDER_RATE_LIMIT


class ProviderInvalidRequestError(ProviderError):
    category = ErrorCategory.PROVIDER_INVALID_REQUEST


class ProviderMalformedResponseError(ProviderError):
    category = ErrorCategory.PROVIDER_MALFORMED_RESPONSE


class ProviderUnsupportedError(ProviderError):
    """The selected provider or model cannot do what was asked (e.g. tool calls)."""

    category = ErrorCategory.PROVIDER_UNSUPPORTED


# --------------------------------------------------------------------------
# Tools and security
# --------------------------------------------------------------------------
class ToolError(AgentError):
    """Base class for tool-layer failures."""

    category = ErrorCategory.TOOL_FAILED


class UnknownToolError(ToolError):
    category = ErrorCategory.UNKNOWN_TOOL


class InvalidArgumentsError(ToolError):
    category = ErrorCategory.INVALID_ARGUMENTS


class ToolTimeoutError(ToolError):
    category = ErrorCategory.TOOL_TIMEOUT


class LimitExceededError(ToolError):
    category = ErrorCategory.LIMIT_EXCEEDED


class SecurityError(AgentError):
    """Base class for refusals produced by the security layer."""

    category = ErrorCategory.PERMISSION_DENIED


class PermissionDeniedError(SecurityError):
    category = ErrorCategory.PERMISSION_DENIED


class ApprovalDeniedError(SecurityError):
    category = ErrorCategory.APPROVAL_DENIED


class PathEscapeError(SecurityError):
    """A path resolved outside the configured workspace."""

    category = ErrorCategory.PATH_ESCAPE


class SecretProtectedError(SecurityError):
    """A path or value is protected because it is credential-bearing."""

    category = ErrorCategory.SECRET_PROTECTED


# --------------------------------------------------------------------------
# Lifecycle
# --------------------------------------------------------------------------
class VerificationFailedError(AgentError):
    category = ErrorCategory.VERIFICATION_FAILED


class CancelledError(AgentError):
    category = ErrorCategory.CANCELLED


class CapabilityUnavailableError(AgentError):
    """A boundary interface exists but the capability is deliberately not enabled."""

    category = ErrorCategory.CAPABILITY_UNAVAILABLE


class SkillError(AgentError):
    category = ErrorCategory.SKILL_INVALID


class StorageError(AgentError):
    category = ErrorCategory.STORAGE
