"""Secret redaction.

Applied at every boundary where data leaves the runtime: the terminal, the JSONL
log, SQLite, and — most importantly — the model's context. Redaction is
conservative: it prefers destroying a harmless string to leaking a credential.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from typing import Any

PLACEHOLDER = "[REDACTED]"

#: Argument/field names whose *values* are always removed, regardless of content.
SENSITIVE_KEY_PATTERN = re.compile(
    r"(api[_-]?key|secret|passwd|password|token|credential|authorization|auth"
    r"|private[_-]?key|session[_-]?key|access[_-]?key|client[_-]?secret|cookie"
    r"|bearer|signature|salt|passphrase)",
    re.IGNORECASE,
)

#: Value shapes that look like credentials wherever they appear in free text.
VALUE_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    # Authorization headers, in any casing. The whole remainder of the line is
    # removed, not just the first token: "Bearer <token>" must not survive by
    # having only the scheme name matched.
    ("auth_header", re.compile(r"(?i)\b(authorization\s*[:=]\s*)([^\n]+)")),
    ("bearer", re.compile(r"(?i)\bBearer\s+[A-Za-z0-9._~+/\-]{8,}=*")),
    # Google / Gemini API keys.
    ("google_key", re.compile(r"\bAIza[0-9A-Za-z_\-]{20,}\b")),
    # Google OAuth credentials. AI Studio now issues keys in the `AQ.` form, and
    # `ya29.` is the long-standing access-token prefix; neither looks like AIza,
    # so both were previously invisible outside a NAME=value assignment.
    ("google_oauth", re.compile(r"\bAQ\.[A-Za-z0-9_\-]{20,}")),
    ("google_access_token", re.compile(r"\bya29\.[A-Za-z0-9_\-]{20,}")),
    # Google OAuth client secrets and refresh tokens.
    ("google_client_secret", re.compile(r"\bGOCSPX-[A-Za-z0-9_\-]{10,}")),
    ("google_refresh", re.compile(r"\b1//[0-9A-Za-z_\-]{20,}")),
    # OpenAI-style and generic long secret keys.
    ("sk_key", re.compile(r"\b(?:sk|rk|pk)-[A-Za-z0-9_\-]{16,}\b")),
    # GitHub tokens.
    ("github", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{16,}\b")),
    # AWS access key IDs.
    ("aws", re.compile(r"\b(?:AKIA|ASIA)[0-9A-Z]{12,}\b")),
    # Slack tokens.
    ("slack", re.compile(r"\bxox[baprs]-[A-Za-z0-9\-]{8,}\b")),
    # JSON Web Tokens.
    ("jwt", re.compile(r"\beyJ[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\.[A-Za-z0-9_\-]{8,}\b")),
    # PEM private key blocks.
    (
        "pem",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
    ),
    # `KEY=value` assignments for credential-shaped names.
    (
        "assignment",
        re.compile(
            r"(?i)\b([A-Z0-9_]*(?:KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL)[A-Z0-9_]*)"
            r"(\s*[:=]\s*)(\"[^\"]*\"|'[^']*'|\S+)"
        ),
    ),
)

#: Environment variables never worth inspecting for secret values.
_IGNORED_ENV_PREFIXES = ("LC_", "LANG", "PATH", "PWD", "HOME", "SHELL", "TERM", "USER")

#: Minimum length for an environment value to be treated as a literal secret.
_MIN_ENV_SECRET_LENGTH = 8


class Redactor:
    """Redacts secrets from strings and nested structures.

    Beyond pattern matching, the redactor also removes the *literal values* of
    credential-shaped environment variables, so a key that does not match any
    known pattern is still scrubbed if it is present in the environment.
    """

    def __init__(
        self,
        *,
        environ: Mapping[str, str] | None = None,
        extra_secrets: Sequence[str] = (),
        placeholder: str = PLACEHOLDER,
    ) -> None:
        self.placeholder = placeholder
        self._literals: list[str] = []
        env = os.environ if environ is None else environ
        for key, value in env.items():
            if key.startswith(_IGNORED_ENV_PREFIXES):
                continue
            if not SENSITIVE_KEY_PATTERN.search(key):
                continue
            stripped = value.strip()
            if len(stripped) >= _MIN_ENV_SECRET_LENGTH:
                self._literals.append(stripped)
        for secret in extra_secrets:
            stripped = secret.strip()
            if len(stripped) >= _MIN_ENV_SECRET_LENGTH:
                self._literals.append(stripped)
        # Longest first, so a superstring is replaced before its prefix.
        self._literals.sort(key=len, reverse=True)

    # -- core ---------------------------------------------------------------
    def redact_text(self, text: str) -> str:
        """Remove every recognised secret from `text`."""
        if not text:
            return text
        result = text
        for literal in self._literals:
            if literal in result:
                result = result.replace(literal, self.placeholder)
        for name, pattern in VALUE_PATTERNS:
            if name == "auth_header":
                result = pattern.sub(lambda m: f"{m.group(1)}{self.placeholder}", result)
            elif name == "assignment":
                result = pattern.sub(
                    lambda m: f"{m.group(1)}{m.group(2)}{self.placeholder}", result
                )
            else:
                result = pattern.sub(self.placeholder, result)
        return result

    def redact(self, value: Any) -> Any:
        """Recursively redact any JSON-like structure.

        Mapping keys that look sensitive have their values replaced outright;
        every string is additionally scanned for secret-shaped values.
        """
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, Mapping):
            out: dict[Any, Any] = {}
            for key, item in value.items():
                if isinstance(key, str) and SENSITIVE_KEY_PATTERN.search(key):
                    out[key] = self.placeholder
                else:
                    out[key] = self.redact(item)
            return out
        if isinstance(value, (list, tuple, set)):
            redacted = [self.redact(item) for item in value]
            if isinstance(value, tuple):
                return tuple(redacted)
            if isinstance(value, set):
                return set(redacted)
            return redacted
        return value

    def __call__(self, value: Any) -> Any:
        return self.redact(value)


#: A module-level default, useful for one-off calls and tests.
_default = Redactor()


def redact(value: Any) -> Any:
    """Redact using a default redactor built from the current environment."""
    return _default.redact(value)
