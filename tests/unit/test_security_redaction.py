"""Secret redaction at every boundary."""

from __future__ import annotations

import pytest

from agent.security.redaction import PLACEHOLDER, Redactor


def test_environment_secret_values_are_removed() -> None:
    redactor = Redactor(environ={"MY_API_KEY": "supersecret-value-9876"})
    assert "supersecret-value-9876" not in redactor.redact_text("key=supersecret-value-9876")


def test_short_environment_values_are_not_treated_as_secrets() -> None:
    """A very short value would redact ordinary words everywhere."""
    redactor = Redactor(environ={"MY_TOKEN": "abc"})
    assert redactor.redact_text("abc def") == "abc def"


def test_google_api_key_pattern() -> None:
    redactor = Redactor(environ={})
    text = redactor.redact_text("use AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q now")
    assert "AIzaSy" not in text
    assert PLACEHOLDER in text


def test_bearer_and_authorization_headers() -> None:
    redactor = Redactor(environ={})
    assert "abcdefghijklmnop" not in redactor.redact_text("Authorization: Bearer abcdefghijklmnop")
    assert "xyzxyzxyzxyzxyz" not in redactor.redact_text("Bearer xyzxyzxyzxyzxyz")


def test_common_token_shapes() -> None:
    redactor = Redactor(environ={})
    for secret in (
        "sk-abcdefghijklmnopqrstuvwx",
        "ghp_abcdefghijklmnopqrstuvwxyz1234",
        "AKIAIOSFODNN7EXAMPLE",
        "xoxb-1234567890-abcdefghij",
        "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N",
    ):
        assert secret not in redactor.redact_text(f"value: {secret}")


def test_private_key_block() -> None:
    redactor = Redactor(environ={})
    pem = "-----BEGIN RSA PRIVATE KEY-----\nMIIEow\n-----END RSA PRIVATE KEY-----"
    assert "MIIEow" not in redactor.redact_text(pem)


def test_assignment_shapes() -> None:
    redactor = Redactor(environ={})
    for text in ("API_KEY=hunter2hunter2", "db_password: 'p@ssw0rd123'", "SECRET_TOKEN = zzzzzzzz"):
        assert PLACEHOLDER in redactor.redact_text(text)


def test_sensitive_keys_are_removed_from_mappings() -> None:
    redactor = Redactor(environ={})
    result = redactor.redact(
        {"api_key": "anything", "nested": {"password": "p", "safe": "keep me"}, "n": 1}
    )
    assert result["api_key"] == PLACEHOLDER
    assert result["nested"]["password"] == PLACEHOLDER
    assert result["nested"]["safe"] == "keep me"
    assert result["n"] == 1


def test_nested_collections_are_walked() -> None:
    redactor = Redactor(environ={"TOKEN_VALUE": "topsecretvalue123"})
    result = redactor.redact({"items": [{"note": "topsecretvalue123"}, ("topsecretvalue123",)]})
    assert "topsecretvalue123" not in str(result)


def test_non_string_scalars_pass_through() -> None:
    redactor = Redactor(environ={})
    assert redactor.redact(42) == 42
    assert redactor.redact(None) is None
    assert redactor.redact(True) is True


def test_extra_secrets_are_honoured() -> None:
    redactor = Redactor(environ={}, extra_secrets=["a-custom-secret-string"])
    assert "a-custom-secret-string" not in redactor.redact_text("x a-custom-secret-string y")


@pytest.mark.parametrize(
    ("secret", "what"),
    [
        ("AIzaSyA1B2C3D4E5F6G7H8I9J0K1L2M3N4O5P6Q", "classic AIza API key"),
        ("AQ.EXAMPLEexampleEXAMPLEexampleEXAMPLEexample1234", "AI Studio AQ. key"),
        ("ya29.a0ARrdaM9xKxKxKxKxKxKxKxKxKxKxKxKxKx", "OAuth access token"),
        ("GOCSPX-abcdefghijklmnopqrstuvwxyz", "OAuth client secret"),
        ("1//0abcdefghijklmnopqrstuvwxyz1234", "OAuth refresh token"),
    ],
)
def test_every_google_credential_shape_is_redacted(secret: str, what: str) -> None:
    """A credential must be caught wherever it appears, not only as NAME=value.

    The `AQ.` form was added after a key of that shape was seen passing through
    untouched in prose and in JSON: it matches none of the older patterns, so
    only the assignment rule caught it, and only when written as an assignment.

    Every value here is synthetic. Never put a real credential in a test — it
    ends up in git history, where redaction cannot reach it.
    """
    redactor = Redactor(environ={})
    for context in (
        f"GEMINI_API_KEY={secret}",
        f'{{"key": "{secret}"}}',
        f"the key is {secret} and it works",
        f"curl -H 'Authorization: Bearer {secret}'",
        secret,
    ):
        assert secret not in redactor.redact_text(context), f"{what} leaked in: {context[:40]}"
