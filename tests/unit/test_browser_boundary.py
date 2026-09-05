"""The browser interface exists; the capability does not."""

from __future__ import annotations

import pytest

from agent.browser import (
    APPROVAL_REQUIRED_ACTIONS,
    BrowserAction,
    BrowserPolicy,
    UnavailableBrowser,
    get_browser,
)
from agent.errors import CapabilityUnavailableError


async def test_the_default_browser_is_unavailable() -> None:
    browser = get_browser()
    assert isinstance(browser, UnavailableBrowser)
    assert await browser.is_available() is False


@pytest.mark.parametrize("action", list(BrowserAction))
async def test_every_action_is_refused(action: BrowserAction) -> None:
    result = await UnavailableBrowser().perform(action, {})
    assert result.ok is False
    assert "not enabled" in result.detail
    assert result.data["enabled"] is False


async def test_raising_variant() -> None:
    with pytest.raises(CapabilityUnavailableError):
        await UnavailableBrowser().perform_or_raise(BrowserAction.OPEN_URL, {"url": "http://x"})


def test_the_normalized_capability_set_is_complete() -> None:
    expected = {
        "open_url",
        "inspect_page",
        "click",
        "type",
        "press_key",
        "scroll",
        "select",
        "screenshot",
        "download",
        "upload",
        "close",
    }
    assert {a.value for a in BrowserAction} == expected


def test_interactive_actions_are_marked_as_needing_approval() -> None:
    for action in (BrowserAction.CLICK, BrowserAction.TYPE, BrowserAction.UPLOAD):
        assert action in APPROVAL_REQUIRED_ACTIONS


def test_an_empty_policy_allows_no_domain() -> None:
    """Deny-by-default: a policy with no allowlist permits nothing."""
    assert BrowserPolicy().allows("example.com") is False


def test_policy_allows_only_listed_domains() -> None:
    policy = BrowserPolicy(allowed_domains=frozenset({"example.com"}))
    assert policy.allows("example.com") is True
    assert policy.allows("www.example.com") is True
    assert policy.allows("docs.example.com") is True
    assert policy.allows("evil.com") is False
    assert policy.allows("notexample.com") is False


def test_uploads_are_disabled_by_default() -> None:
    assert BrowserPolicy().allow_uploads is False


async def test_closing_is_safe() -> None:
    await UnavailableBrowser().aclose()
