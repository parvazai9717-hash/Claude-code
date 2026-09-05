"""Browser capability boundary.

This module defines the *shape* a future browser adapter must take. It does not
implement browser control, and the presence of this interface does not make
browser automation available: :class:`~agent.browser.unavailable.UnavailableBrowser`
is the only implementation shipped, and it refuses every action.

Security requirements any future implementation must satisfy before it is enabled:

- an **isolated browser profile**, never the user's real profile (which holds
  session cookies and saved passwords);
- a **domain allowlist**, checked before navigation, not after;
- a dedicated **download directory** inside the workspace;
- **upload restrictions** — only files the user explicitly selected;
- **navigation and action timeouts** with process cleanup;
- **redacted page data**: form values, cookies, storage and headers never reach
  the model;
- **explicit approval** for login, purchases, messages, form submissions, account
  changes and anything involving personal information;
- **human takeover** for CAPTCHA, authentication, payment and private data.

Unrestricted browser debugging protocols (CDP on an open port, a real user
profile, or `--remote-debugging-port` against an existing session) must never be
used: they hand over every credential the browser holds.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any


class BrowserAction(StrEnum):
    """The normalized capability set a future adapter must cover."""

    OPEN_URL = "open_url"
    INSPECT_PAGE = "inspect_page"
    CLICK = "click"
    TYPE = "type"
    PRESS_KEY = "press_key"
    SCROLL = "scroll"
    SELECT = "select"
    SCREENSHOT = "screenshot"
    DOWNLOAD = "download"
    UPLOAD = "upload"
    CLOSE = "close"


#: Actions that must always require human approval, in any implementation.
APPROVAL_REQUIRED_ACTIONS: frozenset[BrowserAction] = frozenset(
    {
        BrowserAction.CLICK,
        BrowserAction.TYPE,
        BrowserAction.PRESS_KEY,
        BrowserAction.SELECT,
        BrowserAction.DOWNLOAD,
        BrowserAction.UPLOAD,
    }
)


@dataclass(frozen=True)
class BrowserPolicy:
    """The policy a future adapter must be constructed with."""

    #: Domains that may be navigated to. Empty means nothing is allowed.
    allowed_domains: frozenset[str] = field(default_factory=frozenset)
    #: Workspace-relative directory for downloads.
    download_dir: str = "downloads"
    #: Whether uploading files is permitted at all.
    allow_uploads: bool = False
    navigation_timeout_seconds: float = 30.0
    action_timeout_seconds: float = 10.0
    #: An isolated profile directory. Never the user's real browser profile.
    profile_dir: str | None = None

    def allows(self, domain: str) -> bool:
        candidate = domain.lower().removeprefix("www.")
        return any(
            candidate == allowed or candidate.endswith(f".{allowed}")
            for allowed in self.allowed_domains
        )


@dataclass(frozen=True)
class BrowserResult:
    """A normalized, already-redacted result of one browser action."""

    action: BrowserAction
    ok: bool
    detail: str = ""
    data: dict[str, Any] = field(default_factory=dict)


class BrowserProvider(ABC):
    """The interface a future browser adapter must implement."""

    name: str = "browser"

    @abstractmethod
    async def is_available(self) -> bool:
        """Whether browser control can actually be performed right now."""

    @abstractmethod
    async def perform(self, action: BrowserAction, arguments: dict[str, Any]) -> BrowserResult:
        """Perform one normalized action, or return an unsuccessful result."""

    @abstractmethod
    async def aclose(self) -> None:
        """Shut down the browser and clean up its processes."""
