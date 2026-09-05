"""The only browser implementation shipped in the first stable release.

Every action returns a structured capability error. This exists so the runtime,
the CLI and the tests can all depend on the browser interface while the
capability itself stays firmly off.
"""

from __future__ import annotations

from typing import Any

from ..errors import CapabilityUnavailableError
from .base import BrowserAction, BrowserProvider, BrowserResult

UNAVAILABLE_MESSAGE = (
    "Browser automation is not enabled in this release. The interface exists so an "
    "adapter can be added later, but no browser is launched and no page can be "
    "reached. See ARCHITECTURE.md ('Browser boundary') for the security "
    "requirements a future adapter must satisfy first."
)


class UnavailableBrowser(BrowserProvider):
    """Refuses every browser action, with a clear reason."""

    name = "unavailable"

    async def is_available(self) -> bool:
        return False

    async def perform(self, action: BrowserAction, arguments: dict[str, Any]) -> BrowserResult:
        return BrowserResult(
            action=action,
            ok=False,
            detail=UNAVAILABLE_MESSAGE,
            data={"capability": action.value, "enabled": False},
        )

    async def perform_or_raise(
        self, action: BrowserAction, arguments: dict[str, Any]
    ) -> BrowserResult:
        """Variant that raises, for callers that treat unavailability as an error."""
        raise CapabilityUnavailableError(UNAVAILABLE_MESSAGE, details={"capability": action.value})

    async def aclose(self) -> None:
        return None
