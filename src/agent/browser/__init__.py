"""Browser boundary. The interface is defined; the capability is not enabled."""

from .base import (
    APPROVAL_REQUIRED_ACTIONS,
    BrowserAction,
    BrowserPolicy,
    BrowserProvider,
    BrowserResult,
)
from .unavailable import UNAVAILABLE_MESSAGE, UnavailableBrowser

__all__ = [
    "APPROVAL_REQUIRED_ACTIONS",
    "UNAVAILABLE_MESSAGE",
    "BrowserAction",
    "BrowserPolicy",
    "BrowserProvider",
    "BrowserResult",
    "UnavailableBrowser",
]


def get_browser() -> BrowserProvider:
    """Return the configured browser provider.

    Always the unavailable implementation in this release. A future adapter must
    be selected explicitly here, never inferred from an installed package.
    """
    return UnavailableBrowser()
