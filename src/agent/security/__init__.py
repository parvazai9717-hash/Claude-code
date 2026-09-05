"""Security layer.

Everything in this package is enforced by the runtime, in code. The system
prompt is not a security boundary: a model may *request* anything, and only the
modules here decide what is allowed to happen.
"""

from .approvals import (
    ApprovalDecision,
    ApprovalRequest,
    ApprovalResponse,
    Approver,
    AutoDenyApprover,
    ConsoleApprover,
    PolicyApprover,
    UnsafeAutoApprover,
)
from .limits import LimitTracker
from .paths import PathPolicy, SecretPathError, WorkspacePaths
from .permissions import PermissionChecker, PermissionDecision
from .redaction import Redactor, redact

__all__ = [
    "ApprovalDecision",
    "ApprovalRequest",
    "ApprovalResponse",
    "Approver",
    "AutoDenyApprover",
    "ConsoleApprover",
    "LimitTracker",
    "PathPolicy",
    "PermissionChecker",
    "PermissionDecision",
    "PolicyApprover",
    "Redactor",
    "SecretPathError",
    "UnsafeAutoApprover",
    "WorkspacePaths",
    "redact",
]
