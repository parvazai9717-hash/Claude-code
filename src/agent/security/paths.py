"""Filesystem containment.

The configured workspace is a hard boundary. Every path a tool touches is
resolved through :class:`PathPolicy`, which rejects traversal, absolute escapes,
symlink escapes and credential-bearing files — before any I/O happens.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from ..errors import PathEscapeError, SecretProtectedError

SecretPathError = SecretProtectedError

#: Filenames that are always refused, wherever they sit inside the workspace.
PROTECTED_FILENAMES = frozenset(
    {
        ".env",
        ".envrc",
        ".netrc",
        "_netrc",
        ".pgpass",
        ".htpasswd",
        "credentials",
        "credentials.json",
        "client_secret.json",
        "service-account.json",
        "id_rsa",
        "id_dsa",
        "id_ecdsa",
        "id_ed25519",
        "known_hosts",
        "authorized_keys",
        "shadow",
        "passwd",
        ".git-credentials",
        ".npmrc",
        ".pypirc",
        ".dockercfg",
        "secrets.yaml",
        "secrets.yml",
        "secrets.json",
    }
)

#: Directory names that are always refused, at any depth.
PROTECTED_DIRECTORIES = frozenset(
    {
        ".ssh",
        ".gnupg",
        ".aws",
        ".azure",
        ".kube",
        ".docker",
        ".config/gcloud",
        ".gcloud",
        ".mozilla",
        ".pki",
        "keychains",
        "Keychains",
        # Browser profiles hold session cookies and saved passwords.
        "Chrome",
        "Chromium",
        "Firefox",
        "Safari",
        "BraveSoftware",
        ".local-agent",
    }
)

#: Filename suffixes that are always refused.
PROTECTED_SUFFIXES = frozenset(
    {".pem", ".key", ".pfx", ".p12", ".jks", ".keystore", ".asc", ".gpg"}
)

#: Filename prefixes that are always refused (covers `.env.local`, `.env.prod`, ...).
PROTECTED_PREFIXES = ("id_rsa", "id_ed25519", ".env")


@dataclass(frozen=True)
class PathPolicy:
    """Rules applied to every workspace path.

    Attributes:
        root: The workspace root. Resolved once, at construction.
        allow_hidden: Whether dotfiles may be listed or read at all.
        follow_symlinks: Whether a symlink may be traversed. Even when true, the
            *resolved* target must still fall inside the workspace.
        max_file_bytes: Refusal threshold for reads.
    """

    root: Path
    allow_hidden: bool = False
    follow_symlinks: bool = False
    max_file_bytes: int = 1_000_000
    extra_protected_names: frozenset[str] = field(default_factory=frozenset)

    @classmethod
    def for_workspace(cls, workspace: Path, **kwargs: object) -> PathPolicy:
        root = Path(workspace).expanduser().resolve()
        return cls(root=root, **kwargs)  # type: ignore[arg-type]

    # -- protection checks --------------------------------------------------
    def _is_protected_name(self, name: str) -> bool:
        lowered = name.lower()
        if name in PROTECTED_FILENAMES or lowered in PROTECTED_FILENAMES:
            return True
        if name in self.extra_protected_names:
            return True
        if any(lowered.endswith(suffix) for suffix in PROTECTED_SUFFIXES):
            return True
        return any(lowered.startswith(prefix.lower()) for prefix in PROTECTED_PREFIXES)

    def is_protected(self, path: Path) -> tuple[bool, str]:
        """Return `(protected, reason)` for an already-resolved path."""
        try:
            relative = path.relative_to(self.root)
        except ValueError:
            return True, "path is outside the workspace"
        parts = relative.parts
        for part in parts[:-1] if parts else ():
            if part in PROTECTED_DIRECTORIES:
                return True, f"'{part}' is a protected credential directory"
        if parts and parts[-1] in PROTECTED_DIRECTORIES:
            return True, f"'{parts[-1]}' is a protected credential directory"
        if parts and self._is_protected_name(parts[-1]):
            return True, f"'{parts[-1]}' is a protected credential file"
        return False, ""

    def is_hidden(self, path: Path) -> bool:
        try:
            relative = path.relative_to(self.root)
        except ValueError:
            return False
        return any(part.startswith(".") for part in relative.parts)

    # -- resolution ---------------------------------------------------------
    def resolve(self, candidate: str | Path, *, must_exist: bool = False) -> Path:
        """Resolve a tool-supplied path inside the workspace.

        Args:
            candidate: A workspace-relative path. Absolute paths are accepted only
                when they already point inside the workspace.
            must_exist: Raise :class:`FileNotFoundError` when the target is absent.

        Returns:
            The fully resolved, contained path.

        Raises:
            PathEscapeError: On traversal, an absolute path outside the workspace,
                or a symlink whose target leaves the workspace.
            SecretProtectedError: When the path names a credential file/directory.
            FileNotFoundError: When `must_exist` is set and nothing is there.
        """
        raw = str(candidate).strip()
        if not raw:
            raise PathEscapeError("an empty path is not a valid workspace location")
        if "\x00" in raw:
            raise PathEscapeError("path contains a null byte")

        supplied = Path(raw)
        if supplied.is_absolute():
            target = supplied
        else:
            if raw.startswith("~"):
                # `~` would escape to the user's home directory.
                raise PathEscapeError(
                    "home-relative paths are not allowed; use a workspace-relative path"
                )
            target = self.root / supplied

        # `os.path.normpath` collapses `..` textually; resolve() then follows links.
        normalized = Path(os.path.normpath(str(target)))
        resolved = normalized.resolve()

        if not self._contains(resolved):
            raise PathEscapeError(
                f"path escapes the workspace boundary: {raw!r}",
                details={"workspace": str(self.root)},
            )

        # A symlink is only followed when explicitly permitted, and only if its
        # resolved target is also inside the workspace (checked above).
        if not self.follow_symlinks and self._has_symlink(normalized):
            raise PathEscapeError(
                f"path traverses a symbolic link, which is not permitted: {raw!r}"
            )

        protected, reason = self.is_protected(resolved)
        if protected:
            raise SecretProtectedError(f"refusing to touch {raw!r}: {reason}")

        if not self.allow_hidden and self.is_hidden(resolved):
            raise SecretProtectedError(
                f"refusing to touch hidden path {raw!r}; hidden files are excluded by policy"
            )

        if must_exist and not resolved.exists():
            raise FileNotFoundError(f"no such path in the workspace: {raw}")

        return resolved

    def _contains(self, resolved: Path) -> bool:
        try:
            resolved.relative_to(self.root)
        except ValueError:
            return False
        return True

    def _has_symlink(self, path: Path) -> bool:
        """True when `path` or any parent inside the workspace is a symlink."""
        current = path
        while True:
            if current.is_symlink():
                return True
            if current == self.root or current == current.parent:
                return False
            current = current.parent

    def relative(self, path: Path) -> str:
        """Workspace-relative display form. Never leaks the absolute prefix."""
        try:
            return str(path.resolve().relative_to(self.root))
        except ValueError:
            return path.name


@dataclass(frozen=True)
class WorkspacePaths:
    """The standard workspace layout."""

    root: Path

    @property
    def files(self) -> Path:
        return self.root / "files"

    @property
    def projects(self) -> Path:
        return self.root / "projects"

    @property
    def downloads(self) -> Path:
        return self.root / "downloads"

    @property
    def outputs(self) -> Path:
        return self.root / "outputs"

    @property
    def temp(self) -> Path:
        return self.root / "temp"

    @property
    def state(self) -> Path:
        return self.root / "state"
