"""Where connector definitions live.

A JSON file in the data directory, rather than the main config file, because
connectors are the one part of configuration a user is expected to edit *through
the application* — a UI adding a server should not have to rewrite `config.yaml`
and risk clobbering a hand-written comment.

The file holds no credentials. It names environment variables; the values are
read at connection time.
"""

from __future__ import annotations

import json
from pathlib import Path

from ..errors import ConfigurationError, StorageError
from .config import ConnectorCollection, ConnectorConfig

CONNECTORS_FILENAME = "connectors.json"


class ConnectorStore:
    """Loads and saves the connector collection."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path).expanduser()

    @classmethod
    def for_data_dir(cls, data_dir: Path) -> ConnectorStore:
        return cls(Path(data_dir).expanduser() / CONNECTORS_FILENAME)

    def load(self) -> ConnectorCollection:
        """Read the collection. A missing file is an empty collection, not an error."""
        if not self.path.is_file():
            return ConnectorCollection()
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ConfigurationError(
                f"{self.path} is not valid connector JSON: {exc}. "
                "Fix or delete the file to continue."
            ) from exc
        try:
            return ConnectorCollection.model_validate(raw)
        except Exception as exc:
            raise ConfigurationError(f"{self.path} contains an invalid connector: {exc}") from exc

    def save(self, collection: ConnectorCollection) -> None:
        """Write the collection atomically, so a crash cannot truncate it."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = collection.model_dump_json(indent=2)
        temporary = self.path.with_suffix(".json.tmp")
        try:
            temporary.write_text(payload + "\n", encoding="utf-8")
            temporary.replace(self.path)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise StorageError(f"could not save connectors to {self.path}: {exc}") from exc

    # -- convenience --------------------------------------------------------
    def add(self, connector: ConnectorConfig, *, replace: bool = False) -> ConnectorCollection:
        collection = self.load()
        collection.add(connector, replace=replace)
        self.save(collection)
        return collection

    def remove(self, name: str) -> bool:
        collection = self.load()
        removed = collection.remove(name)
        if removed:
            self.save(collection)
        return removed

    def set_enabled(self, name: str, enabled: bool) -> ConnectorConfig | None:
        collection = self.load()
        connector = collection.get(name)
        if connector is None:
            return None
        connector.enabled = enabled
        self.save(collection)
        return connector
