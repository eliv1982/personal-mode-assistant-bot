from __future__ import annotations

import json
import logging
import os
import tempfile
import time
from typing import Any, TypeVar
from uuid import uuid4

logger = logging.getLogger(__name__)

T = TypeVar("T")


class JsonWriteError(RuntimeError):
    """Raised when an atomic JSON write could not be completed."""


def atomic_write_json(path: str, data: Any) -> None:
    """Write `data` as JSON to `path` atomically.

    Writes a complete temporary file in the same directory, then replaces
    the target file in a single filesystem operation so a reader never
    observes a partially written file. Raises JsonWriteError (and cleans
    up the temp file) instead of silently reporting success on failure.
    """
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)

    fd, tmp_path = tempfile.mkstemp(prefix=".tmp-", suffix=".json", dir=directory)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, ensure_ascii=False, indent=2)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_path, path)
    except Exception as exc:
        try:
            os.remove(tmp_path)
        except OSError:
            pass
        logger.error("Atomic write failed for '%s' (%s: %s).", path, type(exc).__name__, exc)
        raise JsonWriteError(f"Failed to write '{path}'") from exc


def load_json_with_quarantine(path: str, expected_type: type, default: T) -> T:
    """Load JSON from `path`, validating its top-level type.

    Missing file -> `default`. Unparseable or wrong-shaped file -> the
    damaged file is moved aside to a timestamped `.corrupt` quarantine
    path (never silently overwritten or discarded) and `default` is
    returned. The incident is logged with the path and exception type
    only -- never the file's contents.
    """
    if not os.path.exists(path):
        return default

    try:
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, expected_type):
            raise ValueError(f"Expected top-level JSON {expected_type.__name__}, got {type(data).__name__}")
        return data
    except Exception as exc:
        logger.error(
            "Corrupt JSON detected in '%s' (%s: %s). Quarantining file and starting fresh.",
            path, type(exc).__name__, exc,
        )
        _quarantine(path)
        return default


def _quarantine(path: str) -> None:
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    quarantine_path = f"{path}.corrupt.{timestamp}-{uuid4().hex[:6]}"
    try:
        os.replace(path, quarantine_path)
        logger.error("Quarantined corrupt file '%s' -> '%s'.", path, quarantine_path)
    except OSError as exc:
        logger.error("Failed to quarantine corrupt file '%s': %s", path, exc)
