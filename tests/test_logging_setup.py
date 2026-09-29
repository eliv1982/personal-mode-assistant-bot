from __future__ import annotations

import logging
import logging.handlers
import os

import main


def test_rotating_file_handler_configured_with_bounded_backups(tmp_path, monkeypatch):
    """`_setup_logging()` must wire up a RotatingFileHandler (bounded size,
    bounded backup count) rather than an unbounded plain FileHandler.

    Root-logger state is a process-global side effect, so this test saves
    and restores it around the call to avoid leaking into other tests.
    """
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level

    for h in list(root.handlers):
        root.removeHandler(h)

    monkeypatch.chdir(tmp_path)

    try:
        main._setup_logging()

        rotating_handlers = [
            h for h in root.handlers if isinstance(h, logging.handlers.RotatingFileHandler)
        ]
        assert len(rotating_handlers) == 1

        handler = rotating_handlers[0]
        assert handler.maxBytes == main._LOG_MAX_BYTES
        assert handler.backupCount == main._LOG_BACKUP_COUNT
        # Bounded means both values are finite and strictly positive --
        # 0 would mean "unbounded" for maxBytes, and 0 backups would mean
        # no rotation history is kept at all.
        assert handler.maxBytes > 0
        assert handler.backupCount > 0

        assert os.path.isdir(tmp_path / "logs")
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
            h.close()
        for h in original_handlers:
            root.addHandler(h)
        root.setLevel(original_level)
