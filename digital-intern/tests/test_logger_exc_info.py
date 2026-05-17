"""_JSONLHandler must serialise exc_info, not crash on it.

Regression guard for a production bug invisible to the 9-file inspection
review (``core/logger.py`` is not in that list): ``_JSONLHandler`` subclasses
``RotatingFileHandler`` — i.e. a ``logging.Handler`` — but its ``format()``
called ``self.formatException(record.exc_info)``. ``formatException`` is a
``logging.Formatter`` method, never present on a ``Handler``, so EVERY record
carrying ``exc_info`` (every ``log.exception(...)`` / ``log.error(...,
exc_info=True)`` in the daemon) raised ``AttributeError`` inside
``emit() -> shouldRollover() -> format()``.

``format()`` raises *before* ``emit`` writes anything, so the whole structured
record was lost — not just the ``exc`` field. Observed live: every
``[urgency] Scoring error`` exception (each one a dropped Sonnet batch) was
absent from ``structured.jsonl`` while ``logging`` spammed a secondary
``--- Logging error ---`` traceback into ``daemon.log`` (48 in one window).
The dashboard / healthcheck read ``structured.jsonl``; error visibility was
broken precisely when errors happened.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from core.logger import _JSONLHandler


def _exc_record(msg: str) -> logging.LogRecord:
    try:
        raise ValueError("boom-sentinel-42")
    except ValueError:
        ei = sys.exc_info()
    return logging.LogRecord(
        name="urgency_scorer", level=logging.ERROR, pathname=__file__,
        lineno=1, msg=msg, args=(), exc_info=ei,
    )


def test_exc_info_record_is_written_not_dropped(tmp_path: Path):
    """The whole record must reach the file — format() used to raise before
    emit could write it, so the structured error stream lost every exception."""
    log_path = tmp_path / "structured.jsonl"
    h = _JSONLHandler(log_path)
    try:
        h.emit(_exc_record("[urgency] Scoring error"))
    finally:
        h.close()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1, "exception log record was dropped from JSONL sink"
    obj = json.loads(lines[0])
    assert obj["msg"] == "[urgency] Scoring error"
    assert obj["level"] == "ERROR"
    # The original exception type + message must be captured for forensics.
    assert "exc" in obj, "exc_info was not serialised"
    assert "ValueError" in obj["exc"]
    assert "boom-sentinel-42" in obj["exc"]
    assert "Traceback" in obj["exc"]


def test_format_does_not_raise_attributeerror_on_exc_record(tmp_path: Path):
    """Direct format() must return valid JSON, never AttributeError."""
    h = _JSONLHandler(tmp_path / "s.jsonl")
    try:
        line = h.format(_exc_record("x"))
    finally:
        h.close()
    obj = json.loads(line)
    assert "ValueError" in obj["exc"] and "boom-sentinel-42" in obj["exc"]
