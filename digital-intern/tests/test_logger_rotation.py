"""Structured JSONL log handler: valid JSON output + size-based rotation.

Regression guard for the unbounded-growth bug — structured.jsonl had reached
72MB / ~400k lines with no rotation, polluting log audits and disk.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

from core.logger import _JSONLHandler


def _make_record(msg: str, **extra) -> logging.LogRecord:
    rec = logging.LogRecord(
        name="test", level=logging.INFO, pathname=__file__, lineno=1,
        msg=msg, args=(), exc_info=None,
    )
    for k, v in extra.items():
        setattr(rec, k, v)
    return rec


def test_emits_valid_jsonl(tmp_path: Path):
    log_path = tmp_path / "structured.jsonl"
    h = _JSONLHandler(log_path)
    try:
        h.emit(_make_record("hello world", worker="rss", count=7))
    finally:
        h.close()

    lines = log_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    obj = json.loads(lines[0])
    assert obj["msg"] == "hello world"
    assert obj["level"] == "INFO"
    assert obj["logger"] == "test"
    assert obj["worker"] == "rss"      # extra fields preserved
    assert obj["count"] == 7


def test_rolls_over_when_exceeding_max_bytes(tmp_path: Path):
    log_path = tmp_path / "structured.jsonl"
    h = _JSONLHandler(log_path)
    # Shrink the threshold so the test stays fast instead of writing 10MB.
    h.maxBytes = 2000
    try:
        big = "x" * 500
        for _ in range(50):
            h.emit(_make_record(big))
    finally:
        h.close()

    # Rollover must have produced at least one backup file.
    backup = Path(str(log_path) + ".1")
    assert backup.exists(), "expected rotated backup structured.jsonl.1"
    # Active file stays bounded (well under the 50 * ~520 bytes written).
    assert log_path.stat().st_size <= h.maxBytes + 600
    # Backups are capped at backupCount; no .8 should ever appear.
    assert not Path(str(log_path) + ".8").exists()
