"""Regression guard for the healthcheck.sh heartbeat watchdog.

Bug: step 4 grepped "[heartbeat] sent" only from the current
structured.jsonl. That event is infrequent (~hourly), so one size-rotation
left the current file with no marker, blanking HB_AGE_H to "?" AND silently
disabling the >6h stale-scorer alert (it was gated on a non-empty ts).

These tests extract the real python heredoc from healthcheck.sh and run it,
so they fail if the rotation-aware scan ever regresses.
"""
import re
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

HEALTHCHECK = Path(__file__).resolve().parent.parent / "healthcheck.sh"


def _extract_snippet() -> str:
    src = HEALTHCHECK.read_text()
    m = re.search(r"<<'PY'[^\n]*\n(.*?)\nPY\b", src, re.DOTALL)
    assert m, "could not locate the heartbeat python heredoc in healthcheck.sh"
    return m.group(1)


def _run(snippet: str, *paths: Path) -> str:
    out = subprocess.run(
        [sys.executable, "-", *map(str, paths)],
        input=snippet,
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert out.returncode == 0, out.stderr
    return out.stdout.strip()


def _hb_line(hours_ago: float) -> str:
    ts = (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()
    return (
        '{"ts": "%s", "level": "INFO", "logger": "daemon", '
        '"msg": "[heartbeat] sent (1987 chars)", "levelno": 20}\n' % ts
    )


def test_marker_only_in_rotated_backup(tmp_path):
    """The exact regression: current file has no marker, .1 does."""
    snippet = _extract_snippet()
    current = tmp_path / "structured.jsonl"
    backup = tmp_path / "structured.jsonl.1"
    current.write_text('{"ts": "x", "msg": "[scorer] alive"}\n')
    backup.write_text(_hb_line(2.0))

    result = _run(snippet, current, backup)
    assert result != "?", "rotation must not blank the heartbeat age"
    assert 1.8 < float(result) < 2.2, result


def test_latest_wins_across_files(tmp_path):
    snippet = _extract_snippet()
    current = tmp_path / "structured.jsonl"
    backup = tmp_path / "structured.jsonl.1"
    current.write_text(_hb_line(0.5))
    backup.write_text(_hb_line(9.0))

    result = _run(snippet, current, backup)
    assert 0.3 < float(result) < 0.7, result


def test_no_marker_anywhere_yields_question(tmp_path):
    snippet = _extract_snippet()
    current = tmp_path / "structured.jsonl"
    current.write_text('{"ts": "x", "msg": "[scorer] alive"}\n')

    result = _run(snippet, current, tmp_path / "structured.jsonl.1")
    assert result == "?", result


def test_missing_backup_is_tolerated(tmp_path):
    snippet = _extract_snippet()
    current = tmp_path / "structured.jsonl"
    current.write_text(_hb_line(1.0))

    result = _run(snippet, current, tmp_path / "does-not-exist.1")
    assert 0.8 < float(result) < 1.2, result
