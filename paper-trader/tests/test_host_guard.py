"""Tests for paper_trader.host_guard — the saturation predicate that decides
whether the live trader's Opus call is doomed before it spends 180 s.

These assert real logic, not just "it runs":
- /proc/meminfo parsing produces exact MB + swap-used %
- the predicate trips on EACH signal independently, with the right reason
- a probe failure (mem_available == 0) degrades to "not saturated" so a
  /proc read error never wedges the live trader
- recent_empty_rate computes the exact fraction off a real temp DB and
  returns a safe default when the DB is missing
"""
from __future__ import annotations

import sqlite3
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import host_guard as hg


def test_parse_meminfo_exact():
    text = (
        "MemTotal:       15584300 kB\n"
        "MemAvailable:    1048576 kB\n"   # exactly 1024 MB
        "SwapTotal:      10485760 kB\n"   # 10240 MB
        "SwapFree:        5242880 kB\n"   # 5120 MB → 50% used
    )
    info = hg._parse_meminfo(text)
    assert info["mem_available_mb"] == pytest.approx(1024.0, abs=0.1)
    assert info["swap_total_mb"] == pytest.approx(10240.0, abs=0.1)
    assert info["swap_used_pct"] == pytest.approx(50.0, abs=0.1)


def test_parse_meminfo_no_swap_is_zero_not_division_error():
    info = hg._parse_meminfo("MemAvailable: 2097152 kB\nSwapTotal: 0 kB\n")
    assert info["swap_used_pct"] == 0.0
    assert info["mem_available_mb"] == pytest.approx(2048.0, abs=0.1)


def _clear_probe():
    return {
        "opus_count": 1,
        "mem_available_mb": 8000.0,
        "swap_used_pct": 5.0,
        "load1": 2.0,
        "cpus": 8,
        "load_per_cpu": 0.25,
    }


def test_host_saturated_clear():
    sat, reason = hg.host_saturated(_probe=_clear_probe())
    assert sat is False
    assert reason == "host clear"


def test_host_saturated_trips_on_opus_only():
    p = _clear_probe()
    p["opus_count"] = 9
    sat, reason = hg.host_saturated(_probe=p, max_opus=4)
    assert sat is True
    assert "9 concurrent Opus" in reason


def test_host_saturated_trips_on_low_mem_only():
    p = _clear_probe()
    p["mem_available_mb"] = 120.0
    sat, reason = hg.host_saturated(_probe=p, min_mem_avail_mb=400)
    assert sat is True
    assert "avail" in reason


def test_host_saturated_trips_on_swap_only():
    p = _clear_probe()
    p["swap_used_pct"] = 97.0
    sat, reason = hg.host_saturated(_probe=p, max_swap_used_pct=90.0)
    assert sat is True
    assert "swap 97%" in reason


def test_host_saturated_trips_on_load_only():
    p = _clear_probe()
    p["load_per_cpu"] = 6.5
    sat, reason = hg.host_saturated(_probe=p, load_per_cpu=4.0)
    assert sat is True
    assert "load/cpu" in reason


def test_zero_mem_available_is_unknown_not_saturated():
    """A /proc read failure yields mem_available_mb == 0; that must NOT be
    read as 'zero memory left' and trip the guard, or a probe glitch would
    silently stop the live trader."""
    p = _clear_probe()
    p["mem_available_mb"] = 0.0
    sat, _ = hg.host_saturated(_probe=p, min_mem_avail_mb=400)
    assert sat is False


def test_recent_empty_rate_exact(tmp_path):
    db = tmp_path / "pt.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE decisions (id INTEGER PRIMARY KEY, "
        "action_taken TEXT, reasoning TEXT)"
    )
    rows = [
        ("NO_DECISION", "claude returned no response (timeout/empty)"),
        ("NO_DECISION", "claude returned no response (timeout/empty)"),
        ("NO_DECISION", "parse_failed: {garbage"),       # not empty/timeout
        ("BUY NVDA → FILLED", "strong momentum"),
        ("HOLD", "no edge"),
    ]
    conn.executemany(
        "INSERT INTO decisions (action_taken, reasoning) VALUES (?, ?)", rows
    )
    conn.commit()
    conn.close()

    out = hg.recent_empty_rate(db, limit=120)
    assert out["ok"] is True
    assert out["n"] == 5
    assert out["empty"] == 2          # only the two "claude returned no response"
    assert out["rate"] == pytest.approx(0.4, abs=1e-6)


def test_recent_empty_rate_missing_db_is_safe_default():
    out = hg.recent_empty_rate("/nonexistent/path/to/pt.db")
    assert out == {"n": 0, "empty": 0, "rate": 0.0, "ok": False}


def test_concurrent_opus_count_is_nonneg_int_and_never_raises():
    n = hg.concurrent_opus_count(marker="this-marker-matches-nothing-xyzzy")
    assert isinstance(n, int)
    assert n == 0


def test_snapshot_and_main_are_degrade_safe():
    snap = hg.snapshot(db_path="/nonexistent/pt.db")
    assert set(snap) == {"saturated", "reason", "probe", "recent_empty_rate"}
    assert isinstance(snap["saturated"], bool)
    rc = hg.main(["--json"])
    assert rc in (0, 1)
