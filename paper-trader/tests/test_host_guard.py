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


# ── recent_starvation_rate — must count BOTH prefixes ────────────────────────
# This is the discriminating regression: `recent_empty_rate` counts only the
# old "claude returned no response" prefix, so once the pre-flight guard is
# live (writing "skipped claude call …") it under-reports a storm. This test
# fails if recent_starvation_rate ever narrows back to the empty-only prefix.

def test_recent_starvation_rate_counts_both_prefixes(tmp_path):
    db = tmp_path / "pt.db"
    conn = sqlite3.connect(db)
    conn.execute(
        "CREATE TABLE decisions (id INTEGER PRIMARY KEY, "
        "action_taken TEXT, reasoning TEXT)"
    )
    rows = [
        ("NO_DECISION", "claude returned no response (timeout/empty)"),  # old
        ("NO_DECISION", "skipped claude call — host saturated: 7 conc"),  # new
        ("NO_DECISION", "skipped claude call — host saturated mid-call"),  # new
        ("NO_DECISION", "parse_failed: {garbage"),    # NOT starvation
        ("HOLD", "no edge"),                          # NOT NO_DECISION
        ("BUY NVDA → FILLED", "skipped claude call"),  # NOT NO_DECISION
    ]
    conn.executemany(
        "INSERT INTO decisions (action_taken, reasoning) VALUES (?, ?)", rows
    )
    conn.commit()
    conn.close()

    out = hg.recent_starvation_rate(db, limit=120)
    assert out["ok"] is True
    assert out["n"] == 6
    # 1 old-prefix + 2 new-prefix NO_DECISION = 3 (parse_failed and the two
    # non-NO_DECISION rows excluded). recent_empty_rate would see only 1.
    assert out["starved"] == 3
    assert out["rate"] == pytest.approx(0.5, abs=1e-6)
    assert hg.recent_empty_rate(db)["empty"] == 1  # proves the divergence


def test_recent_starvation_rate_missing_db_is_safe_default():
    out = hg.recent_starvation_rate("/nonexistent/path/to/pt.db")
    assert out == {"n": 0, "starved": 0, "rate": 0.0, "ok": False}


# ── pulse() — the operator-facing freeze-cause SSOT ──────────────────────────

def _snap(saturated, reason, opus=1):
    return {"saturated": saturated, "reason": reason,
            "probe": {"opus_count": opus}, "recent_empty_rate": {}}


def _starv(rate, n=120, ok=True):
    return {"n": n, "starved": int(round(rate * n)), "rate": rate, "ok": ok}


class TestPulse:
    def test_saturated_wins_regardless_of_log_rate(self):
        # Probe trips now → SATURATED even if the log rate is 0 (the storm
        # only just started; no damage recorded yet).
        out = hg.pulse(
            _snapshot=_snap(True, "host saturated: 7 concurrent Opus (>4)",
                            opus=7),
            _starv=_starv(0.0))
        assert out["state"] == "SATURATED"
        # reason carried VERBATIM into the headline (no-drift lock).
        assert "host saturated: 7 concurrent Opus (>4)" in out["headline"]
        assert out["opus_count"] == 7
        assert hg._OPS_ACTION in out["headline"]

    def test_starved_when_probe_clear_but_log_rate_high(self):
        out = hg.pulse(
            _snapshot=_snap(False, "host clear"),
            _starv=_starv(0.40))
        assert out["state"] == "STARVED"
        assert "40" in out["headline"] and "never reached" in out["headline"]
        assert hg._OPS_ACTION in out["headline"]

    def test_clear_when_probe_clear_and_rate_below_floor(self):
        out = hg.pulse(
            _snapshot=_snap(False, "host clear"),
            _starv=_starv(hg.STARVATION_RATE_FLOOR - 0.01))
        assert out["state"] == "CLEAR"
        assert out["headline"] == ""

    def test_floor_boundary_is_inclusive(self):
        # rate == floor → STARVED (>= comparison, locked).
        out = hg.pulse(
            _snapshot=_snap(False, "host clear"),
            _starv=_starv(hg.STARVATION_RATE_FLOOR))
        assert out["state"] == "STARVED"

    def test_unreadable_log_never_cries_wolf(self):
        # Probe clear + starvation probe failed (DB unreadable) → CLEAR, even
        # though rate reads as 0.0 it must not be trusted as "high" nor as a
        # false STARVED. ok=False is the gate.
        out = hg.pulse(
            _snapshot=_snap(False, "host clear"),
            _starv={"n": 0, "starved": 0, "rate": 0.0, "ok": False})
        assert out["state"] == "CLEAR"
        assert out["headline"] == ""

    def test_saturated_survives_unreadable_log(self):
        # A live probe trip must still report SATURATED even when the decision
        # log is unreadable (the most important case: dashboard/DB wedged).
        out = hg.pulse(
            _snapshot=_snap(True, "host saturated: swap 95% (>90%)"),
            _starv={"n": 0, "starved": 0, "rate": 0.0, "ok": False})
        assert out["state"] == "SATURATED"
        assert "swap 95% (>90%)" in out["headline"]

    def test_degrade_safe_never_raises(self):
        class Boom(dict):
            def get(self, *a, **k):
                raise RuntimeError("probe blew up")

        out = hg.pulse(_snapshot=Boom(), _starv=_starv(0.0))
        assert out["state"] == "CLEAR"
        assert out["headline"] == ""

    def test_real_path_is_inert_under_pytest(self):
        # No injectors + running under pytest → deterministic, offline CLEAR
        # (the _swr_active offline-invariant precedent). This is what the
        # broad reporter integration tests rely on so a saturated CI box
        # can't inject a host-state-dependent line into asserted bodies.
        out = hg.pulse(db_path="/nonexistent/pt.db")
        assert set(out) == {
            "state", "headline", "saturated", "reason",
            "starvation_rate_pct", "starvation_n", "starvation_ok",
            "opus_count"}
        assert out["state"] == "CLEAR"
        assert out["headline"] == ""

    def test_force_flag_overrides_pytest_inertness(self, monkeypatch):
        # A dedicated test can opt back into the real path via the force
        # flag (the _SWR_TEST_FORCE precedent); with a nonexistent DB the
        # probe still degrades safely and never raises.
        monkeypatch.setattr(hg, "_PULSE_TEST_FORCE", True)
        out = hg.pulse(db_path="/nonexistent/pt.db")
        assert out["state"] in ("CLEAR", "STARVED", "SATURATED")
        assert out["starvation_ok"] is False  # nonexistent DB → safe default


# ── CLI text — must surface the BROADER starvation rate ──────────────────────
# `recent_empty_rate` counts the OLD "claude returned no response" prefix only;
# once the pre-flight host_saturated guard is live, storms produce mostly
# "skipped claude call" rows that bucket silently misses. The CLI must show
# the broader starvation figure (which counts BOTH prefixes) so the operator
# isn't given a misleadingly small empty-only percentage — the very blind spot
# `recent_starvation_rate` was added to address.

class TestMainCLI:
    def _seed_db(self, tmp_path, rows):
        db = tmp_path / "pt.db"
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE decisions (id INTEGER PRIMARY KEY, "
            "action_taken TEXT, reasoning TEXT)"
        )
        conn.executemany(
            "INSERT INTO decisions (action_taken, reasoning) VALUES (?, ?)",
            rows,
        )
        conn.commit()
        conn.close()
        return db

    def test_cli_human_text_includes_starvation_rate(
            self, tmp_path, capsys, monkeypatch):
        # 2 empty + 2 skipped + 1 parse_failed (not starved) + 1 filled
        rows = [
            ("NO_DECISION", "claude returned no response (timeout/empty)"),
            ("NO_DECISION", "claude returned no response (nonzero_rc)"),
            ("NO_DECISION", "skipped claude call — host saturated: 5 ..."),
            ("NO_DECISION", "skipped claude call — host saturated: 6 ..."),
            ("NO_DECISION", "parse_failed: {garbage"),
            ("BUY NVDA → FILLED", "ok"),
        ]
        db = self._seed_db(tmp_path, rows)
        monkeypatch.setattr(hg, "_DEFAULT_DB", db)
        # Inert /proc probe so the CLI's verdict text is deterministic.
        monkeypatch.setattr(hg, "probe", lambda: {
            "opus_count": 1, "mem_available_mb": 8000.0,
            "swap_used_pct": 5.0, "load1": 1.0, "load5": 1.0,
            "load15": 1.0, "cpus": 16, "load_per_cpu": 0.1,
        })
        rc = hg.main([])
        out = capsys.readouterr().out
        # 4 starved out of 6 = 67%. Both legs (2 empty + 2 skipped) shown.
        assert "4/6 recent decisions never reached Opus (67%)" in out
        assert "2 empty/timeout + 2 skipped (host guard)" in out
        assert rc == 0  # not saturated by the inert probe

    def test_cli_json_payload_includes_starvation_rate(
            self, tmp_path, capsys, monkeypatch):
        rows = [
            ("NO_DECISION", "claude returned no response (timeout/empty)"),
            ("NO_DECISION", "skipped claude call — host saturated: 5 ..."),
        ]
        db = self._seed_db(tmp_path, rows)
        monkeypatch.setattr(hg, "_DEFAULT_DB", db)
        monkeypatch.setattr(hg, "probe", lambda: {
            "opus_count": 1, "mem_available_mb": 8000.0,
            "swap_used_pct": 5.0, "load1": 1.0, "load5": 1.0,
            "load15": 1.0, "cpus": 16, "load_per_cpu": 0.1,
        })
        import json
        hg.main(["--json"])
        out = capsys.readouterr().out
        payload = json.loads(out)
        # Empty-rate bucket is the original snapshot key; starvation-rate
        # is the additive new key — both must be present so a JSON consumer
        # sees the broader figure too.
        assert "recent_empty_rate" in payload
        assert "recent_starvation_rate" in payload
        assert payload["recent_starvation_rate"]["starved"] == 2
        assert payload["recent_empty_rate"]["empty"] == 1


# ── _NOT_TICKERS — no duplicate tokens ───────────────────────────────────────
# The set deduplicates at runtime but duplicate literals in the source code
# are a maintenance smell: a future editor adding a new token may not notice
# that a "NEW" was already present elsewhere. Test pins the invariant.

def test_not_tickers_has_no_duplicate_source_tokens():
    import re
    text = Path(_ROOT, "paper_trader", "signals.py").read_text()
    m = re.search(r"_NOT_TICKERS = \{(.*?)^\}", text, re.M | re.S)
    assert m is not None, "could not locate _NOT_TICKERS literal"
    tokens = re.findall(r'"([A-Z][A-Z0-9_]*)"', m.group(1))
    duplicates = sorted(t for t in set(tokens) if tokens.count(t) > 1)
    assert duplicates == [], f"_NOT_TICKERS has duplicate tokens: {duplicates}"
