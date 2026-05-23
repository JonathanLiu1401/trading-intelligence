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


# ── recent_starvation_by_cause — operator-actionable bucketing ──────────────
# Each class needs a DIFFERENT action (see _CAUSE_LABELS comment):
#   model_timeout  — Opus wedged; may resolve when load drops
#   cli_nonzero_rc — Anthropic API / CLI transient
#   model_empty    — model-level miss; usually one-cycle
#   cli_missing    — config bug (claude not in PATH)
#   host_skip      — pre-flight guard skipped; ops must reduce parallel Opus
#   unknown        — legacy/future reason; surfaces so sum reconciles
# The aggregate starvation rate hides this cell, so the breakdown is the
# minimum operator-visible thing the CLI must emit during a storm.

class TestStarvationByCause:
    def test_classify_each_known_cause(self):
        c = hg._classify_starvation_cause
        # Host skip prefix wins regardless of an embedded "(timeout)" later.
        assert c("skipped claude call — host saturated: 7 conc") == "host_skip"
        assert c("skipped claude call — host saturated mid-call: ...") == "host_skip"
        # The per-cause sub-buckets strategy.py writes.
        assert c("claude returned no response (timeout)") == "model_timeout"
        assert c("claude returned no response (nonzero_rc)") == "cli_nonzero_rc"
        assert c("claude returned no response (empty_stdout)") == "model_empty"
        assert c("claude returned no response (exception)") == "model_empty"
        assert c("claude returned no response (cli_missing)") == "cli_missing"
        # Legacy generic line — pre-sub-buckets fallback.
        assert c("claude returned no response (timeout/empty)") == "model_timeout"

    def test_classify_unknown_falls_through_to_unknown_bucket(self):
        c = hg._classify_starvation_cause
        # A starvation prefix with a NEW/unrecognised suffix → unknown so
        # the breakdown still reconciles to the aggregate (no row lost).
        assert c("claude returned no response (future_cause_xyz)") == "unknown"
        # No parens → unknown (legacy unparenthesised line, defensive).
        assert c("claude returned no response") == "unknown"
        # Empty reason — defensive; classify() never raises.
        assert c("") == "unknown"
        # Non-starvation prefix (e.g. parse_failed) — unknown by contract;
        # the caller filters via the starvation-prefix gate BEFORE this
        # classifier runs (a non-starved row never enters the per-cause
        # bucket), so this "unknown" return is purely defensive.
        assert c("parse_failed: {garbage") == "unknown"

    def test_by_cause_exact_counts_and_sum_reconcile(self, tmp_path):
        db = tmp_path / "pt.db"
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE decisions (id INTEGER PRIMARY KEY, "
            "action_taken TEXT, reasoning TEXT)"
        )
        rows = [
            ("NO_DECISION", "claude returned no response (timeout)"),
            ("NO_DECISION", "claude returned no response (timeout)"),
            ("NO_DECISION", "claude returned no response (timeout/empty)"),
            ("NO_DECISION", "claude returned no response (nonzero_rc)"),
            ("NO_DECISION", "claude returned no response (nonzero_rc)"),
            ("NO_DECISION", "claude returned no response (empty_stdout)"),
            ("NO_DECISION", "claude returned no response (cli_missing)"),
            ("NO_DECISION", "skipped claude call — host saturated: 5 ..."),
            ("NO_DECISION", "skipped claude call — host saturated mid-call ..."),
            ("NO_DECISION", "skipped claude call — host saturated: 6 ..."),
            ("NO_DECISION", "parse_failed: {garbage"),    # NOT starved
            ("BUY NVDA → FILLED", "ok"),                  # NOT NO_DECISION
            ("HOLD AAPL → HOLD", "no edge"),              # NOT NO_DECISION
        ]
        conn.executemany(
            "INSERT INTO decisions (action_taken, reasoning) VALUES (?, ?)",
            rows,
        )
        conn.commit()
        conn.close()

        out = hg.recent_starvation_by_cause(db, limit=120)
        assert out["ok"] is True
        assert out["n"] == 13
        assert out["starved"] == 10                       # 3 timeout + 2 rc + 1 empty + 1 cli_missing + 3 host_skip
        bc = out["by_cause"]
        # Counts sum to starved exactly — by_cause is a reconciling
        # partition of the starvation slice.
        assert sum(bc.values()) == out["starved"]
        # Each class exact, including the timeout/empty legacy rollup into
        # model_timeout (not model_empty — see _classify_starvation_cause
        # docstring).
        assert bc["model_timeout"] == 3
        assert bc["cli_nonzero_rc"] == 2
        assert bc["model_empty"] == 1
        assert bc["cli_missing"] == 1
        assert bc["host_skip"] == 3
        assert bc["unknown"] == 0
        # The all-labels invariant — every cause key is ALWAYS present
        # (zero-valued for absent classes) so consumers can render the cell
        # without a key-miss guard.
        assert set(bc) == set(hg._CAUSE_LABELS)

    def test_by_cause_missing_db_is_safe_default(self):
        out = hg.recent_starvation_by_cause("/nonexistent/path/to/pt.db")
        assert out["ok"] is False
        assert out["n"] == 0
        assert out["starved"] == 0
        # by_cause is the all-zero shape (NOT empty) so a degraded probe
        # still satisfies the render contract.
        assert set(out["by_cause"]) == set(hg._CAUSE_LABELS)
        assert all(v == 0 for v in out["by_cause"].values())

    def test_unknown_cause_preserves_sum(self, tmp_path):
        db = tmp_path / "pt.db"
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE decisions (id INTEGER PRIMARY KEY, "
            "action_taken TEXT, reasoning TEXT)"
        )
        # A future suffix we don't know about — must roll into `unknown`,
        # not vanish. The breakdown's reconciliation contract is
        # load-bearing: a future cause must never silently zero out a
        # starvation count in the operator-visible aggregate.
        rows = [
            ("NO_DECISION", "claude returned no response (future_xyz)"),
            ("NO_DECISION", "claude returned no response (timeout)"),
        ]
        conn.executemany(
            "INSERT INTO decisions (action_taken, reasoning) VALUES (?, ?)",
            rows,
        )
        conn.commit()
        conn.close()
        out = hg.recent_starvation_by_cause(db)
        assert out["starved"] == 2
        assert out["by_cause"]["unknown"] == 1
        assert out["by_cause"]["model_timeout"] == 1
        assert sum(out["by_cause"].values()) == 2

    def test_cli_human_text_shows_per_cause_when_present(
            self, tmp_path, capsys, monkeypatch):
        db = tmp_path / "pt.db"
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE decisions (id INTEGER PRIMARY KEY, "
            "action_taken TEXT, reasoning TEXT)"
        )
        rows = [
            ("NO_DECISION", "claude returned no response (timeout)"),
            ("NO_DECISION", "claude returned no response (nonzero_rc)"),
            ("NO_DECISION", "skipped claude call — host saturated: 5 ..."),
            ("BUY NVDA → FILLED", "ok"),
        ]
        conn.executemany(
            "INSERT INTO decisions (action_taken, reasoning) VALUES (?, ?)",
            rows,
        )
        conn.commit()
        conn.close()
        monkeypatch.setattr(hg, "_DEFAULT_DB", db)
        monkeypatch.setattr(hg, "probe", lambda: {
            "opus_count": 1, "mem_available_mb": 8000.0,
            "swap_used_pct": 5.0, "load1": 1.0, "load5": 1.0,
            "load15": 1.0, "cpus": 16, "load_per_cpu": 0.1,
        })
        hg.main([])
        out = capsys.readouterr().out
        # The breakdown line must surface non-zero buckets in the human
        # CLI text so an operator gets the actionable cell without needing
        # to read the JSON form.
        assert "by cause:" in out
        assert "model_timeout=1" in out
        assert "cli_nonzero_rc=1" in out
        assert "host_skip=1" in out
        # Zero buckets are NOT printed — the line would be too noisy with
        # six zero classes on a quiet box.
        assert "model_empty=0" not in out
        assert "cli_missing=0" not in out

    def test_cli_human_text_omits_per_cause_line_when_all_zero(
            self, tmp_path, capsys, monkeypatch):
        db = tmp_path / "pt.db"
        conn = sqlite3.connect(db)
        conn.execute(
            "CREATE TABLE decisions (id INTEGER PRIMARY KEY, "
            "action_taken TEXT, reasoning TEXT)"
        )
        # Quiet box — no starved rows at all. The CLI must NOT print a
        # "by cause:" line (every bucket would be zero; pure noise).
        conn.executemany(
            "INSERT INTO decisions (action_taken, reasoning) VALUES (?, ?)",
            [("BUY NVDA → FILLED", "ok"), ("HOLD AAPL → HOLD", "no edge")],
        )
        conn.commit()
        conn.close()
        monkeypatch.setattr(hg, "_DEFAULT_DB", db)
        monkeypatch.setattr(hg, "probe", lambda: {
            "opus_count": 1, "mem_available_mb": 8000.0,
            "swap_used_pct": 5.0, "load1": 1.0, "load5": 1.0,
            "load15": 1.0, "cpus": 16, "load_per_cpu": 0.1,
        })
        hg.main([])
        out = capsys.readouterr().out
        assert "by cause:" not in out


# ── recent_starvation_trend — temporal direction of the storm ────────────────
# The aggregate rate hides direction: 50% could be a clearing storm (100%→0%)
# or an intensifying one (0%→100%). The operator's action diverges across the
# two cases, so a separate verdict is required.

def _seed_decisions(db_path, rows):
    """Insert decisions rows in the given order. The LAST tuple becomes the
    NEWEST row (highest id), matching the live-trader insert order."""
    conn = sqlite3.connect(db_path)
    conn.execute(
        "CREATE TABLE decisions (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "action_taken TEXT, reasoning TEXT)"
    )
    conn.executemany(
        "INSERT INTO decisions (action_taken, reasoning) VALUES (?, ?)", rows,
    )
    conn.commit()
    conn.close()


class TestStarvationTrend:
    """``recent_starvation_trend`` splits the recent decisions into two
    equal halves (older/newer by row id) and reports the rate delta. Pins
    each verdict ladder rung against deterministic seeded DBs."""

    _STARV_ROW = ("NO_DECISION", "skipped claude call — host saturated: x")
    _OK_ROW = ("BUY NVDA → FILLED", "ok")

    def test_worsening_when_newer_half_has_more_starvation(self, tmp_path):
        db = tmp_path / "pt.db"
        # 20 older rows: all OK; 20 newer rows: all starved.
        # Order matters — last inserted is newest.
        rows = [self._OK_ROW] * 20 + [self._STARV_ROW] * 20
        _seed_decisions(db, rows)
        out = hg.recent_starvation_trend(db, limit=120)
        assert out["ok"] is True
        assert out["state"] == "WORSENING"
        assert out["newer_rate"] == 1.0
        assert out["older_rate"] == 0.0
        assert out["delta"] == 1.0
        assert "WORSENING" in out["headline"]

    def test_recovering_when_newer_half_has_less_starvation(self, tmp_path):
        db = tmp_path / "pt.db"
        # 20 older rows: all starved; 20 newer rows: all OK.
        rows = [self._STARV_ROW] * 20 + [self._OK_ROW] * 20
        _seed_decisions(db, rows)
        out = hg.recent_starvation_trend(db, limit=120)
        assert out["state"] == "RECOVERING"
        assert out["newer_rate"] == 0.0
        assert out["older_rate"] == 1.0
        assert out["delta"] == -1.0
        assert "RECOVERING" in out["headline"]

    def test_stable_when_delta_below_threshold(self, tmp_path):
        db = tmp_path / "pt.db"
        # 20 older: 10 starved (50%); 20 newer: 11 starved (55%).
        # Delta 5pp < _TREND_DELTA (10pp) → STABLE.
        older = [self._STARV_ROW] * 10 + [self._OK_ROW] * 10
        newer = [self._STARV_ROW] * 11 + [self._OK_ROW] * 9
        _seed_decisions(db, older + newer)
        out = hg.recent_starvation_trend(db, limit=120)
        assert out["state"] == "STABLE"
        # 50% → 55% (delta 0.05)
        assert abs(out["delta"] - 0.05) < 1e-6

    def test_insufficient_below_min_half(self, tmp_path):
        db = tmp_path / "pt.db"
        # 9 rows per half (under _TREND_MIN_HALF=10). The function still
        # returns rates, but state must be INSUFFICIENT.
        rows = [self._OK_ROW] * 9 + [self._STARV_ROW] * 9
        _seed_decisions(db, rows)
        out = hg.recent_starvation_trend(db, limit=120)
        assert out["ok"] is True
        assert out["state"] == "INSUFFICIENT"
        # Even with a real-direction signal in the data, the verdict
        # suppresses to INSUFFICIENT because the sample is too small.
        assert "need" in out["headline"]

    def test_empty_db_is_insufficient_not_error(self, tmp_path):
        db = tmp_path / "pt.db"
        _seed_decisions(db, [])
        out = hg.recent_starvation_trend(db, limit=120)
        # ok=True (read worked, no rows) but state stays INSUFFICIENT.
        assert out["ok"] is True
        assert out["state"] == "INSUFFICIENT"

    def test_missing_db_returns_safe_default(self):
        out = hg.recent_starvation_trend("/no/such/db.path")
        assert out["ok"] is False
        assert out["state"] == "INSUFFICIENT"
        # Never raises — caller can still render the line.

    def test_threshold_exactly_at_floor_is_worsening(self, tmp_path):
        db = tmp_path / "pt.db"
        # delta == +0.10 exactly should classify WORSENING (>= boundary).
        # 20 older: 0 starved; 20 newer: 2 starved (10%).
        older = [self._OK_ROW] * 20
        newer = [self._STARV_ROW] * 2 + [self._OK_ROW] * 18
        _seed_decisions(db, older + newer)
        out = hg.recent_starvation_trend(db, limit=120)
        # delta is exactly 0.10 → WORSENING (boundary inclusive).
        assert out["state"] == "WORSENING"

    def test_main_cli_renders_trend_line_when_mature(self, tmp_path, capsys,
                                                       monkeypatch):
        db = tmp_path / "pt.db"
        # Mature INTENSIFYING storm — must produce a trend line.
        rows = [self._OK_ROW] * 20 + [self._STARV_ROW] * 20
        _seed_decisions(db, rows)
        monkeypatch.setattr(hg, "_DEFAULT_DB", db)
        monkeypatch.setattr(hg, "probe", lambda: {
            "opus_count": 2, "mem_available_mb": 8000.0,
            "swap_used_pct": 5.0, "load1": 1.0, "load5": 1.0,
            "load15": 1.0, "cpus": 16, "load_per_cpu": 0.1,
        })
        hg.main([])
        out = capsys.readouterr().out
        assert "trend:" in out
        assert "WORSENING" in out

    def test_main_cli_suppresses_trend_line_when_insufficient(
            self, tmp_path, capsys, monkeypatch):
        db = tmp_path / "pt.db"
        # Only 4 decisions — INSUFFICIENT, no trend line.
        rows = [self._OK_ROW] * 2 + [self._STARV_ROW] * 2
        _seed_decisions(db, rows)
        monkeypatch.setattr(hg, "_DEFAULT_DB", db)
        monkeypatch.setattr(hg, "probe", lambda: {
            "opus_count": 1, "mem_available_mb": 8000.0,
            "swap_used_pct": 5.0, "load1": 1.0, "load5": 1.0,
            "load15": 1.0, "cpus": 16, "load_per_cpu": 0.1,
        })
        hg.main([])
        out = capsys.readouterr().out
        assert "trend:" not in out


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
