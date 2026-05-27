"""Reporter wiring for the exit-proximity verdict.

The ``_exit_proximity_line`` helper closes a real dashboard→Discord gap:
when one or more open lots are within striking distance of a mechanical
SL/TP exit, the operator-facing summary historically said nothing about
which lots were about to leave the book. This suite locks:

  * Suppression contract (silence-when-not-actionable) at every level of
    the verdict ladder (COMFORTABLE / NO_DATA / NO_SL_TP_SET silent;
    AT_RISK / NEAR_THRESHOLD fire).
  * SSOT contract — the builder's headline ships verbatim
    (AGENTS.md invariant #10) so this Discord line and
    /api/exit-proximity can never tell different stories.
  * Per-position rendering — up to 3 worst-first lines with the closer
    threshold's signed distance.
  * Wiring — the helper actually fires from BOTH ``send_hourly_summary``
    and ``send_daily_close``.
"""
from __future__ import annotations

from unittest.mock import patch

import pytest

from paper_trader import reporter
from paper_trader.analytics import exit_proximity as ep_mod


class _FakeStore:
    """Minimal store stub — the helper consults ONLY ``open_positions``,
    but the surrounding hourly/daily summary path calls many other reads.
    Return safe defaults so the end-to-end wiring tests can run."""

    def __init__(self, positions=None):
        self._positions = list(positions or [])

    def open_positions(self):
        return [dict(p) for p in self._positions]

    # Surrounding-summary stubs (the helper itself never reaches them).
    def get_portfolio(self):
        return {
            "cash": 1000.0,
            "total_value": 1000.0,
            "positions": [],
            "last_updated": "2026-05-27T12:00:00+00:00",
        }

    def recent_trades(self, limit=50):
        return []

    def recent_decisions(self, limit=20):
        return []

    def equity_curve(self, limit=500):
        return []

    def last_real_decision(self):
        return None


# ── Helpers to build positions in known proximity bands ──────────────


def _stock(ticker, *, cur, sl, tp, qty=1.0, avg=None):
    """Build an open-position dict in the shape ``store.open_positions``
    returns (the same shape ``build_exit_proximity`` consumes)."""
    return {
        "id": 1,
        "ticker": ticker,
        "type": "stock",
        "qty": qty,
        "avg_cost": avg if avg is not None else cur,
        "current_price": cur,
        "stop_loss_price": sl,
        "take_profit_price": tp,
        "expiry": None,
        "strike": None,
        "opened_at": "2026-05-27T10:00:00+00:00",
        "closed_at": None,
        "unrealized_pl": 0.0,
    }


# ── Suppression contract ─────────────────────────────────────────────


def test_silent_when_no_open_positions():
    """Empty book → builder returns NO_DATA → helper silent."""
    store = _FakeStore([])
    out = reporter._exit_proximity_line(store)
    assert out == "", "empty book must suppress (NO_DATA)"


def test_silent_when_no_sl_tp_set():
    """Open positions but none with SL/TP — builder returns NO_SL_TP_SET.
    Helper must suppress (operator can already see this via /api/risk)."""
    pos = {**_stock("AAPL", cur=150.0, sl=140.0, tp=160.0)}
    pos["stop_loss_price"] = None
    pos["take_profit_price"] = None
    store = _FakeStore([pos])
    out = reporter._exit_proximity_line(store)
    assert out == "", "NO_SL_TP_SET verdict must suppress"


def test_silent_when_comfortable():
    """A book sitting mid-corridor on every lot — COMFORTABLE. Must NOT
    fire (the silence-when-nothing-actionable contract; a midband book
    is healthy and any fire would be its own lying-green-light)."""
    # cur=100, sl=90, tp=110: corridor_pos = (100-90)/(110-90) = 0.5 → MID_BAND
    pos = _stock("AAPL", cur=100.0, sl=90.0, tp=110.0)
    store = _FakeStore([pos])
    out = reporter._exit_proximity_line(store)
    assert out == "", "COMFORTABLE verdict must suppress"


# ── Actionable verdicts ──────────────────────────────────────────────


def test_fires_on_at_risk_sl():
    """A lot already past SL → builder verdict AT_RISK → helper fires
    with the page-worthy ⚠️ icon and names the breached ticker."""
    # cur=89, sl=90 → cur < sl → AT_RISK_SL
    pos = _stock("LITE", cur=89.0, sl=90.0, tp=110.0)
    store = _FakeStore([pos])
    out = reporter._exit_proximity_line(store)
    assert "EXIT PROXIMITY" in out, "AT_RISK must fire"
    assert "AT_RISK" in out, "verdict must appear in the header"
    assert "LITE" in out, "breached ticker must appear in the body"
    assert "⚠️" in out, "AT_RISK uses the page-worthy ⚠️ icon"
    assert "🎯" not in out, "🎯 is reserved for NEAR_THRESHOLD"


def test_fires_on_at_risk_tp():
    """A lot already past TP → AT_RISK. Same fire contract as SL; the
    operator wants to know a mechanical TP is about to fire too (a
    forced lock-in is information they want before the trade alert)."""
    # cur=111, tp=110 → cur > tp → AT_RISK_TP
    pos = _stock("NVDA", cur=111.0, sl=90.0, tp=110.0)
    store = _FakeStore([pos])
    out = reporter._exit_proximity_line(store)
    assert "EXIT PROXIMITY" in out
    assert "AT_RISK" in out
    assert "NVDA" in out
    assert "⚠️" in out


def test_fires_on_near_threshold():
    """A lot in the NEAR_SL quartile → NEAR_THRESHOLD verdict → helper
    fires with the 🎯 icon (situational, not page-worthy)."""
    # corridor: sl=90, tp=110 → quartile boundaries 90→95 (NEAR_SL),
    # 95→105 (MID), 105→110 (NEAR_TP). cur=94 → corridor_pos=0.2 → NEAR_SL.
    pos = _stock("MU", cur=94.0, sl=90.0, tp=110.0)
    store = _FakeStore([pos])
    out = reporter._exit_proximity_line(store)
    assert "EXIT PROXIMITY" in out
    assert "NEAR_THRESHOLD" in out
    assert "🎯" in out, "NEAR_THRESHOLD uses 🎯 (situational, not page-worthy)"
    assert "⚠️" not in out


# ── SSOT contract — builder headline verbatim ────────────────────────


def test_ssot_headline_verbatim():
    """The builder's headline must appear verbatim in the helper output
    (AGENTS.md invariant #10 — this Discord line and /api/exit-proximity
    can never disagree on the verdict text). We assert this by patching
    the builder to return a sentinel headline string."""
    sentinel = "SENTINEL_HEADLINE — this exact string must appear in the output"
    fake_snap = {
        "verdict": "AT_RISK",
        "headline": sentinel,
        "n_positions": 1,
        "n_with_sl_tp": 1,
        "band_counts": {"AT_RISK_SL": 1, "AT_RISK_TP": 0, "NEAR_SL": 0,
                         "NEAR_TP": 0, "MID_BAND": 0, "NO_SL_TP": 0},
        "positions": [{
            "ticker": "FAKE", "proximity_band": "AT_RISK_SL",
            "dist_to_sl_pct": -1.0, "dist_to_tp_pct": 10.0,
            "closer_target": "SL", "corridor_pos": -0.05,
        }],
        "thresholds": {"near_sl_max": 0.25, "near_tp_min": 0.75},
    }
    store = _FakeStore([_stock("FAKE", cur=89.0, sl=90.0, tp=110.0)])
    with patch.object(ep_mod, "build_exit_proximity", return_value=fake_snap):
        out = reporter._exit_proximity_line(store)
    assert sentinel in out, \
        "builder's headline must ship verbatim (SSOT — invariant #10)"


# ── Per-position rendering ───────────────────────────────────────────


def test_renders_closer_distance_for_sl_side():
    """When the lot is closer to SL than TP, the per-row tail names the
    SL distance — the actionable number the operator needs to triage."""
    pos = _stock("LITE", cur=89.0, sl=90.0, tp=110.0)
    store = _FakeStore([pos])
    out = reporter._exit_proximity_line(store)
    # cur=89, sl=90 → dist_to_sl_pct = (89-90)/89*100 = -1.12% (past SL)
    assert "from SL" in out, "SL-closer rows must name the SL distance"
    assert "from TP" not in out, "must not render the TP distance for SL-closer rows"


def test_renders_closer_distance_for_tp_side():
    """When the lot is closer to TP than SL, the per-row tail names the
    TP distance instead."""
    # corridor: sl=90, tp=110 → NEAR_TP starts at corridor_pos=0.75 → cur=105.
    # cur=107 → corridor_pos=(107-90)/20=0.85 → NEAR_TP, closer to TP.
    pos = _stock("NVDA", cur=107.0, sl=90.0, tp=110.0)
    store = _FakeStore([pos])
    out = reporter._exit_proximity_line(store)
    assert "from TP" in out, "TP-closer rows must name the TP distance"
    assert "from SL" not in out, "must not render the SL distance for TP-closer rows"


def test_caps_at_3_worst_rows():
    """A book with many actionable lots must render at most 3 per-position
    lines — keep the Discord line lean. The builder sorts worst-first so
    the head of the list is the right slice."""
    positions = [
        _stock("AAA", cur=89.0, sl=90.0, tp=110.0),  # AT_RISK_SL
        _stock("BBB", cur=88.0, sl=90.0, tp=110.0),  # AT_RISK_SL (deeper)
        _stock("CCC", cur=87.0, sl=90.0, tp=110.0),  # AT_RISK_SL (deepest)
        _stock("DDD", cur=91.0, sl=90.0, tp=110.0),  # NEAR_SL
        _stock("EEE", cur=92.0, sl=90.0, tp=110.0),  # NEAR_SL
    ]
    # Use distinct ids so they coexist in the open-positions list.
    for i, p in enumerate(positions, start=1):
        p["id"] = i
    store = _FakeStore(positions)
    out = reporter._exit_proximity_line(store)
    # Count rows of the form "> `XXX` AT_RISK_SL —" / "> `XXX` NEAR_SL —".
    # Header + headline = 2 lines; per-position lines come after.
    per_pos_lines = [ln for ln in out.split("\n")
                     if ln.startswith("> `") and "AT_RISK_SL" in ln + " "
                     or (ln.startswith("> `") and "NEAR_SL" in ln + " ")]
    # Allow up to 3 (the cap); deeper detail belongs on the dashboard.
    assert 1 <= len(per_pos_lines) <= 3, \
        f"per-position rendering must cap at 3 lines (got {len(per_pos_lines)})"


def test_at_risk_book_shows_only_actionable_rows():
    """A book with a mix of AT_RISK + MID_BAND lots must render only the
    AT_RISK rows in the per-position list — a midband lot is not the
    operator's concern when a forced exit is imminent elsewhere."""
    positions = [
        _stock("MID1", cur=100.0, sl=90.0, tp=110.0),  # MID_BAND
        _stock("ATR1", cur=89.0, sl=90.0, tp=110.0),   # AT_RISK_SL
        _stock("MID2", cur=99.0, sl=90.0, tp=110.0),   # MID_BAND
    ]
    for i, p in enumerate(positions, start=1):
        p["id"] = i
    store = _FakeStore(positions)
    out = reporter._exit_proximity_line(store)
    assert "ATR1" in out, "AT_RISK lot must appear in the body"
    assert "MID1" not in out, "MID_BAND lot must be filtered out"
    assert "MID2" not in out, "MID_BAND lot must be filtered out"


# ── Failure contract — never raises ──────────────────────────────────


def test_helper_swallows_builder_exception():
    """A builder fault must degrade to ``""`` (no proximity line this
    report), never an exception (which would take down the whole hourly
    summary). The reporter discipline: notification helpers never raise."""
    store = _FakeStore([_stock("AAPL", cur=89.0, sl=90.0, tp=110.0)])
    with patch.object(ep_mod, "build_exit_proximity",
                       side_effect=RuntimeError("synthetic")):
        out = reporter._exit_proximity_line(store)
    assert out == "", "builder fault must degrade to silence, not raise"


def test_helper_swallows_store_exception():
    """A store fault (e.g. transient DB lock) must degrade to ``""``
    too — same notification-helper contract."""
    class _Boom:
        def open_positions(self):
            raise RuntimeError("synthetic db lock")
    out = reporter._exit_proximity_line(_Boom())
    assert out == "", "store fault must degrade to silence, not raise"


def test_helper_handles_non_dict_builder_return():
    """A bug that returns the wrong type from the builder must NOT crash
    the reporter — degrade to silence (degrade-safe contract)."""
    store = _FakeStore([_stock("AAPL", cur=89.0, sl=90.0, tp=110.0)])
    with patch.object(ep_mod, "build_exit_proximity", return_value=None):
        out = reporter._exit_proximity_line(store)
    assert out == ""
    with patch.object(ep_mod, "build_exit_proximity", return_value=[]):
        out = reporter._exit_proximity_line(store)
    assert out == ""


def test_helper_silent_on_empty_headline():
    """An actionable verdict with an empty headline (defensive — the
    builder always emits one, but a future regression must not surface
    a header-only line). Drop the line entirely instead of rendering
    a bare 'EXIT PROXIMITY ◈ AT_RISK\\n>' with no description."""
    fake_snap = {
        "verdict": "AT_RISK", "headline": "",
        "n_positions": 1, "n_with_sl_tp": 1,
        "band_counts": {"AT_RISK_SL": 1, "AT_RISK_TP": 0, "NEAR_SL": 0,
                         "NEAR_TP": 0, "MID_BAND": 0, "NO_SL_TP": 0},
        "positions": [{"ticker": "FAKE", "proximity_band": "AT_RISK_SL"}],
        "thresholds": {"near_sl_max": 0.25, "near_tp_min": 0.75},
    }
    store = _FakeStore([_stock("FAKE", cur=89.0, sl=90.0, tp=110.0)])
    with patch.object(ep_mod, "build_exit_proximity", return_value=fake_snap):
        out = reporter._exit_proximity_line(store)
    assert out == "", "empty headline must suppress (defensive contract)"


# ── Wiring — actually called from hourly + daily close ───────────────


def test_wired_into_send_hourly_summary():
    """Verify the helper is actually called when ``send_hourly_summary``
    runs. We patch ``_send`` so no real Discord round-trip fires and
    monkey-patch ``_exit_proximity_line`` to a sentinel that lets us
    assert the call happened AND its return value lands in the body."""
    sentinel = "🟣 SENTINEL_EXIT_PROXIMITY_LINE 🟣"
    captured: dict = {}

    def _capture(msg: str) -> bool:
        captured["body"] = msg
        return True

    with patch.object(reporter, "_send", side_effect=_capture), \
         patch.object(reporter, "_exit_proximity_line",
                      return_value=sentinel) as mock_helper, \
         patch.object(reporter, "get_store", return_value=_FakeStore([])):
        ok = reporter.send_hourly_summary()
    assert ok is True, "patched _send returned True; summary should succeed"
    assert mock_helper.called, "send_hourly_summary must call _exit_proximity_line"
    assert sentinel in captured.get("body", ""), \
        "_exit_proximity_line's return value must land in the hourly body"


def test_wired_into_send_daily_close():
    """Same wiring check for ``send_daily_close``."""
    sentinel = "🟣 SENTINEL_EXIT_PROXIMITY_LINE_DAILY 🟣"
    captured: dict = {}

    def _capture(msg: str) -> bool:
        captured["body"] = msg
        return True

    with patch.object(reporter, "_send", side_effect=_capture), \
         patch.object(reporter, "_exit_proximity_line",
                      return_value=sentinel) as mock_helper, \
         patch.object(reporter, "get_store", return_value=_FakeStore([])):
        ok = reporter.send_daily_close()
    assert ok is True
    assert mock_helper.called, "send_daily_close must call _exit_proximity_line"
    assert sentinel in captured.get("body", ""), \
        "_exit_proximity_line's return value must land in the daily-close body"


def test_silent_helper_does_not_add_a_line():
    """When the helper returns ``""``, the body must NOT carry the
    sentinel — confirms the wiring respects the silence contract
    (a stray `body += "\\n"` would surface an empty trailing line)."""
    captured: dict = {}

    def _capture(msg: str) -> bool:
        captured["body"] = msg
        return True

    with patch.object(reporter, "_send", side_effect=_capture), \
         patch.object(reporter, "_exit_proximity_line",
                      return_value="") as mock_helper, \
         patch.object(reporter, "get_store", return_value=_FakeStore([])):
        reporter.send_hourly_summary()
    assert mock_helper.called
    body = captured.get("body", "")
    # A silent helper must not produce a "**EXIT PROXIMITY**" mention.
    assert "EXIT PROXIMITY" not in body, \
        "silent helper return must not inject any EXIT PROXIMITY text"
