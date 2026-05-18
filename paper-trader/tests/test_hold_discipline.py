"""Tests for analytics/hold_discipline.py — the open-book disposition trap.

``build_hold_discipline`` is a *diagnostic* layered on two single sources
of truth (AGENTS.md invariant #10): the empirical median *losing* hold is
consumed **verbatim** from ``loser_autopsy.build_loser_autopsy`` (which
itself consumes ``round_trips.build_round_trips``), and the per-position
dollar is read **directly** from the positions table's ``unrealized_pl``.

Each test below fails loudly if the logic is wrong, not merely if it
crashes:

* an inline re-derivation of the median that drifts from
  ``/api/loser-autopsy`` → ``TestReferenceNoDrift``
* a ``>=`` instead of strict ``>`` boundary, or a winner flagged as an
  overstayed loser → ``TestBoundary``
* a verdict emitted before the sample-size gate → ``TestStateLadder``
* a composed-builder fault that raises instead of degrading → ``TestSafe``
* the route serving a different builder / store than the operator sees →
  ``TestEndpoint``
* the daily-close Discord line leaking on NO_DATA, or a builder fault
  killing the whole close report → ``TestReporterLine``
"""
from __future__ import annotations

import statistics
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics import hold_discipline as hd
from paper_trader.analytics.hold_discipline import (
    MIN_REFERENCE_LOSERS,
    build_hold_discipline,
)
from paper_trader.analytics.loser_autopsy import build_loser_autopsy
from paper_trader.analytics.round_trips import build_round_trips

_BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
_NOW = datetime(2026, 3, 1, 12, 0, 0, tzinfo=timezone.utc)


def _day(offset: int) -> str:
    return (_BASE + timedelta(days=offset)).isoformat()


def _rt(tid, ticker, buy_day, sell_day, qty, buy_px, sell_px):
    """A BUY+SELL pair build_round_trips folds into one closed round-trip."""
    return [
        {"id": tid, "timestamp": _day(buy_day), "ticker": ticker,
         "action": "BUY", "qty": qty, "price": buy_px,
         "value": qty * buy_px, "strike": None, "expiry": None,
         "option_type": None, "reason": ""},
        {"id": tid + 1, "timestamp": _day(sell_day), "ticker": ticker,
         "action": "SELL", "qty": qty, "price": sell_px,
         "value": qty * sell_px, "strike": None, "expiry": None,
         "option_type": None, "reason": ""},
    ]


def _losers_ledger(hold_days_each: list[int]) -> list[dict]:
    """One losing round-trip (buy 100 → sell 90, qty 10 ⇒ −$100) per entry,
    each on a disjoint window of the requested calendar length."""
    trades: list[dict] = []
    tid, day = 1, 0
    for h in hold_days_each:
        trades += _rt(tid, f"L{tid}", day, day + h, 10, 100.0, 90.0)
        tid += 2
        day += h + 5  # disjoint, non-overlapping windows
    return trades


def _pos(ticker, unrealized_pl, age_days, *, type_="stock", opened_at=None):
    """Store.open_positions()-shaped open row. ``age_days`` is turned into a
    concrete ``opened_at`` relative to ``_NOW`` unless ``opened_at`` is
    given explicitly (used for the unparseable-timestamp case)."""
    if opened_at is None:
        opened_at = (_NOW - timedelta(days=age_days)).isoformat()
    return {
        "ticker": ticker, "type": type_, "qty": 1.0, "avg_cost": 100.0,
        "current_price": 100.0 + unrealized_pl, "unrealized_pl": unrealized_pl,
        "opened_at": opened_at, "closed_at": None,
    }


# ───────────────────────── the no-drift lock ──────────────────────────

class TestReferenceNoDrift:
    """The reference median MUST be ``build_loser_autopsy``'s
    ``median_loser_hold_days`` byte-identical (composed verbatim, never
    re-derived — the risk_mirror "embedded headline byte-identical"
    discipline). A future inline median that drifts from /api/loser-autopsy
    fails here."""

    def test_median_equals_loser_autopsy_and_independent_median(self):
        # 5 losing round-trips, holds 2/4/6/8/10 → odd count, exact middle
        # (no even-average rounding), so all three computations must agree.
        trades = _losers_ledger([2, 4, 6, 8, 10])

        la = build_loser_autopsy(trades)
        out = build_hold_discipline([], trades, now=_NOW)

        # 1) verbatim composition — identical object value, not "close".
        assert (out["reference_median_losing_hold_days"]
                == la["median_loser_hold_days"])
        assert out["reference_n_closed_losers"] == la["n_losers"] == 5

        # 2) independent recomputation straight off build_round_trips
        #    (the actual single source) — catches a drift in BOTH layers.
        losing_holds = sorted(
            rt["hold_days"] for rt in build_round_trips(trades)
            if (rt.get("pnl_usd") or 0.0) < 0 and rt["hold_days"] is not None)
        assert losing_holds == [2.0, 4.0, 6.0, 8.0, 10.0]
        assert (out["reference_median_losing_hold_days"]
                == statistics.median(losing_holds) == 6.0)

    def test_winners_excluded_from_reference(self):
        # 3 losers (holds 2/4/6 → median 4) + 1 big fast winner. The winner
        # must NOT pull the losing-hold median (a regression that medianed
        # all round-trips would shift it).
        trades = _losers_ledger([2, 4, 6])
        trades += _rt(99, "WIN", 200, 230, 10, 100.0, 180.0)  # +$800, 30d
        out = build_hold_discipline([], trades, now=_NOW)
        assert out["reference_median_losing_hold_days"] == 4.0
        assert out["reference_n_closed_losers"] == 3


# ───────────────────────── strict boundary ────────────────────────────

class TestBoundary:
    """``age_days > median`` is strict (the loser_autopsy strict-boundary
    idiom): a position aged *exactly* the median is WITHIN discipline; a
    winner is never overstayed regardless of age."""

    def _ref4(self):
        # holds 2/4/6 → median 4.0, n_losers 3 == MIN_REFERENCE_LOSERS
        return _losers_ledger([2, 4, 6])

    def test_age_equal_median_is_not_overstayed(self):
        trades = self._ref4()
        out = build_hold_discipline([_pos("EQ", -10.0, 4.0)], trades,
                                    now=_NOW)
        assert out["reference_median_losing_hold_days"] == 4.0
        card = out["positions"][0]
        assert card["age_days"] == 4.0
        assert card["overstayed"] is False
        assert out["state"] == "DISCIPLINED"

    def test_age_just_past_median_is_overstayed(self):
        trades = self._ref4()
        out = build_hold_discipline([_pos("OVER", -10.0, 4.01)], trades,
                                    now=_NOW)
        card = out["positions"][0]
        assert card["overstayed"] is True
        assert card["overstay_mult"] == round(4.01 / 4.0, 2)
        assert out["state"] == "DISPOSITION_DRAG"
        assert out["n_overstayed"] == 1

    def test_winner_past_median_is_never_overstayed(self):
        trades = self._ref4()
        out = build_hold_discipline([_pos("WINR", +50.0, 99.0)], trades,
                                    now=_NOW)
        card = out["positions"][0]
        assert card["is_losing"] is False
        assert card["overstayed"] is False
        assert out["state"] == "DISCIPLINED"
        assert out["n_overstayed"] == 0

    def test_unparseable_opened_at_never_raises_and_not_overstayed(self):
        trades = self._ref4()
        out = build_hold_discipline(
            [_pos("BAD", -10.0, 0, opened_at="not-a-timestamp")],
            trades, now=_NOW)
        card = out["positions"][0]
        assert card["age_days"] is None
        assert card["overstayed"] is False


# ───────────────────────── state ladder ───────────────────────────────

class TestStateLadder:
    """NO_DATA / INSUFFICIENT / DISCIPLINED / DISPOSITION_DRAG with the
    verdict gated to a stable empirical reference (the loser_autopsy
    sample-size honesty precedent)."""

    def test_no_open_positions_is_no_data(self):
        out = build_hold_discipline([], _losers_ledger([2, 4, 6]), now=_NOW)
        assert out["state"] == "NO_DATA"
        assert out["verdict"] is None
        assert "nothing to check" in out["headline"]

    def test_insufficient_closed_losers_withholds_verdict(self):
        # Only 2 closed losers < MIN_REFERENCE_LOSERS (3): even a wildly
        # overstayed open loser is NOT flagged and the verdict is withheld,
        # but the per-position card is still emitted with its age.
        assert MIN_REFERENCE_LOSERS == 3
        trades = _losers_ledger([3, 9])
        out = build_hold_discipline([_pos("OLD", -25.0, 500.0)], trades,
                                    now=_NOW)
        assert out["state"] == "INSUFFICIENT"
        assert out["verdict"] is None
        assert out["n_overstayed"] == 0
        card = out["positions"][0]
        assert card["age_days"] == 500.0          # numerics still emitted
        assert card["overstayed"] is False        # but never flagged
        assert "verdict withheld" in out["headline"]

    def test_disciplined_when_all_losers_within_reference(self):
        trades = _losers_ledger([2, 4, 6])  # median 4
        out = build_hold_discipline(
            [_pos("A", -3.0, 1.0), _pos("B", +9.0, 50.0)], trades, now=_NOW)
        assert out["state"] == "DISCIPLINED"
        assert out["verdict"] == "DISCIPLINED"
        assert out["n_overstayed"] == 0
        assert out["disposition_drag_usd"] == 0.0
        assert "no disposition drag" in out["headline"]

    def test_disposition_drag_exact_aggregates_and_headline(self):
        trades = _losers_ledger([2, 4, 6])  # median 4.0
        positions = [
            _pos("MU", -30.0, 6.0),    # losing, 6d > 4d  → OVERSTAYED
            _pos("NVDA", -5.0, 3.0),   # losing, 3d ≤ 4d  → within
            _pos("AMD", +20.0, 10.0),  # winning           → never
        ]
        out = build_hold_discipline(positions, trades, now=_NOW)
        assert out["state"] == "DISPOSITION_DRAG"
        assert out["verdict"] == "DISPOSITION_DRAG"
        assert out["n_open"] == 3
        assert out["n_losing_open"] == 2
        assert out["n_overstayed"] == 1
        # $ at risk is the direct unrealized_pl of the overstayed loser only.
        assert out["disposition_drag_usd"] == -30.0
        assert out["worst_overstayed"]["ticker"] == "MU"
        # Overstayed card sorts first (deterministic ordering contract).
        assert out["positions"][0]["ticker"] == "MU"
        assert out["positions"][0]["overstayed"] is True
        assert ("held past the empirical median losing hold (4.00d)"
                in out["headline"])
        assert "MU (6.0d, $-30.00)" in out["headline"]
        assert "Disposition drag $-30.00 unrealized" in out["headline"]

    def test_two_overstayed_drag_sums_and_worst_is_most_negative(self):
        trades = _losers_ledger([2, 4, 6])  # median 4.0
        positions = [
            _pos("X", -10.0, 7.0),   # overstayed
            _pos("Y", -40.0, 9.0),   # overstayed, worse
        ]
        out = build_hold_discipline(positions, trades, now=_NOW)
        assert out["n_overstayed"] == 2
        assert out["disposition_drag_usd"] == -50.0
        assert out["worst_overstayed"]["ticker"] == "Y"
        assert out["positions"][0]["ticker"] == "Y"  # most-negative first


# ───────────────────────── _safe contract ─────────────────────────────

class TestSafe:
    """A fault in the composed builder must degrade to an honest
    verdict-withheld state, never an exception (the event_calendar /
    risk_mirror ``_safe`` contract — a diagnostics fault must not 500 the
    endpoint or kill the daily-close report)."""

    def test_loser_autopsy_fault_degrades_not_raises(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("loser-autopsy exploded")
        monkeypatch.setattr(hd, "build_loser_autopsy", _boom)
        out = build_hold_discipline([_pos("Z", -10.0, 999.0)], [], now=_NOW)
        assert out["state"] == "INSUFFICIENT"
        assert out["verdict"] is None
        assert out["reference_median_losing_hold_days"] is None
        assert str(out["reference_state"]).startswith("ERROR")
        assert "reference unavailable" in out["headline"]
        # The position card still renders (age computed) — never flagged.
        assert out["positions"][0]["overstayed"] is False

    def test_garbage_unrealized_pl_does_not_raise(self):
        trades = _losers_ledger([2, 4, 6])
        bad = _pos("G", 0.0, 5.0)
        bad["unrealized_pl"] = "not-a-number"
        out = build_hold_discipline([bad], trades, now=_NOW)
        # Coerced to 0.0 → not losing → not overstayed, no exception.
        assert out["positions"][0]["is_losing"] is False
        assert out["state"] == "DISCIPLINED"


# ───────────────────────── /api endpoint parity ───────────────────────

def _seed_closed_losers(store, holds: list[int]):
    """Insert BUY/SELL trade pairs with controlled timestamps so
    build_round_trips sees real losing round-trips of known hold lengths
    (record_trade stamps 'now', which would collapse every hold to ~0)."""
    rows: list[tuple] = []
    tid_day = 0
    for i, h in enumerate(holds):
        bt = (_BASE + timedelta(days=tid_day)).isoformat()
        st = (_BASE + timedelta(days=tid_day + h)).isoformat()
        tk = f"CL{i}"
        rows.append((bt, tk, "BUY", 10.0, 100.0, 1000.0))
        rows.append((st, tk, "SELL", 10.0, 90.0, 900.0))
        tid_day += h + 5
    with store._lock:
        store.conn.executemany(
            "INSERT INTO trades (timestamp,ticker,action,qty,price,value) "
            "VALUES (?,?,?,?,?,?)", rows)
        store.conn.commit()


def _seed_open(store, ticker, unrealized_pl, age_days):
    store.upsert_position(ticker, "stock", 1.0, 100.0)
    opened = (datetime.now(timezone.utc)
              - timedelta(days=age_days)).isoformat()
    with store._lock:
        store.conn.execute(
            "UPDATE positions SET opened_at=?, current_price=?, "
            "unrealized_pl=? WHERE ticker=? AND closed_at IS NULL",
            (opened, 100.0 + unrealized_pl, unrealized_pl, ticker))
        store.conn.commit()


class TestEndpoint:
    """The route must serve the SAME builder against the live store so the
    dashboard sees exactly what the reporter / operator sees (the
    loser_autopsy / event_calendar endpoint-parity discipline)."""

    def test_endpoint_reports_disposition_drag(self, tmp_path, monkeypatch):
        from paper_trader import store as store_mod
        from paper_trader.store import Store

        db = tmp_path / "paper_trader.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        s = Store()
        monkeypatch.setattr(store_mod, "_singleton", s)  # endpoint reuses it

        _seed_closed_losers(s, [2, 4, 6])          # median losing hold 4.0
        _seed_open(s, "MU", -30.0, age_days=6)     # losing, 6d > 4d
        _seed_open(s, "NVDA", -5.0, age_days=1)    # losing, within

        from paper_trader import dashboard
        dashboard.app.config["TESTING"] = True
        try:
            with dashboard.app.test_client() as client:
                resp = client.get("/api/hold-discipline")
        finally:
            s.close()

        assert resp.status_code == 200
        data = resp.get_json()
        assert "error" not in data, data
        assert data["state"] == "DISPOSITION_DRAG"
        assert data["reference_median_losing_hold_days"] == 4.0
        assert data["n_overstayed"] == 1
        assert data["disposition_drag_usd"] == -30.0
        assert data["worst_overstayed"]["ticker"] == "MU"


# ───────────────────────── Discord reporter line ──────────────────────

class TestReporterLine:
    """``_hold_discipline_line`` composes the builder verbatim, suppresses
    NO_DATA/INSUFFICIENT (no actionable reference yet — the
    ``_behavioural_block`` NO_DATA precedent), and a builder fault degrades
    to ``""`` while ``send_daily_close`` still sends ("no block, never no
    summary")."""

    def _store(self, tmp_path, monkeypatch):
        from paper_trader import store as store_mod
        from paper_trader.store import Store
        db = tmp_path / "paper_trader.db"
        monkeypatch.setattr(store_mod, "DB_PATH", db)
        monkeypatch.setattr(store_mod, "_singleton", None)
        s = Store()
        monkeypatch.setattr(store_mod, "_singleton", s)
        return s

    def test_line_suppressed_when_no_reference(self, tmp_path, monkeypatch):
        from paper_trader import reporter
        s = self._store(tmp_path, monkeypatch)
        _seed_open(s, "MU", -30.0, age_days=6)  # open, but 0 closed losers
        assert reporter._hold_discipline_line(s) == ""   # INSUFFICIENT → ""
        s.close()

    def test_line_emitted_on_disposition_drag(self, tmp_path, monkeypatch):
        from paper_trader import reporter
        s = self._store(tmp_path, monkeypatch)
        _seed_closed_losers(s, [2, 4, 6])
        _seed_open(s, "MU", -30.0, age_days=6)
        line = reporter._hold_discipline_line(s)
        assert "HOLD DISCIPLINE" in line
        assert "DISPOSITION_DRAG" in line
        assert "MU (6.0d, $-30.00)" in line
        s.close()

    def test_daily_close_survives_builder_fault(self, tmp_path,
                                                 monkeypatch):
        from paper_trader import reporter
        s = self._store(tmp_path, monkeypatch)
        _seed_closed_losers(s, [2, 4, 6])
        _seed_open(s, "MU", -30.0, age_days=6)

        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter, "get_store", lambda: s)
        monkeypatch.setattr(reporter.market, "benchmark_sp500", lambda: None)

        def _boom(*a, **k):
            raise RuntimeError("hold-discipline exploded")
        monkeypatch.setattr(reporter, "build_hold_discipline", _boom)

        # The whole close report must still go out — the block just vanishes.
        assert reporter.send_daily_close() is True
        assert captured and "DAILY CLOSE" in captured[0]
        assert "HOLD DISCIPLINE" not in captured[0]
        s.close()

    def test_daily_close_includes_block_when_present(self, tmp_path,
                                                     monkeypatch):
        from paper_trader import reporter
        s = self._store(tmp_path, monkeypatch)
        _seed_closed_losers(s, [2, 4, 6])
        _seed_open(s, "MU", -30.0, age_days=6)

        captured: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: captured.append(msg) or True)
        monkeypatch.setattr(reporter, "get_store", lambda: s)
        monkeypatch.setattr(reporter.market, "benchmark_sp500", lambda: None)

        assert reporter.send_daily_close() is True
        assert "HOLD DISCIPLINE" in captured[0]
        assert "DISPOSITION_DRAG" in captured[0]
        s.close()
