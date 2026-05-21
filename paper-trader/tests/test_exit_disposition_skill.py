"""Tests for paper_trader.ml.exit_disposition_skill.

Tests assert SPECIFIC verdicts on hand-crafted synthetic trades — each
test exercises one verdict branch or pairing-logic edge with the exact-
value contract the verdict logic claims. The exit-disposition skill
diagnostic must:

  * classify reasons starting with ``stop-loss`` / ``take-profit`` into
    the SL / TP classes respectively, everything else into manual_sell
  * walk FIFO across (run_id, ticker), blending avg_cost on consecutive
    BUYs and clipping a SELL's qty to held — matching backtest.py's
    ``_buy`` / ``_sell`` semantics byte-for-byte
  * exclude (drop) SELL rows with no matching open position
  * exclude (drop) BUY/SELL rows with non-finite or non-positive qty/price
  * report ``MANUAL_SELL_LOSING`` when the manual-SELL class mean realized
    exit < −MANUAL_EDGE_TOL_PP (the actionable signal)
  * report ``MANUAL_SELL_FLAT`` when within ±MANUAL_EDGE_TOL_PP
  * report ``MANUAL_SELL_WINNING`` when > +MANUAL_EDGE_TOL_PP
  * report ``INSUFFICIENT_DATA`` below MIN_TOTAL or below MIN_CLASS_N in
    manual_sell
  * report the SL/TP default-residual (mean realized − default) when ≥1
    fire exists, None otherwise
  * never raise from the public ``exit_disposition_report`` / ``analyze_db``
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from paper_trader.ml.exit_disposition_skill import (
    _classify_reason,
    _pair_trades_fifo,
    exit_disposition_report,
    analyze_db,
    MIN_TOTAL,
    MIN_CLASS_N,
    MANUAL_EDGE_TOL_PP,
    SL_DEFAULT_PCT,
    TP_DEFAULT_PCT,
)


# ─────────────────────────── _classify_reason ───────────────────────────


class TestClassifyReason:
    def test_stop_loss_prefix(self):
        assert _classify_reason("stop-loss @ 92.0 (close 91.8)") == "stop_loss"

    def test_take_profit_prefix(self):
        assert _classify_reason("take-profit @ 115.0 (close 115.5)") == "take_profit"

    def test_ml_decide_manual_sell(self):
        assert _classify_reason(
            "ML+quant: NVDA score=-1.50 regime=bull RSI=72 — reducing"
        ) == "manual_sell"

    def test_empty_reason_is_manual(self):
        assert _classify_reason("") == "manual_sell"
        assert _classify_reason(None) == "manual_sell"

    def test_case_sensitive_prefix(self):
        # The engine writes lowercase prefixes only — uppercase is some
        # external rewrite, treat as manual_sell so we don't misclassify
        # arbitrary "STOP-LOSS:" text in operator notes.
        assert _classify_reason("Stop-Loss @ 92") == "manual_sell"
        assert _classify_reason("STOP-LOSS") == "manual_sell"

    def test_substring_not_prefix(self):
        # `_classify_reason` uses str.startswith, not `in`. A reason that
        # MENTIONS stop-loss elsewhere is manual.
        assert _classify_reason(
            "ML+quant: reducing before potential stop-loss"
        ) == "manual_sell"


# ─────────────────────────── _pair_trades_fifo ──────────────────────────


class TestPairTradesFifo:
    def test_single_buy_then_sell(self):
        trades = [
            {"run_id": 1, "sim_date": "2025-01-01", "ticker": "NVDA",
             "action": "BUY", "qty": 10, "price": 100.0, "reason": "open"},
            {"run_id": 1, "sim_date": "2025-01-05", "ticker": "NVDA",
             "action": "SELL", "qty": 10, "price": 115.0,
             "reason": "take-profit @ 115 (close 115)"},
        ]
        pairs = _pair_trades_fifo(trades)
        assert len(pairs) == 1
        p = pairs[0]
        assert p["ticker"] == "NVDA"
        assert p["sell_qty"] == 10
        assert p["avg_cost"] == 100.0
        assert p["exit_pct"] == 15.0       # (115 - 100)/100 * 100
        assert p["disposition"] == "take_profit"

    def test_blended_avg_cost_on_two_buys(self):
        # Mirror backtest.py::_buy's blended avg formula:
        # (10 * 100 + 10 * 120) / 20 = 110
        trades = [
            {"run_id": 1, "sim_date": "2025-01-01", "ticker": "NVDA",
             "action": "BUY", "qty": 10, "price": 100.0, "reason": "open"},
            {"run_id": 1, "sim_date": "2025-01-02", "ticker": "NVDA",
             "action": "BUY", "qty": 10, "price": 120.0, "reason": "add"},
            {"run_id": 1, "sim_date": "2025-01-10", "ticker": "NVDA",
             "action": "SELL", "qty": 20, "price": 121.0,
             "reason": "ML+quant: trim"},
        ]
        pairs = _pair_trades_fifo(trades)
        assert len(pairs) == 1
        assert pairs[0]["avg_cost"] == 110.0
        # (121 - 110) / 110 * 100 = 10.0
        assert pairs[0]["exit_pct"] == 10.0
        assert pairs[0]["disposition"] == "manual_sell"

    def test_partial_sell_then_full_sell(self):
        trades = [
            {"run_id": 1, "sim_date": "2025-01-01", "ticker": "NVDA",
             "action": "BUY", "qty": 10, "price": 100.0, "reason": "open"},
            {"run_id": 1, "sim_date": "2025-01-05", "ticker": "NVDA",
             "action": "SELL", "qty": 4, "price": 110.0,
             "reason": "ML+quant: trim"},
            {"run_id": 1, "sim_date": "2025-01-10", "ticker": "NVDA",
             "action": "SELL", "qty": 6, "price": 92.0,
             "reason": "stop-loss @ 92 (close 92)"},
        ]
        pairs = _pair_trades_fifo(trades)
        assert len(pairs) == 2
        # First partial sell: held 10 at 100, sell 4 at 110 → +10%, manual
        assert pairs[0]["sell_qty"] == 4
        assert pairs[0]["avg_cost"] == 100.0
        assert pairs[0]["exit_pct"] == 10.0
        assert pairs[0]["disposition"] == "manual_sell"
        # Second sell empties the position: held 6 at 100, sell 6 at 92 → -8%, SL
        assert pairs[1]["sell_qty"] == 6
        assert pairs[1]["avg_cost"] == 100.0
        assert pairs[1]["exit_pct"] == -8.0
        assert pairs[1]["disposition"] == "stop_loss"

    def test_sell_clipped_to_held_qty(self):
        # Sell larger than held → clip to held (mirrors backtest.py::_sell).
        trades = [
            {"run_id": 1, "sim_date": "2025-01-01", "ticker": "NVDA",
             "action": "BUY", "qty": 5, "price": 100.0, "reason": "open"},
            {"run_id": 1, "sim_date": "2025-01-05", "ticker": "NVDA",
             "action": "SELL", "qty": 999, "price": 110.0,
             "reason": "ML+quant: full close"},
        ]
        pairs = _pair_trades_fifo(trades)
        assert len(pairs) == 1
        # sell_qty = min(999, 5) = 5
        assert pairs[0]["sell_qty"] == 5
        assert pairs[0]["exit_pct"] == 10.0

    def test_sell_without_buy_dropped(self):
        # A SELL with no matching open position — drop it (engine semantics).
        trades = [
            {"run_id": 1, "sim_date": "2025-01-05", "ticker": "NVDA",
             "action": "SELL", "qty": 10, "price": 110.0,
             "reason": "ML+quant: phantom"},
        ]
        assert _pair_trades_fifo(trades) == []

    def test_independent_run_ids(self):
        # The same ticker bought in run 1 must NOT match a SELL in run 2.
        trades = [
            {"run_id": 1, "sim_date": "2025-01-01", "ticker": "NVDA",
             "action": "BUY", "qty": 10, "price": 100.0, "reason": "open"},
            {"run_id": 2, "sim_date": "2025-01-05", "ticker": "NVDA",
             "action": "SELL", "qty": 10, "price": 110.0,
             "reason": "ML+quant: trim"},
        ]
        pairs = _pair_trades_fifo(trades)
        # SELL in run 2 has no open position → dropped.
        assert pairs == []

    def test_non_finite_qty_dropped(self):
        trades = [
            {"run_id": 1, "sim_date": "2025-01-01", "ticker": "NVDA",
             "action": "BUY", "qty": float("nan"), "price": 100.0,
             "reason": "bad"},
            {"run_id": 1, "sim_date": "2025-01-05", "ticker": "NVDA",
             "action": "SELL", "qty": 10, "price": 110.0,
             "reason": "ML+quant"},
        ]
        # The BUY was dropped (nan qty), so the SELL has no open position
        # and is itself dropped.
        assert _pair_trades_fifo(trades) == []

    def test_zero_price_dropped(self):
        trades = [
            {"run_id": 1, "sim_date": "2025-01-01", "ticker": "NVDA",
             "action": "BUY", "qty": 10, "price": 0.0, "reason": "bad"},
        ]
        # Zero price → drop (filtered by `if qty <= 0 or price <= 0`).
        assert _pair_trades_fifo(trades) == []


# ─────────────────────────── exit_disposition_report ────────────────────


def _mk_pair(disposition: str, exit_pct: float) -> dict:
    return {"disposition": disposition, "exit_pct": exit_pct,
            "run_id": 1, "ticker": "NVDA", "sim_date": "2025-01-01",
            "sell_qty": 1, "sell_price": 100.0, "avg_cost": 100.0,
            "reason": ""}


class TestExitDispositionReportVerdicts:
    def test_insufficient_data_below_min_total(self):
        # Fewer than MIN_TOTAL paired rows → INSUFFICIENT_DATA.
        pairs = [_mk_pair("manual_sell", -2.0) for _ in range(MIN_TOTAL - 1)]
        rep = exit_disposition_report(pairs)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["n_total"] == MIN_TOTAL - 1

    def test_insufficient_data_no_manual_sell(self):
        # Plenty of SL/TP but no manual_sell → still INSUFFICIENT_DATA.
        pairs = ([_mk_pair("stop_loss", -8.0) for _ in range(20)]
                 + [_mk_pair("take_profit", 15.0) for _ in range(20)])
        rep = exit_disposition_report(pairs)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        # Manual-class mean is still None when nothing in that class.
        assert rep["manual_sell_mean_pct"] is None

    def test_manual_sell_losing(self):
        pairs = ([_mk_pair("stop_loss", -8.0) for _ in range(10)]
                 + [_mk_pair("take_profit", 15.0) for _ in range(10)]
                 + [_mk_pair("manual_sell", -3.0) for _ in range(10)])
        rep = exit_disposition_report(pairs)
        assert rep["verdict"] == "MANUAL_SELL_LOSING"
        assert rep["manual_sell_mean_pct"] == -3.0
        # Fire rates sum to 1.0 modulo per-class rounding (4 decimals).
        # 3 × 1/3 rounded to 4 dp can sum to 0.9999 — within 5e-4 is fine.
        total_share = sum(rep["fire_rate"].values())
        assert abs(total_share - 1.0) < 5e-4

    def test_manual_sell_flat(self):
        pairs = ([_mk_pair("stop_loss", -8.0) for _ in range(10)]
                 + [_mk_pair("take_profit", 15.0) for _ in range(10)]
                 + [_mk_pair("manual_sell", 0.5) for _ in range(10)])
        rep = exit_disposition_report(pairs)
        assert rep["verdict"] == "MANUAL_SELL_FLAT"
        assert abs(rep["manual_sell_mean_pct"] - 0.5) < 1e-6

    def test_manual_sell_winning(self):
        pairs = ([_mk_pair("stop_loss", -8.0) for _ in range(10)]
                 + [_mk_pair("take_profit", 15.0) for _ in range(10)]
                 + [_mk_pair("manual_sell", 5.0) for _ in range(10)])
        rep = exit_disposition_report(pairs)
        assert rep["verdict"] == "MANUAL_SELL_WINNING"
        assert rep["manual_sell_mean_pct"] == 5.0


class TestExitDispositionReportBoundaries:
    def test_manual_at_lower_tolerance_boundary_is_flat(self):
        # mean exactly −MANUAL_EDGE_TOL_PP → FLAT (the |·| ≤ branch).
        pairs = ([_mk_pair("stop_loss", -8.0) for _ in range(10)]
                 + [_mk_pair("take_profit", 15.0) for _ in range(10)]
                 + [_mk_pair("manual_sell", -MANUAL_EDGE_TOL_PP)
                    for _ in range(10)])
        rep = exit_disposition_report(pairs)
        assert rep["verdict"] == "MANUAL_SELL_FLAT"

    def test_manual_just_below_lower_tolerance_is_losing(self):
        pairs = ([_mk_pair("stop_loss", -8.0) for _ in range(10)]
                 + [_mk_pair("take_profit", 15.0) for _ in range(10)]
                 + [_mk_pair("manual_sell", -(MANUAL_EDGE_TOL_PP + 0.01))
                    for _ in range(10)])
        rep = exit_disposition_report(pairs)
        assert rep["verdict"] == "MANUAL_SELL_LOSING"

    def test_manual_at_upper_tolerance_boundary_is_flat(self):
        pairs = ([_mk_pair("stop_loss", -8.0) for _ in range(10)]
                 + [_mk_pair("take_profit", 15.0) for _ in range(10)]
                 + [_mk_pair("manual_sell", MANUAL_EDGE_TOL_PP)
                    for _ in range(10)])
        rep = exit_disposition_report(pairs)
        assert rep["verdict"] == "MANUAL_SELL_FLAT"


class TestSlTpResidual:
    def test_sl_residual_when_fired_exactly_at_default(self):
        pairs = ([_mk_pair("stop_loss", -8.0) for _ in range(10)]
                 + [_mk_pair("take_profit", 15.0) for _ in range(10)]
                 + [_mk_pair("manual_sell", 0.0) for _ in range(10)])
        rep = exit_disposition_report(pairs)
        assert rep["sl_default_residual_pp"] == 0.0
        assert rep["tp_default_residual_pp"] == 0.0

    def test_sl_residual_gap_down_slippage(self):
        # All SL fires hit at −10% (gap-down through the −8% trigger).
        pairs = ([_mk_pair("stop_loss", -10.0) for _ in range(10)]
                 + [_mk_pair("take_profit", 15.0) for _ in range(10)]
                 + [_mk_pair("manual_sell", 0.0) for _ in range(10)])
        rep = exit_disposition_report(pairs)
        # residual = mean(-10.0) - (-8.0) = -2.0pp (worse than default)
        assert rep["sl_default_residual_pp"] == -2.0

    def test_residuals_none_when_class_empty(self):
        pairs = [_mk_pair("manual_sell", 0.0) for _ in range(30)]
        rep = exit_disposition_report(pairs)
        assert rep["sl_default_residual_pp"] is None
        assert rep["tp_default_residual_pp"] is None


class TestFireRateAndShares:
    def test_fire_rates_match_class_counts(self):
        pairs = ([_mk_pair("stop_loss", -8.0) for _ in range(20)]
                 + [_mk_pair("take_profit", 15.0) for _ in range(10)]
                 + [_mk_pair("manual_sell", 0.0) for _ in range(70)])
        rep = exit_disposition_report(pairs)
        assert rep["n_total"] == 100
        assert rep["fire_rate"]["stop_loss"] == 0.2
        assert rep["fire_rate"]["take_profit"] == 0.1
        assert rep["fire_rate"]["manual_sell"] == 0.7

    def test_classes_iteration_order_stable(self):
        # The output `classes` list is in (stop_loss, take_profit,
        # manual_sell) order for readability — pin it so a future
        # refactor doesn't silently shuffle the CLI table.
        pairs = [_mk_pair("manual_sell", 0.0) for _ in range(30)]
        rep = exit_disposition_report(pairs)
        order = [c["class"] for c in rep["classes"]]
        assert order == ["stop_loss", "take_profit", "manual_sell"]


# ─────────────────────────── never_raises ───────────────────────────────


class TestNeverRaises:
    def test_empty_input(self):
        rep = exit_disposition_report([])
        assert rep["status"] == "ok"
        assert rep["n_total"] == 0
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_none_input(self):
        rep = exit_disposition_report(None)
        assert rep["status"] == "ok"
        assert rep["n_total"] == 0

    def test_malformed_rows_dropped(self):
        pairs = [
            None,                                 # not a dict
            {"disposition": "stop_loss"},         # no exit_pct
            {"exit_pct": -8.0},                   # no disposition
            {"disposition": "weird", "exit_pct": 0.0},  # unknown class
            {"disposition": "manual_sell", "exit_pct": float("nan")},  # non-finite
        ]
        rep = exit_disposition_report(pairs)
        assert rep["n_total"] == 0
        assert rep["verdict"] == "INSUFFICIENT_DATA"


# ─────────────────────────── analyze_db ─────────────────────────────────


@pytest.fixture
def synthetic_backtest_db(tmp_path) -> Path:
    """A self-contained backtest.db with a few hand-crafted trades for the
    analyze_db integration path.

    Uses the same SCHEMA shape as paper_trader/backtest.py::SCHEMA so the
    analyzer's read query lands on real columns. We only need backtest_trades
    + backtest_runs (the run_id_limit cutoff query uses backtest_runs); skip
    the rest. Standalone — does NOT touch the real backtest.db.
    """
    db = tmp_path / "backtest.db"
    conn = sqlite3.connect(str(db))
    conn.executescript("""
        CREATE TABLE backtest_runs (
            run_id INTEGER PRIMARY KEY,
            seed INTEGER, start_date TEXT, end_date TEXT,
            start_value REAL, final_value REAL, total_return_pct REAL,
            spy_return_pct REAL, vs_spy_pct REAL, n_trades INTEGER,
            n_decisions INTEGER, status TEXT, started_at TEXT,
            completed_at TEXT, equity_curve_json TEXT, notes TEXT
        );
        CREATE TABLE backtest_trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL, sim_date TEXT NOT NULL,
            ticker TEXT NOT NULL, action TEXT NOT NULL,
            qty REAL NOT NULL, price REAL NOT NULL,
            value REAL NOT NULL, reason TEXT
        );
    """)
    # 1 run; one ticker; 40 BUY-and-SELL pairs with a known disposition mix.
    # Need ≥MIN_TOTAL=30 paired SELLs AND ≥MIN_CLASS_N=5 in manual_sell so
    # the verdict reaches MANUAL_SELL_LOSING rather than INSUFFICIENT_DATA.
    conn.execute(
        "INSERT INTO backtest_runs (run_id, seed, start_date, end_date, "
        "start_value, status, started_at) VALUES (?,?,?,?,?,?,?)",
        (1, 42, "2024-01-01", "2025-12-31", 1000.0, "complete",
         "2024-01-01T00:00:00+00:00"),
    )
    trades: list[tuple] = []
    # 40 BUYs at $100, each fully closed by a SELL.
    # Mix: 16 SL fires (40%), 12 TP fires (30%), 12 manual_sell at -3% (30%, LOSING).
    sells = (
        [(-8.0, "stop-loss @ 92 (close 92)")] * 16
        + [(15.0, "take-profit @ 115 (close 115)")] * 12
        + [(-3.0, "ML+quant: trim")] * 12
    )
    # Generate non-overlapping (BUY_date < SELL_date) pairs across months
    # so a 40-pair set fits inside calendar bounds. Each ticker is unique
    # so positions don't blend in the FIFO walker.
    from datetime import date as _date, timedelta as _td
    base = _date(2024, 1, 1)
    for i, (exit_pct, reason) in enumerate(sells):
        buy_d = base + _td(days=i * 7)         # Mondays-ish, 7d apart
        sell_d = buy_d + _td(days=5)            # 5d later
        price_buy = 100.0
        price_sell = 100.0 * (1 + exit_pct / 100.0)
        # Unique ticker per pair → no FIFO blending across pairs.
        tk = f"T{i:03d}"
        trades.append((1, buy_d.isoformat(), tk, "BUY", 10.0, price_buy,
                       10.0 * price_buy, "open"))
        trades.append((1, sell_d.isoformat(), tk, "SELL", 10.0, price_sell,
                       10.0 * price_sell, reason))
    conn.executemany(
        "INSERT INTO backtest_trades (run_id, sim_date, ticker, action, "
        "qty, price, value, reason) VALUES (?,?,?,?,?,?,?,?)",
        trades,
    )
    conn.commit()
    conn.close()
    return db


class TestAnalyzeDb:
    def test_classifies_synthetic_trades(self, synthetic_backtest_db):
        rep = analyze_db(db_path=synthetic_backtest_db)
        assert rep["status"] == "ok"
        assert rep["n_total"] == 40
        # 16 SL / 12 TP / 12 manual_sell
        assert rep["fire_rate"]["stop_loss"] == 0.4
        assert rep["fire_rate"]["take_profit"] == 0.3
        assert rep["fire_rate"]["manual_sell"] == 0.3
        # Manual mean = -3.0% → MANUAL_SELL_LOSING.
        assert rep["verdict"] == "MANUAL_SELL_LOSING"
        # SL fired exactly at default → residual 0.
        assert rep["sl_default_residual_pp"] == 0.0
        assert rep["tp_default_residual_pp"] == 0.0
        assert rep["n_runs_scanned"] == 1

    def test_missing_db_yields_error_status(self, tmp_path):
        rep = analyze_db(db_path=tmp_path / "does_not_exist.db")
        assert rep["status"] == "error"
        assert "no backtest.db" in rep["hint"]

    def test_run_id_limit_filters(self, tmp_path):
        # Build a 2-run DB; --runs 1 must restrict to the most-recent run.
        db = tmp_path / "two_runs.db"
        conn = sqlite3.connect(str(db))
        conn.executescript("""
            CREATE TABLE backtest_runs (
                run_id INTEGER PRIMARY KEY,
                seed INTEGER, start_date TEXT, end_date TEXT,
                start_value REAL, final_value REAL, total_return_pct REAL,
                spy_return_pct REAL, vs_spy_pct REAL, n_trades INTEGER,
                n_decisions INTEGER, status TEXT, started_at TEXT,
                completed_at TEXT, equity_curve_json TEXT, notes TEXT
            );
            CREATE TABLE backtest_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL, sim_date TEXT NOT NULL,
                ticker TEXT NOT NULL, action TEXT NOT NULL,
                qty REAL NOT NULL, price REAL NOT NULL,
                value REAL NOT NULL, reason TEXT
            );
        """)
        # run 1: 30 paired manual_sell at -5%
        # run 2: 30 paired manual_sell at +5%
        for rid, exit_pct in [(1, -5.0), (2, 5.0)]:
            conn.execute(
                "INSERT INTO backtest_runs (run_id, seed, start_date, end_date, "
                "start_value, status, started_at) VALUES (?,?,?,?,?,?,?)",
                (rid, rid, "2025-01-01", "2025-12-31", 1000.0, "complete",
                 f"2025-01-01T00:00:0{rid}+00:00"),
            )
            for i in range(30):
                conn.execute(
                    "INSERT INTO backtest_trades (run_id, sim_date, ticker, "
                    "action, qty, price, value, reason) VALUES (?,?,?,?,?,?,?,?)",
                    (rid, f"2025-01-{i+1:02d}", f"X{i}", "BUY", 1.0, 100.0,
                     100.0, "open"),
                )
                conn.execute(
                    "INSERT INTO backtest_trades (run_id, sim_date, ticker, "
                    "action, qty, price, value, reason) VALUES (?,?,?,?,?,?,?,?)",
                    (rid, f"2025-02-{i+1:02d}", f"X{i}", "SELL", 1.0,
                     100.0 * (1 + exit_pct / 100.0), 0.0, "ML+quant"),
                )
        conn.commit()
        conn.close()

        # --runs 1 → only run 2 (most-recent) → manual mean +5% (WINNING).
        rep1 = analyze_db(db_path=db, run_id_limit=1)
        assert rep1["n_runs_scanned"] == 1
        assert rep1["verdict"] == "MANUAL_SELL_WINNING"
        assert rep1["manual_sell_mean_pct"] == 5.0

        # No limit → both runs combined → mean (−5 + +5)/2 = 0 → FLAT.
        rep_all = analyze_db(db_path=db, run_id_limit=None)
        assert rep_all["n_runs_scanned"] == 2
        assert rep_all["verdict"] == "MANUAL_SELL_FLAT"
        assert abs(rep_all["manual_sell_mean_pct"]) < 1e-6


# ─────────────────────────── CLI ────────────────────────────────────────


class TestCliExitCodes:
    def test_cli_exits_2_on_manual_sell_losing(self, synthetic_backtest_db,
                                                 monkeypatch, capsys):
        # Monkeypatch BACKTEST_DB so the CLI's default path lands on our
        # synthetic DB — mirrors the conftest call-time-resolution rule.
        import paper_trader.backtest as bt
        monkeypatch.setattr(bt, "BACKTEST_DB", synthetic_backtest_db,
                            raising=False)
        from paper_trader.ml.exit_disposition_skill import _cli
        rc = _cli([])
        assert rc == 2

    def test_cli_json_exits_0_when_winning(self, tmp_path, monkeypatch, capsys):
        db = tmp_path / "winning.db"
        conn = sqlite3.connect(str(db))
        conn.executescript("""
            CREATE TABLE backtest_runs (
                run_id INTEGER PRIMARY KEY, seed INTEGER,
                start_date TEXT, end_date TEXT, start_value REAL,
                final_value REAL, total_return_pct REAL, spy_return_pct REAL,
                vs_spy_pct REAL, n_trades INTEGER, n_decisions INTEGER,
                status TEXT, started_at TEXT, completed_at TEXT,
                equity_curve_json TEXT, notes TEXT
            );
            CREATE TABLE backtest_trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id INTEGER NOT NULL, sim_date TEXT NOT NULL,
                ticker TEXT NOT NULL, action TEXT NOT NULL,
                qty REAL NOT NULL, price REAL NOT NULL, value REAL NOT NULL,
                reason TEXT
            );
        """)
        conn.execute(
            "INSERT INTO backtest_runs (run_id, seed, start_date, end_date, "
            "start_value, status, started_at) VALUES (?,?,?,?,?,?,?)",
            (1, 42, "2025-01-01", "2025-12-31", 1000.0, "complete",
             "2025-01-01T00:00:00"),
        )
        # 30 manual SELLs at +5% each → MANUAL_SELL_WINNING.
        for i in range(30):
            conn.execute(
                "INSERT INTO backtest_trades (run_id, sim_date, ticker, "
                "action, qty, price, value, reason) VALUES (?,?,?,?,?,?,?,?)",
                (1, f"2025-01-{i+1:02d}", f"X{i}", "BUY", 1.0, 100.0, 100.0, "open"),
            )
            conn.execute(
                "INSERT INTO backtest_trades (run_id, sim_date, ticker, "
                "action, qty, price, value, reason) VALUES (?,?,?,?,?,?,?,?)",
                (1, f"2025-02-{i+1:02d}", f"X{i}", "SELL", 1.0, 105.0, 105.0,
                 "ML+quant: trim"),
            )
        conn.commit()
        conn.close()

        import paper_trader.backtest as bt
        monkeypatch.setattr(bt, "BACKTEST_DB", db, raising=False)
        from paper_trader.ml.exit_disposition_skill import _cli
        rc = _cli(["--json"])
        assert rc == 0
        # JSON output is parseable.
        out = capsys.readouterr().out.strip()
        rep = json.loads(out)
        assert rep["verdict"] == "MANUAL_SELL_WINNING"
        assert rep["manual_sell_mean_pct"] == 5.0


# ─────────────────────────── defaults sanity ─────────────────────────────


class TestDefaults:
    def test_sl_default_matches_engine_constant(self):
        # backtest.py::_ml_decide uses `price * 0.92` → −8%.
        assert SL_DEFAULT_PCT == -8.0

    def test_tp_default_matches_engine_constant(self):
        # backtest.py::_ml_decide uses `price * 1.15` → +15%.
        assert TP_DEFAULT_PCT == 15.0

    def test_min_class_n_le_min_total(self):
        # If MIN_CLASS_N > MIN_TOTAL the INSUFFICIENT_DATA branch logic
        # becomes unreachable for valid manual_sell-only distributions.
        assert MIN_CLASS_N <= MIN_TOTAL

    def test_manual_edge_tol_is_positive(self):
        assert MANUAL_EDGE_TOL_PP > 0
