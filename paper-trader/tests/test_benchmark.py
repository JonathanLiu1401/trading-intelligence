"""Tests for the S&P 500 buy-and-hold benchmark
(analytics/benchmark.py + the /api/benchmark endpoint + the reporter line).

These assert *exact behaviour*, not "it returned 200":

* the buy-and-hold arithmetic is locked to **hand-computed** values **and**
  to the real live-book shape (sp0=7444.88, sp1=7409.18, tv=972.69 →
  −2.25pp / −$22.52), so an arithmetic regression fails against a real
  production number, not a self-generated one;
* the sample-size-honesty ladder (NO_DATA → INSUFFICIENT → OK) matches the
  house pattern — numerics still emit under INSUFFICIENT, the verdict is
  withheld;
* the inception anchor is the first row carrying *both* a value and an S&P
  mark (yfinance cold-start robustness), not blindly ``equity_curve[0]``;
* invariant #12 — the builder uses the *passed* ``starting_equity``, never a
  hardcoded 1000 (a literal would fail the init=2000 case);
* the reporter line composes the builder's headline **verbatim** (single
  source of truth, AGENTS.md #10 — a re-derived string would diverge here)
  and its failure degrades to "" while the hourly summary still sends (the
  "no block, never no summary" contract);
* the endpoint and the builder agree on the same store (no drift), and the
  endpoint wires ``INITIAL_CASH``.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import reporter
from paper_trader import store as store_mod
from paper_trader.analytics import benchmark as bm
from paper_trader.analytics.benchmark import build_benchmark
from paper_trader.store import Store

_T0 = datetime(2026, 1, 1, tzinfo=timezone.utc)


def _curve(points, step_h=3.0):
    """points: list of (total_value, sp500_price); timestamps auto-stepped."""
    out = []
    for i, (tv, sp) in enumerate(points):
        out.append({
            "timestamp": (_T0 + timedelta(hours=step_h * i)).isoformat(),
            "total_value": tv, "cash": 0.0, "sp500_price": sp,
        })
    return out


def _ok_curve(init, sp0, sp1, tv1, n=13, step_h=3.0):
    """anchor + (n-2) flat-at-anchor points + latest. With n=13, step 3h the
    span is 12·3 = 36h ≥ _MIN_SPAN_HOURS and n ≥ _MIN_POINTS, so state==OK
    and only the anchor & latest drive the headline numbers."""
    return _curve([(init, sp0)] * (n - 1) + [(tv1, sp1)], step_h=step_h)


# ───────────────────────── pure builder ──────────────────────────

class TestStateLadder:
    def test_no_data_empty(self):
        r = build_benchmark([])
        assert r["state"] == "NO_DATA"
        assert r["verdict"] is None
        assert r["alpha_pp"] is None
        assert "No benchmarkable" in r["headline"]

    def test_no_data_when_no_usable_rows(self):
        # Every row missing an S&P mark or a value → unbenchmarkable.
        rows = [
            {"timestamp": _T0.isoformat(), "total_value": 1000.0,
             "cash": 0.0, "sp500_price": None},
            {"timestamp": _T0.isoformat(), "total_value": None,
             "cash": 0.0, "sp500_price": 5000.0},
        ]
        assert build_benchmark(rows)["state"] == "NO_DATA"

    def test_insufficient_emits_numerics_but_withholds_verdict(self):
        # 4 points, ~9h span → below both gates, but numerics still emitted.
        rows = _curve([(1000, 5000), (1100, 5000), (900, 5000),
                       (1050, 5100)], step_h=3.0)
        r = build_benchmark(rows, starting_equity=1000.0)
        assert r["state"] == "INSUFFICIENT"
        assert r["verdict"] is None
        # latest = (1050, 5100): port +5%, sp +2% → alpha +3pp
        assert r["port_return_pct"] == 5.0
        assert r["sp500_return_pct"] == 2.0
        assert r["alpha_pp"] == 3.0
        # running-alpha extremes & ahead-share computed even under INSUFFICIENT
        assert r["best_alpha_pp"] == 10.0    # pt2: +10% vs flat index
        assert r["worst_alpha_pp"] == -10.0  # pt3: −10% vs flat index
        assert r["pct_cycles_ahead"] == 50.0  # pt2 & pt4 ahead of 4
        assert "maturing" in r["headline"]


class TestVerdictsExactArithmetic:
    def test_beating(self):
        r = build_benchmark(_ok_curve(1000, 5000, 5100, 1100),
                            starting_equity=1000.0)
        assert r["state"] == "OK"
        assert r["verdict"] == "BEATING"
        assert r["port_return_pct"] == 10.0
        assert r["sp500_return_pct"] == 2.0
        assert r["alpha_pp"] == 8.0
        assert r["sp500_equivalent_usd"] == 1020.0
        assert r["usd_vs_sp500"] == 80.0
        assert "Beating buy-and-hold S&P 500 by 8.00pp" in r["headline"]

    def test_lagging(self):
        r = build_benchmark(_ok_curve(1000, 5000, 5100, 990),
                            starting_equity=1000.0)
        assert r["verdict"] == "LAGGING"
        assert r["alpha_pp"] == -3.0
        assert r["sp500_equivalent_usd"] == 1020.0
        assert r["usd_vs_sp500"] == -30.0
        assert "Lagging buy-and-hold S&P 500 by 3.00pp" in r["headline"]

    def test_tracking_band(self):
        # +0.4% port vs +0.2% index → +0.2pp, within ±0.5pp TRACKING band.
        r = build_benchmark(_ok_curve(1000, 5000, 5010, 1004),
                            starting_equity=1000.0)
        assert r["verdict"] == "TRACKING"
        assert r["alpha_pp"] == 0.2
        assert "Tracking buy-and-hold S&P 500 within 0.20pp" in r["headline"]

    def test_real_live_book_shape_locks_arithmetic(self):
        """Regression lock against the *real* 2026-05-17 live book: $1000 →
        $972.69 while ^GSPC 7444.88 → 7409.18. The bot is −2.25pp / −$22.52
        behind a buy-and-hold of the index. A future arithmetic regression
        fails against this real number, not a synthetic one."""
        r = build_benchmark(
            _ok_curve(1000.0, 7444.8798828125, 7409.18017578125,
                      972.6867431640624),
            starting_equity=1000.0)
        assert r["state"] == "OK"
        assert r["verdict"] == "LAGGING"
        assert r["port_return_pct"] == -2.7313
        assert r["sp500_return_pct"] == -0.4795
        assert r["alpha_pp"] == -2.2518
        assert r["sp500_equivalent_usd"] == 995.20
        assert r["usd_vs_sp500"] == -22.52


class TestRobustnessAndInvariants:
    def test_anchor_is_first_usable_not_index_zero(self):
        # Leading null-S&P row + trailing null-value row must be skipped:
        # the anchor/latest are the first/last *usable* rows.
        rows = ([{"timestamp": (_T0 - timedelta(hours=3)).isoformat(),
                  "total_value": 1000.0, "cash": 0.0, "sp500_price": None}]
                + _ok_curve(1000, 5000, 5100, 1100)
                + [{"timestamp": (_T0 + timedelta(hours=999)).isoformat(),
                    "total_value": None, "cash": 0.0, "sp500_price": 9999.0}])
        r = build_benchmark(rows, starting_equity=1000.0)
        assert r["verdict"] == "BEATING"
        assert r["alpha_pp"] == 8.0
        assert r["inception_sp500"] == 5000.0
        assert r["current_sp500"] == 5100.0

    def test_invariant_12_uses_passed_starting_equity(self):
        """A literal 1000 in the builder would make this fail: with
        starting_equity=2000 and tv1=2100 the port return is +5%, not +10%."""
        r = build_benchmark(_ok_curve(2000, 5000, 5100, 2100),
                            starting_equity=2000.0)
        assert r["starting_equity"] == 2000.0
        assert r["port_return_pct"] == 5.0
        assert r["alpha_pp"] == 3.0
        assert r["sp500_equivalent_usd"] == 2040.0
        assert r["usd_vs_sp500"] == 60.0

    def test_never_raises_on_garbage_rows(self):
        rows = [
            {"timestamp": "not-a-date", "total_value": "abc",
             "sp500_price": object()},
            {},  # missing every key
            {"timestamp": _T0.isoformat(), "total_value": 1000.0,
             "cash": 0.0, "sp500_price": 5000.0},
        ]
        r = build_benchmark(rows, starting_equity=1000.0)
        # Only the one clean row survives → single-point span → INSUFFICIENT,
        # but no exception and numerics present.
        assert r["state"] == "INSUFFICIENT"
        assert r["n_points"] == 1

    def test_history_is_bounded_and_downsampled(self):
        rows = _curve([(1000 + i, 5000) for i in range(600)], step_h=0.5)
        r = build_benchmark(rows, starting_equity=1000.0)
        assert len(r["history"]) <= 200
        # last point preserved (the drawdown.py down-sample contract)
        assert r["history"][-1]["ts"] == rows[-1]["timestamp"]


# ─────────────────────── reporter line ────────────────────────────

class _FakeStore:
    def __init__(self, curve):
        self._c = curve

    def equity_curve(self, limit=500):
        return self._c


class TestBenchmarkLine:
    def test_composes_builder_headline_verbatim(self):
        curve = _ok_curve(1000, 5000, 5100, 1100)
        line = reporter._benchmark_line(_FakeStore(curve))
        expected = build_benchmark(curve,
                                   starting_equity=reporter._INITIAL_EQUITY)
        assert "`BEATING`" in line
        # verbatim — the builder owns the string; no drift between Discord,
        # the endpoint and the CLI (single source of truth, #10).
        assert expected["headline"] in line
        assert "**BENCHMARK**" in line

    def test_no_data_suppressed(self):
        assert reporter._benchmark_line(_FakeStore([])) == ""

    def test_fault_degrades_to_empty_never_raises(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("builder fault")
        monkeypatch.setattr(bm, "build_benchmark", _boom)

        class _BadStore:
            def equity_curve(self, limit=500):
                raise RuntimeError("store fault")
        assert reporter._benchmark_line(_BadStore()) == ""

    def test_hourly_summary_still_sends_when_benchmark_builder_raises(
            self, monkeypatch):
        """The 'no block, never no summary' failure contract: a benchmark
        builder fault must drop only its line, never sink the hourly post."""
        def _boom(*a, **k):
            raise RuntimeError("benchmark fault")
        monkeypatch.setattr(bm, "build_benchmark", _boom)
        sent = []
        monkeypatch.setattr(reporter, "_send",
                            lambda msg: sent.append(msg) or True)
        monkeypatch.setattr(reporter.market, "benchmark_sp500", lambda: None)

        class _S:
            def get_portfolio(self):
                return {"total_value": 950.0, "cash": 50.0}

            def open_positions(self):
                return []

            def recent_trades(self, n):
                return []

            def recent_decisions(self, limit=500):
                return []

            def equity_curve(self, limit=500):
                return []
        monkeypatch.setattr(reporter, "get_store", lambda: _S())
        assert reporter.send_hourly_summary() is True
        assert sent and "HOURLY" in sent[0]
        assert "BENCHMARK" not in sent[0]  # the faulting line was dropped


# ─────────────────── Flask endpoint (e2e) ─────────────────────────

@pytest.fixture
def bench_client(tmp_path, monkeypatch):
    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)
    s = Store()
    init = store_mod.INITIAL_CASH
    # Seed equity_curve directly with controlled timestamps spanning >24h
    # (record_equity_point stamps _now(), which would collapse the span):
    # 13 points, +3h apart = 36h. anchor (init, 5000) → latest (1.10·init,
    # 5100): port +10%, index +2% → +8pp BEATING.
    rows = []
    for i in range(12):
        rows.append(((_T0 + timedelta(hours=3 * i)).isoformat(), init,
                     0.0, 5000.0))
    rows.append(((_T0 + timedelta(hours=36)).isoformat(),
                 round(init * 1.10, 6), 0.0, 5100.0))
    with s._lock:
        s.conn.executemany(
            "INSERT INTO equity_curve (timestamp,total_value,cash,"
            "sp500_price) VALUES (?,?,?,?)", rows)
        s.conn.commit()

    from paper_trader import dashboard
    dashboard.app.config["TESTING"] = True
    try:
        with dashboard.app.test_client() as client:
            yield client, s
    finally:
        s.close()


class TestBenchmarkEndpoint:
    def test_ok_state_exact_and_uses_initial_cash(self, bench_client):
        client, _ = bench_client
        resp = client.get("/api/benchmark")
        assert resp.status_code == 200
        d = resp.get_json()
        assert "error" not in d, d
        assert d["state"] == "OK"
        assert d["verdict"] == "BEATING"
        assert d["starting_equity"] == round(store_mod.INITIAL_CASH, 2)
        assert d["port_return_pct"] == 10.0
        assert d["sp500_return_pct"] == 2.0
        assert d["alpha_pp"] == 8.0
        assert d["sp500_equivalent_usd"] == round(
            store_mod.INITIAL_CASH * 1.02, 2)

    def test_endpoint_agrees_with_builder_on_same_store(self, bench_client):
        """Single source of truth: the endpoint must equal the builder run
        on the same store — a divergent inline copy fails here."""
        client, s = bench_client
        d = client.get("/api/benchmark").get_json()
        direct = build_benchmark(s.equity_curve(limit=5000),
                                 starting_equity=store_mod.INITIAL_CASH)
        for k in ("state", "verdict", "alpha_pp", "usd_vs_sp500",
                  "sp500_equivalent_usd", "headline", "n_points"):
            assert d[k] == direct[k], k
