"""Tests for ``_parse_gate_abstention_kind`` and the
``gate_abstention_kind`` outcome-row column.

The legacy ``_parse_gate_decision`` collapses both abstention markers into
a single ``gate_off_dist=True`` boolean for analyzer back-compat. The new
sibling parser disambiguates the cause so a downstream analyzer can
attribute an abstention to the specific guard that fired
(off-distribution clamp vs no-skill kill-switch).

Cross-check invariant: for every row where the gate emitted a
``scorer=`` token (``gate_scorer_pred is not None``),
``gate_off_dist == (gate_abstention_kind is not None)``. Both parsers
read the same reasoning string but expose orthogonal slices of its
abstention semantics.

These tests pin the parser's contract AND the cross-check, so a future
refactor can't silently let the two go out of sync.
"""
from __future__ import annotations

import importlib


def _rcb():
    """Import the module fresh each call so tests can't share state."""
    import run_continuous_backtests as rcb
    return rcb


# ─── _parse_gate_abstention_kind: pure parser contract ───────────────────

class TestParseGateAbstentionKind:
    def test_off_dist_marker_returns_clamp(self):
        rcb = _rcb()
        kind = rcb._parse_gate_abstention_kind(
            "ML+quant: NVDA score=2.50 regime=bull RSI=45 "
            "news_count=0 news_urg=0.0 conviction=25% "
            "scorer=+50.0%(off-dist,gate-skipped)"
        )
        assert kind == "clamp"

    def test_gate_killed_marker_returns_killswitch(self):
        rcb = _rcb()
        kind = rcb._parse_gate_abstention_kind(
            "ML+quant: NVDA score=2.50 regime=bull RSI=45 "
            "news_count=0 news_urg=0.0 conviction=25% "
            "scorer=+5.0%(gate-killed,no-skill)"
        )
        assert kind == "killswitch"

    def test_active_gate_returns_none(self):
        rcb = _rcb()
        kind = rcb._parse_gate_abstention_kind(
            "ML+quant: NVDA score=2.50 regime=bull RSI=45 "
            "news_count=0 news_urg=0.0 conviction=25% scorer=+5.0%"
        )
        assert kind is None

    def test_sub_gate_no_scorer_token_returns_none(self):
        rcb = _rcb()
        # Sub-gate emission has NO scorer= token at all
        kind = rcb._parse_gate_abstention_kind(
            "ML+quant: NVDA score=2.50 regime=bull RSI=45 "
            "news_count=0 news_urg=0.0 conviction=25%"
        )
        assert kind is None

    def test_empty_reasoning_returns_none(self):
        rcb = _rcb()
        assert rcb._parse_gate_abstention_kind(None) is None
        assert rcb._parse_gate_abstention_kind("") is None

    def test_clamp_precedence_when_both_markers(self):
        """Defensive: if a future format ever emitted both markers in one
        reasoning string, ``clamp`` wins — matches the ``_ml_decide``
        if/elif emission order. The current emission can never produce
        both, but the precedence is a pinned contract."""
        rcb = _rcb()
        kind = rcb._parse_gate_abstention_kind(
            "scorer=+99.0%(off-dist,gate-skipped) (gate-killed,no-skill)"
        )
        assert kind == "clamp"

    def test_never_raises_on_non_string_input(self):
        rcb = _rcb()
        # Pure-total: a parser that feeds a per-cycle ledger MUST NOT
        # raise. Bad inputs degrade to None.
        assert rcb._parse_gate_abstention_kind(12345) is None  # type: ignore[arg-type]
        assert rcb._parse_gate_abstention_kind([]) is None  # type: ignore[arg-type]
        assert rcb._parse_gate_abstention_kind({}) is None  # type: ignore[arg-type]

    def test_off_dist_substring_in_unrelated_text_still_matches(self):
        """The marker check is an ``in`` substring search — by design, so a
        ``scorer=...(off-dist,gate-skipped) ...`` reasoning matches even
        with trailing chars. Pinning this behaviour so a future refactor
        doesn't tighten to an over-strict match that drops real rows."""
        rcb = _rcb()
        kind = rcb._parse_gate_abstention_kind(
            "leading text scorer=+50.0%(off-dist,gate-skipped) trailing"
        )
        assert kind == "clamp"


# ─── Cross-check vs _parse_gate_decision ─────────────────────────────────

class TestAbstentionKindCrossCheck:
    """The new ``gate_abstention_kind`` field MUST be consistent with the
    legacy ``gate_off_dist`` boolean. For every BUY row where the gate
    emitted a ``scorer=`` token (``gate_scorer_pred is not None``):

        gate_off_dist == (gate_abstention_kind is not None)

    Both parsers read the same reasoning string, but they expose
    orthogonal slices of its abstention semantics. A drift between them
    silently corrupts every downstream analyzer that joins the columns.
    """

    def test_clamp_marker_consistent_with_legacy(self):
        rcb = _rcb()
        reason = ("ML+quant: NVDA score=2.5 regime=bull RSI=45 "
                  "conviction=25% scorer=+50.0%(off-dist,gate-skipped)")
        pred, off = rcb._parse_gate_decision(reason)
        kind = rcb._parse_gate_abstention_kind(reason)
        assert pred is not None
        assert off is True
        assert kind == "clamp"
        # the invariant:
        assert off == (kind is not None)

    def test_killswitch_marker_consistent_with_legacy(self):
        rcb = _rcb()
        reason = ("ML+quant: NVDA score=2.5 regime=bull RSI=45 "
                  "conviction=25% scorer=+5.0%(gate-killed,no-skill)")
        pred, off = rcb._parse_gate_decision(reason)
        kind = rcb._parse_gate_abstention_kind(reason)
        assert pred is not None
        assert off is True
        assert kind == "killswitch"
        assert off == (kind is not None)

    def test_active_gate_consistent_with_legacy(self):
        rcb = _rcb()
        reason = ("ML+quant: NVDA score=2.5 regime=bull RSI=45 "
                  "conviction=25% scorer=+5.0%")
        pred, off = rcb._parse_gate_decision(reason)
        kind = rcb._parse_gate_abstention_kind(reason)
        assert pred is not None
        assert off is False
        assert kind is None
        assert off == (kind is not None)


# ─── Integration: _compute_decision_outcomes captures the column ────────

class TestComputeDecisionOutcomesEmitsAbstentionKind:
    """Verify the per-cycle outcome capture writes the new field.

    Uses an in-memory engine stub + minimal trading_days so the test
    runs offline (no yfinance, no real backtest.db). Mirrors the
    ``TestComputeDecisionOutcomes`` pattern in tests/test_backtest.py.
    """

    def _make_engine_stub(self, decisions):
        """Build a minimal duck-typed engine the outcome computer can use."""
        from datetime import date
        from types import SimpleNamespace

        # Tiny price cache covering a 30-day calendar with synthetic closes.
        trading_days = [date(2025, 1, 1) + __import__("datetime").timedelta(days=i)
                        for i in range(30)]
        # Use weekdays only (skip weekends) so we don't have to mimic real
        # NYSE calendar — the test doesn't care about specific dates, only
        # that the outcome computer can index forward 5 trading days.
        trading_days = [d for d in trading_days if d.weekday() < 5]

        prices_for_ticker = {
            d.isoformat(): 100.0 + i * 1.0  # linear ramp +1/day
            for i, d in enumerate(trading_days)
        }

        class PriceCache:
            def __init__(self, td, p):
                self.trading_days = td
                self.prices = {"NVDA": p, "SPY": p}

            def price_on(self, ticker, d):
                return self.prices.get(ticker, {}).get(d.isoformat())

            def resolved_close_date(self, ticker, d):
                # Synthetic: every trading day has a close, so resolved == d
                if d.isoformat() in self.prices.get(ticker, {}):
                    return d
                return None

            def returns_pct(self, ticker, s, e):
                ps = self.price_on(ticker, s)
                pe = self.price_on(ticker, e)
                if not ps or not pe:
                    return 0.0
                return (pe - ps) / ps * 100.0

        pc = PriceCache(trading_days, prices_for_ticker)

        # Tiny store stub with a Lock + decisions table emulated as a list.
        import threading

        class _Row:
            def __init__(self, action, ticker, sim_date, reasoning):
                self._d = {"action": action, "ticker": ticker,
                           "sim_date": sim_date, "reasoning": reasoning}

            def __getitem__(self, k):
                return self._d[k]

        class _Cursor:
            def __init__(self, rows):
                self._rows = rows

            def fetchall(self):
                return self._rows

        class _Conn:
            def __init__(self, rows):
                self._rows = rows

            def execute(self, sql, params):
                return _Cursor(self._rows)

        rows = [_Row(d["action"], d["ticker"], d["sim_date"], d["reasoning"])
                for d in decisions]

        class _Store:
            def __init__(self):
                self._lock = threading.Lock()
                self.conn = _Conn(rows)

        engine = SimpleNamespace(prices=pc, store=_Store())
        return engine

    def test_outcome_row_carries_clamp_kind_for_offdist_buy(self):
        from datetime import date
        rcb = _rcb()

        # A BUY whose reasoning carries the off-distribution marker
        sim_date_str = next(
            d.isoformat()
            for d in self._make_engine_stub([]).prices.trading_days
            if d.weekday() < 5
        )
        decisions = [{
            "action": "BUY", "ticker": "NVDA", "sim_date": sim_date_str,
            "reasoning": ("ML+quant: NVDA score=2.5 regime=bull RSI=45 "
                          "conviction=25% scorer=+50.0%(off-dist,gate-skipped)"),
        }]
        engine = self._make_engine_stub(decisions)

        class _Run:
            run_id = 1
            total_return_pct = 0.0
        outcomes = rcb._compute_decision_outcomes(engine, [_Run()])

        assert len(outcomes) == 1
        row = outcomes[0]
        assert row["action"] == "BUY"
        assert row["ticker"] == "NVDA"
        assert row["gate_off_dist"] is True
        assert row["gate_abstention_kind"] == "clamp"
        # Cross-check invariant
        assert row["gate_off_dist"] == (
            row["gate_abstention_kind"] is not None
        )

    def test_outcome_row_carries_killswitch_kind_for_gate_killed_buy(self):
        rcb = _rcb()
        sim_date_str = next(
            d.isoformat()
            for d in self._make_engine_stub([]).prices.trading_days
            if d.weekday() < 5
        )
        decisions = [{
            "action": "BUY", "ticker": "NVDA", "sim_date": sim_date_str,
            "reasoning": ("ML+quant: NVDA score=2.5 regime=bull RSI=45 "
                          "conviction=25% scorer=+5.0%(gate-killed,no-skill)"),
        }]
        engine = self._make_engine_stub(decisions)

        class _Run:
            run_id = 1
            total_return_pct = 0.0
        outcomes = rcb._compute_decision_outcomes(engine, [_Run()])

        assert len(outcomes) == 1
        row = outcomes[0]
        assert row["gate_off_dist"] is True
        assert row["gate_abstention_kind"] == "killswitch"
        assert row["gate_off_dist"] == (
            row["gate_abstention_kind"] is not None
        )

    def test_outcome_row_kind_is_none_when_gate_acted(self):
        rcb = _rcb()
        sim_date_str = next(
            d.isoformat()
            for d in self._make_engine_stub([]).prices.trading_days
            if d.weekday() < 5
        )
        decisions = [{
            "action": "BUY", "ticker": "NVDA", "sim_date": sim_date_str,
            "reasoning": ("ML+quant: NVDA score=2.5 regime=bull RSI=45 "
                          "conviction=25% scorer=+5.0%"),
        }]
        engine = self._make_engine_stub(decisions)

        class _Run:
            run_id = 1
            total_return_pct = 0.0
        outcomes = rcb._compute_decision_outcomes(engine, [_Run()])

        assert len(outcomes) == 1
        row = outcomes[0]
        assert row["gate_off_dist"] is False
        assert row["gate_abstention_kind"] is None
        assert row["gate_off_dist"] == (
            row["gate_abstention_kind"] is not None
        )

    def test_outcome_row_kind_is_none_for_sell(self):
        """SELL reasoning never emits a ``scorer=`` token (gate is BUY-only)
        so the abstention kind must be None — same convention as
        ``gate_scorer_pred`` / ``gate_off_dist`` for SELL rows."""
        rcb = _rcb()
        sim_date_str = next(
            d.isoformat()
            for d in self._make_engine_stub([]).prices.trading_days
            if d.weekday() < 5
        )
        decisions = [{
            "action": "SELL", "ticker": "NVDA", "sim_date": sim_date_str,
            "reasoning": ("ML+quant: NVDA score=-1.5 regime=bull RSI=80 "
                          "news_count=0 news_urg=0.0 — reducing"),
        }]
        # Need a held position for the SELL — but `_compute_decision_outcomes`
        # only reads the decisions table, it doesn't enforce position
        # state. So this works.
        engine = self._make_engine_stub(decisions)

        class _Run:
            run_id = 1
            total_return_pct = 0.0
        outcomes = rcb._compute_decision_outcomes(engine, [_Run()])

        assert len(outcomes) == 1
        row = outcomes[0]
        assert row["action"] == "SELL"
        assert row["gate_scorer_pred"] is None
        assert row["gate_off_dist"] is None
        assert row["gate_abstention_kind"] is None
