"""Regression locks for three ML/backtest seams with real logic that had
*zero* direct test coverage before this pass (verified by grepping every
symbol in tests/ during the 2026-05-16 review):

  1. ``paper_trader.backtest._sector_rotation`` — pure trailing-return
     ranking. Sort direction and the ``start <= 0`` / ``< 2 points`` guards
     are load-bearing (it feeds the Opus prompt's "Sector rotation" line)
     yet were asserted nowhere.
  2. ``paper_trader.backtest._get_decision_scorer`` — the lazy singleton's
     ``_Dummy`` except-path fallback. ``_ml_decide`` calls
     ``scorer.predict(**kwargs)`` with a fixed 11-keyword signature and
     reads ``scorer.is_trained`` / ``_n_train``; a Dummy that doesn't honour
     that exact contract crashes *every* backtest run thread when the real
     scorer fails to import. Nothing pinned the contract.
  3. ``run_continuous_backtests._llm_annotate_outcomes`` — the
     ``allowed_run_ids`` restriction. Its own comment documents a real past
     contamination bug (a verdict derived from the winner/loser run leaking
     onto identically-named trades in the three unreviewed middle runs,
     corrupting their training sample weights). Untested known-pitfall.

All offline/deterministic. No network: the only external dependency
(`anthropic`) is monkeypatched at the module attribute.
"""
from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import pytest

import paper_trader.backtest as bt
import run_continuous_backtests as rcb
from paper_trader.backtest import BacktestRun, _sector_rotation


def _price_cache(prices: dict) -> "bt.PriceCache":
    """Build a bare PriceCache (mirrors the conftest synthetic_prices idiom:
    __new__ + manual attr set, so no yfinance / disk is touched)."""
    cache = bt.PriceCache.__new__(bt.PriceCache)
    cache.tickers = list(prices.keys())
    cache.prices = prices
    cache.trading_days = []
    cache.start = date(2025, 3, 3)
    cache.end = date(2025, 3, 7)
    return cache


# ─────────────────────── _sector_rotation ───────────────────────

class TestSectorRotation:
    SIM = date(2025, 3, 7)

    def test_exact_returns_sorted_descending(self):
        prices = {
            "XLK": {"2025-03-03": 100.0, "2025-03-07": 110.0},  # +10.0%
            "XLE": {"2025-03-03": 100.0, "2025-03-07": 95.0},   #  -5.0%
            "XLF": {"2025-03-03": 200.0, "2025-03-07": 230.0},  # +15.0%
            "XLV": {"2025-03-03": 50.0, "2025-03-07": 50.0},    #   0.0%
            "XLI": {"2025-03-03": 100.0, "2025-03-07": 80.0},   # -20.0%
        }
        out = _sector_rotation(self.SIM, _price_cache(prices))
        # Order is the verdict: a reversed sort (a real, easy regression)
        # would invert this and the prompt would show the worst sector as
        # leading rotation.
        assert [t for t, _ in out] == ["XLF", "XLK", "XLV", "XLE", "XLI"]
        got = dict(out)
        assert got["XLF"] == pytest.approx(15.0)
        assert got["XLK"] == pytest.approx(10.0)
        assert got["XLV"] == pytest.approx(0.0)
        assert got["XLE"] == pytest.approx(-5.0)
        assert got["XLI"] == pytest.approx(-20.0)

    def test_zero_start_and_single_point_sectors_are_dropped(self):
        prices = {
            "XLK": {"2025-03-03": 100.0, "2025-03-07": 120.0},  # +20.0% kept
            "XLE": {"2025-03-03": 0.0, "2025-03-07": 95.0},     # start<=0 → dropped
            "XLF": {"2025-03-07": 200.0},                       # <2 points → dropped
            "XLV": {"2025-03-03": 100.0, "2025-03-07": 90.0},   # -10.0% kept
            "XLI": {"2025-03-03": 100.0, "2025-03-07": 100.0},  #   0.0% kept
        }
        out = _sector_rotation(self.SIM, _price_cache(prices))
        names = [t for t, _ in out]
        assert "XLE" not in names  # divide-by-zero guard (start <= 0: continue)
        assert "XLF" not in names  # insufficient-history guard (len < 2)
        # Descending: XLK +20, XLI 0.0, XLV -10.0.
        assert names == ["XLK", "XLI", "XLV"]
        assert dict(out)["XLK"] == pytest.approx(20.0)
        assert dict(out)["XLI"] == pytest.approx(0.0)
        assert dict(out)["XLV"] == pytest.approx(-10.0)

    def test_future_dated_closes_are_excluded(self):
        # _series_up_to filters d <= sim_date; a close dated AFTER sim_date
        # must not become pairs[-1] (that would be forward-leakage into the
        # prompt's rotation line).
        prices = {
            "XLK": {"2025-03-03": 100.0, "2025-03-07": 110.0,
                    "2025-03-10": 999.0},  # post-sim spike must be ignored
        }
        out = _sector_rotation(self.SIM, _price_cache(prices))
        assert dict(out)["XLK"] == pytest.approx(10.0)  # not (999-100)/100


# ─────────────────── _get_decision_scorer Dummy fallback ───────────────────

class TestDecisionScorerDummyFallback:
    """When the real DecisionScorer import/instantiation raises, the lazy
    singleton must degrade to a Dummy that satisfies the EXACT contract
    `_ml_decide` depends on — not merely 'some object'."""

    def test_dummy_honours_the_exact_ml_decide_contract(self, monkeypatch):
        def _boom(*a, **k):
            raise RuntimeError("forced DecisionScorer init failure")

        # Force the `from .ml.decision_scorer import DecisionScorer as _DS`
        # → `_DS()` line to raise so the except/_Dummy path is taken.
        monkeypatch.setattr(
            "paper_trader.ml.decision_scorer.DecisionScorer", _boom
        )
        prev = bt._DECISION_SCORER
        bt._DECISION_SCORER = None
        try:
            scorer = bt._get_decision_scorer()

            # Contract 1: gate predicate `_scorer.is_trained` is a falsy bool.
            assert scorer.is_trained is False

            # Contract 2: `getattr(_scorer, "_n_train", 0)` (the >=500 gate
            # input in _ml_decide) degrades to 0, never crashes/None.
            assert getattr(scorer, "_n_train", 0) == 0

            # Contract 3: predict() accepts the FULL 11-keyword signature
            # _ml_decide actually calls it with, and returns the no-op 0.0.
            # A Dummy defined with a positional signature would raise here —
            # exactly the regression this locks.
            pred = scorer.predict(
                ml_score=2.0, rsi=50.0, macd=0.1, mom5=1.0, mom20=2.0,
                regime_mult=1.0, ticker="NVDA", vol_ratio=1.0, bb_pos=0.0,
                news_urgency=None, news_article_count=None,
            )
            assert pred == 0.0
            assert isinstance(pred, float)

            # Idempotent: a second call returns the SAME cached dummy
            # (double-checked-locking path), not a fresh object.
            assert bt._get_decision_scorer() is scorer
        finally:
            bt._DECISION_SCORER = prev


# ─────────────────── _llm_annotate_outcomes run isolation ───────────────────

def _fake_anthropic(text: str):
    """Return a drop-in `anthropic.Anthropic` replacement whose
    messages.create(...) yields a single message with `text`."""
    class _Messages:
        def create(self, **kw):
            return SimpleNamespace(content=[SimpleNamespace(text=text)])

    class _Client:
        def __init__(self, *a, **k):
            self.messages = _Messages()

    return _Client


class TestLlmAnnotateRunIsolation:
    def _runs(self):
        winner = BacktestRun(run_id=1, seed=1, start_date="2025-01-01",
                             end_date="2025-12-31", total_return_pct=50.0)
        loser = BacktestRun(run_id=3, seed=3, start_date="2025-01-01",
                            end_date="2025-12-31", total_return_pct=-30.0)
        return winner, loser

    def test_verdict_does_not_leak_to_unreviewed_middle_runs(self, monkeypatch):
        import anthropic

        winner, loser = self._runs()
        # Three runs share the (NVDA, BUY) trade. Only runs 1 (winner) & 3
        # (loser) were summarised in the prompt; run 2 is an unreviewed
        # middle run — its label MUST stay 0 or its training sample weight
        # is corrupted by a verdict it never earned.
        recs = [
            {"run_id": 1, "ticker": "NVDA", "action": "BUY",
             "ml_score": 3.0, "rsi": 40, "forward_return_5d": 5.0},
            {"run_id": 2, "ticker": "NVDA", "action": "BUY",
             "ml_score": 3.0, "rsi": 40, "forward_return_5d": 5.0},
            {"run_id": 3, "ticker": "AMD", "action": "SELL",
             "ml_score": 1.0, "rsi": 70, "forward_return_5d": -2.0},
            {"run_id": 1, "ticker": "TSLA", "action": "BUY",
             "ml_score": 2.0, "rsi": 55, "forward_return_5d": 1.0},
        ]
        text = ("NVDA BUY: ENDORSE strong AI momentum\n"
                "AMD SELL: CONDEMN poor exit timing")
        monkeypatch.setattr(anthropic, "Anthropic", _fake_anthropic(text))

        out = rcb._llm_annotate_outcomes(None, winner, loser, recs, cycle=7)

        by = {(r["run_id"], r["ticker"]): r["llm_quality_label"] for r in out}
        assert by[(1, "NVDA")] == 1     # winner endorsed → +1
        assert by[(3, "AMD")] == -1     # loser condemned → -1
        # The regression lock: identical (NVDA, BUY) in the UNREVIEWED
        # middle run 2 must NOT inherit the winner's +1.
        assert by[(2, "NVDA")] == 0
        # A winner-run trade with no annotation line stays neutral.
        assert by[(1, "TSLA")] == 0
        # setdefault contract: every record carries the key.
        assert all("llm_quality_label" in r for r in out)

    def test_unparseable_response_leaves_all_labels_neutral(self, monkeypatch):
        import anthropic

        winner, loser = self._runs()
        recs = [
            {"run_id": 1, "ticker": "NVDA", "action": "BUY",
             "ml_score": 3.0, "rsi": 40, "forward_return_5d": 5.0},
            {"run_id": 3, "ticker": "AMD", "action": "SELL",
             "ml_score": 1.0, "rsi": 70, "forward_return_5d": -2.0},
        ]
        # No line matches the TICKER ACTION: VERDICT grammar.
        monkeypatch.setattr(
            anthropic, "Anthropic",
            _fake_anthropic("I could not evaluate these trades confidently."),
        )
        out = rcb._llm_annotate_outcomes(None, winner, loser, recs, cycle=1)
        assert [r["llm_quality_label"] for r in out] == [0, 0]

    def test_none_forward_return_does_not_raise_or_drop_batch(self, monkeypatch):
        """Regression lock: a single outcome row with `forward_return_5d=None`
        used to format-string-crash inside the prompt builder
        (``f\"{None:.1f}%\"``), which was caught by the outer except and
        DROPPED THE WHOLE CYCLE'S annotations silently.

        The harden coerces None to 0 only for the prompt display — the
        actual training-time row carries the real None forward.
        Annotations on the OTHER rows must still apply.
        """
        import anthropic

        winner, loser = self._runs()
        recs = [
            # malformed: explicit None forward_return_5d (the live ledger
            # cannot produce this today, but a future record-shape change
            # might. The fix protects against it.)
            {"run_id": 1, "ticker": "NVDA", "action": "BUY",
             "ml_score": None, "rsi": 40, "forward_return_5d": None},
            # well-formed neighbour in the same batch
            {"run_id": 3, "ticker": "AMD", "action": "SELL",
             "ml_score": 1.0, "rsi": 70, "forward_return_5d": -2.0},
        ]
        text = "AMD SELL: CONDEMN poor exit timing"
        monkeypatch.setattr(anthropic, "Anthropic", _fake_anthropic(text))
        out = rcb._llm_annotate_outcomes(None, winner, loser, recs, cycle=1)
        # The malformed row gets neutral. The well-formed row STILL receives
        # its CONDEMN label — the whole batch is not silently dropped.
        by = {(r["run_id"], r["ticker"]): r["llm_quality_label"] for r in out}
        assert by[(3, "AMD")] == -1
        assert by[(1, "NVDA")] == 0
        # The None field is preserved untouched in the returned records.
        nvda = next(r for r in out if r["run_id"] == 1)
        assert nvda["forward_return_5d"] is None


class TestOpusAnnotateOutcomeLookupDefensive:
    """Regression lock for the outcome_lookup builder inside _opus_annotate.

    Previously used direct ``o["sim_date"]`` / ``o["forward_return_5d"]``
    dict access. A malformed record (missing key) would KeyError out of
    this background daemon thread (no outer try wraps lines 2074-2105),
    silently dropping the entire cycle's Opus annotations. The hardened
    code uses ``.get()`` everywhere and only inserts complete entries.
    """

    def test_records_with_missing_keys_do_not_crash(self, monkeypatch, tmp_path):
        """A missing 'sim_date' / 'ticker' / 'forward_return_5d' on ANY record
        must not raise out of `_opus_annotate`. Short-circuits the subprocess
        path so we only exercise the lookup-build phase."""
        # Short-circuit: claude binary "not present" → returns 0 before any
        # outcome_lookup work runs. Instead we directly exercise the helper
        # logic by monkey-patching `shutil.which` to return a path AND
        # mocking subprocess.run to return a valid empty JSON response.
        import shutil
        import subprocess
        from types import SimpleNamespace

        winner = BacktestRun(run_id=42, seed=42,
                             start_date="2025-01-01", end_date="2025-12-31",
                             total_return_pct=10.0)

        # Engine with an in-memory sqlite store containing a single decision.
        import sqlite3
        import paper_trader.backtest as bt
        store = bt.BacktestStore.__new__(bt.BacktestStore)
        store.conn = sqlite3.connect(":memory:")
        store.conn.row_factory = sqlite3.Row
        store.conn.executescript(bt.SCHEMA)
        store.conn.commit()
        import threading as _th
        store._lock = _th.Lock()
        store.conn.execute(
            "INSERT INTO backtest_decisions "
            "(run_id, sim_date, action, ticker, qty, total_value, reasoning, status) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (42, "2025-06-01", "BUY", "NVDA", 1.0, 1000.0, "test", "FILLED"),
        )
        store.conn.commit()
        engine = SimpleNamespace(store=store)

        # Outcome records mixing well-formed, missing keys, and None values.
        outcome_records = [
            # missing sim_date
            {"run_id": 42, "ticker": "NVDA", "forward_return_5d": 5.0},
            # missing ticker
            {"run_id": 42, "sim_date": "2025-06-01", "forward_return_5d": 5.0},
            # missing forward_return_5d
            {"run_id": 42, "sim_date": "2025-06-02", "ticker": "AMD"},
            # explicit None values
            {"run_id": 42, "sim_date": None, "ticker": None,
             "forward_return_5d": None},
            # well-formed (this one SHOULD make it into the lookup)
            {"run_id": 42, "sim_date": "2025-06-01", "ticker": "NVDA",
             "forward_return_5d": 7.5},
            # wrong run_id (filtered out by the run_id == winner.run_id guard)
            {"run_id": 999, "sim_date": "2025-06-01", "ticker": "TSLA",
             "forward_return_5d": 3.0},
        ]

        # Force `shutil.which("claude")` to return a path so the early
        # exit doesn't fire; mock subprocess.run to return a no-op response
        # that has no trade_labels (so no extra annotation rows are written).
        monkeypatch.setattr(rcb.shutil, "which", lambda _: "/usr/bin/claude")
        def _fake_run(*a, **k):
            return SimpleNamespace(
                returncode=0,
                stdout='{"trade_labels": [], "overall_lesson": ""}',
                stderr="",
            )
        monkeypatch.setattr(rcb.subprocess, "run", _fake_run)
        # Redirect WINNER_JSONL to a temp file so we don't pollute real data.
        monkeypatch.setattr(rcb, "WINNER_JSONL", tmp_path / "winner.jsonl")
        # Silence the news context query (it would otherwise try to open
        # the live articles DB).
        monkeypatch.setattr(rcb, "_query_news_context", lambda *a, **k: [])

        # The critical assertion: _opus_annotate runs to completion. Prior
        # to the fix, the first malformed record raised KeyError.
        n_written = rcb._opus_annotate(engine, [winner], cycle=99,
                                       outcome_records=outcome_records)
        # No lesson and no trade_labels → 0 rows. The contract under test
        # is "did not raise", not the write count itself.
        assert isinstance(n_written, int)
        assert n_written == 0
