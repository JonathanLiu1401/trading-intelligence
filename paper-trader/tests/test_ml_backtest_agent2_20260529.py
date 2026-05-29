"""Agent 2 (ML+backtests) review — 2026-05-29.

Test-locks for behaviour findings from this review pass. Focuses on
real business-logic correctness (specific expected values), not just
no-crash contracts.

1. **urgency invariant #2 in winner JSONL emitters.** CLAUDE.md §8
   invariant #2 states "backtest articles always have urgency = 0".
   `_inject_and_train` already hardcodes ``0`` on the write to
   ``articles.db`` so production isn't broken, but the upstream
   ``_append_top_decisions`` and ``_opus_annotate`` JSONL writers used
   to emit ``urgency=1`` for the rank-1 run and Opus GOOD/LESSON
   records. A direct JSONL consumer (or any future refactor that
   propagated the field) would silently violate the invariant.
   These tests pin the JSONL writers to ``urgency=0`` so a regression
   that re-introduces the non-zero branch surfaces immediately.

2. **DecisionScorer.predict idempotence**: identical inputs must yield
   identical outputs across repeat calls (no hidden state drift).

3. **build_features sector one-hot exclusivity**: the sector tail is a
   one-hot — exactly one entry is 1.0, the others 0.0.

4. **build_features is total** on missing/None/NaN/inf inputs:
   the returned vector has length ``N_FEATURES`` and is finite.

5. **predict_with_meta off-distribution surfaces ``failed=False``** for
   a legitimately clamped (but finite) prediction. The gate-relevant
   distinction between "failed" and "off-distribution but real" is
   load-bearing for OOS rank-IC honesty.

6. **_compute_decision_outcomes**: when there are zero filled BUY/SELL
   decisions for a run, the helper returns an empty list (no crash,
   no fabricated rows).

7. **Test the urgency=0 invariant in the in-memory output produced by
   _opus_annotate**: simulate an annotation file and verify the JSONL
   writer never emits ``urgency`` > 0.
"""

from __future__ import annotations

import json
import math
import sqlite3
import sys
import tempfile
import threading
from datetime import date, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# Project root on sys.path for test isolation (mirrors sibling agent2 tests).
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from paper_trader.ml.decision_scorer import (
    DecisionScorer,
    N_FEATURES,
    SECTORS,
    PRED_CLAMP_PCT,
    build_features,
    train_scorer,
)


# ──────────────────────────────────────────────────────────────────────────
# Finding 1 — urgency invariant in JSONL writers (CLAUDE.md §8 invariant #2)
# ──────────────────────────────────────────────────────────────────────────


class _FakeStore:
    """Minimal stand-in for BacktestStore so _append_top_decisions can
    read 'filled' decisions for a synthetic run without spinning a real
    sqlite write."""

    def __init__(self, decisions_by_run: dict):
        self._decisions_by_run = decisions_by_run
        self._lock = threading.RLock()
        # `engine.store.conn.execute(...)` is called by _append_top_decisions
        # — we surface a callable that returns a fake cursor with .fetchall().
        outer = self

        class _Conn:
            def execute(self_inner, sql, params):
                run_id = params[0]
                rows = outer._decisions_by_run.get(run_id, [])
                rows_with_attr = [_dictrow(r) for r in rows]

                class _Cursor:
                    def fetchall(self):
                        return rows_with_attr

                return _Cursor()

        self.conn = _Conn()


class _dictrow:
    def __init__(self, d):
        self._d = d

    def __getitem__(self, k):
        return self._d.get(k)


class _FakeRun:
    def __init__(self, run_id: int, total_return_pct: float):
        self.run_id = run_id
        self.total_return_pct = total_return_pct


class _FakeEngine:
    def __init__(self, store):
        self.store = store


def test_append_top_decisions_emits_urgency_zero(tmp_path, monkeypatch):
    """Fix: winner JSONL emits urgency=0 (was 1 for rank=1) per CLAUDE.md §8 #2."""
    import run_continuous_backtests as rcb

    jsonl_path = tmp_path / "winner_training.jsonl"
    monkeypatch.setattr(rcb, "WINNER_JSONL", jsonl_path)

    store = _FakeStore({
        1: [
            {"action": "BUY", "ticker": "NVDA", "sim_date": "2025-06-15",
             "reasoning": "ML+quant: NVDA score=2.5 regime=bull", "qty": 0.5,
             "confidence": 0.75},
            {"action": "SELL", "ticker": "AMD", "sim_date": "2025-06-16",
             "reasoning": "ML+quant: AMD score=-1.0 regime=bull", "qty": 0.5,
             "confidence": 0.65},
        ],
        2: [
            {"action": "BUY", "ticker": "MSFT", "sim_date": "2025-07-01",
             "reasoning": "ML+quant: MSFT score=1.5 regime=bull", "qty": 0.3,
             "confidence": 0.55},
        ],
    })
    engine = _FakeEngine(store)
    top_runs = [_FakeRun(1, 25.0), _FakeRun(2, 10.0)]

    n = rcb._append_top_decisions(engine, top_runs, cycle=99)
    assert n == 3, f"expected 3 records written, got {n}"

    # Every emitted record must have urgency=0 — even the rank=1 run
    # whose prior code branched to urgency=1.
    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    assert len(rows) == 3
    for rec in rows:
        assert rec.get("urgency") == 0, (
            f"CLAUDE.md §8 invariant #2 violated: urgency must be 0 in winner "
            f"JSONL records, got urgency={rec.get('urgency')} for "
            f"rank={rec.get('rank')} ticker={rec.get('ticker')}"
        )


def test_opus_annotate_emits_urgency_zero(tmp_path, monkeypatch):
    """Opus annotation JSONL writes (both 'opus_lesson' and per-trade rows)
    must obey urgency=0. Mirrors the _append_top_decisions test but for the
    Opus annotation code path."""
    import run_continuous_backtests as rcb

    jsonl_path = tmp_path / "winner_training.jsonl"
    monkeypatch.setattr(rcb, "WINNER_JSONL", jsonl_path)

    # Mock the subprocess.run that calls claude to return a synthetic
    # annotation JSON with both a lesson and a GOOD trade label.
    mock_proc = MagicMock()
    mock_proc.returncode = 0
    mock_proc.stdout = json.dumps({
        "overall_lesson": "Buy semis on RSI<35 bounces.",
        "key_patterns": ["mean-revert"],
        "improvement_suggestions": ["tighten stop"],
        "trade_labels": [
            {"action": "BUY", "ticker": "NVDA", "sim_date": "2025-06-15",
             "quality": "GOOD", "rationale": "Clean breakout"},
            {"action": "SELL", "ticker": "AMD", "sim_date": "2025-06-16",
             "quality": "BAD", "rationale": "Sold too early"},
        ],
    })
    mock_proc.stderr = ""

    # _opus_annotate needs an engine.store with a queryable conn for
    # backtest_decisions; same _FakeStore works.
    decisions = [
        {"sim_date": "2025-06-15", "ticker": "NVDA", "action": "BUY",
         "qty": 0.5, "reasoning": "ML+quant: NVDA score=2.5",
         "total_value": 1050.0},
        {"sim_date": "2025-06-16", "ticker": "AMD", "action": "SELL",
         "qty": 0.5, "reasoning": "ML+quant: AMD score=-1.0",
         "total_value": 1040.0},
    ]
    store = _FakeStore({7: decisions})
    engine = _FakeEngine(store)
    winner = _FakeRun(7, 32.5)

    # `claude` CLI presence is checked first via shutil.which; pretend it
    # exists so the function actually proceeds to subprocess.run.
    with patch("shutil.which", return_value="/usr/bin/claude"), \
         patch("subprocess.run", return_value=mock_proc):
        n = rcb._opus_annotate(engine, [winner], cycle=42, outcome_records=[])

    assert n >= 1, f"_opus_annotate should write at least 1 row, got {n}"
    rows = [json.loads(l) for l in jsonl_path.read_text().splitlines() if l.strip()]
    assert rows, "no rows written"
    for rec in rows:
        assert rec.get("urgency") == 0, (
            f"CLAUDE.md §8 invariant #2 violated in _opus_annotate: "
            f"urgency must be 0, got {rec.get('urgency')} for "
            f"type={rec.get('type')} ticker={rec.get('ticker')}"
        )


# ──────────────────────────────────────────────────────────────────────────
# Finding 2 — DecisionScorer predict idempotence
# ──────────────────────────────────────────────────────────────────────────


def test_predict_is_idempotent():
    """Identical feature vector → identical prediction across repeat calls.

    The scorer must be a pure function of its inputs at inference time.
    A hidden state drift would silently bias OOS metrics and the gate
    against random ordering.
    """
    ds = DecisionScorer()
    if not ds.is_trained:
        pytest.skip("no deployed scorer pickle to verify against")

    common = dict(
        ml_score=3.0, rsi=42.0, macd=0.2, mom5=1.5, mom20=4.0,
        regime_mult=1.0, ticker="NVDA", vol_ratio=1.2, bb_pos=0.3,
        news_urgency=70.0, news_article_count=3.0,
        ema200_above=True, hist_cross_up=False, macd_below_zero_cross=False,
    )
    p1 = ds.predict(**common)
    p2 = ds.predict(**common)
    p3 = ds.predict_with_meta(**common)["pred"]
    assert p1 == p2 == p3, (
        f"predict not idempotent: p1={p1}, p2={p2}, p3={p3}"
    )


# ──────────────────────────────────────────────────────────────────────────
# Finding 3 — build_features sector one-hot exclusivity
# ──────────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("ticker,expected_sector", [
    ("NVDA", "tech"),
    ("XOM", "energy"),
    ("JPM", "financials"),
    ("LLY", "healthcare"),
    ("GLD", "commodities"),
    ("BTC-USD", "crypto"),
    ("TM", "other"),  # explicitly INTENTIONALLY_OTHER per docstring
    ("ZZZZ_UNKNOWN_NEW_TICKER", "other"),  # unmapped → fallthrough
])
def test_sector_one_hot_exact(ticker, expected_sector):
    """Exactly one sector one-hot slot fires, mapped to the right sector."""
    feat = build_features(
        ml_score=1.0, rsi=50.0, macd=0.0, mom5=0.0, mom20=0.0,
        regime_mult=1.0, ticker=ticker,
    )
    assert len(feat) == N_FEATURES
    sector_tail = feat[-len(SECTORS):]
    n_hot = sum(1 for v in sector_tail if v == 1.0)
    n_zero = sum(1 for v in sector_tail if v == 0.0)
    assert n_hot == 1, (
        f"sector one-hot must fire exactly once for {ticker} "
        f"(expected {expected_sector}); fired {n_hot} times: {sector_tail}"
    )
    assert n_zero == len(SECTORS) - 1
    hot_idx = sector_tail.index(1.0)
    assert SECTORS[hot_idx] == expected_sector, (
        f"{ticker} mapped to sector {SECTORS[hot_idx]} but expected "
        f"{expected_sector}"
    )


# ──────────────────────────────────────────────────────────────────────────
# Finding 4 — build_features is total under garbage inputs
# ──────────────────────────────────────────────────────────────────────────


def test_build_features_total_under_garbage():
    """build_features tolerates None / NaN / inf / non-numeric inputs by
    falling back to safe defaults, AND the returned vector is finite +
    correctly shaped."""
    feat = build_features(
        ml_score=float("nan"),
        rsi=float("inf"),
        macd=float("-inf"),
        mom5=None,
        mom20="garbage_string",
        regime_mult=float("nan"),
        ticker="NVDA",
        vol_ratio=float("inf"),
        bb_pos=float("nan"),
        news_urgency=float("-inf"),
        news_article_count=None,
        ema200_above="not_a_bool",
        hist_cross_up=None,
        macd_below_zero_cross=42,  # non-bool int
    )
    assert len(feat) == N_FEATURES
    for i, v in enumerate(feat):
        assert math.isfinite(v), f"feature {i} non-finite: {v}"


# ──────────────────────────────────────────────────────────────────────────
# Finding 5 — predict_with_meta clamp semantics: failed vs clamped
# ──────────────────────────────────────────────────────────────────────────


def test_predict_with_meta_failed_false_on_real_clamp():
    """A REAL finite prediction that exceeded ±PRED_CLAMP_PCT must report
    ``failed=False`` (the rank is still trustworthy — only magnitude is
    suspect). Only an exception / non-finite raw result should be
    ``failed=True``."""
    ds = DecisionScorer()
    if not ds.is_trained:
        pytest.skip("no deployed scorer pickle to verify against")

    # Generate predictions across a sweep of inputs — the deployed pickle
    # is documented to produce off-distribution clamps for certain
    # feature combinations.
    saw_clamped_not_failed = False
    saw_normal_not_failed = False
    for ml in (-10.0, 0.0, 5.0, 10.0):
        for rsi in (10.0, 30.0, 50.0, 70.0):
            m = ds.predict_with_meta(
                ml_score=ml, rsi=rsi, macd=0.0, mom5=0.0, mom20=0.0,
                regime_mult=1.0, ticker="NVDA",
            )
            assert isinstance(m.get("pred"), float)
            assert -PRED_CLAMP_PCT <= m["pred"] <= PRED_CLAMP_PCT
            # The CRITICAL invariant: a legitimately produced (finite raw)
            # prediction is failed=False, even if clamped.
            if m.get("clamped"):
                assert m.get("failed") is False, (
                    "clamped real prediction must NOT be failed=True "
                    f"(meta={m})"
                )
                saw_clamped_not_failed = True
            else:
                assert m.get("failed") is False
                saw_normal_not_failed = True

    # If the deployed model never clamps under this sweep, the
    # invariant still holds — but we want at least the normal case to
    # exercise.
    assert saw_normal_not_failed


# ──────────────────────────────────────────────────────────────────────────
# Finding 6 — _compute_decision_outcomes is empty-safe
# ──────────────────────────────────────────────────────────────────────────


def test_compute_decision_outcomes_empty_run_returns_empty():
    """A run with zero FILLED BUY/SELL decisions yields zero outcome rows
    (no fabricated rows, no exceptions)."""
    import run_continuous_backtests as rcb

    # Empty store: run 99 has no decisions.
    fake_store = _FakeStore({})

    class _FakePrices:
        trading_days = [date(2025, 6, 1) + timedelta(days=i) for i in range(50)]

        def resolved_close_date(self, ticker, d):
            return d

        def price_on(self, ticker, d):
            return 100.0

        def returns_pct(self, ticker, a, b):
            return 1.0

    class _FakeEng:
        def __init__(self):
            self.store = fake_store
            self.prices = _FakePrices()

    out = rcb._compute_decision_outcomes(_FakeEng(), [_FakeRun(99, 5.0)])
    assert out == [], f"expected empty outcomes for empty run, got {out!r}"


# ──────────────────────────────────────────────────────────────────────────
# Finding 7 — train_scorer insufficient-data sentinels don't overwrite pickle
# ──────────────────────────────────────────────────────────────────────────


def test_train_scorer_insufficient_data_does_not_overwrite_pickle(tmp_path):
    """A degenerate training batch (empty, or <30 after dedup) must NOT
    overwrite the on-disk pickle — otherwise a single bad cycle wipes the
    deployed model. The train_scorer contract is that it returns a
    ``status`` sentinel and refuses to pickle. Verified by passing a
    custom ``path`` so the test never touches the real deployed file.
    """
    target = tmp_path / "scorer_target.pkl"
    target.write_bytes(b"sentinel_existing_pickle_bytes")  # pre-existing

    # Empty input — "insufficient_data" path.
    result = train_scorer([], path=target)
    assert result.get("status") == "insufficient_data"
    assert result.get("n") == 0
    # Pickle should be untouched.
    assert target.read_bytes() == b"sentinel_existing_pickle_bytes", (
        "train_scorer overwrote pickle on insufficient_data — must refuse"
    )

    # Single record — fails dedup-len-gate (<30).
    small = [{
        "ticker": "NVDA", "sim_date": "2025-06-15", "action": "BUY",
        "ml_score": 1.0, "rsi": 50.0, "macd": 0.0, "mom5": 0.0,
        "mom20": 0.0, "regime_mult": 1.0, "forward_return_5d": 1.0,
    }]
    result2 = train_scorer(small, path=target)
    # After dedup, still 1 record < 30 threshold.
    assert result2.get("status") == "insufficient_after_dedup", (
        f"expected insufficient_after_dedup sentinel, got {result2}"
    )
    assert target.read_bytes() == b"sentinel_existing_pickle_bytes"
