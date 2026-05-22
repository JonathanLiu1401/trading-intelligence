"""Per-regime OOS rank-IC breakdown in the scorer-skill telemetry.

`_oos_rank_metrics` already splits out-of-sample directional skill into an
aggregate + a BUY/SELL breakdown (the conviction gate is BUY-only). This
suite pins the *regime* breakdown added alongside it: the same temporal
holdout, bucketed by the `regime_mult` every `decision_outcomes.jsonl` row
carries (0.3→bear, 0.6→sideways, 1.0→bull-or-unknown — the `regime_audit`
decode convention).

Why it matters to a quant: an aggregate OOS rank-IC near zero (the
documented current state) can hide a scorer that is genuinely skilled in
one regime and inverted in another. Unlike the standalone `regime_audit`
CLI — a point-in-time snapshot on its own independent split — these tokens
ride the per-cycle `scorer_skill_log.jsonl`, so the regime-conditional
skill becomes *trendable* across cycles.
"""
from __future__ import annotations

import run_continuous_backtests as rcb


class _EchoScorer:
    """Trained scorer whose prediction is exactly the passed `ml_score`.

    Lets a test drive `_oos_rank_metrics` to an exact, known rank-IC by
    choosing each record's `ml_score` relative to its realized return.
    """

    is_trained = True

    def predict(self, **kw):
        return float(kw["ml_score"])


def _rec(ml_score, fwd, regime_mult):
    """One OOS outcome record with only the fields `_oos_rank_metrics` reads."""
    return {
        "ml_score": ml_score,
        "rsi": None, "macd": None, "mom5": None, "mom20": None,
        "regime_mult": regime_mult,
        "ticker": "NVDA",
        "vol_ratio": None, "bb_position": None,
        "news_urgency": None, "news_article_count": None,
        "forward_return_5d": fwd,
        "action": "BUY",
    }


def test_regime_buckets_isolate_per_regime_rank_ic():
    """Bull rows perfectly ranked, bear rows perfectly inverted, sideways
    perfectly ranked — each bucket's rank-IC must reflect ONLY its own rows."""
    records = (
        # bull (regime_mult 1.0): pred rises with realized → rank_ic +1
        [_rec(i, 2.0 * i, 1.0) for i in (1, 2, 3, 4, 5)]
        # bear (regime_mult 0.3): pred rises as realized falls → rank_ic -1
        + [_rec(i, 10.0 - 2.0 * i, 0.3) for i in (1, 2, 3, 4)]
        # sideways (regime_mult 0.6): pred rises with realized → rank_ic +1
        + [_rec(i, float(i), 0.6) for i in (1, 2, 3)]
    )
    m = rcb._oos_rank_metrics(_EchoScorer(), records)

    assert m["regime_bull_n"] == 5
    assert m["regime_bull_rank_ic"] == 1.0
    assert m["regime_bear_n"] == 4
    assert m["regime_bear_rank_ic"] == -1.0
    assert m["regime_sideways_n"] == 3
    assert m["regime_sideways_rank_ic"] == 1.0
    # Aggregate still counts every row (regime split is additive).
    assert m["n"] == 12


def test_unrecognized_regime_mult_is_not_bucketed():
    """A row whose regime_mult is not one of {0.3,0.6,1.0} (or is absent)
    lands in no regime bucket — but still counts in the aggregate, and
    nothing crashes."""
    records = [_rec(i, float(i), 0.99) for i in (1, 2, 3)]  # 0.99 ∉ decode table
    m = rcb._oos_rank_metrics(_EchoScorer(), records)
    assert m["n"] == 3
    assert m["regime_bull_n"] == 0
    assert m["regime_sideways_n"] == 0
    assert m["regime_bear_n"] == 0
    # Too few rows ⇒ rank-IC is None, never a fabricated number.
    assert m["regime_bull_rank_ic"] is None


def test_untrained_scorer_yields_regime_keys_with_zero_n():
    """An untrained scorer must still return every regime key (n=0, ic=None)
    so the status-string formatter and parser never KeyError."""

    class _Untrained:
        is_trained = False

        def predict(self, **kw):
            return 0.0

    m = rcb._oos_rank_metrics(_Untrained(), [_rec(1, 1.0, 1.0)])
    for reg in ("bull", "sideways", "bear"):
        assert m[f"regime_{reg}_n"] == 0
        assert m[f"regime_{reg}_rank_ic"] is None


def test_parse_scorer_status_round_trips_regime_tokens():
    """The regime tokens added to the status string must parse back to the
    exact numeric values — and an older status string lacking them must
    degrade to None, not raise."""
    status = (
        "scorer ok train_n=4000 val_rmse=8.90 oos_n=1000 oos_rmse=13.58 "
        "oos_diracc=0.53 oos_ic=+0.08 "
        "oos_bull_n=620 oos_bull_ic=+0.11 "
        "oos_sideways_n=300 oos_sideways_ic=-0.04 "
        "oos_bear_n=80 oos_bear_ic=+0.02"
    )
    p = rcb._parse_scorer_status(status)
    assert p["oos_bull_n"] == 620
    assert p["oos_bull_ic"] == 0.11
    assert p["oos_sideways_n"] == 300
    assert p["oos_sideways_ic"] == -0.04
    assert p["oos_bear_n"] == 80
    assert p["oos_bear_ic"] == 0.02

    legacy = "scorer ok train_n=4000 val_rmse=8.90 oos_n=1000 oos_rmse=13.58"
    pl = rcb._parse_scorer_status(legacy)
    assert pl["oos_bull_n"] is None
    assert pl["oos_bull_ic"] is None
    assert pl["oos_bear_ic"] is None
