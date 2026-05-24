"""Tests for ``DecisionScorer.feature_group_contributions`` and its
supporting ``FEATURE_GROUP_MAP`` / ``FEATURE_GROUPS`` constants.

Pinned invariants:
  * Group map covers every feature in build_features() output (drift-guard).
  * Every group label resolves to one of the canonical FEATURE_GROUPS.
  * For a trained model, sum-of-group-contributions equals sum-of-per-feature-
    contributions (the algebraic identity that makes the roll-up trustworthy).
  * Untrained / off-distribution / error inputs degrade to honest empty
    payloads — mirrors the parent feature_contributions discipline.
  * share_pct sums to 100.0 when there is any signal at all.
  * Groups are sorted by |contribution| descending.
"""
from __future__ import annotations

import math
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from paper_trader.ml.decision_scorer import (
    DecisionScorer,
    FEATURE_GROUPS,
    FEATURE_GROUP_MAP,
    FEATURE_NAMES,
    N_FEATURES,
    SECTORS,
    SCORER_PATH,
    train_scorer,
)


# ─────────────────────── module-level invariants ───────────────────────


class TestFeatureGroupMapInvariants:
    def test_map_covers_every_feature_name(self):
        """A new build_features feature without a FEATURE_GROUP_MAP entry
        must fail loudly. Module-level assert handles import, this pins it
        at test time too so a `pytest -k feature_group` run catches it."""
        assert set(FEATURE_GROUP_MAP.keys()) == set(FEATURE_NAMES)

    def test_every_group_value_is_in_canonical_set(self):
        """Every mapped group must be one of FEATURE_GROUPS (no typos)."""
        assert set(FEATURE_GROUP_MAP.values()).issubset(set(FEATURE_GROUPS))

    def test_canonical_groups_match_map_values(self):
        """FEATURE_GROUPS must be exactly the set of distinct map values —
        no stale groups, no missing groups."""
        assert set(FEATURE_GROUP_MAP.values()) == set(FEATURE_GROUPS)

    def test_ml_score_in_own_group(self):
        """ml_score is the trade thesis — bucketing it with quant signals
        would hide the most important question (is the thesis or the
        technical signals driving the prediction?)."""
        assert FEATURE_GROUP_MAP["ml_score"] == "ml_score"

    def test_quant_group_has_expected_features(self):
        """The quant group must be exactly the 9 technical signals — the 6
        classic indicators plus the 3 enhanced MACD signals added with the
        MACD-strategy 1.5:1 R:R improvement."""
        quant_features = {f for f, g in FEATURE_GROUP_MAP.items()
                          if g == "quant"}
        assert quant_features == {"rsi", "macd", "mom5", "mom20",
                                  "vol_ratio", "bb_pos",
                                  "ema200_above", "hist_cross_up",
                                  "macd_below_zero_cross"}

    def test_news_group_has_exactly_two_features(self):
        news_features = {f for f, g in FEATURE_GROUP_MAP.items()
                         if g == "news"}
        assert news_features == {"news_urgency", "news_article_count"}

    def test_sector_group_matches_one_hot(self):
        """All 7 sector_* one-hot columns must roll up to 'sector'."""
        sector_features = {f for f, g in FEATURE_GROUP_MAP.items()
                           if g == "sector"}
        assert sector_features == {f"sector_{s}" for s in SECTORS}
        assert len(sector_features) == 7

    def test_regime_group_is_single_feature(self):
        regime_features = {f for f, g in FEATURE_GROUP_MAP.items()
                           if g == "regime"}
        assert regime_features == {"regime_mult"}

    def test_feature_groups_display_order(self):
        """Display order: thesis → quant → regime → news → sector. This
        order is operator-facing (the CLI table) and shouldn't change
        casually."""
        assert FEATURE_GROUPS == ("ml_score", "quant", "regime", "news",
                                  "sector")


# ─────────────────────── untrained / degraded paths ───────────────────────


class TestUntrainedAndDegraded:
    def test_untrained_returns_empty_groups(self, monkeypatch, tmp_path):
        """When no pickle is on disk, feature_group_contributions returns
        a safe empty dict — matches the parent feature_contributions
        discipline (no exception, trained=False, groups=[])."""
        monkeypatch.setattr(
            "paper_trader.ml.decision_scorer.SCORER_PATH",
            tmp_path / "absent.pkl",
        )
        s = DecisionScorer()
        out = s.feature_group_contributions(
            ml_score=2.0, rsi=55.0, macd=0.1, mom5=1.0, mom20=2.0,
            regime_mult=1.0, ticker="NVDA",
        )
        assert out["trained"] is False
        assert out["groups"] == []
        assert out["pred"] == 0.0
        assert out["pred_baseline"] == 0.0
        assert out["off_distribution"] is False


# ─────────────────────── trained-model paths ───────────────────────


def _make_synthetic_records(n: int = 400, seed: int = 7) -> list[dict]:
    """Build a synthetic outcome corpus large enough to train the scorer.
    Mixes tickers across sectors so the sector one-hot has real variance.

    Decorrelates ticker and sim_date so (ticker, sim_date, action) dedup
    keeps the full corpus rather than collapsing to LCM(len(tickers), n_dates)
    unique keys — which would land below the train_scorer min-30 threshold.
    """
    rng = np.random.default_rng(seed)
    tickers = ["NVDA", "AMD", "XOM", "JPM", "LLY", "GLD", "COIN"]
    from datetime import date as _date, timedelta as _td
    base = _date(2024, 1, 1)
    records = []
    for i in range(n):
        tk = tickers[int(rng.integers(0, len(tickers)))]
        d = base + _td(days=int(rng.integers(0, 365)))
        # forward_return correlated weakly with mom5 + ml_score so the
        # model finds *some* signal to lean on.
        mom5 = float(rng.normal(0, 5))
        mlscore = float(rng.uniform(0, 5))
        fr = float(0.3 * mom5 + 0.5 * mlscore + rng.normal(0, 4))
        records.append({
            "ml_score": mlscore,
            "rsi": float(rng.uniform(30, 70)),
            "macd": float(rng.normal(0, 1)),
            "mom5": mom5,
            "mom20": float(rng.normal(0, 6)),
            "regime_mult": float(rng.choice([0.3, 0.6, 1.0])),
            "ticker": tk,
            "sim_date": d.isoformat(),
            "action": "BUY",
            "vol_ratio": float(rng.uniform(0.5, 2.0)),
            "bb_position": float(rng.uniform(-1.5, 1.5)),
            "news_urgency": float(rng.uniform(0, 100)),
            "news_article_count": float(rng.integers(0, 10)),
            "forward_return_5d": fr,
            "return_pct": float(rng.uniform(-30, 30)),
        })
    return records


@pytest.fixture
def trained_scorer(tmp_path, monkeypatch):
    """Train a fresh scorer pickle in tmp_path and return a loaded
    DecisionScorer instance."""
    pickle_path = tmp_path / "scorer.pkl"
    records = _make_synthetic_records()
    result = train_scorer(records, path=pickle_path)
    assert result["status"] == "ok", f"train_scorer failed: {result}"
    monkeypatch.setattr(
        "paper_trader.ml.decision_scorer.SCORER_PATH", pickle_path
    )
    # clear the module-level cache so we actually re-read the new pickle
    from paper_trader.ml.decision_scorer import _LOAD_CACHE
    _LOAD_CACHE.clear()
    return DecisionScorer()


class TestTrainedPath:
    def test_returns_one_row_per_canonical_group(self, trained_scorer):
        out = trained_scorer.feature_group_contributions(
            ml_score=3.0, rsi=55.0, macd=0.5, mom5=2.0, mom20=4.0,
            regime_mult=1.0, ticker="NVDA",
            vol_ratio=1.2, bb_pos=0.3,
            news_urgency=60.0, news_article_count=3.0,
        )
        assert out["trained"] is True
        # NVDA is mapped to sector_tech — every canonical group should
        # appear, since every group has at least one non-zero feature in
        # this row.
        group_names = {g["group"] for g in out["groups"]}
        assert group_names == set(FEATURE_GROUPS)
        assert all("contribution" in g for g in out["groups"])
        assert all("share_pct" in g for g in out["groups"])
        assert all("n_features" in g for g in out["groups"])

    def test_sum_of_groups_equals_sum_of_features(self, trained_scorer):
        """The algebraic identity that makes the roll-up trustworthy."""
        kwargs = dict(
            ml_score=2.5, rsi=42.0, macd=-0.3, mom5=-1.0, mom20=2.0,
            regime_mult=1.0, ticker="JPM",
            vol_ratio=1.1, bb_pos=-0.2,
            news_urgency=40.0, news_article_count=2.0,
        )
        per_feat = trained_scorer.feature_contributions(**kwargs)
        per_group = trained_scorer.feature_group_contributions(**kwargs)
        sum_features = sum(r["contribution"] for r in per_feat["contributions"])
        sum_groups = sum(g["contribution"] for g in per_group["groups"])
        # Both are rounded to 4 decimals in their respective sources, so
        # cumulative rounding can introduce up to ~17 * 0.5e-4 drift.
        assert math.isclose(sum_features, sum_groups, abs_tol=1e-3)

    def test_n_features_per_group_matches_map(self, trained_scorer):
        """The n_features field must report the actual count of features
        in each group (1 / 9 / 1 / 2 / 7) — 9 quant features after the
        MACD-strategy improvement added ema200_above / hist_cross_up /
        macd_below_zero_cross to the quant bucket."""
        out = trained_scorer.feature_group_contributions(
            ml_score=2.0, rsi=55.0, macd=0.1, mom5=1.0, mom20=2.0,
            regime_mult=1.0, ticker="NVDA",
            vol_ratio=1.0, bb_pos=0.0,
            news_urgency=50.0, news_article_count=2.0,
        )
        expected = {"ml_score": 1, "quant": 9, "regime": 1, "news": 2,
                    "sector": 7}
        actual = {g["group"]: g["n_features"] for g in out["groups"]}
        assert actual == expected

    def test_groups_sorted_by_abs_contribution(self, trained_scorer):
        """Display order: dominant group first. A reader scanning the top
        line should see the most-influential group immediately."""
        out = trained_scorer.feature_group_contributions(
            ml_score=4.5, rsi=70.0, macd=0.8, mom5=4.0, mom20=6.0,
            regime_mult=1.0, ticker="NVDA",
            vol_ratio=1.5, bb_pos=0.8,
            news_urgency=90.0, news_article_count=8.0,
        )
        abs_contribs = [abs(g["contribution"]) for g in out["groups"]]
        assert abs_contribs == sorted(abs_contribs, reverse=True)

    def test_share_pct_sums_to_100(self, trained_scorer):
        """A reader interprets share_pct as percentages — they must sum
        to ~100.0 when there's any signal at all (the denominator is
        always positive in this case)."""
        out = trained_scorer.feature_group_contributions(
            ml_score=3.0, rsi=55.0, macd=0.5, mom5=2.0, mom20=4.0,
            regime_mult=1.0, ticker="LLY",
            vol_ratio=1.2, bb_pos=0.3,
            news_urgency=60.0, news_article_count=3.0,
        )
        total = sum(g["share_pct"] for g in out["groups"])
        # Sub-1.0 tolerance — each group's share is rounded to 2 decimals
        # so 5 groups × 0.005 max drift = ~0.025; pad for safety.
        assert math.isclose(total, 100.0, abs_tol=0.1), \
            f"share_pct sum {total} != 100.0"

    def test_share_pct_uses_absolute_value(self, trained_scorer):
        """A group whose features have offsetting positive + negative
        contributions still shows |net|/|all| > 0 — using signed
        contributions in the share denominator would give a misleading
        very-large share for the net group on offsetting signals.

        Sanity check: every share_pct is in [0, 100] and non-negative.
        """
        out = trained_scorer.feature_group_contributions(
            ml_score=2.0, rsi=55.0, macd=0.1, mom5=1.0, mom20=2.0,
            regime_mult=1.0, ticker="NVDA",
            vol_ratio=1.0, bb_pos=0.0,
            news_urgency=50.0, news_article_count=2.0,
        )
        for g in out["groups"]:
            assert 0.0 <= g["share_pct"] <= 100.0
            # Contributions are signed but shares are absolute.
            assert g["share_pct"] >= 0.0


class TestSectorIsolation:
    """The 7 sector one-hot features should ALL belong to 'sector' — a
    sector bias should never be misread as 7 independent signals."""

    def test_all_sector_features_in_one_group(self):
        for s in SECTORS:
            assert FEATURE_GROUP_MAP[f"sector_{s}"] == "sector"

    def test_sector_group_aggregates_across_one_hot(self, trained_scorer):
        """Pick a ticker (NVDA, sector_tech) and verify the sector group's
        contribution equals the sum of all 7 sector_* contributions in
        the per-feature view — the bucketing is honest."""
        kwargs = dict(
            ml_score=2.0, rsi=55.0, macd=0.1, mom5=1.0, mom20=2.0,
            regime_mult=1.0, ticker="NVDA",
            vol_ratio=1.0, bb_pos=0.0,
            news_urgency=50.0, news_article_count=2.0,
        )
        per_feat = trained_scorer.feature_contributions(**kwargs)
        per_group = trained_scorer.feature_group_contributions(**kwargs)
        sector_sum = sum(
            r["contribution"] for r in per_feat["contributions"]
            if r["feature"].startswith("sector_")
        )
        sector_row = next(g for g in per_group["groups"]
                          if g["group"] == "sector")
        assert math.isclose(sector_sum, sector_row["contribution"],
                            abs_tol=1e-3)


class TestPayloadIsJsonSafe:
    """The CLI emits the payload through json.dumps; every field must be
    JSON-safe (no numpy scalars / NaN / Inf leaking through)."""

    def test_payload_dumps_cleanly(self, trained_scorer):
        import json
        out = trained_scorer.feature_group_contributions(
            ml_score=2.0, rsi=55.0, macd=0.1, mom5=1.0, mom20=2.0,
            regime_mult=1.0, ticker="NVDA",
            vol_ratio=1.0, bb_pos=0.0,
            news_urgency=50.0, news_article_count=2.0,
        )
        # allow_nan=False makes json.dumps raise on NaN/Inf — a clean
        # serialisation proves every numeric field is a finite float.
        serialised = json.dumps(out, allow_nan=False)
        assert "trained" in serialised
        assert "groups" in serialised
