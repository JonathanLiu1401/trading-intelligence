"""Unit tests for ``paper_trader.analytics.rotation_skill``.

Mirror the discipline of ``test_cash_redeployment_latency_skill.py`` — pure
builder, deterministic fixtures, tests pin the verdict thresholds AND the
edge cases (window-edge, same-ticker exclusion, cash mismatch, unpriced).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from paper_trader.analytics.rotation_skill import (
    build_rotation_skill,
    MIN_PAIRS_FOR_VERDICT,
    SKILLED_MEDIAN_PP,
    LAZY_MEDIAN_PP,
    NEUTRAL_BAND_PP,
)


NOW = datetime(2026, 5, 1, 16, 0, tzinfo=timezone.utc)


def _trade(ticker, action, days_ago, price, qty=10, hour=10):
    ts = NOW - timedelta(days=days_ago, hours=hour)
    return {
        "ticker": ticker,
        "action": action,
        "timestamp": ts.isoformat(),
        "qty": qty,
        "price": price,
        "value": qty * price,
    }


def _const_prices(table):
    """``table[(ticker, date_str)] -> price``. Falls back to None."""
    def _p(ticker, ts):
        try:
            d = ts.strftime("%Y-%m-%d")
        except Exception:
            return None
        return table.get((ticker, d))
    return _p


# ─── input-validation / total-function contract ─────────────────────────────

class TestBuilderIsTotal:
    def test_none_trades_returns_envelope(self):
        rep = build_rotation_skill(None, price_at=lambda t, ts: None, now=NOW)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["stats"]["n_pairs_scored"] == 0
        assert rep["pairs"] == []

    def test_empty_trades_returns_envelope(self):
        rep = build_rotation_skill([], price_at=lambda t, ts: None, now=NOW)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert "as_of" in rep
        assert "headline" in rep

    def test_non_dict_rows_ignored(self):
        rep = build_rotation_skill(
            [None, "x", 42, ["not", "a", "dict"]],
            price_at=lambda t, ts: None, now=NOW,
        )
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_missing_action_skipped(self):
        rep = build_rotation_skill(
            [{"ticker": "AAA", "timestamp": NOW.isoformat(), "price": 10.0}],
            price_at=lambda t, ts: None, now=NOW,
        )
        assert rep["stats"]["n_sells_in_window"] == 0

    def test_unparseable_timestamp_skipped(self):
        rows = [_trade("AAA", "SELL", 10, 100.0)]
        rows[0]["timestamp"] = "garbage"
        rep = build_rotation_skill(rows, price_at=lambda t, ts: None, now=NOW)
        assert rep["stats"]["n_sells_in_window"] == 0

    def test_price_at_none_collapses_to_unpriced(self):
        rows = [
            _trade("AAA", "SELL", 20, 100.0),
            _trade("BBB", "BUY", 20, 50.0, hour=8),
        ]
        rep = build_rotation_skill(rows, price_at=None, now=NOW)
        assert rep["stats"]["n_pairs_detected"] == 1
        assert rep["stats"]["n_pairs_scored"] == 0
        assert rep["stats"]["n_unpriced"] == 1


# ─── pair detection ─────────────────────────────────────────────────────────

class TestPairDetection:
    def test_same_ticker_rebuy_not_a_rotation(self):
        rows = [
            _trade("AAA", "SELL", 20, 100.0, hour=12),
            _trade("AAA", "BUY", 20, 99.0, hour=10),  # same ticker
        ]
        rep = build_rotation_skill(rows, price_at=lambda t, ts: 110.0, now=NOW)
        assert rep["stats"]["n_pairs_detected"] == 0

    def test_buy_outside_pairing_window_skipped(self):
        rows = [
            _trade("AAA", "SELL", 20, 100.0, hour=12),
            # BUY is 2 days later — way beyond default 24h pairing window
            _trade("BBB", "BUY", 18, 50.0, hour=10),
        ]
        rep = build_rotation_skill(rows, price_at=lambda t, ts: 110.0, now=NOW)
        assert rep["stats"]["n_pairs_detected"] == 0

    def test_first_diff_buy_after_sell_is_the_pair(self):
        rows = [
            _trade("AAA", "SELL", 20, 100.0, hour=12),
            _trade("AAA", "BUY", 20, 99.0, hour=11),    # same-ticker — skip
            _trade("BBB", "BUY", 20, 50.0, hour=10),    # this is the pair
            _trade("CCC", "BUY", 20, 40.0, hour=9),     # later, ignored
        ]
        prices = _const_prices({
            ("AAA", (NOW - timedelta(days=15)).strftime("%Y-%m-%d")): 100.0,
            ("BBB", (NOW - timedelta(days=15)).strftime("%Y-%m-%d")): 50.0,
        })
        rep = build_rotation_skill(rows, price_at=prices, now=NOW)
        scored = [p for p in rep["pairs"] if p["status"] == "SCORED"]
        assert len(scored) == 1
        assert scored[0]["sell_ticker"] == "AAA"
        assert scored[0]["buy_ticker"] == "BBB"

    def test_window_edge_excluded_from_score(self):
        # SELL 2 days ago — forward window has NOT elapsed (default 5d).
        rows = [
            _trade("AAA", "SELL", 2, 100.0),
            _trade("BBB", "BUY", 2, 50.0, hour=8),
        ]
        rep = build_rotation_skill(rows, price_at=lambda t, ts: 110.0, now=NOW)
        assert rep["stats"]["n_window_edge"] == 1
        assert rep["stats"]["n_pairs_scored"] == 0
        assert any(p["status"] == "WINDOW_EDGE" for p in rep["pairs"])

    def test_cash_mismatch_excluded(self):
        # SELL $1000 → BUY $50: ratio 0.05, way below cash_ratio_lo
        rows = [
            _trade("AAA", "SELL", 20, 100.0, qty=10),
            _trade("BBB", "BUY", 20, 50.0, qty=1, hour=8),
        ]
        rep = build_rotation_skill(rows, price_at=lambda t, ts: 110.0, now=NOW)
        assert rep["stats"]["n_pairs_detected"] == 1
        assert rep["stats"]["n_pairs_scored"] == 0
        assert any(p["status"] == "CASH_MISMATCH" for p in rep["pairs"])

    def test_cash_ratio_in_band_keeps_pair(self):
        # SELL $1000 → BUY $1200: ratio 1.2 — inside default (0.3, 3.0)
        rows = [
            _trade("AAA", "SELL", 20, 100.0, qty=10),
            _trade("BBB", "BUY", 20, 60.0, qty=20, hour=8),
        ]
        prices = _const_prices({
            ("AAA", (NOW - timedelta(days=15)).strftime("%Y-%m-%d")): 105.0,
            ("BBB", (NOW - timedelta(days=15)).strftime("%Y-%m-%d")): 63.0,
        })
        rep = build_rotation_skill(rows, price_at=prices, now=NOW)
        assert rep["stats"]["n_pairs_scored"] == 1


# ─── verdict ladder ─────────────────────────────────────────────────────────

class TestVerdictLadder:
    def _three_pairs_with_alpha(self, alphas_pp: list[float]):
        """Build N pairs each with a deterministic alpha_pp. SELLs at 100.0
        with no forward change; bought at 100.0 with forward = 100 + alpha."""
        rows = []
        price_table = {}
        for i, alpha in enumerate(alphas_pp):
            sell_t = f"S{i}"
            buy_t = f"B{i}"
            sell_day = 15 + i * 2
            rows.append(_trade(sell_t, "SELL", sell_day, 100.0, hour=12))
            rows.append(_trade(buy_t, "BUY", sell_day, 100.0, hour=10))
            fwd_date = (NOW - timedelta(days=sell_day - 5)).strftime("%Y-%m-%d")
            price_table[(sell_t, fwd_date)] = 100.0   # sold ticker flat
            price_table[(buy_t, fwd_date)] = 100.0 + alpha  # bought ticker = alpha
        return rows, _const_prices(price_table)

    def test_insufficient_data_under_min_pairs(self):
        rows, p = self._three_pairs_with_alpha([2.0, 2.0])
        rep = build_rotation_skill(rows, price_at=p, now=NOW)
        assert rep["stats"]["n_pairs_scored"] == 2
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_skilled_rotation_strong_positive(self):
        # All 4 pairs strongly positive: median = 3.0pp, pos% = 100
        rows, p = self._three_pairs_with_alpha([3.0, 3.5, 2.5, 4.0])
        rep = build_rotation_skill(rows, price_at=p, now=NOW)
        assert rep["verdict"] == "SKILLED_ROTATION"
        assert rep["stats"]["positive_alpha_pct"] == 100.0
        assert rep["stats"]["median_alpha_pp"] >= SKILLED_MEDIAN_PP

    def test_lazy_rotation_strong_negative(self):
        rows, p = self._three_pairs_with_alpha([-3.0, -2.5, -3.5, -2.0])
        rep = build_rotation_skill(rows, price_at=p, now=NOW)
        assert rep["verdict"] == "LAZY_ROTATION"
        assert rep["stats"]["negative_alpha_pct"] == 100.0
        assert rep["stats"]["median_alpha_pp"] <= LAZY_MEDIAN_PP

    def test_net_negative_mild_negative(self):
        # median ~ -0.75pp — past -0.3pp band but not lazy
        rows, p = self._three_pairs_with_alpha([-0.5, -1.0, -0.5, -1.0])
        rep = build_rotation_skill(rows, price_at=p, now=NOW)
        assert rep["verdict"] == "NET_NEGATIVE"

    def test_net_positive_mild_positive(self):
        rows, p = self._three_pairs_with_alpha([0.5, 1.0, 0.5, 1.0])
        rep = build_rotation_skill(rows, price_at=p, now=NOW)
        assert rep["verdict"] == "NET_POSITIVE"

    def test_neutral_within_band(self):
        rows, p = self._three_pairs_with_alpha([0.1, -0.1, 0.05, -0.05])
        rep = build_rotation_skill(rows, price_at=p, now=NOW)
        assert rep["verdict"] == "NEUTRAL"
        assert abs(rep["stats"]["median_alpha_pp"]) < NEUTRAL_BAND_PP

    def test_skilled_blocked_when_pos_pct_below_threshold(self):
        # high median but only 50% positive → falls back to NET_POSITIVE
        rows, p = self._three_pairs_with_alpha([5.0, 5.0, -5.0, -1.0])
        rep = build_rotation_skill(rows, price_at=p, now=NOW)
        assert rep["verdict"] != "SKILLED_ROTATION"

    def test_lazy_blocked_when_neg_pct_below_threshold(self):
        rows, p = self._three_pairs_with_alpha([-5.0, -5.0, 5.0, 1.0])
        rep = build_rotation_skill(rows, price_at=p, now=NOW)
        assert rep["verdict"] != "LAZY_ROTATION"


# ─── envelope shape / SSOT discipline ───────────────────────────────────────

class TestEnvelopeShape:
    def test_envelope_keys_always_present(self):
        rep = build_rotation_skill([], price_at=lambda t, ts: None, now=NOW)
        for k in ("verdict", "headline", "as_of", "window_days",
                  "forward_days", "pairing_max_h", "stats", "thresholds",
                  "pairs"):
            assert k in rep, f"missing top-level key: {k}"

    def test_thresholds_block_carries_used_values(self):
        rep = build_rotation_skill(
            [], price_at=lambda t, ts: None, now=NOW,
            cash_ratio_lo=0.1, cash_ratio_hi=10.0,
        )
        assert rep["thresholds"]["cash_ratio_lo"] == 0.1
        assert rep["thresholds"]["cash_ratio_hi"] == 10.0
        assert rep["thresholds"]["min_pairs_for_verdict"] == MIN_PAIRS_FOR_VERDICT

    def test_alpha_definition_is_buy_minus_sell(self):
        # Sold flat, bought up 4% → alpha must be +4
        sell_day = 20
        rows = [
            _trade("AAA", "SELL", sell_day, 100.0, hour=12),
            _trade("BBB", "BUY", sell_day, 100.0, hour=10),
        ]
        fwd_date = (NOW - timedelta(days=sell_day - 5)).strftime("%Y-%m-%d")
        prices = _const_prices({
            ("AAA", fwd_date): 100.0,
            ("BBB", fwd_date): 104.0,
        })
        rep = build_rotation_skill(rows, price_at=prices, now=NOW)
        scored = [p for p in rep["pairs"] if p["status"] == "SCORED"]
        assert len(scored) == 1
        assert scored[0]["sold_forward_pct"] == pytest.approx(0.0, abs=0.01)
        assert scored[0]["bought_forward_pct"] == pytest.approx(4.0, abs=0.01)
        assert scored[0]["rotation_alpha_pp"] == pytest.approx(4.0, abs=0.01)

    def test_pairs_sorted_newest_first(self):
        rows = []
        price_table = {}
        for i in range(3):
            sell_day = 10 + i * 5
            rows.append(_trade(f"S{i}", "SELL", sell_day, 100.0, hour=12))
            rows.append(_trade(f"B{i}", "BUY", sell_day, 100.0, hour=10))
            fwd_date = (NOW - timedelta(days=sell_day - 5)).strftime("%Y-%m-%d")
            price_table[(f"S{i}", fwd_date)] = 100.0
            price_table[(f"B{i}", fwd_date)] = 102.0
        rep = build_rotation_skill(rows, price_at=_const_prices(price_table), now=NOW)
        ts_order = [p["sell_ts"] for p in rep["pairs"]]
        assert ts_order == sorted(ts_order, reverse=True)

    def test_never_raises_on_garbage_price_at(self):
        rows = [
            _trade("AAA", "SELL", 20, 100.0),
            _trade("BBB", "BUY", 20, 50.0, hour=8),
        ]
        def _explode(_t, _ts):
            raise RuntimeError("intentional probe")
        rep = build_rotation_skill(rows, price_at=_explode, now=NOW)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["stats"]["n_unpriced"] == 1

    def test_sells_outside_window_skipped(self):
        # SELL 90 days ago; default window is 60d
        rows = [
            _trade("AAA", "SELL", 90, 100.0),
            _trade("BBB", "BUY", 90, 50.0, hour=8),
        ]
        rep = build_rotation_skill(rows, price_at=lambda t, ts: 110.0, now=NOW)
        assert rep["stats"]["n_sells_in_window"] == 0
