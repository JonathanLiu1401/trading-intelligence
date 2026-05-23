"""Tests for paper_trader.analytics.frozen_mark_execution_skill.

Pins:
* the CLEAN × OCCASIONAL × FROZEN_MARK_HEAVY × INSUFFICIENT_DATA ladder
* exact-float equality (one penny of price discovery breaks a cluster)
* cluster_min and cluster_span_hours boundary semantics
* per-cluster summary fields (qty_net, action_mix, span)
* window_days exclusion (older than window is dropped)
* envelope key stability across every verdict
* defensive: malformed trades / NaN / inf / wrong types degrade, never raise
* Flask route smoke (separate class so it can be excluded with -k)
* live-data parity: the 2026-05-21 NVDA frozen-mark cluster is detected
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.frozen_mark_execution_skill import (
    DEFAULT_CLUSTER_MIN,
    DEFAULT_CLUSTER_SPAN_HOURS,
    DEFAULT_HEAVY_PCT,
    DEFAULT_OCCASIONAL_PCT,
    DEFAULT_WINDOW_DAYS,
    MIN_TRADES_FOR_VERDICT,
    _identical_price_clusters,
    _normalize_trade,
    _parse_iso,
    _safe_float,
    build_frozen_mark_execution_skill,
)


def _now() -> datetime:
    return datetime(2026, 5, 23, 18, 0, 0, tzinfo=timezone.utc)


def _trade(action: str, ticker: str, hours_ago: float, price: float,
           qty: float = 1.0, now=None) -> dict:
    now = now or _now()
    ts = now - timedelta(hours=hours_ago)
    return {
        "action": action,
        "ticker": ticker,
        "timestamp": ts.isoformat(),
        "price": price,
        "qty": qty,
        "value": qty * price,
    }


_ENVELOPE_KEYS = {
    "as_of", "verdict", "headline", "window_days",
    "thresholds", "stats", "clusters",
}


# ─────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────


class TestParseIso:
    def test_basic_iso(self):
        d = _parse_iso("2026-05-21T01:36:06.684121+00:00")
        assert d is not None
        assert d.year == 2026 and d.hour == 1 and d.tzinfo is not None

    def test_naive_assumed_utc(self):
        d = _parse_iso("2026-05-21T01:36:06.684121")
        assert d is not None
        assert d.tzinfo is not None

    def test_z_suffix(self):
        d = _parse_iso("2026-05-21T01:36:06Z")
        assert d is not None
        assert d.tzinfo is not None

    def test_garbage_returns_none(self):
        assert _parse_iso("garbage") is None
        assert _parse_iso(None) is None
        assert _parse_iso(123) is None
        assert _parse_iso("") is None


class TestSafeFloat:
    def test_int(self):
        assert _safe_float(5) == 5.0

    def test_float(self):
        assert _safe_float(1.25) == 1.25

    def test_bool_rejected(self):
        # bool is an int subclass but must NOT count as a price.
        assert _safe_float(True) is None
        assert _safe_float(False) is None

    def test_nan_rejected(self):
        assert _safe_float(float("nan")) is None

    def test_inf_rejected(self):
        assert _safe_float(float("inf")) is None
        assert _safe_float(float("-inf")) is None

    def test_str_rejected(self):
        # Trade-table row prices are always numeric; do not parse strings.
        assert _safe_float("12.5") is None


class TestNormalizeTrade:
    def test_basic_fill(self):
        n = _normalize_trade(_trade("BUY", "NVDA", 1.0, 223.435))
        assert n is not None
        assert n["ticker"] == "NVDA"
        assert n["action"] == "BUY"
        assert n["price"] == 223.435

    def test_hold_dropped(self):
        assert _normalize_trade(_trade("HOLD", "NVDA", 1.0, 223.435)) is None

    def test_no_decision_dropped(self):
        assert _normalize_trade({"action": "NO_DECISION"}) is None

    def test_lowercase_ticker_uppercased(self):
        n = _normalize_trade(_trade("buy", "nvda", 1.0, 223.435))
        assert n is not None
        assert n["ticker"] == "NVDA"
        assert n["action"] == "BUY"

    def test_zero_price_dropped(self):
        assert _normalize_trade(_trade("BUY", "NVDA", 1.0, 0.0)) is None

    def test_negative_price_dropped(self):
        assert _normalize_trade(_trade("BUY", "NVDA", 1.0, -5.0)) is None

    def test_option_action_kept(self):
        n = _normalize_trade(_trade("BUY_CALL", "NVDA", 1.0, 12.34))
        assert n is not None
        assert n["action"] == "BUY_CALL"


# ─────────────────────────────────────────────────────────────────────
# Cluster detection
# ─────────────────────────────────────────────────────────────────────


class TestClusterDetection:
    def test_two_at_same_price_within_span_clusters(self):
        trades = [
            _normalize_trade(_trade("BUY", "NVDA", 10.0, 223.435)),
            _normalize_trade(_trade("BUY", "NVDA", 5.0, 223.435)),
        ]
        clusters = _identical_price_clusters(trades, cluster_min=2, cluster_span_hours=24.0)
        assert len(clusters) == 1
        c = clusters[0]
        assert c["ticker"] == "NVDA"
        assert c["price"] == 223.435
        assert c["n_trades"] == 2

    def test_single_trade_does_not_cluster(self):
        trades = [_normalize_trade(_trade("BUY", "NVDA", 5.0, 223.435))]
        clusters = _identical_price_clusters(trades, cluster_min=2, cluster_span_hours=24.0)
        assert clusters == []

    def test_one_penny_difference_breaks_cluster(self):
        # 223.435 vs 223.44 — even tiny price discovery splits the group.
        trades = [
            _normalize_trade(_trade("BUY", "NVDA", 10.0, 223.435)),
            _normalize_trade(_trade("BUY", "NVDA", 5.0, 223.44)),
        ]
        clusters = _identical_price_clusters(trades, cluster_min=2, cluster_span_hours=24.0)
        assert clusters == []

    def test_different_ticker_breaks_cluster(self):
        # Same exact price but different tickers — not a frozen-mark
        # cluster (a coincidence on a round-number price).
        trades = [
            _normalize_trade(_trade("BUY", "NVDA", 10.0, 100.0)),
            _normalize_trade(_trade("BUY", "AAPL", 5.0, 100.0)),
        ]
        clusters = _identical_price_clusters(trades, cluster_min=2, cluster_span_hours=24.0)
        assert clusters == []

    def test_gap_larger_than_span_breaks_cluster(self):
        # Two trades at the same exact price 48h apart, span=24 ⇒ not a
        # single cluster (yfinance caches don't last 2 days).
        trades = [
            _normalize_trade(_trade("BUY", "NVDA", 50.0, 223.435)),
            _normalize_trade(_trade("BUY", "NVDA", 1.0, 223.435)),
        ]
        clusters = _identical_price_clusters(trades, cluster_min=2, cluster_span_hours=24.0)
        assert clusters == []

    def test_cluster_min_3_filters_pairs(self):
        # 2 trades at the same price, but cluster_min=3 ⇒ no cluster.
        trades = [
            _normalize_trade(_trade("BUY", "NVDA", 10.0, 223.435)),
            _normalize_trade(_trade("BUY", "NVDA", 5.0, 223.435)),
        ]
        clusters = _identical_price_clusters(trades, cluster_min=3, cluster_span_hours=24.0)
        assert clusters == []

    def test_split_runs_at_same_price_emit_two_clusters(self):
        # 3 trades at $100 within an 8h window, then another 3 trades at
        # $100 SEPARATED by a 50h gap. Each contiguous run is its own
        # cluster — re-visits of the same round number weeks later do
        # not merge.
        trades = [
            _normalize_trade(_trade("BUY", "AAA", 200.0, 100.0)),
            _normalize_trade(_trade("BUY", "AAA", 195.0, 100.0)),
            _normalize_trade(_trade("BUY", "AAA", 192.0, 100.0)),
            _normalize_trade(_trade("BUY", "AAA", 100.0, 100.0)),
            _normalize_trade(_trade("BUY", "AAA", 96.0, 100.0)),
            _normalize_trade(_trade("BUY", "AAA", 93.0, 100.0)),
        ]
        clusters = _identical_price_clusters(trades, cluster_min=2, cluster_span_hours=24.0)
        assert len(clusters) == 2
        for c in clusters:
            assert c["n_trades"] == 3

    def test_cluster_summary_fields(self):
        trades = [
            _normalize_trade(_trade("BUY", "NVDA", 10.0, 223.435, qty=2.0)),
            _normalize_trade(_trade("BUY", "NVDA", 8.0, 223.435, qty=0.5)),
            _normalize_trade(_trade("SELL", "NVDA", 6.0, 223.435, qty=1.5)),
        ]
        clusters = _identical_price_clusters(trades, cluster_min=2, cluster_span_hours=24.0)
        assert len(clusters) == 1
        c = clusters[0]
        assert c["n_trades"] == 3
        assert c["action_mix"] == {"BUY": 2, "SELL": 1}
        assert c["buy_qty"] == 2.5
        assert c["sell_qty"] == 1.5
        # Net = bought - sold = 1.0
        assert c["qty_net"] == 1.0
        # Realized P&L *inside* the cluster is zero by construction
        # (constant price across all rows). This is the structural
        # invariant the skill exists to surface.
        assert c["realized_pnl_inside_cluster"] == 0.0
        assert c["span_hours"] > 0


# ─────────────────────────────────────────────────────────────────────
# Verdict ladder
# ─────────────────────────────────────────────────────────────────────


class TestEnvelopeStability:
    def test_envelope_keys_on_no_data(self):
        out = build_frozen_mark_execution_skill(None, now=_now())
        assert set(out.keys()) >= _ENVELOPE_KEYS
        assert out["verdict"] == "INSUFFICIENT_DATA"
        assert out["clusters"] == []
        assert out["stats"]["n_trades"] == 0

    def test_envelope_keys_on_clean(self):
        # 5 trades all at distinct prices — no cluster — CLEAN.
        trades = [
            _trade("BUY", "NVDA", 10.0 - i, 220.0 + i)
            for i in range(5)
        ]
        out = build_frozen_mark_execution_skill(trades, now=_now())
        assert set(out.keys()) >= _ENVELOPE_KEYS
        assert out["verdict"] == "CLEAN"
        assert out["stats"]["n_clusters"] == 0
        assert out["stats"]["frozen_trade_pct"] == 0.0

    def test_envelope_keys_on_heavy(self):
        # 5 trades all at the SAME exact price within 1h — 100% frozen.
        trades = [
            _trade("BUY", "NVDA", 10.0 - 0.1 * i, 223.435)
            for i in range(5)
        ]
        out = build_frozen_mark_execution_skill(trades, now=_now())
        assert set(out.keys()) >= _ENVELOPE_KEYS
        assert out["verdict"] == "FROZEN_MARK_HEAVY"
        assert out["stats"]["frozen_trade_pct"] == 100.0


class TestVerdictLadder:
    def test_insufficient_data_below_floor(self):
        # 4 trades < MIN_TRADES_FOR_VERDICT (5)
        trades = [
            _trade("BUY", "NVDA", 10.0, 223.435),
            _trade("BUY", "NVDA", 5.0, 223.435),
            _trade("BUY", "AAPL", 3.0, 180.0),
            _trade("SELL", "AAPL", 1.0, 181.0),
        ]
        out = build_frozen_mark_execution_skill(trades, now=_now())
        assert out["verdict"] == "INSUFFICIENT_DATA"
        assert out["stats"]["n_trades"] == 4

    def test_clean(self):
        # 10 trades, none at the same price ⇒ 0% frozen ⇒ CLEAN.
        trades = [
            _trade("BUY", "NVDA", 240.0 - i, 220.0 + i)
            for i in range(10)
        ]
        out = build_frozen_mark_execution_skill(trades, now=_now())
        assert out["verdict"] == "CLEAN"
        assert out["stats"]["frozen_trade_pct"] == 0.0

    def test_occasional(self):
        # 10 trades total. 2 at the same price (20% frozen).
        # default occasional=5, heavy=25 ⇒ 20% lands in OCCASIONAL.
        trades = [
            _trade("BUY", "NVDA", 100.0, 100.0),
            _trade("BUY", "NVDA", 99.0, 100.0),
        ] + [
            _trade("BUY", "AAPL", 50.0 - i, 180.0 + i)
            for i in range(8)
        ]
        out = build_frozen_mark_execution_skill(trades, now=_now())
        assert out["verdict"] == "OCCASIONAL"
        assert 5.0 <= out["stats"]["frozen_trade_pct"] < 25.0

    def test_heavy(self):
        # 5 frozen trades + 5 unique-price trades ⇒ 50% frozen ⇒ HEAVY.
        trades = [
            _trade("BUY", "NVDA", 10.0 - 0.1 * i, 223.435)
            for i in range(5)
        ] + [
            _trade("BUY", "AAPL", 50.0 - i, 180.0 + i)
            for i in range(5)
        ]
        out = build_frozen_mark_execution_skill(trades, now=_now())
        assert out["verdict"] == "FROZEN_MARK_HEAVY"
        assert out["stats"]["frozen_trade_pct"] == 50.0
        assert out["stats"]["unique_tickers_affected"] == 1


class TestThresholdOverrides:
    def test_lower_heavy_promotes_to_heavy(self):
        # 2/10 frozen = 20%. Default heavy=25 ⇒ OCCASIONAL. Force
        # heavy_pct=10 ⇒ HEAVY.
        trades = [
            _trade("BUY", "NVDA", 100.0, 100.0),
            _trade("BUY", "NVDA", 99.0, 100.0),
        ] + [
            _trade("BUY", "AAPL", 50.0 - i, 180.0 + i) for i in range(8)
        ]
        out = build_frozen_mark_execution_skill(
            trades, now=_now(), heavy_pct=10.0,
        )
        assert out["verdict"] == "FROZEN_MARK_HEAVY"

    def test_flipped_thresholds_dont_raise(self):
        # heavy < occasional is invalid; the builder widens
        # occasional rather than raising.
        trades = [
            _trade("BUY", "NVDA", 10.0 - i, 100.0) for i in range(5)
        ]
        out = build_frozen_mark_execution_skill(
            trades, now=_now(), occasional_pct=50.0, heavy_pct=10.0,
        )
        assert out["verdict"] in ("FROZEN_MARK_HEAVY", "OCCASIONAL", "CLEAN")

    def test_higher_cluster_min_dilutes_frozen(self):
        # 2 trades at a single price + 8 unique = 20% frozen.
        # cluster_min=3 ⇒ no cluster ⇒ CLEAN.
        trades = [
            _trade("BUY", "NVDA", 100.0, 100.0),
            _trade("BUY", "NVDA", 99.0, 100.0),
        ] + [
            _trade("BUY", "AAPL", 50.0 - i, 180.0 + i) for i in range(8)
        ]
        out = build_frozen_mark_execution_skill(
            trades, now=_now(), cluster_min=3,
        )
        assert out["verdict"] == "CLEAN"


class TestWindowExclusion:
    def test_old_trades_dropped(self):
        # Two trades at the same price, but BOTH older than window_days=1
        # ⇒ no in-window trades ⇒ INSUFFICIENT_DATA.
        trades = [
            _trade("BUY", "NVDA", 5 * 24, 100.0),
            _trade("BUY", "NVDA", 5 * 24 - 1, 100.0),
        ]
        out = build_frozen_mark_execution_skill(
            trades, now=_now(), window_days=1.0,
        )
        assert out["verdict"] == "INSUFFICIENT_DATA"
        assert out["stats"]["n_trades"] == 0


# ─────────────────────────────────────────────────────────────────────
# Defensive degradation
# ─────────────────────────────────────────────────────────────────────


class TestDefensiveDegradation:
    def test_malformed_trades_never_raise(self):
        # Mixed garbage rows; builder must degrade gracefully.
        trades = [
            None,
            {},
            {"action": "BUY"},                          # no ticker
            {"action": "BUY", "ticker": "NVDA"},        # no price/ts
            {"action": "BUY", "ticker": "NVDA",
             "price": float("nan"), "qty": 1.0,
             "timestamp": _now().isoformat()},
            {"action": "BUY", "ticker": "NVDA",
             "price": float("inf"), "qty": 1.0,
             "timestamp": _now().isoformat()},
            {"action": "BUY", "ticker": "NVDA",
             "price": True, "qty": 1.0,                 # bool reject
             "timestamp": _now().isoformat()},
            _trade("HOLD", "NVDA", 5.0, 100.0),         # HOLD dropped
            _trade("NO_DECISION", "NVDA", 5.0, 100.0),  # dropped
            "not even a dict",
            42,
        ]
        out = build_frozen_mark_execution_skill(trades, now=_now())
        # No exceptions; envelope intact.
        assert set(out.keys()) >= _ENVELOPE_KEYS
        assert out["verdict"] == "INSUFFICIENT_DATA"
        assert out["stats"]["n_trades"] == 0

    def test_zero_window_days_still_emits_envelope(self):
        trades = [_trade("BUY", "NVDA", 0.0, 100.0)] * 5
        out = build_frozen_mark_execution_skill(
            trades, now=_now(), window_days=0.0,
        )
        # Zero window ⇒ cutoff = now ⇒ all trades AT or BEFORE now
        # would be excluded by strict ``<``. We accept the trades at
        # exactly `now` because `ts < cutoff` is the exclusion rule;
        # zero-window is degenerate but must not raise.
        assert set(out.keys()) >= _ENVELOPE_KEYS


# ─────────────────────────────────────────────────────────────────────
# Live-data parity
# ─────────────────────────────────────────────────────────────────────


class TestLiveParityNvdaFrozenMark:
    """Replays the smoking-gun 2026-05-21 NVDA cluster verbatim from
    paper_trader.db on 2026-05-23 and asserts the builder flags it
    HEAVY. If this test fails, either the executor stopped writing
    identical prices or the cluster detector regressed — both should
    surface immediately."""

    LIVE_NVDA_FROZEN_PRICE = 223.43499755859375
    LIVE_NVDA_TRADES = [
        ("BUY",  "2026-05-20T21:10:10.495989+00:00", 1.0),
        ("BUY",  "2026-05-21T00:11:08.699815+00:00", 0.5),
        ("SELL", "2026-05-21T01:13:38.360935+00:00", 4.5),
        ("BUY",  "2026-05-21T01:36:06.684121+00:00", 2.0),
        ("BUY",  "2026-05-21T10:00:54.069646+00:00", 1.0),
    ]

    def _build(self, now=None):
        now = now or datetime(2026, 5, 23, 18, 0, 0, tzinfo=timezone.utc)
        trades = [
            {
                "action": a,
                "ticker": "NVDA",
                "timestamp": ts,
                "price": self.LIVE_NVDA_FROZEN_PRICE,
                "qty": q,
                "value": q * self.LIVE_NVDA_FROZEN_PRICE,
            }
            for a, ts, q in self.LIVE_NVDA_TRADES
        ]
        return build_frozen_mark_execution_skill(trades, now=now)

    def test_detected_as_heavy(self):
        out = self._build()
        assert out["verdict"] == "FROZEN_MARK_HEAVY"

    def test_cluster_count_and_size(self):
        out = self._build()
        assert out["stats"]["n_clusters"] == 1
        assert out["stats"]["worst_cluster_n_trades"] == 5
        assert out["stats"]["unique_tickers_affected"] == 1

    def test_cluster_action_mix(self):
        out = self._build()
        c = out["clusters"][0]
        assert c["action_mix"] == {"BUY": 4, "SELL": 1}
        # Net: BUY 4.5 - SELL 4.5 = 0 ⇒ desk net-flat on this name
        # at the frozen price. The 5 cycles were pure churn.
        assert c["buy_qty"] == 4.5
        assert c["sell_qty"] == 4.5
        assert c["qty_net"] == 0.0


# ─────────────────────────────────────────────────────────────────────
# Flask route
# ─────────────────────────────────────────────────────────────────────


class TestFlaskRoute:
    def test_route_returns_envelope(self):
        from paper_trader.dashboard import app
        client = app.test_client()
        resp = client.get("/api/frozen-mark-execution-skill")
        assert resp.status_code in (200, 500), resp.status_code
        body = resp.get_json()
        assert isinstance(body, dict)
        for k in ("verdict", "headline", "stats", "thresholds", "clusters"):
            assert k in body, f"missing key: {k}"

    def test_route_clamps_invalid_params(self):
        from paper_trader.dashboard import app
        client = app.test_client()
        # garbage params should not crash, defaults should apply
        resp = client.get(
            "/api/frozen-mark-execution-skill"
            "?window_days=garbage&cluster_min=banana&heavy_pct=oops"
        )
        assert resp.status_code in (200, 500)
        body = resp.get_json()
        assert isinstance(body, dict)
        # thresholds object always present
        assert "thresholds" in body
        # cluster_min must have been clamped/coerced to its default
        assert body["thresholds"]["cluster_min"] == DEFAULT_CLUSTER_MIN
