"""equity_freshness builder + /api/equity-freshness endpoint.

Asserts EXACT verdicts/values for the four states (NO_DATA / FRESH /
STALE_CURVE / DIVERGED), the cadence-aware market-open/closed staleness
threshold, the "both stale AND value-diverged" gate that keeps normal
mid-cycle drift from spamming, the corrupt-point-skipping anchor, the
clock-stepped-back clamp, and the never-raises degrade contract. Pure — no
DB, no network. Endpoint convention mirrors
tests/test_baseline_compare_endpoint.py (real Flask app, real module math,
deterministic offline data; no :8090 bind, no live DB).
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paper_trader.analytics.equity_freshness as ef

NOW = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)


def _pf(total, last="2026-05-18T12:00:00+00:00", cash=18.49):
    return {"cash": cash, "total_value": total, "positions": [],
            "last_updated": last}


def _ep(ts, tv, cash=18.49):
    return {"timestamp": ts, "total_value": tv, "cash": cash,
            "sp500_price": 7400.0}


# ── NO_DATA ───────────────────────────────────────────────────────────────

def test_no_data_when_portfolio_missing():
    out = ef.build_equity_freshness(None, [_ep("2026-05-18T11:00:00+00:00", 1000.0)],
                                    True, now=NOW)
    assert out["verdict"] == "NO_DATA"
    assert out["live_value"] is None


def test_no_data_when_total_non_numeric():
    out = ef.build_equity_freshness(_pf("not-a-number"),
                                    [_ep("2026-05-18T11:00:00+00:00", 1000.0)],
                                    True, now=NOW)
    assert out["verdict"] == "NO_DATA"


def test_no_data_when_total_non_positive():
    out = ef.build_equity_freshness(_pf(0.0),
                                    [_ep("2026-05-18T11:00:00+00:00", 1000.0)],
                                    True, now=NOW)
    assert out["verdict"] == "NO_DATA"


def test_no_data_when_no_equity_points():
    out = ef.build_equity_freshness(_pf(924.13), [], True, now=NOW)
    assert out["verdict"] == "NO_DATA"
    assert out["curve_value"] is None


def test_no_data_when_all_points_non_positive():
    pts = [_ep("2026-05-18T11:00:00+00:00", -5.0),
           _ep("2026-05-18T11:30:00+00:00", 0.0)]
    out = ef.build_equity_freshness(_pf(924.13), pts, True, now=NOW)
    assert out["verdict"] == "NO_DATA"


# ── DIVERGED (stale AND materially off — the actionable one) ───────────────

def test_diverged_exact_values():
    # live $924.13, recorded $928.92, curve 90m old (>60m open threshold).
    pts = [_ep("2026-05-18T10:30:00+00:00", 928.92)]
    out = ef.build_equity_freshness(_pf(924.13), pts, True, now=NOW)
    assert out["verdict"] == "DIVERGED"
    assert out["live_value"] == 924.13
    assert out["curve_value"] == 928.92
    assert out["delta_usd"] == -4.79
    assert out["delta_pct"] == round(-4.79 / 924.13 * 100.0, 4)
    assert abs(out["delta_pct"]) > 0.5            # over the band
    assert out["curve_age_s"] == 5400.0           # 90 min
    assert out["stale_age_s"] == 3600.0           # market open
    assert "misstates the true account by $4.79" in out["headline"]
    assert "trust /api/portfolio" in out["headline"]


def test_diverged_positive_delta_sign():
    # live ABOVE recorded → positive delta (book grew while curve froze).
    pts = [_ep("2026-05-18T10:00:00+00:00", 900.0)]
    out = ef.build_equity_freshness(_pf(950.0), pts, True, now=NOW)
    assert out["verdict"] == "DIVERGED"
    assert out["delta_usd"] == 50.0
    assert out["delta_pct"] == round(50.0 / 950.0 * 100.0, 4)


# ── STALE_CURVE (stale but book barely moved) ─────────────────────────────

def test_stale_curve_small_divergence():
    # 90m old (stale) but only -$0.92 (~0.099% < 0.5% band) → STALE_CURVE.
    pts = [_ep("2026-05-18T10:30:00+00:00", 928.92)]
    out = ef.build_equity_freshness(_pf(928.0), pts, True, now=NOW)
    assert out["verdict"] == "STALE_CURVE"
    assert out["delta_usd"] == -0.92
    assert abs(out["delta_pct"]) <= 0.5
    assert "barely moved" in out["headline"]


def test_stale_curve_at_exactly_band_is_not_diverged():
    # |delta_pct| EXACTLY == divergence_pct must NOT be DIVERGED (strict >).
    # Choose live so delta_pct rounds to exactly 0.5.
    pts = [_ep("2026-05-18T10:30:00+00:00", 1005.0)]
    out = ef.build_equity_freshness(_pf(1000.0), pts, True, now=NOW)
    # delta = -5.0 ; pct = -0.5 ; abs == 0.5, NOT > 0.5
    assert out["delta_pct"] == -0.5
    assert out["verdict"] == "STALE_CURVE"


# ── FRESH (recent → trustworthy, even if value differs) ───────────────────

def test_fresh_recent_curve_even_with_big_divergence():
    # Only 10m old (< 60m open threshold): NOT stale → FRESH regardless of
    # value (it reconciles next cycle — never alarm on a fresh curve).
    pts = [_ep("2026-05-18T11:50:00+00:00", 928.92)]
    out = ef.build_equity_freshness(_pf(800.0), pts, True, now=NOW)
    assert out["verdict"] == "FRESH"
    assert out["curve_age_s"] == 600.0
    assert "trustworthy" in out["headline"]


def test_fresh_when_curve_ts_unparseable():
    pts = [_ep("garbage-timestamp", 928.92)]
    out = ef.build_equity_freshness(_pf(800.0), pts, True, now=NOW)
    assert out["verdict"] == "FRESH"
    assert out["curve_age_s"] is None
    assert "unparseable" in out["headline"]


def test_future_curve_ts_clamps_age_to_zero():
    # Wall clock stepped back (NTP/VM) → curve_ts after now. Age clamps to
    # 0.0 (not stale) instead of a negative age that reads fresh forever.
    pts = [_ep("2026-05-18T13:00:00+00:00", 928.92)]  # 1h in the future
    out = ef.build_equity_freshness(_pf(800.0), pts, True, now=NOW)
    assert out["curve_age_s"] == 0.0
    assert out["verdict"] == "FRESH"


# ── Cadence-aware threshold (market open vs closed) ───────────────────────

def test_market_closed_uses_longer_stale_threshold():
    # 90m old: STALE when open (>60m) but FRESH when closed (<120m).
    pts = [_ep("2026-05-18T10:30:00+00:00", 928.92)]
    out_open = ef.build_equity_freshness(_pf(924.13), pts, True, now=NOW)
    out_closed = ef.build_equity_freshness(_pf(924.13), pts, False, now=NOW)
    assert out_open["verdict"] == "DIVERGED"
    assert out_open["stale_age_s"] == 3600.0
    assert out_closed["verdict"] == "FRESH"
    assert out_closed["stale_age_s"] == 7200.0
    assert out_closed["market_open"] is False


# ── Corrupt-point skipping + ordering defensiveness ───────────────────────

def test_anchors_to_newest_positive_point_skipping_corrupt():
    # Newest two rows are non-positive (corruption equity_integrity owns) →
    # anchor to the newest POSITIVE recorded point, not a poisoned one.
    pts = [
        _ep("2026-05-18T09:00:00+00:00", 1000.0),
        _ep("2026-05-18T10:00:00+00:00", -5.0),   # corrupt
        _ep("2026-05-18T11:00:00+00:00", 0.0),    # corrupt
    ]
    out = ef.build_equity_freshness(_pf(1000.0), pts, True, now=NOW)
    assert out["curve_value"] == 1000.0
    assert out["curve_ts"] == "2026-05-18T09:00:00+00:00"


def test_unsorted_input_picks_lexically_newest_positive():
    pts = [
        _ep("2026-05-18T11:00:00+00:00", 928.92),   # newest
        _ep("2026-05-18T09:00:00+00:00", 980.0),
        _ep("2026-05-18T10:00:00+00:00", 950.0),
    ]
    out = ef.build_equity_freshness(_pf(924.13), pts, True, now=NOW)
    assert out["curve_value"] == 928.92
    assert out["curve_ts"] == "2026-05-18T11:00:00+00:00"


# ── Never raises ──────────────────────────────────────────────────────────

def test_never_raises_on_garbage_inputs():
    for bad in ("x", 123, [], {}, None):
        out = ef.build_equity_freshness(bad, bad, True, now=NOW)
        assert out["verdict"] in ("NO_DATA", "FRESH", "STALE_CURVE",
                                  "DIVERGED")


def test_never_raises_when_now_defaulted():
    # now=None path uses wall clock; must not raise and must classify.
    pts = [_ep("2026-05-18T11:00:00+00:00", 928.92)]
    out = ef.build_equity_freshness(_pf(924.13), pts, True)
    assert out["verdict"] in ("FRESH", "STALE_CURVE", "DIVERGED")


# ── /api/equity-freshness endpoint (thin, faithful, never 500s a panel) ───

@pytest.fixture
def client(monkeypatch):
    import paper_trader.dashboard as d

    fake_store = MagicMock()
    fake_store.get_portfolio.return_value = _pf(924.13)
    fake_store.equity_curve.return_value = [_ep("2026-05-18T10:30:00+00:00",
                                                928.92)]
    monkeypatch.setattr(d, "get_store", lambda: fake_store)
    import paper_trader.market as mkt
    monkeypatch.setattr(mkt, "is_market_open", lambda *a, **k: True)
    with d.app.test_client() as c:
        yield c


def test_route_exists_and_returns_verdict_shape(client):
    r = client.get("/api/equity-freshness")
    assert r.status_code == 200
    j = r.get_json()
    for k in ("verdict", "headline", "live_value", "curve_value",
              "delta_usd", "delta_pct", "curve_age_s", "stale_age_s",
              "market_open"):
        assert k in j


def test_endpoint_is_faithful_thin_wrapper(client):
    # Endpoint output must equal the builder run on the same inputs (modulo
    # the wall-clock as_of/age) — no re-derivation (invariant #10).
    j = client.get("/api/equity-freshness").get_json()
    assert j["verdict"] == "DIVERGED"          # the fixture's stale+diverged
    assert j["live_value"] == 924.13
    assert j["curve_value"] == 928.92
    assert j["delta_usd"] == -4.79


def test_cors_header_present_for_cross_fetch(client):
    r = client.get("/api/equity-freshness")
    assert r.headers.get("Access-Control-Allow-Origin") == "*"


def test_never_raises_into_the_panel_on_store_failure(monkeypatch):
    import paper_trader.dashboard as d

    def _boom():
        raise RuntimeError("store down")

    monkeypatch.setattr(d, "get_store", _boom)
    with d.app.test_client() as c:
        r = c.get("/api/equity-freshness")
    assert r.status_code == 500
    assert r.get_json()["verdict"] == "ERROR"
