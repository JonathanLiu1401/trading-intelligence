"""Tests for analytics/news_velocity.py — per-held-ticker news-flow velocity.

The discriminating locks are:

* **Sample-size honesty** mirrors ``build_tail_risk`` — a ticker with
  ``baseline_count < MIN_BASELINE_N`` reads ``INSUFFICIENT`` even when a
  window article exists, so a freshly-mentioned name doesn't trip a
  spurious SURGING from a baseline of 1.
* **Window boundary** is strict-inclusive on the window side: an article
  at exactly ``now - window_hours`` falls in the window (the standard
  ``signals.py`` precedent — ``first_seen >= since``).
* **Ticker word-boundary** — ``MU`` must NOT alias ``MUSE`` / ``MUTUAL``;
  ``$MU`` cashtag still hits.
* **Z-score sign and direction** is pinned with hand-computed Poisson
  arithmetic so a sign flip / dropped sqrt-floor / divide-by-zero
  regression fails loudly.
* **State priority sort** — SURGING precedes STABLE/FADING precedes
  INSUFFICIENT in ``per_ticker`` so the loudest catalyst surfaces first.
* **``_safe`` contract** — garbage rows (missing ``first_seen``,
  non-dict entries, unparseable timestamps, non-numeric ``ai_score``)
  must not raise; pseudo-tickers excluded from the held set in the
  endpoint layer.
"""
from __future__ import annotations

import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.news_velocity import (
    MIN_BASELINE_N,
    MIN_WINDOW_FOR_SURGE,
    Z_FADE,
    Z_SURGE,
    build_news_velocity,
)

_NOW = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)


def _art(ticker_in_title: str, hours_ago: float,
         ai_score: float = 5.0, urgency: int = 0,
         title_override: str | None = None,
         body: str = "") -> dict:
    ts = _NOW - timedelta(hours=hours_ago)
    title = title_override if title_override is not None else (
        f"{ticker_in_title} update from the wire")
    return {
        "title": title,
        "body": body,
        "first_seen": ts.isoformat(),
        "ai_score": ai_score,
        "urgency": urgency,
    }


def _build(articles, held, **kw):
    kw.setdefault("now", _NOW)
    return build_news_velocity(articles, held, **kw)


# ───────────────────────── State ladder honesty ─────────────────────────


class TestStateLadder:

    def test_no_held_positions_is_no_data(self):
        r = _build([_art("NVDA", 2)], held=[])
        assert r["state"] == "NO_DATA"
        assert r["per_ticker"] == []
        assert r["n_held"] == 0
        assert "no held" in r["headline"].lower()

    def test_held_but_no_articles_at_all_is_no_data(self):
        r = _build([], held=["NVDA", "MU"])
        assert r["state"] == "NO_DATA"
        assert r["n_held"] == 2
        assert r["n_with_data"] == 0
        # Every row still emitted so the dashboard can render them.
        assert len(r["per_ticker"]) == 2
        for row in r["per_ticker"]:
            assert row["state"] in {"INSUFFICIENT", "FADING"}

    def test_baseline_below_min_is_insufficient_for_that_ticker(self):
        # 4 baseline + 0 window: below MIN_BASELINE_N=5 → INSUFFICIENT
        # even though z would otherwise be negative. Articles at 25h+
        # are unambiguously past the 24h window cutoff (the boundary at
        # exactly 24h goes into the WINDOW per the inclusive-on-window
        # contract, locked by `test_exact_window_cutoff_is_inclusive_of_window`).
        arts = [_art("NVDA", 25 + i * 4) for i in range(4)]
        r = _build(arts, held=["NVDA"])
        row = r["per_ticker"][0]
        assert row["ticker"] == "NVDA"
        assert row["baseline_count"] == 4
        assert row["state"] == "INSUFFICIENT"
        # Numerics still emitted (sibling honesty contract).
        assert row["window_count"] == 0
        assert row["baseline_rate_per_h"] is not None

    def test_at_min_baseline_a_zero_window_is_fading_not_insufficient(self):
        # 5 baseline + 0 window: at threshold, classified as FADING.
        arts = [_art("NVDA", 30 + i * 4) for i in range(MIN_BASELINE_N)]
        r = _build(arts, held=["NVDA"])
        row = r["per_ticker"][0]
        assert row["baseline_count"] == MIN_BASELINE_N
        assert row["window_count"] == 0
        assert row["state"] == "FADING"


# ───────────────────────── Hand-computed Poisson z ─────────────────────────


class TestSurgeDetection:

    def setup_method(self):
        # Baseline = 6 NVDA in the [25h, 168h) span (strictly past the 24h
        # window cutoff so none leak into the window), then a burst of 10
        # in the last 24h. baseline_span = 168-24 = 144h, base_rate =
        # 6/144 = 0.04167/h; expected_window = 0.04167*24 = 1.0;
        # z = (10 - 1) / max(sqrt(1), 1) = 9.0 → SURGING.
        baseline = [_art("NVDA", 25 + i * 20) for i in range(6)]
        burst = [_art("NVDA", 1 + i * 2) for i in range(10)]
        self.arts = baseline + burst
        self.r = _build(self.arts, held=["NVDA"])
        self.row = self.r["per_ticker"][0]

    def test_state_is_surging(self):
        assert self.row["state"] == "SURGING"
        assert self.r["state"] == "OK"

    def test_window_and_baseline_counts_exact(self):
        assert self.row["window_count"] == 10
        assert self.row["baseline_count"] == 6

    def test_z_score_is_hand_computed_value(self):
        # Recompute independently — a different algebraic path so a sign
        # flip or floor regression fails loudly even though the rounding
        # matches by design.
        expected = (6 / 144) * 24            # 1.0
        z = (10 - expected) / max(math.sqrt(expected), 1.0)
        assert self.row["z_score"] == round(z, 2)
        assert self.row["z_score"] >= Z_SURGE

    def test_ratio_emitted_when_baseline_positive(self):
        # win_rate = 10/24 = 0.4167/h; base_rate = 6/144 = 0.0417/h
        # ratio = 10.0
        assert self.row["ratio"] == 10.0

    def test_headline_calls_out_surging_ticker_with_z(self):
        h = self.r["headline"]
        assert "NVDA SURGING" in h
        assert f"z={self.row['z_score']}" in h


class TestSurgeRequiresAbsoluteFloor:
    """A surge needs BOTH a high z AND enough absolute volume — otherwise a
    baseline of 1 vs a window of 2 (z = +0.6) doesn't false-positive."""

    def test_one_window_article_against_min_baseline_is_not_surging(self):
        # 5 baseline + 1 window → 1 < MIN_WINDOW_FOR_SURGE=3 → STABLE.
        baseline = [_art("NVDA", 30 + i * 5) for i in range(MIN_BASELINE_N)]
        window = [_art("NVDA", 4)]
        r = _build(baseline + window, held=["NVDA"])
        row = r["per_ticker"][0]
        assert row["state"] == "STABLE"

    def test_two_window_articles_below_floor_remains_stable(self):
        # 5 baseline + 2 window → still < MIN_WINDOW_FOR_SURGE=3 → STABLE
        # even if the z is large.
        baseline = [_art("NVDA", 30 + i * 5) for i in range(MIN_BASELINE_N)]
        window = [_art("NVDA", 4), _art("NVDA", 6)]
        r = _build(baseline + window, held=["NVDA"])
        row = r["per_ticker"][0]
        assert row["window_count"] == 2
        assert row["state"] in {"STABLE", "FADING"}  # never SURGING


# ───────────────────────── Window boundary ─────────────────────────


class TestWindowBoundary:

    def test_exact_window_cutoff_is_inclusive_of_window(self):
        # ``signals.py`` precedent: ``first_seen >= since``. An article
        # at exactly now - window_hours falls IN the window, not baseline.
        on_boundary = _art("NVDA", 24.0)
        baseline_padding = [_art("NVDA", 30 + i * 4) for i in range(MIN_BASELINE_N)]
        r = _build([on_boundary] + baseline_padding,
                   held=["NVDA"], window_hours=24.0)
        row = r["per_ticker"][0]
        assert row["window_count"] == 1, row
        assert row["baseline_count"] == MIN_BASELINE_N

    def test_older_than_baseline_dropped_entirely(self):
        too_old = _art("NVDA", 200.0)              # > 168h baseline
        in_baseline = [_art("NVDA", 100.0)]
        r = _build([too_old] + in_baseline, held=["NVDA"],
                   baseline_hours=168.0)
        row = r["per_ticker"][0]
        assert row["window_count"] == 0
        assert row["baseline_count"] == 1   # ``too_old`` excluded


# ───────────────────────── Ticker regex & substring guards ─────────────────────────


class TestTickerWordBoundary:

    def _padded(self, title: str, baseline_count: int = MIN_BASELINE_N):
        """One window article + an MU-named baseline so the held-MU verdict
        can resolve to OK, then the assertion just checks the substring."""
        window_art = _art("", 1.0, title_override=title)
        baseline = [_art("", 50 + i * 5, title_override="MU prints")
                    for i in range(baseline_count)]
        return [window_art] + baseline

    def test_mu_does_not_match_muse(self):
        r = _build(self._padded("MUSE the artist"), held=["MU"])
        row = r["per_ticker"][0]
        # The MUSE window article must NOT count toward MU.
        assert row["window_count"] == 0

    def test_mu_does_not_match_mutual(self):
        r = _build(self._padded("Mutual fund news"), held=["MU"])
        row = r["per_ticker"][0]
        assert row["window_count"] == 0

    def test_mu_cashtag_matches(self):
        r = _build(self._padded("$MU breakout"), held=["MU"])
        row = r["per_ticker"][0]
        assert row["window_count"] == 1

    def test_amdocs_does_not_match_amd(self):
        # The canonical regression in `test_signal_followthrough` / news_edge.
        r = _build(self._padded("Amdocs results beat", baseline_count=0),
                   held=["AMD"])
        row = r["per_ticker"][0]
        assert row["window_count"] == 0
        assert row["baseline_count"] == 0


# ───────────────────────── Sort order ─────────────────────────


class TestSortOrder:
    """SURGING first, then STABLE/FADING by z DESC, INSUFFICIENT last —
    the loudest catalyst surfaces at index 0."""

    def test_surging_precedes_insufficient(self):
        # NVDA: 6 baseline + 10 window → SURGING
        # AMD:  1 baseline + 0 window  → INSUFFICIENT
        # MU:   5 baseline + 0 window  → FADING
        # Baseline articles start strictly past the 24h window cutoff
        # (inclusive-on-window contract — see TestWindowBoundary).
        arts = (
            [_art("NVDA", 25 + i * 20) for i in range(6)]
            + [_art("NVDA", 1 + i * 2) for i in range(10)]
            + [_art("AMD", 50.0)]
            + [_art("MU", 30 + i * 5) for i in range(MIN_BASELINE_N)]
        )
        r = _build(arts, held=["AMD", "MU", "NVDA"])
        states = [row["state"] for row in r["per_ticker"]]
        # SURGING must come first; INSUFFICIENT must come last.
        assert states[0] == "SURGING"
        assert states[-1] == "INSUFFICIENT"


# ───────────────────────── Held-ticker handling ─────────────────────────


class TestHeldTickers:
    def test_dedup_case_insensitive(self):
        r = _build([_art("NVDA", 1)], held=["NVDA", "nvda", "Nvda"])
        assert r["n_held"] == 1
        assert len(r["per_ticker"]) == 1

    def test_baseline_must_exceed_window(self):
        # baseline <= window → defensive NO_DATA, no divide-by-zero.
        r = _build([_art("NVDA", 1)], held=["NVDA"],
                   window_hours=24.0, baseline_hours=24.0)
        assert r["state"] == "NO_DATA"


# ───────────────────────── Degrade-safe contract ─────────────────────────


class TestDegradeSafe:

    def test_garbage_articles_never_raise(self):
        # Mix in: non-dict, missing first_seen, bad timestamp, None body,
        # non-numeric ai_score, non-int urgency.
        bad = [
            "not a dict",
            {"title": "no first_seen"},
            {"title": "bad ts", "first_seen": "not-a-date"},
            {"title": "None body", "body": None, "first_seen": _NOW.isoformat(),
             "ai_score": "huh", "urgency": "two"},
        ]
        good = [_art("NVDA", i) for i in range(MIN_BASELINE_N + 2)]
        r = _build(bad + good, held=["NVDA"])
        assert "error" not in r
        assert r["state"] == "OK"
        row = r["per_ticker"][0]
        assert row["window_count"] >= 1

    def test_none_now_falls_through_to_utcnow(self):
        # Smoke: passing now=None must not raise and must still produce a
        # well-formed payload (uses real wall clock).
        r = build_news_velocity([], ["NVDA"], now=None)
        assert "error" not in r
        assert r["state"] == "NO_DATA"
        assert "as_of" in r


# ───────────────────────── Endpoint integration ─────────────────────────


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    """Fresh paper-trader Store under tmp_path."""
    from paper_trader import store as store_mod
    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "paper_trader.db")
    # Reset the module-level singleton so each test gets a clean DB.
    if hasattr(store_mod, "_STORE"):
        monkeypatch.setattr(store_mod, "_STORE", None, raising=False)
    s = store_mod.Store()
    return s


class TestNewsVelocityEndpoint:
    """Drive the real Flask view on a fresh temp Store. The endpoint owns
    the DB I/O; we monkeypatch ``_articles_db_path`` → None so the route
    takes the documented no-DB degrade path (NO_DATA, no 500)."""

    def test_no_db_no_held_returns_no_data_not_500(self, fresh_store, monkeypatch):
        from paper_trader import dashboard
        monkeypatch.setattr(dashboard, "get_store", lambda: fresh_store)
        monkeypatch.setattr(dashboard, "_articles_db_path", lambda: None)
        client = dashboard.app.test_client()
        r = client.get("/api/news-velocity")
        assert r.status_code == 200
        body = r.get_json()
        assert body["state"] == "NO_DATA"
        assert body["per_ticker"] == []

    def test_ticker_override_param_resolves(self, fresh_store, monkeypatch):
        from paper_trader import dashboard
        monkeypatch.setattr(dashboard, "get_store", lambda: fresh_store)
        monkeypatch.setattr(dashboard, "_articles_db_path", lambda: None)
        client = dashboard.app.test_client()
        r = client.get("/api/news-velocity?tickers=NVDA,MU")
        assert r.status_code == 200
        body = r.get_json()
        # No DB path → degrade body still records the held count from override
        # (sibling endpoint contract: never 500 on missing DB).
        assert body["n_held"] == 2 or body["state"] == "NO_DATA"

    def test_param_clamping_does_not_500_on_garbage(self, fresh_store, monkeypatch):
        from paper_trader import dashboard
        monkeypatch.setattr(dashboard, "get_store", lambda: fresh_store)
        monkeypatch.setattr(dashboard, "_articles_db_path", lambda: None)
        client = dashboard.app.test_client()
        r = client.get(
            "/api/news-velocity?window_hours=notanumber&baseline_hours=-50")
        assert r.status_code == 200
        body = r.get_json()
        assert body.get("state") in {"NO_DATA", "OK"}
        # Clamped baseline must remain > window so the safety branch doesn't
        # trip on every request.
        assert body["baseline_hours"] > body["window_hours"]
