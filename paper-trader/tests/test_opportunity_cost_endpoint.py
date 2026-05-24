"""Flask wiring tests for /api/opportunity-cost.

Drives the dashboard endpoint with a fake store + a monkeypatched
``_daily_history_cached`` so the test is offline and deterministic.
Asserts that the SSOT builder is wired correctly, the live-only articles
read works (stubbed at the connection layer), forward-return arithmetic
plumbs through cleanly, and the @swr_cached warming path eventually
populates a fresh body on second hit.

Pure arithmetic of the verdict ladder + classification is pinned in
test_opportunity_cost_skill.py — this file only covers the IO seam.
"""
from __future__ import annotations

import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import dashboard


class _NullLock:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


class _FakeStore:
    def __init__(self, decisions=None):
        self._d = decisions or []
        self._lock = _NullLock()
        self.conn = None

    def recent_decisions(self, limit=3000):
        return list(self._d[:limit])


@pytest.fixture
def client():
    return dashboard.app.test_client()


def _wait_swr(client, url, max_wait_s: float = 20.0):
    """Hammer the SWR endpoint until it returns a populated body, not the
    warming envelope. Returns the populated JSON dict.

    The @swr_cached decorator returns a "warming" envelope on the cold
    first hit and computes asynchronously; subsequent hits return the
    fresh body once available.
    """
    deadline = time.time() + max_wait_s
    last = None
    while time.time() < deadline:
        r = client.get(url)
        d = r.get_json()
        last = d
        # Warming envelope carries {warming: True, attempts: ...}; a
        # populated body has at minimum 'verdict' + 'stats' + 'as_of'.
        if d and "verdict" in d and "stats" in d:
            return d
        time.sleep(0.4)
    raise AssertionError(f"SWR never populated for {url}; last={last}")


def _sitout_decision(did, days_ago, action_taken="HOLD CASH → HOLD"):
    ts = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return {
        "id": did,
        "timestamp": ts.isoformat(),
        "action_taken": action_taken,
        "reasoning": "no edge",
    }


class TestEndpointWiring:
    def test_empty_store_no_data(self, client, monkeypatch):
        monkeypatch.setattr(dashboard, "get_store", lambda: _FakeStore())
        # No articles.db read needed when there are zero sit-outs; force
        # the path to None so we don't accidentally read live data.
        monkeypatch.setattr(dashboard, "_articles_db_path", lambda: None)
        # Use a unique window_hours per test so each test gets its own
        # SWR cache slot (the decorator keys on the query string).
        d = _wait_swr(client, "/api/opportunity-cost?window_hours=25")
        assert d["verdict"] == "NO_DATA"
        assert d["stats"]["n_sitout_total"] == 0
        assert d["stats"]["n_classified"] == 0

    def test_missed_alpha_when_holds_preceded_runners(
            self, client, monkeypatch):
        # 6 sit-outs spread across the last 14 days; each one had NVDA as
        # the top news ticker and NVDA ran +6% over 3d after every one.
        # Default thresholds: missed_pct_floor=50 + mean_fwd_pct_floor=2 →
        # MISSED_ALPHA.
        decs = [_sitout_decision(i, 4 + i) for i in range(1, 7)]
        monkeypatch.setattr(dashboard, "get_store",
                            lambda: _FakeStore(decs))
        monkeypatch.setattr(dashboard, "_articles_db_path", lambda: None)
        # Force the news-heat helper to always return ("NVDA", 99.0)
        # regardless of timestamp (no articles needed for the IO test).
        from paper_trader.analytics import opportunity_cost_skill as ocs
        monkeypatch.setattr(
            ocs, "top_ticker_by_heat",
            lambda articles, tickers, ts_end, lookback_hours=2.0:
                ("NVDA", 99.0))
        # Forward returns: every 3d window is +6.1% — a runner. Use
        # compound growth so the fwd_3d % ratio is constant regardless
        # of which session is the base. _daily_history_cached is the seam
        # the endpoint goes through.
        today = datetime.now(timezone.utc).date()
        bars = []
        for i in range(120, 0, -1):
            day = today - timedelta(days=i)
            # 2%/day compounded → fwd_3d ratio ≈ 1.02**3 - 1 = +6.12% on
            # every base session, well above runner_pct_floor=5.
            price = 100.0 * (1.02 ** (120 - i))
            bars.append((day.isoformat(), price))
        monkeypatch.setattr(
            dashboard, "_daily_history_cached",
            lambda tk, period="3mo": bars)

        d = _wait_swr(client, "/api/opportunity-cost?window_hours=720")
        # Every sit-out should classify as MISSED_RUNNER (3d ≥ +5%).
        assert d["stats"]["n_sitout_total"] >= 6
        # Sanity: forward returns reached classification.
        assert d["stats"]["n_classified"] >= 5
        # Verdict should be MISSED_ALPHA (≥50% missed + mean 3d ≥ +2%).
        assert d["verdict"] == "MISSED_ALPHA"

    def test_defensive_win_when_holds_dodged_drawdowns(
            self, client, monkeypatch):
        decs = [_sitout_decision(i, 4 + i) for i in range(1, 7)]
        monkeypatch.setattr(dashboard, "get_store",
                            lambda: _FakeStore(decs))
        monkeypatch.setattr(dashboard, "_articles_db_path", lambda: None)
        from paper_trader.analytics import opportunity_cost_skill as ocs
        monkeypatch.setattr(
            ocs, "top_ticker_by_heat",
            lambda articles, tickers, ts_end, lookback_hours=2.0:
                ("NVDA", 99.0))
        # Compound -2%/day ⇒ fwd_3d ≈ -5.9% on every session, comfortably
        # below defensive_pct_ceil=-1 (DEFENSIVE_HIT).
        today = datetime.now(timezone.utc).date()
        bars = []
        for i in range(120, 0, -1):
            day = today - timedelta(days=i)
            price = 1000.0 * (0.98 ** (120 - i))
            bars.append((day.isoformat(), price))
        monkeypatch.setattr(
            dashboard, "_daily_history_cached",
            lambda tk, period="3mo": bars)

        d = _wait_swr(client, "/api/opportunity-cost?window_hours=721")
        assert d["stats"]["n_classified"] >= 5
        assert d["verdict"] == "DEFENSIVE_WIN"

    def test_no_top_ticker_skips_classification(self, client, monkeypatch):
        # When top_ticker_at returns None for every decision, the
        # n_no_candidate counter advances and n_classified stays at 0.
        decs = [_sitout_decision(i, 4 + i) for i in range(1, 7)]
        monkeypatch.setattr(dashboard, "get_store",
                            lambda: _FakeStore(decs))
        monkeypatch.setattr(dashboard, "_articles_db_path", lambda: None)
        from paper_trader.analytics import opportunity_cost_skill as ocs
        monkeypatch.setattr(
            ocs, "top_ticker_by_heat",
            lambda articles, tickers, ts_end, lookback_hours=2.0: None)

        d = _wait_swr(client, "/api/opportunity-cost?window_hours=722")
        assert d["stats"]["n_no_candidate"] >= 6
        assert d["stats"]["n_classified"] == 0
        # Below the verdict floor → NO_DATA verdict.
        assert d["verdict"] == "NO_DATA"

    def test_yfinance_failure_degrades_to_no_fwd(self, client, monkeypatch):
        decs = [_sitout_decision(i, 4 + i) for i in range(1, 7)]
        monkeypatch.setattr(dashboard, "get_store",
                            lambda: _FakeStore(decs))
        monkeypatch.setattr(dashboard, "_articles_db_path", lambda: None)
        from paper_trader.analytics import opportunity_cost_skill as ocs
        monkeypatch.setattr(
            ocs, "top_ticker_by_heat",
            lambda articles, tickers, ts_end, lookback_hours=2.0:
                ("NVDA", 99.0))
        # Empty bars from yfinance ⇒ forward_returns_for returns (None, None)
        # — the n_no_fwd counter must advance without raising.
        monkeypatch.setattr(
            dashboard, "_daily_history_cached",
            lambda tk, period="3mo": [])

        d = _wait_swr(client, "/api/opportunity-cost?window_hours=723")
        assert d["stats"]["n_no_fwd"] >= 6
        assert d["stats"]["n_classified"] == 0

    def test_envelope_shape_complete(self, client, monkeypatch):
        # The chat helper depends on a specific key set. Pin them.
        monkeypatch.setattr(dashboard, "get_store", lambda: _FakeStore())
        monkeypatch.setattr(dashboard, "_articles_db_path", lambda: None)
        d = _wait_swr(client, "/api/opportunity-cost?window_hours=724")
        for k in ("verdict", "headline", "as_of", "window_hours",
                  "stats", "thresholds", "samples"):
            assert k in d, f"missing key: {k}"
        for sk in ("n_sitout_total", "n_no_candidate", "n_no_fwd",
                   "n_classified", "n_missed_runner", "n_missed_ok",
                   "n_neutral", "n_defensive",
                   "missed_pct", "defensive_pct", "mean_fwd_3d_pct"):
            assert sk in d["stats"], f"missing stats key: {sk}"
