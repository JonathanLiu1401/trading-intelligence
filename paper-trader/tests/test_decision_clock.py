"""Tests for /api/decision-clock — per-hour-of-day decision distribution.

Lock that quota-exhaustion NO_DECISIONs land in a distinct ``quota_exhausted``
bucket and do NOT leak into ``other_no_decision`` (the operator triaging a
NO_DECISION storm must be able to tell host-saturation from quota-frozen apart:
distinct OPS actions — kill review agents vs upgrade plan / wait).

Also locks the existing classification ladder (host_saturated /
empty_response / parse_failed) at the API surface so a future change to
``strategy.py``'s reason format can't silently re-merge them.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from paper_trader import dashboard, store as store_mod
from paper_trader.store import INITIAL_CASH, Store


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)
    s = Store()
    try:
        yield s
    finally:
        s.close()


def _seed_decision(store: Store, ts: str, action: str, reasoning: str):
    """Bypass Store.record_decision so we can pin the timestamp.

    The endpoint reads ``decisions.timestamp`` exactly; the live writer
    uses now() so we can't control it. We poke a row in directly with a
    chosen timestamp — exact same column set."""
    with store._lock:
        store.conn.execute(
            "INSERT INTO decisions (timestamp, market_open, signal_count, "
            "action_taken, reasoning, portfolio_value, cash) "
            "VALUES (?,?,?,?,?,?,?)",
            (ts, 1, 0, action, reasoning, INITIAL_CASH, INITIAL_CASH),
        )
        store.conn.commit()


def _get_clock(fresh_store, monkeypatch, *, days: int = 7) -> dict:
    monkeypatch.setattr(dashboard, "get_store", lambda: fresh_store)
    client = dashboard.app.test_client()
    r = client.get(f"/api/decision-clock?days={days}")
    assert r.status_code == 200, r.get_data(as_text=True)
    return r.get_json()


# ── classification ladder ────────────────────────────────────────────────


class TestQuotaExhaustedBucket:
    """Quota-exhaustion NO_DECISIONs land in ``quota_exhausted``, not in
    ``other_no_decision``. The operator-actionable distinction (kill review
    agents vs upgrade plan) is the whole reason this bucket exists."""

    def test_quota_reason_falls_into_quota_exhausted_not_other(
            self, fresh_store, monkeypatch):
        now = datetime.now(timezone.utc) - timedelta(hours=2)
        ts = now.isoformat()
        # The exact reasoning string strategy.py writes on a quota outage.
        _seed_decision(fresh_store, ts, "NO_DECISION",
                       "claude quota/usage limit exhausted (no decision)")
        body = _get_clock(fresh_store, monkeypatch)

        # Find the bucket carrying this decision (in NY local time).
        from zoneinfo import ZoneInfo
        hour_ny = now.astimezone(ZoneInfo("America/New_York")).hour
        b = body["buckets"][hour_ny]
        assert b["total"] == 1
        assert b["no_decision"] == 1
        assert b["quota_exhausted"] == 1, (
            "quota reason MUST land in quota_exhausted — currently in "
            f"{b!r}")
        # And NOT in any other NO_DECISION sub-bucket — the discriminator.
        assert b["other_no_decision"] == 0
        assert b["host_saturated"] == 0
        assert b["empty_response"] == 0
        assert b["parse_failed"] == 0

    def test_quota_does_not_alias_other_no_decision_buckets(
            self, fresh_store, monkeypatch):
        """A pile of quota rows should not raise the host_saturated /
        empty_response / parse_failed counts at all — they are mutually
        exclusive branches by construction (regression lock against a
        future precedence flip that re-merges them)."""
        now = datetime.now(timezone.utc) - timedelta(hours=1)
        for i in range(5):
            ts = (now + timedelta(seconds=i)).isoformat()
            _seed_decision(fresh_store, ts, "NO_DECISION",
                           "claude quota/usage limit exhausted (no decision)")
        body = _get_clock(fresh_store, monkeypatch)
        total_q = sum(b["quota_exhausted"] for b in body["buckets"])
        total_other = sum(b["other_no_decision"] for b in body["buckets"])
        total_host = sum(b["host_saturated"] for b in body["buckets"])
        total_empty = sum(b["empty_response"] for b in body["buckets"])
        total_parse = sum(b["parse_failed"] for b in body["buckets"])
        assert total_q == 5
        assert total_other == 0
        assert total_host == 0
        assert total_empty == 0
        assert total_parse == 0


class TestExistingClassificationLadder:
    """Lock the unchanged branches at the endpoint surface so the new
    quota_exhausted precedence (now FIRST) hasn't silently shifted the
    others — the load-bearing branch order is documented in dashboard.py."""

    def test_host_saturated_classification(self, fresh_store, monkeypatch):
        ts = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        # strategy.py writes "skipped claude call — host saturated: ..." —
        # both tokens present, host_saturated branch wins.
        _seed_decision(fresh_store, ts, "NO_DECISION",
                       "skipped claude call — host saturated: 6 concurrent Opus (>4)")
        body = _get_clock(fresh_store, monkeypatch)
        total_h = sum(b["host_saturated"] for b in body["buckets"])
        total_q = sum(b["quota_exhausted"] for b in body["buckets"])
        assert total_h == 1
        assert total_q == 0  # "quota" substring is NOT in this reason

    def test_empty_response_classification(self, fresh_store, monkeypatch):
        ts = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        _seed_decision(fresh_store, ts, "NO_DECISION",
                       "claude returned no response (timeout/empty)")
        body = _get_clock(fresh_store, monkeypatch)
        total_e = sum(b["empty_response"] for b in body["buckets"])
        total_h = sum(b["host_saturated"] for b in body["buckets"])
        assert total_e == 1
        assert total_h == 0

    def test_parse_failed_classification(self, fresh_store, monkeypatch):
        ts = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        _seed_decision(fresh_store, ts, "NO_DECISION",
                       "parse_failed: not JSON")
        body = _get_clock(fresh_store, monkeypatch)
        total_p = sum(b["parse_failed"] for b in body["buckets"])
        assert total_p == 1

    def test_filled_does_not_count_as_no_decision(self, fresh_store, monkeypatch):
        ts = (datetime.now(timezone.utc) - timedelta(hours=3)).isoformat()
        _seed_decision(fresh_store, ts, "BUY NVDA → FILLED",
                       "{decision}")
        body = _get_clock(fresh_store, monkeypatch)
        # The substring "FILLED" wins — filled bucket only.
        total_f = sum(b["filled"] for b in body["buckets"])
        total_nd = sum(b["no_decision"] for b in body["buckets"])
        assert total_f == 1
        assert total_nd == 0


# ── verdict gating ───────────────────────────────────────────────────────


class TestVerdictGating:
    """The verdict ladder: INSUFFICIENT_DATA < 5 → EVEN_DISTRIBUTION or
    HOURLY_CONCENTRATION. Tested at the endpoint surface."""

    def test_insufficient_data_below_5_decisions(
            self, fresh_store, monkeypatch):
        now = datetime.now(timezone.utc) - timedelta(hours=1)
        for i in range(4):  # 4 < 5 — below threshold
            ts = (now + timedelta(seconds=i)).isoformat()
            _seed_decision(fresh_store, ts, "NO_DECISION",
                           "claude returned no response (timeout/empty)")
        body = _get_clock(fresh_store, monkeypatch)
        assert body["verdict"] == "INSUFFICIENT_DATA"

    def test_hourly_concentration_at_50pct_no_decision(
            self, fresh_store, monkeypatch):
        # 5 NO_DECISION + 1 FILLED in the SAME hour → 83% no_decision,
        # well above the 50% gate; bucket has 6 samples so passes the
        # ≥3 worst-bucket floor and the 5-total INSUFFICIENT_DATA gate.
        now = datetime.now(timezone.utc) - timedelta(hours=2)
        for i in range(5):
            ts = (now + timedelta(seconds=i)).isoformat()
            _seed_decision(fresh_store, ts, "NO_DECISION",
                           "claude returned no response (timeout/empty)")
        _seed_decision(fresh_store, (now + timedelta(seconds=10)).isoformat(),
                       "BUY NVDA → FILLED", "{decision}")
        body = _get_clock(fresh_store, monkeypatch)
        assert body["verdict"].startswith("HOURLY_CONCENTRATION"), body["verdict"]
        assert "NO_DECISION" in body["verdict"]
        # The worst_hour_local should be the NY-local hour of `now`.
        from zoneinfo import ZoneInfo
        expected_hour = now.astimezone(ZoneInfo("America/New_York")).hour
        assert body["worst_hour_local"] == expected_hour

    def test_even_distribution_when_no_bucket_above_50pct(
            self, fresh_store, monkeypatch):
        # 5 FILLED, 1 NO_DECISION across one hour: only 17% NO_DECISION
        # in the busy bucket → EVEN_DISTRIBUTION.
        now = datetime.now(timezone.utc) - timedelta(hours=2)
        for i in range(5):
            ts = (now + timedelta(seconds=i)).isoformat()
            _seed_decision(fresh_store, ts, "BUY NVDA → FILLED", "{decision}")
        _seed_decision(fresh_store, (now + timedelta(seconds=10)).isoformat(),
                       "NO_DECISION", "claude returned no response")
        body = _get_clock(fresh_store, monkeypatch)
        assert body["verdict"] == "EVEN_DISTRIBUTION", body["verdict"]


class TestBucketShape:
    """The buckets are 24 (hours 0..23 in NY local) and every bucket
    carries the documented keys. A regression that drops a bucket or
    renames a key fails loudly."""

    def test_24_buckets_with_required_keys(self, fresh_store, monkeypatch):
        body = _get_clock(fresh_store, monkeypatch)
        assert len(body["buckets"]) == 24
        required = {"hour", "total", "filled", "no_decision",
                    "host_saturated", "empty_response", "parse_failed",
                    "quota_exhausted", "other_no_decision",
                    "fill_rate_pct", "no_decision_pct",
                    "host_saturated_pct"}
        for h, b in enumerate(body["buckets"]):
            assert b["hour"] == h
            missing = required - set(b.keys())
            assert not missing, f"bucket h={h} missing keys: {missing}"

    def test_days_parameter_clamped(self, fresh_store, monkeypatch):
        # Garbage param → falls through to default; negative / >30 clamp.
        monkeypatch.setattr(dashboard, "get_store", lambda: fresh_store)
        client = dashboard.app.test_client()
        r = client.get("/api/decision-clock?days=abc")
        assert r.status_code == 200
        assert r.get_json()["days"] == 7  # default
        r = client.get("/api/decision-clock?days=-5")
        assert r.status_code == 200
        assert r.get_json()["days"] == 1  # min clamp
        r = client.get("/api/decision-clock?days=9999")
        assert r.status_code == 200
        assert r.get_json()["days"] == 30  # max clamp

    def test_decisions_older_than_window_excluded(
            self, fresh_store, monkeypatch):
        # Old decision (8 days back) when days=7 — must not count.
        old_ts = (datetime.now(timezone.utc) - timedelta(days=8)).isoformat()
        _seed_decision(fresh_store, old_ts, "NO_DECISION",
                       "claude returned no response (timeout/empty)")
        body = _get_clock(fresh_store, monkeypatch, days=7)
        assert body["total_decisions"] == 0
        assert body["verdict"] == "INSUFFICIENT_DATA"
