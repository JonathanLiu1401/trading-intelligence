"""Tests for analytics.alert_freshness + the /api/alert-freshness endpoint.

``alert_freshness`` answers the bottom-line analyst question no other monitor
covers: of the rows that actually fired ``urgency>=1``, how *stale* were they
(``published`` → ``first_seen``) at the moment they were detected? A high p90
here is the quality failure that reads HEALTHY on every volume/uptime monitor.

The pure builder ``compute_alert_freshness`` shipped with no test file — these
assertions are its first coverage. They pin the load-bearing logic: the
``urgency<1`` filter, the ``vetted_fraction`` definition (kept byte-identical
to ``ArticleStore.urgency_label_split`` so the two reads never drift), the
>7d implausible-archive skip, the negative-clock-skew clamp, percentile
maths, and malformed-row tolerance. A second block exercises the Flask
endpoint through the test client to pin the SQL adapter + backtest isolation.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.alert_freshness import compute_alert_freshness
from dashboard import web_server

# Fixed clock so staleness is deterministic regardless of wall time.
SEEN = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)


def _row(stale_min, score_source="ml", urgency=1):
    """A ``(published, first_seen, score_source, urgency)`` tuple — the raw
    column shape SQLite yields — whose detection staleness is ``stale_min``
    minutes (``published`` that many minutes before ``first_seen``)."""
    pub = SEEN - timedelta(minutes=stale_min)
    return (pub.isoformat(), SEEN.isoformat(), score_source, urgency)


# ── pure builder: empty / degenerate input ────────────────────────────────────
class TestEmptyInput:
    def test_empty_iterable_zeroed_envelope(self):
        out = compute_alert_freshness([])
        assert out["n_alerted_in_window"] == 0
        assert out["n_with_published"] == 0
        assert out["skipped_no_published"] == 0
        assert out["skipped_implausible"] == 0
        assert out["vetted_fraction"] == 0.0
        # zero-data discipline: aggregate carries every key, all numerics None
        agg = out["aggregate"]
        assert agg["n"] == 0
        for k in ("p50_min", "p90_min", "p99_min", "max_min", "mean_min"):
            assert agg[k] is None
        # all four score_source buckets are always present
        assert set(out["by_score_source"]) == {
            "llm", "ml", "briefing_boost", "null"}


# ── pure builder: the urgency filter ──────────────────────────────────────────
class TestUrgencyFilter:
    def test_urgency_below_one_excluded(self):
        """urgency=0 rows are not alerts — they must not enter the window."""
        out = compute_alert_freshness([_row(10, urgency=0)])
        assert out["n_alerted_in_window"] == 0
        assert out["aggregate"]["n"] == 0

    def test_urgency_one_and_two_both_counted(self):
        out = compute_alert_freshness(
            [_row(10, urgency=1), _row(20, urgency=2)])
        assert out["n_alerted_in_window"] == 2
        assert out["aggregate"]["n"] == 2

    def test_non_numeric_urgency_treated_as_zero(self):
        """A garbage urgency value coerces to 0 and is dropped, never crashes."""
        out = compute_alert_freshness([("p", "s", "ml", "not-a-number")])
        assert out["n_alerted_in_window"] == 0


# ── pure builder: vetted_fraction is the urgency_label_split contract ─────────
class TestVettedFraction:
    def test_vetted_fraction_matches_label_split_definition(self):
        """``vetted_fraction`` is byte-identical to
        ``ArticleStore.urgency_label_split``: (llm + briefing_boost) / total.
        A drift between this number and the dashboard's aggregate calibration
        tile would silently mislead the analyst — pin the formula so they
        always agree (the same SSOT lock as
        ``test_alert_delivery_audit.py::test_delivered_llm_fraction_*``)."""
        rows = [
            _row(5, score_source="llm"),
            _row(6, score_source="llm"),
            _row(7, score_source="briefing_boost"),
            _row(8, score_source="ml"),
            _row(9, score_source="ml"),
        ]
        out = compute_alert_freshness(rows)
        # (2 llm + 1 briefing_boost) / 5 total = 0.6
        assert out["vetted_fraction"] == 0.6

    def test_vetted_fraction_zero_when_all_model_only(self):
        out = compute_alert_freshness([_row(5, score_source="ml")] * 3)
        assert out["vetted_fraction"] == 0.0

    def test_vetted_fraction_counts_rows_with_no_published(self):
        """A vetted row with unparseable ``published`` is still a vetted
        alert — it must count toward ``vetted_fraction`` even though it
        yields no latency sample (the denominator is n_alerted, not
        n_with_published)."""
        rows = [
            ("", SEEN.isoformat(), "llm", 1),       # no published
            _row(5, score_source="ml"),
        ]
        out = compute_alert_freshness(rows)
        assert out["n_alerted_in_window"] == 2
        assert out["n_with_published"] == 1
        assert out["skipped_no_published"] == 1
        assert out["vetted_fraction"] == 0.5


# ── pure builder: implausible-staleness skip ─────────────────────────────────
class TestImplausibleSkip:
    def test_archive_repost_over_7d_skipped_from_percentiles(self):
        """An 8-day-stale row is almost certainly an archive backfill — it
        must be counted in the window but excluded from the latency
        percentiles so one stale repost cannot dominate p90/p99."""
        rows = [_row(30), _row(8 * 24 * 60)]  # 30min + 8 days
        out = compute_alert_freshness(rows)
        assert out["n_alerted_in_window"] == 2
        assert out["skipped_implausible"] == 1
        assert out["aggregate"]["n"] == 1          # only the 30min sample
        assert out["aggregate"]["max_min"] == 30.0  # archive did not pollute


# ── pure builder: clock-skew clamp ───────────────────────────────────────────
class TestClockSkewClamp:
    def test_published_after_first_seen_clamped_to_zero(self):
        """Upstream clock skew (published ahead of first_seen) yields a
        negative latency — it is a real ingestion and is clamped to 0, not
        dropped (dropping it would bias the view fresher than reality)."""
        rows = [_row(-15)]  # published 15min AFTER first_seen
        out = compute_alert_freshness(rows)
        assert out["aggregate"]["n"] == 1
        assert out["aggregate"]["p50_min"] == 0.0
        assert out["aggregate"]["max_min"] == 0.0


# ── pure builder: malformed-row tolerance ────────────────────────────────────
class TestMalformedRows:
    def test_wrong_shape_row_skipped_not_raised(self):
        rows = [("only", "two"), _row(10), None]
        out = compute_alert_freshness(rows)
        # the one well-formed row survives; the 2-tuple and None are skipped
        assert out["n_alerted_in_window"] == 1
        assert out["aggregate"]["n"] == 1


# ── pure builder: percentile maths ───────────────────────────────────────────
class TestPercentiles:
    def test_known_sample_percentiles(self):
        """11 rows at 0,10,…,100 min staleness — linear-interp percentiles
        land on exact sample points (idx = q*(n-1) is integral)."""
        rows = [_row(m) for m in range(0, 101, 10)]
        agg = compute_alert_freshness(rows)["aggregate"]
        assert agg["n"] == 11
        assert agg["p50_min"] == 50.0    # idx 0.50*10 = 5  -> 50min
        assert agg["p90_min"] == 90.0    # idx 0.90*10 = 9  -> 90min
        assert agg["max_min"] == 100.0
        assert agg["mean_min"] == 50.0   # symmetric sample

    def test_threshold_buckets_count_correctly(self):
        """A fresh row and a 2h-stale row exercise the under/over buckets."""
        out = compute_alert_freshness([_row(2), _row(120)])
        agg = out["aggregate"]
        assert agg["pct_under_5min"] == 50.0   # the 2min row
        assert agg["pct_over_1h"] == 50.0      # the 120min row


# ── pure builder: by_score_source partition ──────────────────────────────────
class TestByScoreSourcePartition:
    def test_buckets_sum_to_aggregate(self):
        rows = [
            _row(5, score_source="llm"),
            _row(6, score_source="ml"),
            _row(7, score_source="briefing_boost"),
            _row(8, score_source="ml"),
        ]
        out = compute_alert_freshness(rows)
        per_bucket = sum(
            out["by_score_source"][k]["n"]
            for k in ("llm", "ml", "briefing_boost", "null"))
        assert per_bucket == out["aggregate"]["n"] == 4
        assert out["by_score_source"]["ml"]["n"] == 2

    def test_unknown_score_source_collapses_to_null_bucket(self):
        """A row with an unrecognised / missing score_source lands in the
        ``null`` bucket — same null-collapse as ``urgency_label_split``."""
        rows = [_row(5, score_source=None), _row(6, score_source="weird")]
        out = compute_alert_freshness(rows)
        assert out["by_score_source"]["null"]["n"] == 2


# ── endpoint wiring: /api/alert-freshness ────────────────────────────────────
def _insert(store, *, id, url, title, source, score_source="ml",
            urgency=1, stale_min=30, age_min=10):
    """Insert one article. ``first_seen`` is ``age_min`` minutes ago;
    ``published`` is a further ``stale_min`` minutes before that."""
    first_seen = datetime.now(timezone.utc) - timedelta(minutes=age_min)
    published = first_seen - timedelta(minutes=stale_min)
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, "
            "urgency, first_seen, cycle, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, published.isoformat(), 2.0, 5.0,
             urgency, first_seen.isoformat(), 0, 0.9, score_source),
        )
        store.conn.commit()


def test_endpoint_shape_and_backtest_isolation(store, monkeypatch):
    monkeypatch.delenv("WEB_API_KEY", raising=False)
    _insert(store, id="l1", url="https://x/1",
            title="NVDA guidance raised", source="rss", stale_min=20)
    # A synthetic backtest row with urgency=2 — only the _LIVE_ONLY_CLAUSE
    # SQL filter can keep it out of the freshness view.
    _insert(store, id="bt", url="backtest://run_9/2026-01-01/BUY/NVDA",
            title="SYNTHETIC SHOULD NOT SURFACE",
            source="backtest_run_9_winner", urgency=2)

    monkeypatch.setattr(web_server, "_store", store, raising=False)
    client = web_server.create_app(store).test_client()
    resp = client.get("/api/alert-freshness")

    assert resp.status_code == 200, resp.data
    data = resp.get_json()
    assert set(data) >= {"n_alerted_in_window", "n_with_published",
                         "aggregate", "by_score_source", "vetted_fraction",
                         "window_hours", "generated_at"}
    # The backtest row is filtered by SQL; only the 1 live row is seen.
    assert data["n_alerted_in_window"] == 1
    assert data["aggregate"]["n"] == 1
    assert data["aggregate"]["p50_min"] == 20.0


def test_endpoint_hours_param_clamped(store, monkeypatch):
    monkeypatch.delenv("WEB_API_KEY", raising=False)
    monkeypatch.setattr(web_server, "_store", store, raising=False)
    client = web_server.create_app(store).test_client()

    assert client.get("/api/alert-freshness?hours=9999").get_json()[
        "window_hours"] == 168
    assert client.get("/api/alert-freshness?hours=0").get_json()[
        "window_hours"] == 1
    # garbage falls back to the 24h default
    assert client.get("/api/alert-freshness?hours=abc").get_json()[
        "window_hours"] == 24


def test_endpoint_db_error_yields_500(store, monkeypatch):
    import sqlite3
    monkeypatch.delenv("WEB_API_KEY", raising=False)

    def _boom(sql, params=()):
        raise sqlite3.OperationalError("database is locked")

    monkeypatch.setattr(web_server, "_ro_query", _boom)
    monkeypatch.setattr(web_server, "_store", store, raising=False)
    resp = web_server.create_app(store).test_client().get(
        "/api/alert-freshness")
    assert resp.status_code == 500
    assert "error" in resp.get_json()


def test_endpoint_api_key_enforced(store, monkeypatch):
    monkeypatch.setenv("WEB_API_KEY", "secret")
    monkeypatch.setattr(web_server, "_store", store, raising=False)
    resp = web_server.create_app(store).test_client().get(
        "/api/alert-freshness")
    assert resp.status_code == 401
