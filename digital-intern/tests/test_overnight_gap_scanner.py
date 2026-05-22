"""Tests for analytics.overnight_gap_scanner.

The pure builder ``build_overnight_gaps`` is exercised against a fixed clock
so the ET market-hours boundary is deterministic; the ``/api/overnight-gaps``
endpoint is exercised through the Flask test client to pin the wiring and the
backtest-isolation SQL filter.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from analytics.overnight_gap_scanner import build_overnight_gaps
from dashboard import web_server

# A fixed clock: 2026-05-20 12:00 UTC (a Wednesday, EDT = UTC-4).
NOW = datetime(2026, 5, 20, 12, 0, tzinfo=timezone.utc)

# 2026-05-20 03:00 UTC -> 2026-05-19 23:00 EDT -> overnight, 9h before NOW.
OVERNIGHT = "2026-05-20T03:00:00+00:00"
# 2026-05-19 18:00 UTC -> 2026-05-19 14:00 EDT -> regular market hours.
INTRADAY = "2026-05-19T18:00:00+00:00"
# 2026-05-18 03:00 UTC -> overnight ET but 33h before NOW (outside 24h window).
STALE_OVERNIGHT = "2026-05-18T03:00:00+00:00"


def _row(first_seen, title, urgency=2, ml_score=0.9, source="rss"):
    return (first_seen, title, urgency, ml_score, source)


# ── Pure builder ────────────────────────────────────────────────────────────


def test_empty_rows_yield_empty_digest():
    out = build_overnight_gaps([], now=NOW)
    assert out["scanned"] == 0
    assert out["overnight_articles_24h"] == 0
    assert out["gap_candidates"] == []


def test_overnight_urgent_article_becomes_a_gap_candidate():
    out = build_overnight_gaps(
        [_row(OVERNIGHT, "NVDA guidance raised after the close")], now=NOW)
    assert out["overnight_articles_24h"] == 1
    tickers = [c["ticker"] for c in out["gap_candidates"]]
    assert "NVDA" in tickers
    cand = next(c for c in out["gap_candidates"] if c["ticker"] == "NVDA")
    assert cand["article_count"] == 1
    assert cand["max_urgency"] == 2


def test_intraday_article_is_not_an_overnight_gap():
    """An article published during regular ET market hours never gaps the
    open — it is already priced. It must not count, even when urgent."""
    out = build_overnight_gaps(
        [_row(INTRADAY, "NVDA surprise mid-session news")], now=NOW)
    assert out["overnight_articles_24h"] == 0
    assert out["gap_candidates"] == []


def test_article_older_than_24h_is_excluded_from_the_window():
    """``scanned`` still counts every input row, but a row outside the 24h
    window contributes nothing to the overnight count or the candidates."""
    out = build_overnight_gaps(
        [_row(STALE_OVERNIGHT, "AMD ancient overnight headline")], now=NOW)
    assert out["scanned"] == 1
    assert out["overnight_articles_24h"] == 0
    assert out["gap_candidates"] == []


def test_low_signal_overnight_row_counts_in_window_but_is_not_a_candidate():
    """A row inside the overnight window is tallied in
    ``overnight_articles_24h``, but with urgency 0 and a near-zero ml_score
    it carries no gap signal and must not become a candidate."""
    out = build_overnight_gaps(
        [_row(OVERNIGHT, "MU minor overnight note", urgency=0, ml_score=0.1)],
        now=NOW)
    assert out["overnight_articles_24h"] == 1
    assert out["gap_candidates"] == []


def test_ranking_puts_higher_urgency_above_higher_volume():
    """Rank key is ``max_urgency*2 + count + max_ml`` — two urgency-1 hits
    must not outrank a single urgency-2 hit."""
    rows = [
        _row(OVERNIGHT, "NVDA halted on urgent filing", urgency=2, ml_score=0.5),
        _row(OVERNIGHT, "AMD chip note one", urgency=1, ml_score=0.5),
        _row(OVERNIGHT, "AMD chip note two", urgency=1, ml_score=0.5),
    ]
    out = build_overnight_gaps(rows, now=NOW)
    order = [c["ticker"] for c in out["gap_candidates"]]
    assert order[0] == "NVDA", order


def test_stop_words_are_not_extracted_as_tickers():
    # Every uppercase token here is in the STOP set (CEO/CFO/BUY/THE/ETF/AND).
    out = build_overnight_gaps(
        [_row(OVERNIGHT, "CEO AND CFO BUY THE ETF")], now=NOW)
    assert out["gap_candidates"] == []


def test_top_articles_capped_at_three_per_ticker():
    rows = [_row(OVERNIGHT, f"NVDA overnight headline {i}") for i in range(6)]
    out = build_overnight_gaps(rows, now=NOW)
    cand = next(c for c in out["gap_candidates"] if c["ticker"] == "NVDA")
    assert cand["article_count"] == 6
    assert len(cand["top_articles"]) == 3


def test_top_n_limit_is_respected():
    rows = [_row(OVERNIGHT, f"{t} overnight catalyst")
            for t in ("NVDA", "AMDX", "MUXY", "WDCZ")]
    out = build_overnight_gaps(rows, now=NOW, top_n=2)
    assert len(out["gap_candidates"]) == 2


def test_malformed_rows_are_skipped_without_raising():
    rows = [None, (1, 2), "garbage", _row(OVERNIGHT, "NVDA real overnight row")]
    out = build_overnight_gaps(rows, now=NOW)
    assert out["scanned"] == 4
    assert [c["ticker"] for c in out["gap_candidates"]] == ["NVDA"]


def test_garbage_urgency_and_ml_score_do_not_raise():
    out = build_overnight_gaps(
        [(OVERNIGHT, "NVDA odd row", "high", "n/a", "rss")], now=NOW)
    # urgency/ml_score coerce to 0 -> below signal floor -> not a candidate,
    # but the call must not raise.
    assert out["overnight_articles_24h"] == 1
    assert out["gap_candidates"] == []


# ── Endpoint ────────────────────────────────────────────────────────────────


def _insert(store, *, id, url, title, source, urgency=2, ml_score=0.9,
            age_min=120):
    fs = (datetime.now(timezone.utc) - timedelta(minutes=age_min)).isoformat()
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", 2.0, 5.0, urgency, fs, 0,
             ml_score, "ml"),
        )
        store.conn.commit()


def test_endpoint_returns_digest_shape_and_excludes_backtest(store, monkeypatch):
    _insert(store, id="l1", url="https://x/1",
            title="NVDA overnight catalyst", source="rss")
    # Synthetic backtest row — the live-only SQL filter must drop it before
    # the builder ever sees it.
    _insert(store, id="bt", url="backtest://run_9/2026-01-01/BUY/NVDA",
            title="SYNTHETIC NVDA SHOULD NOT SURFACE",
            source="backtest_run_9_winner")

    monkeypatch.setattr(web_server, "_store", store, raising=False)
    client = web_server.create_app(store).test_client()
    resp = client.get("/api/overnight-gaps")

    assert resp.status_code == 200, resp.data
    data = resp.get_json()
    assert set(data) >= {"generated_at", "scanned", "overnight_articles_24h",
                         "gap_candidates"}
    assert isinstance(data["gap_candidates"], list)
    # The backtest row is filtered by SQL, so only the 1 live row is scanned.
    assert data["scanned"] == 1
