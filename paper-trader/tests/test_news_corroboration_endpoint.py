"""/api/news-corroboration-skill — endpoint wiring + per-trade counter helper.

The builder (paper_trader/analytics/news_corroboration_skill.py) is pinned by
tests/test_news_corroboration_skill.py. These tests lock the DASHBOARD seam:

* The route is mounted and returns the builder's envelope shape.
* ``_corroborating_article_count_index`` counts articles in the window
  ``[trade_ts - lookback_hours, trade_ts)`` — STRICTLY before the trade,
  INCLUSIVE on the lower edge.
* Word-boundary ticker matching (``MU`` must not alias ``MUSE``).
* Query-param clamping for ``limit`` / ``lookback_hours`` /
  ``min_per_bucket`` / ``verdict_gap_pct``.
* The endpoint never 500s on a store fault — it degrades to an ERROR
  envelope so the panel never goes blank.
"""
from __future__ import annotations

import sqlite3
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import MagicMock

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def _seed_articles_db(path: Path, rows: list[tuple[str, str]]) -> None:
    """rows = [(title, first_seen_iso), ...]. Live-eligible articles only."""
    conn = sqlite3.connect(str(path))
    conn.executescript(
        """
        CREATE TABLE IF NOT EXISTS articles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            url TEXT,
            source TEXT,
            title TEXT,
            first_seen TEXT,
            ai_score REAL,
            urgency INTEGER
        );
        """
    )
    for i, (title, fs) in enumerate(rows):
        conn.execute(
            "INSERT INTO articles (url, source, title, first_seen, ai_score, urgency) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (f"https://x.test/{i}", "test", title, fs, 5.0, 0),
        )
    conn.commit()
    conn.close()


# ── Helper: _corroborating_article_count_index ─────────────────────────────

class TestArticleCountHelper:
    """The per-trade window counter that feeds the endpoint."""

    def test_empty_trades_returns_empty_dict(self, tmp_path, monkeypatch):
        import paper_trader.dashboard as d
        db = tmp_path / "articles.db"
        _seed_articles_db(db, [])
        monkeypatch.setattr(d, "_articles_db_path", lambda: db)
        out = d._corroborating_article_count_index([])
        assert out == {}

    def test_no_articles_yields_zero_count_per_trade(self, tmp_path, monkeypatch):
        import paper_trader.dashboard as d
        db = tmp_path / "articles.db"
        _seed_articles_db(db, [])
        monkeypatch.setattr(d, "_articles_db_path", lambda: db)
        trades = [{"id": 1, "ticker": "NVDA",
                   "timestamp": "2026-05-20T10:00:00+00:00"}]
        out = d._corroborating_article_count_index(trades)
        assert out == {1: 0}

    def test_counts_articles_inside_window(self, tmp_path, monkeypatch):
        import paper_trader.dashboard as d
        db = tmp_path / "articles.db"
        # Trade at 10:00; window=24h → counts in [yesterday 10:00, today 10:00).
        # Three NVDA articles inside the window, one outside (40h ago).
        _seed_articles_db(db, [
            ("NVDA beats earnings", "2026-05-19T11:00:00+00:00"),  # 23h before
            ("NVDA up on AI demand", "2026-05-20T03:00:00+00:00"),  # 7h before
            ("NVDA news leak", "2026-05-20T09:30:00+00:00"),        # 30m before
            ("NVDA old story", "2026-05-18T18:00:00+00:00"),        # 40h before — outside
        ])
        monkeypatch.setattr(d, "_articles_db_path", lambda: db)
        trades = [{"id": 1, "ticker": "NVDA",
                   "timestamp": "2026-05-20T10:00:00+00:00"}]
        out = d._corroborating_article_count_index(
            trades, lookback_hours=24.0)
        assert out == {1: 3}

    def test_article_at_or_after_trade_ts_does_not_count(
            self, tmp_path, monkeypatch):
        """Strictly before — an article that lands at the same microsecond as
        the trade was not knowable at decision time."""
        import paper_trader.dashboard as d
        db = tmp_path / "articles.db"
        _seed_articles_db(db, [
            ("NVDA breaking", "2026-05-20T10:00:00+00:00"),     # exactly at
            ("NVDA after", "2026-05-20T10:00:01+00:00"),         # 1s after
            ("NVDA prior", "2026-05-20T09:59:59+00:00"),         # 1s before
        ])
        monkeypatch.setattr(d, "_articles_db_path", lambda: db)
        trades = [{"id": 1, "ticker": "NVDA",
                   "timestamp": "2026-05-20T10:00:00+00:00"}]
        out = d._corroborating_article_count_index(
            trades, lookback_hours=24.0)
        assert out == {1: 1}

    def test_word_boundary_ticker_match(self, tmp_path, monkeypatch):
        """``MU`` must NOT alias ``MUSE`` / ``MUTUAL`` / ``MULTI``."""
        import paper_trader.dashboard as d
        db = tmp_path / "articles.db"
        _seed_articles_db(db, [
            ("MU beats earnings", "2026-05-20T09:00:00+00:00"),     # match
            ("MUSE.AI launches", "2026-05-20T09:10:00+00:00"),      # no
            ("MUTUAL fund flows", "2026-05-20T09:15:00+00:00"),     # no
            ("MULTIPLE upgrades", "2026-05-20T09:20:00+00:00"),     # no
        ])
        monkeypatch.setattr(d, "_articles_db_path", lambda: db)
        trades = [{"id": 1, "ticker": "MU",
                   "timestamp": "2026-05-20T10:00:00+00:00"}]
        out = d._corroborating_article_count_index(
            trades, lookback_hours=24.0)
        assert out == {1: 1}

    def test_per_trade_window_is_per_trade_anchored(
            self, tmp_path, monkeypatch):
        """Articles between two trades count for the LATER trade only when
        they fall within ITS window."""
        import paper_trader.dashboard as d
        db = tmp_path / "articles.db"
        _seed_articles_db(db, [
            ("NVDA early news", "2026-05-19T08:00:00+00:00"),
            # Trade 1 at 09:00 sees the 08:00 article (1h before).
            # Trade 2 at 11:00 next day → 27h ago, outside 24h window.
            ("NVDA late news", "2026-05-20T10:30:00+00:00"),
            # Trade 2 at 11:00 sees the 10:30 article (30m before).
        ])
        monkeypatch.setattr(d, "_articles_db_path", lambda: db)
        trades = [
            {"id": 1, "ticker": "NVDA",
             "timestamp": "2026-05-19T09:00:00+00:00"},
            {"id": 2, "ticker": "NVDA",
             "timestamp": "2026-05-20T11:00:00+00:00"},
        ]
        out = d._corroborating_article_count_index(
            trades, lookback_hours=24.0)
        assert out == {1: 1, 2: 1}

    def test_missing_db_path_degrades_to_zero(self, monkeypatch):
        import paper_trader.dashboard as d
        monkeypatch.setattr(d, "_articles_db_path", lambda: None)
        trades = [{"id": 1, "ticker": "NVDA",
                   "timestamp": "2026-05-20T10:00:00+00:00"}]
        out = d._corroborating_article_count_index(trades)
        assert out == {1: 0}

    def test_trade_with_no_ticker_emits_zero_count(self, tmp_path, monkeypatch):
        """An empty-ticker trade gets ``article_count=0`` (operationally
        equivalent to NO_NEWS). Downstream the endpoint drops it for a
        different reason — no realized_pct entry — so the bucket never
        sees it; but the helper itself is honest about "0 known articles
        for this ticker" rather than silently skipping the row."""
        import paper_trader.dashboard as d
        db = tmp_path / "articles.db"
        _seed_articles_db(db, [])
        monkeypatch.setattr(d, "_articles_db_path", lambda: db)
        trades = [
            {"id": 1, "ticker": "", "timestamp": "2026-05-20T10:00:00+00:00"},
            {"id": 2, "ticker": "NVDA",
             "timestamp": "2026-05-20T10:00:00+00:00"},
        ]
        out = d._corroborating_article_count_index(trades)
        assert out == {1: 0, 2: 0}


# ── Endpoint: /api/news-corroboration-skill ───────────────────────────────

@pytest.fixture
def client(tmp_path, monkeypatch):
    import paper_trader.dashboard as d

    fake_store = MagicMock()
    fake_store.recent_trades.return_value = []
    fake_store.open_positions.return_value = []
    monkeypatch.setattr(d, "get_store", lambda: fake_store)

    db = tmp_path / "articles.db"
    _seed_articles_db(db, [])
    monkeypatch.setattr(d, "_articles_db_path", lambda: db)
    with d.app.test_client() as c:
        yield c


def test_route_exists_and_returns_envelope_shape(client):
    r = client.get("/api/news-corroboration-skill")
    assert r.status_code == 200
    j = r.get_json()
    for k in ("verdict", "headline", "n_samples", "buckets", "thresholds",
              "samples", "as_of"):
        assert k in j


def test_empty_book_reads_insufficient_data(client):
    j = client.get("/api/news-corroboration-skill").get_json()
    assert j["verdict"] == "INSUFFICIENT_DATA"
    assert j["n_samples"] == 0


def test_thresholds_echo_query_params(client):
    j = client.get(
        "/api/news-corroboration-skill"
        "?min_per_bucket=5&verdict_gap_pct=4.5"
    ).get_json()
    assert j["thresholds"]["min_per_bucket"] == 5
    # gap is a float — exact echo since we clamp inside the supported range.
    assert j["thresholds"]["verdict_gap_pct"] == 4.5


def test_query_params_clamped(client):
    """``min_per_bucket`` cap is 20, ``verdict_gap_pct`` cap is 20."""
    j = client.get(
        "/api/news-corroboration-skill"
        "?min_per_bucket=999&verdict_gap_pct=999"
    ).get_json()
    assert j["thresholds"]["min_per_bucket"] == 20
    assert j["thresholds"]["verdict_gap_pct"] == 20.0


def test_invalid_query_params_fall_back_to_defaults(client):
    j = client.get(
        "/api/news-corroboration-skill"
        "?min_per_bucket=abc&verdict_gap_pct=xyz"
    ).get_json()
    # Defaults: 3 / 2.0 (see paper_trader/analytics/news_corroboration_skill.py)
    assert j["thresholds"]["min_per_bucket"] == 3
    assert j["thresholds"]["verdict_gap_pct"] == 2.0


def test_endpoint_renders_verdict_with_real_samples(tmp_path, monkeypatch):
    """End-to-end: seed the store with closed round-trips that have known
    corroboration counts, expect the builder's verdict to surface."""
    import paper_trader.dashboard as d

    # 6 closed round-trips on NVDA — three on a SINGLE article (-2% mean),
    # three on a CHORUS of 5 articles (+3% mean). gap = +5pp ⇒
    # CORROBORATION_HELPS at default gap=2.0pp.
    trades = []
    tid = 0
    base_buy_price = 100.0
    sells_per_outcome = []  # (loss_pct, gain_pct) BUY-SELL pairs
    # SINGLE-article BUYs (one article each, all losses → -2%)
    for i in range(3):
        ts_buy = f"2026-05-1{i}T10:00:00+00:00"
        ts_sell = f"2026-05-1{i}T15:00:00+00:00"
        tid += 1
        trades.append({"id": tid, "ticker": "NVDA", "action": "BUY",
                       "qty": 1.0, "price": base_buy_price,
                       "value": base_buy_price, "timestamp": ts_buy,
                       "option_type": None, "strike": None, "expiry": None})
        tid += 1
        sell_px = base_buy_price * 0.98
        trades.append({"id": tid, "ticker": "NVDA", "action": "SELL",
                       "qty": 1.0, "price": sell_px,
                       "value": sell_px, "timestamp": ts_sell,
                       "option_type": None, "strike": None, "expiry": None})
    # CHORUS-of-5 BUYs (5 articles each, all wins → +3%)
    for i in range(3):
        day = 14 + i
        ts_buy = f"2026-05-{day}T10:00:00+00:00"
        ts_sell = f"2026-05-{day}T15:00:00+00:00"
        tid += 1
        trades.append({"id": tid, "ticker": "NVDA", "action": "BUY",
                       "qty": 1.0, "price": base_buy_price,
                       "value": base_buy_price, "timestamp": ts_buy,
                       "option_type": None, "strike": None, "expiry": None})
        tid += 1
        sell_px = base_buy_price * 1.03
        trades.append({"id": tid, "ticker": "NVDA", "action": "SELL",
                       "qty": 1.0, "price": sell_px,
                       "value": sell_px, "timestamp": ts_sell,
                       "option_type": None, "strike": None, "expiry": None})

    fake_store = MagicMock()
    # recent_trades returns newest-first; the endpoint reverses.
    fake_store.recent_trades.return_value = list(reversed(trades))
    fake_store.open_positions.return_value = []
    monkeypatch.setattr(d, "get_store", lambda: fake_store)

    # Seed articles.db: 1 article for each of the SINGLE BUYs (May 10-12),
    # 5 articles each for the CHORUS BUYs (May 14-16) — all within 1h before.
    rows = []
    for i in range(3):
        rows.append((f"NVDA news {i}",
                     f"2026-05-1{i}T09:30:00+00:00"))
    for i in range(3):
        day = 14 + i
        for k in range(5):
            rows.append(
                (f"NVDA chorus {day}-{k}",
                 f"2026-05-{day}T09:{10 + k:02d}:00+00:00")
            )
    db = tmp_path / "articles.db"
    _seed_articles_db(db, rows)
    monkeypatch.setattr(d, "_articles_db_path", lambda: db)

    with d.app.test_client() as c:
        j = c.get("/api/news-corroboration-skill").get_json()
    assert j["verdict"] == "CORROBORATION_HELPS", j
    assert j["n_samples"] == 6
    assert j["buckets"]["SINGLE"]["n"] == 3
    assert j["buckets"]["CHORUS"]["n"] == 3
    assert j["buckets"]["SINGLE"]["mean_pct"] == pytest.approx(-2.0, abs=0.01)
    assert j["buckets"]["CHORUS"]["mean_pct"] == pytest.approx(3.0, abs=0.01)


def test_endpoint_never_500s_on_store_failure(monkeypatch):
    import paper_trader.dashboard as d

    def _boom():
        raise RuntimeError("store down")

    monkeypatch.setattr(d, "get_store", _boom)
    with d.app.test_client() as c:
        r = c.get("/api/news-corroboration-skill")
    assert r.status_code == 500
    body = r.get_json()
    assert body["verdict"] == "ERROR"
    assert "store down" in body["headline"]


def test_cors_header_present_for_cross_fetch(client):
    r = client.get("/api/news-corroboration-skill")
    assert r.headers.get("Access-Control-Allow-Origin") == "*"
