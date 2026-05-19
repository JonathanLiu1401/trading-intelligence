"""Tests for analytics/idle_opportunity.py + /api/idle-opportunity.

Contract:
* Pure ``build_idle_opportunity`` composes ``build_decision_drought``'s
  ``current_drought`` block verbatim (SSOT, AGENTS.md #10). State ladder:
  NO_DATA → NO_DROUGHT → OK (with or without opportunities).
* Articles in the drought window with ai_score ≥ floor mentioning a
  watchlist ticker are bucketed per ticker. Per-ticker keep the top
  article (score DESC, tie-break newer first_seen).
* Sort: top_score DESC then most-recent first_seen DESC then ticker ASC.
* Word-boundary regex (MU ≠ MUTUAL, $NVDA cashtag hits, AMD ≠ AMDOCS) —
  matches news_velocity / trade_attribution / ticker_sentiments.
* held_tickers flags rows for HELD positions.
* NaN/Inf ai_score rejected (digital-intern column has been observed with
  stale NaNs).
* Garbage rows (None, missing fields, unparseable ts) degrade — never raise.
* Endpoint Flask test_client: NO_DROUGHT short-circuit (no DB read), real
  ongoing-drought scenario with seeded articles.db.
"""
from __future__ import annotations

import os
import sqlite3
import tempfile
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from paper_trader.analytics.idle_opportunity import (
    DEFAULT_MIN_AI_SCORE,
    _ticker_regex,
    _safe_float,
    build_idle_opportunity,
)

NOW = datetime(2026, 5, 19, 12, 0, 0, tzinfo=timezone.utc)


def _drought(start: datetime, end: datetime | None = None,
             ongoing: bool = True, n_no_decision: int = 33,
             n_hold: int = 7, n_blocked: int = 0,
             kind: str = "PARALYSIS") -> dict:
    """Shape matches build_decision_drought.current_drought."""
    end = end or NOW
    dur = round((end - start).total_seconds() / 3600.0, 2)
    return {
        "current_drought": {
            "start": start.isoformat(timespec="seconds"),
            "end": end.isoformat(timespec="seconds"),
            "duration_hours": dur,
            "n_cycles": n_no_decision + n_hold + n_blocked,
            "n_no_decision": n_no_decision,
            "n_hold": n_hold,
            "n_blocked": n_blocked,
            "kind": kind,
            "ongoing": ongoing,
        }
    }


def _article(ticker_in_title: str, ai_score: float, first_seen: datetime,
             title_extra: str = "", urgency: int = 0,
             source: str = "rss", url: str | None = None) -> dict:
    return {
        "title": f"{ticker_in_title} {title_extra}".strip(),
        "ai_score": ai_score,
        "urgency": urgency,
        "first_seen": first_seen.isoformat(timespec="seconds"),
        "source": source,
        "url": url or f"https://example.com/{abs(hash(title_extra)) % 99999}",
    }


# ─── State ladder ──────────────────────────────────────────────────


class TestStateLadder:
    def test_no_drought_block_returns_no_data(self):
        r = build_idle_opportunity(None, [], ["NVDA"], now=NOW)
        assert r["state"] == "NO_DATA"
        assert r["n_opportunities"] == 0
        assert "no decisions recorded yet" in r["headline"]

    def test_empty_drought_dict_returns_no_drought(self):
        # build_decision_drought returns this when no current drought.
        r = build_idle_opportunity({"current_drought": None},
                                   [], ["NVDA"], now=NOW)
        assert r["state"] == "NO_DROUGHT"
        assert "filling normally" in r["headline"]
        assert r["drought"] is None

    def test_closed_drought_returns_no_drought(self):
        # Even if the dict has a drought block, ongoing=False → no_drought.
        d = _drought(NOW - timedelta(hours=2), ongoing=False)
        r = build_idle_opportunity(d, [], ["NVDA"], now=NOW)
        assert r["state"] == "NO_DROUGHT"

    def test_ongoing_drought_no_articles_is_ok_quiet(self):
        d = _drought(NOW - timedelta(hours=8))
        r = build_idle_opportunity(d, [], ["NVDA"], now=NOW)
        assert r["state"] == "OK"
        assert r["n_opportunities"] == 0
        assert "no live watchlist signals" in r["headline"]
        assert r["missed_top_score"] is None

    def test_ongoing_drought_with_high_score_article_is_ok_regret(self):
        d = _drought(NOW - timedelta(hours=8))
        arts = [_article("NVDA", 9.0, NOW - timedelta(hours=3))]
        r = build_idle_opportunity(d, arts, ["NVDA"], now=NOW)
        assert r["state"] == "OK"
        assert r["n_opportunities"] == 1
        assert r["missed_top_ticker"] == "NVDA"
        assert r["missed_top_score"] == 9.0
        assert "loudest: NVDA" in r["headline"]

    def test_held_tag_appears_in_headline(self):
        d = _drought(NOW - timedelta(hours=8))
        arts = [_article("NVDA", 9.0, NOW - timedelta(hours=3))]
        r = build_idle_opportunity(d, arts, ["NVDA"], held_tickers=["NVDA"],
                                   now=NOW)
        assert "(HELD)" in r["headline"]
        assert r["opportunities"][0]["held"] is True


# ─── Drought window strict-inclusive boundary ─────────────────────


class TestDroughtWindow:
    def test_article_before_drought_start_excluded(self):
        d = _drought(NOW - timedelta(hours=4))
        # Article from 5h ago — pre-drought.
        arts = [_article("NVDA", 9.0, NOW - timedelta(hours=5))]
        r = build_idle_opportunity(d, arts, ["NVDA"], now=NOW)
        assert r["n_opportunities"] == 0

    def test_article_at_drought_start_included(self):
        # ≥ start is inclusive (matches signals.py first_seen >= cutoff).
        start = NOW - timedelta(hours=4)
        d = _drought(start)
        arts = [_article("NVDA", 9.0, start)]
        r = build_idle_opportunity(d, arts, ["NVDA"], now=NOW)
        assert r["n_opportunities"] == 1

    def test_article_after_drought_end_included(self):
        # Drought is ongoing; "end" in the block is just snapshot end —
        # articles right now are still inside the live drought window.
        d = _drought(NOW - timedelta(hours=8))
        arts = [_article("NVDA", 9.0, NOW - timedelta(minutes=1))]
        r = build_idle_opportunity(d, arts, ["NVDA"], now=NOW)
        assert r["n_opportunities"] == 1


# ─── ai_score floor ───────────────────────────────────────────────


class TestScoreFloor:
    def test_below_floor_excluded(self):
        d = _drought(NOW - timedelta(hours=8))
        arts = [_article("NVDA", 5.9, NOW - timedelta(hours=1))]
        r = build_idle_opportunity(d, arts, ["NVDA"], now=NOW,
                                   min_ai_score=6.0)
        assert r["n_opportunities"] == 0

    def test_at_floor_included(self):
        d = _drought(NOW - timedelta(hours=8))
        arts = [_article("NVDA", 6.0, NOW - timedelta(hours=1))]
        r = build_idle_opportunity(d, arts, ["NVDA"], now=NOW,
                                   min_ai_score=6.0)
        assert r["n_opportunities"] == 1

    def test_min_ai_score_carried_in_payload(self):
        d = _drought(NOW - timedelta(hours=8))
        r = build_idle_opportunity(d, [], ["NVDA"], now=NOW,
                                   min_ai_score=7.5)
        assert r["min_ai_score"] == 7.5

    def test_nan_ai_score_rejected(self):
        d = _drought(NOW - timedelta(hours=8))
        bad = _article("NVDA", float("nan"), NOW - timedelta(hours=1))
        r = build_idle_opportunity(d, [bad], ["NVDA"], now=NOW)
        assert r["n_opportunities"] == 0

    def test_inf_ai_score_rejected(self):
        d = _drought(NOW - timedelta(hours=8))
        bad = _article("NVDA", float("inf"), NOW - timedelta(hours=1))
        r = build_idle_opportunity(d, [bad], ["NVDA"], now=NOW)
        assert r["n_opportunities"] == 0


# ─── Word-boundary regex ──────────────────────────────────────────


class TestWordBoundary:
    def test_mu_does_not_alias_mutual(self):
        d = _drought(NOW - timedelta(hours=8))
        # "Mutual funds" must NOT bucket under MU.
        arts = [_article("Mutual", 9.0, NOW - timedelta(hours=1),
                         title_extra="funds rebound")]
        r = build_idle_opportunity(d, arts, ["MU"], now=NOW)
        assert r["n_opportunities"] == 0

    def test_amd_does_not_alias_amdocs(self):
        d = _drought(NOW - timedelta(hours=8))
        arts = [_article("AMDOCS", 9.0, NOW - timedelta(hours=1),
                         title_extra="signs new contract")]
        r = build_idle_opportunity(d, arts, ["AMD"], now=NOW)
        assert r["n_opportunities"] == 0

    def test_cashtag_dollar_hits(self):
        d = _drought(NOW - timedelta(hours=8))
        arts = [_article("$NVDA", 9.0, NOW - timedelta(hours=1),
                         title_extra="earnings preview")]
        r = build_idle_opportunity(d, arts, ["NVDA"], now=NOW)
        assert r["n_opportunities"] == 1

    def test_compiled_pattern_helper(self):
        # Lock the helper directly so future drift in the regex shape is
        # caught even if no caller is exercised.
        pat = _ticker_regex("MU")
        assert pat.search("MU SURGES ON DRAM")
        assert pat.search("$MU UPGRADED")
        assert not pat.search("MUTUAL FUNDS REBALANCE")
        assert not pat.search("MUNICH STOCKS")


# ─── Per-ticker bucketing + tie-break ─────────────────────────────


class TestBucketing:
    def test_multiple_articles_same_ticker_bucket_count(self):
        d = _drought(NOW - timedelta(hours=8))
        arts = [
            _article("NVDA", 7.5, NOW - timedelta(hours=4),
                     title_extra="Lead 1"),
            _article("NVDA", 8.5, NOW - timedelta(hours=3),
                     title_extra="Lead 2 — top score"),
            _article("NVDA", 8.0, NOW - timedelta(hours=2),
                     title_extra="Lead 3"),
        ]
        r = build_idle_opportunity(d, arts, ["NVDA"], now=NOW)
        assert r["n_opportunities"] == 1
        row = r["opportunities"][0]
        assert row["article_count"] == 3
        assert row["top_score"] == 8.5
        assert "Lead 2" in row["top_title"]

    def test_tie_break_score_equal_newer_wins(self):
        d = _drought(NOW - timedelta(hours=8))
        old = _article("NVDA", 8.0, NOW - timedelta(hours=4),
                       title_extra="OLDER")
        newer = _article("NVDA", 8.0, NOW - timedelta(hours=1),
                         title_extra="NEWER")
        # Order in input should not matter — bucketing is order-independent.
        for arts in ([old, newer], [newer, old]):
            r = build_idle_opportunity(d, arts, ["NVDA"], now=NOW)
            assert "NEWER" in r["opportunities"][0]["top_title"]

    def test_max_urgency_carried(self):
        d = _drought(NOW - timedelta(hours=8))
        arts = [
            _article("NVDA", 7.5, NOW - timedelta(hours=3), urgency=0),
            _article("NVDA", 7.5, NOW - timedelta(hours=2), urgency=2),
            _article("NVDA", 6.5, NOW - timedelta(hours=1), urgency=1),
        ]
        r = build_idle_opportunity(d, arts, ["NVDA"], now=NOW)
        assert r["opportunities"][0]["max_urgency"] == 2


# ─── Sort + cap ───────────────────────────────────────────────────


class TestSortAndCap:
    def test_sort_top_score_desc(self):
        d = _drought(NOW - timedelta(hours=8))
        arts = [
            _article("MU", 7.0, NOW - timedelta(hours=1)),
            _article("NVDA", 9.0, NOW - timedelta(hours=1)),
            _article("TSLA", 8.0, NOW - timedelta(hours=1)),
        ]
        r = build_idle_opportunity(d, arts, ["NVDA", "MU", "TSLA"], now=NOW)
        tickers = [o["ticker"] for o in r["opportunities"]]
        assert tickers == ["NVDA", "TSLA", "MU"]

    def test_max_opportunities_cap(self):
        d = _drought(NOW - timedelta(hours=8))
        wl = [f"TKR{i}" for i in range(5)]
        # Build one article per ticker with same score
        arts = [_article(t, 9.0, NOW - timedelta(hours=1)) for t in wl]
        r = build_idle_opportunity(d, arts, wl, now=NOW, max_opportunities=3)
        assert r["n_opportunities"] == 3


# ─── Held flag ────────────────────────────────────────────────────


class TestHeld:
    def test_held_set_correctly(self):
        d = _drought(NOW - timedelta(hours=8))
        arts = [
            _article("NVDA", 9.0, NOW - timedelta(hours=1)),
            _article("MU", 8.0, NOW - timedelta(hours=1)),
        ]
        r = build_idle_opportunity(d, arts, ["NVDA", "MU"],
                                   held_tickers=["NVDA"], now=NOW)
        by_t = {o["ticker"]: o for o in r["opportunities"]}
        assert by_t["NVDA"]["held"] is True
        assert by_t["MU"]["held"] is False


# ─── Garbage / degrade ────────────────────────────────────────────


class TestDegradeNeverRaises:
    def test_none_articles(self):
        d = _drought(NOW - timedelta(hours=4))
        r = build_idle_opportunity(d, None, ["NVDA"], now=NOW)
        assert r["state"] == "OK"
        assert r["n_opportunities"] == 0

    def test_garbage_row_skipped(self):
        d = _drought(NOW - timedelta(hours=4))
        arts = [
            None,
            "not a dict",
            {"title": "missing score", "first_seen": NOW.isoformat()},
            {"ai_score": 9.0, "first_seen": NOW.isoformat()},  # no title
            {"title": "NVDA up big", "ai_score": "not-a-number",
             "first_seen": NOW.isoformat()},
            _article("NVDA", 9.0, NOW - timedelta(hours=1)),  # the real one
        ]
        r = build_idle_opportunity(d, arts, ["NVDA"], now=NOW)
        assert r["n_opportunities"] == 1

    def test_unparseable_first_seen_skipped(self):
        d = _drought(NOW - timedelta(hours=4))
        arts = [{"title": "NVDA news", "ai_score": 9.0,
                 "first_seen": "not-a-timestamp"}]
        r = build_idle_opportunity(d, arts, ["NVDA"], now=NOW)
        assert r["n_opportunities"] == 0

    def test_empty_watchlist_is_ok_empty(self):
        d = _drought(NOW - timedelta(hours=4))
        arts = [_article("NVDA", 9.0, NOW - timedelta(hours=1))]
        r = build_idle_opportunity(d, arts, [], now=NOW)
        assert r["state"] == "OK"
        assert r["n_opportunities"] == 0

    def test_safe_float_helper(self):
        assert _safe_float(None) is None
        assert _safe_float("nope") is None
        assert _safe_float(float("nan")) is None
        assert _safe_float(float("inf")) is None
        assert _safe_float(7.5) == 7.5
        assert _safe_float("7.5") == 7.5
        assert _safe_float(0) == 0.0

    def test_drought_with_no_start_returns_no_data(self):
        # Defensive — a malformed drought block should not flood the table.
        d = {"current_drought": {"ongoing": True, "start": None}}
        arts = [_article("NVDA", 9.0, NOW - timedelta(hours=1))]
        r = build_idle_opportunity(d, arts, ["NVDA"], now=NOW)
        assert r["state"] == "NO_DATA"


# ─── Endpoint Flask test_client ───────────────────────────────────


@pytest.fixture
def seeded_articles_db(tmp_path):
    """Build a minimal articles.db that matches digital-intern's schema
    enough for the endpoint's query to succeed."""
    db = tmp_path / "articles.db"
    conn = sqlite3.connect(str(db))
    conn.executescript(
        """
        CREATE TABLE articles (
            id TEXT PRIMARY KEY,
            url TEXT NOT NULL,
            title TEXT NOT NULL,
            source TEXT,
            published TEXT,
            kw_score REAL DEFAULT 0,
            ai_score REAL DEFAULT 0,
            urgency INTEGER DEFAULT 0,
            full_text BLOB,
            first_seen TEXT NOT NULL,
            cycle INTEGER DEFAULT 0,
            time_sensitivity REAL
        );
        """
    )
    rows = [
        # Inside drought window, high score, watchlist ticker → SHOULD appear.
        ("a1", "https://x/1", "NVDA earnings beat preview",
         "rss", 9.0, 0,
         (NOW - timedelta(hours=2)).isoformat(timespec="seconds")),
        # Below floor → drops.
        ("a2", "https://x/2", "NVDA chatter",
         "rss", 5.0, 0,
         (NOW - timedelta(hours=1)).isoformat(timespec="seconds")),
        # Synthetic — backtest row must be filtered by SQL.
        ("a3", "backtest://run_1/NVDA", "NVDA backtest winner",
         "backtest_run_1_winner", 9.5, 0,
         (NOW - timedelta(hours=1)).isoformat(timespec="seconds")),
    ]
    conn.executemany(
        "INSERT INTO articles (id,url,title,source,ai_score,urgency,first_seen) "
        "VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()
    return db


def _bootstrap_dashboard_app(tmp_path, monkeypatch, seeded_db: Path | None = None):
    """Spin up the Flask app pointed at a temp paper_trader.db so the live
    DB never leaks in; monkeypatch ``_articles_db_path`` to the seeded news DB.

    Same pattern as ``test_recovery.TestRecoveryEndpoint._seed_underwater``:
    Store reads from module-global ``DB_PATH``, so monkeypatch that and the
    singleton before constructing.
    """
    from paper_trader import store as store_mod
    from paper_trader.store import Store
    from paper_trader import dashboard as dash
    monkeypatch.setattr(store_mod, "DB_PATH", tmp_path / "paper_trader.db")
    monkeypatch.setattr(store_mod, "_singleton", None)
    fresh_store = Store()
    monkeypatch.setattr(dash, "get_store", lambda: fresh_store)
    if seeded_db is not None:
        monkeypatch.setattr(dash, "_articles_db_path", lambda: seeded_db)
    dash.app.config["TESTING"] = True
    return dash, fresh_store


class TestIdleOpportunityEndpoint:
    def test_no_drought_short_circuits_with_no_db_read(
            self, tmp_path, monkeypatch):
        """When there's no ongoing drought we must NOT read articles.db —
        verify by passing a path that would explode if read."""
        dash, store = _bootstrap_dashboard_app(tmp_path, monkeypatch)
        # Don't seed any decisions → build_decision_drought returns NO_DATA
        # → no current_drought → endpoint short-circuits.

        # Sentinel: setting _articles_db_path to a nonexistent path; the
        # short-circuit means it must never be opened. (We don't monkeypatch
        # sqlite3 itself because the short-circuit path doesn't touch it.)
        monkeypatch.setattr(dash, "_articles_db_path",
                            lambda: tmp_path / "does-not-exist.db")

        client = dash.app.test_client()
        resp = client.get("/api/idle-opportunity")
        assert resp.status_code == 200
        body = resp.get_json()
        # NO_DATA when no decisions exist (build_decision_drought path) — or
        # NO_DROUGHT after a fresh-fill scenario. Both prove no DB read
        # happened (the path would have raised).
        assert body["state"] in {"NO_DATA", "NO_DROUGHT"}
        assert body["n_opportunities"] == 0

    def test_endpoint_against_real_drought_returns_opportunities(
            self, tmp_path, monkeypatch, seeded_articles_db):
        dash, store = _bootstrap_dashboard_app(
            tmp_path, monkeypatch, seeded_db=seeded_articles_db,
        )
        # Seed an ongoing PARALYSIS drought directly via the connection so
        # we can pin custom timestamps (store.record_decision uses _now()).
        base = NOW - timedelta(hours=6)
        rows = [
            (base.isoformat(timespec="seconds"), "BUY NVDA → FILLED"),
            *[
                ((base + timedelta(hours=i)).isoformat(timespec="seconds"),
                 "NO_DECISION")
                for i in range(1, 6)
            ],
        ]
        for ts, act in rows:
            store.conn.execute(
                "INSERT INTO decisions (timestamp, market_open, signal_count, "
                "action_taken, reasoning, portfolio_value, cash) "
                "VALUES (?, 0, 1, ?, 'seed', 1000.0, 500.0)",
                (ts, act),
            )
        store.conn.commit()

        client = dash.app.test_client()
        resp = client.get("/api/idle-opportunity")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["state"] == "OK"
        # The seeded NVDA row at ai_score=9.0 should be there.
        tickers = [o["ticker"] for o in body["opportunities"]]
        # NVDA is on the live strategy.WATCHLIST.
        assert "NVDA" in tickers
        nvda_row = [o for o in body["opportunities"] if o["ticker"] == "NVDA"][0]
        assert nvda_row["top_score"] == 9.0
        # Synthetic backtest row at score 9.5 must NOT appear — SQL-side
        # live-only clause (invariant #3) filtered it out.
        assert nvda_row["top_score"] != 9.5

    def test_endpoint_clamps_min_ai_score_param(
            self, tmp_path, monkeypatch, seeded_articles_db):
        dash, store = _bootstrap_dashboard_app(
            tmp_path, monkeypatch, seeded_db=seeded_articles_db,
        )
        # Need an ongoing drought for the endpoint to read the DB.
        base = NOW - timedelta(hours=3)
        for ts_off, act in [(0, "BUY MU → FILLED"),
                            (1, "NO_DECISION"), (2, "NO_DECISION")]:
            store.conn.execute(
                "INSERT INTO decisions (timestamp, market_open, signal_count, "
                "action_taken, reasoning, portfolio_value, cash) "
                "VALUES (?, 0, 1, ?, 'seed', 1000.0, 500.0)",
                ((base + timedelta(hours=ts_off)).isoformat(timespec="seconds"),
                 act),
            )
        store.conn.commit()

        client = dash.app.test_client()
        # Garbage param → clamped to default; should still serve 200.
        resp = client.get("/api/idle-opportunity?min_ai_score=nope")
        assert resp.status_code == 200
        body = resp.get_json()
        assert body["min_ai_score"] == DEFAULT_MIN_AI_SCORE

        # Out-of-range → clamped.
        resp = client.get("/api/idle-opportunity?min_ai_score=99")
        body = resp.get_json()
        assert body["min_ai_score"] == 10.0
