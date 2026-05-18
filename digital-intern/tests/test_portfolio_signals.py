"""Deterministic, always-fresh per-held-ticker live-news digest.

The 5h Opus heartbeat briefing (``/api/briefings``) is the synthesised
"what matters" view but is up to 5h stale AND skipped entirely whenever the
Claude org usage limit is exhausted — a chronic, repeatedly-documented failure
mode (paper-trader CLAUDE.md §11; "NO_DECISION = quota, not JSON").
``/api/articles`` is the raw unbucketed feed; ``/api/sector-pulse`` is
sector-level density; ``/api/trends`` is aggregate sentiment over time. None
answers the held-book trader's between-briefings question: *"what fresh news
touches MY exact positions right now, ranked, with the one headline to read
first?"* — and answer it WITHOUT any Claude/LLM call, so it still works while
the briefing is dark on quota.

``build_portfolio_signals`` composes the briefing SSOT helpers verbatim
(``_filter_quote_widget_noise`` + ``_book_tickers`` + ``_rank_by_decayed_score``
from ``analysis.claude_analyst``) so the held universe and the recency decay
can never drift from what the 5h Opus digest itself uses. These tests pin:
bucketing correctness (word-boundary, MU≠Micron), informative absence (a held
ticker with no fresh news is still listed), daemon-parity of the held universe
(anti-drift), recency-decay ordering, quote-widget noise suppression, urgency
surfacing, read-only purity, the no-LLM/no-subprocess guarantee (the headline
"survives quota" claim, made falsifiable), and the route's auth + window clamp.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest

from analysis.claude_analyst import _BOOK_TICKERS
from dashboard.web_server import build_portfolio_signals, create_app

NOW = datetime(2026, 5, 18, 12, 0, 0, tzinfo=timezone.utc)


def _art(title, *, ai_score=8.0, urgency=0, age_h=0.0, ts=1.0,
         source="rss", summary=""):
    """One article row in the shape dashboard reads out of articles.db."""
    return {
        "title": title,
        "summary": summary,
        "source": source,
        "ai_score": ai_score,
        "urgency": urgency,
        "time_sensitivity": ts,
        "first_seen": (NOW - timedelta(hours=age_h)).isoformat(),
    }


def _by_ticker(result):
    return {t["ticker"]: t for t in result["tickers"]}


# ── bucketing correctness ────────────────────────────────────────────────────
class TestBucketing:
    def test_article_buckets_to_its_held_ticker(self):
        r = build_portfolio_signals([_art("MU beats Q3 estimates")], now=NOW)
        mu = _by_ticker(r)["MU"]
        assert mu["n_articles"] == 1
        assert mu["top_headline"] == "MU beats Q3 estimates"

    def test_unmentioned_held_ticker_present_with_zero(self):
        # Informative absence: a pinned book explicitly wants to see
        # "no fresh news on LITE" — the ticker must still be listed.
        r = build_portfolio_signals([_art("MU beats Q3 estimates")], now=NOW)
        lite = _by_ticker(r)["LITE"]
        assert lite["n_articles"] == 0
        assert lite["top_headline"] is None
        assert lite["top_score"] == 0.0
        assert lite["max_urgency"] == 0

    def test_all_book_tickers_present_ssot(self):
        # Anti-drift: the bucket universe IS claude_analyst._BOOK_TICKERS
        # (daemon-parity), never a re-derived list — same discipline as
        # tests/test_briefing_book_tag.py pins the briefing side.
        r = build_portfolio_signals([], now=NOW)
        assert {t["ticker"] for t in r["tickers"]} == set(_BOOK_TICKERS)

    def test_word_boundary_mu_not_inside_micron(self):
        r = build_portfolio_signals(
            [_art("Micron museum MUSEUM micromu opens")], now=NOW)
        assert _by_ticker(r)["MU"]["n_articles"] == 0

    def test_one_article_two_tickers_buckets_into_both(self):
        r = build_portfolio_signals(
            [_art("MU and NVDA both rip on AI demand", ai_score=9.0)], now=NOW)
        bt = _by_ticker(r)
        assert bt["MU"]["n_articles"] == 1
        assert bt["NVDA"]["n_articles"] == 1


# ── ranking / recency decay (SSOT _rank_by_decayed_score) ────────────────────
class TestRanking:
    def test_tickers_sorted_by_decayed_score_desc(self):
        r = build_portfolio_signals(
            [_art("MU surges", ai_score=9.0),
             _art("NVDA dips", ai_score=4.0)], now=NOW)
        order = [t["ticker"] for t in r["tickers"]]
        assert order.index("MU") < order.index("NVDA")
        # Both name-bearing tickers must rank ahead of every zero-news one.
        assert order.index("NVDA") < order.index("LITE")

    def test_recency_decay_promotes_fresher_within_ticker(self):
        # Same base score; the fresher article must be the surfaced headline.
        fresh = _art("MU fresh catalyst", ai_score=8.0, age_h=1.0, ts=1.0)
        stale = _art("MU old news", ai_score=8.0, age_h=25.0, ts=1.0)
        r = build_portfolio_signals([stale, fresh], now=NOW)
        mu = _by_ticker(r)["MU"]
        assert mu["top_headline"] == "MU fresh catalyst"
        assert mu["headlines"][0]["title"] == "MU fresh catalyst"

    def test_headlines_capped(self):
        arts = [_art(f"MU headline {i}", ai_score=5.0, age_h=i)
                for i in range(12)]
        r = build_portfolio_signals(arts, now=NOW)
        mu = _by_ticker(r)["MU"]
        assert mu["n_articles"] == 12
        assert len(mu["headlines"]) <= 5


# ── noise + urgency ──────────────────────────────────────────────────────────
class TestNoiseAndUrgency:
    def test_quote_widget_pseudo_article_suppressed(self):
        # The analyst's documented #1 noise complaint: a live ticker-tape
        # entry ("MUMicron Technology72.10-3.10(-4.12%)") masquerading as a
        # high-score article. SSOT _filter_quote_widget_noise must drop it.
        widget = _art("MUMicron Technology72.10-3.10(-4.12%)", ai_score=9.9)
        real = _art("MU lands hyperscaler HBM deal", ai_score=8.0)
        r = build_portfolio_signals([widget, real], now=NOW)
        mu = _by_ticker(r)["MU"]
        assert mu["n_articles"] == 1
        assert mu["top_headline"] == "MU lands hyperscaler HBM deal"
        assert r["n_quote_widget_suppressed"] == 1

    def test_max_urgency_surfaced(self):
        r = build_portfolio_signals(
            [_art("MU minor note", urgency=0),
             _art("MU URGENT recall", urgency=2)], now=NOW)
        assert _by_ticker(r)["MU"]["max_urgency"] == 2


# ── purity / no-LLM guarantee (the "survives quota" claim, falsifiable) ──────
class TestPurity:
    def test_builder_does_not_mutate_input_articles(self):
        arts = [_art("MU beats Q3")]
        before = dict(arts[0])
        build_portfolio_signals(arts, now=NOW)
        assert arts[0] == before

    def test_no_subprocess_or_llm_invoked(self, monkeypatch):
        # The headline claim is "deterministic, no LLM — works while the
        # briefing is dark on quota". Make that falsifiable: poison every
        # path to a Claude/subprocess call and assert the digest still builds.
        import subprocess

        def _boom(*a, **k):
            raise AssertionError("portfolio-signals must not shell out / call an LLM")

        monkeypatch.setattr(subprocess, "run", _boom)
        monkeypatch.setattr(subprocess, "Popen", _boom)
        import core.claude_cli as cc
        monkeypatch.setattr(cc, "claude_call", _boom)
        r = build_portfolio_signals(
            [_art("MU beats Q3"), _art("NVDA AI demand", ai_score=7.0)],
            now=NOW)
        assert r["n_tickers_with_news"] == 2


# ── route wiring (auth + window clamp), DB monkeypatched ─────────────────────
class TestEndpoint:
    def _client(self, monkeypatch, rows):
        import dashboard.web_server as ws
        monkeypatch.setattr(ws, "_ro_query", lambda sql, params=(): rows)
        return create_app(store=None).test_client()

    def test_endpoint_returns_200_and_shape(self, monkeypatch):
        rows = [("MU beats Q3 estimates", 8.0, 2, NOW.isoformat(), 1.0, "rss")]
        c = self._client(monkeypatch, rows)
        resp = c.get("/api/portfolio-signals")
        assert resp.status_code == 200
        data = resp.get_json()
        assert {t["ticker"] for t in data["tickers"]} == set(_BOOK_TICKERS)
        assert data["window_hours"] == 24
        mu = next(t for t in data["tickers"] if t["ticker"] == "MU")
        assert mu["top_headline"] == "MU beats Q3 estimates"

    def test_endpoint_hours_clamped(self, monkeypatch):
        c = self._client(monkeypatch, [])
        assert c.get("/api/portfolio-signals?hours=999").get_json()[
            "window_hours"] == 168
        assert c.get("/api/portfolio-signals?hours=0").get_json()[
            "window_hours"] == 1
        assert c.get("/api/portfolio-signals?hours=junk").get_json()[
            "window_hours"] == 24
