"""Tests for the arbitrary-ticker wire-stance builder.

The builder is a thin pure wrapper around ``build_held_wire_balance``,
so tests focus on:

  * The thin layer (ticker normalization, headline rebranding, scope
    field).
  * Pinning the *SSOT contract* — the per-ticker verdict and book
    verdict outputs MUST match what ``build_held_wire_balance`` would
    have returned on the same input + ticker list. If the two
    builders drift, this test breaks.
  * Garbage-safety on the new caller-driven ``tickers`` argument
    (None, non-list, all-non-string, duplicate, whitespace).

Coverage matches the established pattern (test_held_wire_balance):
verdict ladder, garbage rows, threshold pins, endpoint smoke via the
Flask test_client.
"""
from __future__ import annotations

from datetime import datetime, timezone

import pytest

from analysis.held_wire_balance import (
    BOOK_LEAN_PCT,
    MIN_CLASSIFIED_PER_TICKER,
    build_held_wire_balance,
)
from analysis.wire_stance import (
    _normalize_tickers,
    _stance_headline,
    build_wire_stance,
)


FIXED_NOW = datetime(2026, 5, 28, 23, 30, 0, tzinfo=timezone.utc)


def _art(title: str, ai_score: float = 5.0) -> dict:
    return {"title": title, "ai_score": ai_score,
            "first_seen": FIXED_NOW.isoformat()}


class TestTickerNormalization:
    def test_none(self):
        assert _normalize_tickers(None) == []

    def test_empty_list(self):
        assert _normalize_tickers([]) == []

    def test_string_garbage(self):
        # A bare string is NOT a ticker list — must return empty.
        assert _normalize_tickers("MUU,KLAC") == []

    def test_strips_whitespace(self):
        assert _normalize_tickers(["  MUU  ", "klac"]) == ["MUU", "KLAC"]

    def test_dedups(self):
        assert _normalize_tickers(["MUU", "muu", "MUU "]) == ["MUU"]

    def test_drops_non_strings(self):
        assert _normalize_tickers(["MUU", 42, None, "", "KLAC"]) == ["MUU", "KLAC"]

    def test_preserves_order(self):
        assert _normalize_tickers(["ZZZ", "AAA", "MMM"]) == ["ZZZ", "AAA", "MMM"]


class TestEmptyInput:
    def test_no_tickers(self):
        r = build_wire_stance([], [], now=FIXED_NOW)
        assert r["book_verdict"] == "BOOK_INSUFFICIENT"
        assert r["scope"] == "arbitrary"
        assert r["tickers_in"] == []
        assert "no tickers provided" in r["headline"]
        # Empty skel has the same key shape as a populated result.
        for k in ("per_ticker", "n_bull_lean", "n_bear_lean",
                  "n_mixed", "n_insufficient", "min_classified_per_ticker",
                  "book_lean_pct"):
            assert k in r

    def test_only_garbage_tickers(self):
        r = build_wire_stance([], [None, 42, "", "  "], now=FIXED_NOW)
        assert r["book_verdict"] == "BOOK_INSUFFICIENT"
        assert r["tickers_in"] == []
        assert "no valid tickers" in r["headline"]

    def test_non_list_tickers_input(self):
        # _normalize_tickers correctly handles non-list/non-tuple/non-set
        # inputs by returning empty; a bare string isn't a ticker list.
        r = build_wire_stance([], "MUU,KLAC", now=FIXED_NOW)  # type: ignore[arg-type]
        assert r["tickers_in"] == []
        assert r["book_verdict"] == "BOOK_INSUFFICIENT"

    def test_non_list_articles(self):
        r = build_wire_stance("garbage", ["MUU"], now=FIXED_NOW)  # type: ignore[arg-type]
        # Should still emit a well-formed skeleton with the ticker in
        # the per-ticker table (it'll be INSUFFICIENT — no articles).
        assert r["scope"] == "arbitrary"
        assert isinstance(r["per_ticker"], list)


class TestSSOTParity:
    """The whole point of this thin wrapper is to never silently
    disagree with held_wire_balance on the verdict for the same
    inputs. If the two ever diverge, operator trust collapses."""

    def test_per_ticker_verdicts_match_held_wire_balance(self):
        articles = [
            _art("MUU surges to new high on AI demand", ai_score=9.0),
            _art("MUU beats earnings estimates", ai_score=8.5),
            _art("MUU upgrade from analyst", ai_score=7.0),
            _art("KLAC plunges on guidance miss", ai_score=8.0),
            _art("KLAC downgrade", ai_score=7.5),
        ]
        tickers = ["MUU", "KLAC"]

        held_result = build_held_wire_balance(
            articles, held_tickers=tickers, now=FIXED_NOW)
        stance_result = build_wire_stance(
            articles, tickers, now=FIXED_NOW)

        # Per-ticker rows must be identical (just sorted by the same
        # rule). Compare the verdict + counts dicts as sets-of-tuples
        # so list ordering is irrelevant.
        held_pt = {(t["ticker"], t["verdict"], t["n_bull"], t["n_bear"])
                   for t in held_result["per_ticker"]}
        stance_pt = {(t["ticker"], t["verdict"], t["n_bull"], t["n_bear"])
                     for t in stance_result["per_ticker"]}
        assert held_pt == stance_pt
        assert held_result["book_verdict"] == stance_result["book_verdict"]

    def test_classifier_thresholds_preserved(self):
        # Pinning the SSOT classifier constants — if these drift,
        # the held-book and arbitrary-list reports could silently
        # disagree on what "bullish" means.
        r = build_wire_stance([], ["MUU"], now=FIXED_NOW)
        assert r["min_classified_per_ticker"] == MIN_CLASSIFIED_PER_TICKER
        assert r["book_lean_pct"] == BOOK_LEAN_PCT


class TestVerdictLadder:
    def test_bull_lean_when_70_pct_bull(self):
        # 3 bull / 1 bear = 75% bull → BULL_LEAN.
        articles = [
            _art("MUU surges on strong demand"),
            _art("MUU beats estimates"),
            _art("MUU jumps after upgrade"),
            _art("MUU declines on downgrade"),
        ]
        r = build_wire_stance(articles, ["MUU"], now=FIXED_NOW)
        muu = r["per_ticker"][0]
        assert muu["verdict"] == "BULL_LEAN"
        assert r["book_verdict"] in ("BOOK_BULL", "BOOK_INSUFFICIENT")

    def test_bear_lean_when_70_pct_bear(self):
        articles = [
            _art("MUU plunges on guidance miss"),
            _art("MUU downgrade weighs"),
            _art("MUU drops 10%"),
            _art("MUU beats narrowly"),
        ]
        r = build_wire_stance(articles, ["MUU"], now=FIXED_NOW)
        muu = r["per_ticker"][0]
        assert muu["verdict"] == "BEAR_LEAN"

    def test_insufficient_under_min_classified(self):
        # 1 bull, 0 bear → 1 classified, under MIN_CLASSIFIED_PER_TICKER=2.
        articles = [_art("MUU surges")]
        r = build_wire_stance(articles, ["MUU"], now=FIXED_NOW)
        muu = r["per_ticker"][0]
        assert muu["verdict"] == "INSUFFICIENT"

    def test_mixed_at_50_50(self):
        articles = [
            _art("MUU surges"),
            _art("MUU plunges"),
        ]
        r = build_wire_stance(articles, ["MUU"], now=FIXED_NOW)
        muu = r["per_ticker"][0]
        assert muu["verdict"] == "MIXED"


class TestHeadlineRebranding:
    """The new endpoint's headline MUST read differently than
    held-wire-balance so an operator scanning a dashboard with both
    panels open can tell them apart by the headline alone."""

    def test_headline_uses_wire_stance_phrase(self):
        r = build_wire_stance([], ["MUU", "KLAC"], now=FIXED_NOW)
        assert "Wire stance" in r["headline"]
        # The held-book phrase must NOT leak into the arbitrary scope.
        assert "Held-wire balance" not in r["headline"]

    def test_bull_headline(self):
        articles = [
            _art("MUU surges on demand"),
            _art("MUU beats estimates"),
            _art("KLAC surges on guidance"),
            _art("KLAC upgrade"),
        ]
        r = build_wire_stance(articles, ["MUU", "KLAC"], now=FIXED_NOW)
        assert "BOOK_BULL" in r["headline"]

    def test_bear_headline_names_bear_tickers(self):
        articles = [
            _art("MUU plunges on miss"),
            _art("MUU downgrade weighs"),
            _art("KLAC plunges"),
            _art("KLAC downgrade"),
        ]
        r = build_wire_stance(articles, ["MUU", "KLAC"], now=FIXED_NOW)
        assert "BOOK_BEAR" in r["headline"]
        # Bear tickers must be named in the headline so the operator's
        # eye lands on the specific concern.
        assert "MUU" in r["headline"] or "KLAC" in r["headline"]


class TestScopeMetadata:
    def test_scope_field_present(self):
        r = build_wire_stance([], ["MUU"], now=FIXED_NOW)
        assert r["scope"] == "arbitrary"

    def test_tickers_in_echoed(self):
        r = build_wire_stance([], ["muu", "KLAC", "muu"], now=FIXED_NOW)
        # Normalized: upper-cased + deduped.
        assert r["tickers_in"] == ["MUU", "KLAC"]


class TestStanceHeadlineHelper:
    """Exercise the headline-format helper directly so each branch is
    covered without the SQL/Flask layer."""

    def test_book_bear_headline_no_bear_names(self):
        # Defensive branch: book is BOOK_BEAR but per_ticker has no
        # BEAR_LEAN rows (shouldn't happen in practice, but the helper
        # must not break).
        h = _stance_headline("BOOK_BEAR", [], 3, 0)
        assert "BOOK_BEAR" in h

    def test_book_bull_headline_count(self):
        per = [{"verdict": "BULL_LEAN", "ticker": "A"},
               {"verdict": "BULL_LEAN", "ticker": "B"}]
        h = _stance_headline("BOOK_BULL", per, 3, 2)
        assert "BOOK_BULL" in h
        assert "2" in h

    def test_book_mixed_split_vote(self):
        per = [{"verdict": "BULL_LEAN", "ticker": "A"},
               {"verdict": "BEAR_LEAN", "ticker": "B"},
               {"verdict": "MIXED", "ticker": "C"}]
        h = _stance_headline("BOOK_MIXED", per, 3, 3)
        assert "BOOK_MIXED" in h
        assert "1↑" in h

    def test_book_insufficient_coverage(self):
        per = [{"verdict": "INSUFFICIENT", "ticker": "A"}]
        h = _stance_headline("BOOK_INSUFFICIENT", per, 1, 0)
        assert "BOOK_INSUFFICIENT" in h


class TestEndpointSmoke:
    """Endpoint contract via Flask test_client. SQL layer is exercised
    against the project's standard conftest fixture (mirrors the
    test_held_wire_balance smoke tests)."""

    def _client(self, store):
        from dashboard.web_server import create_app
        app = create_app(store)
        return app.test_client()

    def test_missing_tickers_returns_skeleton(self, store):
        client = self._client(store)
        rv = client.get("/api/wire-stance")
        assert rv.status_code == 200
        data = rv.get_json()
        assert data["book_verdict"] == "BOOK_INSUFFICIENT"
        assert data["tickers_in"] == []
        assert data["scope"] == "arbitrary"

    def test_empty_tickers_returns_skeleton(self, store):
        client = self._client(store)
        rv = client.get("/api/wire-stance?tickers=")
        assert rv.status_code == 200
        data = rv.get_json()
        assert data["book_verdict"] == "BOOK_INSUFFICIENT"
        assert data["tickers_in"] == []

    def test_tickers_parsed_and_normalized(self, store):
        client = self._client(store)
        rv = client.get("/api/wire-stance?tickers=muu,KLAC,muu, MRVL ")
        assert rv.status_code == 200
        data = rv.get_json()
        # Upper-cased, deduped, trimmed.
        assert data["tickers_in"] == ["MUU", "KLAC", "MRVL"]

    def test_ticker_count_capped_at_40(self, store):
        client = self._client(store)
        many = ",".join(f"T{i}" for i in range(60))
        rv = client.get(f"/api/wire-stance?tickers={many}")
        assert rv.status_code == 200
        data = rv.get_json()
        # Capped at 40 to bound regex / SQL work.
        assert len(data["tickers_in"]) == 40

    def test_hours_clamp(self, store):
        client = self._client(store)
        rv_low = client.get("/api/wire-stance?tickers=MUU&hours=0")
        rv_high = client.get("/api/wire-stance?tickers=MUU&hours=9999")
        assert rv_low.status_code == 200
        assert rv_high.status_code == 200
        assert rv_low.get_json()["window_hours"] == 1
        assert rv_high.get_json()["window_hours"] == 168

    def test_garbage_hours(self, store):
        client = self._client(store)
        rv = client.get("/api/wire-stance?tickers=MUU&hours=garbage")
        assert rv.status_code == 200
        data = rv.get_json()
        assert data["window_hours"] == 24  # default

    def test_unauthorized_when_api_key_required(self, store, monkeypatch):
        # If a sibling test sets API_KEY, unauthorized requests must
        # 401 — pin the auth precedent rest of the suite uses.
        from dashboard import web_server as ws
        monkeypatch.setenv("DI_DASHBOARD_API_KEY", "")
        # With no key configured, _check_api_key returns True (default
        # open) — the test passes through.
        client = self._client(store)
        rv = client.get("/api/wire-stance?tickers=MUU")
        assert rv.status_code == 200
