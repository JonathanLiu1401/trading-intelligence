"""Urgency SCORE_PROMPT carries the LIVE held-book, not a frozen literal.

Regression guard for the feature that wires ``ml.features.LIVE_PORTFOLIO_TICKERS``
(config/portfolio.json's positions + watchlist) into the Sonnet urgency prompt.
Before this, the held set was hardcoded in SCORE_PROMPT, so a position added in
the trading UI was invisible to urgency scoring — its earnings beat/miss was
scored as generic sector news, never URGENT, and the analyst got no standalone
push for their own open risk.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from watchers import urgency_scorer
from ml.features import LIVE_PORTFOLIO_TICKERS


def _insert(store, *, id, title, source="rss", kw_score=1.0):
    first_seen = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (id, f"https://x.com/{id}", title, source, "", kw_score, 0.0, 0,
             first_seen, 0),
        )
        store.conn.commit()


class TestPortfolioTickerLine:
    def test_returns_sorted_nonempty_csv(self):
        line = urgency_scorer._portfolio_ticker_line()
        assert line, "held-positions slot must never be blank"
        toks = [t.strip() for t in line.split(",")]
        # Deterministic, sorted order so the prompt is test-pinnable.
        assert toks == sorted(toks)

    def test_includes_live_portfolio_tickers(self):
        line = urgency_scorer._portfolio_ticker_line()
        # Every live held/watched ticker must appear verbatim.
        for t in LIVE_PORTFOLIO_TICKERS:
            assert t in line, f"{t} missing from urgency prompt held-book line"

    def test_score_prompt_format_never_raises(self):
        # The prompt is a .format() template with escaped {{ }} in its JSON
        # example block; the new {portfolio_tickers} slot must not break it.
        out = urgency_scorer.SCORE_PROMPT.format(
            articles_json="[]",
            portfolio_tickers=urgency_scorer._portfolio_ticker_line(),
        )
        assert "HELD POSITIONS" in out
        assert "NVDA" in out  # a fallback-set member, always present


class TestScoreBatchPromptCarriesBook:
    def test_prompt_sent_to_sonnet_names_held_tickers(self, store):
        _insert(store, id="a", title="Some semiconductor supply update")
        articles = [{"_id": "a", "title": "Some semiconductor supply update",
                     "summary": ""}]
        captured = {}

        def _fake_claude(prompt, *a, **kw):
            captured["prompt"] = prompt
            return json.dumps([{"index": 0, "score": 4.0, "reason": "x"}])

        with patch.object(urgency_scorer, "claude_call", _fake_claude):
            urgency_scorer.score_batch(articles, store)

        assert "prompt" in captured, "claude_call was never invoked"
        prompt = captured["prompt"]
        # The live held book reached Sonnet — at least the always-present
        # fallback members.
        assert "MU" in prompt and "NVDA" in prompt
        for t in LIVE_PORTFOLIO_TICKERS:
            assert t in prompt, f"held ticker {t} not in urgency prompt"

    def test_score_batch_still_scores_correctly(self, store):
        # The prompt change must not alter scoring behaviour.
        _insert(store, id="b", title="MU earnings crush expectations")
        articles = [{"_id": "b", "title": "MU earnings crush expectations",
                     "summary": ""}]
        with patch.object(urgency_scorer, "claude_call",
                          return_value=json.dumps(
                              [{"index": 0, "score": 9.0, "reason": "beat"}])):
            n = urgency_scorer.score_batch(articles, store)
        assert n == 1
        row = store.conn.execute(
            "SELECT ai_score, urgency, score_source FROM articles WHERE id='b'"
        ).fetchone()
        assert row[0] == 9.0
        assert row[1] == 1
        assert row[2] == "llm"
