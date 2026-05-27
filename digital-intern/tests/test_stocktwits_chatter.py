"""Tests for the stocktwits forum-chatter pre-floor gate.

Pins ``_looks_like_stocktwits_chatter`` plus its three integration surfaces:
  * ``watchers.alert_agent._filter_stocktwits_chatter`` — formatter-side partition
  * ``watchers.urgency_scorer.score_batch`` — Sonnet pre-floor
  * ``storage.article_store.prefloor_pseudo_articles`` — ML pre-floor

Live evidence (2026-05-27 articles.db 24h scan): 2444 raw ``stocktwits`` rows
collected, 162 reached urgency=1 with chatter titles ("$MU lol" / "$MU yum" /
"$MU mooooooooo") at ml_score=9.95-9.97. The alert-side
``_filter_low_authority_lone`` gate (cred=0.30 < 0.45) suppressed the Discord
push, but only AFTER they reached urgency=1, inflating urgent_queue_health and
``urgent_score_distribution`` calibration metrics into BORDERLINE_HEAVY.
Pre-flooring at the ML stage exits these rows before they reach urgency=1.
"""
from __future__ import annotations

import pytest


# ── Pure helper: title-shape + source discriminator ─────────────────────────
class TestLooksLikeStocktwitsChatter:
    def _art(self, source: str, title: str) -> dict:
        return {"source": source, "title": title, "_id": "x"}

    # ── live noise corpus — MUST be caught ──────────────────────────────────
    @pytest.mark.parametrize("title", [
        "$MU lol",
        "$MU yum",
        "$MU mooooooooo",
        "$MU LETS V",
        "$MU 1k 😅",
        "$MU enjoy it while it lasts.",
        "$MU please size accordingly",
        "$MU getting ridiculous",
        "$MU Squeeze it",
        "$MU $1200 by Friday 🫡🚀🚀",
        "$MU euphoria",
        "$MU die",
        "$MU yup",
        "$MU what what any new news omg",
        "@shitstock $MU",
        "$MU 850",
        "$MU f me",
        "$MU split would be appreciated!",
    ])
    def test_live_chatter_titles_are_caught(self, title):
        from watchers.alert_agent import _looks_like_stocktwits_chatter
        assert _looks_like_stocktwits_chatter(
            self._art("stocktwits", title)
        ) is True, f"chatter NOT caught: {title!r}"

    # ── must-survive corpus — real news in stocktwits raw stream ────────────
    @pytest.mark.parametrize("title", [
        "$MU Micron price target raised to $1,175 from $675 at Barclays",
        "$MU upgraded to Buy by Goldman, PT $1100",
        "MU beats Q3 EPS estimates by 22%, guidance raised",
        "$NVDA downgraded by Morgan Stanley citing valuation concerns",
        "MU acquires Memory Inc for $5 billion",
        "AXTI dividend declared $0.15 per share quarterly",
        "LITE files 10-Q with SEC ahead of earnings call",
        "MU Q1 earnings: revenue beats $9B estimate, EPS misses",
        "Micron resigns CFO Smith effective immediately",
        "QBTS partnership with IBM announced, stock halted",
        # Long stocktwits posts that aren't keyword-bearing should still
        # survive on length alone (>= 50 chars). 50-79 char tier is mixed
        # in the live corpus — conservatively keep them.
        "MU users keep buying every dip and I think this run continues",
    ])
    def test_real_news_survives(self, title):
        from watchers.alert_agent import _looks_like_stocktwits_chatter
        assert _looks_like_stocktwits_chatter(
            self._art("stocktwits", title)
        ) is False, f"real news WRONGLY caught: {title!r}"

    def test_stocktwits_sentiment_source_NOT_gated(self):
        """The structured ``stocktwits/sentiment`` digest source carries real
        signal (extreme-sentiment bursts) and must NEVER be caught — even at
        short title length and no keyword."""
        from watchers.alert_agent import _looks_like_stocktwits_chatter
        # Even an unstructured-looking title from the sentiment digest source
        # must escape (the source IS the signal there).
        assert _looks_like_stocktwits_chatter(
            self._art("stocktwits/sentiment", "$MU bullish")
        ) is False

    def test_non_stocktwits_source_NOT_gated(self):
        """Title-shape alone must NEVER catch — gate is source-scoped. A
        reddit / nitter / rss row with a short '$MU lol' title goes through
        the existing low-authority gate, not this one."""
        from watchers.alert_agent import _looks_like_stocktwits_chatter
        for src in ("reddit/r/stocks", "nitter", "rss", "GDELT/reuters.com",
                    "yfinance/MU", "twitter"):
            assert _looks_like_stocktwits_chatter(
                self._art(src, "$MU lol")
            ) is False, f"non-stocktwits source incorrectly gated: {src}"

    def test_yfinance_stocktwits_syndication_IS_gated(self):
        """Syndicated stocktwits feeds (``yfinance/Stocktwits``,
        ``GoogleNews/Stocktwits``) carry the same raw chatter — discriminator
        is ANY source containing 'stocktwits' (case-insensitive) EXCEPT the
        ``stocktwits/sentiment`` digest."""
        from watchers.alert_agent import _looks_like_stocktwits_chatter
        assert _looks_like_stocktwits_chatter(
            self._art("yfinance/Stocktwits", "$MU lol")
        ) is True
        assert _looks_like_stocktwits_chatter(
            self._art("GoogleNews/Stocktwits", "$MU yum")
        ) is True

    def test_long_title_NOT_gated(self):
        """Length >= 50 chars escapes regardless of source — longer messages
        statistically carry signal (analyst attribution, structured prose)."""
        from watchers.alert_agent import _looks_like_stocktwits_chatter
        long_title = "$MU" + " word" * 12  # 51+ chars
        assert len(long_title) >= 50
        assert _looks_like_stocktwits_chatter(
            self._art("stocktwits", long_title)
        ) is False

    def test_empty_title_NOT_gated(self):
        """Empty / whitespace-only title is NOT chatter — let the existing
        empty-title handlers reject it elsewhere (don't double-classify)."""
        from watchers.alert_agent import _looks_like_stocktwits_chatter
        assert _looks_like_stocktwits_chatter(
            self._art("stocktwits", "")
        ) is False
        assert _looks_like_stocktwits_chatter(
            self._art("stocktwits", "   ")
        ) is False

    def test_empty_source_NOT_gated(self):
        from watchers.alert_agent import _looks_like_stocktwits_chatter
        assert _looks_like_stocktwits_chatter(
            self._art("", "$MU lol")
        ) is False
        assert _looks_like_stocktwits_chatter(
            {"title": "$MU lol", "_id": "x"}
        ) is False

    def test_case_insensitive_source_match(self):
        """Source matching is case-insensitive (sources arrive in mixed case)."""
        from watchers.alert_agent import _looks_like_stocktwits_chatter
        for src in ("Stocktwits", "STOCKTWITS", "stockTwits"):
            assert _looks_like_stocktwits_chatter(
                {"source": src, "title": "$MU lol", "_id": "x"}
            ) is True

    def test_news_keyword_at_any_position_escapes(self):
        """Keyword anywhere in the title (not just leading) is the escape
        hatch — a short '$MU earnings tmrw' must survive."""
        from watchers.alert_agent import _looks_like_stocktwits_chatter
        assert _looks_like_stocktwits_chatter(
            self._art("stocktwits", "$MU earnings tmrw")
        ) is False
        assert _looks_like_stocktwits_chatter(
            self._art("stocktwits", "MU goldman PT $1100")
        ) is False


# ── Partition helper: _filter_stocktwits_chatter ─────────────────────────────
class TestFilterStocktwitsChatter:
    def test_partition_keeps_real_and_suppresses_chatter(self):
        from watchers.alert_agent import _filter_stocktwits_chatter
        rows = [
            {"source": "stocktwits", "title": "$MU lol", "_id": "1"},
            {"source": "stocktwits", "title": (
                "$MU upgraded to Buy by Goldman PT $1100"
            ), "_id": "2"},
            {"source": "rss", "title": "$MU lol", "_id": "3"},
            {"source": "stocktwits/sentiment", "title": "$MU yum", "_id": "4"},
        ]
        kept, suppressed = _filter_stocktwits_chatter(rows)
        assert {r["_id"] for r in suppressed} == {"1"}
        assert {r["_id"] for r in kept} == {"2", "3", "4"}

    def test_empty_input(self):
        from watchers.alert_agent import _filter_stocktwits_chatter
        kept, suppressed = _filter_stocktwits_chatter([])
        assert kept == []
        assert suppressed == []

    def test_all_chatter(self):
        from watchers.alert_agent import _filter_stocktwits_chatter
        rows = [
            {"source": "stocktwits", "title": "$MU lol", "_id": "1"},
            {"source": "stocktwits", "title": "$MU yum", "_id": "2"},
        ]
        kept, suppressed = _filter_stocktwits_chatter(rows)
        assert kept == []
        assert len(suppressed) == 2

    def test_all_real(self):
        from watchers.alert_agent import _filter_stocktwits_chatter
        rows = [
            {"source": "rss", "title": "real news", "_id": "1"},
            {"source": "stocktwits", "title": (
                "MU beats Q3 earnings estimates and raises guidance"
            ), "_id": "2"},
        ]
        kept, suppressed = _filter_stocktwits_chatter(rows)
        assert len(kept) == 2
        assert suppressed == []


# ── Integration: storage.prefloor_pseudo_articles ────────────────────────────
def _insert_chatter_unscored(store, *, aid: str, title: str,
                              source: str = "stocktwits"):
    """Insert a raw, unscored row so prefloor_pseudo_articles has something
    to act on. Mirrors the existing test_article_store.py raw-insert pattern."""
    from datetime import datetime, timezone
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (aid, f"https://stocktwits.com/{aid}", title, source, "", 1.0,
             0.0, 0, datetime.now(timezone.utc).isoformat(), 0, None, None),
        )
        store.conn.commit()


class TestPrefloorIntegration:
    def test_storage_prefloor_catches_stocktwits_chatter(self, store):
        """``prefloor_pseudo_articles`` writes the chatter row to ml_score=0.01
        and removes it from the returned batch."""
        _insert_chatter_unscored(store, aid="ch1", title="$MU lol")
        _insert_chatter_unscored(store, aid="ch2", title="$MU yum")
        _insert_chatter_unscored(store, aid="real", title=(
            "MU beats Q3 earnings, guidance raised — Reuters"
        ))
        batch = store.get_unscored(min_kw=0.0)
        assert len(batch) == 3
        real, n_floored = store.prefloor_pseudo_articles(batch)
        assert n_floored == 2
        assert [a["_id"] for a in real] == ["real"]

        # Verify the score-source invariant: pre-floored rows must have
        # ml_score=0.01 / score_source='ml', and ai_score must NOT be touched
        # (model output never pollutes the LLM column).
        for aid in ("ch1", "ch2"):
            row = store.conn.execute(
                "SELECT ai_score, ml_score, score_source FROM articles "
                "WHERE id=?", (aid,),
            ).fetchone()
            ai, ml, src = row
            assert ai == 0.0, "stocktwits-chatter pre-floor must not write ai_score"
            assert ml == pytest.approx(0.01)
            assert src == "ml"

    def test_storage_prefloor_skips_stocktwits_sentiment(self, store):
        """``stocktwits/sentiment`` source carries real signal and must NOT
        be pre-floored, even at short title length."""
        _insert_chatter_unscored(
            store, aid="sent1", title="$NVDA bullish",
            source="stocktwits/sentiment",
        )
        batch = store.get_unscored(min_kw=0.0)
        real, n_floored = store.prefloor_pseudo_articles(batch)
        assert n_floored == 0
        assert [a["_id"] for a in real] == ["sent1"]


# ── Integration: urgency_scorer.score_batch ──────────────────────────────────
class TestUrgencyScorerIntegration:
    def test_score_batch_prefloors_chatter_without_sonnet(self, store,
                                                           monkeypatch):
        """``score_batch`` must pre-floor stocktwits-chatter rows BEFORE the
        Sonnet call (saves quota + keeps the LLM label pool honest). The
        Sonnet call must not even happen if every input is chatter."""
        from watchers import urgency_scorer

        # Spy on the Claude CLI — must not be invoked.
        called = {"n": 0}
        def _spy(*a, **k):
            called["n"] += 1
            return None
        monkeypatch.setattr(urgency_scorer, "claude_call", _spy)

        _insert_chatter_unscored(store, aid="ch1", title="$MU lol")
        _insert_chatter_unscored(store, aid="ch2", title="$MU yum")
        batch = store.get_unscored(min_kw=0.0)

        urgency_scorer.score_batch(batch, store)
        assert called["n"] == 0, (
            "Sonnet was called even though every input was stocktwits-chatter "
            "— pre-floor must save quota"
        )

        # Both rows now have ai_score=0.01 (urgency_scorer.score_batch uses
        # update_ai_scores_batch — tags 'llm' as documented; the storage-side
        # prefloor uses update_ml_scores_batch instead which tags 'ml').
        for aid in ("ch1", "ch2"):
            row = store.conn.execute(
                "SELECT ai_score, urgency, score_source FROM articles "
                "WHERE id=?", (aid,),
            ).fetchone()
            ai, urg, src = row
            assert ai == pytest.approx(0.01)
            assert urg == 0
            assert src == "llm"

    def test_score_batch_passes_mixed_real_only_to_sonnet(self, store,
                                                          monkeypatch):
        """Mixed batch — only the non-chatter rows reach Sonnet."""
        from watchers import urgency_scorer

        captured_payload = {"prompt": None}
        def _fake_call(prompt, **kw):
            captured_payload["prompt"] = prompt
            return '[]'
        monkeypatch.setattr(urgency_scorer, "claude_call", _fake_call)

        _insert_chatter_unscored(store, aid="ch1", title="$MU lol")
        _insert_chatter_unscored(store, aid="real", title=(
            "Federal Reserve cuts rates 50bps in emergency action"
        ), source="rss")
        batch = store.get_unscored(min_kw=0.0)

        urgency_scorer.score_batch(batch, store)

        # Sonnet was called once with the real article only.
        assert captured_payload["prompt"] is not None
        # ml prompt carries titles in the JSON payload — verify the chatter
        # is absent and the real article is present.
        assert "$MU lol" not in captured_payload["prompt"]
        assert "Federal Reserve" in captured_payload["prompt"]


# ── Lockstep: ML-path and LLM-path pre-floor agree on which rows to gate ────
def test_lockstep_ml_and_llm_path_agree_on_chatter():
    """Both pre-floor surfaces (storage.prefloor_pseudo_articles uses the ML
    column, watchers.urgency_scorer.score_batch uses the LLM column) MUST
    classify each candidate the same way — driven by the same
    ``_looks_like_stocktwits_chatter`` SSOT. If a row is chatter the storage
    path gates it BEFORE inference; if it ever slips through (e.g. comes in
    via a separate ingestion path), the urgency_scorer catches it BEFORE the
    Sonnet call. This is the defense-in-depth contract — prevents a future
    refactor from gating one path but not the other."""
    from watchers.alert_agent import _looks_like_stocktwits_chatter
    import storage.article_store as A_store
    import watchers.urgency_scorer as A_urg
    # The urgency_scorer module-level import must contain the same symbol.
    assert A_urg._looks_like_stocktwits_chatter is _looks_like_stocktwits_chatter
    # The storage prefloor lazy-imports inside the method — verify by
    # source-grep that it references the symbol (this catches the case where
    # a future edit drops the chatter line from the helper).
    import inspect
    src = inspect.getsource(A_store.ArticleStore.prefloor_pseudo_articles)
    assert "_looks_like_stocktwits_chatter" in src
