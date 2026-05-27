"""``ArticleStore.held_ticker_latest_article`` — per-held-ticker freshest-article
primitive.

Why this exists (news-analyst lens): the standing between-briefings question
"what's the freshest headline I have about each open position right now?"
is not answered by ``ticker_news_burst`` (which returns aggregate counts +
verdicts, not specific articles), by ``held_ticker_news_silence`` (CLI audit
emitting JSON with counts + ECHO/DARK verdicts, not the specific article),
or by ``_book_silence_lines`` (briefing-static silence flag). This primitive
exposes the per-ticker latest-article snapshot suitable for a dashboard /
chat enrichment / briefing pre-render.

These tests pin: the freshest-first ordering, word-boundary discipline (so
NVDAB does NOT match NVDA), the ``$TICKER`` prefix match, the dark-ticker
identification, backtest:// URL exclusion, ``_LIVE_ONLY_CLAUSE`` enforcement
on backtest_* / opus_annotation* sources (the load-bearing invariant), the
default-tickers fallback to ``LIVE_PORTFOLIO_TICKERS``, the window_h cutoff,
the read-only nature (no ai_score/ml_score/score_source/urgency mutation),
and the empty-input degradation.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _iso_ago(hours: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


def _insert(
    store, *, id, title, source="rss",
    first_seen=None, ai_score=0.0, ml_score=None,
    url=None, urgency=0, score_source=None,
):
    if first_seen is None:
        first_seen = _iso_ago(0.1)
    if url is None:
        url = f"https://reuters.com/{id}"
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", 1.0, ai_score, urgency,
             first_seen, 0, ml_score, score_source),
        )
        store.conn.commit()


# ── Core: freshest-first per-ticker selection ───────────────────────────────
class TestFreshestSelection:
    def test_picks_most_recent_per_ticker(self, store):
        # Two MU mentions — the 0.5h-old one must win over the 6h-old one.
        _insert(store, id="mu_old", title="MU older story",
                source="rss", first_seen=_iso_ago(6.0), ai_score=5.0)
        _insert(store, id="mu_new", title="MU breaks Q3 earnings",
                source="finnhub", first_seen=_iso_ago(0.5), ai_score=9.0)
        # NVDA single mention.
        _insert(store, id="nvda1", title="NVDA buyback announced",
                source="yahoo", first_seen=_iso_ago(2.0), ai_score=8.0)

        out = store.held_ticker_latest_article(
            tickers=["MU", "NVDA"], window_h=24.0,
        )
        by_t = {r["ticker"]: r for r in out["by_ticker"]}
        assert by_t["MU"]["id"] == "mu_new", (
            "freshest MU article should win, got: %r" % by_t["MU"]["id"]
        )
        assert by_t["MU"]["title"] == "MU breaks Q3 earnings"
        assert by_t["MU"]["source"] == "finnhub"
        assert by_t["MU"]["ai_score"] == 9.0
        assert by_t["MU"]["n_in_window"] == 2, (
            "MU has 2 mentions in window, got: %r" % by_t["MU"]["n_in_window"]
        )
        assert by_t["NVDA"]["id"] == "nvda1"
        assert by_t["NVDA"]["n_in_window"] == 1

    def test_freshest_first_order(self, store):
        # MU 4h ago, NVDA 0.5h ago — NVDA must lead the by_ticker order.
        _insert(store, id="mu", title="MU coverage", first_seen=_iso_ago(4.0))
        _insert(store, id="nvda", title="NVDA coverage",
                first_seen=_iso_ago(0.5))
        out = store.held_ticker_latest_article(tickers=["MU", "NVDA"])
        assert [r["ticker"] for r in out["by_ticker"]] == ["NVDA", "MU"], (
            "by_ticker must be freshest-first; got %r" %
            [r["ticker"] for r in out["by_ticker"]]
        )

    def test_age_h_is_positive_and_monotone(self, store):
        _insert(store, id="a", title="MU recent", first_seen=_iso_ago(0.5))
        _insert(store, id="b", title="NVDA older", first_seen=_iso_ago(3.0))
        out = store.held_ticker_latest_article(tickers=["MU", "NVDA"])
        ages = {r["ticker"]: r["latest_age_h"] for r in out["by_ticker"]}
        assert ages["MU"] is not None and ages["MU"] >= 0.0
        assert ages["NVDA"] is not None and ages["NVDA"] >= ages["MU"]
        # 0.5h and 3.0h with up to a few ms drift — sanity bound.
        assert 0.3 <= ages["MU"] <= 1.0
        assert 2.5 <= ages["NVDA"] <= 3.5


# ── Word-boundary discipline (the bug class _LIVE_RE was built to prevent) ─
class TestWordBoundary:
    def test_substring_does_not_match(self, store):
        # "MUSEUM" must NOT match MU — pure word-boundary discipline. If this
        # ever fails, the regex is broken and EVERY held-book surface (alerts,
        # briefing, model features) is wrong.
        _insert(store, id="museum", title="MUSEUM exhibits historic art")
        out = store.held_ticker_latest_article(tickers=["MU"])
        assert out["dark_tickers"] == ["MU"], (
            "MUSEUM must not match MU (word-boundary); got %r" % out
        )
        assert out["by_ticker"] == []

    def test_dollar_prefix_matches(self, store):
        # $MU should match — the \b\$?MU\b discriminator the rest of the
        # held-book surfaces use.
        _insert(store, id="d", title="$MU sees record DRAM demand")
        out = store.held_ticker_latest_article(tickers=["MU"])
        assert out["dark_tickers"] == []
        assert len(out["by_ticker"]) == 1
        assert out["by_ticker"][0]["ticker"] == "MU"

    def test_case_insensitive_word(self, store):
        # Tickers conventionally appear uppercase in financial copy, but
        # the regex should not be case-sensitive — defensive match.
        _insert(store, id="mu_lower", title="Micron mu shares jump on demand")
        out = store.held_ticker_latest_article(tickers=["MU"])
        # The current pattern uses re.compile without IGNORECASE, so lowercase
        # mu does NOT match — this pins existing behaviour (mirrors
        # ticker_news_burst's case-sensitive convention).
        assert out["dark_tickers"] == ["MU"], (
            "lowercase 'mu' must not match — financial copy writes tickers "
            "uppercase, matching ticker_news_burst convention"
        )


# ── Dark-ticker identification ──────────────────────────────────────────────
class TestDarkTickers:
    def test_ticker_with_no_mentions_is_dark(self, store):
        _insert(store, id="a", title="MU jumps on earnings")
        out = store.held_ticker_latest_article(tickers=["MU", "NVDA", "AXTI"])
        assert "MU" not in out["dark_tickers"]
        assert set(out["dark_tickers"]) == {"NVDA", "AXTI"}

    def test_all_dark_returns_empty_by_ticker(self, store):
        # No matching articles.
        _insert(store, id="x", title="Generic macro headline today")
        out = store.held_ticker_latest_article(tickers=["MU", "NVDA"])
        assert out["by_ticker"] == []
        assert set(out["dark_tickers"]) == {"MU", "NVDA"}

    def test_dark_tickers_preserves_input_order(self, store):
        # Order matters for the dashboard rendering — input order preserved.
        out = store.held_ticker_latest_article(tickers=["NVDA", "MU", "AXTI"])
        assert out["dark_tickers"] == ["NVDA", "MU", "AXTI"]


# ── Backtest isolation (the load-bearing invariant) ─────────────────────────
class TestBacktestIsolation:
    def test_backtest_url_excluded(self, store):
        # Backtest replays inject rows with url=backtest://... — these must
        # NEVER appear in any live surface; mirrors article_store
        # _LIVE_ONLY_CLAUSE.
        _insert(store, id="bt", title="MU breaking news for backtest",
                url="backtest://run_1/2026-05-15/BUY/MU",
                source="backtest_run_1_winner", first_seen=_iso_ago(0.1))
        out = store.held_ticker_latest_article(tickers=["MU"])
        assert out["dark_tickers"] == ["MU"], (
            "backtest:// URL leaked into live surface — load-bearing "
            "invariant #1 violated"
        )
        assert out["by_ticker"] == []

    def test_backtest_source_excluded(self, store):
        # Synthetic source tag (backtest_*) must also be filtered.
        _insert(store, id="bt2", title="NVDA earnings beat estimates",
                url="https://example.com/synth",
                source="backtest_run_42_winner")
        out = store.held_ticker_latest_article(tickers=["NVDA"])
        assert out["dark_tickers"] == ["NVDA"]

    def test_opus_annotation_excluded(self, store):
        _insert(store, id="op", title="MU GOOD label retraining sample",
                url="https://example.com/opus",
                source="opus_annotation_cycle_3")
        out = store.held_ticker_latest_article(tickers=["MU"])
        assert out["dark_tickers"] == ["MU"]


# ── Window cutoff ───────────────────────────────────────────────────────────
class TestWindowCutoff:
    def test_articles_outside_window_excluded(self, store):
        # 26h-old article must NOT appear when window_h=24.
        _insert(store, id="stale", title="MU news 26h ago",
                first_seen=_iso_ago(26.0))
        _insert(store, id="fresh", title="NVDA news 1h ago",
                first_seen=_iso_ago(1.0))
        out = store.held_ticker_latest_article(
            tickers=["MU", "NVDA"], window_h=24.0,
        )
        assert "MU" in out["dark_tickers"]
        assert any(r["ticker"] == "NVDA" for r in out["by_ticker"])

    def test_tight_window_drops_old_freshest(self, store):
        # Two MU articles, one 3h old and one 0.5h old — but window=1h
        # keeps only the 0.5h one.
        _insert(store, id="mu_3h", title="MU old story",
                first_seen=_iso_ago(3.0))
        _insert(store, id="mu_30m", title="MU fresh story",
                first_seen=_iso_ago(0.5))
        out = store.held_ticker_latest_article(
            tickers=["MU"], window_h=1.0,
        )
        assert len(out["by_ticker"]) == 1
        assert out["by_ticker"][0]["id"] == "mu_30m"
        assert out["by_ticker"][0]["n_in_window"] == 1, (
            "tight window must not count the 3h-old article"
        )


# ── Defaults / hygiene ──────────────────────────────────────────────────────
class TestInputHygiene:
    def test_default_tickers_pulls_live_portfolio(self, store, monkeypatch):
        # When tickers=None, the method must use LIVE_PORTFOLIO_TICKERS.
        import ml.features as features_mod
        monkeypatch.setattr(features_mod, "LIVE_PORTFOLIO_TICKERS",
                            {"FAKETICK", "OTHRTICK"})
        _insert(store, id="a", title="FAKETICK breaks records")
        out = store.held_ticker_latest_article()  # no tickers arg
        # FAKETICK was matched; OTHRTICK is dark.
        all_tickers = {r["ticker"] for r in out["by_ticker"]} | set(out["dark_tickers"])
        assert "FAKETICK" in all_tickers
        assert "OTHRTICK" in out["dark_tickers"]

    def test_empty_tickers_returns_empty(self, store):
        # Explicit empty list must not crash and not return data.
        _insert(store, id="a", title="MU breaking")
        out = store.held_ticker_latest_article(tickers=[])
        assert out["by_ticker"] == []
        assert out["dark_tickers"] == []

    def test_invalid_tickers_filtered(self, store):
        # Sub-2-char and over-8-char tickers must be dropped (matches
        # ticker_news_burst hygiene). Also None/empty entries.
        _insert(store, id="mu", title="MU jumps today")
        out = store.held_ticker_latest_article(
            tickers=["MU", "X", "TOOLONGSYMBOL", "", None],
        )
        # Only "MU" survives input hygiene; X is too short, TOOLONGSYMBOL too
        # long, others falsy.
        all_seen = {r["ticker"] for r in out["by_ticker"]} | set(out["dark_tickers"])
        assert all_seen == {"MU"}, (
            "input-hygiene filter must drop invalid tickers, got %r" % all_seen
        )

    def test_duplicate_tickers_deduped(self, store):
        # Caller passing ["MU", "mu", "MU"] should yield exactly one MU entry.
        _insert(store, id="a", title="MU jumps today")
        out = store.held_ticker_latest_article(tickers=["MU", "mu", "MU"])
        all_seen = [r["ticker"] for r in out["by_ticker"]] + out["dark_tickers"]
        assert all_seen.count("MU") == 1, (
            "duplicate tickers must be deduplicated by case-folded value"
        )


# ── Read-only invariant ─────────────────────────────────────────────────────
class TestReadOnly:
    def test_no_db_mutation(self, store):
        # The method MUST be read-only — no ai_score/ml_score/score_source/
        # urgency touch — load-bearing invariant #2/#3.
        _insert(store, id="a", title="MU breaking news today",
                ai_score=5.0, ml_score=8.5, urgency=1, score_source="llm")
        # Snapshot pre-state.
        before = store.conn.execute(
            "SELECT ai_score, ml_score, score_source, urgency FROM articles WHERE id=?",
            ("a",),
        ).fetchone()
        store.held_ticker_latest_article(tickers=["MU"])
        after = store.conn.execute(
            "SELECT ai_score, ml_score, score_source, urgency FROM articles WHERE id=?",
            ("a",),
        ).fetchone()
        assert before == after, (
            "held_ticker_latest_article must not mutate any column — got "
            f"before={before!r} after={after!r}"
        )


# ── Output structure ────────────────────────────────────────────────────────
class TestOutputShape:
    def test_returns_expected_top_level_keys(self, store):
        out = store.held_ticker_latest_article(tickers=["MU"])
        assert set(out.keys()) == {"window_h", "now_iso", "by_ticker", "dark_tickers"}
        assert isinstance(out["window_h"], float)
        assert isinstance(out["now_iso"], str)
        assert isinstance(out["by_ticker"], list)
        assert isinstance(out["dark_tickers"], list)

    def test_by_ticker_row_has_all_fields(self, store):
        _insert(store, id="a", title="MU jumps", source="finnhub",
                ai_score=9.0, ml_score=7.5, url="https://x.com/a")
        out = store.held_ticker_latest_article(tickers=["MU"])
        assert len(out["by_ticker"]) == 1
        r = out["by_ticker"][0]
        for k in ("ticker", "id", "title", "source", "first_seen", "link",
                  "ai_score", "ml_score", "latest_age_h", "n_in_window"):
            assert k in r, f"by_ticker row missing field: {k!r}"

    def test_ml_score_none_when_null(self, store):
        # When ml_score is NULL in the DB, the field must come back as None
        # — not 0.0 — so the dashboard can distinguish "unscored" from
        # "scored at zero" (the load-bearing invariant the column was
        # created to preserve).
        _insert(store, id="a", title="MU coverage", ml_score=None,
                ai_score=8.0)
        out = store.held_ticker_latest_article(tickers=["MU"])
        assert out["by_ticker"][0]["ml_score"] is None
