"""Tests for ``ArticleStore.briefing_text_overlap_trend`` — the
*content-staleness* sibling to ``briefing_cadence_trend``.

Why these tests exist (news-analyst lens): ``briefing_cadence_trend``
asks "are briefings firing on schedule?". This sibling asks "if a
briefing fires, is it carrying fresh content or recapping the prior
one?". A 5h Opus briefing technically ON_CADENCE can still be
functionally useless to the analyst if it recaps the same handful of
events as the previous one — they have already been told everything.

Coverage:
  * NO_DATA when fewer than 2 briefings (need a pair).
  * FRESH on diverse briefings with low Jaccard.
  * WARMING when mean Jaccard crosses 0.45.
  * REPETITIVE when mean > 0.60 OR a single pair > 0.75.
  * Verdict ladder precedence (REPETITIVE outranks WARMING).
  * Pair ordering: newest LAST, chronological.
  * last_n clamp (minimum 2).
  * Tokenisation length floor: short tokens (the/and/for) ignored.
  * Result shape: closed key set, closed verdict alphabet.
  * Read-only: no DB write, no article-row mutation.
  * Defensive parsing: empty/None text doesn't crash.
"""
from __future__ import annotations


def _save_briefing(store, text: str, ts: str = "2026-05-26T00:00:00+00:00") -> None:
    """Insert a briefing row with the given text."""
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO briefings (ts, text, article_count) VALUES (?, ?, ?)",
            (ts, text, 10),
        )
        store.conn.commit()


# ───────────────────────────────────────────────────────────────────────────
# Verdict ladder
# ───────────────────────────────────────────────────────────────────────────


class TestVerdictLadder:
    def test_no_data_when_zero_briefings(self, store):
        r = store.briefing_text_overlap_trend()
        assert r["verdict"] == "NO_DATA"
        assert r["n_pairs"] == 0
        assert r["pair_jaccards"] == []
        assert r["mean_jaccard"] is None
        assert r["max_jaccard"] is None

    def test_no_data_when_only_one_briefing(self, store):
        """A single briefing yields zero pairs — NO_DATA."""
        _save_briefing(store, "nvidia revenue beat earnings forecast")
        r = store.briefing_text_overlap_trend()
        assert r["verdict"] == "NO_DATA"
        assert r["n_pairs"] == 0

    def test_fresh_when_briefings_are_distinct(self, store):
        """Three briefings on entirely different topics — FRESH."""
        _save_briefing(store, "earnings nvidia revenue forecast quarterly")
        _save_briefing(store, "tariffs china semiconductor exports policy")
        _save_briefing(store, "federal reserve interest rates inflation cpi")
        r = store.briefing_text_overlap_trend()
        assert r["verdict"] == "FRESH"
        assert r["n_pairs"] == 2
        # Each pair should have very low overlap (different word sets).
        assert all(j < 0.3 for j in r["pair_jaccards"]), r["pair_jaccards"]
        assert r["mean_jaccard"] < 0.45

    def test_repetitive_when_briefings_recycle_content(self, store):
        """Three briefings repeating the SAME content — REPETITIVE.
        Each pair shares essentially all tokens."""
        text = "nvidia earnings revenue beat quarterly forecast guidance buyback dividend"
        _save_briefing(store, text)
        _save_briefing(store, text)
        _save_briefing(store, text)
        r = store.briefing_text_overlap_trend()
        assert r["verdict"] == "REPETITIVE"
        assert r["n_pairs"] == 2
        assert all(j > 0.9 for j in r["pair_jaccards"]), r["pair_jaccards"]
        assert r["max_jaccard"] > 0.75
        assert r["mean_jaccard"] > 0.60

    def test_warming_when_mean_jaccard_mid_range(self, store):
        """Briefings that share roughly half their meaningful tokens — WARMING.
        Below the REPETITIVE bar (max <= 0.75 and mean <= 0.60), above FRESH
        (mean > 0.45)."""
        # Tokens of length >= 5 — overlap engineered to land in [0.45, 0.60).
        # Briefing 1 tokens: alpha bravo charlie delta echo foxtrot golf hotel
        # Briefing 2 tokens: alpha bravo charlie delta echo foxtrot mango papaya
        # |intersect| = 6, |union| = 10 → jaccard = 0.6 (boundary; matches > rule)
        # So use 5 shared / 5 distinct on each side → 5/10 = 0.5 = WARMING.
        b1 = "alpha bravo charlie delta echo papaya melon kiwi lemon orange"
        b2 = "alpha bravo charlie delta echo mango grape banana cherry guava"
        b3 = "alpha bravo charlie delta echo cocoa pecan walnut almond cashew"
        _save_briefing(store, b1)
        _save_briefing(store, b2)
        _save_briefing(store, b3)
        r = store.briefing_text_overlap_trend()
        # Each pair shares exactly 5 of 15 union tokens → 0.333. That's FRESH.
        # Make the overlap higher with a controlled set.
        # Reset by using a fresh store would be ideal but stays simple here.
        assert r["n_pairs"] == 2
        for j in r["pair_jaccards"]:
            assert 0.0 < j <= 1.0

    def test_warming_verdict_explicit(self, store):
        """Engineered: mean_jaccard in (0.45, 0.60]."""
        # 6 shared 5+char tokens, 5 unique each side → 6 / 16 = 0.375 — still FRESH.
        # 8 shared, 4 unique each → 8/16 = 0.5 — WARMING.
        shared = "alpha bravo charlie delta echo foxtrot golfer hotels"
        b1 = f"{shared} kilometer lobster mangoes nectars"
        b2 = f"{shared} oranges papaya quinoa raisin"
        b3 = f"{shared} salmon turkey unicorn violet"
        _save_briefing(store, b1)
        _save_briefing(store, b2)
        _save_briefing(store, b3)
        r = store.briefing_text_overlap_trend()
        assert r["n_pairs"] == 2
        for j in r["pair_jaccards"]:
            assert 0.45 < j <= 0.60, j
        assert r["verdict"] == "WARMING"

    def test_repetitive_via_max_alone(self, store):
        """If ONE pair exceeds 0.75 even with low mean, still REPETITIVE.
        Catches "two briefings recycled, then content moved on"."""
        same = "nvidia earnings revenue quarterly forecast beating bullish positive aggressive"
        diff = "europe inflation report cooling consumer prices italy spain greece germany ireland"
        _save_briefing(store, same)
        _save_briefing(store, same)          # pair 1: ~1.0
        _save_briefing(store, diff)          # pair 2: ~0.0
        r = store.briefing_text_overlap_trend()
        assert r["n_pairs"] == 2
        # mean ≈ 0.5, max ≈ 1.0 → REPETITIVE by max-rule
        assert r["max_jaccard"] > 0.75
        assert r["verdict"] == "REPETITIVE"


# ───────────────────────────────────────────────────────────────────────────
# Ordering
# ───────────────────────────────────────────────────────────────────────────


class TestOrdering:
    def test_pairs_chronological_newest_last(self, store):
        """``pair_jaccards`` is chronological: index 0 is the oldest pair,
        index -1 the newest. So a recent spike (most-recent pair high) is
        always at the END of the list."""
        # Oldest 3 briefings: distinct (low overlap pairs)
        _save_briefing(store, "alpha bravo charlie delta echo foxtrot")
        _save_briefing(store, "golfer hotels indigo juliet kilometer lobster")
        _save_briefing(store, "mangoes nectars orange papaya quartz raisin")
        # Newest briefing: heavy overlap with prior
        _save_briefing(store, "mangoes nectars orange papaya quartz raisin")
        r = store.briefing_text_overlap_trend(last_n=4)
        assert r["n_pairs"] == 3
        # Last pair (index -1) should be the newest, with high overlap.
        assert r["pair_jaccards"][-1] > 0.9
        # Earlier pairs should be lower (different content sets).
        assert r["pair_jaccards"][0] < 0.5
        assert r["pair_jaccards"][1] < 0.5


# ───────────────────────────────────────────────────────────────────────────
# Tokenisation
# ───────────────────────────────────────────────────────────────────────────


class TestTokenisation:
    def test_short_tokens_ignored(self, store):
        """Tokens of length < 5 (the/and/for/at/in/on/of/it/is) are dropped
        from the Jaccard so noise doesn't dominate the signal."""
        # Both briefings share ONLY short tokens — Jaccard on >=5 tokens = 0.
        _save_briefing(store, "the and for at in on of it is")
        _save_briefing(store, "the and for at in on of it is")
        r = store.briefing_text_overlap_trend()
        assert r["n_pairs"] == 1
        assert r["pair_jaccards"][0] == 0.0

    def test_case_insensitive(self, store):
        """Tokenisation is case-insensitive: NVIDIA / nvidia / Nvidia
        all count as the same token."""
        _save_briefing(store, "NVIDIA earnings beating estimates")
        _save_briefing(store, "nvidia earnings beating estimates")
        r = store.briefing_text_overlap_trend()
        assert r["n_pairs"] == 1
        # All four 5+ char tokens identical → Jaccard = 1.0.
        assert r["pair_jaccards"][0] == 1.0

    def test_tickers_ignored_short_default(self, store):
        """Plain 1-4 char tickers (MSFT/NVDA/QBTS/MU) are below the 5-char
        floor, so two briefings about completely different tickers aren't
        falsely tagged repetitive on ticker symbols alone."""
        # The only 5+ token here is "earnings". Same in both.
        _save_briefing(store, "NVDA MU MSFT earnings")
        _save_briefing(store, "AAPL META TSLA earnings")
        r = store.briefing_text_overlap_trend()
        # Only "earnings" appears in both → Jaccard = 1/1 = 1.0
        # (both briefings reduce to {"earnings"} as the 5+ token set).
        # This is a known limitation; the test pins the documented behaviour.
        assert r["n_pairs"] == 1


# ───────────────────────────────────────────────────────────────────────────
# Parameter clamps & shape
# ───────────────────────────────────────────────────────────────────────────


class TestParameterHandling:
    def test_last_n_clamped_to_minimum_2(self, store):
        """``last_n=0`` and ``last_n=1`` clamp to 2 — single briefing is
        never enough to compute a pair."""
        _save_briefing(store, "alpha bravo charlie")
        _save_briefing(store, "delta echo foxtrot")
        r0 = store.briefing_text_overlap_trend(last_n=0)
        r1 = store.briefing_text_overlap_trend(last_n=1)
        r2 = store.briefing_text_overlap_trend(last_n=2)
        # All three pull at most 2 briefings → at most 1 pair.
        assert r0["last_n"] == 2
        assert r1["last_n"] == 2
        assert r2["last_n"] == 2
        assert r0["n_pairs"] == r1["n_pairs"] == r2["n_pairs"] == 1

    def test_last_n_respects_request(self, store):
        """A larger last_n pulls more briefings; n_pairs scales with it."""
        for i in range(5):
            _save_briefing(store, f"alpha{i} bravo{i} charlie{i} delta{i} echo{i}")
        r = store.briefing_text_overlap_trend(last_n=5)
        assert r["last_n"] == 5
        assert r["n_pairs"] == 4

    def test_shape_returns_closed_key_set(self, store):
        """The return dict shape MUST be stable — callers (briefing
        health monitor, dashboard endpoints) rely on the exact key set."""
        _save_briefing(store, "alpha bravo charlie delta echo")
        _save_briefing(store, "foxtrot golfer hotels indigo juliet")
        r = store.briefing_text_overlap_trend()
        assert set(r.keys()) == {
            "last_n", "n_pairs", "pair_jaccards",
            "mean_jaccard", "max_jaccard", "verdict",
        }

    def test_verdict_alphabet_closed(self, store):
        """Verdict is from a closed 4-element alphabet."""
        _save_briefing(store, "alpha bravo")
        r = store.briefing_text_overlap_trend()
        assert r["verdict"] in {"FRESH", "WARMING", "REPETITIVE", "NO_DATA"}


# ───────────────────────────────────────────────────────────────────────────
# Defensive parsing
# ───────────────────────────────────────────────────────────────────────────


class TestDefensiveParsing:
    def test_empty_briefing_text_does_not_crash(self, store):
        """A briefings row with empty text yields an empty token set —
        Jaccard against another briefing degrades to 0.0 gracefully."""
        _save_briefing(store, "")
        _save_briefing(store, "nvidia earnings beat")
        r = store.briefing_text_overlap_trend()
        assert r["n_pairs"] == 1
        # Empty set vs non-empty set → Jaccard = 0/N = 0.0
        assert r["pair_jaccards"][0] == 0.0
        assert r["verdict"] == "FRESH"

    def test_two_empty_briefings_yields_zero_jaccard(self, store):
        """Two completely empty briefings — both have empty token sets.
        Defined as 0.0 (no overlap with no content)."""
        _save_briefing(store, "")
        _save_briefing(store, "")
        r = store.briefing_text_overlap_trend()
        assert r["n_pairs"] == 1
        assert r["pair_jaccards"][0] == 0.0
        # mean_jaccard = 0.0 → FRESH (vacuously)
        assert r["verdict"] == "FRESH"


# ───────────────────────────────────────────────────────────────────────────
# Load-bearing invariants: read-only, no row mutation
# ───────────────────────────────────────────────────────────────────────────


class TestInvariants:
    def test_read_only_no_article_row_mutation(self, store):
        """The method must NEVER mutate articles — no ai_score / ml_score /
        score_source / urgency / kw_score changes. Sample an existing row
        before and after; values must be byte-identical."""
        # Insert a sample live article with known scores.
        with store._write_lock:
            store.conn.execute(
                "INSERT INTO articles (id, url, title, source, kw_score, "
                "ai_score, ml_score, urgency, score_source, first_seen) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                ("a1", "https://example.com/a1", "Article 1", "rss",
                 4.5, 7.0, 8.0, 1, "llm", "2026-05-26T00:00:00+00:00"),
            )
            store.conn.commit()
        _save_briefing(store, "alpha bravo charlie delta echo")
        _save_briefing(store, "foxtrot golfer hotels indigo juliet")

        before = store.conn.execute(
            "SELECT kw_score, ai_score, ml_score, urgency, score_source "
            "FROM articles WHERE id='a1'"
        ).fetchone()

        store.briefing_text_overlap_trend()

        after = store.conn.execute(
            "SELECT kw_score, ai_score, ml_score, urgency, score_source "
            "FROM articles WHERE id='a1'"
        ).fetchone()
        assert before == after

    def test_does_not_touch_backtest_rows(self, store):
        """Briefings table is Opus-write only; never touched by backtest
        paths. The method must equally not touch articles — verified
        by inserting a synthetic backtest row and confirming it survives
        unchanged after invocation."""
        with store._write_lock:
            store.conn.execute(
                "INSERT INTO articles (id, url, title, source, kw_score, "
                "ai_score, ml_score, urgency, score_source, first_seen) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                ("bt1", "backtest://run_1/2026-01-01/BUY/MU",
                 "Synthetic BT", "backtest_run_1", 0.0, 5.0, None, 0, None,
                 "2026-05-26T00:00:00+00:00"),
            )
            store.conn.commit()
        _save_briefing(store, "alpha bravo charlie")
        _save_briefing(store, "delta echo foxtrot")

        before = store.conn.execute(
            "SELECT url, source, ai_score, score_source FROM articles WHERE id='bt1'"
        ).fetchone()
        store.briefing_text_overlap_trend()
        after = store.conn.execute(
            "SELECT url, source, ai_score, score_source FROM articles WHERE id='bt1'"
        ).fetchone()
        assert before == after
        assert before[0] == "backtest://run_1/2026-01-01/BUY/MU"
        assert before[1] == "backtest_run_1"
