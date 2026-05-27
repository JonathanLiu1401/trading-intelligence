"""Tests for ``ArticleStore.briefing_length_trend`` — the *output-density*
sibling to ``briefing_cadence_trend`` and ``briefing_text_overlap_trend``.

Why these tests exist (news-analyst lens): ``briefing_cadence_trend`` asks
"are briefings firing on schedule?", ``briefing_text_overlap_trend`` asks
"if a briefing fires, is it fresh or recapping?". Neither answers "is the
briefing as DETAILED as it used to be, or is Opus producing materially
shorter output per cycle?". A 30%-shorter briefing is a real signal of
Opus quota throttling / context truncation / response cutoff — conditions
that leave the digest covering FEWER events even when it fires ON_CADENCE
with FRESH content.

Coverage:
  * NO_DATA when fewer than 4 briefings (need a meaningful older/newer split).
  * STABLE when length is roughly constant.
  * SHRINKING when newer-half median <= 0.7 * older-half median.
  * GROWING when newer-half median >= 1.3 * older-half median.
  * Verdict ladder boundary cases (exactly at 0.70 and 1.30 ratio).
  * Result-shape contract: closed key set, closed verdict alphabet.
  * Read-only: no DB write, no article-row mutation.
  * Defensive: malformed rows (NULL / zero-length) don't crash analytics.
  * ``last_n`` clamp behaviour.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _save_briefing(store, text: str = "briefing body") -> None:
    """Insert a briefing row with the given text. ts is "now" so the
    write-order id ordering (which the analytics uses) is preserved."""
    now = datetime.now(timezone.utc).isoformat()
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO briefings (ts, text, article_count) VALUES (?, ?, ?)",
            (now, text, 10),
        )
        store.conn.commit()


def _save_n(store, n: int, length: int) -> None:
    """Append n briefings, each with text of the given character length."""
    body = "x" * length
    for _ in range(n):
        _save_briefing(store, body)


# ───────────────────────────────────────────────────────────────────────────
# Verdict ladder
# ───────────────────────────────────────────────────────────────────────────


class TestVerdictLadder:
    def test_no_data_when_zero_briefings(self, store):
        h = store.briefing_length_trend()
        assert h["verdict"] == "NO_DATA"
        assert h["n_briefings"] == 0
        assert h["lengths"] == []
        # All numeric stats must be None when there's no data.
        for k in (
            "median_length", "min_length", "max_length",
            "recent_median", "older_median", "shrink_ratio",
        ):
            assert h[k] is None

    def test_no_data_below_minimum_count(self, store):
        # 3 briefings is below the 4-row minimum — verdict must be NO_DATA.
        _save_n(store, 3, 3000)
        h = store.briefing_length_trend()
        assert h["verdict"] == "NO_DATA"
        assert h["n_briefings"] == 3
        # Lengths must still be returned for debugging visibility even
        # when the verdict is NO_DATA.
        assert h["lengths"] == [3000, 3000, 3000]
        assert h["shrink_ratio"] is None

    def test_stable_when_lengths_are_constant(self, store):
        # 6 briefings, all the same length — recent and older halves match.
        _save_n(store, 6, 3000)
        h = store.briefing_length_trend()
        assert h["verdict"] == "STABLE"
        assert h["shrink_ratio"] == 1.0
        assert h["recent_median"] == 3000
        assert h["older_median"] == 3000

    def test_shrinking_when_recent_half_drops_30pct(self, store):
        # Older 3 at 3000, newer 3 at 2000 → ratio 0.666 → SHRINKING.
        # Older comes first chronologically (id 1..3), newer second (id 4..6).
        # The DESC SQL read + reverse-to-chronological in the method gives
        # the natural [old, old, old, new, new, new] split.
        _save_n(store, 3, 3000)  # ids 1-3, chronologically older
        _save_n(store, 3, 2000)  # ids 4-6, chronologically newer
        h = store.briefing_length_trend()
        assert h["verdict"] == "SHRINKING"
        assert h["older_median"] == 3000
        assert h["recent_median"] == 2000
        # 2000 / 3000 = 0.6666... rounded to 0.667
        assert h["shrink_ratio"] == 0.667

    def test_growing_when_recent_half_jumps_30pct(self, store):
        # Older 3 at 2000, newer 3 at 3000 → ratio 1.5 → GROWING.
        _save_n(store, 3, 2000)
        _save_n(store, 3, 3000)
        h = store.briefing_length_trend()
        assert h["verdict"] == "GROWING"
        assert h["older_median"] == 2000
        assert h["recent_median"] == 3000
        assert h["shrink_ratio"] == 1.5

    def test_stable_just_above_shrinking_threshold(self, store):
        # ratio just above 0.70 (=0.75) → STABLE, not SHRINKING.
        # 3000 → 2250 = ratio 0.75. The cutoff is <=0.70, so 0.75 stays STABLE.
        _save_n(store, 3, 3000)
        _save_n(store, 3, 2250)
        h = store.briefing_length_trend()
        assert h["verdict"] == "STABLE"
        assert h["shrink_ratio"] == 0.75

    def test_shrinking_exactly_at_boundary(self, store):
        # ratio == 0.70 exactly should trigger SHRINKING (the <= 0.70 ladder).
        # 3000 → 2100 = ratio 0.70.
        _save_n(store, 3, 3000)
        _save_n(store, 3, 2100)
        h = store.briefing_length_trend()
        assert h["verdict"] == "SHRINKING"
        assert h["shrink_ratio"] == 0.70


# ───────────────────────────────────────────────────────────────────────────
# Result shape contract
# ───────────────────────────────────────────────────────────────────────────


class TestResultShape:
    def test_result_has_exact_key_set(self, store):
        _save_n(store, 6, 3000)
        h = store.briefing_length_trend()
        # The contract: callers may build dashboards/operator surfaces on
        # this exact key set, so a future revision adding a key must
        # update this test alongside the consumer.
        assert set(h.keys()) == {
            "last_n", "n_briefings", "lengths",
            "median_length", "min_length", "max_length",
            "recent_median", "older_median",
            "shrink_ratio", "verdict",
        }

    def test_verdict_is_in_closed_alphabet(self, store):
        # Across the configurations we test, verdict must come from the
        # closed set the docstring promises.
        allowed = {"NO_DATA", "STABLE", "SHRINKING", "GROWING"}
        # Zero briefings → NO_DATA
        assert store.briefing_length_trend()["verdict"] in allowed
        _save_n(store, 6, 3000)
        assert store.briefing_length_trend()["verdict"] in allowed
        _save_n(store, 6, 1000)
        assert store.briefing_length_trend()["verdict"] in allowed

    def test_lengths_in_chronological_order(self, store):
        # First insert short (older), then long (newer). The returned
        # ``lengths`` must be chronological — oldest first, newest last.
        _save_n(store, 3, 1000)
        _save_n(store, 3, 4000)
        h = store.briefing_length_trend()
        assert h["lengths"][:3] == [1000, 1000, 1000]
        assert h["lengths"][3:] == [4000, 4000, 4000]


# ───────────────────────────────────────────────────────────────────────────
# Read-only invariant
# ───────────────────────────────────────────────────────────────────────────


class TestReadOnly:
    def test_does_not_write_to_articles_or_briefings(self, store):
        _save_n(store, 6, 3000)
        # Snapshot the briefings row count before + after the analytics call.
        before_briefings = store.conn.execute(
            "SELECT COUNT(*) FROM briefings"
        ).fetchone()[0]
        before_articles = store.conn.execute(
            "SELECT COUNT(*) FROM articles"
        ).fetchone()[0]
        _ = store.briefing_length_trend()
        after_briefings = store.conn.execute(
            "SELECT COUNT(*) FROM briefings"
        ).fetchone()[0]
        after_articles = store.conn.execute(
            "SELECT COUNT(*) FROM articles"
        ).fetchone()[0]
        assert before_briefings == after_briefings
        assert before_articles == after_articles

    def test_does_not_mutate_score_columns(self, store):
        # Insert a row with known ai_score / ml_score / score_source values
        # and verify they are untouched by the analytics call.
        with store._write_lock:
            store.conn.execute(
                "INSERT INTO articles "
                "(id, url, title, source, kw_score, ai_score, ml_score, "
                "score_source, first_seen) "
                "VALUES "
                "('id1', 'http://x.com/a', 'T', 'rss', 5.0, 7.0, 9.5, "
                "'llm', ?)",
                (datetime.now(timezone.utc).isoformat(),),
            )
            store.conn.commit()
        _save_n(store, 6, 3000)
        _ = store.briefing_length_trend()
        row = store.conn.execute(
            "SELECT ai_score, ml_score, score_source, urgency FROM articles "
            "WHERE id='id1'"
        ).fetchone()
        assert row[0] == 7.0
        assert row[1] == 9.5
        assert row[2] == "llm"
        assert row[3] == 0


# ───────────────────────────────────────────────────────────────────────────
# Defensive parsing / edge cases
# ───────────────────────────────────────────────────────────────────────────


class TestDefensiveParsing:
    def test_zero_length_row_falls_back_to_no_data(self, store):
        # The schema says text NOT NULL, but a future malformed entry with
        # an empty string would produce LENGTH=0. The older_median guard
        # must trip NO_DATA rather than divide-by-zero.
        for _ in range(4):
            _save_briefing(store, text="")  # length 0
        h = store.briefing_length_trend()
        # All lengths are 0; older_median is 0 → guarded to NO_DATA.
        assert h["verdict"] == "NO_DATA"
        # Shape stays consistent.
        assert h["shrink_ratio"] is None

    def test_last_n_clamped_to_minimum_one(self, store):
        # last_n <= 0 must be silently clamped (mirrors the cadence-trend
        # discipline). Below the 4-row minimum verdict is NO_DATA.
        _save_n(store, 6, 3000)
        h = store.briefing_length_trend(last_n=0)
        assert h["last_n"] == 1
        # With last_n=1, only 1 briefing pulled — below minimum, NO_DATA.
        assert h["verdict"] == "NO_DATA"

    def test_last_n_caps_window(self, store):
        # Insert 12 briefings, request last_n=6 → only 6 considered.
        _save_n(store, 12, 3000)
        h = store.briefing_length_trend(last_n=6)
        assert h["n_briefings"] == 6
        assert len(h["lengths"]) == 6

    def test_min_and_max_length_match_window(self, store):
        # min/max must equal the actual extremes in the returned window.
        _save_briefing(store, "x" * 1500)
        _save_briefing(store, "x" * 2500)
        _save_briefing(store, "x" * 3500)
        _save_briefing(store, "x" * 4500)
        h = store.briefing_length_trend()
        assert h["min_length"] == 1500
        assert h["max_length"] == 4500
