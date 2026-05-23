"""Per-held-ticker alert-pipeline coverage primitive.

The analyst's "is the alert path actually surfacing news on MY positions?"
question. The novel signal is the ``MENTIONS_ONLY`` verdict — a held name
with significant article volume in the window but ZERO urgency>=1
classifications, i.e. the scorer ran but never flagged any of its stories
as urgent. Nothing else surfaces this exact failure mode:
``urgent_queue_health`` tracks ALREADY-urgent backlog (rows that DID reach
urgency=1); ``held_ticker_news_silence`` tracks 24h DARK (zero mentions);
``urgency_label_split_by_ticker`` only sees urgent rows.
"""
from __future__ import annotations

import sqlite3
import zlib
from datetime import datetime, timedelta, timezone

import pytest


def _recent_iso(hours_ago: float = 0.05) -> str:
    return (
        datetime.now(timezone.utc) - timedelta(hours=hours_ago)
    ).isoformat()


def _insert(store, *, id, url, title, source="rss", urgency=0,
            summary="", first_seen=None):
    if first_seen is None:
        first_seen = _recent_iso()
    body = zlib.compress(summary.encode()) if summary else None
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, "
            "urgency, first_seen, cycle, ml_score, score_source, full_text) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", 1.0, 0.0, urgency,
             first_seen, 0, None, None, body),
        )
        store.conn.commit()


class TestVerdictPartition:
    def test_mentions_only_when_volume_but_no_urgent(self, store):
        """The actionable verdict: >= mentions_only_min mentions in the
        window, zero urgency>=1. Means the urgency scorer ran on these
        articles and never classified any of them urgent — either a
        calibration miss or genuinely all colour, but either way the
        analyst should look at the position."""
        for i in range(6):
            _insert(store, id=f"a{i}",
                    url=f"https://x.com/{i}",
                    title=f"NVDA quarterly outlook number {i}")

        r = store.book_alert_coverage(["NVDA"], hours=24,
                                       mentions_only_min=5)
        assert r["n_mentions_only"] == 1, r
        item = r["by_ticker"][0]
        assert item["ticker"] == "NVDA"
        assert item["mentions"] == 6
        assert item["urgent"] == 0
        assert item["alerted"] == 0
        assert item["verdict"] == "MENTIONS_ONLY"

    def test_urgent_verdict_when_any_urgency_reached(self, store):
        """A single urgency>=1 row flips the verdict away from
        MENTIONS_ONLY — the analyst HAS been served some urgent signal
        on this name, regardless of how many normal-urgency rows pile up
        alongside."""
        for i in range(6):
            _insert(store, id=f"low{i}",
                    url=f"https://x.com/low{i}",
                    title=f"AMD generic peer mention {i}")
        _insert(store, id="hot", url="https://x.com/hot",
                title="AMD earnings beat by 12%", urgency=1)

        r = store.book_alert_coverage(["AMD"], hours=24,
                                       mentions_only_min=5)
        item = next(x for x in r["by_ticker"] if x["ticker"] == "AMD")
        assert item["mentions"] == 7
        assert item["urgent"] == 1
        assert item["verdict"] == "URGENT"
        assert r["n_urgent"] == 1
        assert r["n_mentions_only"] == 0

    def test_low_volume_when_under_threshold(self, store):
        """Mentions in (0, mentions_only_min) with no urgent — the
        normal sleepy-ticker case, NOT actionable."""
        for i in range(3):
            _insert(store, id=f"q{i}", url=f"https://x.com/q{i}",
                    title=f"MSFT brief mention {i}")

        r = store.book_alert_coverage(["MSFT"], hours=24,
                                       mentions_only_min=5)
        item = next(x for x in r["by_ticker"] if x["ticker"] == "MSFT")
        assert item["mentions"] == 3
        assert item["urgent"] == 0
        assert item["verdict"] == "LOW_VOLUME"
        assert r["n_low_volume"] == 1

    def test_quiet_when_zero_mentions(self, store):
        """A ticker with zero mentions in the window — book name not
        currently in the news. Still emitted (zero-data discipline,
        callers iterate fixed-length series)."""
        _insert(store, id="other", url="https://x.com/o",
                title="generic market news no held names")

        r = store.book_alert_coverage(["NVDA"], hours=24)
        assert r["n_quiet"] == 1
        item = r["by_ticker"][0]
        assert item["ticker"] == "NVDA"
        assert item["mentions"] == 0
        assert item["latest_mention_age_h"] is None
        assert item["latest_urgent_age_h"] is None
        assert item["verdict"] == "QUIET"


class TestBacktestIsolation:
    """The most critical invariant. A synthetic backtest row that mentions
    a held ticker must NEVER inflate ``mentions`` / ``urgent`` / ``alerted``
    — otherwise an injection burst could mask a real MENTIONS_ONLY gap
    or fabricate a phantom URGENT verdict."""

    def test_backtest_url_does_not_inflate(self, store):
        # 6 real low-urgency live rows → real MENTIONS_ONLY.
        for i in range(6):
            _insert(store, id=f"live{i}",
                    url=f"https://reuters.com/{i}",
                    title=f"NVDA peer colour {i}")
        # Synthetic backtest:// row with urgency=1 — must not flip
        # MENTIONS_ONLY → URGENT, and must not inflate counts.
        _insert(store, id="bt1",
                url="backtest://run_1/2026-01-01/BUY/NVDA",
                source="backtest_run_1_winner",
                title="NVDA synthetic urgent buy entry",
                urgency=1)

        r = store.book_alert_coverage(["NVDA"], hours=24,
                                       mentions_only_min=5)
        item = next(x for x in r["by_ticker"] if x["ticker"] == "NVDA")
        assert item["mentions"] == 6, "backtest:// row leaked into mentions"
        assert item["urgent"] == 0, (
            "synthetic urgency=1 row flipped MENTIONS_ONLY → URGENT"
        )
        assert item["verdict"] == "MENTIONS_ONLY"

    def test_opus_annotation_source_does_not_inflate(self, store):
        _insert(store, id="ann1", url="https://x.com/ann1",
                source="opus_annotation_cycle_5",
                title="NVDA opus annotation GOOD label",
                urgency=2)
        r = store.book_alert_coverage(["NVDA"], hours=24)
        item = next(x for x in r["by_ticker"] if x["ticker"] == "NVDA")
        assert item["mentions"] == 0, "opus annotation row leaked"
        assert item["verdict"] == "QUIET"


class TestTickerMatching:
    """Whole-word, ALL-CAPS, optional leading $, len>=2 — byte-identical
    to urgency_label_split_by_ticker / ticker_mention_velocity /
    urgent_queue_health so the per-ticker primitives never disagree."""

    def test_whole_word_only(self, store):
        """MU must NOT match Micron / MUSEUM / MUMBAI substrings — the
        exact bug ``\\b`` boundaries exist to prevent."""
        _insert(store, id="false1", url="https://x.com/1",
                title="Micron exceeds guidance")
        _insert(store, id="false2", url="https://x.com/2",
                title="MUSEUM exhibits new tech")
        _insert(store, id="real", url="https://x.com/3",
                title="MU beats Q3 estimates")

        r = store.book_alert_coverage(["MU"], hours=24,
                                       mentions_only_min=5)
        item = next(x for x in r["by_ticker"] if x["ticker"] == "MU")
        assert item["mentions"] == 1, (
            "substring match leaked Micron/MUSEUM into MU count"
        )

    def test_dollar_prefix_matches(self, store):
        """``$NVDA`` should match ``NVDA`` — same convention the sibling
        primitives carry."""
        _insert(store, id="dollar", url="https://x.com/d",
                title="$NVDA breaks out on volume")
        r = store.book_alert_coverage(["NVDA"], hours=24)
        item = next(x for x in r["by_ticker"] if x["ticker"] == "NVDA")
        assert item["mentions"] == 1

    def test_short_ticker_skipped(self, store):
        """``len < 2`` tickers are skipped — would over-match (a single
        ``F`` would hit every prose sentence)."""
        _insert(store, id="any", url="https://x.com/a",
                title="F is a common letter")
        r = store.book_alert_coverage(["F", "NVDA"], hours=24)
        # F was skipped; only NVDA shows up.
        assert {x["ticker"] for x in r["by_ticker"]} == {"NVDA"}

    def test_match_surface_is_title_plus_summary(self, store):
        """Match surface must be title + decompressed summary — same as
        urgency_label_split_by_ticker / urgent_queue_health. A ticker in
        the body but not the title MUST count."""
        _insert(store, id="body", url="https://x.com/b",
                title="Semiconductor sector wrap",
                summary="The story centers on AXTI's indium phosphide line.")
        r = store.book_alert_coverage(["AXTI"], hours=24)
        item = next(x for x in r["by_ticker"] if x["ticker"] == "AXTI")
        assert item["mentions"] == 1, (
            "body-only ticker mention was not counted; match surface "
            "should include the decompressed summary"
        )


class TestWindowAndCounts:
    def test_window_h_excludes_older_rows(self, store):
        """Rows older than ``hours`` must be excluded. A row inside the
        window counts; a row outside doesn't."""
        old = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        for i in range(5):
            _insert(store, id=f"old{i}",
                    url=f"https://x.com/o{i}",
                    title=f"NVDA stale headline {i}",
                    first_seen=old)
        for i in range(2):
            _insert(store, id=f"new{i}",
                    url=f"https://x.com/n{i}",
                    title=f"NVDA fresh headline {i}")
        r = store.book_alert_coverage(["NVDA"], hours=24,
                                       mentions_only_min=5)
        item = next(x for x in r["by_ticker"] if x["ticker"] == "NVDA")
        assert item["mentions"] == 2
        # 2 mentions, no urgent, below the threshold → LOW_VOLUME.
        assert item["verdict"] == "LOW_VOLUME"

    def test_alerted_counts_urgency_two_only(self, store):
        """``alerted`` reports urgency=2 rows (queue-exited) — both
        urgency=1 (queued) and urgency=2 contribute to ``urgent``."""
        _insert(store, id="q", url="https://x.com/q",
                title="ORCL queued urgent",
                urgency=1)
        _insert(store, id="a", url="https://x.com/a",
                title="ORCL alerted urgent",
                urgency=2)
        r = store.book_alert_coverage(["ORCL"], hours=24)
        item = next(x for x in r["by_ticker"] if x["ticker"] == "ORCL")
        assert item["mentions"] == 2
        assert item["urgent"] == 2
        assert item["alerted"] == 1  # only urgency=2 counts

    def test_latest_mention_age_h_reflects_newest(self, store):
        """``latest_mention_age_h`` must be the age of the most-recent
        live row — used by a UI to render "last seen Xh ago"."""
        old = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
        _insert(store, id="o", url="https://x.com/o",
                title="NVDA older headline", first_seen=old)
        _insert(store, id="n", url="https://x.com/n",
                title="NVDA newer headline")
        r = store.book_alert_coverage(["NVDA"], hours=24)
        item = next(x for x in r["by_ticker"] if x["ticker"] == "NVDA")
        assert item["latest_mention_age_h"] is not None
        assert item["latest_mention_age_h"] < 1.0, (
            "latest_mention should be the freshest of the two rows"
        )


class TestSortAndShape:
    def test_worst_first_sort_order(self, store):
        """MENTIONS_ONLY (actionable) before LOW_VOLUME before URGENT
        before QUIET — analyst sees the actionable signal at the top."""
        for i in range(6):
            _insert(store, id=f"mo{i}",
                    url=f"https://x.com/mo{i}",
                    title=f"NVDA general colour {i}")
        _insert(store, id="urg", url="https://x.com/urg",
                title="MSFT earnings shock", urgency=1)
        for i in range(2):
            _insert(store, id=f"lv{i}",
                    url=f"https://x.com/lv{i}",
                    title=f"AXTI thin mention {i}")
        # QBTS has zero mentions → QUIET.

        r = store.book_alert_coverage(
            ["AXTI", "MSFT", "NVDA", "QBTS"], hours=24,
            mentions_only_min=5,
        )
        verdicts = [x["verdict"] for x in r["by_ticker"]]
        assert verdicts == ["MENTIONS_ONLY", "LOW_VOLUME",
                            "URGENT", "QUIET"]
        # Counts add up.
        assert (r["n_mentions_only"] + r["n_low_volume"]
                + r["n_urgent"] + r["n_quiet"]) == 4

    def test_empty_ticker_list_returns_empty(self, store):
        """An empty input must NOT raise — returns zero-shape envelope so
        a caller can pass an empty held set."""
        r = store.book_alert_coverage([], hours=24)
        assert r["by_ticker"] == []
        assert r["n_quiet"] == 0
        assert r["n_mentions_only"] == 0

    def test_window_h_floor_at_one(self, store):
        """``hours=0`` clamps to 1 (no zero-second window)."""
        r = store.book_alert_coverage(["NVDA"], hours=0)
        assert r["window_h"] == 1

    def test_mentions_only_min_floor_at_one(self, store):
        """``mentions_only_min=0`` clamps to 1 — a held name with even
        ONE mention can never be QUIET (zero), so MENTIONS_ONLY would
        fire on a single colour mention without the floor."""
        _insert(store, id="one", url="https://x.com/o",
                title="NVDA single mention")
        r = store.book_alert_coverage(["NVDA"], hours=24,
                                       mentions_only_min=0)
        item = r["by_ticker"][0]
        # With floor=1, 1 mention >= 1 → MENTIONS_ONLY (the floor still
        # protects against zero-second-window pathology, but a sane
        # caller passes >= 5 and gets a meaningful actionable bar).
        assert r["mentions_only_min"] == 1
        assert item["verdict"] == "MENTIONS_ONLY"


class TestReadOnlyInvariant:
    """The fourth load-bearing invariant: no DB write, no
    ai_score/ml_score/score_source/urgency mutation. A regression that
    accidentally added a write would silently corrupt the store; this
    test snapshots column state across the call and asserts equality."""

    def test_call_does_not_mutate_any_row(self, store):
        _insert(store, id="r1", url="https://x.com/1",
                title="NVDA earnings beat", urgency=1)
        _insert(store, id="r2", url="https://x.com/2",
                title="other story no held tickers")

        def _snapshot():
            return {row[0]: row[1:] for row in store.conn.execute(
                "SELECT id, ai_score, ml_score, score_source, urgency "
                "FROM articles"
            ).fetchall()}

        before = _snapshot()
        store.book_alert_coverage(["NVDA", "MU", "MSFT"], hours=24)
        after = _snapshot()
        assert after == before, (
            "book_alert_coverage mutated row state — read-only invariant "
            "broken"
        )
