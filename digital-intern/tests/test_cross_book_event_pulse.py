"""Tests for ArticleStore.cross_book_event_pulse — the cross-position
"basket" event surfacer.

The novel signal this primitive exposes is "what events touched MULTIPLE
held positions simultaneously?" — every other per-ticker metric slices
by ONE name at a time. These tests pin:

  * the basket grouping is canonical (sorted-tuple key, never order-
    dependent);
  * load-bearing invariants (backtest isolation, no DB write);
  * the held-ticker hygiene (whole-word, optional ``$``, len>=2);
  * the deterministic strongest-event-first sort;
  * urgent / alerted / score_source aggregations match what the alert
    path would see for the same rows.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _recent_iso(minutes_ago: int = 5) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


def _insert(store, *, id, title, urgency=0, ai_score=0.0, ml_score=None,
            score_source=None, source="rss", url=None, summary="",
            first_seen=None):
    if url is None:
        url = f"https://x.com/{id}"
    if first_seen is None:
        first_seen = _recent_iso()
    import zlib
    blob = zlib.compress(summary.encode("utf-8")) if summary else None
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle, ml_score, score_source, full_text) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", 1.0, ai_score, urgency,
             first_seen, 0, ml_score, score_source, blob),
        )
        store.conn.commit()


class TestBasicGrouping:
    def test_single_ticker_row_is_excluded_by_default(self, store):
        """min_tickers defaults to 2 — a row mentioning only NVDA is NOT a
        cross-position event."""
        _insert(store, id="a", title="NVDA earnings beat",
                urgency=2, ai_score=9.0, score_source="llm")
        out = store.cross_book_event_pulse(["NVDA", "MU"])
        assert out["total_articles"] == 0
        assert out["by_basket"] == []

    def test_two_ticker_row_forms_one_basket(self, store):
        _insert(store, id="a", title="NVDA and MU both rise on memory cycle",
                urgency=2, ai_score=8.0, score_source="llm",
                source="reuters")
        out = store.cross_book_event_pulse(["NVDA", "MU", "MSFT"])
        assert out["total_articles"] == 1
        assert len(out["by_basket"]) == 1
        b = out["by_basket"][0]
        # Basket is sorted, NOT ticker-list order.
        assert b["basket"] == ["MU", "NVDA"]
        assert b["basket_size"] == 2
        assert b["count"] == 1
        assert b["urgent_count"] == 1
        assert b["alerted_count"] == 1
        assert b["sample_title"].startswith("NVDA and MU")
        assert b["score_sources"]["llm"] == 1

    def test_syndicated_copies_collapse_to_count(self, store):
        """5 syndicated copies of the same basket land on ONE row with
        count=5 — the recurring-coverage signal preserved without
        flooding the digest."""
        for i in range(5):
            _insert(store, id=f"c{i}", title="MU STX WDC sink on Samsung strike",
                    urgency=2 if i < 3 else 0, ai_score=8.0,
                    score_source="llm", source=f"feed_{i}")
        out = store.cross_book_event_pulse(["MU", "STX", "WDC", "NVDA"])
        assert len(out["by_basket"]) == 1
        b = out["by_basket"][0]
        assert b["basket"] == ["MU", "STX", "WDC"]
        assert b["count"] == 5
        assert b["urgent_count"] == 3
        assert b["alerted_count"] == 3


class TestBacktestIsolation:
    def test_backtest_url_excluded(self, store):
        """A backtest row with co-mentioned tickers MUST never inflate a
        basket — would manufacture a fake cross-position event every cycle
        the runner injected. CRITICAL invariant."""
        _insert(store, id="bt", title="NVDA MU both rise (backtest)",
                urgency=2, ai_score=9.0, source="backtest_run_1_winner",
                url="backtest://run_1/2026-05-21/BUY/NVDA")
        _insert(store, id="live", title="NVDA MU both rise on real news",
                urgency=2, ai_score=8.0, source="reuters",
                score_source="llm")
        out = store.cross_book_event_pulse(["NVDA", "MU"])
        assert out["total_articles"] == 1, (
            "backtest row leaked into cross-book pulse — invariant violation"
        )
        assert out["by_basket"][0]["count"] == 1

    def test_opus_annotation_excluded(self, store):
        _insert(store, id="op", title="NVDA MU opus annotation",
                urgency=2, ai_score=5.0, source="opus_annotation_cycle_5")
        out = store.cross_book_event_pulse(["NVDA", "MU"])
        assert out["total_articles"] == 0


class TestTickerHygiene:
    def test_substring_not_matched(self, store):
        """'AMD' must not match inside 'AMDOCS', 'MU' must not match inside
        'Museum'."""
        _insert(store, id="a", title="AMDOCS and Museum collaborate",
                ai_score=5.0, score_source="llm")
        out = store.cross_book_event_pulse(["AMD", "MU"])
        assert out["total_articles"] == 0

    def test_dollar_prefix_matched(self, store):
        _insert(store, id="a", title="$NVDA and $MU lead semis higher",
                ai_score=5.0, score_source="llm")
        out = store.cross_book_event_pulse(["NVDA", "MU"])
        assert out["total_articles"] == 1
        assert out["by_basket"][0]["basket"] == ["MU", "NVDA"]

    def test_summary_contributes_to_match(self, store):
        """Match surface is title + decompressed summary — same as the
        alert path's _book_tickers and urgency_label_split_by_ticker."""
        _insert(store, id="a", title="Semiconductor cycle update",
                summary="NVDA and MU both guide higher for next quarter.",
                ai_score=5.0, score_source="llm")
        out = store.cross_book_event_pulse(["NVDA", "MU"])
        assert out["total_articles"] == 1
        assert out["by_basket"][0]["basket"] == ["MU", "NVDA"]


class TestSortingAndAggregation:
    def test_strongest_event_first(self, store):
        """Sort: urgent_count desc → basket_size desc → count desc → alphabetical
        first ticker. Pin the deterministic ordering."""
        # basket (NVDA, MU): 3 articles, 2 urgent
        for i in range(3):
            _insert(store, id=f"nm{i}", title="NVDA MU semis cycle update",
                    urgency=2 if i < 2 else 0, ai_score=7.0,
                    score_source="llm")
        # basket (MU, STX, WDC): 1 article, 1 urgent — bigger basket but
        # lower urgent_count
        _insert(store, id="trip", title="MU STX WDC memory shock",
                urgency=2, ai_score=9.0, score_source="llm")
        # basket (MSFT, ORCL): 5 articles, 0 urgent — most count but no
        # urgency, sinks to bottom
        for i in range(5):
            _insert(store, id=f"mo{i}", title="MSFT ORCL cloud earnings",
                    urgency=0, ai_score=3.0, score_source="llm")
        out = store.cross_book_event_pulse(
            ["NVDA", "MU", "STX", "WDC", "MSFT", "ORCL"]
        )
        assert len(out["by_basket"]) == 3
        # urgent_count: (NVDA,MU)=2, (MU,STX,WDC)=1, (MSFT,ORCL)=0
        baskets = [tuple(b["basket"]) for b in out["by_basket"]]
        assert baskets[0] == ("MU", "NVDA")  # highest urgent_count
        assert baskets[1] == ("MU", "STX", "WDC")  # next urgent
        assert baskets[2] == ("MSFT", "ORCL")  # no urgent, last

    def test_max_score_uses_coalesce_convention(self, store):
        """max_score mirrors COALESCE(NULLIF(ai_score,0), ml_score, 0) —
        same as get_unalerted_urgent / get_top_for_briefing. A row with
        ai_score=0 and ml_score=9 must surface 9, not 0."""
        _insert(store, id="a", title="NVDA MU model-flagged",
                urgency=2, ai_score=0.0, ml_score=9.0, score_source="ml")
        out = store.cross_book_event_pulse(["NVDA", "MU"])
        assert out["by_basket"][0]["max_score"] == 9.0

    def test_score_sources_tallied(self, store):
        """Each row's score_source feeds the basket's per-tag tally so the
        analyst can see what verifies the basket (LLM vs model-only)."""
        _insert(store, id="a", title="NVDA MU LLM-verified",
                ai_score=7.0, score_source="llm")
        _insert(store, id="b", title="NVDA MU model-only",
                ai_score=0.0, ml_score=8.0, score_source="ml")
        _insert(store, id="c", title="NVDA MU model-only #2",
                ai_score=0.0, ml_score=8.5, score_source="ml")
        out = store.cross_book_event_pulse(["NVDA", "MU"])
        b = out["by_basket"][0]
        assert b["count"] == 3
        assert b["score_sources"]["llm"] == 1
        assert b["score_sources"]["ml"] == 2

    def test_newest_age_h_reflects_most_recent(self, store):
        """A basket's newest_age_h is the freshest article in it, not the
        oldest — the analyst wants to know "how recent is this event?"."""
        old = (datetime.now(timezone.utc) - timedelta(hours=10)).isoformat()
        fresh = (datetime.now(timezone.utc) - timedelta(minutes=15)).isoformat()
        _insert(store, id="old", title="NVDA MU prior coverage",
                ai_score=5.0, score_source="llm", first_seen=old)
        _insert(store, id="new", title="NVDA MU fresh wire",
                ai_score=5.0, score_source="llm", first_seen=fresh)
        out = store.cross_book_event_pulse(["NVDA", "MU"], hours=24)
        b = out["by_basket"][0]
        assert b["count"] == 2
        assert b["newest_age_h"] < 1.0  # 15 minutes — fresh wins


class TestEdgeCases:
    def test_empty_ticker_list_returns_empty(self, store):
        out = store.cross_book_event_pulse([])
        assert out == {
            "window_h": 24, "min_tickers": 2,
            "by_basket": [], "total_baskets": 0, "total_articles": 0,
        }

    def test_min_tickers_three_requires_triple(self, store):
        _insert(store, id="d", title="NVDA MU double mention",
                ai_score=5.0, score_source="llm")
        _insert(store, id="t", title="NVDA MU STX triple mention",
                ai_score=5.0, score_source="llm")
        out = store.cross_book_event_pulse(
            ["NVDA", "MU", "STX"], min_tickers=3
        )
        assert out["total_articles"] == 1
        assert out["by_basket"][0]["basket"] == ["MU", "NVDA", "STX"]

    def test_short_ticker_filtered(self, store):
        """Tickers shorter than 2 chars are skipped (no signal, over-match)."""
        _insert(store, id="a", title="NVDA and MU both rise",
                ai_score=5.0, score_source="llm")
        out = store.cross_book_event_pulse(["X", "NVDA", "MU"])
        # 'X' is skipped, so this still matches on NVDA+MU.
        assert out["total_articles"] == 1
        assert out["by_basket"][0]["basket"] == ["MU", "NVDA"]

    def test_top_n_truncates_but_total_baskets_reflects_full(self, store):
        # Build 4 distinct baskets, ask for top 2.
        baskets_to_build = [
            ("NVDA MU semis", ["NVDA", "MU"]),
            ("MSFT ORCL cloud", ["MSFT", "ORCL"]),
            ("AXTI TSEM wafers", ["AXTI", "TSEM"]),
            ("DRAM SNDU memory", ["DRAM", "SNDU"]),
        ]
        for i, (title, _) in enumerate(baskets_to_build):
            _insert(store, id=f"b{i}", title=title,
                    ai_score=5.0, score_source="llm")
        out = store.cross_book_event_pulse(
            ["NVDA", "MU", "MSFT", "ORCL", "AXTI", "TSEM", "DRAM", "SNDU"],
            top_n=2,
        )
        assert len(out["by_basket"]) == 2
        assert out["total_baskets"] == 4

    def test_outside_window_excluded(self, store):
        """An article older than ``hours`` is excluded — the pulse is a
        recency snapshot, not a lifetime tally."""
        old = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat()
        _insert(store, id="o", title="NVDA MU old wire",
                ai_score=5.0, score_source="llm", first_seen=old)
        _insert(store, id="n", title="NVDA MU fresh wire",
                ai_score=5.0, score_source="llm")
        out = store.cross_book_event_pulse(
            ["NVDA", "MU"], hours=24
        )
        assert out["total_articles"] == 1
        assert out["by_basket"][0]["count"] == 1


class TestInvariants:
    def test_read_only_no_mutation(self, store):
        """The pulse is read-only — must NEVER mutate ai_score / ml_score /
        score_source / urgency. Pin this so a future refactor doesn't
        slip a write in."""
        _insert(store, id="a", title="NVDA MU rise",
                urgency=2, ai_score=8.0, score_source="llm")
        before = store.conn.execute(
            "SELECT urgency, ai_score, ml_score, score_source FROM articles "
            "WHERE id='a'"
        ).fetchone()
        store.cross_book_event_pulse(["NVDA", "MU"])
        after = store.conn.execute(
            "SELECT urgency, ai_score, ml_score, score_source FROM articles "
            "WHERE id='a'"
        ).fetchone()
        assert before == after
