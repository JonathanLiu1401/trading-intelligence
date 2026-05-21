"""``ArticleStore.urgency_label_split_by_ticker`` — per-held-ticker slice of
the urgency-label calibration metric.

The aggregate ``urgency_label_split`` answers "is the alert path mostly
LLM-vetted?" (pinned at ~29% for days); ``urgency_label_split_by_source``
answers "which feeders generate the unverified noise?". This module is the
*third* natural slice: per-held-ticker — the question the analyst persona
"I depend on these alerts to react to events affecting MY positions" cares
about most. Live evidence (2026-05-21 24h): NVDA 89 urgent rows at 25%
LLM-vetted (67 ML-only) while AXTI 10 urgent rows at 60% LLM-vetted — the
analyst's biggest held name has the WORST verification rate, a per-position
view no other metric exposes.

Discriminating asserts (mirror the per-source suite where they overlap, but
each pin targets a per-TICKER concern):

  1. Per-ticker counts equal the aggregate (no rows lost / double-counted).
  2. The four canonical buckets exist on every row even when zero —
     dashboard-stable shape.
  3. Whole-word + ALL-CAPS ticker matching — ``NVDAQ`` does NOT inflate
     ``NVDA``; a leading ``$`` is allowed (``$NVDA`` matches ``NVDA``).
  4. Match surface is title+summary (mirrors ``_book_tickers`` in
     alert_agent.py — same SSOT the alert path uses).
  5. Sort: ML-DESC, alphabetical tiebreak — worst-vetted held name first.
  6. Backtest isolation: synthetic ``backtest://`` URLs and
     ``backtest_*`` / ``opus_annotation*`` sources NEVER inflate the
     per-ticker count.
  7. Non-urgent rows (urgency=0) NEVER counted.
  8. Window: an old urgent row outside ``hours`` is excluded.
  9. Held names with zero urgent mentions are OMITTED (not zero-row stubs)
     — the analyst wants signal, not boilerplate.
 10. Empty / invalid (too-short) ticker inputs degrade silently.
 11. One urgent row mentioning N held tickers counts in each — same as
     ``_book_tickers`` / ticker_mention_velocity discipline.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from storage.article_store import compress


def _recent(minutes_ago: int = 5) -> str:
    return (
        datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)
    ).isoformat()


def _insert_raw(
    store, *, id, url, title, source, urgency=1, ai_score=0.0,
    ml_score=None, score_source=None, kw_score=1.0, first_seen=None,
    summary=None,
):
    """Build any (urgency, score_source, summary) state without the live API.

    ``summary`` is zlib-compressed into ``full_text`` — the match surface
    for the new method is title+summary (the same surface ``_book_tickers``
    uses), so tests that need to exercise the summary path can pass it here.
    """
    if first_seen is None:
        first_seen = _recent()
    blob = compress(summary) if summary else None
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle, full_text, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", kw_score, ai_score, urgency,
             first_seen, 0, blob, ml_score, score_source),
        )
        store.conn.commit()


# Held set used across the suite — same shape as production's
# ml.features.LIVE_PORTFOLIO_TICKERS / daemon.PORTFOLIO_TICKERS.
HELD = ["LITE", "LNOK", "MUU", "DRAM", "SNDU", "MU", "NVDA",
        "MSFT", "AXTI", "ORCL", "TSEM", "QBTS"]


class TestShape:
    def test_empty_store_returns_empty(self, store):
        out = store.urgency_label_split_by_ticker(HELD, hours=24)
        assert out["window_h"] == 24
        assert out["by_ticker"] == []
        assert out["total_urgent"] == 0
        assert out["total_tickers"] == 0

    def test_empty_ticker_list_returns_empty(self, store):
        """No tickers in → no signal possible — degrade silently (don't
        scan the entire urgent table for nothing)."""
        _insert_raw(
            store, id="x", url="https://r.com/1",
            title="NVDA jumps 10%", source="rss",
            urgency=1, ai_score=9.0, score_source="llm",
        )
        out = store.urgency_label_split_by_ticker([], hours=24)
        assert out["by_ticker"] == []
        assert out["total_urgent"] == 0
        assert out["total_tickers"] == 0

    def test_only_invalid_tickers_returns_empty(self, store):
        """Tickers shorter than 2 chars / empty strings are skipped — if
        every input is junk, the result is empty (not an error)."""
        _insert_raw(
            store, id="x", url="https://r.com/1",
            title="NVDA jumps 10%", source="rss",
            urgency=1, ai_score=9.0, score_source="llm",
        )
        out = store.urgency_label_split_by_ticker(["", "A", None], hours=24)
        assert out["by_ticker"] == []

    def test_single_ticker_has_all_four_buckets(self, store):
        """Even a ticker with only LLM-vetted urgent rows must expose all
        four canonical buckets — dashboard-stable column set."""
        _insert_raw(
            store, id="a", url="https://r.com/1",
            title="NVDA jumps 10% on guide raise", source="rss",
            urgency=1, ai_score=9.0, score_source="llm",
        )
        out = store.urgency_label_split_by_ticker(HELD, hours=24)
        assert len(out["by_ticker"]) == 1
        row = out["by_ticker"][0]
        assert row["ticker"] == "NVDA"
        for key in ("llm", "ml", "briefing_boost", "null"):
            assert key in row, f"bucket {key} missing — shape unstable"
        assert row["llm"] == 1
        assert row["ml"] == 0
        assert row["llm_fraction"] == 1.0


class TestMatchingDiscipline:
    def test_word_boundary_prevents_substring_match(self, store):
        """``NVDAQ`` must NOT inflate ``NVDA`` — the recurring substring-leak
        bug the word-boundary regex exists to prevent (same class as the
        ``ap matched inside snap`` source-cred bug pinned in ml.features)."""
        _insert_raw(
            store, id="ok", url="https://r.com/1",
            title="NVDA earnings beat", source="rss",
            urgency=1, ai_score=9.0, score_source="llm",
        )
        _insert_raw(
            store, id="bad", url="https://r.com/2",
            title="NVDAQ fictional ticker surge", source="rss",
            urgency=1, ai_score=9.0, score_source="llm",
        )
        out = store.urgency_label_split_by_ticker(HELD, hours=24)
        nvda = {r["ticker"]: r for r in out["by_ticker"]}.get("NVDA")
        assert nvda is not None
        assert nvda["total"] == 1, (
            "NVDA matched 'NVDAQ' substring — word boundary broken"
        )

    def test_leading_dollar_sign_matches(self, store):
        """``$NVDA`` is a common urgent-headline convention; it must
        match ``NVDA`` (mirrors ml.features._LIVE_RE and
        ticker_mention_velocity)."""
        _insert_raw(
            store, id="d", url="https://r.com/1",
            title="$NVDA upgraded to Buy", source="rss",
            urgency=1, ai_score=9.0, score_source="llm",
        )
        out = store.urgency_label_split_by_ticker(HELD, hours=24)
        nvda = {r["ticker"]: r for r in out["by_ticker"]}.get("NVDA")
        assert nvda is not None
        assert nvda["total"] == 1

    def test_match_surface_includes_summary(self, store):
        """A held ticker mentioned only in the summary (not the title) must
        still count — the alert path's ``_book_tickers`` matches on
        title+summary, and this metric describes the same surface (SSOT
        with the alert path: the two surfaces never disagree about whether
        a row touches a held name)."""
        _insert_raw(
            store, id="s", url="https://r.com/1",
            title="Macro: Fed signals dovish pivot",
            source="rss",
            summary="The repricing lifted semis broadly with MU and "
                    "NVDA both rallying on the news.",
            urgency=1, ai_score=9.0, score_source="llm",
        )
        out = store.urgency_label_split_by_ticker(HELD, hours=24)
        tickers = {r["ticker"]: r for r in out["by_ticker"]}
        assert tickers.get("MU") is not None and tickers["MU"]["total"] == 1
        assert tickers.get("NVDA") is not None and tickers["NVDA"]["total"] == 1

    def test_one_row_with_multiple_held_tickers_counts_in_each(self, store):
        """A single urgent row touching N held tickers must contribute 1
        urgent row to each. Same multi-mention discipline as
        ``_book_tickers``: the row's score_source applies to every held
        name on it — anything else would mis-attribute the verification
        signal between names."""
        _insert_raw(
            store, id="m", url="https://r.com/1",
            title="MU and NVDA both up on memory-cycle tailwind",
            source="rss",
            urgency=1, ai_score=8.0, score_source="llm",
        )
        out = store.urgency_label_split_by_ticker(HELD, hours=24)
        tickers = {r["ticker"]: r for r in out["by_ticker"]}
        assert tickers["MU"]["llm"] == 1
        assert tickers["NVDA"]["llm"] == 1
        # total_urgent counts row-occurrences (i.e. ticker-mentions), so
        # the same row touching two held names contributes 2 to the
        # cross-ticker sum — same shape as alert_book_velocity. This is
        # the per-ticker analogue.
        assert out["total_urgent"] == 2
        assert out["total_tickers"] == 2


class TestPerTickerCounts:
    def test_mixed_score_sources_partition_exactly(self, store):
        """Three held tickers, mixed score_sources. The per-ticker counts
        must equal what a simple regex-and-bucket loop produces, and the
        ``llm_fraction`` definition must match the aggregate metric
        exactly (``(llm + briefing_boost) / total``)."""
        # NVDA: 3 LLM, 5 ML-only — heavily under-vetted
        for i in range(3):
            _insert_raw(
                store, id=f"n_llm{i}", url=f"https://r.com/n{i}",
                title=f"NVDA wire {i}", source="rss",
                urgency=1, ai_score=9.0, score_source="llm",
            )
        for i in range(5):
            _insert_raw(
                store, id=f"n_ml{i}", url=f"https://r.com/nm{i}",
                title=f"NVDA model-only urgent {i}", source="GN: Nvidia",
                urgency=1, ai_score=0.0, ml_score=9.0, score_source="ml",
            )
        # MU: 2 LLM-vetted, 1 briefing_boost — fully vetted
        for i in range(2):
            _insert_raw(
                store, id=f"m_llm{i}", url=f"https://r.com/m{i}",
                title=f"MU wire {i}", source="rss",
                urgency=1, ai_score=9.0, score_source="llm",
            )
        _insert_raw(
            store, id="m_bb", url="https://r.com/mb",
            title="MU opus boost", source="rss",
            urgency=1, ai_score=4.5, score_source="briefing_boost",
        )
        # AXTI: 1 ML-only, 1 legacy NULL tag
        _insert_raw(
            store, id="a_ml", url="https://r.com/a1",
            title="AXTI surge model-only", source="rss",
            urgency=1, ml_score=9.0, score_source="ml",
        )
        _insert_raw(
            store, id="a_null", url="https://r.com/a2",
            title="AXTI legacy untagged urgent row", source="rss",
            urgency=1, ai_score=7.0, score_source=None,
        )

        out = store.urgency_label_split_by_ticker(HELD, hours=24)
        # 8 NVDA + 3 MU + 2 AXTI = 13 ticker-mentions
        assert out["total_urgent"] == 13
        assert out["total_tickers"] == 3
        rows = {r["ticker"]: r for r in out["by_ticker"]}

        assert rows["NVDA"]["llm"] == 3
        assert rows["NVDA"]["ml"] == 5
        assert rows["NVDA"]["total"] == 8
        # 3 vetted / 8 total = 0.375
        assert rows["NVDA"]["llm_fraction"] == 0.375

        assert rows["MU"]["llm"] == 2
        assert rows["MU"]["briefing_boost"] == 1
        assert rows["MU"]["total"] == 3
        assert rows["MU"]["llm_fraction"] == 1.0

        assert rows["AXTI"]["ml"] == 1
        assert rows["AXTI"]["null"] == 1
        assert rows["AXTI"]["total"] == 2
        assert rows["AXTI"]["llm_fraction"] == 0.0

    def test_zero_mention_held_name_omitted(self, store):
        """A held ticker that nobody mentioned this window must NOT appear
        as a zero-row stub — the analyst wants signal, not boilerplate
        rows. (Differs from ``ticker_mention_velocity`` which DOES emit
        zero-rows because its caller iterates a per-position widget; this
        metric is consumed by "worst-vetted-first" displays where empties
        are pure clutter.)"""
        _insert_raw(
            store, id="n", url="https://r.com/1",
            title="NVDA urgent", source="rss",
            urgency=1, ai_score=9.0, score_source="llm",
        )
        out = store.urgency_label_split_by_ticker(HELD, hours=24)
        names = [r["ticker"] for r in out["by_ticker"]]
        assert names == ["NVDA"], (
            f"omission discipline broken — got {names}, expected only NVDA"
        )


class TestSortOrder:
    def test_worst_vetted_held_name_first(self, store):
        """The analyst-facing question is 'which of my positions is getting
        the WORST urgent vetting?' — sort by ml-desc so that name leads.
        Alphabetical tiebreak (same discipline as
        ``urgency_label_split_by_source``) so the order is reproducible."""
        # NVDA: 5 ml — worst offender
        for i in range(5):
            _insert_raw(
                store, id=f"n{i}", url=f"https://r.com/n{i}",
                title=f"NVDA wire {i}", source="rss",
                urgency=1, ml_score=9.0, score_source="ml",
            )
        # AXTI: 2 ml — tied with QBTS, alphabetical wins
        for i in range(2):
            _insert_raw(
                store, id=f"a{i}", url=f"https://r.com/a{i}",
                title=f"AXTI wire {i}", source="rss",
                urgency=1, ml_score=9.0, score_source="ml",
            )
        for i in range(2):
            _insert_raw(
                store, id=f"q{i}", url=f"https://r.com/q{i}",
                title=f"QBTS wire {i}", source="rss",
                urgency=1, ml_score=9.0, score_source="ml",
            )
        out = store.urgency_label_split_by_ticker(HELD, hours=24)
        tickers = [r["ticker"] for r in out["by_ticker"]]
        assert tickers == ["NVDA", "AXTI", "QBTS"], (
            f"unexpected sort: {tickers}; expected ml-desc with "
            f"alphabetical tiebreak"
        )

    def test_zero_ml_tickers_sort_alphabetically(self, store):
        """A held name 100% LLM-vetted (ml=0) must still appear — at the
        bottom — alphabetical with its peers."""
        _insert_raw(
            store, id="n", url="https://r.com/n",
            title="NVDA urgent", source="rss",
            urgency=1, ai_score=8.5, score_source="llm",
        )
        _insert_raw(
            store, id="m", url="https://r.com/m",
            title="MU urgent", source="rss",
            urgency=1, ai_score=8.5, score_source="llm",
        )
        out = store.urgency_label_split_by_ticker(HELD, hours=24)
        # Both have ml=0 — alphabetical tiebreak: MU < NVDA.
        tickers = [r["ticker"] for r in out["by_ticker"]]
        assert tickers == ["MU", "NVDA"]


class TestBacktestIsolation:
    def test_synthetic_rows_never_inflate_a_ticker(self, store):
        """A backtest:// URL or ``backtest_*`` / ``opus_annotation*``
        source must NEVER inflate the per-ticker figure — invariant #1.
        Same class as the per-source / aggregate / sentiment-trends /
        signals-vendored tests."""
        # Live row: NVDA urgent.
        _insert_raw(
            store, id="live", url="https://r.com/live",
            title="NVDA real urgent wire", source="rss",
            urgency=1, ai_score=9.0, score_source="llm",
        )
        # Three classes of synthetic poison — all carry urgency=1 and
        # would each add 1 to NVDA's bucket if isolation broke.
        _insert_raw(
            store, id="bt_url", url="backtest://run_1/2026/BUY/NVDA",
            title="synthetic NVDA backtest URL row", source="rss",
            urgency=1, ai_score=8.0, score_source="llm",
        )
        _insert_raw(
            store, id="bt_src", url="https://r.com/bt",
            title="synthetic NVDA backtest source row",
            source="backtest_run_42_winner",
            urgency=1, ai_score=8.0, score_source="llm",
        )
        _insert_raw(
            store, id="opus", url="https://r.com/opus",
            title="synthetic NVDA opus-annotation row",
            source="opus_annotation_cycle_3",
            urgency=1, ai_score=8.0, score_source="llm",
        )

        out = store.urgency_label_split_by_ticker(HELD, hours=24)
        assert out["total_urgent"] == 1, (
            "synthetic rows leaked into per-ticker metric — backtest "
            "isolation broken"
        )
        rows = {r["ticker"]: r for r in out["by_ticker"]}
        assert "NVDA" in rows
        assert rows["NVDA"]["total"] == 1


class TestUrgencyAndWindow:
    def test_non_urgent_rows_not_counted(self, store):
        """urgency=0 rows are NEVER counted — same predicate as the
        aggregate ``urgency_label_split``. The metric describes the alert
        path's calibration; a model-low-relevance call the alert path
        never saw must not pollute it."""
        _insert_raw(
            store, id="urg", url="https://r.com/1",
            title="NVDA urgent", source="rss",
            urgency=1, ai_score=9.0, score_source="llm",
        )
        _insert_raw(
            store, id="not_urg", url="https://r.com/2",
            title="NVDA boring", source="rss",
            urgency=0, ai_score=3.0, score_source="llm",
        )
        out = store.urgency_label_split_by_ticker(HELD, hours=24)
        assert out["total_urgent"] == 1
        assert out["by_ticker"][0]["total"] == 1

    def test_alerted_urgency_2_included_alongside_urgency_1(self, store):
        """urgency=2 (alerted) rows must count alongside urgency=1
        (queued) — same definition as the aggregate metric (urgency>=1).
        The analyst persona's question is "of all the urgent events the
        pipeline DECIDED on, what fraction were LLM-vetted" — that
        includes both pushed and (formatter-)suppressed-but-marked rows."""
        _insert_raw(
            store, id="alerted", url="https://r.com/1",
            title="NVDA alerted urgent", source="rss",
            urgency=2, ai_score=9.0, score_source="llm",
        )
        _insert_raw(
            store, id="queued", url="https://r.com/2",
            title="NVDA queued urgent", source="rss",
            urgency=1, ml_score=9.0, score_source="ml",
        )
        out = store.urgency_label_split_by_ticker(HELD, hours=24)
        rows = {r["ticker"]: r for r in out["by_ticker"]}
        assert rows["NVDA"]["total"] == 2
        assert rows["NVDA"]["llm"] == 1
        assert rows["NVDA"]["ml"] == 1

    def test_old_urgent_row_excluded_by_window(self, store):
        """An urgent row older than ``hours`` must not appear — same
        window semantics as the aggregate / per-source metrics."""
        old = (datetime.now(timezone.utc) - timedelta(hours=48)).isoformat()
        _insert_raw(
            store, id="old", url="https://r.com/old",
            title="NVDA stale urgent", source="rss",
            urgency=1, ai_score=8.0, score_source="llm",
            first_seen=old,
        )
        out = store.urgency_label_split_by_ticker(HELD, hours=6)
        assert out["total_urgent"] == 0
        assert out["by_ticker"] == []


class TestAggregateParity:
    def test_per_ticker_sum_lte_aggregate(self, store):
        """The aggregate ``urgency_label_split`` counts ROWS; this metric
        counts ticker-MENTIONS (a row with 2 held tickers contributes 2).
        So the per-ticker sum >= aggregate row count. The non-trivial
        invariant is the LOWER bound: every aggregate-counted row
        mentioning any held ticker must show up at least once in the
        per-ticker sum (no rows lost). This pins that anti-drift between
        the two metrics."""
        # 2 rows, 1 touching NVDA, 1 touching neither (macro)
        _insert_raw(
            store, id="held", url="https://r.com/1",
            title="NVDA beats Q1", source="rss",
            urgency=1, ai_score=9.0, score_source="llm",
        )
        _insert_raw(
            store, id="unheld", url="https://r.com/2",
            title="Generic macro news with no held ticker", source="rss",
            urgency=1, ai_score=8.0, score_source="llm",
        )
        agg = store.urgency_label_split(hours=24)
        per_t = store.urgency_label_split_by_ticker(HELD, hours=24)
        # 2 aggregate rows; 1 ticker-mention (only NVDA). Per-ticker is a
        # subset — the metric is the held-book slice of the aggregate.
        assert agg["total"] == 2
        assert per_t["total_urgent"] == 1
        assert per_t["total_urgent"] <= agg["total"], (
            "per-ticker sum exceeded aggregate row count — a held name "
            "appears to have been double-counted"
        )
