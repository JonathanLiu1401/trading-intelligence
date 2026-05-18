"""Per-publisher-domain diversity cap for the heartbeat briefing.

Evidence (live DB, 2026-05-18 snapshot): the top-50 articles feeding the 5h
Opus briefing were dominated by a single scrape channel — 10/50 slots were
``scraped/finance.yahoo.com`` price-quote widget pages (e.g. the literal
"ETH-USDEthereum USD2,169.83" string scored 9.96, the #1 slot), several of
them near-identical. A single high-volume / high-scoring publisher domain
crowding out diverse real headlines is exactly the noise the consuming
analyst complains about.

``get_top_for_briefing`` must therefore cap how many rows any one resolved
publisher domain may occupy in the digest — WITHOUT ever returning fewer
articles than the pre-cap behaviour would (a low-diversity window must still
fill the briefing via score-ordered overflow backfill), and without weakening
the backtest-isolation invariant.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone


def _recent_iso(minutes_ago: int = 5) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


def _insert_raw(store, *, id, url, title, source, urgency=0, ai_score=0.0,
                ml_score=None, score_source=None, kw_score=1.0,
                first_seen=None, published=""):
    if first_seen is None:
        first_seen = _recent_iso()
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, published, kw_score, ai_score, urgency,
             first_seen, 0, ml_score, score_source),
        )
        store.conn.commit()


class TestBriefingDomainDiversityCap:
    def test_single_scrape_domain_is_capped_when_diversity_available(self, store):
        """12 high-score rows from one host + 12 from distinct hosts. The
        noisy host must not occupy more than BRIEFING_MAX_PER_DOMAIN slots
        while the briefing still fills to ``limit`` from the diverse rows."""
        from storage.article_store import BRIEFING_MAX_PER_DOMAIN

        # 12 near-identical scrape-junk rows, all from finance.yahoo.com,
        # scored very high by the model (this is the live failure mode).
        for i in range(12):
            _insert_raw(
                store, id=f"yh{i}", url=f"https://finance.yahoo.com/q{i}",
                title=f"ETH-USDEthereum USD2,169.{i:02d} price quote widget",
                source="scraped/finance.yahoo.com",
                ai_score=0.0, ml_score=9.9, score_source="ml",
            )
        # 30 genuinely distinct publishers with solid (slightly lower) scores
        # — ample diversity (>= limit), the normal production condition where
        # the cap binds without needing the never-shrink overflow backfill.
        for i in range(30):
            _insert_raw(
                store, id=f"o{i}", url=f"https://pub{i}.com/a{i}",
                title=f"Distinct real market headline number {i} for digest",
                source=f"scraped/pub{i}.com",
                ai_score=8.0, ml_score=None, score_source="llm",
            )

        top = store.get_top_for_briefing(hours=24, limit=20)

        yahoo = [a for a in top if "finance.yahoo.com" in a["source"]]
        assert len(yahoo) <= BRIEFING_MAX_PER_DOMAIN, (
            f"one domain occupied {len(yahoo)} of 20 briefing slots "
            f"(cap={BRIEFING_MAX_PER_DOMAIN})"
        )
        # The cap must not shrink the briefing: diverse rows backfill it.
        assert len(top) == 20, f"briefing shrank to {len(top)} (expected 20)"
        # Distinct publishers should now dominate the digest.
        distinct_sources = {a["source"] for a in top}
        assert len(distinct_sources) >= 15

    def test_low_diversity_window_does_not_shrink_briefing(self, store):
        """If only ONE domain has articles, the cap must NOT reduce the
        briefing to BRIEFING_MAX_PER_DOMAIN — score-ordered overflow backfills
        it so the analyst still gets a full-size digest."""
        for i in range(15):
            _insert_raw(
                store, id=f"only{i}", url=f"https://finance.yahoo.com/n{i}",
                title=f"Only-source market wrap article number {i:02d} today",
                source="scraped/finance.yahoo.com",
                ai_score=float(9.0 - i * 0.1), score_source="llm",
            )
        top = store.get_top_for_briefing(hours=24, limit=10)
        assert len(top) == 10, (
            f"low-diversity window shrank the briefing to {len(top)}"
        )

    def test_highest_scored_rows_survive_the_cap(self, store):
        """Within a capped domain the rows that survive must be the
        highest-scored ones (the cap drops the weakest, not the strongest)."""
        from storage.article_store import BRIEFING_MAX_PER_DOMAIN

        # 8 same-domain rows with strictly descending scores 9.0 .. 2.0
        scores = [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0]
        for i, sc in enumerate(scores):
            _insert_raw(
                store, id=f"d{i}", url=f"https://finance.yahoo.com/x{i}",
                title=f"Same domain article ranked at score {sc} for test {i}",
                source="scraped/finance.yahoo.com",
                ai_score=sc, score_source="llm",
            )
        # 20 distinct lower-scored fillers so the digest fills from diverse
        # domains and the cap binds (overflow is NOT backfilled — limit met).
        for i in range(20):
            _insert_raw(
                store, id=f"f{i}", url=f"https://host{i}.com/f{i}",
                title=f"Filler distinct publisher headline number {i} here",
                source=f"scraped/host{i}.com", ai_score=1.0, score_source="llm",
            )
        top = store.get_top_for_briefing(hours=24, limit=10)
        kept = [a["ai_score"] for a in top if "finance.yahoo.com" in a["source"]]
        assert len(kept) == BRIEFING_MAX_PER_DOMAIN
        # The kept scores must be the TOP BRIEFING_MAX_PER_DOMAIN scores.
        assert sorted(kept, reverse=True) == scores[:BRIEFING_MAX_PER_DOMAIN]

    def test_backtest_rows_still_excluded_with_cap(self, store):
        """The diversity cap must not weaken backtest isolation."""
        for i in range(3):
            _insert_raw(
                store, id=f"live{i}", url=f"https://reuters.com/r{i}",
                title=f"Genuine live market headline number {i} for briefing",
                source="rss", ai_score=8.0, score_source="llm",
            )
        _insert_raw(
            store, id="bt", url="backtest://run_1/2026-01-01/BUY/MU",
            title="Synthetic backtest training row must never surface here",
            source="backtest_run_1", ai_score=9.9, score_source=None,
        )
        top = store.get_top_for_briefing(hours=24, limit=10)
        urls = [a["link"] for a in top]
        assert not any(u.startswith("backtest://") for u in urls)
        assert all(not a["source"].startswith("backtest_") for a in top)
