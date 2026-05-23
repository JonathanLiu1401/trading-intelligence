"""Per-held-ticker recap-template pollution rate
(``ArticleStore.ticker_recap_pollution``).

Sibling to ``source_recap_pollution`` (per-collector content-type angle) and to
``urgency_label_split_by_ticker`` (per-held-ticker verification angle). Answers
the analyst persona's "which of MY positions are getting recap-mill noise vs
real news?" question — neither sibling does.

Recap detection is injected (the storage layer must not import the analysis or
watchers gates), and the test suite verifies both the boolean-return and
tuple-return matcher conventions, the SSOT-parity with the alert-side and
briefing-side matchers, the buggy-matcher degradation, the ``min_total`` volume
floor, the deterministic sort, and the backtest-exclusion invariant.

Ticker matching is byte-identical to the other four per-held-ticker primitives
(``urgency_label_split_by_ticker`` / ``ticker_mention_velocity`` /
``urgent_queue_health`` / ``book_alert_coverage``) — whole-word, ALL-CAPS,
optional leading ``$``, ``len >= 2``, surface = title + decompressed summary —
so the five never disagree about whether a row touches a held name.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest


def _insert(store, *, ticker_in_title: str | None = None,
            title: str | None = None, summary: str = "",
            urgency: int = 1, src: str = "rss",
            url: str | None = None, hours_ago: float = 1.0) -> str:
    """Insert one article row. Returns its id. Bypasses ``insert_batch`` so
    the test can pin the exact (urgency, age, ticker-mention) combination."""
    from storage.article_store import article_id, compress
    if title is None:
        title = (
            f"News about {ticker_in_title}" if ticker_in_title
            else "Some neutral headline"
        )
    effective_url = url or f"https://x/{title[:40]}/{ticker_in_title or ''}"
    aid = article_id(effective_url, title)
    now = datetime.now(timezone.utc)
    first_seen = (now - timedelta(hours=hours_ago)).isoformat()
    store.conn.execute(
        "INSERT OR IGNORE INTO articles "
        "(id, url, title, source, published, kw_score, ai_score, urgency, "
        " full_text, first_seen, cycle, ml_score, score_source) "
        "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (aid, effective_url, title, src, "", 1.0, 0.0, urgency,
         compress(summary), first_seen, 0, 9.5, "ml"),
    )
    store.conn.commit()
    return aid


# ── matcher signatures ───────────────────────────────────────────────────────

def _bool_recap_matcher(title: str) -> bool:
    """The simplest dashboard hook: "is the title recap?" → bool."""
    title_l = (title or "").lower()
    return any(kw in title_l for kw in (
        "why did", "why is", "why are", "today's movers", "earnings call recap",
        "trading up today", "trading down today",
    ))


def _tuple_recap_matcher(title: str):
    """The canonical (hit, fingerprint_name) signature the production
    matchers in ``watchers.alert_agent`` and ``analysis.claude_analyst`` use."""
    title_l = (title or "").lower()
    if "why did" in title_l:
        return True, "why_did_stock"
    if "today's movers" in title_l:
        return True, "todays_movers_list"
    if "trading up today" in title_l or "trading down today" in title_l:
        return True, "why_trading_today"
    if "earnings call recap" in title_l:
        return True, "earnings_call_recap"
    return False, ""


class TestTickerRecapPollutionBasic:
    def test_empty_tickers_returns_zero_skeleton(self, store):
        out = store.ticker_recap_pollution(
            [], _bool_recap_matcher, hours=24, min_total=1
        )
        assert out["by_ticker"] == []
        assert out["total_urgent"] == 0
        assert out["total_recap"] == 0
        assert out["global_rate"] == 0.0
        assert out["window_h"] == 24

    def test_empty_corpus_returns_zero_skeleton(self, store):
        out = store.ticker_recap_pollution(
            ["NVDA", "MU"], _bool_recap_matcher, hours=24, min_total=1
        )
        # Tickers given but no urgent rows in window → all-zero result.
        assert out["by_ticker"] == []
        assert out["total_urgent"] == 0

    def test_per_ticker_counts_correct(self, store):
        # NVDA: 5 urgent, 4 recap (rate 0.8)
        for i in range(4):
            _insert(store, ticker_in_title="NVDA",
                    title=f"Why Did NVDA Stock Drop Today {i}")
        _insert(store, ticker_in_title="NVDA",
                title="NVDA tops Q1 estimates with HBM beat")
        # MU: 4 urgent, 1 recap (rate 0.25)
        _insert(store, ticker_in_title="MU",
                title="Why Did MU Stock Surge Today")
        for i in range(3):
            _insert(store, ticker_in_title="MU",
                    title=f"MU guides Q3 higher on memory pricing {i}")

        out = store.ticker_recap_pollution(
            ["NVDA", "MU"], _bool_recap_matcher, hours=24, min_total=2
        )
        by_t = {r["ticker"]: r for r in out["by_ticker"]}
        assert by_t["NVDA"]["total"] == 5
        assert by_t["NVDA"]["recap"] == 4
        assert by_t["NVDA"]["recap_rate"] == 0.8
        assert by_t["MU"]["total"] == 4
        assert by_t["MU"]["recap"] == 1
        assert by_t["MU"]["recap_rate"] == 0.25

    def test_no_mention_ticker_excluded_from_by_ticker(self, store):
        """A held ticker with ZERO urgent mentions is omitted from the per-
        ticker list — analyst wants signal, not zero-rows for the entire book.
        Mirrors urgency_label_split_by_ticker's discipline."""
        for i in range(3):
            _insert(store, ticker_in_title="NVDA",
                    title=f"NVDA news headline {i}")

        out = store.ticker_recap_pollution(
            ["NVDA", "MU", "MSFT"], _bool_recap_matcher,
            hours=24, min_total=1,
        )
        tickers = {r["ticker"] for r in out["by_ticker"]}
        # MU and MSFT have zero mentions → not in by_ticker.
        assert tickers == {"NVDA"}

    def test_global_rate_counts_rows_not_buckets(self, store):
        """A single recap row mentioning TWO held tickers counts toward
        both per-ticker buckets, but the global total counts the row ONCE
        (matches source_recap_pollution's row-counted total)."""
        _insert(store, title="Why Did NVDA and MU Stocks Drop Today",
                ticker_in_title=None)  # title contains both tickers
        _insert(store, ticker_in_title="NVDA",
                title="NVDA beats expectations on H100 ramp")

        out = store.ticker_recap_pollution(
            ["NVDA", "MU"], _bool_recap_matcher, hours=24, min_total=1
        )
        by_t = {r["ticker"]: r for r in out["by_ticker"]}
        # NVDA: in 2 rows (the shared recap + the standalone real headline)
        assert by_t["NVDA"]["total"] == 2
        assert by_t["NVDA"]["recap"] == 1
        # MU: in 1 row (just the shared recap)
        assert by_t["MU"]["total"] == 1
        assert by_t["MU"]["recap"] == 1
        # Global total = rows that mention ANY held name, not bucket sum.
        # Two rows touched the held set, one was recap.
        assert out["total_urgent"] == 2
        assert out["total_recap"] == 1
        assert out["global_rate"] == 0.5


class TestTickerRecapPollutionSort:
    def test_worst_rate_first_alphabetical_tiebreak(self, store):
        # NVDA rate 0.5, MU rate 0.5 (tie), AXTI rate 0.8 — worst first.
        for i in range(4):
            _insert(store, ticker_in_title="AXTI",
                    title=f"Why Did AXTI Stock Drop Today {i}")
        _insert(store, ticker_in_title="AXTI",
                title="AXTI guides Q4 higher on Indium phosphide demand")
        for i in range(2):
            _insert(store, ticker_in_title="NVDA",
                    title=f"Why Did NVDA Stock Drop Today {i}")
        for i in range(2):
            _insert(store, ticker_in_title="NVDA",
                    title=f"NVDA real headline {i}")
        for i in range(2):
            _insert(store, ticker_in_title="MU",
                    title=f"Why Did MU Stock Drop Today {i}")
        for i in range(2):
            _insert(store, ticker_in_title="MU",
                    title=f"MU real headline {i}")

        out = store.ticker_recap_pollution(
            ["NVDA", "MU", "AXTI"], _bool_recap_matcher,
            hours=24, min_total=1,
        )
        order = [r["ticker"] for r in out["by_ticker"]]
        # AXTI 0.8 first; MU and NVDA tie at 0.5 → alphabetical tiebreak.
        assert order == ["AXTI", "MU", "NVDA"]


class TestTickerRecapPollutionMatcherShapes:
    def test_tuple_matcher_fills_fingerprints(self, store):
        for i in range(2):
            _insert(store, ticker_in_title="NVDA",
                    title=f"Why Did NVDA Stock Drop Today {i}")
        _insert(store, ticker_in_title="NVDA",
                title="These Stocks Are Today's Movers: NVDA")

        out = store.ticker_recap_pollution(
            ["NVDA"], _tuple_recap_matcher, hours=24, min_total=1
        )
        row = out["by_ticker"][0]
        assert row["fingerprints"] == {
            "why_did_stock": 2, "todays_movers_list": 1,
        }
        assert row["recap"] == 3

    def test_bool_matcher_no_fingerprints(self, store):
        for i in range(3):
            _insert(store, ticker_in_title="NVDA",
                    title=f"Why Did NVDA Stock Drop Today {i}")
        out = store.ticker_recap_pollution(
            ["NVDA"], _bool_recap_matcher, hours=24, min_total=1
        )
        # Boolean matcher emits no name → empty dict (not absent key).
        assert out["by_ticker"][0]["fingerprints"] == {}
        assert out["by_ticker"][0]["recap"] == 3

    def test_buggy_matcher_does_not_crash(self, store):
        """A matcher that raises must NEVER take down the metric. The
        row is treated as non-recap (best-effort discipline mirrors
        source_recap_pollution)."""
        def _bad_matcher(_t):
            raise RuntimeError("regex blew up")

        for i in range(3):
            _insert(store, ticker_in_title="NVDA",
                    title=f"NVDA real headline {i}")

        out = store.ticker_recap_pollution(
            ["NVDA"], _bad_matcher, hours=24, min_total=1
        )
        # No crash; every row counted as non-recap.
        assert out["by_ticker"][0]["total"] == 3
        assert out["by_ticker"][0]["recap"] == 0
        assert out["by_ticker"][0]["recap_rate"] == 0.0


class TestTickerRecapPollutionVolumeAndCap:
    def test_min_total_excludes_low_volume_tickers(self, store):
        """A 1-of-1 ticker reads "100% polluted" without volume to justify
        the verdict — excluded by the volume floor. Mirrors the discipline
        source_recap_pollution and book_alert_coverage both apply."""
        # NVDA: 5 urgent, 5 recap → 100% (passes min_total=3)
        for i in range(5):
            _insert(store, ticker_in_title="NVDA",
                    title=f"Why Did NVDA Stock Drop Today {i}")
        # MU: 1 urgent, 1 recap → 100% (BELOW min_total=3, excluded)
        _insert(store, ticker_in_title="MU",
                title="Why Did MU Stock Drop Today")

        out = store.ticker_recap_pollution(
            ["NVDA", "MU"], _bool_recap_matcher, hours=24, min_total=3
        )
        tickers = {r["ticker"] for r in out["by_ticker"]}
        assert tickers == {"NVDA"}  # MU excluded by volume floor
        # But MU still counts toward the GLOBAL totals (any-ticker-mention
        # row-count). NVDA 5 + MU 1 = 6 unique rows.
        assert out["total_urgent"] == 6
        assert out["total_recap"] == 6

    def test_top_n_caps_response_size(self, store):
        for tk in ("AAA", "BBB", "CCC", "DDD"):
            for i in range(3):
                _insert(store, ticker_in_title=tk,
                        title=f"Why Did {tk} Stock Drop Today {i}")

        out = store.ticker_recap_pollution(
            ["AAA", "BBB", "CCC", "DDD"], _bool_recap_matcher,
            hours=24, min_total=1, top_n=2,
        )
        assert len(out["by_ticker"]) == 2


class TestTickerRecapPollutionInvariants:
    def test_backtest_rows_excluded(self, store):
        """``_LIVE_ONLY_CLAUSE`` discipline — synthetic backtest/opus rows
        with held-ticker mentions must NEVER inflate the metric. Pinned by
        a dedicated test because the rest of the recap-fingerprint corpus
        is small and the invariant is load-bearing (CLAUDE.md §5).
        """
        # 2 live urgent NVDA recap rows.
        for i in range(2):
            _insert(store, ticker_in_title="NVDA",
                    title=f"Why Did NVDA Stock Drop Today {i}", src="rss")
        # 5 synthetic NVDA rows the metric MUST ignore.
        for i in range(3):
            _insert(store, ticker_in_title="NVDA",
                    title=f"NVDA backtest synthetic {i}",
                    url=f"backtest://run_1/d/BUY/NVDA/{i}",
                    src="backtest_run_1_winner")
        for i in range(2):
            _insert(store, ticker_in_title="NVDA",
                    title=f"NVDA opus synthetic {i}",
                    src="opus_annotation_cycle_3",
                    url=f"https://opus.test/{i}")

        out = store.ticker_recap_pollution(
            ["NVDA"], _bool_recap_matcher, hours=24, min_total=1
        )
        row = out["by_ticker"][0]
        # Only the 2 live rows count, regardless of how many synthetic rows
        # share the held name + recap fingerprint.
        assert row["total"] == 2
        assert row["recap"] == 2

    def test_only_urgency_geq_1_rows_counted(self, store):
        """``urgent_queue_health`` and friends count urgency>=1; this metric
        does the same so a low-relevance NVDA row (urgency=0) never reads
        as "covered". Differentiates from total ticker mention volume."""
        _insert(store, ticker_in_title="NVDA",
                title="Why Did NVDA Stock Drop Today", urgency=1)
        _insert(store, ticker_in_title="NVDA",
                title="Why Did NVDA Stock Surge Today (low rel)", urgency=0)
        out = store.ticker_recap_pollution(
            ["NVDA"], _bool_recap_matcher, hours=24, min_total=1
        )
        assert out["by_ticker"][0]["total"] == 1

    def test_hours_window_excludes_old_rows(self, store):
        # 3 in-window urgent NVDA rows.
        for i in range(3):
            _insert(store, ticker_in_title="NVDA",
                    title=f"Why Did NVDA Stock Drop Today {i}", hours_ago=2.0)
        # 4 out-of-window rows (>24h) that must NOT count.
        for i in range(4):
            _insert(store, ticker_in_title="NVDA",
                    title=f"Why Did NVDA Stock Surge Today {i}",
                    hours_ago=48.0)
        out = store.ticker_recap_pollution(
            ["NVDA"], _bool_recap_matcher, hours=24, min_total=1
        )
        assert out["by_ticker"][0]["total"] == 3

    def test_match_surface_includes_summary(self, store):
        """Ticker matching surface is title + summary (decompressed). A
        held ticker mentioned ONLY in the body must still count — same
        discipline as urgency_label_split_by_ticker."""
        _insert(store, title="Generic news headline that names no symbol",
                summary="The full body discusses NVDA earnings in depth.",
                ticker_in_title=None)
        out = store.ticker_recap_pollution(
            ["NVDA"], _bool_recap_matcher, hours=24, min_total=1
        )
        assert out["by_ticker"] and out["by_ticker"][0]["ticker"] == "NVDA"
        assert out["by_ticker"][0]["total"] == 1

    def test_short_or_empty_tickers_ignored(self, store):
        """``len < 2`` would over-match (e.g. ``A`` matches "A new study").
        Empty / whitespace tickers must be silently dropped. Mirrors the
        4 sibling per-ticker primitives' discipline."""
        _insert(store, ticker_in_title="NVDA",
                title="Why Did NVDA Stock Drop Today")
        out = store.ticker_recap_pollution(
            ["", "A", "NVDA", "  ", None],  # type: ignore[list-item]
            _bool_recap_matcher, hours=24, min_total=1,
        )
        # Only NVDA survives the ``len >= 2`` + non-empty filter.
        assert {r["ticker"] for r in out["by_ticker"]} == {"NVDA"}


class TestTickerRecapPollutionSSOTParity:
    """The metric must accept the SSOT matchers from BOTH the alert path
    AND the briefing path. This is the structural anti-drift guard that
    keeps the storage layer dependency-free of analysis/watchers while
    proving the production matchers are wire-compatible with the metric.
    """

    def test_alert_side_ssot_matcher_works(self, store):
        """The alert-side SSOT matcher signature is ``(article_dict) ->
        (hit, name)``; the metric passes a TITLE STRING. Production callers
        adapt with a lambda — mirrors the dual-signature discipline
        ``test_source_recap_pollution`` uses for the same SSOT matcher."""
        from watchers.alert_agent import _looks_like_recap_template

        def matcher(title):
            return _looks_like_recap_template({"title": title})

        _insert(store, ticker_in_title="NVDA",
                title="Why Did NVDA Stock Drop Today")
        _insert(store, ticker_in_title="NVDA",
                title="Why Is NVDA Down 7.2% Since Last Earnings Report?")
        _insert(store, ticker_in_title="NVDA",
                title="NVDA tops Q1 estimates")  # not recap

        out = store.ticker_recap_pollution(
            ["NVDA"], matcher, hours=24, min_total=1,
        )
        row = out["by_ticker"][0]
        assert row["total"] == 3
        assert row["recap"] == 2
        # Both fingerprints recorded with their alert-side canonical names.
        assert set(row["fingerprints"].keys()) == {
            "why_did_stock", "why_is_pct_since",
        }

    def test_briefing_side_ssot_matcher_works(self, store):
        """Briefing-side SSOT matcher — same dict→title-string adapter as
        the alert-side test. Both production matchers MUST yield identical
        fingerprint names on identical input (the parity is structurally
        pinned by ``tests/test_briefing_recap_template.py``); this test
        proves the per-ticker metric works with both."""
        from analysis.claude_analyst import _looks_like_recap_template

        def matcher(title):
            return _looks_like_recap_template({"title": title})

        _insert(store, ticker_in_title="NVDA",
                title="Why Did NVDA Stock Drop Today")
        _insert(store, ticker_in_title="NVDA",
                title="NVDA Q1 2026 Earnings Call Highlights")
        _insert(store, ticker_in_title="NVDA",
                title="NVDA real Q1 beat headline")  # not recap

        out = store.ticker_recap_pollution(
            ["NVDA"], matcher, hours=24, min_total=1,
        )
        row = out["by_ticker"][0]
        assert row["total"] == 3
        assert row["recap"] == 2
        # Briefing-side matcher emits the same fingerprint names as alert side.
        assert set(row["fingerprints"].keys()) == {
            "why_did_stock", "earnings_call_recap",
        }
