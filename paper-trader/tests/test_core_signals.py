"""Tests for paper_trader.signals — articles.db queries and ticker extraction.

These tests use a temp SQLite DB that mirrors digital-intern's schema so we
can drive the queries deterministically without touching the real DB. The
backtest-filter clause is exercised directly: a backtest:// row must NOT be
returned, and a synthetic source row must NOT be returned.
"""
from __future__ import annotations

import sqlite3
import sys
import zlib
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import signals


def _build_articles_db(path: Path, rows: list[dict]) -> None:
    """Create an articles.db with just the columns paper_trader/signals.py uses."""
    conn = sqlite3.connect(str(path))
    conn.execute(
        """
        CREATE TABLE articles (
            id INTEGER PRIMARY KEY,
            url TEXT,
            title TEXT,
            source TEXT,
            ai_score REAL,
            urgency REAL,
            first_seen TEXT,
            full_text BLOB
        )
        """
    )
    for r in rows:
        conn.execute(
            "INSERT INTO articles (id, url, title, source, ai_score, urgency, first_seen, full_text) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (
                r.get("id"),
                r.get("url"),
                r.get("title"),
                r.get("source"),
                r.get("ai_score"),
                r.get("urgency"),
                r.get("first_seen"),
                zlib.compress(r.get("body", "").encode("utf-8")) if r.get("body") else None,
            ),
        )
    conn.commit()
    conn.close()


@pytest.fixture
def fake_articles_db(tmp_path, monkeypatch):
    db = tmp_path / "articles.db"
    # Override the path discovery so signals._db_path() returns our temp file.
    monkeypatch.setattr(signals, "USB_DB", Path("/nonexistent/articles.db"))
    monkeypatch.setattr(signals, "LOCAL_DB", db)
    return db


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hours_ago(h: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=h)).isoformat()


class TestExtractTickers:
    def test_dollar_prefixed_ticker_extracted(self):
        assert "NVDA" in signals._extract_tickers("Big move in $NVDA today")

    def test_plain_allcaps_extracted(self):
        assert "AMD" in signals._extract_tickers("AMD beats earnings")

    def test_common_acronyms_filtered_out(self):
        # The whole text is acronyms; the result should contain no tickers.
        out = signals._extract_tickers("FOMC PCE CPI Q1 GDP say AND THE FED")
        assert out == set()

    def test_single_letter_filtered(self):
        # Single letters are below the length floor (2 chars min).
        assert "A" not in signals._extract_tickers("A and I went to lunch")

    def test_mixed_tickers_and_noise(self):
        out = signals._extract_tickers("NVDA and AMD beat Q1 estimates, said the FED")
        assert "NVDA" in out
        assert "AMD" in out
        assert "Q1" not in out
        assert "FED" not in out

    def test_cashtag_overrides_noise_filter(self):
        # `AI` is in _NOT_TICKERS so a bare mention is dropped, but an
        # explicit cashtag ($AI) is an intentional signal and is kept.
        # This asymmetry is deliberate — pin it so it isn't "fixed" away.
        assert "AI" not in signals._extract_tickers("the AI boom continues")
        assert "AI" in signals._extract_tickers("watching $AI into the print")

    def test_empty_string_returns_empty(self):
        assert signals._extract_tickers("") == set()
        assert signals._extract_tickers(None) == set()

    def test_time_and_date_abbreviations_filtered(self):
        # Time / date / chat abbreviations are common in news headlines and
        # are NOT tickers — extracting them pollutes the per-article `tickers`
        # field Opus reads in the live decision prompt and biases the model
        # toward "names" that don't exist. Pin so a future trim of
        # _NOT_TICKERS cannot silently re-introduce these false positives.
        for noise in ("AM", "ET", "JUN", "JUL", "BTW", "FYI", "ETA"):
            out = signals._extract_tickers(f"hours are 10 {noise} today")
            assert noise not in out, (
                f"expected {noise!r} to be filtered as noise, "
                f"got {sorted(out)!r}"
            )

    def test_time_phrase_extracts_no_tickers(self):
        # A typical news-headline time phrase produces nothing —
        # historically gave {'AM','ET'} which silently misled Opus.
        assert signals._extract_tickers(
            "Fed meeting at 10 AM ET on Wednesday"
        ) == set()

    def test_real_ticker_still_extracted_alongside_noise(self):
        out = signals._extract_tickers(
            "NVDA earnings call 5 PM ET; AMD reports Q3 later"
        )
        assert "NVDA" in out
        assert "AMD" in out
        # Time/date noise is dropped.
        for noise in ("AM", "ET", "PM", "Q3"):
            assert noise not in out

    def test_common_english_words_filtered_out(self):
        # All-caps English words observed live polluting the `tickers` field
        # Opus reads in the live decision prompt. Each entry is verified NOT
        # to collide with a known real-money ticker. Pin so a future trim of
        # _NOT_TICKERS cannot silently re-introduce the live regression.
        noise_phrases = [
            ("JOIN NVDA AT 5PM", "JOIN", "NVDA"),
            ("CEO TOLD INVESTORS", "TOLD", None),
            ("Apple HIGH today", "HIGH", None),
            ("NEAR HIGH again", "NEAR", None),
            ("WHEN WILL FED CUT", "WILL", None),
            ("THIS WAS UNDER PRESSURE", "WAS", None),
            ("OVER 50% UP", "OVER", None),
        ]
        for phrase, false_positive, real_ticker in noise_phrases:
            out = signals._extract_tickers(phrase)
            assert false_positive not in out, (
                f"{false_positive!r} should be filtered from {phrase!r}; "
                f"got {sorted(out)!r}"
            )
            if real_ticker is not None:
                assert real_ticker in out, (
                    f"{real_ticker!r} should still be extracted from "
                    f"{phrase!r}; got {sorted(out)!r}"
                )

    def test_known_collision_tickers_still_extracted(self):
        # OPEN (Opendoor) and LOW (Lowe's) are real publicly-traded tickers
        # whose symbols collide with common English words. They MUST NOT be
        # added to _NOT_TICKERS, or the live trader goes blind to legit
        # news on those names. Pin so the trade-off is explicit.
        assert "OPEN" in signals._extract_tickers("OPEN announces new partnership")
        assert "LOW" in signals._extract_tickers("LOW lifts dividend, raises guidance")

    def test_cashtag_overrides_expanded_noise_filter(self):
        # Even for words newly added to _NOT_TICKERS, an explicit $cashtag
        # is an intentional signal and is kept (the AI / $AI asymmetry
        # pin extended).
        assert "TOLD" in signals._extract_tickers("watching $TOLD into the print")
        assert "JOIN" in signals._extract_tickers("$JOIN if you must")

    def test_finance_verbs_and_nouns_filtered(self):
        # Verbs and nouns that dominate financial-news headlines but are not
        # real tickers. Each entry observed live polluting the `tickers`
        # field Opus reads (e.g. "Fed CUT RATES today" → tickers=CUT,RATES).
        # Pin so a future trim cannot silently re-introduce the regression.
        cases = [
            ("Fed CUT RATES today", ("CUT", "RATES"), None),
            ("NVDA BEATS earnings, MISSES guidance", ("BEATS", "MISSES"), "NVDA"),
            ("AMD shares JUMP on STRONG REVENUE", ("JUMP", "REVENUE"), "AMD"),
            ("MU PRICE TARGET RAISED to $200", ("PRICE", "RAISED"), "MU"),
            ("MSFT REPORTS Q3 PROFIT", ("REPORTS", "PROFIT"), "MSFT"),
            ("AAPL DROPS on guidance miss", ("DROPS",), "AAPL"),
            ("Markets RALLY as Fed HIKES rates", ("RALLY", "HIKES"), None),
        ]
        for phrase, false_positives, real_ticker in cases:
            out = signals._extract_tickers(phrase)
            for fp in false_positives:
                assert fp not in out, (
                    f"{fp!r} should be filtered from {phrase!r}; "
                    f"got {sorted(out)!r}"
                )
            if real_ticker is not None:
                assert real_ticker in out, (
                    f"{real_ticker!r} should still be extracted from "
                    f"{phrase!r}; got {sorted(out)!r}"
                )

    def test_finance_verb_cashtag_overrides_filter(self):
        # Even for finance verbs like CUT / RATES, an explicit $cashtag is
        # an intentional signal and is kept (pin the cashtag-override
        # asymmetry for the new finance-verb additions).
        assert "CUT" in signals._extract_tickers("watching $CUT into the print")
        assert "BEAT" in signals._extract_tickers("$BEAT calls active")

    def test_additional_headline_verbs_filtered(self):
        # A second batch of finance-headline verbs / plural nouns observed
        # live polluting the `tickers` field Opus reads (e.g. "Nvidia BLOWS
        # PAST estimates" → tickers=BLOWS,PAST). Pin so a future trim of
        # _NOT_TICKERS cannot silently re-introduce them while preserving
        # the real ticker in each phrase.
        cases = [
            ("Nvidia BLOWS PAST Wall Street estimates", ("BLOWS", "PAST"), "NVDA"),
            ("NVDA TANKS on guidance miss", ("TANKS",), "NVDA"),
            ("CEO TALKS UP the outlook", ("TALKS",), None),
            ("Q3 SALES TOPPED estimates", ("SALES", "TOPPED"), None),
            ("AMD CLIMBS in pre-market", ("CLIMBS",), "AMD"),
            ("Tech stocks TUMBLE; chips SLIP", ("TUMBLE", "SLIP"), None),
            ("Banks SLID as yields SOAR", ("SLID", "SOAR"), None),
            ("Retail prints fresh LOWS", ("LOWS",), None),
        ]
        for phrase, false_positives, real_ticker in cases:
            out = signals._extract_tickers(phrase)
            for fp in false_positives:
                assert fp not in out, (
                    f"{fp!r} should be filtered from {phrase!r}; "
                    f"got {sorted(out)!r}"
                )
            if real_ticker is not None:
                assert real_ticker in out, (
                    f"{real_ticker!r} should still be extracted from "
                    f"{phrase!r}; got {sorted(out)!r}"
                )

    def test_new_noise_words_cashtag_still_overrides(self):
        # The cashtag asymmetry must hold for the new entries too — a
        # deliberate $TANKS / $SLIP is kept even though the bare token is
        # now filtered. (No collision with a real ticker; this only pins
        # that the bare-token filter does not leak into the cashtag path.)
        assert "TANKS" in signals._extract_tickers("eyeing $TANKS today")
        assert "SLIP" in signals._extract_tickers("$SLIP momentum building")

    def test_real_low_volume_tickers_not_collateral_damaged(self):
        # ME / REAL / CHIP / BIRD are genuine listed tickers deliberately
        # NOT added to the noise filter — a bare-ALLCAPS mention still
        # surfaces them (the OPEN / LOW precedent). Pin the trade-off so a
        # future "cleanup" does not silently blind the trader to them.
        assert "REAL" in signals._extract_tickers("REAL beats on subscriber growth")
        assert "CHIP" in signals._extract_tickers("CHIP lifts full-year guidance")

    def test_exchange_and_index_names_filtered(self):
        # Exchange and index name tokens that previously leaked into the
        # ``tickers`` field Opus reads. Each entry is NOT a real public
        # ticker (NYSE is an exchange, DJIA is the Dow Jones index name);
        # the cashtag bypass keeps an explicit $NYSE deliberate-signal use
        # case working. Pin so a future trim cannot re-introduce the live
        # 2026-05-28 regression where every market-hours "NYSE opens..."
        # headline polluted the prompt with NYSE-as-a-ticker.
        for noise in ("NYSE", "DJIA"):
            out = signals._extract_tickers(f"{noise} opens at 9:30 AM ET")
            assert noise not in out, (
                f"{noise!r} should be filtered; got {sorted(out)!r}")

    def test_dow_jones_phrase_does_not_emit_jones(self):
        # "DOW JONES" headlines previously extracted both DOW (real ticker,
        # Dow Inc.) and JONES (no such public ticker). The DOW must still
        # surface (it is a real listed name); JONES must NOT.
        out = signals._extract_tickers("DOW JONES rallies on Fed signal")
        assert "DOW" in out
        assert "JONES" not in out

    def test_wall_street_uppercase_extracts_nothing(self):
        # ALL-CAPS "WALL STREET" headlines previously emitted both WALL
        # and STREET — neither is a public ticker (State Street is STT).
        out = signals._extract_tickers("WALL STREET ENDS WEEK HIGHER")
        assert "WALL" not in out
        assert "STREET" not in out

    def test_buys_sells_plural_verb_filtered(self):
        # Plural verb forms used in headline ledes ("Berkshire BUYS more
        # Apple", "Insider SELLS up to $1M") are NOT real tickers. Bare
        # BUY / SELL are deliberately preserved (analyst-rating headlines:
        # "RBC upgrades NVDA to BUY") — that asymmetry is the documented
        # rationale in _NOT_TICKERS. AAPL must still surface from the
        # "Berkshire BUYS more Apple" alias-path so the trader still
        # gets the news.
        out = signals._extract_tickers("Berkshire BUYS more Apple")
        assert "BUYS" not in out
        assert "AAPL" in out  # alias-path on "Apple"

        out2 = signals._extract_tickers("Insider SELLS up to 1M shares")
        assert "SELLS" not in out2

    def test_exchange_cashtag_bypass_still_works(self):
        # The cashtag asymmetry holds for the new entries — a deliberate
        # $NYSE / $DJIA / $JONES cashtag is an intentional signal even
        # though the bare token is now filtered. (Mirrors the AI / $AI
        # asymmetry pin.) Without this, the cashtag override could
        # silently regress without anyone noticing.
        assert "NYSE" in signals._extract_tickers("watching $NYSE today")
        assert "DJIA" in signals._extract_tickers("$DJIA breakout")
        assert "JONES" in signals._extract_tickers("eyeing $JONES")


class TestTickerAliasExtraction:
    """Locks the company-name → ticker alias path. A headline that names a
    company by its everyday name (e.g. "Nvidia") historically extracted no
    ticker because the regex needed $cashtag or an ALLCAPS token; the alias
    feature folds the company name into the same tag set the rest of the
    pipeline consumes."""

    def test_nvidia_alias_extracts_nvda(self):
        assert "NVDA" in signals._extract_tickers(
            "Nvidia surges to record on chip demand"
        )

    def test_apple_alias_extracts_aapl(self):
        assert "AAPL" in signals._extract_tickers(
            "Apple expands service revenue beats"
        )

    def test_tesla_alias_extracts_tsla(self):
        assert "TSLA" in signals._extract_tickers(
            "Tesla cuts model 3 prices in Europe"
        )

    def test_multi_word_alias_taiwan_semiconductor_extracts_tsm(self):
        # Multi-word alias matches because \b sits between any word/non-word
        # transition, so spaces don't break the boundary.
        assert "TSM" in signals._extract_tickers(
            "Taiwan Semiconductor raises capex guidance"
        )

    def test_alias_is_case_insensitive(self):
        # "NVIDIA" (all caps), "nvidia" (lower), "Nvidia" (title) all map to NVDA.
        # The all-caps token would already be caught by the regex extractor (and
        # NVIDIA is NOT in _NOT_TICKERS), so this test specifically locks the
        # lower-case alias path.
        assert "NVDA" in signals._extract_tickers("nvidia rallies")
        assert "NVDA" in signals._extract_tickers("Nvidia rallies")

    def test_alias_substring_does_not_falsely_match(self):
        # Word-boundary discipline: "appletini" or "apples" must NOT map to AAPL.
        # The substring "apple" inside another word should be ignored — same
        # contract as strategy._WORD_TO_TICKER_LIVE_PATTERNS after the
        # rain→TQQQ fix.
        assert "AAPL" not in signals._extract_tickers(
            "the applesauce supply chain")
        assert "AAPL" not in signals._extract_tickers(
            "pineapple cocktail launches")

    def test_alias_does_not_duplicate_existing_ticker(self):
        # Headline carries both the ticker symbol AND the company name.
        # _extract_tickers returns a set, so the ticker is present exactly once.
        out = signals._extract_tickers(
            "NVDA / Nvidia beats on earnings"
        )
        assert "NVDA" in out
        # Sanity — no surprise extra tickers from the company-name pass.
        # AMD is uppercase 3-char, would also extract if present — it isn't.
        assert "AMD" not in out

    def test_alias_extraction_keeps_existing_ticker_extraction(self):
        # The alias pass is additive — it must not break the ALLCAPS / cashtag
        # extraction or the _NOT_TICKERS filter for the same body.
        out = signals._extract_tickers(
            "Nvidia rallies; AMD also up Q3; FOMC and PCE noise"
        )
        assert "NVDA" in out
        assert "AMD" in out
        assert "FOMC" not in out
        assert "PCE" not in out
        assert "Q3" not in out

    def test_empty_text_alias_path_does_not_raise(self):
        # The alias loop must respect the same empty-text contract as the
        # regex extractor (returns empty set, never raises).
        assert signals._extract_tickers("") == set()
        assert signals._extract_tickers(None) == set()

    def test_allcaps_company_name_does_not_pollute_with_fake_ticker(self):
        # An ALLCAPS headline like "APPLE BEATS EARNINGS" used to extract
        # BOTH `APPLE` (fake — Apple's ticker is AAPL, never APPLE) and
        # `AAPL` (via the alias path). Opus then read `tickers=APPLE,AAPL`
        # in the prompt block — non-existent-ticker pollution that confuses
        # the decision engine. The alias-false-positive filter strips the
        # shouted-company-name form so only the canonical ticker survives.
        # Locks the fix for the four documented collisions (alias len 2-5,
        # alias.upper() != ticker): apple/tesla/intel/tsmc.
        out = signals._extract_tickers("APPLE BEATS EARNINGS today")
        assert "AAPL" in out
        assert "APPLE" not in out

        out = signals._extract_tickers("TESLA stock plunges 5%")
        assert "TSLA" in out
        assert "TESLA" not in out

        out = signals._extract_tickers("INTEL upgrades chip design")
        assert "INTC" in out
        assert "INTEL" not in out

        out = signals._extract_tickers("TSMC raises capex guidance")
        assert "TSM" in out
        assert "TSMC" not in out

    def test_alias_filter_keeps_alias_that_equals_real_ticker(self):
        # ASML's company-name alias is "asml" — upper-cased == the canonical
        # ticker ASML, so it is a *legitimate* extraction, NOT a false
        # positive. The filter must only strip aliases whose upper form
        # differs from the canonical ticker (e.g. APPLE/AAPL); a regression
        # that silently filters ASML out would silently undercount semis
        # news for the live trader.
        assert "ASML" in signals._extract_tickers("ASML beats Q3 expectations")
        assert "ASML" in signals._extract_tickers("ASML guidance raised")

    def test_cashtag_alias_upper_does_not_pollute_with_fake_ticker(self):
        # The shouted-company-name fix originally ran on the bare-ALLCAPS
        # extractor only — a ``$cashtag`` form bypassed the filter and silently
        # re-introduced the same bug: ``$APPLE`` produced both ``APPLE`` (a
        # non-existent ticker; the alias path already maps the body to AAPL)
        # AND ``AAPL``. The fix narrows the cashtag bypass to ``_NOT_TICKERS``
        # ONLY (preserving the documented ``$AI``/``$TOLD``/``$CUT`` override);
        # ``_ALIAS_UPPER_FALSE_POSITIVES`` is filtered on both paths because
        # there is no legitimate use case for a ``$APPLE``/``$TESLA``/``$INTEL``
        # cashtag (the real symbol is the alias). Pin so a future refactor
        # cannot silently re-introduce the pollution.
        out = signals._extract_tickers("watching $APPLE into earnings")
        assert "AAPL" in out
        assert "APPLE" not in out, (
            f"$APPLE cashtag must not produce the fake APPLE ticker; "
            f"got {sorted(out)!r}"
        )

        out = signals._extract_tickers("rumor $TESLA breakout")
        assert "TSLA" in out
        assert "TESLA" not in out

        out = signals._extract_tickers("$INTEL guidance recap")
        assert "INTC" in out
        assert "INTEL" not in out

        out = signals._extract_tickers("$TSMC update")
        assert "TSM" in out
        assert "TSMC" not in out

    def test_cashtag_not_tickers_override_still_pinned_after_alias_filter(self):
        # The narrow scope of the cashtag-alias filter: it must NOT also
        # filter ``_NOT_TICKERS`` from cashtags. ``$AI`` (Sportradar), ``$TOLD``
        # and ``$CUT`` remain the documented cashtag-override asymmetry.
        # Regression guard for the prior, over-broad fix that filtered both
        # sets on cashtags.
        assert "AI" in signals._extract_tickers("watching $AI into the print")
        assert "TOLD" in signals._extract_tickers("$TOLD reports earnings")
        assert "CUT" in signals._extract_tickers("$CUT pops on guide")

    def test_alias_false_positive_set_only_contains_distinct_aliases(self):
        # Lock the membership of _ALIAS_UPPER_FALSE_POSITIVES so a future
        # alias addition with a same-as-ticker entry (length 2-5) doesn't
        # accidentally regress the ASML case. The set is the four observed
        # collisions; verify by inclusion (additions are fine, deletions
        # must not silently happen).
        fp = signals._ALIAS_UPPER_FALSE_POSITIVES
        # ASML must NOT be in the false-positive set.
        assert "ASML" not in fp
        # The four documented collisions must be filtered.
        for shouted in ("APPLE", "TESLA", "INTEL", "TSMC"):
            assert shouted in fp


class TestTickerSentimentsAliasPath:
    """The alias pass must propagate through ``ticker_sentiments`` and
    ``get_ticker_sentiment`` — these are the per-name aggregates Opus reads
    in the live prompt, and a company-name headline silently dropped from
    them was the documented under-count this feature closes."""

    def test_company_name_headline_counts_toward_ticker(self, fake_articles_db):
        _build_articles_db(fake_articles_db, [
            {"id": 1, "url": "http://a", "title": "Nvidia surges to record",
             "source": "x", "ai_score": 8.0, "urgency": 0,
             "first_seen": _now_iso(), "body": ""},
        ])
        out = signals.ticker_sentiments(["NVDA"], hours=24)
        assert len(out) == 1
        # Pre-fix: n=0 (silent miss). Post-fix: n=1 with the article's score.
        assert out[0]["n"] == 1
        assert out[0]["avg_score"] == 8.0
        assert out[0]["max_score"] == 8.0

    def test_get_ticker_sentiment_alias_path(self, fake_articles_db):
        _build_articles_db(fake_articles_db, [
            {"id": 1, "url": "http://a", "title": "Apple expands services",
             "source": "x", "ai_score": 7.0, "urgency": 1,
             "first_seen": _now_iso(), "body": ""},
            {"id": 2, "url": "http://b", "title": "AAPL hits new high",
             "source": "x", "ai_score": 9.0, "urgency": 0,
             "first_seen": _now_iso(), "body": ""},
        ])
        out = signals.get_ticker_sentiment("AAPL", hours=24)
        # BOTH headlines count — one by alias, one by symbol.
        assert out["n"] == 2
        assert out["urgent"] == 1
        assert out["max_score"] == 9.0

    def test_alias_does_not_double_count_when_symbol_also_present(
            self, fake_articles_db):
        # A single article with BOTH the ticker symbol AND the company name
        # is one match, not two. The alias path is gated by ``or _alias_match``
        # — same row in the outer loop, no double-counting.
        _build_articles_db(fake_articles_db, [
            {"id": 1, "url": "http://a", "title": "NVDA / Nvidia beats EPS",
             "source": "x", "ai_score": 8.0, "urgency": 0,
             "first_seen": _now_iso(), "body": ""},
        ])
        out = signals.ticker_sentiments(["NVDA"], hours=24)
        assert out[0]["n"] == 1
        assert out[0]["avg_score"] == 8.0

    def test_unrelated_alias_does_not_pollute_other_ticker(
            self, fake_articles_db):
        # An article about Nvidia must NOT pollute MU sentiment (a sibling
        # semi). Aliases are scoped per-ticker; this would only fail if the
        # alias_match dispatch silently OR'd patterns across tickers.
        _build_articles_db(fake_articles_db, [
            {"id": 1, "url": "http://a", "title": "Nvidia surges",
             "source": "x", "ai_score": 8.0, "urgency": 0,
             "first_seen": _now_iso(), "body": ""},
        ])
        out = {r["ticker"]: r for r in
               signals.ticker_sentiments(["NVDA", "MU"], hours=24)}
        assert out["NVDA"]["n"] == 1
        assert out["MU"]["n"] == 0

    def test_alias_substring_does_not_falsely_match_in_body_scan(
            self, fake_articles_db):
        # The same word-boundary discipline applied at the body-scan path
        # (not just extraction): an "applesauce" headline must NOT count for
        # AAPL. Otherwise the under-count fix would re-introduce noise into
        # the opposite direction.
        _build_articles_db(fake_articles_db, [
            {"id": 1, "url": "http://a",
             "title": "applesauce supply chain stays steady",
             "source": "x", "ai_score": 5.0, "urgency": 0,
             "first_seen": _now_iso(),
             "body": "pineapple cocktail launches expected"},
        ])
        out = signals.ticker_sentiments(["AAPL"], hours=24)
        assert out[0]["n"] == 0, (
            "substring of 'apple' (applesauce/pineapple) must not match the "
            f"AAPL alias word-boundary, got n={out[0]['n']}"
        )


class TestDecompress:
    def test_roundtrip(self):
        blob = zlib.compress(b"hello world")
        assert signals._decompress(blob) == "hello world"

    def test_empty_blob_returns_empty(self):
        assert signals._decompress(b"") == ""
        assert signals._decompress(None) == ""

    def test_corrupt_blob_returns_empty(self):
        assert signals._decompress(b"not-zlib-data") == ""


class TestGetTopSignals:
    def test_empty_db_returns_empty(self, fake_articles_db):
        _build_articles_db(fake_articles_db, [])
        assert signals.get_top_signals(n=10) == []

    def test_missing_db_returns_empty(self, monkeypatch, tmp_path):
        # Point both candidate paths at nonexistent files.
        monkeypatch.setattr(signals, "USB_DB", tmp_path / "nope.db")
        monkeypatch.setattr(signals, "LOCAL_DB", tmp_path / "nope2.db")
        assert signals.get_top_signals(n=10) == []

    def test_min_score_threshold_filters_below(self, fake_articles_db):
        _build_articles_db(fake_articles_db, [
            {"id": 1, "url": "http://a", "title": "low", "source": "x",
             "ai_score": 2.0, "urgency": 0, "first_seen": _now_iso(), "body": "low signal"},
            {"id": 2, "url": "http://b", "title": "high", "source": "x",
             "ai_score": 8.0, "urgency": 0, "first_seen": _now_iso(), "body": "high signal NVDA"},
        ])
        rows = signals.get_top_signals(n=10, hours=24, min_score=4.0)
        assert len(rows) == 1
        assert rows[0]["title"] == "high"

    def test_score_descending_order(self, fake_articles_db):
        _build_articles_db(fake_articles_db, [
            {"id": 1, "url": "http://a", "title": "mid", "source": "x",
             "ai_score": 5.0, "urgency": 0, "first_seen": _now_iso(), "body": ""},
            {"id": 2, "url": "http://b", "title": "high", "source": "x",
             "ai_score": 9.0, "urgency": 0, "first_seen": _now_iso(), "body": ""},
            {"id": 3, "url": "http://c", "title": "low_pass", "source": "x",
             "ai_score": 4.5, "urgency": 0, "first_seen": _now_iso(), "body": ""},
        ])
        rows = signals.get_top_signals(n=10, hours=24, min_score=4.0)
        scores = [r["ai_score"] for r in rows]
        assert scores == sorted(scores, reverse=True)

    def test_backtest_url_filtered(self, fake_articles_db):
        # Backtest synthetic rows must never reach the live trader.
        _build_articles_db(fake_articles_db, [
            {"id": 1, "url": "backtest://NVDA/2025-05-01", "title": "synthetic",
             "source": "backtest_opus", "ai_score": 9.0, "urgency": 1,
             "first_seen": _now_iso(), "body": "should not appear"},
            {"id": 2, "url": "http://real.com", "title": "real article",
             "source": "reuters", "ai_score": 9.0, "urgency": 1,
             "first_seen": _now_iso(), "body": "real"},
        ])
        rows = signals.get_top_signals(n=10, hours=24, min_score=4.0)
        assert len(rows) == 1
        assert rows[0]["url"] == "http://real.com"

    def test_opus_annotation_source_filtered(self, fake_articles_db):
        _build_articles_db(fake_articles_db, [
            {"id": 1, "url": "http://x", "title": "annot",
             "source": "opus_annotation_v1", "ai_score": 9.0, "urgency": 1,
             "first_seen": _now_iso(), "body": "should not appear"},
            {"id": 2, "url": "http://y", "title": "real",
             "source": "bloomberg", "ai_score": 5.0, "urgency": 0,
             "first_seen": _now_iso(), "body": "real"},
        ])
        rows = signals.get_top_signals(n=10, hours=24, min_score=4.0)
        assert len(rows) == 1
        assert rows[0]["source"] == "bloomberg"

    def test_old_articles_filtered_by_hours(self, fake_articles_db):
        _build_articles_db(fake_articles_db, [
            {"id": 1, "url": "http://old", "title": "stale", "source": "x",
             "ai_score": 9.0, "urgency": 0, "first_seen": _hours_ago(24), "body": ""},
            {"id": 2, "url": "http://new", "title": "fresh", "source": "x",
             "ai_score": 9.0, "urgency": 0, "first_seen": _now_iso(), "body": ""},
        ])
        rows = signals.get_top_signals(n=10, hours=2, min_score=4.0)
        assert len(rows) == 1
        assert rows[0]["url"] == "http://new"

    def test_tickers_extracted_into_output(self, fake_articles_db):
        _build_articles_db(fake_articles_db, [
            {"id": 1, "url": "http://x", "title": "NVDA crushes earnings",
             "source": "x", "ai_score": 9.0, "urgency": 0,
             "first_seen": _now_iso(), "body": "AMD and NVDA up 5%"},
        ])
        rows = signals.get_top_signals(n=10, hours=24, min_score=4.0)
        assert len(rows) == 1
        tickers = set(rows[0]["tickers"])
        assert "NVDA" in tickers
        assert "AMD" in tickers


class TestTickerSentiments:
    def test_unmentioned_ticker_returns_zero_defaults(self, fake_articles_db):
        _build_articles_db(fake_articles_db, [
            {"id": 1, "url": "http://x", "title": "AAPL beats", "source": "x",
             "ai_score": 9.0, "urgency": 0, "first_seen": _now_iso(), "body": ""},
        ])
        out = signals.ticker_sentiments(["NVDA"], hours=24)
        assert len(out) == 1
        assert out[0] == {"ticker": "NVDA", "avg_score": 0.0, "max_score": 0.0, "n": 0, "urgent": 0}

    def test_average_score_calculation(self, fake_articles_db):
        _build_articles_db(fake_articles_db, [
            {"id": 1, "url": "http://a", "title": "NVDA earnings",
             "source": "x", "ai_score": 4.0, "urgency": 0,
             "first_seen": _now_iso(), "body": ""},
            {"id": 2, "url": "http://b", "title": "NVDA downgrade",
             "source": "x", "ai_score": 8.0, "urgency": 0,
             "first_seen": _now_iso(), "body": ""},
        ])
        out = signals.ticker_sentiments(["NVDA"], hours=24)
        nvda = out[0]
        assert nvda["n"] == 2
        # avg = (4 + 8) / 2 = 6.0
        assert nvda["avg_score"] == pytest.approx(6.0)
        assert nvda["max_score"] == 8.0

    def test_urgent_counter_increments(self, fake_articles_db):
        _build_articles_db(fake_articles_db, [
            {"id": 1, "url": "http://a", "title": "NVDA flash crash",
             "source": "x", "ai_score": 9.0, "urgency": 1,
             "first_seen": _now_iso(), "body": ""},
            {"id": 2, "url": "http://b", "title": "NVDA boring news",
             "source": "x", "ai_score": 4.0, "urgency": 0,
             "first_seen": _now_iso(), "body": ""},
        ])
        out = signals.ticker_sentiments(["NVDA"], hours=24)
        assert out[0]["urgent"] == 1
        assert out[0]["n"] == 2

    def test_backtest_rows_filtered(self, fake_articles_db):
        _build_articles_db(fake_articles_db, [
            {"id": 1, "url": "backtest://x", "title": "NVDA synthetic",
             "source": "backtest_run1", "ai_score": 9.0, "urgency": 0,
             "first_seen": _now_iso(), "body": ""},
            {"id": 2, "url": "http://real", "title": "NVDA real",
             "source": "bloomberg", "ai_score": 5.0, "urgency": 0,
             "first_seen": _now_iso(), "body": ""},
        ])
        out = signals.ticker_sentiments(["NVDA"], hours=24)
        # Only the bloomberg row contributes.
        assert out[0]["n"] == 1
        assert out[0]["avg_score"] == pytest.approx(5.0)

    def test_dollar_prefixed_ticker_matched(self, fake_articles_db):
        # The pattern is `(?:\$|\b)NVDA\b` so $NVDA must also count.
        _build_articles_db(fake_articles_db, [
            {"id": 1, "url": "http://a", "title": "$NVDA pop",
             "source": "x", "ai_score": 7.0, "urgency": 0,
             "first_seen": _now_iso(), "body": ""},
        ])
        out = signals.ticker_sentiments(["NVDA"], hours=24)
        assert out[0]["n"] == 1
        assert out[0]["max_score"] == 7.0

    def test_word_boundary_prevents_substring_match(self, fake_articles_db):
        # "MUSE" should NOT count as a mention of "MU".
        _build_articles_db(fake_articles_db, [
            {"id": 1, "url": "http://a", "title": "MUSEUM opens",
             "source": "x", "ai_score": 5.0, "urgency": 0,
             "first_seen": _now_iso(), "body": "MUSEUMS everywhere"},
        ])
        out = signals.ticker_sentiments(["MU"], hours=24)
        assert out[0]["n"] == 0


class TestGetUrgentArticles:
    def test_only_urgency_ge_1_returned(self, fake_articles_db):
        _build_articles_db(fake_articles_db, [
            {"id": 1, "url": "http://a", "title": "flat", "source": "x",
             "ai_score": 5.0, "urgency": 0,
             "first_seen": _now_iso(), "body": ""},
            {"id": 2, "url": "http://b", "title": "BREAKING", "source": "x",
             "ai_score": 5.0, "urgency": 1,
             "first_seen": _now_iso(), "body": ""},
        ])
        out = signals.get_urgent_articles(minutes=60)
        assert len(out) == 1
        assert out[0]["title"] == "BREAKING"

    def test_null_ai_score_coerced_to_zero(self, fake_articles_db):
        # If a row has NULL ai_score, the get_urgent_articles output must not
        # crash downstream formatting that does f"{ai_score:.1f}".
        _build_articles_db(fake_articles_db, [
            {"id": 1, "url": "http://a", "title": "BREAKING", "source": "x",
             "ai_score": None, "urgency": 2,
             "first_seen": _now_iso(), "body": ""},
        ])
        out = signals.get_urgent_articles(minutes=60)
        assert len(out) == 1
        # Must be coerced to a float so downstream formatting works.
        assert out[0]["ai_score"] == 0.0
        # And the format string used downstream must not raise.
        f"{out[0]['ai_score']:.1f}"


def _write_gzip_jsonl(path: Path, lines: list[str]) -> None:
    """Write raw text lines (already serialized) into a gzip file so the
    corrupt-line / empty-line resilience branches can be exercised verbatim."""
    import gzip

    with gzip.open(path, "wt", encoding="utf-8") as gz:
        gz.write("\n".join(lines) + "\n")


class TestGetHistoricalSignals:
    """`get_historical_signals` is the backtest-fallback gzip reader. It has
    branching nothing else exercises: a `score`/`ai_score` `or`-fallback, a
    strict `< min_score` threshold, a `limit` cap, and per-line resilience to
    corrupt JSON / non-numeric scores. Each test pins one branch with an exact
    expectation so a `<`→`<=` or `continue`→`break` regression fails loudly."""

    def test_missing_file_returns_empty(self, tmp_path, monkeypatch):
        monkeypatch.setattr(signals, "HISTORICAL_GZ", tmp_path / "nope.json.gz")
        assert signals.get_historical_signals() == []

    def test_min_score_threshold_is_strict_less_than(self, tmp_path, monkeypatch):
        import json

        gz = tmp_path / "h.json.gz"
        _write_gzip_jsonl(gz, [
            json.dumps({"id": "lo", "score": 3.99}),
            json.dumps({"id": "eq", "score": 4.0}),   # == threshold → KEPT (cond is `< min_score`)
            json.dumps({"id": "hi", "score": 9.5}),
        ])
        monkeypatch.setattr(signals, "HISTORICAL_GZ", gz)
        out = signals.get_historical_signals(min_score=4.0)
        assert [r["id"] for r in out] == ["eq", "hi"]

    def test_score_key_absent_falls_back_to_ai_score(self, tmp_path, monkeypatch):
        import json

        gz = tmp_path / "h.json.gz"
        # `score` falsy/absent → `rec.get("score") or rec.get("ai_score")`.
        _write_gzip_jsonl(gz, [
            json.dumps({"id": "via_ai", "ai_score": 7.0}),         # no "score" key
            json.dumps({"id": "zero_score", "score": 0, "ai_score": 8.0}),  # 0 is falsy → uses ai_score
            json.dumps({"id": "neither"}),                          # both absent → score None → skipped
        ])
        monkeypatch.setattr(signals, "HISTORICAL_GZ", gz)
        out = signals.get_historical_signals(min_score=4.0)
        assert [r["id"] for r in out] == ["via_ai", "zero_score"]

    def test_limit_caps_result_count(self, tmp_path, monkeypatch):
        import json

        gz = tmp_path / "h.json.gz"
        _write_gzip_jsonl(gz, [json.dumps({"id": i, "score": 9.0}) for i in range(5)])
        monkeypatch.setattr(signals, "HISTORICAL_GZ", gz)
        out = signals.get_historical_signals(min_score=4.0, limit=2)
        assert [r["id"] for r in out] == [0, 1]   # stops the moment len(out) >= limit

    def test_corrupt_and_nonnumeric_lines_skipped_reading_continues(self, tmp_path, monkeypatch):
        import json

        gz = tmp_path / "h.json.gz"
        _write_gzip_jsonl(gz, [
            json.dumps({"id": "ok1", "score": 5.0}),
            "{not valid json",                                  # JSONDecodeError → skip, keep reading
            "",                                                 # blank line → skip
            json.dumps({"id": "bad_score", "score": "NaNish"}),  # float() raises → skip, keep reading
            json.dumps({"id": "ok2", "score": 6.0}),            # must still be reached
        ])
        monkeypatch.setattr(signals, "HISTORICAL_GZ", gz)
        out = signals.get_historical_signals(min_score=4.0)
        assert [r["id"] for r in out] == ["ok1", "ok2"]


# ───────────────────────── freshness-aware DB resolver ─────────────────────
# `_db_path()` historically returned the USB copy whenever it merely
# `exists()`. The digital-intern daemon falls back to writing the LOCAL copy
# when the USB mount is unavailable, leaving a stale USB mirror that still
# exists — and the live trader then read day-old news while every other
# surface read the fresh LOCAL DB ("split brain"; detected by /api/feed-health
# but never root-fixed). These tests pin the freshness-aware replacement and
# the advisor's full decision matrix with exact expectations.

def _iso_ago(hours: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


def _make_db(path: Path, live_ago_h: float | None, backtest_ago_h: float | None = None):
    """Build an articles.db whose newest *live* row is `live_ago_h` old, with
    an optional NEWER `backtest://` row that the live-only filter must ignore."""
    rows = []
    if live_ago_h is not None:
        rows.append({"id": 1, "url": "https://x/a", "title": "live", "source": "rss",
                     "ai_score": 8.0, "urgency": 0, "first_seen": _iso_ago(live_ago_h)})
    if backtest_ago_h is not None:
        rows.append({"id": 2, "url": "backtest://run_1/2026-05-16/BUY/NVDA",
                     "title": "synthetic", "source": "backtest_run_1_winner",
                     "ai_score": 5.0, "urgency": 0, "first_seen": _iso_ago(backtest_ago_h)})
    _build_articles_db(path, rows)


@pytest.fixture
def two_dbs(tmp_path, monkeypatch):
    """USB + LOCAL temp paths wired into the resolver, cache reset each test."""
    usb = tmp_path / "usb" / "articles.db"
    local = tmp_path / "local" / "articles.db"
    usb.parent.mkdir()
    local.parent.mkdir()
    monkeypatch.setattr(signals, "USB_DB", usb)
    monkeypatch.setattr(signals, "LOCAL_DB", local)
    signals._reset_resolver_cache()
    yield usb, local
    signals._reset_resolver_cache()


class TestChoosePure:
    """`_choose` is the pure decision given a freshness map — no IO."""

    def test_tie_prefers_local(self, two_dbs):
        usb, local = two_dbs
        ts = _iso_ago(1)
        # strict > keeps the first candidate on equality, and _candidates() is
        # (LOCAL, USB) since 6227cd5 — LOCAL is the live daemon's write path.
        assert signals._choose({usb: ts, local: ts}) == local

    def test_fresher_local_wins(self, two_dbs):
        usb, local = two_dbs
        assert signals._choose({usb: _iso_ago(30), local: _iso_ago(1)}) == local

    def test_fresher_usb_wins(self, two_dbs):
        usb, local = two_dbs
        assert signals._choose({usb: _iso_ago(1), local: _iso_ago(30)}) == usb

    def test_single_candidate_returned(self, two_dbs):
        usb, local = two_dbs
        assert signals._choose({local: _iso_ago(5)}) == local
        assert signals._choose({usb: _iso_ago(5)}) == usb

    def test_both_unreadable_falls_back_to_local_first(self, two_dbs):
        usb, local = two_dbs
        # Both exist but neither yielded a timestamp → LOCAL-first order
        # (6227cd5: _candidates() is (LOCAL, USB); LOCAL is the daemon write path).
        assert signals._choose({usb: None, local: None}) == local

    def test_neither_exists_returns_local(self, two_dbs):
        # Empty freshness map → preserve the legacy "neither → LOCAL_DB" contract.
        _, local = two_dbs
        assert signals._choose({}) == local


class TestDbPathFreshness:
    """End-to-end resolver over real temp DBs — the bug-fix matrix."""

    def test_stale_usb_loses_to_fresh_local_and_ignores_backtest_rows(self, two_dbs):
        usb, local = two_dbs
        # USB: live row 30h old, but a *newer* backtest row 0.1h old that the
        # live-only filter MUST exclude (else the stale mirror wins falsely).
        _make_db(usb, live_ago_h=30, backtest_ago_h=0.1)
        _make_db(local, live_ago_h=1)
        assert signals._db_path() == local

    def test_both_fresh_prefers_usb(self, two_dbs):
        usb, local = two_dbs
        _make_db(usb, live_ago_h=1)
        _make_db(local, live_ago_h=2)
        assert signals._db_path() == usb

    def test_usb_only_present(self, two_dbs):
        usb, _ = two_dbs
        _make_db(usb, live_ago_h=3)            # local path never created
        assert signals._db_path() == usb

    def test_local_only_present(self, two_dbs):
        _, local = two_dbs
        _make_db(local, live_ago_h=3)
        assert signals._db_path() == local

    def test_cache_keyed_on_candidates_not_just_time(self, two_dbs, tmp_path, monkeypatch):
        usb, local = two_dbs
        _make_db(usb, live_ago_h=1)
        assert signals._db_path() == usb       # resolves + caches on (usb, local)
        # Repoint the candidates at a DIFFERENT path (what every other signals
        # test does — each gets a unique tmp LOCAL_DB) WITHOUT resetting the
        # cache. A TTL cache keyed only on time would wrongly keep returning
        # the stale `usb` for 120s and cross-contaminate sibling tests; keyed
        # on the candidate tuple it must re-resolve to the new DB.
        other = tmp_path / "other" / "articles.db"
        other.parent.mkdir()
        _make_db(other, live_ago_h=1)
        monkeypatch.setattr(signals, "USB_DB", other.parent / "missing.db")
        monkeypatch.setattr(signals, "LOCAL_DB", other)
        assert signals._db_path() == other     # re-resolved, not time-cached


class TestAgeHours:
    def test_offset_and_z_and_naive_and_garbage(self):
        now = datetime.now(timezone.utc)
        off = (now - timedelta(hours=2)).isoformat()                 # ...+00:00
        z = (now - timedelta(hours=2)).strftime("%Y-%m-%dT%H:%M:%SZ")
        naive = (now - timedelta(hours=2)).replace(tzinfo=None).isoformat()
        assert abs(signals._age_hours(off) - 2.0) < 0.05
        assert abs(signals._age_hours(z) - 2.0) < 0.05
        assert abs(signals._age_hours(naive) - 2.0) < 0.05           # assumed UTC
        assert signals._age_hours("not-a-date") is None
        assert signals._age_hours(None) is None
        assert signals._age_hours("") is None


class TestFeedStatusAndWarn:
    def test_split_brain_flags_restart_needed(self, two_dbs):
        usb, local = two_dbs
        _make_db(usb, live_ago_h=30)           # what legacy resolver would pick
        _make_db(local, live_ago_h=0.2)        # what the fix picks
        st = signals.feed_status()
        assert st["chosen"] == str(local)
        assert st["legacy_choice"] == str(usb)
        assert st["split_brain"] is True       # actionable: a stale process is blind
        assert st["stale"] is False            # the freshest copy itself is current

    def test_all_stale_is_stale_not_split_brain(self, two_dbs):
        usb, local = two_dbs
        _make_db(usb, live_ago_h=30)
        _make_db(local, live_ago_h=40)
        st = signals.feed_status()
        assert st["chosen"] == str(usb)        # USB is the freshest of the two
        assert st["legacy_choice"] == str(usb)
        assert st["split_brain"] is False      # legacy == chosen
        assert st["stale"] is True             # pipeline down — restart won't help

    def test_warn_fires_once_then_dedups(self, two_dbs, capfd):
        usb, local = two_dbs
        _make_db(usb, live_ago_h=30)
        fresh = {usb: _iso_ago(30)}
        signals._maybe_warn_stale(usb, fresh)
        first = capfd.readouterr().err
        signals._maybe_warn_stale(usb, fresh)  # same path → deduped, silent
        second = capfd.readouterr().err
        assert "WARNING reading STALE feed" in first
        assert "30.0h old" in first
        assert second == ""

    def test_no_warn_when_fresh(self, two_dbs, capfd):
        usb, _ = two_dbs
        signals._maybe_warn_stale(usb, {usb: _iso_ago(1)})
        assert capfd.readouterr().err == ""

    def test_unparseable_other_timestamp_does_not_crash(self, two_dbs, capfd):
        # Regression: when the *chosen* feed is stale and ANOTHER candidate
        # carries an unparseable ISO timestamp (corrupt DB row /
        # implementation drift), _age_hours returns None for that other
        # candidate. The warning composer used to feed that None straight
        # into ``{:.1f}`` and TypeError'd, taking down the whole decide()
        # cycle. The composer must now silently skip the unparseable
        # candidate while still emitting the warning for the chosen one.
        usb, local = two_dbs
        # chosen is the stale USB; the LOCAL candidate has a garbage ts
        freshness = {usb: _iso_ago(30), local: "not-an-iso-timestamp"}
        signals._maybe_warn_stale(usb, freshness)
        err = capfd.readouterr().err
        # Warning still fires for the stale chosen feed
        assert "WARNING reading STALE feed" in err
        assert "30.0h old" in err
        # And the unparseable sibling is not surfaced (silently skipped)
        assert "not-an-iso-timestamp" not in err


class TestCheckFreshnessCLI:
    """`_print_freshness_report` is the `--check-freshness` body; its return
    value is the shell exit code (3 split-brain, 2 stale, 0 ok)."""

    def test_exit_3_on_split_brain(self, two_dbs, capsys):
        usb, local = two_dbs
        _make_db(usb, live_ago_h=30)
        _make_db(local, live_ago_h=0.2)
        rc = signals._print_freshness_report()
        out = capsys.readouterr().out
        assert rc == 3
        assert "SPLIT-BRAIN" in out and "RESTART" in out

    def test_exit_2_on_all_stale(self, two_dbs, capsys):
        usb, local = two_dbs
        _make_db(usb, live_ago_h=30)
        _make_db(local, live_ago_h=40)
        rc = signals._print_freshness_report()
        assert rc == 2
        assert "STALE" in capsys.readouterr().out

    def test_exit_0_when_fresh(self, two_dbs, capsys):
        usb, local = two_dbs
        _make_db(usb, live_ago_h=0.5)
        _make_db(local, live_ago_h=1)
        rc = signals._print_freshness_report()
        assert rc == 0
        assert "OK" in capsys.readouterr().out


class TestGetTickerSentiment:
    """Single-ticker `get_ticker_sentiment` — a DISTINCT code path from the
    bulk `ticker_sentiments` (its own per-row compiled regex + aggregation),
    with ZERO prior direct coverage. The word-boundary case (AMDOCS must not
    count as AMD) is the regression that historically bites the
    ``(?:\\$|\\b)TKR\\b`` pattern, so it is locked here exactly as the bulk
    variant locks "MUSE" ≠ "MU".
    """

    def test_no_connection_returns_zero_defaults(self, monkeypatch):
        # No DB anywhere -> _connect_ro() is None -> zeroed dict, never raises.
        monkeypatch.setattr(signals, "USB_DB", Path("/nonexistent/u.db"))
        monkeypatch.setattr(signals, "LOCAL_DB", Path("/nonexistent/l.db"))
        out = signals.get_ticker_sentiment("NVDA", hours=4)
        assert out == {"ticker": "NVDA", "avg_score": 0.0,
                       "max_score": 0.0, "n": 0, "urgent": 0}

    def test_avg_max_n_exact(self, fake_articles_db):
        _build_articles_db(fake_articles_db, [
            {"id": 1, "url": "http://a", "title": "NVDA earnings beat",
             "source": "x", "ai_score": 4.0, "urgency": 0,
             "first_seen": _now_iso(), "body": ""},
            {"id": 2, "url": "http://b", "title": "NVDA guidance raise",
             "source": "x", "ai_score": 9.0, "urgency": 0,
             "first_seen": _now_iso(), "body": ""},
        ])
        out = signals.get_ticker_sentiment("NVDA", hours=24)
        assert out["n"] == 2
        assert out["avg_score"] == pytest.approx(6.5)  # (4 + 9) / 2
        assert out["max_score"] == 9.0
        assert out["urgent"] == 0

    def test_urgent_only_counts_urgency_ge_1(self, fake_articles_db):
        _build_articles_db(fake_articles_db, [
            {"id": 1, "url": "http://a", "title": "NVDA trading halt",
             "source": "x", "ai_score": 8.0, "urgency": 2,
             "first_seen": _now_iso(), "body": ""},
            {"id": 2, "url": "http://b", "title": "NVDA quiet session",
             "source": "x", "ai_score": 5.0, "urgency": 0,
             "first_seen": _now_iso(), "body": ""},
        ])
        out = signals.get_ticker_sentiment("NVDA", hours=24)
        assert out["n"] == 2
        assert out["urgent"] == 1  # only the urgency>=1 row

    def test_unmentioned_ticker_zero_defaults_no_crash(self, fake_articles_db):
        _build_articles_db(fake_articles_db, [
            {"id": 1, "url": "http://a", "title": "AAPL up on services",
             "source": "x", "ai_score": 7.0, "urgency": 0,
             "first_seen": _now_iso(), "body": ""},
        ])
        out = signals.get_ticker_sentiment("NVDA", hours=24)
        assert out == {"ticker": "NVDA", "avg_score": 0.0,
                       "max_score": 0.0, "n": 0, "urgent": 0}

    def test_word_boundary_amdocs_is_not_amd(self, fake_articles_db):
        # The single-ticker pattern is `(?:\$|\b)AMD\b`; "AMDOCS" must NOT match
        # — the exact substring-leak regression the bulk path also guards.
        _build_articles_db(fake_articles_db, [
            {"id": 1, "url": "http://a", "title": "AMDOCS signs telco deal",
             "source": "x", "ai_score": 9.0, "urgency": 1,
             "first_seen": _now_iso(), "body": "AMDOCS revenue grew sharply"},
        ])
        out = signals.get_ticker_sentiment("AMD", hours=24)
        assert out["n"] == 0
        assert out["urgent"] == 0

    def test_dollar_tag_in_body_matches(self, fake_articles_db):
        _build_articles_db(fake_articles_db, [
            {"id": 1, "url": "http://a", "title": "chip roundup",
             "source": "x", "ai_score": 6.0, "urgency": 0,
             "first_seen": _now_iso(), "body": "watching $AMD into the print"},
        ])
        out = signals.get_ticker_sentiment("AMD", hours=24)
        assert out["n"] == 1
        assert out["max_score"] == 6.0

    def test_backtest_rows_excluded(self, fake_articles_db):
        _build_articles_db(fake_articles_db, [
            {"id": 1, "url": "backtest://x", "title": "NVDA synthetic",
             "source": "backtest_run1", "ai_score": 10.0, "urgency": 0,
             "first_seen": _now_iso(), "body": ""},
            {"id": 2, "url": "http://real", "title": "NVDA real move",
             "source": "reuters", "ai_score": 3.0, "urgency": 0,
             "first_seen": _now_iso(), "body": ""},
        ])
        out = signals.get_ticker_sentiment("NVDA", hours=24)
        # Only the live reuters row contributes (live-only clause, invariant #3).
        assert out["n"] == 1
        assert out["avg_score"] == pytest.approx(3.0)


def _fake_score(**attrs):
    import types
    return types.SimpleNamespace(**attrs)


class TestGetMlPredictions:
    """`get_ml_predictions` bridges to digital-intern's ``ml.inference``. It
    has four guard branches (import-fail / empty input / empty default /
    scoring-raises) and one zip-mapping body, none previously exercised. The
    ml import is faked through ``sys.modules`` so the test stays fully offline.
    """

    def _install(self, monkeypatch, score_articles):
        import types
        fake = types.ModuleType("ml.inference")
        fake.score_articles = score_articles
        monkeypatch.setitem(sys.modules, "ml", types.ModuleType("ml"))
        monkeypatch.setitem(sys.modules, "ml.inference", fake)

    def test_ml_import_failure_returns_empty(self, monkeypatch):
        # A None entry makes `from ml.inference import score_articles` raise
        # ModuleNotFoundError -> caught -> [] (caller falls back to rules).
        monkeypatch.setitem(sys.modules, "ml", __import__("types").ModuleType("ml"))
        monkeypatch.setitem(sys.modules, "ml.inference", None)
        assert signals.get_ml_predictions([{"id": 1}]) == []

    def test_explicit_empty_articles_short_circuits(self, monkeypatch):
        def _no_call(arts):
            raise AssertionError("score_articles must not run on empty input")
        self._install(monkeypatch, _no_call)
        assert signals.get_ml_predictions([]) == []

    def test_none_articles_defaults_to_top_signals(self, monkeypatch):
        captured = {}

        def score(arts):
            captured["arts"] = arts
            return [_fake_score(relevance=0.9, urgency=0.4, rel_std=0.1,
                                urg_std=0.2, needs_llm=False)]

        self._install(monkeypatch, score)
        sentinel = [{"id": 7, "title": "T", "tickers": ["NVDA"]}]
        monkeypatch.setattr(signals, "get_top_signals", lambda *a, **k: sentinel)

        out = signals.get_ml_predictions(None)
        # None -> get_top_signals(30, hours=6, min_score=0.0) feeds the scorer.
        assert captured["arts"] is sentinel
        assert out == [{
            "id": 7, "title": "T", "tickers": ["NVDA"],
            "relevance": 0.9, "urgency": 0.4, "rel_std": 0.1,
            "urg_std": 0.2, "needs_llm": False,
        }]

    def test_none_articles_empty_default_returns_empty(self, monkeypatch):
        def _no_call(arts):
            raise AssertionError("must short-circuit before scoring")
        self._install(monkeypatch, _no_call)
        monkeypatch.setattr(signals, "get_top_signals", lambda *a, **k: [])
        assert signals.get_ml_predictions(None) == []

    def test_score_articles_exception_returns_empty(self, monkeypatch):
        def boom(arts):
            raise RuntimeError("inference model not loaded")
        self._install(monkeypatch, boom)
        assert signals.get_ml_predictions([{"id": 1, "title": "x"}]) == []

    def test_zip_truncates_to_shorter_scores(self, monkeypatch):
        # Two articles but only ONE score -> zip yields exactly one mapped row
        # (the second article is silently dropped — locked behaviour).
        self._install(monkeypatch, lambda arts: [
            _fake_score(relevance=1.0, urgency=0.0, rel_std=0.0,
                        urg_std=0.0, needs_llm=True)
        ])
        arts = [{"id": 1, "title": "A", "tickers": ["X"]},
                {"id": 2, "title": "B", "tickers": ["Y"]}]
        out = signals.get_ml_predictions(arts)
        assert len(out) == 1
        assert out[0]["id"] == 1 and out[0]["needs_llm"] is True

    def test_missing_tickers_key_defaults_to_empty_list(self, monkeypatch):
        self._install(monkeypatch, lambda arts: [
            _fake_score(relevance=0.5, urgency=0.5, rel_std=0.0,
                        urg_std=0.0, needs_llm=False)
        ])
        # Article has no "tickers" key -> a.get("tickers", []) must yield [].
        out = signals.get_ml_predictions([{"id": 3, "title": "no tickers"}])
        assert out[0]["tickers"] == []
