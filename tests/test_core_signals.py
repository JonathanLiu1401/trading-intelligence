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
