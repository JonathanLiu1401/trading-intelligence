"""Tests for the market_movers collector.

Pins two recently-fixed properties:

1. ``_fetch_screener`` now logs at WARNING when the upstream request raises
   (formerly a bare ``except: return []`` made every network outage
   indistinguishable from an empty result set). The contract: the function
   STILL returns an empty list (the worker downstream handles that gracefully),
   but it MUST log so source_health diagnostics surface the outage.

2. ``collect_market_movers`` filters out small moves below MIN_GAINER_PCT
   and MIN_LOSER_PCT — a defensive threshold the source helper requires.
"""
import logging
from unittest.mock import MagicMock, patch

import pytest

from collectors import market_movers


def _quote(symbol="NVDA", name="NVIDIA Corp", price=100.0, chg_pct=5.0,
           volume=5_000_000, avg_vol=10_000_000):
    return {
        "symbol": symbol,
        "shortName": name,
        "regularMarketPrice": price,
        "regularMarketChangePercent": chg_pct,
        "regularMarketVolume": volume,
        "averageDailyVolume3Month": avg_vol,
    }


class TestFetchScreenerLogging:
    """The silent-exception fix: a network error MUST surface in the log
    even though the return value is still an empty list (the worker
    treats that as 'no new articles this cycle')."""

    def test_returns_empty_and_logs_on_network_error(self, monkeypatch, caplog):
        # requests.get raises a generic exception (e.g. timeout / DNS / SSL).
        def boom(*a, **kw):
            raise RuntimeError("simulated network outage")
        monkeypatch.setattr(market_movers.requests, "get", boom)

        with caplog.at_level(logging.WARNING, logger="market_movers"):
            out = market_movers._fetch_screener("day_gainers")

        # Returns an empty list — worker downstream handles this as no data.
        assert out == []
        # A WARNING-level record must be emitted, naming the screener and
        # the exception type so an operator tailing daemon.log can attribute
        # the outage.
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "day_gainers" in messages
        assert "RuntimeError" in messages

    def test_returns_empty_and_logs_on_bad_status(self, monkeypatch, caplog):
        # Yahoo occasionally returns 401/429 — raise_for_status must surface
        # them and the wrapper must log.
        resp = MagicMock()
        # Raising in raise_for_status mirrors requests' real behavior on >=400.
        resp.raise_for_status.side_effect = RuntimeError("HTTP 429")
        monkeypatch.setattr(market_movers.requests, "get", lambda *a, **k: resp)

        with caplog.at_level(logging.WARNING, logger="market_movers"):
            out = market_movers._fetch_screener("most_actives")

        assert out == []
        messages = " ".join(r.getMessage() for r in caplog.records)
        assert "most_actives" in messages

    def test_happy_path_returns_quotes_no_warning(self, monkeypatch, caplog):
        resp = MagicMock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {
            "finance": {"result": [{"quotes": [_quote(), _quote(symbol="MU")]}]}
        }
        monkeypatch.setattr(market_movers.requests, "get", lambda *a, **k: resp)

        with caplog.at_level(logging.WARNING, logger="market_movers"):
            out = market_movers._fetch_screener("day_gainers")

        assert len(out) == 2
        # On the happy path NO warning should be emitted — the
        # silent-network-error fix must not turn a healthy fetch into log noise.
        warnings = [r for r in caplog.records if r.levelno >= logging.WARNING]
        assert warnings == []


class TestMoverThresholds:
    """Defensive thresholds — sub-3% gainers / -3% losers are filtered out
    so the collector does not flood the briefing on low-volatility days."""

    def _setup_screener(self, monkeypatch, scr_id_to_quotes, tmp_path):
        # Redirect the seen_articles.db to a tmp file so the test doesn't
        # write into the repo's data/ dir.
        monkeypatch.setattr(market_movers, "DB_PATH",
                            tmp_path / "seen_articles.db")

        def fake_fetch(scr_id):
            return scr_id_to_quotes.get(scr_id, [])

        monkeypatch.setattr(market_movers, "_fetch_screener", fake_fetch)

    def test_small_gainer_below_threshold_is_filtered(self, monkeypatch, tmp_path):
        # 2.0% < MIN_GAINER_PCT=3.0 — must be dropped.
        small = _quote(chg_pct=2.0)
        big = _quote(symbol="MU", chg_pct=8.5)
        self._setup_screener(monkeypatch, {"day_gainers": [small, big]}, tmp_path)

        out = market_movers.collect_market_movers()

        # Only the >3% mover (MU) makes it through.
        symbols = [a["symbol"] for a in out]
        assert "MU" in symbols
        assert "NVDA" not in symbols

    def test_small_loser_above_threshold_is_filtered(self, monkeypatch, tmp_path):
        # -1.5% > MIN_LOSER_PCT=-3.0 — must be dropped.
        small_loss = _quote(symbol="AAPL", chg_pct=-1.5)
        big_loss = _quote(symbol="TSLA", chg_pct=-7.2)
        self._setup_screener(monkeypatch, {"day_losers": [small_loss, big_loss]},
                             tmp_path)

        out = market_movers.collect_market_movers()

        symbols = [a["symbol"] for a in out]
        assert "TSLA" in symbols
        assert "AAPL" not in symbols

    def test_skips_quotes_with_no_symbol_or_price(self, monkeypatch, tmp_path):
        bad1 = {"symbol": "", "regularMarketPrice": 10.0,
                "regularMarketChangePercent": 5.0}
        bad2 = {"symbol": "GOOG", "regularMarketPrice": None,
                "regularMarketChangePercent": 5.0}
        good = _quote(symbol="MSFT", chg_pct=4.0)
        self._setup_screener(monkeypatch,
                             {"day_gainers": [bad1, bad2, good]}, tmp_path)

        out = market_movers.collect_market_movers()
        symbols = [a["symbol"] for a in out]
        assert symbols == ["MSFT"]


class TestDedup:
    """Re-running the collector with the same upstream payload must NOT
    produce duplicate articles — seen_articles.db is the cross-cycle gate."""

    def test_second_run_returns_no_new_articles(self, monkeypatch, tmp_path):
        monkeypatch.setattr(market_movers, "DB_PATH",
                            tmp_path / "seen_articles.db")
        quotes = [_quote(symbol="NVDA", chg_pct=6.0)]
        monkeypatch.setattr(market_movers, "_fetch_screener",
                            lambda scr_id: quotes if scr_id == "day_gainers" else [])

        first = market_movers.collect_market_movers()
        second = market_movers.collect_market_movers()

        assert len(first) == 1
        # Second run sees the exact same article and must skip it via the
        # seen_articles dedup index.
        assert second == []


class TestMoverCooldown:
    """Per-(symbol, source_tag) cooldown — the SAME ticker bouncing inside one
    screener within a few minutes used to emit a NEW urgent article per tick
    because the title encodes the moving price (live 2026-05-19 18:01:25: MU
    surfaced FOUR distinct urgent rows across day_gainers + most_actives at
    +5.7% then +5.2% within the same minute). The cooldown gate suppresses
    re-emission of the same key within MOVER_COOLDOWN_MIN minutes."""

    def _setup(self, monkeypatch, tmp_path):
        monkeypatch.setattr(market_movers, "DB_PATH",
                            tmp_path / "seen_articles.db")

    def test_same_symbol_same_screener_changing_price_is_suppressed(
        self, monkeypatch, tmp_path
    ):
        """The live failure mode: MU at +5.7% emits, then 3 minutes later MU
        at +5.2% (different title → passes article-id dedup) — the cooldown
        gate must catch the re-emission."""
        self._setup(monkeypatch, tmp_path)
        ticks = iter([
            [_quote(symbol="MU", price=720.63, chg_pct=5.7)],
            [_quote(symbol="MU", price=716.84, chg_pct=5.2)],
        ])
        monkeypatch.setattr(
            market_movers, "_fetch_screener",
            lambda scr_id: next(ticks) if scr_id == "day_gainers" else [],
        )

        first = market_movers.collect_market_movers()
        second = market_movers.collect_market_movers()

        assert [a["symbol"] for a in first] == ["MU"]
        # The second tick has a different title (different price) so the
        # article-id dedup would NOT catch it. The cooldown gate must.
        assert second == [], (
            f"expected the second MU tick within MOVER_COOLDOWN_MIN to be "
            f"suppressed by the cooldown gate, got {second!r}"
        )

    def test_different_screener_for_same_symbol_is_not_suppressed(
        self, monkeypatch, tmp_path
    ):
        """The cooldown is scoped per source_tag so a ticker on both
        day_gainers and most_actives is still allowed to emit once per
        screener — they are independent signals."""
        self._setup(monkeypatch, tmp_path)
        nvda_g = _quote(symbol="NVDA", chg_pct=7.0)
        # most_actives is volume-driven, no chg_pct threshold; reuse same vol
        nvda_a = _quote(symbol="NVDA", chg_pct=0.5, volume=50_000_000)
        monkeypatch.setattr(
            market_movers, "_fetch_screener",
            lambda scr_id: {"day_gainers": [nvda_g],
                            "most_actives": [nvda_a]}.get(scr_id, []),
        )

        out = market_movers.collect_market_movers()
        sources = sorted({a["source"] for a in out})
        assert sources == ["YF/day_gainers", "YF/most_actives"], (
            f"NVDA on two different screeners must emit once per screener; "
            f"got sources={sources}"
        )

    def test_different_symbol_same_screener_is_not_suppressed(
        self, monkeypatch, tmp_path
    ):
        """The cooldown gate must not bleed across tickers — MU on cooldown
        cannot suppress a fresh NVDA mover on the same screener."""
        self._setup(monkeypatch, tmp_path)
        run_quotes = iter([
            [_quote(symbol="MU", chg_pct=5.7)],
            [_quote(symbol="NVDA", chg_pct=6.5)],
        ])
        monkeypatch.setattr(
            market_movers, "_fetch_screener",
            lambda scr_id: next(run_quotes) if scr_id == "day_gainers" else [],
        )

        first = market_movers.collect_market_movers()
        second = market_movers.collect_market_movers()

        assert [a["symbol"] for a in first] == ["MU"]
        assert [a["symbol"] for a in second] == ["NVDA"], (
            "a different ticker on the same screener must NOT be suppressed by "
            "an unrelated ticker's cooldown — they are independent emissions"
        )

    def test_cooldown_arms_only_after_successful_emit(
        self, monkeypatch, tmp_path
    ):
        """A sub-threshold gainer (filtered out by MIN_GAINER_PCT) must NOT
        arm the cooldown — otherwise the next run with a *real* mover would
        be silently suppressed even though we never told the analyst."""
        self._setup(monkeypatch, tmp_path)
        run_quotes = iter([
            [_quote(symbol="MU", chg_pct=1.5)],   # under threshold → dropped
            [_quote(symbol="MU", chg_pct=8.5)],   # genuine mover next cycle
        ])
        monkeypatch.setattr(
            market_movers, "_fetch_screener",
            lambda scr_id: next(run_quotes) if scr_id == "day_gainers" else [],
        )

        first = market_movers.collect_market_movers()
        second = market_movers.collect_market_movers()

        assert first == [], "sub-threshold gainer must not emit"
        assert [a["chg_pct"] for a in second] == [8.5], (
            "the genuine mover on the next cycle must emit — cooldown must "
            "only arm AFTER a successful emission, not after a filtered drop"
        )

    def test_cooldown_expires_after_window(self, monkeypatch, tmp_path):
        """Outside the MOVER_COOLDOWN_MIN window a re-emission is allowed —
        otherwise a mover that surges twice in one day would be silenced for
        the second push. Simulates time progression by mutating the stored
        last_emit_iso directly (cleaner than monkeypatching datetime)."""
        import sqlite3
        from datetime import datetime, timezone, timedelta
        self._setup(monkeypatch, tmp_path)
        monkeypatch.setattr(
            market_movers, "_fetch_screener",
            lambda scr_id: [_quote(symbol="MU", chg_pct=5.7,
                                   price=720.63 if scr_id == "day_gainers" else 0)]
                            if scr_id == "day_gainers" else [],
        )

        first = market_movers.collect_market_movers()
        assert [a["symbol"] for a in first] == ["MU"]

        # Backdate the cooldown row past the window
        old = (datetime.now(timezone.utc)
               - timedelta(minutes=market_movers.MOVER_COOLDOWN_MIN + 5))
        old_iso = old.strftime("%Y-%m-%dT%H:%M:%SZ")
        # Use the same DB the collector touches (per the monkeypatched
        # DB_PATH); a fresh connection that arms its OWN busy_timeout/WAL.
        conn = sqlite3.connect(str(market_movers.DB_PATH), timeout=10)
        try:
            conn.execute(
                "UPDATE mover_cooldown SET last_emit_iso=? "
                "WHERE key=?",
                (old_iso, market_movers._cooldown_key("MU", "YF/day_gainers")),
            )
            conn.commit()
        finally:
            conn.close()

        # The next emission has a different title (encoded price differs)
        # AND the cooldown has expired — the row should be allowed through.
        monkeypatch.setattr(
            market_movers, "_fetch_screener",
            lambda scr_id: [_quote(symbol="MU", chg_pct=5.2, price=716.84)]
                            if scr_id == "day_gainers" else [],
        )
        second = market_movers.collect_market_movers()
        assert [a["symbol"] for a in second] == ["MU"], (
            "after the cooldown window expires the same ticker must be "
            "allowed to emit a fresh mover article again"
        )

    def test_unparseable_cooldown_row_falls_through_to_emit(
        self, monkeypatch, tmp_path
    ):
        """A corrupted last_emit_iso (e.g. truncated string) must NOT
        permanently silence the ticker — the gate fails open. Same safe-default
        discipline as alert_agent / urgency_scorer date parsers."""
        import sqlite3
        self._setup(monkeypatch, tmp_path)

        monkeypatch.setattr(
            market_movers, "_fetch_screener",
            lambda scr_id: [] if scr_id != "day_gainers" else
                            [_quote(symbol="MU", chg_pct=5.7)],
        )
        # Force-arm the cooldown with a junk timestamp
        from collectors.market_movers import _ensure_db
        DB = tmp_path / "seen_articles.db"
        conn = sqlite3.connect(str(DB), timeout=10)
        _ensure_db(conn)
        conn.execute(
            "INSERT INTO mover_cooldown (key, last_emit_iso) VALUES (?, ?)",
            (market_movers._cooldown_key("MU", "YF/day_gainers"),
             "not-a-real-timestamp"),
        )
        conn.commit()
        conn.close()

        out = market_movers.collect_market_movers()
        assert [a["symbol"] for a in out] == ["MU"], (
            "a corrupted cooldown timestamp must fall through to emit (gate "
            "fails open) — otherwise a single bad row permanently silences "
            "the ticker"
        )


class TestSeenDbHardening:
    """Pins the SQLite hardening on the shared ``seen_articles.db``.

    Live failure (2026-05-19 daemon.log): market_movers_worker hit "database is
    locked" seven times in eleven minutes (10s/20s/40s/80s/160s/320s/600s back-
    off), then was flagged DEAD by the supervisor (last_ok=1471s — 25 min with
    no successful collection cycle). The cause was the bare
    ``sqlite3.connect(str(DB_PATH))`` — default ``busy_timeout=0`` so the first
    concurrent writer on the shared dedup store raised immediately. Every other
    collector that touches ``seen_articles.db`` already uses ``timeout=30`` +
    ``PRAGMA busy_timeout=30000`` (google_news, rss_collector, finnhub, ...);
    market_movers was the lone holdout. The hardening MUST stick — without it
    the worker silently degrades to ~0 cycles/h on a busy daemon."""

    def test_collect_uses_timeout_30(self, monkeypatch, tmp_path):
        """collect_market_movers must open seen_articles.db with timeout=30
        — without it the Python wrapper bails on the FIRST contended write."""
        captured: dict = {}
        real_connect = market_movers.sqlite3.connect

        def spy_connect(*args, **kwargs):
            captured["kwargs"] = dict(kwargs)
            return real_connect(*args, **kwargs)

        monkeypatch.setattr(market_movers, "DB_PATH",
                            tmp_path / "seen_articles.db")
        monkeypatch.setattr(market_movers.sqlite3, "connect", spy_connect)
        monkeypatch.setattr(market_movers, "_fetch_screener", lambda s: [])

        market_movers.collect_market_movers()

        assert captured.get("kwargs", {}).get("timeout") == 30, (
            "market_movers must connect with timeout=30; got "
            f"{captured.get('kwargs')!r}. Without the Python-wrapper timeout "
            "the FIRST concurrent write on the shared seen_articles.db raises "
            "OperationalError('database is locked') immediately and the cycle "
            "aborts — observed live as exponential backoff to DEAD."
        )

    def test_ensure_db_sets_busy_timeout_and_wal(self, tmp_path):
        """``_ensure_db`` must arm the connection with busy_timeout=30000 ms
        and WAL journal mode — what SQLite actually consults during a lock
        wait. busy_timeout=0 (default) means a contended write fails instantly,
        not after a 30s grace period, regardless of the Python-wrapper timeout
        above. Verifies the PRAGMA state directly so a regression to
        ``conn.execute("PRAGMA ...")`` being silently dropped is caught."""
        import sqlite3
        db_path = tmp_path / "seen_articles.db"
        conn = sqlite3.connect(str(db_path), timeout=30,
                               check_same_thread=False)
        try:
            market_movers._ensure_db(conn)
            busy = conn.execute("PRAGMA busy_timeout").fetchone()[0]
            journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
        finally:
            conn.close()

        assert busy == 30000, (
            f"expected busy_timeout=30000 ms, got {busy}. With busy_timeout=0 "
            f"(SQLite default) any concurrent writer on the shared "
            f"seen_articles.db raises 'database is locked' immediately."
        )
        assert (journal or "").lower() == "wal", (
            f"expected WAL journal mode, got {journal!r}. Other collectors on "
            f"the same shared DB use WAL — a non-WAL writer would block every "
            f"other reader for the duration of its transaction."
        )
