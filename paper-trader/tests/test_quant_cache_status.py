"""Tests for `paper_trader.analytics.quant_cache_status` and the
`/api/quant-cache-status` dashboard endpoint.

Each test asserts SPECIFIC values (counts, statuses, ages, ttl_remaining_s)
to actually exercise the builder logic — never just "didn't raise". The
builder pulls module state from `paper_trader.strategy` so every test
clears both caches first (autouse fixture) to isolate.
"""
from __future__ import annotations

import time
import types

import pytest

import paper_trader.strategy as strategy
from paper_trader.analytics.quant_cache_status import (
    build_quant_cache_status,
    VERDICT_DEGRADED,
    VERDICT_EMPTY,
    VERDICT_HEALTHY,
)


@pytest.fixture(autouse=True)
def _reset_quant_caches():
    strategy._QUANT_CACHE.clear()
    strategy._QUANT_NEG_CACHE.clear()
    yield
    strategy._QUANT_CACHE.clear()
    strategy._QUANT_NEG_CACHE.clear()


# Synthetic quant record shaped like a real `get_quant_signals_live` output —
# only the fields the builder surfaces need to be plausible.
def _mkrec(rsi=55.0, macd="bullish", mom5=2.5, mom20=8.0, cross=False):
    return {
        "RSI": rsi,
        "MACD": macd,
        "rsi": rsi,
        "mom_5d": mom5,
        "mom_20d": mom20,
        "macd_below_zero_cross": cross,
    }


# ─────────────────────────── empty-state path ───────────────────────────


class TestEmptyState:
    """No entries in either cache → VERDICT_EMPTY, zero counts."""

    def test_empty_caches_no_filter(self):
        snap = build_quant_cache_status(strategy, requested=None)
        assert snap["verdict"] == VERDICT_EMPTY
        assert snap["n_total"] == 0
        assert snap["n_fresh"] == 0
        assert snap["n_stale"] == 0
        assert snap["n_dark"] == 0
        assert snap["n_never"] == 0
        assert snap["rows"] == []

    def test_empty_caches_with_request_returns_never(self):
        snap = build_quant_cache_status(
            strategy, requested=["NVDA", "AMD"]
        )
        # When all requested tickers are absent → all NEVER, verdict HEALTHY
        # (no dark entries) — DEGRADED is reserved for actually-dark feeds.
        assert snap["n_total"] == 2
        assert snap["n_never"] == 2
        assert snap["n_dark"] == 0
        statuses = {r["ticker"]: r["status"] for r in snap["rows"]}
        assert statuses == {"NVDA": "NEVER", "AMD": "NEVER"}


# ─────────────────────────── FRESH path ─────────────────────────────────


class TestFreshPath:
    """A positive-cache entry younger than _QUANT_TTL must surface as FRESH
    with the indicator fields the trader cares about."""

    def test_fresh_entry_surfaces_signals(self):
        rec = _mkrec(rsi=72.0, macd="bullish", mom5=4.5, mom20=12.0,
                     cross=True)
        now = 1_000_000.0
        strategy._QUANT_CACHE["NVDA"] = (rec, now - 60.0)  # 60s old

        snap = build_quant_cache_status(
            strategy, requested=["NVDA"], now=now
        )
        assert snap["verdict"] == VERDICT_HEALTHY
        assert snap["n_fresh"] == 1
        row = snap["rows"][0]
        assert row["status"] == "FRESH"
        assert row["age_s"] == 60
        # TTL remaining = 300 - 60 = 240
        assert row["ttl_remaining_s"] == 240
        # Surfaced signal fields match the source record.
        assert row["signals"]["RSI"] == 72.0
        assert row["signals"]["MACD"] == "bullish"
        assert row["signals"]["mom_5d"] == 4.5
        assert row["signals"]["mom_20d"] == 12.0
        assert row["signals"]["macd_below_zero_cross"] is True


# ─────────────────────────── STALE path ─────────────────────────────────


class TestStalePath:
    """A positive-cache entry older than _QUANT_TTL (300s default) must
    surface as STALE — would be re-fetched on the next live call."""

    def test_stale_entry_surfaces(self):
        rec = _mkrec()
        now = 1_000_000.0
        # 400s old — past the 300s default TTL.
        strategy._QUANT_CACHE["AMD"] = (rec, now - 400.0)

        snap = build_quant_cache_status(
            strategy, requested=["AMD"], now=now
        )
        # STALE alone is not DEGRADED — only DARK trips DEGRADED.
        assert snap["verdict"] == VERDICT_HEALTHY
        assert snap["n_stale"] == 1
        row = snap["rows"][0]
        assert row["status"] == "STALE"
        assert row["age_s"] == 400
        assert row["ttl_remaining_s"] == 0


# ─────────────────────────── DARK path ──────────────────────────────────


class TestDarkPath:
    """A negative-cache entry younger than _QUANT_NEG_TTL must surface as
    DARK — yfinance is being suppressed this window. DARK always wins
    over STALE because the next cycle will skip the lookup entirely."""

    def test_dark_entry_surfaces_with_ttl_countdown(self):
        now = 2_000_000.0
        # 90s into a 300s neg TTL window → 210s remaining.
        strategy._QUANT_NEG_CACHE["ZOMBIE"] = now - 90.0

        snap = build_quant_cache_status(
            strategy, requested=["ZOMBIE"], now=now
        )
        # Dark feed → DEGRADED verdict.
        assert snap["verdict"] == VERDICT_DEGRADED
        assert snap["n_dark"] == 1
        row = snap["rows"][0]
        assert row["status"] == "DARK"
        assert row["neg_age_s"] == 90
        assert row["neg_ttl_remaining_s"] == 210
        # DARK rows omit the `signals` block — we have no fresh values.
        assert "signals" not in row

    def test_dark_wins_over_stale_when_both_present(self):
        """If a ticker has BOTH a stale positive entry AND a fresh negative
        entry, DARK must win (the next cycle will skip yfinance and the
        positive entry won't refresh — surfacing it would mislead)."""
        now = 3_000_000.0
        strategy._QUANT_CACHE["WEIRD"] = (_mkrec(), now - 400.0)  # stale pos
        strategy._QUANT_NEG_CACHE["WEIRD"] = now - 30.0           # fresh neg

        snap = build_quant_cache_status(
            strategy, requested=["WEIRD"], now=now
        )
        assert snap["verdict"] == VERDICT_DEGRADED
        row = snap["rows"][0]
        assert row["status"] == "DARK"
        # n_stale must NOT count this ticker — it was claimed by DARK.
        assert snap["n_stale"] == 0
        assert snap["n_dark"] == 1


# ───────────────────────── mixed roll-up ────────────────────────────────


class TestMixedRollup:
    """A realistic mix: some fresh, some stale, some dark, some never."""

    def test_mixed_counts_match_expected(self):
        now = 5_000_000.0
        strategy._QUANT_CACHE["NVDA"] = (_mkrec(rsi=55.0), now - 60.0)
        strategy._QUANT_CACHE["AMD"] = (_mkrec(rsi=42.0), now - 350.0)   # stale
        strategy._QUANT_CACHE["TSM"] = (_mkrec(rsi=68.0), now - 10.0)
        strategy._QUANT_NEG_CACHE["MUU"] = now - 10.0                    # dark

        snap = build_quant_cache_status(
            strategy,
            requested=["NVDA", "AMD", "TSM", "MUU", "GOOGU"],
            now=now,
        )
        assert snap["n_total"] == 5
        assert snap["n_fresh"] == 2   # NVDA, TSM
        assert snap["n_stale"] == 1   # AMD
        assert snap["n_dark"] == 1    # MUU
        assert snap["n_never"] == 1   # GOOGU (no entry anywhere)
        assert snap["verdict"] == VERDICT_DEGRADED  # MUU is dark

    def test_no_requested_filter_returns_everything_in_either_cache(self):
        now = 6_000_000.0
        strategy._QUANT_CACHE["A"] = (_mkrec(), now - 10.0)
        strategy._QUANT_NEG_CACHE["B"] = now - 10.0

        snap = build_quant_cache_status(strategy, requested=None, now=now)
        tickers = {r["ticker"] for r in snap["rows"]}
        assert tickers == {"A", "B"}
        assert snap["n_total"] == 2
        assert snap["verdict"] == VERDICT_DEGRADED  # B is dark


# ───────────────────────── degrade-safe paths ──────────────────────────


class TestDegradeSafe:
    """The builder must never raise: caches without the expected attributes,
    a malformed record, a clock step-back — all degrade to a valid envelope."""

    def test_missing_caches_module_state_degrades_cleanly(self):
        # Pass a stand-in object with no _QUANT_CACHE / _QUANT_NEG_CACHE.
        fake = types.SimpleNamespace()
        snap = build_quant_cache_status(fake, requested=None)
        # Empty state envelope, no raise.
        assert snap["verdict"] == VERDICT_EMPTY
        assert snap["n_total"] == 0

    def test_malformed_record_does_not_raise(self):
        now = 7_000_000.0
        # Record is a non-dict — the builder must NOT crash on field access.
        strategy._QUANT_CACHE["BROKEN"] = ("not-a-dict", now - 5.0)
        snap = build_quant_cache_status(
            strategy, requested=["BROKEN"], now=now
        )
        row = snap["rows"][0]
        assert row["status"] == "FRESH"  # ts is still valid → counts as fresh
        # signals MUST be present but empty (no fields could be extracted).
        assert row["signals"] == {}

    def test_clock_step_back_clamps_age_to_zero(self):
        now = 8_000_000.0
        # Stamped 60s in the FUTURE (a wall-clock NTP correction the rest
        # of the runner already hardens against — alarm_latch_state pattern).
        strategy._QUANT_CACHE["FUTURE"] = (_mkrec(), now + 60.0)
        snap = build_quant_cache_status(
            strategy, requested=["FUTURE"], now=now
        )
        row = snap["rows"][0]
        # Clamps to 0 instead of -60 — never a negative age token.
        assert row["age_s"] == 0
        assert row["status"] == "FRESH"


# ─────────────────────── requested-input hygiene ───────────────────────


class TestRequestedInputHygiene:
    """The builder accepts an arbitrary list (the endpoint splits on commas);
    blanks, dupes, lowercase, non-strings must not pollute the output."""

    def test_strips_blanks_and_dedupes(self):
        snap = build_quant_cache_status(
            strategy,
            requested=["", "NVDA", " ", "NVDA", "nvda"],  # blanks + dupes
            now=time.time(),
        )
        # After normalization → exactly one row for NVDA (status NEVER).
        assert len(snap["rows"]) == 1
        assert snap["rows"][0]["ticker"] == "NVDA"
        assert snap["rows"][0]["status"] == "NEVER"

    def test_non_string_entries_are_skipped(self):
        snap = build_quant_cache_status(
            strategy,
            requested=["NVDA", 42, None, "AMD"],  # type: ignore[list-item]
            now=time.time(),
        )
        tickers = [r["ticker"] for r in snap["rows"]]
        assert tickers == ["NVDA", "AMD"]


# ───────────────────────── dashboard endpoint ──────────────────────────


class TestEndpointWiring:
    """The `/api/quant-cache-status` endpoint must compose the builder
    verbatim and degrade safely. Uses the Flask test client (no live
    server hop)."""

    def _client(self):
        from paper_trader.dashboard import app
        return app.test_client()

    def test_endpoint_returns_builder_envelope(self):
        now = 9_000_000.0
        strategy._QUANT_CACHE["NVDA"] = (_mkrec(rsi=55.0), time.time() - 30.0)

        client = self._client()
        rv = client.get("/api/quant-cache-status?tickers=NVDA")
        assert rv.status_code == 200
        body = rv.get_json()
        assert body["service"] == "paper_trader"
        assert body["verdict"] == VERDICT_HEALTHY
        assert body["n_fresh"] == 1
        assert body["rows"][0]["ticker"] == "NVDA"

    def test_endpoint_with_empty_filter_returns_everything(self):
        strategy._QUANT_CACHE["A"] = (_mkrec(), time.time() - 1.0)
        client = self._client()
        rv = client.get("/api/quant-cache-status")
        assert rv.status_code == 200
        body = rv.get_json()
        # No ?tickers= → unfiltered; A must be present.
        assert any(r["ticker"] == "A" for r in body["rows"])

    def test_endpoint_with_dark_ticker_reports_degraded(self):
        strategy._QUANT_NEG_CACHE["ZOMBIE"] = time.time() - 5.0
        client = self._client()
        rv = client.get("/api/quant-cache-status?tickers=ZOMBIE")
        assert rv.status_code == 200
        body = rv.get_json()
        assert body["verdict"] == VERDICT_DEGRADED
        assert body["n_dark"] == 1
        assert body["rows"][0]["status"] == "DARK"
