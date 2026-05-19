"""Tests for analytics.sector_signal_fit — the per-sector position vs.
live-signal-density divergence view exposed at /api/sector-signal-fit.

The discriminating locks:

* sector classification reuses the SSOT ``sector_exposure.classify`` so
  a drifted map would surface here too (covered by
  test_sector_exposure.py — we only assert wiring, not the map itself);
* signal-share denominator is total weighted ai_score (so per-sector
  shares ALWAYS sum to ~100.0 across surfaced sectors — never a fabricated
  fraction that doesn't add up);
* multi-sector signal splits weight evenly across MENTIONED sectors,
  never multiplies (a 4-ticker article all in SEMIS contributes ONCE to
  SEMIS, not 4x — discriminator vs. a naive per-mention sum);
* OVERWEIGHT / UNDERWEIGHT / ALIGNED flip exactly at the configurable
  gap_threshold_pct boundary (defaults to 15.0);
* rows sorted by descending |gap_pct|, ties by sector name — the analyst
  reads worst-divergence first;
* signals with no extracted tickers contribute NOTHING (they're counted
  separately as ``n_signals_with_no_tickers`` for honesty, never absorbed
  as "other-sector noise");
* top-level state is ALIGNED iff EVERY sector is within threshold (the
  threshold itself is configurable);
* pure / total — bad inputs degrade to NO_DATA or empty rows, never raise.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.sector_signal_fit import (  # noqa: E402
    build_sector_signal_fit,
    GAP_THRESHOLD_PCT,
)


def _exposure(sector_pct: dict[str, float]) -> dict:
    """Minimal stand-in for build_sector_exposure output — only sector_pct
    is consumed; carrying extra keys keeps the contract resilient."""
    return {"sector_pct": dict(sector_pct), "state": "DIVERSIFIED",
            "summary": "test fixture"}


def _sig(tickers: list[str], score: float) -> dict:
    return {"title": "t", "tickers": list(tickers), "ai_score": score}


# ── basic shape & arithmetic ──────────────────────────────────────────
class TestBasicShape:
    def test_signal_shares_sum_to_100_when_signals_present(self):
        """The signal_share_pct column MUST sum to ~100.0 across surfaced
        sectors — otherwise the "share of the wire" claim is a lie.
        Discriminator: an off-by-one in the denominator would make this fail."""
        exp = _exposure({})
        sigs = [
            _sig(["NVDA"], 9.0),       # semis
            _sig(["AAPL"], 6.0),       # tech
            _sig(["SPY"], 4.0),        # broad
        ]
        rep = build_sector_signal_fit(exp, sigs)
        assert rep["state"] == "MISALIGNED"  # heavy signal, zero position
        shares = sum(r["signal_share_pct"] for r in rep["sectors"])
        assert shares == pytest.approx(100.0, abs=0.05)

    def test_aligned_when_position_matches_signal_share(self):
        """Position 50/50 SEMIS/TECH; signals also 50/50 → all gaps ≈ 0 →
        top-level ALIGNED."""
        exp = _exposure({"semis": 50.0, "tech": 50.0})
        sigs = [_sig(["NVDA"], 5.0), _sig(["AAPL"], 5.0)]
        rep = build_sector_signal_fit(exp, sigs)
        assert rep["state"] == "ALIGNED"
        for row in rep["sectors"]:
            assert row["verdict"] == "ALIGNED"
            assert abs(row["gap_pct"]) <= GAP_THRESHOLD_PCT

    def test_overweight_when_position_exceeds_signal_share(self):
        """Held 80% SEMIS, but the wire only allocates 20% of its score to
        SEMIS → SEMIS is OVERWEIGHT (you're long where it's quiet)."""
        exp = _exposure({"semis": 80.0, "tech": 20.0})
        sigs = [
            _sig(["NVDA"], 2.0),
            _sig(["AAPL"], 8.0),
        ]
        rep = build_sector_signal_fit(exp, sigs)
        sem = next(r for r in rep["sectors"] if r["sector"] == "semis")
        assert sem["verdict"] == "OVERWEIGHT"
        assert sem["gap_pct"] > GAP_THRESHOLD_PCT
        assert rep["state"] == "MISALIGNED"
        assert rep["max_gap_sector"] == "semis"

    def test_underweight_when_signal_share_exceeds_position(self):
        """Held 5% TECH but wire is 80% TECH-focused → TECH UNDERWEIGHT."""
        exp = _exposure({"semis": 95.0, "tech": 5.0})
        sigs = [
            _sig(["NVDA"], 2.0),
            _sig(["AAPL"], 8.0),
        ]
        rep = build_sector_signal_fit(exp, sigs)
        tech = next(r for r in rep["sectors"] if r["sector"] == "tech")
        assert tech["verdict"] == "UNDERWEIGHT"
        assert tech["gap_pct"] < -GAP_THRESHOLD_PCT


# ── signal weighting subtleties ───────────────────────────────────────
class TestSignalWeighting:
    def test_multi_ticker_single_sector_signal_contributes_once(self):
        """A 4-ticker SEMIS article (NVDA, AMD, MU, AMAT) must contribute
        its ai_score ONCE to SEMIS, not 4×. Discriminator: a naive per-
        mention sum would 4× the SEMIS share and silently distort the gap."""
        exp = _exposure({})
        # One 4-ticker SEMIS article (score 10) + one 1-ticker TECH article (score 10).
        # If multi-ticker double-counted, SEMIS would be 80% (40/50) of weight.
        # With correct one-per-sector accounting, SEMIS == TECH == 50%.
        sigs = [
            _sig(["NVDA", "AMD", "MU", "AMAT"], 10.0),
            _sig(["AAPL"], 10.0),
        ]
        rep = build_sector_signal_fit(exp, sigs)
        sem = next(r for r in rep["sectors"] if r["sector"] == "semis")
        tech = next(r for r in rep["sectors"] if r["sector"] == "tech")
        assert sem["signal_share_pct"] == pytest.approx(50.0, abs=0.1)
        assert tech["signal_share_pct"] == pytest.approx(50.0, abs=0.1)

    def test_cross_sector_signal_splits_weight_evenly(self):
        """A SEMIS+TECH article (NVDA + AAPL, score 10) splits 5 to SEMIS
        and 5 to TECH. Plus a TECH-only article (AAPL, score 5) → TECH
        gets 5+5=10, SEMIS gets 5 → TECH 66.7%, SEMIS 33.3%."""
        exp = _exposure({})
        sigs = [
            _sig(["NVDA", "AAPL"], 10.0),
            _sig(["AAPL"], 5.0),
        ]
        rep = build_sector_signal_fit(exp, sigs)
        shares = {r["sector"]: r["signal_share_pct"] for r in rep["sectors"]}
        assert shares["semis"] == pytest.approx(33.33, abs=0.1)
        assert shares["tech"] == pytest.approx(66.67, abs=0.1)

    def test_signals_with_no_tickers_are_counted_separately(self):
        """A signal with no extracted tickers must NOT silently land in
        'other' — it goes into n_signals_with_no_tickers for honesty."""
        exp = _exposure({})
        sigs = [
            _sig(["NVDA"], 5.0),
            _sig([], 9.0),
            {"title": "no ticker key at all", "ai_score": 7.0},
        ]
        rep = build_sector_signal_fit(exp, sigs)
        assert rep["n_signals_used"] == 1
        assert rep["n_signals_with_no_tickers"] == 2
        # NO 'other' row should appear from the no-ticker signals.
        sectors = {r["sector"] for r in rep["sectors"]}
        assert "other" not in sectors

    def test_zero_or_negative_score_signals_dropped(self):
        """A signal with ai_score=0 contributes nothing to the share
        denominator (it's noise the model wasn't sure was real)."""
        exp = _exposure({})
        sigs = [
            _sig(["NVDA"], 5.0),
            _sig(["AAPL"], 0.0),
            _sig(["MU"], -1.0),
        ]
        rep = build_sector_signal_fit(exp, sigs)
        assert rep["n_signals_used"] == 1
        sectors = {r["sector"] for r in rep["sectors"]}
        assert sectors == {"semis"}

    def test_unknown_ticker_classifies_to_other(self):
        """An unknown ticker (not in SECTOR_MAP) MUST classify to 'other'
        — that's the SSOT classify() contract, and the signal-share
        accounting must respect it. Discriminator: drop-the-row behaviour
        would mask coverage of names the operator hasn't curated yet."""
        exp = _exposure({})
        sigs = [_sig(["UNKNOWNXYZ"], 6.0), _sig(["NVDA"], 6.0)]
        rep = build_sector_signal_fit(exp, sigs)
        other = next(
            (r for r in rep["sectors"] if r["sector"] == "other"), None)
        assert other is not None
        assert other["signal_share_pct"] == pytest.approx(50.0, abs=0.1)


# ── ranking & summary ─────────────────────────────────────────────────
class TestRankingAndSummary:
    def test_rows_sorted_by_absolute_gap_descending(self):
        """The analyst should read worst-divergence first. Discriminator:
        an unsorted output would surface near-zero gaps above the headline
        divergence."""
        exp = _exposure({"semis": 70.0, "tech": 25.0, "broad": 5.0})
        sigs = [
            _sig(["NVDA"], 2.0),
            _sig(["AAPL"], 5.0),
            _sig(["SPY"], 3.0),
        ]
        rep = build_sector_signal_fit(exp, sigs)
        abs_gaps = [abs(r["gap_pct"]) for r in rep["sectors"]]
        assert abs_gaps == sorted(abs_gaps, reverse=True)

    def test_summary_names_the_most_divergent_sector_when_misaligned(self):
        exp = _exposure({"semis": 80.0, "tech": 20.0})
        sigs = [_sig(["AAPL"], 10.0)]
        rep = build_sector_signal_fit(exp, sigs)
        assert rep["state"] == "MISALIGNED"
        # The summary should mention semis (the OVERWEIGHT divergence) AND
        # the direction so it's an answer, not just a flag.
        assert "semis" in rep["summary"].lower()
        assert "overweight" in rep["summary"].lower()

    def test_max_gap_sector_matches_max_gap_pct(self):
        """The reported max_gap_sector / max_gap_pct pair must be consistent
        with the per-sector rows — no drift between the headline and the
        breakdown."""
        exp = _exposure({"semis": 90.0, "tech": 10.0})
        sigs = [_sig(["AAPL"], 10.0)]
        rep = build_sector_signal_fit(exp, sigs)
        owner = next(r for r in rep["sectors"]
                     if r["sector"] == rep["max_gap_sector"])
        assert abs(owner["gap_pct"]) == pytest.approx(rep["max_gap_pct"], abs=0.01)


# ── threshold sensitivity ─────────────────────────────────────────────
class TestThresholdSensitivity:
    def test_custom_gap_threshold_flips_verdict_at_the_boundary(self):
        """A 10-pt gap is ALIGNED at the default 15-pt threshold but
        OVERWEIGHT at a stricter 5-pt threshold. Pins the configurable
        boundary."""
        exp = _exposure({"semis": 60.0, "tech": 40.0})
        sigs = [_sig(["NVDA"], 5.0), _sig(["AAPL"], 5.0)]  # 50/50 shares
        # Default (15): both within ±10 → ALIGNED
        rep_default = build_sector_signal_fit(exp, sigs)
        assert rep_default["state"] == "ALIGNED"
        # Tight (5): both ±10 → MISALIGNED, semis OVERWEIGHT
        rep_tight = build_sector_signal_fit(exp, sigs, gap_threshold_pct=5.0)
        assert rep_tight["state"] == "MISALIGNED"
        sem = next(r for r in rep_tight["sectors"] if r["sector"] == "semis")
        assert sem["verdict"] == "OVERWEIGHT"


# ── degraded inputs ───────────────────────────────────────────────────
class TestDegradedInputs:
    def test_no_positions_and_no_signals_is_no_data(self):
        rep = build_sector_signal_fit(_exposure({}), [])
        assert rep["state"] == "NO_DATA"
        assert rep["sectors"] == []
        assert "undefined" in rep["summary"].lower()

    def test_positions_but_no_signals_is_overweight_everywhere(self):
        """With zero signal weight, every position-bearing sector has a
        100-share gap against 0 → MISALIGNED, all OVERWEIGHT."""
        exp = _exposure({"semis": 100.0})
        rep = build_sector_signal_fit(exp, [])
        assert rep["state"] == "MISALIGNED"
        sem = next(r for r in rep["sectors"] if r["sector"] == "semis")
        assert sem["verdict"] == "OVERWEIGHT"
        assert sem["gap_pct"] == pytest.approx(100.0, abs=0.01)

    def test_signals_but_no_positions_is_underweight_everywhere(self):
        rep = build_sector_signal_fit(_exposure({}), [_sig(["NVDA"], 5.0)])
        assert rep["state"] == "MISALIGNED"
        sem = next(r for r in rep["sectors"] if r["sector"] == "semis")
        assert sem["verdict"] == "UNDERWEIGHT"
        assert sem["gap_pct"] == pytest.approx(-100.0, abs=0.01)

    @pytest.mark.parametrize("bad", [None, "x", 42, [], object()])
    def test_non_dict_exposure_treated_as_empty(self, bad):
        rep = build_sector_signal_fit(bad, [_sig(["NVDA"], 5.0)])
        # No position info → MISALIGNED with single underweight sector.
        assert rep["state"] == "MISALIGNED"
        assert any(r["sector"] == "semis" for r in rep["sectors"])

    @pytest.mark.parametrize("bad", [None, "x", 42, object()])
    def test_non_list_signals_treated_as_empty(self, bad):
        rep = build_sector_signal_fit(_exposure({"semis": 100.0}), bad)
        assert rep["state"] == "MISALIGNED"
        assert rep["n_signals_used"] == 0

    def test_garbage_signal_rows_dropped_silently(self):
        exp = _exposure({})
        sigs = [
            _sig(["NVDA"], 5.0),
            "not a dict",                                       # → no_tickers
            None,                                               # → no_tickers
            {"tickers": "not a list", "ai_score": 9.0},          # → no_tickers
            {"tickers": [None, 42, "NVDA"], "ai_score": "bad"},  # tickers OK, score 0 → dropped
            {"tickers": ["NVDA"], "ai_score": 3.0},
        ]
        rep = build_sector_signal_fit(exp, sigs)
        # Two valid signals (5.0 + 3.0) used; three rows have no tickers
        # (the non-dict, the None, and the "tickers: not a list" entry).
        # The "ai_score: 'bad'" row HAS tickers (extracted as ['NVDA']) but
        # its score coerces to 0 → silently dropped (zero-score policy);
        # since it had a ticker it does NOT count as no_tickers either.
        assert rep["n_signals_used"] == 2
        assert rep["n_signals_with_no_tickers"] == 3
        # Final sanity — the surviving signals all map to semis.
        assert {r["sector"] for r in rep["sectors"]} == {"semis"}


# ── Flask integration via test_client ─────────────────────────────────
class TestEndpoint:
    def test_endpoint_returns_json_shape(self, tmp_path, monkeypatch):
        """End-to-end smoke through the Flask test_client — the route must
        wire build_sector_exposure + build_sector_signal_fit together and
        echo the input knobs."""
        # Isolate the trader store to tmp so this test never touches the
        # live paper_trader.db (the conftest fixture pattern).
        monkeypatch.setenv("PAPER_TRADER_DB", str(tmp_path / "pt.db"))
        from paper_trader import dashboard as dash

        # Monkeypatch signal source so this test doesn't touch articles.db.
        from paper_trader import signals as _sig
        monkeypatch.setattr(
            _sig, "get_top_signals",
            lambda n=20, hours=2, min_score=4.0: [
                {"id": "x", "url": "u", "title": "NVDA blowout",
                 "source": "reuters", "ai_score": 9.0, "urgency": 1,
                 "first_seen": "2026-05-18T10:00:00+00:00",
                 "summary": "...", "tickers": ["NVDA"]},
            ],
        )
        # Reset Store singleton so it picks up the tmp env var.
        try:
            dash._store = None  # type: ignore[attr-defined]
        except AttributeError:
            pass
        client = dash.app.test_client()
        resp = client.get("/api/sector-signal-fit?hours=4&min_score=3.0")
        assert resp.status_code == 200
        body = resp.get_json()
        assert isinstance(body, dict)
        assert body.get("state") in ("ALIGNED", "MISALIGNED", "NO_DATA")
        assert isinstance(body.get("sectors"), list)
        # Knob echoes — the response must be self-describing.
        assert body["window_hours"] == 4
        assert body["min_score"] == pytest.approx(3.0)
        # The injected single signal → semis underweight (no positions).
        assert body["n_signals_input"] == 1
