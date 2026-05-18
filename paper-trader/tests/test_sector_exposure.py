"""Tests for analytics.sector_exposure — the live-book sector-concentration
awareness fed into the live Opus decision prompt (the prompt-facing
complement to the dashboard-only /api/analytics sector breakdown).

Every assertion pins a *specific* expected value so a wrong comparison
operator, a drifted sector map, a broken HHI, or a parity break with the
dashboard fails loudly. The discriminating locks:

* SECTOR_MAP / classify are byte-identical to dashboard's (the single
  source of truth — a duplicated map that silently drifts would make
  /api/sector-exposure and /api/analytics disagree);
* SECTOR_HEAVY_PCT is pinned to game_plan._SECTOR_HEAVY_PCT (the prompt
  "heavy" flag and the dashboard game-plan card must agree);
* every WATCHLIST ticker classifies to a real sector (a future watchlist
  add that forgets a SECTOR_MAP entry must fail here, not silently become
  "% other");
* parity: the builder's sector_pct equals an independent recompute of
  analytics_api's exact formula on the same snapshot;
* hand-computed exposure %, top-sector pick, and sector-HHI on a known
  book; the option ×100 path; deterministic tie-break;
* the marginal in-play view flags an add to the heaviest sector and tags a
  0%-sector name "diversifying";
* CONCENTRATED/DIVERSIFIED flips exactly at the 60.0% boundary (>=);
* the block is observational: autonomy preamble, no imperative trade verb
  (the risk_mirror / buying_power #2/#12 contract);
* _build_payload renders it after the risk-mirror block; None renders no
  stray text;
* it never raises on garbage (the _safe contract — a diagnostics fault
  must not sink a live decision cycle).
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics import sector_exposure as se
from paper_trader.analytics.sector_exposure import (build_sector_exposure,
                                                    classify,
                                                    SECTOR_HEAVY_PCT,
                                                    SECTOR_MAP)


def _snap(cash, total, positions=None):
    return {"cash": cash, "total_value": total, "positions": positions or []}


def _pos(ticker, qty, avg, cur, type_="stock"):
    return {"ticker": ticker, "type": type_, "qty": qty,
            "avg_cost": avg, "current_price": cur}


# ───────────────────── single-source-of-truth drift locks ───────────────

class TestDriftLocks:
    def test_sector_map_matches_dashboard(self):
        """The duplicated SECTOR_MAP MUST be byte-identical to the canonical
        dashboard one — the whole parity guarantee rests on this."""
        from paper_trader import dashboard
        assert SECTOR_MAP == dashboard.SECTOR_MAP

    def test_classify_matches_dashboard_classify(self):
        from paper_trader import dashboard
        for t in ["MU", "soxl", "TQQQ", "LITE", "ZZZZ", "", "spy"]:
            assert classify(t) == dashboard._classify(t)

    def test_heavy_threshold_pinned_to_game_plan(self):
        """The in-prompt 'heavy sector' flag and the dashboard game-plan card
        must use the same number, or they contradict each other."""
        from paper_trader.analytics import game_plan
        assert SECTOR_HEAVY_PCT == game_plan._SECTOR_HEAVY_PCT == 60.0

    def test_every_watchlist_ticker_is_classified(self):
        """A future WATCHLIST addition that forgot a SECTOR_MAP entry would
        silently fold into 'other' and read as a fake sector — fail instead."""
        from paper_trader.strategy import WATCHLIST
        unmapped = [t for t in WATCHLIST if classify(t) == "other"]
        assert unmapped == [], f"WATCHLIST tickers missing a sector: {unmapped}"


# ─────────────────────────── exact exposure math ────────────────────────

class TestExposureMath:
    def test_hand_computed_book(self):
        """$1000 book: MU(semis) 500, SOXL(semis_lev) 300, LITE(optical) 100,
        cash 100. Pin every derived number."""
        snap = _snap(100.0, 1000.0, [
            _pos("MU", 5, 90, 100),      # 500 semis
            _pos("SOXL", 10, 25, 30),    # 300 semis_lev
            _pos("LITE", 2, 40, 50),     # 100 optical
        ])
        r = build_sector_exposure(snap, set())
        assert r["state"] == "DIVERSIFIED"          # top 50% < 60
        assert r["sector_usd"] == {"semis": 500.0, "semis_lev": 300.0,
                                   "optical": 100.0}
        assert r["sector_pct"] == {"semis": 50.0, "semis_lev": 30.0,
                                   "optical": 10.0}
        assert r["top_sector"] == "semis"
        assert r["top_sector_pct"] == 50.0
        assert r["n_sectors"] == 3
        assert r["cash_pct"] == 10.0
        # HHI over invested weights (5/9, 3/9, 1/9): 0.5556^2+0.3333^2+0.1111^2
        assert r["hhi"] == 0.4321

    def test_option_x100_multiplier(self):
        """A held call contributes price*qty*100 to its sector USD — the
        analytics_api formula. Without ×100 this reads 100x too small."""
        snap = _snap(0.0, 700.0, [
            _pos("NVDA", 1, 5.0, 7.0, type_="call"),  # 7*1*100 = 700 semis
        ])
        r = build_sector_exposure(snap, set())
        assert r["sector_usd"] == {"semis": 700.0}
        assert r["sector_pct"] == {"semis": 100.0}
        assert r["hhi"] == 1.0

    def test_avg_cost_fallback_when_no_mark(self):
        """current_price missing → analytics_api falls back to avg_cost."""
        snap = _snap(0.0, 200.0, [
            {"ticker": "MU", "type": "stock", "qty": 2,
             "avg_cost": 100.0, "current_price": None},
        ])
        r = build_sector_exposure(snap, set())
        assert r["sector_usd"] == {"semis": 200.0}

    def test_concentrated_state_at_threshold(self):
        """CONCENTRATED flips exactly at 60.0 (>=), not 60.01."""
        snap = _snap(0.0, 1000.0, [
            _pos("MU", 6, 90, 100),   # 600 semis = exactly 60.0%
            _pos("LITE", 8, 40, 50),  # 400 optical
        ])
        r = build_sector_exposure(snap, set())
        assert r["top_sector_pct"] == 60.0
        assert r["state"] == "CONCENTRATED"
        assert "past the 60% heavy mark" in r["prompt_block"]

        snap2 = _snap(0.0, 1000.0, [
            {"ticker": "MU", "type": "stock", "qty": 1, "avg_cost": 0,
             "current_price": 599.0},   # 59.9%
            {"ticker": "LITE", "type": "stock", "qty": 1, "avg_cost": 0,
             "current_price": 401.0},
        ])
        assert build_sector_exposure(snap2, set())["state"] == "DIVERSIFIED"

    def test_deterministic_tiebreak(self):
        """Two sectors at an identical % → top is the alphabetically-first,
        and the breakdown line leads with the same one (no disagreement)."""
        snap = _snap(0.0, 1000.0, [
            _pos("MU", 5, 90, 100),     # 500 semis
            _pos("SOXL", 5, 90, 100),   # 500 semis_lev
        ])
        r = build_sector_exposure(snap, set())
        assert r["sector_pct"] == {"semis": 50.0, "semis_lev": 50.0}
        assert r["top_sector"] == "semis"           # 'semis' < 'semis_lev'
        assert "Breakdown: SEMIS 50.0% · SEMIS_LEV 50.0%" in r["prompt_block"]


# ───────────────── parity with analytics_api (the SSoT) ─────────────────

class TestAnalyticsParity:
    def _analytics_formula(self, positions, total):
        """Independent re-implementation of analytics_api's exact sector
        math. If the builder ever diverges from this, the dashboard's
        /api/analytics and the in-prompt block would disagree."""
        from paper_trader import dashboard
        sector_usd = {}
        for p in positions:
            mult = 100 if p["type"] in ("call", "put") else 1
            price = p.get("current_price") or p["avg_cost"]
            val = price * p["qty"] * mult
            sec = dashboard._classify(p["ticker"])
            sector_usd[sec] = sector_usd.get(sec, 0.0) + val
        return {s: round(v / total * 100, 2) for s, v in sector_usd.items()}

    def test_sector_pct_matches_analytics_api(self):
        positions = [
            _pos("MU", 3, 80, 95),
            _pos("SOXL", 7, 20, 31),
            _pos("NVDA", 1, 4, 6, type_="call"),
            _pos("LITE", 4, 45, 48),
        ]
        total = 1234.56
        snap = _snap(0.0, total, positions)
        r = build_sector_exposure(snap, set())
        assert r["sector_pct"] == self._analytics_formula(positions, total)


# ─────────────────────── marginal in-play view ──────────────────────────

class TestMarginalView:
    def test_in_play_flags_heaviest_sector(self):
        """An in-play name whose sector is the book's heaviest is flagged
        heavy and tagged 'your heaviest sector'; a 0%-sector name is
        'diversifying'."""
        snap = _snap(0.0, 1000.0, [
            _pos("MU", 7, 90, 100),     # 700 semis (heaviest, > 60)
            _pos("LITE", 6, 40, 50),    # 300 optical
        ])
        r = build_sector_exposure(snap, {"NVDA", "TQQQ", "LITE"})
        by_tk = {m["ticker"]: m for m in r["in_play"]}
        # NVDA → semis (the 70% heaviest sector)
        assert by_tk["NVDA"]["sector"] == "semis"
        assert by_tk["NVDA"]["sector_pct"] == 70.0
        assert by_tk["NVDA"]["heavy"] is True
        # TQQQ → broad_lev, not held at all → 0%, diversifying
        assert by_tk["TQQQ"]["sector"] == "broad_lev"
        assert by_tk["TQQQ"]["sector_pct"] == 0.0
        assert by_tk["TQQQ"]["heavy"] is False
        block = r["prompt_block"]
        assert "NVDA→SEMIS (70.0% — your heaviest sector)" in block
        assert "TQQQ→BROAD_LEV (0.0% — diversifying)" in block

    def test_in_play_sorted_riskiest_first(self):
        snap = _snap(0.0, 1000.0, [
            _pos("MU", 6, 90, 100),     # 600 semis
            _pos("SOXL", 8, 25, 30),    # 240 semis_lev
            _pos("LITE", 32, 5, 5),     # 160 optical
        ])
        r = build_sector_exposure(snap, {"LITE", "MU", "SOXL"})
        order = [m["ticker"] for m in r["in_play"]]
        assert order == ["MU", "SOXL", "LITE"]  # 60 > 24 > 16


# ─────────────────────── degenerate / no-data ───────────────────────────

class TestNoData:
    def test_all_cash_book(self):
        r = build_sector_exposure(_snap(1000.0, 1000.0, []), {"MU"})
        assert r["state"] == "NO_DATA"
        assert r["sector_pct"] == {}
        assert r["top_sector"] is None
        assert "no priced positions this cycle" in r["prompt_block"]

    def test_zero_total_value(self):
        r = build_sector_exposure(_snap(0.0, 0.0, [_pos("MU", 1, 10, 10)]),
                                  set())
        assert r["state"] == "NO_DATA"

    def test_none_snapshot(self):
        r = build_sector_exposure(None, None)
        assert r["state"] == "NO_DATA"
        assert r["prompt_block"]  # honest line, not an empty string


# ─────────────────────── _safe contract (never raises) ──────────────────

class TestNeverRaises:
    def test_garbage_positions_do_not_raise(self):
        snap = {"cash": "?", "total_value": 1000.0,
                "positions": [None, {"ticker": None, "type": "stock"},
                              {"ticker": "MU", "type": "stock",
                               "qty": "x", "avg_cost": "y",
                               "current_price": "z"}]}
        r = build_sector_exposure(snap, {None, 123, "MU"})
        # Either NO_DATA (everything coerced to 0) or a typed marker — never
        # an exception, and always a usable prompt_block.
        assert r["state"] in ("NO_DATA", "DIVERSIFIED", "CONCENTRATED",
                              "ERROR")
        assert isinstance(r["prompt_block"], str) and r["prompt_block"]

    def test_classify_never_raises(self):
        assert classify(None) == "other"
        assert classify(12345) == "other"


# ─────────────────────── observational voice ────────────────────────────

class TestObservationalVoice:
    def test_preamble_disclaims_directive_and_keeps_autonomy(self):
        r = build_sector_exposure(_snap(0.0, 1000.0,
                                        [_pos("MU", 6, 90, 100),
                                         _pos("LITE", 8, 40, 50)]), {"MU"})
        block = r["prompt_block"]
        assert "NOT a directive or limit" in block
        assert "you retain complete autonomy" in block

    def test_no_imperative_trade_verb(self):
        """Observational only (invariants #2/#12): the block must never tell
        Opus to trade. No imperative BUY/SELL/TRIM/AVOID/REDUCE/MUST."""
        r = build_sector_exposure(_snap(0.0, 1000.0,
                                        [_pos("MU", 9, 90, 100),
                                         _pos("LITE", 1, 40, 50)]),
                                  {"MU", "NVDA"})
        block = r["prompt_block"]
        for verb in (" MUST ", "SHOULD ", "DO NOT", "AVOID ", "REDUCE ",
                     "TRIM ", "SELL ", "BUY "):
            assert verb not in block.upper(), f"directive leaked: {verb!r}"


# ─────────────────────── _build_payload wiring ──────────────────────────

class TestPayloadWiring:
    def _base_kwargs(self):
        snap = {"cash": 100.0, "open_value": 0.0, "total_value": 100.0,
                "positions": []}
        return snap

    def test_block_rendered_when_present(self):
        from paper_trader.strategy import _build_payload
        snap = self._base_kwargs()
        out = _build_payload(snap, [], [], {}, {}, None, True,
                             quant_signals={},
                             sector_exposure_block="SECTOR_MARKER_XYZ")
        assert "SECTOR_MARKER_XYZ" in out

    def test_none_renders_no_stray_text(self):
        from paper_trader.strategy import _build_payload
        snap = self._base_kwargs()
        out = _build_payload(snap, [], [], {}, {}, None, True,
                             quant_signals={},
                             sector_exposure_block=None)
        assert "SECTOR EXPOSURE" not in out

    def test_rendered_after_risk_mirror(self):
        """Placement contract: the structural sector view sits with the
        risk-mirror (name concentration), before the forward event block."""
        from paper_trader.strategy import _build_payload
        snap = self._base_kwargs()
        out = _build_payload(snap, [], [], {}, {}, None, True,
                             quant_signals={},
                             risk_mirror_block="RISK_MARKER",
                             sector_exposure_block="SECTOR_MARKER",
                             event_calendar_block="EVENT_MARKER")
        assert (out.index("RISK_MARKER") < out.index("SECTOR_MARKER")
                < out.index("EVENT_MARKER"))
