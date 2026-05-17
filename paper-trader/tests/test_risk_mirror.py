"""Tests for analytics/risk_mirror.py — the concentration + churn mirror fed
back into the live Opus decision prompt.

This block is the third advisory mirror (after self-review and track-record),
targeting the live failure observed 2026-05-17: a book that is 60%+ in one
sector with a 0.52-day median hold and a 16.7% win rate — i.e. *concentrated*
and *churning*. The system already computes both pathologies
(``build_correlation`` / ``build_churn``) but the decision engine never saw
them.

Single source of truth (AGENTS.md #10): the block composes the two existing
pure builders **verbatim** and never re-derives a number. A re-derived metric,
a drifted headline, the correlation builder's "verdict withheld" sentence
leaking into the prompt when only weight-based concentration is available, or a
builder fault sinking the whole mirror (and a live trading cycle) all fail an
assertion here.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.churn import build_churn
from paper_trader.analytics.correlation import build_correlation
from paper_trader.analytics import risk_mirror
from paper_trader.analytics.risk_mirror import build_risk_mirror

_BASE = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
_NOW = _BASE + timedelta(days=200)


def _ts(day_offset: float) -> str:
    return (_BASE + timedelta(days=day_offset)).isoformat()


def _pair(tid, ticker, buy_day, sell_day, qty=1, bpx=100.0, spx=100.0):
    """One BUY+SELL that build_round_trips folds into one round-trip."""
    return [
        {"id": tid, "timestamp": _ts(buy_day), "ticker": ticker,
         "action": "BUY", "qty": qty, "price": bpx, "value": qty * bpx,
         "strike": None, "expiry": None, "option_type": None},
        {"id": tid + 1, "timestamp": _ts(sell_day), "ticker": ticker,
         "action": "SELL", "qty": qty, "price": spx, "value": qty * spx,
         "strike": None, "expiry": None, "option_type": None},
    ]


def _churning_ledger_oldest_first() -> list[dict]:
    """10 names each traded twice back-to-back → 20 round-trips, 50%
    re-entry → CHURNING (the exact shape locked by
    test_churn.py::test_churning_via_reentry_rate)."""
    trades: list[dict] = []
    tid = 1
    block = 0.0
    for i in range(10):
        tk = f"N{i}"
        trades += _pair(tid, tk, block, block + 1)
        trades += _pair(tid + 2, tk, block + 2, block + 3)
        tid += 4
        block += 10
    return trades


# A 61/39 two-name stock book — top weight 0.61 > DOMINANT_WEIGHT (0.60).
_CONCENTRATED_POSITIONS = [
    {"ticker": "LITE", "market_value": 610.0, "type": "stock"},
    {"ticker": "MU", "market_value": 390.0, "type": "stock"},
]


class TestComposition:
    def test_churning_concentrated_book_composes_both_signals(self):
        oldest_first = _churning_ledger_oldest_first()
        store_native = list(reversed(oldest_first))  # newest-first, as Store gives

        out = build_risk_mirror(store_native, _CONCENTRATED_POSITIONS,
                                price_history={}, now=_NOW)
        block = out["prompt_block"]

        # Preamble first, reaffirming autonomy (the self-review precedent).
        assert block.startswith(risk_mirror._PREAMBLE)
        assert "autonomy" in risk_mirror._PREAMBLE.lower()

        # Churn signal: the build_churn headline appears VERBATIM (single
        # source of truth — no re-derived turnover numbers).
        ch = build_churn(oldest_first, now=_NOW)
        assert ch["verdict"] == "CHURNING"          # fixture sanity
        assert ch["headline"] in block

        # Concentration signal from weights alone (no price history): the
        # dominant name, its weight %, effective-by-weight and HHI must be
        # surfaced — this is the whole point of the block.
        assert "LITE" in block
        assert "61" in block                        # 610/1000 = 61%
        assert "effective" in block.lower()
        assert "HHI" in block or "hhi" in block.lower()

        # Summary one-liner carries both verdicts/states for logs+Discord.
        assert "CHURNING" in out["summary"]
        assert out["churn"]["verdict"] == "CHURNING"
        assert "as_of" in out

    def test_no_verdict_withheld_sentence_leaks_when_only_weights(self):
        """Regression lock: with empty price_history build_correlation's
        headline is the bare 'correlation verdict withheld' sentence, which
        BURIES the concentration signal. The mirror must surface the
        weight-based concentration instead and must NOT paste that sentence
        into the live prompt."""
        out = build_risk_mirror([], _CONCENTRATED_POSITIONS,
                                price_history={}, now=_NOW)
        block = out["prompt_block"]

        co = build_correlation(_CONCENTRATED_POSITIONS, {}, now=_NOW)
        assert co["state"] == "INSUFFICIENT"        # fixture sanity
        assert "withheld" in (co["headline"] or "")  # the buried-signal source

        assert "withheld" not in block
        assert "LITE" in block and "61" in block

    def test_correlation_headline_used_verbatim_when_price_history_present(self):
        """When real price history is available the rich ρ headline (the
        'book moves as one' story) is used verbatim — NOT the weight-only
        fallback. Discriminates the two concentration paths."""
        # A balanced 55/45 book so top weight (0.55) stays below
        # DOMINANT_WEIGHT (0.60) and the verdict is unambiguously
        # CONCENTRATED rather than SINGLE_NAME_RISK. Two perfectly-correlated
        # 12-close series (≥ MIN_RETURNS=10 returns): B = 2*A is an affine
        # map ⇒ Pearson ρ = +1.0 ⇒ CONCENTRATED.
        balanced = [
            {"ticker": "LITE", "market_value": 550.0, "type": "stock"},
            {"ticker": "MU", "market_value": 450.0, "type": "stock"},
        ]
        a = [float(x) for x in range(1, 13)]
        ph = {"LITE": a, "MU": [2.0 * x for x in a]}

        out = build_risk_mirror([], balanced, price_history=ph, now=_NOW)
        block = out["prompt_block"]

        co = build_correlation(balanced, ph, now=_NOW)
        assert co["state"] == "OK"                  # fixture sanity
        assert co["verdict"] == "CONCENTRATED"
        assert co["headline"] in block              # verbatim, not re-derived
        assert "moves as one" in block
        # Proves the rich ρ path was taken, NOT the weight-pending fallback.
        assert "pending" not in block

    def test_single_source_of_truth_churn_headline_byte_identical(self):
        """The embedded churn headline must equal build_churn's own headline
        exactly — an inline re-implementation would drift from /api/churn."""
        oldest_first = _churning_ledger_oldest_first()
        out = build_risk_mirror(list(reversed(oldest_first)), [],
                                price_history={}, now=_NOW)
        expected = build_churn(oldest_first, now=_NOW)["headline"]
        assert expected and expected in out["prompt_block"]


class TestDegrade:
    def test_empty_book_is_honest_short_line_not_empty(self):
        out = build_risk_mirror([], [], price_history={}, now=_NOW)
        block = out["prompt_block"]
        assert block.startswith(risk_mirror._PREAMBLE)
        # An honest one-liner beats an empty section or a None the caller
        # must special-case (the self-review precedent).
        assert "nothing to mirror" in block.lower()
        assert out["summary"]  # never empty

    def test_no_stock_positions_omits_concentration_line(self):
        """An options-only / cash book has undefined concentration — the
        line is omitted, not faked."""
        opts = [{"ticker": "NVDA", "market_value": 500.0, "type": "call"}]
        out = build_risk_mirror(list(reversed(_churning_ledger_oldest_first())),
                                opts, price_history={}, now=_NOW)
        assert "Concentration" not in out["prompt_block"]
        assert "Turnover" in out["prompt_block"]  # churn still shown

    def test_builder_fault_is_non_fatal_never_sinks_the_cycle(self, monkeypatch):
        """A single bad builder must degrade to 'that line missing', never an
        exception — the live decision cycle must survive (self-review
        _safe contract)."""
        def boom(*a, **k):
            raise RuntimeError("simulated churn builder fault")

        monkeypatch.setattr(risk_mirror, "build_churn", boom)
        out = build_risk_mirror(
            list(reversed(_churning_ledger_oldest_first())),
            _CONCENTRATED_POSITIONS, price_history={}, now=_NOW)
        # No raise; concentration line still present; churn line gone.
        assert isinstance(out, dict)
        assert "LITE" in out["prompt_block"]
        assert "Turnover" not in out["prompt_block"]


class TestPayloadIntegration:
    def test_build_payload_renders_risk_mirror_after_track_record(self):
        from paper_trader import strategy

        snap = {"positions": [], "cash": 1000.0,
                "open_value": 0.0, "total_value": 1000.0}
        marker = "RISK-MIRROR-BLOCK-MARKER concentration+churn"
        payload = strategy._build_payload(
            snap, [], [], {}, {}, None, True,
            quant_signals={},
            self_review_block=None,
            track_record_block="TRACK-RECORD-MARKER",
            risk_mirror_block=marker,
        )
        assert marker in payload
        # Behavioural mirrors sit before market data biases the trader.
        assert payload.index("TRACK-RECORD-MARKER") < payload.index(marker)
        assert payload.index(marker) < payload.index("WATCHLIST PRICES")

    def test_build_payload_none_risk_mirror_renders_no_stray_text(self):
        from paper_trader import strategy

        snap = {"positions": [], "cash": 1000.0,
                "open_value": 0.0, "total_value": 1000.0}
        payload = strategy._build_payload(
            snap, [], [], {}, {}, None, True,
            quant_signals={}, risk_mirror_block=None)
        assert "RISK MIRROR" not in payload
        assert "None" not in payload.split("PORTFOLIO")[1].split("WATCHLIST")[0]
