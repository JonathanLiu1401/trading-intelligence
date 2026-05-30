"""Tests for ``_drought_alpha_bleed_line`` — the new alpha-bleed Discord
surface that composes ``build_decision_drought`` and surfaces BLEEDING /
STUCK verdicts in the hourly summary.

Locks the contract from three angles:

1. Pure helper composition — verdict suppression whitelist (only
   ``BLEEDING`` / ``STUCK`` surface; everything else is silent), the
   builder's ``verdict_reason`` is rendered verbatim (single source of
   truth — invariant #10), prefix differs by verdict severity.
2. Reporter wiring — driven through a fake ``store`` against the live
   ``build_decision_drought`` so a future builder refactor still keeps
   the surface contract.
3. Degrade safety — non-dict / missing / empty / unknown verdicts and a
   store fault all collapse to ``""`` (never raises — the notification
   helper contract).
"""
from __future__ import annotations

from unittest.mock import patch

import pytest


# --------------------------------------------------------------------------
# Helper unit tests — verdict suppression whitelist + render contract.
# --------------------------------------------------------------------------


def _fake_store_with_calls():
    """Minimal store stub: returns empty data so the inner builder is
    called for its real shape, but we still patch the builder return."""

    class _S:
        def __init__(self):
            self.recent_decisions_called_with: int | None = None
            self.equity_curve_called_with: int | None = None

        def recent_decisions(self, *, limit):
            self.recent_decisions_called_with = limit
            return []

        def equity_curve(self, *, limit):
            self.equity_curve_called_with = limit
            return []

    return _S()


class TestDroughtAlphaBleedLineHelper:
    """Verdict-driven render contract — pinned by patching the builder
    return so the helper's branching is exercised directly."""

    def test_bleeding_surfaces_with_blood_prefix(self):
        from paper_trader import reporter

        fake = {
            "verdict": "BLEEDING",
            "verdict_reason": (
                "6.56% of alpha lost across 8 involuntary (parse-failure) "
                "droughts — the NO_DECISION problem is costing real performance"
            ),
        }
        store = _fake_store_with_calls()
        with patch(
            "paper_trader.analytics.decision_drought.build_decision_drought",
            return_value=fake,
        ):
            line = reporter._drought_alpha_bleed_line(store)

        assert line.startswith("🩸")
        assert "**ALPHA BLEED**" in line
        assert "BLEEDING" in line
        assert "6.56% of alpha lost" in line
        # builder's reason rendered verbatim (SSOT — no re-derivation)
        assert fake["verdict_reason"] in line

    def test_stuck_surfaces_with_warning_prefix(self):
        from paper_trader import reporter

        fake = {
            "verdict": "STUCK",
            "verdict_reason": (
                "currently paralyzed for 19.5h (35/46 cycles NO_DECISION)"
            ),
        }
        store = _fake_store_with_calls()
        with patch(
            "paper_trader.analytics.decision_drought.build_decision_drought",
            return_value=fake,
        ):
            line = reporter._drought_alpha_bleed_line(store)

        assert line.startswith("⚠️")
        assert "**ALPHA BLEED**" in line
        assert "STUCK" in line
        assert "paralyzed for 19.5h" in line

    def test_ok_verdict_silent(self):
        from paper_trader import reporter

        fake = {
            "verdict": "OK",
            "verdict_reason": "9 fills across 246 cycles; no material involuntary alpha bleed",
        }
        with patch(
            "paper_trader.analytics.decision_drought.build_decision_drought",
            return_value=fake,
        ):
            line = reporter._drought_alpha_bleed_line(_fake_store_with_calls())
        assert line == ""

    def test_no_data_silent(self):
        from paper_trader import reporter

        fake = {"verdict": "NO_DATA", "verdict_reason": "no decisions recorded yet"}
        with patch(
            "paper_trader.analytics.decision_drought.build_decision_drought",
            return_value=fake,
        ):
            line = reporter._drought_alpha_bleed_line(_fake_store_with_calls())
        assert line == ""

    def test_never_traded_silent(self):
        from paper_trader import reporter

        fake = {
            "verdict": "NEVER_TRADED",
            "verdict_reason": "246 cycles, zero FILLED trades — the bot has never opened or closed a position",
        }
        with patch(
            "paper_trader.analytics.decision_drought.build_decision_drought",
            return_value=fake,
        ):
            line = reporter._drought_alpha_bleed_line(_fake_store_with_calls())
        assert line == ""

    def test_unknown_verdict_silent_whitelist_only(self):
        """Defensive: a future builder addition (e.g. 'WARNING') must NOT
        be surfaced — suppression is a whitelist, not a blacklist, so the
        Discord surface never becomes its own lying green light when the
        builder ladder grows."""
        from paper_trader import reporter

        fake = {
            "verdict": "MILDLY_CONCERNING",
            "verdict_reason": "a verdict the helper has never seen",
        }
        with patch(
            "paper_trader.analytics.decision_drought.build_decision_drought",
            return_value=fake,
        ):
            line = reporter._drought_alpha_bleed_line(_fake_store_with_calls())
        assert line == ""

    def test_missing_verdict_key_silent(self):
        from paper_trader import reporter

        with patch(
            "paper_trader.analytics.decision_drought.build_decision_drought",
            return_value={},
        ):
            line = reporter._drought_alpha_bleed_line(_fake_store_with_calls())
        assert line == ""

    def test_non_dict_result_silent(self):
        from paper_trader import reporter

        with patch(
            "paper_trader.analytics.decision_drought.build_decision_drought",
            return_value=None,
        ):
            line = reporter._drought_alpha_bleed_line(_fake_store_with_calls())
        assert line == ""

    def test_bleeding_with_empty_reason_silent(self):
        from paper_trader import reporter

        fake = {"verdict": "BLEEDING", "verdict_reason": ""}
        with patch(
            "paper_trader.analytics.decision_drought.build_decision_drought",
            return_value=fake,
        ):
            line = reporter._drought_alpha_bleed_line(_fake_store_with_calls())
        assert line == ""

    def test_bleeding_with_whitespace_only_reason_silent(self):
        from paper_trader import reporter

        fake = {"verdict": "BLEEDING", "verdict_reason": "   \n  "}
        with patch(
            "paper_trader.analytics.decision_drought.build_decision_drought",
            return_value=fake,
        ):
            line = reporter._drought_alpha_bleed_line(_fake_store_with_calls())
        assert line == ""

    def test_bleeding_with_non_string_reason_silent(self):
        from paper_trader import reporter

        fake = {"verdict": "BLEEDING", "verdict_reason": 123}
        with patch(
            "paper_trader.analytics.decision_drought.build_decision_drought",
            return_value=fake,
        ):
            line = reporter._drought_alpha_bleed_line(_fake_store_with_calls())
        assert line == ""

    def test_store_fault_returns_empty(self):
        from paper_trader import reporter

        class _BadStore:
            def recent_decisions(self, *, limit):
                raise RuntimeError("DB unreachable")

            def equity_curve(self, *, limit):
                raise RuntimeError("DB unreachable")

        line = reporter._drought_alpha_bleed_line(_BadStore())
        assert line == ""

    def test_builder_fault_returns_empty(self):
        from paper_trader import reporter

        store = _fake_store_with_calls()
        with patch(
            "paper_trader.analytics.decision_drought.build_decision_drought",
            side_effect=ValueError("malformed shape"),
        ):
            line = reporter._drought_alpha_bleed_line(store)
        assert line == ""

    def test_calls_store_with_documented_limits(self):
        """Pin the read limits so a future cap doesn't silently narrow
        the surface (3000 decisions, 5000 equity points — matches
        ``_idle_opportunity_line`` which composes the same builder)."""
        from paper_trader import reporter

        store = _fake_store_with_calls()
        with patch(
            "paper_trader.analytics.decision_drought.build_decision_drought",
            return_value={"verdict": "OK", "verdict_reason": "fine"},
        ):
            reporter._drought_alpha_bleed_line(store)
        assert store.recent_decisions_called_with == 3000
        assert store.equity_curve_called_with == 5000


# --------------------------------------------------------------------------
# Reporter wiring — drive the helper through the real
# ``build_decision_drought`` so a future builder refactor still keeps the
# contract. Hand-built decision rows mirror the store's ``action_taken``
# free-text shape.
# --------------------------------------------------------------------------


class TestReporterWiringWithRealBuilder:
    """Lock the helper end-to-end against the real builder — no patching,
    so the suppression-whitelist contract holds against the live verdict
    ladder."""

    def _decision(self, ts, action_taken):
        return {
            "timestamp": ts,
            "market_open": 1,
            "signal_count": 0,
            "action_taken": action_taken,
            "reasoning": "",
            "portfolio_value": 1000.0,
            "cash": 1000.0,
        }

    def _equity(self, ts, total_value, sp500):
        return {
            "timestamp": ts,
            "total_value": total_value,
            "cash": total_value,
            "sp500_price": sp500,
        }

    def _store(self, decisions_newest_first, equity_asc):
        decs = list(decisions_newest_first)
        eq = list(equity_asc)

        class _S:
            def recent_decisions(self, *, limit):
                return decs[:limit]

            def equity_curve(self, *, limit):
                return eq[-limit:]

        return _S()

    def test_real_bleeding_surfaces(self):
        """Walk an ongoing PARALYSIS drought BIG enough that the
        aggregate involuntary_alpha_bleed_pct trips the BLEEDING
        threshold (≤ -1.0%). Two closed PARALYSIS droughts, each
        losing ~5% alpha vs SPY, summed to ~-10% bleed."""
        from paper_trader import reporter

        # First PARALYSIS drought (closed): 10 NO_DECISION cycles, then
        # a FILLED — port -5%, SPY 0% over the drought window.
        decisions: list[dict] = []
        equity: list[dict] = []
        # Drought 1: 2026-01-01T00..09 NO_DECISION, then 10:00 FILLED.
        for h in range(10):
            ts = f"2026-01-01T{h:02d}:00:00+00:00"
            decisions.append(self._decision(ts, "NO_DECISION"))
            # port slides 0% → -5% linearly; SPY flat at 5000.
            tv = 1000.0 * (1.0 - 0.005 * h)
            equity.append(self._equity(ts, tv, 5000.0))
        ts = "2026-01-01T10:00:00+00:00"
        decisions.append(self._decision(ts, "BUY NVDA → FILLED"))
        equity.append(self._equity(ts, 950.0, 5000.0))

        # Drought 2: 2026-01-02 same shape.
        for h in range(10):
            ts = f"2026-01-02T{h:02d}:00:00+00:00"
            decisions.append(self._decision(ts, "NO_DECISION"))
            tv = 950.0 * (1.0 - 0.005 * h)
            equity.append(self._equity(ts, tv, 5000.0))
        ts = "2026-01-02T10:00:00+00:00"
        decisions.append(self._decision(ts, "BUY MSFT → FILLED"))
        equity.append(self._equity(ts, 902.5, 5000.0))

        # newest-first for recent_decisions
        decisions_nf = list(reversed(decisions))
        store = self._store(decisions_nf, equity)

        line = reporter._drought_alpha_bleed_line(store)
        assert "BLEEDING" in line
        assert "🩸" in line
        assert "**ALPHA BLEED**" in line

    def test_real_stuck_surfaces(self):
        """Ongoing PARALYSIS drought ≥3h — STUCK verdict. No prior
        droughts so aggregate bleed is 0% (below BLEEDING threshold),
        making STUCK the dominant verdict for the helper to render."""
        from paper_trader import reporter

        decisions: list[dict] = []
        equity: list[dict] = []
        # One FILLED at 0h so the bot has at least one fill ever (else
        # NEVER_TRADED preempts STUCK).
        decisions.append(self._decision(
            "2026-01-01T00:00:00+00:00", "BUY AAPL → FILLED"
        ))
        equity.append(self._equity(
            "2026-01-01T00:00:00+00:00", 1000.0, 5000.0
        ))
        # 5 hours of NO_DECISION — ongoing PARALYSIS drought ≥3h.
        for h in range(1, 6):
            ts = f"2026-01-01T{h:02d}:00:00+00:00"
            decisions.append(self._decision(ts, "NO_DECISION"))
            equity.append(self._equity(ts, 1005.0, 5005.0))

        decisions_nf = list(reversed(decisions))
        store = self._store(decisions_nf, equity)

        line = reporter._drought_alpha_bleed_line(store)
        assert "STUCK" in line
        assert "⚠️" in line
        assert "**ALPHA BLEED**" in line

    def test_real_ok_book_silent(self):
        """Steady stream of FILLED decisions — verdict OK, line silent.
        The hourly must never become its own lying green light."""
        from paper_trader import reporter

        decisions: list[dict] = []
        equity: list[dict] = []
        for h in range(20):
            ts = f"2026-01-01T{h:02d}:00:00+00:00"
            decisions.append(self._decision(ts, "BUY NVDA → FILLED"))
            equity.append(self._equity(ts, 1000.0 + h, 5000.0 + h))

        decisions_nf = list(reversed(decisions))
        store = self._store(decisions_nf, equity)

        line = reporter._drought_alpha_bleed_line(store)
        assert line == ""

    def test_real_never_traded_silent(self):
        """All NO_DECISION cycles, no FILLED ever — verdict
        NEVER_TRADED, line silent (a brand-new bot is not bleeding)."""
        from paper_trader import reporter

        decisions: list[dict] = []
        equity: list[dict] = []
        for h in range(20):
            ts = f"2026-01-01T{h:02d}:00:00+00:00"
            decisions.append(self._decision(ts, "NO_DECISION"))
            equity.append(self._equity(ts, 1000.0, 5000.0))

        decisions_nf = list(reversed(decisions))
        store = self._store(decisions_nf, equity)

        line = reporter._drought_alpha_bleed_line(store)
        assert line == ""

    def test_real_empty_book_silent(self):
        """Empty decisions list — NO_DATA, line silent."""
        from paper_trader import reporter

        store = self._store([], [])
        line = reporter._drought_alpha_bleed_line(store)
        assert line == ""


if __name__ == "__main__":  # pragma: no cover
    pytest.main([__file__, "-v"])
