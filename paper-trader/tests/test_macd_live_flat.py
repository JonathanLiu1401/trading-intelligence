"""Pins the steady-state-flat fix in `strategy._macd_live`.

Mirrors the `backtest._macd` fix landed in AGENTS.md pass #38 — without
this guard a STRICTLY LINEAR price series flapped to ``"bullish"`` /
``"bearish"`` from EMA accumulation roundoff alone, polluting Opus's
live decision prompt with a false MACD label. The existing
``test_accelerating_*`` tests deliberately avoid the steady-state by
using accelerating trends; this file covers what the existing suite
explicitly side-stepped:

* a strictly linear uptrend is ``"flat"`` (not bullish-by-noise)
* a strictly linear downtrend is ``"flat"`` (not bearish-by-noise)
* a steady constant series is ``"flat"`` (the most extreme degenerate)
* a sharp UP crossover stays ``"bullish"`` (real signal survives the
  epsilon tolerance — the bar the fix MUST clear)
* a sharp DOWN crossover stays ``"bearish"`` (symmetric guarantee)
"""
from __future__ import annotations

from paper_trader import strategy


class TestMacdLiveFlatLabel:
    """The fix's headline behaviour: linear / steady-state ⇒ ``"flat"``."""

    def test_strictly_linear_uptrend_is_flat_not_bullish(self):
        # Strict linear uptrend: MACD line converges to signal line at
        # machine precision. The pre-fix code returned "bullish" purely
        # from EMA accumulation roundoff — a false catalyst label fed
        # directly into the live decision prompt.
        closes = [100.0 + i for i in range(80)]
        assert strategy._macd_live(closes) == "flat"

    def test_strictly_linear_downtrend_is_flat_not_bearish(self):
        # Symmetric to the uptrend case — pre-fix code returned "bearish"
        # purely from roundoff on a clean linear DOWNtrend.
        closes = [200.0 - i for i in range(80)]
        assert strategy._macd_live(closes) == "flat"

    def test_constant_series_is_flat(self):
        # The most degenerate steady state — every close identical. Both
        # MACD line and signal line are exactly 0; bare ``>`` is False
        # so pre-fix code returned "bearish" (a phantom sell label on a
        # genuinely dead tape).
        closes = [150.0 for _ in range(60)]
        assert strategy._macd_live(closes) == "flat"


class TestMacdLiveStillDetectsRealCrossovers:
    """The fix MUST NOT swallow real signal — epsilon scales with the
    magnitudes involved, so a real crossover (diff well above the noise
    floor) is still classified."""

    def test_sharp_up_crossover_is_bullish(self):
        # 50 bars of flat then a sharp jump: MACD line spikes well above
        # the signal line. Diff is orders of magnitude above the
        # epsilon → must return "bullish".
        closes = [100.0] * 50 + [100.0 + 2 * i for i in range(1, 31)]
        assert strategy._macd_live(closes) == "bullish"

    def test_sharp_down_crossover_is_bearish(self):
        # Symmetric to the up case: sharp downside break after a flat
        # base. Must return "bearish" — fix must not over-suppress.
        closes = [200.0] * 50 + [200.0 - 2 * i for i in range(1, 31)]
        assert strategy._macd_live(closes) == "bearish"

    def test_accelerating_uptrend_still_bullish(self):
        # Locks the pre-fix accelerating-uptrend assertion: this is the
        # case the original test suite hand-picked because it survives
        # the bare ``>``. The fix MUST leave it bullish (the headline
        # behaviour of the comparison stays correct on real signal).
        closes = [100.0 + i + 0.02 * i * i for i in range(60)]
        assert strategy._macd_live(closes) == "bullish"

    def test_accelerating_downtrend_still_bearish(self):
        # Symmetric to the uptrend pin — guards against an over-zealous
        # epsilon that would swallow real downside.
        closes = [100.0 - i - 0.02 * i * i for i in range(60)]
        assert strategy._macd_live(closes) == "bearish"


class TestMacdLiveLengthGuards:
    """Pin the existing ``len(closes) < 35`` short-circuit so a future
    refactor cannot silently let it through."""

    def test_too_few_closes_returns_none(self):
        # 34 closes is below the MACD minimum (26 + 9 - 1) so the
        # function must short-circuit to None rather than computing on
        # an incomplete signal line.
        assert strategy._macd_live([float(i) for i in range(34)]) is None

    def test_empty_returns_none(self):
        assert strategy._macd_live([]) is None
