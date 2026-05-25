"""Agent 2 (ML+backtests) review 2026-05-25 — bugs found and fixed.

Three real bugs were uncovered while auditing paper_trader/backtest.py and
run_continuous_backtests.py:

1. **_macd label flapping on steady-state trends.**  In a linear-trend price
   series the MACD line and signal line converge to the same value; tiny
   floating-point roundoff in the EMA-of-EMA chain flips the
   ``m > s`` comparison the label was derived from, producing
   ``"bearish"`` for an obvious uptrend and ``"bullish"`` for an obvious
   downtrend.  The label appears in the live trader's prompt-build path
   (``_build_prompt`` → ``MARKET STRUCTURE`` block) so a false ``bearish``
   in a clean uptrend misleads the LLM.  Fix: compare with an epsilon
   tolerance and label as ``"flat"`` when ``|m - s|`` is at noise.

2. **NaN/Inf in price closes silently propagates through indicators.**
   A poisoned price cache (one NaN close, e.g. yfinance returning corrupt
   data on a bad day) flows through ``_rsi`` and ``_macd`` unchecked,
   yielding ``rsi=NaN`` and ``macd_signal=NaN``.  ``_ml_decide`` then sees
   ``isinstance(NaN, float) == True`` and applies the negative branch
   (``adj -= 0.5``) because ``NaN > 0`` is False — silently penalising
   every name with a NaN-poisoned indicator.  Fix: defensively return
   ``None`` when any input close is non-finite.

3. **``datetime.utcnow()`` deprecation in the monkey-benchmark refresh
   path (run_continuous_backtests.py:~4120).**  Python 3.12 emits a
   DeprecationWarning and 3.13+ will remove it.  Fix: switch to
   timezone-aware ``datetime.now(timezone.utc)`` and parse ``gen_at`` as
   aware too so subtraction is consistent.

These tests assert the corrected behaviour: they fail against the buggy
code and pass against the fix.
"""
from __future__ import annotations

import math
import random
import re
import warnings
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest


# ---------------------------------------------------------------------------
# Bug 1: _macd label flapping on steady-state linear trends
# ---------------------------------------------------------------------------


class TestMacdLabelStability:
    """The MACD label must not flip from floating-point roundoff alone.

    On a clean linear trend (no noise) the MACD line and signal line
    converge to the same value in steady state.  Any tiny rounding error
    can flip ``m > s`` either way — the label has no real information.
    A robust implementation labels this as ``"flat"`` rather than
    fabricating a bullish/bearish verdict from EMA accumulation order.
    """

    def test_clean_linear_uptrend_does_not_report_bearish(self):
        """A 60-bar pure linear uptrend is not bearish."""
        from paper_trader.backtest import _macd

        closes = [100.0 + i for i in range(60)]
        res = _macd(closes)
        assert res is not None, "_macd returned None for 60-bar series"
        label, m, s = res
        # The macd line is positive (EMA12 > EMA26 in an uptrend),
        # so m > 0.  But m ≈ s in steady state — labeling this
        # "bearish" because of float roundoff is the bug.
        assert m > 0, f"linear uptrend should have macd>0, got {m}"
        assert label != "bearish", (
            f"linear uptrend labeled {label!r} (m={m}, s={s}) — "
            f"diff {m - s:.3e} is at floating-point noise floor"
        )

    def test_clean_linear_downtrend_does_not_report_bullish(self):
        """A 60-bar pure linear downtrend is not bullish."""
        from paper_trader.backtest import _macd

        closes = [200.0 - i for i in range(60)]
        res = _macd(closes)
        assert res is not None
        label, m, s = res
        assert m < 0, f"linear downtrend should have macd<0, got {m}"
        assert label != "bullish", (
            f"linear downtrend labeled {label!r} (m={m}, s={s}) — "
            f"diff {m - s:.3e} is at floating-point noise floor"
        )

    def test_real_bullish_crossover_still_labels_bullish(self):
        """The epsilon guard must NOT swallow real crossovers.

        Build a series whose MACD line genuinely crosses above the
        signal line (a noisy uptrend) — the label must be ``"bullish"``
        regardless of the new tolerance.
        """
        from paper_trader.backtest import _macd

        rng = random.Random(42)
        # Sideways → strong uptrend (a real crossover setup).
        closes = [100.0 + rng.gauss(0, 0.5) for _ in range(40)]
        closes += [closes[-1] + i * 0.8 + rng.gauss(0, 0.3)
                   for i in range(1, 41)]
        res = _macd(closes)
        assert res is not None
        label, m, s = res
        # The macd line should be measurably above the signal at the
        # end of a real bullish acceleration.
        assert m - s > 1e-3, (
            f"engineered bullish crossover did not produce m-s > 1e-3 "
            f"(m={m}, s={s})"
        )
        assert label == "bullish", (
            f"engineered bullish crossover labeled {label!r}"
        )

    def test_flat_zero_input_yields_flat_label(self):
        """A perfectly constant input must yield ``"flat"``."""
        from paper_trader.backtest import _macd

        closes = [100.0] * 60
        res = _macd(closes)
        assert res is not None
        label, m, s = res
        assert m == 0.0 and s == 0.0
        assert label == "flat"


# ---------------------------------------------------------------------------
# Bug 2: NaN/Inf closes silently propagate through indicators
# ---------------------------------------------------------------------------


class TestIndicatorNanGuards:
    """An indicator computed over a price series containing NaN or Inf
    must return ``None`` rather than propagating the non-finite value
    downstream.  ``_ml_decide``'s gate uses ``isinstance(x, (int, float))``
    to decide whether to apply quant adjustments — NaN passes that check
    (NaN is a float) and then ``NaN > 0`` is False, so the negative
    branch fires.  A defensive ``None`` skips the adjustment entirely
    instead of silently penalising every NaN-poisoned name.
    """

    def test_rsi_returns_none_for_nan_closes(self):
        from paper_trader.backtest import _rsi

        closes = [float("nan")] * 60
        v = _rsi(closes)
        assert v is None, (
            f"NaN closes should yield None RSI, got {v!r} "
            f"(NaN would feed `adj -= 0.5` in _ml_decide because "
            f"`NaN > 0` is False)"
        )

    def test_rsi_returns_none_for_inf_closes(self):
        from paper_trader.backtest import _rsi

        closes = [float("inf")] * 60
        v = _rsi(closes)
        assert v is None, f"Inf closes should yield None RSI, got {v!r}"

    def test_rsi_returns_none_for_mixed_nan(self):
        from paper_trader.backtest import _rsi

        closes = [100.0] * 30 + [float("nan")] + [100.0] * 29
        v = _rsi(closes)
        assert v is None, (
            f"Single NaN in closes should poison RSI honestly to None, "
            f"got {v!r}"
        )

    def test_macd_returns_none_for_nan_closes(self):
        from paper_trader.backtest import _macd

        closes = [float("nan")] * 60
        v = _macd(closes)
        assert v is None, (
            f"NaN closes should yield None MACD, got {v!r}"
        )

    def test_macd_returns_none_for_inf_closes(self):
        from paper_trader.backtest import _macd

        closes = [float("inf")] * 60
        v = _macd(closes)
        assert v is None, f"Inf closes should yield None MACD, got {v!r}"

    def test_rsi_clean_uptrend_unchanged(self):
        """The NaN guard must NOT change behaviour for clean inputs."""
        from paper_trader.backtest import _rsi

        closes = [100.0 + i for i in range(30)]
        v = _rsi(closes)
        assert v is not None
        assert 95.0 <= v <= 100.0, (
            f"clean uptrend should pin RSI to ~100, got {v!r}"
        )

    def test_rsi_flat_series_unchanged(self):
        """The textbook neutral RSI=50 fix must survive."""
        from paper_trader.backtest import _rsi

        closes = [100.0] * 30
        v = _rsi(closes)
        # The existing "flat → 50" guard MUST still work.
        assert v == 50.0


# ---------------------------------------------------------------------------
# Bug 3: datetime.utcnow() deprecation
# ---------------------------------------------------------------------------


class TestUtcnowDeprecation:
    """``datetime.utcnow()`` is deprecated in Python 3.12 and slated for
    removal.  The monkey-benchmark refresh path inside
    ``run_continuous_backtests.main`` used it once.  Fix: use a
    timezone-aware ``datetime.now(timezone.utc)`` and parse ``gen_at``
    as aware too so subtraction is well-defined.
    """

    def test_no_utcnow_in_run_continuous(self):
        """The source must no longer call ``datetime.utcnow()`` —
        deprecated and removed in a future Python.
        """
        from pathlib import Path

        src = Path(__file__).resolve().parent.parent / "run_continuous_backtests.py"
        text = src.read_text()
        # Strip comments before checking so a description that mentions
        # the historical usage doesn't cause a false positive.
        code_only = "\n".join(
            re.sub(r"#.*$", "", ln) for ln in text.splitlines()
        )
        # Allow ``timezone.utc`` (e.g. ``datetime.now(timezone.utc)``) and
        # ``_dt.UTC`` — but not the bare ``utcnow()`` call.
        assert "utcnow(" not in code_only, (
            "run_continuous_backtests.py still calls datetime.utcnow() — "
            "deprecated in Python 3.12 and slated for removal. "
            "Use datetime.now(timezone.utc) instead."
        )
