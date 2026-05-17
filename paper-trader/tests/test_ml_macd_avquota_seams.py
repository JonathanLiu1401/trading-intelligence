"""ML/backtest seam locks — 2026-05-16 review pass (Agent 2).

Found by grepping every backtest symbol against tests/: two pieces of
real, load-bearing logic with **zero** prior direct coverage —

1. `_macd`  — the numeric MACD signal it returns (`element [2]`,
   `macd_signal`) is DecisionScorer feature slot 2 and drives
   `_ml_decide`'s `adj += 0.5 if macd > 0 else -0.5`. Its correctness
   rests on a subtle EMA-alignment offset (`offset = len(ema12) -
   len(ema26)` = 14) that a refactor could silently shift, corrupting
   every quant feature without raising. `_ema` itself was imported by
   `test_backtest.py` but only for the `len < period → []` guard — the
   actual seed-as-SMA + `v·k + prev·(1−k)` recurrence was never
   value-asserted. Both are pinned here.

2. `AlphaVantageNewsFetcher._quota` / `_inc_quota` — the cross-restart
   daily-quota tracker that implements CLAUDE.md §8 invariant #9 ("the
   Alpha Vantage daily quota persists across restarts ... a day change
   resets the client's view"). A regression in the single load-bearing
   line `if q.get("date") == date.today().isoformat()` would either
   carry yesterday's count forward forever (client thinks quota is
   always exhausted) or never persist it (silent AV ban from
   over-calling). Fully offline — the conftest autouse fixture
   redirects `AV_QUOTA_PATH` / `AV_CACHE_DIR` into tmp.

Deliberately NOT locked (documented sharp edges, no functional impact):
the MACD *label* ("bullish"/"bearish"/"flat") is float-noise-sensitive
on a perfectly linear ramp (m and s differ only at ~1e-15, flipping the
label) — but the label's *only* reader is `_build_prompt`'s currently
unused Opus path; `_ml_decide` and the scorer consume the numeric
signal. Locking a label on a linear ramp would cement float noise as
semantics, so the label is asserted only on *non-degenerate convex*
series where m−s is a real (>1.0) margin, plus the exact-zero `flat`
branch on all-identical closes (no float noise there).

All assertions are exact values / independent reconstructions, not
ranges — a normalization or offset change must update the literals
deliberately.
"""
from __future__ import annotations

import json
from datetime import date, timedelta

import pytest

import paper_trader.backtest as bt
from paper_trader.backtest import _ema, _macd, AlphaVantageNewsFetcher


# ───────────────────────────── _ema ─────────────────────────────

class TestEmaComputation:
    def test_exact_recurrence(self):
        # seed = SMA(values[:period]); thereafter v*k + prev*(1-k),
        # k = 2/(period+1). period=3 → k=0.5.
        #   seed       = (1+2+3)/3 = 2.0
        #   v=4 → 4*.5 + 2.0*.5 = 3.0
        #   v=5 → 5*.5 + 3.0*.5 = 4.0
        #   v=6 → 6*.5 + 4.0*.5 = 5.0
        assert _ema([1, 2, 3, 4, 5, 6], 3) == [2.0, 3.0, 4.0, 5.0]

    def test_output_length_is_n_minus_period_plus_one(self):
        out = _ema([float(i) for i in range(30)], 12)
        assert len(out) == 30 - 12 + 1

    def test_too_short_returns_empty(self):
        # Boundary: len < period → []. (len == period yields one seed point.)
        assert _ema([1.0, 2.0], 3) == []
        assert _ema([1.0, 2.0, 3.0], 3) == [2.0]


# ───────────────────────────── _macd ─────────────────────────────

# Non-degenerate convex fixtures: macd_line has a *real* trend so the
# label is determined by genuine momentum, not float rounding.
_ACC_UP = [100.0 + i * i * 0.1 for i in range(50)]    # accelerating rise
_ACC_DN = [1000.0 - i * i * 0.1 for i in range(50)]   # accelerating decline


class TestMacd:
    def test_history_guard(self):
        # Needs len(closes) >= 35 (26-period EMA → macd_line, then a
        # 9-period signal EMA). 34 closes is one short → None.
        assert _macd([100.0 + i for i in range(34)]) is None
        assert _macd([100.0 + i * i * 0.1 for i in range(35)]) is not None

    def test_alignment_is_ema12_last_minus_ema26_last(self):
        """The strongest, input-agnostic regression lock.

        macd_line[i] = ema12[i + offset] - ema26[i], offset = 14, so
        macd_line[-1] MUST equal ema12[-1] - ema26[-1]. A regression
        that drops `+ offset` makes it ema12[k] - ema26[k] for a
        mis-shifted k and this equality breaks loudly.
        """
        label, m, s = _macd(_ACC_UP)
        ema12 = _ema(_ACC_UP, 12)
        ema26 = _ema(_ACC_UP, 26)
        assert round(m, 9) == round(ema12[-1] - ema26[-1], 9)

        # And s (returned as element [2]) is the last point of the
        # 9-EMA of the fully-reconstructed macd_line — locks both the
        # signal computation and the tuple ordering (m, s).
        offset = len(ema12) - len(ema26)
        assert offset == 14
        macd_line = [ema12[i + offset] - ema26[i] for i in range(len(ema26))]
        signal_line = _ema(macd_line, 9)
        assert round(s, 9) == round(signal_line[-1], 9)

    def test_bullish_on_accelerating_rise(self):
        label, m, s = _macd(_ACC_UP)
        assert label == "bullish"
        assert m > 0.0
        assert m - s > 1.0           # a real margin, not float noise
        assert m == pytest.approx(44.46994635985152, rel=1e-9)
        assert s == pytest.approx(39.682088657587066, rel=1e-9)

    def test_bearish_on_accelerating_decline(self):
        label, m, s = _macd(_ACC_DN)
        assert label == "bearish"
        assert m < 0.0
        assert s - m > 1.0
        assert m == pytest.approx(-44.46994635985163, rel=1e-9)
        assert s == pytest.approx(-39.68208865758706, rel=1e-9)

    def test_flat_branch_exact_zero_on_constant_series(self):
        # All-identical closes → macd_line is exactly 0.0 everywhere
        # (no float noise here, unlike a linear ramp) → m == s == 0.0
        # → the `label == "flat"` branch. Locks the m==s tie-break.
        assert _macd([50.0] * 40) == ("flat", 0.0, 0.0)


# ─────────────── AlphaVantageNewsFetcher quota tracker ───────────────
# CLAUDE.md §8 invariant #9. conftest autouse redirects bt.AV_QUOTA_PATH
# / bt.AV_CACHE_DIR into tmp, so every case here is fully offline.

class TestAlphaVantageQuota:
    def test_fresh_when_no_file(self):
        f = AlphaVantageNewsFetcher()
        assert f._quota() == {"date": date.today().isoformat(), "calls": 0}

    def test_inc_quota_persists_and_accumulates(self):
        f = AlphaVantageNewsFetcher()
        f._inc_quota()
        f._inc_quota()
        f._inc_quota()
        # Verify the on-disk file DIRECTLY — not via _quota(), whose
        # broad `except Exception: pass` would mask a corrupt write.
        raw = json.loads(bt.AV_QUOTA_PATH.read_text())
        assert raw == {"date": date.today().isoformat(), "calls": 3}

    def test_quota_honors_same_day_file(self):
        bt.AV_QUOTA_PATH.write_text(
            json.dumps({"date": date.today().isoformat(), "calls": 7})
        )
        f = AlphaVantageNewsFetcher()
        assert f._quota()["calls"] == 7

    def test_quota_rolls_over_on_new_day(self):
        # Invariant #9: yesterday's count must NOT carry into today.
        # This is the single load-bearing line
        # `if q.get("date") == date.today().isoformat()`.
        stale = (date.today() - timedelta(days=1)).isoformat()
        bt.AV_QUOTA_PATH.write_text(json.dumps({"date": stale, "calls": 19}))
        f = AlphaVantageNewsFetcher()
        assert f._quota() == {"date": date.today().isoformat(), "calls": 0}

    def test_quota_corrupt_file_degrades_to_fresh(self):
        bt.AV_QUOTA_PATH.write_text("{ not valid json ::::")
        f = AlphaVantageNewsFetcher()
        assert f._quota() == {"date": date.today().isoformat(), "calls": 0}

    def test_inc_after_rollover_starts_at_one_not_stale_plus_one(self):
        """The end-to-end invariant-#9 lock.

        `_inc_quota` calls `_quota()` (which must roll over) then +1.
        A new day therefore starts counting at 1 — NOT 22. A regression
        that drops the date check in `_quota` would yield 22 here
        (stale count carried forward), silently keeping the client
        permanently over-quota or never resetting it.
        """
        stale = (date.today() - timedelta(days=1)).isoformat()
        bt.AV_QUOTA_PATH.write_text(json.dumps({"date": stale, "calls": 21}))
        f = AlphaVantageNewsFetcher()
        f._inc_quota()
        raw = json.loads(bt.AV_QUOTA_PATH.read_text())
        assert raw == {"date": date.today().isoformat(), "calls": 1}
