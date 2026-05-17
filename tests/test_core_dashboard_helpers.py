"""Tests for pure dashboard helpers that carry real branching / state logic
but had no direct unit coverage:

- ``_scorer_verdict``            — 5-way return bucketing (boundary values)
- ``_position_ages_from_trades`` — chronological open-lot state machine
- ``_next_market_open``          — NYSE session arithmetic (open/close/weekend/holiday)
- ``_classify_action``           — co-pilot action selection (incl. the
                                    documented EXIT-before-TRIM ordering bug)

Every assertion pins a *specific* expected value so a wrong comparison
operator, off-by-one threshold, or broken state transition fails the test.
The article DB / git / network are never touched: these helpers are pure
over their arguments + a patched wall clock.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader import dashboard

NY = ZoneInfo("America/New_York")
UTC = ZoneInfo("UTC")


def _ny(year, month, day, hour, minute):
    """An aware UTC datetime corresponding to a given NY wall-clock time."""
    return datetime(year, month, day, hour, minute, tzinfo=NY).astimezone(UTC)


class _FakeDT:
    """Stand-in for dashboard.datetime so now() is deterministic.

    Only ``now`` and ``fromisoformat`` are referenced by the helpers under
    test; everything else stays the real datetime via __getattr__-free
    explicit passthrough.
    """

    fixed: datetime | None = None

    @classmethod
    def now(cls, tz=None):
        assert cls.fixed is not None
        return cls.fixed.astimezone(tz) if tz else cls.fixed

    @staticmethod
    def fromisoformat(s):
        return datetime.fromisoformat(s)


@pytest.fixture
def patch_clock(monkeypatch):
    def _set(when_utc: datetime):
        _FakeDT.fixed = when_utc
        monkeypatch.setattr(dashboard, "datetime", _FakeDT)
    yield _set
    _FakeDT.fixed = None


# ─────────────────────────── _scorer_verdict ───────────────────────────

class TestScorerVerdict:
    @pytest.mark.parametrize("pred,expected", [
        (50.0, "STRONG_HOLD"),
        (3.0, "STRONG_HOLD"),     # boundary: >= 3.0
        (2.999, "HOLD"),
        (1.0, "HOLD"),            # boundary: >= 1.0
        (0.999, "NEUTRAL"),
        (0.0, "NEUTRAL"),
        (-1.0, "NEUTRAL"),        # boundary: >= -1.0
        (-1.001, "TRIM"),
        (-3.0, "TRIM"),           # boundary: >= -3.0
        (-3.001, "EXIT"),
        (-50.0, "EXIT"),
    ])
    def test_buckets(self, pred, expected):
        assert dashboard._scorer_verdict(pred) == expected


# ─────────────────────── _position_ages_from_trades ───────────────────────

class TestPositionAgesFromTrades:
    def _open(self, ticker, type_="stock"):
        return {"ticker": ticker, "type": type_}

    def test_single_buy_age_is_days_since_buy(self, patch_clock):
        patch_clock(datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc))
        trades = [
            {"ticker": "NVDA", "action": "BUY", "qty": 10,
             "timestamp": "2026-05-05T12:00:00+00:00"},
        ]
        ages = dashboard._position_ages_from_trades([self._open("NVDA")], trades)
        assert ages == {"NVDA": 10}

    def test_partial_sell_keeps_original_entry_date(self, patch_clock):
        # Age must still be measured from the FIRST buy — a partial sell that
        # leaves qty > 0 must NOT reset the open-lot timestamp.
        patch_clock(datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc))
        trades = [
            {"ticker": "MU", "action": "BUY", "qty": 10,
             "timestamp": "2026-05-01T12:00:00+00:00"},
            {"ticker": "MU", "action": "SELL", "qty": 4,
             "timestamp": "2026-05-10T12:00:00+00:00"},
        ]
        ages = dashboard._position_ages_from_trades([self._open("MU")], trades)
        assert ages == {"MU": 14}  # 2026-05-15 − 2026-05-01

    def test_full_sell_then_rebuy_resets_to_second_buy(self, patch_clock):
        # qty returns to ~0 → earliest dropped → next BUY reseeds the clock.
        patch_clock(datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc))
        trades = [
            {"ticker": "AMD", "action": "BUY", "qty": 5,
             "timestamp": "2026-04-01T12:00:00+00:00"},
            {"ticker": "AMD", "action": "SELL", "qty": 5,
             "timestamp": "2026-04-20T12:00:00+00:00"},
            {"ticker": "AMD", "action": "BUY", "qty": 3,
             "timestamp": "2026-05-13T12:00:00+00:00"},
        ]
        ages = dashboard._position_ages_from_trades([self._open("AMD")], trades)
        assert ages == {"AMD": 2}  # measured from the 2026-05-13 re-buy

    def test_ticker_not_open_excluded(self, patch_clock):
        patch_clock(datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc))
        trades = [
            {"ticker": "TSLA", "action": "BUY", "qty": 1,
             "timestamp": "2026-05-01T12:00:00+00:00"},
        ]
        # No open TSLA position → not reported at all.
        ages = dashboard._position_ages_from_trades([self._open("NVDA")], trades)
        assert ages == {}

    def test_option_trades_do_not_corrupt_stock_qty(self, patch_clock):
        # A BUY_CALL on the same ticker must be ignored so it can't push the
        # running stock qty negative / reset the open-lot date.
        patch_clock(datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc))
        trades = [
            {"ticker": "NVDA", "action": "BUY", "qty": 10,
             "timestamp": "2026-05-05T12:00:00+00:00"},
            {"ticker": "NVDA", "action": "BUY_CALL", "qty": 2,
             "timestamp": "2026-05-08T12:00:00+00:00"},
            {"ticker": "NVDA", "action": "SELL_CALL", "qty": 2,
             "timestamp": "2026-05-09T12:00:00+00:00"},
        ]
        ages = dashboard._position_ages_from_trades([self._open("NVDA")], trades)
        assert ages == {"NVDA": 10}  # unaffected by the option round-trip

    def test_open_option_position_not_treated_as_stock(self, patch_clock):
        patch_clock(datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc))
        trades = [
            {"ticker": "NVDA", "action": "BUY", "qty": 10,
             "timestamp": "2026-05-05T12:00:00+00:00"},
        ]
        # Only an open *call* position exists → no stock open-ticker → empty.
        ages = dashboard._position_ages_from_trades(
            [self._open("NVDA", type_="call")], trades)
        assert ages == {}


# ─────────────────────────── _next_market_open ───────────────────────────

class TestNextMarketOpen:
    def test_weekday_before_open_returns_today_930(self, patch_clock):
        # Thursday 2026-05-14 08:00 NY — market closed, opens 1.5h later.
        patch_clock(_ny(2026, 5, 14, 8, 0))
        nxt, secs = dashboard._next_market_open()
        assert nxt == _ny(2026, 5, 14, 9, 30)
        assert secs == 90 * 60  # 08:00 → 09:30

    def test_weekday_after_close_returns_next_morning(self, patch_clock):
        # Thursday 2026-05-14 17:00 NY — opens Friday 09:30 NY.
        patch_clock(_ny(2026, 5, 14, 17, 0))
        nxt, secs = dashboard._next_market_open()
        assert nxt == _ny(2026, 5, 15, 9, 30)
        # 17:00 Thu → 09:30 Fri = 16h30m
        assert secs == int((16 * 60 + 30) * 60)

    def test_friday_after_close_skips_weekend_to_monday(self, patch_clock):
        # Friday 2026-05-15 17:00 NY — Sat/Sun skipped → Monday 2026-05-18.
        patch_clock(_ny(2026, 5, 15, 17, 0))
        nxt, _ = dashboard._next_market_open()
        assert nxt == _ny(2026, 5, 18, 9, 30)

    def test_holiday_is_skipped(self, patch_clock):
        # Friday 2026-05-22 17:00 NY. Mon 2026-05-25 is Memorial Day (a
        # NYSE_HOLIDAYS_2026 entry) → next open is Tue 2026-05-26.
        patch_clock(_ny(2026, 5, 22, 17, 0))
        nxt, _ = dashboard._next_market_open()
        assert nxt == _ny(2026, 5, 26, 9, 30)

    def test_market_open_returns_next_close(self, patch_clock):
        # Thursday 2026-05-14 11:00 NY — market is open; helper returns the
        # 16:00 NY close with a positive seconds-until.
        patch_clock(_ny(2026, 5, 14, 11, 0))
        nxt, secs = dashboard._next_market_open()
        assert nxt == _ny(2026, 5, 14, 16, 0)
        assert secs == 5 * 3600  # 11:00 → 16:00


# ─────────────────────────── _classify_action ───────────────────────────

class TestClassifyAction:
    def test_strong_bearish_held_is_exit_not_trim(self):
        """Regression: EXIT must be evaluated before TRIM. A bias < -0.5 with
        quiet news also satisfies the TRIM guard (bias < -0.3, news < 0.4);
        if the order were swapped this returns TRIM and understates severity."""
        quant = {"RSI": 75, "MACD": "bearish"}  # -0.4 + -0.25 = -0.65 bias
        action, conv, notes = dashboard._classify_action(
            "NVDA", held_qty=10, quant=quant, news_score=0.0, news_urgent=False)
        assert action == "EXIT"
        assert conv == pytest.approx(0.845)  # min(0.65 + 0.65*0.3, 0.95)
        assert any("overbought" in n for n in notes)

    def test_mild_bearish_held_quiet_news_is_trim(self):
        quant = {"RSI": 65, "MACD": "bearish"}  # -0.1 + -0.25 = -0.35 bias
        action, conv, _ = dashboard._classify_action(
            "MU", held_qty=5, quant=quant, news_score=0.0, news_urgent=False)
        assert action == "TRIM"
        assert conv == pytest.approx(0.705)  # min(0.6 + 0.35*0.3, 0.95)

    def test_neutral_held_is_hold(self):
        action, conv, notes = dashboard._classify_action(
            "AMD", held_qty=5, quant={}, news_score=0.0, news_urgent=False)
        assert action == "HOLD"
        assert conv == pytest.approx(0.4)
        assert notes == []

    def test_bullish_held_with_news_is_add(self):
        quant = {"RSI": 25, "MACD": "bullish"}  # +0.4 + 0.25 = 0.65 bias
        action, conv, _ = dashboard._classify_action(
            "NVDA", held_qty=3, quant=quant, news_score=6.0, news_urgent=False)
        assert action == "ADD"
        # min(0.5 + 0.65*0.3 + 0.6*0.2, 0.95)
        assert conv == pytest.approx(0.815)

    def test_strong_news_plus_technical_not_held_is_buy(self):
        quant = {"RSI": 25}  # +0.4 bias
        action, conv, _ = dashboard._classify_action(
            "AMD", held_qty=0, quant=quant, news_score=8.0, news_urgent=False)
        assert action == "BUY"
        # min(0.5 + 0.8*0.3 + 0.4*0.2, 0.95)
        assert conv == pytest.approx(0.82)

    def test_buy_never_without_technical_confirm(self):
        """Docstring invariant: never BUY on news alone. Max news, zero
        technical bias must downgrade to WATCH, not BUY."""
        action, conv, _ = dashboard._classify_action(
            "AMD", held_qty=0, quant={}, news_score=10.0, news_urgent=False)
        assert action == "WATCH"
        assert conv == pytest.approx(0.6)  # min(0.3 + 1.0*0.3, 0.8)

    def test_quiet_not_held_is_low_conviction_watch(self):
        action, conv, _ = dashboard._classify_action(
            "AMD", held_qty=0, quant={}, news_score=0.0, news_urgent=False)
        assert action == "WATCH"
        assert conv == pytest.approx(0.2)

    def test_urgent_news_bumps_weight_and_prepends_note(self):
        action, conv, notes = dashboard._classify_action(
            "AMD", held_qty=0, quant={}, news_score=0.0, news_urgent=True)
        # news_weight: 0.0 → +0.2 (urgent) → 0.2
        assert notes[0] == "URGENT news"
        assert action == "WATCH"
        assert conv == pytest.approx(0.24)  # 0.2 + 0.2*0.2


class TestTemplateIdsUnique:
    """Regression lock for the duplicate-DOM-id collision bug.

    Successive feature agents added the 'Drawdown anatomy' card (2026-05-15)
    and the 'Decision drought drift' card (2026-05-16). The drought card
    reused the drawdown card's `dd-` id prefix, so `id="dd-card"` and
    `id="dd-current"` each appeared twice. `getElementById("dd-current")`
    resolves to the *first* element in document order (the drawdown card's
    "current equity" stat), so `refreshDecisionDrought()` wrote its status
    into the wrong card and the drought card's own status box never left
    "loading…". Every static `id="..."` in TEMPLATE must be globally unique
    or a co-pilot panel silently corrupts another panel's DOM.
    """

    @staticmethod
    def _static_ids():
        import re
        # Only static ids matter for getElementById collisions — exclude any
        # id containing Jinja interpolation ({...}), which is never queried by
        # a literal getElementById.
        return re.findall(r'\bid="([^"{}]+)"', dashboard.TEMPLATE)

    def test_no_duplicate_static_element_ids(self):
        from collections import Counter
        dups = {k: v for k, v in Counter(self._static_ids()).items() if v > 1}
        assert dups == {}, f"duplicate element id(s) in TEMPLATE: {dups}"

    def test_drought_and_drawdown_cards_do_not_share_ids(self):
        """The specific fix: drought card → `drought-*`, drawdown keeps `dd-*`."""
        ids = set(self._static_ids())
        # Drawdown anatomy card retains the original `dd-` namespace.
        assert "dd-card" in ids and "dd-current" in ids
        # Decision drought card was renamed off the collision.
        assert "drought-card" in ids and "drought-current" in ids
        # And the JS that drives the drought card targets the renamed id,
        # not the drawdown stat it used to clobber.
        assert 'getElementById("drought-current")' in dashboard.TEMPLATE
        assert 'getElementById("dd-current")' in dashboard.TEMPLATE  # drawdown's own
