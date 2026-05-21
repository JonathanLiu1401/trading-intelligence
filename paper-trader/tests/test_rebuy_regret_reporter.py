"""Tests for `reporter._rebuy_regret_line` — Discord surface for the existing
`build_rebuy_regret` analytic.

`build_rebuy_regret` (paper_trader/analytics/rebuy_regret.py) is wired only
to `/api/rebuy-regret` on the dashboard. The operator who lives in Discord
never sees the verdict that they "sold low, bought back higher" — the exact
disposition-effect pattern this verdict was built to flag. This module
covers the new `_rebuy_regret_line` that routes the builder's own headline
to the surface the operator actually reads — the same dashboard→Discord
trajectory `_streak_line` / `_repeat_loser_line` each followed.

Tests verify:
  * SAVINGS / NET_NEUTRAL / NO_DATA / NO_REBUYS states are silent (no
    hourly noise — the silence precedent).
  * REGRETTING verdict surfaces verbatim from the builder (invariant #10).
  * Builder fault / store raise / non-dict result degrade to "" (additive
    failure contract).
  * Wired into both send_hourly_summary and send_daily_close.
"""
from __future__ import annotations

from unittest.mock import patch

from paper_trader import reporter


class _FakeStore:
    """Minimal store stub — only the reads `_rebuy_regret_line` and the
    hourly/daily code paths touch."""
    def __init__(self, **kw):
        self._portfolio = kw.get("portfolio", {
            "cash": 500.0, "total_value": 1000.0, "positions": [],
            "last_updated": "2026-05-20T12:00:00+00:00",
        })
        self._open_positions = kw.get("open_positions", [])
        self._recent_trades = kw.get("recent_trades", [])
        self._recent_decisions = kw.get("recent_decisions", [])
        self._equity_curve = kw.get("equity_curve", [])

    def get_portfolio(self):
        return dict(self._portfolio)

    def open_positions(self):
        return list(self._open_positions)

    def recent_trades(self, limit=50):
        return list(self._recent_trades[:limit])

    def recent_decisions(self, limit=20):
        return list(self._recent_decisions[:limit])

    def equity_curve(self, limit=500):
        return list(self._equity_curve)


def _make_close_then_rebuy_trades(
    sell_px: float,
    rebuy_px: float,
    ticker: str = "NVDA",
    qty: float = 1.0,
) -> list[dict]:
    """Build a synthetic trade ledger: BUY at $100 → SELL at sell_px →
    BUY (re-entry) at rebuy_px → SELL at $100 (closes the second leg so the
    builder counts it).

    Returns NEWEST-FIRST to match the `Store.recent_trades()` contract — the
    consumer `_rebuy_regret_line` reverses it before calling
    `build_rebuy_regret`."""
    chronological: list[dict] = [
        {
            "id": 1, "timestamp": "2026-05-10T09:30:00+00:00",
            "ticker": ticker, "action": "BUY",
            "qty": qty, "price": 100.0, "value": 100.0 * qty,
            "option_type": None, "strike": None, "expiry": None,
        },
        {
            "id": 2, "timestamp": "2026-05-10T10:00:00+00:00",
            "ticker": ticker, "action": "SELL",
            "qty": qty, "price": sell_px, "value": sell_px * qty,
            "option_type": None, "strike": None, "expiry": None,
        },
        {
            "id": 3, "timestamp": "2026-05-10T11:00:00+00:00",
            "ticker": ticker, "action": "BUY",
            "qty": qty, "price": rebuy_px, "value": rebuy_px * qty,
            "option_type": None, "strike": None, "expiry": None,
        },
        {
            "id": 4, "timestamp": "2026-05-10T12:00:00+00:00",
            "ticker": ticker, "action": "SELL",
            "qty": qty, "price": 100.0, "value": 100.0 * qty,
            "option_type": None, "strike": None, "expiry": None,
        },
    ]
    return list(reversed(chronological))


class TestRebuyRegretLineSuppression:
    """A balanced or savings book must produce no Discord line — the silence
    precedent. The summary must never become its own lying green light."""

    def test_empty_book_returns_empty(self):
        store = _FakeStore(recent_trades=[])
        assert reporter._rebuy_regret_line(store) == ""

    def test_no_rebuys_returns_empty(self):
        # One open→close round-trip with no re-entry → NO_REBUYS → silent.
        trades = list(reversed([
            {"id": 1, "timestamp": "2026-05-10T09:30:00+00:00",
             "ticker": "NVDA", "action": "BUY",
             "qty": 1.0, "price": 100.0, "value": 100.0,
             "option_type": None, "strike": None, "expiry": None},
            {"id": 2, "timestamp": "2026-05-10T10:00:00+00:00",
             "ticker": "NVDA", "action": "SELL",
             "qty": 1.0, "price": 105.0, "value": 105.0,
             "option_type": None, "strike": None, "expiry": None},
        ]))
        store = _FakeStore(recent_trades=trades)
        assert reporter._rebuy_regret_line(store) == ""

    def test_savings_returns_empty(self):
        # Sold at $110, re-bought at $100 → SAVED $10 ⇒ SAVINGS → silent
        # (good outcome, no operator alert needed).
        trades = _make_close_then_rebuy_trades(sell_px=110.0, rebuy_px=100.0)
        store = _FakeStore(recent_trades=trades)
        out = reporter._rebuy_regret_line(store)
        assert out == "", (
            "SAVINGS must be silent; got: " + repr(out)
        )

    def test_neutral_returns_empty(self):
        # Sell px == rebuy px → regret == 0 → NET_NEUTRAL → silent.
        trades = _make_close_then_rebuy_trades(sell_px=100.0, rebuy_px=100.0)
        store = _FakeStore(recent_trades=trades)
        out = reporter._rebuy_regret_line(store)
        assert out == "", (
            "NET_NEUTRAL must be silent; got: " + repr(out)
        )


class TestRebuyRegretLineSurfaces:
    """REGRETTING must surface the builder's own headline verbatim — invariant
    #10. The Discord line and `/api/rebuy-regret` can never tell different
    stories."""

    def test_regretting_surfaces_verbatim(self):
        # Sold at $100, re-bought at $110 (10/share x 1qty = $10 regret,
        # above _MATERIAL_USD=$5) → REGRETTING → must fire.
        trades = _make_close_then_rebuy_trades(sell_px=100.0, rebuy_px=110.0)
        store = _FakeStore(recent_trades=trades)
        out = reporter._rebuy_regret_line(store)
        assert out, "should surface REGRETTING verdict; got: " + repr(out)
        # The Discord prefix announces the surface so the operator can
        # eye-scan the hourly.
        assert "REBUY REGRET" in out
        assert "sold low, bought higher" in out
        # The builder's own headline says "Net REGRET $X.XX across N
        # re-entry event(s). Worst: NVDA ..." — must be present verbatim.
        assert "Net REGRET" in out
        assert "NVDA" in out

    def test_regretting_quotes_worst_event_prices(self):
        # The headline quotes the worst event's sold→re-bought prices.
        trades = _make_close_then_rebuy_trades(sell_px=200.0, rebuy_px=215.0)
        store = _FakeStore(recent_trades=trades)
        out = reporter._rebuy_regret_line(store)
        assert "200" in out, (
            "headline must quote sold price; got:\n" + out
        )
        assert "215" in out, (
            "headline must quote re-bought price; got:\n" + out
        )


class TestRebuyRegretLineFailureContract:
    """Any builder/store fault degrades to ``""`` — never an exception that
    takes down the whole hourly/daily summary (the `_streak_line` /
    `_repeat_loser_line` additive failure contract)."""

    def test_store_raise_returns_empty(self):
        class BadStore:
            def recent_trades(self, limit=50):
                raise RuntimeError("store down")
        assert reporter._rebuy_regret_line(BadStore()) == ""

    def test_builder_raise_returns_empty(self):
        trades = _make_close_then_rebuy_trades(sell_px=100.0, rebuy_px=110.0)
        store = _FakeStore(recent_trades=trades)
        with patch(
            "paper_trader.analytics.rebuy_regret.build_rebuy_regret",
            side_effect=RuntimeError("builder boom"),
        ):
            assert reporter._rebuy_regret_line(store) == ""

    def test_non_dict_result_returns_empty(self):
        trades = _make_close_then_rebuy_trades(sell_px=100.0, rebuy_px=110.0)
        store = _FakeStore(recent_trades=trades)
        with patch(
            "paper_trader.analytics.rebuy_regret.build_rebuy_regret",
            return_value=None,
        ):
            assert reporter._rebuy_regret_line(store) == ""

    def test_missing_headline_returns_empty(self):
        trades = _make_close_then_rebuy_trades(sell_px=100.0, rebuy_px=110.0)
        store = _FakeStore(recent_trades=trades)
        with patch(
            "paper_trader.analytics.rebuy_regret.build_rebuy_regret",
            return_value={"verdict": "REGRETTING", "headline": ""},
        ):
            assert reporter._rebuy_regret_line(store) == ""

    def test_non_regretting_verdict_returns_empty(self):
        # Even with a healthy headline, only REGRETTING fires — any other
        # verdict is silent (the suppression contract).
        trades = _make_close_then_rebuy_trades(sell_px=100.0, rebuy_px=110.0)
        store = _FakeStore(recent_trades=trades)
        with patch(
            "paper_trader.analytics.rebuy_regret.build_rebuy_regret",
            return_value={"verdict": "SAVINGS", "headline": "looks fine"},
        ):
            assert reporter._rebuy_regret_line(store) == ""


class TestRebuyRegretLineWiredIntoSummaries:
    """The line must be wired into both `send_hourly_summary` and
    `send_daily_close` — the operator must see the disposition-effect
    pattern on the surface they actually read (the `_streak_line` /
    `_repeat_loser_line` precedent)."""

    def test_hourly_summary_surfaces_regretting(self, monkeypatch):
        trades = _make_close_then_rebuy_trades(sell_px=100.0, rebuy_px=110.0)
        sent: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda body: sent.append(body) or True)
        store = _FakeStore(recent_trades=trades)
        monkeypatch.setattr(reporter, "get_store", lambda: store)
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda: 5000.0)

        ok = reporter.send_hourly_summary()
        assert ok, "send_hourly_summary returned False"
        assert sent, "send must have been called"
        body = sent[0]
        assert "REBUY REGRET" in body, (
            "REGRETTING verdict must appear in the hourly summary; got:\n"
            + body
        )

    def test_hourly_summary_silent_when_savings(self, monkeypatch):
        trades = _make_close_then_rebuy_trades(sell_px=110.0, rebuy_px=100.0)
        sent: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda body: sent.append(body) or True)
        store = _FakeStore(recent_trades=trades)
        monkeypatch.setattr(reporter, "get_store", lambda: store)
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda: 5000.0)
        reporter.send_hourly_summary()
        body = sent[0]
        assert "REBUY REGRET" not in body, (
            "SAVINGS must not surface a REBUY REGRET line; got:\n" + body
        )

    def test_daily_close_surfaces_regretting(self, monkeypatch):
        trades = _make_close_then_rebuy_trades(sell_px=200.0, rebuy_px=215.0)
        sent: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda body: sent.append(body) or True)
        store = _FakeStore(recent_trades=trades)
        monkeypatch.setattr(reporter, "get_store", lambda: store)
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda: 5000.0)

        ok = reporter.send_daily_close()
        assert ok, "send_daily_close returned False"
        assert sent, "send must have been called"
        body = sent[0]
        assert "REBUY REGRET" in body, (
            "REGRETTING verdict must appear in the daily-close; got:\n"
            + body
        )
