"""Tests for ``reporter._regime_leverage_fit_line`` — Discord surface for the
existing ``build_regime_leverage_fit_skill`` analytic.

``build_regime_leverage_fit_skill`` (paper_trader/analytics/regime_leverage_fit_skill.py)
is wired to ``/api/regime-leverage-fit-skill`` and the digital-intern chat
block ``_regime_leverage_fit_chat_lines``, but NOT to the hourly / daily-close
Discord report. The operator who lives in Discord never sees the structural
verdict that the book's leverage class is misaligned with the SPY 20d regime
— the exact backtest-validated risk pattern this verdict was built to flag.
This module covers the new ``_regime_leverage_fit_line`` that routes the
builder's own headline to the surface the operator actually reads — the same
dashboard→Discord trajectory ``_cash_conviction_fit_line`` /
``_concentration_line`` / ``_rebuy_regret_line`` each followed.

Tests verify:
  * Silence on ALIGNED / DEFENSIVE / NEUTRAL / NO_DATA (no hourly noise —
    the ``_cash_conviction_fit_line`` suppression precedent).
  * Each of the three actionable verdicts (BLIND_LEVERING /
    DANGEROUS_HEADWIND / MISSED_TAILWIND) surfaces verbatim from the
    builder (invariant #10 — chat / dashboard / Discord can never disagree).
  * Builder fault / store raise / non-dict result / missing headline /
    network fault on the SPY momentum fetch all degrade to "" (additive
    failure contract — never an exception that takes down the whole
    hourly / daily summary).
  * Wired into both ``send_hourly_summary`` and ``send_daily_close``.
"""
from __future__ import annotations

from unittest.mock import patch

from paper_trader import reporter


# ── Test doubles ─────────────────────────────────────────────────────────


class _FakeStore:
    """Minimal store stub covering the reads ``_regime_leverage_fit_line``
    and the hourly / daily-close code paths touch."""

    def __init__(self, **kw):
        self._portfolio = kw.get("portfolio", {
            "cash": 500.0,
            "total_value": 1000.0,
            "positions": [],
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


def _patch_spy_mom(value):
    """Patch the live SPY momentum fetch the reporter line uses — returns a
    context manager. ``value`` is the mom_20d % (positive = bull, negative =
    bear, in (-3, 3) = sideways under default thresholds). Pass ``None`` /
    raise via ``side_effect`` for the unknown-regime fallback path."""
    return patch(
        "paper_trader.strategy.get_quant_signals_live",
        return_value={"SPY": {"mom_20d": value}},
    )


# ── Suppression: non-actionable verdicts MUST stay silent ────────────────


class TestRegimeLeverageFitLineSuppression:
    """A regime-fit book must produce no Discord line — the silence precedent.
    The summary must never become its own lying green light."""

    def test_no_data_empty_book_and_unknown_regime_returns_empty(self):
        """Empty positions + cash=0 + spy_mom_20d=None → NO_DATA → silent."""
        store = _FakeStore(portfolio={
            "cash": 0.0, "total_value": 0.0, "positions": [],
        })
        with patch(
            "paper_trader.strategy.get_quant_signals_live",
            return_value={"SPY": {}},
        ):
            assert reporter._regime_leverage_fit_line(store) == ""

    def test_aligned_bull_lev_returns_empty(self):
        """Bull regime + 30% leveraged → ALIGNED → silent (good outcome)."""
        store = _FakeStore(
            portfolio={"cash": 700.0, "total_value": 1000.0, "positions": []},
            open_positions=[
                {"ticker": "TQQQ", "type": "stock", "market_value": 300.0,
                 "qty": 5.0, "avg_cost": 60.0, "current_price": 60.0,
                 "unrealized_pl": 0.0},
            ],
        )
        with _patch_spy_mom(5.0):
            out = reporter._regime_leverage_fit_line(store)
        assert out == "", f"ALIGNED must be silent; got: {out!r}"

    def test_defensive_bear_low_lev_returns_empty(self):
        """Bear regime + 0% leveraged → DEFENSIVE → silent (correct posture)."""
        store = _FakeStore(
            portfolio={"cash": 1000.0, "total_value": 1000.0, "positions": []},
            open_positions=[],
        )
        with _patch_spy_mom(-5.0):
            out = reporter._regime_leverage_fit_line(store)
        assert out == "", f"DEFENSIVE must be silent; got: {out!r}"

    def test_neutral_sideways_mid_lev_returns_empty(self):
        """Sideways regime + mid-band exposure → NEUTRAL → silent."""
        store = _FakeStore(
            portfolio={"cash": 850.0, "total_value": 1000.0, "positions": []},
            open_positions=[
                {"ticker": "TQQQ", "type": "stock", "market_value": 150.0,
                 "qty": 2.0, "avg_cost": 75.0, "current_price": 75.0,
                 "unrealized_pl": 0.0},
            ],
        )
        # spy_mom_20d=0 → sideways under default ±3% thresholds; 15% lev
        # falls between low_lev_ceil (10) and high_lev_floor (30) → NEUTRAL.
        with _patch_spy_mom(0.0):
            out = reporter._regime_leverage_fit_line(store)
        assert out == "", f"NEUTRAL must be silent; got: {out!r}"


# ── Surfaces: each of the three actionable verdicts must fire ────────────


class TestRegimeLeverageFitLineSurfaces:
    """Actionable verdicts must surface the builder's own headline verbatim —
    invariant #10. The Discord line, the dashboard endpoint, and the chat
    helper must never tell different stories."""

    def test_missed_tailwind_surfaces_with_builder_headline(self):
        """The live 100%-cash drought regime: bull tape, 0% leveraged →
        MISSED_TAILWIND. Must surface."""
        store = _FakeStore(
            portfolio={"cash": 987.39, "total_value": 987.39, "positions": []},
            open_positions=[],
            recent_trades=[],
        )
        with _patch_spy_mom(5.0):
            out = reporter._regime_leverage_fit_line(store)
        assert out, f"MISSED_TAILWIND must surface; got: {out!r}"
        # Discord prefix announces the surface for eye-scan.
        assert "REGIME FIT" in out
        assert "MISSED_TAILWIND" in out
        # Builder's own headline phrasing surfaces verbatim (no chat-side
        # re-derived verdict — invariant #10).
        assert "bull tape" in out
        assert "spy_mom_20d=5.00%" in out
        # The leveraged-pct number is in the headline.
        assert "0" in out  # 0% leveraged

    def test_dangerous_headwind_surfaces_with_builder_headline(self):
        """Bear regime + high leveraged exposure → DANGEROUS_HEADWIND."""
        store = _FakeStore(
            portfolio={"cash": 500.0, "total_value": 1000.0, "positions": []},
            open_positions=[
                # 50% leveraged book — well above 30% high_lev_floor.
                {"ticker": "TQQQ", "type": "stock", "market_value": 500.0,
                 "qty": 8.0, "avg_cost": 62.5, "current_price": 62.5,
                 "unrealized_pl": 0.0},
            ],
        )
        with _patch_spy_mom(-5.0):
            out = reporter._regime_leverage_fit_line(store)
        assert out, f"DANGEROUS_HEADWIND must surface; got: {out!r}"
        assert "REGIME FIT" in out
        assert "DANGEROUS_HEADWIND" in out
        assert "into bear" in out
        assert "spy_mom_20d=-5.00%" in out

    def test_blind_levering_surfaces_with_builder_headline(self):
        """Bear / sideways regime + recent leveraged BUY flow ≥ 5% of book
        → BLIND_LEVERING (highest priority — direction of change matters)."""
        # Recent BUY into TQQQ ~ $80 = 8% of $1000 book — above 5% threshold.
        from datetime import datetime, timezone
        recent_ts = datetime.now(timezone.utc).isoformat()
        store = _FakeStore(
            portfolio={"cash": 920.0, "total_value": 1000.0, "positions": []},
            open_positions=[
                {"ticker": "TQQQ", "type": "stock", "market_value": 80.0,
                 "qty": 1.0, "avg_cost": 80.0, "current_price": 80.0,
                 "unrealized_pl": 0.0},
            ],
            recent_trades=[{
                "id": 1, "timestamp": recent_ts, "ticker": "TQQQ",
                "action": "BUY", "qty": 1.0, "price": 80.0, "value": 80.0,
                "option_type": None, "strike": None, "expiry": None,
            }],
        )
        with _patch_spy_mom(-5.0):
            out = reporter._regime_leverage_fit_line(store)
        assert out, f"BLIND_LEVERING must surface; got: {out!r}"
        assert "REGIME FIT" in out
        assert "BLIND_LEVERING" in out
        assert "levering into bear" in out

    def test_surface_includes_discord_prefix_and_verbatim_headline(self):
        """The block format is ``**REGIME FIT** ◈ <VERDICT>\\n> <headline>``
        — locked so the operator can eye-scan against the established
        ``_cash_conviction_fit_line`` / ``_concentration_line`` style."""
        store = _FakeStore(
            portfolio={"cash": 1000.0, "total_value": 1000.0, "positions": []},
            open_positions=[],
            recent_trades=[],
        )
        with _patch_spy_mom(5.0):
            out = reporter._regime_leverage_fit_line(store)
        # Two-line block exactly: header + indented headline.
        lines = out.split("\n")
        assert len(lines) == 2, f"expected 2 lines, got {len(lines)}: {out!r}"
        assert lines[0].startswith("**REGIME FIT** ◈ ")
        assert lines[1].startswith("> ")


# ── Failure contract: never raise, never crash the report ────────────────


class TestRegimeLeverageFitLineFailureContract:
    """Any fault degrades to ``""`` — never an exception that takes down
    the whole hourly / daily summary (the ``_rebuy_regret_line`` /
    ``_cash_conviction_fit_line`` additive failure contract)."""

    def test_store_get_portfolio_raise_returns_empty(self):
        class BadStore:
            def get_portfolio(self):
                raise RuntimeError("store down")
            def open_positions(self):
                return []
            def recent_trades(self, limit=50):
                return []
        with _patch_spy_mom(5.0):
            assert reporter._regime_leverage_fit_line(BadStore()) == ""

    def test_open_positions_raise_degrades_to_empty_book(self):
        """The line catches an open_positions fault internally and falls
        through to the snapshot path — the builder still emits a verdict
        from cash + recent flow, so this is NOT an empty return; it just
        means the held set is treated as empty. With an empty book + bull
        regime, the line MISSED_TAILWIND fires."""
        class HalfBadStore(_FakeStore):
            def open_positions(self):
                raise RuntimeError("positions read down")
        store = HalfBadStore(
            portfolio={"cash": 987.0, "total_value": 987.0, "positions": []},
            recent_trades=[],
        )
        with _patch_spy_mom(5.0):
            out = reporter._regime_leverage_fit_line(store)
        # open_positions fault → empty held set, bull regime, cash-heavy
        # → MISSED_TAILWIND (not "" — that would be the wrong contract).
        assert "MISSED_TAILWIND" in out

    def test_recent_trades_raise_degrades_silently_to_no_flow(self):
        """A trades-read fault drops flow data — but the position-based
        verdict still computes. Empty book + bull → MISSED_TAILWIND fires."""
        class TradesBadStore(_FakeStore):
            def recent_trades(self, limit=50):
                raise RuntimeError("trades read down")
        store = TradesBadStore(
            portfolio={"cash": 987.0, "total_value": 987.0, "positions": []},
            open_positions=[],
        )
        with _patch_spy_mom(5.0):
            out = reporter._regime_leverage_fit_line(store)
        assert "MISSED_TAILWIND" in out

    def test_get_quant_signals_live_raise_degrades_to_neutral(self):
        """yfinance / strategy fault → spy_mom_20d=None → builder emits
        NEUTRAL (regime=unknown with book data) → silent."""
        store = _FakeStore(
            portfolio={"cash": 987.0, "total_value": 987.0, "positions": []},
            open_positions=[],
            recent_trades=[],
        )
        with patch(
            "paper_trader.strategy.get_quant_signals_live",
            side_effect=RuntimeError("yfinance timeout"),
        ):
            assert reporter._regime_leverage_fit_line(store) == ""

    def test_builder_raise_returns_empty(self):
        store = _FakeStore(
            portfolio={"cash": 987.0, "total_value": 987.0, "positions": []},
            open_positions=[],
            recent_trades=[],
        )
        with _patch_spy_mom(5.0), patch(
            "paper_trader.analytics.regime_leverage_fit_skill"
            ".build_regime_leverage_fit_skill",
            side_effect=RuntimeError("builder boom"),
        ):
            assert reporter._regime_leverage_fit_line(store) == ""

    def test_non_dict_result_returns_empty(self):
        store = _FakeStore(
            portfolio={"cash": 987.0, "total_value": 987.0, "positions": []},
        )
        with _patch_spy_mom(5.0), patch(
            "paper_trader.analytics.regime_leverage_fit_skill"
            ".build_regime_leverage_fit_skill",
            return_value=None,
        ):
            assert reporter._regime_leverage_fit_line(store) == ""

    def test_missing_headline_returns_empty(self):
        """An actionable verdict with no headline must degrade to silent —
        an empty headline would render as the bare prefix line, a lying
        green light."""
        store = _FakeStore(
            portfolio={"cash": 987.0, "total_value": 987.0, "positions": []},
        )
        with _patch_spy_mom(5.0), patch(
            "paper_trader.analytics.regime_leverage_fit_skill"
            ".build_regime_leverage_fit_skill",
            return_value={"verdict": "MISSED_TAILWIND", "headline": "   "},
        ):
            assert reporter._regime_leverage_fit_line(store) == ""

    def test_non_actionable_verdict_with_headline_returns_empty(self):
        """Even with a populated headline, ALIGNED / DEFENSIVE / NEUTRAL /
        NO_DATA stay silent — the suppression contract is verdict-based,
        not headline-based."""
        store = _FakeStore(
            portfolio={"cash": 987.0, "total_value": 987.0, "positions": []},
        )
        for v in ("ALIGNED", "DEFENSIVE", "NEUTRAL", "NO_DATA", "UNKNOWN"):
            with _patch_spy_mom(5.0), patch(
                "paper_trader.analytics.regime_leverage_fit_skill"
                ".build_regime_leverage_fit_skill",
                return_value={"verdict": v, "headline": "x"},
            ):
                assert reporter._regime_leverage_fit_line(store) == "", (
                    f"{v} must stay silent"
                )


# ── Wired into both summaries ────────────────────────────────────────────


class TestRegimeLeverageFitLineWiredIntoSummaries:
    """The line must be wired into both ``send_hourly_summary`` and
    ``send_daily_close`` — the operator must see the structural verdict
    on the surface they actually read (the ``_rebuy_regret_line`` /
    ``_repeat_loser_line`` precedent)."""

    def _wire(self, monkeypatch, store):
        sent: list[str] = []
        monkeypatch.setattr(reporter, "_send",
                            lambda body: sent.append(body) or True)
        monkeypatch.setattr(reporter, "get_store", lambda: store)
        monkeypatch.setattr(reporter.market, "benchmark_sp500",
                            lambda: 5000.0)
        return sent

    def test_hourly_summary_surfaces_missed_tailwind(self, monkeypatch):
        """The 100%-cash drought + bull tape state — exactly the 2026-05-24
        live pattern — must surface on the hourly."""
        store = _FakeStore(
            portfolio={"cash": 987.39, "total_value": 987.39, "positions": []},
            open_positions=[],
            recent_trades=[],
        )
        sent = self._wire(monkeypatch, store)
        with _patch_spy_mom(5.0):
            ok = reporter.send_hourly_summary()
        assert ok, "send_hourly_summary returned False"
        assert sent, "send must have been called"
        body = sent[0]
        assert "REGIME FIT" in body, (
            "MISSED_TAILWIND must appear in the hourly summary; got:\n"
            + body
        )
        assert "MISSED_TAILWIND" in body

    def test_hourly_summary_silent_when_aligned(self, monkeypatch):
        """An aligned book must produce no REGIME FIT line on the hourly —
        the suppression contract end-to-end."""
        store = _FakeStore(
            portfolio={"cash": 700.0, "total_value": 1000.0, "positions": []},
            open_positions=[
                {"ticker": "TQQQ", "type": "stock", "market_value": 300.0,
                 "qty": 5.0, "avg_cost": 60.0, "current_price": 60.0,
                 "unrealized_pl": 0.0},
            ],
            recent_trades=[],
        )
        sent = self._wire(monkeypatch, store)
        with _patch_spy_mom(5.0):
            reporter.send_hourly_summary()
        body = sent[0]
        assert "REGIME FIT" not in body, (
            "ALIGNED must not surface a REGIME FIT line; got:\n" + body
        )

    def test_daily_close_surfaces_dangerous_headwind(self, monkeypatch):
        """Bear regime + high leveraged exposure must surface on the daily
        close."""
        store = _FakeStore(
            portfolio={"cash": 500.0, "total_value": 1000.0, "positions": []},
            open_positions=[
                {"ticker": "TQQQ", "type": "stock", "market_value": 500.0,
                 "qty": 8.0, "avg_cost": 62.5, "current_price": 62.5,
                 "unrealized_pl": 0.0},
            ],
            recent_trades=[],
        )
        sent = self._wire(monkeypatch, store)
        with _patch_spy_mom(-5.0):
            ok = reporter.send_daily_close()
        assert ok, "send_daily_close returned False"
        body = sent[0]
        assert "REGIME FIT" in body
        assert "DANGEROUS_HEADWIND" in body

    def test_daily_close_silent_when_defensive(self, monkeypatch):
        """A correctly de-risked book on a bear tape stays silent on the
        daily close — DEFENSIVE is silent."""
        store = _FakeStore(
            portfolio={"cash": 1000.0, "total_value": 1000.0, "positions": []},
            open_positions=[],
            recent_trades=[],
        )
        sent = self._wire(monkeypatch, store)
        with _patch_spy_mom(-5.0):
            reporter.send_daily_close()
        body = sent[0]
        assert "REGIME FIT" not in body, (
            "DEFENSIVE must not surface a REGIME FIT line; got:\n" + body
        )

    def test_hourly_summary_survives_builder_fault(self, monkeypatch):
        """A builder fault must drop ONLY the REGIME FIT line, never the
        whole summary — the additive failure contract end-to-end."""
        store = _FakeStore(
            portfolio={"cash": 987.0, "total_value": 987.0, "positions": []},
            open_positions=[],
            recent_trades=[],
        )
        sent = self._wire(monkeypatch, store)
        with _patch_spy_mom(5.0), patch(
            "paper_trader.analytics.regime_leverage_fit_skill"
            ".build_regime_leverage_fit_skill",
            side_effect=RuntimeError("builder boom"),
        ):
            ok = reporter.send_hourly_summary()
        assert ok, "summary must still ship despite builder fault"
        body = sent[0]
        assert "REGIME FIT" not in body
        # The rest of the summary must still be present.
        assert "**HOURLY**" in body
        assert "Equity" in body
