"""Tests that ``strategy._build_payload`` and ``decide()`` wire the
``exit_proximity_block`` (from ``analytics.exit_proximity.build_exit_proximity``)
into the live decision prompt — placed AFTER buying_power and BEFORE
WATCHLIST PRICES.

The actual block rendering is locked by ``test_exit_proximity_prompt_block.py``;
this file pins the strategy-side wiring: kwarg accepted, payload ordering,
silence on a healthy book, and that ``decide()``'s diagnostics-fault arm
degrades to no block (never aborts the cycle).
"""
from __future__ import annotations

import pytest

from paper_trader import strategy


def _empty_snap():
    return {"positions": [], "cash": 1000.0,
            "open_value": 0.0, "total_value": 1000.0}


class TestBuildPayloadKwargAccepted:
    def test_build_payload_accepts_exit_proximity_block_kwarg(self):
        payload = strategy._build_payload(
            _empty_snap(), [], [], {}, {}, None, True,
            quant_signals={},
            exit_proximity_block="EXIT-PROX-MARKER hello",
        )
        assert "EXIT-PROX-MARKER hello" in payload

    def test_none_exit_proximity_block_emits_no_stray_text(self):
        payload = strategy._build_payload(
            _empty_snap(), [], [], {}, {}, None, True,
            quant_signals={}, exit_proximity_block=None,
        )
        # No marker, no stray "EXIT PROXIMITY" header, no literal None.
        assert "EXIT PROXIMITY" not in payload
        middle = payload.split("PORTFOLIO")[1].split("WATCHLIST")[0]
        assert "None" not in middle


class TestBuildPayloadOrdering:
    def test_exit_proximity_falls_after_buying_power_and_before_watchlist(self):
        # The block surfaces "what is at risk on the CURRENT book" — it
        # belongs right after "what cash can I add with" (buying_power)
        # and before the forward-looking market data (WATCHLIST PRICES).
        payload = strategy._build_payload(
            _empty_snap(), [], [], {}, {}, None, True,
            quant_signals={},
            buying_power_block="BUYING-POWER-MARKER",
            exit_proximity_block="EXIT-PROX-MARKER",
        )
        bp_idx = payload.index("BUYING-POWER-MARKER")
        ep_idx = payload.index("EXIT-PROX-MARKER")
        wl_idx = payload.index("WATCHLIST PRICES")
        assert bp_idx < ep_idx < wl_idx, (
            f"order should be BP < EP < WL but got {bp_idx},{ep_idx},{wl_idx}"
        )

    def test_exit_proximity_renders_alongside_other_advisory_blocks(self):
        payload = strategy._build_payload(
            _empty_snap(), [], [], {}, {}, None, True,
            quant_signals={},
            self_review_block="SELF-REVIEW",
            risk_mirror_block="RISK-MIRROR",
            event_calendar_block="EVENT-CAL",
            macro_calendar_block="MACRO-CAL",
            buying_power_block="BUYING-POWER",
            exit_proximity_block="EXIT-PROX",
        )
        # All blocks present and in the expected order.
        for marker in ("SELF-REVIEW", "RISK-MIRROR", "EVENT-CAL",
                       "MACRO-CAL", "BUYING-POWER", "EXIT-PROX",
                       "WATCHLIST PRICES"):
            assert marker in payload, f"missing {marker}"
        # Ordering: advisory stack → buying_power → exit_proximity → watchlist
        assert (
            payload.index("SELF-REVIEW")
            < payload.index("RISK-MIRROR")
            < payload.index("EVENT-CAL")
            < payload.index("MACRO-CAL")
            < payload.index("BUYING-POWER")
            < payload.index("EXIT-PROX")
            < payload.index("WATCHLIST PRICES")
        )


class TestDecideWiresExitProximity:
    """``decide()`` should call ``build_exit_proximity`` and forward its
    ``prompt_block`` to ``_build_payload``. We do not run the whole
    ``decide()`` (it needs a real store, market, Claude); instead we
    confirm via direct integration that the symbol is wired."""

    def test_strategy_imports_build_exit_proximity_callable(self):
        # If strategy.decide()'s import statement was renamed/removed, the
        # block would silently never render. Pin the import path.
        from paper_trader.analytics.exit_proximity import build_exit_proximity
        assert callable(build_exit_proximity)

    def test_decide_passes_exit_proximity_to_build_payload(self, monkeypatch):
        """End-to-end: decide() snapshot → exit_proximity builder →
        _build_payload(exit_proximity_block=...) chain holds."""
        from paper_trader.analytics import exit_proximity as ep_mod

        # Forge the at-risk shape so the builder emits a non-None block.
        # Includes unrealized_pl / pl_pct so _build_payload's position-line
        # f-string renders without KeyError (it consumes the post-mark
        # snapshot shape; an upstream test must mirror that contract).
        forged_positions = [{
            "ticker": "FORGED",
            "type": "stock",
            "qty": 5.0,
            "avg_cost": 100.0,
            "current_price": 97.0,         # below SL → AT_RISK_SL
            "stop_loss_price": 98.0,
            "take_profit_price": 103.0,
            "unrealized_pl": -15.0,
            "pl_pct": -3.0,
        }]
        forged_snap = {
            "cash": 500.0,
            "open_value": 485.0,
            "total_value": 985.0,
            "positions": forged_positions,
        }

        # Stub out external network / claude / store.
        monkeypatch.setattr(strategy, "_portfolio_snapshot",
                            lambda store: forged_snap)
        monkeypatch.setattr(strategy, "_check_and_execute_hard_exits",
                            lambda store, snap, **kwargs: [])
        monkeypatch.setattr(strategy.signals, "get_top_signals",
                            lambda n=20, hours=2, min_score=4.0: [])
        monkeypatch.setattr(strategy.signals, "get_urgent_articles",
                            lambda minutes=30: [])
        monkeypatch.setattr(strategy.signals, "ticker_sentiments",
                            lambda tickers, hours=4: [])
        monkeypatch.setattr(strategy.market, "get_prices",
                            lambda tickers: {t: 100.0 for t in tickers})
        monkeypatch.setattr(strategy.market, "get_futures_price",
                            lambda sym: 100.0)
        monkeypatch.setattr(strategy.market, "benchmark_sp500",
                            lambda: 5000.0)
        monkeypatch.setattr(strategy.market, "is_market_open",
                            lambda: True)
        monkeypatch.setattr(strategy, "get_quant_signals_live",
                            lambda tickers: {})
        # Force a NO_DECISION so we never spawn claude / mutate the store
        # write paths beyond record_decision/record_equity_point.
        monkeypatch.setattr(strategy, "host_saturated",
                            lambda: (True, "host saturated: test"))
        monkeypatch.setattr(strategy, "_ml_is_qualified",
                            lambda: (False, "not qualified"))

        # Capture what _build_payload sees.
        captured: dict = {}
        original_bp = strategy._build_payload

        def _spy(*args, **kwargs):
            captured["exit_proximity_block"] = kwargs.get(
                "exit_proximity_block"
            )
            return original_bp(*args, **kwargs)

        monkeypatch.setattr(strategy, "_build_payload", _spy)

        # Stub the singleton store's writes to in-memory.
        class _MemStore:
            def get_portfolio(self):
                return {"cash": forged_snap["cash"],
                        "total_value": forged_snap["total_value"],
                        "positions": [], "last_updated": ""}

            def open_positions(self):
                return forged_positions

            def recent_trades(self, n):
                return []

            def recent_decisions(self, limit=20):
                return []

            def equity_curve(self, limit=500):
                return []

            def record_decision(self, *a, **k):
                pass

            def record_equity_point(self, *a, **k):
                pass

            def update_position_marks(self, marks):
                pass

            def update_portfolio(self, *a, **k):
                pass

            def positions_needing_hard_exit(self):
                return []

        monkeypatch.setattr(strategy, "get_store", lambda: _MemStore())

        summary = strategy.decide()

        # The snapshot was AT_RISK_SL → builder emits a non-None block →
        # forwarded as exit_proximity_block.
        block = captured.get("exit_proximity_block")
        assert block is not None, (
            "decide() should pass the AT_RISK exit_proximity prompt_block "
            "to _build_payload"
        )
        assert "EXIT PROXIMITY (AT_RISK)" in block
        assert "FORGED" in block

        # And it should not abort the cycle.
        assert summary["status"] == "NO_DECISION"  # host_saturated path

    def test_decide_degrades_when_builder_raises(self, monkeypatch):
        """A diagnostics fault in build_exit_proximity must degrade to
        'no block this cycle', NEVER 'no decision this cycle' — the
        non-fatal-by-construction contract that protects every other
        prompt block."""
        from paper_trader.analytics import exit_proximity as ep_mod

        def _boom(*args, **kwargs):
            raise RuntimeError("synthetic-fault")

        # Replace the function symbol that strategy.decide() imports
        # locally (`from .analytics.exit_proximity import build_exit_proximity`).
        # Patching the module attribute reaches the import via the live
        # import-time reference inside decide().
        monkeypatch.setattr(ep_mod, "build_exit_proximity", _boom)

        forged_snap = {
            "cash": 1000.0, "open_value": 0.0,
            "total_value": 1000.0, "positions": []
        }
        monkeypatch.setattr(strategy, "_portfolio_snapshot",
                            lambda store: forged_snap)
        monkeypatch.setattr(strategy, "_check_and_execute_hard_exits",
                            lambda store, snap, **kwargs: [])
        monkeypatch.setattr(strategy.signals, "get_top_signals",
                            lambda n=20, hours=2, min_score=4.0: [])
        monkeypatch.setattr(strategy.signals, "get_urgent_articles",
                            lambda minutes=30: [])
        monkeypatch.setattr(strategy.signals, "ticker_sentiments",
                            lambda tickers, hours=4: [])
        monkeypatch.setattr(strategy.market, "get_prices",
                            lambda tickers: {t: 100.0 for t in tickers})
        monkeypatch.setattr(strategy.market, "get_futures_price",
                            lambda sym: 100.0)
        monkeypatch.setattr(strategy.market, "benchmark_sp500",
                            lambda: 5000.0)
        monkeypatch.setattr(strategy.market, "is_market_open",
                            lambda: True)
        monkeypatch.setattr(strategy, "get_quant_signals_live",
                            lambda tickers: {})
        monkeypatch.setattr(strategy, "host_saturated",
                            lambda: (True, "host saturated: test"))
        monkeypatch.setattr(strategy, "_ml_is_qualified",
                            lambda: (False, "not qualified"))

        # Stub the store.
        class _MemStore:
            def get_portfolio(self):
                return {"cash": 1000.0, "total_value": 1000.0,
                        "positions": [], "last_updated": ""}

            def open_positions(self):
                return []

            def recent_trades(self, n):
                return []

            def recent_decisions(self, limit=20):
                return []

            def equity_curve(self, limit=500):
                return []

            def record_decision(self, *a, **k):
                pass

            def record_equity_point(self, *a, **k):
                pass

            def update_position_marks(self, marks):
                pass

            def update_portfolio(self, *a, **k):
                pass

            def positions_needing_hard_exit(self):
                return []

        monkeypatch.setattr(strategy, "get_store", lambda: _MemStore())

        # No exception. Cycle still produced a (NO_DECISION) summary.
        summary = strategy.decide()
        assert summary["status"] == "NO_DECISION"
