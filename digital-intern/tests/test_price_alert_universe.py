"""Price-alert universe covers every live config/portfolio.json holding.

Regression guard for the missed-price-alert bug (2026-05-21): held positions
added in the trading UI — GOOG, COHR, NVDL — were absent from BOTH the frozen
``daemon.PORTFOLIO_TICKERS`` tuple AND ``config/watchlist.json`` (which drives
``get_stock_data()``), so ``price_alert_worker`` fired NO 3% price alert on
them: a silent blind spot on names the analyst has real money in.

The fix unions ``PORTFOLIO_TICKERS`` with ``ml.features.LIVE_PORTFOLIO_TICKERS``
(the SSOT that reads positions + option underlyings + sector_watchlist from
``config/portfolio.json``) and fetches any ticker the watchlist sweep missed
directly via ``stock_data._fetch_one``.
"""
import json
from pathlib import Path

import daemon


def _config_positions() -> set[str]:
    pf = json.loads(
        (Path(daemon.__file__).resolve().parent / "config" / "portfolio.json")
        .read_text()
    )
    return {
        (p.get("ticker") or "").strip().upper()
        for p in pf.get("positions", [])
        if p.get("ticker")
    }


def test_universe_covers_every_held_position():
    """Every open position in config/portfolio.json is price-alerted."""
    universe = set(daemon._price_alert_universe())
    missing = _config_positions() - universe
    assert not missing, f"held positions absent from price alerts: {sorted(missing)}"


def test_universe_is_superset_of_static_tuple():
    """The frozen PORTFOLIO_TICKERS tuple is still fully covered — the fix
    only widens the universe, never narrows it."""
    universe = set(daemon._price_alert_universe())
    assert set(daemon.PORTFOLIO_TICKERS) <= universe


def test_universe_is_sorted_and_deterministic():
    u = daemon._price_alert_universe()
    assert u == sorted(u)
    assert u == daemon._price_alert_universe()


def test_worker_fetches_held_tickers_missing_from_watchlist_sweep(monkeypatch):
    """price_alert_worker must directly fetch a held ticker that
    get_stock_data() (watchlist.json-driven) does not return — otherwise the
    ``if not row: continue`` skip silently drops it from alerting."""
    # get_stock_data returns NOTHING — simulates every held ticker being
    # absent from config/watchlist.json (the live GOOG/COHR/NVDL case).
    monkeypatch.setattr(daemon, "get_stock_data", lambda: {"equities": []})

    fetched: list[str] = []

    def _fake_fetch(tkr: str):
        fetched.append(tkr)
        return {"ticker": tkr, "price": 100.0}

    monkeypatch.setattr(daemon, "_fetch_one", _fake_fetch)
    monkeypatch.setattr(daemon, "discord_send", lambda *a, **k: True)
    monkeypatch.setattr(daemon, "_last_prices", {})
    monkeypatch.setattr(daemon, "_running", True)

    def _stop_after_one_cycle(*_a, **_k):
        daemon._running = False

    monkeypatch.setattr(daemon, "_sleep", _stop_after_one_cycle)

    daemon.price_alert_worker(store=None)

    held = _config_positions()
    assert held <= set(fetched), (
        f"held positions never fetched for price alerts: "
        f"{sorted(held - set(fetched))}"
    )
