"""Behaviour lock for paper_trader.analytics.game_plan.build_game_plan and the
/api/game-plan Flask endpoint.

build_game_plan is a *pure synthesis* layer: it fuses four diagnostics that
already exist as separate endpoints (the co-pilot action from
``_classify_action``, hold-discipline disposition flags, concentration/risk,
and the earnings calendar) into ONE prioritised, trader-facing action plan.

These tests assert the exact fusion + prioritisation behaviour on synthetic
component dicts — not "the call returned 200". They would fail if:
  * an overstayed losing position did not escalate a HOLD to a sell-side review,
  * the single largest position under HIGH concentration was not pushed to TRIM,
  * a held name with imminent earnings did not gain priority/awareness,
  * opportunities leaked held names or ignored conviction ordering,
  * the deterministic priority sort reordered ties,
  * a stronger suggestion verb (EXIT) was silently downgraded,
  * garbage inputs raised instead of degrading.
"""
from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.game_plan import build_game_plan

NOW = datetime(2026, 5, 18, 6, 0, 0, tzinfo=timezone.utc)


def _pos(ticker, qty, avg, cur, upl, opened="2026-05-14T00:00:00+00:00", typ="stock"):
    return {
        "ticker": ticker, "type": typ, "qty": qty, "avg_cost": avg,
        "current_price": cur, "unrealized_pl": upl, "opened_at": opened,
    }


def _hd(state="DISCIPLINED", drag=0.0, n_over=0, cards=None):
    return {
        "state": state, "disposition_drag_usd": drag, "n_overstayed": n_over,
        "positions": cards or [],
    }


def _hd_card(ticker, overstayed, mult=None, age=2.0, upl=-1.0, losing=True):
    return {
        "ticker": ticker, "overstayed": overstayed, "overstay_mult": mult,
        "age_days": age, "unrealized_pl": upl, "is_losing": losing,
    }


def _conc(top1="", sev="NONE", warn=False, t1=0.0, t3=0.0, cash_pct=20.0, sectors=None):
    return {
        "top1_ticker": top1, "severity": sev, "warning": warn,
        "top1_pct": t1, "top3_pct": t3, "cash_pct": cash_pct,
        "sector_pct": sectors or {},
    }


def _cls(action, conv=0.5, reasons=None, held_qty=1.0, news=0.0, urgent=False, price=100.0):
    return {
        "action": action, "conviction": conv, "reasons": reasons or [],
        "held_qty": held_qty, "news_max_score": news, "news_urgent": urgent,
        "price": price,
    }


# ───────────────────────── per-position fusion ─────────────────────────

def test_overstayed_loser_escalates_hold_to_sell_side_review():
    """A co-pilot HOLD on a position the disposition detector flags as an
    overstayed loser must escalate to a sell-side review and rank first."""
    pos = [_pos("LITE", 0.61, 980.9, 970.71, -6.21)]
    hd = _hd("DISPOSITION_DRAG", drag=-6.21, n_over=1,
             cards=[_hd_card("LITE", True, mult=6.59, age=3.51, upl=-6.21)])
    gp = build_game_plan(
        positions=pos, total_value=972.69, cash=18.49,
        hold_discipline=hd, concentration=_conc(),
        earnings_events=[], classified={"LITE": _cls("HOLD", 0.4)}, now=NOW,
    )
    pa = gp["position_actions"]
    assert len(pa) == 1
    assert pa[0]["ticker"] == "LITE"
    assert pa[0]["action"] == "REVIEW EXIT"
    # overstay(3) + losing(1) ⇒ priority 4
    assert pa[0]["priority"] == 4
    assert any("disposition trap" in r for r in pa[0]["reasons"])
    assert gp["state"] == "ACTIONS_PRESENT"


def test_suggestion_exit_is_not_downgraded_by_overstay():
    """EXIT (strongest sell verb) must survive — overstay only escalates, it
    never weakens a verb the co-pilot already made stronger."""
    pos = [_pos("AMD", 2, 100, 80, -40)]
    hd = _hd("DISPOSITION_DRAG", drag=-40, n_over=1,
             cards=[_hd_card("AMD", True, mult=3.0, age=9.0, upl=-40)])
    gp = build_game_plan(
        positions=pos, total_value=1000, cash=500, hold_discipline=hd,
        concentration=_conc(), earnings_events=[],
        classified={"AMD": _cls("EXIT", 0.9)}, now=NOW,
    )
    assert gp["position_actions"][0]["action"] == "EXIT"


def test_concentration_top1_forces_trim_and_high_directive():
    pos = [_pos("MU", 0.5, 724, 724, 0.0)]
    conc = _conc(top1="MU", sev="HIGH", warn=True, t1=65.0, t3=98.1, cash_pct=2.0)
    gp = build_game_plan(
        positions=pos, total_value=1000, cash=20, hold_discipline=_hd(),
        concentration=conc, earnings_events=[],
        classified={"MU": _cls("HOLD", 0.4)}, now=NOW,
    )
    assert gp["position_actions"][0]["action"] == "TRIM"
    kinds = {d["kind"]: d for d in gp["portfolio_directives"]}
    assert "CONCENTRATION" in kinds
    assert kinds["CONCENTRATION"]["severity"] == "HIGH"
    # cash 2% < 5% ⇒ a dry-powder directive too
    assert "DRY_POWDER" in kinds


def test_held_earnings_within_3d_raises_priority_and_adds_reason():
    """Imminent earnings on a HELD name is awareness — it bumps priority and
    annotates, but does NOT by itself invent a sell verb."""
    pos = [_pos("NVDA", 1, 200, 210, 10.0)]
    ev = [{"ticker": "NVDA", "days_away": 1.7,
           "earnings_date": "2026-05-20T00:00:00+00:00",
           "held": True, "tier": "HELD_SOON"}]
    gp = build_game_plan(
        positions=pos, total_value=1000, cash=500, hold_discipline=_hd(),
        concentration=_conc(), earnings_events=ev,
        classified={"NVDA": _cls("HOLD", 0.5)}, now=NOW,
    )
    card = gp["position_actions"][0]
    assert card["action"] == "HOLD"          # awareness, not a directive
    assert card["priority"] >= 2             # earnings ≤3d ⇒ +2
    assert any("earnings in 1.7d" in r for r in card["reasons"])


# ───────────────────────── opportunities ─────────────────────────

def test_opportunities_exclude_held_and_sort_by_conviction():
    classified = {
        "LITE": _cls("HOLD", 0.4, held_qty=1.0),               # held → not an opp
        "SOXL": _cls("BUY", 0.82, held_qty=0.0, news=8.0, price=30.0),
        "TQQQ": _cls("WATCH", 0.55, held_qty=0.0, price=75.0),
        "QQQ": _cls("WATCH", 0.20, held_qty=0.0),               # below the floor
    }
    gp = build_game_plan(
        positions=[_pos("LITE", 1, 10, 10, 0.0)], total_value=1000, cash=500,
        hold_discipline=_hd(), concentration=_conc(), earnings_events=[],
        classified=classified, now=NOW,
    )
    opp_tickers = [o["ticker"] for o in gp["opportunities"]]
    assert "LITE" not in opp_tickers
    assert opp_tickers[:2] == ["SOXL", "TQQQ"]   # conviction-desc
    assert all(o["action"] in ("BUY", "WATCH") for o in gp["opportunities"])


# ───────────────────────── state / ordering / robustness ─────────────────────────

def test_steady_state_when_nothing_actionable():
    gp = build_game_plan(
        positions=[_pos("SPY", 1, 400, 405, 5.0)], total_value=1000, cash=595,
        hold_discipline=_hd(), concentration=_conc(),
        earnings_events=[], classified={"SPY": _cls("HOLD", 0.4)}, now=NOW,
    )
    assert gp["state"] == "STEADY"
    assert gp["n_actions"] == 0
    assert "steady" in gp["headline"].lower()


def test_no_data_when_empty_book_and_no_setups():
    gp = build_game_plan(
        positions=[], total_value=1000, cash=1000, hold_discipline=_hd(),
        concentration=_conc(), earnings_events=[], classified={}, now=NOW,
    )
    assert gp["state"] == "NO_DATA"
    assert gp["position_actions"] == []


def test_priority_ordering_is_deterministic():
    """Overstayed loser must outrank a clean holding regardless of input order."""
    pos = [_pos("CLEAN", 1, 100, 110, 10.0),
           _pos("TRAP", 1, 100, 70, -30.0)]
    hd = _hd("DISPOSITION_DRAG", drag=-30, n_over=1,
             cards=[_hd_card("TRAP", True, mult=4.0, age=8.0, upl=-30)])
    gp = build_game_plan(
        positions=pos, total_value=1000, cash=500, hold_discipline=hd,
        concentration=_conc(), earnings_events=[],
        classified={"CLEAN": _cls("HOLD"), "TRAP": _cls("HOLD")}, now=NOW,
    )
    assert [c["ticker"] for c in gp["position_actions"]] == ["TRAP", "CLEAN"]


def test_never_raises_on_garbage_inputs():
    gp = build_game_plan(
        positions=[{"ticker": "X", "qty": None, "current_price": "bad",
                    "unrealized_pl": None, "type": "stock"}],
        total_value=0.0, cash=None,
        hold_discipline={"positions": [{"ticker": "X", "overstayed": "yes"}]},
        concentration={"severity": None, "top1_ticker": None},
        earnings_events=[{"ticker": "X"}],
        classified={"X": {"action": None, "conviction": "nope"}}, now=NOW,
    )
    assert isinstance(gp, dict)
    assert "state" in gp and "position_actions" in gp


# ───────────────────────── Flask endpoint wiring ─────────────────────────

@pytest.fixture
def seeded_client(tmp_path, monkeypatch):
    """Fresh Store at a temp DB + offline Flask test client. Mocks the news
    and market layers so the endpoint test never touches the network."""
    from paper_trader import store as store_mod
    from paper_trader.store import Store

    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)
    s = Store()
    # A deeply-losing single-name book == 100% concentration + a loss: the
    # plan must not read this as a calm HOLD.
    s.record_trade("LITE", "BUY", 1.0, 1000.0)
    s.upsert_position("LITE", "stock", 1.0, 1000.0)
    pid = s.open_positions()[0]["id"]
    s.update_position_marks({pid: (900.0, -100.0)})  # -$100 unrealized

    from paper_trader import dashboard, signals as sig_mod, market as mkt_mod
    from paper_trader import strategy as strat_mod
    monkeypatch.setattr(sig_mod, "get_top_signals", lambda *a, **k: [])
    monkeypatch.setattr(mkt_mod, "get_prices", lambda *a, **k: {"LITE": 900.0})
    monkeypatch.setattr(strat_mod, "get_quant_signals_live",
                        lambda *a, **k: {})
    monkeypatch.setattr(dashboard, "get_store", lambda: s)
    dashboard.app.config.update(TESTING=True)
    return dashboard.app.test_client()


def test_endpoint_returns_plan_and_flags_seeded_trap(seeded_client):
    r = seeded_client.get("/api/game-plan")
    assert r.status_code == 200
    body = r.get_json()
    for key in ("as_of", "state", "headline", "n_actions",
                "position_actions", "portfolio_directives", "opportunities"):
        assert key in body, f"missing {key}"
    lite = [c for c in body["position_actions"] if c["ticker"] == "LITE"]
    assert lite, "seeded LITE position absent from the plan"
    # A 10-day-old -$100 single-name book must not read as HOLD/steady.
    assert lite[0]["action"] != "HOLD"
