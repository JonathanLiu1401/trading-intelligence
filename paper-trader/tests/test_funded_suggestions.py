"""Exact-value tests for analytics/funded_suggestions.build_funded_suggestions.

The feature pairs each unfundable BUY/ADD idea with the *specific minimum
prefix of sales* (from capital_paralysis' desk-cut-priority unlock ladder) that
funds it. Pins: FREE → all FUNDED no chain; PINNED single-sale-enough vs
needs-two vs whole-ladder-insufficient (→ UNFUNDABLE, enough=False, full
ladder); EMPTY/NO_DATA → UNFUNDABLE; only BUY/ADD are funding-checked
(HOLD/WATCH/TRIM/EXIT bypass — they don't consume buying power); deterministic
conviction tie-break for top_actionable; recommended_pairing only when PINNED.
"""
from paper_trader.analytics.funded_suggestions import build_funded_suggestions


def _sugg(ticker, action, conviction):
    return {"ticker": ticker, "action": action, "conviction": conviction,
            "price": 100.0, "held_qty": 0.0, "reasons": []}


def _ladder(*rungs):
    """rungs: (ticker, frees_usd) in desk cut-priority; cumulative auto-summed."""
    out, cum = [], 0.0
    for tk, frees in rungs:
        cum = round(cum + frees, 2)
        out.append({"ticker": tk, "type": "stock", "frees_usd": frees,
                    "pl_pct": -5.0, "cumulative_freed_usd": cum,
                    "restores_action_alone": False})
    return out


def _paralysis(state, *, can_act, total_value, cash=6.23, ladder=None,
               recommended=None):
    return {"state": state, "can_act_on_signal": can_act,
            "total_value": total_value, "cash": cash, "cash_pct": 0.6,
            "deployed_pct": 99.4, "min_actionable_usd": 9.73,
            "unlock_ladder": ladder or [],
            "recommended_unlock": recommended}


def test_free_all_actionable_funded_no_unlock_chain():
    sg = [_sugg("NVDA", "BUY", 0.70), _sugg("AMD", "ADD", 0.30),
          _sugg("MSFT", "HOLD", 0.0), _sugg("T", "WATCH", 0.1),
          _sugg("X", "TRIM", 0.4), _sugg("Z", "EXIT", 0.9)]
    # cash genuinely covers every advisory notional → still all FUNDED.
    # (Originally cash was left at the helper default 6.23, which is an
    #  under-specified FREE fixture: a FREE book with $6 cash can't actually
    #  fund a $700 notional. Specifying cash makes the FUNDED claim honest.)
    p = _paralysis("FREE", can_act=True, total_value=1000.0, cash=1000.0)
    out = build_funded_suggestions(sg, p)

    assert out["state"] == "FREE"
    assert out["can_act"] is True
    assert out["n_actionable"] == 2          # only BUY + ADD
    assert out["n_funded"] == 2
    assert out["n_unlockable"] == 0
    assert out["n_unfundable"] == 0
    ideas = out["ideas"]
    assert [i["ticker"] for i in ideas] == ["NVDA", "AMD"]   # conviction desc
    for i in ideas:
        assert i["fundability"] == "FUNDED"
        assert i["funded_by"] == []
        assert i["frees_usd"] == 0.0
        assert i["enough"] is True
    # notional is advisory but still computed
    assert ideas[0]["suggested_notional_usd"] == round(0.70 * 1000.0, 2)
    assert out["top_actionable"] == {"ticker": "NVDA", "action": "BUY",
                                     "conviction": 0.70}
    assert out["recommended_pairing"] is None  # pairing is PINNED-only


def test_pinned_single_sale_is_enough():
    sg = [_sugg("NVDA", "BUY", 0.50)]            # notional = 0.5 * 1000 = 500
    lad = _ladder(("LITE", 786.28), ("AMAT", 120.0))
    p = _paralysis("PINNED", can_act=False, total_value=1000.0, ladder=lad,
                   recommended={"ticker": "LITE"})
    out = build_funded_suggestions(sg, p)

    idea = out["ideas"][0]
    assert idea["suggested_notional_usd"] == 500.0
    assert idea["fundability"] == "UNLOCKABLE"
    assert idea["funded_by"] == ["LITE"]        # minimum prefix
    assert idea["frees_usd"] == 786.28
    assert idea["enough"] is True
    assert out["n_unlockable"] == 1
    assert out["recommended_pairing"] == {
        "sell": "LITE", "buy": "NVDA", "frees_usd": 786.28,
        "buy_conviction": 0.50}


def test_pinned_needs_two_sales():
    sg = [_sugg("SOXL", "BUY", 0.90)]            # notional = 900
    lad = _ladder(("A", 500.0), ("B", 450.0))    # cum: 500, 950
    p = _paralysis("PINNED", can_act=False, total_value=1000.0, ladder=lad,
                   recommended={"ticker": "A"})
    out = build_funded_suggestions(sg, p)

    idea = out["ideas"][0]
    assert idea["suggested_notional_usd"] == 900.0
    assert idea["fundability"] == "UNLOCKABLE"
    assert idea["funded_by"] == ["A", "B"]       # minimum prefix is both
    assert idea["frees_usd"] == 950.0
    assert idea["enough"] is True


def test_pinned_whole_ladder_insufficient_is_unfundable():
    sg = [_sugg("TQQQ", "BUY", 0.95)]            # notional = 950
    lad = _ladder(("A", 500.0), ("B", 300.0))    # cum max 800 < 950
    p = _paralysis("PINNED", can_act=False, total_value=1000.0, ladder=lad,
                   recommended={"ticker": "A"})
    out = build_funded_suggestions(sg, p)

    idea = out["ideas"][0]
    assert idea["fundability"] == "UNFUNDABLE"
    assert idea["funded_by"] == ["A", "B"]       # full ladder, best effort
    assert idea["frees_usd"] == 800.0
    assert idea["enough"] is False
    assert out["n_unfundable"] == 1


def test_pinned_empty_ladder_is_unfundable():
    sg = [_sugg("NVDA", "BUY", 0.40)]
    p = _paralysis("PINNED", can_act=False, total_value=1000.0, ladder=[])
    out = build_funded_suggestions(sg, p)
    idea = out["ideas"][0]
    assert idea["fundability"] == "UNFUNDABLE"
    assert idea["funded_by"] == []
    assert idea["frees_usd"] == 0.0
    assert idea["enough"] is False
    assert out["recommended_pairing"] is None    # no recommended_unlock


def test_empty_state_all_unfundable():
    sg = [_sugg("NVDA", "BUY", 0.6), _sugg("AMD", "ADD", 0.2)]
    p = _paralysis("EMPTY", can_act=False, total_value=0.0)
    out = build_funded_suggestions(sg, p)
    assert out["n_actionable"] == 2
    assert out["n_unfundable"] == 2
    assert all(i["fundability"] == "UNFUNDABLE" for i in out["ideas"])
    assert out["recommended_pairing"] is None


def test_no_data_state_unfundable_but_top_actionable_still_reported():
    sg = [_sugg("NVDA", "BUY", 0.8)]
    p = _paralysis("NO_DATA", can_act=False, total_value=0.0)
    out = build_funded_suggestions(sg, p)
    assert out["ideas"][0]["fundability"] == "UNFUNDABLE"
    assert out["top_actionable"] == {"ticker": "NVDA", "action": "BUY",
                                     "conviction": 0.8}


def test_conviction_tie_break_is_deterministic():
    # Same conviction → alphabetical ticker wins (stable, reproducible).
    sg = [_sugg("ZZZ", "BUY", 0.60), _sugg("AAA", "BUY", 0.60),
          _sugg("MMM", "ADD", 0.60)]
    p = _paralysis("FREE", can_act=True, total_value=1000.0, cash=1000.0)
    out = build_funded_suggestions(sg, p)
    assert [i["ticker"] for i in out["ideas"]] == ["AAA", "MMM", "ZZZ"]
    assert out["top_actionable"]["ticker"] == "AAA"


# ── Honest buying-power: can_act only means cash ≥ a tiny floor, NOT that cash
#    covers the advisory notional. The panel must not paint a $856 idea green
#    "FUNDED — no sale required" on $18 cash. (live repro 2026-05-18) ──────────

def test_free_but_cash_below_notional_is_partial_with_unlock_chain():
    # Live repro: $973 book, $18.49 cash, BUY NVDA conv 0.88 → $856 notional.
    sg = [_sugg("NVDA", "BUY", 0.88)]
    lad = _ladder(("LITE", 592.13), ("MU", 362.06))   # cum: 592.13, 954.19
    p = _paralysis("FREE", can_act=True, total_value=973.0, cash=18.49,
                   ladder=lad, recommended={"ticker": "LITE"})
    out = build_funded_suggestions(sg, p)

    idea = out["ideas"][0]
    assert idea["suggested_notional_usd"] == round(0.88 * 973.0, 2)  # 856.24
    assert idea["fundability"] == "PARTIAL"          # NOT the old green FUNDED
    # minimum sale prefix s.t. cash + cumulative_freed covers the notional
    assert idea["funded_by"] == ["LITE", "MU"]
    assert idea["frees_usd"] == 954.19
    assert idea["enough"] is True
    assert out["n_partial"] == 1
    assert out["n_funded"] == 0
    # headline must not keep claiming it is "fundable from cash now"
    assert "fundable from cash now" not in out["headline"]
    assert "NVDA" in out["headline"]


def test_free_cash_covers_notional_stays_fully_funded():
    sg = [_sugg("AMD", "BUY", 0.50)]                  # notional = 500
    p = _paralysis("FREE", can_act=True, total_value=1000.0, cash=1000.0)
    out = build_funded_suggestions(sg, p)

    idea = out["ideas"][0]
    assert idea["fundability"] == "FUNDED"            # cash truly covers it
    assert idea["funded_by"] == []
    assert idea["frees_usd"] == 0.0
    assert idea["enough"] is True
    assert out["n_funded"] == 1
    assert out["n_partial"] == 0


def test_free_cash_below_notional_ladder_insufficient_is_partial_not_enough():
    sg = [_sugg("TQQQ", "BUY", 0.95)]                 # notional = 950
    lad = _ladder(("A", 100.0), ("B", 50.0))          # cum max 150
    p = _paralysis("FREE", can_act=True, total_value=1000.0, cash=10.0,
                   ladder=lad)
    out = build_funded_suggestions(sg, p)

    idea = out["ideas"][0]
    # cash funds *part* of it → still PARTIAL (better than UNFUNDABLE: there
    # IS cash), but selling the whole ladder still can't reach the notional.
    assert idea["fundability"] == "PARTIAL"
    assert idea["funded_by"] == ["A", "B"]            # full ladder, best effort
    assert idea["frees_usd"] == 150.0
    assert idea["enough"] is False
    assert out["n_partial"] == 1


def test_free_cash_below_notional_no_ladder_is_partial_not_enough():
    sg = [_sugg("NVDA", "BUY", 0.50)]                 # notional = 500
    p = _paralysis("FREE", can_act=True, total_value=1000.0, cash=20.0,
                   ladder=[])
    out = build_funded_suggestions(sg, p)

    idea = out["ideas"][0]
    assert idea["fundability"] == "PARTIAL"
    assert idea["funded_by"] == []
    assert idea["frees_usd"] == 0.0
    assert idea["enough"] is False


def test_non_buy_actions_never_funding_checked():
    sg = [_sugg("X", "HOLD", 0.0), _sugg("Y", "WATCH", 0.9),
          _sugg("Z", "TRIM", 0.5), _sugg("W", "EXIT", 0.7)]
    p = _paralysis("PINNED", can_act=False, total_value=1000.0,
                   ladder=_ladder(("A", 999.0)))
    out = build_funded_suggestions(sg, p)
    assert out["n_actionable"] == 0
    assert out["ideas"] == []
    assert out["top_actionable"] is None
    assert out["recommended_pairing"] is None
    assert "no actionable" in out["headline"].lower()
