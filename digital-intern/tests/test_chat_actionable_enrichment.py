"""Pure-helper tests for the /api/chat actionable enrichment (web_server.py).

Three total/pure helpers, extracted from the one big inline chat closure so
their behaviour is locked without standing up Flask or cross-fetching :8090
(the `_tail_risk_chat_lines` / `_behavioural_chat_lines` precedent):

* ``_paper_trader_position_lines`` — renders the live trader's OPEN book
  from the **marked** ``portfolio.positions`` array (carries ``stale_mark``
  and a real ``pl_pct``) instead of the raw top-level ``positions`` array
  (which has neither). Two discriminating locks:
    1. the **always-(0.0%) bug**: the raw array has no ``pl_pct`` key, so
       the old inline code printed ``(0.0%)`` for every stock regardless of
       actual P/L. A real ``pl_pct=-1.04`` must surface, not ``0.0``.
    2. the **stale-mark misread**: a position whose live price lookup
       failed (``stale_mark=True``, ``current_price == avg_cost``,
       P/L $0.00) looks exactly like a genuinely flat position. The chat —
       the user's primary surface — must annotate it, mirroring the
       trader-side prompt's ``[STALE MARK …]`` suffix and the reporter's
       ``⚠ STALE`` (both shipped for this exact live MU pathology).

* ``_game_plan_chat_lines`` — the system's own prioritised next-session
  action plan (``/api/game-plan``) composed **verbatim** (paper-trader
  invariant #10): the builder's ``headline`` and each HIGH directive's
  ``text`` pass through unchanged. An inline re-derivation that drifts
  from the trader endpoint fails here.

* ``_hold_discipline_chat_lines`` — the disposition-trap verdict
  (``/api/hold-discipline``). Mirrors the reporter's ``_hold_discipline_line``
  contract exactly: emit the verbatim headline ONLY on ``DISPOSITION_DRAG``;
  ``DISCIPLINED`` / ``INSUFFICIENT`` / ``NO_DATA`` / error → silence (a
  "you're fine" verdict is not chat-worthy noise).

All three obey the shared total contract: a non-dict, an ``error`` key, a
missing ``state``, or a ``NO_DATA`` gate contributes nothing — the block is
omitted, never an exception into the chat handler.
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dashboard.web_server import (
    _game_plan_chat_lines,
    _hold_discipline_chat_lines,
    _paper_trader_position_lines,
)


class TestPaperTraderPositionLines:
    def _marked_state(self) -> dict:
        # Mirrors a real :8090 /api/state body: top-level `positions` is the
        # raw store read (no pl_pct / stale_mark); `portfolio.positions` is
        # the snapshot-marked array.
        return {
            "positions": [
                {"ticker": "MU", "type": "stock", "qty": 0.5,
                 "avg_cost": 724.12, "current_price": 724.12,
                 "unrealized_pl": 0.0},
                {"ticker": "LITE", "type": "stock", "qty": 0.61,
                 "avg_cost": 980.90, "current_price": 970.71,
                 "unrealized_pl": -6.21},
            ],
            "portfolio": {
                "positions": [
                    {"ticker": "MU", "type": "stock", "qty": 0.5,
                     "avg_cost": 724.12, "current_price": 724.12,
                     "unrealized_pl": 0.0, "pl_pct": 0.0,
                     "stale_mark": True},
                    {"ticker": "LITE", "type": "stock", "qty": 0.61,
                     "avg_cost": 980.90, "current_price": 970.71,
                     "unrealized_pl": -6.21, "pl_pct": -1.04,
                     "stale_mark": False},
                ]
            },
        }

    def test_uses_marked_array_real_pct_not_zero_bug(self):
        """The raw top-level array has no pl_pct → the old inline code
        printed (0.0%) for every stock. The marked array must surface the
        real -1.04% on LITE."""
        out = _paper_trader_position_lines(self._marked_state())
        joined = "\n".join(out)
        assert out[0] == "Open positions:"
        lite = [ln for ln in out if "LITE" in ln][0]
        assert "-1.04%" in lite or "-1.0%" in lite
        # The discriminating regression: LITE is down ~1% — it must NOT
        # render as a flat 0.0% (the bug the raw-array read caused).
        assert "(0.0%)" not in lite

    def test_stale_mark_is_annotated(self):
        """MU's mark equals its cost and P/L is $0.00 only because the live
        price lookup failed (stale_mark=True). The chat must flag it so the
        analyst doesn't confidently report MU as flat."""
        out = _paper_trader_position_lines(self._marked_state())
        mu = [ln for ln in out if ln.strip().startswith("MU")][0]
        assert "STALE" in mu.upper()
        # A genuinely-marked position is never falsely flagged.
        lite = [ln for ln in out if "LITE" in ln][0]
        assert "STALE" not in lite.upper()

    def test_option_position_formatting(self):
        st = {
            "portfolio": {
                "positions": [
                    {"ticker": "NVDA", "type": "call", "qty": 2,
                     "strike": 1000, "expiry": "2026-06-19",
                     "avg_cost": 25.0, "current_price": 31.5,
                     "unrealized_pl": 1300.0, "pl_pct": 26.0,
                     "stale_mark": False},
                ]
            }
        }
        out = _paper_trader_position_lines(st)
        line = [ln for ln in out if "NVDA" in ln][0]
        assert "CALL" in line
        assert "1000" in line and "2026-06-19" in line
        assert "26.0%" in line

    def test_falls_back_to_raw_when_marked_empty(self):
        """A degraded store hands get_portfolio() positions=[] while the
        raw open_positions() read still has rows — keep showing them
        (current behaviour preserved), just without a fabricated %."""
        st = {
            "positions": [
                {"ticker": "AMD", "type": "stock", "qty": 3,
                 "avg_cost": 150.0, "current_price": 165.0,
                 "unrealized_pl": 45.0},
            ],
            "portfolio": {"positions": []},
        }
        out = _paper_trader_position_lines(st)
        amd = [ln for ln in out if "AMD" in ln][0]
        assert "AMD" in amd
        # No pl_pct available in the raw array → must NOT invent (0.0%).
        assert "(0.0%)" not in amd

    def test_no_positions_returns_none_line(self):
        assert _paper_trader_position_lines(
            {"portfolio": {"positions": []}, "positions": []}
        ) == ["Open positions: (none)"]

    def test_total_contract(self):
        assert _paper_trader_position_lines(None) == ["Open positions: (none)"]
        assert _paper_trader_position_lines("nope") == [
            "Open positions: (none)"]
        assert _paper_trader_position_lines({}) == ["Open positions: (none)"]

    def test_caps_at_15_positions(self):
        st = {"portfolio": {"positions": [
            {"ticker": f"T{i}", "type": "stock", "qty": 1,
             "avg_cost": 10.0, "current_price": 11.0,
             "unrealized_pl": 1.0, "pl_pct": 10.0, "stale_mark": False}
            for i in range(30)
        ]}}
        out = _paper_trader_position_lines(st)
        # header + 15 position lines
        assert len(out) == 16


class TestGamePlanChatLines:
    def _actions_present(self) -> dict:
        return {
            "state": "ACTIONS_PRESENT",
            "headline": "3 action(s) for the next session: TRIM LITE · EXIT MU",
            "n_actions": 3,
            "position_actions": [
                {"ticker": "LITE", "action": "TRIM", "priority": 3,
                 "conviction": 0.7, "unrealized_pl": -6.21,
                 "pct_port": 60.9, "reasons": ["overstayed"]},
                {"ticker": "MU", "action": "EXIT", "priority": 2,
                 "conviction": 0.6, "unrealized_pl": 0.0,
                 "pct_port": 37.2, "reasons": ["thesis broken"]},
                {"ticker": "AMD", "action": "HOLD", "priority": 0,
                 "conviction": 0.5, "unrealized_pl": 12.0,
                 "pct_port": 0.0, "reasons": []},
            ],
            "portfolio_directives": [
                {"kind": "CONCENTRATION", "severity": "HIGH",
                 "text": "Top position LITE is 60.9% of book (top-3 98.1%) "
                         "— single-name risk; consider trimming into "
                         "strength."},
                {"kind": "DRY_POWDER", "severity": "MEDIUM",
                 "text": "Only 1.9% cash — limited room to act."},
            ],
            "opportunities": [
                {"ticker": "NVDA", "action": "BUY", "conviction": 0.8,
                 "news_max_score": 8.0, "price": 1010.0, "reasons": []},
            ],
        }

    def test_verbatim_headline_and_high_directive(self):
        gp = self._actions_present()
        out = _game_plan_chat_lines(gp)
        joined = "\n".join(out)
        # Headline composed verbatim (invariant #10 — no re-derivation).
        assert gp["headline"] in joined
        # HIGH directive text passes through unchanged.
        assert gp["portfolio_directives"][0]["text"] in joined
        # MEDIUM directive is NOT promoted as a top-priority bullet.
        assert "Only 1.9% cash" not in joined

    def test_actionable_positions_listed_hold_excluded(self):
        out = _game_plan_chat_lines(self._actions_present())
        joined = "\n".join(out)
        assert "TRIM LITE" in joined
        assert "EXIT MU" in joined
        # A HOLD action is not a "do something" line.
        assert "HOLD AMD" not in joined

    def test_opportunities_surfaced(self):
        out = _game_plan_chat_lines(self._actions_present())
        assert any("NVDA" in ln for ln in out)

    def test_steady_state_headline_only(self):
        gp = {
            "state": "STEADY",
            "headline": "Book steady — 2 position(s) within discipline; "
                        "nothing high-priority for the next session.",
            "position_actions": [
                {"ticker": "MU", "action": "HOLD", "priority": 0,
                 "conviction": 0.5, "unrealized_pl": 1.0,
                 "pct_port": 30.0, "reasons": []},
            ],
            "portfolio_directives": [],
            "opportunities": [],
        }
        out = _game_plan_chat_lines(gp)
        assert out == [f"Game plan: {gp['headline']}"]

    def test_total_contract(self):
        assert _game_plan_chat_lines(None) == []
        assert _game_plan_chat_lines("x") == []
        assert _game_plan_chat_lines({}) == []
        assert _game_plan_chat_lines({"error": "boom"}) == []
        assert _game_plan_chat_lines(
            {"state": "NO_DATA", "headline": "nothing"}) == []


class TestHoldDisciplineChatLines:
    def test_disposition_drag_verbatim(self):
        hd = {
            "state": "DISPOSITION_DRAG",
            "verdict": "DISPOSITION_DRAG",
            "headline": "1 losing position held past the desk's own 0.5d "
                        "median losing-cut: LITE (1.0d, $-6.21). "
                        "Disposition drag $-6.21 unrealized.",
            "disposition_drag_usd": -6.21,
            "n_overstayed": 1,
        }
        out = _hold_discipline_chat_lines(hd)
        assert len(out) == 1
        assert hd["headline"] in out[0]

    def test_disciplined_is_silent(self):
        """Mirrors reporter._hold_discipline_line: only DISPOSITION_DRAG is
        chat-worthy. A 'you're fine' verdict is noise."""
        for state in ("DISCIPLINED", "INSUFFICIENT", "NO_DATA"):
            assert _hold_discipline_chat_lines(
                {"state": state, "headline": "x"}) == []

    def test_total_contract(self):
        assert _hold_discipline_chat_lines(None) == []
        assert _hold_discipline_chat_lines("x") == []
        assert _hold_discipline_chat_lines({}) == []
        assert _hold_discipline_chat_lines({"error": "boom"}) == []
