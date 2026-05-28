"""Behaviour locks for the 2026-05-28 Phase-2 additions:

1. ``paper_trader.gate_status`` CLI reports the live gate state truthfully
   in both human-readable and ``--json`` form.
2. ``_append_scorer_skill_log`` captures the kill-switch's verdict per
   cycle as ``gate_killswitch_active`` / ``gate_killswitch_reason`` /
   ``gate_effectively_active`` so a researcher can trend "is the gate
   actually firing right now?" across the durable JSONL ledger.
"""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ---------------------------------------------------------------------------
# Section A — gate_status CLI
# ---------------------------------------------------------------------------

class TestGateStatusStateBuilder:
    """``_gate_effective_state`` rolls the two gate guards (n_train >= 500,
    kill-switch verdict) into one operator-readable dict. Pin the field
    contract so a downstream consumer (the JSON output, the ledger field)
    can rely on the schema."""

    def _fake_scorer(self, trained: bool, n_train: int):
        """Build a DecisionScorer test double bypassing disk load."""
        from paper_trader.ml.decision_scorer import DecisionScorer
        ds = DecisionScorer.__new__(DecisionScorer)
        ds._model = object() if trained else None
        ds._scaler = None
        ds._trained = trained
        ds._n_train = n_train
        ds._pred_quantiles = None
        ds._label_quantiles = None
        return ds

    def test_untrained_scorer_yields_empty_threshold(self):
        import paper_trader.gate_status as gs
        fake = self._fake_scorer(trained=False, n_train=0)
        # gate_status does lazy imports inside the function, so patch the
        # source modules rather than the local reference.
        with patch("paper_trader.ml.decision_scorer.DecisionScorer",
                   return_value=fake), \
             patch("paper_trader.backtest._should_gate_modulate_conviction",
                   return_value=(True, "default gate-active")):
            state = gs._gate_effective_state()
        assert state["trained"] is False
        assert state["n_train"] is None
        assert state["n_train_threshold_met"] is False
        assert state["killswitch_active"] is True
        assert state["gate_effectively_active"] is False

    def test_trained_high_n_train_and_active_killswitch_gives_active(self):
        import paper_trader.gate_status as gs
        fake = self._fake_scorer(trained=True, n_train=2500)
        with patch("paper_trader.ml.decision_scorer.DecisionScorer",
                   return_value=fake), \
             patch("paper_trader.backtest._should_gate_modulate_conviction",
                   return_value=(True, "median oos_buy_ic=+0.12 ...")):
            state = gs._gate_effective_state()
        assert state["trained"] is True
        assert state["n_train"] == 2500
        assert state["n_train_threshold_met"] is True
        assert state["killswitch_active"] is True
        assert state["gate_effectively_active"] is True

    def test_killswitch_killed_yields_inactive_gate(self):
        import paper_trader.gate_status as gs
        fake = self._fake_scorer(trained=True, n_train=2500)
        with patch("paper_trader.ml.decision_scorer.DecisionScorer",
                   return_value=fake), \
             patch("paper_trader.backtest._should_gate_modulate_conviction",
                   return_value=(False, "median oos_buy_ic=-0.06 ...")):
            state = gs._gate_effective_state()
        assert state["killswitch_active"] is False
        assert state["n_train_threshold_met"] is True
        assert state["gate_effectively_active"] is False

    def test_below_500_n_train_threshold_unmet(self):
        import paper_trader.gate_status as gs
        fake = self._fake_scorer(trained=True, n_train=100)
        with patch("paper_trader.ml.decision_scorer.DecisionScorer",
                   return_value=fake), \
             patch("paper_trader.backtest._should_gate_modulate_conviction",
                   return_value=(True, "default")):
            state = gs._gate_effective_state()
        assert state["n_train_threshold_met"] is False
        assert state["gate_effectively_active"] is False

    def test_killswitch_exception_yields_none(self):
        """A kill-switch read raising MUST degrade to None (not fabricate
        True or False). The ``gate_effectively_active`` field then also
        reads None — honesty discipline mirrors ``predict_with_meta``."""
        import paper_trader.gate_status as gs
        fake = self._fake_scorer(trained=True, n_train=2500)
        with patch("paper_trader.ml.decision_scorer.DecisionScorer",
                   return_value=fake), \
             patch("paper_trader.backtest._should_gate_modulate_conviction",
                   side_effect=RuntimeError("simulated")):
            state = gs._gate_effective_state()
        assert state["killswitch_active"] is None
        assert state["gate_effectively_active"] is None
        assert "kill-switch read error" in (state["killswitch_reason"] or "")


class TestGateStatusCli:
    """End-to-end CLI: ``python3 -m paper_trader.gate_status`` returns
    exit code 0 when the gate is effectively active, 1 otherwise (mirrors
    the host_guard / decision_scorer CLI exit-code convention so shell
    callers can ``gate_status && do-something``)."""

    def test_main_returns_0_when_gate_active(self, monkeypatch):
        import paper_trader.gate_status as gs
        monkeypatch.setattr(gs, "_gate_effective_state", lambda: {
            "trained": True, "n_train": 2500,
            "n_train_threshold_met": True,
            "killswitch_active": True, "killswitch_reason": "active",
            "gate_effectively_active": True,
        })
        # Default human-readable mode
        assert gs.main([]) == 0
        # JSON mode
        assert gs.main(["--json"]) == 0

    def test_main_returns_1_when_gate_inactive(self, monkeypatch):
        import paper_trader.gate_status as gs
        monkeypatch.setattr(gs, "_gate_effective_state", lambda: {
            "trained": True, "n_train": 2500,
            "n_train_threshold_met": True,
            "killswitch_active": False, "killswitch_reason": "killed",
            "gate_effectively_active": False,
        })
        assert gs.main([]) == 1
        assert gs.main(["--json"]) == 1

    def test_main_returns_1_on_unknown(self, monkeypatch):
        """``None`` (kill-switch read failure) yields exit 1 — the
        conservative choice for ``$?``-gated shells (fail closed)."""
        import paper_trader.gate_status as gs
        monkeypatch.setattr(gs, "_gate_effective_state", lambda: {
            "trained": True, "n_train": 2500,
            "n_train_threshold_met": True,
            "killswitch_active": None, "killswitch_reason": "error",
            "gate_effectively_active": None,
        })
        assert gs.main([]) == 1
        assert gs.main(["--json"]) == 1

    def test_json_output_is_valid_json(self, capsys, monkeypatch):
        import paper_trader.gate_status as gs
        monkeypatch.setattr(gs, "_gate_effective_state", lambda: {
            "trained": True, "n_train": 1500,
            "n_train_threshold_met": True,
            "killswitch_active": True, "killswitch_reason": "active",
            "gate_effectively_active": True,
        })
        gs.main(["--json"])
        out = capsys.readouterr().out
        # Last block is the JSON dump
        parsed = json.loads(out.strip().splitlines()[-1] if "{" not in
                            out.strip().splitlines()[0] else out)
        # More tolerant: just verify json.loads accepts the whole printed payload
        # (only one JSON object is emitted, possibly preceded by a load banner).
        # Find the JSON object in the output:
        start = out.find("{")
        end = out.rfind("}") + 1
        parsed = json.loads(out[start:end])
        assert parsed["gate_effectively_active"] is True
        assert parsed["n_train"] == 1500


# ---------------------------------------------------------------------------
# Section B — _append_scorer_skill_log kill-switch field capture
# ---------------------------------------------------------------------------

class TestScorerSkillLogKillswitchFields:
    """``_append_scorer_skill_log`` must capture the kill-switch verdict
    in every row alongside the existing ``gate_active`` (n_train ≥ 500)
    field. The TRUE effective gate state ⇔ both guards green ⇒
    ``gate_effectively_active``. Locks the schema additions so a future
    refactor cannot silently drop them."""

    def _read_last_row(self, path: Path) -> dict:
        lines = [ln for ln in path.read_text().splitlines() if ln.strip()]
        return json.loads(lines[-1])

    def test_killswitch_active_recorded_when_active(self, tmp_path,
                                                    monkeypatch):
        import run_continuous_backtests as rcb
        log_path = tmp_path / "scorer_skill_log.jsonl"
        monkeypatch.setattr(rcb, "SCORER_SKILL_LOG", log_path)
        with patch("paper_trader.backtest._should_gate_modulate_conviction",
                   return_value=(True, "median oos_buy_ic=+0.12")):
            ok = rcb._append_scorer_skill_log(
                "scorer ok train_n=800 val_rmse=10.1 oos_n=200 "
                "oos_rmse=12.0 oos_diracc=0.55 oos_ic=+0.10",
                cycle=1, win_start=date(2020, 1, 1),
                win_end=date(2021, 1, 1),
            )
        assert ok is True
        row = self._read_last_row(log_path)
        assert row["gate_killswitch_active"] is True
        assert "median oos_buy_ic" in row["gate_killswitch_reason"]
        # gate_active = train_n >= 500 → True (800 ≥ 500)
        assert row["gate_active"] is True
        # Effectively active because BOTH guards green
        assert row["gate_effectively_active"] is True

    def test_killswitch_killed_recorded_when_killed(self, tmp_path,
                                                    monkeypatch):
        """The most operationally interesting state — gate_active is True
        (n_train threshold met) BUT killswitch killed the modulation.
        This row pattern is what a quant queries with
        ``jq 'select(.gate_active and (.gate_killswitch_active==false))'``
        to find cycles where the kill-switch suppressed a would-be-firing
        gate."""
        import run_continuous_backtests as rcb
        log_path = tmp_path / "scorer_skill_log.jsonl"
        monkeypatch.setattr(rcb, "SCORER_SKILL_LOG", log_path)
        with patch("paper_trader.backtest._should_gate_modulate_conviction",
                   return_value=(False, "median oos_buy_ic=-0.06")):
            ok = rcb._append_scorer_skill_log(
                "scorer ok train_n=2500 val_rmse=10.0 oos_n=500 "
                "oos_rmse=15.0 oos_diracc=0.49 oos_ic=-0.05",
                cycle=2, win_start=date(2020, 1, 1),
                win_end=date(2021, 1, 1),
            )
        assert ok is True
        row = self._read_last_row(log_path)
        assert row["gate_active"] is True            # n_train >= 500
        assert row["gate_killswitch_active"] is False
        assert row["gate_effectively_active"] is False  # ks killed
        assert "median oos_buy_ic" in row["gate_killswitch_reason"]

    def test_killswitch_exception_recorded_as_none(self, tmp_path,
                                                    monkeypatch):
        """A read fault in the kill-switch must degrade the row to
        ``None`` for both ``killswitch_active`` and
        ``gate_effectively_active`` — honest degradation, not fabricated
        True/False. Mirrors the gate_status CLI's None semantics."""
        import run_continuous_backtests as rcb
        log_path = tmp_path / "scorer_skill_log.jsonl"
        monkeypatch.setattr(rcb, "SCORER_SKILL_LOG", log_path)
        with patch("paper_trader.backtest._should_gate_modulate_conviction",
                   side_effect=RuntimeError("simulated")):
            ok = rcb._append_scorer_skill_log(
                "scorer ok train_n=800 val_rmse=10.1",
                cycle=3, win_start=date(2020, 1, 1),
                win_end=date(2021, 1, 1),
            )
        assert ok is True
        row = self._read_last_row(log_path)
        assert row["gate_killswitch_active"] is None
        assert row["gate_effectively_active"] is None
        # Reason should still carry the failure cause
        assert "kill-switch read error" in (
            row["gate_killswitch_reason"] or "")

    def test_n_train_below_threshold_inactive_regardless_of_ks(
        self, tmp_path, monkeypatch
    ):
        """Below n_train=500 the gate is structurally inactive
        (invariant #5). Even with a healthy kill-switch verdict, the
        effective gate must be False."""
        import run_continuous_backtests as rcb
        log_path = tmp_path / "scorer_skill_log.jsonl"
        monkeypatch.setattr(rcb, "SCORER_SKILL_LOG", log_path)
        with patch("paper_trader.backtest._should_gate_modulate_conviction",
                   return_value=(True, "median oos_buy_ic=+0.10")):
            ok = rcb._append_scorer_skill_log(
                "scorer ok train_n=400 val_rmse=11.0",
                cycle=4, win_start=date(2020, 1, 1),
                win_end=date(2021, 1, 1),
            )
        assert ok is True
        row = self._read_last_row(log_path)
        assert row["gate_active"] is False  # 400 < 500
        assert row["gate_killswitch_active"] is True
        assert row["gate_effectively_active"] is False  # n_train guard wins
