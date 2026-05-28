"""CLI: ``python3 -m paper_trader.gate_status``

Reports the conviction gate's current effective state in one shell
command — the operator-facing complement to the per-cycle
``gate_killswitch_active`` field added to ``scorer_skill_log.jsonl``.

Why this exists: the conviction gate (``_ml_decide``) has two independent
guards (the ``n_train >= 500`` engagement threshold and the trailing-OOS-IC
kill-switch). The deployed pickle's ``n_train`` is one read away from
``DecisionScorer().n_train``, but the kill-switch decision is buried
inside a private helper that production code calls per-decision. Without
this CLI, an operator on a shell triaging "is the gate actually firing
right now?" had to either tail ``continuous.log`` for a recent
``scorer=…(gate-killed,...)`` token (only visible during BUY emissions)
or read the latest ``scorer_skill_log.jsonl`` row (only updated once per
cycle). This wraps the live kill-switch evaluation as a one-shot read so
the answer is always current.

Pattern mirrors ``paper_trader.ml.decision_scorer`` and
``paper_trader.host_guard``: ``int`` return + ``--json`` flag + read-only
(never trains, never persists), so callers can gate on ``$?`` and pipe
the JSON output through ``jq``.
"""
from __future__ import annotations

import argparse
import json
import sys


def _gate_effective_state() -> dict:
    """Return ``{"n_train", "n_train_threshold_met",
    "killswitch_active", "killswitch_reason", "gate_effectively_active",
    "trained"}`` describing the gate's current effective state.

    Every field degrades to ``None`` (or ``False``) on read failure
    rather than raising — same best-effort discipline as the in-process
    ``_should_gate_modulate_conviction`` itself, so a corrupt pickle /
    missing ledger doesn't break the diagnostic.
    """
    out: dict = {
        "n_train": None,
        "n_train_threshold_met": False,
        "trained": False,
        "killswitch_active": None,
        "killswitch_reason": None,
        "gate_effectively_active": None,
    }
    try:
        from paper_trader.ml.decision_scorer import DecisionScorer
        ds = DecisionScorer()
        out["trained"] = bool(ds.is_trained)
        if ds.is_trained:
            try:
                n = int(ds.n_train)
                out["n_train"] = n
                out["n_train_threshold_met"] = n >= 500
            except (TypeError, ValueError):
                pass
    except Exception:
        pass
    try:
        from paper_trader.backtest import _should_gate_modulate_conviction
        active, reason = _should_gate_modulate_conviction()
        out["killswitch_active"] = bool(active)
        out["killswitch_reason"] = str(reason)
    except Exception as exc:
        out["killswitch_reason"] = f"kill-switch read error: {exc}"
    # Effective gate: both guards must say active. None if kill-switch
    # state is unknown (honest degradation, not a fabricated False).
    if out["killswitch_active"] is None:
        out["gate_effectively_active"] = None
    else:
        out["gate_effectively_active"] = (
            out["n_train_threshold_met"] and out["killswitch_active"]
        )
    return out


def main(argv: list[str] | None = None) -> int:
    """Print the gate's effective state. Returns 0 when the gate IS
    effectively active (both guards green), 1 otherwise — so shell
    callers can gate on ``$?`` (``gate_status && do-something``)."""
    parser = argparse.ArgumentParser(
        prog="python3 -m paper_trader.gate_status",
        description=(
            "Show the conviction gate's current effective state. The gate "
            "fires only when (a) the deployed scorer has n_train >= 500 "
            "AND (b) the kill-switch (signed trailing OOS BUY rank-IC) is "
            "active. Read-only — never trains, never writes."
        ),
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit machine-readable JSON instead of a human-readable table.",
    )
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)

    state = _gate_effective_state()

    if args.json:
        print(json.dumps(state, indent=2, sort_keys=True))
        return 0 if state["gate_effectively_active"] is True else 1

    print("[gate_status] paper-trader conviction-gate effective state")
    if not state["trained"]:
        print("  scorer: NOT TRAINED (no pickle on disk)")
    else:
        n = state["n_train"]
        threshold_state = "MET" if state["n_train_threshold_met"] else "UNMET"
        print(f"  scorer: trained, n_train={n}  (>=500 ? {threshold_state})")
    ks = state["killswitch_active"]
    if ks is None:
        print(f"  kill-switch: unknown (read error)")
    else:
        print(f"  kill-switch: {'ACTIVE (letting gate fire)' if ks else 'KILLED (gate suppressed)'}")
    if state["killswitch_reason"]:
        print(f"    reason: {state['killswitch_reason']}")
    eff = state["gate_effectively_active"]
    if eff is None:
        print("  gate effectively: UNKNOWN")
    elif eff:
        print("  gate effectively: ACTIVE — conviction modulation IS firing")
    else:
        print("  gate effectively: INACTIVE — conviction modulation is NOT firing")
    return 0 if eff is True else 1


if __name__ == "__main__":
    raise SystemExit(main())
