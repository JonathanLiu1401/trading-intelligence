"""Unified quant scorecard — one consolidated read-only view of the
deployed scorer + gate health.

Until this module landed, getting a single "current ML health" picture
required ten separate CLI invocations against the per-cycle skill ledgers
(``baseline_compare``, ``calibration_reliability``, ``stop_out_audit``,
``mfe_conversion``, ``gate_pnl``, ``gate_arm_historical``,
``persona_skill``, ``persona_regime_skill``, ``conviction_calibration``,
``llm_annotation_skill``). Each printed a different verdict format. An
unattended operator had to remember which CLI answers which question and
manually fuse them.

This module:

  * Reads the LATEST row of each ``data/*_skill_log.jsonl`` ledger the
    continuous loop maintains.
  * Computes a top-level verdict (HEALTHY / DEGRADED / CRITICAL / UNKNOWN)
    from the rolled-up state.
  * Emits the result as JSON (machine-readable) or a human-readable table.

Read-only by construction — never writes to ``backtest.db``, never trains,
never touches the deployed pickle. Same operational discipline as
``paper_trader.ml.run_risk_metrics`` and ``paper_trader.ml.deploy_audit``.

CLI::

    python3 -m paper_trader.ml.scorecard               # table
    python3 -m paper_trader.ml.scorecard --json        # JSON
    python3 -m paper_trader.ml.scorecard --reasons     # only show non-empty reasons

Exit code: 0 for HEALTHY / DEGRADED / UNKNOWN; 1 for CRITICAL — a shell
consumer can gate on ``$?`` like every other diagnostic CLI in this dir
(``host_guard``, ``decision_scorer.main``).
"""
from __future__ import annotations

import json
from collections import deque
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data"

# Ledger files mapped to scorecard section names. Single source of truth so
# a future ledger addition only needs ONE entry here (plus a tiny
# ``_compute_verdict`` rule for the new section's red/green threshold).
_LEDGER_PATHS: dict[str, Path] = {
    "scorer": DATA_DIR / "scorer_skill_log.jsonl",
    "baseline": DATA_DIR / "baseline_skill_log.jsonl",
    "gate_pnl": DATA_DIR / "gate_pnl_skill_log.jsonl",
    "gate_arm": DATA_DIR / "gate_arm_skill_log.jsonl",
    "calibrated_reliability": DATA_DIR / "calibrated_reliability_log.jsonl",
    "stop_out": DATA_DIR / "stop_out_skill_log.jsonl",
    "mfe": DATA_DIR / "mfe_skill_log.jsonl",
    "persona": DATA_DIR / "persona_skill_log.jsonl",
    "persona_regime": DATA_DIR / "persona_regime_skill_log.jsonl",
    "llm_annotation": DATA_DIR / "llm_annotation_skill_log.jsonl",
    "conviction_calibration": DATA_DIR / "conviction_calibration_log.jsonl",
}


def _tail_row(path: Path) -> dict | None:
    """Return the LATEST non-empty, parseable JSON row in ``path`` or None.

    Defensive: a missing file, a permission error, or a torn final row (a
    common pattern when a process is killed mid-``open`` write — see
    ``_append_*_skill_log`` discipline) MUST NOT crash the scorecard. Returns
    the last successfully-parsed row, walking backwards from the tail.
    """
    try:
        if not path.exists():
            return None
        # Bounded tail — ledgers are at most ``*_LOG_KEEP * 2`` rows (the
        # continuous loop trims past that). Reading the whole file is
        # acceptable; using a deque-1 keeps memory bounded if a malformed
        # ledger somehow grows beyond the cap.
        last_good: dict | None = None
        with path.open("r") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    row = json.loads(ln)
                except Exception:
                    # Torn row — skip and keep the previous good one.
                    continue
                if isinstance(row, dict):
                    last_good = row
        return last_good
    except Exception:
        return None


def collect(data_dir: Path | None = None) -> dict[str, Any]:
    """Read every ledger and return a section-keyed snapshot dict.

    ``data_dir`` overrides ``DATA_DIR`` for tests (tmp redirect pattern).
    Every section value is the LATEST row of the corresponding ledger, or
    None when the ledger is absent / unreadable.
    """
    base = data_dir if data_dir is not None else DATA_DIR
    out: dict[str, Any] = {}
    for name, rel_path in _LEDGER_PATHS.items():
        # When ``data_dir`` is overridden, redirect each ledger under it.
        path = (base / rel_path.name) if data_dir is not None else rel_path
        out[name] = _tail_row(path)
    return out


# Verdict thresholds — kept narrow and explicit so the scorecard reads like a
# checklist a quant would actually run by hand. None of these are tunable
# from the outside; if the bar moves, edit it here.
_GATE_PNL_RED_VERDICTS = frozenset({"GATE_SUBTRACTS_RETURN"})
_GATE_PNL_AMBER_VERDICTS = frozenset({"GATE_RETURN_NEUTRAL"})
_BASELINE_RED_VERDICTS = frozenset({"MLP_WORSE_THAN_TRIVIAL"})
_BASELINE_AMBER_VERDICTS = frozenset({"MLP_NO_BETTER_THAN_TRIVIAL"})
_STOP_RED_VERDICTS = frozenset({"STOP_HURTS"})
_MFE_RED_VERDICTS = frozenset({"TP_HURTS"})
_CONVICTION_RED_VERDICTS = frozenset({"MISCALIBRATED"})


def _section_state(section: dict | None,
                   red_predicate, amber_predicate) -> str:
    """Classify one ledger's latest row into ``RED`` / ``AMBER`` / ``GREEN``
    / ``UNKNOWN``. ``None`` ⇒ UNKNOWN (ledger absent). Predicates run
    against the row dict and return bool; first-match wins (RED > AMBER
    > GREEN). Empty / non-dict rows degrade to UNKNOWN."""
    if not isinstance(section, dict):
        return "UNKNOWN"
    try:
        if red_predicate(section):
            return "RED"
        if amber_predicate(section):
            return "AMBER"
        return "GREEN"
    except Exception:
        return "UNKNOWN"


def _compute_verdict(sections: dict[str, dict | None]) -> dict:
    """Roll the per-section RED/AMBER/GREEN states into one overall verdict.

    Logic mirrors a quant's mental triage:

      * **CRITICAL** — the gate is sizing real conviction (``gate_active=True``)
        AND the data says the gate is actively SUBTRACTING return
        (``gate_pnl`` GATE_SUBTRACTS_RETURN) OR the model is worse than a
        trivial one-liner (``baseline`` MLP_WORSE_THAN_TRIVIAL). Either is
        the "stop the bot" red flag.

      * **DEGRADED** — at least one section is RED (e.g. stop band hurts,
        a persona is inverted) but the central scorer + gate aren't actively
        net-negative. Operator should triage, not page.

      * **HEALTHY** — every section is GREEN or UNKNOWN-but-not-known-bad.

      * **UNKNOWN** — the scorer ledger itself is missing (fresh install or
        wiped data dir); we can't honestly verdict.

    Returns ``{"overall": ..., "reasons": [...]}``. ``reasons`` lists the
    specific findings that drove the verdict so an operator can triage
    without re-reading the ledger payload.
    """
    scorer = sections.get("scorer")
    baseline = sections.get("baseline")
    gate_pnl = sections.get("gate_pnl")

    # UNKNOWN gate — can't compute anything load-bearing without the scorer ledger.
    if scorer is None and baseline is None and gate_pnl is None:
        return {"overall": "UNKNOWN",
                "reasons": ["scorer ledger absent — no health signal"],
                "sections": {}}

    reasons: list[str] = []
    states: dict[str, str] = {}

    # Scorer (RED when MLP_NO_BETTER_THAN_TRIVIAL + gate active; AMBER when
    # gate is sub-min n_train; GREEN otherwise).
    gate_active = bool(scorer.get("gate_active")) if isinstance(scorer, dict) else False
    states["scorer"] = _section_state(
        scorer,
        red_predicate=lambda r: (
            gate_active and r.get("oos_rmse_ratio") is not None
            and float(r["oos_rmse_ratio"]) >= 1.5
        ),
        amber_predicate=lambda r: (
            gate_active and r.get("oos_buy_ic") is not None
            and abs(float(r["oos_buy_ic"])) < 0.05
        ),
    )
    if isinstance(scorer, dict):
        if states["scorer"] == "RED":
            reasons.append(
                f"scorer oos_rmse_ratio={scorer.get('oos_rmse_ratio')} >= 1.5"
                f" while gate is active — model worse than σ-baseline"
            )
        elif states["scorer"] == "AMBER":
            reasons.append(
                f"scorer oos_buy_ic={scorer.get('oos_buy_ic')} below "
                f"±0.05 — BUY rank skill at noise"
            )

    # Baseline (MLP vs trivial one-liner).
    states["baseline"] = _section_state(
        baseline,
        red_predicate=lambda r: (str(r.get("verdict") or "")
                                 in _BASELINE_RED_VERDICTS),
        amber_predicate=lambda r: (str(r.get("verdict") or "")
                                   in _BASELINE_AMBER_VERDICTS),
    )
    if isinstance(baseline, dict):
        if states["baseline"] == "RED":
            reasons.append(
                f"baseline MLP_WORSE_THAN_TRIVIAL "
                f"(ic_gap={baseline.get('ic_gap')})"
            )
        elif states["baseline"] == "AMBER":
            reasons.append(
                f"baseline MLP_NO_BETTER_THAN_TRIVIAL "
                f"(ic_gap={baseline.get('ic_gap')})"
            )

    # Gate PnL (the keep-or-kill economic verdict).
    states["gate_pnl"] = _section_state(
        gate_pnl,
        red_predicate=lambda r: (str(r.get("verdict") or "")
                                 in _GATE_PNL_RED_VERDICTS),
        amber_predicate=lambda r: (str(r.get("verdict") or "")
                                   in _GATE_PNL_AMBER_VERDICTS),
    )
    if isinstance(gate_pnl, dict):
        if states["gate_pnl"] == "RED":
            reasons.append(
                f"gate_pnl GATE_SUBTRACTS_RETURN "
                f"({gate_pnl.get('equal_weight_gate_contribution_pp')}pp)"
            )
        elif states["gate_pnl"] == "AMBER":
            reasons.append(
                f"gate_pnl GATE_RETURN_NEUTRAL"
            )

    # Stop-out / MFE / conviction / personas / llm-annotation are SECONDARY
    # signals — RED here downgrades HEALTHY → DEGRADED but cannot trigger
    # CRITICAL alone.
    stop_out = sections.get("stop_out")
    states["stop_out"] = _section_state(
        stop_out,
        red_predicate=lambda r: (str(r.get("verdict") or "")
                                 in _STOP_RED_VERDICTS),
        amber_predicate=lambda r: bool(r.get("stop_dark")),
    )
    mfe = sections.get("mfe")
    states["mfe"] = _section_state(
        mfe,
        red_predicate=lambda r: (str(r.get("verdict") or "")
                                 in _MFE_RED_VERDICTS),
        amber_predicate=lambda r: bool(r.get("tp_dark")),
    )
    conviction = sections.get("conviction_calibration")
    states["conviction_calibration"] = _section_state(
        conviction,
        red_predicate=lambda r: (str(r.get("verdict") or "")
                                 in _CONVICTION_RED_VERDICTS),
        amber_predicate=lambda r: bool(r.get("sizing_dark")),
    )
    persona = sections.get("persona")
    # Inverted personas — actively anti-predictive. Any inverted persona
    # is a flag (RED). Empty / no data ⇒ AMBER (we can't see signal yet).
    states["persona"] = _section_state(
        persona,
        red_predicate=lambda r: int(r.get("n_inverted") or 0) >= 1,
        amber_predicate=lambda r: bool(r.get("signal_dark")),
    )
    if isinstance(persona, dict) and states["persona"] == "RED":
        reasons.append(
            f"persona has {persona.get('n_inverted')} inverted "
            f"persona(s) — signal is anti-predictive"
        )

    llm = sections.get("llm_annotation")
    states["llm_annotation"] = _section_state(
        llm,
        # LLM pipeline dark isn't a model-skill RED — it just means the
        # ENDORSE/CONDEMN weight multiplier is structurally inert. AMBER
        # so an operator notices.
        red_predicate=lambda r: False,
        amber_predicate=lambda r: bool(r.get("pipeline_dark")),
    )

    cal_rel = sections.get("calibrated_reliability")
    states["calibrated_reliability"] = _section_state(
        cal_rel,
        red_predicate=lambda r: False,
        amber_predicate=lambda r: bool(r.get("calibrated_dark")),
    )

    # Overall verdict.
    # CRITICAL: scorer RED OR (baseline RED AND gate_active) OR gate_pnl RED.
    scorer_critical = states.get("scorer") == "RED"
    baseline_critical = (states.get("baseline") == "RED" and gate_active)
    gate_pnl_critical = states.get("gate_pnl") == "RED"

    if scorer_critical or baseline_critical or gate_pnl_critical:
        overall = "CRITICAL"
    elif any(s == "RED" for s in states.values()):
        overall = "DEGRADED"
    elif all(s in ("GREEN", "UNKNOWN") for s in states.values()):
        # All green or quietly-unknown — but if scorer ledger is fully
        # missing call it UNKNOWN, not HEALTHY.
        if scorer is None:
            overall = "UNKNOWN"
        else:
            overall = "HEALTHY"
    else:
        overall = "DEGRADED"

    return {"overall": overall, "reasons": reasons, "section_states": states}


def operator_summary(verdict: dict) -> str:
    """Single-line Discord/Slack-ready summary of the scorecard verdict.

    Single line (no embedded newlines) so a chat post survives intact.
    Includes the verdict and up to 3 of the most important reasons.
    """
    overall = verdict.get("overall", "UNKNOWN")
    reasons = verdict.get("reasons") or []
    if reasons:
        head = "; ".join(r.replace("\n", " ") for r in reasons[:3])
        return f"[scorecard] {overall} — {head}"
    return f"[scorecard] {overall}"


def _format_text(verdict: dict, sections: dict) -> str:
    """Operator-facing table. One section per row + verdict header."""
    lines = []
    overall = verdict.get("overall", "UNKNOWN")
    lines.append(f"=== ML/BACKTEST SCORECARD — {overall} ===")
    states = verdict.get("section_states") or {}
    lines.append(f"  {'section':<28}{'state':<10}{'note':<50}")
    for name in _LEDGER_PATHS:
        state = states.get(name, "UNKNOWN")
        row = sections.get(name)
        if row is None:
            note = "ledger absent"
        else:
            # Pick the most operationally-relevant field per section.
            if name == "scorer":
                note = (f"n_train={row.get('train_n')} "
                        f"oos_buy_ic={row.get('oos_buy_ic')} "
                        f"rmse_ratio={row.get('oos_rmse_ratio')}")
            elif name == "baseline":
                note = (f"verdict={row.get('verdict')} "
                        f"ic_gap={row.get('ic_gap')}")
            elif name == "gate_pnl":
                note = (f"verdict={row.get('verdict')} "
                        f"contrib_pp={row.get('equal_weight_gate_contribution_pp')}")
            elif name == "gate_arm":
                note = (f"verdict={row.get('verdict')} "
                        f"arm_monotone={row.get('arm_monotone_fraction')}")
            elif name == "calibrated_reliability":
                note = (f"verdict={row.get('verdict')} "
                        f"dark={row.get('calibrated_dark')}")
            elif name == "stop_out":
                note = (f"verdict={row.get('verdict')} "
                        f"dark={row.get('stop_dark')}")
            elif name == "mfe":
                note = (f"verdict={row.get('verdict')} "
                        f"dark={row.get('tp_dark')}")
            elif name == "persona":
                note = (f"n_inverted={row.get('n_inverted')} "
                        f"top={row.get('top_persona')}")
            elif name == "persona_regime":
                note = (f"verdict={row.get('verdict')}")
            elif name == "llm_annotation":
                note = (f"dark={row.get('pipeline_dark')} "
                        f"endorsed={row.get('n_endorsed')}")
            elif name == "conviction_calibration":
                note = (f"verdict={row.get('verdict')} "
                        f"dark={row.get('sizing_dark')}")
            else:
                note = ""
        lines.append(f"  {name:<28}{state:<10}{note:<50}")
    reasons = verdict.get("reasons") or []
    if reasons:
        lines.append("")
        lines.append("Reasons:")
        for r in reasons:
            lines.append(f"  - {r}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint. Returns exit code 1 on CRITICAL, 0 otherwise."""
    import argparse
    import sys

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.scorecard",
        description="Unified read-only quant scorecard. Reads the latest row "
                    "of every per-cycle skill ledger and emits one consolidated "
                    "health verdict. Never trains, never writes.",
    )
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON.")
    p.add_argument("--reasons", action="store_true",
                   help="Show only the reason list (one per line). Suppresses "
                        "the section table.")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    sections = collect()
    verdict = _compute_verdict(sections)

    if args.json:
        payload = {
            "overall": verdict.get("overall"),
            "reasons": verdict.get("reasons") or [],
            "section_states": verdict.get("section_states") or {},
            "sections": sections,
        }
        print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    elif args.reasons:
        for r in verdict.get("reasons") or []:
            print(r)
    else:
        print(_format_text(verdict, sections))

    return 1 if verdict.get("overall") == "CRITICAL" else 0


if __name__ == "__main__":
    raise SystemExit(main())
