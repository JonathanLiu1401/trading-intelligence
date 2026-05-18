"""DecisionScorer training-pipeline liveness / staleness monitor — read-only.

This is a **read-only diagnostic**. It never trains, never touches
`decision_scorer.pkl`, `decision_outcomes.jsonl`, `scorer_skill_log.jsonl`,
`build_features`, `N_FEATURES`, or any trade path — same operational
discipline as `paper_trader/ml/calibration.py` / `gate_audit.py` /
`feature_coverage.py` / `deploy_audit.py`. Safe to run against the live
unattended loop.

**Why this is not any existing tool.** The ML/backtest domain has a
saturated *quality* suite — `calibration` (decile rank skill), `skill_trend`
(oos_rmse vs the mean predictor), `baseline_compare` (MLP vs one-line rules),
`overfit_gap` (val≪oos), `feature_coverage` (dead inputs), `gate_stability`
(bootstrap ARM stability) — and one *config-drift* tool (`deploy_audit`:
deployed pkl arch vs `MLP_CONFIG`). **Every one of them assumes the on-disk
`decision_scorer.pkl` is current and asks whether it is *good*.** None ask the
prior question: *is the continuous loop still alive and actually re-pickling
the model as new outcomes accumulate?*

That gap is real and load-bearing. The conviction gate (invariant #5) acts on
whatever `decision_scorer.pkl` happens to be on disk every cycle once
`n_train ≥ 500`. If `run_continuous_backtests.py` has silently died, OOM'd, or
hung between cycles, the gate keeps modulating trades against a **frozen**
model while `decision_outcomes.jsonl` keeps growing — and *no existing
diagnostic surfaces it*, because they all analyse the stale pickle's quality,
which looks unchanged. runner.log shows exactly this failure class (a
long-lived process going stale relative to disk). This module is the
heartbeat watcher for the training pipeline itself.

Signals (all read-only):
  * heartbeat — newest `scorer_skill_log.jsonl` row's `timestamp` vs now.
    `run_continuous_backtests.py::_append_scorer_skill_log` writes one row per
    cycle; a healthy cycle is a few hours, so a heartbeat older than
    `STALE_HEARTBEAT_H` means the loop has stopped producing cycles.
  * pkl freshness — `decision_scorer.pkl` mtime + embedded `n_train` vs the
    newest skill-log row's `train_n`. If the loop logged a retrain the
    on-disk pickle does not reflect, the running gate is reading a model
    older than the last logged training (a deploy/liveness inversion).
  * gate exposure — the newest row's `gate_active`: a stale model that is
    *also* gating live-equivalent trades is the escalated case.

Verdicts (exit code mirrors the sibling diagnostics so cron can branch):
  FRESH               loop alive, pkl reflects the last logged retrain   → 0
  INSUFFICIENT_DATA   no skill-log and no pkl yet                        → 0
  STALE_PKL           loop alive but on-disk pkl lags the last retrain   → 2
  LOOP_STALLED        heartbeat older than STALE_HEARTBEAT_H             → 2
  LOOP_DEAD           heartbeat older than DEAD_HEARTBEAT_H              → 2
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

# data/ resolved the same way feature_coverage.py / decision_scorer.py do —
# independent of the (optional) decision_scorer import below so path
# resolution never depends on sklearn being importable.
_DATA = Path(__file__).resolve().parent.parent.parent / "data"
SKILL_LOG = _DATA / "scorer_skill_log.jsonl"
OUTCOMES = _DATA / "decision_outcomes.jsonl"
PKL = _DATA / "ml" / "decision_scorer.pkl"

# A healthy continuous cycle (5 runs + training + 60s cooldown) is observed at
# ~2-3h/cycle in scorer_skill_log.jsonl. These thresholds are deliberately
# generous so a single slow cycle never false-alarms, while a dead loop is
# still caught within a trading session.
STALE_HEARTBEAT_H = 6.0    # loop should have produced a cycle by now → WARN
DEAD_HEARTBEAT_H = 24.0    # loop is almost certainly down → CRITICAL
# The on-disk pkl is allowed to lag the last logged retrain by at most this
# (one in-flight cycle). Beyond it, the gate is reading a model older than the
# loop's own ledger says it trained — a liveness/deploy inversion.
PKL_LAG_GRACE_H = 4.0


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _parse_ts(s: str) -> datetime | None:
    try:
        dt = datetime.fromisoformat(s)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _last_skill_row(path: Path) -> dict | None:
    """Newest parseable row of scorer_skill_log.jsonl, or None."""
    if not path.exists():
        return None
    last: dict | None = None
    try:
        with path.open("r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                except (json.JSONDecodeError, ValueError):
                    continue
                if isinstance(row, dict) and row.get("timestamp"):
                    last = row
    except OSError:
        return None
    return last


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        with path.open("rb") as f:
            return sum(1 for _ in f)
    except OSError:
        return 0


def _pkl_n_train(path: Path) -> int | None:
    """Embedded `n_train` of the deployed pickle, or None if unreadable.

    Uses the canonical `DecisionScorer` accessor (the same one the live gate
    uses) so this reports exactly what the running system sees. Import is
    guarded: a diagnostic must degrade, never crash the unattended loop."""
    if not path.exists():
        return None
    try:
        from paper_trader.ml.decision_scorer import DecisionScorer
        return int(DecisionScorer().n_train)
    except Exception:
        return None


def scorer_freshness_report() -> dict:
    """Liveness/staleness of the DecisionScorer training pipeline. Pure read."""
    now = _now_utc()
    row = _last_skill_row(SKILL_LOG)
    pkl_exists = PKL.exists()
    pkl_mtime = (
        datetime.fromtimestamp(PKL.stat().st_mtime, tz=timezone.utc)
        if pkl_exists else None
    )
    pkl_age_h = (
        round((now - pkl_mtime).total_seconds() / 3600.0, 2)
        if pkl_mtime else None
    )
    pkl_n = _pkl_n_train(PKL)
    outcomes_rows = _count_lines(OUTCOMES)

    out: dict = {
        "now_utc": now.isoformat(),
        "skill_log_present": row is not None,
        "pkl_present": pkl_exists,
        "pkl_age_h": pkl_age_h,
        "pkl_n_train": pkl_n,
        "outcomes_rows": outcomes_rows,
        "last_cycle": row.get("cycle") if row else None,
        "last_train_n": row.get("train_n") if row else None,
        "last_status": row.get("status") if row else None,
        "gate_active": bool(row.get("gate_active")) if row else None,
        "heartbeat_age_h": None,
        "verdict": "INSUFFICIENT_DATA",
        "hint": "",
    }

    if row is None and not pkl_exists:
        out["hint"] = (
            "no scorer_skill_log.jsonl and no decision_scorer.pkl yet — "
            "the continuous loop has not completed its first training cycle"
        )
        return out

    if row is None:
        # pkl exists but the loop never wrote a heartbeat ledger — can't
        # assess liveness, only report the static pkl state.
        out["verdict"] = "INSUFFICIENT_DATA"
        out["hint"] = (
            f"decision_scorer.pkl present (n_train={pkl_n}, age={pkl_age_h}h) "
            "but scorer_skill_log.jsonl absent — heartbeat unverifiable"
        )
        return out

    hb = _parse_ts(row["timestamp"])
    if hb is None:
        out["verdict"] = "INSUFFICIENT_DATA"
        out["hint"] = f"newest skill-log timestamp unparseable: {row['timestamp']!r}"
        return out

    age_h = round((now - hb).total_seconds() / 3600.0, 2)
    out["heartbeat_age_h"] = age_h
    gate = out["gate_active"]
    gate_note = (
        " — the conviction gate is ACTIVE, so trades are being modulated "
        "against this frozen model" if gate else ""
    )

    if age_h >= DEAD_HEARTBEAT_H:
        out["verdict"] = "LOOP_DEAD"
        out["hint"] = (
            f"no continuous cycle in {age_h}h (>{DEAD_HEARTBEAT_H}h) — "
            f"run_continuous_backtests.py is almost certainly down; last "
            f"cycle was #{out['last_cycle']}{gate_note}"
        )
        return out

    if age_h >= STALE_HEARTBEAT_H:
        out["verdict"] = "LOOP_STALLED"
        out["hint"] = (
            f"no continuous cycle in {age_h}h (>{STALE_HEARTBEAT_H}h) — the "
            f"loop has stalled/hung between cycles; last cycle was "
            f"#{out['last_cycle']}{gate_note}"
        )
        return out

    # Heartbeat is fresh — is the on-disk pkl consistent with it? If the loop
    # logged a retrain meaningfully before the pkl's mtime advanced, the gate
    # is reading a model older than the loop's own ledger claims.
    if pkl_mtime is not None and (hb - pkl_mtime).total_seconds() / 3600.0 > PKL_LAG_GRACE_H:
        lag_h = round((hb - pkl_mtime).total_seconds() / 3600.0, 2)
        out["verdict"] = "STALE_PKL"
        out["hint"] = (
            f"loop alive (heartbeat {age_h}h old, cycle #{out['last_cycle']}) "
            f"but decision_scorer.pkl is {lag_h}h older than the last logged "
            f"retrain — the gate is reading a model the ledger says was "
            f"superseded{gate_note}"
        )
        return out

    out["verdict"] = "FRESH"
    out["hint"] = (
        f"loop alive — last cycle #{out['last_cycle']} {age_h}h ago "
        f"(status={out['last_status']}), pkl n_train={pkl_n} reflects the "
        f"last retrain (train_n={out['last_train_n']}), "
        f"{outcomes_rows} outcomes on disk"
    )
    return out


def analyze() -> dict:
    """Public entry point — full freshness report (read-only)."""
    return scorer_freshness_report()


def _cli() -> int:
    """`python3 -m paper_trader.ml.scorer_freshness` — read-only liveness
    check of the DecisionScorer training pipeline.

    Exit mirrors the sibling diagnostics so a cron can branch on "the gate is
    acting on a frozen / superseded model right now": 0 on FRESH /
    INSUFFICIENT_DATA, 2 on STALE_PKL / LOOP_STALLED / LOOP_DEAD."""
    rep = scorer_freshness_report()
    print(f"VERDICT: {rep['verdict']}  ({rep['hint']})")
    print(
        f"  heartbeat_age_h={rep['heartbeat_age_h']}  "
        f"last_cycle={rep['last_cycle']}  status={rep['last_status']}  "
        f"gate_active={rep['gate_active']}"
    )
    print(
        f"  pkl: present={rep['pkl_present']}  age_h={rep['pkl_age_h']}  "
        f"n_train={rep['pkl_n_train']}  |  outcomes_rows={rep['outcomes_rows']}"
        f"  last_train_n={rep['last_train_n']}"
    )
    return 0 if rep["verdict"] in ("FRESH", "INSUFFICIENT_DATA") else 2


if __name__ == "__main__":
    raise SystemExit(_cli())
