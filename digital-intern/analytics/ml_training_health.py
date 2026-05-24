"""ML training-cycle health snapshot — the analyst's "is the local model
actually still training?" view.

``ArticleNet`` is the local pre-filter that routes the >99% of articles
Sonnet never sees; if it stops retraining the alert path's *unverified ML
push* rate drifts and the analyst silently loses signal calibration. The
trainer writes ``data/ml/training_metrics.jsonl`` per cycle (see
``ml/trainer.py::_log_metrics``), but no analytics surface reads it — the
existing siblings cover orthogonal angles:

  * ``storage.article_store.briefing_health`` — 5h Opus briefing pipeline
    cadence. Cannot detect a wedged local trainer.
  * ``analytics.ml_score_calibration`` — threshold sweep over the labels
    already in the DB. Says nothing about *whether the trainer ran*.
  * ``analytics.label_audit`` / ``analytics.score_agreement`` — strong-
    pool integrity and ML-vs-LLM drift. Diagnoses what the model
    *learned*, not whether it's still learning.

This is the missing third axis: read the trainer's own per-cycle log and
emit a single roll-up verdict so the operator can answer "is the trainer
still running, and is it improving?" in one call.

Verdict ladder (precedence high → low):
  * ``NO_DATA`` — log file missing or empty (a brand-new daemon must not
    false-flag as DEAD before its first cycle has run; honest "no
    verdict yet", mirrors the ``briefing_health`` NO_DATA discipline).
  * ``DEAD`` — most recent successful ``phase=train`` is older than
    ``dead_age_h`` (default 48h). The cycle has materially stopped —
    the production failure mode the ``[DI ml-trainer subprocess timeout]``
    memory documents.
  * ``STALE`` — most recent successful ``phase=train`` between
    ``stale_age_h`` (default 12h) and ``dead_age_h``. Early warning: a
    single missed full-retrain window or a subprocess timeout that hit
    once; not yet DEAD but cycle has begun to gap.
  * ``ERROR_HEAVY`` — > 50% of records in the window have
    ``status != 'ok'``. The cycle IS running but failing systematically
    (e.g. subprocess_timeout / child_exception / no_result errors). Less
    severe than DEAD because at least the runner is alive.
  * ``DIVERGING`` — last 3 ``phase=train`` ``val_loss`` values are
    strictly increasing each by >= ``DIVERGE_MIN_RATIO`` (5%). Model is
    getting worse cycle-over-cycle — typically a label-distribution
    drift or feature-pipeline regression. Surfaces BEFORE the analyst
    notices via degraded calibration.
  * ``HEALTHY`` — most recent < ``stale_age_h`` AND none of the above.

The pure builder ``compute_ml_training_health(records, ...)`` takes a list
of dicts so unit tests can inject synthetic records without touching the
filesystem. ``compute(...)`` is the file-reader shell that opens the
JSONL log, parses up to ``limit`` most-recent lines, and calls the
builder. Malformed lines are silently skipped; missing file returns
NO_DATA. Read-only — no DB / network / model touch. None of the four
load-bearing digital-intern invariants are touched (this module never
writes ai_score / ml_score / score_source / urgency, never opens the
articles DB).
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

BASE = Path(__file__).resolve().parents[1]
if str(BASE) not in sys.path:
    sys.path.insert(0, str(BASE))


# Default location matches ml/trainer.py::TRAINING_METRICS_LOG. Honour the
# ``DIGITAL_INTERN_ML_DIR`` env override too so an operator who relocated the
# ml dir (e.g. to a roomier disk) still gets a meaningful snapshot.
_DEFAULT_ML_DIR = Path(os.environ.get(
    "DIGITAL_INTERN_ML_DIR",
    str(BASE / "data" / "ml"),
))
DEFAULT_METRICS_PATH = _DEFAULT_ML_DIR / "training_metrics.jsonl"

# Verdict thresholds. Conservative defaults — the trainer cadence is highly
# variable (RETRAIN_INTERVAL=180s but gated by MIN_NEW_LABELS=50, so a quiet
# news cycle legitimately produces 6-12h gaps), so STALE starts at 12h and
# DEAD at 48h to avoid false alarms on a normal weekend lull. The values can
# be tuned by callers without changing the verdict-alphabet contract.
DEFAULT_STALE_AGE_H = 12.0
DEFAULT_DEAD_AGE_H = 48.0
DEFAULT_WINDOW_H = 72  # error / divergence aggregation window
DEFAULT_ERROR_HEAVY_PCT = 0.5

# Minimum step-up ratio in val_loss between consecutive cycles to count as
# "diverging". Noise floor — losses jitter ±2% normally, so 5% is the
# smallest move that's distinguishable from regular variance.
DIVERGE_MIN_RATIO = 1.05
DIVERGE_MIN_STEPS = 3

# Closed verdict alphabet — tests pin this so a typo or stray case can never
# silently break the dashboard / CLI consumers.
VERDICTS = ("NO_DATA", "DEAD", "STALE", "ERROR_HEAVY", "DIVERGING", "HEALTHY")


def _parse_ts(raw: str | None) -> datetime | None:
    """Parse the trainer's ``ts`` field. The log writes
    ``%Y-%m-%dT%H:%M:%SZ`` (always UTC), but the parser also accepts ISO
    8601 with explicit offset / Z so a future schema change is forward-
    compatible. Returns None on any failure — never raises (a corrupt log
    must never crash the health snapshot)."""
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(str(raw).replace("Z", "+00:00"))
    except Exception:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _age_h(ts: datetime | None, now: datetime) -> float | None:
    if ts is None:
        return None
    return round(max(0.0, (now - ts).total_seconds() / 3600.0), 2)


def compute_ml_training_health(
    records: Iterable[dict],
    *,
    now: datetime | None = None,
    window_h: int = DEFAULT_WINDOW_H,
    stale_age_h: float = DEFAULT_STALE_AGE_H,
    dead_age_h: float = DEFAULT_DEAD_AGE_H,
    error_heavy_pct: float = DEFAULT_ERROR_HEAVY_PCT,
) -> dict:
    """Pure builder: turn a list of trainer metric dicts into the health
    snapshot. Caller-injected ``now`` and thresholds make the verdict ladder
    fully deterministic for tests.

    ``records`` may be a generator — it is materialised once. Records with
    unparseable ``ts`` are tolerated for the trend / count buckets that
    don't need a timestamp; they are skipped for age-dependent buckets.
    """
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    window_h = max(int(window_h), 1)
    stale_age_h = max(float(stale_age_h), 0.0)
    dead_age_h = max(float(dead_age_h), stale_age_h)
    cutoff = now.timestamp() - window_h * 3600.0

    materialised = list(records)
    if not materialised:
        return _empty_snapshot(window_h, stale_age_h, dead_age_h)

    # Most recent successful run of each phase across the full record set
    # (not bounded by the window — answering "when did the trainer last
    # succeed?" must work even on a stale 5-day log).
    last_train_ts: datetime | None = None
    last_continuous_ts: datetime | None = None
    # Train val_losses ordered newest-first for the diverging check + the
    # surface trend (last 5 shown). ``status='ok'`` only — a skipped /
    # errored cycle has no val_loss to compare against.
    train_with_loss: list[tuple[datetime | None, float]] = []
    train_in_window = 0
    continuous_in_window = 0
    errors_in_window = 0
    total_in_window = 0

    for rec in materialised:
        if not isinstance(rec, dict):
            continue
        phase = rec.get("phase")
        status = rec.get("status")
        ts = _parse_ts(rec.get("ts"))

        in_window = ts is not None and ts.timestamp() >= cutoff
        if in_window:
            total_in_window += 1
            if status != "ok":
                errors_in_window += 1

        if status != "ok":
            # Errored records still count toward error_heavy but never
            # toward last-success / val-loss trend.
            continue

        if phase == "train":
            if ts is not None and (last_train_ts is None or ts > last_train_ts):
                last_train_ts = ts
            if in_window:
                train_in_window += 1
            try:
                val = rec.get("val_loss")
                if val is not None:
                    train_with_loss.append((ts, float(val)))
            except (TypeError, ValueError):
                pass
        elif phase == "continuous":
            if ts is not None and (
                last_continuous_ts is None or ts > last_continuous_ts
            ):
                last_continuous_ts = ts
            if in_window:
                continuous_in_window += 1

    # Newest-first for the trend / diverging check. Records without a
    # parseable ts sort last so they never bump real recent ones.
    train_with_loss.sort(
        key=lambda r: r[0].timestamp() if r[0] else -1.0,
        reverse=True,
    )
    val_loss_trend = [round(v, 4) for _, v in train_with_loss[:5]]

    # Divergence: last DIVERGE_MIN_STEPS val_losses (newest-first) must be
    # strictly increasing AS WE GO BACK IN TIME — i.e. oldest_of_three <
    # middle < newest, each step up by >= DIVERGE_MIN_RATIO. Mirror order:
    # newest-first means we want trend[0] > trend[1] > trend[2] (each ratio
    # >= 1.05).
    diverging = False
    if len(val_loss_trend) >= DIVERGE_MIN_STEPS:
        recent = val_loss_trend[:DIVERGE_MIN_STEPS]
        diverging = all(
            recent[i] >= recent[i + 1] * DIVERGE_MIN_RATIO
            and recent[i + 1] > 0
            for i in range(DIVERGE_MIN_STEPS - 1)
        )

    last_train_age_h = _age_h(last_train_ts, now)
    last_continuous_age_h = _age_h(last_continuous_ts, now)

    # Verdict ladder — precedence: NO_DATA, DEAD, STALE, ERROR_HEAVY,
    # DIVERGING, HEALTHY. STALE/DEAD outrank ERROR_HEAVY/DIVERGING because
    # a not-running trainer is a worse fact than a misbehaving one (a
    # diverging trainer at least produces fresh checkpoints; a dead one
    # leaves the model frozen and silently rots calibration).
    if last_train_ts is None and last_continuous_ts is None:
        verdict = "NO_DATA"
    elif last_train_age_h is not None and last_train_age_h > dead_age_h:
        verdict = "DEAD"
    elif (
        last_train_age_h is not None
        and last_train_age_h > stale_age_h
        and last_train_age_h <= dead_age_h
    ):
        verdict = "STALE"
    elif (
        total_in_window > 0
        and errors_in_window / total_in_window > error_heavy_pct
    ):
        verdict = "ERROR_HEAVY"
    elif diverging:
        verdict = "DIVERGING"
    else:
        verdict = "HEALTHY"

    return {
        "window_h": window_h,
        "last_train_age_h": last_train_age_h,
        "last_continuous_age_h": last_continuous_age_h,
        "train_in_window": train_in_window,
        "continuous_in_window": continuous_in_window,
        "errors_in_window": errors_in_window,
        "total_in_window": total_in_window,
        "val_loss_trend": val_loss_trend,
        "diverging": diverging,
        "verdict": verdict,
    }


def _empty_snapshot(
    window_h: int, stale_age_h: float, dead_age_h: float
) -> dict:
    return {
        "window_h": window_h,
        "last_train_age_h": None,
        "last_continuous_age_h": None,
        "train_in_window": 0,
        "continuous_in_window": 0,
        "errors_in_window": 0,
        "total_in_window": 0,
        "val_loss_trend": [],
        "diverging": False,
        "verdict": "NO_DATA",
    }


def load_recent_records(
    metrics_path: Path | str = DEFAULT_METRICS_PATH,
    limit: int = 500,
) -> list[dict]:
    """Tail-read up to ``limit`` JSONL records from the trainer log.

    Cheap: each record is < 500 bytes so even a 500-line tail is ~250 KB.
    Missing file returns []. Malformed lines are skipped (best-effort —
    a half-written line at the tip of an active log must not crash the
    snapshot). The trainer appends in chronological order, so the returned
    list is oldest-first within ``limit``."""
    p = Path(metrics_path)
    if not p.exists():
        return []
    try:
        with p.open("r", encoding="utf-8") as f:
            lines = f.readlines()
    except OSError:
        return []
    if limit > 0:
        lines = lines[-limit:]
    out: list[dict] = []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            continue
        if isinstance(rec, dict):
            out.append(rec)
    return out


def compute(
    window_h: int = DEFAULT_WINDOW_H,
    metrics_path: Path | str | None = None,
    *,
    now: datetime | None = None,
) -> dict:
    """File-reader orchestrator: load + build + return."""
    path = Path(metrics_path) if metrics_path else DEFAULT_METRICS_PATH
    records = load_recent_records(path)
    snap = compute_ml_training_health(records, now=now, window_h=window_h)
    snap["metrics_path"] = str(path)
    snap["records_read"] = len(records)
    return snap


def main() -> int:
    result = compute()
    print(f"ml_training_health: verdict={result['verdict']}")
    last = result["last_train_age_h"]
    print(
        f"  last_train={('%.2fh ago' % last) if last is not None else 'never'} "
        f"(window={result['window_h']}h "
        f"train_in_window={result['train_in_window']} "
        f"errors={result['errors_in_window']}/{result['total_in_window']})"
    )
    trend = result["val_loss_trend"]
    if trend:
        print(f"  val_loss trend (newest→oldest): {trend}")
    if result["diverging"]:
        print("  WARNING: val_loss is rising 3+ consecutive cycles")
    return 0


if __name__ == "__main__":
    sys.exit(main())
