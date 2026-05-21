"""Dead-sector audit — cross-references per-sector training mass against the
deployed scorer's per-sector first-layer weight share to surface sectors the
model has gone *blind* to.

This is a **read-only diagnostic**. It never trains, never touches
``decision_scorer.pkl``, ``decision_outcomes.jsonl``, ``build_features``,
``N_FEATURES``, or any trade path — same operational discipline as
``paper_trader/ml/feature_importance.py`` / ``feature_coverage.py`` /
``regime_audit.py``. Safe to run against the live unattended loop.

**Why this is not ``feature_importance`` / ``feature_coverage``.**
``feature_importance`` reports per-feature importance but never tells the
operator whether a near-zero score means "the model genuinely doesn't rely on
it" or "the training pool literally has no rows that exercise it".
``feature_coverage`` measures default-frequency — for a 7-way sector one-hot
*every* non-majority sector is at its zero "default" in ~90% of rows simply by
construction, so the coverage signal is misleading for sectors.
**Neither tool answers the gate-relevant question for the operator who
maintains the deployed scorer: "do I have sectors that carry real outcome
mass yet contribute nothing to predictions?"** That is the gap this module
fills.

Decisive on the live corpus (observed at write time):
  * ``sector_crypto`` carries ~626 records in ``decision_outcomes.jsonl``
    (mostly MSTR/COIN positions) AND ~9% of all training mass, yet the
    deployed scorer's first-layer absolute-weight share for that sector slot
    is exactly ``0.0000`` (mean(|W|) across hidden units = 0).
  * Likewise ``sector_energy`` carries ~117 records yet shares ``0.0000``.

A future operator running ``feature_importance`` alone would see those zeros
and reasonably conclude "the data has nothing here" — but the data does. The
deployed pickle predates the corpus growth, so the model is structurally mute
on those sectors right now (the gate cannot help a crypto trade because no
non-zero ``sector_crypto`` weight exists to act on). This audit makes that
state explicit and gives the operator a single verdict to act on.

Logic:
  * record per-sector count by mapping every outcome's ``ticker`` through
    ``decision_scorer.SECTOR_MAP`` (same source of truth ``build_features``
    uses, so the count and the weight slot index can never drift).
  * pull per-sector importance share from the deployed
    ``DecisionScorer().feature_importance()`` payload (also a read-only call —
    re-uses the existing diagnostic, never re-derives the metric).
  * classify each sector with a crisp threshold-driven verdict:

      | Verdict | Trigger |
      |---|---|
      | ``DEAD_FEATURE``  | ``n_outcomes >= MIN_RECORDS`` AND ``importance_share < MIN_IMPORTANCE_SHARE``  (THE ALERT) |
      | ``SPARSE_DATA``   | ``n_outcomes < MIN_RECORDS``  (expected near-zero importance — honest) |
      | ``HEALTHY``       | otherwise |

  * overall verdict:

      | Verdict | Trigger |
      |---|---|
      | ``INSUFFICIENT_DATA`` | no scorer pickle / no outcomes / scorer untrained |
      | ``HAS_DEAD_SECTORS``  | ≥1 sector flagged ``DEAD_FEATURE``                |
      | ``HEALTHY``           | every sector with enough data carries weight     |

Operational note: ``DEAD_FEATURE`` is a HINT, not a strict bug — a retrain
on the now-larger outcomes pool will almost always fix it (the model gets
enough examples to learn the sector). The audit therefore also surfaces
``corpus_growth_ratio = n_outcomes_total / scorer.n_train`` so the operator
can see whether the scorer is simply stale (high ratio ⇒ retrain), or whether
even a fresh train would not help (low ratio ⇒ data-pipeline issue).

```bash
cd /home/zeph/paper-trader && python3 -m paper_trader.ml.dead_sector_audit
cd /home/zeph/paper-trader && python3 -m pytest tests/test_dead_sector_audit.py -v
```
"""
from __future__ import annotations

import json
from pathlib import Path

# Thresholds are module-level so tests assert exact verdicts and a tuning
# change is a single reviewable edit (calibration.py / gate_audit.py /
# skill_trend.py / regime_audit.py / feature_coverage.py precedent).
MIN_RECORDS = 30                # need this many in a sector before its
                                # importance reading is meaningful
MIN_IMPORTANCE_SHARE = 0.005    # 0.5% — sector is "dead" below this. Set
                                # well below the empirical healthy share
                                # (a uniformly-trained 17-feature model
                                # averages 1/17 ≈ 5.9%, so 0.5% catches a
                                # 10x-collapsed weight without false-
                                # alarming on naturally-rare sectors.
                                # Importance_share=0.0 (the live
                                # observation) is ~100x below this.
CORPUS_GROWTH_RETRAIN_HINT = 2.0  # outcomes / scorer.n_train above this ⇒
                                  # the audit suggests a retrain in its hint

_DATA = Path(__file__).resolve().parent.parent.parent / "data"
OUTCOMES = _DATA / "decision_outcomes.jsonl"


def _classify_sector(n_outcomes: int, importance_share: float) -> str:
    """Per-sector verdict — pure, total, no side-effects so tests can
    assert each branch directly without standing up a scorer."""
    if n_outcomes < MIN_RECORDS:
        return "SPARSE_DATA"
    if importance_share < MIN_IMPORTANCE_SHARE:
        return "DEAD_FEATURE"
    return "HEALTHY"


def _per_sector_counts(records) -> dict[str, int]:
    """Map outcome records to per-sector counts via SECTOR_MAP. Lazy import
    so an unrelated decision_scorer failure can't kill this whole
    diagnostic (same pattern as analyze() below)."""
    from paper_trader.ml.decision_scorer import SECTOR_MAP, SECTORS
    counts: dict[str, int] = {s: 0 for s in SECTORS}
    for r in records or []:
        try:
            t = str(r.get("ticker") or "")
        except Exception:
            continue
        sector = SECTOR_MAP.get(t, "other")
        # SECTORS is the closed enum; unknown sector falls through to
        # "other" so the count totals match len(records).
        counts[sector] = counts.get(sector, 0) + 1
    return counts


def report(records, importance_payload) -> dict:
    """Build the per-sector + overall report from already-loaded inputs.

    Pure function — takes already-parsed records and the
    ``DecisionScorer.feature_importance()`` payload so tests can inject
    controlled inputs without touching the disk. Never raises; any fault
    in shape parsing degrades to a sentinel row rather than a crash
    (the AGENTS.md "diagnostic must never break the loop" discipline).
    """
    from paper_trader.ml.decision_scorer import SECTORS

    rec_list = list(records or [])
    counts = _per_sector_counts(rec_list)

    # Index per-sector importance shares from the payload, keyed by the
    # `sector_<name>` feature name. Missing rows degrade to 0 share
    # (consistent with feature_importance's all-zero degenerate path).
    imp_share: dict[str, float] = {s: 0.0 for s in SECTORS}
    try:
        rows = (importance_payload or {}).get("importances") or []
        for row in rows:
            name = str(row.get("feature") or "")
            if not name.startswith("sector_"):
                continue
            sec = name[len("sector_"):]
            try:
                imp_share[sec] = float(row.get("importance_normalized") or 0.0)
            except (TypeError, ValueError):
                imp_share[sec] = 0.0
    except Exception:
        # Defensive — never raise from a diagnostic
        pass

    sectors_out: list[dict] = []
    n_dead = 0
    for s in SECTORS:
        n_out = counts.get(s, 0)
        share = imp_share.get(s, 0.0)
        verdict = _classify_sector(n_out, share)
        if verdict == "DEAD_FEATURE":
            n_dead += 1
        sectors_out.append({
            "sector": s,
            "n_outcomes": n_out,
            "importance_share": round(share, 6),
            "verdict": verdict,
        })
    # Sort: dead first (operator focus), then by importance descending
    # so the report mirrors `feature_importance`'s ranked layout.
    sectors_out.sort(key=lambda r: (r["verdict"] != "DEAD_FEATURE",
                                    -r["importance_share"]))

    n_total = len(rec_list)
    n_train = 0
    try:
        n_train = int((importance_payload or {}).get("n_train") or 0)
    except (TypeError, ValueError):
        n_train = 0

    # corpus_growth_ratio is the operator's tell that a retrain WILL help —
    # the bigger it is, the more new outcomes the deployed pickle hasn't
    # seen. Defined as outcomes / scorer.n_train, capped at -1 sentinel
    # when n_train is unknown so a dashboard / cron caller can distinguish
    # "unknown" from "no growth".
    growth_ratio: float | None
    if n_train > 0:
        growth_ratio = round(n_total / n_train, 3)
    else:
        growth_ratio = None

    if n_total == 0:
        overall = "INSUFFICIENT_DATA"
        hint = "no outcomes loaded — pipeline empty or path wrong"
    elif n_dead > 0:
        overall = "HAS_DEAD_SECTORS"
        dead_names = [r["sector"] for r in sectors_out
                      if r["verdict"] == "DEAD_FEATURE"]
        if growth_ratio is not None and growth_ratio >= CORPUS_GROWTH_RETRAIN_HINT:
            hint = (f"{n_dead} dead sector(s): {','.join(dead_names)}. "
                    f"corpus_growth_ratio={growth_ratio:.2f}× — the deployed "
                    f"scorer pickle predates the bulk of these records; a "
                    f"retrain (via the continuous loop or "
                    f"`run_continuous_backtests`) will almost certainly fix it.")
        else:
            hint = (f"{n_dead} dead sector(s): {','.join(dead_names)}. "
                    f"the scorer has seen most of the corpus already — "
                    f"investigate the training pipeline or sector mapping.")
    else:
        overall = "HEALTHY"
        hint = "every sector with sufficient data contributes weight"

    return {
        "status": "ok",
        "verdict": overall,
        "n_outcomes": n_total,
        "n_train_in_pickle": n_train,
        "corpus_growth_ratio": growth_ratio,
        "n_dead_sectors": n_dead,
        "sectors": sectors_out,
        "hint": hint,
    }


def analyze(outcomes_path: "Path | str | None" = None) -> dict:
    """Load the deployed scorer + outcomes file and return the audit.

    Read-only; never raises (any fault degrades to a status='error' row,
    mirroring regime_audit.analyze / baseline_compare.analyze)."""
    out: dict = {"status": "error", "verdict": "INSUFFICIENT_DATA",
                 "n_outcomes": 0, "sectors": [], "hint": ""}
    try:
        path = Path(outcomes_path) if outcomes_path else OUTCOMES
        if not path.exists():
            out["hint"] = f"no outcomes file at {path}"
            return out
        records: list[dict] = []
        # Stream to bound peak memory on a large outcomes file (the file
        # is allowed to grow to MAX_OUTCOMES_FOR_TRAINING * 2 ≈ 10k rows
        # before the loop trims; safe to read in full, but stream anyway
        # for symmetry with the other read-only diagnostics).
        with path.open("r") as fh:
            for ln in fh:
                ln = ln.strip()
                if not ln:
                    continue
                try:
                    records.append(json.loads(ln))
                except Exception:
                    continue
        try:
            from paper_trader.ml.decision_scorer import DecisionScorer
            scorer = DecisionScorer()
        except Exception as exc:
            out["hint"] = f"scorer load failed: {type(exc).__name__}: {exc}"
            return out
        if not getattr(scorer, "is_trained", False):
            out["hint"] = "scorer not trained — nothing to audit"
            return out
        imp = scorer.feature_importance()
        rep = report(records, imp)
        return rep
    except Exception as e:
        out["hint"] = f"audit failed: {type(e).__name__}: {e}"
        return out


def main(argv=None) -> int:
    """CLI entry — matches host_guard / decision_scorer convention:
    exit 0 on HEALTHY, 1 on INSUFFICIENT_DATA, 2 on HAS_DEAD_SECTORS so
    cron / shell can branch on $?."""
    import argparse
    import sys

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.dead_sector_audit",
        description=(
            "Cross-reference per-sector training mass in "
            "decision_outcomes.jsonl against the deployed scorer's "
            "per-sector first-layer weight share. Flags sectors that "
            "carry real outcome mass but contribute nothing to the "
            "deployed model — typically a stale-pickle / "
            "needs-retrain signal."
        ),
    )
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table.")
    p.add_argument("--outcomes", default=None,
                   help="Path to outcomes JSONL (default: data/decision_outcomes.jsonl).")
    args = p.parse_args(sys.argv[1:] if argv is None else argv)

    rep = analyze(args.outcomes)

    if args.json:
        print(json.dumps(rep, indent=2, sort_keys=True))
    else:
        v = rep.get("verdict", "INSUFFICIENT_DATA")
        nt = rep.get("n_train_in_pickle") or 0
        ng = rep.get("corpus_growth_ratio")
        gr_s = f"{ng:.2f}x" if isinstance(ng, (int, float)) else "n/a"
        print(f"[dead_sector_audit] verdict={v}  "
              f"n_outcomes={rep.get('n_outcomes', 0)}  "
              f"n_train_in_pickle={nt}  corpus_growth_ratio={gr_s}")
        rows = rep.get("sectors") or []
        if rows:
            print(f"  {'sector':<14}{'n_outcomes':>12}{'share':>10}  verdict")
            for r in rows:
                share_pct = r["importance_share"] * 100
                print(f"  {r['sector']:<14}{r['n_outcomes']:>12}"
                      f"{share_pct:>9.2f}%  {r['verdict']}")
        if rep.get("hint"):
            print(f"  hint: {rep['hint']}")

    verdict = rep.get("verdict", "INSUFFICIENT_DATA")
    if verdict == "HAS_DEAD_SECTORS":
        return 2
    if verdict == "INSUFFICIENT_DATA":
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
