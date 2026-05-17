#!/usr/bin/env python3
"""One-off overfitting audit for the paper-trader backtest engine.

Sweeps a handful of historical windows. For each window:

  1. Reports the label-contamination rate (what fraction of articles in
     that window were retroactively collected and thus carry hindsight
     `ai_score` values).

  2. Runs one real backtest plus a permutation test (shuffling article
     dates) and reports whether the real return is statistically above
     the shuffled distribution.

  3. Prints a final PASS/WARN summary across windows and writes the full
     machine-readable result to ``data/overfitting_audit.json``.

Usage:
    cd /home/zeph/paper-trader
    python3 scripts/audit_overfitting.py [--quick]

``--quick`` reduces the number of windows and permutations for a
faster check (useful when iterating on the validation logic itself).
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from datetime import date
from pathlib import Path

# Allow running the script directly without installing the package.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from paper_trader.backtest import BacktestEngine, LOCAL_ARTICLES_DB
from paper_trader.validation import (
    audit_label_contamination,
    run_permutation_test,
)


# Five well-spaced historical windows, each ~1 year, covering different
# regimes (quiet bull, COVID crash, post-rate-hike, etc.). Tweak as needed.
DEFAULT_WINDOWS: list[tuple[date, date]] = [
    (date(2017, 1, 1), date(2017, 12, 31)),
    (date(2019, 6, 1), date(2020, 5, 31)),    # straddles COVID crash
    (date(2021, 1, 1), date(2021, 12, 31)),
    (date(2022, 6, 1), date(2023, 5, 31)),    # rate-hike regime
    (date(2024, 1, 1), date(2024, 12, 31)),
]

QUICK_WINDOWS: list[tuple[date, date]] = [
    (date(2021, 1, 1), date(2021, 12, 31)),
    (date(2024, 1, 1), date(2024, 12, 31)),
]


def _audit_window(start: date, end: date, n_perm: int) -> dict:
    """Run audit + permutation test for one window."""
    print(f"\n══════ {start} → {end} ══════")

    # 1. Label contamination — fast.
    if LOCAL_ARTICLES_DB.exists():
        try:
            audit = audit_label_contamination(str(LOCAL_ARTICLES_DB), start, end)
            print(f"  label contamination: {audit['contamination_rate']:.1%} "
                  f"({audit['contaminated_count']}/{audit['total_articles']} articles, "
                  f"verdict {audit['verdict']})")
        except Exception as e:
            audit = {"error": str(e)}
            print(f"  label audit failed: {e}")
    else:
        audit = {"error": f"articles DB not found at {LOCAL_ARTICLES_DB}"}
        print(f"  label audit skipped: {audit['error']}")

    # 2. Build engine + permutation test.
    perm: dict = {}
    try:
        t0 = time.time()
        engine = BacktestEngine(start=start, end=end)
        with tempfile.TemporaryDirectory(prefix="audit_perm_") as tmp:
            perm = run_permutation_test(
                engine, seed=hash((start, end)) & 0xFFFFFFFF,
                n_permutations=n_perm,
                isolated_db_path=Path(tmp) / "perm.db",
            )
        elapsed = time.time() - t0
        if "error" in perm:
            print(f"  permutation test FAILED: {perm['error']}")
        else:
            print(f"  permutation test ({elapsed/60:.1f}min): "
                  f"verdict={perm['verdict']} "
                  f"strategy={perm['original_return']:+.1f}% "
                  f"shuffled_mean={perm['permuted_mean']:+.1f}% "
                  f"p={perm['p_value']:.3f} z={perm['z_score']:+.2f}")
    except Exception as e:
        perm = {"error": str(e)}
        print(f"  permutation test crashed: {e}")

    return {
        "window_start": start.isoformat(),
        "window_end": end.isoformat(),
        "label_audit": audit,
        "permutation_test": perm,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="Use 2 windows × 5 permutations instead of 5 × 10 (fast).")
    ap.add_argument("--windows", type=int, default=None,
                    help="Limit to first N windows from the default set.")
    ap.add_argument("--permutations", type=int, default=None,
                    help="Override permutations per window.")
    args = ap.parse_args()

    if args.quick:
        windows = QUICK_WINDOWS
        n_perm = args.permutations or 5
    else:
        windows = DEFAULT_WINDOWS
        n_perm = args.permutations or 10
    if args.windows:
        windows = windows[: args.windows]

    print(f"Auditing {len(windows)} windows × {n_perm} permutations each")
    print(f"(this is slow — each permutation runs one full backtest)\n")

    results: list[dict] = []
    for start, end in windows:
        results.append(_audit_window(start, end, n_perm))

    significant = sum(
        1 for r in results
        if r.get("permutation_test", {}).get("verdict") == "SIGNIFICANT"
    )
    high_contam = sum(
        1 for r in results
        if r.get("label_audit", {}).get("verdict") == "HIGH_CONTAMINATION"
    )

    print(f"\n──────── SUMMARY ────────")
    print(f"  significant signal: {significant}/{len(windows)} windows")
    print(f"  high contamination: {high_contam}/{len(windows)} windows")
    if significant >= len(windows) // 2 and high_contam <= 1:
        print(f"  ✅ PASS: backtest signal looks real and labels are mostly clean")
    elif significant == 0:
        print(f"  ❌ FAIL: NO window shows statistically significant signal")
    else:
        print(f"  ⚠️  WARN: mixed — review per-window detail above")

    out_path = ROOT / "data" / "overfitting_audit.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({
        "ran_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_windows": len(windows),
        "n_permutations_per_window": n_perm,
        "significant_count": significant,
        "high_contamination_count": high_contam,
        "results": results,
    }, indent=2))
    print(f"\nFull results → {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
