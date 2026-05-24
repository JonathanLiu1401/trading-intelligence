"""Recap noise rate by source — per-source content-aware noise factory detector.

For every source with a meaningful sample of HIGH-RELEVANCE articles
(ai_score >= AI_FLOOR Sonnet-labeled, OR ml_score >= ML_FLOOR model-flagged),
compute what fraction matches ANY recap-template fingerprint from
``watchers.alert_agent._RECAP_TEMPLATE_PATTERNS``. Sources with a high
``recap_rate`` are SEO/algorithmic mills the operator should consider gating
at the COLLECTOR layer (credibility reduction or skipping entirely) instead
of relying on the alert / briefing gates to drop them every cycle.

Distinct signal from ``junk_source_detector`` (which uses raw title-uniqueness
prefix counts — any source with diverse template titles passes that check
unchanged). MarketBeat, simplywall.st, bloomingbit, GuruFocus and similar mills
emit *content-diverse* but *structurally retrospective* titles, so the
uniqueness-ratio detector cannot see them; this module is exactly that gap.

Pure read-only: single SELECT, no DB writes. backtest:// URLs and
backtest_* / opus_annotation* sources excluded via ``_LIVE_ONLY_CLAUSE`` —
load-bearing invariant intact by construction (no live signal mutation, no
ai_score / ml_score / score_source / urgency touches).

The ``build_report`` builder is a pure function that takes a list of
``(source, title, ai_score, ml_score)`` tuples and returns the JSON snapshot —
fully unit-testable without SQLite. ``main()`` wires the live DB to it.
"""
from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from storage.article_store import _LIVE_ONLY_CLAUSE
from watchers.alert_agent import _RECAP_TEMPLATE_PATTERNS

_LOCAL_DB = Path(__file__).resolve().parents[1] / "data" / "articles.db"
OUT_PATH = Path("/home/zeph/logs/recap_noise_by_source.json")

# A source needs at least this many high-relevance rows in the window to be
# evaluated; below this the sample is too small for a stable rate.
MIN_PER_SOURCE = 20
# Recap-rate at or above this fraction → flagged as a noise factory. 30% is the
# rough discriminator validated against the live evidence: MarketBeat and
# simplywall.st sit well above; mainstream wires (Reuters, Bloomberg, CNBC)
# sit well below.
NOISE_THRESHOLD = 0.30
# Lookback window. 72h is short enough that a one-day SEO-mill burst is
# visible; long enough that an evening-only feed isn't excluded.
LOOKBACK_HOURS = 72
# A row qualifies as "high relevance" (the sample we judge) when either the
# Sonnet label is >= AI_FLOOR or the model score is >= ML_FLOOR. These
# thresholds match the gates that decide "interesting" in the alert path
# (URGENT_THRESHOLD=8.0 for ml; ai>=5 is the RELEVANT band Sonnet emits).
AI_FLOOR = 5.0
ML_FLOOR = 8.0


def _resolve_db_path() -> Path:
    usb = Path(
        os.environ.get(
            "DIGITAL_INTERN_USB", "/media/zeph/projects/digital-intern/db"
        )
    ) / "articles.db"
    if usb.exists():
        return usb
    return _LOCAL_DB


def _matches_recap(title: str) -> tuple[bool, str]:
    """``(True, fingerprint_name)`` on first match against the SSOT pattern
    tuple. ``(False, "")`` otherwise. Single source of truth — imports the
    same ``_RECAP_TEMPLATE_PATTERNS`` the alert path uses, so a new fingerprint
    added there is automatically tracked here on next run (no double-bookkeeping).
    """
    if not title:
        return False, ""
    for name, pat in _RECAP_TEMPLATE_PATTERNS:
        if pat.search(title):
            return True, name
    return False, ""


def build_report(rows: list[tuple]) -> dict:
    """Pure builder.

    ``rows`` is a list of ``(source, title, ai_score, ml_score)`` tuples,
    pre-filtered by the caller to the high-relevance sample. The ``ai_score`` /
    ``ml_score`` columns are accepted but currently not used in the math —
    they're carried through so a future enhancement (e.g. weighting recap
    rate by score magnitude) is additive.

    Returns the JSON-serialisable report dict. Sources below ``MIN_PER_SOURCE``
    rows are excluded so we never report a 100% rate from a sample of 2.
    Result is deterministic: sources are sorted by descending ``recap_rate``
    so the worst offenders are first.
    """
    per_source: dict[str, dict] = {}
    for source, title, _ai, _ml in rows:
        src = source or "(unknown)"
        s = per_source.setdefault(
            src, {"total": 0, "recap": 0, "fingerprints": {}}
        )
        s["total"] += 1
        hit, name = _matches_recap(title or "")
        if hit:
            s["recap"] += 1
            s["fingerprints"][name] = s["fingerprints"].get(name, 0) + 1

    sources: list[dict] = []
    noise_sources: list[str] = []
    for src, s in per_source.items():
        if s["total"] < MIN_PER_SOURCE:
            continue
        rate = s["recap"] / s["total"]
        is_noise = rate >= NOISE_THRESHOLD
        entry = {
            "source": src,
            "high_relevance_count": s["total"],
            "recap_count": s["recap"],
            "recap_rate": round(rate, 4),
            "is_noise_factory": is_noise,
            # Top-3 fingerprints by count so the operator can see WHICH gates
            # this source primarily trips (e.g. simplywall.st → earnings_release_pt;
            # MarketBeat → subject_pct_after + holdings_by_fund).
            "top_fingerprints": sorted(
                s["fingerprints"].items(), key=lambda kv: -kv[1]
            )[:3],
        }
        sources.append(entry)
        if is_noise:
            noise_sources.append(src)
    # Deterministic ordering: worst-offender (highest recap_rate) first; tie
    # break by source name so the output is stable across runs.
    sources.sort(key=lambda e: (-e["recap_rate"], e["source"]))
    return {
        "lookback_hours": LOOKBACK_HOURS,
        "min_per_source": MIN_PER_SOURCE,
        "noise_threshold": NOISE_THRESHOLD,
        "ai_floor": AI_FLOOR,
        "ml_floor": ML_FLOOR,
        "evaluated_sources": len(sources),
        "noise_source_count": len(noise_sources),
        "noise_sources": noise_sources,
        "sources": sources,
    }


def main() -> int:
    db_path = _resolve_db_path()
    if not db_path.exists():
        print(f"recap_noise_by_source: no DB at {db_path}")
        return 1
    conn = sqlite3.connect(
        f"file:{db_path}?mode=ro", uri=True, timeout=15
    )
    try:
        rows = conn.execute(
            f"SELECT source, title, ai_score, ml_score FROM articles "
            f"WHERE first_seen > datetime('now', '-{LOOKBACK_HOURS} hours') "
            f"AND (ai_score >= ? OR ml_score >= ?) "
            f"AND {_LIVE_ONLY_CLAUSE}",
            (AI_FLOOR, ML_FLOOR),
        ).fetchall()
    finally:
        conn.close()

    report = build_report(rows)
    report["generated_at"] = datetime.now(timezone.utc).isoformat()
    report["scanned_rows"] = len(rows)
    report["db_path"] = str(db_path)

    try:
        OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        OUT_PATH.write_text(json.dumps(report, indent=2))
    except Exception as e:
        print(f"recap_noise_by_source: write failed: {e}")

    n_noise = report["noise_source_count"]
    print(
        f"recap_noise_by_source: scanned={len(rows)} "
        f"sources_evaluated={report['evaluated_sources']} "
        f"noise_factories={n_noise}"
    )
    if n_noise:
        threshold_pct = int(NOISE_THRESHOLD * 100)
        print(f"  NOISE FACTORIES (recap_rate >= {threshold_pct}%):")
        for entry in report["sources"]:
            if not entry["is_noise_factory"]:
                continue
            fps = ", ".join(
                f"{n}={c}" for n, c in entry["top_fingerprints"]
            )
            print(
                f"    {entry['source']}: "
                f"rate={entry['recap_rate']:.1%} "
                f"({entry['recap_count']}/{entry['high_relevance_count']}) "
                f"top: {fps}"
            )
    print(f"  Report written to {OUT_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
