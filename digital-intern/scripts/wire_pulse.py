"""One-line news-intelligence pipeline status — the operator's "is the
wire alive?" snapshot.

The codebase carries 120+ analytics modules, each focused on a slice
(``label_production_rate``, ``urgency_label_split``, ``briefing_health``,
``source_freshness``, ...). What's missing is a SINGLE-LINE roll-up that
fits in a Discord status push, a cron heartbeat, or a fast grep — the
"glance at it and know" view the analyst persona needs when they wake up
and want to confirm the pipeline didn't quietly die overnight.

Output format::

    [wire_pulse <iso_ts>] articles_1h=N urgent_1h=N llm_vetted_pct=NN
    briefing_age_h=N.N last_alert_age_h=N.N → <VERDICT>

VERDICT ladder:
  * ``HEALTHY``      — fresh ingest, recent alert, briefing within cadence.
  * ``BRIEFING_STALE`` — last briefing > 12h ago (briefing path dark).
  * ``ALERT_QUIET``  — no urgent rows in the last hour AND no alert in
                       the last 6h. Could be a quiet news day; could be
                       the scoring path dark — operator should check.
  * ``INGEST_DARK``  — fewer than 30 articles ingested in the last hour
                       (live news stopped flowing — collector outage).
  * ``UNKNOWN``      — unable to read the DB (mount issue / file corrupt).

This is the inverse of the existing analytics modules' "deep dive once you
know something is off" shape — it tells you WHETHER something is off at a
glance, and points at WHICH primitive to dive into.

Read-only over articles.db and alert_recency.db. NO DB write, no
ai_score/ml_score/score_source/urgency mutation. All four load-bearing
invariants intact by construction.

Usage::

    python3 scripts/wire_pulse.py             # one-line stdout
    python3 scripts/wire_pulse.py --json      # machine-readable JSON
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# Verdict thresholds — conservative, evidence-driven. The point of this
# script is "loud when something is actually broken"; chatty false alarms
# would teach the operator to ignore the heartbeat.
INGEST_DARK_MIN_1H = 30          # < this many articles/h = collectors down
BRIEFING_STALE_AFTER_H = 12.0    # 2× the 5h cadence
ALERT_QUIET_AFTER_H = 6.0        # ALERT_RECENCY_TTL_HOURS — alerts are
                                  # silently retained for 6h, so longer
                                  # silence is a real quiet stretch.


def _compute(store) -> dict:
    """Compose the three existing primitives into a single rollup.

    Each sub-read is wrapped so a partial failure (e.g. alert_recency DB
    missing on a fresh install) still produces a meaningful snapshot
    instead of crashing the whole rollup. The verdict is computed AFTER
    the reads, so a missing component shows up as ``None`` in the JSON
    and the verdict ladder treats it as "data unavailable" rather than
    silently degrading to HEALTHY."""
    out: dict = {
        "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    # Ingest rate — recent traffic volume. ``stats_since`` is a fast index
    # read so a 1h window is cheap; the live-only clause keeps synthetic
    # backtest injections out of the count.
    try:
        s1h = store.stats_since(hours=1)
        out["articles_1h"] = int(s1h.get("total") or 0)
        out["urgent_1h"] = int(s1h.get("urgent") or 0)
    except Exception as e:
        out["articles_1h"] = None
        out["urgent_1h"] = None
        out["_ingest_err"] = str(e)[:120]

    # LLM-vetted fraction — calibration of the urgent backlog. A persistent
    # near-zero ``llm_fraction`` means the Sonnet path is dark, exactly the
    # signal the urgency_label_split docstring calls out.
    try:
        split = store.urgency_label_split(hours=24)
        total = int(split.get("total") or 0)
        out["urgent_24h"] = total
        out["llm_vetted_pct"] = round(
            100.0 * float(split.get("llm_fraction") or 0.0), 1
        )
    except Exception as e:
        out["urgent_24h"] = None
        out["llm_vetted_pct"] = None
        out["_split_err"] = str(e)[:120]

    # Briefing freshness — the 5h Opus product. ``briefing_health`` is the
    # canonical "is the briefing path healthy?" read.
    try:
        bh = store.briefing_health(window_h=24)
        age_h = bh.get("last_briefing_age_h")
        out["briefing_age_h"] = (round(float(age_h), 1)
                                  if age_h is not None else None)
        out["briefing_verdict"] = bh.get("verdict")
    except Exception as e:
        out["briefing_age_h"] = None
        out["briefing_verdict"] = None
        out["_briefing_err"] = str(e)[:120]

    # Last-pushed alert — pulled from alert_recency.db (the canonical record
    # of REAL Discord pushes; gate-suppressed rows do NOT write here, see
    # send_urgent_alert's success path). A long stretch without a push is
    # part of the verdict ladder.
    try:
        from watchers.alert_recency import recent_alerts
        recent = recent_alerts(ttl_hours=24)  # newest-first
        if recent:
            out["last_alert_age_h"] = round(
                float(recent[0].get("age_hours") or 0.0), 2
            )
            out["alerts_24h"] = len(recent)
        else:
            out["last_alert_age_h"] = None
            out["alerts_24h"] = 0
    except Exception as e:
        out["last_alert_age_h"] = None
        out["alerts_24h"] = None
        out["_alert_err"] = str(e)[:120]

    out["verdict"] = _verdict(out)
    return out


def _verdict(snap: dict) -> str:
    """Walk the verdict ladder. INGEST_DARK overrides everything else
    (no point grading downstream when nothing is coming in), then briefing
    staleness (the briefing path is the slowest, so a stale briefing IS a
    failure signal even if the wire is live), then alert silence. HEALTHY
    is the no-news-is-good-news terminal."""
    articles_1h = snap.get("articles_1h")
    if articles_1h is None:
        # Couldn't read the DB at all — operator should investigate.
        return "UNKNOWN"
    if articles_1h < INGEST_DARK_MIN_1H:
        return "INGEST_DARK"
    age_h = snap.get("briefing_age_h")
    if age_h is not None and age_h > BRIEFING_STALE_AFTER_H:
        return "BRIEFING_STALE"
    # Briefing verdict from briefing_health overrides the raw age check
    # for the DEAD case (it uses the same threshold but is the canonical
    # read).
    if snap.get("briefing_verdict") == "DEAD":
        return "BRIEFING_STALE"
    last_alert = snap.get("last_alert_age_h")
    urgent_1h = snap.get("urgent_1h") or 0
    if last_alert is None and urgent_1h == 0:
        # No fresh urgent rows AND no recent pushes — could be quiet, could
        # be dark. Loud but not catastrophic.
        return "ALERT_QUIET"
    if last_alert is not None and last_alert > ALERT_QUIET_AFTER_H and urgent_1h == 0:
        return "ALERT_QUIET"
    return "HEALTHY"


def _format_line(snap: dict) -> str:
    """One-line human-readable summary. Field order is fixed so a grep /
    awk pipeline can rely on it."""
    def _f(key: str, fmt: str = "?", missing: str = "?") -> str:
        v = snap.get(key)
        if v is None:
            return missing
        return fmt.format(v)

    return (
        f"[wire_pulse {snap['ts']}] "
        f"articles_1h={_f('articles_1h', '{}')} "
        f"urgent_1h={_f('urgent_1h', '{}')} "
        f"llm_vetted_pct={_f('llm_vetted_pct', '{}')} "
        f"briefing_age_h={_f('briefing_age_h', '{}')} "
        f"last_alert_age_h={_f('last_alert_age_h', '{}')} "
        f"→ {snap.get('verdict', 'UNKNOWN')}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="One-line wire-pipeline pulse for ops / cron heartbeat."
    )
    parser.add_argument(
        "--json", action="store_true",
        help="Emit the snapshot as JSON instead of the one-line summary.",
    )
    args = parser.parse_args(argv)

    try:
        from storage.article_store import ArticleStore
        store = ArticleStore()
    except Exception as e:
        snap = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "verdict": "UNKNOWN",
            "_store_err": str(e)[:200],
        }
        if args.json:
            print(json.dumps(snap))
        else:
            print(f"[wire_pulse {snap['ts']}] STORE_UNAVAILABLE — {snap['_store_err']}")
        return 2

    snap = _compute(store)
    if args.json:
        print(json.dumps(snap, indent=2))
    else:
        print(_format_line(snap))
    # Process exit code so cron / alerting can react: 0=healthy, 1=degraded,
    # 2=store unavailable. ALERT_QUIET is a soft warning (1); only the
    # ingest-dark / briefing-dead cases are hard fails (1 too — ops should
    # investigate; cron + grep verdict is the operator workflow).
    return 0 if snap.get("verdict") == "HEALTHY" else 1


if __name__ == "__main__":
    sys.exit(main())
