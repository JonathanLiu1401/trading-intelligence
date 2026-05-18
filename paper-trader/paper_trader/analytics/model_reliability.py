"""Model reliability — *which model actually made each live decision*, and
how often the cycle produced nothing at all.

The live trader is tuned end-to-end around Opus 4.7's reasoning depth
(paper-trader AGENTS.md / CLAUDE.md invariant #3 — "the entire system prompt
is tuned around Opus's reasoning depth; do not downgrade to Sonnet"). But
``strategy.decide()`` has a degrade ladder: Opus → (timeout) Sonnet fallback
on a *condensed* prompt → (still nothing) NO_DECISION. Every existing panel
is blind to *which* arm fired:

* ``/api/decision-health`` buckets by **outcome** (FILLED/HOLD/BLOCKED/
  NO_DECISION). A trade Sonnet made on a stripped-down prompt is counted
  identically to one full Opus made — the operator cannot tell a
  reasoning-depth-tuned book is quietly being run by the fallback.
* ``/api/decision-forensics`` only sub-classifies the *NO_DECISION* excerpts.
  It never looks at the *successful* population, so a Sonnet-rescued FILLED
  is invisible to it too.

This module reads the same ``decisions`` rows and answers the question the
two above cannot: **of the decisions that were actually made, what share was
full Opus vs the degraded Sonnet fallback — and is that share getting
worse?** Plus the money-relevant cut: **what fraction of executed trades
(FILLED) were placed by the fallback model, not Opus?**

Data source (no new state — pure walk of ``store.recent_decisions()``):

* A *made* decision stores ``reasoning`` as
  ``json.dumps({"decision": {...}, "auto_exits": [...], "detail": "...",
  "fallback_used": bool})`` (strategy.py). ``fallback_used`` is the
  authoritative arm signal. **Rows written before that key existed read
  back ``None``** (verified live: 18/60 recent rows) — they are bucketed
  ``legacy_unknown`` and EXCLUDED from the Opus/fallback ratio so a stale
  history can never fake a healthy (or unhealthy) reliability number.
* A *NO_DECISION* row stores a plain string: ``"claude returned no response
  (timeout/empty)"`` (Opus timed out *and* Sonnet fallback also failed —
  the hard failure), ``"parse_failed: <excerpt>"`` (Opus replied,
  unparseable, no retry rescue) or ``"retry_failed: <excerpt>"`` (the
  JSON-only retry also failed). These mirror strategy.py's exact prefixes.

``build_model_reliability`` is pure and never raises: pass the newest-first
row list from ``store.recent_decisions(limit)`` and it returns a JSON-ready
dict. It never touches the DB or the network.

**Observational only.** It reports; it never gates Opus, imposes no caps,
and feeds nothing back into the decision prompt (paper-trader AGENTS.md
invariants #2/#12 — the ``decision_health`` / ``self_review`` precedent). A
mirror, not a cage.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

# Arm of the degrade ladder that produced each row. Order = display order.
_ARMS = [
    "opus",            # full Opus made the decision (fallback_used == False)
    "sonnet_fallback", # Opus timed out, Sonnet rescued it (fallback_used True)
    "legacy_unknown",  # made, but pre-`fallback_used` row → arm unknowable
    "timeout",         # NO_DECISION: "claude returned no response"
    "parse_failed",    # NO_DECISION: Opus replied unparseable, no retry rescue
    "retry_failed",    # NO_DECISION: the JSON-only retry also failed
    "other_no_dec",    # NO_DECISION with an unrecognised reason string
]

# Minimum *attributable* decisions (opus + sonnet_fallback) before a verdict
# is asserted — the decision_health / benchmark sample-size-honesty
# precedent. Below this the ratio is too noisy to act on.
_MIN_ATTRIBUTABLE = 10

# Opus-share verdict bands (share of attributable decisions made by Opus).
_HEALTHY_OPUS_SHARE = 90.0   # ≥ this % Opus → HEALTHY
_DEGRADED_OPUS_SHARE = 70.0  # ≥ this % (and < HEALTHY) → DEGRADED, else FAILING


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _outcome(action_taken: str | None) -> str:
    """FILLED / HOLD / BLOCKED / NO_DECISION from the free-text column.

    ``action_taken`` is ``"BUY NVDA → FILLED"`` / ``"HOLD MU → HOLD"`` /
    ``"SELL X → BLOCKED"`` / ``"NO_DECISION"`` (store.py / strategy.py). A
    non-str value (corrupt/garbage row) degrades to NO_DECISION, never
    raises — the builder's "never raises" contract."""
    raw = (action_taken if isinstance(action_taken, str) else "").strip()
    if not raw or raw == "NO_DECISION":
        return "NO_DECISION"
    if "→" in raw:
        tail = raw.split("→")[-1].strip().upper()
        if tail in ("FILLED", "HOLD", "BLOCKED"):
            return tail
    return "NO_DECISION"


def _classify(row: dict) -> str:
    """Map one decisions row to an arm in ``_ARMS``.

    A made decision (outcome != NO_DECISION) is attributed by the
    authoritative ``fallback_used`` flag in its reasoning JSON; a row
    predating that flag is ``legacy_unknown``. A NO_DECISION row is
    sub-classified by strategy.py's exact reason prefix."""
    outcome = _outcome(row.get("action_taken"))
    raw_reasoning = row.get("reasoning")
    # Coerce defensively: a corrupt row may carry a non-str reasoning. All
    # string ops below (json.loads / startswith) must tolerate it without
    # raising — the builder's "never raises" contract.
    reasoning = raw_reasoning if isinstance(raw_reasoning, str) else ""
    if outcome != "NO_DECISION":
        try:
            obj = json.loads(reasoning)
            used = obj.get("fallback_used") if isinstance(obj, dict) else None
        except Exception:
            used = None
        if used is True:
            return "sonnet_fallback"
        if used is False:
            return "opus"
        return "legacy_unknown"
    # NO_DECISION — sub-classify by reason prefix (strategy.py is the source
    # of these exact strings).
    s = reasoning.strip()
    if s.startswith("claude returned no response"):
        return "timeout"
    if s.startswith("parse_failed:"):
        return "parse_failed"
    if s.startswith("retry_failed:"):
        return "retry_failed"
    return "other_no_dec"


def _window(rows: list[dict], now: datetime, hours: float | None) -> dict:
    """Arm + outcome tallies for rows newer than ``hours`` ago (None ⇒ all)."""
    arms = {a: 0 for a in _ARMS}
    total = 0
    filled_total = 0
    filled_fallback = 0
    for r in rows:
        if hours is not None:
            ts = _parse_ts(r.get("timestamp"))
            if ts is None or (now - ts).total_seconds() > hours * 3600:
                continue
        arm = _classify(r)
        arms[arm] += 1
        total += 1
        if _outcome(r.get("action_taken")) == "FILLED":
            filled_total += 1
            if arm == "sonnet_fallback":
                filled_fallback += 1

    made = arms["opus"] + arms["sonnet_fallback"] + arms["legacy_unknown"]
    attributable = arms["opus"] + arms["sonnet_fallback"]
    no_decision = (arms["timeout"] + arms["parse_failed"]
                   + arms["retry_failed"] + arms["other_no_dec"])

    def pct(n: int, d: int) -> float:
        return round(n / d * 100, 1) if d else 0.0

    return {
        "total": total,
        "made": made,
        "attributable": attributable,
        "no_decision": no_decision,
        **{a: arms[a] for a in _ARMS},
        # Share of *attributable* decisions made by full Opus — the headline
        # reliability number (legacy_unknown deliberately excluded).
        "opus_share_pct": pct(arms["opus"], attributable),
        "fallback_share_pct": pct(arms["sonnet_fallback"], attributable),
        # Share of *all* cycles that produced nothing.
        "no_decision_pct": pct(no_decision, total),
        # Of executed trades, how many were placed by the degraded fallback.
        "filled_total": filled_total,
        "filled_fallback": filled_fallback,
        "filled_fallback_pct": pct(filled_fallback, filled_total),
    }


def build_model_reliability(decisions: list[dict],
                            now: datetime | None = None) -> dict:
    """Reliability report over the ``decisions`` table (newest-first rows).

    ``state`` ∈ ``NO_DATA`` → ``INSUFFICIENT`` (fewer than
    ``_MIN_ATTRIBUTABLE`` attributable decisions all-time; numerics still
    emitted, ``verdict`` withheld) → ``OK`` (``verdict`` ∈
    ``OPUS_HEALTHY`` / ``DEGRADED`` / ``FAILING``). ``headline`` is the
    single string the endpoint / CLI / any Discord line render, so they
    cannot drift. Pure; never raises."""
    now = now or datetime.now(timezone.utc)
    n = len(decisions or [])
    out: dict = {
        "as_of": now.isoformat(timespec="seconds"),
        "n_decisions": n,
        "state": "NO_DATA",
        "verdict": None,
        "headline": "No decisions recorded yet — nothing to attribute.",
        "windows": {},
        "trend": None,
        "recent_fallback": [],
    }
    if not decisions:
        return out

    w24 = _window(decisions, now, 24)
    w7d = _window(decisions, now, 24 * 7)
    wall = _window(decisions, now, None)
    out["windows"] = {"24h": w24, "7d": w7d, "all": wall}

    attributable_all = wall["attributable"]

    # Opus-share trend: recent half vs older half of the *attributable*
    # decisions. decisions is newest-first, so the first attributable rows
    # encountered are the recent ones.
    attr_rows = [_classify(r) for r in decisions]
    attr_rows = [a for a in attr_rows if a in ("opus", "sonnet_fallback")]
    trend = None
    if len(attr_rows) >= 4:
        half = len(attr_rows) // 2
        recent, older = attr_rows[:half], attr_rows[half:]
        r_share = recent.count("opus") / len(recent) * 100 if recent else 0.0
        o_share = older.count("opus") / len(older) * 100 if older else 0.0
        delta = r_share - o_share
        direction = ("improving" if delta > 5
                     else "worsening" if delta < -5 else "flat")
        trend = {
            "recent_opus_share_pct": round(r_share, 1),
            "older_opus_share_pct": round(o_share, 1),
            "delta_pp": round(delta, 1),
            "direction": direction,
        }
    out["trend"] = trend

    # A few of the most recent fallback / hard-failure rows so the operator
    # can see concrete timestamps without opening the DB.
    flagged = []
    for r in decisions:
        arm = _classify(r)
        if arm in ("sonnet_fallback", "timeout", "retry_failed"):
            flagged.append({
                "ts": r.get("timestamp"),
                "arm": arm,
                "action": r.get("action_taken"),
            })
        if len(flagged) >= 8:
            break
    out["recent_fallback"] = flagged

    if attributable_all < _MIN_ATTRIBUTABLE:
        out["state"] = "INSUFFICIENT"
        out["headline"] = (
            f"Model attribution maturing — only {attributable_all} "
            f"attributable decision(s) (need ≥{_MIN_ATTRIBUTABLE}); "
            f"{wall['legacy_unknown']} legacy rows excluded. "
            f"NO_DECISION {wall['no_decision_pct']:.0f}% all-time."
        )
        return out

    opus_share = wall["opus_share_pct"]
    out["state"] = "OK"
    if opus_share >= _HEALTHY_OPUS_SHARE:
        out["verdict"] = "OPUS_HEALTHY"
    elif opus_share >= _DEGRADED_OPUS_SHARE:
        out["verdict"] = "DEGRADED"
    else:
        out["verdict"] = "FAILING"

    nd24 = w24["no_decision_pct"]
    out["headline"] = (
        f"{out['verdict']}: Opus made {opus_share:.0f}% of "
        f"{attributable_all} attributable decisions "
        f"({wall['fallback_share_pct']:.0f}% Sonnet fallback); "
        f"{wall['filled_fallback']}/{wall['filled_total']} executed trades "
        f"were placed by the fallback. NO_DECISION {nd24:.0f}% last 24h "
        f"(timeout {w24['timeout']}, parse {w24['parse_failed']}, "
        f"retry {w24['retry_failed']})."
    )
    return out


if __name__ == "__main__":  # one-screen answer, usable when :8090 is wedged
    import sqlite3
    import sys
    from pathlib import Path

    db = Path(__file__).resolve().parents[2] / "data" / "paper_trader.db"
    try:
        c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        c.row_factory = sqlite3.Row
        rows = [dict(r) for r in c.execute(
            "SELECT timestamp, action_taken, reasoning FROM decisions "
            "ORDER BY timestamp DESC, id DESC LIMIT 3000").fetchall()]
        c.close()
    except Exception as e:  # the benchmark / signals --check-freshness CLI precedent
        print(f"model-reliability: cannot read {db}: {e}")
        sys.exit(2)

    rep = build_model_reliability(rows)
    if "--json" in sys.argv:
        print(json.dumps(rep, indent=2, default=str))
    else:
        tag = rep["state"] + (f"/{rep['verdict']}" if rep["verdict"] else "")
        print(f"MODEL RELIABILITY  [{tag}]  {rep['headline']}")
        for name in ("24h", "7d", "all"):
            w = (rep.get("windows") or {}).get(name) or {}
            if not w:
                continue
            print(f"  {name:>3}: opus {w['opus']}  fallback "
                  f"{w['sonnet_fallback']}  legacy {w['legacy_unknown']}  "
                  f"| timeout {w['timeout']}  parse {w['parse_failed']}  "
                  f"retry {w['retry_failed']}  "
                  f"(opus_share {w['opus_share_pct']}%)")
        if rep.get("trend"):
            t = rep["trend"]
            print(f"  trend: {t['direction']} "
                  f"({t['older_opus_share_pct']}% → "
                  f"{t['recent_opus_share_pct']}%)")
