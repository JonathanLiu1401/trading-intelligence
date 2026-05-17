"""Decision-pipeline health — is the live Opus trader actually deciding?

The ``decisions`` table records one row per cycle (every 60s when the NYSE is
open, 3600s when closed). Most cycles end in HOLD — fine — but a large share end
in ``NO_DECISION``: the reasoning column reads "claude returned no parseable
JSON". That is a *silent reliability failure*. Opus was invoked, the cycle was
spent, and nothing parseable came back. Nothing on the dashboard surfaces it, so
a trader has no idea whether the bot is healthy or quietly broken.

This module reads the ``decisions`` table and reports:

* **action mix** — NO_DECISION / HOLD / FILLED / BLOCKED counts and shares
* **parse-failure rate** over rolling windows (24h / 7d / all)
* **confidence trend** — mean self-assessed confidence, recent vs older
* **cadence** — decisions/day, time since last cycle, time since last *fill*
* **signal volume** — how many news signals the bot sees per cycle

A coarse verdict (HEALTHY / DEGRADED / CRITICAL) rolls it up so the trader sees
one word before reading the detail.

``build_decision_health`` is pure: pass in the row list from
``store.recent_decisions(limit)`` (newest-first) and it returns a JSON-ready
dict. It never touches the DB.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

# Outcome categories, in display order. NO_DECISION is the failure mode.
_CATEGORIES = ["FILLED", "HOLD", "BLOCKED", "NO_DECISION", "OTHER"]


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _classify(action_taken: str | None) -> tuple[str, str]:
    """Map a free-text ``action_taken`` to (category, action_verb).

    ``action_taken`` is one of: ``"NO_DECISION"``, ``"HOLD MU → HOLD"``,
    ``"BUY SOXL → FILLED"``, ``"SELL NVDA → BLOCKED"`` — see store.py / strategy.py.
    """
    raw = (action_taken or "").strip()
    if not raw or raw == "NO_DECISION":
        return "NO_DECISION", ""
    verb = raw.split()[0].upper() if raw.split() else ""
    outcome = raw.split("→")[-1].strip().upper() if "→" in raw else raw.upper()
    if outcome in ("FILLED", "HOLD", "BLOCKED"):
        return outcome, verb
    return "OTHER", verb


def _confidence(reasoning: str | None) -> float | None:
    """Pull Opus's self-assessed confidence out of a decision's reasoning JSON."""
    if not reasoning:
        return None
    try:
        inner = (json.loads(reasoning).get("decision") or {})
        c = inner.get("confidence")
        return float(c) if isinstance(c, (int, float)) else None
    except Exception:
        return None


def _window(rows: list[dict], now: datetime, hours: float | None) -> dict:
    """Action-mix counts for rows newer than ``hours`` ago (None ⇒ all rows)."""
    counts = {c: 0 for c in _CATEGORIES}
    total = 0
    for r in rows:
        if hours is not None:
            ts = _parse_ts(r.get("timestamp"))
            if ts is None or (now - ts).total_seconds() > hours * 3600:
                continue
        cat, _ = _classify(r.get("action_taken"))
        counts[cat] += 1
        total += 1
    pct = lambda n: round(n / total * 100, 1) if total else 0.0
    return {
        "total": total,
        **{c.lower(): counts[c] for c in _CATEGORIES},
        "parse_fail_pct": pct(counts["NO_DECISION"]),
        "fill_pct": pct(counts["FILLED"]),
        "hold_pct": pct(counts["HOLD"]),
    }


def build_decision_health(decisions: list[dict]) -> dict:
    """Health report over the ``decisions`` table (newest-first row list)."""
    now = datetime.now(timezone.utc)
    n = len(decisions)
    out: dict = {
        "as_of": now.isoformat(timespec="seconds"),
        "n_decisions": n,
        "verdict": "NO_DATA",
        "verdict_reason": "no decisions recorded yet",
        "windows": {},
        "action_mix": [],
        "confidence": {},
        "cadence": {},
        "signal_count": {},
        "recent": [],
    }
    if not decisions:
        return out

    w24, w7d, wall = (_window(decisions, now, 24),
                      _window(decisions, now, 24 * 7),
                      _window(decisions, now, None))
    out["windows"] = {"24h": w24, "7d": w7d, "all": wall}

    # ── action mix (all-time) ───────────────────────────────────────────
    out["action_mix"] = [
        {"category": c, "n": wall[c.lower()], "pct": round(wall[c.lower()] / n * 100, 1)}
        for c in _CATEGORIES if wall[c.lower()] > 0
    ]

    # ── confidence: mean, plus recent-half vs older-half trend ──────────
    confs = [(_parse_ts(r.get("timestamp")), _confidence(r.get("reasoning")))
             for r in decisions]
    confs = [(t, c) for t, c in confs if c is not None]
    if confs:
        vals = [c for _, c in confs]
        # decisions is newest-first → first half is the *recent* half.
        half = max(1, len(vals) // 2)
        recent_avg = sum(vals[:half]) / half
        older = vals[half:]
        older_avg = (sum(older) / len(older)) if older else recent_avg
        delta = recent_avg - older_avg
        trend = "rising" if delta > 0.03 else "falling" if delta < -0.03 else "flat"
        out["confidence"] = {
            "n": len(vals),
            "avg": round(sum(vals) / len(vals), 3),
            "recent_avg": round(recent_avg, 3),
            "older_avg": round(older_avg, 3),
            "trend": trend,
        }

    # ── cadence: per-day rate, gap since last cycle / last fill ─────────
    ts_sorted = sorted([t for t in (_parse_ts(r.get("timestamp")) for r in decisions) if t])
    cadence: dict = {}
    if ts_sorted:
        span_h = (ts_sorted[-1] - ts_sorted[0]).total_seconds() / 3600
        cadence["per_day"] = round(n / (span_h / 24), 1) if span_h > 1 else None
        cadence["last_decision_ts"] = ts_sorted[-1].isoformat(timespec="seconds")
        cadence["minutes_since_last"] = round((now - ts_sorted[-1]).total_seconds() / 60, 1)
    last_fill = next((r for r in decisions if _classify(r.get("action_taken"))[0] == "FILLED"), None)
    if last_fill:
        ft = _parse_ts(last_fill.get("timestamp"))
        cadence["last_fill_ts"] = ft.isoformat(timespec="seconds") if ft else None
        cadence["hours_since_fill"] = round((now - ft).total_seconds() / 3600, 1) if ft else None
        cadence["last_fill_action"] = last_fill.get("action_taken")
    else:
        cadence["last_fill_ts"] = None
        cadence["hours_since_fill"] = None
    out["cadence"] = cadence

    # ── signal volume per cycle ─────────────────────────────────────────
    sigs = [r.get("signal_count") for r in decisions
            if isinstance(r.get("signal_count"), (int, float))]
    if sigs:
        half = max(1, len(sigs) // 2)
        out["signal_count"] = {
            "avg": round(sum(sigs) / len(sigs), 1),
            "min": min(sigs),
            "max": max(sigs),
            "recent_avg": round(sum(sigs[:half]) / half, 1),
        }

    # ── recent decision tape ────────────────────────────────────────────
    for r in decisions[:18]:
        cat, verb = _classify(r.get("action_taken"))
        ts = _parse_ts(r.get("timestamp"))
        out["recent"].append({
            "timestamp": ts.isoformat(timespec="seconds") if ts else r.get("timestamp"),
            "category": cat,
            "action": r.get("action_taken"),
            "confidence": _confidence(r.get("reasoning")),
            "signal_count": r.get("signal_count"),
            "market_open": bool(r.get("market_open")),
        })

    # ── verdict — judged on the freshest window with ≥10 samples ────────
    judged = w24 if w24["total"] >= 10 else (w7d if w7d["total"] >= 10 else wall)
    fail = judged["parse_fail_pct"]
    if judged["total"] == 0:
        out["verdict"], out["verdict_reason"] = "NO_DATA", "no decisions in any window"
    elif fail >= 50:
        out["verdict"] = "CRITICAL"
        out["verdict_reason"] = (f"{fail:.0f}% of recent cycles produced no parseable "
                                 f"decision — Opus output is failing to parse")
    elif fail >= 25:
        out["verdict"] = "DEGRADED"
        out["verdict_reason"] = (f"{fail:.0f}% parse-failure rate — elevated; "
                                 f"check strategy._parse_decision and Opus timeouts")
    else:
        out["verdict"] = "HEALTHY"
        out["verdict_reason"] = f"parse-failure rate {fail:.0f}% — within normal range"
    out["verdict_window"] = ("24h" if judged is w24
                             else "7d" if judged is w7d else "all")
    return out


if __name__ == "__main__":  # smoke test
    from paper_trader.store import get_store
    rep = build_decision_health(get_store().recent_decisions(limit=2000))
    print(json.dumps(rep, indent=2, default=str))
