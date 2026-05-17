"""Decision-failure forensics — *why* the live Opus trader produces no decision.

``analytics/decision_health.py`` already reports the NO_DECISION *rate* and a
coarse verdict. It does not say **why** a cycle failed, and the dashboard never
surfaces the raw text Opus actually returned. ``strategy.py`` captures that text
into ``decisions.reasoning`` for every failed cycle:

* ``"parse_failed: <up-to-1000-char excerpt>"`` — Opus replied, first parse failed
* ``"retry_failed: <excerpt>"``                 — the JSON-only retry *also* failed
* ``"claude returned no response (timeout/empty)"`` — CLI timeout / empty stdout
* legacy ``"claude returned no parseable JSON"``    — pre-diagnostics code path
  (no excerpt; still present in older rows)

This module turns those opaque strings into an actionable failure taxonomy so an
operator can tell *truncation* (raise the timeout) from *prose-wrapping* (tighten
the prompt) from *timeouts* (CLI load / auth) at a glance.

``build_decision_forensics`` is pure: pass the row list from
``store.recent_decisions(limit)`` (newest-first) and it returns a JSON-ready
dict. ``classify_failure`` is the testable core — a single reasoning string in,
``{mode, tag, excerpt}`` out.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

# Failure *modes* — what an operator can act on. Ordered by display priority.
MODES = [
    "TIMEOUT_EMPTY",    # CLI timed out or returned nothing — retry can't help
    "TRUNCATED",        # response cut off mid-object (unbalanced braces)
    "NO_JSON",          # no '{' at all — refusal or pure prose
    "FENCED",            # wrapped in ``` fences and still unparseable
    "PROSE_WRAPPED",    # JSON present but preceded by commentary
    "MALFORMED_JSON",   # starts at '{', braces balanced, bad JSON syntax
    "EMPTY",            # tag present but excerpt blank
    "LEGACY_UNKNOWN",   # pre-diagnostics row — no excerpt was captured
    "OTHER",            # unrecognised reasoning text
]

# Most-actionable hint per dominant mode.
_HINTS = {
    "TIMEOUT_EMPTY": ("Opus is timing out or returning empty stdout — check "
                      "`claude` CLI auth and the 3-concurrent subprocess cap; "
                      "consider raising DECISION_TIMEOUT_S."),
    "TRUNCATED": ("Opus responses are being cut off mid-JSON — raise "
                  "DECISION_TIMEOUT_S or shorten the prompt payload."),
    "NO_JSON": ("Opus is replying with prose / a refusal and no JSON object — "
                "review the system prompt's JSON-only instruction."),
    "FENCED": ("Opus is wrapping JSON in ``` fences with malformed content — "
               "_parse_decision strips a clean fence, so the body itself is "
               "bad; tighten the schema example in the prompt."),
    "PROSE_WRAPPED": ("Opus is prefacing the JSON with commentary — reinforce "
                      "the 'start your response with {' instruction."),
    "MALFORMED_JSON": ("Opus emits JSON-shaped text with syntax errors — add a "
                       "stricter schema example to the system prompt."),
    "EMPTY": "Failure rows carry no excerpt — likely empty model output.",
    "LEGACY_UNKNOWN": ("These rows predate parse diagnostics; restart the "
                       "runner so new failures capture the raw excerpt."),
    "OTHER": "Unrecognised failure reasoning — inspect the raw rows.",
}

_EXCERPT_CAP = 280  # display cap; strategy.py already capped the stored text


def _parse_ts(ts: str | None) -> datetime | None:
    if not ts:
        return None
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _is_no_decision(action_taken: str | None) -> bool:
    """A failed cycle: strategy.py records action_taken exactly ``"NO_DECISION"``."""
    raw = (action_taken or "").strip()
    return not raw or raw == "NO_DECISION"


def _clean(text: str) -> str:
    """Strip control chars the way strategy.py does before display."""
    return "".join(ch for ch in text if ch >= " " or ch in "\t\n").strip()


def _classify_payload(payload: str) -> str:
    """Sub-classify the excerpt of a parse_failed/retry_failed row.

    Precedence is deliberate and pinned by tests: structural cut-off
    (TRUNCATED) outranks cosmetic issues (FENCED/PROSE_WRAPPED) because it is
    the most actionable — it points at the timeout / prompt length, not the
    schema wording.
    """
    p = payload.strip()
    if not p:
        return "EMPTY"
    first = p.find("{")
    if first == -1:
        return "NO_JSON"
    opens, closes = p.count("{"), p.count("}")
    if opens > closes:
        return "TRUNCATED"
    if "```" in p:
        return "FENCED"
    if p[:first].strip():
        return "PROSE_WRAPPED"
    return "MALFORMED_JSON"


def classify_failure(reasoning: str | None) -> dict:
    """Map one ``decisions.reasoning`` string → ``{mode, tag, excerpt}``.

    ``tag`` mirrors strategy.py's prefix (parse_failed / retry_failed /
    no_response / legacy / not_a_failure / other). ``mode`` is the
    operator-facing failure class from ``MODES``. ``excerpt`` is the cleaned,
    display-capped model text (empty when none was captured).
    """
    raw = (reasoning or "").strip()
    if not raw:
        return {"mode": "EMPTY", "tag": "none", "excerpt": ""}

    low = raw.lower()
    if raw.startswith(("parse_failed:", "retry_failed:")):
        tag = raw.split(":", 1)[0]
        payload = _clean(raw.split(":", 1)[1])
        return {
            "mode": _classify_payload(payload),
            "tag": tag,
            "excerpt": payload[:_EXCERPT_CAP],
        }
    if "no response" in low and ("timeout" in low or "empty" in low):
        return {"mode": "TIMEOUT_EMPTY", "tag": "no_response", "excerpt": ""}
    if "no parseable json" in low:  # legacy pre-diagnostics rows
        return {"mode": "LEGACY_UNKNOWN", "tag": "legacy", "excerpt": ""}
    # Non-failure rows store decision JSON ({"decision": {...}, ...}).
    if raw.startswith("{") and ('"decision"' in raw or '"action"' in raw):
        return {"mode": "OTHER", "tag": "not_a_failure", "excerpt": ""}
    return {"mode": "OTHER", "tag": "other", "excerpt": _clean(raw)[:_EXCERPT_CAP]}


def _bucket_hourly(failrows: list[tuple[datetime, bool]],
                   allrows: list[tuple[datetime, bool]],
                   now: datetime) -> list[dict]:
    """Per-hour total/failure counts over the last 24h, oldest→newest.

    Only hours with at least one decision are emitted (sparse, like the rest
    of the dashboard's time series)."""
    cutoff = now - timedelta(hours=24)
    tot: dict[datetime, int] = {}
    fail: dict[datetime, int] = {}
    for ts, _ in allrows:
        if ts < cutoff:
            continue
        h = ts.replace(minute=0, second=0, microsecond=0)
        tot[h] = tot.get(h, 0) + 1
    for ts, _ in failrows:
        if ts < cutoff:
            continue
        h = ts.replace(minute=0, second=0, microsecond=0)
        fail[h] = fail.get(h, 0) + 1
    out = []
    for h in sorted(tot):
        t = tot[h]
        f = fail.get(h, 0)
        out.append({
            "hour": h.isoformat(timespec="minutes"),
            "total": t,
            "failures": f,
            "fail_pct": round(f / t * 100, 1) if t else 0.0,
        })
    return out


def build_decision_forensics(decisions: list[dict],
                             now: datetime | None = None) -> dict:
    """Forensic breakdown of NO_DECISION cycles (newest-first row list).

    Pure: never touches the DB. ``now`` is injectable for deterministic tests.
    """
    now = now or datetime.now(timezone.utc)
    n = len(decisions)
    out: dict = {
        "as_of": now.isoformat(timespec="seconds"),
        "n_decisions": n,
        "n_failures": 0,
        "failure_rate_pct": 0.0,
        "failure_rate_24h_pct": 0.0,
        "verdict": "NO_DATA",
        "verdict_reason": "no decisions recorded yet",
        "mode_mix": [],
        "tag_mix": {},
        "retry_exhausted": 0,
        "by_market": {},
        "hourly": [],
        "recent_failures": [],
        "dominant_mode": None,
        "hint": "",
    }
    if not decisions:
        return out

    cutoff24 = now - timedelta(hours=24)
    mode_n: dict[str, int] = {}
    tag_n: dict[str, int] = {}
    by_mkt = {
        "open": {"total": 0, "failures": 0},
        "closed": {"total": 0, "failures": 0},
    }
    failrows: list[tuple[datetime, bool]] = []
    allrows: list[tuple[datetime, bool]] = []
    recent: list[dict] = []
    n_fail = n_fail_24 = n_tot_24 = retry_exhausted = 0

    for d in decisions:
        ts = _parse_ts(d.get("timestamp"))
        is_open = bool(d.get("market_open"))
        mkt = "open" if is_open else "closed"
        by_mkt[mkt]["total"] += 1
        if ts is not None:
            allrows.append((ts, is_open))
            if ts >= cutoff24:
                n_tot_24 += 1

        if not _is_no_decision(d.get("action_taken")):
            continue

        n_fail += 1
        by_mkt[mkt]["failures"] += 1
        cls = classify_failure(d.get("reasoning"))
        mode, tag = cls["mode"], cls["tag"]
        mode_n[mode] = mode_n.get(mode, 0) + 1
        tag_n[tag] = tag_n.get(tag, 0) + 1
        if tag == "retry_failed":
            retry_exhausted += 1
        if ts is not None:
            failrows.append((ts, is_open))
            if ts >= cutoff24:
                n_fail_24 += 1
        if len(recent) < 14:
            recent.append({
                "timestamp": ts.isoformat(timespec="seconds") if ts else d.get("timestamp"),
                "mode": mode,
                "tag": tag,
                "market_open": is_open,
                "excerpt": cls["excerpt"],
            })

    out["n_failures"] = n_fail
    out["failure_rate_pct"] = round(n_fail / n * 100, 1)
    out["failure_rate_24h_pct"] = round(n_fail_24 / n_tot_24 * 100, 1) if n_tot_24 else 0.0
    out["retry_exhausted"] = retry_exhausted
    out["tag_mix"] = tag_n
    out["recent_failures"] = recent

    out["mode_mix"] = sorted(
        ({"mode": m, "n": c, "pct": round(c / n_fail * 100, 1)}
         for m, c in mode_n.items()),
        key=lambda r: (-r["n"], MODES.index(r["mode"]) if r["mode"] in MODES else 99),
    )

    for side in ("open", "closed"):
        t = by_mkt[side]["total"]
        f = by_mkt[side]["failures"]
        by_mkt[side]["fail_pct"] = round(f / t * 100, 1) if t else 0.0
    out["by_market"] = by_mkt

    out["hourly"] = _bucket_hourly(failrows, allrows, now)

    if out["mode_mix"]:
        dom = out["mode_mix"][0]["mode"]
        out["dominant_mode"] = dom
        out["hint"] = _HINTS.get(dom, "")

    # Verdict — judged on the 24h window when it has ≥10 cycles, else lifetime.
    if n_tot_24 >= 10:
        rate, win = out["failure_rate_24h_pct"], "24h"
    else:
        rate, win = out["failure_rate_pct"], "all"
    out["verdict_window"] = win
    if n_fail == 0:
        out["verdict"] = "HEALTHY"
        out["verdict_reason"] = "no NO_DECISION cycles — every cycle parsed"
    elif rate >= 50:
        out["verdict"] = "CRITICAL"
        out["verdict_reason"] = (
            f"{rate:.0f}% of {win} cycles failed to parse — dominant mode "
            f"{out['dominant_mode']}")
    elif rate >= 25:
        out["verdict"] = "DEGRADED"
        out["verdict_reason"] = (
            f"{rate:.0f}% {win} parse-failure rate — dominant mode "
            f"{out['dominant_mode']}")
    else:
        out["verdict"] = "HEALTHY"
        out["verdict_reason"] = (
            f"parse-failure rate {rate:.0f}% ({win}) — within normal range")
    return out


if __name__ == "__main__":  # smoke test against the live DB
    import json
    from paper_trader.store import get_store
    rep = build_decision_forensics(get_store().recent_decisions(limit=2000))
    print(json.dumps(rep, indent=2, default=str))
