"""Decision-failure forensics — *why* the live Opus trader produces no decision.

``analytics/decision_health.py`` already reports the NO_DECISION *rate* and a
coarse verdict. It does not say **why** a cycle failed, and the dashboard never
surfaces the raw text Opus actually returned. ``strategy.py`` captures that text
into ``decisions.reasoning`` for every failed cycle:

* ``"parse_failed: <up-to-1000-char excerpt>"`` — Opus replied, first parse failed
* ``"retry_failed: <excerpt>"``                 — the JSON-only retry *also* failed
* ``"claude returned no response (timeout/empty)"`` — CLI timeout / empty stdout
* ``"claude returned no response (nonzero_rc)"``     — the ``claude`` CLI
  subprocess ran but exited with a non-zero return code (a crash / OOM kill /
  ``cli_missing`` / wrapper ``exception``) — distinct from a timeout: the call
  is not slow, it *failed*, so a longer ``DECISION_TIMEOUT_S`` cannot help
* ``"skipped claude call — host saturated: …"``      — pre-flight guard declined
* ``"skipped claude call — host saturated mid-call: …"`` — box saturated *during*
  the call; the doomed Sonnet fallback was skipped (root-cause-attributed, not a
  model timeout — kept out of the empty-response bucket on purpose)
* ``"claude quota/usage limit exhausted (no decision)"`` — CLI usage-limit reject
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
    "SUBPROCESS_ERROR",      # claude CLI ran but crashed (non-zero rc / OOM)
    "HOST_SATURATED_SKIP",   # pre-flight guard declined the call (box overloaded)
    "HOST_STARVED_MIDCALL",  # box saturated *during* the call; fallback skipped
    "QUOTA_EXHAUSTED",       # claude CLI hit a usage/quota limit
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
    "SUBPROCESS_ERROR": ("The `claude` CLI subprocess ran but exited abnormally "
                         "(non-zero return code, missing binary, or a wrapper "
                         "exception) — the call *failed* rather than timed "
                         "out, so raising DECISION_TIMEOUT_S cannot help. A "
                         "non-zero rc under host pressure is usually an OOM "
                         "kill (each Opus subprocess ~1.5 GB against the "
                         "3-process cap) — same cure as the host-saturation "
                         "modes, fewer concurrent Opus jobs; a `cli_missing` "
                         "excerpt instead means `claude` is not on PATH."),
    "HOST_SATURATED_SKIP": ("Host was saturated at pre-flight — the Opus call "
                            "was deliberately skipped so it wouldn't feed the "
                            "storm. Reduce concurrent out-of-band Opus "
                            "subprocesses (hourly review / HYBRID agents / the "
                            "backtest committee) or stagger them; NOT a prompt "
                            "or model bug — raising the timeout won't help."),
    "HOST_STARVED_MIDCALL": ("Host passed pre-flight but saturated *during* "
                             "the Opus call (out-of-band Opus agents); the "
                             "doomed Sonnet fallback was skipped. Same fix as "
                             "HOST_SATURATED_SKIP — fewer concurrent Opus "
                             "subprocesses, not a longer DECISION_TIMEOUT_S."),
    "QUOTA_EXHAUSTED": ("The `claude` CLI hit a usage/quota limit — no decision "
                        "will succeed until the quota window resets. Not a host "
                        "or prompt issue; runner._cycle alarms this once."),
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

# Parenthesised cause codes strategy.py appends to a "claude returned no
# response (<cause>)" row (strategy.py: `reason_text = f"claude returned no
# response ({cause})"`). These three mean the `claude` CLI subprocess *ran and
# failed* — a non-zero exit, a missing binary, or a wrapper exception — as
# opposed to `timeout` / `empty_stdout` / `timeout/empty` which mean the call
# was slow / produced nothing. The split matters because the remediation
# differs: a crashed subprocess is never cured by a longer DECISION_TIMEOUT_S.
_SUBPROCESS_CAUSES = ("nonzero_rc", "cli_missing", "exception")

# ── Decision-loss clock ──────────────────────────────────────────────────────
# Minimum decisions in a UTC-hour bucket before it can be named a "worst hour"
# or drive the actionable hint. Mirrors decision_reliability.MIN_CURRENT's
# sample-honesty convention so a 1/1 = 100% bucket can never trigger a (wrong)
# "reschedule the cron" recommendation. Tests read this constant so a retune
# can't silently false-fail them.
HOUR_MIN_SAMPLE = 6
# A worst hour only earns an actionable clock_hint when its parse-fail rate
# exceeds the window-wide current-regime rate by at least this margin (pp) —
# a recurring concentration, not ambient noise.
CLOCK_HINT_MARGIN_PP = 15.0


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
    if "no response" in low:
        # A "claude returned no response (<cause>)" row. The CLI-fault causes
        # (nonzero_rc / cli_missing / exception) are a *crashed* subprocess —
        # split them out of OTHER into SUBPROCESS_ERROR and carry the specific
        # cause as the excerpt so the operator sees which one. timeout /
        # empty_stdout / timeout/empty stay TIMEOUT_EMPTY (model-output fault).
        for cause in _SUBPROCESS_CAUSES:
            if "(%s)" % cause in low:
                return {"mode": "SUBPROCESS_ERROR",
                        "tag": "subprocess_error", "excerpt": cause}
        if "timeout" in low or "empty" in low:
            return {"mode": "TIMEOUT_EMPTY", "tag": "no_response",
                    "excerpt": ""}
    # Operational (non-model) classes. strategy.py records these as
    # "skipped claude call — host saturated[ mid-call]: …" (host-saturation
    # guard) and "claude quota/usage limit exhausted …" (CLI usage cap). They
    # are NOT a prompt/model fault — keep them OUT of TIMEOUT_EMPTY so the
    # "raise DECISION_TIMEOUT_S" hint can't fire on an overloaded box, and so
    # the dominant live failure stops hiding in OTHER. The mid-call variant is
    # checked first: it also contains "host saturated", and the more specific
    # bucket must win (precedence pinned by tests).
    if raw.startswith("skipped claude call"):
        if "mid-call" in low:
            return {"mode": "HOST_STARVED_MIDCALL",
                    "tag": "host_starved_midcall", "excerpt": ""}
        return {"mode": "HOST_SATURATED_SKIP",
                "tag": "host_skip", "excerpt": ""}
    if "quota" in low and ("exhaust" in low or "usage limit" in low):
        return {"mode": "QUOTA_EXHAUSTED", "tag": "quota", "excerpt": ""}
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


def _regime_boundary(decisions: list[dict]) -> datetime | None:
    """Newest timestamp of a *legacy* (pre-diagnostics) parse failure, or None.

    ``strategy.py`` tags pre-diagnostics NO_DECISION rows with the legacy string
    ``"claude returned no parseable JSON"`` (``classify_failure`` → tag
    ``legacy``). Once the runner restarts onto diagnostic code those rows stop
    accruing — a fixed historical mass. The hour-of-day clock MUST be windowed
    past it: a clock-hour failure-rate computed over that dead mass would be
    skewed by *when those rows happened to land in UTC*, not by current host
    load, producing a confidently wrong "shift your cron" recommendation.

    This is the **same regime contract** ``decision_reliability`` partitions on:
    both derive ``boundary = max(ts where classify_failure(reasoning).tag ==
    'legacy')`` from *this module's* ``classify_failure`` / ``_parse_ts``, so
    the clock window and the reliability headline can never tell different
    stories on the same data. ``None`` ⇒ no dead legacy mass ⇒ the clock spans
    the full history.
    """
    legacy_ts: list[datetime] = []
    for d in decisions:
        if not _is_no_decision(d.get("action_taken")):
            continue
        if classify_failure(d.get("reasoning"))["tag"] == "legacy":
            ts = _parse_ts(d.get("timestamp"))
            if ts is not None:
                legacy_ts.append(ts)
    return max(legacy_ts) if legacy_ts else None


def _hour_of_day_clock(
    decisions: list[dict], boundary: datetime | None
) -> tuple[list[dict], int, int, list[dict], str]:
    """Current-regime parse-fail rate aggregated by UTC clock hour (0–23).

    The existing ``hourly`` series is a sparse 24h *calendar* timeline — it can
    show that today spiked at 14:00 but not that 14:00 spikes *every* day. This
    folds the whole current-regime history onto a 24-hour clock so a recurring
    host-load window becomes visible and actionable. That window is the system's
    dominant unsolved problem: TIMEOUT_EMPTY failures from the ``claude`` CLI
    contending for the 3-subprocess OOM cap whenever the hourly self-review and
    continuous-backtest loops fire — knowing *which UTC hours* lets the operator
    deconflict the cron instead of just watching the bleed.

    Returns ``(hour_of_day, window_n, window_failures, worst_hours,
    clock_hint)``. Window = rows with a parsed ts and
    ``(boundary is None or ts > boundary)`` — strictly ``>``, identical to
    ``decision_reliability``'s current-regime partition. Sparse (only hours with
    ≥1 decision), matching ``_bucket_hourly``'s convention. Pure; never raises.
    """
    tot: dict[int, int] = {}
    fail: dict[int, int] = {}
    win_n = win_fail = 0
    for d in decisions:
        ts = _parse_ts(d.get("timestamp"))
        if ts is None:
            continue
        if boundary is not None and ts <= boundary:
            continue
        h = ts.hour
        tot[h] = tot.get(h, 0) + 1
        win_n += 1
        if _is_no_decision(d.get("action_taken")):
            fail[h] = fail.get(h, 0) + 1
            win_fail += 1

    hod = [
        {
            "hour": h,
            "total": tot[h],
            "failures": fail.get(h, 0),
            "fail_pct": round(fail.get(h, 0) / tot[h] * 100, 1) if tot[h] else 0.0,
        }
        for h in sorted(tot)
    ]

    # Worst hours: only buckets with enough samples to not be 1/1 = 100% noise.
    eligible = [b for b in hod if b["total"] >= HOUR_MIN_SAMPLE]
    worst = sorted(
        eligible, key=lambda b: (-b["fail_pct"], -b["failures"], b["hour"])
    )[:3]
    overall = round(win_fail / win_n * 100, 1) if win_n else 0.0

    hint = ""
    if worst and worst[0]["fail_pct"] - overall >= CLOCK_HINT_MARGIN_PP:
        w = worst[0]
        others = [
            b
            for b in worst[1:]
            if b["fail_pct"] - overall >= CLOCK_HINT_MARGIN_PP
        ]
        extra = ""
        if others:
            tags = ", ".join("%02d:00" % b["hour"] for b in others)
            extra = " (also %s)" % tags
        hint = (
            "Parse-failures concentrate at %02d:00–%02d:00 UTC — %.0f%% fail "
            "vs %.0f%% window-wide (n=%d)%s. Shift host-heavy jobs (hourly "
            "self-review, continuous backtests) off this UTC hour to relieve "
            "the claude-subprocess contention."
            % (
                w["hour"],
                (w["hour"] + 1) % 24,
                w["fail_pct"],
                overall,
                w["total"],
                extra,
            )
        )
    return hod, win_n, win_fail, worst, hint


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
        "regime_boundary": None,
        "hour_of_day": [],
        "hour_of_day_window_n": 0,
        "hour_of_day_window_failures": 0,
        "hour_of_day_min_sample": HOUR_MIN_SAMPLE,
        "worst_hours": [],
        "clock_hint": "",
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

    # Decision-loss clock: fold the *current-regime* history onto a 24h UTC
    # clock so a recurring host-load window is visible/actionable (additive —
    # the legacy-inclusive `hourly`/`by_market`/verdict above are untouched).
    boundary = _regime_boundary(decisions)
    hod, hod_n, hod_fail, worst_hours, clock_hint = _hour_of_day_clock(
        decisions, boundary)
    out["regime_boundary"] = boundary.isoformat() if boundary else None
    out["hour_of_day"] = hod
    out["hour_of_day_window_n"] = hod_n
    out["hour_of_day_window_failures"] = hod_fail
    out["hour_of_day_min_sample"] = HOUR_MIN_SAMPLE
    out["worst_hours"] = worst_hours
    out["clock_hint"] = clock_hint

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
