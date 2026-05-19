"""Per-held-position composite action brief.

Existing analytics surface fragments the operator's "what's the situation on
each name I hold, right now" view across at least seven endpoints:

* ``/api/portfolio`` — exposure and unrealized P&L
* ``/api/game-plan`` — per-position HOLD/SELL/TRIM action with conviction
* ``/api/news-velocity`` — per-held-ticker article surge/fade
* ``/api/earnings-risk`` — per-held-ticker earnings-event proximity
* ``/api/position-attention`` — per-held-ticker last-real-look freshness
* ``/api/empty-claude-rate`` — does the bot actually have a working Opus path
* ``/api/host-guard`` — is the box starving claude calls

Each is excellent on its own surface; none answers the per-position composite
question a trader poses to themselves before bed:

    NVDA — earnings in 11.5h, $445 exposure (44.5% of book), bot is 81%
    empty over 24h, news SURGING. Recommended: TRIM_BEFORE_EVENT or restart
    the runner. Urgency: 0.95.

That composition lives nowhere — and the dangerous case (held-imminent print
× wedged Opus × no decision in 4h) is exactly when the operator most needs a
single readable view. This builder fills the gap.

Pure, no network, no DB. The route wrapper owns the I/O (the documented
thesis_drift split) — this composes already-fetched scalars / dicts from the
surfaces above. Returns a JSON-ready dict with per-position briefs ranked by
``urgency_score`` (most-urgent first) plus an overall ``headline`` /
``overall_urgency`` for the home-page operator card.

Advisory only — never gates Opus, never injected into a decision prompt,
adds no caps (AGENTS.md invariants #2 / #12).
"""
from __future__ import annotations

from datetime import datetime, timezone

# Action verbs. Stable for tests / chart-mapping.
ACTION_OK = "OK"
ACTION_MONITOR = "MONITOR"
ACTION_HOLD_THROUGH_EVENT = "HOLD_THROUGH_EVENT"
ACTION_TRIM_BEFORE_EVENT = "TRIM_BEFORE_EVENT"
ACTION_RESTART_RUNNER = "RESTART_RUNNER"

# Decision-history status tags. Stable.
DECISION_DECIDED = "DECIDED"
DECISION_EMPTY = "EMPTY"
DECISION_HOST_SKIP = "HOST_SKIP"
DECISION_PARSE_FAIL = "PARSE_FAIL"
DECISION_NEVER = "NEVER"

# Thresholds — tested-pinned. Mirror restart_recommendation where overlapping
# so two surfaces agree on a wedge instead of disagreeing by 1%.
EVENT_IMMINENT_HOURS = 24.0          # any held event within 24h is imminent
EVENT_NEAR_HOURS = 72.0              # the wider monitoring horizon
WEDGED_EMPTY_RATE = 50.0             # %  empty-rate that flips TRIM/RESTART
NEGLECTED_AGE_MIN = 240.0            # minutes since last real decision = STALE


def _coerce_float(x, default: float | None = None) -> float | None:
    try:
        if x is None:
            return default
        return float(x)
    except (TypeError, ValueError):
        return default


def _parse_ts(ts) -> datetime | None:
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        dt = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _classify_decision(action: str | None, reasoning: str | None) -> str:
    """Stable decision-status taxonomy.

    Mirrors the buckets ``/api/decision-forensics`` and
    ``/api/no-decision-reasons`` already report on, so the surfaces agree on
    what a row "means". A real BUY/SELL/HOLD/BUY_CALL/... decision reads as
    DECIDED; the documented NO_DECISION sub-modes read as EMPTY (no response),
    HOST_SKIP (skipped before claude was invoked), or PARSE_FAIL. Anything
    else reads as DECIDED (the trader actually committed to an action).
    """
    a = (action or "").strip().upper()
    r = (reasoning or "").lower()
    if not a:
        return DECISION_NEVER
    if a.startswith("BLOCKED"):
        return DECISION_DECIDED  # risk gate fired; bot's brain is alive
    if "NO_DECISION" not in a:
        return DECISION_DECIDED
    # NO_DECISION row — bucket the why.
    if r.startswith("claude returned no response") or "timeout" in r or "empty" in r:
        return DECISION_EMPTY
    if "host saturat" in r or "skipped" in r or "host_skip" in r:
        return DECISION_HOST_SKIP
    if "parse_failed" in r or "retry_failed" in r or "parse fail" in r:
        return DECISION_PARSE_FAIL
    return DECISION_EMPTY  # default the unknown NO_DECISION to EMPTY


def _last_decision_for_ticker(decisions: list[dict],
                              ticker: str,
                              now: datetime) -> dict:
    """Most recent decision row that names this ticker — DECIDED preferred,
    else most recent NO_DECISION, else NEVER.

    A ticker held but never decided on returns ``status=NEVER`` so the brief
    is honest about freshness. The endpoint owns the row fetch; this function
    only scans the supplied list newest-first.
    """
    most_recent = None
    most_recent_decided = None
    tk = ticker.upper()
    for d in (decisions or []):
        action = (d.get("action_taken") or "")
        # `_parse_action_ticker` lives in dashboard.py; we re-implement a
        # cheap substring match here — the upper-cased action verbs include
        # the ticker (e.g. "BUY NVDA → FILLED", "HOLD NVDA"). Spurious
        # substring hits (e.g. AMD inside AMDOCS) cannot happen because the
        # action text uses watchlist tickers only.
        if not action:
            continue
        if tk not in action.upper():
            continue
        status = _classify_decision(action, d.get("reasoning"))
        if most_recent is None:
            most_recent = (d, status)
        if status == DECISION_DECIDED and most_recent_decided is None:
            most_recent_decided = (d, status)
        if most_recent_decided is not None and most_recent is not None:
            break

    chosen = most_recent_decided or most_recent
    if chosen is None:
        return {
            "status": DECISION_NEVER,
            "action": None,
            "age_min": None,
            "timestamp": None,
        }
    row, status = chosen
    ts = _parse_ts(row.get("timestamp"))
    age_min = ((now - ts).total_seconds() / 60.0) if ts is not None else None
    return {
        "status": status,
        "action": (row.get("action_taken") or "").strip() or None,
        "age_min": round(age_min, 1) if age_min is not None else None,
        "timestamp": ts.isoformat(timespec="seconds") if ts is not None else None,
    }


def _event_for_ticker(events: list[dict] | None,
                      ticker: str) -> tuple[float | None, str | None, str | None]:
    """Return (hours_to_event, event_verdict, earnings_date) for the soonest
    matching event.

    The events list comes from ``build_event_readiness`` (whose row schema
    emits ``verdict`` ∈ {BLIND, DEGRADED, READY, IMMINENT_OVERDUE} — the
    readiness ladder, not the tier ladder). It can also come from
    ``/api/earnings-risk`` (which emits ``tier`` ∈ {HELD_IMMINENT,
    HELD_SOON, WATCH}). We prefer the readiness ``verdict`` when present —
    that's strictly more actionable for the operator than the tier (a
    HELD_IMMINENT row is no use if the bot is BLIND on it).
    """
    if not events:
        return (None, None, None)
    tk = ticker.upper()
    best: tuple[float, str | None, str | None] | None = None
    for e in events:
        if not isinstance(e, dict):
            continue
        et = str(e.get("ticker") or "").upper()
        if et != tk:
            continue
        days_away = _coerce_float(e.get("days_away"))
        hours_to_event = _coerce_float(e.get("hours_until_event"))
        if hours_to_event is None and days_away is not None:
            hours_to_event = days_away * 24.0
        if hours_to_event is None:
            continue
        # readiness verdict preferred over tier — see docstring.
        verdict = e.get("verdict") or e.get("tier")
        ed = e.get("earnings_date") or e.get("event_date")
        if best is None or hours_to_event < best[0]:
            best = (hours_to_event, verdict, ed)
    if best is None:
        return (None, None, None)
    return (best[0], best[1], best[2])


def _news_for_ticker(news_velocity: dict | None,
                     ticker: str) -> dict:
    """Pluck the per-ticker row from a /api/news-velocity response. Missing /
    malformed input degrades to INSUFFICIENT — never raises.
    """
    blank = {
        "state": "INSUFFICIENT",
        "window_count": 0,
        "z_score": None,
        "max_ai_score_window": None,
        "top_window_title": None,
    }
    if not isinstance(news_velocity, dict):
        return blank
    rows = news_velocity.get("per_ticker")
    if not isinstance(rows, list):
        return blank
    tk = ticker.upper()
    for r in rows:
        if not isinstance(r, dict):
            continue
        if str(r.get("ticker") or "").upper() != tk:
            continue
        return {
            "state": r.get("state") or "INSUFFICIENT",
            "window_count": int(r.get("window_count") or 0),
            "z_score": r.get("z_score"),
            "max_ai_score_window": r.get("max_ai_score_window"),
            "top_window_title": r.get("top_window_title"),
        }
    return blank


def _decide_action(hours_to_event: float | None,
                   empty_rate_pct: float | None,
                   host_saturated: bool | None,
                   news_state: str,
                   decision_status: str,
                   age_min: float | None) -> tuple[str, float, list[str]]:
    """Per-position recommended action + urgency score + reason bundle.

    Precedence (highest urgency first):

    1. ``TRIM_BEFORE_EVENT`` — held-imminent (<24h) earnings AND wedged bot.
    2. ``RESTART_RUNNER`` — wedged bot (high empty rate or stale decision)
       without an imminent event but with positive news flow.
    3. ``HOLD_THROUGH_EVENT`` — held-imminent earnings with a working bot.
    4. ``MONITOR`` — news SURGING, OR decision is stale-but-not-wedged.
    5. ``OK`` — none of the above.
    """
    reasons: list[str] = []
    er = empty_rate_pct
    wedged_by_rate = (er is not None and er >= WEDGED_EMPTY_RATE)
    wedged_by_age = (age_min is not None and age_min >= NEGLECTED_AGE_MIN
                     and decision_status != DECISION_DECIDED)
    bot_wedged = wedged_by_rate or wedged_by_age

    if hours_to_event is not None and hours_to_event <= EVENT_IMMINENT_HOURS:
        if bot_wedged:
            reasons.append(
                f"earnings in {hours_to_event:.1f}h AND bot is wedged "
                f"({'empty rate ' + format(er, '.0f') + '%' if wedged_by_rate else 'no real decision in ' + str(int(age_min)) + 'm'})")
            if news_state == "SURGING":
                reasons.append("news SURGING — catalyst already pricing in")
            return (ACTION_TRIM_BEFORE_EVENT, 0.95, reasons)
        reasons.append(
            f"earnings in {hours_to_event:.1f}h — hold through with a "
            f"working bot")
        if news_state == "SURGING":
            reasons.append("news SURGING into the print")
        return (ACTION_HOLD_THROUGH_EVENT, 0.6, reasons)

    if bot_wedged:
        reasons.append(
            "bot is wedged" + (
                f" (empty rate {er:.0f}%)" if wedged_by_rate
                else f" (no real decision in {int(age_min) if age_min else '?'}m)")
        )
        if news_state == "SURGING":
            reasons.append("news SURGING — the bot is missing the catalyst")
            return (ACTION_RESTART_RUNNER, 0.85, reasons)
        if host_saturated:
            reasons.append("host saturated — kill out-of-band Opus jobs")
        return (ACTION_RESTART_RUNNER, 0.7, reasons)

    if news_state == "SURGING":
        reasons.append("news SURGING — review for thesis change")
        return (ACTION_MONITOR, 0.5, reasons)

    if (hours_to_event is not None
            and hours_to_event <= EVENT_NEAR_HOURS):
        reasons.append(f"earnings in {hours_to_event:.1f}h — monitor")
        return (ACTION_MONITOR, 0.4, reasons)

    if (decision_status != DECISION_DECIDED
            and age_min is not None
            and age_min >= 60.0):
        reasons.append(
            f"last real decision {int(age_min)}m ago — review freshness")
        return (ACTION_MONITOR, 0.3, reasons)

    return (ACTION_OK, 0.05, ["no flagged condition"])


def build_position_action_brief(
    positions: list[dict],
    decisions: list[dict],
    news_velocity: dict | None,
    held_events: list[dict] | None,
    empty_rate_pct: float | None,
    host_saturated: bool | None,
    starting_equity_usd: float | None = None,
    now: datetime | None = None,
) -> dict:
    """Compose the per-held-position composite.

    Parameters
    ----------
    positions : list[dict]
        Open positions from ``store.open_positions()``. Each row needs
        ``ticker``, ``quantity``, ``avg_cost``, ``current_price``, ``side``.
    decisions : list[dict]
        Newest-first recent decisions from ``store.recent_decisions``. Each
        needs ``timestamp``, ``action_taken``, ``reasoning``.
    news_velocity : dict | None
        Output of ``build_news_velocity`` (or the cached endpoint payload).
    held_events : list[dict] | None
        Output of ``/api/earnings-risk`` (or ``/api/event-calendar``) — each
        dict needs ``ticker``, ``days_away`` (or ``hours_until_event``), and
        optionally ``tier`` and ``earnings_date``.
    empty_rate_pct : float | None
        24h ``/api/empty-claude-rate``.
    host_saturated : bool | None
        ``/api/host-guard`` saturation flag.
    starting_equity_usd : float | None
        Total book value for ``pct_portfolio``. Defaults to sum of exposures
        when omitted; never zero-divides.
    now : datetime, optional
        Test injection.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    out: dict = {
        "as_of": now.isoformat(timespec="seconds"),
        "n_positions": 0,
        "n_imminent_events": 0,
        "overall_action": ACTION_OK,
        "overall_urgency": 0.0,
        "headline": "OK — no held positions.",
        "inputs": {
            "empty_rate_pct": (round(float(empty_rate_pct), 1)
                               if empty_rate_pct is not None else None),
            "host_saturated": (bool(host_saturated)
                               if host_saturated is not None else None),
        },
        "briefs": [],
    }

    if not positions:
        return out

    # Map ticker → aggregate exposure (option lots fold into the underlying).
    by_ticker: dict[str, dict] = {}
    for p in positions:
        if not isinstance(p, dict):
            continue
        tk = (p.get("ticker") or "").upper().strip()
        if not tk or tk in {"CASH", "NONE", "NO_DECISION", "BLOCKED"}:
            continue
        # Live store rows use ``qty``; tests/fixtures use ``quantity`` —
        # accept either so the builder is callable from both.
        qty = _coerce_float(p.get("qty"), default=None)
        if qty is None:
            qty = _coerce_float(p.get("quantity"), default=0.0) or 0.0
        avg = _coerce_float(p.get("avg_cost"), default=0.0) or 0.0
        cur = _coerce_float(p.get("current_price"), default=avg) or avg
        # Option lots have a ×100 contract multiplier.
        mult = 100 if (p.get("type") in ("call", "put")) else 1
        mkt = abs(qty) * cur * mult
        cost = abs(qty) * avg * mult
        side = (p.get("side") or "").upper() or "LONG"
        agg = by_ticker.setdefault(tk, {
            "ticker": tk,
            "exposure_usd": 0.0,
            "cost_basis_usd": 0.0,
            "unrealized_pl_usd": 0.0,
            "n_lots": 0,
            "side": side,
        })
        agg["exposure_usd"] += mkt
        agg["cost_basis_usd"] += cost
        agg["unrealized_pl_usd"] += (mkt - cost) if side == "LONG" else (cost - mkt)
        agg["n_lots"] += 1

    if not by_ticker:
        return out

    # Portfolio denominator. Caller may pass actual total_value (cash + open)
    # to make pct_portfolio honest; without it, fall back to open-only sum.
    denom = _coerce_float(starting_equity_usd, default=None)
    if denom is None or denom <= 0:
        denom = sum(v["exposure_usd"] for v in by_ticker.values()) or 1.0

    briefs: list[dict] = []
    n_imminent = 0
    for tk, agg in by_ticker.items():
        dec_info = _last_decision_for_ticker(decisions, tk, now)
        hours_to_event, event_verdict, ed = _event_for_ticker(held_events, tk)
        news_info = _news_for_ticker(news_velocity, tk)
        if hours_to_event is not None and hours_to_event <= EVENT_IMMINENT_HOURS:
            n_imminent += 1
        action, urgency, reasons = _decide_action(
            hours_to_event=hours_to_event,
            empty_rate_pct=empty_rate_pct,
            host_saturated=host_saturated,
            news_state=news_info["state"],
            decision_status=dec_info["status"],
            age_min=dec_info["age_min"],
        )

        # Augment reasons with news head — useful in a Discord post even if
        # the action ladder didn't pivot on news state.
        head = news_info.get("top_window_title")
        if head and news_info["state"] in ("SURGING", "STABLE") and \
                not any("SURGING" in r or "FADING" in r for r in reasons):
            reasons.append(f"news head: {head[:80]}")

        briefs.append({
            "ticker": tk,
            "exposure_usd": round(agg["exposure_usd"], 2),
            "cost_basis_usd": round(agg["cost_basis_usd"], 2),
            "unrealized_pl_usd": round(agg["unrealized_pl_usd"], 2),
            "pct_portfolio": round(100.0 * agg["exposure_usd"] / denom, 2),
            "n_lots": agg["n_lots"],
            "side": agg["side"],
            "hours_to_event": (round(hours_to_event, 2)
                               if hours_to_event is not None else None),
            "event_verdict": event_verdict,
            "earnings_date": ed,
            "news_state": news_info["state"],
            "news_window_count": news_info["window_count"],
            "news_z_score": news_info["z_score"],
            "news_top_title": news_info["top_window_title"],
            "news_max_ai_score": news_info["max_ai_score_window"],
            "last_decision_status": dec_info["status"],
            "last_decision_action": dec_info["action"],
            "last_decision_age_min": dec_info["age_min"],
            "last_decision_timestamp": dec_info["timestamp"],
            "recommended_action": action,
            "urgency_score": round(urgency, 2),
            "reasons": reasons,
        })

    # Rank by urgency desc, then exposure desc, then alphabetical ticker so
    # ties are deterministic (same shape build_event_threads / event-readiness
    # uses elsewhere on the analytics surface).
    briefs.sort(key=lambda b: (
        -b["urgency_score"], -b["exposure_usd"], b["ticker"],
    ))

    overall_urgency = max((b["urgency_score"] for b in briefs), default=0.0)
    overall_action = ACTION_OK
    for b in briefs:
        if b["recommended_action"] != ACTION_OK:
            overall_action = b["recommended_action"]
            break

    out["n_positions"] = len(briefs)
    out["n_imminent_events"] = n_imminent
    out["overall_action"] = overall_action
    out["overall_urgency"] = round(overall_urgency, 2)
    out["briefs"] = briefs

    # Headline composition: most-urgent name leads, an event count tail.
    top = briefs[0]
    if overall_urgency >= 0.9:
        out["headline"] = (
            f"URGENT — {top['ticker']} {top['recommended_action']} "
            f"(urgency {top['urgency_score']:.2f}): "
            f"{top['reasons'][0] if top['reasons'] else 'see briefs'}")
    elif overall_urgency >= 0.6:
        out["headline"] = (
            f"ACTION — {top['ticker']} {top['recommended_action']}: "
            f"{top['reasons'][0] if top['reasons'] else 'see briefs'}")
    elif overall_urgency >= 0.3:
        out["headline"] = (
            f"MONITOR — {top['ticker']}: "
            f"{top['reasons'][0] if top['reasons'] else 'see briefs'}")
    else:
        out["headline"] = (
            f"OK — {len(briefs)} held position(s), no flagged conditions.")

    return out
