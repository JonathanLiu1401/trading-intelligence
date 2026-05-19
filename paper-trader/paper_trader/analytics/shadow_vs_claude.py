"""Right-now snapshot: the deterministic shadow recommendation vs the most
recent Claude decision.

The live trader's primary failure mode in the 2026-05-18/19 window has been
HOST_SATURATED: Opus subprocesses spawned by review agents starve the live
trader's claude call, which records ``NO_DECISION (claude returned no
response)`` in the ``decisions`` table. ``/api/empty-claude-rate`` /
``/api/host-guard`` already surface *that the box is starved* — but they say
nothing about *what the bot would have done if a decision had come back*.

``/api/suggestions`` exposes the deterministic co-pilot rules engine
(``_classify_action`` in ``dashboard.py``) that emits BUY/ADD/TRIM/EXIT/WATCH
cards from the same news + quant signals the live trader sees. This builder
joins the two: it asks "right now, what is the top deterministic rec, and
what was the last Claude decision, and do they line up?" and emits a verdict
that flags the operationally-meaningful case where Claude is silent while
the rules engine has a strong actionable BUY.

Snapshot-only by design. The two inputs are produced from different points
in time (last Claude decision could be minutes-to-hours old; the suggestion
list reflects current market state), so this builder deliberately does **not**
compute "agreement %" over a historical window — that comparison would be
incoherent (signals at decision time ≠ signals now). The honest framing is
"right now, here's both, here's whether the operator should care."

Observational only — never gates Opus, never injected into the decision
prompt, no caps (invariants #2/#12 — the ``stress_scenarios`` / ``recovery``
/ ``event_calendar`` precedent). Pure: no I/O, never raises.
"""
from __future__ import annotations

from datetime import datetime, timezone

# Conviction threshold above which a shadow BUY/ADD is considered "strong
# enough that a NO_DECISION on it is a meaningful missed opportunity."
# Below this the rules engine doesn't itself trust the call (the action is
# still WATCH-territory), so silence from Claude isn't notable.
STRONG_CONVICTION = 0.70

# Actions the rules engine emits that are *directional* (would have caused a
# state change on the book). HOLD/WATCH don't change anything so a
# NO_DECISION on them isn't a missed opportunity.
DIRECTIONAL_ACTIONS = frozenset({"BUY", "ADD", "TRIM", "EXIT"})


def _classify_claude_action(action_taken: str | None) -> str:
    """Bucket the free-text ``decisions.action_taken`` into a canonical
    category. ``store.recent_decisions`` rows look like
    ``"BUY NVDA → FILLED"`` / ``"HOLD NVDA → HOLD"`` / ``"NO_DECISION"`` /
    ``"BLOCKED"``. We need a verb to compare against the shadow action."""
    if not action_taken:
        return "UNKNOWN"
    s = str(action_taken).strip().upper()
    if s.startswith("NO_DECISION"):
        return "NO_DECISION"
    if s.startswith("BLOCKED"):
        return "BLOCKED"
    if s.startswith("SKIPPED"):
        return "SKIPPED"
    # First whitespace token is the verb in the standard "VERB TICKER → STATUS" shape.
    verb = s.split()[0]
    if verb in ("BUY", "ADD", "SELL", "TRIM", "EXIT", "HOLD",
                "REBALANCE", "BUY_CALL", "BUY_PUT", "SELL_CALL", "SELL_PUT"):
        return verb
    return "UNKNOWN"


def _claude_ticker(action_taken: str | None) -> str | None:
    """Extract the ticker from a ``"VERB TICKER → STATUS"`` action_taken
    string. Mirrors ``dashboard._parse_action_ticker`` minimally — we only
    need it for the alignment check. ``NO_DECISION`` / ``BLOCKED`` / cash
    pseudo-tickers return ``None``."""
    if not action_taken:
        return None
    parts = str(action_taken).strip().split()
    if len(parts) < 2:
        return None
    verb = parts[0].upper()
    if verb in ("NO_DECISION", "BLOCKED", "SKIPPED"):
        return None
    raw = parts[1].strip(",→:.").upper()
    if not raw or raw in ("CASH", "NONE", "→"):
        return None
    return raw


def _minutes_since(ts_iso: str | None, now: datetime) -> float | None:
    """Minutes between ``now`` and an ISO-8601 timestamp. ``None`` on parse
    failure (rather than fabricating a 0.0 that would read as "just now")."""
    if not ts_iso:
        return None
    try:
        ts = datetime.fromisoformat(str(ts_iso))
    except (TypeError, ValueError):
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return round((now - ts).total_seconds() / 60.0, 1)


def _z(v: float | None, ndigits: int = 2) -> float | None:
    """Round; fold ``-0.0`` → ``0.0`` so the JSON never carries a signed
    zero (the ``earnings_shock._z`` / ``stress_scenarios._z`` precedent)."""
    if v is None:
        return None
    try:
        r = round(float(v), ndigits)
    except (TypeError, ValueError):
        return None
    return 0.0 if r == 0 else r


def _top_directional(suggestions: list[dict]) -> dict | None:
    """The highest-conviction directional shadow rec. None if the suggestion
    list has only HOLD/WATCH cards (no directional intent)."""
    if not suggestions:
        return None
    directional = [
        s for s in suggestions
        if isinstance(s, dict) and (s.get("action") or "").upper() in DIRECTIONAL_ACTIONS
    ]
    if not directional:
        return None
    directional.sort(key=lambda s: -float(s.get("conviction") or 0))
    return directional[0]


def build_shadow_vs_claude(
    suggestions: list[dict],
    last_decision: dict | None,
    now: datetime | None = None,
) -> dict:
    """Pure: no I/O, never raises.

    ``suggestions`` is the ``/api/suggestions`` ``suggestions`` list shape
    (already action-priority sorted in the endpoint, but we re-sort here on
    conviction within directional actions so the builder doesn't depend on
    upstream ordering). Each row has ``action``, ``conviction``, ``ticker``,
    ``reasons``, ``rsi``, ``macd``, etc.

    ``last_decision`` is the most recent row from ``store.recent_decisions``
    (or ``None`` if the table is empty). Carries ``timestamp``,
    ``action_taken``, ``reasoning``, ``confidence``.
    """
    now = now or datetime.now(timezone.utc)

    base = {
        "as_of": now.isoformat(timespec="seconds"),
        "shadow": None,
        "claude": None,
        "aligned": None,
        "verdict": None,
        "headline": None,
    }

    # ── Claude side ───────────────────────────────────────────────────────
    if isinstance(last_decision, dict):
        claude_action = _classify_claude_action(last_decision.get("action_taken"))
        claude_ticker = _claude_ticker(last_decision.get("action_taken"))
        claude_ts = last_decision.get("timestamp")
        claude_min_ago = _minutes_since(claude_ts, now)
        try:
            claude_conf = float(last_decision.get("confidence")) \
                if last_decision.get("confidence") is not None else None
        except (TypeError, ValueError):
            claude_conf = None
        base["claude"] = {
            "action": claude_action,
            "ticker": claude_ticker,
            "raw_action_taken": last_decision.get("action_taken"),
            "timestamp": claude_ts,
            "minutes_ago": claude_min_ago,
            "confidence": _z(claude_conf, 3),
        }
    else:
        claude_action = None
        claude_ticker = None

    # ── Shadow side ───────────────────────────────────────────────────────
    top = _top_directional(suggestions or [])
    if top is not None:
        shadow_action = (top.get("action") or "").upper()
        shadow_ticker = (top.get("ticker") or "").upper() or None
        try:
            shadow_conv = float(top.get("conviction") or 0)
        except (TypeError, ValueError):
            shadow_conv = 0.0
        base["shadow"] = {
            "action": shadow_action,
            "ticker": shadow_ticker,
            "conviction": _z(shadow_conv, 2),
            "strong": shadow_conv >= STRONG_CONVICTION,
            "reasons": list(top.get("reasons") or [])[:6],
            "rsi": _z(top.get("rsi"), 1),
            "macd": top.get("macd"),
            "news_urgent": bool(top.get("news_urgent")),
            "news_max_score": _z(top.get("news_max_score"), 1),
            "top_headline": top.get("top_headline"),
        }
    else:
        shadow_action = None
        shadow_ticker = None
        shadow_conv = 0.0

    # ── Alignment + verdict ───────────────────────────────────────────────
    # "Aligned" means the verb agrees on the same ticker. HOLD/WATCH on
    # either side maps to "not directional, no alignment claim made".
    aligned: bool | None
    if claude_action is None or shadow_action is None:
        aligned = None
    elif claude_action in ("NO_DECISION", "BLOCKED", "SKIPPED", "UNKNOWN"):
        aligned = None
    elif claude_ticker is None or shadow_ticker is None:
        aligned = None
    else:
        # Verb equivalence: BUY ≡ ADD (both are net-long-on-the-name);
        # SELL ≡ TRIM ≡ EXIT (all net-down). The shadow can't emit BUY for
        # a held ticker (the rules engine routes those to ADD/HOLD), so a
        # literal "BUY == ADD" check would never fire — collapsing here.
        def _equiv(v: str) -> str:
            if v in ("BUY", "ADD"):
                return "LONG"
            if v in ("SELL", "TRIM", "EXIT", "SELL_CALL", "SELL_PUT"):
                return "DOWN"
            return v
        aligned = (
            _equiv(claude_action) == _equiv(shadow_action)
            and claude_ticker.upper() == shadow_ticker.upper()
        )

    base["aligned"] = aligned

    # Verdict ladder:
    #   MISSED_OPPORTUNITY → most recent Claude was NO_DECISION AND the
    #                       shadow currently has a strong directional rec.
    #                       Operationally: "the bot would have acted, but
    #                       didn't because Opus came back empty."
    #   DROUGHT_OK         → Claude was NO_DECISION but shadow is quiet too
    #                       (no strong directional rec). Silence is fine.
    #   ALIGNED            → Claude and shadow agree on the same directional
    #                       call on the same name. (Cross-engine confirm.)
    #   DIVERGENT          → both produced a directional call but they
    #                       disagree on verb or ticker.
    #   CLAUDE_HOLDS       → Claude said HOLD (no state change) while shadow
    #                       has a directional rec; the operator may want to
    #                       second-guess the hold.
    #   NO_CLAUDE_DATA     → decisions table empty / unparseable.
    #   NO_SHADOW_DATA     → suggestions empty (signals unavailable) and we
    #                       can't say anything useful.
    if claude_action is None:
        verdict = "NO_CLAUDE_DATA"
    elif shadow_action is None and claude_action == "NO_DECISION":
        verdict = "DROUGHT_OK"
    elif shadow_action is None:
        verdict = "NO_SHADOW_DATA"
    elif claude_action == "NO_DECISION":
        verdict = "MISSED_OPPORTUNITY" if shadow_conv >= STRONG_CONVICTION else "DROUGHT_OK"
    elif claude_action == "HOLD":
        verdict = "CLAUDE_HOLDS"
    elif aligned is True:
        verdict = "ALIGNED"
    elif aligned is False:
        verdict = "DIVERGENT"
    else:
        verdict = "UNKNOWN"
    base["verdict"] = verdict

    # ── Headline ──────────────────────────────────────────────────────────
    sm = base["shadow"]
    cm = base["claude"]
    if verdict == "MISSED_OPPORTUNITY":
        base["headline"] = (
            f"MISSED_OPPORTUNITY — last Claude was NO_DECISION "
            f"({cm['minutes_ago']:.0f}m ago); shadow rules engine says "
            f"{sm['action']} {sm['ticker']} (conviction {sm['conviction']:.2f})."
        ) if cm and cm.get("minutes_ago") is not None else (
            f"MISSED_OPPORTUNITY — last Claude was NO_DECISION; "
            f"shadow says {sm['action']} {sm['ticker']} "
            f"(conviction {sm['conviction']:.2f})."
        )
    elif verdict == "ALIGNED":
        base["headline"] = (
            f"ALIGNED — Claude {cm['action']} {cm['ticker']} matches shadow "
            f"{sm['action']} {sm['ticker']} (conviction {sm['conviction']:.2f})."
        )
    elif verdict == "DIVERGENT":
        base["headline"] = (
            f"DIVERGENT — Claude {cm['action']} {cm['ticker']} vs shadow "
            f"{sm['action']} {sm['ticker']} (conviction {sm['conviction']:.2f})."
        )
    elif verdict == "CLAUDE_HOLDS":
        base["headline"] = (
            f"CLAUDE_HOLDS — Claude HOLD while shadow flags "
            f"{sm['action']} {sm['ticker']} (conviction {sm['conviction']:.2f})."
        )
    elif verdict == "DROUGHT_OK":
        if cm and cm.get("action") == "NO_DECISION":
            base["headline"] = (
                "DROUGHT_OK — Claude NO_DECISION but shadow rules engine "
                "has no strong directional rec either; nothing to act on."
            )
        else:
            base["headline"] = "DROUGHT_OK — quiet on both sides."
    elif verdict == "NO_CLAUDE_DATA":
        base["headline"] = "NO_CLAUDE_DATA — no recent Claude decision to compare against."
    elif verdict == "NO_SHADOW_DATA":
        base["headline"] = "NO_SHADOW_DATA — shadow suggestions list is empty (signals unavailable)."
    else:
        base["headline"] = f"{verdict} — see fields."

    return base
