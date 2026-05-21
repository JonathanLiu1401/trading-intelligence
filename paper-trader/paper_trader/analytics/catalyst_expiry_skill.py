"""Catalyst-expiry skill — for each currently OPEN position, classify
the entry-rationale catalyst class and flag positions whose dated
catalyst has materially aged past its expiry window.

Existing analytics either classify the *closed* trip by catalyst type
or measure unrelated axes for the open book:

* ``catalyst_class_autopsy`` — per-class realised P&L of *closed*
  round trips. Backward-looking. Cannot answer "you opened NVDA on
  the earnings catalyst 4 days ago — is that catalyst still live?".
* ``thesis_drift`` — re-tests each open position against its own
  rationale on technicals (RSI/MACD/momentum/PL) but never reads the
  catalyst's TIME MARKER. A held position whose ``earnings tomorrow``
  catalyst is now 5 days past prints INTACT if RSI / MACD are still
  bullish.
* ``position_thesis`` / ``position_attention`` — surface news
  intensity NOW, not the staleness of the original catalyst.
* ``held_theme_decay`` — clusters held tickers by theme keyword
  drift across all live news, not by the entry-rationale's TIME
  MARKER.

A dated catalyst (``earnings tomorrow``, ``FOMC this week``,
``Q1 results in 2 days``, ``CPI in 3 days``) has a structural
TTL: once the event has passed and the price has digested the
result, holding the same position requires a *new* thesis. Without
that, the position is a zombie: the original reason to hold has
expired but the position lingers.

Per-position classification:

1. ``CATALYST_CLASS`` — the strongest catalyst keyword family present
   in the verbatim entry reason (lower-cased, whole-word match):

   * ``EARNINGS`` — earnings, results, EPS, revenue, Q1/Q2/Q3/Q4,
     beat, miss, guidance, guides.
   * ``MACRO`` — FOMC, fed rate, CPI, PPI, payrolls, jobs report,
     inflation, GDP, ECB, BOJ.
   * ``PRODUCT`` — launch, unveil, announce(d), summit, event, GTC,
     keynote, debut.
   * ``REGULATORY`` — FDA, approval, rejection, ruling, court,
     antitrust, sanction, tariff, ban.
   * ``CORPORATE`` — buyback, repurchase, dividend, merger, acquire,
     acquisition, IPO, spinoff, split.
   * ``TECHNICAL`` — RSI, MACD, breakout, golden cross, death cross,
     momentum, mom_20d, mom_5d, mean reversion. (Structural thesis,
     no inherent TIME MARKER.)
   * ``UNCATEGORIZED`` — none of the above keyword families matched.

2. ``HAS_TIME_MARKER`` — does the reason cite an imminent time anchor?
   Whole-word match against ``tomorrow``, ``today``, ``next week``,
   ``this week``, ``in N day(s)``, ``in N hour(s)``, ``in 0.Nd``,
   ``post-earnings``, ``ahead of``, ``before``, ``Q[1-4] 20\\d\\d``,
   or an ISO date / month-day pattern.

3. ``DAYS_SINCE_OPEN`` — wall-clock days between the position's
   opened_at and the supplied ``now``.

Per-position verdict ladder:

* ``ZOMBIE`` — CATALYST_CLASS in EARNINGS/MACRO/PRODUCT/REGULATORY
  AND HAS_TIME_MARKER AND DAYS_SINCE_OPEN ≥
  ``ZOMBIE_DAYS_FLOOR`` (default 3). The dated catalyst is past;
  the operator is still holding.
* ``FRESH_CATALYST`` — CATALYST_CLASS in same dated set AND
  DAYS_SINCE_OPEN < ``FRESH_DAYS_CEIL`` (default 2). Catalyst
  window still open.
* ``STRUCTURAL`` — CATALYST_CLASS in TECHNICAL/CORPORATE
  (CORPORATE: buyback/dividend are persistent flow catalysts
  with no expiry) OR the reason has no time marker. The position's
  thesis is structural, not dated.
* ``UNCATEGORIZED`` — no catalyst keyword family matched. Often
  fine for purely tactical re-entries; surfaced as a separate
  bucket so the operator can spot poorly-anchored entries.
* ``NO_REASON`` — entry trade has no parseable reason text.

Aggregate verdict over all open positions:

* ``ZOMBIE_HOLDINGS`` — ≥1 position is ZOMBIE.
* ``ALL_FRESH`` — no ZOMBIE, ≥1 FRESH_CATALYST, no STRUCTURAL.
* ``STRUCTURAL_BOOK`` — every position is STRUCTURAL /
  UNCATEGORIZED / NO_REASON; no dated catalysts.
* ``MIXED_BOOK`` — neither extreme. Some fresh, some structural,
  but no zombies.
* ``NO_DATA`` — no open positions.

Pure builder. Open positions + trades in, dict out, never raises.
Observational only — never gates Opus, no caps (AGENTS.md #2/#12 —
the ``thesis_drift`` / ``catalyst_class_autopsy`` precedent).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Sequence

# Per-position verdict labels.
ZOMBIE = "ZOMBIE"
FRESH_CATALYST = "FRESH_CATALYST"
STRUCTURAL = "STRUCTURAL"
UNCATEGORIZED = "UNCATEGORIZED"
NO_REASON = "NO_REASON"

# Aggregate verdict labels.
ZOMBIE_HOLDINGS = "ZOMBIE_HOLDINGS"
ALL_FRESH = "ALL_FRESH"
STRUCTURAL_BOOK = "STRUCTURAL_BOOK"
MIXED_BOOK = "MIXED_BOOK"
NO_DATA = "NO_DATA"

# Default per-position thresholds. Tuneable per-call.
DEFAULT_ZOMBIE_DAYS_FLOOR = 3.0
DEFAULT_FRESH_DAYS_CEIL = 2.0

# Reason-excerpt cap on the wire.
_REASON_EXCERPT_MAX = 140

# Dated-catalyst classes — these have a finite TTL after the event.
_DATED_CLASSES = frozenset({"EARNINGS", "MACRO", "PRODUCT", "REGULATORY"})

# Structural catalyst classes — persistent thesis with no inherent
# expiry. CORPORATE (buyback / dividend / M&A talks) is bucketed
# here because the flow thesis persists for weeks after the
# announcement; it's not a one-day event.
_STRUCTURAL_CLASSES = frozenset({"TECHNICAL", "CORPORATE"})

# Catalyst keyword families — case-insensitive whole-word match.
# Order: EARNINGS first because Q1/Q2 etc are common false positives
# for MACRO (CPI Q1, fed Q1, …) — earliest match wins per a single
# pass.
_CATALYST_KEYWORDS: tuple[tuple[str, frozenset[str]], ...] = (
    ("EARNINGS", frozenset({
        "earnings", "earning", "eps", "revenue",
        "beat", "beats", "miss", "missed", "misses",
        "guide", "guides", "guidance", "guided",
        "q1", "q2", "q3", "q4",
        "outlook", "preprint", "preannounce",
    })),
    ("MACRO", frozenset({
        "fomc", "fed", "rate", "rates", "hike", "hikes", "cut", "cuts",
        "cpi", "ppi", "pce", "payrolls", "payroll", "nfp",
        "inflation", "gdp", "jobless", "claims",
        "ecb", "boj", "pboc",
    })),
    ("PRODUCT", frozenset({
        "launch", "launches", "launched",
        "unveil", "unveils", "unveiled",
        "announce", "announces", "announced",
        "summit", "keynote", "gtc",
        "debut", "debuts", "rollout",
    })),
    ("REGULATORY", frozenset({
        "fda", "approval", "approved", "approves",
        "rejection", "rejected", "rejects",
        "ruling", "rules", "ruled",
        "court", "antitrust", "doj",
        "sanction", "sanctioned", "sanctions",
        "tariff", "tariffs",
        "ban", "banned", "bans",
    })),
    ("CORPORATE", frozenset({
        "buyback", "buybacks", "repurchase", "repurchases",
        "dividend", "dividends",
        "merger", "mergers",
        "acquire", "acquires", "acquired",
        "acquisition", "acquisitions",
        "ipo", "spinoff", "split",
    })),
    ("TECHNICAL", frozenset({
        "rsi", "macd",
        "breakout", "breakdown",
        "golden", "death",
        "momentum", "mom_5d", "mom_20d", "mom",
        "reversion", "support", "resistance",
        "trendline", "channel",
    })),
)

# Time-marker patterns — case-insensitive. Each pattern is sufficient
# to set HAS_TIME_MARKER=True.
_TIME_MARKER_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"\btomorrow\b", re.IGNORECASE),
    re.compile(r"\btoday\b", re.IGNORECASE),
    re.compile(r"\b(this|next)\s+week\b", re.IGNORECASE),
    re.compile(r"\bin\s+\d+(?:\.\d+)?\s*d(?:ays?)?\b", re.IGNORECASE),
    re.compile(r"\bin\s+\d+(?:\.\d+)?\s*hour", re.IGNORECASE),
    re.compile(r"\bin\s+0?\.\d+d\b", re.IGNORECASE),
    re.compile(r"\bpost[-\s]?earnings\b", re.IGNORECASE),
    re.compile(r"\bahead\s+of\b", re.IGNORECASE),
    re.compile(r"\bbefore\s+the\s+open\b", re.IGNORECASE),
    re.compile(r"\bafter[-\s]hours?\b", re.IGNORECASE),
    re.compile(r"\bpre[-\s]market\b", re.IGNORECASE),
    re.compile(r"\bovernight\b", re.IGNORECASE),
    re.compile(r"\bQ[1-4]\s+20\d{2}\b", re.IGNORECASE),
    re.compile(r"\bAH\b"),               # after-hours abbreviation (case-sensitive)
    re.compile(r"\b20\d{2}-\d{2}-\d{2}\b"),  # ISO date
)

# Whole-word tokenizer (alpha and alphanumeric tokens, plus
# qN-style earnings-quarter shorthand).
_WORD_RE = re.compile(r"[A-Za-z]+(?:\d)?")


def _num(x: Any) -> float | None:
    if x is None:
        return None
    if isinstance(x, bool):
        return None
    if isinstance(x, (int, float)):
        return float(x)
    if isinstance(x, str):
        try:
            return float(x.strip())
        except (TypeError, ValueError):
            return None
    return None


def _parse_iso(ts: Any) -> datetime | None:
    if not isinstance(ts, str) or not ts:
        return None
    try:
        s = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except (TypeError, ValueError):
        return None


def classify_catalyst(text: str) -> str:
    """Pure text → catalyst class. Earliest-family-wins keyword match.

    Returns one of ``EARNINGS`` / ``MACRO`` / ``PRODUCT`` /
    ``REGULATORY`` / ``CORPORATE`` / ``TECHNICAL`` / ``UNCATEGORIZED``.
    """
    if not isinstance(text, str) or not text:
        return "UNCATEGORIZED"
    text_lower = text.lower()
    tokens = {m.group(0) for m in _WORD_RE.finditer(text_lower)}
    if not tokens:
        return "UNCATEGORIZED"
    for klass, kws in _CATALYST_KEYWORDS:
        if tokens & kws:
            return klass
    return "UNCATEGORIZED"


def has_time_marker(text: str) -> bool:
    """Pure text → True if any time-marker pattern matches."""
    if not isinstance(text, str) or not text:
        return False
    for pat in _TIME_MARKER_PATTERNS:
        if pat.search(text):
            return True
    return False


def _position_key(pos: dict) -> tuple[str, str, str, float | None]:
    return (
        str(pos.get("ticker") or "").upper(),
        str(pos.get("type") or "stock").lower(),
        str(pos.get("expiry") or ""),
        _num(pos.get("strike")),
    )


def _trade_key(trade: dict) -> tuple[str, str, str, float | None]:
    return (
        str(trade.get("ticker") or "").upper(),
        str(trade.get("type") or trade.get("option_type") or "stock").lower(),
        str(trade.get("expiry") or ""),
        _num(trade.get("strike")),
    )


def _earliest_open_buy_reason(
    pos: dict, trades_by_key: dict[tuple, list[dict]]
) -> tuple[str | None, datetime | None, int | None]:
    """Find the earliest BUY trade matching the position's key that
    occurred at-or-after the position's opened_at.

    Returns (reason_text, ts, trade_id) for the earliest such trade,
    else (None, None, None).
    """
    k = _position_key(pos)
    opened = _parse_iso(pos.get("opened_at"))
    candidates = trades_by_key.get(k, [])
    best: tuple[datetime, int, dict] | None = None
    for t in candidates:
        action = t.get("action")
        if not isinstance(action, str) or not action.upper().startswith("BUY"):
            continue
        ts = _parse_iso(t.get("timestamp"))
        if ts is None:
            continue
        # Allow a small backwards window vs opened_at (sometimes the
        # opened_at gets stamped after the first trade by ~ms in the
        # live store). Anything >5s before opened_at is too far.
        if opened is not None and ts < opened.replace(microsecond=0):
            # tolerate small ts skew but reject clearly-stale rows
            if (opened - ts).total_seconds() > 5.0:
                continue
        tid_raw = t.get("id") or 0
        try:
            tid = int(tid_raw)
        except (TypeError, ValueError):
            tid = 0
        if best is None or ts < best[0]:
            best = (ts, tid, t)
    if best is None:
        return None, None, None
    reason = best[2].get("reason")
    if not isinstance(reason, str):
        return None, best[0], best[1]
    return reason, best[0], best[1]


def _classify_position(
    pos: dict,
    reason: str | None,
    *,
    days_held: float,
    zombie_days_floor: float,
    fresh_days_ceil: float,
) -> dict[str, Any]:
    if not isinstance(reason, str) or not reason.strip():
        return {
            "verdict": NO_REASON,
            "catalyst_class": "UNCATEGORIZED",
            "has_time_marker": False,
            "days_held": round(days_held, 3),
            "reason_excerpt": "",
        }

    catalyst = classify_catalyst(reason)
    has_marker = has_time_marker(reason)
    excerpt = reason.strip()
    if len(excerpt) > _REASON_EXCERPT_MAX:
        excerpt = excerpt[:_REASON_EXCERPT_MAX] + "…"

    if catalyst in _DATED_CLASSES and has_marker:
        if days_held >= zombie_days_floor:
            verdict = ZOMBIE
        elif days_held < fresh_days_ceil:
            verdict = FRESH_CATALYST
        else:
            # Between fresh_days_ceil and zombie_days_floor — neither
            # fresh nor zombie. Fold into MIXED via FRESH_CATALYST
            # (the dated catalyst is still in its plausible-impact
            # window for >24h post-event drift).
            verdict = FRESH_CATALYST
    elif catalyst in _STRUCTURAL_CLASSES:
        verdict = STRUCTURAL
    elif catalyst in _DATED_CLASSES and not has_marker:
        # Catalyst keyword present but no time marker — treat as
        # STRUCTURAL because we can't anchor it to a specific event
        # window (e.g. "earnings season tailwind" is structural).
        verdict = STRUCTURAL
    else:  # UNCATEGORIZED
        verdict = UNCATEGORIZED

    return {
        "verdict": verdict,
        "catalyst_class": catalyst,
        "has_time_marker": has_marker,
        "days_held": round(days_held, 3),
        "reason_excerpt": excerpt,
    }


def build_catalyst_expiry_skill(
    open_positions: Sequence[dict] | None,
    trades: Sequence[dict] | None,
    *,
    now: datetime | None = None,
    zombie_days_floor: float = DEFAULT_ZOMBIE_DAYS_FLOOR,
    fresh_days_ceil: float = DEFAULT_FRESH_DAYS_CEIL,
) -> dict[str, Any]:
    """Pure per-open-position catalyst-staleness classifier. Never raises.

    Inputs:
      ``open_positions`` — list of open position dicts (each needs
        ``ticker``, ``type``, ``opened_at``; ``expiry`` / ``strike``
        / ``current_price`` / ``unrealized_pl`` optional).
      ``trades`` — list of trade dicts used to recover the opening
        rationale per position.
      ``now`` — defaults to ``datetime.now(utc)``.
      ``zombie_days_floor`` — dated catalysts open longer than this
        flag as ZOMBIE (default 3 days).
      ``fresh_days_ceil`` — dated catalysts open less than this stay
        FRESH (default 2 days).

    Returns the envelope dict. Always emits the envelope keys; never
    raises.
    """
    now = now or datetime.now(timezone.utc)

    positions = list(open_positions or [])
    trades_list = list(trades or [])

    if not positions:
        return {
            "as_of": now.isoformat(),
            "verdict": NO_DATA,
            "headline": "no open positions",
            "n_positions": 0,
            "counts": {
                ZOMBIE: 0, FRESH_CATALYST: 0, STRUCTURAL: 0,
                UNCATEGORIZED: 0, NO_REASON: 0,
            },
            "positions": [],
            "thresholds": {
                "zombie_days_floor": zombie_days_floor,
                "fresh_days_ceil": fresh_days_ceil,
            },
        }

    by_key: dict[tuple, list[dict]] = {}
    for t in trades_list:
        if not isinstance(t, dict):
            continue
        k = _trade_key(t)
        by_key.setdefault(k, []).append(t)

    counts = {
        ZOMBIE: 0, FRESH_CATALYST: 0, STRUCTURAL: 0,
        UNCATEGORIZED: 0, NO_REASON: 0,
    }
    per_position: list[dict] = []

    for pos in positions:
        if not isinstance(pos, dict):
            continue
        ticker = str(pos.get("ticker") or "").upper()
        if not ticker:
            continue
        opened = _parse_iso(pos.get("opened_at"))
        if opened is None:
            days_held = 0.0
        else:
            days_held = max(0.0, (now - opened).total_seconds() / 86400.0)

        reason, entry_ts, trade_id = _earliest_open_buy_reason(pos, by_key)
        cls = _classify_position(
            pos, reason,
            days_held=days_held,
            zombie_days_floor=zombie_days_floor,
            fresh_days_ceil=fresh_days_ceil,
        )
        counts[cls["verdict"]] = counts.get(cls["verdict"], 0) + 1
        per_position.append({
            "ticker": ticker,
            "type": str(pos.get("type") or "stock").lower(),
            "expiry": pos.get("expiry"),
            "strike": _num(pos.get("strike")),
            "opened_at": pos.get("opened_at"),
            "entry_trade_id": trade_id,
            "entry_ts": entry_ts.isoformat() if entry_ts else None,
            **cls,
        })

    # Aggregate verdict
    n_zombie = counts[ZOMBIE]
    n_fresh = counts[FRESH_CATALYST]
    n_structural = counts[STRUCTURAL]
    n_uncat = counts[UNCATEGORIZED]
    n_no_reason = counts[NO_REASON]

    if n_zombie >= 1:
        verdict = ZOMBIE_HOLDINGS
        zombie_names = [
            p["ticker"] for p in per_position
            if p["verdict"] == ZOMBIE
        ]
        headline = (
            f"{n_zombie} ZOMBIE position(s) — dated catalyst past "
            f"{zombie_days_floor:g}d but still held: "
            f"{', '.join(zombie_names[:5])}"
        )
    elif n_fresh >= 1 and n_structural == 0 and n_uncat == 0 and n_no_reason == 0:
        verdict = ALL_FRESH
        headline = (
            f"all {n_fresh} open position(s) on fresh dated catalysts "
            f"(<{fresh_days_ceil:g}d)"
        )
    elif n_fresh == 0 and n_zombie == 0:
        verdict = STRUCTURAL_BOOK
        headline = (
            f"all {n_structural + n_uncat + n_no_reason} open position(s) on "
            f"structural/uncategorized theses — no dated catalysts"
        )
    else:
        verdict = MIXED_BOOK
        parts = []
        if n_fresh:
            parts.append(f"{n_fresh} fresh")
        if n_structural:
            parts.append(f"{n_structural} structural")
        if n_uncat:
            parts.append(f"{n_uncat} uncategorized")
        if n_no_reason:
            parts.append(f"{n_no_reason} no-reason")
        headline = f"mixed book — {', '.join(parts)}"

    return {
        "as_of": now.isoformat(),
        "verdict": verdict,
        "headline": headline,
        "n_positions": len(per_position),
        "counts": counts,
        "positions": per_position,
        "thresholds": {
            "zombie_days_floor": zombie_days_floor,
            "fresh_days_ceil": fresh_days_ceil,
        },
        "note": (
            "Advisory only — never gates Opus, never injected into the "
            "decision prompt as a directive (AGENTS.md #2 / #12)."
        ),
    }
