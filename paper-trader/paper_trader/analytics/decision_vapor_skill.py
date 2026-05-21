"""Decision-vapor skill — for FILLED decisions, does the reasoning ground
in specifics (numeric figures + a named catalyst + an explicit ticker),
or does it read as generic vapor ("strong setup, building position")?

Existing reasoning analytics measure different axes and leave the
specificity question open:

* ``reasoning_coherence`` — pair-wise Jaccard between consecutive HOLD
  reasonings (a stability metric, not a specificity one).
* ``reasoning_action_verbs`` — directional-verb consistency between the
  structured ``action`` and the prose (action-vs-prose agreement).
* ``reasoning_themes`` — n-gram phrase frequency across decisions (which
  topics recur, not whether any one decision is well-grounded).
* ``decision_confidence`` — aggregates the self-rated ``confidence``
  scalar (a number, not the prose backing it).
* ``conviction_language_skill`` — does the prose language match the
  conviction number (calibration of *expressed* conviction).

None grade the per-decision **specificity** question: when Opus says
BUY NVDA at 90% conviction, does the rationale cite an actual catalyst
(earnings, beat, guide, FDA, rate, fed, court, deal, …) and an actual
number ($58.3B, RSI 61.85, +5.5%, …) and the ticker by name — or is
it three sentences of generic conviction with no anchors?

A FILLED decision whose reasoning is VAPOR is structurally indistin-
guishable from one that's SPECIFIC at JSON-parse time. Both produce a
trade, both pass risk gates, both move money. But the post-mortem
diagnostics are entirely different: a SPECIFIC trade that fails has a
falsifiable thesis you can trace ("we sold the earnings beat into a
guidance cut"); a VAPOR trade that fails has nothing for the next
review pass to learn from.

Per-decision classification — 3 signals, each independently checked:

* ``has_numeric`` — at least one numeric figure (digits with optional
  decimal, optional %, $ prefix, optional sign). E.g. ``58.3B``,
  ``+5.5%``, ``$220``, ``RSI 61.85``. Pure single digits inside common
  filler tokens like ``one`` / ``2025`` (year-only) are still numeric
  by this definition — that's deliberate; reasoning that cites an
  earnings *year* is more grounded than reasoning with no number at
  all.
* ``has_catalyst`` — at least one of the catalyst tokens (earnings,
  beat, miss, missed, guide, guidance, upgrade, downgrade, FDA,
  approval, court, ruling, fed, rate, hike, cut, deal, merger,
  acquire, buyback, dividend, warn, warning, breakout, surge, plunge,
  jump, drop, ban, sanction, tariff, layoff, lawsuit, settle, recall,
  partnership). Case-insensitive whole-word match.
* ``has_ticker`` — at least one upper-case ticker symbol (1-5 chars).
  We match against an injectable watchlist when supplied, else fall
  back to a regex (``[A-Z]{1,5}`` whole-token) that filters out 1-2
  letter false positives by requiring length ≥ 2 unless followed by
  a ``$`` cashtag prefix.

Classification:

* ``SPECIFIC`` — all 3 signals present.
* ``SEMI`` — exactly 2 signals.
* ``VAPOR`` — 0 or 1 signals.

Aggregate over a recent window of FILLED decisions (BUY / SELL /
BUY_CALL / BUY_PUT / SELL_CALL / SELL_PUT — anything that resulted in
``→ FILLED``):

* ``SPECIFIC`` — ``specific_pct`` ≥ ``SPECIFIC_PCT_FLOOR`` (default
  50%) AND ``vapor_pct`` < ``VAPOR_PCT_CEIL`` (default 15%). Most
  FILLED trades are well-grounded.
* ``VAPOR_DECISIONS`` — ``vapor_pct`` ≥ ``VAPOR_PCT_FLOOR``
  (default 35%). A material share of FILLED trades are unanchored.
* ``MIXED`` — neither extreme. The desk is grounded most of the time
  but not consistently.
* ``NO_DATA`` — fewer than ``MIN_FILLED_FOR_VERDICT`` (5) FILLED
  decisions in the window.

Pure builder. Decisions in, dict out, never raises. Observational only
— never gates Opus, no caps (AGENTS.md #2 / #12 — the
``reasoning_coherence`` / ``conviction_language_skill`` precedent).
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from typing import Any, Iterable, Sequence

DEFAULT_WINDOW_HOURS = 168.0  # 7 days
DEFAULT_VAPOR_PCT_FLOOR = 35.0
DEFAULT_VAPOR_PCT_CEIL = 15.0
DEFAULT_SPECIFIC_PCT_FLOOR = 50.0
MIN_FILLED_FOR_VERDICT = 5

# Numeric token: optional $ prefix, optional sign, digits with optional
# decimal, optional %, B/M/K/T suffix. Examples that match:
#   58.3B, +5.5%, $220, 61.85, -2, 1.2k
# Examples that do NOT match (intentionally — context-free numerals like
# bare "2" in "for 2 sessions" are weak as standalone grounding signals):
#   We use a more permissive match that catches any digit run, because
#   live operator experience is that *any* number in reasoning beats no
#   number at all.
_NUMERIC_RE = re.compile(r"[-+]?\$?\d+(?:[.,]\d+)?[%BMKTbmkt]?")

# Catalyst whole-word tokens. Case-insensitive. Order-independent.
_CATALYST_TOKENS = frozenset({
    "earnings", "earning", "beat", "beats", "miss", "missed", "misses",
    "guide", "guidance", "guided",
    "upgrade", "upgrades", "upgraded", "downgrade", "downgrades", "downgraded",
    "fda", "approval", "approved", "rejection", "rejected",
    "court", "ruling", "rules", "ruled",
    "fed", "fomc", "rate", "rates", "hike", "hikes", "cut", "cuts",
    "deal", "merger", "acquire", "acquisition", "acquired", "acquires",
    "buyback", "buybacks", "repurchase",
    "dividend", "dividends",
    "warn", "warns", "warning", "warned",
    "breakout", "breakdown",
    "surge", "surges", "surged", "plunge", "plunges", "plunged",
    "jump", "jumps", "jumped", "drop", "drops", "dropped",
    "ban", "banned", "bans", "sanction", "sanctions", "sanctioned",
    "tariff", "tariffs",
    "layoff", "layoffs",
    "lawsuit", "settled", "settles", "settle", "settlement",
    "recall", "recalls", "recalled",
    "partnership", "partner", "partners",
    "ipo", "spinoff", "split",
    "catalyst", "catalysts",
    # macro / market structure cues that ground a thesis
    "yield", "yields", "vix", "spx", "qqq", "spy",
    "kospi", "nikkei", "shanghai", "hang",  # global indices
    "futures",
    "buyer", "seller", "flow", "flows",
})

# Token finder for catalyst + ticker scans. Captures whole words.
_WORD_RE = re.compile(r"[A-Za-z]{2,}|\$[A-Za-z]{1,5}")

# Ticker regex when no explicit watchlist supplied — match 2-5 capital
# letters bordered by non-alphanumerics. Cashtag $TICKER also matches.
# 1-letter cap tokens are dropped (too many false positives: "A", "I").
_TICKER_RE = re.compile(r"(?:(?<=^)|(?<=[^A-Za-z0-9]))(?:\$?[A-Z]{2,5})(?=$|[^A-Za-z0-9])")

# Common 2-5 letter cap-token false positives that show up in reasoning
# prose ("ML", "AI", "BUY", "SELL", "FILL", "OPEN", "HOLD", "CLOSE",
# "JSON", "TODO", etc.). Filtered when matching against the regex
# fallback path; *not* filtered against an explicit watchlist (if the
# operator chose to include "AI" as a ticker, that's their call).
_TICKER_BLACKLIST = frozenset({
    "BUY", "SELL", "HOLD", "FILL", "OPEN", "CLOSE", "TRIM", "ADD",
    "AND", "OR", "BUT", "FOR", "WITH", "FROM", "INTO", "OVER",
    "THE", "TO", "AT", "ON", "OF", "IN", "BY", "IS", "AS",
    "ML", "AI", "EU", "US", "USA", "UK", "CA", "JP", "CN",
    "CEO", "CFO", "COO", "CTO", "CIO", "CSO",
    "JSON", "TODO", "FIXME", "TBD", "NA", "NM",
    "NYSE", "NASDAQ", "DOW", "SPX", "RTY",  # indices are not tradeable tickers in this universe
    "AH", "PM", "AM", "ET", "PT", "GMT", "UTC",
    "FOMC", "FED", "ECB", "BOJ", "PBOC",
    "RSI", "MACD", "BB", "EMA", "SMA", "ATR", "ADX",
    "CPI", "PPI", "GDP", "ISM", "NFP",
    "WSJ", "FT", "AP", "CNBC", "CNN", "BBC", "DJ",  # outlet abbreviations
    "YOY", "MOM", "QOQ", "AVG", "MAX", "MIN", "ROI",
    "PE", "EPS", "PEG", "EBITDA", "FCF",
    "NO", "YES", "OK", "OFF",
})


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


def _extract_reasoning_text(decision: dict) -> str:
    """Extract the natural-language reasoning from a decision row.

    The decision's ``reasoning`` field is canonically the raw Opus JSON
    envelope (a string). The natural-language prose lives at
    ``reasoning_envelope["decision"]["reasoning"]``. If parsing fails
    we fall back to the raw string itself — a parse_failed: prefix row
    still has *some* prose worth scanning, even if the structure is
    broken.
    """
    raw = decision.get("reasoning")
    if not isinstance(raw, str) or not raw:
        return ""
    # Try the JSON-envelope path first.
    try:
        env = json.loads(raw)
        if isinstance(env, dict):
            inner = env.get("decision")
            if isinstance(inner, dict):
                txt = inner.get("reasoning")
                if isinstance(txt, str) and txt:
                    return txt
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return raw


def _is_filled(action_taken: Any) -> bool:
    """A decision row is FILLED iff ``action_taken`` ends with FILLED.
    The canonical format is ``"BUY NVDA → FILLED"`` per AGENTS.md #11.
    """
    if not isinstance(action_taken, str):
        return False
    return action_taken.endswith("FILLED") or action_taken.endswith("→ FILLED")


def detect_signals(
    text: str,
    watchlist: Iterable[str] | None = None,
) -> dict[str, bool]:
    """Pure per-text signal-detection — has_numeric / has_catalyst /
    has_ticker, each boolean.

    ``watchlist`` is optional; if supplied, ``has_ticker`` requires a
    whole-word match against the uppercased set. If absent, the regex
    fallback is used (2-5 uppercase letters, with a curated blacklist).
    """
    if not isinstance(text, str) or not text:
        return {"has_numeric": False, "has_catalyst": False, "has_ticker": False}

    # has_numeric
    has_numeric = bool(_NUMERIC_RE.search(text))

    # has_catalyst — case-insensitive whole-word scan
    text_lower = text.lower()
    has_catalyst = False
    for tok in _WORD_RE.finditer(text_lower):
        w = tok.group(0).lstrip("$")
        if w in _CATALYST_TOKENS:
            has_catalyst = True
            break

    # has_ticker
    if watchlist:
        wl = {t.upper() for t in watchlist if isinstance(t, str) and t}
        # Use the same uppercased-token regex but allow 1-letter symbols
        # if the watchlist explicitly contains them.
        has_ticker = False
        for tok in re.finditer(r"(?:(?<=^)|(?<=[^A-Za-z0-9]))(?:\$?[A-Z]{1,5})(?=$|[^A-Za-z0-9])", text):
            sym = tok.group(0).lstrip("$")
            if sym in wl:
                has_ticker = True
                break
    else:
        has_ticker = False
        for tok in _TICKER_RE.finditer(text):
            sym = tok.group(0).lstrip("$")
            if sym in _TICKER_BLACKLIST:
                continue
            has_ticker = True
            break

    return {
        "has_numeric": has_numeric,
        "has_catalyst": has_catalyst,
        "has_ticker": has_ticker,
    }


def classify_specificity(signals: dict[str, bool]) -> str:
    """Pure score → SPECIFIC / SEMI / VAPOR."""
    n = sum(1 for k in ("has_numeric", "has_catalyst", "has_ticker") if signals.get(k))
    if n >= 3:
        return "SPECIFIC"
    if n == 2:
        return "SEMI"
    return "VAPOR"


def build_decision_vapor_skill(
    decisions: Sequence[Any] | None,
    *,
    now: datetime | None = None,
    watchlist: Iterable[str] | None = None,
    window_hours: float = DEFAULT_WINDOW_HOURS,
    vapor_pct_floor: float = DEFAULT_VAPOR_PCT_FLOOR,
    vapor_pct_ceil: float = DEFAULT_VAPOR_PCT_CEIL,
    specific_pct_floor: float = DEFAULT_SPECIFIC_PCT_FLOOR,
    sample_limit: int = 20,
) -> dict[str, Any]:
    """Pure FILLED-reasoning specificity classifier. Never raises.

    Inputs:
      ``decisions`` — list of decision dicts ``{action_taken, reasoning,
        timestamp, ...}``. Caller does not need to filter to FILLED.
      ``now`` — defaults to ``datetime.now(utc)``.
      ``watchlist`` — optional set of ticker symbols. When supplied,
        ``has_ticker`` requires a whole-word match against this set.
      ``window_hours`` — analysis window. SELLs/BUYs older than this
        are ignored.

    Threshold overrides exposed for tests + caller knobs.
    """
    now = now or datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=max(0.0, window_hours))
    watchlist_set = None
    if watchlist:
        watchlist_set = {t.upper() for t in watchlist if isinstance(t, str) and t}

    n_specific = 0
    n_semi = 0
    n_vapor = 0
    samples: list[dict[str, Any]] = []

    for d in (decisions or []):
        if not isinstance(d, dict):
            continue
        if not _is_filled(d.get("action_taken")):
            continue
        ts_dt = _parse_iso(d.get("timestamp"))
        if ts_dt is None:
            continue
        if ts_dt < cutoff:
            continue
        text = _extract_reasoning_text(d)
        signals = detect_signals(text, watchlist_set)
        klass = classify_specificity(signals)
        if klass == "SPECIFIC":
            n_specific += 1
        elif klass == "SEMI":
            n_semi += 1
        else:
            n_vapor += 1
        if len(samples) < sample_limit:
            samples.append({
                "id": d.get("id"),
                "ts": ts_dt.isoformat(),
                "action_taken": d.get("action_taken"),
                "has_numeric": signals["has_numeric"],
                "has_catalyst": signals["has_catalyst"],
                "has_ticker": signals["has_ticker"],
                "klass": klass,
                "excerpt": (text[:160] + "…") if len(text) > 160 else text,
            })

    n_filled = n_specific + n_semi + n_vapor
    specific_pct: float | None
    semi_pct: float | None
    vapor_pct: float | None
    if n_filled > 0:
        specific_pct = round((n_specific / n_filled) * 100.0, 2)
        semi_pct = round((n_semi / n_filled) * 100.0, 2)
        vapor_pct = round((n_vapor / n_filled) * 100.0, 2)
    else:
        specific_pct = semi_pct = vapor_pct = None

    # Verdict ladder
    if n_filled < MIN_FILLED_FOR_VERDICT:
        verdict = "NO_DATA"
        headline = (
            f"insufficient: {n_filled} FILLED decisions in last "
            f"{window_hours:g}h (min {MIN_FILLED_FOR_VERDICT})"
        )
    else:
        vp = vapor_pct if vapor_pct is not None else 0.0
        sp = specific_pct if specific_pct is not None else 0.0
        if sp >= specific_pct_floor and vp < vapor_pct_ceil:
            verdict = "SPECIFIC"
            headline = (
                f"{n_specific}/{n_filled} ({sp:.0f}%) FILLED decisions "
                f"cite specifics; vapor {vp:.0f}%"
            )
        elif vp >= vapor_pct_floor:
            verdict = "VAPOR_DECISIONS"
            headline = (
                f"{n_vapor}/{n_filled} ({vp:.0f}%) FILLED decisions read "
                f"as vapor — missing numbers or catalysts"
            )
        else:
            verdict = "MIXED"
            headline = (
                f"mixed: {sp:.0f}% specific / {vp:.0f}% vapor "
                f"across {n_filled} FILLED decisions"
            )

    # Samples newest-first for the dashboard.
    samples.sort(key=lambda s: s["ts"], reverse=True)

    return {
        "verdict": verdict,
        "headline": headline,
        "as_of": now.isoformat(),
        "window_hours": window_hours,
        "stats": {
            "n_filled": n_filled,
            "n_specific": n_specific,
            "n_semi": n_semi,
            "n_vapor": n_vapor,
            "specific_pct": specific_pct,
            "semi_pct": semi_pct,
            "vapor_pct": vapor_pct,
        },
        "thresholds": {
            "vapor_pct_floor": vapor_pct_floor,
            "vapor_pct_ceil": vapor_pct_ceil,
            "specific_pct_floor": specific_pct_floor,
            "min_filled_for_verdict": MIN_FILLED_FOR_VERDICT,
        },
        "samples": samples,
    }
