"""Decision confidence — aggregate Opus's self-rated conviction.

Every parseable decision blob carries a ``confidence`` field in the
``{"decision": {"confidence": 0.7, ...}}`` envelope. Across the existing
~80-endpoint analytics surface **nothing** aggregates this signal:

  * ``/api/decision-forensics`` reads ONE decision (no across-time view).
  * ``/api/scorer-confidence`` is the **DecisionScorer**'s confidence
    (the small CPU MLP on the backtest side), not Opus's.
  * ``/api/reasoning-coherence`` measures pair-wise Jaccard between
    HOLD reasonings — a stability metric, not a conviction one.

An operator scanning a paralysed week of HOLDs cannot tell whether the
bot is **confidently** sitting on its hands (high-conviction HOLDs around
a binary event) or **uncertainly** doing nothing (low-conviction churn
that should be flagged). This builder answers that question.

Pure: no DB, no LLM, no network. Caller passes the last N decision rows
from ``store.recent_decisions``; this filters to rows whose reasoning
parses as JSON with a numeric ``confidence`` 0..1.

Observational only — never gates Opus, never injected into the decision
prompt, no caps (invariants #2/#12 — the ``reasoning_coherence`` /
``stress_scenarios`` precedent).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

# Confidence bucket cut points (inclusive lower bound, exclusive upper).
# Tuned to live HOLD samples (mostly 0.5..0.8): a discriminating mid-band
# rather than 4 equal quartiles, so the "convicted" tail is its own bucket.
_BUCKETS = [
    ("low", 0.0, 0.4),
    ("medium", 0.4, 0.6),
    ("high", 0.6, 0.8),
    ("very_high", 0.8, 1.0 + 1e-9),  # include exact 1.0
]

# Regime thresholds applied to the **median** confidence.
CAUTIOUS_THRESHOLD = 0.45
CONVICTED_THRESHOLD = 0.70

# Below this many parsed rows the median is too noisy to read.
MIN_SAMPLES_FOR_VERDICT = 5

# A trend delta (recent half median − older half median) of this size or
# larger flags TRENDING_UP / TRENDING_DOWN as a secondary verdict overlay.
TREND_DELTA = 0.10


def _action_verb(action_taken: str | None) -> str:
    """Extract the leading verb from ``action_taken`` free-text.

    Real shapes (live 2026-05):
      * ``"HOLD NVDA → HOLD"`` → ``HOLD``
      * ``"BUY NVDA → FILLED"`` → ``BUY``
      * ``"SELL_CALL NVDA 200C 2026-05-23 → FILLED"`` → ``SELL_CALL``
      * ``"NO_DECISION"`` → ``NO_DECISION``
      * ``"BLOCKED"`` / ``""`` / ``None`` → ``UNKNOWN``
    """
    if not action_taken:
        return "UNKNOWN"
    s = str(action_taken).strip()
    if not s:
        return "UNKNOWN"
    first = s.split(None, 1)[0]
    return first.upper()


def _extract_confidence(blob: str | None) -> float | None:
    """Pull the ``confidence`` float from a decisions.reasoning blob.

    The canonical envelope is ``{"decision": {"confidence": 0.7, ...}}``
    (the live ``strategy.py`` output schema). Falls back to the top-level
    ``confidence`` key if the row is recorded without the wrapping. Returns
    ``None`` for any unparseable / non-numeric / out-of-band value — never
    raises. Out-of-band 0..1 values are silently clamped to the closed
    interval rather than dropped (a model returning 1.2 is conviction, not
    invalid data — read it but bound it).
    """
    if not blob:
        return None
    s = str(blob).strip()
    if not s:
        return None
    for prefix in ("parse_failed:", "retry_failed:"):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s).strip()
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        return None
    if not isinstance(obj, dict):
        return None

    raw = None
    dec = obj.get("decision")
    if isinstance(dec, dict) and "confidence" in dec:
        raw = dec["confidence"]
    elif "confidence" in obj:
        raw = obj["confidence"]
    if raw is None:
        return None
    try:
        val = float(raw)
    except (TypeError, ValueError):
        return None
    if val != val:  # NaN
        return None
    if val < 0.0:
        return 0.0
    if val > 1.0:
        return 1.0
    return val


def _median(values: list[float]) -> float:
    """Population median over a NON-empty list of floats."""
    s = sorted(values)
    n = len(s)
    if n % 2:
        return s[n // 2]
    return (s[n // 2 - 1] + s[n // 2]) / 2.0


def _bucket(val: float) -> str:
    for name, lo, hi in _BUCKETS:
        if lo <= val < hi:
            return name
    return _BUCKETS[-1][0]


def build_decision_confidence(
    decisions: list[dict[str, Any]] | None,
    *,
    cautious_threshold: float = CAUTIOUS_THRESHOLD,
    convicted_threshold: float = CONVICTED_THRESHOLD,
    min_samples: int = MIN_SAMPLES_FOR_VERDICT,
    trend_delta: float = TREND_DELTA,
) -> dict:
    """Aggregate Opus's self-rated confidence across a decision window.

    Args:
        decisions: rows in ``store.recent_decisions`` order (newest-first).
            Order is preserved so the trend split (recent half vs older
            half) is computed on the caller-supplied chronology — a
            ``store.recent_decisions`` newest-first list reads the FIRST
            half as RECENT.

    Returns a dict with ``state`` in:
      * ``NO_DATA`` — zero parseable confidence values
      * ``INSUFFICIENT`` — < min_samples; raw stats emitted, regime None
      * ``OK`` — verdict emitted

    Regime tag (when OK):
      * ``CAUTIOUS`` — median < cautious_threshold
      * ``NEUTRAL`` — cautious ≤ median < convicted
      * ``CONVICTED`` — median ≥ convicted_threshold

    Trend tag (when OK & ≥2 samples per half):
      * ``TRENDING_UP`` — recent median exceeds older median by ≥ trend_delta
      * ``TRENDING_DOWN`` — older exceeds recent by ≥ trend_delta
      * ``FLAT`` — otherwise

    Pure: never raises on malformed rows.
    """
    rows = list(decisions or [])
    n_rows = len(rows)

    values: list[float] = []
    per_action: dict[str, list[float]] = {}
    n_unparseable = 0

    for d in rows:
        v = _extract_confidence(d.get("reasoning"))
        if v is None:
            n_unparseable += 1
            continue
        values.append(v)
        verb = _action_verb(d.get("action_taken"))
        per_action.setdefault(verb, []).append(v)

    if not values:
        return {
            "state": "NO_DATA",
            "n_decisions": n_rows,
            "n_with_confidence": 0,
            "n_unparseable": n_unparseable,
            "median": None,
            "mean": None,
            "min": None,
            "max": None,
            "buckets": {name: 0 for name, _, _ in _BUCKETS},
            "by_action": {},
            "trend": None,
            "regime": None,
            "headline": (
                "No parseable confidence values in window — "
                "decision confidence cannot be assessed."
            ),
            "cautious_threshold": cautious_threshold,
            "convicted_threshold": convicted_threshold,
            "min_samples": min_samples,
        }

    median = round(_median(values), 3)
    mean = round(sum(values) / len(values), 3)
    lo = round(min(values), 3)
    hi = round(max(values), 3)
    buckets = Counter(_bucket(v) for v in values)
    bucket_dict = {name: buckets.get(name, 0) for name, _, _ in _BUCKETS}

    by_action = {
        verb: {
            "n": len(vs),
            "median": round(_median(vs), 3),
            "mean": round(sum(vs) / len(vs), 3),
        }
        for verb, vs in sorted(per_action.items())
    }

    if len(values) < min_samples:
        return {
            "state": "INSUFFICIENT",
            "n_decisions": n_rows,
            "n_with_confidence": len(values),
            "n_unparseable": n_unparseable,
            "median": median,
            "mean": mean,
            "min": lo,
            "max": hi,
            "buckets": bucket_dict,
            "by_action": by_action,
            "trend": None,
            "regime": None,
            "headline": (
                f"Only {len(values)} confidence sample(s) in window — "
                f"need ≥{min_samples} for a regime verdict "
                f"(median {median:.2f} so far)."
            ),
            "cautious_threshold": cautious_threshold,
            "convicted_threshold": convicted_threshold,
            "min_samples": min_samples,
        }

    # Caller-supplied order is preserved. ``recent_decisions`` is
    # newest-first, so the FIRST half of ``values`` is the recent half.
    half = len(values) // 2
    if half >= 2:
        recent_vals = values[:half]
        older_vals = values[-half:]
        recent_med = _median(recent_vals)
        older_med = _median(older_vals)
        delta = round(recent_med - older_med, 3)
        if delta >= trend_delta:
            trend_tag = "TRENDING_UP"
        elif delta <= -trend_delta:
            trend_tag = "TRENDING_DOWN"
        else:
            trend_tag = "FLAT"
        trend_block = {
            "tag": trend_tag,
            "recent_median": round(recent_med, 3),
            "older_median": round(older_med, 3),
            "delta": delta,
            "split_size": half,
        }
    else:
        trend_block = None

    if median < cautious_threshold:
        regime = "CAUTIOUS"
        head = (
            f"Cautious regime: median confidence {median:.2f} across "
            f"{len(values)} decision(s) — Opus is hedging."
        )
    elif median < convicted_threshold:
        regime = "NEUTRAL"
        head = (
            f"Neutral regime: median confidence {median:.2f} across "
            f"{len(values)} decision(s)."
        )
    else:
        regime = "CONVICTED"
        head = (
            f"Convicted regime: median confidence {median:.2f} across "
            f"{len(values)} decision(s) — Opus is decisive."
        )

    if trend_block and trend_block["tag"] != "FLAT":
        head = (
            f"{head} {trend_block['tag'].replace('_', ' ').title()}: "
            f"recent {trend_block['recent_median']:.2f} vs "
            f"older {trend_block['older_median']:.2f} "
            f"(Δ {trend_block['delta']:+.2f})."
        )

    return {
        "state": "OK",
        "n_decisions": n_rows,
        "n_with_confidence": len(values),
        "n_unparseable": n_unparseable,
        "median": median,
        "mean": mean,
        "min": lo,
        "max": hi,
        "buckets": bucket_dict,
        "by_action": by_action,
        "trend": trend_block,
        "regime": regime,
        "headline": head,
        "cautious_threshold": cautious_threshold,
        "convicted_threshold": convicted_threshold,
        "min_samples": min_samples,
    }
