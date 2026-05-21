"""Conviction-language skill — is Opus self-aware about its own conviction?

The desk question this answers: **when Opus said "high conviction" vs
"speculative / probe / small", did the realized returns actually match
that ranking?**

Existing neighbours each see a *different* slice:

* ``/api/cash-conviction-fit`` — *point-in-time* check on whether cash
  level fits the loudest live signal. Forward-looking, not a calibration.
* ``/api/conviction-deployment-curve`` — does *capital deployed* scale with
  some external conviction proxy. About sizing, not language honesty.
* ``/api/decision-confidence`` — distribution of Opus' self-reported
  confidence scalar over time. Says nothing about whether high confidence
  correlated with better outcomes.

None answer the *language-honesty* question: parse the verbatim reasoning
text for conviction-strength phrases, bucket the round-trip outcomes,
report whether the HIGH-CONVICTION bucket actually outperformed the
LOW-CONVICTION bucket.

Verdict matrix:

* ``SELF_AWARE`` — HIGH bucket mean ≥ LOW bucket mean + verdict_gap_pct
  (and both ≥ min_per_bucket). The bot's confidence language is calibrated.
* ``OVERCONFIDENT`` — HIGH mean ≤ LOW mean - verdict_gap_pct. The bot
  says "high conviction" precisely when it shouldn't — anti-calibrated.
* ``NO_PATTERN`` — both buckets full but the gap is within tolerance.
  Language is uninformative.
* ``INSUFFICIENT_DATA`` — either bucket below min_per_bucket.

Pure builder. Pre-joined samples in, dict out, never raises.
Observational only — never gates Opus, no caps (AGENTS.md #2/#12).
"""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import Any, Sequence

# Conviction-language phrase patterns. Order matters — first match wins
# in ``classify_reason``. Patterns are case-insensitive and use word
# boundaries to avoid substring poisoning (e.g. ``probe`` must not match
# ``problem``).
#
# Each pattern is a (label, compiled_regex) pair. The labels are the
# bucket names exposed in the output envelope.
#
# HIGH precedes ADD so "scaling on high conviction" gets HIGH credit.
# LOW precedes ADD so "small probe to add" gets LOW (the probe is the
# operative conviction, not the add). ADD is the catch-all for adds/scales
# that don't carry an explicit conviction adjective.
_HIGH_PATTERNS = [
    r"\bhigh[- ]?conviction\b",
    r"\bstrong[- ]?conviction\b",
    r"\bmax(?:imum)?[- ]?conviction\b",
    r"\bfull[- ]?(?:size|conviction)\b",
    r"\bconviction (?:is )?(?:high|strong|max|maximum)\b",
    r"\baggressive(?:ly)? (?:size|sized|sizing|deploy|deployed|deploying)\b",
]
_LOW_PATTERNS = [
    r"\b(?:small|tiny|micro)[- ]?(?:position|size|sized|stake|tranche)\b",
    r"\btest(?:ing)?[- ]?(?:position|trade|tranche|stake)\b",
    r"\bprobe(?:s|d)?\b",
    r"\bstarter[- ]?(?:position|tranche|stake)?\b",
    r"\bspeculat(?:ive|ion)\b",
    r"\bcautious(?:ly)?\b",
    r"\btentative(?:ly)?\b",
    r"\blow[- ]?conviction\b",
    r"\bweak[- ]?conviction\b",
]
_ADD_PATTERNS = [
    r"\b(?:scale|scaling|scaled)[- ]?(?:in|up|into)?\b",
    r"\baccumulat(?:e|ing|ed)\b",
    r"\bramp(?:ing|ed)?\b",
    r"\bbuild(?:ing)?[- ]?(?:up|into|position)\b",
    r"\badd(?:ing)? (?:to|more|exposure)\b",
]

HIGH = "HIGH"
LOW = "LOW"
ADD = "ADD"
NEUTRAL = "NEUTRAL"

_HIGH_RE = [re.compile(p, re.IGNORECASE) for p in _HIGH_PATTERNS]
_LOW_RE = [re.compile(p, re.IGNORECASE) for p in _LOW_PATTERNS]
_ADD_RE = [re.compile(p, re.IGNORECASE) for p in _ADD_PATTERNS]

# Verdict thresholds.
VERDICT_GAP_PCT = 2.0  # mean-return gap (%) needed for a directional verdict
MIN_PER_BUCKET = 3      # min samples in EACH of HIGH and LOW to fire

# Verdict labels.
SELF_AWARE = "SELF_AWARE"
OVERCONFIDENT = "OVERCONFIDENT"
NO_PATTERN = "NO_PATTERN"
INSUFFICIENT_DATA = "INSUFFICIENT_DATA"


def _num(x: Any) -> float | None:
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        return None
    if x != x:
        return None
    return float(x)


def classify_reason(reason: Any) -> tuple[str, str | None]:
    """Classify a free-text reason into (bucket, matched_phrase).

    Precedence: HIGH > LOW > ADD > NEUTRAL. Within HIGH/LOW/ADD the first
    pattern that matches in source order wins; the matched substring is
    returned so the operator can spot-check the classifier.

    ``None``, non-string, and empty strings all return ``(NEUTRAL, None)``.
    Pure — never raises."""
    if not isinstance(reason, str) or not reason.strip():
        return (NEUTRAL, None)
    for rx in _HIGH_RE:
        m = rx.search(reason)
        if m:
            return (HIGH, m.group(0))
    for rx in _LOW_RE:
        m = rx.search(reason)
        if m:
            return (LOW, m.group(0))
    for rx in _ADD_RE:
        m = rx.search(reason)
        if m:
            return (ADD, m.group(0))
    return (NEUTRAL, None)


def _summarise(samples: list[dict]) -> dict:
    n = len(samples)
    if n == 0:
        return {"n": 0, "mean_pct": None, "median_pct": None, "win_rate": None}
    rets = sorted([s["realized_pct"] for s in samples])
    mean = sum(rets) / n
    if n % 2 == 1:
        med = rets[n // 2]
    else:
        med = (rets[n // 2 - 1] + rets[n // 2]) / 2.0
    wins = sum(1 for r in rets if r > 0)
    return {
        "n": n,
        "mean_pct": round(mean, 2),
        "median_pct": round(med, 2),
        "win_rate": round(wins / n * 100.0, 1),
    }


def build_conviction_language_skill(
    samples: Sequence[dict] | None,
    *,
    now: datetime | None = None,
    min_per_bucket: int = MIN_PER_BUCKET,
    verdict_gap_pct: float = VERDICT_GAP_PCT,
) -> dict:
    """Build the conviction-language-vs-realized-return verdict.

    Each sample is shaped::

        {
            "trade_id": int | None,
            "trade_ts": iso str,
            "ticker": str,
            "action": str,                  # BUY / SELL / ...
            "reason": str | None,           # verbatim trades.reason text
            "realized_pct": float,          # exit return OR mark return
            "closed": bool,
        }

    The route is responsible for joining trades to their round-trip
    outcomes and current marks.

    Returns a stable envelope::

        {
            as_of, verdict, headline,
            n_samples,
            buckets: {HIGH/LOW/ADD/NEUTRAL: {n, mean_pct, median_pct, win_rate}},
            thresholds, samples: [first 50 raw rows]
        }

    Pure — never raises. Malformed samples are silently dropped.
    """
    now = now or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    norm: list[dict] = []
    for s in samples or ():
        if not isinstance(s, dict):
            continue
        realized = _num(s.get("realized_pct"))
        if realized is None:
            continue
        bucket, matched = classify_reason(s.get("reason"))
        norm.append({
            "trade_id": s.get("trade_id"),
            "trade_ts": s.get("trade_ts"),
            "ticker": s.get("ticker"),
            "action": s.get("action"),
            "reason_excerpt": (
                (s.get("reason")[:200] + ("…" if len(s.get("reason")) > 200 else ""))
                if isinstance(s.get("reason"), str)
                else None
            ),
            "matched_phrase": matched,
            "realized_pct": realized,
            "closed": bool(s.get("closed", False)),
            "bucket": bucket,
        })

    n_total = len(norm)
    bucket_agg = {
        b: _summarise([x for x in norm if x["bucket"] == b])
        for b in (HIGH, LOW, ADD, NEUTRAL)
    }

    high = bucket_agg[HIGH]
    low = bucket_agg[LOW]
    if (
        n_total == 0
        or high["n"] < min_per_bucket
        or low["n"] < min_per_bucket
    ):
        verdict = INSUFFICIENT_DATA
        headline = (
            f"INSUFFICIENT_DATA — need ≥{min_per_bucket} samples in both "
            f"HIGH ({high['n']}) and LOW ({low['n']}) buckets to call "
            f"language calibration."
        )
    else:
        gap = high["mean_pct"] - low["mean_pct"]
        if gap >= verdict_gap_pct:
            verdict = SELF_AWARE
            headline = (
                f"SELF_AWARE — HIGH-conviction trades returned "
                f"{high['mean_pct']:+.2f}% mean ({high['n']}) vs LOW "
                f"{low['mean_pct']:+.2f}% ({low['n']}); "
                f"+{gap:.2f}pp — language is calibrated."
            )
        elif gap <= -verdict_gap_pct:
            verdict = OVERCONFIDENT
            headline = (
                f"OVERCONFIDENT — HIGH-conviction trades returned "
                f"{high['mean_pct']:+.2f}% mean ({high['n']}) vs LOW "
                f"{low['mean_pct']:+.2f}% ({low['n']}); "
                f"{gap:.2f}pp — confidence language inverts the outcome."
            )
        else:
            verdict = NO_PATTERN
            headline = (
                f"NO_PATTERN — HIGH mean {high['mean_pct']:+.2f}% "
                f"({high['n']}) vs LOW {low['mean_pct']:+.2f}% "
                f"({low['n']}); gap {gap:+.2f}pp within "
                f"±{verdict_gap_pct:.1f}pp tolerance — language uninformative."
            )

    by_priority = sorted(
        norm,
        key=lambda x: (0 if x["closed"] else 1, x.get("trade_ts") or ""),
    )

    return {
        "as_of": now.isoformat(timespec="seconds"),
        "verdict": verdict,
        "headline": headline,
        "n_samples": n_total,
        "buckets": bucket_agg,
        "thresholds": {
            "min_per_bucket": min_per_bucket,
            "verdict_gap_pct": verdict_gap_pct,
        },
        "samples": [
            {
                "trade_id": x["trade_id"],
                "trade_ts": x["trade_ts"],
                "ticker": x["ticker"],
                "action": x["action"],
                "reason_excerpt": x["reason_excerpt"],
                "matched_phrase": x["matched_phrase"],
                "realized_pct": round(x["realized_pct"], 2),
                "closed": x["closed"],
                "bucket": x["bucket"],
            }
            for x in by_priority[:50]
        ],
    }
