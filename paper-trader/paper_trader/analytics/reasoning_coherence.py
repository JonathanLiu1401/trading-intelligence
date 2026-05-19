"""Reasoning coherence — how stable is Opus's HOLD justification cycle-to-cycle?

When Opus repeats HOLD for many cycles, is it *reiterating* the same thesis
("NVDA earnings in 0.8d, hold through print") or *citing different content
each pass* ("today: macro; previous: catalyst; before that: technicals")? A
stable thesis is a conviction signal; a rapidly drifting one is a confusion
signal — the bot is groping for a different justification each cycle.

Existing analytics are silent on this:
  * ``/api/decision-drought`` counts consecutive NO_DECISION cycles
    (frequency of silence, not coherence of HOLDs that bracket it).
  * ``/api/decision-forensics`` diagnoses *why* Opus said HOLD on a specific
    signal, but only for the latest decision (no across-time view).
  * ``/api/thesis-drift`` re-tests an OPEN POSITION'S entry thesis against
    current state (different question entirely — it asks "is the position
    still justified", not "is the bot saying the same thing twice").

This builder is the across-time complement: token-set Jaccard similarity
between consecutive HOLD-reasoning prose, summarised into one ``regime``
verdict the operator can read in two seconds.

Pure: no DB, no LLM, no network. Caller passes the last N decision rows
from ``store.recent_decisions``; this filters to HOLDs whose reasoning
parses as the standard ``{"decision": {"reasoning": "..."}}`` JSON envelope,
then compares consecutive pairs.

Observational only — never gates Opus, never injected into the decision
prompt, no caps (invariants #2/#12 — the ``stress_scenarios`` /
``shadow_vs_claude`` precedent).
"""
from __future__ import annotations

import json
import re
from typing import Any

# Threshold above which median similarity reads as a stable, reiterated
# thesis. Below ``DRIFTING_THRESHOLD`` reads as rapid drift (each HOLD cites
# essentially different content). Tuned conservatively against live HOLD
# reasonings in 2026-05 logs — same-thesis pairs land 0.45-0.75, different-
# thesis pairs land 0.05-0.20.
STABLE_THRESHOLD = 0.60
DRIFTING_THRESHOLD = 0.30

# Minimum number of consecutive HOLD pairs to emit a regime verdict. Below
# this the median is too noisy to read; raw stats are still emitted so the
# operator sees "only 1 HOLD pair this window" rather than mistaking the
# silence for OK.
MIN_PAIRS_FOR_VERDICT = 3

# Token filter for reasoning prose. Stopwords are the function words that
# carry no thesis identity; a substantive overlap between two reasonings
# must come from content words. Length>=3 drops "a"/"is"/"of" implicitly
# without needing them in the stopword list.
_WORD = re.compile(r"[a-z0-9]+")
_MIN_TOKEN_LEN = 3
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "this", "that", "into", "over",
    "than", "but", "are", "was", "were", "has", "have", "had", "its",
    "not", "any", "all", "also", "still", "even", "very", "more", "less",
    "now", "yet", "may", "can", "could", "would", "should", "will", "just",
    "out", "off", "too", "via", "per", "while", "since", "until", "between",
})


def _tokens(text: str | None) -> set[str]:
    """Lowercased content-token set: alphanumeric, length >= MIN_TOKEN_LEN,
    non-stopword. Empty / ``None`` -> empty set."""
    if not text:
        return set()
    return {
        t for t in _WORD.findall(text.lower())
        if len(t) >= _MIN_TOKEN_LEN and t not in _STOPWORDS
    }


def _jaccard(a: set, b: set) -> float:
    """|a ∩ b| / |a ∪ b|. Two empty sets -> 0.0 (contentless reasonings must
    not register as a 1.0 match)."""
    if not a and not b:
        return 0.0
    union = a | b
    if not union:
        return 0.0
    return len(a & b) / len(union)


def _is_hold(action_taken: str | None) -> bool:
    """Decisions table free-text. Canonical HOLD shape is ``"HOLD <TICKER> →
    HOLD"`` but bare ``"HOLD"`` also appears. NO_DECISION, BLOCKED, SKIPPED,
    and any FILLED row are not HOLDs."""
    if not action_taken:
        return False
    return str(action_taken).strip().upper().startswith("HOLD")


def _extract_reasoning(reasoning_blob: str | None) -> str | None:
    """The ``decisions.reasoning`` column carries the raw Opus output.
    Canonical shape is a JSON envelope ``{"decision": {"reasoning": "..."}}``;
    parse failures may carry a ``parse_failed:`` / ``retry_failed:`` prefix
    on the raw response (see ``strategy._should_retry_parse``). Return the
    inner ``reasoning`` string, or ``None`` when the row cannot yield a
    thesis comparison."""
    if not reasoning_blob:
        return None
    s = str(reasoning_blob).strip()
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
    dec = obj.get("decision")
    if isinstance(dec, dict):
        r = dec.get("reasoning")
        if isinstance(r, str) and r.strip():
            return r
    r = obj.get("reasoning")
    if isinstance(r, str) and r.strip():
        return r
    return None


def build_reasoning_coherence(
    decisions: list[dict[str, Any]] | None,
    *,
    stable_threshold: float = STABLE_THRESHOLD,
    drifting_threshold: float = DRIFTING_THRESHOLD,
    min_pairs: int = MIN_PAIRS_FOR_VERDICT,
) -> dict:
    """Compute reasoning-coherence verdict over a list of ``decisions`` rows.

    Expected row shape (matches ``store.recent_decisions``):
    ``{timestamp, action_taken, reasoning, ...}``. Rows are paired in the
    order given — for ``recent_decisions`` (newest-first) the pair list is
    newest-first; verdict statistics are order-invariant.

    Returns a dict with ``state`` in:
      * ``NO_DATA`` — no HOLD rows with parseable reasoning in window
      * ``INSUFFICIENT`` — fewer than ``min_pairs`` HOLD pairs, raw stats emitted
      * ``OK`` — verdict emitted

    On ``OK`` the ``regime`` is one of ``STABLE_THESIS`` / ``DRIFTING`` /
    ``RAPID_DRIFT`` based on ``median_similarity``.
    """
    decs = list(decisions or [])

    reasoned: list[tuple[str, str]] = []
    n_hold_total = 0
    for d in decs:
        if not _is_hold(d.get("action_taken")):
            continue
        n_hold_total += 1
        text = _extract_reasoning(d.get("reasoning"))
        if not text:
            continue
        ts = d.get("timestamp") or ""
        reasoned.append((ts, text))

    if not reasoned:
        return {
            "state": "NO_DATA",
            "n_hold_decisions": n_hold_total,
            "n_pairs": 0,
            "headline": (
                "No HOLD decisions with parseable reasoning in window — "
                "reasoning coherence cannot be assessed."
            ),
            "regime": None,
            "median_similarity": None,
            "min_similarity": None,
            "max_similarity": None,
            "pairs": [],
            "stable_threshold": stable_threshold,
            "drifting_threshold": drifting_threshold,
        }

    pairs: list[dict] = []
    for i in range(len(reasoned) - 1):
        a_ts, a_text = reasoned[i]
        b_ts, b_text = reasoned[i + 1]
        sim = _jaccard(_tokens(a_text), _tokens(b_text))
        pairs.append({
            "a_ts": a_ts,
            "b_ts": b_ts,
            "similarity": round(sim, 3),
        })

    if not pairs:
        return {
            "state": "INSUFFICIENT",
            "n_hold_decisions": n_hold_total,
            "n_pairs": 0,
            "headline": (
                f"Only {len(reasoned)} HOLD reasoning in window — "
                "need ≥2 to measure coherence."
            ),
            "regime": None,
            "median_similarity": None,
            "min_similarity": None,
            "max_similarity": None,
            "pairs": [],
            "stable_threshold": stable_threshold,
            "drifting_threshold": drifting_threshold,
        }

    sims = sorted(p["similarity"] for p in pairs)
    n = len(sims)
    median = sims[n // 2] if n % 2 else (sims[n // 2 - 1] + sims[n // 2]) / 2.0
    median = round(median, 3)

    if len(pairs) < min_pairs:
        return {
            "state": "INSUFFICIENT",
            "n_hold_decisions": n_hold_total,
            "n_pairs": len(pairs),
            "min_pairs": min_pairs,
            "headline": (
                f"{len(pairs)} HOLD pair(s) — need ≥{min_pairs} for regime "
                f"verdict (median sim {median:.2f} so far)."
            ),
            "regime": None,
            "median_similarity": median,
            "min_similarity": round(sims[0], 3),
            "max_similarity": round(sims[-1], 3),
            "pairs": pairs,
            "stable_threshold": stable_threshold,
            "drifting_threshold": drifting_threshold,
        }

    if median >= stable_threshold:
        regime = "STABLE_THESIS"
        head = (
            f"Stable thesis: median Jaccard {median:.2f} across "
            f"{len(pairs)} HOLD pair(s) — Opus is reiterating the same "
            "justification cycle-to-cycle."
        )
    elif median >= drifting_threshold:
        regime = "DRIFTING"
        head = (
            f"Thesis drifting: median Jaccard {median:.2f} across "
            f"{len(pairs)} HOLD pair(s) — reasoning evolves between holds."
        )
    else:
        regime = "RAPID_DRIFT"
        head = (
            f"Rapid drift: median Jaccard {median:.2f} across "
            f"{len(pairs)} HOLD pair(s) — each HOLD cites different content."
        )

    return {
        "state": "OK",
        "n_hold_decisions": n_hold_total,
        "n_pairs": len(pairs),
        "regime": regime,
        "median_similarity": median,
        "min_similarity": round(sims[0], 3),
        "max_similarity": round(sims[-1], 3),
        "headline": head,
        "pairs": pairs,
        "stable_threshold": stable_threshold,
        "drifting_threshold": drifting_threshold,
    }
