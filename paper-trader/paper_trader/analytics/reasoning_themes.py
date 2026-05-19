"""Reasoning themes — what phrases dominate Opus's decision rationale.

The decisions table carries the raw Opus reasoning prose; over weeks it
accumulates the bot's recurring mental models ("earnings premium",
"concentration drag", "memory super-cycle", "macro overhang"). Neither
``/api/reasoning-coherence`` (which measures **pair-wise Jaccard** between
consecutive HOLDs — a stability metric, not a vocabulary one) nor
``/api/decision-forensics`` (one-decision diagnostic) surfaces the
distribution. Operators reading the dashboard cannot tell at a glance
what topics Opus has *been talking about* — only how stable any one HOLD
is vs the previous.

This builder is the descriptive complement: across a window of recent
decisions, count the content 1- and 2-grams and surface the top phrases
ranked by **how many decisions mentioned them** (not raw token count — a
phrase that appears 30 times in one verbose reasoning is less interesting
than one that recurs across 12 different decisions).

Pure: no DB, no LLM, no network. Caller passes the last N decision rows
from ``store.recent_decisions``; this filters to rows with parseable JSON
reasoning (HOLD/FILLED) plus rows whose ``reasoning`` is a non-empty raw
string (NO_DECISION timeout messages, ``parse_failed:`` raw captures).

Observational only — never gates Opus, never injected into the decision
prompt, no caps (invariants #2/#12 — the ``reasoning_coherence`` /
``stress_scenarios`` precedent).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

_WORD = re.compile(r"[a-z0-9]+")
_MIN_TOKEN_LEN = 3

# Same content-word filter as reasoning_coherence — function words add no
# thematic identity. Augmented with prose-glue that recurs across every
# reasoning and would otherwise dominate the unigram leaderboard.
_STOPWORDS = frozenset({
    "the", "and", "for", "with", "from", "this", "that", "into", "over",
    "than", "but", "are", "was", "were", "has", "have", "had", "its",
    "not", "any", "all", "also", "still", "even", "very", "more", "less",
    "now", "yet", "may", "can", "could", "would", "should", "will", "just",
    "out", "off", "too", "via", "per", "while", "since", "until", "between",
    "amid", "against", "during", "before", "after", "about", "above", "below",
    "their", "them", "they", "these", "those", "such", "some", "other",
    "which", "what", "who", "whom", "whose", "where", "when", "how",
    "been", "being", "does", "did", "doing", "done", "here", "there",
    "then", "thus", "hence", "rather", "either", "neither", "both",
    "much", "many", "few", "most", "least", "only", "own", "same",
    "another", "each", "every",
})


def _extract_reasoning_text(blob: str | None) -> str | None:
    """Pull the textual reasoning from a ``decisions.reasoning`` column.

    Three real shapes (live 2026-05 inspection):
      * JSON envelope ``{"decision": {"reasoning": "..."}}`` — the HOLD /
        FILLED canonical path
      * ``parse_failed:`` / ``retry_failed:`` prefix on raw Opus output
        when ``strategy._parse_decision`` ran out of retries
      * A bare prose string for NO_DECISION timeouts and circuit-breaker
        log lines (e.g. ``"claude returned no response (timeout/empty)"``)

    Returns the prose to mine, or ``None`` when the column is empty.
    Never raises on malformed JSON / unexpected shape — degrades to the
    raw string (which may itself be themable; the timeout-message corpus
    *is* a real theme).
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
        return s or None
    if isinstance(obj, dict):
        dec = obj.get("decision")
        if isinstance(dec, dict):
            r = dec.get("reasoning")
            if isinstance(r, str) and r.strip():
                return r
        r = obj.get("reasoning")
        if isinstance(r, str) and r.strip():
            return r
        return None
    return s or None


def _content_tokens(text: str) -> list[str]:
    """Ordered list of lowercased content tokens — length≥3, non-stopword.

    Preserves order so consecutive-pair (bigram) extraction is meaningful.
    """
    return [
        t for t in _WORD.findall(text.lower())
        if len(t) >= _MIN_TOKEN_LEN and t not in _STOPWORDS
    ]


def build_reasoning_themes(
    decisions: list[dict[str, Any]] | None,
    *,
    top_k: int = 10,
    include_bigrams: bool = True,
) -> dict:
    """Top recurring phrases across a window of recent decision reasonings.

    Args:
        decisions: rows in the shape ``store.recent_decisions`` returns —
            ``{timestamp, action_taken, reasoning, ...}``. Order does not
            affect the leaderboard (themes are aggregate); the first
            occurrence's timestamp is captured as the example anchor.
        top_k: how many phrases to surface, clamped to a sensible band.
        include_bigrams: when True (default), 2-grams compete with 1-grams
            in the same leaderboard, ranked by decisions_mentioning. A
            bigram is more informative than its parts ("super cycle" beats
            "super" + "cycle"), so on ties bigrams are deliberately broken
            in favour of the bigram via the secondary sort.

    Returns a dict with ``state`` in:
      * ``NO_DATA`` — zero rows with extractable reasoning text
      * ``OK`` — leaderboard emitted

    Headline names the top theme + its share of the window. Pure: never
    raises on malformed rows (a row whose reasoning fails to extract is
    counted in ``n_unparseable`` and skipped, not propagated).
    """
    try:
        top_k = int(top_k) if top_k is not None else 10
    except (TypeError, ValueError):
        top_k = 10
    top_k = max(3, min(50, top_k))

    rows = list(decisions or [])
    n_rows = len(rows)
    n_unparseable = 0

    # Per-decision: each unique phrase mentioned at least once. A phrase
    # that appears 30× in one verbose reasoning still counts as **one**
    # mentioning decision — the question is breadth, not loudness.
    phrase_decisions: Counter[str] = Counter()
    phrase_total_mentions: Counter[str] = Counter()
    phrase_first_ts: dict[str, str] = {}
    phrase_example: dict[str, str] = {}
    n_with_text = 0

    for d in rows:
        text = _extract_reasoning_text(d.get("reasoning"))
        if not text:
            n_unparseable += 1
            continue
        toks = _content_tokens(text)
        if not toks:
            n_unparseable += 1
            continue
        n_with_text += 1
        ts = d.get("timestamp") or ""

        # Unique phrases in THIS decision (per-decision tally) + raw count.
        seen_here: set[str] = set()
        local_mentions: Counter[str] = Counter()

        for tok in toks:
            local_mentions[tok] += 1
            seen_here.add(tok)
        if include_bigrams:
            for a, b in zip(toks, toks[1:]):
                bg = f"{a} {b}"
                local_mentions[bg] += 1
                seen_here.add(bg)

        for phrase in seen_here:
            phrase_decisions[phrase] += 1
            phrase_total_mentions[phrase] += local_mentions[phrase]
            if phrase not in phrase_first_ts:
                phrase_first_ts[phrase] = ts
                # Trim the example to a digestible window; if the phrase
                # appears mid-sentence we want the surrounding context.
                phrase_example[phrase] = _example_for(text, phrase)

    if not phrase_decisions:
        return {
            "state": "NO_DATA",
            "n_decisions": n_rows,
            "n_with_reasoning": 0,
            "n_unparseable": n_unparseable,
            "top_k": top_k,
            "include_bigrams": include_bigrams,
            "themes": [],
            "headline": (
                "No reasoning text in window — no themes to surface."
            ),
        }

    # Rank: more decisions mentioning first, then total mentions, then
    # bigrams over unigrams (informativeness tie-break), then alpha for
    # deterministic ordering.
    def _sort_key(item):
        phrase, dec_count = item
        is_bigram = " " in phrase
        return (
            -dec_count,
            -phrase_total_mentions[phrase],
            0 if is_bigram else 1,
            phrase,
        )

    ranked = sorted(phrase_decisions.items(), key=_sort_key)[:top_k]

    themes = [
        {
            "phrase": phrase,
            "decisions_mentioning": dec_count,
            "share_of_decisions": (
                round(dec_count / n_with_text, 3) if n_with_text else 0.0
            ),
            "total_mentions": phrase_total_mentions[phrase],
            "first_seen_ts": phrase_first_ts[phrase],
            "example": phrase_example[phrase],
            "is_bigram": " " in phrase,
        }
        for phrase, dec_count in ranked
    ]

    top = themes[0]
    head = (
        f"Top theme: \"{top['phrase']}\" mentioned in "
        f"{top['decisions_mentioning']}/{n_with_text} decisions "
        f"({top['share_of_decisions']*100:.0f}%)."
    )

    return {
        "state": "OK",
        "n_decisions": n_rows,
        "n_with_reasoning": n_with_text,
        "n_unparseable": n_unparseable,
        "top_k": top_k,
        "include_bigrams": include_bigrams,
        "themes": themes,
        "headline": head,
    }


def _example_for(text: str, phrase: str, span: int = 80) -> str:
    """Locate the phrase in the source text and return a ±``span`` excerpt.

    Case-insensitive find. Falls back to the leading ``2·span`` chars when
    the phrase cannot be located literally (a bigram may bridge a token
    we dropped via stopword filtering and not appear contiguously in the
    raw text). Empty / missing phrase returns the leading excerpt.
    """
    if not text:
        return ""
    if phrase:
        lo = text.lower().find(phrase.lower())
        if lo >= 0:
            start = max(0, lo - span)
            end = min(len(text), lo + len(phrase) + span)
            excerpt = text[start:end].strip()
            prefix = "…" if start > 0 else ""
            suffix = "…" if end < len(text) else ""
            return f"{prefix}{excerpt}{suffix}"
    return text[: 2 * span].strip() + ("…" if len(text) > 2 * span else "")
