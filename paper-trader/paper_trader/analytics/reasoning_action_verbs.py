"""Reasoning-action verb consistency — does Opus's *natural language* agree
with the structured ``action`` field on the same decision?

Every existing reasoning surface measures a different thing:

* ``/api/reasoning-coherence`` — pairwise Jaccard between consecutive HOLD
  reasonings (a stability metric across decisions).
* ``/api/reasoning-themes`` — bag-of-words across decisions (what topics
  Opus is talking about).
* ``/api/decision-confidence`` — aggregates the self-rated ``confidence``
  scalar.
* ``/api/decision-forensics`` — reads the most recent NO_DECISION's failure
  mode.

None grade the **single-decision internal-consistency** question: when the
structured ``action`` is ``HOLD``, does the natural-language reasoning *also
read* as a HOLD, or does it actually verbalise a BUY/SELL intent that the
structured field then quietly contradicts? That is a real LLM failure mode
(an action-verb lapse) that produces a decision-row that *looks fine to a
JSON-parsing reviewer* but reads alarming to a human operator.

The builder counts action verbs inside ``decision.reasoning`` text and
classifies the dominant leaning per decision:

* ``HOLDING`` — wait/hold/patience verbs dominate; or no directional verbs.
* ``BULLISH`` — buy/add/scale-in verbs dominate.
* ``BEARISH`` — sell/trim/exit verbs dominate.
* ``MIXED`` — directional verbs cancel out (e.g. "add or trim depending on…").

Then compares against the structured action:

* action=HOLD + leaning=BULLISH → ``BULLISH_INSIDE_HOLD`` (operator flag)
* action=HOLD + leaning=BEARISH → ``BEARISH_INSIDE_HOLD`` (operator flag)
* action=BUY + leaning=BEARISH → ``BEARISH_INSIDE_BUY`` (alarming)
* action=SELL + leaning=BULLISH → ``BULLISH_INSIDE_SELL`` (alarming)
* action=NO_DECISION + leaning≠HOLDING → ``DIRECTION_INSIDE_NO_DECISION``
* everything else → ``CONSISTENT``

Negation handling: the cue is rejected if any negation token sits within
``_NEG_WINDOW=3`` whitespace-separated words *before* the cue start. So
"would not add" / "shouldn't trim" / "rather than buy" do NOT contribute a
directional vote.

Conditional/temporal hedge handling: the cue is rejected when the
preceding ``_HEDGE_WINDOW=3`` words contain a hedge token (``if``,
``unless``, ``until``, ``before``, ``after``, ``once``, ``when``). The
common HOLD pattern "wait for X before adding or trimming" therefore
yields zero directional votes — neither "adding" nor "trimming" pull the
leaning toward BUY/SELL, because both are explicitly conditional on a
future event.

Pure / no I/O. Composes ``store.recent_decisions`` rows; never raises on
malformed input (the ``_safe`` discipline shared with ``decision_
confidence`` / ``reasoning_coherence`` / ``reasoning_themes``).

Observational only — never gates Opus, never injected into the prompt,
adds no caps (AGENTS.md invariants #2/#12 — the ``decision_confidence`` /
``reasoning_coherence`` / ``shadow_vs_claude`` precedent).
"""
from __future__ import annotations

import json
import re
from collections import Counter
from typing import Any

# Bullish (BUY/ADD) cue verbs / phrases. Anchored to a word boundary so
# "additional" does not match "add". Patterns are intentionally narrow —
# we want PRESENT-action verbiage, not subjunctive opinion.
_BULLISH_CUES = (
    r"\badd(?:ing)?\b",
    r"\bbuy(?:ing)?\b",
    r"\baccumulat(?:e|ing)\b",
    r"\bscal(?:e|ing)\s+into\b",
    r"\brotat(?:e|ing)\s+into\b",
    r"\bbuild(?:ing)?\s+(?:a\s+)?(?:position|stake)\b",
    r"\binitiat(?:e|ing)\s+(?:a\s+)?(?:long|position)\b",
    r"\bincreas(?:e|ing)\s+(?:exposure|the\s+position|the\s+stake)\b",
    r"\baveraging\s+down\b",
    r"\bpyramid(?:ing)?\s+up\b",
)

# Bearish (TRIM/SELL/EXIT) cue verbs / phrases.
_BEARISH_CUES = (
    r"\bsell(?:ing)?\b",
    r"\btrim(?:ming)?\b",
    r"\bexit(?:ing|ed)?\b",
    r"\bclos(?:e|ing)\s+(?:the\s+)?position\b",
    r"\bcut(?:ting)?\s+(?:the\s+)?(?:position|exposure|stake|loser)\b",
    r"\blighten(?:ing)?(?:\s+up)?\b",
    r"\breduc(?:e|ing)\s+(?:exposure|the\s+position|the\s+stake)\b",
    r"\bscal(?:e|ing)\s+out\b",
    r"\btak(?:e|ing)\s+(?:profits?|some\s+off)\b",
    r"\brais(?:e|ing)\s+cash\b",
)

# Hold / patience verbs. We do NOT flag a HOLD action with hold cues —
# we use this count to verify the leaning is HOLDING vs MIXED.
_HOLD_CUES = (
    r"\bhold(?:ing)?\b",
    r"\bwait(?:ing)?\b",
    r"\bsit(?:ting)?\s+(?:tight|on)\b",
    r"\bstay(?:ing)?\s+(?:put|the\s+course)\b",
    r"\bpatien(?:ce|t)\b",
    r"\bstand(?:ing)?\s+(?:pat|by|aside)\b",
    r"\blet(?:ting)?\s+it\s+(?:run|ride|play\s+out)\b",
    r"\bdefer(?:ring)?\b",
    r"\bobserv(?:e|ing)\b",
    r"\bmonitor(?:ing)?\b",
)

# Negation tokens. A cue preceded by any of these within _NEG_WINDOW
# words is dropped (the model isn't asking for that action, it is
# negating it).
_NEGATIONS = {
    "not", "never", "no", "nor", "n't", "without", "against",
    "rather", "instead", "avoid", "avoided", "avoiding",
    "wouldn't", "won't", "don't", "doesn't", "shouldn't",
    "didn't", "can't", "cannot",
}

# Hedge / conditional tokens. A cue preceded by any of these within
# _HEDGE_WINDOW words is conditional ("would add IF earnings beat") and
# does not contribute to the present-tense leaning.
_HEDGES = {
    "if", "unless", "until", "before", "after", "once",
    "when", "whenever", "should", "would", "could", "might",
    "may", "perhaps", "maybe", "possibly", "consider", "considering",
}

_NEG_WINDOW = 3
_HEDGE_WINDOW = 3

# Combined compiled patterns. We compile each cue separately so the
# matcher can name the specific cue in the output.
_BULL_PATS = [re.compile(p, re.IGNORECASE) for p in _BULLISH_CUES]
_BEAR_PATS = [re.compile(p, re.IGNORECASE) for p in _BEARISH_CUES]
_HOLD_PATS = [re.compile(p, re.IGNORECASE) for p in _HOLD_CUES]

# Verdict ladder for state.
STATE_OK_THRESHOLD = 0.05      # < 5 % mismatch ⇒ CLEAN
STATE_MILD_THRESHOLD = 0.15    # 5-15 % ⇒ MILD
STATE_NOTABLE_THRESHOLD = 0.30  # 15-30 % ⇒ NOTABLE; ≥30 % ⇒ ALARMING

MIN_SAMPLES_FOR_VERDICT = 10
SNIPPET_CHARS = 140


def _action_verb(action_taken: str | None) -> str:
    if not action_taken:
        return "UNKNOWN"
    s = str(action_taken).strip()
    if not s:
        return "UNKNOWN"
    return s.split(None, 1)[0].upper()


def _extract_inner_reasoning(blob: str | None) -> tuple[str, str | None, str | None]:
    """Return ``(reasoning_text, inner_action, inner_ticker)``.

    The canonical decision envelope is
    ``{"decision": {"action": ..., "ticker": ..., "reasoning": ...}, ...}``.
    Falls back to the top-level ``reasoning`` / ``action`` keys when the
    inner envelope is missing (older rows). Returns an empty string when
    the blob is unparseable / non-JSON / not a dict — never raises.
    """
    if not blob:
        return "", None, None
    s = str(blob).strip()
    if not s:
        return "", None, None
    # parse_failed / retry_failed prefixes — strategy.py persists the raw
    # Opus response after this tag for forensic readability. The text after
    # the tag is the raw response, often free-form prose — try JSON, then
    # fall back to using the prose as the reasoning text.
    for prefix in ("parse_failed:", "retry_failed:"):
        if s.startswith(prefix):
            s = s[len(prefix):].strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s).strip()
    try:
        obj = json.loads(s)
    except (ValueError, TypeError):
        # Not JSON — use the raw text as the reasoning. inner_action /
        # inner_ticker stay None; the caller relies on the outer
        # ``action_taken`` verb for the action.
        return s, None, None
    if not isinstance(obj, dict):
        return "", None, None
    inner_action = None
    inner_ticker = None
    text = ""
    dec = obj.get("decision")
    if isinstance(dec, dict):
        ia = dec.get("action")
        if isinstance(ia, str):
            inner_action = ia.strip().upper() or None
        it = dec.get("ticker")
        if isinstance(it, str):
            inner_ticker = it.strip().upper() or None
        t = dec.get("reasoning")
        if isinstance(t, str):
            text = t
    if not text:
        t = obj.get("reasoning")
        if isinstance(t, str):
            text = t
    if not text:
        # Some rows store the prose under ``detail`` only.
        t = obj.get("detail")
        if isinstance(t, str):
            text = t
    return text, inner_action, inner_ticker


_WORD_RE = re.compile(r"[A-Za-z'’]+")


def _tokenize(text: str) -> list[tuple[int, str]]:
    """Return list of ``(char_offset, lowercased_word)`` for ``text``.

    Used to walk backwards from a cue start to check negation / hedge
    windows. The char offset of each word lets us re-anchor against the
    cue's ``re.Match.start()``.
    """
    return [(m.start(), m.group(0).lower()) for m in _WORD_RE.finditer(text)]


def _preceding_words(tokens: list[tuple[int, str]], cue_start: int,
                     n: int) -> list[str]:
    """The last ``n`` words ending strictly before ``cue_start``."""
    out = []
    for off, w in reversed(tokens):
        if off < cue_start:
            out.append(w)
            if len(out) >= n:
                break
    return out


def _is_rejected(text: str, tokens: list[tuple[int, str]], match) -> bool:
    """True iff a cue is negated or hedged (window-local context)."""
    start = match.start()
    neg_window = _preceding_words(tokens, start, _NEG_WINDOW)
    if any(w in _NEGATIONS for w in neg_window):
        return True
    # n't is a suffix on the prior word ("shouldn't", "won't"); the suffix
    # case is already in _NEGATIONS as the whole contraction. The bare
    # "n't" token form only appears if a tokenizer split it, which our
    # _WORD_RE does not — but we keep the membership test for safety.
    hedge_window = _preceding_words(tokens, start, _HEDGE_WINDOW)
    if any(w in _HEDGES for w in hedge_window):
        return True
    return False


def _scan_cues(text: str, patterns: list[re.Pattern]) -> list[tuple[str, int]]:
    """Return non-rejected cue matches as ``[(matched_text, char_offset)]``."""
    if not text:
        return []
    tokens = _tokenize(text)
    out = []
    for pat in patterns:
        for m in pat.finditer(text):
            if _is_rejected(text, tokens, m):
                continue
            out.append((m.group(0), m.start()))
    return out


def _snippet(text: str, anchor_offset: int, span: int = SNIPPET_CHARS) -> str:
    """Centred snippet around ``anchor_offset``, collapsed whitespace."""
    if not text:
        return ""
    lo = max(0, anchor_offset - span // 2)
    hi = min(len(text), anchor_offset + span // 2)
    s = text[lo:hi]
    s = re.sub(r"\s+", " ", s).strip()
    if lo > 0:
        s = "…" + s
    if hi < len(text):
        s = s + "…"
    return s


def _classify_leaning(n_bull: int, n_bear: int, n_hold: int) -> str:
    """Pick the dominant leaning from cue counts.

    * Both directional sides zero → ``HOLDING`` (no directional intent;
      includes the "zero cues at all" case).
    * Bull > Bear → ``BULLISH``.
    * Bear > Bull → ``BEARISH``.
    * Bull == Bear ≠ 0 → ``MIXED``.
    """
    if n_bull == 0 and n_bear == 0:
        return "HOLDING"
    if n_bull > n_bear:
        return "BULLISH"
    if n_bear > n_bull:
        return "BEARISH"
    return "MIXED"


def _verdict(action_verb: str, leaning: str) -> str:
    """Map (structured action, natural-language leaning) → verdict tag."""
    # Normalise option verbs to their direction: BUY_CALL/SELL_PUT are
    # bullish; SELL_CALL/BUY_PUT are bearish; the inner-reasoning leaning
    # is compared against this normalised direction.
    direction = action_verb
    if action_verb in ("BUY_CALL", "SELL_PUT"):
        direction = "BUY"
    elif action_verb in ("SELL_CALL", "BUY_PUT"):
        direction = "SELL"

    if direction == "HOLD":
        if leaning == "BULLISH":
            return "BULLISH_INSIDE_HOLD"
        if leaning == "BEARISH":
            return "BEARISH_INSIDE_HOLD"
        return "CONSISTENT"
    if direction == "BUY":
        if leaning == "BEARISH":
            return "BEARISH_INSIDE_BUY"
        return "CONSISTENT"
    if direction == "SELL":
        if leaning == "BULLISH":
            return "BULLISH_INSIDE_SELL"
        return "CONSISTENT"
    if direction == "NO_DECISION":
        if leaning in ("BULLISH", "BEARISH"):
            return "DIRECTION_INSIDE_NO_DECISION"
        return "CONSISTENT"
    # BLOCKED / REBALANCE / UNKNOWN — leaning isn't a fair mismatch test.
    return "CONSISTENT"


def _state(mismatch_rate: float, n_parsed: int) -> str:
    if n_parsed < MIN_SAMPLES_FOR_VERDICT:
        return "INSUFFICIENT"
    if mismatch_rate < STATE_OK_THRESHOLD:
        return "CLEAN"
    if mismatch_rate < STATE_MILD_THRESHOLD:
        return "MILD"
    if mismatch_rate < STATE_NOTABLE_THRESHOLD:
        return "NOTABLE"
    return "ALARMING"


def build_reasoning_action_verbs(
    decisions: list[dict[str, Any]] | None,
    *,
    snippet_chars: int = SNIPPET_CHARS,
) -> dict:
    """Grade single-decision internal consistency between structured action
    and natural-language reasoning.

    Args:
        decisions: rows from ``store.recent_decisions`` (any order).

    Returns a dict with:
      * ``state`` ∈ ``INSUFFICIENT`` / ``CLEAN`` / ``MILD`` /
        ``NOTABLE`` / ``ALARMING``
      * ``n_decisions``, ``n_parsed``, ``n_unparseable``,
        ``n_mismatched``, ``mismatch_rate_pct``
      * ``by_verdict`` — Counter of verdict tags
      * ``by_action`` — per-action {n, n_mismatched, mismatch_rate_pct}
      * ``mismatches`` — flagged-row list, newest-first by ts
      * ``headline`` — one-line operator summary
    """
    rows = list(decisions or [])
    n_rows = len(rows)

    n_unparseable = 0
    n_parsed = 0
    per_verdict: Counter = Counter()
    per_action_total: Counter = Counter()
    per_action_mismatch: Counter = Counter()
    mismatches: list[dict] = []

    for d in rows:
        if not isinstance(d, dict):
            n_unparseable += 1
            continue
        outer_action = _action_verb(d.get("action_taken"))
        text, inner_action, inner_ticker = _extract_inner_reasoning(
            d.get("reasoning"))
        if not text:
            n_unparseable += 1
            continue
        n_parsed += 1

        # Cue scan.
        bull_hits = _scan_cues(text, _BULL_PATS)
        bear_hits = _scan_cues(text, _BEAR_PATS)
        hold_hits = _scan_cues(text, _HOLD_PATS)

        leaning = _classify_leaning(len(bull_hits), len(bear_hits), len(hold_hits))
        # Prefer the inner-JSON action verb when present; the outer
        # action_taken can carry FILL/BLOCKED appendices that aren't the
        # verb we're comparing leaning against.
        action_for_verdict = inner_action or outer_action
        # Option-verb compounds inside ``action_taken`` ("BUY_CALL NVDA")
        # parse out as ``BUY_CALL`` by _action_verb; the inner JSON may
        # store just ``BUY_CALL`` too. The verdict helper handles both.
        verdict = _verdict(action_for_verdict, leaning)

        per_action_total[action_for_verdict] += 1
        per_verdict[verdict] += 1
        if verdict != "CONSISTENT":
            per_action_mismatch[action_for_verdict] += 1
            # Choose the snippet anchor: the first cue from the dominant
            # leaning, falling back to the first directional cue overall.
            anchor = 0
            if leaning == "BULLISH" and bull_hits:
                anchor = bull_hits[0][1]
            elif leaning == "BEARISH" and bear_hits:
                anchor = bear_hits[0][1]
            elif bull_hits:
                anchor = bull_hits[0][1]
            elif bear_hits:
                anchor = bear_hits[0][1]
            mismatches.append({
                "id": d.get("id"),
                "ts": d.get("timestamp"),
                "action": action_for_verdict,
                "outer_action_taken": d.get("action_taken"),
                "ticker": inner_ticker,
                "leaning": leaning,
                "n_bullish_cues": len(bull_hits),
                "n_bearish_cues": len(bear_hits),
                "n_hold_cues": len(hold_hits),
                "cues_bullish": [c for c, _ in bull_hits][:5],
                "cues_bearish": [c for c, _ in bear_hits][:5],
                "verdict": verdict,
                "snippet": _snippet(text, anchor, snippet_chars),
            })

    # newest-first sort on ts (strings are lexicographically sortable for
    # ISO-8601 with a trailing "+00:00" / "Z"). Rows without a ts sink.
    def _sort_key(row):
        ts = row.get("ts") or ""
        return ts
    mismatches.sort(key=_sort_key, reverse=True)

    n_mismatched = sum(1 for v, c in per_verdict.items()
                       if v != "CONSISTENT" for _ in range(c))
    # The above generator double-counts; use sum(c for ... ) instead.
    n_mismatched = sum(c for v, c in per_verdict.items() if v != "CONSISTENT")
    rate = (n_mismatched / n_parsed) if n_parsed else 0.0
    state = _state(rate, n_parsed)

    # Per-action breakdown — only actions we actually saw.
    by_action = {}
    for verb, total in per_action_total.items():
        m = per_action_mismatch.get(verb, 0)
        by_action[verb] = {
            "n": total,
            "n_mismatched": m,
            "mismatch_rate_pct": round((m / total) * 100.0, 2) if total else 0.0,
        }

    by_verdict = dict(per_verdict)

    # Headline composition.
    if n_parsed == 0:
        headline = (
            "No parseable decision reasoning in window — "
            "action-verb consistency cannot be assessed."
        )
    elif state == "INSUFFICIENT":
        headline = (
            f"Only {n_parsed} parseable decision(s) — need ≥"
            f"{MIN_SAMPLES_FOR_VERDICT} for a verdict (so far "
            f"{n_mismatched} mismatch(es))."
        )
    elif state == "CLEAN":
        headline = (
            f"CLEAN: {n_mismatched}/{n_parsed} decision(s) "
            f"({rate * 100:.1f}%) carry an action-verb mismatch — "
            "structured action matches the natural-language leaning."
        )
    else:
        # Find the most common mismatch verdict for the headline.
        top_verdict = max(
            ((v, c) for v, c in per_verdict.items() if v != "CONSISTENT"),
            key=lambda kv: kv[1],
            default=(None, 0),
        )[0]
        headline = (
            f"{state}: {n_mismatched}/{n_parsed} decision(s) "
            f"({rate * 100:.1f}%) carry an action-verb mismatch"
            + (f" — top mode {top_verdict}." if top_verdict
               else " — no dominant mode.")
        )

    return {
        "state": state,
        "n_decisions": n_rows,
        "n_parsed": n_parsed,
        "n_unparseable": n_unparseable,
        "n_mismatched": n_mismatched,
        "mismatch_rate_pct": round(rate * 100.0, 2),
        "by_verdict": by_verdict,
        "by_action": by_action,
        "mismatches": mismatches,
        "headline": headline,
        "min_samples_for_verdict": MIN_SAMPLES_FOR_VERDICT,
        "thresholds_pct": {
            "clean_under": round(STATE_OK_THRESHOLD * 100, 1),
            "mild_under": round(STATE_MILD_THRESHOLD * 100, 1),
            "notable_under": round(STATE_NOTABLE_THRESHOLD * 100, 1),
        },
    }
