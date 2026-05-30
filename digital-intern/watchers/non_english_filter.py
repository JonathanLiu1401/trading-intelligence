"""Non-English title pre-floor — defense-in-depth gate for the noise pre-floor chain.

PR Newswire / GlobeNewswire / GDELT fire multi-language copies of the same
corporate announcement (English + Spanish + French + Portuguese + German).
The ML urgency head over-scores the foreign-language copies to
``ml_score >= 9`` because the titles are dense with product / company
tokens the model has learned correlate with relevance — but it has no
English-language language-ID prior, so a Spanish "Arasan Chip Systems
anuncia la primera solución IP Sureboot™ Total de 16 bits xSPI + PSRAM"
looks identical to a real wire from the model's perspective.

Live evidence (2026-05-30, 30d articles.db urgency>=2 audit): 5 of 8 PR
Newswire BREAKING alerts on the consumer's Discord channel were non-English
copies of the same announcement — Spanish ("anuncia"), German ("kündigt"),
French ("annonce", "acquiert", "accroître"), Portuguese ("gastará bilhões").
Same retrospective-noise class as
``alert_agent._looks_like_quote_widget`` / ``_looks_like_recap_template`` /
``_looks_like_stocktwits_chatter``: pure title-fingerprint discriminator,
defense-in-depth at the ML / LLM scoring pre-floor stage.

Single source of truth — called from
``storage.article_store.prefloor_pseudo_articles`` (the daemon's ML-path
pre-floor) so non-English noise gets ``ml_score=0.01``, ``urgency=0``,
``score_source='ml'`` before it can ever reach urgency=1 — the same code
path the existing three siblings use.

Pure read-side: no DB write, no ai_score / ml_score / score_source /
urgency mutation in THIS module (the caller does the update via the
existing ``update_ml_scores_batch`` path). Backtest rows are already
filtered by the caller's upstream live-only clause. All four load-bearing
invariants intact by construction.
"""
from __future__ import annotations

import re


# Two-signal AND keeps precision: an unambiguously non-English stopword
# AND a diacritic somewhere in the title. A real English headline that
# happens to mention an accented proper noun ("São Paulo", "Volkswagen
# Münster", "Beyoncé tour") will not match any stopword and survives.
#
# Every entry below either CONTAINS an accent in the matched form (so
# case-insensitive matching cannot collapse it onto an English word —
# "è" never collapses to "e" under casefold), or is unambiguously
# French/Spanish/German with no English homograph. The English-language
# false-positive risk on the must-survive headline corpus is zero.
_NON_ENGLISH_STOPWORD_RE = re.compile(
    r"\b(?:"
    # Spanish — accent-anchored or unambiguous verbs
    r"anunci[aoó]|solución|según|también|cápsula|asimismo|"
    r"primera\s+solución|"
    # Portuguese — accent-anchored
    r"bilh(?:ão|ões)|gastar[áa]|tamb[ée]m|atrav[ée]s|"
    # French — accent-anchored or French-unique verbs
    r"premi[èe]re|accro[îi]tre|capacit[ée]|acquiert|annonce\s+la|"
    # German — umlaut-anchored
    r"k[üu]ndigt|gesch[äa]ftsjahr"
    r")\b",
    re.IGNORECASE | re.UNICODE,
)
# Latin-extended diacritics — the second signal. A title must contain at
# least one of these (in any token) to be flagged. Brand marks ™ ® © °
# and curly quotes are deliberately excluded since they appear in
# English-language headlines.
_NON_ENGLISH_DIACRITIC_RE = re.compile(
    r"[áàâãäåçéèêëíìîïñóòôõöúùûüýÿß"
    r"ÁÀÂÃÄÅÇÉÈÊËÍÌÎÏÑÓÒÔÕÖÚÙÛÜÝŸ]"
)
# Don't fire on ultra-short titles — false positives possible on a 4-char
# "kündigt" wire summary; require some headline-shaped length so the
# diacritic + stopword co-occurrence is statistically meaningful.
_MIN_TITLE_LEN = 12


def looks_non_english(art: dict) -> bool:
    """True for a title in Spanish/Portuguese/French/German.

    Two-signal AND: an accent-anchored or non-English-unique stopword
    AND a Latin-extended diacritic anywhere in the title. Pure,
    side-effect-free; reads only ``art['title']`` via ``.get()``."""
    title = (art.get("title") or "").strip()
    if len(title) < _MIN_TITLE_LEN:
        return False
    if not _NON_ENGLISH_STOPWORD_RE.search(title):
        return False
    if not _NON_ENGLISH_DIACRITIC_RE.search(title):
        return False
    return True


def filter_non_english_noise(
    arts: list[dict],
) -> tuple[list[dict], list[dict]]:
    """Partition rows into ``(kept, suppressed)`` where ``suppressed`` is
    every row whose title looks non-English. Mirrors
    ``alert_agent._filter_quote_widget_noise`` / ``_filter_recap_template_noise``
    / ``_filter_stocktwits_chatter`` so the pre-floor surfaces behave
    identically. Pure — no DB / IO."""
    kept: list[dict] = []
    suppressed: list[dict] = []
    for a in arts:
        (suppressed if looks_non_english(a) else kept).append(a)
    return kept, suppressed
