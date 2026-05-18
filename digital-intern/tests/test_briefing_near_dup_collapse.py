"""Pins the briefing's second-stage order-independent near-duplicate collapse.

`analysis.claude_analyst._build_payload` runs, in order:
  1. `_filter_quote_widget_noise`
  2. `_collapse_syndicated`            (exact first-8-token prefix signature)
  3. `_dedupe_near_duplicates`         (ml.dedup, Jaccard token-set — NEW)
  4. `_rank_by_decayed_score`

`ml.dedup` is separately unit-tested in `tests/test_dedup.py`; this suite pins
the *integration*: that a reordered / attribution-suffixed copy of one story —
which `_collapse_syndicated` deliberately keeps separate — is collapsed in the
rendered newswire, while a single-token ANTONYM flip is provably NOT merged at
the conservative `BRIEFING_NEAR_DUP_THRESHOLD`, the prepended PORTFOLIO/OPTIONS
snapshot rows survive untouched and pinned first, and the caller's
`source_articles` list/dicts are never mutated (the load-bearing read-side
invariant — heartbeat_worker feeds the same objects onward to the
briefing-label / training path).
"""
import copy

from analysis import claude_analyst
from analysis.claude_analyst import (
    BRIEFING_NEAR_DUP_THRESHOLD,
    _build_payload,
    _collapse_syndicated,
    _dedupe_near_duplicates,
)

_STOCKS = {"macro": [], "equities": []}


def _newswire_titles(payload: str) -> list[str]:
    """Title of every rendered ' N. [score=...] [src] <title>' newswire row."""
    out = []
    for line in payload.splitlines():
        s = line.strip()
        if not s or not s[0].isdigit() or "[score=" not in s:
            continue
        # The render is "{i}. [score=..]{tags} [{source}] {title}"; the title
        # is everything after the LAST "] " on the line.
        out.append(s.rsplit("] ", 1)[-1])
    return out


# ── The conservative threshold is itself load-bearing ───────────────────────
def test_threshold_is_the_conservative_value():
    """0.7 is the documented antonym-flip-safe value. A future loosening to
    0.6 would reopen opposite-direction over-collapse — pin it explicitly so
    that change can't land silently (defense-in-depth with the antonym test)."""
    assert BRIEFING_NEAR_DUP_THRESHOLD == 0.7


# ── The feature is NOT redundant with _collapse_syndicated ──────────────────
def test_reorder_variant_survives_collapse_syndicated_but_not_the_new_stage():
    """A word-reordered copy has a DIFFERENT first-8-token prefix signature,
    so `_collapse_syndicated` keeps it — proving the new Jaccard stage is
    necessary, not duplicative — then the new stage collapses it."""
    arts = [
        {"title": "Apple beats Q2 expectations on strong iPhone sales",
         "source": "rss", "ai_score": 8.0, "link": "u1"},
        {"title": "Q2 expectations beaten by Apple on strong iPhone sales",
         "source": "gdelt", "ai_score": 6.0, "link": "u2"},
    ]
    # Prefix-signature stage does NOT merge the reorder.
    assert len(_collapse_syndicated(arts)) == 2
    # Jaccard stage at the real threshold DOES (J = 0.75 >= 0.7).
    merged = _dedupe_near_duplicates(
        _collapse_syndicated(arts), threshold=BRIEFING_NEAR_DUP_THRESHOLD
    )
    assert len(merged) == 1
    # Highest-ai_score copy is the survivor (matches the displayed score).
    assert merged[0]["ai_score"] == 8.0
    assert merged[0]["title"] == "Apple beats Q2 expectations on strong iPhone sales"


def test_build_payload_collapses_reorder_duplicate_to_one_row():
    arts = [
        {"title": "Apple beats Q2 expectations on strong iPhone sales",
         "source": "rss", "ai_score": 8.0, "link": "u1", "summary": "a"},
        {"title": "Q2 expectations beaten by Apple on strong iPhone sales",
         "source": "gdelt", "ai_score": 6.0, "link": "u2", "summary": "b"},
    ]
    payload = _build_payload(arts, _STOCKS, [])
    titles = _newswire_titles(payload)
    assert "Apple beats Q2 expectations on strong iPhone sales" in titles
    assert "Q2 expectations beaten by Apple on strong iPhone sales" not in titles
    # Exactly one newswire row for that story.
    assert titles.count("Apple beats Q2 expectations on strong iPhone sales") == 1


# ── The antonym-flip safety property (the over-collapse guard) ──────────────
def test_single_token_antonym_flip_is_NOT_merged():
    """"Fed raises rates 25bp today" vs "Fed cuts rates 25bp today" share 4 of
    6 union tokens (J = 0.667 < 0.7): opposite-direction stories must BOTH
    reach the analyst. This is the property that makes 0.7, not 0.6, correct."""
    arts = [
        {"title": "Fed raises rates 25bp today", "source": "rss",
         "ai_score": 9.0, "link": "u1", "summary": ""},
        {"title": "Fed cuts rates 25bp today", "source": "wsj",
         "ai_score": 9.0, "link": "u2", "summary": ""},
    ]
    kept = _dedupe_near_duplicates(
        _collapse_syndicated(arts), threshold=BRIEFING_NEAR_DUP_THRESHOLD
    )
    assert len(kept) == 2
    payload = _build_payload(arts, _STOCKS, [])
    titles = _newswire_titles(payload)
    assert "Fed raises rates 25bp today" in titles
    assert "Fed cuts rates 25bp today" in titles


# ── Prepended snapshot rows survive untouched and stay first ────────────────
def test_portfolio_and_options_snapshots_pass_through_and_stay_first():
    arts = [
        {"title": "PORTFOLIO P&L SNAPSHOT", "source": "portfolio",
         "summary": "MU -6.6% NVDA -4%", "ai_score": 10},
        {"title": "OPTIONS SNAPSHOT", "source": "options_monitor",
         "summary": "DRAM C59", "ai_score": 10},
        {"title": "Apple beats Q2 expectations on strong iPhone sales",
         "source": "rss", "ai_score": 8.0, "link": "u1", "summary": "a"},
        {"title": "Q2 expectations beaten by Apple on strong iPhone sales",
         "source": "gdelt", "ai_score": 6.0, "link": "u2", "summary": "b"},
    ]
    payload = _build_payload(arts, _STOCKS, [])
    titles = _newswire_titles(payload)
    # Both snapshots present (low Jaccard with each other and with articles —
    # never collapsed) and pinned ahead of the real article.
    assert titles[0] == "PORTFOLIO P&L SNAPSHOT"
    assert titles[1] == "OPTIONS SNAPSHOT"
    assert "Apple beats Q2 expectations on strong iPhone sales" in titles[2:]
    # The reorder dup is still collapsed even with snapshots present.
    assert "Q2 expectations beaten by Apple on strong iPhone sales" not in titles


# ── Read-side invariant: never mutate the caller's source_articles ──────────
def test_build_payload_does_not_mutate_caller_articles():
    """heartbeat_worker passes the SAME list to the briefing-label / training
    path after analyze(); a mutation here would corrupt the labels. Mirrors
    test_briefing_book_heat / _collapse_syndicated's no-mutation contract."""
    arts = [
        {"title": "Apple beats Q2 expectations on strong iPhone sales",
         "source": "rss", "ai_score": 8.0, "link": "u1", "summary": "a"},
        {"title": "Q2 expectations beaten by Apple on strong iPhone sales",
         "source": "gdelt", "ai_score": 6.0, "link": "u2", "summary": "b"},
        {"title": "Totally unrelated macro headline about jobs data",
         "source": "rss", "ai_score": 7.0, "link": "u3", "summary": "c"},
    ]
    before = copy.deepcopy(arts)
    _build_payload(arts, _STOCKS, [])
    assert arts == before  # list length, order, and every dict value unchanged
    # No score-bearing key was added or rewritten on any row.
    for a in arts:
        assert set(a.keys()) <= {"title", "source", "ai_score", "link", "summary"}


def test_dedupe_returns_original_objects_no_copy():
    """The integration relies on dedupe_articles returning the ORIGINAL dicts
    (so downstream tags/decay see the same objects) — identity, not a copy."""
    a = {"title": "Totally unique headline number one here", "ai_score": 5.0,
         "link": "u1"}
    b = {"title": "A completely different unrelated second headline", "ai_score": 5.0,
         "link": "u2"}
    out = _dedupe_near_duplicates([a, b], threshold=BRIEFING_NEAR_DUP_THRESHOLD)
    assert len(out) == 2
    assert out[0] is a and out[1] is b


# ── Unrelated stories are never collapsed ───────────────────────────────────
def test_distinct_stories_all_survive():
    arts = [
        {"title": "NVDA earnings beat sends chip stocks higher",
         "source": "rss", "ai_score": 9.0, "link": "u1", "summary": ""},
        {"title": "Oil prices tumble on OPEC supply surprise",
         "source": "rss", "ai_score": 8.0, "link": "u2", "summary": ""},
        {"title": "Fed minutes signal patience on rate path",
         "source": "rss", "ai_score": 7.0, "link": "u3", "summary": ""},
    ]
    payload = _build_payload(arts, _STOCKS, [])
    titles = _newswire_titles(payload)
    for a in arts:
        assert a["title"] in titles
