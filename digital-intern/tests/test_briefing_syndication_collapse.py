"""Briefing newswire syndication collapse — analysis/claude_analyst.

A breaking wire item is carried within minutes by GDELT, Reuters, Yahoo, RSS
and the scrapers; each lands as its own high-scoring row. The alert path has
``watchers.alert_dedup`` and the store has a per-publisher cap, but the SAME
wire headline arriving under DIFFERENT domain keys was never collapsed in the
heartbeat digest Opus reads — the analyst's #1 noise complaint applied to the
one path that lacked dedup.

``_collapse_syndicated`` fixes that, reusing the single well-tested
``alert_dedup._signature`` (no signature drift). These pin the contract with
specific-value asserts (not "no crash"):

  * verbatim + wire-marker + attribution variants collapse to ONE survivor;
  * the highest-score copy represents the cluster; ties keep the earlier one;
  * ``_corroboration`` counts every copy; the rendered row shows
    ``[syndicated xN]`` for N>1 and nothing for N==1;
  * distinct headlines are NEVER collapsed (guards the cap-60 contract);
  * untitled / snapshot rows are never merged and keep leading position;
  * the caller's list is not mutated (load-bearing: heartbeat_worker feeds
    that exact list to the briefing-label / training path — this function must
    only ever reshape the text Opus reads).
"""
from __future__ import annotations

import copy

from analysis import claude_analyst
from analysis.claude_analyst import _collapse_syndicated


def test_verbatim_and_wire_variants_collapse_to_one():
    arts = [
        {"title": "Micron shares surge after Q3 earnings blowout",
         "source": "GDELT/reuters.com", "ai_score": 7.0, "summary": "a"},
        {"title": "Micron shares surge after Q3 earnings blowout",
         "source": "scraped/finance.yahoo.com", "ai_score": 9.0, "summary": "b"},
        {"title": "UPDATE 2-Micron shares surge after Q3 earnings blowout",
         "source": "rss", "ai_score": 6.0, "summary": "c"},
        {"title": "Micron shares surge after Q3 earnings blowout - Reuters",
         "source": "google_news", "ai_score": 5.0, "summary": "d"},
    ]
    out = _collapse_syndicated(arts)
    assert len(out) == 1, f"4 syndicated copies must collapse to 1, got {len(out)}"
    rep = out[0]
    # Highest ai_score (9.0, the yahoo copy) represents the cluster.
    assert rep["ai_score"] == 9.0
    assert rep["source"] == "scraped/finance.yahoo.com"
    assert rep["_corroboration"] == 4


def test_distinct_headlines_never_collapse():
    """Regression guard for the cap-60 contract: N distinct titles ⇒ N rows."""
    arts = [
        {"title": f"distinct headline number {i} about chips",
         "source": "rss", "ai_score": 7.0, "summary": ""}
        for i in range(12)
    ]
    out = _collapse_syndicated(arts)
    assert len(out) == 12
    assert all(a["_corroboration"] == 1 for a in out)


def test_order_preserved_and_tie_keeps_earlier():
    arts = [
        {"title": "Fed holds rates steady amid inflation concerns",
         "source": "rss", "ai_score": 8.0, "summary": "first"},
        {"title": "Nvidia unveils next-gen Blackwell accelerator lineup",
         "source": "rss", "ai_score": 9.0, "summary": ""},
        {"title": "Fed holds rates steady amid inflation concerns",
         "source": "gdelt", "ai_score": 8.0, "summary": "second"},  # tie
    ]
    out = _collapse_syndicated(arts)
    assert [a["title"][:9] for a in out] == ["Fed holds", "Nvidia un"]
    # Tie on score (8.0 == 8.0) ⇒ the earlier copy ("first") stays the rep.
    assert out[0]["summary"] == "first"
    assert out[0]["_corroboration"] == 2


def test_untitled_rows_never_merged_and_lead():
    """Prepended snapshot rows (and any titleless item) must each survive and
    keep their leading order — identical policy to dedupe_urgent."""
    arts = [
        {"title": "PORTFOLIO P&L SNAPSHOT", "source": "portfolio",
         "ai_score": 10, "summary": "pnl"},
        {"title": "OPTIONS SNAPSHOT", "source": "options_monitor",
         "ai_score": 10, "summary": "opts"},
        {"title": "", "source": "rss", "ai_score": 5, "summary": "blank"},
        {"title": "", "source": "gdelt", "ai_score": 5, "summary": "blank2"},
        {"title": "Real wire story about semiconductor exports today",
         "source": "rss", "ai_score": 7, "summary": ""},
    ]
    out = _collapse_syndicated(arts)
    # 2 snapshots + 2 distinct empty-title rows + 1 real = 5 (none merged).
    assert len(out) == 5
    assert out[0]["title"] == "PORTFOLIO P&L SNAPSHOT"
    assert out[1]["title"] == "OPTIONS SNAPSHOT"
    assert all(a["_corroboration"] == 1 for a in out)


def test_caller_list_not_mutated():
    arts = [
        {"title": "Micron shares surge after Q3 earnings blowout",
         "source": "rss", "ai_score": 7.0, "summary": "x"},
        {"title": "Micron shares surge after Q3 earnings blowout",
         "source": "gdelt", "ai_score": 9.0, "summary": "y"},
    ]
    snapshot = copy.deepcopy(arts)
    _collapse_syndicated(arts)
    assert arts == snapshot, (
        "input list/dicts were mutated — heartbeat_worker feeds this exact "
        "list to the briefing-label/training path; mutation would leak"
    )


def test_payload_renders_syndication_tag_only_when_corroborated():
    articles = [
        {"title": "Chip export ban widened to ten more China firms",
         "source": "GDELT/reuters.com", "ai_score": 9.0, "summary": "a"},
        {"title": "Chip export ban widened to ten more China firms",
         "source": "scraped/finance.yahoo.com", "ai_score": 8.0, "summary": "b"},
        {"title": "Chip export ban widened to ten more China firms",
         "source": "rss", "ai_score": 7.0, "summary": "c"},
        {"title": "Lone single-sourced analyst note on memory pricing",
         "source": "substack", "ai_score": 6.0, "summary": "d"},
    ]
    payload = claude_analyst._build_payload(
        articles, {"macro": [], "equities": []}, []
    )
    # The 3 syndicated copies collapse into one tagged row...
    assert "[syndicated x3]" in payload
    # ...rendered exactly once (the digest is de-noised, not repeated).
    assert payload.count("Chip export ban widened to ten more China firms") == 1
    # The lone item carries NO tag.
    assert "[syndicated x" not in payload.split(
        "Lone single-sourced analyst note"
    )[0].rsplit("\n", 1)[-1]
    # NEWSWIRE now lists 2 distinct stories, not 4 near-dupes.
    assert "\n 1. " in payload and "\n 2. " in payload
    assert "\n 3. " not in payload


def test_cap_60_contract_still_holds_with_distinct_titles():
    """65 distinct headlines ⇒ no collapse ⇒ line 60 present, 61 absent
    (the existing test_claude_analyst regression, re-pinned post-feature).

    Titles carry a per-row `alpha{i}`/`topic{i}` token: the original
    `f"...headline {i}..."` distinguisher was a bare digit, a len-1 token
    dropped by ml.dedup's `_MIN_TOKEN_LEN=2`, so every "distinct" title
    normalized to the SAME token set and the order-independent near-dup
    stage correctly collapsed them — a latent fixture defect, not a feature
    bug. Genuinely-distinct tokens (J~0.5 < the 0.7 threshold) restore the
    test's stated intent; the cap-60 assertions are unchanged.
    """
    arts = [
        {"title": f"unique chip headline alpha{i} for the desk topic{i}",
         "source": "rss", "ai_score": 7.0, "summary": "body"}
        for i in range(65)
    ]
    payload = claude_analyst._build_payload(
        arts, {"macro": [], "equities": []}, []
    )
    assert "\n60. " in payload
    assert "61. " not in payload
