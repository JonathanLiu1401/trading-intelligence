"""``triage.heuristic_scorer`` — pin the core relevance-scoring contract.

``score_article`` / ``score_and_rank`` are the first-stage funnel for the
entire pipeline: every collected article is filtered here before it ever
reaches the model or the LLM. At 412 lines it is one of the largest modules in
the repo and, until now, had **no dedicated test** — it was only exercised
indirectly through ``test_score_pending`` / ``test_inference_grey_zone``,
neither of which asserts anything about the heuristics themselves.

These tests lock the behaviors whose comments in the module explicitly record
a past bug + deliberate fix, i.e. the ones most likely to be silently
regressed by a future "cleanup":

  * **Blacklist survival asymmetry.** A blacklist hit no longer hard-zeros
    unconditionally — that previously discarded genuine supply-chain signals
    like "Earthquake damage halts TSMC fab". It zeros only when there is no
    real domain signal (``domain_kw < 3``); with a portfolio/memory/semis
    keyword present the article survives at ``blacklist_penalty=0.5``.
  * **Ticker word-boundary guard.** Bare tickers match on ``\b`` boundaries so
    "museum" does not become a "mu" hit and "MU beats" still does.
  * **no_keywords precedes blacklist.** With zero domain/macro/general
    keywords the function returns early (``reason='no_keywords'``) before the
    blacklist is consulted — a pure-noise blacklisted story is reported as
    ``no_keywords``, not ``blacklisted``.
  * **Recency decay** is bounded ``[0.1, 1.0]``, monotonically non-increasing
    with age, and degrades to the 0.85 unknown-age default on missing/garbage
    timestamps rather than raising.
  * **score_and_rank** drops sub-threshold rows, sorts descending, and
    attaches ``_relevance_score`` / ``_score_detail`` to survivors.

heuristic_scorer is pure (no DB / network), so these need no fixtures.
"""
from __future__ import annotations

from triage.heuristic_scorer import (
    score_article,
    score_and_rank,
    _recency_factor,
)


# ── Blacklist survival asymmetry ─────────────────────────────────────────────

def test_blacklist_with_strong_domain_signal_survives_penalized():
    """Earthquake/TSMC-class story: blacklist term present but real semis +
    memory signal → kept, not zeroed, at half weight."""
    r = score_article(
        "Earthquake damage halts TSMC fab in Taiwan, DRAM supply at risk",
        "wafer starts cut after quake; memory pricing expected to surge",
        "Reuters",
        "",
    )
    assert r["score"] > 0.0
    assert r["blacklist_penalty"] == 0.5
    assert r["reason"] == "scored_penalized"


def test_blacklist_with_weak_signal_is_zeroed():
    """Blacklist term + only a non-domain (general) keyword → domain_kw < 3,
    so the hard-zero path still fires with reason 'blacklisted'."""
    r = score_article(
        "Taylor Swift stock market earnings lawsuit",
        "celebrity equity dispute",
        "blog",
        "",
    )
    assert r["score"] == 0.0
    assert r["reason"] == "blacklisted"


def test_pure_noise_reports_no_keywords_not_blacklisted():
    """Zero keywords short-circuits *before* the blacklist check, so a
    blacklisted-but-keywordless story is 'no_keywords', not 'blacklisted'."""
    r = score_article(
        "Celebrity announces world tour",
        "pure entertainment gossip",
        "TMZ",
        "",
    )
    assert r["score"] == 0.0
    assert r["reason"] == "no_keywords"


# ── Ticker word-boundary guard ───────────────────────────────────────────────

def test_substring_ticker_false_positive_is_rejected():
    """'museum' must not register as the 'mu' (Micron) ticker."""
    r = score_article("Museum opens new wing", "art exhibit downtown", "blog", "")
    assert r["score"] == 0.0
    assert r["reason"] == "no_keywords"


def test_real_ticker_token_is_matched():
    """'MU beats' is a genuine whole-word ticker hit and must score."""
    r = score_article("MU beats earnings", "micron strong quarter", "Reuters", "")
    assert r["score"] > 0.0
    assert r["kw"] > 0.0


# ── Score range / normalisation ──────────────────────────────────────────────

def test_score_is_clamped_to_ten():
    r = score_article(
        "Micron raises DRAM ASP guidance 20% on HBM3E demand surge",
        "Q2 beat, EPS $1.42 vs $1.18E, NAND oversupply easing",
        "Reuters",
        "",
    )
    assert 0.0 <= r["score"] <= 10.0
    assert r["score"] == 10.0  # saturates for a top-tier portfolio+event story


# ── Recency decay ────────────────────────────────────────────────────────────

def test_recency_unknown_and_garbage_default_without_raising():
    assert _recency_factor("") == 0.85
    assert _recency_factor("not-a-date") == 0.85
    assert _recency_factor(None) == 0.85


def test_recency_is_bounded_and_monotonic_in_age():
    fresh = _recency_factor("Mon, 01 Jan 2035 00:00:00 +0000")  # future → t=0
    old = _recency_factor("Mon, 01 Jan 2020 00:00:00 +0000")    # ancient → floor
    assert 0.1 <= old <= fresh <= 1.0
    assert old == 0.1  # decay floors, never reaches 0


# ── score_and_rank contract ──────────────────────────────────────────────────

def test_score_and_rank_filters_sorts_and_annotates():
    arts = [
        {"title": "random noise", "summary": "nothing", "source": "x", "published": ""},
        {"title": "Micron DRAM HBM3E guidance raised", "summary": "beat",
         "source": "Reuters", "published": ""},
        {"title": "Fed rate cut, inflation, cpi report", "summary": "macro",
         "source": "Bloomberg", "published": ""},
    ]
    ranked = score_and_rank(arts, min_score=1.5, top_n=10)

    # sub-threshold "random noise" row dropped
    assert len(ranked) == 2
    titles = [a["title"] for a in ranked]
    assert "random noise" not in titles
    # descending by relevance, with detail attached to every survivor
    scores = [a["_relevance_score"] for a in ranked]
    assert scores == sorted(scores, reverse=True)
    for a in ranked:
        assert a["_relevance_score"] == a["_score_detail"]["score"]


def test_score_and_rank_respects_top_n():
    arts = [
        {"title": f"Micron DRAM HBM3E guidance beat {i}", "summary": "beat",
         "source": "Reuters", "published": ""}
        for i in range(5)
    ]
    assert len(score_and_rank(arts, min_score=1.0, top_n=3)) == 3


# ── Plural-form event detection ──────────────────────────────────────────────
# Past gap: supply/regulatory/capex EVENT_PATTERNS ended a count-noun group
# with ``\b``, so the singular "new fab" / "chip shortage" / "export control
# on chips" fired the multiplier but the (more common) plural phrasing —
# "new fabs", "chip shortages", "export controls on chips" — silently did
# not, systematically under-scoring real supply/capex/regulatory headlines.
# Events are gated behind kw>0 and only ever multiply already-relevant
# articles up, so widening to plurals adds signal without new false positives.

def test_plural_capex_fires_like_singular():
    sing = score_article("TSMC breaks ground on new fab in Arizona", "", "Reuters", "")
    plur = score_article("TSMC and Samsung break ground on new fabs in Arizona", "", "Reuters", "")
    fnd = score_article("Intel to build new foundries amid demand", "", "Reuters", "")
    assert "capex" in sing["events"]
    assert "capex" in plur["events"]
    assert "capex" in fnd["events"]
    assert plur["score"] == sing["score"]


def test_plural_supply_and_regulatory_fire():
    sh = score_article("Chip shortages worsen as DRAM demand outstrips supply", "", "Reuters", "")
    rg = score_article("US imposes new export controls on advanced memory chips", "", "Reuters", "")
    assert "supply" in sh["events"]
    assert "regulatory" in rg["events"]


# ── Corporate distress / legal / governance events ───────────────────────────
# Past gap: bankruptcy filings, SEC/DOJ probes, accounting restatements and
# CEO/CFO departures are among the sharpest single-name re-rating catalysts,
# yet none had an EVENT_PATTERNS entry — a "Chipmaker files for Chapter 11"
# headline scored on keywords+source only, with no event multiplier. Events
# are gated behind kw>0 and only ever multiply an already-relevant article
# up, so these add signal to real semis/portfolio stories without new noise.

def test_distress_bankruptcy_and_default_fire():
    bk = score_article("Memory chipmaker files for Chapter 11 bankruptcy", "", "Reuters", "")
    df = score_article("Chip supplier defaults on its bonds amid DRAM glut", "", "Reuters", "")
    assert "distress" in bk["events"]
    assert "distress" in df["events"]


def test_legal_probe_fraud_and_restatement_fire():
    sec = score_article("SEC opens probe into Micron revenue recognition", "", "Reuters", "")
    rs = score_article("Semiconductor firm restates its financials after audit", "", "Reuters", "")
    fr = score_article("DOJ charges chip executive with securities fraud", "", "Bloomberg", "")
    assert "legal" in sec["events"]
    assert "legal" in rs["events"]
    assert "legal" in fr["events"]


def test_exec_departure_fires_both_orderings():
    a = score_article("Intel CEO steps down amid foundry turnaround struggles", "", "Reuters", "")
    b = score_article("Micron CFO resigns abruptly ahead of earnings", "", "CNBC", "")
    assert "exec_change" in a["events"]
    assert "exec_change" in b["events"]


def test_distress_is_gated_behind_domain_keywords():
    """Events only fire after the kw>0 gate: a non-domain bankruptcy story
    never reaches event detection (reported as no_keywords, not scored)."""
    r = score_article("Local bakery files for Chapter 11 bankruptcy", "", "Reuters", "")
    assert r["events"] == []
    assert r["reason"] == "no_keywords"


# ── Multi-catalyst compounding ───────────────────────────────────────────────
# An article that fires several *distinct* event categories is a materially
# stronger signal than any one alone; the old max()-only event_bonus scored
# both identically. The compounding uplift gates on n_distinct>=2, so single-
# category scoring (and every test pinned to it) is unchanged.

def test_single_event_category_leaves_bonus_untouched():
    """One distinct category → no uplift: n_events==1 and the multiplier is
    exactly the matched pattern's value, preserving the legacy contract."""
    r = score_article("Buyback announced amid component shortage", "", "reddit", "")
    assert r["n_events"] == 1
    assert r["event_bonus"] == 2.0  # the lone "supply" multiplier, un-upflifted


def test_multi_event_categories_score_strictly_higher():
    """Same article, mid-range (un-clamped) score: adding a second distinct
    category (analyst downgrade) must rank it strictly above the single-
    category version, and the n_events count must reflect the difference."""
    single = score_article("Buyback announced amid component shortage", "", "reddit", "")
    multi = score_article(
        "Buyback announced amid component shortage and analyst downgrade to sell",
        "", "reddit", "",
    )
    assert single["n_events"] == 1
    assert multi["n_events"] == 2
    assert 0.0 < single["score"] < 10.0
    assert 0.0 < multi["score"] < 10.0
    assert multi["score"] > single["score"]
    assert multi["event_bonus"] > single["event_bonus"]


def test_multi_event_bonus_is_hard_capped():
    """Stacking many catalysts can never push event_bonus past the cap, so a
    multi-catalyst article cannot run away above a top-tier single crisis."""
    from triage.heuristic_scorer import MULTI_EVENT_BONUS_CAP
    r = score_article(
        "Chip maker earnings beat estimates, raises guidance above consensus, "
        "analyst upgrade to buy, NAND shortage, capex increase 30% billion, "
        "DRAM pricing up 20%, SEC investigation, CEO resigns, files for bankruptcy",
        "", "Reuters", "",
    )
    assert r["n_events"] >= 4
    assert r["event_bonus"] <= MULTI_EVENT_BONUS_CAP


# ── Portfolio-ticker drift: live config/portfolio.json positions must be honored ──

def test_live_portfolio_positions_get_ticker_tier_score():
    """Held positions read from config/portfolio.json (via ml.features.
    LIVE_PORTFOLIO_TICKERS) must score at the +4.0 ticker-tier bonus, not just
    the additive 1.5 boost. Prior bug: PORTFOLIO_TICKERS was a frozen literal
    so a position added in the trading UI (e.g. GOOG/COHR/NVDL on 2026-05-21)
    silently fell out of the ticker tier and was scored as generic sector news.
    """
    from triage.heuristic_scorer import PORTFOLIO_TICKERS, TIER_PORTFOLIO_TICKERS
    from ml.features import LIVE_PORTFOLIO_TICKERS

    if not LIVE_PORTFOLIO_TICKERS:
        # No live config; the union is a no-op and we can't make a meaningful
        # assertion. The test_features.py fallback test covers the empty case.
        return
    # Every live-held ticker must be in BOTH portfolio sets after the union.
    lowered_live = {t.lower() for t in LIVE_PORTFOLIO_TICKERS}
    missing_p = lowered_live - PORTFOLIO_TICKERS
    missing_t = lowered_live - TIER_PORTFOLIO_TICKERS
    assert not missing_p, (
        f"Live config tickers missing from PORTFOLIO_TICKERS (heuristic drift): "
        f"{sorted(missing_p)}"
    )
    assert not missing_t, (
        f"Live config tickers missing from TIER_PORTFOLIO_TICKERS: "
        f"{sorted(missing_t)}"
    )


def test_static_fallback_tickers_still_present_after_union():
    """The union must EXTEND the hardcoded fallback, never replace it. Tickers
    that pre-existed in the static set (LITE/MU/NVDA/...) must still receive
    the ticker-tier score even if config/portfolio.json is missing or pares
    them out — the static set is a safety floor."""
    from triage.heuristic_scorer import PORTFOLIO_TICKERS, TIER_PORTFOLIO_TICKERS

    must_have = {"lite", "lnok", "muu", "dram", "sndu", "mu", "msft",
                 "axti", "orcl", "tsem", "qbts", "nvda"}
    assert must_have.issubset(PORTFOLIO_TICKERS)
    assert must_have.issubset(TIER_PORTFOLIO_TICKERS)
