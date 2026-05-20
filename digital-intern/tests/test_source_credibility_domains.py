"""Publisher-domain resolution in ml.features._source_credibility.

~95% of the production corpus is aggregator-prefixed (``gdelt_gkg/<host>``,
``GDELT/<host>``, ``scraped/<host>``, ``SEC-EDGAR/<form>``). The verbatim
word-boundary scan silently defaulted 86% of the live top-40 source tags,
flattening feature[0] for the model and blinding watchers.alert_agent's
lone-alert authority gate to the real publisher.

Phase-1 contract — the rescue is *strictly additive*:
  * NO tag that already resolved to a non-default grade may move (every
    existing test_features / test_alert_source_authority pin stays green);
  * a genuinely-defaulting prefixed tag now resolves to the real publisher
    grade;
  * every _DOMAIN_CRED value is >= DEFAULT and _LOW_AUTHORITY_DOMAINS is empty,
    so the 0.45 lone-alert gate is byte-identical to pre-fix (the junk tier
    that *lowers* it is Phase 2).
"""
from __future__ import annotations

import pytest

from ml import features
from ml.features import (
    DEFAULT_SOURCE_CRED,
    _DOMAIN_CRED,
    _LOW_AUTHORITY_DOMAINS,
    _PREFIX_ALIASES,
    _domain_candidates,
    _source_credibility,
)

# True PRE-FIX grades (verbatim word-boundary scan only, before the domain
# tier and the SEC-EDGAR alias were added). Confirmed against a 1.4M-row live
# DB snapshot where these exact tags were observed. Hardcoded — NOT recomputed
# from the patched module — so this is an honest before/after reference.
PRE_FIX = {
    "reuters": 0.90,
    "reddit": 0.40,
    "nitter": 0.40,
    "twitter": 0.35,
    "stocktwits": 0.30,
    "rss": 0.65,
    "scraped": 0.50,
    "reddit/r/Daytrading": 0.40,
    "gdelt_gkg/reuters.com": 0.90,          # 'reuters' matched by luck pre-fix
    "gdelt_gkg/cnbc.com": 0.85,             # 'cnbc' single-token, matched pre-fix
    "scraped/finance.yahoo.com": 0.65,      # 'yahoo' matched pre-fix
    "Finnhub/Yahoo": 0.65,
    "Finnhub/SeekingAlpha": 0.78,           # 'finnhub' matched; SA key has space
    "some-new-feed-2026": DEFAULT_SOURCE_CRED,
    "gdelt_gkg/iheart.com": DEFAULT_SOURCE_CRED,
    "gdelt_gkg/seekingalpha.com": DEFAULT_SOURCE_CRED,  # key "seeking alpha"
    "SEC-EDGAR/8-K": DEFAULT_SOURCE_CRED,                # prefix shape, no match
}


class TestPhase1IsStrictlyAdditive:
    def test_rescue_tier_never_below_default(self):
        assert _DOMAIN_CRED, "rescue tier must be populated"
        for host, cred in _DOMAIN_CRED.items():
            assert cred >= DEFAULT_SOURCE_CRED, f"{host}={cred} < DEFAULT"

    def test_tier_signs_never_overlap(self):
        """Durable separation invariant: the rescue tier is strictly >= DEFAULT
        and the junk tier is strictly < DEFAULT, with no host in both. If a
        future edit puts a junk host >= DEFAULT it would silently stop being
        gated; a rescue host < DEFAULT would wrongly suppress a real wire."""
        for host, cred in _LOW_AUTHORITY_DOMAINS.items():
            assert cred < DEFAULT_SOURCE_CRED, f"junk {host}={cred} >= DEFAULT"
        assert not (set(_DOMAIN_CRED) & set(_LOW_AUTHORITY_DOMAINS)), (
            "a host must not be in both the rescue and junk tiers"
        )

    def test_no_already_differentiated_tag_moves(self):
        """The load-bearing safety property (holds across Phase 1 AND 2): every
        tag that was ALREADY non-default keeps its EXACT grade. Only tags that
        used to silently default may be re-tiered — upward via the rescue tier,
        or downward only if their host is explicitly junk-listed."""
        for source, old in PRE_FIX.items():
            new = _source_credibility(source)
            if old != DEFAULT_SOURCE_CRED:
                assert new == pytest.approx(old), (
                    f"{source}: {old} -> {new} (already-differentiated, frozen)"
                )
                continue
            # old == DEFAULT: a downward move is allowed ONLY for an
            # explicitly junk-listed host (the Phase-2 noise-gate feature).
            host_is_junk = any(
                h in source for h in _LOW_AUTHORITY_DOMAINS
            )
            if not host_is_junk:
                assert new >= old - 1e-9, (
                    f"{source}: {old} -> {new} (unexpected un-listed regression)"
                )

    def test_domain_tier_agrees_with_verbatim_where_they_overlap(self):
        """If a bare domain ALSO matches the legacy word-boundary scan, the
        domain grade MUST equal it — otherwise the same publisher scores
        differently as ``reuters`` vs ``gdelt_gkg/reuters.com``: silent
        train/serve drift."""
        for host, cred in _DOMAIN_CRED.items():
            verbatim = DEFAULT_SOURCE_CRED
            for pat, score in features._SOURCE_CRED_PATTERNS:
                if pat.search(host):
                    verbatim = score
                    break
            if verbatim != DEFAULT_SOURCE_CRED:
                assert cred == verbatim, (
                    f"{host}: domain grade {cred} != verbatim {verbatim}"
                )


class TestPreviouslyDefaultingTagsNowResolve:
    @pytest.mark.parametrize(
        "source,expected",
        [
            ("gdelt_gkg/seekingalpha.com", 0.72),   # key "seeking alpha" (space)
            ("GDELT/seekingalpha.com", 0.72),
            ("SEC-EDGAR/8-K", 0.95),                # prefix-shape alias rescue
            ("SEC-EDGAR/10-Q", 0.95),
        ],
    )
    def test_genuine_rescue_from_default(self, source, expected):
        """These tags ALL defaulted pre-fix (key has a space the concatenated
        domain lacks, or the SEC-EDGAR/<form> prefix shape never matched
        sec_edgar/sec edgar). They now resolve to the true publisher grade —
        the model gets real source signal and the alert gate sees SEC."""
        assert _source_credibility(source) == pytest.approx(expected), (
            f"{source}: expected rescue to {expected}"
        )

    def test_subdomain_strip_resolves_registrable_host(self):
        assert _domain_candidates("gdelt_gkg/finance.yahoo.com") == [
            "finance.yahoo.com",
            "yahoo.com",
        ]
        assert _source_credibility("gdelt_gkg/finance.yahoo.com") == pytest.approx(
            0.65
        )

    def test_non_dotted_tokens_yield_no_host_candidates(self):
        """reddit / sec-edgar / nitter have no dotted host — they must route
        via the verbatim scan, never the domain tier."""
        assert _domain_candidates("reddit/r/Daytrading") == []
        assert _domain_candidates("SEC-EDGAR/8-K") == []
        assert _domain_candidates("nitter") == []
        assert _domain_candidates("") == []

    def test_unknown_host_still_defaults_and_is_never_gated(self):
        """An unmapped publisher must still land at DEFAULT — strictly above
        the 0.45 lone-alert gate, so brand-new sources are NEVER auto-
        suppressed. Pins the documented 'unknown is never gated' invariant
        through the new code path."""
        c = _source_credibility("gdelt_gkg/brand-new-outlet-2027.example")
        assert c == pytest.approx(DEFAULT_SOURCE_CRED)
        assert c >= 0.45


class TestPrefixAliasesRescueAggregatorTags:
    """``_PREFIX_ALIASES`` rescues aggregator tag conventions that carry no
    dotted host AND are missed by the verbatim word-boundary scan. Same Phase-1
    contract as the rescue tier: strictly additive, every alias resolves to
    an existing SOURCE_CRED grade, every value >= DEFAULT.

    Live evidence (2026-05-20, 24h): ``GN: <topic>`` (~5,376 rows from
    config/sources.json Google News topic feeds), ``YF/<bucket>`` (95 rows
    from collectors/market_movers.py screener-tape), ``YahooFinance/<sym>``
    (~hundreds from collectors/yahoo_ticker_rss.py) all silently fell to
    DEFAULT_SOURCE_CRED — flattening feature[0] for the model and blinding
    the lone-alert authority gate."""

    @pytest.mark.parametrize(
        "source,expected",
        [
            # Google News topic feeds — sources.json: "GN: Nvidia" etc.
            ("GN: Nvidia", features.SOURCE_CRED["googlenews"]),
            ("GN: earnings", features.SOURCE_CRED["googlenews"]),
            ("GN: stock market", features.SOURCE_CRED["googlenews"]),
            # Yahoo Finance market_movers screener-tape buckets.
            ("YF/most_actives", features.SOURCE_CRED["yfinance"]),
            ("YF/day_gainers", features.SOURCE_CRED["yfinance"]),
            # Yahoo per-ticker RSS — embedded "yahoo" misses the word
            # boundary in "yahoofinance"; alias rescues it.
            ("YahooFinance/005930.KS", features.SOURCE_CRED["yahoo"]),
            ("YahooFinance/MU", features.SOURCE_CRED["yahoo"]),
        ],
    )
    def test_aggregator_prefix_resolves_to_publisher(self, source, expected):
        assert _source_credibility(source) == pytest.approx(expected), (
            f"{source}: expected prefix-alias rescue to {expected}"
        )

    def test_prefix_aliases_never_below_default(self):
        """Phase-1 contract: a prefix alias only ever RAISES a defaulting tag."""
        for prefix, score in _PREFIX_ALIASES:
            assert score >= DEFAULT_SOURCE_CRED, (
                f"alias {prefix!r}={score} < DEFAULT — would *suppress* a wire"
            )

    def test_prefix_aliases_only_resolve_to_existing_grades(self):
        """Anti-drift discipline: an alias must point at a publisher tier that
        ALREADY exists in SOURCE_CRED — so adding the alias is purely a
        spelling rescue, never an opinionated grade for an unrated publisher."""
        existing = set(features.SOURCE_CRED.values())
        for prefix, score in _PREFIX_ALIASES:
            assert score in existing, (
                f"alias {prefix!r}={score} doesn't match any SOURCE_CRED grade"
            )

    def test_prefix_alias_is_anchored_not_substring(self):
        """An alias must only fire on a tag that actually LEADS with that
        prefix — never as a mid-string substring. Pins the discriminator so
        a future regression replacing startswith with a contains check
        wouldn't silently re-grade a real publisher whose name happens to
        contain ``gn:`` mid-string."""
        # "EFGN: x" / "AGN: x" must NOT be re-graded as Google News — the
        # alias is anchored to the START of the (lstripped) tag.
        for non_alias in ("EFGN: news", "AGN: pharma", "MGN: x", "TheGN:"):
            assert _source_credibility(non_alias) == pytest.approx(
                DEFAULT_SOURCE_CRED
            ), f"{non_alias!r} must not match GN: alias (substring guard)"

    def test_already_differentiated_tags_still_unchanged(self):
        """Belt-and-braces parity check on the most-trafficked rescued tags:
        every tag that already resolved to a non-default grade must keep its
        EXACT pre-alias value through the new code path."""
        # The "Finnhub/Yahoo" case is the spelling-order discriminator pinned
        # in PRE_FIX above: yahoo (0.65) appears in SOURCE_CRED BEFORE finnhub
        # (0.78), so the word-boundary scan returns 0.65 first. The new alias
        # step runs BEFORE the verbatim scan but only after testing for the
        # tag's own prefix, so this resolution is unchanged.
        assert _source_credibility("Finnhub/Yahoo") == pytest.approx(0.65)
        assert _source_credibility("yfinance/AFP") == pytest.approx(0.65)
        assert _source_credibility("reddit/r/Daytrading") == pytest.approx(0.40)
        assert _source_credibility("GDELT/finance.yahoo.com") == pytest.approx(0.65)
