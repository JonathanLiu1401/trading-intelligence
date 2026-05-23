"""Wikipedia recent-changes reference content does not fire BREAKING alerts
or surface as a fresh TOP SIGNAL in the heartbeat briefing.

Regression guard for the ``[Wikipedia]``-prefix recap fingerprint added to
``watchers.alert_agent._RT_WIKIPEDIA_REF`` and the lockstep briefing-side
mirror ``analysis.claude_analyst._BRIEFING_RT_WIKIPEDIA_REF``.

Live evidence (2026-05-23, 7-day articles.db scan): ``[Wikipedia] DRAM
(musician)`` (ml_score=10.0) and ``[Wikipedia] Nvidia RTX`` (ml_score=8.6)
both reached urgency=2 with score_source='ml'. Wikipedia's source-credibility
tier (0.60) sits above the 0.45 ALERT_MIN_LONE_SOURCE_CRED bar, so the
authority gate doesn't catch them — content type IS the failure mode.

Critical: the sibling ``collectors.wikipedia_pageviews`` collector — a useful
predictive signal that surfaces 2.5σ page-view surges as early indicators of
breaking news — emits titles in a DIFFERENT shape ("Wiki pageview SURGE NVDA
(NVIDIA_Corporation): ...") without the leading bracketed-source tag. Those
must survive the gate verbatim.
"""
from __future__ import annotations

from analysis import claude_analyst as ca
from watchers import alert_agent as aa


# Real noise — must all be matched by both gates.
WIKIPEDIA_NOISE = [
    "[Wikipedia] DRAM (musician)",
    "[Wikipedia] Nvidia RTX",
    "[Wikipedia] List of AMD Ryzen processors",
    "[Wikipedia] List of Nvidia graphics processing units",
    "[Wikipedia] Semiconductor fabrication plant",
    "[Wikipedia] Tesla, Inc.",
    "[Wikipedia] Stock market index",
    "[Wikipedia] MUST IPO SC",
    "[Wikipedia] Inflation Reduction Act",
    "[Wikipedia] Shanghai Stock Exchange",
    "  [Wikipedia] Sound chip",  # leading whitespace tolerated
]

# Must-survive — none of these may be matched by either gate.
MUST_SURVIVE = [
    # The wikipedia_pageviews collector's title shape — useful early-warning signal.
    "Wiki pageview SURGE NVDA (NVIDIA_Corporation): 12,345 vs 4,567 baseline (z=+3.2, x2.7) 2026-05-23",
    "Wiki pageview DROP MU (Micron_Technology): 2,001 vs 8,500 baseline (z=-3.5, x0.2) 2026-05-23",
    # Real wire headlines that happen to mention Wikipedia mid-text.
    "Wikipedia adds new NVDA reference page after IPO",
    "MU references Wikipedia in 10-K filing — material disclosure",
    "Bloomberg: Wikipedia Foundation receives $50M donation from tech billionaire",
    # Real breaking copy with brackets / source tags that are NOT [Wikipedia].
    "[Reuters] Nvidia Q1 revenue rises 22%",
    "[BREAKING] Fed cuts rates 50bp",
    "Nvidia (NVDA) tops Q1 earnings and revenue estimates",
    "MU shares halted on guidance update",
    # Forward-looking question form (the canonical must-survive recap-corpus shape).
    "Why investors are bullish on Nvidia",
]


class TestAlertWikipediaRefGate:
    def test_noise_titles_all_matched(self):
        for title in WIKIPEDIA_NOISE:
            hit, name = aa._looks_like_recap_template({"title": title})
            assert hit, f"alert gate failed to match Wikipedia noise: {title!r}"
            assert name == "wikipedia_ref", (
                f"alert gate matched {title!r} with wrong fingerprint {name!r} "
                f"— precedence order may have shifted, the wikipedia_ref entry "
                f"should be the discriminator"
            )

    def test_must_survive_corpus_not_matched(self):
        for title in MUST_SURVIVE:
            hit, name = aa._looks_like_recap_template({"title": title})
            assert not hit, (
                f"alert gate over-suppressed must-survive title {title!r} "
                f"(matched fingerprint: {name})"
            )

    def test_pageview_signal_specifically_preserved(self):
        # The wikipedia_pageviews collector is a USEFUL predictive signal and
        # its rows must NEVER be caught by this gate (its title shape lacks
        # the leading bracketed-source tag).
        pageview_titles = [
            "Wiki pageview SURGE NVDA (NVIDIA_Corporation): 12,345 vs 4,567 baseline (z=+3.2, x2.7) 2026-05-23",
            "Wiki pageview DROP TSLA (Tesla,_Inc.): 5,001 vs 12,500 baseline (z=-2.8, x0.4) 2026-05-23",
            "Wiki pageview SURGE AMD (Advanced_Micro_Devices): 8,400 vs 3,200 baseline (z=+2.9, x2.6) 2026-05-22",
        ]
        for title in pageview_titles:
            hit, _ = aa._looks_like_recap_template({"title": title})
            assert not hit, (
                f"alert gate caught wikipedia_pageviews signal {title!r} "
                f"— this is a useful predictive surface that MUST survive"
            )


class TestBriefingWikipediaRefGate:
    def test_briefing_mirror_matches_alert(self):
        for title in WIKIPEDIA_NOISE:
            hit_a, name_a = aa._looks_like_recap_template({"title": title})
            hit_b, name_b = ca._looks_like_recap_template({"title": title})
            assert hit_a == hit_b, (
                f"alert/briefing gates disagree on {title!r}: alert={hit_a} "
                f"briefing={hit_b} — the documented lockstep is broken"
            )
            assert name_a == name_b, (
                f"alert/briefing gates assigned different fingerprint names "
                f"to {title!r}: alert={name_a!r} briefing={name_b!r}"
            )

    def test_briefing_must_survive(self):
        for title in MUST_SURVIVE:
            hit, name = ca._looks_like_recap_template({"title": title})
            assert not hit, (
                f"briefing gate over-suppressed must-survive title {title!r} "
                f"(matched fingerprint: {name})"
            )


class TestFingerprintRegistration:
    def test_alert_pattern_in_registry(self):
        names = {n for n, _ in aa._RECAP_TEMPLATE_PATTERNS}
        assert "wikipedia_ref" in names, (
            "alert_agent._RECAP_TEMPLATE_PATTERNS missing wikipedia_ref entry"
        )

    def test_briefing_pattern_in_registry(self):
        names = {n for n, _ in ca._BRIEFING_RECAP_TEMPLATE_PATTERNS}
        assert "wikipedia_ref" in names, (
            "claude_analyst._BRIEFING_RECAP_TEMPLATE_PATTERNS missing wikipedia_ref entry"
        )

    def test_registry_byte_identical_pattern_source(self):
        # The two regex objects must compile from byte-identical source so the
        # lockstep cannot drift on a future tightening.
        alert_pat = dict(aa._RECAP_TEMPLATE_PATTERNS)["wikipedia_ref"]
        briefing_pat = dict(ca._BRIEFING_RECAP_TEMPLATE_PATTERNS)["wikipedia_ref"]
        assert alert_pat.pattern == briefing_pat.pattern, (
            f"wikipedia_ref pattern drift: "
            f"alert={alert_pat.pattern!r} briefing={briefing_pat.pattern!r}"
        )


class TestFilterPartition:
    def test_alert_partition_correctness(self):
        articles = [
            {"_id": "1", "title": "[Wikipedia] DRAM (musician)"},
            {"_id": "2", "title": "Nvidia Q1 earnings beat — revenue $81.6B"},
            {"_id": "3", "title": "[Wikipedia] Inflation Reduction Act"},
            {"_id": "4", "title": "Wiki pageview SURGE NVDA: 12k vs 4k baseline"},
        ]
        kept, suppressed = aa._filter_recap_template_noise(articles)
        kept_ids = {a["_id"] for a in kept}
        suppressed_ids = {a["_id"] for a in suppressed}
        assert kept_ids == {"2", "4"}
        assert suppressed_ids == {"1", "3"}
        # Suppressed rows carry the fingerprint tag for log clarity.
        for a in suppressed:
            assert a.get("_recap_fingerprint") == "wikipedia_ref"
        # Caller's articles must not be mutated (no _recap_fingerprint on inputs).
        for a in articles:
            assert "_recap_fingerprint" not in a

    def test_briefing_partition_correctness(self):
        articles = [
            {"title": "[Wikipedia] Nvidia RTX", "ai_score": 8.6},
            {"title": "Nvidia posts record $81.6B quarter", "ai_score": 9.5},
            {"title": "[Wikipedia] List of AMD Ryzen processors", "ai_score": 7.0},
        ]
        kept, suppressed = ca._filter_recap_template_noise(articles)
        assert [a["title"] for a in kept] == ["Nvidia posts record $81.6B quarter"]
        assert len(suppressed) == 2
        for a in suppressed:
            assert a.get("_recap_fingerprint") == "wikipedia_ref"
        # Inputs unmutated — heartbeat_worker feeds source_articles onward.
        for a in articles:
            assert "_recap_fingerprint" not in a
