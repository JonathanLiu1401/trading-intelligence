"""YouTube-share-card SEO mill noise gate — assert specific titles match /
must-survive corpus is byte-clean, and that the storage pre-floor wiring
actually drops these to ``ml_score=0.01`` / ``urgency=0`` /
``score_source='ml'`` so they never reach ``urgency=1`` → alert path."""
from __future__ import annotations

import pytest

from watchers.youtube_mill_filter import (
    _YOUTUBE_MILL_TOKEN,
    filter_youtube_mill_noise,
    looks_like_youtube_mill,
)


# Live-evidence noise samples — every one of these was observed in
# articles.db carrying urgency>=1 in the 30d window leading up to
# 2026-05-30. The fingerprint must catch all of them.
NOISE_TITLES = [
    "S&P 500 - Quadruple Top Airdrie Fc (8KsWofaZjy) - Mshale",
    "NVIDIA Stock Price Analysis | Top $NVDA Levels To Watch For April 30th, 2025 Mortal Kombat (Tjm0z6tJEB) - Mshale",
    "CNBC Today On NVIDIA Stock, NVIDIA Groq Deal - NVDA Update Craig Berube (Jhb5fvrGZ8) - Mshale",
    "Nvidia Stock (NVDA) Earnings Call | Q1 2026 Breakdown Max Angioni (GtRnFn9fD5) - Mshale",
    "Nvidia Rises, Elf Beauty Climbs, Walmart Drops As It Flags Higher Fuel Costs (fF7scCMtmL) - Mshale",
    "AMD Stock: Up 262% — Better Than NVIDIA In 2026? ($AMD) Who Is Regina Hall (D11TWWAfgv) - Mshale",
    "MSFT Price Predictions - Microsoft Stock Analysis For Tuesday (KfMvp6zuqV) - Mshale",
    "Lumentum (LITE) Up 866%… But 28% Crash Ahead? Full Financial Breakdown (An6ydTgV2N) - fathomjournal.org",
    "Bears Not Done: Stock Market Crash: SPY QQQ IWM SMH VIX IWM DIA (VyYoYeVVY5) - fathomjournal.org",
    "ASML Stock SHOCKER! Revenue MISS But Shares SOAR? (y1fDXou0Mz) - Fathom Journal",
    "Other AI Stocks Impacted By NVDA Earnings (6xGXSeyKSt) - fernandovasconcelos.com",
]

# Curated must-survive corpus — every one of these is a legitimate
# headline or sentence that the gate MUST leave untouched. Assertions
# are byte-exact (no fuzzy "below threshold" — false positives here
# are silent and dangerous).
MUST_SURVIVE_TITLES = [
    # Real exchange-qualified tickers in parens — colon/dot/space breaks alnum-only
    "Nvidia (NASDAQ:NVDA) gives Q1 2026 guidance — analysts mixed",
    "Samsung (005930.KS) launches new HBM4E for AI",
    "Zscaler (NASDAQ:ZS) Price Target Cut to $223.00",
    "Apple Stock Forecast (NASDAQ: AAPL) — Goldman Raises Target",
    # Short parens (under length floor)
    "$NVDA breaks out (NYSE) on AI revenue beat",
    "Federal Reserve cuts rates (50bp) — Reuters",
    "$MU upgraded to Buy by Barclays (PT $215)",
    "GPT-4o release follows DeepSeek (R1) by weeks",
    # No-digit parens (foreign-symbol case, lacks one of the three signals)
    "Lumentum Holdings Inc (LITE) Shares Fall 8.8%",
    "Volkswagen (VOLKSWAGEN) announces Q3 earnings beat",
    # No-lowercase parens (all-caps ticker / exchange-only)
    "Samsung (005930KS) launches new HBM4E for AI",
    # Forward-looking earnings preview (no parenthetical hash)
    "Q3 2026 earnings preview: Nvidia, Micron, Lumentum",
    "NVIDIA Q1 Earnings Call Highlights",
    # Real news with accented characters
    "São Paulo factory expansion drives BRL gains — Reuters",
    "Beyoncé tour merch boosts LiveNation Q1",
    # Mid-sentence parenthetical with mixed-case alnum (no end anchor)
    "Some thing (alphaGo2) is OpenAI's next model said the CEO during the speech",
    "Apple's new (iPhone16) launch date confirmed by suppliers",
    # Parenthetical without trailing "- Publisher" or end-of-title anchor
    "(8KsWofaZjy) is the YouTube ID for a video about NVDA earnings call recap",
    # Legitimate corroborative headlines with currency or %
    "Tesla beats Q1 2025 estimates by 8.3% — Reuters",
    "Bank of America says rate cut means recession risk eases",
    "Wedbush raises NVDA price target to $215 - MarketBeat",
    "MU drops 5% on memory weakness - Yahoo",
    "Samsung Unveils HBM4E : A Leap in AI Memory Performance",
]


class TestYoutubeMillFingerprint:
    """The discriminator must catch every observed live-noise title and
    leave the must-survive corpus untouched. These are pinned exact tests."""

    @pytest.mark.parametrize("title", NOISE_TITLES)
    def test_catches_live_noise(self, title: str) -> None:
        assert looks_like_youtube_mill({"title": title}), (
            f"YouTube-mill noise title was NOT caught: {title!r}"
        )

    @pytest.mark.parametrize("title", MUST_SURVIVE_TITLES)
    def test_survives_real_headlines(self, title: str) -> None:
        assert not looks_like_youtube_mill({"title": title}), (
            f"Real headline FALSE-POSITIVE matched the gate: {title!r}"
        )

    def test_empty_title(self) -> None:
        assert looks_like_youtube_mill({"title": ""}) is False
        assert looks_like_youtube_mill({}) is False
        assert looks_like_youtube_mill({"title": None}) is False

    def test_token_length_floor(self) -> None:
        # 7-char alnum mixed-case+digit — below the 8-char minimum, no match
        # (the parenthetical is too short to be a YouTube ID).
        assert not looks_like_youtube_mill(
            {"title": "Some headline (GpT4ai7) - Publisher"}
        )
        # 8-char token — at the floor, MUST match
        assert looks_like_youtube_mill(
            {"title": "Some headline (AbCd1234) - Publisher"}
        )

    def test_token_length_ceiling(self) -> None:
        # 15-char token at the ceiling — must match
        assert looks_like_youtube_mill(
            {"title": "Headline (AbCdEf012345678) - Publisher"}
        )
        # 16-char token — above ceiling, no match
        assert not looks_like_youtube_mill(
            {"title": "Headline (AbCdEfGh01234567) - Publisher"}
        )

    def test_must_have_all_three_character_classes(self) -> None:
        # Missing lowercase
        assert not looks_like_youtube_mill(
            {"title": "Headline (ABCDEF1234) - Publisher"}
        )
        # Missing uppercase
        assert not looks_like_youtube_mill(
            {"title": "Headline (abcdef1234) - Publisher"}
        )
        # Missing digit
        assert not looks_like_youtube_mill(
            {"title": "Headline (AbCdEfGhIj) - Publisher"}
        )

    def test_anchor_requires_end_or_publisher_tail(self) -> None:
        # Mid-sentence — even with a valid token shape, no match
        assert not looks_like_youtube_mill(
            {"title": "(8KsWofaZjy) starts the headline but more text follows here"}
        )
        # End-of-title — matches
        assert looks_like_youtube_mill(
            {"title": "Some stock analysis video (8KsWofaZjy)"}
        )
        # Followed by " - Publisher" — matches
        assert looks_like_youtube_mill(
            {"title": "Some stock analysis video (8KsWofaZjy) - Mshale"}
        )

    def test_non_alnum_inside_parens_breaks_match(self) -> None:
        # Colon, dot, space, dash all break the alnum-only run
        assert not looks_like_youtube_mill(
            {"title": "Samsung (NASDAQ:005930) press release"}
        )
        assert not looks_like_youtube_mill(
            {"title": "Samsung (005930.KS) press release"}
        )
        assert not looks_like_youtube_mill(
            {"title": "Samsung (005930 KS) press release"}
        )

    def test_filter_partitions_correctly(self) -> None:
        batch = [
            {"title": NOISE_TITLES[0]},
            {"title": MUST_SURVIVE_TITLES[0]},
            {"title": NOISE_TITLES[1]},
            {"title": MUST_SURVIVE_TITLES[5]},
        ]
        kept, suppressed = filter_youtube_mill_noise(batch)
        kept_titles = [a["title"] for a in kept]
        suppressed_titles = [a["title"] for a in suppressed]
        assert kept_titles == [MUST_SURVIVE_TITLES[0], MUST_SURVIVE_TITLES[5]]
        assert suppressed_titles == [NOISE_TITLES[0], NOISE_TITLES[1]]


class TestStorePreFloorWiresInYoutubeMill:
    """The defense-in-depth gate must actually fire from
    ``storage.article_store.prefloor_pseudo_articles`` — the daemon's
    ML-path entry point — so the row exits the unscored queue at
    ``ml_score=0.01`` / ``urgency=0`` / ``score_source='ml'``, never
    reaching ``urgency=1`` or the alert path. This is the load-bearing
    integration test: a regex tightening on the filter side
    automatically engages here."""

    def test_youtube_mill_row_is_pre_floored(self, store) -> None:
        # Insert one YouTube-mill row + one real-news row directly.
        with store._write_lock:
            store.conn.execute(
                "INSERT INTO articles "
                "(id, url, title, source, published, kw_score, ai_score, "
                " urgency, first_seen, cycle) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    "noise1", "https://mshale.example/x",
                    "S&P 500 - Quadruple Top Airdrie Fc (8KsWofaZjy) - Mshale",
                    "GN: SP500", "", 1.0, 0.0, 0,
                    "2026-05-30T05:00:00+00:00", 0,
                ),
            )
            store.conn.execute(
                "INSERT INTO articles "
                "(id, url, title, source, published, kw_score, ai_score, "
                " urgency, first_seen, cycle) "
                "VALUES (?,?,?,?,?,?,?,?,?,?)",
                (
                    "real1", "https://reuters.com/x",
                    "Fed surprise cut: 50bp",
                    "rss", "", 1.0, 0.0, 0,
                    "2026-05-30T05:00:00+00:00", 0,
                ),
            )
            store.conn.commit()

        # Simulate the get_unscored payload shape used by the daemon
        batch = [
            {"_id": "noise1",
             "title": "S&P 500 - Quadruple Top Airdrie Fc (8KsWofaZjy) - Mshale",
             "source": "GN: SP500", "link": "https://mshale.example/x",
             "summary": "", "published": "", "first_seen": "2026-05-30T05:00:00+00:00"},
            {"_id": "real1",
             "title": "Fed surprise cut: 50bp",
             "source": "rss", "link": "https://reuters.com/x",
             "summary": "", "published": "", "first_seen": "2026-05-30T05:00:00+00:00"},
        ]
        real, n_pre = store.prefloor_pseudo_articles(batch)
        assert n_pre == 1, f"expected exactly 1 pre-floored row, got {n_pre}"
        assert [a["_id"] for a in real] == ["real1"]

        # Verify the noise row was written: ml_score=0.01, urgency=0,
        # score_source='ml'. The real row is untouched.
        row = store.conn.execute(
            "SELECT ai_score, ml_score, urgency, score_source "
            "FROM articles WHERE id=?",
            ("noise1",),
        ).fetchone()
        ai, ml, urg, src = row
        assert ai == 0.0, f"ai_score must remain 0 (invariant: LLM-only column), got {ai}"
        assert ml == pytest.approx(0.01), f"ml_score must be 0.01 (noise floor), got {ml}"
        assert urg == 0, f"urgency must stay 0 (never alerted), got {urg}"
        assert src == "ml", f"score_source must be 'ml' (model-floored), got {src!r}"

        row2 = store.conn.execute(
            "SELECT ai_score, ml_score, urgency, score_source "
            "FROM articles WHERE id=?",
            ("real1",),
        ).fetchone()
        assert row2 == (0.0, None, 0, None), (
            f"real-news row must be untouched by pre-floor: got {row2!r}"
        )


class TestRegexExposed:
    """The compiled regex is exposed for ``analytics`` audit modules to
    consume, mirroring ``alert_agent._QUOTE_WIDGET_TITLE_PATTERNS``."""

    def test_regex_is_compiled_pattern(self) -> None:
        import re as _re
        assert isinstance(_YOUTUBE_MILL_TOKEN, _re.Pattern)

    def test_regex_matches_canonical_sample(self) -> None:
        # Sanity — the canonical live-evidence noise row matches at the
        # tail-anchored position.
        m = _YOUTUBE_MILL_TOKEN.search(
            "S&P 500 - Quadruple Top Airdrie Fc (8KsWofaZjy) - Mshale"
        )
        assert m is not None
        assert "8KsWofaZjy" in m.group(0)
