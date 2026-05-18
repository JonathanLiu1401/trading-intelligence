"""analysis/claude_analyst.py — PRIOR DIGEST continuity (anti-rehash).

A news analyst reading consecutive 5h heartbeats complains most about
repetition: the briefing re-LEADS with the SAME story it led with last time.
Confirmed live 2026-05-18 — briefing id26 (07:13Z) and id27 (12:51Z, 5.6h
later) BOTH led with the global-bond-rout-into-NVDA-earnings story, the
documented #1 noise complaint on the primary consumed product.

A per-article-title match against the rendered prior briefing was empirically
measured at 0% recall (Opus paraphrases every headline), so the feature parses
the prior briefing's OWN deterministic SYSTEM_PROMPT format (the literal
``**LEAD:**`` line + ``**TOP SIGNALS**`` fenced block) and feeds it back as a
framing hint — Opus does the semantic "same story?" comparison (its strength),
exactly as it already does for BOOK HEAT / AGING TOP ROWS.

These tests pin the behaviour with specific-value assertions:

  * `_parse_prior_digest` — extracts the LEAD sentence + TOP SIGNALS lines
    verbatim from a real-format briefing; caps signals; degrades to empty on
    garbage/missing sections; never raises.
  * `_prior_digest_lines` — pure renderer; [] on None / non-dict / empty.
  * `_recent_briefing_digest` — best-effort own-DB read: newest NON-sentinel
    briefing wins (sentinel filter pinned), correct age sign, any
    failure/missing/empty → None and never propagates.
  * `_build_payload` — emits the block ONLY when an explicit prior_digest is
    passed (None ⇒ deterministic, no DB read, section omitted); never mutates
    source_articles.
  * SYSTEM_PROMPT — carries the new framing rule AND the existing
    BOOK HEAT / AGING TOP ROWS / [ALERTED] / COVERAGE GAP rules byte-unchanged.
  * `analyze` — wires the live prior-digest read into the prompt Opus receives
    and still degrades when there is no prior digest.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import analysis.claude_analyst as ca


# A real-format prior briefing (the exact id26 shape read live 2026-05-18).
_REAL_PRIOR = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**DIGITAL INTERN** ◈ 2026-05-18 07:04 UTC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**LEAD:** Global bond rout deepens — 10Y UST +13bp to 4.59% on oil-fed inflation fears — dragging Nasdaq -1.54% in a semis-led selloff two days before NVDA earnings.

**MACRO**
```
INDEX        LAST       CHG%
S&P 500    7,408.50   -1.24%
```

**TOP SIGNALS**
```
9.24 MACRO Japan leads global bond rout, inflation fears rise
8.72 NVDA  UBS sees more upside, AI demand stays healthy
8.67 MU    CXMT revenue to surge as memory demand soars
```

**DESK NOTE:** Rates, not earnings, drive this tape.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""


class TestParsePriorDigest:
    def test_extracts_lead_verbatim(self):
        out = ca._parse_prior_digest(_REAL_PRIOR)
        assert out["lead"] == (
            "Global bond rout deepens — 10Y UST +13bp to 4.59% on oil-fed "
            "inflation fears — dragging Nasdaq -1.54% in a semis-led selloff "
            "two days before NVDA earnings."
        )

    def test_extracts_top_signals_lines(self):
        out = ca._parse_prior_digest(_REAL_PRIOR)
        assert out["top_signals"] == [
            "9.24 MACRO Japan leads global bond rout, inflation fears rise",
            "8.72 NVDA  UBS sees more upside, AI demand stays healthy",
            "8.67 MU    CXMT revenue to surge as memory demand soars",
        ]

    def test_top_signals_capped(self):
        many = "\n".join(f"{i}.0 T sig{i}" for i in range(20))
        txt = f"**LEAD:** x\n\n**TOP SIGNALS**\n```\n{many}\n```\n"
        out = ca._parse_prior_digest(txt)
        assert len(out["top_signals"]) == ca._PRIOR_DIGEST_MAX_SIGNALS == 6
        assert out["top_signals"][0] == "0.0 T sig0"

    def test_missing_sections_degrade_empty(self):
        assert ca._parse_prior_digest("just some prose, no markers") == {
            "lead": "", "top_signals": []
        }

    def test_lead_without_top_signals(self):
        out = ca._parse_prior_digest("**LEAD:** Only a lead here.\n")
        assert out["lead"] == "Only a lead here."
        assert out["top_signals"] == []

    def test_none_and_non_str_never_raise(self):
        for bad in (None, 123, b"bytes", [], {}):
            assert ca._parse_prior_digest(bad) == {"lead": "", "top_signals": []}

    def test_unterminated_fence_yields_no_signals(self):
        # Only one ``` after the header → no closing fence → no signals,
        # but the LEAD is still recovered (independent parse).
        txt = "**LEAD:** L\n\n**TOP SIGNALS**\n```\n9.0 T a\n(no close)"
        out = ca._parse_prior_digest(txt)
        assert out["lead"] == "L"
        assert out["top_signals"] == []


class TestPriorDigestLines:
    def test_none_and_non_dict_empty(self):
        assert ca._prior_digest_lines(None) == []
        assert ca._prior_digest_lines("nope") == []
        assert ca._prior_digest_lines(42) == []

    def test_empty_dict_empty(self):
        assert ca._prior_digest_lines({"lead": "", "top_signals": []}) == []
        assert ca._prior_digest_lines({}) == []

    def test_renders_lead_and_signals(self):
        out = ca._prior_digest_lines(
            {"lead": "Bond rout", "top_signals": ["9.2 MACRO x", "8.1 NVDA y"]}
        )
        assert out == [
            "LEAD (last briefing): Bond rout",
            "TOP SIGNAL (last briefing): 9.2 MACRO x",
            "TOP SIGNAL (last briefing): 8.1 NVDA y",
        ]

    def test_lead_only(self):
        assert ca._prior_digest_lines({"lead": "Solo", "top_signals": []}) == [
            "LEAD (last briefing): Solo"
        ]

    def test_blank_signal_entries_filtered(self):
        out = ca._prior_digest_lines(
            {"lead": "", "top_signals": ["  ", "", "9 X real"]}
        )
        assert out == ["TOP SIGNAL (last briefing): 9 X real"]


def _make_briefings_db(tmp_path, rows):
    """rows: list of (ts, text) inserted oldest-first (autoincrement id)."""
    p = tmp_path / "articles.db"
    c = sqlite3.connect(str(p))
    c.execute(
        "CREATE TABLE briefings (id INTEGER PRIMARY KEY AUTOINCREMENT, "
        "ts TEXT NOT NULL, text TEXT NOT NULL)"
    )
    c.executemany("INSERT INTO briefings (ts, text) VALUES (?, ?)", rows)
    c.commit()
    c.close()
    return p


class TestRecentBriefingDigest:
    def test_returns_parsed_with_age(self, tmp_path, monkeypatch):
        now = datetime(2026, 5, 18, 12, 51, tzinfo=timezone.utc)
        five_h_ago = (now - timedelta(hours=5, minutes=30)).isoformat()
        p = _make_briefings_db(tmp_path, [(five_h_ago, _REAL_PRIOR)])
        from storage import article_store
        monkeypatch.setattr(article_store, "_get_db_path", lambda: p)
        out = ca._recent_briefing_digest(now=now)
        assert out is not None
        assert out["lead"].startswith("Global bond rout deepens")
        assert out["top_signals"][0].startswith("9.24 MACRO Japan")
        assert 5.4 < out["age_h"] < 5.6  # ~5.5h, positive

    def test_newest_non_sentinel_wins_over_newer_sentinel(self, tmp_path,
                                                          monkeypatch):
        ts0 = "2026-05-18T01:00:00+00:00"
        ts1 = "2026-05-18T07:00:00+00:00"
        ts2 = "2026-05-18T12:00:00+00:00"
        p = _make_briefings_db(tmp_path, [
            ("2026-05-17T20:00:00+00:00", "old real **LEAD:** stale\n"),
            (ts1, _REAL_PRIOR),                                   # real, id=2
            (ts2, "[analyst] No response from Claude."),           # sentinel, id=3
        ])
        from storage import article_store
        monkeypatch.setattr(article_store, "_get_db_path", lambda: p)
        out = ca._recent_briefing_digest(
            now=datetime(2026, 5, 18, 12, 30, tzinfo=timezone.utc)
        )
        # The newest row is a sentinel and MUST be skipped → id=2 (_REAL_PRIOR).
        assert out is not None
        assert out["lead"].startswith("Global bond rout deepens")

    def test_sentinel_only_table_returns_none(self, tmp_path, monkeypatch):
        p = _make_briefings_db(tmp_path, [
            ("2026-05-18T07:00:00+00:00", "[analyst] No response from Claude."),
            ("2026-05-18T12:00:00+00:00", "No response from Claude."),
        ])
        from storage import article_store
        monkeypatch.setattr(article_store, "_get_db_path", lambda: p)
        assert ca._recent_briefing_digest() is None

    def test_empty_table_returns_none(self, tmp_path, monkeypatch):
        p = _make_briefings_db(tmp_path, [])
        from storage import article_store
        monkeypatch.setattr(article_store, "_get_db_path", lambda: p)
        assert ca._recent_briefing_digest() is None

    def test_missing_db_returns_none_never_raises(self, tmp_path, monkeypatch):
        from storage import article_store
        monkeypatch.setattr(
            article_store, "_get_db_path", lambda: tmp_path / "nope.db"
        )
        assert ca._recent_briefing_digest() is None

    def test_path_resolver_raising_returns_none(self, monkeypatch):
        from storage import article_store

        def boom():
            raise RuntimeError("usb gone")

        monkeypatch.setattr(article_store, "_get_db_path", boom)
        assert ca._recent_briefing_digest() is None

    def test_real_row_with_no_parseable_markers_returns_none(self, tmp_path,
                                                             monkeypatch):
        # A persisted briefing whose body has neither LEAD nor TOP SIGNALS
        # carries no usable continuity signal → None (not a crash, not a
        # phantom block).
        p = _make_briefings_db(tmp_path, [
            ("2026-05-18T07:00:00+00:00", "totally malformed briefing body")
        ])
        from storage import article_store
        monkeypatch.setattr(article_store, "_get_db_path", lambda: p)
        assert ca._recent_briefing_digest() is None


_STOCK = {"macro": [], "equities": []}


class TestBuildPayloadWiring:
    def test_none_omits_block_deterministic(self):
        payload = ca._build_payload([], _STOCK, [], prior_digest=None)
        assert "PRIOR DIGEST" not in payload

    def test_explicit_prior_emits_block_verbatim(self):
        prior = {"age_h": 5.5, "lead": "Bond rout deepens",
                 "top_signals": ["9.2 MACRO bond rout"]}
        payload = ca._build_payload([], _STOCK, [], prior_digest=prior)
        assert "=== PRIOR DIGEST" in payload
        assert "~5.5h ago" in payload
        assert "LEAD (last briefing): Bond rout deepens" in payload
        assert "TOP SIGNAL (last briefing): 9.2 MACRO bond rout" in payload

    def test_unknown_age_renders_earlier(self):
        prior = {"age_h": None, "lead": "X", "top_signals": []}
        payload = ca._build_payload([], _STOCK, [], prior_digest=prior)
        assert "PRIOR DIGEST" in payload
        assert "earlier" in payload.split("PRIOR DIGEST")[1][:120]

    def test_empty_prior_dict_omits_block(self):
        payload = ca._build_payload(
            [], _STOCK, [], prior_digest={"lead": "", "top_signals": []}
        )
        assert "PRIOR DIGEST" not in payload

    def test_does_not_mutate_source_articles(self):
        arts = [{"_id": "a", "link": "http://x/1", "title": "Real story one",
                 "summary": "body", "ai_score": 7.0, "first_seen": ""}]
        before_ids = [id(a) for a in arts]
        before_keys = set(arts[0].keys())
        prior = {"age_h": 3.0, "lead": "L", "top_signals": ["1 T s"]}
        ca._build_payload(list(arts), _STOCK, [], prior_digest=prior)
        # Same objects, no key injected by the prior-digest path.
        assert [id(a) for a in arts] == before_ids
        assert set(arts[0].keys()) == before_keys

    def test_block_appended_after_coverage_gap(self):
        # Both blocks present → PRIOR DIGEST comes AFTER COVERAGE GAP
        # (it is the last appended data block).
        report = {"sec_edgar": {"disabled": True, "consecutive_failures": 100,
                                "total_articles": 0}}
        prior = {"age_h": 5.0, "lead": "Lead text", "top_signals": []}
        payload = ca._build_payload(
            [], _STOCK, [], source_health_report=report, prior_digest=prior
        )
        assert "COVERAGE GAP" in payload and "PRIOR DIGEST" in payload
        assert payload.index("COVERAGE GAP") < payload.index("PRIOR DIGEST")


class TestSystemPrompt:
    def test_new_prior_digest_rule_present(self):
        sp = ca.SYSTEM_PROMPT
        assert '"PRIOR DIGEST"' in sp
        assert "do NOT restate it as the LEAD" in sp
        assert "most-cited \"repetitive digest\" complaint" in sp
        # Explicitly a non-echoed framing hint (BOOK HEAT / AGING shape).
        assert "do NOT echo a literal \"PRIOR DIGEST\" section" in sp

    def test_existing_rules_unchanged(self):
        # Anti-regression: adding the new rule must not have altered the
        # pinned prose of the sibling rules other suites assert on.
        sp = ca.SYSTEM_PROMPT
        assert 'A newswire row tagged "[ALERTED]"' in sp
        assert 'If a "BOOK HEAT" block is present' in sp
        assert 'If an "AGING TOP ROWS" block is present' in sp
        assert 'If a "COVERAGE GAP" block is present in the data input' in sp
        # PRIOR DIGEST is a hint, NOT an OUTPUT FORMAT section (unlike
        # COVERAGE GAP which IS echoed) — it must not appear in the format
        # skeleton's section list.
        fmt = sp.split("OUTPUT FORMAT")[1]
        assert "**PRIOR DIGEST**" not in fmt


class TestAnalyzeWiring:
    def test_analyze_passes_prior_digest_into_prompt(self):
        sentinel = {"age_h": 4.0, "lead": "Carried bond-rout lead",
                    "top_signals": ["9.0 MACRO rout"]}
        captured = {}

        def fake_call(prompt, **kw):
            captured["p"] = prompt
            return "OK BRIEFING"

        with patch.object(ca, "_recent_briefing_digest",
                          return_value=sentinel), \
             patch.object(ca, "_collect_source_health", return_value={}), \
             patch.object(ca, "claude_call", side_effect=fake_call):
            out = ca.analyze([], _STOCK, [])
        assert out == "OK BRIEFING"
        assert "PRIOR DIGEST" in captured["p"]
        assert "Carried bond-rout lead" in captured["p"]

    def test_analyze_degrades_when_no_prior_digest(self):
        with patch.object(ca, "_recent_briefing_digest", return_value=None), \
             patch.object(ca, "_collect_source_health", return_value={}), \
             patch.object(ca, "claude_call", return_value=""):
            out = ca.analyze([], {}, [])
        # Empty Claude response still yields the documented sentinel, never
        # None, and the missing prior digest never broke the call.
        assert out == "[analyst] No response from Claude."


# The exact id27 briefing read live 2026-05-18 (wider ━x40 divider than id26's
# ━x30, a real LEAD with parenthetical/percent punctuation, and bracketed
# [score] TICKER signal rows). Pins parser robustness across the real format
# variants Opus actually emits — distinct from _REAL_PRIOR (id26 shape).
_REAL_PRIOR_ID27 = """━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**DIGITAL INTERN** ◈ 2026-05-18 12:44 UTC
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
**LEAD:** Iran-war inflation scare drives a global bond rout (US 30Y 5.13%, post-2023 high) and a -1.24% S&P / -3.80% SMH semis dump into NVDA earnings Wed — but the live tape is already cooling (WTI -4.15%, bond selloff easing).

**TOP SIGNALS**
```
[9.37] NVDA  GF Securities ups PT ahead of Q1 [x2]
[9.10] INTC  Trump: should've asked for bigger Intel stake [x2]
[8.96] MACRO Global bond yields at multiyear highs
```

**DESK NOTE:** Headline flow is still bond-rout heavy.
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"""


class TestRealFormatVariants:
    def test_parses_real_id27_lead_and_signals(self):
        out = ca._parse_prior_digest(_REAL_PRIOR_ID27)
        assert out["lead"].startswith("Iran-war inflation scare drives")
        assert out["lead"].endswith("(WTI -4.15%, bond selloff easing).")
        assert out["top_signals"] == [
            "[9.37] NVDA  GF Securities ups PT ahead of Q1 [x2]",
            "[9.10] INTC  Trump: should've asked for bigger Intel stake [x2]",
            "[8.96] MACRO Global bond yields at multiyear highs",
        ]
