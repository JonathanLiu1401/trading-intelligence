"""analysis/claude_analyst.py — per-signal wire-arrival timestamp.

`SYSTEM_PROMPT`'s TOP SIGNALS line asks Opus for `[HH:MM] [score] [TICKER]
headline` per signal, but `_build_payload` historically fed **zero** per-article
time data — so Opus had to fabricate or omit every timestamp on the analyst's
primary 5h digest (the same "prompt asks for X, payload omits X" class that
0792a57 closed on the alert path). `_seen_utc_str` surfaces the real
`first_seen` clock.

These pin specific values, not "no crash":
  * exact `HH:MM` from ISO / RFC822 / naive-UTC / `Z`-suffix / TZ-offset inputs;
  * `None` for absent / blank / unparseable (caller omits the token silently —
    the synthetic PORTFOLIO/OPTIONS snapshot rows carry no `first_seen` and
    must pass through with NO fabricated `00:00`);
  * the `[seen HH:MM UTC]` token actually lands in the rendered newswire row
    and survives the `_collapse_syndicated` shallow-copy path;
  * `_build_payload` does not mutate the caller's article dicts — the
    heartbeat worker feeds that same list to the briefing-label / training
    path, so a read-only contract here is load-bearing (backtest isolation /
    ml_score≠ai_score / score_source are all enforced upstream and untouched
    by construction: this layer only reshapes the text Opus reads).
"""
from __future__ import annotations

from unittest.mock import patch

from analysis import claude_analyst


class TestSeenUtcStr:
    def test_iso_with_offset_converted_to_utc(self):
        # 01:54 at +00:00 → 01:54 UTC
        assert claude_analyst._seen_utc_str("2026-05-18T01:54:07+00:00") == "01:54"

    def test_iso_naive_assumed_utc(self):
        assert claude_analyst._seen_utc_str("2026-05-18T14:32:00") == "14:32"

    def test_z_suffix_tolerated(self):
        assert claude_analyst._seen_utc_str("2026-05-18T09:07:30Z") == "09:07"

    def test_non_utc_offset_normalised_to_utc(self):
        # 23:30 at +09:00 (JST) is 14:30 UTC — the same instant must render in
        # UTC so a Nikkei row and a US-wire row are on one clock.
        assert claude_analyst._seen_utc_str("2026-05-18T23:30:00+09:00") == "14:30"

    def test_rfc822_parsed(self):
        # RSS feeds emit RFC822; first_seen is ISO but the helper accepts both
        # (same dual-format convention as alert_agent._article_age_hours).
        assert claude_analyst._seen_utc_str("Mon, 18 May 2026 06:05:00 +0000") == "06:05"

    def test_none_blank_and_unparseable_yield_none(self):
        assert claude_analyst._seen_utc_str(None) is None
        assert claude_analyst._seen_utc_str("") is None
        assert claude_analyst._seen_utc_str("   ") is None
        assert claude_analyst._seen_utc_str("not-a-date") is None


class TestBuildPayloadSeenToken:
    def test_real_article_gets_seen_token(self):
        arts = [{
            "title": "Micron guides Q4 DRAM ASP up 20%",
            "source": "rss", "ai_score": 9.0,
            "summary": "body",
            "first_seen": "2026-05-18T14:32:09+00:00",
        }]
        payload = claude_analyst._build_payload(arts, {}, [])
        assert "[seen 14:32 UTC]" in payload
        # token sits between score and source on the row
        assert "[score=9.0] [seen 14:32 UTC] [rss]" in payload

    def test_synthetic_snapshot_row_has_no_seen_token(self):
        # Exactly the shape daemon.heartbeat_worker prepends — no first_seen.
        arts = [
            {"title": "PORTFOLIO P&L SNAPSHOT", "source": "portfolio",
             "summary": "snap", "ai_score": 10},
            {"title": "Real headline about NVDA earnings beat",
             "source": "rss", "ai_score": 8.0, "summary": "b",
             "first_seen": "2026-05-18T09:07:00+00:00"},
        ]
        payload = claude_analyst._build_payload(arts, {}, [])
        # the synthetic row renders, but with NO fabricated time
        assert "PORTFOLIO P&L SNAPSHOT" in payload
        snap_line = next(l for l in payload.splitlines()
                         if "PORTFOLIO P&L SNAPSHOT" in l)
        assert "seen" not in snap_line
        assert "UTC" not in snap_line
        # the real row still gets its real clock
        assert "[seen 09:07 UTC]" in payload

    def test_seen_token_survives_syndication_collapse(self):
        # Two syndicated copies of one story (same 8-token signature) collapse
        # to the higher-score rep; first_seen must survive the shallow copy so
        # the surviving row still carries its wire clock.
        arts = [
            {"title": "Fed holds rates steady amid inflation concerns",
             "source": "GDELT/reuters.com", "ai_score": 7.0, "summary": "a",
             "first_seen": "2026-05-18T03:01:00+00:00"},
            {"title": "Fed holds rates steady amid inflation concerns",
             "source": "scraped/finance.yahoo.com", "ai_score": 9.0,
             "summary": "b", "first_seen": "2026-05-18T03:11:00+00:00"},
        ]
        payload = claude_analyst._build_payload(arts, {}, [])
        # one collapsed row, score=9.0 rep, its own first_seen (03:11)
        assert payload.count("Fed holds rates steady") == 1
        assert "[score=9.0] [seen 03:11 UTC]" in payload
        assert "[syndicated x2]" in payload

    def test_build_payload_does_not_mutate_input_dicts(self):
        # heartbeat_worker feeds this exact list onward to the briefing-label /
        # training path. The feature must be read-only on the dicts.
        a = {"title": "headline that is long enough", "source": "rss",
             "ai_score": 6.0, "summary": "x",
             "first_seen": "2026-05-18T12:00:00+00:00"}
        before = dict(a)
        claude_analyst._build_payload([a], {}, [])
        assert a == before  # no _seen / mutated keys leaked back


class TestAnalyzeStillWorks:
    def test_analyze_passes_payload_with_seen_token_to_claude(self):
        """End-to-end: analyze() builds the payload (now with seen tokens) and
        hands it to claude_call — the token must reach the prompt."""
        arts = [{"title": "AXTI lands new gallium-arsenide supply deal",
                 "source": "finnhub", "ai_score": 8.0, "summary": "deal",
                 "first_seen": "2026-05-18T22:45:00+00:00"}]
        with patch.object(claude_analyst, "claude_call",
                          return_value="BRIEF") as cc:
            out = claude_analyst.analyze(arts, {}, [])
        assert out == "BRIEF"
        sent_prompt = cc.call_args[0][0]
        assert "[seen 22:45 UTC]" in sent_prompt
