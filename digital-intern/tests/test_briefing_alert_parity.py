"""Alert↔briefing parity tag in the 5h Opus digest.

A news analyst reading the 5h heartbeat digest cannot today tell a genuinely
NEW LEAD from a rehash of a story they were ALREADY pushed as a standalone
🚨 BREAKING alert hours ago. ``watchers.alert_recency`` already persists the
canonical ``alert_dedup._signature`` of every story that actually fired (TTL
``ALERT_RECENCY_TTL_HOURS`` = 6h ≈ the 5h briefing window) and uses it for
cross-cycle alert suppression — but the briefing path never consulted it, so
the analyst's #1 complaint (duplicate / repeated alerts) reached the digest
LEAD unmitigated.

``analysis.claude_analyst._build_payload`` now:
  * reads the recent fired-alert signature set ONCE per briefing
    (``_recent_alert_signatures`` — best-effort, ``set()`` on any failure, a
    single read of a separate alert_recency.db, NEVER articles.db);
  * tags a digest row ``[ALERTED]`` when its headline's canonical signature
    (the SAME ``alert_dedup._signature`` the suppression path uses — so the
    tag and that gate agree by construction) is in that set;
  * SYSTEM_PROMPT instructs Opus to never LEAD with an ``[ALERTED]`` row over
    a comparable untagged one and to frame it as continuation.

Pure read-side: no DB write, no ai_score/ml_score/score_source/urgency
mutation, backtest excluded upstream by ``get_top_for_briefing``'s
``_LIVE_ONLY_CLAUSE`` — all four load-bearing invariants intact by
construction. Specific-value pins, not "no crash".
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from analysis import claude_analyst
from watchers.alert_dedup import _signature


def _recent_iso(minutes_ago: int = 5) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


def _row(**kw) -> dict:
    """A digest row shaped exactly like ``get_top_for_briefing`` returns."""
    base = {
        "_id": "x",
        "link": "https://reuters.com/x",
        "title": "MU guides Q4 revenue sharply above the Street consensus",
        "source": "rss",
        "ai_score": 9.0,
        "_relevance_score": 4.0,
        "summary": "Micron lifted its outlook citing HBM demand.",
        "first_seen": _recent_iso(),
        "time_sensitivity": 0.5,
        "_llm_vetted": True,
    }
    base.update(kw)
    return base


def _payload(articles, monkeypatch, recent_sigs):
    """Drive _build_payload with a controlled recent-alert signature set."""
    monkeypatch.setattr(
        claude_analyst.alert_recency, "recent_signatures",
        lambda *a, **k: set(recent_sigs),
    )
    return claude_analyst._build_payload(articles, {}, [])


class TestParityTag:
    def test_alerted_story_is_tagged(self, monkeypatch):
        title = "Drone strike on UAE nuclear plant sends Brent crude surging"
        arts = [_row(_id="hit", title=title, link="https://benzinga.com/oil")]
        out = _payload(arts, monkeypatch, {_signature(title)})
        assert "[ALERTED]" in out, "an already-pushed story must be tagged"
        # tag attaches to THAT row's line, not floating loose
        line = next(l for l in out.splitlines() if title[:30] in l)
        assert "[ALERTED]" in line

    def test_unalerted_story_is_not_tagged(self, monkeypatch):
        arts = [_row(_id="fresh", title="AXTI signs multi-year InP wafer supply deal")]
        # recent set contains a DIFFERENT story's signature
        out = _payload(arts, monkeypatch, {_signature("Totally unrelated Fed headline")})
        assert "[ALERTED]" not in out, "a fresh, never-pushed story must not be tagged"

    def test_wire_marker_variant_still_tagged(self, monkeypatch):
        """Reuses alert_dedup._signature verbatim, so a wire-revision variant
        ("UPDATE 2-...") of an alerted headline canonicalises to the same
        signature and is still recognised as already-pushed — proving the tag
        and the cross-cycle suppression agree by construction."""
        alerted = "Micron shares surge after Q3 earnings blowout"
        variant = "UPDATE 2-Micron shares surge after Q3 earnings blowout (Reuters)"
        # precondition: the two canonicalise identically (the shared primitive)
        assert _signature(variant) == _signature(alerted)
        arts = [_row(_id="v", title=variant, link="https://gdelt/x")]
        out = _payload(arts, monkeypatch, {_signature(alerted)})
        assert "[ALERTED]" in out

    def test_distinct_same_ticker_events_do_not_false_tag(self, monkeypatch):
        """The discriminating case: the same ticker with a DIFFERENT event
        must NOT be silenced. A prior "MU surges 8% on Q3 beat" alert must not
        false-tag a fresh "MU drops 6% on guidance cut" LEAD — otherwise Opus
        skips a legitimately new story. Pins that _signature is a
        full-headline canonicalisation, not an over-aggressive ticker prefix."""
        alerted = "MU surges 8% after Q3 earnings beat"
        other = "MU drops 6% after guidance cut warning"
        assert _signature(alerted) != _signature(other), \
            "precondition: distinct same-ticker events have distinct signatures"
        arts = [_row(_id="o", title=other, link="https://reuters.com/y")]
        out = _payload(arts, monkeypatch, {_signature(alerted)})
        assert "[ALERTED]" not in out, \
            "a different same-ticker event must not be false-tagged as pushed"

    def test_snapshot_rows_never_tagged(self, monkeypatch):
        """The prepended PORTFOLIO/OPTIONS snapshot rows carry no link/url;
        even if their title signature collided with an alerted one they must
        pass through clean (same guard as _extract_briefing_labels)."""
        snap_title = "PORTFOLIO P&L SNAPSHOT"
        snap = {"title": snap_title, "source": "portfolio",
                "summary": "LITE +2%", "ai_score": 10}
        real = _row(_id="r", title="Fed delivers surprise 50bp emergency cut",
                    link="https://reuters.com/fed")
        out = _payload([snap, real], monkeypatch,
                       {_signature(snap_title), _signature("Fed delivers surprise 50bp emergency cut")})
        snap_line = next(l for l in out.splitlines() if "PORTFOLIO P&L SNAPSHOT" in l)
        assert "[ALERTED]" not in snap_line, "snapshot row must never be tagged"
        # the real alerted row IS still tagged in the same payload
        assert "[ALERTED]" in out

    def test_empty_recent_set_degrades_to_no_tag(self, monkeypatch):
        arts = [_row(_id="a", title="MU earnings blow past Q3 estimates")]
        out = _payload(arts, monkeypatch, set())
        assert "[ALERTED]" not in out, "no recent alerts → behaviour unchanged"

    def test_recent_signatures_failure_is_swallowed(self, monkeypatch):
        """A broken/locked alert_recency.db must NOT break or delay the 5h
        briefing — _recent_alert_signatures returns set(), payload still
        builds, no tag."""
        def _boom(*a, **k):
            raise RuntimeError("alert_recency.db locked")
        monkeypatch.setattr(claude_analyst.alert_recency,
                             "recent_signatures", _boom)
        assert claude_analyst._recent_alert_signatures() == set()
        out = claude_analyst._build_payload(
            [_row(_id="z", title="Some urgent MU headline here today")], {}, []
        )
        assert "NEWSWIRE" in out and "[ALERTED]" not in out

    def test_input_list_not_mutated(self, monkeypatch):
        title = "NVDA H200 export approval expands to 10 China firms"
        arts = [_row(_id="m", title=title)]
        snapshot = dict(arts[0])
        _payload(arts, monkeypatch, {_signature(title)})
        assert arts[0] == snapshot, "parity tag must be read-side only (no row mutation)"


class TestSystemPromptRule:
    def test_alerted_rule_present_with_lead_consequence(self):
        sp = claude_analyst.SYSTEM_PROMPT
        assert "[ALERTED]" in sp
        low = sp.lower()
        assert "lead" in low and "continuation" in low, (
            "the rule must state the LEAD/continuation consequence, not just "
            "define the tag"
        )


def test_recent_alert_signatures_returns_set(monkeypatch):
    monkeypatch.setattr(claude_analyst.alert_recency, "recent_signatures",
                        lambda *a, **k: {"sig1", "sig2"})
    assert claude_analyst._recent_alert_signatures() == {"sig1", "sig2"}
