"""LLM-vetted vs model-only score calibration tag in the 5h Opus digest.

`article_store.get_top_for_briefing` ranks the newswire by
`COALESCE(NULLIF(ai_score,0), ml_score, 0)` — so a row the LOCAL relevance
model scored 9.8 and a row Sonnet/Opus vetted at 9 render with the same
`[score=...]` and the COALESCE erases which is which. The relevance head
demonstrably over-scores forum/wiki/social rows (a recurring live finding:
reddit 9.8 / wikipedia 8.6 with `ai_score=0, ml_score` high). The alert path
gates that noise (`_filter_low_authority_lone`), but the BRIEFING newswire
Opus reads did not expose the distinction at all, so neither Opus nor the
consuming analyst could down-weight a raw-model 9.8 against an LLM-vetted 9.

Two additive, read-only pieces, all four load-bearing invariants intact by
construction (no DB write, no ai_score/ml_score/score_source mutation,
backtest excluded upstream by `_LIVE_ONLY_CLAUSE`):

  * `get_top_for_briefing` adds `_llm_vetted = bool(raw ai_score)` — True iff
    a real Opus/Sonnet label exists (model predictions only ever write
    ml_score, never ai_score: invariant #2), without changing the displayed
    `ai_score` field or any ordering/diversity/decay logic;
  * `_build_payload` renders a ` [model]` token when `_llm_vetted is False`,
    and SYSTEM_PROMPT instructs Opus to prefer untagged rows for LEAD /
    TOP SIGNALS.

Specific-value pins, not "no crash". Calibration signal for the documented
failure mode — NOT a claim that it changes any particular healthy briefing.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

from analysis import claude_analyst


def _recent_iso(minutes_ago: int = 5) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes_ago)).isoformat()


def _insert_raw(store, *, id, url, title, source, urgency=0, ai_score=0.0,
                ml_score=None, score_source=None, kw_score=1.0,
                first_seen=None, published=""):
    if first_seen is None:
        first_seen = _recent_iso()
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, published, kw_score, ai_score, urgency,
             first_seen, 0, ml_score, score_source),
        )
        store.conn.commit()


# ── store layer: _llm_vetted reflects the LLM-label presence exactly ─────────
class TestLlmVettedFlag:
    def _by_id(self, rows, _id):
        for r in rows:
            if r["_id"] == _id:
                return r
        raise AssertionError(f"{_id} not returned by get_top_for_briefing")

    def test_llm_scored_row_is_vetted(self, store):
        _insert_raw(store, id="llm1", url="https://r.com/a",
                    title="Sonnet vetted real market headline for digest",
                    source="rss", ai_score=8.0, score_source="llm")
        row = self._by_id(store.get_top_for_briefing(hours=24, limit=50), "llm1")
        assert row["_llm_vetted"] is True
        assert row["ai_score"] == 8.0  # display value unchanged

    def test_model_only_row_is_not_vetted_and_displays_ml_score(self, store):
        # The exact live failure mode: model over-scored a forum row; no LLM
        # label (ai_score=0), high ml_score, score_source='ml'.
        _insert_raw(store, id="ml1", url="https://reddit.com/x",
                    title="reddit r ValueInvesting MSFT is a screaming buy now",
                    source="reddit/r/ValueInvesting",
                    ai_score=0.0, ml_score=9.76, score_source="ml")
        row = self._by_id(store.get_top_for_briefing(hours=24, limit=50), "ml1")
        assert row["_llm_vetted"] is False
        # COALESCE still surfaces the (over-)score so ranking is unchanged.
        assert row["ai_score"] == 9.76

    def test_briefing_boost_row_is_vetted(self, store):
        _insert_raw(store, id="bb1", url="https://r.com/b",
                    title="Opus curated this into a prior heartbeat briefing",
                    source="rss", ai_score=4.5, score_source="briefing_boost")
        row = self._by_id(store.get_top_for_briefing(hours=24, limit=50), "bb1")
        assert row["_llm_vetted"] is True

    def test_sonnet_floored_noise_is_still_vetted(self, store):
        # urgency_scorer floors LLM-reviewed non-urgent noise at ai_score=0.01.
        # It WAS looked at by Sonnet → vetted (and correctly low), must NOT be
        # tagged [model]. bool(0.01) is True — the discriminating case.
        _insert_raw(store, id="fl1", url="https://r.com/c",
                    title="Sonnet reviewed this and scored it as plain noise",
                    source="rss", ai_score=0.01, score_source="llm")
        row = self._by_id(store.get_top_for_briefing(hours=24, limit=50), "fl1")
        assert row["_llm_vetted"] is True


# ── payload layer: the rendered [model] token ────────────────────────────────
def _newswire_line(payload: str, needle: str) -> str:
    for ln in payload.splitlines():
        if needle in ln and ln.lstrip()[:1].isdigit():
            return ln
    raise AssertionError(f"no numbered newswire row containing {needle!r}")


class TestModelTagRendering:
    def test_vetted_row_has_no_model_tag(self):
        arts = [{"title": "Fed signals surprise hold, yields whip lower",
                 "source": "rss", "ai_score": 8.0, "_llm_vetted": True,
                 "summary": "x"}]
        out = claude_analyst._build_payload(arts, {}, [])
        line = _newswire_line(out, "Fed signals surprise hold")
        assert "[model]" not in line
        assert "[score=8.0]" in line

    def test_model_only_row_is_tagged(self):
        arts = [{"title": "reddit thread says NVDA to 300 imminent for sure",
                 "source": "reddit/r/wallstreetbets", "ai_score": 9.8,
                 "_llm_vetted": False, "summary": "y"}]
        out = claude_analyst._build_payload(arts, {}, [])
        line = _newswire_line(out, "reddit thread says NVDA")
        assert "[score=9.8] [model]" in line, line

    def test_snapshot_row_without_flag_is_never_tagged(self):
        # daemon prepends PORTFOLIO/OPTIONS snapshots as raw dicts with NO
        # _llm_vetted key and ai_score=10 — .get → None, `None is False` is
        # False → must pass through untagged.
        arts = [{"title": "PORTFOLIO P&L SNAPSHOT", "source": "portfolio",
                 "ai_score": 10, "summary": "MU -6.6%"}]
        out = claude_analyst._build_payload(arts, {}, [])
        line = _newswire_line(out, "PORTFOLIO P&L SNAPSHOT")
        assert "[model]" not in line

    def test_mixed_cluster_tag_reflects_the_rendered_representative(self):
        # Same headline arrives twice: an LLM-vetted ai=7 copy and a
        # model-only ml-surfaced ai=9 copy. _collapse_syndicated keeps the
        # higher-_score copy (9, the model one) as the cluster rep — so the
        # rendered score IS the unvetted 9 and the tag MUST say [model].
        # Pins that vetting is NOT OR-ed across siblings (which would show a
        # 9 with no [model] — a false "trust this" calibration signal).
        title = "Treasury auction tails hard, ten year breaks four sixty"
        arts = [
            {"title": title, "source": "rss", "ai_score": 7.0,
             "_llm_vetted": True, "summary": "vetted copy"},
            {"title": title, "source": "reddit/r/bonds", "ai_score": 9.0,
             "_llm_vetted": False, "summary": "model copy"},
        ]
        out = claude_analyst._build_payload(arts, {}, [])
        line = _newswire_line(out, "Treasury auction tails hard")
        assert "[score=9.0] [model]" in line, line
        assert "[syndicated x2]" in line, line

    def test_build_payload_does_not_mutate_caller_dicts(self):
        # heartbeat_worker feeds this same list to the briefing-label /
        # training path — a read-only contract is load-bearing.
        a = {"title": "Genuine LLM vetted headline number one for digest",
             "source": "rss", "ai_score": 8.0, "_llm_vetted": True,
             "summary": "z"}
        b = {"title": "reddit only model surfaced this one no llm look",
             "source": "reddit/r/x", "ai_score": 9.1, "_llm_vetted": False,
             "summary": "w"}
        before = (dict(a), dict(b))
        claude_analyst._build_payload([a, b], {}, [])
        assert (a, b) == before


# ── SYSTEM_PROMPT must actually instruct Opus to act on the tag ──────────────
def test_system_prompt_documents_model_tag_with_consequence():
    sp = claude_analyst.SYSTEM_PROMPT
    assert "[model]" in sp
    # The rule is useless if it only defines the tag — it must state the
    # LEAD / TOP SIGNALS consequence (the advisor's explicit requirement).
    low = sp.lower()
    assert "lead" in low and "top signals" in low
    assert "local relevance model" in low or "no llm verification" in low
