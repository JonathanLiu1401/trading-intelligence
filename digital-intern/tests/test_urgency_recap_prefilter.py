"""Recap-template pre-filter on the urgency_scorer (Sonnet) path.

Sibling to ``tests/test_alert_recap_template.py`` (the alert formatter gate)
and ``tests/test_briefing_recap_template.py`` (the briefing payload gate).
Same fingerprint set, same must-survive corpus — three surfaces share ONE
source of truth (``watchers.alert_agent._looks_like_recap_template``) so a
regex tightening on the alert path can never silently re-admit recap
templates here.

The discriminating asserts:

  1. A recap-template article in the input is floored to ``ai_score=0.01,
     urgency=0, score_source='llm'`` WITHOUT calling ``claude_call`` —
     proves the pre-filter actually saves Sonnet quota and stops the row
     from being mis-labeled urgent in the training pool.
  2. A real urgent article in the same batch IS still sent to Sonnet and
     receives its Sonnet score.
  3. An all-recap batch returns 0 urgent and never calls Sonnet at all
     (the analyst-facing "wasted-call rate" goes to zero on a pathological
     window).
  4. The must-survive corpus (real earnings beats, Fed cuts, mid-sentence
     "why", earnings PREVIEWS, value/analyst headlines) is NEVER pre-floored
     — they reach Sonnet exactly as they did before this fix.
  5. Lockstep parity: the live-noise titles from the alert test corpus are
     ALSO caught here (a regex drift on the shared gate fails both suites).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from watchers import urgency_scorer, alert_agent


def _insert(store, *, id, url=None, title="t", source="rss",
            kw_score=1.0, urgency=0, first_seen=None):
    if first_seen is None:
        first_seen = (
            datetime.now(timezone.utc) - timedelta(minutes=5)
        ).isoformat()
    if url is None:
        url = f"https://example.com/{id}"
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            "first_seen, cycle) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", kw_score, 0.0, urgency,
             first_seen, 0),
        )
        store.conn.commit()


def _patched_claude(response):
    body = json.dumps(response)
    return patch.object(urgency_scorer, "claude_call", return_value=body)


# ── Live-noise pre-floor (the actual win) ───────────────────────────────────


class TestRecapPreFloorSkipsSonnet:
    def test_recap_row_floored_to_noise_without_calling_sonnet(self, store):
        """The single-row case the live evidence directly documents: a
        recap-template article must exit with ai_score=0.01 / urgency=0 /
        score_source='llm' AND zero Sonnet calls — proves the pre-filter
        actually engages BEFORE claude_call."""
        _insert(store, id="r1",
                title="Why Did Micron Stock Drop Today ? | The Motley Fool")
        articles = [{"_id": "r1",
                     "title": "Why Did Micron Stock Drop Today ? | The Motley Fool",
                     "summary": ""}]
        with patch.object(urgency_scorer, "claude_call") as mock_claude:
            n_urgent = urgency_scorer.score_batch(articles, store)
        assert n_urgent == 0
        assert mock_claude.call_count == 0, (
            "Sonnet was called on a recap-template row — pre-filter did not engage"
        )
        row = store.conn.execute(
            "SELECT ai_score, urgency, score_source FROM articles WHERE id='r1'"
        ).fetchone()
        assert row[0] == pytest.approx(0.01)
        assert row[1] == 0
        assert row[2] == "llm", (
            "recap row missing score_source='llm' — it would re-enter "
            "get_unscored every cycle and waste Sonnet calls forever"
        )

    def test_all_recap_batch_skips_sonnet_entirely(self, store):
        """A whole batch of recap-template noise (the pathological window
        the live evidence documents — 10 in 24h) must short-circuit BEFORE
        the Sonnet call. Hard guard against an analyst-facing wasted-quota
        spike on a recap-heavy window."""
        for i, t in enumerate([
            "Why Nvidia (NVDA) Stock Is Trading Up Today",
            "Stock Market Today, May 18: Micron Falls",
            "D-Wave Quantum Inc (QBTS) Q1 2026 Earnings Call Highlights",
            "Lumentum Holdings Inc (LITE) Shares Fall 8.8% -- GF Value Says",
            "Here What the Street Thinks About NVIDIA",
            "Why Did AMD Stock Surge Today",
        ]):
            _insert(store, id=f"r{i}", title=t)
        articles = [
            {"_id": f"r{i}", "title": t, "summary": ""}
            for i, t in enumerate([
                "Why Nvidia (NVDA) Stock Is Trading Up Today",
                "Stock Market Today, May 18: Micron Falls",
                "D-Wave Quantum Inc (QBTS) Q1 2026 Earnings Call Highlights",
                "Lumentum Holdings Inc (LITE) Shares Fall 8.8% -- GF Value Says",
                "Here What the Street Thinks About NVIDIA",
                "Why Did AMD Stock Surge Today",
            ])
        ]
        with patch.object(urgency_scorer, "claude_call") as mock_claude:
            n_urgent = urgency_scorer.score_batch(articles, store)
        assert n_urgent == 0
        assert mock_claude.call_count == 0, (
            "Sonnet was called on an all-recap batch — the pathological "
            "wasted-quota window the fix exists to prevent"
        )
        # Every row should have been floored.
        rows = {
            r[0]: (r[1], r[2], r[3]) for r in store.conn.execute(
                "SELECT id, ai_score, urgency, score_source FROM articles"
            ).fetchall()
        }
        for i in range(6):
            ai, urg, src = rows[f"r{i}"]
            assert ai == pytest.approx(0.01), (
                f"recap row {i} not floored: ai_score={ai}"
            )
            assert urg == 0
            assert src == "llm"

    def test_mixed_batch_sonnet_gets_only_real_rows(self, store):
        """A batch with one recap + one real headline: Sonnet sees only the
        real one in its payload, the real one gets its real score, and the
        recap one is floored to 0.01."""
        _insert(store, id="recap",
                title="Why Did Micron Stock Drop Today ? | The Motley Fool")
        _insert(store, id="real", title="MU earnings beat Q3 estimates")
        articles = [
            {"_id": "recap",
             "title": "Why Did Micron Stock Drop Today ? | The Motley Fool",
             "summary": ""},
            {"_id": "real",
             "title": "MU earnings beat Q3 estimates", "summary": ""},
        ]
        # Sonnet should see ONLY the real article — index 0 in the payload.
        captured: dict = {}

        def _capture(prompt, **kwargs):
            captured["prompt"] = prompt
            return json.dumps([
                {"index": 0, "score": 9.5, "reason": "earnings"}
            ])

        with patch.object(urgency_scorer, "claude_call", side_effect=_capture):
            n_urgent = urgency_scorer.score_batch(articles, store)
        assert n_urgent == 1, "real urgent row not classified urgent"
        # The recap title must NOT appear in the Sonnet prompt — proves it
        # was excluded from the LLM payload, not silently passed through.
        assert "Why Did Micron Stock Drop Today" not in captured["prompt"]
        assert "MU earnings beat Q3 estimates" in captured["prompt"]
        rows = {
            r[0]: (r[1], r[2]) for r in store.conn.execute(
                "SELECT id, ai_score, urgency FROM articles"
            ).fetchall()
        }
        assert rows["recap"][0] == pytest.approx(0.01)
        assert rows["recap"][1] == 0
        assert rows["real"][0] == pytest.approx(9.5)
        assert rows["real"][1] == 1


# ── Must-survive corpus (no false-positives on real news) ───────────────────


class TestMustSurviveReachesSonnet:
    """Real breaking headlines MUST reach Sonnet exactly as before the fix.
    Mirrors the alert-path test_alert_recap_template.py must-survive corpus
    so a regex tightening that catches one of these fails both suites."""

    def _run(self, store, title):
        _insert(store, id="s", title=title)
        articles = [{"_id": "s", "title": title, "summary": ""}]
        with _patched_claude([{"index": 0, "score": 9.0, "reason": "real"}]):
            urgency_scorer.score_batch(articles, store)
        row = store.conn.execute(
            "SELECT ai_score, urgency FROM articles WHERE id='s'"
        ).fetchone()
        return row

    def test_real_earnings_movers_survive(self, store):
        ai, urg = self._run(store, "MU earnings blow past Q3 estimates sharply")
        assert ai == pytest.approx(9.0), (
            "real earnings headline was incorrectly pre-floored"
        )
        assert urg == 1

    def test_macro_breaking_survives(self, store):
        ai, urg = self._run(store, "Fed cuts rates by 50bp, citing labor weakness")
        assert ai == pytest.approx(9.0)
        assert urg == 1

    def test_ticker_action_survives(self, store):
        ai, urg = self._run(store, "MU shares halted on pending news")
        assert ai == pytest.approx(9.0)
        assert urg == 1

    def test_question_form_mid_sentence_survives(self, store):
        """Mid-sentence "why" is not the recap template — must reach Sonnet."""
        ai, urg = self._run(
            store, "Why investors are bullish on Nvidia ahead of earnings"
        )
        assert ai == pytest.approx(9.0)
        assert urg == 1

    def test_earnings_preview_not_recap_survives(self, store):
        """A PREVIEW of an upcoming call is forward-looking and analyst-
        actionable — it must NOT be caught by the call-highlights pattern."""
        ai, urg = self._run(
            store, "Nvidia Q1 earnings preview: all eyes on data center"
        )
        assert ai == pytest.approx(9.0)
        assert urg == 1

    def test_value_analyst_headline_survives(self, store):
        """A real analyst-rating headline must reach Sonnet — the GF Value
        pattern is precision-targeted to the GuruFocus mill."""
        ai, urg = self._run(
            store, "Zscaler (NASDAQ:ZS) Price Target Cut to $223.00"
        )
        assert ai == pytest.approx(9.0)
        assert urg == 1


# ── Lockstep parity with the alert-path gate (anti-drift) ───────────────────


def test_urgency_scorer_uses_alert_agent_gate():
    """All three surfaces (alert formatter, briefing builder, urgency
    scorer pre-filter) MUST resolve recap-template fingerprints through
    the SAME ``_looks_like_recap_template`` function — single source of
    truth. A future agent that copies/forks the patterns into urgency_scorer
    breaks this assertion. Mirrors the
    ``test_briefing_recap_template.test_alert_and_briefing_gates_agree``
    anti-drift discipline, extended to 3 surfaces.
    """
    assert (urgency_scorer._looks_like_recap_template
            is alert_agent._looks_like_recap_template), (
        "urgency_scorer imported a fork of _looks_like_recap_template — "
        "fingerprint drift across alert/briefing/scorer surfaces is now possible"
    )


def test_lockstep_with_alert_path_on_live_noise():
    """The six live-noise titles caught on the alert path 2026-05-18/19 are
    ALSO caught on the urgency-scorer pre-filter (same gate, by parity).
    A regex tightening that re-admits any of these fails both suites in
    lockstep."""
    titles = [
        "Why Nvidia (NVDA) Stock Is Trading Up Today",
        "Why Did Micron Stock Drop Today ? | The Motley Fool",
        ("Stock Market Today, May 18: Micron Falls as Memory Concerns "
         "Test AI Rally"),
        "D-Wave Quantum Inc (QBTS) Q1 2026 Earnings Call Highlights",
        "Here What the Street Thinks About ​NVIDIA Corporation ( NVDA )",
        ("Lumentum Holdings Inc (LITE) Shares Fall 8.8% -- GF Value Says S "
         "- GuruFocus"),
    ]
    for t in titles:
        hit, _ = urgency_scorer._looks_like_recap_template({"title": t})
        assert hit, f"urgency_scorer pre-filter missed live recap: {t!r}"
