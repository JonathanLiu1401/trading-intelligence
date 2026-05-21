"""Per-held-ticker BREAKING-alert burst awareness on the 🚨 BREAKING path.

When the wire concentrates on a single held name, the analyst gets a rapid
series of distinct BREAKING pushes. The existing gates already collapse
exact-signature dupes / paraphrases / wire syndication — but a series of
genuinely different headlines about the same earnings event (revenue beat,
then guidance, then buyback, then segment colour) are NOT duplicates and
correctly fire as separate alerts. Each currently presents as a fresh
break, though, so the 4th distinct NVDA earnings push reads identically
to the 1st.

This suite pins the contract of the *non-suppressing* burst annotation:

  1. ``alert_recency.ticker_burst_counts`` is pure (no DB / IO) and counts
     case-insensitive word-boundary matches of each ticker across the
     stored ``title`` of every recent alert. Pin the counting logic in
     isolation.
  2. The threshold-based emission in ``alert_agent._fmt`` only adds a
     ``burst:`` line when a held-book ticker has >= ``BURST_MIN_PRIOR_ALERTS``
     prior pushes; below the bar the line is silent (no chat filler).
  3. The ``burst:`` line names exactly which held tickers cleared the bar
     and the count for each (multiple tickers compose with ``; ``).
  4. ``BURST WIRE`` prompt rule is present in ``ALERT_PROMPT`` so the LLM
     has the contract that drives the annotation.
  5. Best-effort: a ``recent_alerts`` failure / empty input degrades to
     no annotation (a missed alert is worse than missed framing).
  6. The annotation is read-only — it never touches articles.db,
     ai_score, ml_score, score_source, or the urgency state machine.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from watchers import alert_agent, alert_recency


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def _recent(title: str, age_h: float = 1.0, sig: str | None = None) -> dict:
    """Build a recent-alert dict like ``recent_alerts`` returns."""
    return {"title": title, "age_hours": age_h, "sig": sig or "_sig_" + title}


def _row(_id="x", title="generic", source="rss", summary="", **kw) -> dict:
    base = {
        "_id": _id, "link": f"https://news.example.com/{_id}",
        "title": title, "source": source, "ai_score": 9.0,
        "summary": summary, "published": _iso(0.2), "first_seen": _iso(0.1),
    }
    base.update(kw)
    return base


# ── ticker_burst_counts pure function ──────────────────────────────────────


class TestTickerBurstCountsPure:
    """Unit-pin the counting logic in isolation. The caller in alert_agent
    drives the prompt rule on this contract — drifting either side breaks
    the analyst's burst framing without any other test catching it."""

    def test_no_recent_no_tickers_returns_empty(self):
        assert alert_recency.ticker_burst_counts([], []) == {}
        assert alert_recency.ticker_burst_counts(
            [_recent("NVDA earnings beat")], []
        ) == {}
        assert alert_recency.ticker_burst_counts([], ["NVDA"]) == {}

    def test_counts_ticker_across_multiple_recent_alerts(self):
        recent = [
            _recent("NVDA earnings beat estimates"),
            _recent("nvda adds $80B buyback program"),         # case-insensitive match
            _recent("Nvidia guidance lifts; data center surge"),  # NO ticker — company name only
            _recent("$NVDA segment colour breakdown"),         # leading $ — \b still matches
            _recent("MU memory pricing softens after Samsung deal"),
        ]
        # Matches the TICKER SYMBOL (\bNVDA\b case-insensitive), not the
        # company name "Nvidia" — that decision is documented in the helper
        # and pinned here so a future "company-name match" change would
        # need to update this test and re-justify the looser semantics.
        counts = alert_recency.ticker_burst_counts(recent, ["NVDA", "MU"])
        assert counts.get("NVDA") == 3, f"NVDA count wrong: {counts}"
        assert counts.get("MU") == 1, f"MU count wrong: {counts}"

    def test_per_alert_dedup_same_ticker_mentioned_twice_counts_once(self):
        # One alert mentions NVDA twice in title — should still be 1.
        recent = [_recent("NVDA Q1 NVDA buyback NVDA segment")]
        assert alert_recency.ticker_burst_counts(recent, ["NVDA"]) == {"NVDA": 1}

    def test_word_boundary_prevents_substring_false_positives(self):
        recent = [
            _recent("MUUSE memory analysis"),       # MU substring — must NOT match MU
            _recent("DAMD analysis"),               # AMD substring — must NOT match AMD
            _recent("$MU shares jump"),             # leading $ — boundary works
            _recent("MU reports earnings"),         # word boundary — match
        ]
        counts = alert_recency.ticker_burst_counts(recent, ["MU", "AMD"])
        # MU should be 2 ($MU, MU reports) — MUUSE substring excluded.
        assert counts.get("MU") == 2, f"MU substring leak: {counts}"
        # AMD should be missing (zero) — DAMD substring excluded.
        assert "AMD" not in counts, f"AMD substring leak: {counts}"

    def test_case_insensitive_matches(self):
        recent = [_recent("nvidia announces buyback")]
        counts = alert_recency.ticker_burst_counts(recent, ["NVDA"])
        # NVDA literal isn't in title — only "nvidia". So NVDA count should be 0.
        # (The fingerprint is ticker symbols, not company names — same shape
        # as ml.features._LIVE_RE: pin this so a future "company-name match"
        # change doesn't silently leak generic company-name false positives.)
        assert counts.get("NVDA", 0) == 0

        recent2 = [_recent("nvda is up"), _recent("NVDA earnings")]
        counts2 = alert_recency.ticker_burst_counts(recent2, ["NVDA"])
        assert counts2.get("NVDA") == 2  # both cases match

    def test_missing_title_skipped_without_crash(self):
        recent = [
            _recent("NVDA earnings beat"),
            {"title": "", "age_hours": 1.0},
            {"title": None, "age_hours": 1.0},
            {"age_hours": 1.0},  # no title key at all
        ]
        counts = alert_recency.ticker_burst_counts(recent, ["NVDA"])
        assert counts == {"NVDA": 1}

    def test_uppercases_input_tickers_and_deduplicates(self):
        recent = [_recent("NVDA earnings beat")]
        # lowercase / mixed-case input + duplicate
        counts = alert_recency.ticker_burst_counts(
            recent, ["nvda", "NVDA", "Nvda"]
        )
        # Should not double-count nor crash; one normalised "NVDA" key
        assert counts == {"NVDA": 1}


# ── _fmt integration: burst line emission threshold ─────────────────────────


def _send_capture_prompt(rows, recent_alerts_data=None):
    """Drive ``send_urgent_alert`` with mocks and capture the rendered prompt.

    Returns ``prompt`` (str) — the verbatim text passed to Claude — or
    ``None`` if the call short-circuited before claude_call was invoked.
    Mirrors the spy shape used by other alert tests."""

    class _Store:
        def __init__(self):
            self.marked: list[str] = []

        def mark_alerted_batch(self, ids):
            self.marked.extend(ids)

        def mark_alerted(self, aid):
            self.marked.append(aid)

        def ticker_mention_velocity(self, tickers, window_min=60):
            # Pretend nothing else is being mentioned right now — isolates
            # the burst-vs-velocity signal so the test asserts burst alone.
            return []

    captured: dict[str, str] = {}

    def fake_claude(prompt, **kw):
        captured["prompt"] = prompt
        return "[mock alert body]"

    def fake_discord(*args, **kwargs):
        return True

    recent = recent_alerts_data if recent_alerts_data is not None else []

    with patch.object(alert_agent, "DISCORD_WEBHOOK", "https://webhook.test"), \
         patch.object(alert_agent, "claude_call", side_effect=fake_claude), \
         patch.object(alert_agent, "discord_send", create=True, side_effect=fake_discord), \
         patch.object(alert_recency, "recent_alerts", return_value=recent), \
         patch.object(alert_recency, "recent_signatures", return_value=set()), \
         patch.object(alert_recency, "record_alerted", return_value=0):
        store = _Store()
        # discord_notifier.send is imported inside send_urgent_alert; patch both
        with patch("notifier.discord_notifier.send", side_effect=fake_discord):
            alert_agent.send_urgent_alert(rows, store)
    return captured.get("prompt")


class TestBurstLineEmission:
    """Pin the burst:` line into the prompt only when the threshold is met."""

    def test_below_threshold_no_burst_line(self):
        # Only 2 prior NVDA alerts — below BURST_MIN_PRIOR_ALERTS=3.
        recent = [
            _recent("NVDA Q1 earnings beat", age_h=1.0),
            _recent("NVDA adds $80B buyback", age_h=0.5),
        ]
        rows = [_row(
            _id="alert1",
            title="Nvidia Vera Rubin GPU details revealed",
            summary="NVDA discloses new chip roadmap details.",
        )]
        prompt = _send_capture_prompt(rows, recent_alerts_data=recent)
        assert prompt is not None, "alert should have fired"
        assert "\nburst:" not in prompt, (
            f"unexpected burst line below threshold:\n{prompt}"
        )

    def test_at_threshold_burst_line_appears(self):
        # 3 prior NVDA pushes = threshold met. Note: matches the SYMBOL
        # (NVDA) not the company name ("Nvidia"); the recap titles below
        # use the literal ticker so they pin the threshold count.
        recent = [
            _recent("NVDA Q1 earnings beat", age_h=2.0),
            _recent("NVDA adds $80B buyback", age_h=1.5),
            _recent("NVDA data center segment colour", age_h=1.0),
        ]
        rows = [_row(
            _id="alert1",
            title="Nvidia Vera Rubin GPU full-cycle supply constraints",
            summary="NVDA discloses new chip roadmap details.",
        )]
        prompt = _send_capture_prompt(rows, recent_alerts_data=recent)
        assert prompt is not None
        assert "\nburst:" in prompt, (
            f"missing burst line at threshold:\n{prompt}"
        )
        assert "NVDA: 3 prior BREAKING alerts" in prompt, (
            f"burst count wrong:\n{prompt}"
        )

    def test_multiple_tickers_compose_with_semicolons(self):
        recent = [
            _recent("NVDA Q1 earnings beat", age_h=2.0),
            _recent("NVDA adds $80B buyback", age_h=1.5),
            _recent("NVDA data center colour", age_h=1.0),
            _recent("MU Samsung deal news", age_h=2.0),
            _recent("MU memory pricing flat", age_h=1.0),
            _recent("MU guidance lifts", age_h=0.5),
        ]
        rows = [_row(
            _id="alert1",
            title="MU upgrades on Nvidia datacenter pull-through",
            summary="MU NVDA semiconductor news.",
        )]
        prompt = _send_capture_prompt(rows, recent_alerts_data=recent)
        assert prompt is not None
        assert "\nburst:" in prompt
        # Both held tickers should appear in the burst line.
        burst_line = [
            ln for ln in prompt.splitlines() if ln.startswith("burst:")
        ][0]
        assert "MU:" in burst_line, f"MU missing: {burst_line!r}"
        assert "NVDA:" in burst_line, f"NVDA missing: {burst_line!r}"
        assert "; " in burst_line, f"separator missing: {burst_line!r}"

    def test_no_book_no_burst_line(self):
        # Row touches no held ticker; even if recent alerts mention NVDA,
        # this row's prompt should not carry a burst: line (the analyst's
        # held-book is what matters).
        recent = [
            _recent("NVDA earnings", age_h=1.0),
            _recent("NVDA buyback", age_h=0.5),
            _recent("NVDA data center", age_h=0.1),
            _recent("NVDA segment colour", age_h=0.05),
        ]
        rows = [_row(
            _id="alert1",
            title="Fed signals possible pause on rate hikes",
            summary="Macro: no held ticker mentioned.",
        )]
        prompt = _send_capture_prompt(rows, recent_alerts_data=recent)
        assert prompt is not None
        assert "\nburst:" not in prompt, (
            f"burst line emitted on non-book row:\n{prompt}"
        )

    def test_empty_recent_degrades_silently_no_burst(self):
        # Best-effort path: no recent_alerts → no burst signal → no annotation.
        rows = [_row(
            _id="alert1",
            title="Nvidia adds $80B buyback program",
            summary="NVDA capital return news.",
        )]
        prompt = _send_capture_prompt(rows, recent_alerts_data=[])
        assert prompt is not None
        assert "\nburst:" not in prompt


class TestPromptContract:
    """The BURST WIRE rule must be in the ALERT_PROMPT so the LLM knows what
    to do with the ``burst:`` line. This pin ensures a future prompt edit
    that drops the rule is caught (the annotation without the rule is
    dead weight)."""

    def test_burst_wire_rule_present_in_prompt(self):
        assert "BURST WIRE:" in alert_agent.ALERT_PROMPT, (
            "ALERT_PROMPT missing BURST WIRE rule"
        )

    def test_burst_wire_rule_names_development_verbs(self):
        # The framing verbs the LLM is told to use. Anchored so the rule
        # can't be reduced to a vague "frame as continuation" — the
        # analyst's complaint is non-specific framing.
        rule = alert_agent.ALERT_PROMPT
        for verb in ("DETAILS", "ADDS", "FOLLOWS", "EXTENDS"):
            assert verb in rule, f"BURST WIRE verb {verb!r} missing"
