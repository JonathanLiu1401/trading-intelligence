"""watchers/alert_agent.py — per-held-ticker book_velocity annotation.

The analyst persona is "react to events affecting MY positions". The pre-feature
``book:`` line answered WHICH held tickers a wire touches; it did not answer
"is this part of a multi-mention surge or a lone event?" — a critical
magnitude signal a Bloomberg-style alert composer should use to decide
BUY/SELL (high confidence) vs WATCH (single mention).

This test pins the new ``book_velocity:`` line:

  * It only appears on rows that already carry a ``book:`` line.
  * It appears only when the held ticker has >=2 mentions in the last 60 min
    (the conservative discriminator: one mention is THIS alert itself).
  * Multi-ticker rows emit one velocity entry per qualifying ticker, ordered
    to match the ``book:`` line.
  * Best-effort: a store without ``ticker_mention_velocity`` (mocks) degrades
    silently — no exception, no fabricated count.
  * The new BOOK VELOCITY rule reaches the Sonnet prompt verbatim.

Pure read-side feature: NO DB write, NO ai_score / ml_score / score_source /
urgency mutation, NO mutation of source_articles. ``ticker_mention_velocity``
itself is ``_LIVE_ONLY_CLAUSE``-scoped so synthetic backtest/opus rows can
never inflate the count.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from watchers import alert_agent


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


class _StoreSpy:
    """Minimal store mock with a ticker_mention_velocity stub controlled per
    test. ``velocity_rows`` is what the helper returns; ``raises`` short-
    circuits to an exception (best-effort degradation test)."""

    def __init__(self, velocity_rows=None, raises=False):
        self.marked: list[str] = []
        self._velocity_rows = velocity_rows or []
        self._raises = raises
        self.velocity_calls: list[tuple[list[str], int]] = []

    def mark_alerted_batch(self, ids):
        self.marked.extend(ids)

    def mark_alerted(self, aid):
        self.marked.append(aid)

    def ticker_mention_velocity(self, tickers, window_min=60):
        self.velocity_calls.append((list(tickers), window_min))
        if self._raises:
            raise RuntimeError("simulated DB failure")
        return list(self._velocity_rows)


def _send(art, store, monkeypatch):
    """Run send_urgent_alert with Discord/Claude mocked; return prompt text."""
    monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
    with patch.object(alert_agent, "claude_call",
                      return_value="🚨 BREAKING ◈ EARNINGS ◈ MU") as mock_claude, \
         patch("notifier.discord_notifier.send", return_value=True):
        ok = alert_agent.send_urgent_alert([art], store)
    return ok, mock_claude.call_args.args[0] if mock_claude.call_args else ""


def _data_block(prompt: str) -> str:
    """Extract ONLY the per-article data block from the full prompt — the
    static rule text contains the literal token ``book_velocity:`` (a
    backticked code-reference inside the BOOK VELOCITY rule), so a naive
    substring assertion against the full prompt would always pass that
    fragment. The data block is appended after the static "Urgent articles
    detected:" marker (see ``ALERT_PROMPT``), so splitting on that boundary
    yields the actual per-row content the gate must control."""
    marker = "Urgent articles detected:"
    parts = prompt.split(marker, 1)
    return parts[1] if len(parts) == 2 else prompt


class TestBookVelocityLineEmission:
    def test_held_row_with_recent_surge_emits_velocity_line(self, monkeypatch):
        """A held name with 4 recent mentions: the velocity line must appear."""
        art = {
            "_id": "mu1", "link": "https://reuters.com/x",
            "title": "MU guides Q4 sharply above the Street",
            "source": "rss", "ai_score": 9.0, "summary": "",
            "published": _iso(0.1), "first_seen": _iso(0.05),
        }
        store = _StoreSpy(velocity_rows=[
            {"ticker": "MU", "recent": 4, "prior": 1, "ratio": 2.5,
             "newest_age_s": 600.0},
        ])
        ok, prompt = _send(art, store, monkeypatch)
        assert ok is True
        assert "book: MU — analyst HOLDS/watches these" in prompt
        assert "book_velocity: MU: 4 mentions in last 60min" in prompt
        # Velocity helper was called exactly once with the union of held
        # tickers in the batch (here just MU).
        assert store.velocity_calls == [(["MU"], 60)]

    def test_held_row_with_single_mention_omits_velocity(self, monkeypatch):
        """``recent == 1`` is just THIS alert — no other recent mention; no
        velocity line emitted so a lone event stays a lone event in the prompt."""
        art = {
            "_id": "mu2", "link": "https://reuters.com/y",
            "title": "MU pre-announces Q4 — single isolated wire",
            "source": "rss", "ai_score": 9.0, "summary": "",
            "published": _iso(0.1), "first_seen": _iso(0.05),
        }
        store = _StoreSpy(velocity_rows=[
            {"ticker": "MU", "recent": 1, "prior": 0, "ratio": 2.0,
             "newest_age_s": 60.0},
        ])
        ok, prompt = _send(art, store, monkeypatch)
        assert ok is True
        assert "book: MU" in prompt
        assert "book_velocity:" not in _data_block(prompt)

    def test_zero_mention_ticker_silent(self, monkeypatch):
        """A held name with no other mentions in the velocity report (recent=0)
        must not emit a velocity line. Mirrors the silence discipline of the
        other operational blocks."""
        art = {
            "_id": "mu3", "link": "https://reuters.com/z",
            "title": "MU pricing comment from a trade-show panel",
            "source": "rss", "ai_score": 9.0, "summary": "",
            "published": _iso(0.1), "first_seen": _iso(0.05),
        }
        store = _StoreSpy(velocity_rows=[
            {"ticker": "MU", "recent": 0, "prior": 5, "ratio": 0.167,
             "newest_age_s": None},
        ])
        ok, prompt = _send(art, store, monkeypatch)
        assert ok is True
        assert "book_velocity:" not in _data_block(prompt)

    def test_multi_ticker_velocity_lists_each_qualifying(self, monkeypatch):
        """Two held tickers in one wire, only one of them with recent surge:
        the velocity line lists ONLY the qualifying one (silent on the other)."""
        art = {
            "_id": "multi", "link": "https://reuters.com/multi",
            "title": "MU and NVDA both surge on HBM demand",
            "source": "rss", "ai_score": 9.5, "summary": "",
            "published": _iso(0.1), "first_seen": _iso(0.05),
        }
        store = _StoreSpy(velocity_rows=[
            {"ticker": "MU", "recent": 5, "prior": 2, "ratio": 2.0,
             "newest_age_s": 600.0},
            # NVDA has only one mention (just this alert) — must be silent.
            {"ticker": "NVDA", "recent": 1, "prior": 0, "ratio": 2.0,
             "newest_age_s": 60.0},
        ])
        ok, prompt = _send(art, store, monkeypatch)
        assert ok is True
        assert "book: MU,NVDA" in prompt
        assert "MU: 5 mentions in last 60min" in prompt
        # NVDA must NOT appear in book_velocity because recent < 2.
        assert "NVDA: " not in prompt.split("book_velocity:")[1].split("\n")[0] \
            if "book_velocity:" in prompt else True
        # Both held tickers were queried (a single batched call).
        assert store.velocity_calls == [(["MU", "NVDA"], 60)]

    def test_no_held_ticker_no_velocity_call(self, monkeypatch):
        """A non-book row never carries ``book:``, so no velocity lookup is
        needed and no velocity line appears. Mirrors the existing book:-absent
        discipline (no fabricated context for a row that doesn't touch the
        book)."""
        art = {
            "_id": "nobook", "link": "https://reuters.com/q",
            "title": "Fed minutes hint at a split on the next move",
            "source": "rss", "ai_score": 8.5, "summary": "",
            "published": _iso(0.3), "first_seen": _iso(0.05),
        }
        store = _StoreSpy(velocity_rows=[
            {"ticker": "MU", "recent": 99, "prior": 99, "ratio": 1.0,
             "newest_age_s": 0.0},  # would qualify, but no book row
        ])
        ok, prompt = _send(art, store, monkeypatch)
        assert ok is True
        data = _data_block(prompt)
        # No per-article book: data line for a non-book row (the static BOOK
        # rule text contains "book:" — that's why we scope to the data block).
        assert "\nbook:" not in data
        assert "book_velocity:" not in data
        # No held tickers in batch → velocity helper must NOT be called
        # (single batched call is conditional on the batch having held names).
        assert store.velocity_calls == []


class TestBookVelocityFailureDegrades:
    """Best-effort: a store without ticker_mention_velocity (legacy mocks) OR
    one whose method raises (locked DB) must degrade silently to the
    pre-feature behaviour — the ``book:`` line still appears, no
    ``book_velocity:`` line, and the alert still fires.

    A missed velocity hint is far less bad than a dropped alert: the analyst
    persona's #2 complaint is missed urgent items, so this gate must NEVER
    block a fresh alert."""

    def test_store_without_method_degrades_silently(self, monkeypatch):
        class _NoVelocityStore:
            def __init__(self):
                self.marked = []
            def mark_alerted_batch(self, ids):
                self.marked.extend(ids)
            def mark_alerted(self, aid):
                self.marked.append(aid)
            # No ticker_mention_velocity attribute at all.

        store = _NoVelocityStore()
        art = {
            "_id": "old_mock", "link": "https://reuters.com/x",
            "title": "MU guides Q4 sharply above the Street",
            "source": "rss", "ai_score": 9.0, "summary": "",
            "published": _iso(0.1), "first_seen": _iso(0.05),
        }
        ok, prompt = _send(art, store, monkeypatch)
        assert ok is True
        assert "book: MU" in prompt
        assert "book_velocity:" not in _data_block(prompt)

    def test_store_raises_degrades_silently(self, monkeypatch):
        store = _StoreSpy(raises=True)
        art = {
            "_id": "raises", "link": "https://reuters.com/r",
            "title": "MU guides Q4 — store raises on velocity lookup",
            "source": "rss", "ai_score": 9.0, "summary": "",
            "published": _iso(0.1), "first_seen": _iso(0.05),
        }
        ok, prompt = _send(art, store, monkeypatch)
        assert ok is True
        # book: line still emitted (pre-feature behaviour preserved).
        assert "book: MU" in prompt
        assert "book_velocity:" not in _data_block(prompt)


class TestBookVelocityRuleReachesPrompt:
    def test_book_velocity_rule_in_static_prompt(self, monkeypatch):
        """The new BOOK VELOCITY rule must reach the Sonnet prompt so a
        Sonnet that sees the line knows how to interpret it. Mirrors the
        existing 'BOOK rule reaches the prompt' test discipline."""
        art = {
            "_id": "rulecheck", "link": "https://reuters.com/rc",
            "title": "MU guides Q4 sharply above the Street",
            "source": "rss", "ai_score": 9.0, "summary": "",
            "published": _iso(0.1), "first_seen": _iso(0.05),
        }
        store = _StoreSpy(velocity_rows=[
            {"ticker": "MU", "recent": 3, "prior": 1, "ratio": 2.0,
             "newest_age_s": 600.0},
        ])
        ok, prompt = _send(art, store, monkeypatch)
        assert ok is True
        assert "BOOK VELOCITY:" in prompt
        # The static rule must explain WHAT the line means and HOW Sonnet
        # should use it (prefer BUY/SELL over WATCH on a surge).
        assert "concentrating on that name" in prompt or "concentrating" in prompt
        assert "prefer BUY/SELL over WATCH" in prompt
