"""``analysis.claude_analyst._alert_velocity_lines`` and its wiring into
``_build_payload`` — the BREAKING-wire firing-rate magnitude hint.

The 🚨 BREAKING alert path is the analyst's most time-critical product. Its
RAW firing rate over a 5h window vs the prior 5h window carries a magnitude
signal no individual story score can express: 24 alerts vs 8 prior tells
Opus the wire is materially hot (a real macro event under way — Fed
surprise, geopolitical escalation, broad selloff). 2 vs 12 tells Opus the
wire is unusually quiet, so individual stories this window deserve closer
scrutiny than the same scores in a busy window.

Tests pin the operational contract with SPECIFIC numbers (not "no crash"):
  * tiny totals stay silent (a 1→3 swing on a sleepy wire is noise);
  * mild changes stay silent (a 25% swing is normal variance);
  * a materially-hot wire emits a hot line with exact recent/prior/% values;
  * a materially-cooling wire emits a cooling line, ditto;
  * the newly-lit / newly-silent edges bypass the percentage gate;
  * ``_build_payload`` emits the section ONLY when a velocity dict is
    explicitly supplied — the no-arg path stays byte-deterministic (same
    discipline as ``source_throughput`` / ``source_health_report`` /
    ``prior_digest``);
  * the SYSTEM_PROMPT reproduces the new section (a non-prompt that silently
    dropped the input would defeat the whole feature);
  * ``_collect_alert_velocity`` reads from ``alert_recency.db`` (ACTUAL
    fires) — NOT from ``articles.db urgency=2`` (which conflates fires with
    pre-fire suppressions); a synthetic mixed fixture pins the discriminator.
"""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from analysis import claude_analyst


class TestAlertVelocityLines:
    def test_returns_empty_on_none_or_non_dict(self):
        assert claude_analyst._alert_velocity_lines(None) == []
        assert claude_analyst._alert_velocity_lines([]) == []  # type: ignore[arg-type]
        assert claude_analyst._alert_velocity_lines("x") == []  # type: ignore[arg-type]

    def test_returns_empty_when_total_below_minimum(self):
        """1→3 is +200% but only 4 alerts total — a sleepy wire's micro-
        fluctuation, not analyst-actionable. Must stay silent."""
        assert claude_analyst._alert_velocity_lines(
            {"recent": 3, "prior": 1, "window_h": 5}
        ) == []

    def test_returns_empty_when_change_below_min_delta(self):
        """100→80 is -20%, normal news-rate variance. Must stay silent."""
        assert claude_analyst._alert_velocity_lines(
            {"recent": 80, "prior": 100, "window_h": 5}
        ) == []

    def test_hot_wire_emits_exact_message(self):
        """24 vs 8 is +200% — well above the 50% gate. The output line MUST
        carry the exact numbers and "+200%" + "hot" verdict."""
        lines = claude_analyst._alert_velocity_lines(
            {"recent": 24, "prior": 8, "window_h": 5}
        )
        assert len(lines) == 1
        assert lines[0] == (
            "BREAKING wire fired 24 alerts in last 5h vs 8 in prior 5h "
            "(+200%) — wire materially hot"
        )

    def test_cooling_wire_emits_exact_message(self):
        """2 vs 12 is -83% (recent+prior=14 >= 5; |delta|>=50). Cooling
        verdict, no leading '+' on negative delta."""
        lines = claude_analyst._alert_velocity_lines(
            {"recent": 2, "prior": 12, "window_h": 5}
        )
        assert len(lines) == 1
        assert lines[0] == (
            "BREAKING wire fired 2 alerts in last 5h vs 12 in prior 5h "
            "(-83%) — wire materially cooling"
        )

    def test_newly_lit_wire_bypasses_percentage_gate(self):
        """recent=5, prior=0 — ratio undefined, but the absolute change IS
        the signal. Must emit a "newly active" line."""
        lines = claude_analyst._alert_velocity_lines(
            {"recent": 5, "prior": 0, "window_h": 5}
        )
        assert len(lines) == 1
        assert lines[0] == (
            "BREAKING wire fired 5 alert(s) in last 5h vs 0 in prior 5h — "
            "wire newly active"
        )

    def test_newly_silent_wire_bypasses_percentage_gate(self):
        """recent=0, prior=8 — the wire went dark. Must emit a "silent" line
        (the percentage would be -100% but we handle this edge explicitly
        so the wording reads naturally — "0 alerts" not "wire materially
        cooling")."""
        lines = claude_analyst._alert_velocity_lines(
            {"recent": 0, "prior": 8, "window_h": 5}
        )
        assert len(lines) == 1
        assert lines[0] == (
            "BREAKING wire fired 0 alerts in last 5h vs 8 in prior 5h — "
            "wire silent"
        )

    def test_newly_lit_below_min_total_stays_silent(self):
        """recent=2, prior=0 — too few alerts to be a real wire-event signal.
        Must stay silent (the special-case branch must respect min_total)."""
        assert claude_analyst._alert_velocity_lines(
            {"recent": 2, "prior": 0, "window_h": 5}
        ) == []

    def test_newly_silent_below_min_total_stays_silent(self):
        """recent=0, prior=3 — the prior wire was already too quiet for
        the absence to be meaningful. Must stay silent."""
        assert claude_analyst._alert_velocity_lines(
            {"recent": 0, "prior": 3, "window_h": 5}
        ) == []

    def test_doubling_at_threshold_emits(self):
        """6 vs 4 is +50% exactly; recent+prior=10 (>=5). At-threshold must
        emit (the comparison is >=, not >). Pinned so a future thresh tweak
        is visible."""
        lines = claude_analyst._alert_velocity_lines(
            {"recent": 6, "prior": 4, "window_h": 5}
        )
        assert len(lines) == 1
        assert "+50%" in lines[0]
        assert "hot" in lines[0]

    def test_window_hours_is_reflected_in_text(self):
        """A 1h window briefing variant must render '1h' in the output (the
        period label is data-driven, not hardcoded)."""
        lines = claude_analyst._alert_velocity_lines(
            {"recent": 12, "prior": 4, "window_h": 1}
        )
        assert len(lines) == 1
        assert "in last 1h" in lines[0]
        assert "in prior 1h" in lines[0]

    def test_malformed_dict_returns_empty(self):
        """A garbage payload (non-numeric, missing keys, negative values)
        must degrade gracefully to [] — best-effort upstream means the dict
        could legitimately be missing fields."""
        assert claude_analyst._alert_velocity_lines({}) == []
        assert claude_analyst._alert_velocity_lines(
            {"recent": "x", "prior": 8, "window_h": 5}
        ) == []
        assert claude_analyst._alert_velocity_lines(
            {"recent": -1, "prior": 8, "window_h": 5}
        ) == []
        assert claude_analyst._alert_velocity_lines(
            {"recent": 10, "prior": 5, "window_h": 0}
        ) == []
        assert claude_analyst._alert_velocity_lines(
            {"recent": 10, "prior": 5, "window_h": -3}
        ) == []


class TestBuildPayloadAlertVelocityWiring:
    def test_emit_when_velocity_signals_hot(self):
        out = claude_analyst._build_payload(
            articles=[],
            stock_data={"macro": [], "equities": []},
            earnings=[],
            alert_velocity={"recent": 24, "prior": 8, "window_h": 5},
        )
        assert "=== ALERT VELOCITY" in out
        # The exact rendered line must appear so a non-conforming Opus could
        # still recover it via the prompt's "verbatim" rule.
        assert (
            "BREAKING wire fired 24 alerts in last 5h vs 8 in prior 5h "
            "(+200%) — wire materially hot"
        ) in out

    def test_omit_when_velocity_below_threshold(self):
        """A 10→8 wire (-20%) emits NO velocity section — same "below the
        bar means silent" discipline as COVERAGE GAP / THROUGHPUT
        DEGRADATION, so an unremarkable window stays clean."""
        out = claude_analyst._build_payload(
            articles=[],
            stock_data={"macro": [], "equities": []},
            earnings=[],
            alert_velocity={"recent": 8, "prior": 10, "window_h": 5},
        )
        assert "ALERT VELOCITY" not in out

    def test_omit_entirely_when_arg_is_none(self):
        """The default path (no alert_velocity kwarg) MUST be byte-identical
        to the pre-feature behaviour for that section — every caller / test
        that doesn't pass it stays unaffected. Same discipline as
        source_health_report / prior_digest / source_throughput."""
        out_default = claude_analyst._build_payload(
            articles=[],
            stock_data={"macro": [], "equities": []},
            earnings=[],
        )
        out_explicit_none = claude_analyst._build_payload(
            articles=[],
            stock_data={"macro": [], "equities": []},
            earnings=[],
            alert_velocity=None,
        )
        assert "ALERT VELOCITY" not in out_default
        assert out_default == out_explicit_none

    def test_emit_for_cooling_wire(self):
        out = claude_analyst._build_payload(
            articles=[],
            stock_data={"macro": [], "equities": []},
            earnings=[],
            alert_velocity={"recent": 2, "prior": 12, "window_h": 5},
        )
        assert "=== ALERT VELOCITY" in out
        assert "wire materially cooling" in out
        assert "(-83%)" in out

    def test_emit_for_newly_lit_wire(self):
        out = claude_analyst._build_payload(
            articles=[],
            stock_data={"macro": [], "equities": []},
            earnings=[],
            alert_velocity={"recent": 7, "prior": 0, "window_h": 5},
        )
        assert "=== ALERT VELOCITY" in out
        assert "wire newly active" in out


class TestCollectAlertVelocityDataSource:
    """Pin that ``_collect_alert_velocity`` reads from ``alert_recency.db``
    (the canonical fires log) and NOT from ``articles.db urgency=2`` (which
    conflates fires with the four pre-fire suppression gates'
    ``mark_alerted_batch`` calls in ``watchers.alert_agent``).

    Live evidence (2026-05-19, 10h window): 58 ``urgency=2`` rows in
    ``articles.db`` vs 37 hits in ``alert_recency.db`` — the prior
    implementation over-counted fires by ~57%, inflating the wire-heat
    signal Opus sees on the briefing's primary consumed product. These
    tests pin the data-source switch so a future regression that points
    ``_collect_alert_velocity`` back at ``urgency=2`` fails loud."""

    def _make_recency_db(self, path: Path, rows: list[tuple[str, str]]) -> None:
        """Build a fake ``alert_recency.db`` at ``path`` mirroring the live
        schema. Each row is ``(sig, last_ts_iso)``."""
        conn = sqlite3.connect(str(path))
        try:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS alerted_sig (
                    sig      TEXT PRIMARY KEY,
                    last_ts  TEXT NOT NULL,
                    title    TEXT,
                    hits     INTEGER NOT NULL DEFAULT 1
                );
                CREATE INDEX IF NOT EXISTS idx_alerted_sig_ts
                    ON alerted_sig(last_ts);
            """)
            conn.executemany(
                "INSERT OR REPLACE INTO alerted_sig (sig, last_ts, title, hits) "
                "VALUES (?, ?, ?, 1)",
                [(sig, ts, f"title-{sig}") for sig, ts in rows],
            )
            conn.commit()
        finally:
            conn.close()

    def test_counts_distinct_fires_in_each_window(
        self, tmp_path, monkeypatch
    ):
        """Three fires in the last 5h, two in the prior 5h — collector must
        report recent=3, prior=2, window_h=5 (and never read ``articles.db``).
        Pinned against last_ts boundary semantics: recent = ``last_ts >= now-5h``,
        prior = ``now-10h <= last_ts < now-5h``."""
        db_path = tmp_path / "alert_recency.db"
        now = datetime.now(timezone.utc)
        rows = [
            # recent (within 5h)
            ("sig-recent-1", (now - timedelta(hours=0.5)).isoformat()),
            ("sig-recent-2", (now - timedelta(hours=2.0)).isoformat()),
            ("sig-recent-3", (now - timedelta(hours=4.9)).isoformat()),
            # prior (5h..10h ago)
            ("sig-prior-1", (now - timedelta(hours=5.5)).isoformat()),
            ("sig-prior-2", (now - timedelta(hours=8.0)).isoformat()),
            # outside both windows (>10h ago) — must not count
            ("sig-stale", (now - timedelta(hours=11.0)).isoformat()),
        ]
        self._make_recency_db(db_path, rows)

        # Point the collector at our fixture path. The module's ``DB_PATH``
        # is what the runtime reads; monkeypatch it for the duration of the
        # test.
        monkeypatch.setattr(
            "watchers.alert_recency.DB_PATH", db_path, raising=True
        )

        result = claude_analyst._collect_alert_velocity(window_hours=5)
        assert result == {"recent": 3, "prior": 2, "window_h": 5}

    def test_ignores_articles_db_urgency_2_suppressions(
        self, tmp_path, monkeypatch
    ):
        """**The discriminating regression.** Seed an ``articles.db`` with
        many ``urgency=2`` rows (the four ``alert_agent`` suppression gates
        all mark rows alerted WITHOUT firing) and a SEPARATE
        ``alert_recency.db`` with only ONE recorded fire. The collector must
        report 1, not 5+ — i.e. it must NOT be looking at the articles
        table. The prior buggy implementation would have counted the 5
        urgency=2 suppressed rows as fires."""
        recency_path = tmp_path / "alert_recency.db"
        now = datetime.now(timezone.utc)
        self._make_recency_db(
            recency_path,
            [("sig-real-fire", (now - timedelta(hours=1)).isoformat())],
        )

        # Seed a parallel articles.db with FIVE urgency=2 rows in the same
        # recent window — these are what the prior bug counted. The collector
        # must ignore them by virtue of pointing at alert_recency.db instead.
        articles_path = tmp_path / "articles.db"
        aconn = sqlite3.connect(str(articles_path))
        try:
            aconn.executescript("""
                CREATE TABLE articles (
                    id TEXT PRIMARY KEY, url TEXT, title TEXT, source TEXT,
                    first_seen TEXT, urgency INTEGER
                );
            """)
            for i in range(5):
                aconn.execute(
                    "INSERT INTO articles VALUES (?, ?, ?, ?, ?, 2)",
                    (f"id{i}", f"https://example.com/{i}", f"t{i}", "rss",
                     (now - timedelta(hours=0.5 + i * 0.1)).isoformat()),
                )
            aconn.commit()
        finally:
            aconn.close()

        monkeypatch.setattr(
            "watchers.alert_recency.DB_PATH", recency_path, raising=True
        )
        # Belt-and-braces: force-point the articles DB resolver at the
        # poisoned suppression-only fixture. If the collector ever queries
        # it, the assertion below catches the regression.
        monkeypatch.setattr(
            "storage.article_store._get_db_path",
            lambda: articles_path,
            raising=True,
        )

        result = claude_analyst._collect_alert_velocity(window_hours=5)
        assert result is not None
        # Exactly 1 real fire — the 5 urgency=2 articles-only suppressions
        # MUST NOT count.
        assert result["recent"] == 1, (
            f"Expected 1 real fire; counted {result['recent']} — collector "
            "is reading articles.db urgency=2 again (the over-count bug)"
        )
        assert result["prior"] == 0
        assert result["window_h"] == 5

    def test_returns_none_on_missing_recency_db(self, tmp_path, monkeypatch):
        """A missing/locked recency DB is the documented best-effort failure
        mode: return ``None`` so ``_build_payload`` omits the section entirely
        (the 5h briefing is never broken or delayed)."""
        missing = tmp_path / "does_not_exist.db"
        monkeypatch.setattr(
            "watchers.alert_recency.DB_PATH", missing, raising=True
        )
        assert claude_analyst._collect_alert_velocity(window_hours=5) is None

    def test_window_hours_propagates_to_result(self, tmp_path, monkeypatch):
        """A non-default ``window_hours`` (e.g. 2h) must round-trip into the
        returned dict so ``_alert_velocity_lines`` renders the correct
        period label (the test fixtures above already check the renderer's
        end, this checks the collector's end)."""
        recency_path = tmp_path / "alert_recency.db"
        now = datetime.now(timezone.utc)
        rows = [
            # Inside 2h window
            ("a", (now - timedelta(minutes=30)).isoformat()),
            ("b", (now - timedelta(minutes=90)).isoformat()),
            # Inside prior 2h..4h window
            ("c", (now - timedelta(hours=2.5)).isoformat()),
            # Outside both
            ("d", (now - timedelta(hours=5)).isoformat()),
        ]
        self._make_recency_db(recency_path, rows)
        monkeypatch.setattr(
            "watchers.alert_recency.DB_PATH", recency_path, raising=True
        )

        result = claude_analyst._collect_alert_velocity(window_hours=2)
        assert result == {"recent": 2, "prior": 1, "window_h": 2}


class TestSystemPromptCoverage:
    def test_system_prompt_includes_alert_velocity_rule(self):
        """The prompt MUST instruct Opus to reproduce the new section,
        otherwise the input block is silently dropped from the output and
        the analyst never sees it. Same protection as the other reproduced
        sections (COVERAGE GAP, THROUGHPUT DEGRADATION)."""
        assert "ALERT VELOCITY" in claude_analyst.SYSTEM_PROMPT
        # The reproduction rule must be explicit and bidirectional (emit
        # when present, omit when absent) so Opus doesn't fabricate it.
        assert "Omit the section entirely if no velocity block" in (
            claude_analyst.SYSTEM_PROMPT
        )
