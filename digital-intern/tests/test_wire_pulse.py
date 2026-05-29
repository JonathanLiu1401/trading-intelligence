"""``scripts/wire_pulse.py`` — pin the verdict ladder and the format
contract.

The wire_pulse script is consumed by cron / grep / Discord status pushes,
so the field ORDER in the one-line output and the EXACT verdict tokens are
load-bearing: a downstream awk pipeline that reads the third field as
``urgent_1h=N`` would silently misbehave if the format changes. The verdict
ladder is the operator's "do I need to wake up?" signal — a regression that
slides INGEST_DARK rows into HEALTHY would silently hide a collector outage.

Each verdict transition is exercised on a fixture snapshot so the ladder
remains assertable without a live DB.
"""
from __future__ import annotations

import json
from io import StringIO
from unittest.mock import patch

import pytest

from scripts import wire_pulse


def _snap(**overrides) -> dict:
    """Build a known-good HEALTHY snapshot, then override specific fields.
    Verdict is derived from the resulting fields so the fixture stays
    consistent with the ladder under test."""
    base = {
        "ts": "2026-05-29T01:50:00+00:00",
        "articles_1h": 8421,
        "urgent_1h": 12,
        "urgent_24h": 145,
        "llm_vetted_pct": 42.0,
        "briefing_age_h": 2.3,
        "briefing_verdict": "HEALTHY",
        "last_alert_age_h": 0.4,
        "alerts_24h": 8,
    }
    base.update(overrides)
    base["verdict"] = wire_pulse._verdict(base)
    return base


class TestVerdictLadder:
    def test_healthy_snapshot(self):
        snap = _snap()
        assert wire_pulse._verdict(snap) == "HEALTHY"

    def test_unknown_when_articles_1h_is_none(self):
        """Couldn't read the DB — operator must see UNKNOWN, not a false
        HEALTHY just because the other fields happen to look reasonable."""
        snap = _snap(articles_1h=None)
        assert wire_pulse._verdict(snap) == "UNKNOWN"

    def test_ingest_dark_below_threshold(self):
        """Below INGEST_DARK_MIN_1H = 30 articles/h — a collector outage.
        Must dominate the ladder (we don't care about downstream health
        when nothing is coming in)."""
        snap = _snap(articles_1h=10)
        assert wire_pulse._verdict(snap) == "INGEST_DARK"

    def test_ingest_dark_overrides_briefing_stale(self):
        """Both conditions are bad; INGEST_DARK is the more actionable one
        (fix the collectors first; the briefing dependency follows)."""
        snap = _snap(articles_1h=5, briefing_age_h=24.0)
        assert wire_pulse._verdict(snap) == "INGEST_DARK"

    def test_briefing_stale_by_age(self):
        snap = _snap(briefing_age_h=13.0)
        assert wire_pulse._verdict(snap) == "BRIEFING_STALE"

    def test_briefing_dead_verdict_promotes_to_briefing_stale(self):
        """briefing_health emits its own verdict; if it says DEAD we honor
        it even if the age is at the boundary."""
        snap = _snap(briefing_age_h=11.5, briefing_verdict="DEAD")
        assert wire_pulse._verdict(snap) == "BRIEFING_STALE"

    def test_alert_quiet_when_no_recent_pushes_and_no_urgent(self):
        snap = _snap(urgent_1h=0, last_alert_age_h=None)
        assert wire_pulse._verdict(snap) == "ALERT_QUIET"

    def test_alert_quiet_when_last_alert_old_and_no_urgent(self):
        snap = _snap(urgent_1h=0, last_alert_age_h=7.5)
        assert wire_pulse._verdict(snap) == "ALERT_QUIET"

    def test_healthy_with_recent_alert_even_if_urgent_zero(self):
        """A recent push (under the 6h ttl) plus zero urgent in the last
        hour is normal — the wire just happens to be quiet RIGHT NOW."""
        snap = _snap(urgent_1h=0, last_alert_age_h=2.0)
        assert wire_pulse._verdict(snap) == "HEALTHY"


class TestOneLineFormat:
    """The one-line output is consumed by grep / awk pipelines (the
    docstring's stated workflow) — the field order MUST stay stable."""

    def test_field_order_and_prefix(self):
        snap = _snap()
        line = wire_pulse._format_line(snap)
        assert line.startswith("[wire_pulse 2026-05-29T01:50:00+00:00] ")
        # Field order: articles_1h, urgent_1h, llm_vetted_pct,
        # briefing_age_h, last_alert_age_h, → VERDICT.
        order = [
            "articles_1h=",
            "urgent_1h=",
            "llm_vetted_pct=",
            "briefing_age_h=",
            "last_alert_age_h=",
            "→ HEALTHY",
        ]
        last_idx = -1
        for tok in order:
            idx = line.find(tok)
            assert idx > last_idx, (
                f"field {tok!r} out of order or missing in {line!r}"
            )
            last_idx = idx

    def test_missing_field_renders_as_question_mark(self):
        """A partial-read snapshot must still emit a valid line — a
        downstream parser must not have to special-case None."""
        snap = _snap(articles_1h=None, last_alert_age_h=None)
        line = wire_pulse._format_line(snap)
        assert "articles_1h=?" in line
        assert "last_alert_age_h=?" in line

    def test_verdict_appears_at_end(self):
        snap = _snap()
        line = wire_pulse._format_line(snap)
        assert line.rstrip().endswith("→ HEALTHY")


class TestComputeIntegration:
    """End-to-end compose-from-primitives. Uses real ArticleStore (via the
    conftest tmp_path fixture) so the SQL-side primitives actually run."""

    def test_empty_store_yields_unknown_or_ingest_dark(self, store):
        """A fresh empty DB has zero articles in the last hour, so the
        verdict must reflect that the wire is dark — not silently HEALTHY."""
        snap = wire_pulse._compute(store)
        # articles_1h=0 < INGEST_DARK_MIN_1H → INGEST_DARK
        assert snap["articles_1h"] == 0
        assert snap["verdict"] == "INGEST_DARK"
        # All numeric fields must be defined (never None) once the store
        # itself opens cleanly — partial reads only happen on per-primitive
        # exceptions, which a fresh empty store won't raise.
        assert snap["urgent_1h"] == 0
        assert snap["llm_vetted_pct"] == 0.0
        assert snap["alerts_24h"] == 0

    def test_compute_yields_healthy_with_seeded_traffic(self, store, monkeypatch):
        """Seed enough live traffic to clear the INGEST_DARK floor and stub
        out the alert-recency / briefing reads so the verdict path can
        reach HEALTHY without a fully populated alert ledger."""
        from datetime import datetime, timedelta, timezone

        recent = (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat()
        # Seed 50 fresh live rows. ml/llm split: 1 LLM-vetted urgent (so
        # llm_fraction > 0 and not the dark-Sonnet case), rest are scored
        # quiet rows.
        with store._write_lock:
            for i in range(50):
                store.conn.execute(
                    "INSERT INTO articles "
                    "(id, url, title, source, published, kw_score, "
                    " ai_score, urgency, first_seen, cycle, "
                    " ml_score, score_source) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                    (f"row_{i}", f"https://reuters.com/{i}",
                     f"News item {i}", "rss", "", 1.0,
                     9.0 if i == 0 else 0.0,    # one LLM-labelled urgent
                     1 if i == 0 else 0, recent, 0, None,
                     "llm" if i == 0 else None),
                )
            store.conn.commit()

        # Stub the briefing+alert reads via monkeypatched store methods —
        # we are testing the verdict ladder, not the real alert_recency DB.
        monkeypatch.setattr(
            store, "briefing_health",
            lambda window_h=24: {"last_briefing_age_h": 2.0,
                                  "verdict": "HEALTHY"},
        )
        monkeypatch.setattr(
            "watchers.alert_recency.recent_alerts",
            lambda ttl_hours=24: [{"age_hours": 0.5,
                                    "title": "MU surges",
                                    "sig": "mu surges"}],
        )

        snap = wire_pulse._compute(store)
        assert snap["articles_1h"] == 50
        assert snap["urgent_1h"] == 1
        assert snap["llm_vetted_pct"] > 0.0
        assert snap["briefing_age_h"] == 2.0
        assert snap["last_alert_age_h"] == 0.5
        assert snap["verdict"] == "HEALTHY"


class TestCliEntrypoint:
    """The CLI entrypoint is the cron/operator contract: human one-line
    by default, --json for machine consumers, and a meaningful exit code."""

    def test_main_emits_one_line_summary(self, store, monkeypatch, capsys):
        # Redirect the store the script opens to the in-test instance.
        import storage.article_store as _store_mod
        monkeypatch.setattr(_store_mod, "ArticleStore", lambda: store)

        rc = wire_pulse.main([])
        out = capsys.readouterr().out
        # One line of output, starts with the canonical prefix.
        assert out.startswith("[wire_pulse ")
        assert "verdict" not in out  # the key name is internal
        assert "→ " in out
        # Empty store → INGEST_DARK → non-zero exit.
        assert rc == 1

    def test_main_json_mode_emits_parseable_payload(self, store, monkeypatch, capsys):
        import storage.article_store as _store_mod
        monkeypatch.setattr(_store_mod, "ArticleStore", lambda: store)

        rc = wire_pulse.main(["--json"])
        out = capsys.readouterr().out
        payload = json.loads(out)
        assert "verdict" in payload
        assert "ts" in payload
        assert "articles_1h" in payload
        # Same fresh-store invariant as above.
        assert payload["articles_1h"] == 0
        assert payload["verdict"] == "INGEST_DARK"
        assert rc == 1

    def test_main_store_unavailable_path_returns_two(self, monkeypatch, capsys):
        """If ArticleStore() raises (USB unmounted, locked DB), the CLI
        must STILL produce a deterministic line + JSON and return exit
        code 2 (distinct from 1 = degraded) so cron can branch."""
        import storage.article_store as _store_mod

        def _boom():
            raise RuntimeError("USB drive not mounted")

        monkeypatch.setattr(_store_mod, "ArticleStore", _boom)
        rc = wire_pulse.main([])
        out = capsys.readouterr().out
        assert "STORE_UNAVAILABLE" in out
        assert "USB drive not mounted" in out
        assert rc == 2
