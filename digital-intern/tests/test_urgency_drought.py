"""Urgency drought monitor — bounded reads, status classification, tz parsing.

Pins the latent bug fixed in this pass: ``_parse_ts`` previously checked only
``"+" not in s[10:]`` so a NEGATIVE timezone offset (``-05:00`` US EST,
``-08:00`` PST) survived the "no offset present" branch, had ``+00:00``
appended to it, and the resulting ``...-05:00+00:00`` raised ``ValueError``
and returned ``None``. In production ``first_seen`` is always written as
``datetime.now(timezone.utc).isoformat()`` (UTC + ``+00:00``) so this never
fired live, but any non-UTC row sourced from a migration or external import
would silently classify as ``status='unknown'`` — a defense-in-depth gap.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from analytics import urgency_drought as ud


class TestParseTs:
    def test_utc_iso_with_plus_offset(self):
        assert ud._parse_ts("2026-05-23T18:00:00+00:00") == datetime(
            2026, 5, 23, 18, 0, tzinfo=timezone.utc
        )

    def test_z_suffix_is_utc(self):
        assert ud._parse_ts("2026-05-23T18:00:00Z") == datetime(
            2026, 5, 23, 18, 0, tzinfo=timezone.utc
        )

    def test_naive_assumed_utc(self):
        """A timestamp with NO offset and no Z is treated as UTC. This is
        what the SQLite default ``CURRENT_TIMESTAMP`` produces and what the
        legacy migration script wrote, so it must continue to parse."""
        assert ud._parse_ts("2026-05-23T18:00:00") == datetime(
            2026, 5, 23, 18, 0, tzinfo=timezone.utc
        )

    def test_space_separator(self):
        """SQLite's default separator is a space, not T — must still parse."""
        assert ud._parse_ts("2026-05-23 18:00:00") == datetime(
            2026, 5, 23, 18, 0, tzinfo=timezone.utc
        )

    def test_negative_tz_offset_now_parses(self):
        """The pre-fix bug: only ``"+"`` was treated as a tz marker, so a
        negative offset had ``+00:00`` appended and the result raised
        ValueError. A US EST ``-05:00`` row must convert to its UTC instant
        (23:00) — NOT silently return None."""
        got = ud._parse_ts("2026-05-23T18:00:00-05:00")
        assert got is not None, "negative-tz input was dropped to None"
        assert got == datetime(2026, 5, 23, 23, 0, tzinfo=timezone.utc)

    def test_negative_tz_no_colon_offset_parses(self):
        """RFC-3339 allows ``-0500`` (no colon) — also a negative offset, also
        previously broken by the ``"+" not in ...`` heuristic."""
        got = ud._parse_ts("2026-05-23T18:00:00-0500")
        assert got == datetime(2026, 5, 23, 23, 0, tzinfo=timezone.utc)

    def test_positive_non_utc_offset_normalises(self):
        """JST ``+09:00`` 12:15 = UTC 03:15 — must convert, not just strip."""
        got = ud._parse_ts("2026-05-23T12:15:00+09:00")
        assert got == datetime(2026, 5, 23, 3, 15, tzinfo=timezone.utc)

    def test_garbage_returns_none(self):
        assert ud._parse_ts("not-a-date") is None
        assert ud._parse_ts("") is None
        assert ud._parse_ts(None) is None


class TestDroughtClassification:
    """``compute`` reads from a real DB. Test the status logic directly via
    the inline ``_drought`` helper by exercising ``compute`` with a temporary
    fixture connection."""

    def _run_with_fixture(self, monkeypatch, tmp_path, *,
                          last_u1: str | None, last_u2: str | None):
        """Build a tiny temp DB with one row per requested urgency, then run
        ``compute``. Returns the payload it would have written."""
        import sqlite3
        db = tmp_path / "articles.db"
        conn = sqlite3.connect(str(db))
        conn.executescript("""
            CREATE TABLE articles (
                id TEXT PRIMARY KEY, url TEXT NOT NULL, title TEXT NOT NULL,
                source TEXT, published TEXT, kw_score REAL DEFAULT 0,
                ai_score REAL DEFAULT 0, urgency INTEGER DEFAULT 0,
                full_text BLOB, first_seen TEXT NOT NULL, cycle INTEGER DEFAULT 0,
                time_sensitivity REAL, ml_score REAL, score_source TEXT
            );
            CREATE INDEX idx_first_seen ON articles(first_seen);
        """)
        if last_u1 is not None:
            conn.execute(
                "INSERT INTO articles (id,url,title,source,urgency,first_seen) "
                "VALUES ('u1','https://x.com/u1','t','rss',1,?)", (last_u1,))
        if last_u2 is not None:
            conn.execute(
                "INSERT INTO articles (id,url,title,source,urgency,first_seen) "
                "VALUES ('u2','https://x.com/u2','t','rss',2,?)", (last_u2,))
        conn.commit()
        conn.close()

        monkeypatch.setattr(ud, "_get_db_path", lambda: db)
        # Redirect the JSON sink off the live ~/logs/ path.
        monkeypatch.setattr(ud, "OUT", tmp_path / "urgency_drought.json")
        return ud.compute()

    def test_ok_status_when_recent(self, monkeypatch, tmp_path):
        now = datetime.now(timezone.utc)
        recent = (now - timedelta(hours=1)).isoformat()
        p = self._run_with_fixture(
            monkeypatch, tmp_path, last_u1=recent, last_u2=recent)
        assert p["status"] == "ok"
        assert p["urgency_2"]["status"] == "ok"
        assert p["urgency_1"]["status"] == "ok"

    def test_warn_threshold(self, monkeypatch, tmp_path):
        now = datetime.now(timezone.utc)
        old = (now - timedelta(hours=ud.WARN_HOURS + 0.1)).isoformat()
        p = self._run_with_fixture(
            monkeypatch, tmp_path, last_u1=old, last_u2=old)
        assert p["urgency_2"]["status"] == "warn"
        assert p["status"] == "warn"

    def test_alert_threshold_dominates(self, monkeypatch, tmp_path):
        now = datetime.now(timezone.utc)
        very_old = (now - timedelta(hours=ud.ALERT_HOURS + 0.1)).isoformat()
        recent = (now - timedelta(hours=0.5)).isoformat()
        # u1 fresh, u2 stale — overall must escalate to the worst (alert)
        p = self._run_with_fixture(
            monkeypatch, tmp_path, last_u1=recent, last_u2=very_old)
        assert p["urgency_2"]["status"] == "alert"
        assert p["urgency_1"]["status"] == "ok"
        assert p["status"] == "alert"

    def test_unknown_when_no_rows(self, monkeypatch, tmp_path):
        p = self._run_with_fixture(
            monkeypatch, tmp_path, last_u1=None, last_u2=None)
        assert p["urgency_2"]["status"] == "unknown"
        assert p["urgency_2"]["last_seen"] is None
        assert p["status"] in ("unknown", "warn")

    def test_negative_tz_first_seen_does_not_drop_to_unknown(
            self, monkeypatch, tmp_path):
        """End-to-end pin of the parse fix: a row stored with a NEGATIVE-tz
        ``first_seen`` (e.g. from an external import that preserved the
        publisher's local time) must produce a normal status, NOT
        ``status='unknown'`` from a silently-failed parse."""
        # 1h ago expressed in US EST — same instant as UTC, written with -05:00.
        local = datetime.now(timezone.utc) - timedelta(hours=1)
        offset_iso = local.astimezone(timezone(timedelta(hours=-5))).isoformat()
        assert "-05:00" in offset_iso  # sanity — the test input must be negative-tz
        p = self._run_with_fixture(
            monkeypatch, tmp_path,
            last_u1=offset_iso, last_u2=offset_iso)
        assert p["urgency_2"]["status"] == "ok", (
            f"negative-tz first_seen silently classed as "
            f"{p['urgency_2']['status']!r}; the parse fix regressed"
        )
        assert p["urgency_2"]["hours_ago"] == pytest.approx(1.0, abs=0.1)
