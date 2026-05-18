"""Alert continuation context — non-suppressing "related prior alert" hint.

Cross-cycle suppression (``alert_recency.partition_already_alerted``) drops only
EXACT-signature repeats. A *different* headline about the same developing event
(live: the 01:55 UAE-strike alert then a 09:19 Brent/markets follow-up —
distinct signatures, correctly NOT collapsed) still fires a fresh standalone 🚨
BREAKING with zero continuity framing: the consuming analyst's top "duplicate
alerts" complaint, on the one product that never got the mitigation the
briefing's ``[ALERTED]`` tag added.

This feature annotates (NEVER drops) such a survivor with the related prior
alert so the prompt's CONTINUITY rule frames it as a developing update. These
tests assert specific behaviour — the exact relatedness decision, that the
alert STILL fires (non-suppressing), and that the load-bearing invariants are
untouched (no ai_score/ml_score/score_source mutation; alert_recency.db only,
never articles.db).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from watchers import alert_agent, alert_recency


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


# ── Pure relatedness decision ────────────────────────────────────────────────
class TestRelatedPriorAlert:
    def test_distinct_but_related_headline_matches(self):
        recent = [{"sig": alert_recency._signature(
            "Nvidia H200 chip export approval to China firms"),
            "title": "Nvidia H200 chip export approval to China firms",
            "age_hours": 2.0}]
        m = alert_recency.related_prior_alert(
            "Nvidia H200 chip export approval widens amid Beijing talks", recent)
        assert m is not None
        assert m["title"] == "Nvidia H200 chip export approval to China firms"
        assert m["age_hours"] == 2.0
        # nvidia / h200 / chip / export / approval are the shared salient tokens
        assert {"nvidia", "h200", "chip", "export", "approval"} <= set(m["shared"])

    def test_exact_signature_is_not_a_continuation(self):
        # An exact-signature repeat is a true duplicate — already dropped by
        # partition_already_alerted upstream; it must NOT also produce a hint.
        title = "Micron DRAM pricing surges on HBM shortage this quarter"
        recent = [{"sig": alert_recency._signature(title), "title": title,
                   "age_hours": 1.0}]
        assert alert_recency.related_prior_alert(title, recent) is None

    def test_unrelated_headline_no_match(self):
        recent = [{"sig": alert_recency._signature(
            "Fed holds rates steady amid inflation pressure"),
            "title": "Fed holds rates steady amid inflation pressure",
            "age_hours": 0.5}]
        assert alert_recency.related_prior_alert(
            "Micron DRAM pricing surges on HBM shortage", recent) is None

    def test_stopword_only_overlap_is_not_related(self):
        # "Stock Market Today ..." vs "Stock Market Wrap ..." share only
        # _REL_STOPWORDS — must NOT be called a continuation (the false-
        # positive the analyst would read as noise).
        recent = [{"sig": alert_recency._signature(
            "Stock Market Today: Dow Jones futures climb"),
            "title": "Stock Market Today: Dow Jones futures climb",
            "age_hours": 1.0}]
        assert alert_recency.related_prior_alert(
            "Stock Market Wrap: S&P 500 closes lower", recent) is None

    def test_live_uae_drone_vs_futures_drop_no_false_continuation(self):
        # The two real 2026-05-18 alerts — genuinely different events that
        # share no salient signature token. Must NOT cross-link.
        recent = [{"sig": alert_recency._signature(
            "Drone Attack On UAE Nuclear Plant, Trump's Iran Warning Send Brent"),
            "title": "Drone Attack On UAE Nuclear Plant", "age_hours": 7.0}]
        assert alert_recency.related_prior_alert(
            "Stock Market Today: Dow Jones, S&P 500 Futures Drop", recent) is None

    def test_picks_prior_with_most_shared_salient_tokens(self):
        recent = [
            {"sig": alert_recency._signature("Micron earnings beat boosts chip stocks"),
             "title": "weak link", "age_hours": 1.0},
            {"sig": alert_recency._signature("Micron earnings beat lifts DRAM pricing outlook"),
             "title": "strong link", "age_hours": 5.0},
        ]
        m = alert_recency.related_prior_alert(
            "Micron earnings beat sends DRAM pricing higher", recent)
        assert m is not None and m["title"] == "strong link"

    def test_untitled_current_returns_none(self):
        recent = [{"sig": "anything here token set", "title": "x", "age_hours": 1.0}]
        assert alert_recency.related_prior_alert("", recent) is None
        assert alert_recency.related_prior_alert(None, recent) is None

    def test_empty_recent_is_noop(self):
        assert alert_recency.related_prior_alert("Nvidia surges on AI demand", []) is None


# ── recent_alerts() store read ───────────────────────────────────────────────
class TestRecentAlerts:
    def test_record_then_recent_alerts_returns_title_and_age(self):
        alert_recency.record_alerted(
            [{"title": "Nvidia H200 export approval expands to China"}],
            now=datetime.now(timezone.utc) - timedelta(hours=2),
        )
        out = alert_recency.recent_alerts()
        assert len(out) == 1
        r = out[0]
        assert r["title"] == "Nvidia H200 export approval expands to China"
        assert 1.5 < r["age_hours"] < 2.5  # ~2h ago
        assert r["sig"] == alert_recency._signature(
            "Nvidia H200 export approval expands to China")

    def test_ttl_filters_old_alerts(self):
        alert_recency.record_alerted(
            [{"title": "Very old breaking story about chips"}],
            now=datetime.now(timezone.utc) - timedelta(hours=20),
        )
        # 20h ago, default TTL 6h → excluded.
        assert alert_recency.recent_alerts() == []
        # Widen the window past it → included.
        assert len(alert_recency.recent_alerts(ttl_hours=48)) == 1

    def test_broken_db_degrades_to_empty_list(self, monkeypatch):
        def _boom():
            raise RuntimeError("db down")
        monkeypatch.setattr(alert_recency, "_connect", _boom)
        assert alert_recency.recent_alerts() == []


# ── Integration: prompt carries the hint, alert still fires ──────────────────
def _insert_urgent(store, *, id, title, url="https://reuters.com/x",
                   source="rss", ai_score=9.0):
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            " first_seen, cycle, ml_score, score_source, full_text) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", 1.0, ai_score, 1,
             _iso(0.08), 0, None, "llm", None),
        )
        store.conn.commit()


class TestContinuationIntegration:
    def test_related_prior_injects_continuity_line_and_still_fires(
        self, store, monkeypatch
    ):
        # A related alert fired ~2h ago.
        alert_recency.record_alerted(
            [{"title": "Nvidia H200 chip export approval to China firms"}],
            now=datetime.now(timezone.utc) - timedelta(hours=2),
        )
        # A DIFFERENT headline about the same developing story is now urgent.
        _insert_urgent(
            store, id="dev",
            title="Nvidia H200 chip export approval widens amid Beijing talks",
        )
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call",
                          return_value="alert body") as mc, \
             patch("notifier.discord_notifier.send", return_value=True) as ms:
            ok = alert_agent.send_urgent_alert(store.get_unalerted_urgent(), store)

        assert ok is True                       # NON-suppressing: still fires
        assert mc.called and ms.called          # reached Claude + Discord
        prompt = mc.call_args[0][0]
        assert "related: a 🚨 BREAKING alert fired ~2" in prompt
        assert "Nvidia H200 chip export approval to China firms" in prompt
        assert "CONTINUITY:" in prompt          # prompt rule present
        # Invariant pin: only urgency moved (1 → 2); scores untouched.
        row = store.conn.execute(
            "SELECT ai_score, ml_score, score_source, urgency "
            "FROM articles WHERE id=?", ("dev",)
        ).fetchone()
        assert row == (9.0, None, "llm", 2)

    def test_unrelated_prior_adds_no_line_and_still_fires(self, store, monkeypatch):
        alert_recency.record_alerted(
            [{"title": "Fed holds rates steady amid inflation pressure"}],
            now=datetime.now(timezone.utc) - timedelta(hours=1),
        )
        _insert_urgent(
            store, id="u1",
            title="Micron DRAM pricing surges on HBM shortage this quarter",
        )
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call",
                          return_value="alert body") as mc, \
             patch("notifier.discord_notifier.send", return_value=True):
            ok = alert_agent.send_urgent_alert(store.get_unalerted_urgent(), store)
        assert ok is True
        assert "\nrelated:" not in mc.call_args[0][0]

    def test_recency_store_failure_does_not_block_alert(self, store, monkeypatch):
        _insert_urgent(store, id="g", title="Major chip supply shock hits market now")
        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")

        def _boom(*a, **k):
            raise RuntimeError("recency down")
        monkeypatch.setattr(alert_recency, "recent_alerts", _boom)
        with patch.object(alert_agent, "claude_call",
                          return_value="alert body") as mc, \
             patch("notifier.discord_notifier.send", return_value=True):
            ok = alert_agent.send_urgent_alert(store.get_unalerted_urgent(), store)
        # A genuine breaking story must still reach the analyst.
        assert ok is True and mc.called
        assert "\nrelated:" not in mc.call_args[0][0]
