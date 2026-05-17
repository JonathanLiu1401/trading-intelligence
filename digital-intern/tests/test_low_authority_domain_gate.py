"""Phase-2: the GDELT junk-domain tier tightens the lone-alert gate.

A 1.4M-row live snapshot showed ~95% of the corpus is ``gdelt_gkg/<host>``
and the urgency head over-scores low-signal hosts (algorithmic stock-mention
press mills, radio networks). Before this tier a lone, un-syndicated urgent
row from ``gdelt_gkg/wkrb13.com`` cleared the 0.45 gate (it defaulted to
0.55) and fired a standalone Bloomberg "🚨 BREAKING" push — exactly the noise
the analyst complains about.

Pinned here (assert behaviour, not no-crash):
  * a lone junk-domain urgent row is suppressed end-to-end through
    ``send_urgent_alert`` — no Claude, no Discord, marked ``urgency=2`` so it
    exits the queue and never re-fires;
  * a lone CREDIBLE gdelt host (reuters.com) is unaffected — still fires
    (this is the regression tripwire for over-reach);
  * corroboration is still the escape valve: a junk copy + a credible copy of
    the SAME story collapses (dup_count>1) and fires.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import patch

import pytest

from watchers import alert_agent
from watchers.alert_agent import ALERT_MIN_LONE_SOURCE_CRED, _filter_low_authority_lone
from ml.features import _LOW_AUTHORITY_DOMAINS, _source_credibility


def _iso(hours_ago: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def _insert_urgent(store, *, id, source, title="MU earnings blow past Q3 sharply",
                   url=None, ai_score=9.0):
    if url is None:
        url = f"https://example.com/{id}"
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            " first_seen, cycle, ml_score, score_source, full_text) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, _iso(1), 1.0, ai_score, 1,
             _iso(0.08), 0, None, "llm", None),
        )
        store.conn.commit()


def _urgency_of(store, aid):
    return store.conn.execute(
        "SELECT urgency FROM articles WHERE id=?", (aid,)
    ).fetchone()[0]


# ── unit: the tier resolves below the gate ──────────────────────────────────
class TestJunkTierValues:
    def test_all_junk_values_below_lone_alert_gate(self):
        """Every junk host MUST resolve below ALERT_MIN_LONE_SOURCE_CRED, else
        it would not actually be suppressed when lone."""
        assert _LOW_AUTHORITY_DOMAINS, "junk tier must be populated in Phase 2"
        for host, cred in _LOW_AUTHORITY_DOMAINS.items():
            assert cred < ALERT_MIN_LONE_SOURCE_CRED, f"{host}={cred} >= gate"

    def test_prefixed_junk_host_resolves_to_junk_grade(self):
        assert _source_credibility("gdelt_gkg/wkrb13.com") == pytest.approx(0.25)
        assert _source_credibility("GDELT/iheart.com") == pytest.approx(0.30)
        # …while a credible host sharing the same prefix is unaffected.
        assert _source_credibility("gdelt_gkg/reuters.com") == pytest.approx(0.90)

    def test_filter_partitions_lone_junk_vs_credible_vs_corroborated(self):
        rows = [
            {"_id": "lone_junk", "source": "gdelt_gkg/wkrb13.com", "dup_count": 1},
            {"_id": "lone_radio", "source": "GDELT/iheart.com", "dup_count": 1},
            {"_id": "synd_junk", "source": "gdelt_gkg/wkrb13.com", "dup_count": 4},
            {"_id": "lone_wire", "source": "gdelt_gkg/reuters.com", "dup_count": 1},
        ]
        kept, suppressed = _filter_low_authority_lone(rows)
        assert {a["_id"] for a in suppressed} == {"lone_junk", "lone_radio"}
        assert {a["_id"] for a in kept} == {"synd_junk", "lone_wire"}


# ── e2e through send_urgent_alert (mirrors test_alert_source_authority) ──────
class TestLoneJunkDomainSuppressedE2E:
    def test_lone_junk_domain_never_reaches_claude_or_discord(self, store,
                                                              monkeypatch):
        _insert_urgent(store, id="j1", source="gdelt_gkg/wkrb13.com",
                       title="Micron Technology Inc shares bought by Vanguard")
        urgent = store.get_unalerted_urgent()
        assert len(urgent) == 1, "precondition: store returns the live row"

        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call") as mock_claude, \
             patch("notifier.discord_notifier.send") as mock_send:
            ok = alert_agent.send_urgent_alert(urgent, store)

        assert ok is False
        mock_claude.assert_not_called()
        mock_send.assert_not_called()
        assert _urgency_of(store, "j1") == 2, "suppressed row must exit the queue"
        assert store.get_unalerted_urgent() == []

    def test_lone_credible_gdelt_host_still_fires(self, store, monkeypatch):
        """Over-reach tripwire: a lone gdelt_gkg/reuters.com (0.90) is NOT in
        the junk tier and must still compose+post an alert exactly as before."""
        _insert_urgent(store, id="r1", source="gdelt_gkg/reuters.com",
                       title="MU guides Q4 revenue sharply above the Street")
        urgent = store.get_unalerted_urgent()
        assert len(urgent) == 1

        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call",
                          return_value="🚨 BREAKING ◈ EARNINGS ◈ MU") as mock_claude, \
             patch("notifier.discord_notifier.send",
                   return_value=True) as mock_send:
            ok = alert_agent.send_urgent_alert(urgent, store)

        assert ok is True
        mock_claude.assert_called_once()
        mock_send.assert_called_once()
        assert _urgency_of(store, "r1") == 2

    def test_corroborated_junk_story_still_fires(self, store, monkeypatch):
        """Escape valve: the SAME headline from a junk host AND a credible
        host collapses via dedupe_urgent (dup_count=2) — the gate keeps it and
        BOTH underlying ids end urgency=2 (cannot re-fire)."""
        shared = "Samsung begins HBM4 shipments as SK Hynix lags"
        _insert_urgent(store, id="jc_junk", source="gdelt_gkg/wkrb13.com",
                       title=shared, ai_score=8.0)
        _insert_urgent(store, id="jc_wire", source="gdelt_gkg/reuters.com",
                       title=shared, ai_score=9.0)
        urgent = store.get_unalerted_urgent()
        assert {a["_id"] for a in urgent} == {"jc_junk", "jc_wire"}

        monkeypatch.setattr(alert_agent, "DISCORD_WEBHOOK", "https://x/webhook")
        with patch.object(alert_agent, "claude_call",
                          return_value="🚨 BREAKING ◈ SUPPLY CHAIN ◈ HBM4") as mock_claude, \
             patch("notifier.discord_notifier.send",
                   return_value=True) as mock_send:
            ok = alert_agent.send_urgent_alert(urgent, store)

        assert ok is True, "a syndicated story must still fire even via a junk copy"
        mock_claude.assert_called_once()
        mock_send.assert_called_once()
        assert _urgency_of(store, "jc_junk") == 2
        assert _urgency_of(store, "jc_wire") == 2
