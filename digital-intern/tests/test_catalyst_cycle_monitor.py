from __future__ import annotations

from datetime import datetime, timedelta, timezone

from analytics import catalyst_cycle_monitor as ccm


NOW = datetime(2026, 6, 1, 20, 0, tzinfo=timezone.utc)


def _ts(minutes_ago: float) -> str:
    return (NOW - timedelta(minutes=minutes_ago)).isoformat()


def test_fresh_high_score_ai_catalyst_is_urgent():
    rows = [
        (_ts(2), "NVDA unveils new Blackwell AI accelerator platform at Computex", "reuters", "https://x/nvda", 0.0, 9.4),
        (_ts(8), "NVDA AI server partner says data center demand accelerates", "bloomberg", "https://x/nvda2", 0.0, 8.8),
    ]
    out = ccm.build_cycle_events(rows, now=NOW)
    assert len(out["events"]) == 1
    ev = out["events"][0]
    assert ev["ticker"] == "NVDA"
    assert ev["level"] == "urgent"
    assert ev["kind"] == "fresh_ai_catalyst"
    assert ev["recent_mentions_30m"] == 2


def test_stale_high_volume_ai_ticker_becomes_profit_taking_watch():
    rows = []
    for i in range(8):
        rows.append((
            _ts(130 + i * 20),
            f"AXTI AI optical chip sympathy story number {i}",
            "rss",
            f"https://x/axti/{i}",
            0.0,
            8.0,
        ))
    out = ccm.build_cycle_events(rows, now=NOW)
    assert len(out["events"]) == 1
    ev = out["events"][0]
    assert ev["ticker"] == "AXTI"
    assert ev["level"] == "watch"
    assert ev["kind"] == "stale_catalyst_profit_taking_risk"


def test_low_score_single_article_does_not_alert():
    rows = [
        (_ts(3), "AMD stock mentioned in broad market recap", "rss", "https://x/amd", 0.0, 4.5)
    ]
    out = ccm.build_cycle_events(rows, now=NOW)
    assert out["events"] == []


def test_throttle_suppresses_repeat_and_keeps_state():
    event = {
        "ticker": "MU",
        "kind": "fresh_ai_catalyst",
        "level": "urgent",
        "fresh_score": 10.0,
    }
    first, state = ccm.select_new_events([event], {}, now_epoch=1000.0)
    assert first == [event]
    second, state2 = ccm.select_new_events([event], state, now_epoch=1100.0)
    assert second == []
    assert state2["fresh_ai_catalyst:MU"] == 1000.0


def test_urgent_message_mentions_jonathan_and_sao():
    event = {
        "ticker": "MU",
        "kind": "fresh_ai_catalyst",
        "level": "urgent",
        "reason": "fresh high-score AI catalyst",
        "fresh_score": 10.2,
        "latest_age_min": 2.0,
        "recent_mentions_30m": 2,
        "source_count_30m": 2,
        "catalyst": "EARNINGS",
        "catalyst_confidence": 1.0,
        "source": "reuters",
        "title": "MU memory demand accelerates on AI server demand",
        "url": "https://example.com/mu",
    }
    msg = ccm.format_event(event)
    assert "<@454961974048980992>" in msg
    assert "<@702863115276124211>" in msg
    assert "[CATALYST URGENT]" in msg
