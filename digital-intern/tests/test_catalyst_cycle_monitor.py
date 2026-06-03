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


def test_known_earnings_recap_does_not_become_urgent():
    rows = [
        (
            _ts(0.5),
            "'Nobody wants to be left behind' in AI adoption: HPE CEO on Q2 earnings",
            "Finnhub/Yahoo",
            "https://x/hpe",
            0.0,
            8.5,
        ),
        (
            _ts(4),
            "HPE Q2 earnings recap highlights AI server demand",
            "rss",
            "https://x/hpe2",
            0.0,
            8.2,
        ),
    ]
    out = ccm.build_cycle_events(rows, now=NOW)
    assert out["events"] == []


def test_broad_market_technical_headline_does_not_ping():
    rows = [
        (
            _ts(2),
            "S&P 500, Nasdaq, Dow Futures Ease After Another Record Close As AI Momentum Cushions Iran's Expanding Strikes: MRVL, AVGO, MSFT In Focus",
            "yfinance/Stocktwits",
            "https://x/avgo",
            0.0,
            9.2,
        ),
        (
            _ts(6),
            "AVGO and MRVL in focus as AI momentum lifts chip stocks",
            "rss",
            "https://x/avgo2",
            0.0,
            8.7,
        ),
    ]
    out = ccm.build_cycle_events(rows, now=NOW)
    assert out["events"] == []


def test_syndicated_microsoft_build_recap_does_not_ping():
    rows = [
        (
            _ts(2),
            "Microsoft Build: MSFT partners with NVDA on AI hardware push, launches new models and quantum chip - MSN",
            "GN: Microsoft",
            "https://x/nvda-msn",
            0.0,
            9.4,
        ),
        (
            _ts(5),
            "MSFT and NVDA AI hardware partnership recap from Microsoft Build",
            "rss",
            "https://x/nvda-build",
            0.0,
            8.8,
        ),
    ]
    out = ccm.build_cycle_events(rows, now=NOW)
    assert out["events"] == []


def test_bearish_stocktwits_chatter_does_not_become_buy_watch():
    rows = [
        (
            _ts(0.5),
            "$AVGO it can easily drop to 440 support Design chip, also $SNPS $CDNS design chip, crash at earnings",
            "stocktwits",
            "https://x/avgo-stocktwits",
            0.0,
            9.8,
        ),
        (
            _ts(3),
            "$AVGO $SNPS $CDNS shorts watching support before earnings",
            "stocktwits",
            "https://x/avgo-stocktwits2",
            0.0,
            9.1,
        ),
    ]
    out = ccm.build_cycle_events(rows, now=NOW)
    assert out["events"] == []


def test_stocktwits_wrappers_do_not_count_as_independent_sources():
    rows = [
        (
            _ts(0.7),
            "$AVGO To me Hock Tan's biggest surprises are not the headline earnings number. They are casual Q&A comments about AI demand.",
            "stocktwits",
            "https://stocktwits.com/message/655283710",
            0.0,
            9.9,
        ),
        (
            _ts(3),
            "$AVGO Bears eat crumbs. Bulls eat a buffet. If AVGO has great earnings everything rips.",
            "yfinance/Stocktwits",
            "https://stocktwits.com/message/655301716",
            0.0,
            9.7,
        ),
    ]
    out = ccm.build_cycle_events(rows, now=NOW)
    assert out["events"] == []


def test_stocktwits_plus_one_default_source_does_not_corroborate():
    rows = [
        (
            _ts(1),
            "$AVGO Broadcom earnings setup looks bullish as AI revenue remains the main growth driver.",
            "stocktwits",
            "https://stocktwits.com/message/1",
            0.0,
            9.8,
        ),
        (
            _ts(4),
            "After-Hours Earnings Report for June 3, 2026: AVGO, CRWD, VEEV, FIVE",
            "Nasdaq/ETFs",
            "https://x/earnings-calendar",
            0.0,
            6.0,
        ),
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


def test_select_new_events_sends_only_urgent_by_default():
    watch = {
        "ticker": "HPE",
        "kind": "fresh_ai_catalyst_watch",
        "level": "watch",
        "fresh_score": 8.2,
    }
    urgent = {
        "ticker": "NVDA",
        "kind": "fresh_ai_catalyst",
        "level": "urgent",
        "fresh_score": 10.0,
    }
    selected, state = ccm.select_new_events([watch, urgent], {}, now_epoch=1000.0)
    assert selected == [urgent]
    assert "fresh_ai_catalyst:NVDA" in state
    assert "fresh_ai_catalyst_watch:HPE" not in state


def test_select_new_events_can_include_watch_when_requested():
    watch = {
        "ticker": "HPE",
        "kind": "fresh_ai_catalyst_watch",
        "level": "watch",
        "fresh_score": 8.2,
    }
    selected, state = ccm.select_new_events([watch], {}, now_epoch=1000.0, min_level="watch")
    assert selected == [watch]
    assert state["fresh_ai_catalyst_watch:HPE"] == 1000.0


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
    assert "**$MU - BUY WATCH**  [URGENT]" in msg
    assert "Stats: score 10.2 | 2.0m old | 2 mentions, 2 sources | EARNINGS 100%" in msg
    assert "Source: reuters" in msg
    assert 'Latest: "MU memory demand accelerates on AI server demand"' in msg
