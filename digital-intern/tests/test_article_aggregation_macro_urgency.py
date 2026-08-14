from __future__ import annotations

from article_aggregation_daemon import _is_breaking_macro_market_event


def test_iran_war_halt_oil_market_story_is_macro_urgent():
    art = {
        "title": (
            "Iran, US agree to halt war and reopen Hormuz, "
            "sending oil prices tumbling"
        ),
        "summary": "",
    }
    assert _is_breaking_macro_market_event(art, 7.7) is True


def test_low_score_macro_like_story_stays_nonurgent():
    art = {
        "title": "Iran war opinion column mentions oil prices",
        "summary": "",
    }
    assert _is_breaking_macro_market_event(art, 4.9) is False


def test_high_score_single_name_story_does_not_use_macro_fast_path():
    art = {
        "title": "Nvidia price target lifted as AI demand grows",
        "summary": "",
    }
    assert _is_breaking_macro_market_event(art, 9.2) is False
