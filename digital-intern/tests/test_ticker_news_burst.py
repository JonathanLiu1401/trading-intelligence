"""``ArticleStore.ticker_news_burst`` — per-ticker news-volume burst detector.

The analyst-facing "is the wire heating up on my book RIGHT NOW?" surface for
the between-briefings window. Live evidence (2026-05-26 1h vs 23h):
SOXX 18×, MU 12.67×, QBTS 12.55×, DRAM 10.31× — none surfaced anywhere else
in the system.

Pins:
  * COLD when ticker has zero current mentions (regardless of baseline)
  * BLAZING needs both spike >= 10 AND count_window >= 5
  * HOT needs spike >= 5 AND count_window >= 3
  * WARMING needs spike >= 2 AND count_window >= 2
  * sort is spike desc, then count_window desc, then alpha
  * backtest:// rows MUST NOT count toward either window or baseline
  * read-only — no ai_score/ml_score/score_source/urgency mutation
"""
from __future__ import annotations

from datetime import datetime, timezone, timedelta

import pytest


def _insert(store, *, id, url, title, source, first_seen):
    with store._write_lock:
        store.conn.execute(
            "INSERT INTO articles "
            "(id, url, title, source, published, kw_score, ai_score, urgency, "
            " first_seen, cycle, ml_score, score_source) "
            "VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
            (id, url, title, source, "", 1.0, 0.0, 0,
             first_seen, 0, None, None),
        )
        store.conn.commit()


def _iso_minus(minutes: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(minutes=minutes)).isoformat()


def test_empty_db_all_cold(store):
    res = store.ticker_news_burst(tickers=["NVDA", "MU"])
    assert res["n_window"] == 0
    assert res["n_baseline"] == 0
    assert res["hottest"] is None
    assert res["n_hot"] == 0
    verdicts = {r["ticker"]: r["verdict"] for r in res["by_ticker"]}
    assert verdicts == {"NVDA": "COLD", "MU": "COLD"}


def test_blazing_spike_when_window_overwhelms_baseline(store):
    """5 fresh NVDA mentions vs 1 mention spread over the 24h baseline →
    spike = 5 / max(1/24, 0.5) = 5 / 0.5 = 10.0, count_window=5 → BLAZING."""
    # 5 in last hour
    for i in range(5):
        _insert(store, id=f"win{i}",
                url=f"https://reuters.com/nvda-{i}",
                title="NVDA hits new record on earnings beat",
                source="reuters", first_seen=_iso_minus(30 + i))
    # 1 in the prior 23h
    _insert(store, id="base1", url="https://reuters.com/nvda-base",
            title="NVDA quarterly outlook",
            source="reuters", first_seen=_iso_minus(60 * 5))

    res = store.ticker_news_burst(tickers=["NVDA"], window_h=1.0, baseline_h=24.0)
    nvda = next(r for r in res["by_ticker"] if r["ticker"] == "NVDA")
    assert nvda["count_window"] == 5
    assert nvda["count_baseline"] == 1
    # base_per_h = 1/24 ≈ 0.04 → floor at 0.5 → spike = 5/0.5 = 10.0 → BLAZING
    assert nvda["spike"] == pytest.approx(10.0, abs=0.01)
    assert nvda["verdict"] == "BLAZING"
    assert res["hottest"] == "NVDA"
    assert res["n_hot"] == 1


def test_hot_verdict_threshold(store):
    """Just over the HOT threshold: 3 in window, 1 in baseline.
    base_per_h = 1/24 → floor 0.5 → spike = 3/0.5 = 6 (>=5), cw=3 → HOT."""
    for i in range(3):
        _insert(store, id=f"w{i}", url=f"https://x/w{i}",
                title="MU beats estimates", source="reuters",
                first_seen=_iso_minus(15 + i))
    _insert(store, id="b", url="https://x/b",
            title="MU long-term thesis", source="reuters",
            first_seen=_iso_minus(60 * 8))
    res = store.ticker_news_burst(tickers=["MU"])
    mu = next(r for r in res["by_ticker"] if r["ticker"] == "MU")
    assert mu["verdict"] == "HOT"


def test_warming_verdict_threshold(store):
    """2 in window, modest baseline → WARMING."""
    for i in range(2):
        _insert(store, id=f"w{i}", url=f"https://x/w{i}",
                title="QBTS earnings", source="rss",
                first_seen=_iso_minus(20 + i))
    # 4 in baseline → 4/24 = 0.167 → floor 0.5 → spike = 2/0.5 = 4 → ≥2 → WARMING
    # (HOT would need spike>=5 AND cw>=3; cw=2 so falls to WARMING)
    for i in range(4):
        _insert(store, id=f"b{i}", url=f"https://x/b{i}",
                title="QBTS history", source="rss",
                first_seen=_iso_minus(60 * (i + 2)))
    res = store.ticker_news_burst(tickers=["QBTS"])
    qbts = next(r for r in res["by_ticker"] if r["ticker"] == "QBTS")
    assert qbts["verdict"] == "WARMING"


def test_normal_when_count_window_equals_baseline_rate(store):
    """Window matches the baseline per-hour rate → NORMAL (not WARMING).

    Insert 1 in window + 23 spread across the prior 23h so spike ≈
    1 / (23/23) = 1.0 → NORMAL (below the WARMING ≥2 threshold). The 23h
    fixture (not 24) is deliberate: at runtime ``baseline_h=24`` means the
    baseline window spans now-25h..now-1h, and a row inserted exactly at
    -25h can fall outside that window due to the brief delay between insert
    and query — keep all rows safely inside the window."""
    _insert(store, id="w0", url="https://x/w0",
            title="NVDA market update", source="rss",
            first_seen=_iso_minus(20))
    for i in range(23):
        _insert(store, id=f"b{i}", url=f"https://x/b{i}",
                title="NVDA market update", source="rss",
                first_seen=_iso_minus(60 * (i + 2)))
    res = store.ticker_news_burst(tickers=["NVDA"], window_h=1.0, baseline_h=24.0)
    nvda = next(r for r in res["by_ticker"] if r["ticker"] == "NVDA")
    assert nvda["count_window"] == 1
    assert nvda["count_baseline"] == 23
    # spike = 1 / max(23/24, 0.5) = 1 / 0.958 ≈ 1.04 → NORMAL
    assert nvda["spike"] == pytest.approx(1.04, abs=0.05)
    assert nvda["verdict"] == "NORMAL"


def test_backtest_rows_excluded(store):
    """``_LIVE_ONLY_CLAUSE`` — backtest:// rows MUST NOT count."""
    # 3 fresh backtest rows that would otherwise trip BLAZING
    for i in range(3):
        _insert(store, id=f"bt{i}",
                url=f"backtest://run_1/2026-05-26/BUY/NVDA-{i}",
                title="NVDA backtest synthetic", source="backtest_run_1",
                first_seen=_iso_minus(5 + i))
    # No real rows
    res = store.ticker_news_burst(tickers=["NVDA"])
    nvda = next(r for r in res["by_ticker"] if r["ticker"] == "NVDA")
    assert nvda["count_window"] == 0
    assert nvda["verdict"] == "COLD"


def test_sort_by_spike_then_count(store):
    """Multi-ticker sort: highest spike first, ties broken by count_window,
    then alphabetical."""
    # NVDA: 5 in window, 1 in baseline → spike 10 BLAZING
    for i in range(5):
        _insert(store, id=f"n{i}", url=f"https://x/n{i}",
                title="NVDA news", source="rss", first_seen=_iso_minus(10+i))
    _insert(store, id="nb", url="https://x/nb",
            title="NVDA history", source="rss", first_seen=_iso_minus(180))
    # MU: 3 in window, 1 in baseline → spike 6 HOT
    for i in range(3):
        _insert(store, id=f"m{i}", url=f"https://x/m{i}",
                title="MU news", source="rss", first_seen=_iso_minus(10+i))
    _insert(store, id="mb", url="https://x/mb",
            title="MU history", source="rss", first_seen=_iso_minus(200))
    res = store.ticker_news_burst(tickers=["MU", "NVDA"])
    order = [r["ticker"] for r in res["by_ticker"]]
    # NVDA must come first (higher spike)
    assert order[0] == "NVDA"
    assert order[1] == "MU"
    assert res["hottest"] == "NVDA"
    assert res["n_hot"] == 2


def test_word_boundary_no_substring_match(store):
    """A real word containing a ticker substring must NOT count.

    e.g. "AMAT" is held; "AUTOMATIC" must not match "MAT". Compiled regex
    uses \\b so partial matches don't trigger false spikes."""
    _insert(store, id="real", url="https://x/r",
            title="AMAT reports quarterly revenue beat", source="rss",
            first_seen=_iso_minus(5))
    _insert(store, id="false_hit", url="https://x/f",
            title="AUTOMATIC SHIPMENTS RESUME", source="rss",
            first_seen=_iso_minus(10))
    res = store.ticker_news_burst(tickers=["AMAT"])
    amat = next(r for r in res["by_ticker"] if r["ticker"] == "AMAT")
    assert amat["count_window"] == 1  # only AMAT matched, not AUTOMATIC


def test_dollar_prefix_ticker_matches(store):
    """``$NVDA`` style ticker references must count — common in StockTwits
    / Twitter feeds and an analyst expects them to register."""
    _insert(store, id="usd", url="https://x/u",
            title="$NVDA breaking out on volume", source="stocktwits",
            first_seen=_iso_minus(5))
    res = store.ticker_news_burst(tickers=["NVDA"])
    nvda = next(r for r in res["by_ticker"] if r["ticker"] == "NVDA")
    assert nvda["count_window"] == 1


def test_zero_baseline_zero_window_returns_none_spike(store):
    """When BOTH windows are empty, spike is None (not 0, not divide-by-zero)
    so a consumer can distinguish 'no data' from 'literally zero activity'.
    Verdict is COLD."""
    res = store.ticker_news_burst(tickers=["XYZ"])
    xyz = next(r for r in res["by_ticker"] if r["ticker"] == "XYZ")
    assert xyz["spike"] is None
    assert xyz["verdict"] == "COLD"


def test_read_only_no_mutation(store):
    """The method must be pure read — no ai_score/ml_score/score_source/
    urgency mutation. Mirrors every other analytics method's invariants."""
    _insert(store, id="x", url="https://x/x",
            title="NVDA news", source="rss", first_seen=_iso_minus(5))
    # Snapshot row state
    before = store.conn.execute(
        "SELECT id, ai_score, ml_score, score_source, urgency FROM articles"
    ).fetchall()
    store.ticker_news_burst(tickers=["NVDA"])
    after = store.conn.execute(
        "SELECT id, ai_score, ml_score, score_source, urgency FROM articles"
    ).fetchall()
    assert before == after, (
        "ticker_news_burst must not mutate ai_score / ml_score / "
        "score_source / urgency — pure analytics read"
    )


def test_default_tickers_use_live_portfolio(store, monkeypatch):
    """When ``tickers=None``, the method must fall back to
    ml.features.LIVE_PORTFOLIO_TICKERS so an analyst who calls
    ``store.ticker_news_burst()`` gets the held universe by default."""
    # Patch the live set so the test is hermetic
    import ml.features as features
    monkeypatch.setattr(features, "LIVE_PORTFOLIO_TICKERS", {"FOO", "BAR"})
    _insert(store, id="x", url="https://x/x",
            title="FOO announces deal", source="rss",
            first_seen=_iso_minus(5))
    res = store.ticker_news_burst()  # no tickers arg
    tickers_seen = {r["ticker"] for r in res["by_ticker"]}
    # FOO and BAR were the live set; both present in the result
    assert tickers_seen == {"FOO", "BAR"}


def test_baseline_excludes_window(store):
    """Baseline rows must NOT include window rows — otherwise the rate
    estimate is artificially inflated, killing real spike detection."""
    # 5 in window
    for i in range(5):
        _insert(store, id=f"w{i}", url=f"https://x/w{i}",
                title="NVDA breaking", source="rss",
                first_seen=_iso_minus(10 + i))
    # 0 in baseline window
    res = store.ticker_news_burst(tickers=["NVDA"], window_h=1.0, baseline_h=24.0)
    nvda = next(r for r in res["by_ticker"] if r["ticker"] == "NVDA")
    assert nvda["count_baseline"] == 0
    assert nvda["count_window"] == 5
    assert nvda["verdict"] == "BLAZING"
