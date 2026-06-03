from __future__ import annotations

from types import SimpleNamespace

import requests

from collectors import yahoo_ticker_rss


class _FakeResp:
    def __init__(self, *, status_code=200, content=b"", headers=None):
        self.status_code = status_code
        self.content = content
        self.headers = headers or {}

    def raise_for_status(self):
        if self.status_code >= 400:
            raise requests.HTTPError(f"HTTP {self.status_code}")


def _parsed(entries, *, bozo=0):
    return SimpleNamespace(entries=entries, bozo=bozo)


def _entry(title="NVDA headline", link="https://finance.yahoo.com/news/nvda"):
    return {
        "title": title,
        "link": link,
        "summary": "summary text",
        "published": "Wed, 03 Jun 2026 10:00:00 GMT",
    }


def test_fetch_ticker_retries_transient_network_failure_and_keeps_shape(
    monkeypatch, capsys
):
    attempts = {"n": 0}

    def fake_get(url, **kwargs):
        attempts["n"] += 1
        assert url == "https://finance.yahoo.com/rss/headline?s=NVDA"
        assert kwargs["timeout"] == yahoo_ticker_rss.REQUEST_TIMEOUT
        assert kwargs["headers"]["User-Agent"] == yahoo_ticker_rss.USER_AGENT
        if attempts["n"] == 1:
            raise requests.ConnectionError("temporary reset")
        return _FakeResp(content=b"<rss>ok</rss>")

    monkeypatch.setattr(requests, "get", fake_get)
    monkeypatch.setattr(yahoo_ticker_rss.feedparser, "parse",
                        lambda content: _parsed([_entry()]))
    monkeypatch.setattr(yahoo_ticker_rss.time, "sleep", lambda _delay: None)

    articles = yahoo_ticker_rss._fetch_ticker("NVDA")

    assert attempts["n"] == 2
    assert articles == [{
        "title": "NVDA headline",
        "link": "https://finance.yahoo.com/news/nvda",
        "summary": "summary text",
        "published": "Wed, 03 Jun 2026 10:00:00 GMT",
        "source": "YahooFinance/NVDA",
        "_ticker": "NVDA",
    }]
    assert "NVDA attempt 1/" in capsys.readouterr().out


def test_fetch_ticker_retries_http_429_and_prints_final_failure(
    monkeypatch, capsys
):
    sleeps: list[float] = []

    monkeypatch.setattr(
        requests,
        "get",
        lambda url, **kwargs: _FakeResp(
            status_code=429, headers={"Retry-After": "0.25"}
        ),
    )
    monkeypatch.setattr(yahoo_ticker_rss.time, "sleep", sleeps.append)

    articles = yahoo_ticker_rss._fetch_ticker("AMD")

    assert articles == []
    assert len(sleeps) == yahoo_ticker_rss.MAX_FETCH_ATTEMPTS - 1
    assert sleeps == [0.25, 0.25]
    out = capsys.readouterr().out
    assert "[yahoo_ticker_rss] AMD attempt 1/" in out
    assert "HTTP 429" in out
    assert "AMD failed after" in out


def test_collect_logs_ticker_when_worker_raises(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(yahoo_ticker_rss, "DB_PATH", tmp_path / "seen.db")
    monkeypatch.setattr(yahoo_ticker_rss, "CURSOR_PATH", tmp_path / "cursor.json")
    monkeypatch.setattr(yahoo_ticker_rss, "PER_TICKER_COOLDOWN_SEC", 0)
    monkeypatch.setattr(yahoo_ticker_rss, "_load_tickers", lambda: ["NVDA", "AMD"])

    def fake_fetch(ticker):
        if ticker == "AMD":
            raise RuntimeError("unexpected parser crash")
        return [_entry()]

    monkeypatch.setattr(yahoo_ticker_rss, "_fetch_ticker", fake_fetch)

    articles = yahoo_ticker_rss.collect_yahoo_ticker_rss(batch=2)

    assert len(articles) == 1
    assert articles[0]["title"] == "NVDA headline"
    assert "AMD worker error: unexpected parser crash" in capsys.readouterr().out


def test_collect_preserves_round_robin_cursor_and_seen_dedup(tmp_path, monkeypatch):
    monkeypatch.setattr(yahoo_ticker_rss, "DB_PATH", tmp_path / "seen.db")
    monkeypatch.setattr(yahoo_ticker_rss, "CURSOR_PATH", tmp_path / "cursor.json")
    monkeypatch.setattr(yahoo_ticker_rss, "PER_TICKER_COOLDOWN_SEC", 0)
    monkeypatch.setattr(yahoo_ticker_rss, "_load_tickers", lambda: ["NVDA", "AMD"])
    monkeypatch.setattr(
        yahoo_ticker_rss,
        "_fetch_ticker",
        lambda ticker: [{
            **_entry(
                title=f"{ticker} headline",
                link=f"https://finance.yahoo.com/news/{ticker}",
            ),
            "source": f"YahooFinance/{ticker}",
            "_ticker": ticker,
        }],
    )

    first = yahoo_ticker_rss.collect_yahoo_ticker_rss(batch=2)
    second = yahoo_ticker_rss.collect_yahoo_ticker_rss(batch=2)

    assert {a["_ticker"] for a in first} == {"NVDA", "AMD"}
    assert second == []
    state = yahoo_ticker_rss._load_cursor()
    assert state["index"] == 0
    assert set(state["last_polled"]) == {"NVDA", "AMD"}
