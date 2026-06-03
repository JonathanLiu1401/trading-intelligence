import sqlite3

from analytics import source_quality


def _init_db(path):
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE articles (
            source TEXT,
            ai_score REAL,
            ml_score REAL,
            kw_score REAL,
            urgency INTEGER,
            url TEXT,
            first_seen TEXT
        )
        """
    )
    rows = [
        ("rss", 8.0, 7.0, 6.0, 2, "https://example.com/a", "2026-06-03T07:00:00+00:00"),
        ("rss", 6.0, None, 4.0, 1, "https://example.com/b", "2026-06-03T06:59:00+00:00"),
        ("gdelt", None, 3.0, 1.0, 0, "https://example.com/c", "2026-06-03T06:58:00+00:00"),
        ("backtest_x", 10.0, 10.0, 10.0, 2, "backtest://x", "2026-06-03T06:57:00+00:00"),
    ]
    conn.executemany(
        "INSERT INTO articles(source, ai_score, ml_score, kw_score, urgency, url, first_seen) "
        "VALUES (?,?,?,?,?,?,?)",
        rows,
    )
    conn.commit()
    conn.close()


def test_compute_reports_quality_deltas_and_excludes_backtests(tmp_path, monkeypatch):
    db_path = tmp_path / "articles.db"
    _init_db(db_path)
    monkeypatch.setattr(source_quality, "DB_PATH", db_path)

    report = source_quality.compute(
        previous_sources={
            "rss": {
                "count": 1,
                "avg_ai_score": 6.0,
                "avg_ml_score": 8.0,
                "avg_kw_score": 5.0,
                "pct_urgent": 0.0,
            }
        }
    )

    assert "backtest_x" not in report["sources"]
    rss = report["sources"]["rss"]
    assert rss["count"] == 2
    assert rss["avg_ai_score"] == 7.0
    assert rss["avg_ml_score"] == 7.0
    assert rss["avg_kw_score"] == 5.0
    assert rss["pct_urgent"] == 0.5
    assert rss["count_delta"] == 1
    assert rss["avg_ai_score_delta"] == 1.0
    assert rss["avg_ml_score_delta"] == -1.0
    assert rss["avg_kw_score_delta"] == 0.0
    assert rss["pct_urgent_delta"] == 0.5

    assert report["sources"]["gdelt"]["avg_ai_score_delta"] is None
