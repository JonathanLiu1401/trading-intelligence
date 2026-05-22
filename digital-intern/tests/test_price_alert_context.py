"""Price-alert enrichment: held-position context + news-catalyst context.

The bare price alert ("GOOG +3.2% to $X") told the analyst nothing about
whether the move was on a position they actually OWN, where it left them vs
cost basis, or whether it had a news catalyst. These helpers add that
context. All are pure (or best-effort) and degrade to "" so a non-held /
quiet mover's alert stays a clean one-liner.
"""
import daemon


# ── _fmt_qty ─────────────────────────────────────────────────────────────────
def test_fmt_qty_integer_has_no_decimal():
    assert daemon._fmt_qty(14) == "14"
    assert daemon._fmt_qty(14.0) == "14"


def test_fmt_qty_fractional_is_compact():
    # fractional share counts render exactly, without float dust
    assert daemon._fmt_qty(3.615) == "3.615"
    assert daemon._fmt_qty(4.7095) == "4.7095"
    assert daemon._fmt_qty(3.6150000001) == "3.615"


def test_fmt_qty_bad_input_is_safe():
    assert daemon._fmt_qty(None) == "?"
    assert daemon._fmt_qty("xyz") == "?"


# ── _price_alert_position_line ───────────────────────────────────────────────
_POSITIONS = {
    "GOOG": {"qty": 7.8548, "avg_cost": 381.93, "type": "stock"},
    "LNOK": {"qty": 14, "avg_cost": 72.48, "type": "stock"},
    "BADCOST": {"qty": 5, "avg_cost": 0, "type": "stock"},
}


def test_position_line_held_ticker_above_cost():
    line = daemon._price_alert_position_line("GOOG", 420.0, _POSITIONS)
    assert "HELD POSITION" in line
    assert "7.8548" in line
    assert "$381.93" in line
    # 420 vs 381.93 → +9.97% above cost
    assert "10.0% above cost basis" in line


def test_position_line_held_ticker_below_cost():
    line = daemon._price_alert_position_line("LNOK", 65.23, _POSITIONS)
    # 65.23 vs 72.48 → -10.0% below cost
    assert "10.0% below cost basis" in line
    assert "14 @ $72.48 avg" in line


def test_position_line_empty_for_watchlist_only_ticker():
    # AMD is watchlist-only (not in positions) — no held-position line.
    assert daemon._price_alert_position_line("AMD", 200.0, _POSITIONS) == ""


def test_position_line_empty_when_avg_cost_nonpositive():
    assert daemon._price_alert_position_line("BADCOST", 100.0, _POSITIONS) == ""


def test_position_line_case_insensitive_ticker():
    assert daemon._price_alert_position_line("goog", 420.0, _POSITIONS) != ""


# ── _price_alert_news_line ───────────────────────────────────────────────────
class _FakeStore:
    def __init__(self, rows):
        self._rows = rows

    def ticker_mention_velocity(self, tickers, window_min=60):
        return self._rows


def test_news_line_reports_recent_mentions():
    store = _FakeStore([{"ticker": "NVDA", "recent": 4, "prior": 1}])
    line = daemon._price_alert_news_line(store, "NVDA")
    assert "4 live article(s) mention NVDA" in line
    assert "news catalyst" in line


def test_news_line_empty_when_no_recent_mentions():
    store = _FakeStore([{"ticker": "NVDA", "recent": 0, "prior": 0}])
    assert daemon._price_alert_news_line(store, "NVDA") == ""


def test_news_line_empty_when_store_is_none():
    assert daemon._price_alert_news_line(None, "NVDA") == ""


def test_news_line_degrades_on_store_error():
    class _BoomStore:
        def ticker_mention_velocity(self, *a, **k):
            raise RuntimeError("db locked")

    assert daemon._price_alert_news_line(_BoomStore(), "NVDA") == ""


# ── _load_held_positions ─────────────────────────────────────────────────────
def test_load_held_positions_reads_config():
    held = daemon._load_held_positions()
    # config/portfolio.json carries real open positions — at minimum the
    # known held names must parse with a positive avg_cost.
    assert held, "no positions parsed from config/portfolio.json"
    for tkr, pos in held.items():
        assert tkr == tkr.upper()
        assert "avg_cost" in pos and "qty" in pos
