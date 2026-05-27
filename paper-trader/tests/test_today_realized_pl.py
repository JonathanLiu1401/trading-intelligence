"""Tests for paper_trader.analytics.today_realized_pl + the
``/api/today-realized-pl`` endpoint.

Covers (1) the pure builder's per-row date filtering, aggregation, and
verdict precedence, (2) the dashboard endpoint envelope, and (3) the
silence-by-default contract callers can rely on.
"""
from __future__ import annotations

import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from paper_trader.analytics.today_realized_pl import (  # noqa: E402
    _BREAKEVEN_EPSILON_USD,
    _MAX_CLOSES_RENDERED,
    _ny_today,
    _parse_iso_to_ny_date,
    build_today_realized_pl,
)

NY = ZoneInfo("America/New_York")
UTC = timezone.utc


def _ts(year, month, day, hour=12, minute=0, tz=UTC):
    return datetime(year, month, day, hour, minute, tzinfo=tz)


def _close_row(ticker, realized_pl, closed_at, *, cost=100.0, hold_seconds=3600,
               opened_at=None, realized_pl_pct=None, type_="stock"):
    """A row in the shape ``store.closed_positions(N)`` returns."""
    return {
        "ticker": ticker,
        "type": type_,
        "closed_at": (closed_at.isoformat()
                      if isinstance(closed_at, datetime) else closed_at),
        "opened_at": (opened_at.isoformat()
                      if isinstance(opened_at, datetime) else opened_at),
        "realized_pl": realized_pl,
        "realized_pl_pct": realized_pl_pct,
        "cost": cost,
        "proceeds": cost + realized_pl,
        "hold_seconds": hold_seconds,
        "n_trades": 2,
    }


class TestParseDate:
    def test_iso_utc_string_to_ny_date(self):
        # 2026-05-27T03:00:00Z is 2026-05-26 23:00 ET → NY date 2026-05-26.
        d = _parse_iso_to_ny_date("2026-05-27T03:00:00+00:00")
        assert d.isoformat() == "2026-05-26"

    def test_iso_utc_midday_same_date(self):
        # Midday UTC == morning NY, same date.
        d = _parse_iso_to_ny_date("2026-05-27T14:00:00+00:00")
        assert d.isoformat() == "2026-05-27"

    def test_iso_with_zulu_z(self):
        d = _parse_iso_to_ny_date("2026-05-27T14:00:00Z")
        assert d.isoformat() == "2026-05-27"

    def test_naive_iso_treated_as_utc(self):
        # No tz → treated as UTC. 2026-05-27 14:00 UTC = 2026-05-27 NY.
        d = _parse_iso_to_ny_date("2026-05-27T14:00:00")
        assert d.isoformat() == "2026-05-27"

    def test_datetime_object_input(self):
        dt = datetime(2026, 5, 27, 14, 0, tzinfo=UTC)
        d = _parse_iso_to_ny_date(dt)
        assert d.isoformat() == "2026-05-27"

    def test_empty_returns_none(self):
        assert _parse_iso_to_ny_date(None) is None
        assert _parse_iso_to_ny_date("") is None

    def test_unparseable_returns_none(self):
        assert _parse_iso_to_ny_date("not a date") is None
        assert _parse_iso_to_ny_date("2026-13-99") is None


class TestEmptyOrNoCloses:
    def test_empty_list_returns_no_closes(self):
        out = build_today_realized_pl([])
        assert out["verdict"] == "NO_CLOSES_TODAY"
        assert out["n_closes"] == 0
        assert out["net_realized_usd"] == 0.0
        assert out["closes"] == []
        assert out["biggest_win"] is None
        assert out["biggest_loss"] is None

    def test_none_input_returns_no_closes(self):
        out = build_today_realized_pl(None)
        assert out["verdict"] == "NO_CLOSES_TODAY"

    def test_non_list_input_returns_no_closes(self):
        out = build_today_realized_pl({"not": "a list"})  # type: ignore[arg-type]
        assert out["verdict"] == "NO_CLOSES_TODAY"

    def test_all_rows_from_other_days_filtered_out(self):
        now = datetime.now(UTC)
        yesterday = now - timedelta(days=2)  # safely before NY today
        rows = [
            _close_row("NVDA", 10.0, yesterday),
            _close_row("MU", -5.0, yesterday - timedelta(days=1)),
        ]
        out = build_today_realized_pl(rows, now=now)
        assert out["verdict"] == "NO_CLOSES_TODAY"
        assert out["n_closes"] == 0


class TestVerdictPrecedence:
    def _today_close(self, ticker, pl, now, *, cost=100.0, hold_seconds=3600):
        # Close at midday NY today so it's unambiguously today regardless of
        # when the test runs.
        ny_today = now.astimezone(NY).date()
        closed_at = datetime(
            ny_today.year, ny_today.month, ny_today.day, 12, 0, tzinfo=NY
        ).astimezone(UTC)
        return _close_row(ticker, pl, closed_at, cost=cost,
                          hold_seconds=hold_seconds)

    def test_winning_day_when_net_positive(self):
        now = datetime.now(UTC)
        rows = [self._today_close("NVDA", 10.5, now),
                self._today_close("MU", 5.0, now)]
        out = build_today_realized_pl(rows, now=now)
        assert out["verdict"] == "WINNING_DAY"
        assert out["net_realized_usd"] == pytest.approx(15.5)
        assert out["n_winners"] == 2
        assert out["n_losers"] == 0
        assert out["biggest_win"]["ticker"] == "NVDA"
        assert out["biggest_loss"] is None

    def test_losing_day_when_net_negative(self):
        now = datetime.now(UTC)
        rows = [self._today_close("NVDA", -20.0, now),
                self._today_close("MU", 5.0, now)]
        out = build_today_realized_pl(rows, now=now)
        assert out["verdict"] == "LOSING_DAY"
        assert out["net_realized_usd"] == pytest.approx(-15.0)
        assert out["biggest_loss"]["ticker"] == "NVDA"

    def test_breakeven_day_within_epsilon(self):
        now = datetime.now(UTC)
        rows = [self._today_close("NVDA", 5.0, now),
                self._today_close("MU", -5.0, now)]
        out = build_today_realized_pl(rows, now=now)
        assert out["verdict"] == "BREAKEVEN_DAY"
        assert abs(out["net_realized_usd"]) <= _BREAKEVEN_EPSILON_USD

    def test_breakeven_at_sub_epsilon(self):
        """A net of $0.0009 (float-rounding noise) reads as BREAKEVEN, not
        WINNING — the precision-noise floor matters for trader trust."""
        now = datetime.now(UTC)
        rows = [self._today_close("NVDA", 0.0009, now)]
        out = build_today_realized_pl(rows, now=now)
        assert out["verdict"] == "BREAKEVEN_DAY"
        assert out["n_winners"] == 0
        assert out["n_scratch"] == 1


class TestWinLossClassification:
    def _today(self, now):
        ny_today = now.astimezone(NY).date()
        return datetime(
            ny_today.year, ny_today.month, ny_today.day, 12, 0, tzinfo=NY
        ).astimezone(UTC)

    def test_n_winners_losers_scratch_split(self):
        now = datetime.now(UTC)
        closed_at = self._today(now)
        rows = [
            _close_row("A", 10.0, closed_at),     # winner
            _close_row("B", -8.0, closed_at),     # loser
            _close_row("C", 0.001, closed_at),    # scratch (sub-epsilon)
            _close_row("D", 0.0, closed_at),      # scratch
            _close_row("E", 7.0, closed_at),      # winner
        ]
        out = build_today_realized_pl(rows, now=now)
        assert out["n_closes"] == 5
        assert out["n_winners"] == 2
        assert out["n_losers"] == 1
        assert out["n_scratch"] == 2
        assert out["verdict"] == "WINNING_DAY"

    def test_biggest_win_loss_sorted_correctly(self):
        now = datetime.now(UTC)
        closed_at = self._today(now)
        rows = [
            _close_row("BIGGEST_WIN", 50.0, closed_at),
            _close_row("SMALL_WIN", 5.0, closed_at),
            _close_row("BIGGEST_LOSS", -30.0, closed_at),
            _close_row("SMALL_LOSS", -2.0, closed_at),
        ]
        out = build_today_realized_pl(rows, now=now)
        assert out["biggest_win"]["ticker"] == "BIGGEST_WIN"
        assert out["biggest_win"]["realized_pl"] == 50.0
        assert out["biggest_loss"]["ticker"] == "BIGGEST_LOSS"
        assert out["biggest_loss"]["realized_pl"] == -30.0
        # `closes` sorted best-first (descending realized_pl).
        assert [c["ticker"] for c in out["closes"]] == [
            "BIGGEST_WIN", "SMALL_WIN", "SMALL_LOSS", "BIGGEST_LOSS"
        ]


class TestPercentAndAverages:
    def _today_at(self, now):
        ny_today = now.astimezone(NY).date()
        return datetime(
            ny_today.year, ny_today.month, ny_today.day, 12, 0, tzinfo=NY
        ).astimezone(UTC)

    def test_net_realized_pct_uses_total_cost(self):
        """net_realized_pct = net / total_cost, NOT % of starting book —
        the right framing is "return on capital deployed today"."""
        now = datetime.now(UTC)
        closed_at = self._today_at(now)
        rows = [
            _close_row("A", 30.0, closed_at, cost=200.0),
            _close_row("B", -10.0, closed_at, cost=100.0),
        ]
        out = build_today_realized_pl(rows, now=now)
        # Net = +20, total_cost = 300, pct = 6.6666% → rounded 4dp.
        assert out["total_cost_basis_usd"] == pytest.approx(300.0)
        assert out["net_realized_pct"] == pytest.approx(6.6667, abs=1e-3)

    def test_net_realized_pct_none_when_total_cost_zero(self):
        """A round-trip with cost=0 (e.g. fully recovered from a free-stock
        promo) makes the % undefined — emit None rather than divide-by-zero."""
        now = datetime.now(UTC)
        closed_at = self._today_at(now)
        rows = [_close_row("A", 5.0, closed_at, cost=0.0)]
        out = build_today_realized_pl(rows, now=now)
        assert out["net_realized_pct"] is None

    def test_avg_hold_computed_from_hold_seconds(self):
        now = datetime.now(UTC)
        closed_at = self._today_at(now)
        rows = [
            _close_row("A", 5.0, closed_at, hold_seconds=3600),     # 1h
            _close_row("B", -2.0, closed_at, hold_seconds=10800),   # 3h
            _close_row("C", 1.0, closed_at, hold_seconds=1800),     # 0.5h
        ]
        out = build_today_realized_pl(rows, now=now)
        # mean of 3600, 10800, 1800 = 5400s = 1.5h
        assert out["avg_hold_seconds"] == 5400
        assert out["avg_hold_hours"] == pytest.approx(1.5, abs=1e-3)

    def test_avg_hold_none_when_no_hold_seconds(self):
        now = datetime.now(UTC)
        closed_at = self._today_at(now)
        rows = [_close_row("A", 5.0, closed_at, hold_seconds=None)]
        out = build_today_realized_pl(rows, now=now)
        assert out["avg_hold_seconds"] is None
        assert out["avg_hold_hours"] is None


class TestNyTzCorrectness:
    def test_close_at_23_et_yesterday_excluded_from_today(self):
        """A close that landed at 23:00 ET yesterday — even though it's
        ~04:00 UTC today — is NOT today's. The NY trading day is the
        canonical anchor (close-anchored daily-close report uses the same)."""
        # Pick a fixed "now" so the test is deterministic.
        now = datetime(2026, 5, 27, 14, 0, tzinfo=UTC)  # NY-date 2026-05-27
        # 03:00 UTC 2026-05-27 == 23:00 ET 2026-05-26 (NY-date 2026-05-26).
        yesterday_late_et = datetime(2026, 5, 27, 3, 0, tzinfo=UTC)
        rows = [_close_row("A", 10.0, yesterday_late_et)]
        out = build_today_realized_pl(rows, now=now)
        assert out["verdict"] == "NO_CLOSES_TODAY"

    def test_close_at_00_et_today_included(self):
        """A close at 00:00 ET today (=04:00 UTC) is today's first close."""
        now = datetime(2026, 5, 27, 14, 0, tzinfo=UTC)
        ny_midnight_today_utc = datetime(2026, 5, 27, 4, 0, tzinfo=UTC)
        rows = [_close_row("A", 10.0, ny_midnight_today_utc)]
        out = build_today_realized_pl(rows, now=now)
        assert out["verdict"] == "WINNING_DAY"
        assert out["n_closes"] == 1

    def test_ny_today_helper_returns_ny_date(self):
        # 2026-05-27 03:00 UTC = 2026-05-26 23:00 ET → ny_today=2026-05-26.
        now = datetime(2026, 5, 27, 3, 0, tzinfo=UTC)
        assert _ny_today(now).isoformat() == "2026-05-26"


class TestRowFiltering:
    def _today_at(self, now):
        ny_today = now.astimezone(NY).date()
        return datetime(
            ny_today.year, ny_today.month, ny_today.day, 12, 0, tzinfo=NY
        ).astimezone(UTC)

    def test_rows_missing_realized_pl_filtered(self):
        now = datetime.now(UTC)
        closed_at = self._today_at(now)
        rows = [
            _close_row("A", 5.0, closed_at),
            {"ticker": "B", "closed_at": closed_at.isoformat(),
             "realized_pl": None, "cost": 0.0},
        ]
        out = build_today_realized_pl(rows, now=now)
        assert out["n_closes"] == 1

    def test_rows_with_non_numeric_realized_pl_filtered(self):
        now = datetime.now(UTC)
        closed_at = self._today_at(now)
        rows = [
            _close_row("A", 5.0, closed_at),
            {"ticker": "B", "closed_at": closed_at.isoformat(),
             "realized_pl": "garbage", "cost": 100.0},
        ]
        out = build_today_realized_pl(rows, now=now)
        assert out["n_closes"] == 1

    def test_rows_with_bad_closed_at_filtered(self):
        now = datetime.now(UTC)
        closed_at = self._today_at(now)
        rows = [
            _close_row("A", 5.0, closed_at),
            _close_row("B", 5.0, "not a date"),
            _close_row("C", 5.0, None),
        ]
        out = build_today_realized_pl(rows, now=now)
        assert out["n_closes"] == 1

    def test_non_dict_rows_filtered(self):
        now = datetime.now(UTC)
        closed_at = self._today_at(now)
        rows = [
            _close_row("A", 5.0, closed_at),
            "garbage",
            None,
            42,
        ]
        out = build_today_realized_pl(rows, now=now)
        assert out["n_closes"] == 1


class TestCappedRendering:
    def test_closes_capped_at_max_rendered(self):
        now = datetime.now(UTC)
        ny_today = now.astimezone(NY).date()
        closed_at = datetime(
            ny_today.year, ny_today.month, ny_today.day, 12, 0, tzinfo=NY
        ).astimezone(UTC)
        # Generate one more than the cap so we can verify truncation.
        n = _MAX_CLOSES_RENDERED + 5
        rows = [_close_row(f"T{i}", float(i), closed_at) for i in range(n)]
        out = build_today_realized_pl(rows, now=now)
        # Aggregate counts include ALL closes, only `closes` array is capped.
        assert out["n_closes"] == n
        assert len(out["closes"]) == _MAX_CLOSES_RENDERED


class TestHeadlineComposition:
    def _today_at(self, now):
        ny_today = now.astimezone(NY).date()
        return datetime(
            ny_today.year, ny_today.month, ny_today.day, 12, 0, tzinfo=NY
        ).astimezone(UTC)

    def test_headline_includes_best_and_worst(self):
        now = datetime.now(UTC)
        closed_at = self._today_at(now)
        rows = [
            _close_row("BEST", 50.0, closed_at),
            _close_row("WORST", -20.0, closed_at),
        ]
        out = build_today_realized_pl(rows, now=now)
        assert "BEST" in out["headline"]
        assert "WORST" in out["headline"]
        assert "+$50" in out["headline"]
        assert "-$20" in out["headline"]

    def test_headline_omits_worst_on_clean_winning_day(self):
        """No worst-token on a day with no real losers (silence-when-
        nothing-actionable — a clean day should not invent a 'worst')."""
        now = datetime.now(UTC)
        closed_at = self._today_at(now)
        rows = [
            _close_row("A", 50.0, closed_at),
            _close_row("B", 10.0, closed_at),
        ]
        out = build_today_realized_pl(rows, now=now)
        assert "best A" in out["headline"]
        assert "worst" not in out["headline"]

    def test_headline_omits_best_on_clean_losing_day(self):
        now = datetime.now(UTC)
        closed_at = self._today_at(now)
        rows = [
            _close_row("A", -50.0, closed_at),
            _close_row("B", -10.0, closed_at),
        ]
        out = build_today_realized_pl(rows, now=now)
        assert "worst" in out["headline"]
        assert "best" not in out["headline"]

    def test_headline_scratch_count_when_present(self):
        now = datetime.now(UTC)
        closed_at = self._today_at(now)
        rows = [
            _close_row("A", 10.0, closed_at),
            _close_row("B", 0.0, closed_at),
            _close_row("C", -5.0, closed_at),
        ]
        out = build_today_realized_pl(rows, now=now)
        # 1W/1L/1S — scratch shown explicitly so the operator doesn't
        # mis-read n_closes - winners - losers as "another loss".
        assert "1W/1L/1S" in out["headline"]


class TestEndpointEnvelope:
    """The dashboard endpoint wires the builder to /api/today-realized-pl
    and adds an ``as_of`` field. Verifies the contract callers depend on."""

    @pytest.fixture
    def client(self, monkeypatch, tmp_path):
        from paper_trader import dashboard as dash_mod
        from paper_trader.store import Store

        # Use an isolated store backed by a fresh tmp DB so the test never
        # touches the live paper_trader.db.
        db_path = tmp_path / "paper_trader.db"
        monkeypatch.setattr("paper_trader.store.DB_PATH", db_path)
        monkeypatch.setattr("paper_trader.store._singleton", None)

        # Force the dashboard's store factory to a fresh in-tmp store.
        fresh = Store()
        monkeypatch.setattr(dash_mod, "get_store", lambda: fresh)
        # Also patch the module-imported one used inside the endpoint.
        monkeypatch.setattr("paper_trader.store.get_store",
                            lambda: fresh)
        dash_mod.app.config["TESTING"] = True
        return dash_mod.app.test_client()

    def test_endpoint_returns_no_closes_envelope_on_empty_store(self, client):
        r = client.get("/api/today-realized-pl")
        assert r.status_code == 200
        body = r.get_json()
        assert body["verdict"] == "NO_CLOSES_TODAY"
        assert body["n_closes"] == 0
        # Stable shape — every key the docstring promises is present.
        for k in ("verdict", "headline", "ny_date", "net_realized_usd",
                  "net_realized_pct", "n_closes", "n_winners", "n_losers",
                  "biggest_win", "biggest_loss", "avg_hold_seconds",
                  "avg_hold_hours", "closes", "as_of"):
            assert k in body, f"missing key in envelope: {k!r}"

    def test_endpoint_respects_limit_query_param(self, client):
        # ``limit`` clamps to [1, 1000]; out-of-range silently coerces.
        r = client.get("/api/today-realized-pl?limit=2000")
        assert r.status_code == 200
        # Endpoint still returns a stable envelope even when the limit
        # is invalid — never 500s on a malformed query string.
        body = r.get_json()
        assert body["verdict"] in ("NO_CLOSES_TODAY", "WINNING_DAY",
                                    "LOSING_DAY", "BREAKEVEN_DAY", "ERROR")

    def test_endpoint_invalid_limit_does_not_500(self, client):
        r = client.get("/api/today-realized-pl?limit=garbage")
        assert r.status_code == 200
