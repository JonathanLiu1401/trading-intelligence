"""Tests for analytics/decision_clock.py — the pure verdict builder.

The pure builder mirrors the inline /api/decision-clock endpoint's
categorisation + verdict logic (the dashboard endpoint is independently
maintained for legacy verdict-string shape — see the builder docstring).
These tests lock the pure-builder shape so a future endpoint extraction
is byte-aligned.

Discriminating locks:
  * **verdict is enum-only**: ``HOURLY_CONCENTRATION`` literal — the
    endpoint string embeds detail; the builder splits ``verdict`` (enum)
    from ``headline`` (human). The reporter's Discord line keys off
    ``verdict``, so an enum-clean field is required.
  * **quota_exhausted precedence**: a quota reasoning lands in
    ``quota_exhausted`` even when the row would naively match other
    branches (the same fix the inline endpoint shipped).
  * **breakdown in HOURLY_CONCENTRATION headline**: the worst bucket's
    sub-bucket counts are interpolated so the operator immediately sees
    *why* the hour is starved (host / quota / empty / parse).
  * **window cutoff is strict** (``ts < cutoff`` excludes; equality
    keeps) per the standard signals.py precedent.
  * **garbage rows degrade-never-raise**: missing timestamp, missing
    keys, unparseable ISO, non-string action — every row failure drops
    the row, never the whole verdict.

Each test asserts exact values, not "no exception". The reporter line
integration is exercised in TestReporterLine — the Discord-side
contract: surface ONLY HOURLY_CONCENTRATION (the actionable verdict),
silent on INSUFFICIENT_DATA / EVEN_DISTRIBUTION (the
``_hold_discipline_line`` NO_DATA / ``_heartbeat_line`` HEALTHY
suppression precedent — the summary must never become its own lying
green light).
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from paper_trader.analytics.decision_clock import (
    HOURLY_CONCENTRATION_PCT,
    MIN_TOTAL_DECISIONS,
    MIN_WORST_BUCKET_SAMPLES,
    NY,
    _classify_no_decision,
    build_decision_clock,
)


def _ts(now, **kwargs):
    return (now + timedelta(**kwargs)).isoformat()


def _dec(ts, action="NO_DECISION", reason="claude returned no response"):
    return {"timestamp": ts, "action_taken": action, "reasoning": reason}


# ── _classify_no_decision: bucket precedence ─────────────────────────────


class TestClassifyNoDecision:
    """Branches mutually exclusive, ordered most-specific first.
    A change to ordering must fail this class loudly."""

    @pytest.mark.parametrize("reason,expected", [
        # quota wins even though future formats might add "no response":
        ("claude quota/usage limit exhausted (no decision)", "quota_exhausted"),
        # host-saturation: skipped or saturated tokens trip the bucket
        ("skipped claude call — host saturated: 6 concurrent Opus (>4)", "host_saturated"),
        ("host saturated mid-call", "host_saturated"),
        # empty-response after the more-specific branches
        ("claude returned no response (timeout/empty)", "empty_response"),
        # parse / retry
        ("parse_failed: not JSON", "parse_failed"),
        ("retry_failed: still garbage", "parse_failed"),
        # anything else
        ("", "other_no_decision"),
        ("some new failure mode", "other_no_decision"),
    ])
    def test_branch_classification(self, reason, expected):
        assert _classify_no_decision(reason) == expected

    def test_quota_precedence_wins_against_concatenated_tokens(self):
        """A hypothetical future row that says 'quota ... no response'
        must still land in quota_exhausted — the operator-actionable
        distinction (kill agents vs upgrade plan) must not silently
        flip if reason strings get concatenated."""
        assert _classify_no_decision(
            "claude quota exhausted; no response from CLI") == "quota_exhausted"


# ── insufficient_data verdict ────────────────────────────────────────────


class TestInsufficientData:
    def test_zero_decisions_returns_insufficient_data(self):
        now = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
        out = build_decision_clock([], now=now)
        assert out["verdict"] == "INSUFFICIENT_DATA"
        assert out["total_decisions"] == 0
        assert "withheld" in out["headline"].lower()

    def test_below_min_total_decisions_returns_insufficient_data(self):
        # MIN_TOTAL_DECISIONS = 5 — exactly 4 must read INSUFFICIENT
        now = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
        decs = [_dec(_ts(now, hours=-1, seconds=i))
                for i in range(MIN_TOTAL_DECISIONS - 1)]
        out = build_decision_clock(decs, now=now)
        assert out["verdict"] == "INSUFFICIENT_DATA"
        assert out["total_decisions"] == MIN_TOTAL_DECISIONS - 1

    def test_at_min_total_decisions_no_longer_insufficient(self):
        # Exactly MIN_TOTAL_DECISIONS — boundary inclusive (5 OK, 4 not)
        now = datetime(2026, 5, 18, 12, 0, tzinfo=timezone.utc)
        decs = [_dec(_ts(now, hours=-1, seconds=i), action="BUY → FILLED",
                     reason="{}")
                for i in range(MIN_TOTAL_DECISIONS)]
        out = build_decision_clock(decs, now=now)
        assert out["verdict"] != "INSUFFICIENT_DATA"


# ── hourly_concentration verdict ─────────────────────────────────────────


class TestHourlyConcentration:
    def test_exact_50pct_triggers_concentration(self):
        """Boundary inclusive: ``>= HOURLY_CONCENTRATION_PCT`` triggers."""
        # 3 NO_DECISION + 3 FILLED in same hour → 50% NO_DECISION,
        # 6 total ≥ MIN_WORST_BUCKET_SAMPLES, 6 ≥ MIN_TOTAL_DECISIONS
        now = datetime(2026, 5, 18, 16, 30, tzinfo=timezone.utc)
        target = now - timedelta(hours=2)  # bucket the same hour in NY
        decs = []
        for i in range(3):
            decs.append(_dec(_ts(target, seconds=i), action="NO_DECISION",
                             reason="claude returned no response"))
        for i in range(3):
            decs.append(_dec(_ts(target, seconds=10 + i),
                             action="BUY → FILLED", reason="{}"))
        out = build_decision_clock(decs, now=now)
        assert out["verdict"] == "HOURLY_CONCENTRATION"
        expected_hour = target.astimezone(NY).hour
        assert out["worst_hour_local"] == expected_hour

    def test_just_below_50pct_no_concentration(self):
        # 4 NO_DECISION + 5 FILLED → 44.4% NO_DECISION (< 50)
        now = datetime(2026, 5, 18, 16, 30, tzinfo=timezone.utc)
        target = now - timedelta(hours=2)
        decs = [_dec(_ts(target, seconds=i), action="NO_DECISION",
                     reason="claude returned no response") for i in range(4)]
        decs += [_dec(_ts(target, seconds=10 + i), action="BUY → FILLED",
                      reason="{}") for i in range(5)]
        out = build_decision_clock(decs, now=now)
        assert out["verdict"] == "EVEN_DISTRIBUTION"

    def test_headline_breakdown_names_dominant_subbucket(self):
        """Worst-bucket breakdown surfaces the sub-bucket counts so the
        operator sees the failure-mode mix at a glance."""
        now = datetime(2026, 5, 18, 16, 30, tzinfo=timezone.utc)
        target = now - timedelta(hours=2)
        # 3 host + 2 quota + 1 empty → 6 NO_DECISION total
        decs = []
        for i in range(3):
            decs.append(_dec(_ts(target, seconds=i),
                             reason="skipped claude call — host saturated"))
        for i in range(2):
            decs.append(_dec(_ts(target, seconds=10 + i),
                             reason="claude quota/usage limit exhausted"))
        for i in range(1):
            decs.append(_dec(_ts(target, seconds=20 + i),
                             reason="claude returned no response"))
        out = build_decision_clock(decs, now=now)
        assert out["verdict"] == "HOURLY_CONCENTRATION"
        h = out["headline"]
        # All three sub-bucket counts visible in the headline
        assert "3 host" in h, h
        assert "2 quota" in h, h
        assert "1 empty" in h, h

    def test_low_sample_bucket_excluded_from_worst(self):
        """A bucket with < MIN_WORST_BUCKET_SAMPLES samples is excluded
        from `worst` even if it's 100% NO_DECISION — too few samples to
        trust."""
        # 2 NO_DECISION in hour X (below floor) + 5 FILLED in hour Y
        # Hour X is 100% NO_DECISION but only 2 samples — must NOT be
        # the worst pick. Hour Y is 0% NO_DECISION.
        now = datetime(2026, 5, 18, 16, 30, tzinfo=timezone.utc)
        hour_x_anchor = now - timedelta(hours=5)
        hour_y_anchor = now - timedelta(hours=2)
        decs = [_dec(_ts(hour_x_anchor, seconds=i), action="NO_DECISION",
                     reason="claude returned no response") for i in range(2)]
        decs += [_dec(_ts(hour_y_anchor, seconds=i), action="BUY → FILLED",
                      reason="{}") for i in range(5)]
        out = build_decision_clock(decs, now=now)
        # Hour Y has 5 samples but 0% NO_DECISION; hour X has 100% but
        # only 2 samples (excluded). Worst = hour Y (passes floor).
        # No concentration (Y is 0%), so EVEN_DISTRIBUTION.
        assert out["verdict"] == "EVEN_DISTRIBUTION"
        expected_hour_y = hour_y_anchor.astimezone(NY).hour
        assert out["worst_hour_local"] == expected_hour_y


# ── window cutoff + degrade-never-raise ──────────────────────────────────


class TestWindowAndDegradeNeverRaise:
    def test_decision_older_than_window_excluded(self):
        now = datetime(2026, 5, 18, 16, 30, tzinfo=timezone.utc)
        old_ts = _ts(now, days=-8)  # window=7 days → excluded
        out = build_decision_clock([_dec(old_ts)], now=now, days=7)
        assert out["total_decisions"] == 0
        assert out["verdict"] == "INSUFFICIENT_DATA"

    def test_decision_at_exact_cutoff_included(self):
        """Strict ``<`` exclusion: ``ts == cutoff`` keeps the row."""
        now = datetime(2026, 5, 18, 16, 30, tzinfo=timezone.utc)
        cutoff_ts = (now - timedelta(days=7)).isoformat()
        out = build_decision_clock([_dec(cutoff_ts)], now=now, days=7)
        # Single row included — total counts it
        assert out["total_decisions"] == 1

    def test_days_clamped(self):
        now = datetime(2026, 5, 18, tzinfo=timezone.utc)
        for raw, expected in [(0, 1), (-5, 1), (9999, 30), (15, 15)]:
            out = build_decision_clock([], now=now, days=raw)
            assert out["days"] == expected

    def test_garbage_rows_degrade_never_raise(self):
        now = datetime(2026, 5, 18, 16, 30, tzinfo=timezone.utc)
        good_ts = _ts(now, hours=-1)
        decs = [
            None,                                       # non-dict
            {},                                         # missing timestamp
            {"timestamp": None},                        # null timestamp
            {"timestamp": "not-iso", "action_taken": "x"},  # unparseable
            _dec(good_ts),                              # one good row
        ]
        out = build_decision_clock(decs, now=now)  # MUST NOT raise
        assert out["total_decisions"] == 1
        # All other rows silently dropped, verdict computed on the one.
        assert out["verdict"] == "INSUFFICIENT_DATA"  # 1 < 5

    def test_unknown_action_does_not_count_in_filled_or_no_decision(self):
        now = datetime(2026, 5, 18, 16, 30, tzinfo=timezone.utc)
        # HOLD / BLOCKED rows count toward total but NOT toward filled
        # or no_decision (the dashboard's design — the sub-counts only
        # cover the two extremes).
        decs = [
            {"timestamp": _ts(now, hours=-2), "action_taken": "HOLD MU → HOLD",
             "reasoning": "{}"},
            {"timestamp": _ts(now, hours=-2, seconds=1),
             "action_taken": "SELL X → BLOCKED", "reasoning": "{}"},
        ]
        out = build_decision_clock(decs, now=now)
        total_f = sum(b["filled"] for b in out["buckets"])
        total_nd = sum(b["no_decision"] for b in out["buckets"])
        assert total_f == 0
        assert total_nd == 0
        assert out["total_decisions"] == 2


# ── parity with the inline endpoint ──────────────────────────────────────


class TestEndpointParity:
    """The pure builder and the inline /api/decision-clock endpoint
    must compute the SAME bucket counts on the same input — a regression
    in either layer (or a divergent quota-precedence change) fails here
    loudly. The endpoint's ``verdict`` field embeds detail in
    HOURLY_CONCENTRATION so we compare the structural bucket counts +
    `worst_hour_local`, not the verdict string itself."""

    def _seed_via_store(self, fresh_store, decs):
        for ts, action, reason in decs:
            with fresh_store._lock:
                fresh_store.conn.execute(
                    "INSERT INTO decisions (timestamp, market_open, "
                    "signal_count, action_taken, reasoning, "
                    "portfolio_value, cash) VALUES (?,?,?,?,?,?,?)",
                    (ts, 1, 0, action, reason, 1000.0, 1000.0),
                )
                fresh_store.conn.commit()

    def test_bucket_counts_match_endpoint(
            self, fresh_store, monkeypatch):
        from paper_trader import dashboard, store as store_mod
        monkeypatch.setattr(dashboard, "get_store", lambda: fresh_store)

        now = datetime.now(timezone.utc)
        decs_input = [
            (_ts(now, hours=-2), "NO_DECISION",
             "claude quota/usage limit exhausted (no decision)"),
            (_ts(now, hours=-2, seconds=1), "NO_DECISION",
             "skipped claude call — host saturated"),
            (_ts(now, hours=-3), "BUY NVDA → FILLED", "{}"),
        ]
        self._seed_via_store(fresh_store, decs_input)

        # Endpoint path
        client = dashboard.app.test_client()
        ep = client.get("/api/decision-clock?days=7").get_json()

        # Builder path: read decisions, compute over the same data
        store_decs = fresh_store.recent_decisions(limit=20000)
        bd = build_decision_clock(store_decs, now=now, days=7)

        # Compare bucket-by-bucket on the shared keys
        shared = {"hour", "total", "filled", "no_decision",
                  "host_saturated", "empty_response", "parse_failed",
                  "quota_exhausted", "other_no_decision"}
        for h in range(24):
            for k in shared:
                assert ep["buckets"][h][k] == bd["buckets"][h][k], (
                    f"hour {h} key {k} drifted: endpoint="
                    f"{ep['buckets'][h][k]} builder={bd['buckets'][h][k]}")
        assert ep["worst_hour_local"] == bd["worst_hour_local"]
        assert ep["total_decisions"] == bd["total_decisions"]


@pytest.fixture
def fresh_store(tmp_path, monkeypatch):
    from paper_trader.store import Store
    from paper_trader import store as store_mod
    db = tmp_path / "paper_trader.db"
    monkeypatch.setattr(store_mod, "DB_PATH", db)
    monkeypatch.setattr(store_mod, "_singleton", None)
    s = Store()
    try:
        yield s
    finally:
        s.close()


# ── reporter Discord line integration ────────────────────────────────────


class TestReporterDecisionClockLine:
    """`_decision_clock_line` from reporter.py surfaces the
    HOURLY_CONCENTRATION verdict — and ONLY that — to Discord.
    INSUFFICIENT_DATA / EVEN_DISTRIBUTION / any fault → empty string
    (the ``_hold_discipline_line`` NO_DATA / ``_heartbeat_line``
    HEALTHY suppression precedent). The Discord summary must never
    become its own lying green light."""

    def _seed(self, store, ts, action, reason):
        with store._lock:
            store.conn.execute(
                "INSERT INTO decisions (timestamp, market_open, "
                "signal_count, action_taken, reasoning, portfolio_value, "
                "cash) VALUES (?,?,?,?,?,?,?)",
                (ts, 1, 0, action, reason, 1000.0, 1000.0),
            )
            store.conn.commit()

    def test_empty_store_silent(self, fresh_store):
        from paper_trader import reporter
        out = reporter._decision_clock_line(fresh_store)
        assert out == ""

    def test_even_distribution_silent(self, fresh_store):
        from paper_trader import reporter
        now = datetime.now(timezone.utc)
        for i in range(6):
            self._seed(fresh_store, _ts(now, hours=-2, seconds=i),
                       "BUY NVDA → FILLED", "{}")
        out = reporter._decision_clock_line(fresh_store)
        assert out == ""

    def test_hourly_concentration_surfaces(self, fresh_store):
        from paper_trader import reporter
        now = datetime.now(timezone.utc)
        for i in range(5):
            self._seed(fresh_store, _ts(now, hours=-2, seconds=i),
                       "NO_DECISION", "claude returned no response")
        self._seed(fresh_store, _ts(now, hours=-2, seconds=10),
                   "BUY NVDA → FILLED", "{}")
        out = reporter._decision_clock_line(fresh_store)
        assert "HOURLY_CONCENTRATION" in out, out
        # Verdict embedded, headline embedded
        assert "**DECISION CLOCK**" in out, out
        assert "ET" in out  # the hour-of-day label
        assert "%" in out   # the no-decision percentage

    def test_builder_fault_degrades_to_empty(self, fresh_store, monkeypatch):
        """A builder/store fault must drop the line, NEVER raise — the
        rest-of-reporter additive contract."""
        from paper_trader import reporter
        from paper_trader.analytics import decision_clock as dc

        def _boom(*a, **k):
            raise RuntimeError("simulated builder fault")

        monkeypatch.setattr(dc, "build_decision_clock", _boom)
        # Also patch the symbol the reporter looks up at call time, in
        # case it imports build_decision_clock at module-load.
        monkeypatch.setattr(
            "paper_trader.analytics.decision_clock.build_decision_clock",
            _boom)
        out = reporter._decision_clock_line(fresh_store)
        assert out == ""

    def test_hourly_summary_includes_line_when_concentrated(
            self, fresh_store, monkeypatch):
        """End-to-end: when HOURLY_CONCENTRATION is the verdict,
        send_hourly_summary's body includes the DECISION CLOCK line —
        and the summary still ships even if every other reporter block
        silently degrades. We monkeypatch _send to capture the body."""
        from paper_trader import reporter
        now = datetime.now(timezone.utc)
        for i in range(5):
            self._seed(fresh_store, _ts(now, hours=-2, seconds=i),
                       "NO_DECISION", "claude returned no response")
        self._seed(fresh_store, _ts(now, hours=-2, seconds=10),
                   "BUY NVDA → FILLED", "{}")
        # Route get_store everywhere reporter uses it.
        monkeypatch.setattr(reporter, "get_store", lambda: fresh_store)
        captured = {}

        def _fake_send(body):
            captured["body"] = body
            return True

        monkeypatch.setattr(reporter, "_send", _fake_send)
        ok = reporter.send_hourly_summary()
        assert ok is True
        assert "DECISION CLOCK" in captured["body"], captured["body"][:500]
