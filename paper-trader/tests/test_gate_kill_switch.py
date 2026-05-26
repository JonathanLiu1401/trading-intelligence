"""Tests for the per-cycle no-skill kill-switch.

`paper_trader.backtest._should_gate_modulate_conviction` reads the trailing
N rows of `data/scorer_skill_log.jsonl` and returns ``(gate_active, reason)``.
When the trailing OOS BUY rank-IC's median absolute value is below
``_GATE_SKILL_IC_TOLERANCE`` the kill-switch reports ``gate_active=False``
and `_ml_decide` short-circuits the ×0.6/×0.85/×1.15/×1.3 modulation block.

The test contract pins the SAFE DEFAULTS (always gate-active on fault /
missing ledger / insufficient trailing cycles) so a transient ledger read
failure never silently disables the conviction gate — same discipline as
the live-trader `_ml_is_qualified` cache (strategy.py CLAUDE.md §15).
"""
from __future__ import annotations

import json

import pytest

from paper_trader import backtest as bt


@pytest.fixture(autouse=True)
def _reset_kill_switch_cache_per_test():
    """Each test starts with a cold kill-switch cache so the path under test
    actually exercises the JSONL read instead of returning a stale value
    from a sibling test (the production TTL is 1h; tests would otherwise
    leak state via the module-global ``_gate_skill_cache``)."""
    bt._reset_gate_skill_cache()
    yield
    bt._reset_gate_skill_cache()


def _write_skill_log(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


class TestKillSwitchDefaults:
    """Safe-default invariants: a fault or missing data must NEVER silently
    disable the gate. The kill-switch defaults to gate-active in every
    degenerate case so invariant #5 semantics hold during fresh starts."""

    def test_missing_ledger_defaults_to_gate_active(self, tmp_path, monkeypatch):
        """No skill log on disk ⇒ gate stays active (safe default)."""
        monkeypatch.setattr(bt, "_GATE_SKILL_LOG_PATH",
                            tmp_path / "no_such_file.jsonl")
        active, reason = bt._should_gate_modulate_conviction()
        assert active is True
        assert "missing" in reason.lower()

    def test_empty_ledger_defaults_to_gate_active(self, tmp_path, monkeypatch):
        """An empty file produces 0 parseable rows ⇒ insufficient cycles ⇒
        gate stays active (safe default). Catches a literal-empty bug
        where the deque-tail path could otherwise crash on empty iter."""
        p = tmp_path / "scorer_skill_log.jsonl"
        p.write_text("")
        monkeypatch.setattr(bt, "_GATE_SKILL_LOG_PATH", p)
        active, reason = bt._should_gate_modulate_conviction()
        assert active is True
        assert "0" in reason
        assert "valid" in reason.lower() or "default" in reason.lower()

    def test_fewer_than_min_cycles_defaults_to_gate_active(self, tmp_path,
                                                          monkeypatch):
        """N-1 valid rows ⇒ insufficient cycles ⇒ gate stays active.
        Pins the off-by-one boundary: MIN_CYCLES rows would activate the
        evaluation, MIN_CYCLES-1 must keep the safe default."""
        rows = [{"oos_buy_ic": 0.0} for _ in range(bt._GATE_SKILL_MIN_CYCLES - 1)]
        p = tmp_path / "scorer_skill_log.jsonl"
        _write_skill_log(p, rows)
        monkeypatch.setattr(bt, "_GATE_SKILL_LOG_PATH", p)
        active, reason = bt._should_gate_modulate_conviction()
        assert active is True
        assert str(bt._GATE_SKILL_MIN_CYCLES - 1) in reason

    def test_corrupt_json_rows_count_as_invalid(self, tmp_path, monkeypatch):
        """A row that fails json.loads must NOT crash and must NOT be
        counted toward the trailing-cycles total — same defensive shape as
        the production ``_parse_scorer_status`` and ``_inject_and_train``
        per-line parses. With every row corrupt, the kill-switch must
        report insufficient cycles, not raise."""
        p = tmp_path / "scorer_skill_log.jsonl"
        p.write_text("not json\n{also bad\n}\n")
        monkeypatch.setattr(bt, "_GATE_SKILL_LOG_PATH", p)
        active, reason = bt._should_gate_modulate_conviction()
        assert active is True
        assert "0" in reason

    def test_null_oos_buy_ic_rows_count_as_invalid(self, tmp_path, monkeypatch):
        """A row with ``oos_buy_ic=null`` (the documented n/a sentinel when
        the OOS slice is too small) must be skipped — `float(None)` would
        otherwise raise TypeError out of the kill-switch."""
        rows = [{"oos_buy_ic": None} for _ in range(bt._GATE_SKILL_MIN_CYCLES)]
        p = tmp_path / "scorer_skill_log.jsonl"
        _write_skill_log(p, rows)
        monkeypatch.setattr(bt, "_GATE_SKILL_LOG_PATH", p)
        active, reason = bt._should_gate_modulate_conviction()
        # All null → 0 valid rows → safe default
        assert active is True


class TestKillSwitchVerdict:
    """The economic contract: |median oos_buy_ic| < tolerance ⇒ gate killed;
    otherwise gate stays active. Pins the tolerance boundary so a future
    change to the constant fails this test loudly."""

    def test_high_median_ic_keeps_gate_active(self, tmp_path, monkeypatch):
        """A clearly skilled scorer (median IC well above tolerance) keeps
        the gate active. Uses +0.20 — far above the +0.03 production
        tolerance so the test is robust to small tuning changes."""
        rows = [{"oos_buy_ic": 0.20} for _ in range(bt._GATE_SKILL_MIN_CYCLES)]
        p = tmp_path / "scorer_skill_log.jsonl"
        _write_skill_log(p, rows)
        monkeypatch.setattr(bt, "_GATE_SKILL_LOG_PATH", p)
        active, reason = bt._should_gate_modulate_conviction()
        assert active is True
        assert "gate active" in reason.lower()
        assert "+0.200" in reason

    def test_zero_median_ic_kills_gate(self, tmp_path, monkeypatch):
        """A median of exactly 0.0 is below any positive tolerance — the
        canonical no-skill case. The production `MLP_NO_BETTER_THAN_TRIVIAL`
        verdict corresponds to this region."""
        rows = [{"oos_buy_ic": 0.0} for _ in range(bt._GATE_SKILL_MIN_CYCLES)]
        p = tmp_path / "scorer_skill_log.jsonl"
        _write_skill_log(p, rows)
        monkeypatch.setattr(bt, "_GATE_SKILL_LOG_PATH", p)
        active, reason = bt._should_gate_modulate_conviction()
        assert active is False
        assert "killed" in reason.lower()
        assert "+0.000" in reason

    def test_negative_median_ic_below_tolerance_kills_gate(self, tmp_path,
                                                          monkeypatch):
        """A consistently NEGATIVE small IC is worse than noise — also
        kills the gate. ``abs(median_ic) < tolerance`` is the contract,
        so a -0.01 median (anti-skill but below noise threshold) must kill."""
        rows = [{"oos_buy_ic": -0.01} for _ in range(bt._GATE_SKILL_MIN_CYCLES)]
        p = tmp_path / "scorer_skill_log.jsonl"
        _write_skill_log(p, rows)
        monkeypatch.setattr(bt, "_GATE_SKILL_LOG_PATH", p)
        active, reason = bt._should_gate_modulate_conviction()
        assert active is False

    def test_outlier_does_not_flip_verdict(self, tmp_path, monkeypatch):
        """The median is robust to a single outlier. 19 rows at 0.0 + 1 row
        at +0.50 yields median 0.0 ⇒ gate killed. A naive mean would
        instead read 0.025 and (depending on tolerance) flip the verdict
        — the median's robustness is the load-bearing reason this is
        ``statistics.median``, not a mean."""
        rows = [{"oos_buy_ic": 0.0} for _ in range(bt._GATE_SKILL_MIN_CYCLES - 1)]
        rows.append({"oos_buy_ic": 0.50})
        p = tmp_path / "scorer_skill_log.jsonl"
        _write_skill_log(p, rows)
        monkeypatch.setattr(bt, "_GATE_SKILL_LOG_PATH", p)
        active, _ = bt._should_gate_modulate_conviction()
        # Median of 20 (19 zeros + 1 outlier 0.5) = 0.0 → gate killed
        assert active is False

    def test_uses_recent_tail_not_old_rows(self, tmp_path, monkeypatch):
        """Old rows with high IC must NOT override a fresh tail of zeros —
        a scorer's skill DECAYS, and reviving a stale-good verdict from
        cycle 1 when the latest 20 cycles all show no skill would defeat
        the kill-switch's purpose."""
        # 50 old rows at high IC, MIN_CYCLES recent rows at 0.0
        rows = ([{"oos_buy_ic": 0.50} for _ in range(50)]
                + [{"oos_buy_ic": 0.0}
                   for _ in range(bt._GATE_SKILL_MIN_CYCLES)])
        p = tmp_path / "scorer_skill_log.jsonl"
        _write_skill_log(p, rows)
        monkeypatch.setattr(bt, "_GATE_SKILL_LOG_PATH", p)
        active, _ = bt._should_gate_modulate_conviction()
        # Tail-only median is 0.0 → gate killed despite the prior high-IC tail
        assert active is False


class TestKillSwitchCaching:
    """The kill-switch caches its result for 1h to amortize the JSONL read.
    Verify the cache is read once per TTL window and reset_gate_skill_cache
    forces a re-evaluation (the test seam)."""

    def test_repeat_call_returns_cached_value(self, tmp_path, monkeypatch):
        rows = [{"oos_buy_ic": 0.20} for _ in range(bt._GATE_SKILL_MIN_CYCLES)]
        p = tmp_path / "scorer_skill_log.jsonl"
        _write_skill_log(p, rows)
        monkeypatch.setattr(bt, "_GATE_SKILL_LOG_PATH", p)
        a1, _ = bt._should_gate_modulate_conviction()
        # Mutate the file to a kill verdict — without cache reset, the
        # cached gate-active reading must persist.
        kill_rows = [{"oos_buy_ic": 0.0}
                     for _ in range(bt._GATE_SKILL_MIN_CYCLES)]
        _write_skill_log(p, kill_rows)
        a2, _ = bt._should_gate_modulate_conviction()
        assert a1 == a2 is True

    def test_reset_cache_picks_up_fresh_verdict(self, tmp_path, monkeypatch):
        rows = [{"oos_buy_ic": 0.20} for _ in range(bt._GATE_SKILL_MIN_CYCLES)]
        p = tmp_path / "scorer_skill_log.jsonl"
        _write_skill_log(p, rows)
        monkeypatch.setattr(bt, "_GATE_SKILL_LOG_PATH", p)
        a1, _ = bt._should_gate_modulate_conviction()
        assert a1 is True
        # Flip the file to kill verdict + force cache reset → new reading
        kill_rows = [{"oos_buy_ic": 0.0}
                     for _ in range(bt._GATE_SKILL_MIN_CYCLES)]
        _write_skill_log(p, kill_rows)
        bt._reset_gate_skill_cache()
        a2, _ = bt._should_gate_modulate_conviction()
        assert a2 is False


class TestParseGateDecisionRecognizesKillMarker:
    """`run_continuous_backtests._parse_gate_decision` must treat the
    kill-switch `(gate-killed,no-skill)` marker as an abstention so
    downstream analyzers (`gate_pnl`, `gate_arm_historical`) drop those
    rows. Without this the gate's economic effect would be measured
    against decisions where the gate didn't actually act — fake skill."""

    def test_kill_marker_yields_off_dist_true(self):
        from run_continuous_backtests import _parse_gate_decision
        reasoning = (
            "ML+quant: NVDA score=2.50 regime=bull RSI=55 news_count=3 "
            "news_urg=8.0 conviction=25% scorer=+5.2%(gate-killed,no-skill)"
        )
        pred, off_dist = _parse_gate_decision(reasoning)
        assert pred == 5.2
        # Both abstention types must surface via gate_off_dist=True so
        # existing analyzers correctly drop them.
        assert off_dist is True

    def test_off_dist_marker_still_works(self):
        """Regression guard: extending the kill-marker recognition must
        NOT break the original (off-dist abstention) parse contract."""
        from run_continuous_backtests import _parse_gate_decision
        reasoning = (
            "ML+quant: NVDA score=2.50 regime=bull RSI=55 news_count=3 "
            "news_urg=8.0 conviction=25% scorer=+5.2%(off-dist,gate-skipped)"
        )
        pred, off_dist = _parse_gate_decision(reasoning)
        assert pred == 5.2
        assert off_dist is True

    def test_real_acted_gate_still_reads_false(self):
        """A real (acted) gate decision must still parse as
        ``off_dist=False`` — the kill marker must not pollute the
        non-abstention case."""
        from run_continuous_backtests import _parse_gate_decision
        reasoning = (
            "ML+quant: NVDA score=2.50 regime=bull RSI=55 news_count=3 "
            "news_urg=8.0 conviction=25% scorer=+5.2%"
        )
        pred, off_dist = _parse_gate_decision(reasoning)
        assert pred == 5.2
        assert off_dist is False
