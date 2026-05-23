"""Unit tests for paper_trader.analytics.stack_liveness.

Locks the per-component status mapping and the worst-component-wins
top-level verdict reduction. The builder is pure (keyword-only inputs);
endpoint-side network/IO is tested separately by integration.
"""
from __future__ import annotations

from paper_trader.analytics.stack_liveness import (
    ARTICLES_DB_DARK_MIN,
    ARTICLES_DB_DEGRADED_MIN,
    SCORER_PKL_MIN_N_TRAIN,
    TRADER_SHA_DEGRADED_BEHIND,
    build_stack_liveness,
)


def _green_kwargs(**overrides):
    """All-HEALTHY input set; tests override individual components."""
    base = dict(
        build_info={"boot_sha": "abc1234", "head_sha": "abc1234",
                    "behind": 0, "stale": False},
        runner_heartbeat={"verdict": "HEALTHY", "headline": "loop OK"},
        scorer_pkl_info={"exists": True, "n_train": 5000,
                         "pred_collapsed": False, "error": None},
        intern_reachable=True,
        intern_error=None,
        articles_db_age_minutes=2.0,
    )
    base.update(overrides)
    return base


def test_all_green_reports_healthy():
    out = build_stack_liveness(**_green_kwargs())
    assert out["verdict"] == "HEALTHY"
    assert out["worst_component"] is None
    assert all(c["status"] == "HEALTHY" for c in out["components"].values())


# ─── per-component status mapping ────────────────────────────────────────


def test_trader_sha_degraded_when_behind_head():
    out = build_stack_liveness(**_green_kwargs(
        build_info={"boot_sha": "old1234", "head_sha": "new5678",
                    "behind": 5, "stale": True}))
    assert out["components"]["trader_sha"]["status"] == "DEGRADED"
    assert out["verdict"] == "DEGRADED"


def test_trader_sha_unknown_when_git_unreachable():
    out = build_stack_liveness(**_green_kwargs(
        build_info={"boot_sha": "abc", "head_sha": None,
                    "behind": None, "stale": False}))
    assert out["components"]["trader_sha"]["status"] == "UNKNOWN"


def test_trader_loop_dark_on_stalled():
    out = build_stack_liveness(**_green_kwargs(
        runner_heartbeat={"verdict": "STALLED", "headline": "loop dead"}))
    assert out["components"]["trader_loop"]["status"] == "DARK"
    assert out["verdict"] == "DARK"


def test_trader_loop_degraded_on_lagging():
    out = build_stack_liveness(**_green_kwargs(
        runner_heartbeat={"verdict": "LAGGING", "headline": "loop slow"}))
    assert out["components"]["trader_loop"]["status"] == "DEGRADED"


def test_trader_loop_degraded_on_idle_storm():
    out = build_stack_liveness(**_green_kwargs(
        runner_heartbeat={"verdict": "IDLE_STORM", "headline": "no-decision storm"}))
    assert out["components"]["trader_loop"]["status"] == "DEGRADED"


def test_scorer_pkl_dark_when_collapsed_predictions():
    out = build_stack_liveness(**_green_kwargs(
        scorer_pkl_info={"exists": True, "n_train": 39,
                         "pred_collapsed": True, "error": None}))
    assert out["components"]["scorer_pkl"]["status"] == "DARK"
    assert out["verdict"] == "DARK"


def test_scorer_pkl_degraded_when_n_train_below_floor():
    out = build_stack_liveness(**_green_kwargs(
        scorer_pkl_info={"exists": True, "n_train": SCORER_PKL_MIN_N_TRAIN - 1,
                         "pred_collapsed": False, "error": None}))
    assert out["components"]["scorer_pkl"]["status"] == "DEGRADED"


def test_scorer_pkl_degraded_when_missing():
    out = build_stack_liveness(**_green_kwargs(
        scorer_pkl_info={"exists": False, "n_train": None,
                         "pred_collapsed": None, "error": None}))
    assert out["components"]["scorer_pkl"]["status"] == "DEGRADED"


def test_intern_dark_when_unreachable():
    out = build_stack_liveness(**_green_kwargs(
        intern_reachable=False, intern_error="connection refused"))
    assert out["components"]["intern"]["status"] == "DARK"
    assert out["verdict"] == "DARK"


def test_intern_unknown_when_probe_skipped():
    out = build_stack_liveness(**_green_kwargs(intern_reachable=None))
    assert out["components"]["intern"]["status"] == "UNKNOWN"


def test_articles_db_dark_when_very_stale():
    out = build_stack_liveness(**_green_kwargs(
        articles_db_age_minutes=ARTICLES_DB_DARK_MIN + 5.0))
    assert out["components"]["articles_db"]["status"] == "DARK"


def test_articles_db_degraded_when_moderately_stale():
    out = build_stack_liveness(**_green_kwargs(
        articles_db_age_minutes=ARTICLES_DB_DEGRADED_MIN + 5.0))
    assert out["components"]["articles_db"]["status"] == "DEGRADED"


def test_articles_db_unknown_when_age_none():
    out = build_stack_liveness(**_green_kwargs(articles_db_age_minutes=None))
    assert out["components"]["articles_db"]["status"] == "UNKNOWN"


# ─── top-level verdict reduction ─────────────────────────────────────────


def test_dark_dominates_degraded():
    # trader_sha DEGRADED, intern DARK ⇒ overall DARK.
    out = build_stack_liveness(**_green_kwargs(
        build_info={"boot_sha": "old", "head_sha": "new",
                    "behind": 5, "stale": True},
        intern_reachable=False, intern_error="oops"))
    assert out["verdict"] == "DARK"
    assert out["worst_component"] == "intern"


def test_degraded_dominates_unknown():
    # build_info UNKNOWN + articles_db DEGRADED ⇒ overall DEGRADED.
    out = build_stack_liveness(**_green_kwargs(
        build_info={"boot_sha": "abc", "head_sha": None,
                    "behind": None, "stale": False},
        articles_db_age_minutes=ARTICLES_DB_DEGRADED_MIN + 1.0))
    assert out["verdict"] == "DEGRADED"
    assert out["worst_component"] == "articles_db"


def test_unknown_dominates_healthy():
    out = build_stack_liveness(**_green_kwargs(
        build_info={"boot_sha": "abc", "head_sha": None,
                    "behind": None, "stale": False}))
    assert out["verdict"] == "UNKNOWN"


def test_priority_order_picks_trader_loop_over_intern_when_both_dark():
    out = build_stack_liveness(**_green_kwargs(
        runner_heartbeat={"verdict": "STALLED", "headline": "dead loop"},
        intern_reachable=False, intern_error="dead intern"))
    assert out["verdict"] == "DARK"
    # trader_loop sits earlier in _PRIORITY so a dead loop wins.
    assert out["worst_component"] == "trader_loop"


def test_healthy_headline_is_canned():
    out = build_stack_liveness(**_green_kwargs())
    assert out["verdict"] == "HEALTHY"
    assert "HEALTHY" in out["headline"]


def test_degraded_headline_prefixes_with_worst_component():
    out = build_stack_liveness(**_green_kwargs(
        articles_db_age_minutes=ARTICLES_DB_DEGRADED_MIN + 1.0))
    h = out["headline"]
    assert h.startswith("[articles_db]")
    assert "DEGRADED" in h


# ─── robustness ──────────────────────────────────────────────────────────


def test_never_raises_on_all_none_inputs():
    out = build_stack_liveness(
        build_info=None, runner_heartbeat=None, scorer_pkl_info=None,
        intern_reachable=None, articles_db_age_minutes=None)
    # All UNKNOWN → top-level UNKNOWN, no exception.
    assert out["verdict"] == "UNKNOWN"


def test_scorer_pkl_error_string_degraded_not_raise():
    out = build_stack_liveness(**_green_kwargs(
        scorer_pkl_info={"exists": True, "n_train": None,
                         "pred_collapsed": None,
                         "error": "pickle load failed: corrupted"}))
    assert out["components"]["scorer_pkl"]["status"] == "DEGRADED"


def test_module_constants_are_module_owned():
    """Tests read constants from the module so a retune cannot false-fail."""
    assert TRADER_SHA_DEGRADED_BEHIND >= 1
    assert SCORER_PKL_MIN_N_TRAIN > 0
    assert ARTICLES_DB_DARK_MIN > ARTICLES_DB_DEGRADED_MIN > 0


# ─── Flask route ─────────────────────────────────────────────────────────


class TestStackLivenessRoute:
    def test_route_returns_well_formed_envelope(self):
        from paper_trader.dashboard import app
        client = app.test_client()
        resp = client.get("/api/stack-liveness")
        assert resp.status_code in (200, 500), resp.status_code
        body = resp.get_json()
        assert isinstance(body, dict)
        # Must always carry verdict + headline + components, regardless of
        # whether the live system is up or stubbed in CI.
        assert "verdict" in body
        assert "headline" in body
        if resp.status_code == 200:
            assert "components" in body
            for name in ("trader_sha", "trader_loop", "scorer_pkl",
                         "intern", "articles_db"):
                assert name in body["components"]
