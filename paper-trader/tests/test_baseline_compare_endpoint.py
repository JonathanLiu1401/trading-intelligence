"""End-to-end Flask-client tests for /api/baseline-compare — the honest
"does the 17-feature DecisionScorer earn its complexity OUT OF SAMPLE, or
would a one-line rule do as well?" trust panel.

`paper_trader/ml/baseline_compare.py` (the read-only OOS-skill diagnostic,
20 pure-unit tests in test_baseline_compare.py) had no dashboard surface and
no chat surface — its verdict (`MLP_NO_BETTER_THAN_TRIVIAL` on the live
outcomes, per data/run_log.md) was buried in a CLI the operator never runs.
This locks the endpoint contract: the route exists, is a faithful thin
wrapper over `scorer_baseline_compare(..., oos_only=True)` (the trustworthy
generalization-relevant slice — NOT the in-sample view that flatters the
net), never raises into a panel, and carries the keys the dashboard card +
the digital-intern chat block read.

Convention mirrors tests/test_decision_context_endpoint.py — real Flask app,
real module math, deterministic offline data; no :8090 bind, no live DB.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import paper_trader.dashboard as d
import paper_trader.ml.decision_scorer as ds_mod
from paper_trader.ml.baseline_compare import scorer_baseline_compare


class _FakeScorer:
    """Trained scorer whose prediction echoes 20-day momentum.

    Faithful to the live `DecisionScorer.predict(**11 kwargs)` signature so
    `scorer_baseline_compare` exercises the exact code path. Echoing mom20
    means a one-line baseline (`mom20`) carries the same rank skill as the
    "net" → a deterministic, real (non-INSUFFICIENT) verdict the test can
    pin without coupling to live data."""

    is_trained = True

    @property
    def n_train(self) -> int:
        return 1234

    def predict(self, *, ml_score=0.0, rsi=None, macd=None, mom5=None,
                mom20=None, regime_mult=1.0, ticker="", vol_ratio=None,
                bb_pos=None, news_urgency=None, news_article_count=None, **_extra_kwargs):
        try:
            return float(mom20 or 0.0)
        except (TypeError, ValueError):
            return 0.0


def _synthetic_outcomes(n: int = 260) -> list[dict]:
    """`data/decision_outcomes.jsonl` row shape. A monotone-ish mom20→return
    relation so the slice is non-degenerate; n is large enough that the
    temporal-OOS tail (oos_fraction=0.2) clears MIN_PAIRS=30 and the verdict
    is a *real computed* one — not a vacuous INSUFFICIENT_DATA."""
    rows = []
    for i in range(n):
        m = (i % 17) - 8                       # spread of momentum values
        rows.append({
            "run_id": 9000 + i,
            "sim_date": f"2025-{1 + (i % 12):02d}-{1 + (i % 27):02d}",
            "ticker": ["NVDA", "MU", "SPY", "AMD"][i % 4],
            "action": "SELL" if i % 5 == 0 else "BUY",
            "ml_score": (i % 7) - 3,
            "rsi": 30.0 + (i % 40),
            "macd": (i % 9) - 4,
            "mom5": (i % 11) - 5,
            "mom20": m,
            "regime_mult": 1.0,
            "vol_ratio": 1.0 + (i % 3) * 0.1,
            "bb_position": ((i % 21) - 10) / 10.0,
            "news_urgency": None,
            "news_article_count": None,
            # realized 5d return tracks momentum (+ deterministic wobble)
            "forward_return_5d": round(m * 0.4 + ((i % 5) - 2) * 0.3, 4),
        })
    return rows


@pytest.fixture
def client(monkeypatch):
    outcomes = _synthetic_outcomes()
    monkeypatch.setattr(d, "_load_decision_outcomes", lambda *a, **k: outcomes)
    monkeypatch.setattr(ds_mod, "DecisionScorer", _FakeScorer)
    d.app.config["TESTING"] = True
    with d.app.test_client() as c:
        yield c, outcomes


_VERDICTS = {
    "INSUFFICIENT_DATA", "MLP_WORSE_THAN_TRIVIAL",
    "MLP_NO_BETTER_THAN_TRIVIAL", "MLP_ADDS_SKILL",
}


def test_route_exists_and_returns_verdict_shape(client):
    c, _ = client
    r = c.get("/api/baseline-compare")
    assert r.status_code == 200
    j = r.get_json()
    assert "error" not in j, j
    # keys the dashboard card + chat block read — a missing key is a panel
    # KeyError, so this is the operator-facing contract.
    for k in ("verdict", "mlp", "baselines", "best_baseline",
              "best_baseline_ic", "ic_gap", "hint", "slice",
              "n_records_considered", "n_train"):
        assert k in j, f"missing {k!r} in {sorted(j)}"
    assert j["verdict"] in _VERDICTS
    assert {b["name"] for b in j["baselines"]} >= {"ml_score", "mom20", "mom5"}


def test_endpoint_is_faithful_thin_wrapper_over_oos_slice(client):
    """The endpoint must call scorer_baseline_compare with oos_only=True (the
    trustworthy slice) over the loaded outcomes, and add n_train — nothing
    else. Equality with the module function on identical inputs catches a
    wrong arg, the in-sample slice, or the path-based analyze() being used."""
    c, outcomes = client
    expect = scorer_baseline_compare(_FakeScorer(), outcomes, oos_only=True)
    expect.setdefault("n_train", _FakeScorer().n_train)
    got = c.get("/api/baseline-compare").get_json()
    got.pop("cached", None)
    got.pop("cache_age_s", None)
    assert got == expect
    # the OOS slice (not the flattering in-sample one) was the one scored
    assert got["slice"] == "oos"
    assert got["n_train"] == 1234


def test_mom20_echo_net_reads_as_no_better_than_trivial(client):
    """Behavioural assertion (not just shape): when the "net" is literally
    mom20, the mom20 one-liner must match it → the diagnostic must NOT award
    MLP_ADDS_SKILL. This is the bug the panel exists to make visible — a
    complex model that carries no edge a single feature doesn't."""
    c, _ = client
    j = c.get("/api/baseline-compare").get_json()
    assert j["verdict"] != "MLP_ADDS_SKILL", j["hint"]


def test_cors_header_present_for_cross_fetch(client):
    """The digital-intern chat + the unified dashboard cross-read this; the
    global _cors after_request must stamp it like every sibling endpoint."""
    c, _ = client
    r = c.get("/api/baseline-compare")
    assert r.headers.get("Access-Control-Allow-Origin") == "*"


def test_dashboard_card_is_wired_into_the_page():
    """Regression-lock the third surface: the `/` page must carry the
    bc-card, and its refresh fn must be defined AND registered (init call +
    poll interval) — an endpoint nobody renders is invisible to the
    operator, exactly the gap this feature closes."""
    d.app.config["TESTING"] = True
    with d.app.test_client() as c:
        h = c.get("/").get_data(as_text=True)
    assert 'id="bc-card"' in h and 'id="bc-state"' in h
    assert 'id="bc-headline"' in h and 'id="bc-mlp"' in h
    assert "async function refreshBaselineCompare()" in h
    assert '"/api/baseline-compare"' in h
    assert "refreshBaselineCompare();" in h                       # init call
    assert "setInterval(refreshBaselineCompare, 120_000)" in h     # poll
    # the diagnostic-not-recommendation framing (invariant #5) must stay —
    # the card must never read as "turn the gate off".
    assert "invariant #5" in h


def test_never_raises_into_the_panel_on_load_failure(monkeypatch):
    """A read fault must degrade to a verdict-keyed body, never a 500 with a
    bare stack — the card/chat must always find `verdict` to render."""
    def _boom(*a, **k):
        raise RuntimeError("decision_outcomes.jsonl unreadable")
    monkeypatch.setattr(d, "_load_decision_outcomes", _boom)
    d.app.config["TESTING"] = True
    with d.app.test_client() as c:
        r = c.get("/api/baseline-compare")
        j = r.get_json()
        assert "verdict" in j and j["verdict"] == "INSUFFICIENT_DATA"
        assert "error" in j
