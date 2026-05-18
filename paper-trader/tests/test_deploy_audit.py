"""Exact-value offline locks for the deployed-scorer config-vs-source audit
(`paper_trader/ml/deploy_audit.py`, 2026-05-18 quant feature).

Mirrors test_overfit_gap.py / test_baseline_trend.py: deterministic synthetic
pickles, exact verdicts (not ranges) so a logic change must update the
literals deliberately. All offline; the module never trains, never writes a
pickle, never touches the trade path.

The decisive scenario this feature exists for — deployed `(64,32,16)/
alpha=1e-4/early_stopping=False` while source `MLP_CONFIG` says
`(32,16)/alpha=1e-2/early_stopping=True` — is locked exactly in
`TestStaleConfig`.
"""
from __future__ import annotations

import pickle

import pytest

from paper_trader.ml import deploy_audit as da
from paper_trader.ml import decision_scorer as ds


class _FakeModel:
    """Minimal stand-in for a fitted sklearn MLPRegressor — only the
    config attributes the audit introspects matter."""

    def __init__(self, **attrs):
        for k, v in attrs.items():
            setattr(self, k, v)


def _write_pickle(path, model, n_train=600):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        pickle.dump({"model": model, "scaler": None, "n_train": n_train}, fh)


# The exact source config the regularized net (commit 5a0af2d) must carry.
_SOURCE = {
    "hidden_layer_sizes": (32, 16),
    "activation": "relu",
    "max_iter": 1000,
    "random_state": 42,
    "alpha": 1e-2,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "n_iter_no_change": 25,
}

# The documented stale net the running loop keeps re-pickling.
_STALE = {
    "hidden_layer_sizes": (64, 32, 16),
    "activation": "relu",
    "max_iter": 600,
    "random_state": 42,
    "alpha": 1e-4,
    "early_stopping": False,
    "validation_fraction": 0.15,
    "n_iter_no_change": 25,
}


class TestSingleSourceOfTruth:
    """`MLP_CONFIG` is the one place the kwargs live; `train_scorer` builds
    from it and this audit compares against it — no hand-maintained mirror."""

    def test_mlp_config_exact_values(self):
        # A drift here must be a deliberate literal edit (the anti-overfit
        # retune's contract). Locks every audited key.
        assert ds.MLP_CONFIG == _SOURCE

    def test_train_scorer_references_the_constant(self):
        import inspect
        src = inspect.getsource(ds.train_scorer)
        assert "MLPRegressor(**MLP_CONFIG)" in src


class TestMatchingDeploy:
    def test_matches_source(self, tmp_path):
        p = tmp_path / "scorer.pkl"
        _write_pickle(p, _FakeModel(**_SOURCE))
        rep = da.audit_deployed_config(p, ds.MLP_CONFIG)
        assert rep["verdict"] == "DEPLOYED_MATCHES_SOURCE"
        assert rep["n_audited"] == 8
        assert rep["n_mismatched"] == 0
        assert rep["mismatches"] == []
        assert rep["deployed"]["hidden_layer_sizes"] == (32, 16)

    def test_list_normalizes_to_tuple(self, tmp_path):
        # sklearn stores hidden_layer_sizes as passed; a legacy list form
        # must still read as the same architecture, not a false STALE.
        m = _FakeModel(**{**_SOURCE, "hidden_layer_sizes": [32, 16]})
        p = tmp_path / "scorer.pkl"
        _write_pickle(p, m)
        rep = da.audit_deployed_config(p, ds.MLP_CONFIG)
        assert rep["verdict"] == "DEPLOYED_MATCHES_SOURCE"

    def test_float_isclose_not_exact(self, tmp_path):
        # A pickle round-trip's last-bit drift on alpha must not read STALE.
        m = _FakeModel(**{**_SOURCE, "alpha": 0.01 + 1e-15})
        p = tmp_path / "scorer.pkl"
        _write_pickle(p, m)
        assert (da.audit_deployed_config(p, ds.MLP_CONFIG)["verdict"]
                == "DEPLOYED_MATCHES_SOURCE")


class TestStaleConfig:
    """The decisive documented scenario — exact mismatch payload locked."""

    def test_stale_64_32_16_net(self, tmp_path):
        p = tmp_path / "scorer.pkl"
        _write_pickle(p, _FakeModel(**_STALE))
        rep = da.audit_deployed_config(p, ds.MLP_CONFIG)
        assert rep["verdict"] == "DEPLOYED_STALE_CONFIG"
        # The pre-retune (64,32,16)/600-iter/alpha=1e-4/no-early-stop net
        # drifts in exactly these four hyper-params vs the source.
        keyed = {m["key"]: m for m in rep["mismatches"]}
        assert set(keyed) == {"hidden_layer_sizes", "max_iter", "alpha",
                              "early_stopping"}
        assert rep["n_mismatched"] == 4
        assert keyed["hidden_layer_sizes"]["deployed"] == (64, 32, 16)
        assert keyed["hidden_layer_sizes"]["expected"] == (32, 16)
        assert keyed["alpha"]["deployed"] == 1e-4
        assert keyed["alpha"]["expected"] == 1e-2
        assert keyed["early_stopping"]["deployed"] is False
        assert keyed["early_stopping"]["expected"] is True
        assert keyed["max_iter"]["deployed"] == 600
        assert da.is_deploy_stale(p, ds.MLP_CONFIG) is True

    def test_missing_attribute_is_mismatch(self, tmp_path):
        m = _FakeModel(**{k: v for k, v in _SOURCE.items()
                          if k != "early_stopping"})
        p = tmp_path / "scorer.pkl"
        _write_pickle(p, m)
        rep = da.audit_deployed_config(p, ds.MLP_CONFIG)
        assert rep["verdict"] == "DEPLOYED_STALE_CONFIG"
        keyed = {m["key"]: m for m in rep["mismatches"]}
        assert keyed["early_stopping"]["deployed"] is None

    def test_early_stopping_false_vs_true_is_mismatch(self, tmp_path):
        m = _FakeModel(**{**_SOURCE, "early_stopping": False})
        p = tmp_path / "scorer.pkl"
        _write_pickle(p, m)
        rep = da.audit_deployed_config(p, ds.MLP_CONFIG)
        assert rep["verdict"] == "DEPLOYED_STALE_CONFIG"
        assert rep["n_mismatched"] == 1


class TestCannotTellVerdicts:
    def test_no_pickle_is_insufficient(self, tmp_path):
        rep = da.audit_deployed_config(tmp_path / "absent.pkl", ds.MLP_CONFIG)
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert rep["deployed"] is None
        assert da.is_deploy_stale(tmp_path / "absent.pkl", ds.MLP_CONFIG) is None

    def test_unreadable_pickle(self, tmp_path):
        p = tmp_path / "scorer.pkl"
        p.write_bytes(b"\x00not a pickle\xff")
        rep = da.audit_deployed_config(p, ds.MLP_CONFIG)
        assert rep["verdict"] == "UNREADABLE_PICKLE"
        assert da.is_deploy_stale(p, ds.MLP_CONFIG) is None

    def test_lstsq_fallback_detected(self, tmp_path):
        import numpy as np
        model = ds._LstsqModel(np.zeros(ds.N_FEATURES + 1, dtype=np.float32))
        p = tmp_path / "scorer.pkl"
        _write_pickle(p, model)
        rep = da.audit_deployed_config(p, ds.MLP_CONFIG)
        assert rep["verdict"] == "LSTSQ_FALLBACK"
        assert da.is_deploy_stale(p, ds.MLP_CONFIG) is None

    def test_empty_expected_config(self, tmp_path):
        p = tmp_path / "scorer.pkl"
        _write_pickle(p, _FakeModel(**_SOURCE))
        rep = da.audit_deployed_config(p, {})
        assert rep["verdict"] == "INSUFFICIENT_DATA"

    def test_never_raises_on_garbage(self):
        # Pure/total contract — every can't-tell input degrades, never raises.
        assert da.audit_deployed_config(None, None)["verdict"] \
            in ("INSUFFICIENT_DATA", "UNREADABLE_PICKLE")
        assert da.is_deploy_stale(12345, "not a dict") is None


class TestRoundTripWithRealPickle:
    """A real `train_scorer` pickle (built from MLP_CONFIG) must read as
    MATCHES — the end-to-end no-drift guarantee, not just synthetic fakes."""

    def test_trained_pickle_matches_source(self, tmp_path, monkeypatch):
        pytest.importorskip("sklearn")
        import numpy as np
        from datetime import date, timedelta
        rng = np.random.RandomState(0)
        base = date(2024, 1, 1)
        recs = []
        for i in range(80):
            # Distinct sim_date per row so train_scorer's
            # (ticker, sim_date, action) dedup keeps all 80 (≥30 gate).
            recs.append({
                "ml_score": float(rng.uniform(0, 5)),
                "rsi": float(rng.uniform(20, 80)),
                "macd": float(rng.uniform(-1, 1)),
                "mom5": float(rng.uniform(-5, 5)),
                "mom20": float(rng.uniform(-10, 10)),
                "regime_mult": 1.0,
                "ticker": "NVDA",
                "sim_date": (base + timedelta(days=i)).isoformat(),
                "action": "BUY",
                "forward_return_5d": float(rng.uniform(-8, 8)),
            })
        p = tmp_path / "scorer.pkl"
        monkeypatch.setattr(ds, "SCORER_PATH", p)
        res = ds.train_scorer(recs)
        assert res["status"] == "ok"
        rep = da.audit_deployed_config(p, ds.MLP_CONFIG)
        assert rep["verdict"] == "DEPLOYED_MATCHES_SOURCE", rep["mismatches"]


class TestCli:
    def test_cli_exit_2_on_stale(self, tmp_path, monkeypatch, capsys):
        p = tmp_path / "scorer.pkl"
        _write_pickle(p, _FakeModel(**_STALE))
        monkeypatch.setattr(ds, "SCORER_PATH", p)
        rc = da._cli()
        assert rc == 2
        assert "DEPLOYED_STALE_CONFIG" in capsys.readouterr().out

    def test_cli_exit_0_on_match(self, tmp_path, monkeypatch):
        p = tmp_path / "scorer.pkl"
        _write_pickle(p, _FakeModel(**_SOURCE))
        monkeypatch.setattr(ds, "SCORER_PATH", p)
        assert da._cli() == 0

    def test_cli_exit_0_when_insufficient(self, tmp_path, monkeypatch):
        monkeypatch.setattr(ds, "SCORER_PATH", tmp_path / "nope.pkl")
        assert da._cli() == 0


class TestLedgerWiring:
    """`_append_scorer_skill_log` must carry a `deploy_stale` field so the
    stale-net state is trendable per-cycle, not only a CLI."""

    def test_row_has_deploy_stale_key(self, tmp_path, monkeypatch):
        import run_continuous_backtests as rcb
        import json
        from datetime import date
        log = tmp_path / "skill.jsonl"
        monkeypatch.setattr(rcb, "SCORER_SKILL_LOG", log)
        # conftest redirects ds.SCORER_PATH to an empty tmp dir → no pickle →
        # is_deploy_stale() returns None (honest can't-tell, not False).
        ok = rcb._append_scorer_skill_log(
            "scorer ok train_n=600 val_rmse=5.0 oos_n=100 oos_rmse=12.0 "
            "oos_diracc=0.5 oos_ic=+0.01",
            cycle=3, win_start=date(2015, 1, 2), win_end=date(2016, 1, 2),
        )
        assert ok is True
        row = json.loads(log.read_text().strip())
        assert "deploy_stale" in row
        assert row["deploy_stale"] is None
