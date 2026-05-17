"""Decision Scorer — MLP trained on (quant features) → predicted 5-day forward return.

Architecturally separate from ArticleNet (the text classifier). This model learns
from price outcomes, not text patterns, giving it signal that ArticleNet structurally
cannot learn. Trained on actual backtest BUY/SELL decisions with their real
5-trading-day forward returns.

Architecture: sklearn MLPRegressor (64, 32, 16) on 17 features (10 numeric:
8 quant + 2 news signals (urgency, article_count) + 7-way sector one-hot).
Falls back to a numpy weighted least-squares linear model when sklearn is
unavailable.
"""
from __future__ import annotations

import math
import pickle
import threading
from pathlib import Path

import numpy as np

SCORER_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "ml" / "decision_scorer.pkl"

# Process-wide load cache. Every polled dashboard endpoint constructs a fresh
# DecisionScorer() (paper_trader/dashboard.py builds one per request in
# /api/scorer-predictions, /api/scorer-confidence, /api/calibration,
# /api/disagreement, …). The old constructor re-read + re-unpickled
# scorer.pkl AND printed a "[decision_scorer] loaded n=" line on EVERY
# construction — 657 such lines in a single runner.log, which fed the
# disk-full logging failures (OSError: [Errno 28]) and burned needless disk
# I/O on every frontend poll.
#
# Key the cached (model, scaler, n_train) by the pickle's
# (path, st_mtime_ns, st_size). A retrain writes atomically (tmp + .replace,
# see train_scorer below) so a new model always changes the key and is
# picked up on the next construction — preserving the per-cycle retrain
# pickup the continuous loop relies on (it nulls backtest._DECISION_SCORER
# after every retrain). Only one entry is ever kept (the current file), so
# this is bounded regardless of how many retrain cycles run.
_LOAD_CACHE: dict[tuple, tuple] = {}
_LOAD_CACHE_LOCK = threading.Lock()


def _scorer_cache_key(path: Path):
    """(path, mtime_ns, size) signature, or None if the file is absent."""
    try:
        st = path.stat()
    except OSError:
        return None
    return (str(path), st.st_mtime_ns, st.st_size)

SECTORS = ["tech", "energy", "financials", "healthcare", "commodities", "crypto", "other"]

SECTOR_MAP: dict[str, str] = {
    # Tech / semis
    "NVDA": "tech", "AMD": "tech", "MU": "tech", "INTC": "tech", "QCOM": "tech",
    "AAPL": "tech", "MSFT": "tech", "META": "tech", "GOOGL": "tech", "AMZN": "tech",
    "TSLA": "tech", "CRM": "tech", "SNOW": "tech",
    "TSM": "tech", "ASML": "tech", "SMH": "tech", "SOXL": "tech", "TECL": "tech",
    "TQQQ": "tech", "QQQ": "tech", "XLK": "tech", "SHOP": "tech", "PLTR": "tech",
    "NVDU": "tech", "MSFU": "tech", "AMZU": "tech", "GOOGU": "tech", "METAU": "tech",
    "TSLL": "tech", "TSLT": "tech",
    "SOXS": "tech", "TECS": "tech", "FNGD": "tech", "FNGU": "tech",
    "SPY": "tech", "UPRO": "tech", "SPXL": "tech",  # broad index, treated as tech-correlated
    # Energy
    "XOM": "energy", "CVX": "energy", "XLE": "energy", "USO": "energy", "UNG": "energy",
    "BOIL": "energy", "UCO": "energy", "BP": "energy",
    # Financials
    "GS": "financials", "JPM": "financials", "BAC": "financials", "XLF": "financials",
    "FAS": "financials", "V": "financials", "MA": "financials", "UYG": "financials",
    "DPST": "financials", "HIBL": "financials",
    # Healthcare
    "LLY": "healthcare", "UNH": "healthcare", "NVO": "healthcare", "XLV": "healthcare",
    "CURE": "healthcare", "LABU": "healthcare",
    # Commodities / macro
    "GLD": "commodities", "SLV": "commodities", "TLT": "commodities", "GC=F": "commodities",
    "AGQ": "commodities", "RIO": "commodities", "BHP": "commodities",
    # Crypto
    "BTC-USD": "crypto", "COIN": "crypto", "MSTR": "crypto", "BITX": "crypto",
    "BITU": "crypto", "ETHU": "crypto", "CONL": "crypto",
}

N_FEATURES = 10 + len(SECTORS)  # 10 base (quant + news_urgency + news_article_count) + 7 sector one-hot = 17

# A 5-trading-day forward return is physically bounded. Across the 9,000+
# real outcomes in data/decision_outcomes.jsonl the distribution is
# p1=-25%, p99=+32%, and only ~0.4% of samples exceed |50%| (those are
# 3x-leveraged-ETF crash/rip weeks — genuinely real). An MLPRegressor with
# ReLU has no output bound, so for off-distribution feature vectors it
# extrapolates to nonsense like -89% for an optical-networking stock. Such a
# value isn't information; it's noise that pins the conviction board's ML
# axis to full conviction and destroys trader trust in every panel that
# surfaces it. Clamp the prediction to the empirical label support. ±50 is a
# round bound that still encloses 99.6% of all real outcomes AND every
# _ml_decide gate boundary (±10/±5/0), so gating behaviour is unchanged —
# a clamped -50 is still in the "p < -10 → ×0.6" bucket, exactly as -89 was.
PRED_CLAMP_PCT = 50.0


class _LstsqScaler:
    """Pickle-safe stand-in for sklearn's StandardScaler, used in the numpy fallback."""

    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.mean_ = np.asarray(mean, dtype=np.float32)
        self.std_ = np.asarray(std, dtype=np.float32)

    def transform(self, Xin) -> np.ndarray:
        X = np.asarray(Xin, dtype=np.float32)
        return (X - self.mean_) / self.std_


class _LstsqModel:
    """Pickle-safe linear least-squares predictor used when sklearn is unavailable."""

    def __init__(self, weights: np.ndarray) -> None:
        self.w_ = np.asarray(weights, dtype=np.float32)

    def predict(self, Xin) -> np.ndarray:
        X = np.asarray(Xin, dtype=np.float32)
        Xa = np.hstack([X, np.ones((len(X), 1), dtype=np.float32)])
        return Xa @ self.w_


def _to_float(v, default: float) -> float:
    # bool is a subclass of int — exclude it so True/False don't become 1.0/0.0.
    if isinstance(v, bool):
        return default
    # np.float64 inherits from float, but np.float32 / np.integer do not — so
    # a bare isinstance(v, (int, float)) check silently drops np.float32 values
    # (which come back from pandas/numpy operations) to the default. Add a
    # numpy fallback explicitly.
    #
    # math.isfinite rejects BOTH NaN and ±inf. A bare `v == v` only excluded
    # NaN: `float('inf') == float('inf')` is True, so an inf leaked straight
    # through. That broke the "always finite" contract predict_with_meta
    # advertises and — worse — a single decision_outcomes.jsonl row with a
    # non-finite forward_return_5d made train_scorer raise inside
    # MLPRegressor.fit, which _train_decision_scorer swallows, silently
    # wedging scorer retraining for that cycle and every cycle after (the
    # poisoned row persists in the 5000-record tail). The numpy branch below
    # already used np.isfinite; this aligns the Python branch with it.
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    # Guard on np.number, NOT np.generic. np.generic also covers np.str_ /
    # np.bool_ / np.object_, and np.isfinite raises an *unhandled* TypeError
    # on a numpy string ("ufunc 'isfinite' not supported") — that would
    # propagate straight out of build_features and crash train_scorer. np.number
    # is the precise numeric guard, mirroring the (int, float) check above; a
    # numpy bool/string falls through to the safe default just like a Python
    # bool/str already does.
    if isinstance(v, np.number) and np.isfinite(v):
        return float(v)
    return default


def build_features(
    ml_score: float,
    rsi: float | None,
    macd: float | None,
    mom5: float | None,
    mom20: float | None,
    regime_mult: float,
    ticker: str,
    vol_ratio: float | None = None,
    bb_pos: float | None = None,
    news_urgency: float | None = None,
    news_article_count: float | None = None,
) -> list[float]:
    """Build a fixed-length feature vector for one decision."""
    rsi_v = _to_float(rsi, 50.0)
    macd_v = _to_float(macd, 0.0)
    mom5_v = _to_float(mom5, 0.0)
    mom20_v = _to_float(mom20, 0.0)
    sector = SECTOR_MAP.get(ticker, "other")
    sector_oh = [1.0 if s == sector else 0.0 for s in SECTORS]
    vol_v = max(0.0, min(5.0, _to_float(vol_ratio, 1.0)))
    bb_v = max(-2.0, min(2.0, _to_float(bb_pos, 0.0)))
    urg_v = max(0.0, min(100.0, _to_float(news_urgency, 50.0)))
    cnt_v = max(0.0, min(20.0, _to_float(news_article_count, 1.0)))
    return [_to_float(ml_score, 0.0), rsi_v, macd_v, mom5_v, mom20_v,
            _to_float(regime_mult, 1.0), vol_v, bb_v, urg_v, cnt_v] + sector_oh


class DecisionScorer:
    """Lightweight MLP: (quant_features) → predicted 5-day forward return (%)."""

    def __init__(self) -> None:
        self._model = None
        self._scaler = None
        self._trained = False
        self._n_train = 0
        if SCORER_PATH.exists():
            self._load()

    def _load(self) -> None:
        key = _scorer_cache_key(SCORER_PATH)
        if key is None:
            return
        # Hold the lock across the unpickle so concurrent Flask request
        # threads on a cold start (or right after a retrain) do exactly one
        # disk read and never observe a torn cache tuple. Steady state is a
        # single dict lookup under an uncontended lock.
        with _LOAD_CACHE_LOCK:
            cached = _LOAD_CACHE.get(key)
            if cached is not None:
                self._model, self._scaler, self._n_train = cached
                self._trained = True
                return
            try:
                with SCORER_PATH.open("rb") as f:
                    state = pickle.load(f)
                self._model = state["model"]
                self._scaler = state.get("scaler")
                self._n_train = int(state.get("n_train", 0))
                self._trained = True
                # Only the current file is ever relevant; clearing bounds the
                # cache to one entry across unbounded retrain cycles.
                _LOAD_CACHE.clear()
                _LOAD_CACHE[key] = (self._model, self._scaler, self._n_train)
                print(f"[decision_scorer] loaded n={self._n_train} from {SCORER_PATH}")
            except Exception as e:
                print(f"[decision_scorer] load failed: {e}")

    _predict_err_logged: bool = False

    def predict_with_meta(
        self,
        ml_score: float,
        rsi: float | None,
        macd: float | None,
        mom5: float | None,
        mom20: float | None,
        regime_mult: float,
        ticker: str,
        vol_ratio: float | None = None,
        bb_pos: float | None = None,
        news_urgency: float | None = None,
        news_article_count: float | None = None,
    ) -> dict:
        """Predicted 5d forward return (%) plus calibration metadata.

        Returns ``{"pred", "raw", "clamped", "off_distribution"}``:
        - ``pred``  — the value callers should act on (clamped, always finite)
        - ``raw``   — the model's unbounded output (for diagnostics / honesty)
        - ``clamped`` — True when ``|raw| > PRED_CLAMP_PCT`` (or non-finite)
        - ``off_distribution`` — alias of ``clamped``; a True here means the
          model extrapolated past the empirical label support and the point
          estimate should be treated as low-trust, not gospel.

        ``predict()`` is the scalar fast path every existing consumer uses;
        this sibling exists for panels that want to surface the trust flag
        without changing ``predict()``'s float contract.
        """
        if not self._trained or self._model is None:
            return {"pred": 0.0, "raw": 0.0, "clamped": False,
                    "off_distribution": False}
        try:
            X = np.array(
                [build_features(ml_score, rsi, macd, mom5, mom20, regime_mult, ticker,
                                vol_ratio=vol_ratio, bb_pos=bb_pos,
                                news_urgency=news_urgency,
                                news_article_count=news_article_count)],
                dtype=np.float32,
            )
            if self._scaler is not None:
                X = self._scaler.transform(X)
            raw = float(self._model.predict(X)[0])
        except Exception as e:
            # Log once per instance — silent swallow was masking shape / dtype
            # mismatches when feature additions were rolled out without retraining.
            if not self._predict_err_logged:
                print(f"[decision_scorer] predict error (silenced after first): {e}")
                self._predict_err_logged = True
            return {"pred": 0.0, "raw": 0.0, "clamped": False,
                    "off_distribution": False}

        # A non-finite model output (inf/nan from a pathological feature
        # vector) is unusable — treat it as a 0% / off-distribution result
        # rather than letting nan propagate silently through max/min.
        if not np.isfinite(raw):
            return {"pred": 0.0, "raw": raw, "clamped": True,
                    "off_distribution": True}
        clamped_pred = max(-PRED_CLAMP_PCT, min(PRED_CLAMP_PCT, raw))
        was_clamped = abs(raw) > PRED_CLAMP_PCT
        return {"pred": clamped_pred, "raw": raw, "clamped": was_clamped,
                "off_distribution": was_clamped}

    def predict(
        self,
        ml_score: float,
        rsi: float | None,
        macd: float | None,
        mom5: float | None,
        mom20: float | None,
        regime_mult: float,
        ticker: str,
        vol_ratio: float | None = None,
        bb_pos: float | None = None,
        news_urgency: float | None = None,
        news_article_count: float | None = None,
    ) -> float:
        """Return predicted 5d forward return (%), clamped to the empirical
        label support (see ``PRED_CLAMP_PCT``). Returns 0.0 if not trained.

        Scalar contract preserved for every existing consumer (``_ml_decide``
        gate, ``_live_scorer_predictions``, ``scorer_confidence``)."""
        return self.predict_with_meta(
            ml_score, rsi, macd, mom5, mom20, regime_mult, ticker,
            vol_ratio=vol_ratio, bb_pos=bb_pos, news_urgency=news_urgency,
            news_article_count=news_article_count,
        )["pred"]

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def n_train(self) -> int:
        return self._n_train


def train_scorer(records: list[dict]) -> dict:
    """Train DecisionScorer on outcome records.

    Each record must have: ml_score, rsi, macd, mom5, mom20, regime_mult, ticker,
    forward_return_5d. Optional: action (BUY/SELL — SELL flips target sign so the
    model learns "goodness of THIS action"), return_pct (overall backtest run
    quality, used to weight samples). Returns stats dict.
    """
    if not records:
        return {"status": "insufficient_data", "n": 0}

    # Deduplicate identical decisions seen across multiple runs (same features,
    # same label) so they don't dominate training — keep the highest-return-run
    # copy. The key MUST include action: build_features ignores action, and the
    # SELL branch below flips the target sign, so a BUY and a SELL of the same
    # ticker on the same day share features but carry OPPOSITE labels. Keying on
    # (ticker, sim_date) alone silently discarded one of the pair — and with
    # opposing personas (momentum vs contrarian) trading the same names, that
    # collision is hit constantly.
    seen: dict[tuple, dict] = {}
    for r in records:
        key = (
            str(r.get("ticker") or ""),
            str(r.get("sim_date") or ""),
            str(r.get("action") or "BUY").upper(),
        )
        rp = _to_float(r.get("return_pct"), 0.0)
        if key not in seen or rp > _to_float(seen[key].get("return_pct"), 0.0):
            seen[key] = r
    records = list(seen.values())

    # Length gate applied AFTER dedup — dedup can shrink the set well below the
    # raw input count, and the model needs a real minimum of distinct samples.
    if len(records) < 30:
        return {"status": "insufficient_after_dedup", "n": len(records)}

    X_raw, y, weights = [], [], []
    for r in records:
        X_raw.append(build_features(
            _to_float(r.get("ml_score"), 0.0),
            r.get("rsi"),
            r.get("macd"),
            r.get("mom5"),
            r.get("mom20"),
            _to_float(r.get("regime_mult"), 1.0),
            str(r.get("ticker") or ""),
            vol_ratio=r.get("vol_ratio"),
            bb_pos=r.get("bb_position"),
            news_urgency=r.get("news_urgency"),
            news_article_count=r.get("news_article_count"),
        ))
        # Use _to_float so JSON nulls / missing keys / strings don't crash.
        # Prior float(r.get(..., default)) crashed on `null` values because
        # dict.get returns the value (None) even when a default is supplied.
        fr = _to_float(r.get("forward_return_5d"), 0.0)
        action = str(r.get("action") or "BUY").upper()
        # SELL: negative forward returns were the *correct* outcome, so flip
        # sign — the model then learns one consistent meaning of "good".
        y.append(-fr if action == "SELL" else fr)
        # Sample weight from overall run quality:
        # +200% run → 2.0×, 0% → 1.0×, -100%+ → 0.5×.
        rp = _to_float(r.get("return_pct"), 0.0)
        # LLM annotation multiplier: endorsed → 3x signal, condemned → 0.1x
        # `or 0` guards against an explicit JSON null in the record.
        llm_label = int(r.get("llm_quality_label") or 0)
        llm_mult = {1: 3.0, -1: 0.1, 0: 1.0}.get(llm_label, 1.0)
        weights.append(max(0.5, min(2.0, 1.0 + rp / 200.0)) * llm_mult)

    X = np.array(X_raw, dtype=np.float32)
    y = np.array(y, dtype=np.float32)
    weights = np.array(weights, dtype=np.float32)

    try:
        from sklearn.neural_network import MLPRegressor
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import train_test_split

        # Split BEFORE scaling. Fitting StandardScaler on the full set leaks
        # validation-fold statistics into the reported val_rmse. Fit on the
        # training fold only; the model is itself trained on that fold, so the
        # pickled scaler stays consistent with the model at inference.
        scaler = StandardScaler()
        X_tr_raw, X_v_raw, y_tr, y_v, w_tr, _ = train_test_split(
            X, y, weights, test_size=0.2, random_state=42
        )
        X_tr = scaler.fit_transform(X_tr_raw)
        X_v = scaler.transform(X_v_raw)
        # MLPRegressor.fit doesn't accept sample_weight — emulate by deterministic
        # oversampling: weight 0.5→1× replica, 1.0→2×, 1.5→3×, 2.0→4×. Done
        # only on the training fold so val_rmse stays clean.
        rep = np.maximum(1, np.round(w_tr * 2).astype(int))
        X_tr_w = np.repeat(X_tr, rep, axis=0)
        y_tr_w = np.repeat(y_tr, rep, axis=0)

        model = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation="relu",
            max_iter=600,
            random_state=42,
        )
        model.fit(X_tr_w, y_tr_w)
        y_pred = model.predict(X_v)
        val_rmse = float(np.sqrt(np.mean((y_pred - y_v) ** 2)))

    except ImportError:
        # Numpy weighted least-squares linear fallback when sklearn not installed.
        # Uses module-level _LstsqScaler / _LstsqModel so the resulting pickle
        # can be loaded later — closures cannot be pickled by name.
        scaler_mean = X.mean(axis=0)
        scaler_std = X.std(axis=0) + 1e-8
        X_s = (X - scaler_mean) / scaler_std
        X_aug = np.hstack([X_s, np.ones((len(X_s), 1), dtype=np.float32)])
        sw = np.sqrt(weights).astype(np.float32).reshape(-1, 1)
        w, _, _, _ = np.linalg.lstsq(X_aug * sw, y * sw.ravel(), rcond=None)
        scaler = _LstsqScaler(scaler_mean, scaler_std)
        model = _LstsqModel(w)
        val_rmse = float("nan")

    SCORER_PATH.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: a torn pickle (process killed mid-write, or a backtest
    # thread loading the file concurrently) would leave DecisionScorer
    # permanently untrained. Write to a temp file then atomically replace.
    _tmp = SCORER_PATH.with_suffix(".pkl.tmp")
    with _tmp.open("wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "n_train": len(records)}, f)
    _tmp.replace(SCORER_PATH)

    return {"status": "ok", "n": len(records), "val_rmse": val_rmse}
