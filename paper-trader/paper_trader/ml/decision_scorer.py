"""Decision Scorer — MLP trained on (quant features) → predicted 5-day forward return.

Architecturally separate from ArticleNet (the text classifier). This model learns
from price outcomes, not text patterns, giving it signal that ArticleNet structurally
cannot learn. Trained on actual backtest BUY/SELL decisions with their real
5-trading-day forward returns.

Architecture: regularized sklearn MLPRegressor (32, 16) — L2 alpha + early
stopping (anti-overfit, 2026-05-18; see the train_scorer config comment) — on
20 features (10 base numeric: 8 quant + 2 news signals (urgency,
article_count); 3 enhanced MACD/EMA200 booleans (ema200_above,
hist_cross_up, macd_below_zero_cross); 7-way sector one-hot). Falls back
to a numpy weighted least-squares linear model when sklearn is unavailable.
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
    # Tech: semi capital-equipment + optical (LITE/AMAT/LRCX) — group with NVDA/AMD/MU.
    # Until now these 35% of the watchlist landed in sector_other, collapsing
    # to one bucket with utilities/defense/Toyota and erasing every learnable
    # semi-equipment correlation.
    "LITE": "tech", "AMAT": "tech", "LRCX": "tech",
    # Tech: international tech ADRs — SAP/SONY/BABA trade with the global tech
    # tape, not "other".
    "BABA": "tech", "SAP": "tech", "SONY": "tech",
    # Tech: EV / innovation — treated like TSLA (already mapped).
    "RIVN": "tech", "NIO": "tech", "ARKK": "tech",
    # Tech: broad-index 3x leveraged ETFs — same "treated as tech-correlated"
    # pattern documented inline for SPY/UPRO/SPXL above.
    "UDOW": "tech", "URTY": "tech", "TNA": "tech", "MIDU": "tech", "WANT": "tech",
    # Tech: broad-index 2x leveraged ETFs — same pattern (QQQ/SPY/Dow/small/Russell).
    "QLD": "tech", "SSO": "tech", "MVV": "tech", "SAA": "tech", "UWM": "tech",
    # Tech: single-stock 2x leveraged (AAPL/SMCI/PLTR/Nokia) — mirrors NVDU/MSFU pattern.
    "AAPLU": "tech", "SMCI2X": "tech", "PLTU": "tech", "LNOK": "tech",
    # Tech: 2x rotation ETFs (tech-themed) — USD/ROM.
    "USD": "tech", "ROM": "tech",
    # Tech: 3x inverse broad-index — same sector correlation magnitude (just
    # opposite direction). Same pattern existing SOXS/TECS/FNGD already follow.
    "SQQQ": "tech", "SPXS": "tech", "SDOW": "tech", "SRTY": "tech",
    "TZA": "tech", "HIBS": "tech",
    # Tech: power semis / GaN / SiC / RF semis — same fundamental driver as
    # NVDA/AMD/MU (silicon cycle, semi-equipment capex). WOLF (Wolfspeed SiC),
    # MPWR (Monolithic Power), STM (STMicro), NVTS (Navitas), MCHP (Microchip),
    # AMBA (Ambarella vision SoC), SWKS/QRVO (RF semis).
    "NVTS": "tech", "MPWR": "tech", "WOLF": "tech", "STM": "tech",
    "MCHP": "tech", "AMBA": "tech", "SWKS": "tech", "QRVO": "tech",
    # Tech: AI infrastructure / accelerators / EDA / semi-test — same family
    # as the deployed NVDA/AMD bucket. SMCI (AI servers), AVGO (Broadcom AI
    # silicon), ARM, MRVL (Marvell DPU), CDNS/SNPS (EDA), AEHR/COHU (semi
    # test equipment, group with LITE/AMAT/LRCX).
    "SMCI": "tech", "AVGO": "tech", "ARM": "tech", "MRVL": "tech",
    "CDNS": "tech", "SNPS": "tech", "AEHR": "tech", "COHU": "tech",
    # Tech: quantum compute — speculative growth-tech tape, same momentum-name
    # bucket as ARKK/PLTR/RIVN. IONQ, RGTI (Rigetti), QUBT (Quantum Computing
    # Inc), ARQQ (Arqit), QMCO (Quantum Corp data tape — quantum-adjacent).
    "IONQ": "tech", "RGTI": "tech", "QUBT": "tech", "ARQQ": "tech",
    "QMCO": "tech",
    # Tech: AI software / voice / next-gen comms — same growth-tech tape.
    # SOUN (SoundHound voice AI), BBAI (BigBear AI defense analytics), ASTS
    # (AST SpaceMobile direct-to-cell satellite), and the legacy AST listing
    # which is the same underlying — both trade with the satellite-comms tape.
    "SOUN": "tech", "BBAI": "tech", "ASTS": "tech", "AST": "tech",
    # Tech: space launch / eVTOL autonomy — high-beta growth-tech speculative
    # names, same tape as ARKK / RIVN / NIO. RKLB (Rocket Lab), LUNR
    # (Intuitive Machines lunar lander), ACHR (Archer Aviation eVTOL),
    # JOBY (Joby Aviation eVTOL).
    "RKLB": "tech", "LUNR": "tech", "ACHR": "tech", "JOBY": "tech",
    # Tech: single-stock 2x leveraged NVDA — same NVDU pattern (NVDX is the
    # second 2x-NVDA-leveraged ETF, mirroring the existing AAPLU/SMCI2X/PLTU
    # placement).
    "NVDX": "tech",
    # Healthcare: medical AI / surgical robotics — clearly healthcare-tape.
    # BFLY (Butterfly Network hand-held ultrasound), PRCT (Procept BioRobotics
    # surgical waterjet). Same bucket as LLY/UNH/NVO.
    "BFLY": "healthcare", "PRCT": "healthcare",
    # Energy
    "XOM": "energy", "CVX": "energy", "XLE": "energy", "USO": "energy", "UNG": "energy",
    "BOIL": "energy", "UCO": "energy", "BP": "energy",
    # Financials
    "GS": "financials", "JPM": "financials", "BAC": "financials", "XLF": "financials",
    "FAS": "financials", "V": "financials", "MA": "financials", "UYG": "financials",
    "DPST": "financials", "HIBL": "financials",
    # Financials: mega-cap / international bank / fintech / 3x inverse (mirrors FAS).
    "BRK-B": "financials", "HSBC": "financials", "SQ": "financials",
    "FAZ": "financials",
    # Healthcare
    "LLY": "healthcare", "UNH": "healthcare", "NVO": "healthcare", "XLV": "healthcare",
    "CURE": "healthcare", "LABU": "healthcare",
    # Commodities / macro
    "GLD": "commodities", "SLV": "commodities", "TLT": "commodities", "GC=F": "commodities",
    "AGQ": "commodities", "RIO": "commodities", "BHP": "commodities",
    # Crypto
    "BTC-USD": "crypto", "COIN": "crypto", "MSTR": "crypto", "BITX": "crypto",
    "BITU": "crypto", "ETHU": "crypto", "CONL": "crypto",
    # "other" by design (no industrials/utilities/real-estate sector enum):
    # TM (Toyota auto), UXI (2x industrials, mirrors XLI's implicit "other"),
    # NAIL (homebuilders), DFEN (defense), UTSL (utilities), ^VIX (vol gauge),
    # NEWS_VEH_PSEUDO_TICKERS (ES=F/NQ=F/CL=F futures). These are *intentionally*
    # in sector_other — adding an entry above for any of them would mis-couple
    # them with tech/financials. The coverage test pins this allow-list.
}

# Watchlist tickers that *intentionally* fall through to sector_other because
# their economic sector has no SECTORS enum (no industrials, utilities,
# real-estate, defense, vol-gauge categories). The coverage test asserts
# WATCHLIST ⊆ (SECTOR_MAP ∪ INTENTIONALLY_OTHER ∪ {^…}), so a NEW watchlist
# ticker added without explicit classification fails loudly rather than
# silently degrading to "other".
INTENTIONALLY_OTHER: frozenset[str] = frozenset({
    "TM",       # Toyota — auto industrial, no auto sector
    "UXI",      # 2x industrials — XLI already implicitly "other"
    "NAIL",     # homebuilders 3x — no real-estate sector
    "DFEN",     # defense 3x — no defense sector (XLI route)
    "UTSL",     # utilities 3x — no utilities sector
    "XLI",      # industrials sector ETF itself — no industrials enum
    # Nuclear small-caps — they trade like utility-adjacent / spec-energy
    # hybrids and do NOT co-vary with the XOM/CVX oil&gas bucket; routing
    # them to "energy" would mis-couple. No utility sector enum either, so
    # honestly "other" is the right bucket (the same logic UTSL already follows).
    "OKLO", "NNE",
})

N_FEATURES = 10 + len(SECTORS) + 3  # 10 base + 7 sector one-hot + 3 enhanced MACD = 20

# Human-readable name per build_features() output slot, in order. Single
# source of truth shared by feature_contributions() (and any future
# attribution consumer) so a feature reorder can't silently mislabel an
# attribution panel. Kept in lockstep with build_features()'s return list.
# The 3 enhanced MACD features (ema200_above, hist_cross_up,
# macd_below_zero_cross) are inserted AFTER the 10 base numeric features
# but BEFORE the 7 sector one-hot block — preserves the invariant "the last
# len(SECTORS) entries are the sector one-hot", which several attribution
# consumers (and the existing per-sector test) rely on. Existing pickles
# ARE shape-mismatched (17 → 20 input dim); predict_with_meta catches the
# shape error and degrades to failed=True / pred=0.0 until the next
# continuous-loop retrain rebuilds the model on the new 20-feature vector.
FEATURE_NAMES = [
    "ml_score", "rsi", "macd", "mom5", "mom20", "regime_mult",
    "vol_ratio", "bb_pos", "news_urgency", "news_article_count",
    "ema200_above", "hist_cross_up", "macd_below_zero_cross",
] + [f"sector_{s}" for s in SECTORS]
assert len(FEATURE_NAMES) == N_FEATURES, "FEATURE_NAMES drifted from build_features()"

# Operational feature groups for at-a-glance prediction triage.
#
# `feature_contributions()` returns 17 per-feature contributions — useful
# for forensics but noisy for a researcher who wants to answer "is the
# scorer leaning on news, on quant signals, on the sector bias, or on the
# trade-thesis ml_score?". `feature_group_contributions()` rolls the 17
# features into 5 semantic buckets that map 1:1 to the *operator's* mental
# model of the inputs:
#
#   ml_score  — the ArticleNet trade-thesis score itself (1 feature)
#   quant     — purely-technical price signals (6 features)
#   regime    — SPY-50/200 market-regime multiplier (1 feature)
#   news      — news-context features (urgency + article count, 2 features)
#   sector    — sector one-hot (7 features — a 60% sector_tech contribution
#               is a sector bias, not 7 unrelated signals)
#
# A FEATURE_GROUP_MAP entry must exist for every name in FEATURE_NAMES; the
# assert below is the same drift-guard FEATURE_NAMES itself uses against
# build_features(). FEATURE_GROUPS is the canonical display order
# (matches the FEATURE_NAMES axis: thesis → quant → regime → news → sector).
FEATURE_GROUPS: tuple[str, ...] = (
    "ml_score", "quant", "regime", "news", "sector",
)
FEATURE_GROUP_MAP: dict[str, str] = {
    "ml_score": "ml_score",
    "rsi": "quant", "macd": "quant", "mom5": "quant", "mom20": "quant",
    "vol_ratio": "quant", "bb_pos": "quant",
    "regime_mult": "regime",
    "news_urgency": "news", "news_article_count": "news",
    # Enhanced MACD / EMA200 features land in the "quant" bucket — they are
    # purely-technical price signals, same family as RSI / MACD / momentum.
    "ema200_above": "quant",
    "hist_cross_up": "quant",
    "macd_below_zero_cross": "quant",
}
for _s in SECTORS:
    FEATURE_GROUP_MAP[f"sector_{_s}"] = "sector"
assert set(FEATURE_GROUP_MAP.keys()) == set(FEATURE_NAMES), \
    "FEATURE_GROUP_MAP drifted from FEATURE_NAMES"
assert set(FEATURE_GROUP_MAP.values()) == set(FEATURE_GROUPS), \
    "FEATURE_GROUPS drifted from FEATURE_GROUP_MAP values"

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

# Single source of truth for the sklearn MLPRegressor hyper-parameters.
# `train_scorer` builds the model with `MLPRegressor(**MLP_CONFIG)`, and the
# read-only `paper_trader.ml.deploy_audit` diagnostic introspects a deployed
# pickle's fitted model attributes against THIS dict to answer the single
# most-repeated ML/backtest finding: "the running loop predates the
# anti-overfit retune — it is still gating real conviction on the memorizing
# (64,32,16)/alpha=1e-4/early_stopping=False net while the source says
# (32,16)/alpha=1e-2/early_stopping=True". Keeping the kwargs here (not inline
# in train_scorer) makes that comparison a true no-drift check rather than a
# hand-maintained mirror. Anti-overfit config (2026-05-18); see the
# train_scorer comment for the OOS-RMSE evidence behind each value.
MLP_CONFIG: dict = {
    "hidden_layer_sizes": (32, 16),
    "activation": "relu",
    "max_iter": 1000,
    "random_state": 42,
    "alpha": 1e-2,
    "early_stopping": True,
    "validation_fraction": 0.15,
    "n_iter_no_change": 25,
}


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
        # Mirror sklearn's `MLPRegressor.predict` 1D/2D acceptance so a
        # caller that hands in a single feature vector (shape (n,)) doesn't
        # crash on the np.hstack mixed-dimension error
        # ("ValueError: all the input arrays must have same number of
        # dimensions") that bare 1D + ones((len(X), 1)) produces. Production
        # paths inside DecisionScorer always batch (predict_with_meta wraps
        # build_features in `np.array([…])`; feature_contributions vstacks),
        # so this is a robustness fix for the public contract — same shape
        # acceptance as the sklearn path the lstsq fallback substitutes for.
        if X.ndim == 1:
            X = X.reshape(1, -1)
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


def _bool_to_float(v) -> float:
    """Bool → 1.0/0.0; None/non-bool → 0.0. Matches the spec for the 3
    enhanced MACD features: missing data is treated as "signal not present"
    rather than imputed positive."""
    if v is True:
        return 1.0
    if v is False:
        return 0.0
    # Be tolerant of stringified bools / 0/1 ints if a caller pre-coerces.
    try:
        return 1.0 if float(v) > 0.0 else 0.0
    except (TypeError, ValueError):
        return 0.0


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
    ema200_above: bool | None = None,
    hist_cross_up: bool | None = None,
    macd_below_zero_cross: bool | None = None,
) -> list[float]:
    """Build a fixed-length feature vector for one decision.

    Slot order matches FEATURE_NAMES exactly: 10 base numeric features,
    then the 3 enhanced MACD/EMA200 booleans, then the 7-way sector
    one-hot tail. Concretely:
    ``[ml_score, rsi, macd, mom5, mom20, regime_mult, vol_ratio, bb_pos,
       news_urgency, news_article_count,
       ema200_above, hist_cross_up, macd_below_zero_cross,
       sector_tech, sector_energy, …, sector_other]``.
    The invariant "the last len(SECTORS) entries are the sector one-hot"
    is preserved (several attribution consumers and the per-sector test
    rely on it). Existing pickles trained on the legacy 17-feature
    vector will fail to predict on the 20-feature input (shape mismatch);
    predict_with_meta catches this and degrades to failed=True / pred=0.0
    until the next retrain rebuilds the model on the new vector (the
    continuous loop's per-cycle retrain picks this up automatically)."""
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
    ema200_above_f = _bool_to_float(ema200_above)
    hist_cross_up_f = _bool_to_float(hist_cross_up)
    macd_below_zero_cross_f = _bool_to_float(macd_below_zero_cross)
    return ([_to_float(ml_score, 0.0), rsi_v, macd_v, mom5_v, mom20_v,
             _to_float(regime_mult, 1.0), vol_v, bb_v, urg_v, cnt_v,
             ema200_above_f, hist_cross_up_f, macd_below_zero_cross_f]
            + sector_oh)


class DecisionScorer:
    """Lightweight MLP: (quant_features) → predicted 5-day forward return (%)."""

    def __init__(self) -> None:
        self._model = None
        self._scaler = None
        self._trained = False
        self._n_train = 0
        # Sorted training-set prediction quantiles (101 points, percentiles
        # 0..100) persisted by train_scorer. Used by predict_percentile() to
        # map a raw prediction to its rank within the training distribution.
        # None for legacy pickles written before this field existed — every
        # consumer degrades gracefully (returns None) in that case.
        self._pred_quantiles = None
        # Sorted training-LABEL quantiles (101 points, percentiles 0..100)
        # persisted by train_scorer alongside _pred_quantiles. Used by
        # predict_calibrated() / _raw_to_calibrated() to quantile-map a raw
        # prediction onto the realized forward-return distribution — the
        # textbook fix for the documented DIRECTIONAL_BUT_BIASED verdict
        # (rank trustworthy, magnitude biased). None for legacy pickles
        # written before this field existed; every consumer degrades to None.
        self._label_quantiles = None
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
                (self._model, self._scaler, self._n_train,
                 self._pred_quantiles, self._label_quantiles) = cached
                self._trained = True
                return
            try:
                with SCORER_PATH.open("rb") as f:
                    state = pickle.load(f)
                self._model = state["model"]
                self._scaler = state.get("scaler")
                self._n_train = int(state.get("n_train", 0))
                # Additive, backward-compatible: legacy pickles have no
                # `pred_quantiles` key, so `.get` yields None and
                # predict_percentile() degrades to None for those.
                pq = state.get("pred_quantiles")
                self._pred_quantiles = (
                    np.asarray(pq, dtype=np.float64)
                    if pq is not None else None
                )
                # Additive, backward-compatible: legacy pickles have no
                # `label_quantiles` key, so `.get` yields None and
                # predict_calibrated() degrades to None for those.
                lq = state.get("label_quantiles")
                self._label_quantiles = (
                    np.asarray(lq, dtype=np.float64)
                    if lq is not None else None
                )
                self._trained = True
                # Only the current file is ever relevant; clearing bounds the
                # cache to one entry across unbounded retrain cycles.
                _LOAD_CACHE.clear()
                _LOAD_CACHE[key] = (self._model, self._scaler,
                                    self._n_train, self._pred_quantiles,
                                    self._label_quantiles)
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
        ema200_above: bool | None = None,
        hist_cross_up: bool | None = None,
        macd_below_zero_cross: bool | None = None,
    ) -> dict:
        """Predicted 5d forward return (%) plus calibration metadata.

        Returns ``{"pred", "raw", "clamped", "off_distribution",
        "percentile", "calibrated", "failed"}``:
        - ``pred``  — the value callers should act on (clamped, always finite)
        - ``raw``   — the model's unbounded output (for diagnostics / honesty)
        - ``clamped`` — True when ``|raw| > PRED_CLAMP_PCT`` (or non-finite)
        - ``off_distribution`` — True when the prediction is low-trust, either
          because the model extrapolated past the empirical label support OR
          the prediction could not be computed at all. ``off_distribution=True``
          means "do not treat ``pred`` as gospel"; the ``failed`` flag below
          tells you WHICH side of that bucket the row landed in.
        - ``failed`` — True only when the prediction could not be produced
          (untrained model, predict raised, raw was non-finite). False on any
          successful prediction, including off-distribution clamps. The OOS
          rank-IC computations consume this to drop fabricated 0.0 fallbacks
          (they would otherwise dilute the rank metric with tied zeros) WITHOUT
          dropping legitimately extreme but trustworthy clamped predictions.
        - ``percentile`` — where ``raw`` falls (0..100) within the training
          set's own prediction distribution, or None for legacy pickles that
          carry no quantile table. The OOS calibration verdict is
          ``DIRECTIONAL_BUT_BIASED`` — the predicted % magnitude is unreliable
          but the *ranking* is trustworthy — so the percentile is the
          honest, low-bias way to read a prediction.
        - ``calibrated`` — the quantile-mapped prediction: ``raw`` is mapped
          to its percentile in the training PREDICTION distribution, then
          read off at that same percentile of the training LABEL (realized
          forward-return) distribution. This is the honest *magnitude* a
          reader should believe — it preserves the model's trustworthy rank
          ordering while pulling the biased % back onto the empirical return
          support (a +42% raw extrapolation calibrates down to whatever
          realized return its rank actually corresponds to). None for legacy
          pickles that carry no ``label_quantiles`` table.

        ``predict()`` is the scalar fast path every existing consumer uses;
        this sibling exists for panels that want to surface the trust flag
        without changing ``predict()``'s float contract.
        """
        if not self._trained or self._model is None:
            return {"pred": 0.0, "raw": 0.0, "clamped": False,
                    "off_distribution": False, "percentile": None,
                    "calibrated": None, "failed": True}
        try:
            X = np.array(
                [build_features(ml_score, rsi, macd, mom5, mom20, regime_mult, ticker,
                                vol_ratio=vol_ratio, bb_pos=bb_pos,
                                news_urgency=news_urgency,
                                news_article_count=news_article_count,
                                ema200_above=ema200_above,
                                hist_cross_up=hist_cross_up,
                                macd_below_zero_cross=macd_below_zero_cross)],
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
            # A prediction that could not be computed at all (shape/dtype
            # mismatch from a build_features change without a retrain — the
            # exact case the log above guards) is the maximally-untrustworthy
            # result. Flag it low-trust like the non-finite branch does, so
            # honesty panels (/api/scorer-predictions, the conviction board)
            # never render a broken scorer's safe-fallback 0.0 as a confident
            # in-distribution call. ``failed=True`` is the specific signal OOS
            # rank-metric consumers need: a fabricated 0.0 here is NOT a real
            # prediction (the scalar predict() path returns that same 0.0 with
            # no flag, so without ``failed`` the OOS rank-IC would tie every
            # exception row at zero — fake concordance that biases the metric).
            # predict()'s scalar contract is unchanged (still 0.0).
            return {"pred": 0.0, "raw": 0.0, "clamped": True,
                    "off_distribution": True, "percentile": None,
                    "calibrated": None, "failed": True}

        # A non-finite model output (inf/nan from a pathological feature
        # vector) is unusable — treat it as a 0% / off-distribution result
        # rather than letting nan propagate silently through max/min. Same
        # ``failed=True`` discipline as the exception branch above: the 0.0
        # fallback is NOT a real prediction; flag it so OOS rank-IC consumers
        # drop the row rather than tying it at zero.
        if not np.isfinite(raw):
            return {"pred": 0.0, "raw": raw, "clamped": True,
                    "off_distribution": True, "percentile": None,
                    "calibrated": None, "failed": True}
        clamped_pred = max(-PRED_CLAMP_PCT, min(PRED_CLAMP_PCT, raw))
        was_clamped = abs(raw) > PRED_CLAMP_PCT
        # A legitimate clamped prediction (raw exceeded ±50 but was a real,
        # finite model output) is low-trust on MAGNITUDE but still has a
        # trustworthy rank — `failed=False` so OOS rank-IC retains it.
        return {"pred": clamped_pred, "raw": raw, "clamped": was_clamped,
                "off_distribution": was_clamped,
                "percentile": self._raw_to_percentile(raw),
                "calibrated": self._raw_to_calibrated(raw),
                "failed": False}

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
        ema200_above: bool | None = None,
        hist_cross_up: bool | None = None,
        macd_below_zero_cross: bool | None = None,
    ) -> float:
        """Return predicted 5d forward return (%), clamped to the empirical
        label support (see ``PRED_CLAMP_PCT``). Returns 0.0 if not trained.

        Scalar contract preserved for every existing consumer (``_ml_decide``
        gate, ``_live_scorer_predictions``, ``scorer_confidence``)."""
        return self.predict_with_meta(
            ml_score, rsi, macd, mom5, mom20, regime_mult, ticker,
            vol_ratio=vol_ratio, bb_pos=bb_pos, news_urgency=news_urgency,
            news_article_count=news_article_count,
            ema200_above=ema200_above, hist_cross_up=hist_cross_up,
            macd_below_zero_cross=macd_below_zero_cross,
        )["pred"]

    def _raw_to_percentile(self, raw: float) -> float | None:
        """Map a raw prediction to its percentile (0..100) within the
        training set's own prediction distribution.

        ``train_scorer`` persists 101 sorted quantile breakpoints of the
        model's predictions on the (deduped) training features. Linear
        interpolation between them turns any new raw prediction into a
        rank-position: 50.0 means "a median prediction for this model",
        90.0 means "in the top decile of how bullish this model ever gets".

        Returns None when the model carries no quantile table (legacy
        pickle written before this field existed), ``raw`` is non-finite,
        or the persisted ``pred_quantiles`` table is COLLAPSED to a single
        value (``q.max() == q.min()`` — the synthetic-clobber footprint
        documented in the 2026-05-23 finding #1, where a smoke-test
        retrain wrote 101 identical entries and every prediction silently
        mapped to 0 or 100 percentile via ``np.interp``'s clamp behaviour
        on a non-strictly-increasing ``xp``). The analytics-side
        ``stack_liveness.pred_collapsed`` check detects the on-disk
        pickle state externally; this guard is the internal complement so
        ``predict_percentile()`` / ``predict_calibrated()`` themselves
        degrade to a truthful ``None`` rather than silently surface
        fabricated rank values to every dashboard consumer. Never raises.
        """
        q = self._pred_quantiles
        try:
            if q is None or len(q) < 2 or not np.isfinite(raw):
                return None
            # Collapsed-quantile guard. `np.interp(x, xp, fp)` requires `xp`
            # to be monotonically increasing; on a constant `xp` it returns
            # `fp[0]` for x < xp[0], `fp[-1]` for x >= xp[0]. That maps every
            # real prediction to 0 or 100 percentile — fake rank information
            # the gate-relevant `predict_with_meta.calibrated` field then
            # interpolates through. Honest fail: return None so the OOS
            # `failed=False` legitimately-clamped path still works, but
            # rank-derived consumers see "no rank available" rather than a
            # fabricated extreme.
            q_arr = np.asarray(q, dtype=np.float64)
            if float(q_arr.max()) <= float(q_arr.min()):
                return None
            pcts = np.linspace(0.0, 100.0, len(q))
            return round(float(np.interp(float(raw), q_arr, pcts)), 2)
        except Exception:
            return None

    def _raw_to_calibrated(self, raw: float) -> float | None:
        """Quantile-map a raw prediction onto the realized-return distribution.

        Two 101-point quantile tables are persisted by ``train_scorer``:
        ``pred_quantiles`` (the model's predictions on the training set) and
        ``label_quantiles`` (the realized forward returns it was trained on).
        ``_raw_to_percentile`` locates ``raw`` within the prediction
        distribution; this method then reads the realized-return value at
        that *same* percentile of the label distribution.

        The result is the honest magnitude reading the documented
        ``DIRECTIONAL_BUT_BIASED`` verdict calls for: it is **monotonic** in
        ``raw`` (percentile is monotonic in raw, both quantile tables are
        sorted ascending) so the model's trustworthy rank ordering is fully
        preserved, while the biased % magnitude is pulled back onto the
        empirical label support. A raw ``+42%`` extrapolation calibrates down
        to whatever realized return its rank actually corresponds to.

        Returns None when either quantile table is absent (legacy pickle),
        ``raw`` is non-finite, the persisted ``label_quantiles`` table is
        COLLAPSED to a single value (``lq.max() == lq.min()``), or anything
        else fails — never raises, the same discipline as
        ``_raw_to_percentile``. The collapsed-label guard mirrors the
        collapsed-pred guard in ``_raw_to_percentile``: ``np.interp`` over a
        constant ``fp`` array would silently return that constant for every
        rank, fabricating a single magnitude for every prediction (so a
        degenerate single-label corpus surfaced as e.g. ``calibrated=7.0%``
        for an entire dashboard regardless of input rank). Returning None
        instead lets honesty-aware consumers degrade gracefully — the same
        contract a legacy pre-feature pickle already promised."""
        lq = self._label_quantiles
        try:
            if lq is None or len(lq) < 2:
                return None
            lq_arr = np.asarray(lq, dtype=np.float64)
            if float(lq_arr.max()) <= float(lq_arr.min()):
                return None
            pct = self._raw_to_percentile(raw)
            if pct is None:
                return None
            pcts = np.linspace(0.0, 100.0, len(lq_arr))
            return round(float(np.interp(float(pct), pcts, lq_arr)), 4)
        except Exception:
            return None

    def predict_calibrated(
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
        ema200_above: bool | None = None,
        hist_cross_up: bool | None = None,
        macd_below_zero_cross: bool | None = None,
    ) -> float | None:
        """Quantile-mapped reading of one prediction: the realized 5-day
        forward return (%) at the rank the raw prediction occupies.

        This is the honest *magnitude* counterpart to ``predict_percentile``:
        where the percentile answers "how bullish, in rank terms", this
        answers "what return does that rank historically correspond to".
        Use it over the raw ``predict()`` for any human-facing magnitude —
        the deployed scorer's OOS verdict is ``DIRECTIONAL_BUT_BIASED``, so a
        raw ``+8%`` is unreliable as ``+8%`` but trustworthy as a rank.

        Returns None when the model is untrained, the prediction failed /
        went non-finite, or the loaded pickle predates the ``label_quantiles``
        table (legacy compatibility). Never raises — mirrors
        ``predict_percentile`` / ``predict_with_meta``.
        """
        return self.predict_with_meta(
            ml_score, rsi, macd, mom5, mom20, regime_mult, ticker,
            vol_ratio=vol_ratio, bb_pos=bb_pos, news_urgency=news_urgency,
            news_article_count=news_article_count,
            ema200_above=ema200_above, hist_cross_up=hist_cross_up,
            macd_below_zero_cross=macd_below_zero_cross,
        )["calibrated"]

    def predict_percentile(
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
        ema200_above: bool | None = None,
        hist_cross_up: bool | None = None,
        macd_below_zero_cross: bool | None = None,
    ) -> float | None:
        """Rank-calibrated reading of one prediction: the percentile (0..100)
        the raw forward-return estimate occupies in the training-set
        prediction distribution.

        Motivation: the deployed scorer's OOS calibration verdict is
        ``DIRECTIONAL_BUT_BIASED`` — the predicted % is biased (decile error
        ~7.5pp) but the ordering carries real skill (rank-IC ≈ 0.22). A raw
        ``+8%`` should NOT be read as "+8%"; reading it as "82nd percentile"
        is the honest, low-bias signal.

        Returns None when the model is untrained, the prediction failed /
        went non-finite, or the loaded pickle predates the quantile table
        (legacy compatibility). Never raises — mirrors ``predict_with_meta``.
        """
        return self.predict_with_meta(
            ml_score, rsi, macd, mom5, mom20, regime_mult, ticker,
            vol_ratio=vol_ratio, bb_pos=bb_pos, news_urgency=news_urgency,
            news_article_count=news_article_count,
            ema200_above=ema200_above, hist_cross_up=hist_cross_up,
            macd_below_zero_cross=macd_below_zero_cross,
        )["percentile"]

    def feature_contributions(
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
        ema200_above: bool | None = None,
        hist_cross_up: bool | None = None,
        macd_below_zero_cross: bool | None = None,
    ) -> dict:
        """Per-feature signed attribution for one prediction.

        Answers "WHY does the scorer predict this?" — the model is otherwise a
        black box and an operator can't tell whether a -50 EXIT verdict is
        driven by RSI, momentum, news, or a sector bias.

        Ablation-from-baseline: the scaler centres every feature on its
        training mean, so the all-zeros standardized vector is the "neutral"
        input. ``contribution[i]`` is how far moving feature *i alone* from
        that neutral baseline to its live value moves the raw prediction.
        ``interaction_residual = pred_full - pred_baseline - sum(contrib)``
        is ~0 for the linear fallback (exactly additive) and captures genuine
        MLP non-linearity / feature interactions for the sklearn model — it is
        surfaced rather than hidden so the attribution stays honest.

        Returns a JSON-safe dict; ``trained=False`` (no model) yields an empty
        ``contributions`` list. Every failure degrades to an ``error`` field,
        never an exception (mirrors ``predict_with_meta``)."""
        if not self._trained or self._model is None:
            return {"trained": False, "contributions": [], "pred": 0.0,
                    "pred_baseline": 0.0, "interaction_residual": 0.0,
                    "off_distribution": False}
        try:
            raw_feat = build_features(
                ml_score, rsi, macd, mom5, mom20, regime_mult, ticker,
                vol_ratio=vol_ratio, bb_pos=bb_pos, news_urgency=news_urgency,
                news_article_count=news_article_count,
                ema200_above=ema200_above, hist_cross_up=hist_cross_up,
                macd_below_zero_cross=macd_below_zero_cross,
            )
            x = np.array([raw_feat], dtype=np.float32)
            if self._scaler is not None:
                xs = np.asarray(self._scaler.transform(x), dtype=np.float32)[0]
            else:
                xs = x[0]
            n = xs.shape[0]
            baseline = np.zeros(n, dtype=np.float32)
            # Batch one predict: [full, baseline, then feature-i-only-active].
            batch = np.vstack([xs, baseline,
                               baseline + np.eye(n, dtype=np.float32) * xs])
            out = np.asarray(self._model.predict(batch), dtype=np.float64)
            if not np.all(np.isfinite(out)):
                return {"trained": True, "contributions": [], "pred": 0.0,
                        "pred_baseline": 0.0, "interaction_residual": 0.0,
                        "off_distribution": True}
            pred_full = float(out[0])
            pred_base = float(out[1])
            contribs = out[2:] - pred_base  # marginal effect of each feature
            rows = [
                {"feature": FEATURE_NAMES[i],
                 "raw_value": round(float(raw_feat[i]), 4),
                 "contribution": round(float(contribs[i]), 4)}
                for i in range(n)
            ]
            rows.sort(key=lambda r: -abs(r["contribution"]))
            residual = pred_full - pred_base - float(contribs.sum())
            clamped = abs(pred_full) > PRED_CLAMP_PCT
            return {
                "trained": True,
                "pred": round(max(-PRED_CLAMP_PCT,
                                  min(PRED_CLAMP_PCT, pred_full)), 4),
                "pred_raw": round(pred_full, 4),
                "pred_baseline": round(pred_base, 4),
                "interaction_residual": round(residual, 4),
                "off_distribution": bool(clamped),
                "contributions": rows,
            }
        except Exception as e:
            return {"trained": True, "contributions": [], "pred": 0.0,
                    "pred_baseline": 0.0, "interaction_residual": 0.0,
                    "off_distribution": True, "error": str(e)}

    def feature_group_contributions(
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
        ema200_above: bool | None = None,
        hist_cross_up: bool | None = None,
        macd_below_zero_cross: bool | None = None,
    ) -> dict:
        """Per-group signed attribution for one prediction.

        Thin operator-facing roll-up of ``feature_contributions``: sums the
        17 per-feature contributions into the 5 ``FEATURE_GROUPS`` buckets
        (``ml_score``, ``quant``, ``regime``, ``news``, ``sector``). The
        per-feature view answers "which input is driving this prediction?";
        this answers the orthogonal triage question "is the scorer leaning
        on news, on quant signals, on the trade thesis, or on the sector
        bias?" — at a glance, without scanning a 17-row table.

        Algebraic identity (sum-of-groups = sum-of-features):

            sum(group.contribution for group in groups)
                == sum(feature.contribution for feature in contributions)
                == pred_raw - pred_baseline - interaction_residual

        Returns a JSON-safe dict ``{"trained", "groups", "pred",
        "pred_baseline", "interaction_residual", "off_distribution"}``.
        Each ``groups[i]`` carries ``{"group", "contribution",
        "share_pct", "n_features"}`` where ``share_pct`` is
        ``|contribution| / sum(|contribution|)`` as a 0..100 share (so
        groups whose contributions cancel internally show ``|net|/|all|``
        — the operator-meaningful "how much net signal does this group
        contribute" rather than the misleading "0%" a literal share would
        report). ``share_pct`` sums to 100.0 across all groups when the
        per-group denominator is non-zero, and to 0.0 in the degenerate
        all-zero case (e.g. an untrained or zeroed-contribution row).

        ``groups`` is sorted by ``-|contribution|`` so the dominant group
        always comes first — same display ordering as
        ``feature_contributions``. ``trained=False`` (no model) yields an
        empty groups list, mirroring the parent method. Every failure path
        degrades to an ``error`` field, never an exception."""
        contrib = self.feature_contributions(
            ml_score=ml_score, rsi=rsi, macd=macd, mom5=mom5, mom20=mom20,
            regime_mult=regime_mult, ticker=ticker, vol_ratio=vol_ratio,
            bb_pos=bb_pos, news_urgency=news_urgency,
            news_article_count=news_article_count,
            ema200_above=ema200_above, hist_cross_up=hist_cross_up,
            macd_below_zero_cross=macd_below_zero_cross,
        )
        if not contrib.get("trained"):
            return {"trained": False, "groups": [], "pred": 0.0,
                    "pred_baseline": 0.0, "interaction_residual": 0.0,
                    "off_distribution": False}
        if not contrib.get("contributions"):
            # Either off-distribution (model raised / non-finite) or some
            # other failure inside feature_contributions; propagate the
            # honest signal up (no contributions means no groups).
            out = {"trained": True, "groups": [], "pred": contrib.get("pred", 0.0),
                   "pred_baseline": contrib.get("pred_baseline", 0.0),
                   "interaction_residual": contrib.get("interaction_residual", 0.0),
                   "off_distribution": bool(contrib.get("off_distribution"))}
            if contrib.get("error"):
                out["error"] = contrib["error"]
            return out
        try:
            # Defense-in-depth: an unknown feature name in the contributions
            # list (would only happen if FEATURE_NAMES drifts at runtime
            # without FEATURE_GROUP_MAP updating, despite the module-level
            # assert) lands in a synthetic 'unknown' bucket so the group
            # totals still add correctly rather than silently losing rows.
            sums: dict[str, float] = {g: 0.0 for g in FEATURE_GROUPS}
            counts: dict[str, int] = {g: 0 for g in FEATURE_GROUPS}
            for row in contrib["contributions"]:
                feat = row["feature"]
                grp = FEATURE_GROUP_MAP.get(feat, "unknown")
                if grp not in sums:
                    sums[grp] = 0.0
                    counts[grp] = 0
                sums[grp] += float(row["contribution"])
                counts[grp] += 1
            denom = sum(abs(v) for v in sums.values())
            rows: list[dict] = []
            for grp in (*FEATURE_GROUPS, "unknown"):
                if grp not in sums:
                    continue
                # Drop the synthetic 'unknown' bucket if nothing landed
                # there — keeps the operator-facing payload clean in the
                # 99.99% case where FEATURE_GROUP_MAP is complete.
                if grp == "unknown" and counts[grp] == 0:
                    continue
                share = (abs(sums[grp]) / denom * 100.0) if denom > 0 else 0.0
                rows.append({
                    "group": grp,
                    "contribution": round(sums[grp], 4),
                    "share_pct": round(share, 2),
                    "n_features": counts[grp],
                })
            rows.sort(key=lambda r: -abs(r["contribution"]))
            return {
                "trained": True,
                "pred": contrib.get("pred", 0.0),
                "pred_baseline": contrib.get("pred_baseline", 0.0),
                "interaction_residual": contrib.get("interaction_residual", 0.0),
                "off_distribution": bool(contrib.get("off_distribution")),
                "groups": rows,
            }
        except Exception as e:
            return {"trained": True, "groups": [], "pred": 0.0,
                    "pred_baseline": 0.0, "interaction_residual": 0.0,
                    "off_distribution": True, "error": str(e)}

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def n_train(self) -> int:
        return self._n_train

    def feature_importance(self) -> dict:
        """Global feature importance derived from the trained model's
        input-layer weights — a model-wide answer to "which of the 17
        features does the scorer actually rely on?", complementing
        feature_contributions() which only answers it for one input.

        For the sklearn MLPRegressor, importance is mean(|first-layer weight|)
        per input neuron — a standard "this input drives many hidden units"
        proxy. For the numpy lstsq fallback (a single linear layer), it is
        |coefficient| directly. Both metrics live in the *standardized*
        feature space the scaler produces, so cross-feature comparison is
        meaningful even though raw units differ (RSI 0..100 vs sector 0/1).

        Returns a JSON-safe dict ``{"trained", "method", "n_train",
        "importances": [{"feature", "importance", "importance_normalized"}]}``
        sorted desc by raw importance. ``importance_normalized`` sums to 1.0
        across all features so a reader can read it as a share. Every failure
        path degrades to an ``error`` field, never an exception (mirrors
        predict_with_meta / feature_contributions)."""
        if not self._trained or self._model is None:
            return {"trained": False, "method": None, "n_train": 0,
                    "importances": []}
        try:
            if hasattr(self._model, "coefs_") and self._model.coefs_:
                W = np.asarray(self._model.coefs_[0], dtype=np.float64)
                raw = np.abs(W).mean(axis=1)
                method = "mlp_first_layer_mean_abs_weight"
            elif hasattr(self._model, "w_"):
                # _LstsqModel: w_ is shape (n_features + 1,); last entry is bias.
                w = np.asarray(self._model.w_, dtype=np.float64)
                if w.shape[0] < N_FEATURES:
                    return {"trained": True, "method": "lstsq_abs_weight",
                            "n_train": int(self._n_train), "importances": [],
                            "error": f"weights len {w.shape[0]} < "
                                     f"N_FEATURES {N_FEATURES}"}
                raw = np.abs(w[:N_FEATURES])
                method = "lstsq_abs_weight"
            else:
                return {"trained": True, "method": None,
                        "n_train": int(self._n_train), "importances": [],
                        "error": "unknown model type"}
            if raw.size != N_FEATURES:
                return {"trained": True, "method": method,
                        "n_train": int(self._n_train), "importances": [],
                        "error": f"importance len {raw.size} != "
                                 f"N_FEATURES {N_FEATURES}"}
            # Non-finite weights (NaN/Inf from a degenerate lstsq solve on a
            # rank-deficient design, or pathological MLP coefs from training
            # on contaminated inputs) would otherwise poison everything
            # downstream: `raw.sum()` is NaN, `norm = raw/total` is NaN,
            # `json.dumps` then either emits non-standard "NaN" or raises,
            # and `sort(key=raw)` becomes order-dependent (NaN compares
            # unpredictably). Replace with 0.0 so the feature shows as
            # zero importance but the JSON contract — sortable, sums to 1.0,
            # all finite — holds. Mirrors the `_to_float`/`predict_with_meta`
            # discipline: never raise on a degenerate input; degrade to a
            # safe sentinel and let downstream consumers see honest zeros.
            raw = np.where(np.isfinite(raw), raw, 0.0)
            total = float(raw.sum())
            norm = (raw / total) if total > 0 else raw
            rows = [
                {"feature": FEATURE_NAMES[i],
                 "importance": round(float(raw[i]), 6),
                 "importance_normalized": round(float(norm[i]), 6)}
                for i in range(N_FEATURES)
            ]
            rows.sort(key=lambda r: -r["importance"])
            return {"trained": True, "method": method,
                    "n_train": int(self._n_train), "importances": rows}
        except Exception as e:
            return {"trained": True, "method": None,
                    "n_train": int(self._n_train), "importances": [],
                    "error": str(e)}


def train_scorer(records: list[dict], path: "Path | None" = None) -> dict:
    """Train DecisionScorer on outcome records.

    Each record must have: ml_score, rsi, macd, mom5, mom20, regime_mult, ticker,
    forward_return_5d. Optional: action (BUY/SELL — SELL flips target sign so the
    model learns "goodness of THIS action"), return_pct (overall backtest run
    quality, used to weight samples). Returns stats dict.

    ``path`` overrides the deployed ``SCORER_PATH`` for one call. Default (None)
    pickles to the deployed path — preserves the existing contract every cycle
    of the continuous loop and every test that monkey-patched the module-level
    constant. The override exists so diagnostic analyzers (the learning-curve
    sweep) can train into a throwaway temp file without trampling the live
    pickle the conviction gate reads; the same atomic tmp+``.replace`` idiom
    is used regardless of which path is targeted.
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
    #
    # Tie-break precedence is `(has_valid_label, return_pct)` rather than
    # `return_pct` alone: an externally injected / corrupted row whose
    # `forward_return_5d` is null/NaN/inf would otherwise WIN dedup if its
    # run's `return_pct` was highest, and then the validation loop below would
    # drop it for a missing label — silently throwing away the surviving real
    # outcomes for that (ticker, sim_date, action) key. Production data has
    # 0 such rows today (`_compute_decision_outcomes` only emits walk-back
    # validated finite outcomes), but the guard makes the "valid label always
    # beats missing label" invariant explicit and refactor-proof rather than
    # emergent. Boolean-as-int comparison: True > False, so a valid-labelled
    # record always wins regardless of run return_pct.
    def _has_valid_label(rec: dict) -> bool:
        v = rec.get("forward_return_5d")
        if isinstance(v, bool) or v is None:
            return False
        try:
            return math.isfinite(float(v))
        except (TypeError, ValueError):
            return False

    seen: dict[tuple, dict] = {}
    for r in records:
        key = (
            str(r.get("ticker") or ""),
            str(r.get("sim_date") or ""),
            str(r.get("action") or "BUY").upper(),
        )
        cand_prec = (_has_valid_label(r), _to_float(r.get("return_pct"), 0.0))
        if key not in seen:
            seen[key] = r
            continue
        cur = seen[key]
        cur_prec = (_has_valid_label(cur),
                    _to_float(cur.get("return_pct"), 0.0))
        if cand_prec > cur_prec:
            seen[key] = r
    records = list(seen.values())

    # Length gate applied AFTER dedup — dedup can shrink the set well below the
    # raw input count, and the model needs a real minimum of distinct samples.
    if len(records) < 30:
        return {"status": "insufficient_after_dedup", "n": len(records)}

    X_raw, y, weights = [], [], []
    n_label_clamped = 0
    n_label_dropped = 0
    for r in records:
        # Drop rows whose forward_return_5d is missing / non-finite BEFORE
        # building features. The prior `_to_float(..., 0.0)` silently
        # coerced None / NaN / inf into a 0.0 (a fake flat-return label),
        # which contaminated training with neutral-label phantom rows.
        # `_compute_decision_outcomes` already filters genuine missing
        # outcomes; this is defence-in-depth for any externally injected /
        # malformed record (a single bad row otherwise distorts the entire
        # 5000-record retrain). Drops are counted so the per-cycle skill
        # ledger can trend training-set integrity.
        fr_raw = r.get("forward_return_5d")
        if isinstance(fr_raw, bool) or fr_raw is None:
            n_label_dropped += 1
            continue
        try:
            fr = float(fr_raw)
        except (TypeError, ValueError):
            n_label_dropped += 1
            continue
        if not math.isfinite(fr):
            n_label_dropped += 1
            continue
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
            ema200_above=r.get("ema200_above"),
            hist_cross_up=r.get("hist_cross_up"),
            macd_below_zero_cross=r.get("macd_below_zero_cross"),
        ))
        # Symmetric label clamp — mirror the inference-side ±PRED_CLAMP_PCT
        # clamp at train time. predict() always clamps its output to
        # ±PRED_CLAMP_PCT, so a training label like MSTR +175% (the live
        # corpus carries 2 such rows in the 5000-record trainer tail; see
        # data audit) can never be predicted — yet it still drives huge MSE
        # gradients during fit, pulling weights toward magnitudes the gate
        # can never act on and perturbing the entire feature subspace those
        # outliers inhabit. Aligning the label space with the prediction
        # space removes this outlier-induced training noise. Applied BEFORE
        # the SELL sign-flip so the clamp bound is identical on either side
        # — abs() is symmetric. Only ~0.5% of the live trainer tail has
        # |fr|>50% (25/5000), so impact is concentrated on the genuine
        # outliers without touching the heart of the distribution. The
        # ±50 boundary is the same constant predict() / predict_with_meta
        # already enforces, so the alignment is verifiable.
        if abs(fr) > PRED_CLAMP_PCT:
            n_label_clamped += 1
        fr = max(-PRED_CLAMP_PCT, min(PRED_CLAMP_PCT, fr))
        action = str(r.get("action") or "BUY").upper()
        # SELL: negative forward returns were the *correct* outcome, so flip
        # sign — the model then learns one consistent meaning of "good".
        y.append(-fr if action == "SELL" else fr)
        # Sample weight from overall run quality:
        # +200% run → 2.0×, 0% → 1.0×, -100%+ → 0.5×.
        rp = _to_float(r.get("return_pct"), 0.0)
        # LLM annotation multiplier: endorsed → 3x signal, condemned → 0.1x.
        # Defensive parse: a corrupted record carrying a string label
        # ("ENDORSE") or any non-int (list / dict) used to crash `int(...)`
        # with ValueError, which the outer caller (`_train_decision_scorer`)
        # then surfaced as `scorer err: invalid literal for int()` — wedging
        # the per-cycle retrain (CLAUDE.md §6) and freezing the conviction
        # gate (#5) until the bad row was manually purged. Mirror the
        # `forward_return_5d` validation discipline above: a single bad
        # value silently degrades to "no label" (weight ×1.0) rather than
        # poisoning the whole batch.
        _raw_label = r.get("llm_quality_label")
        if isinstance(_raw_label, bool) or _raw_label is None:
            llm_label = 0
        else:
            try:
                llm_label = int(_raw_label)
            except (TypeError, ValueError):
                llm_label = 0
        llm_mult = {1: 3.0, -1: 0.1, 0: 1.0}.get(llm_label, 1.0)
        weights.append(max(0.5, min(2.0, 1.0 + rp / 200.0)) * llm_mult)

    # Honest empty-after-validation guard. If every record dropped (e.g. all
    # rows carried `forward_return_5d=null` from a malformed outcomes batch),
    # building np.array on []/[] is itself fine, but train_test_split below
    # would raise on n=0. Return a status the caller can surface to the skill
    # ledger rather than letting that exception poison the cycle.
    if not X_raw:
        return {"status": "no_valid_labels", "n": 0,
                "n_label_dropped": n_label_dropped}

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
        #
        # Rounding produces rep=0 for very low weights — the documented LLM
        # `CONDEMN` annotation (`llm_mult=0.1`) at base weight 0.5 gives
        # 0.05 → ×2=0.1 → round to 0. Such rows are DROPPED from the
        # training fold so the documented 0.1× weight is actually realized.
        # The prior `np.maximum(1, …)` floor promoted CONDEMN to rep=1,
        # rendering the 0.1× multiplier indistinguishable from a 0.5× weight
        # on a losing run (measured CONDEMN/unlabeled ≈ 0.5×, NOT the
        # documented 0.1×). Keeping the ×2 scaling (not ×10) preserves the
        # original training-fold size so the unweighted L2 `alpha` term
        # stays comparable across samples (a 5× larger fold would weaken
        # regularization 5×, breaking the anti-overfit guarantees the
        # noise-memorization test pins). Defensive empty-fold guard: if
        # every weight rounds to 0 (impossible in any real corpus, since
        # unlabeled records always weight ≥0.5 → rep≥1), fall back to the
        # raw training fold so MLPRegressor.fit never sees an empty array.
        rep = np.round(w_tr * 2).astype(int)
        _keep = rep > 0
        if _keep.any():
            X_tr_w = np.repeat(X_tr[_keep], rep[_keep], axis=0)
            y_tr_w = np.repeat(y_tr[_keep], rep[_keep], axis=0)
        else:
            X_tr_w = X_tr
            y_tr_w = y_tr

        # Anti-overfit config (2026-05-18). The prior unregularized
        # (64,32,16)/600-iter net memorised the noisy training fold:
        # measured on the live 5000-outcome tail (temporal 80/20 holdout,
        # `_train_decision_scorer`'s honest split) it posted val_rmse≈10.7
        # but oos_rmse≈16.7 — the textbook overfit the per-cycle
        # scorer-skill ledger records every cycle (val_rmse≪oos_rmse). A
        # smaller (32,16) net + L2 `alpha` + `early_stopping` shrinks that
        # gap hard: across 4 MLP seeds OOS RMSE drops uniformly (mean
        # ≈14.97→≈12.58, up to 16.7→10.5 on the worst prior seed) and the
        # val/oos gap closes from ~6pp to <1pp, while OOS rank-IC and
        # directional accuracy stay within ±0.04 / coin-flip noise — i.e.
        # this removes magnitude overfit WITHOUT touching the (unchanged,
        # data-limited) MLP_NO_BETTER_THAN_TRIVIAL rank-skill finding. The
        # `_ml_decide` conviction gate acts on the prediction's MAGNITUDE
        # bucket (±10/±5/0), so a uniformly lower-error, less-extrapolating
        # head makes those bucket assignments materially less noisy. Gate
        # arms, the ±PRED_CLAMP_PCT output clamp, build_features, SECTORS,
        # N_FEATURES and the {model,scaler,n_train} pickle schema are all
        # untouched — a drop-in the next retrain cycle picks up. Realigns
        # the code with CLAUDE.md §3's long-documented "MLPRegressor 32→16"
        # architecture (the code had silently drifted to (64,32,16)). The
        # numpy-lstsq fallback (sklearn-absent hosts) is unaffected.
        # Hyper-parameters live in the module-level MLP_CONFIG so the
        # deploy_audit diagnostic can compare a deployed pickle's fitted
        # model against the exact values used here (single source of truth).
        model = MLPRegressor(**MLP_CONFIG)
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

    out_path = path if path is not None else SCORER_PATH
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: a torn pickle (process killed mid-write, or a backtest
    # thread loading the file concurrently) would leave DecisionScorer
    # permanently untrained. Write to a temp file then atomically replace.
    # Pickle n_train as the ACTUAL count of rows that produced training
    # features (post forward_return validation), not the pre-validation
    # `len(records)` count, so the gate-relevant n_train >= 500 invariant
    # (#5) reflects the model's true exposure to data.
    n_pickle = len(X_raw)

    # Rank-calibration table: 101 sorted quantile breakpoints (percentiles
    # 0..100) of the model's predictions on the full deduped training set.
    # predict_percentile() interpolates into this to turn a raw — and, per
    # the DIRECTIONAL_BUT_BIASED OOS verdict, magnitude-unreliable —
    # prediction into a trustworthy rank position. Best-effort: any failure
    # leaves `pred_quantiles=None` so training (and the gate) is never
    # blocked by a diagnostic-only artifact, and a legacy-style pickle
    # results (predict_percentile then degrades to None).
    pred_quantiles = None
    try:
        train_pred = model.predict(scaler.transform(X))
        train_pred = np.asarray(train_pred, dtype=np.float64)
        train_pred = train_pred[np.isfinite(train_pred)]
        if train_pred.size >= 2:
            pred_quantiles = [
                float(v) for v in
                np.percentile(train_pred, np.arange(0, 101))
            ]
    except Exception as e:
        print(f"[decision_scorer] pred-quantile build failed: {e}")
        pred_quantiles = None

    # Label-calibration table: 101 sorted quantile breakpoints of the
    # realized training LABELS (`y` — clamped, SELL-sign-flipped forward
    # returns). predict_calibrated() / _raw_to_calibrated() pair this with
    # `pred_quantiles` to quantile-map a raw prediction onto the empirical
    # return distribution — the honest magnitude reading the documented
    # DIRECTIONAL_BUT_BIASED verdict calls for. Same best-effort discipline
    # as pred_quantiles: any failure leaves it None and a legacy-style
    # pickle results (predict_calibrated then degrades to None). `y` is the
    # exact label space the model was fit against (post clamp + SELL flip),
    # so the calibration is apples-to-apples with the prediction space.
    label_quantiles = None
    try:
        y_finite = y[np.isfinite(y)]
        if y_finite.size >= 2:
            label_quantiles = [
                float(v) for v in
                np.percentile(y_finite, np.arange(0, 101))
            ]
    except Exception as e:
        print(f"[decision_scorer] label-quantile build failed: {e}")
        label_quantiles = None

    _tmp = out_path.with_suffix(".pkl.tmp")
    with _tmp.open("wb") as f:
        pickle.dump({"model": model, "scaler": scaler, "n_train": n_pickle,
                     "pred_quantiles": pred_quantiles,
                     "label_quantiles": label_quantiles}, f)
    _tmp.replace(out_path)

    return {"status": "ok", "n": n_pickle, "val_rmse": val_rmse,
            "n_label_clamped": n_label_clamped,
            "n_label_dropped": n_label_dropped,
            "n_pred_quantiles": len(pred_quantiles) if pred_quantiles else 0,
            "n_label_quantiles": len(label_quantiles) if label_quantiles else 0}


# ---------------------------------------------------------------------------
# CLI: `python3 -m paper_trader.ml.decision_scorer --explain --ticker NVDA ...`
#
# Until now the only way to ask the scorer "what would you predict for THIS
# input, and WHY" was the Flask dashboard (/api/scorer-predictions,
# /api/scorer-attribution). On a box where the 78%-NO_DECISION operational
# reality means an operator is usually on a shell triaging, not in a browser,
# there was no way to interrogate the model directly. This exposes the already
# existing honesty machinery — predict_with_meta() (clamp / off-distribution
# trust flags) and feature_contributions() (per-feature signed attribution) —
# as a read-only command. It adds NO new model behaviour, touches NO existing
# function, imports stdlib lazily inside the entrypoints (module import cost
# stays zero, mirroring train_scorer's local sklearn import), and never
# trains or writes the pickle. Pattern (int return + --json + SystemExit)
# matches paper_trader/host_guard.py's CLI so operators get one muscle memory.
# ---------------------------------------------------------------------------

def _build_arg_parser():
    import argparse

    p = argparse.ArgumentParser(
        prog="python3 -m paper_trader.ml.decision_scorer",
        description="Explain one DecisionScorer prediction: predicted 5-day "
                    "forward return (%) plus per-feature signed attribution. "
                    "Loads the pickled model at data/ml/decision_scorer.pkl; "
                    "read-only — never trains or writes.",
    )
    p.add_argument("--explain", action="store_true",
                   help="(Accepted for clarity; the default mode.)")
    p.add_argument("--feature-importance", action="store_true",
                   dest="feature_importance",
                   help="Global mode: print which features the trained model "
                        "relies on most (mean-|first-layer-weight| for the "
                        "MLP, |coef| for the lstsq fallback). Ignores all "
                        "per-prediction args.")
    p.add_argument("--groups", action="store_true", dest="groups",
                   help="In per-prediction mode, also print the per-group "
                        "(ml_score / quant / regime / news / sector) "
                        "attribution roll-up — operator triage view of "
                        "what's driving this prediction.")
    p.add_argument("--ticker", default="",
                   help="Ticker symbol — drives the 7-way sector one-hot.")
    p.add_argument("--ml-score", type=float, default=0.0, dest="ml_score",
                   help="ArticleNet ai_score for the name (default 0).")
    p.add_argument("--rsi", type=float, default=None)
    p.add_argument("--macd", type=float, default=None)
    p.add_argument("--mom5", type=float, default=None,
                   help="5-day momentum %%.")
    p.add_argument("--mom20", type=float, default=None,
                   help="20-day momentum %%.")
    p.add_argument("--regime-mult", type=float, default=1.0,
                   dest="regime_mult", help="Market-regime multiplier.")
    p.add_argument("--vol-ratio", type=float, default=None, dest="vol_ratio")
    p.add_argument("--bb-pos", type=float, default=None, dest="bb_pos",
                   help="Bollinger-band position 0..1.")
    p.add_argument("--news-urgency", type=float, default=None,
                   dest="news_urgency")
    p.add_argument("--news-article-count", type=float, default=None,
                   dest="news_article_count")
    # Enhanced MACD / EMA200 features. These are real inputs to
    # ``build_features`` (slots 10/11/12 — see ``FEATURE_NAMES``) and they
    # carry non-zero learned weights in the deployed pickle (pass #35 closed
    # the training-side dead-trained gap). Until they were wired here the
    # CLI silently defaulted them to None → 0.0 via ``_bool_to_float``: the
    # explainer claimed to attribute a prediction but actually predicted
    # against a *different* feature vector than the live gate sees for any
    # name where these signals fire — exactly the operator-blind class of
    # bug the pass-#35 inference plumb closed for ``_ml_decide``. Three
    # store_true flags (no value needed — they are booleans by contract);
    # absence ⇒ None ⇒ ``_bool_to_float`` falls back to 0.0 (the documented
    # "signal not present" sentinel, unchanged from prior CLI behaviour).
    p.add_argument("--ema200-above", action="store_true",
                   dest="ema200_above", default=None,
                   help="EMA200 long-term trend filter — pass when "
                        "close > EMA200 (textbook MACD setup filter).")
    p.add_argument("--hist-cross-up", action="store_true",
                   dest="hist_cross_up", default=None,
                   help="MACD histogram crossed up through zero on the "
                        "latest bar (bullish trigger).")
    p.add_argument("--macd-below-zero-cross", action="store_true",
                   dest="macd_below_zero_cross", default=None,
                   help="MACD line below zero AND hist_cross_up — "
                        "classic oversold-bottom MACD setup.")
    p.add_argument("--json", action="store_true",
                   help="Emit machine-readable JSON instead of a table.")
    return p


def main(argv: list[str] | None = None) -> int:
    """Explain a single prediction. Returns 0 when a trusted prediction was
    produced, 1 when the model is untrained or the result is off-distribution
    (low-trust) — so shell callers can gate on `$?` like host_guard's CLI."""
    import sys

    args = _build_arg_parser().parse_args(
        sys.argv[1:] if argv is None else argv)

    scorer = DecisionScorer()

    if args.feature_importance:
        imp = scorer.feature_importance()
        if args.json:
            import json

            print(json.dumps(imp, indent=2, sort_keys=True))
            return 0 if imp.get("trained") and not imp.get("error") else 1
        if not imp.get("trained"):
            print("[decision_scorer] model NOT trained — no pickle at "
                  f"{SCORER_PATH} or load failed.")
            return 1
        if imp.get("error"):
            print(f"[decision_scorer] feature importance unavailable: "
                  f"{imp['error']}")
            return 1
        print(f"[decision_scorer] feature importance  "
              f"n_train={imp.get('n_train', 0)}  method={imp.get('method')}")
        print(f"  {'feature':<22}{'importance':>14}{'share':>10}")
        for r in imp.get("importances") or []:
            print(f"  {r['feature']:<22}{r['importance']:>14.6f}"
                  f"{r['importance_normalized']*100:>9.2f}%")
        return 0

    common = dict(
        ml_score=args.ml_score, rsi=args.rsi, macd=args.macd,
        mom5=args.mom5, mom20=args.mom20, regime_mult=args.regime_mult,
        ticker=args.ticker, vol_ratio=args.vol_ratio, bb_pos=args.bb_pos,
        news_urgency=args.news_urgency,
        news_article_count=args.news_article_count,
        # Enhanced MACD / EMA200 features — see ``_build_arg_parser`` for
        # the gap this closes. ``None`` (absent flag) plumbs to
        # ``_bool_to_float``'s 0.0 default, preserving prior CLI behaviour
        # for names where these signals aren't being passed.
        ema200_above=args.ema200_above,
        hist_cross_up=args.hist_cross_up,
        macd_below_zero_cross=args.macd_below_zero_cross,
    )
    meta = scorer.predict_with_meta(**common)
    contrib = scorer.feature_contributions(**common)
    groups = scorer.feature_group_contributions(**common) if args.groups else None

    if args.json:
        import json

        payload = {
            "trained": scorer.is_trained,
            "n_train": scorer.n_train,
            "ticker": args.ticker,
            "prediction": meta,
            "attribution": contrib,
        }
        if groups is not None:
            payload["group_attribution"] = groups
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if (scorer.is_trained
                     and not meta.get("off_distribution")) else 1

    if not scorer.is_trained:
        print("[decision_scorer] model NOT trained — no pickle at "
              f"{SCORER_PATH} or load failed. predict() is a no-op 0.0%; "
              "accumulate >=30 deduped outcomes then retrain.")
        return 1

    off = bool(meta.get("off_distribution"))
    flag = "  [OFF-DISTRIBUTION — extrapolated past label support, low trust]" \
        if off else ""
    print(f"[decision_scorer] {args.ticker or '(no ticker)'}  "
          f"n_train={scorer.n_train}")
    print(f"  predicted 5d forward return: {meta['pred']:+.2f}%  "
          f"(raw {meta['raw']:+.2f}%){flag}")
    _pct = meta.get("percentile")
    if _pct is not None:
        print(f"  rank percentile: {_pct:.1f}  (position in the training "
              f"prediction distribution — trust this over the raw %, the "
              f"OOS verdict is DIRECTIONAL_BUT_BIASED)")
    _cal = meta.get("calibrated")
    if _cal is not None:
        print(f"  calibrated 5d return: {_cal:+.2f}%  (raw quantile-mapped "
              f"onto the realized-return distribution — the honest magnitude; "
              f"preserves rank, corrects the documented % bias)")
    rows = contrib.get("contributions") or []
    if rows:
        print(f"  baseline={contrib.get('pred_baseline', 0.0):+.2f}%   "
              "interaction_residual="
              f"{contrib.get('interaction_residual', 0.0):+.2f}%")
        print("  per-feature attribution (sorted by |impact|):")
        print(f"    {'feature':<22}{'value':>12}{'contribution':>16}")
        for r in rows:
            print(f"    {r['feature']:<22}{r['raw_value']:>12.4f}"
                  f"{r['contribution']:>+16.4f}")
    elif contrib.get("error"):
        print(f"  attribution unavailable: {contrib['error']}")
    if groups is not None and groups.get("groups"):
        print("  per-group attribution (operator triage):")
        print(f"    {'group':<12}{'contribution':>14}{'share':>10}{'n_feats':>10}")
        for g in groups["groups"]:
            print(f"    {g['group']:<12}{g['contribution']:>+14.4f}"
                  f"{g['share_pct']:>9.2f}%{g['n_features']:>10}")
    elif groups is not None and groups.get("error"):
        print(f"  group attribution unavailable: {groups['error']}")
    return 1 if off else 0


if __name__ == "__main__":
    raise SystemExit(main())
