"""
ArticleNet — sklearn MLPRegressor for financial article scoring.

Two models (one per output head):
  - relevance_model: predicts relevance score 0–10
  - urgency_model:   predicts urgency score 0–10

Uses sklearn MLPRegressor with warm_start for incremental training.
Persisted as joblib files in data/ml/.

Inference: sklearn predict → numpy → fast (~0.1ms/article).
Uncertainty: bootstrap ensemble of 5 sub-models, measure spread.

No AVX512 required — runs on any x86_64 with AVX2.
"""
import os
import pickle
import time
import warnings
import numpy as np
from pathlib import Path

try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.exceptions import ConvergenceWarning
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    SKLEARN_OK = True
except ImportError:
    SKLEARN_OK = False

MODEL_DIR = Path(os.environ.get("DIGITAL_INTERN_ML_DIR",
                                Path(__file__).resolve().parent.parent / "data" / "ml"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

REL_MODEL_PATH = MODEL_DIR / "relevance_model.pkl"
URG_MODEL_PATH = MODEL_DIR / "urgency_model.pkl"

# Ensemble size for uncertainty estimation
N_ENSEMBLE = 5

MLP_PARAMS = dict(
    hidden_layer_sizes=(512, 256, 128),
    activation="relu",
    solver="adam",
    alpha=1e-4,           # L2 regularisation
    batch_size="auto",    # sklearn picks min(200, n_samples)
    learning_rate_init=3e-4,
    max_iter=15,
    warm_start=True,      # incremental training
    random_state=42,
)


class ArticleNet:
    """Ensemble of sklearn MLPs for relevance + urgency scoring with uncertainty."""

    def __init__(self):
        self._rel_models: list[MLPRegressor] = []
        self._urg_models: list[MLPRegressor] = []
        self._fitted = False
        self._load()

    def _load(self):
        if REL_MODEL_PATH.exists() and URG_MODEL_PATH.exists():
            try:
                with open(REL_MODEL_PATH, "rb") as f:
                    self._rel_models = pickle.load(f)
                with open(URG_MODEL_PATH, "rb") as f:
                    self._urg_models = pickle.load(f)
                self._fitted = True
                print(f"[ml:model] Loaded ensemble ({N_ENSEMBLE} models) from {MODEL_DIR}")
            except Exception as e:
                print(f"[ml:model] Load error: {e} — starting fresh")
                self._fitted = False

    def save(self):
        with open(REL_MODEL_PATH, "wb") as f:
            pickle.dump(self._rel_models, f)
        with open(URG_MODEL_PATH, "wb") as f:
            pickle.dump(self._urg_models, f)

    def fit(self, X: np.ndarray, y_rel: np.ndarray, y_urg: np.ndarray):
        """Train (or continue training) the ensemble. X shape: (N, features)."""
        if not SKLEARN_OK:
            return

        t0 = time.time()
        n = X.shape[0]

        # Build ensemble — each model trained on a different bootstrap sample
        if not self._fitted:
            self._rel_models = [MLPRegressor(**MLP_PARAMS) for _ in range(N_ENSEMBLE)]
            self._urg_models = [MLPRegressor(**MLP_PARAMS) for _ in range(N_ENSEMBLE)]

        rng = np.random.default_rng(42)
        for i in range(N_ENSEMBLE):
            idx = rng.integers(0, n, size=n)  # bootstrap sample
            self._rel_models[i].fit(X[idx], y_rel[idx])
            self._urg_models[i].fit(X[idx], y_urg[idx])

        self._fitted = True
        self.save()
        elapsed = time.time() - t0
        print(f"[ml:model] Ensemble trained on {n} samples in {elapsed:.1f}s")

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Returns (rel_mean, rel_std, urg_mean, urg_std) — all shape (N,).
        High std → uncertain → send to LLM.
        """
        if not self._fitted or not SKLEARN_OK:
            n = X.shape[0]
            return (np.zeros(n), np.full(n, 99.0),
                    np.zeros(n), np.full(n, 99.0))

        rel_preds = np.stack([m.predict(X) for m in self._rel_models])  # (E, N)
        urg_preds = np.stack([m.predict(X) for m in self._urg_models])

        rel_mean = np.clip(rel_preds.mean(axis=0), 0, 10)
        urg_mean = np.clip(urg_preds.mean(axis=0), 0, 10)
        rel_std  = rel_preds.std(axis=0)
        urg_std  = urg_preds.std(axis=0)

        return rel_mean, rel_std, urg_mean, urg_std

    @property
    def fitted(self) -> bool:
        return self._fitted


_net: ArticleNet | None = None


def get_model() -> ArticleNet:
    global _net
    if _net is None:
        _net = ArticleNet()
    return _net
