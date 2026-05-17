"""
Text embedder — Phase 1: TF-IDF with financial vocabulary.

Fits on the existing article corpus and persists the vectorizer to disk.
Phase 2 upgrade: replace _vectorize() with sentence-transformers — same interface.
"""
import os
import pickle
import numpy as np
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer

MODEL_DIR = Path(os.environ.get("DIGITAL_INTERN_ML_DIR",
                                Path(__file__).resolve().parent.parent / "data" / "ml"))
MODEL_DIR.mkdir(parents=True, exist_ok=True)

VECTORIZER_PATH = MODEL_DIR / "tfidf_vectorizer.pkl"

# Financial domain vocabulary to boost rare but important terms
FINANCIAL_VOCAB_BOOST = [
    # Memory / DRAM
    "dram", "nand", "hbm", "hbm3e", "hbm2e", "lpddr5", "lpddr4x",
    "asp", "average selling price", "wafer", "bit growth", "capex",
    "memory pricing", "dram supply", "nand oversupply", "memory chip",
    # Portfolio tickers
    "micron", "lumentum", "lite", "axti", "tsem", "qbts", "msft",
    "sk hynix", "kioxia", "western digital", "tower semiconductor",
    # Semis ecosystem
    "semiconductor", "tsmc", "asml", "euv", "foundry", "fab",
    "amat", "lrcx", "klac", "chip equipment", "advanced packaging",
    "ai chip", "hpc", "data center", "nvidia", "amd", "intel",
    # Events
    "earnings beat", "earnings miss", "guidance raise", "guidance cut",
    "upgrade", "downgrade", "price target", "initiate", "coverage",
    "acquisition", "merger", "buyout", "takeover",
    "shortage", "glut", "oversupply", "capacity cut", "production halt",
    # Macro
    "federal reserve", "fomc", "rate cut", "rate hike", "inflation",
    "cpi", "pce", "nonfarm payroll", "gdp", "recession",
    "tariff", "export control", "entity list", "china ban", "chip war",
    "emergency", "surprise", "unexpected", "flash crash", "circuit breaker",
    # Crypto
    "bitcoin", "ethereum", "crypto", "etf inflows",
    # Asia
    "nikkei", "kospi", "hang seng", "samsung", "korea", "japan",
]

TFIDF_PARAMS = {
    "max_features": 15_000,
    "ngram_range": (1, 3),
    "sublinear_tf": True,
    "min_df": 2,
    "max_df": 0.95,
    "analyzer": "word",
    "strip_accents": "unicode",
    "vocabulary": None,  # learned from corpus
}


# Minimum acceptable vocabulary size. A pickle fitted on a tiny early corpus
# (vocab=3) was leaving the model effectively blind to article text, so a refit
# is forced whenever vocab drops below this floor.
MIN_VOCAB_SIZE = 500
# Refit when the available corpus has grown by this multiple since the last fit.
REFIT_GROWTH_MULTIPLE = 3.0


class Embedder:
    def __init__(self):
        self._vec: TfidfVectorizer | None = None
        self._fitted = False
        self._last_fit_n: int = 0
        self._load()

    def _load(self):
        if not VECTORIZER_PATH.exists():
            return
        try:
            with open(VECTORIZER_PATH, "rb") as f:
                obj = pickle.load(f)
            if isinstance(obj, dict) and "vec" in obj:
                self._vec = obj["vec"]
                self._last_fit_n = int(obj.get("last_fit_n", 0))
            else:
                self._vec = obj
                self._last_fit_n = 0
            self._fitted = True
        except Exception as e:
            print(f"[embedder] Failed to load vectorizer ({e}); will refit on next call")
            self._vec = None
            self._fitted = False
            self._last_fit_n = 0

    def _save(self):
        with open(VECTORIZER_PATH, "wb") as f:
            pickle.dump({"vec": self._vec, "last_fit_n": self._last_fit_n}, f)

    def should_refit(self, n: int) -> bool:
        """Return True when the saved vocab is unusable or the corpus has grown
        materially since the last fit. Triggers an expensive refit, so guard
        against thrashing by requiring real change."""
        if not self._fitted or self._vec is None:
            return True
        try:
            vocab_size = len(self._vec.vocabulary_)
        except Exception:
            return True
        if vocab_size < MIN_VOCAB_SIZE:
            return True
        if self._last_fit_n > 0 and n >= self._last_fit_n * REFIT_GROWTH_MULTIPLE:
            return True
        return False

    def fit(self, texts: list[str]):
        """Fit the vectorizer on a corpus. Call once on the article store at startup.

        The new vectorizer is fitted on a local variable and swapped into
        ``self._vec`` only when fully ready, so concurrent inference threads
        never observe a half-initialized vectorizer that would raise
        NotFittedError mid-refit."""
        new_vec = TfidfVectorizer(**TFIDF_PARAMS)
        new_vec.fit(texts)
        self._vec = new_vec
        self._fitted = True
        self._last_fit_n = len(texts)
        self._save()
        print(f"[embedder] Fitted TF-IDF on {len(texts)} texts, "
              f"vocab size={len(self._vec.vocabulary_)}")

    def transform(self, texts: list[str]) -> np.ndarray:
        """Transform list of texts → dense float32 matrix (N, max_features)."""
        if not self._fitted:
            raise RuntimeError("Embedder not fitted yet — call fit() first")
        sparse = self._vec.transform(texts)
        return sparse.toarray().astype(np.float32)

    def fit_transform(self, texts: list[str]) -> np.ndarray:
        self.fit(texts)
        return self.transform(texts)

    @property
    def dim(self) -> int:
        return len(self._vec.vocabulary_) if self._fitted else 15_000

    @property
    def fitted(self) -> bool:
        return self._fitted


_embedder: Embedder | None = None


def get_embedder() -> Embedder:
    global _embedder
    if _embedder is None:
        _embedder = Embedder()
    return _embedder
