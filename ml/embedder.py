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


class Embedder:
    def __init__(self):
        self._vec: TfidfVectorizer | None = None
        self._fitted = False
        self._load()

    def _load(self):
        if VECTORIZER_PATH.exists():
            with open(VECTORIZER_PATH, "rb") as f:
                self._vec = pickle.load(f)
                self._fitted = True

    def _save(self):
        with open(VECTORIZER_PATH, "wb") as f:
            pickle.dump(self._vec, f)

    def fit(self, texts: list[str]):
        """Fit the vectorizer on a corpus. Call once on the article store at startup."""
        self._vec = TfidfVectorizer(**TFIDF_PARAMS)
        self._vec.fit(texts)
        self._fitted = True
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
