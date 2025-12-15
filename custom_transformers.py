# custom_transformers.py
import re
import numpy as np
import pandas as pd
from typing import List
from sklearn.base import BaseEstimator, TransformerMixin

class AddressTopTokens(BaseEstimator, TransformerMixin):
    """Ekstrak Top-N token dari kolom alamat → fitur biner 0/1 per token."""
    def __init__(self, col_name: str, top_n: int = 40, min_len: int = 3):
        self.col_name = col_name
        self.top_n = top_n
        self.min_len = min_len
        self.tokens_: List[str] = []

    def _series(self, X):
        if isinstance(X, pd.DataFrame):
            s = X[self.col_name] if self.col_name in X.columns else X.iloc[:, 0]
        else:
            s = pd.Series(X)
        return s.fillna("")

    def _clean(self, s: str) -> str:
        s = str(s).upper()
        s = re.sub(r'\b(JALAN|JLN|JL|JL\.|GG|GANG|NO\.?\s*\d+|RT\s*\d+/?\d*|RW\s*\d+)\b', ' ', s)
        s = re.sub(r'[^A-Z0-9 ]+', ' ', s)
        s = re.sub(r'\s+', ' ', s).strip()
        return s

    def fit(self, X, y=None):
        s = self._series(X).astype(str).map(self._clean)
        tokens = s.str.split().explode()
        tokens = tokens[tokens.str.len() >= self.min_len]
        vc = tokens.value_counts()
        self.tokens_ = list(vc.head(self.top_n).index)
        return self

    def transform(self, X):
        s = self._series(X).astype(str).map(self._clean)
        out = {}
        for t in self.tokens_:
            out[f"{self.col_name}__TOK_{t}"] = s.str.contains(rf"\b{re.escape(t)}\b").astype(int).values
        if not out:
            return pd.DataFrame({f"{self.col_name}__TOK_NONE": np.zeros(len(s), dtype=int)},
                                index=getattr(X, "index", None))
        return pd.DataFrame(out, index=getattr(X, "index", None))

    def get_feature_names_out(self, input_features=None):
        if self.tokens_:
            return np.array([f"{self.col_name}__TOK_{t}" for t in self.tokens_], dtype=object)
        return np.array([f"{self.col_name}__TOK_NONE"], dtype=object)

class FrequencyEncoder(BaseEstimator, TransformerMixin):
    """Frequency Encoding 1 kolom kategori → 1 kolom numerik <col>__freq."""
    def __init__(self, col_name: str):
        self.col_name = col_name
        self.freq_map_ = None

    def _series(self, X):
        if isinstance(X, pd.DataFrame):
            s = X[self.col_name] if self.col_name in X.columns else X.iloc[:, 0]
        else:
            s = pd.Series(X)
        return s.fillna("NA").astype(str)

    def fit(self, X, y=None):
        s = self._series(X)
        self.freq_map_ = s.value_counts()
        return self

    def transform(self, X):
        s = self._series(X)
        vals = s.map(self.freq_map_).fillna(1).astype(float).values
        return pd.DataFrame({f"{self.col_name}__freq": vals}, index=getattr(X, "index", None))

    def get_feature_names_out(self, input_features=None):
        return np.array([f"{self.col_name}__freq"], dtype=object)
