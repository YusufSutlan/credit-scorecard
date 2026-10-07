"""Weight of evidence (WoE) binning with monotonic merging and information value (IV).

WoE = ln(share of goods in the bin / share of bads in the bin). Higher = safer. Binning turns
each variable into a handful of interpretable, monotone risk bands, which is what lets a
scorecard be audited and explained to a regulator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

SMOOTH = 0.5


class WoE:
    def __init__(self, kind: str = "numeric", n_bins: int = 10, monotonic: str | None = "auto", min_frac: float = 0.03):
        """monotonic: 'auto' (direction from the data), 'up', 'down', or None (e.g. U-shaped age)."""
        self.kind, self.n_bins, self.monotonic, self.min_frac = kind, n_bins, monotonic, min_frac

    # ------------------------------------------------------------------ fit
    def fit(self, x: pd.Series, y: np.ndarray, weight: np.ndarray | None = None) -> "WoE":
        y = np.asarray(y, float)
        w = np.ones(len(y)) if weight is None else np.asarray(weight, float)
        n_bad, n_good = (y * w).sum(), ((1 - y) * w).sum()
        if self.kind == "numeric":
            self._fit_numeric(x.to_numpy(float), y, w)
        else:
            self._fit_categorical(x.astype(object).fillna("__missing__").to_numpy(), y, w)
        idx = self._bin_index(x)
        self.bad = np.bincount(idx, weights=y * w, minlength=self.n_total).astype(float)
        self.total = np.bincount(idx, weights=w, minlength=self.n_total).astype(float)
        good = self.total - self.bad
        self.woe = np.log(((good + SMOOTH) / (n_good + SMOOTH)) / ((self.bad + SMOOTH) / (n_bad + SMOOTH)))
        if self.total[-1] == 0:  # nothing missing in training: a missing value at scoring time is neutral
            self.woe[-1] = 0.0
        self.iv = float((((good + SMOOTH) / (n_good + SMOOTH) - (self.bad + SMOOTH) / (n_bad + SMOOTH)) * self.woe).sum())
        return self

    def _fit_numeric(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        ok = ~np.isnan(x)
        values = np.unique(x[ok])
        if len(values) <= 12:  # discrete variable: every value starts as its own bin
            edges = values[1:]
        else:
            edges = np.unique(np.quantile(x[ok], np.linspace(0, 1, self.n_bins + 1))[1:-1])
        while True:
            idx = np.digitize(x[ok], edges)
            k = len(edges) + 1
            cnt = np.bincount(idx, weights=w[ok], minlength=k).astype(float)
            bad = np.bincount(idx, weights=(y * w)[ok], minlength=k)
            rate = bad / np.maximum(cnt, 1)
            merge = None
            small = np.flatnonzero(cnt < self.min_frac * w[ok].sum())
            if len(small) and k > 2:
                i = small[np.argmin(cnt[small])]
                merge = i - 1 if i == k - 1 else i  # merge with the right neighbour (left for the last bin)
            elif self.monotonic and k > 2:
                direction = self.monotonic
                if direction == "auto":
                    direction = "up" if spearmanr(np.arange(k), rate)[0] >= 0 else "down"
                diff = np.diff(rate) * (1 if direction == "up" else -1)
                bad_pairs = np.flatnonzero(diff < 0)
                if len(bad_pairs):
                    merge = int(bad_pairs[0])
            if merge is None:
                break
            edges = np.delete(edges, merge)
        self.edges = edges
        self.has_missing_bin = bool((~ok).any())
        self.n_bins_value = len(edges) + 1
        self.n_total = self.n_bins_value + 1  # last slot = missing

    def _fit_categorical(self, x: np.ndarray, y: np.ndarray, w: np.ndarray) -> None:
        cats = np.unique(x)
        keep = [c for c in cats if w[x == c].sum() >= self.min_frac * w.sum()]
        self.categories = keep
        self.n_bins_value = len(keep) + 1  # + "other"
        self.n_total = self.n_bins_value + 1

    # ------------------------------------------------------------ transform
    def _bin_index(self, x: pd.Series) -> np.ndarray:
        if self.kind == "numeric":
            v = x.to_numpy(float)
            idx = np.digitize(np.nan_to_num(v, nan=0.0), self.edges)
            return np.where(np.isnan(v), self.n_total - 1, idx)
        values = x.astype(object).fillna("__missing__").to_numpy()
        lookup = {c: i for i, c in enumerate(self.categories)}
        return np.array([self.n_total - 1 if v == "__missing__" else lookup.get(v, len(self.categories)) for v in values])

    def transform(self, x: pd.Series) -> np.ndarray:
        return self.woe[self._bin_index(x)]

    def labels(self) -> list[str]:
        if self.kind == "numeric":
            e = [-np.inf, *self.edges, np.inf]
            out = [f"[{a:g}, {b:g})" for a, b in zip(e[:-1], e[1:])]
        else:
            out = [str(c) for c in self.categories] + ["other"]
        return out + ["missing"]

    def table(self) -> pd.DataFrame:
        rate = self.bad / np.maximum(self.total, 1)
        return pd.DataFrame({"bin": self.labels(), "n": self.total.astype(int), "bad_rate": rate, "woe": self.woe})
