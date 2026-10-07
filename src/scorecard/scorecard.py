"""A classical points-based credit scorecard.

1. Bin every variable and replace it by its WoE (see woe.py); keep variables with IV >= min_iv.
2. Fit a logistic regression on the WoE values (ln odds of default is linear in WoE).
3. Scale to points: score = offset + factor * ln(odds of being good), where `pdo` points
   double the odds, and `base_score` corresponds to `base_odds`:1 good:bad.
   Because the model is additive, every applicant's score is a sum of per-bin points, so a
   decline can be explained with "reason codes" (the variables that cost the most points).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .woe import WoE

SPECS: dict[str, dict] = {
    "age": dict(monotonic=None),  # risk is U-shaped in age: do not force monotonicity
    "income": {}, "employment_years": {}, "loan_amount": {}, "term_months": {}, "dti": {},
    "utilization": {}, "delinq_12m": {}, "inquiries_6m": {}, "home_owner": {},
    "region": dict(kind="categorical"),
}  # fmt: skip


class Scorecard:
    def __init__(self, specs: dict[str, dict] = SPECS, min_iv: float = 0.02, pdo: float = 20.0,
                 base_score: float = 600.0, base_odds: float = 50.0, C: float = 1.0):  # fmt: skip
        self.specs, self.min_iv, self.C = specs, min_iv, C
        self.factor = pdo / np.log(2)
        self.offset = base_score - self.factor * np.log(base_odds)
        self.shift = 0.0  # intercept shift from recalibration (log-odds of default)

    def fit(self, df: pd.DataFrame, y, sample_weight=None) -> "Scorecard":
        y = np.asarray(y)
        woes = {f: WoE(**spec).fit(df[f], y, sample_weight) for f, spec in self.specs.items()}
        self.iv = pd.Series({f: w.iv for f, w in woes.items()}).sort_values(ascending=False)
        self.woes = {f: w for f, w in woes.items() if w.iv >= self.min_iv}
        self.features = list(self.woes)
        self.model = LogisticRegression(C=self.C, max_iter=1000).fit(self._woe_matrix(df), y, sample_weight=sample_weight)
        self.coef = pd.Series(self.model.coef_[0], index=self.features)
        return self

    def _woe_matrix(self, df: pd.DataFrame) -> np.ndarray:
        return np.column_stack([self.woes[f].transform(df[f]) for f in self.features])

    # ------------------------------------------------------------ predictions
    def logit_default(self, df: pd.DataFrame) -> np.ndarray:
        return self.model.decision_function(self._woe_matrix(df)) + self.shift

    def predict_pd(self, df: pd.DataFrame) -> np.ndarray:
        return 1 / (1 + np.exp(-self.logit_default(df)))

    def score(self, df: pd.DataFrame) -> np.ndarray:
        return self.offset + self.factor * (-self.logit_default(df))  # ln(odds good) = -ln(odds bad)

    def recalibrate(self, df: pd.DataFrame, y) -> float:
        """Shift the intercept so the mean predicted PD equals the observed default rate."""
        self.shift = 0.0
        target, lo, hi = float(np.mean(y)), -5.0, 5.0
        base = self.logit_default(df)
        for _ in range(60):
            mid = (lo + hi) / 2
            lo, hi = (mid, hi) if np.mean(1 / (1 + np.exp(-(base + mid)))) < target else (lo, mid)
        self.shift = (lo + hi) / 2
        return self.shift

    # --------------------------------------------------------------- points
    def points_table(self) -> pd.DataFrame:
        n, b0 = len(self.features), self.model.intercept_[0]
        rows = []
        for f in self.features:
            w = self.woes[f]
            pts = (self.offset - self.factor * b0) / n + self.factor * (-self.coef[f]) * w.woe
            t = w.table()
            rows.append(t.assign(feature=f, points=pts))
        return pd.concat(rows, ignore_index=True)[["feature", "bin", "n", "bad_rate", "woe", "points"]]

    def points_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        n, b0 = len(self.features), self.model.intercept_[0]
        cols = {f: (self.offset - self.factor * b0) / n + self.factor * (-self.coef[f]) * self.woes[f].transform(df[f]) for f in self.features}
        return pd.DataFrame(cols, index=df.index)

    def reason_codes(self, row: pd.DataFrame, k: int = 3) -> list[dict]:
        """The k variables where this applicant lost the most points against the best possible bin."""
        table = self.points_table()
        best = table.groupby("feature")["points"].max()
        mine = self.points_matrix(row).iloc[0]
        loss = (best - mine).sort_values(ascending=False)
        return [{"feature": f, "points_lost": round(float(l), 1), "value": None if pd.isna(row[f].iloc[0]) else row[f].iloc[0]}
                for f, l in loss.head(k).items() if l > 0]  # fmt: skip
