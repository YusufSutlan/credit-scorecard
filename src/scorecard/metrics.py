"""Credit-risk model metrics: discrimination (AUC, Gini, KS), calibration, stability (PSI)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve


def auc(y, p) -> float:
    return float(roc_auc_score(y, p))


def gini(y, p) -> float:
    return 2 * auc(y, p) - 1


def ks(y, p) -> float:
    """Largest gap between the cumulative score distributions of bads and goods."""
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def brier(y, p) -> float:
    return float(brier_score_loss(y, p))


def psi(expected, actual, n_bins: int = 10) -> float:
    """Population stability index. Rule of thumb: < 0.1 stable, 0.1-0.25 watch, > 0.25 shifted."""
    expected, actual = np.asarray(expected, float), np.asarray(actual, float)
    expected, actual = expected[~np.isnan(expected)], actual[~np.isnan(actual)]
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, n_bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, 1e-4, None), np.clip(a, 1e-4, None)
    return float(((a - e) * np.log(a / e)).sum())


def calibration_table(y, p, n_bins: int = 10) -> pd.DataFrame:
    df = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)})
    df["bucket"] = pd.qcut(df["p"], n_bins, duplicates="drop")
    out = df.groupby("bucket", observed=True).agg(predicted=("p", "mean"), observed=("y", "mean"), n=("y", "size"))
    return out.reset_index(drop=True)
