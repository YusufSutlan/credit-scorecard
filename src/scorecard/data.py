"""Synthetic retail-loan applications with a known default process.

Because the true default probability of every applicant is known (`true_pd`), the experiments
can answer questions that real data cannot: what would a model trained on *all* applicants have
learned, how far off are the predicted probabilities after a macro shock, and so on.

Vintage 0 is the development period. Vintage 1 is a later period with a macro shock (higher
default rates), income inflation and larger loans: a covariate shift plus a concept shift.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

REGIONS = ["Marmara", "Ege", "İç Anadolu", "Akdeniz", "Doğu"]
REGION_EFFECT = dict(zip(REGIONS, [0.0, -0.10, 0.05, 0.10, 0.30]))
NUMERIC = ["age", "income", "employment_years", "loan_amount", "term_months", "dti", "utilization",
           "delinq_12m", "inquiries_6m", "home_owner"]  # fmt: skip
CATEGORICAL = ["region"]
FEATURES = NUMERIC + CATEGORICAL


def generate(n: int, seed: int = 0, vintage: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    income_scale, loan_scale, macro = (1.25, 1.2, 0.45) if vintage == 1 else (1.0, 1.0, 0.0)

    age = np.clip(rng.normal(40, 12, n), 21, 70).round()
    income = rng.lognormal(np.log(25_000 * income_scale), 0.5, n).round(-2)
    employment_years = np.minimum(age - 20, rng.exponential(6, n)).round(1)
    loan_amount = rng.lognormal(np.log(60_000 * loan_scale), 0.7, n).round(-2)
    term = rng.choice([12, 24, 36, 48], n, p=[0.15, 0.3, 0.35, 0.2])
    existing_debt = income * rng.beta(2, 6, n)
    dti = np.clip((existing_debt + loan_amount / term * 1.15) / income, 0, 2).round(3)
    utilization = np.clip(rng.beta(2, 3, n) + (0.05 if vintage == 1 else 0), 0, 1).round(3)
    delinq = rng.poisson(0.3 * (1 + utilization))
    inquiries = rng.poisson(1.2, n)
    home_owner = (rng.random(n) < 0.25 + 0.4 * (age > 35)).astype(int)
    region = rng.choice(REGIONS, n, p=[0.35, 0.2, 0.2, 0.15, 0.1])

    util_missing = rng.random(n) < 0.05  # informative missingness: applicants who skip the field are riskier
    z = (-3.0 + 2.4 * (dti - 0.35) + 1.6 * (utilization - 0.4) + 0.45 * delinq + 0.10 * inquiries
         - 0.018 * (age - 40) + 0.0009 * (age - 40) ** 2 - 0.35 * np.log(income / (25_000 * income_scale))
         - 0.04 * np.clip(employment_years, 0, 15) - 0.25 * home_owner
         + np.array([REGION_EFFECT[r] for r in region]) + 0.6 * ((utilization > 0.8) & (delinq > 0))
         + 0.3 * util_missing + macro + rng.normal(0, 0.35, n))  # fmt: skip
    true_pd = 1 / (1 + np.exp(-z))

    df = pd.DataFrame({
        "age": age, "income": income, "employment_years": employment_years, "loan_amount": loan_amount,
        "term_months": term, "dti": dti, "utilization": np.where(util_missing, np.nan, utilization),
        "delinq_12m": delinq, "inquiries_6m": inquiries, "home_owner": home_owner, "region": region,
        "default": (rng.random(n) < true_pd).astype(int), "true_pd": true_pd,
    })  # fmt: skip
    df.loc[rng.random(n) < 0.03, "employment_years"] = np.nan
    return df


def legacy_policy(df: pd.DataFrame, seed: int = 0, override_rate: float = 0.12) -> np.ndarray:
    """The bank's old approval rule. A small share of rule-failing applicants is approved anyway
    (manual overrides), which is what makes reject inference possible in principle."""
    rng = np.random.default_rng(seed)
    rule = (df["dti"] < 0.50) & (df["delinq_12m"] <= 1) & (df["utilization"].fillna(0) < 0.90) & (df["income"] >= 8_000)
    p = np.where(rule, 0.97, override_rate)
    return rng.random(len(df)) < p
