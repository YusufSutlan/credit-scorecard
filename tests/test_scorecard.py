import numpy as np
import pandas as pd

from scorecard.data import generate, legacy_policy
from scorecard.experiments import exp_models, exp_reject_inference, profit, MARGIN, LOSS
from scorecard.metrics import auc, calibration_table, gini, ks, psi
from scorecard.scorecard import Scorecard
from scorecard.woe import WoE


def test_data_is_deterministic_and_realistic():
    a, b = generate(2000, 3), generate(2000, 3)
    assert a.equals(b)
    d = generate(30_000, 0)
    assert 0.06 < d["default"].mean() < 0.12 and generate(30_000, 0, vintage=1)["default"].mean() > d["default"].mean() + 0.02
    assert d["utilization"].isna().mean() > 0.03


def test_legacy_policy_selects_safer_applicants_but_keeps_overrides():
    d = generate(30_000, 1)
    ok = legacy_policy(d)
    assert 0.6 < ok.mean() < 0.8 and d.loc[ok, "default"].mean() < 0.8 * d["default"].mean()
    rule_fail = (d["dti"] >= 0.5) | (d["delinq_12m"] > 1)
    assert 0.03 < ok[rule_fail].mean() < 0.25  # overrides exist, which is what makes reject inference possible


def test_woe_bins_are_monotone_and_signed_correctly():
    d = generate(30_000, 2)
    w = WoE().fit(d["dti"], d["default"].to_numpy())
    t = w.table().iloc[:-1]
    assert (np.diff(t["bad_rate"]) >= -1e-12).all() and (np.diff(t["woe"]) <= 1e-12).all() and w.iv > 0.2
    age = WoE(monotonic=None).fit(d["age"], d["default"].to_numpy())
    assert len(age.table()) >= len(w.table()) - 4


def test_woe_missing_and_unseen_values_are_handled():
    d = generate(10_000, 3)
    w = WoE().fit(d["delinq_12m"], d["default"].to_numpy())  # nothing missing in training
    assert w.woe[-1] == 0.0 and np.isfinite(w.transform(pd.Series([np.nan, 99.0]))).all()
    c = WoE("categorical").fit(d["region"], d["default"].to_numpy())
    assert np.isfinite(c.transform(pd.Series(["Mars", None]))).all()


def test_unit_weights_change_nothing():
    d = generate(10_000, 4)
    a = WoE().fit(d["income"], d["default"].to_numpy())
    b = WoE().fit(d["income"], d["default"].to_numpy(), np.ones(len(d)))
    assert np.allclose(a.woe, b.woe) and np.array_equal(a.edges, b.edges)


def fitted():
    train = generate(20_000, 1)
    return train, Scorecard().fit(train, train["default"])


def test_points_scaling_follows_pdo_and_base_score():
    train, sc = fitted()
    assert abs(sc.offset + sc.factor * np.log(50) - 600) < 1e-9
    base = sc.score(train.head(500))
    sc.shift = -np.log(2)  # odds of default halve, i.e. odds of being good double
    assert np.allclose(sc.score(train.head(500)) - base, 20.0)
    sc.shift = 0.0


def test_score_is_the_sum_of_bin_points_and_decreases_with_risk():
    train, sc = fitted()
    s = sc.score(train.head(2000))
    assert np.allclose(sc.points_matrix(train.head(2000)).sum(axis=1), s)
    pd_hat = sc.predict_pd(train.head(2000))
    assert np.corrcoef(s, pd_hat)[0, 1] < -0.9
    assert (sc.coef < 0).all()  # WoE is "higher = safer", so every coefficient on default must be negative
    assert "inquiries_6m" not in sc.features  # IV below the cut-off


def test_reason_codes_name_the_costliest_variables():
    train, sc = fitted()
    bad = train[(train["delinq_12m"] >= 2) & (train["dti"] > 0.6)].head(1)
    codes = sc.reason_codes(bad, k=3)
    assert 1 <= len(codes) <= 3 and codes[0]["feature"] in ("delinq_12m", "dti")
    assert all(a["points_lost"] >= b["points_lost"] for a, b in zip(codes, codes[1:]))


def test_recalibration_fixes_the_level_but_not_the_ranking():
    train, sc = fitted()
    v1 = generate(10_000, 9, vintage=1)
    before_auc = auc(v1["default"], sc.predict_pd(v1))
    sc.recalibrate(v1, v1["default"])
    assert abs(sc.predict_pd(v1).mean() - v1["default"].mean()) < 1e-3
    assert abs(auc(v1["default"], sc.predict_pd(v1)) - before_auc) < 1e-9


def test_metrics():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 5000)
    p = np.clip(y * 0.3 + rng.normal(0.3, 0.2, 5000), 0, 1)
    assert abs(gini(y, p) - (2 * auc(y, p) - 1)) < 1e-12 and 0 < ks(y, p) < 1
    assert psi(p, p) < 1e-6 and psi(p, p + 0.4) > 0.25
    t = calibration_table(y, p, 5)
    assert len(t) == 5 and (np.diff(t["predicted"]) > 0).all()


def test_profit_arithmetic():
    y = np.array([0, 0, 1, 1])
    assert profit(np.array([0.01, 0.02, 0.5, 0.6]), y, 0.1) == 2 * MARGIN
    assert profit(np.array([0.01, 0.02, 0.05, 0.6]), y, 0.1) == 2 * MARGIN - LOSS


def test_model_comparison_and_true_pd_bound():
    res, _ = exp_models(generate(10_000, 1), generate(8_000, 2))
    assert all(res[k]["auc"] > 0.68 for k in ("scorecard (WoE)", "logistic (raw)", "gradient boosting"))
    assert res["true PD (upper bound)"]["auc"] >= max(res[k]["auc"] for k in res if not k.startswith("true"))


def test_reject_inference_bias_is_visible():
    r = exp_reject_inference(seed=5)
    m = r["models"]
    assert r["bad_rate_accepted"] < r["bad_rate_all"]
    assert m["accepted only"]["mean_pd"] < m["accepted only"]["actual"] * 0.95  # underestimates risk of the full population
    assert m["oracle (all outcomes)"]["auc"] > m["accepted only"]["auc"]
