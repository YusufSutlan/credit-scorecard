"""Four experiments a model-risk team would ask for.

1. Does a WoE scorecard give up accuracy against a raw logistic regression and gradient boosting?
2. What happens after a macro shock (drift), and what does recalibration fix and not fix?
3. Reject inference: the bank only sees outcomes for applicants it approved. How wrong is a
   model trained on those, and does inverse-probability weighting help?
4. Where should the approval cutoff be, in money terms?

Usage:
    python -m scorecard.experiments
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

from .data import CATEGORICAL, FEATURES, NUMERIC, generate, legacy_policy
from .metrics import auc, brier, calibration_table, gini, ks, psi
from .scorecard import Scorecard

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "reports" / "figures"
MARGIN, LOSS = 2_500.0, 30_000.0  # profit per repaid loan, loss per default (TRY)


def raw_logistic() -> Pipeline:
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median", add_indicator=True)), ("sc", StandardScaler())]), NUMERIC),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
    ])  # fmt: skip
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=1.0, max_iter=2000))])


def boosting() -> Pipeline:
    pre = ColumnTransformer([
        ("num", "passthrough", NUMERIC),
        ("cat", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=np.nan), CATEGORICAL),
    ])  # fmt: skip
    clf = HistGradientBoostingClassifier(learning_rate=0.05, max_iter=300, max_leaf_nodes=15, l2_regularization=1.0,
                                         categorical_features=[len(NUMERIC)], random_state=0)  # fmt: skip
    return Pipeline([("pre", pre), ("clf", clf)])


def pd_of(model, df) -> np.ndarray:
    return model.predict_proba(df[FEATURES])[:, 1]


def metrics(y, p) -> dict:
    return {"auc": auc(y, p), "gini": gini(y, p), "ks": ks(y, p), "brier": brier(y, p),
            "mean_pd": float(np.mean(p)), "actual": float(np.mean(y))}  # fmt: skip


# ------------------------------------------------------------------------------------ 1
def exp_models(train, test) -> tuple[dict, dict]:
    sc = Scorecard().fit(train, train["default"])
    lr = raw_logistic().fit(train[FEATURES], train["default"])
    gb = boosting().fit(train[FEATURES], train["default"])
    preds = {"scorecard (WoE)": sc.predict_pd(test), "logistic (raw)": pd_of(lr, test), "gradient boosting": pd_of(gb, test)}
    res = {k: metrics(test["default"], v) for k, v in preds.items()}
    res["true PD (upper bound)"] = metrics(test["default"], test["true_pd"].to_numpy())
    return res, {"scorecard": sc, "preds": preds, "lr": lr, "gb": gb}


# ------------------------------------------------------------------------------------ 2
def exp_drift(models: dict, test0, v1) -> dict:
    sc: Scorecard = models["scorecard"]
    out = {"features_psi": {f: psi(test0[f], v1[f]) for f in FEATURES if f != "region"},
           "score_psi": psi(sc.score(test0), sc.score(v1))}  # fmt: skip
    out["models"] = {"scorecard": metrics(v1["default"], sc.predict_pd(v1)),
                     "logistic (raw)": metrics(v1["default"], pd_of(models["lr"], v1)),
                     "gradient boosting": metrics(v1["default"], pd_of(models["gb"], v1))}  # fmt: skip
    recent, rest = v1.iloc[:5000], v1.iloc[5000:]
    before = metrics(rest["default"], sc.predict_pd(rest))
    shift = sc.recalibrate(recent, recent["default"])
    after = metrics(rest["default"], sc.predict_pd(rest))
    sc.shift = 0.0
    out["recalibration"] = {"shift_logodds": shift, "before": before, "after": after}
    return out


# ------------------------------------------------------------------------------------ 3
def exp_reject_inference(seed: int = 5) -> dict:
    pool = generate(80_000, seed=seed)
    approved = legacy_policy(pool, seed=seed)
    test = generate(40_000, seed=seed + 1)
    acc = pool[approved]

    # Approval model (known in practice: it is the bank's own policy plus overrides).
    feat = pool[["dti", "delinq_12m", "income", "utilization", "age", "loan_amount"]].fillna(0)
    policy_lr = LogisticRegression(C=10, max_iter=2000).fit(StandardScaler().fit_transform(feat), approved)
    p_app = policy_lr.predict_proba(StandardScaler().fit(feat).transform(feat))[:, 1][approved]
    weights = 1 / np.clip(p_app, 0.05, 1.0)
    weights = np.clip(weights, None, 10.0)

    fits = {
        "accepted only": Scorecard().fit(acc, acc["default"]),
        "inverse-probability weighted": Scorecard().fit(acc, acc["default"], sample_weight=weights),
        "oracle (all outcomes)": Scorecard().fit(pool, pool["default"]),
    }
    res = {"approval_rate": float(approved.mean()), "bad_rate_accepted": float(acc["default"].mean()),
           "bad_rate_all": float(pool["default"].mean()), "models": {}}  # fmt: skip
    for name, m in fits.items():
        p = m.predict_pd(test)
        order = np.argsort(p)
        keep = order[: int(0.7 * len(test))]  # approve the 70% the model likes best
        res["models"][name] = {**metrics(test["default"], p), "bad_rate_when_approving_best_70pct": float(test["default"].to_numpy()[keep].mean())}
    return res


def exp_reject_multi(seeds=range(5, 10)) -> dict:
    runs = [exp_reject_inference(s) for s in seeds]
    agg = {"approval_rate": runs[0]["approval_rate"], "bad_rate_accepted": float(np.mean([r["bad_rate_accepted"] for r in runs])),
           "bad_rate_all": float(np.mean([r["bad_rate_all"] for r in runs])), "n_runs": len(runs), "models": {}, "std": {}}  # fmt: skip
    for name in runs[0]["models"]:
        keys = runs[0]["models"][name]
        agg["models"][name] = {k: float(np.mean([r["models"][name][k] for r in runs])) for k in keys}
        agg["std"][name] = {k: float(np.std([r["models"][name][k] for r in runs])) for k in keys}
    # Paired difference to the accepted-only model: same data, so far less noisy than comparing means.
    agg["paired_auc_gain_vs_accepted_only"] = {
        n: [float(r["models"][n]["auc"] - r["models"]["accepted only"]["auc"]) for r in runs] for n in runs[0]["models"]}
    return agg


# ------------------------------------------------------------------------------------ 4
def profit(pd_hat, y, cutoff) -> float:
    take = pd_hat < cutoff
    return float(MARGIN * (take & (y == 0)).sum() - LOSS * (take & (y == 1)).sum())


def exp_cutoff(models: dict, test) -> dict:
    y = test["default"].to_numpy()
    half = len(test) // 2
    out = {"breakeven_pd": MARGIN / (MARGIN + LOSS)}
    grid = np.linspace(0.02, 0.5, 97)
    out["curves"] = {}
    for name, p in models["preds"].items():
        val, hold = slice(0, half), slice(half, None)
        curve = [profit(p[val], y[val], c) for c in grid]
        best = float(grid[int(np.argmax(curve))])
        out[name] = {"best_cutoff": best, "profit_holdout": profit(p[hold], y[hold], best),
                     "approval_rate": float((p[hold] < best).mean())}  # fmt: skip
        out["curves"][name] = [profit(p, y, c) for c in grid]
    yh = y[half:]
    out["approve_all_profit_holdout"] = profit(np.zeros(len(yh)), yh, 1.0)
    rule = legacy_policy(test)[half:]
    out["legacy_rule_profit_holdout"] = float(MARGIN * (rule & (yh == 0)).sum() - LOSS * (rule & (yh == 1)).sum())
    out["grid"] = grid.tolist()
    return out


# ---------------------------------------------------------------------------------- figures
def plot_all(res1, models, drift, reject, cutoff, test) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    names = [n for n in res1 if not n.startswith("true")]
    axes[0].bar(names, [res1[n]["auc"] for n in names], color=["#1f77b4", "#7f7f7f", "#ff7f0e"])
    axes[0].axhline(res1["true PD (upper bound)"]["auc"], color="k", ls="--", lw=1, label="true PD (upper bound)")
    axes[0].set_ylim(0.6, 0.8)
    axes[0].set_ylabel("AUC (30k test applicants)")
    axes[0].legend()
    for n, p in models["preds"].items():
        t = calibration_table(test["default"], p)
        axes[1].plot(t["predicted"], t["observed"], "o-", ms=4, label=n)
    axes[1].plot([0, 0.3], [0, 0.3], "k:", lw=1)
    axes[1].set_xlabel("predicted default probability (decile mean)")
    axes[1].set_ylabel("observed default rate")
    axes[1].set_title("Calibration")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "model_comparison.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.3))
    f = drift["features_psi"]
    order = sorted(f, key=f.get)
    axes[0].barh(order, [f[k] for k in order], color=["#d62728" if f[k] > 0.25 else "#ff7f0e" if f[k] > 0.1 else "#2ca02c" for k in order])
    axes[0].axvline(0.1, color="k", ls=":", lw=1)
    axes[0].axvline(0.25, color="k", ls="--", lw=1)
    axes[0].set_xlabel(f"PSI per feature (score PSI = {drift['score_psi']:.2f})")
    axes[0].set_title("Population stability after the macro shock")
    r = drift["recalibration"]
    labels = ["train period", "after shock", "after intercept\nrecalibration"]
    vals = [res1["scorecard (WoE)"]["mean_pd"], r["before"]["mean_pd"], r["after"]["mean_pd"]]
    actual = [res1["scorecard (WoE)"]["actual"], r["before"]["actual"], r["after"]["actual"]]
    x = np.arange(3)
    axes[1].bar(x - 0.18, vals, 0.36, label="mean predicted PD")
    axes[1].bar(x + 0.18, actual, 0.36, label="actual default rate")
    axes[1].set_xticks(x, labels)
    axes[1].legend()
    axes[1].set_title("Calibration under drift")
    fig.tight_layout()
    fig.savefig(FIGURES / "drift.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    m = reject["models"]
    names = list(m)
    axes[0].bar(names, [m[n]["mean_pd"] for n in names], yerr=[reject["std"][n]["mean_pd"] for n in names], color="#1f77b4", capsize=4)
    axes[0].axhline(m[names[0]]["actual"], color="k", ls="--", label="actual default rate, all applicants")
    axes[0].set_title("Mean predicted PD on through-the-door applicants")
    axes[0].legend()
    axes[1].bar(names, [m[n]["auc"] for n in names], yerr=[reject["std"][n]["auc"] for n in names], color="#ff7f0e", capsize=4)
    axes[1].set_ylim(0.6, 0.8)
    axes[1].set_title("AUC on through-the-door applicants")
    for ax in axes:
        ax.tick_params(axis="x", labelsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "reject_inference.png", dpi=130)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    for n, c in cutoff["curves"].items():
        ax.plot(cutoff["grid"], np.array(c) / 1e6, label=n)
    ax.axvline(cutoff["breakeven_pd"], color="k", ls=":", label=f"break-even PD {cutoff['breakeven_pd']:.1%}")
    ax.set_xlabel("approve if predicted PD below")
    ax.set_ylabel("profit on 30k applicants (million TRY)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES / "cutoff_profit.png", dpi=130)
    plt.close(fig)


def main() -> None:
    train, test = generate(40_000, seed=1), generate(30_000, seed=2)
    v1 = generate(30_000, seed=3, vintage=1)

    res1, models = exp_models(train, test)
    print(f"{'model':<24}{'AUC':>7}{'Gini':>7}{'KS':>7}{'Brier':>8}")
    for n, m in res1.items():
        print(f"{n:<24}{m['auc']:>7.3f}{m['gini']:>7.3f}{m['ks']:>7.3f}{m['brier']:>8.4f}")
    sc = models["scorecard"]
    print("\nIV (kept:", ", ".join(sc.features), ")\n", sc.iv.round(3).to_string())

    drift = exp_drift(models, test, v1)
    print(f"\nDrift: default rate {res1['scorecard (WoE)']['actual']:.3f} -> {drift['models']['scorecard']['actual']:.3f}, score PSI {drift['score_psi']:.3f}")
    for n, m in drift["models"].items():
        print(f"  {n:<20} AUC {m['auc']:.3f}  mean PD {m['mean_pd']:.3f}  actual {m['actual']:.3f}")
    r = drift["recalibration"]
    print(f"  recalibrated scorecard: mean PD {r['before']['mean_pd']:.3f} -> {r['after']['mean_pd']:.3f} (actual {r['after']['actual']:.3f}), AUC {r['before']['auc']:.3f} -> {r['after']['auc']:.3f}")  # fmt: skip
    print("  feature PSI:", {k: round(v, 2) for k, v in sorted(drift['features_psi'].items(), key=lambda kv: -kv[1])[:5]})

    reject = exp_reject_multi()
    print(f"\nReject inference ({reject['n_runs']} runs): approved {reject['approval_rate']:.0%}, bad rate accepted {reject['bad_rate_accepted']:.3f} vs all applicants {reject['bad_rate_all']:.3f}")
    for n, m in reject["models"].items():
        sd = reject["std"][n]
        print(f"  {n:<30} AUC {m['auc']:.3f}±{sd['auc']:.3f}  mean PD {m['mean_pd']:.3f}±{sd['mean_pd']:.3f} (actual {m['actual']:.3f})  bad rate if approve best 70%: {m['bad_rate_when_approving_best_70pct']:.4f}±{sd['bad_rate_when_approving_best_70pct']:.4f}")  # fmt: skip
    print("  paired AUC gain vs accepted-only per run:", {k: [round(x, 4) for x in v] for k, v in reject["paired_auc_gain_vs_accepted_only"].items() if k != "accepted only"})

    cutoff = exp_cutoff(models, test)
    print(f"\nCutoff (break-even PD {cutoff['breakeven_pd']:.1%}), profit on holdout half of the test set (million TRY):")
    for n in models["preds"]:
        c = cutoff[n]
        print(f"  {n:<20} cutoff {c['best_cutoff']:.3f}  approval {c['approval_rate']:.0%}  profit {c['profit_holdout'] / 1e6:.1f}")
    print(f"  approve everyone: {cutoff['approve_all_profit_holdout'] / 1e6:.1f} | legacy rule: {cutoff['legacy_rule_profit_holdout'] / 1e6:.1f}  (same holdout half)")

    example = test[(test["delinq_12m"] >= 2) & (test["dti"] > 0.6)].head(2)
    reasons = {str(i): sc.reason_codes(example.loc[[i]]) for i in example.index}
    print("\nReason codes for two declined-looking applicants:", json.dumps(reasons, ensure_ascii=False, default=str))
    plot_all(res1, models, drift, reject, cutoff, test)
    (ROOT / "reports" / "results.json").write_text(json.dumps(
        {"models": res1, "drift": drift, "reject_inference": reject, "cutoff": {k: v for k, v in cutoff.items() if k not in ("curves", "grid")},
         "reason_codes": reasons}, indent=2, default=float, ensure_ascii=False))
    sc.points_table().to_csv(ROOT / "reports" / "scorecard_points.csv", index=False)


if __name__ == "__main__":
    main()
