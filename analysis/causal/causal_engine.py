"""
ContentPulse — Part 1: Causal Inference Engine
Estimates causal effect of content diversity on 7-day return rate.

Methods:
1. Difference-in-Differences (DiD) — primary estimator
2. Propensity Score Matching — robustness check
3. Bayesian A/B Test (Beta-Binomial) — uncertainty quantification

Treatment: high diversity session (genre entropy > median)
Control: low diversity session (genre entropy <= median)
Outcome: 7-day return rate
"""
import logging
import numpy as np
import pandas as pd
import pymc as pm
import arviz as az
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

logger = logging.getLogger(__name__)
OUTPUT_DIR = Path("data/outputs")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def test_parallel_trends(sessions_df: pd.DataFrame) -> dict:
    """
    Test parallel trends assumption for DiD.
    Compare pre-treatment trends between high/low diversity groups.
    """
    sessions_df = sessions_df.copy()
    sessions_df["week"] = pd.to_datetime(sessions_df["session_date"]).dt.isocalendar().week
    median_entropy = sessions_df["genre_entropy"].median()
    sessions_df["treatment"] = (sessions_df["genre_entropy"] > median_entropy).astype(int)

    weekly = sessions_df.groupby(["week", "treatment"])["returned_7day"].mean().reset_index()
    treat = weekly[weekly["treatment"] == 1]["returned_7day"].values
    control = weekly[weekly["treatment"] == 0]["returned_7day"].values

    min_len = min(len(treat), len(control))
    if min_len < 4:
        return {"parallel_trends_pvalue": None, "parallel_trends_satisfied": None}

    treat_diff = np.diff(treat[:min_len])
    control_diff = np.diff(control[:min_len])
    t_stat, p_value = stats.ttest_ind(treat_diff, control_diff)

    satisfied = p_value > 0.05
    logger.info(f"Parallel trends test: p={p_value:.4f}, satisfied={satisfied}")
    return {
        "parallel_trends_pvalue": round(float(p_value), 4),
        "parallel_trends_satisfied": bool(satisfied),
    }


def difference_in_differences(sessions_df: pd.DataFrame) -> dict:
    """
    DiD estimator for causal effect of content diversity on 7-day return rate.
    Pre-period: first half of data. Post-period: second half.
    """
    sessions_df = sessions_df.copy()
    sessions_df["session_date"] = pd.to_datetime(sessions_df["session_date"])
    median_date = sessions_df["session_date"].median()
    median_entropy = sessions_df["genre_entropy"].median()

    sessions_df["post"] = (sessions_df["session_date"] > median_date).astype(int)
    sessions_df["treatment"] = (sessions_df["genre_entropy"] > median_entropy).astype(int)
    sessions_df["did_interaction"] = sessions_df["post"] * sessions_df["treatment"]

    groups = sessions_df.groupby(["treatment", "post"])["returned_7day"].agg(["mean", "count"])

    try:
        y_treat_post = groups.loc[(1, 1), "mean"]
        y_treat_pre = groups.loc[(1, 0), "mean"]
        y_ctrl_post = groups.loc[(0, 1), "mean"]
        y_ctrl_pre = groups.loc[(0, 0), "mean"]
        did_estimate = (y_treat_post - y_treat_pre) - (y_ctrl_post - y_ctrl_pre)
    except KeyError:
        did_estimate = 0.0
        y_treat_post = y_treat_pre = y_ctrl_post = y_ctrl_pre = 0.0

    n_treat = sessions_df[sessions_df["treatment"] == 1]["returned_7day"]
    n_ctrl = sessions_df[sessions_df["treatment"] == 0]["returned_7day"]
    se = np.sqrt(n_treat.var() / len(n_treat) + n_ctrl.var() / len(n_ctrl))
    t_stat = did_estimate / (se + 1e-9)
    p_value = 2 * (1 - stats.norm.cdf(abs(t_stat)))
    ci_lower = did_estimate - 1.96 * se
    ci_upper = did_estimate + 1.96 * se

    result = {
        "did_estimate": round(float(did_estimate), 4),
        "se": round(float(se), 4),
        "t_stat": round(float(t_stat), 4),
        "p_value": round(float(p_value), 4),
        "ci_lower": round(float(ci_lower), 4),
        "ci_upper": round(float(ci_upper), 4),
        "significant": bool(p_value < 0.05),
        "treat_return_rate_post": round(float(y_treat_post), 4),
        "ctrl_return_rate_post": round(float(y_ctrl_post), 4),
        "n_treatment": int(len(n_treat)),
        "n_control": int(len(n_ctrl)),
    }
    logger.info(f"DiD estimate: {did_estimate:.4f} (p={p_value:.4f}, significant={p_value < 0.05})")
    return result


def propensity_score_matching(sessions_df: pd.DataFrame) -> dict:
    """
    Propensity score matching as robustness check.
    Match treated and control sessions on observable confounders.
    """
    df = sessions_df.copy()
    median_entropy = df["genre_entropy"].median()
    df["treatment"] = (df["genre_entropy"] > median_entropy).astype(int)

    covariates = ["n_tracks", "session_duration_min", "skip_rate", "repeat_rate"]
    available = [c for c in covariates if c in df.columns]
    df_clean = df[available + ["treatment", "returned_7day"]].dropna()

    scaler = StandardScaler()
    X = scaler.fit_transform(df_clean[available])
    y = df_clean["treatment"].values

    lr = LogisticRegression(random_state=42, max_iter=500)
    lr.fit(X, y)
    propensity_scores = lr.predict_proba(X)[:, 1]
    df_clean = df_clean.copy()
    df_clean["propensity_score"] = propensity_scores

    treated = df_clean[df_clean["treatment"] == 1].copy()
    control = df_clean[df_clean["treatment"] == 0].copy()

    matched_controls = []
    caliper = 0.05
    for _, treat_row in treated.iterrows():
        diffs = (control["propensity_score"] - treat_row["propensity_score"]).abs()
        best_match = diffs.idxmin()
        if diffs[best_match] <= caliper:
            matched_controls.append(control.loc[best_match])

    if not matched_controls:
        return {"psm_ate": None, "psm_significant": None, "n_matched": 0}

    matched_ctrl_df = pd.DataFrame(matched_controls)
    ate = treated["returned_7day"].mean() - matched_ctrl_df["returned_7day"].mean()
    t_stat, p_value = stats.ttest_ind(
        treated["returned_7day"], matched_ctrl_df["returned_7day"]
    )

    result = {
        "psm_ate": round(float(ate), 4),
        "psm_t_stat": round(float(t_stat), 4),
        "psm_p_value": round(float(p_value), 4),
        "psm_significant": bool(p_value < 0.05),
        "n_matched": len(matched_controls),
        "n_treated": len(treated),
    }
    logger.info(f"PSM ATE: {ate:.4f} (p={p_value:.4f})")
    return result


def bayesian_ab_test(sessions_df: pd.DataFrame) -> dict:
    """
    Bayesian Beta-Binomial A/B test on return rates.
    Posterior probability that high-diversity treatment > control.
    """
    median_entropy = sessions_df["genre_entropy"].median()
    treat = sessions_df[sessions_df["genre_entropy"] > median_entropy]["returned_7day"]
    ctrl = sessions_df[sessions_df["genre_entropy"] <= median_entropy]["returned_7day"]

    n_treat_success = int(treat.sum())
    n_treat_total = int(len(treat))
    n_ctrl_success = int(ctrl.sum())
    n_ctrl_total = int(len(ctrl))

    with pm.Model() as model:
        p_treat = pm.Beta("p_treat", alpha=1 + n_treat_success,
                           beta=1 + n_treat_total - n_treat_success)
        p_ctrl = pm.Beta("p_ctrl", alpha=1 + n_ctrl_success,
                          beta=1 + n_ctrl_total - n_ctrl_success)
        delta = pm.Deterministic("delta", p_treat - p_ctrl)
        trace = pm.sample(2000, tune=500, chains=2, progressbar=False, random_seed=42)

    delta_samples = trace.posterior["delta"].values.flatten()
    prob_positive = float((delta_samples > 0).mean())
    hdi_result = az.hdi(delta_samples, prob=0.95)
    median_delta = float(np.median(delta_samples))

    result = {
        "prob_treatment_better": round(prob_positive, 4),
        "median_delta": round(median_delta, 4),
        "hdi_lower": round(float(hdi_result[0]), 4),
        "hdi_upper": round(float(hdi_result[1]), 4),
        "treat_return_rate": round(float(treat.mean()), 4),
        "ctrl_return_rate": round(float(ctrl.mean()), 4),
        "n_treat": n_treat_total,
        "n_ctrl": n_ctrl_total,
    }
    logger.info(f"Bayesian: P(treatment>control)={prob_positive:.4f}, "
                f"median delta={median_delta:.4f}")
    return result


def run_causal_analysis(sessions_df: pd.DataFrame) -> dict:
    mlflow.set_experiment("contentpulse_causal")
    with mlflow.start_run(run_name="diversity_effect_on_retention"):
        parallel = test_parallel_trends(sessions_df)
        did = difference_in_differences(sessions_df)
        psm = propensity_score_matching(sessions_df)
        bayesian = bayesian_ab_test(sessions_df)

        for k, v in {**did, **psm, **bayesian}.items():
            if isinstance(v, (int, float)) and v is not None:
                try:
                    mlflow.log_metric(k, v)
                except Exception:
                    pass

        return {
            "parallel_trends": parallel,
            "did": did,
            "psm": psm,
            "bayesian": bayesian,
        }
