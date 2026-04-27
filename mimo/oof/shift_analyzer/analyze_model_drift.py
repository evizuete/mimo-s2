#!/usr/bin/env python3
"""
Analyze model drift between TRAIN and HOLDOUT.

Outputs:
    drift_report.json
    drift_summary.csv
"""

import argparse
import json
import numpy as np
import pandas as pd


# -----------------------------
# PSI calculation
# -----------------------------
def calculate_psi(train, holdout, bins=20):

    quantiles = np.linspace(0, 1, bins + 1)
    breakpoints = np.quantile(train, quantiles)

    train_counts, _ = np.histogram(train, breakpoints)
    hold_counts, _ = np.histogram(holdout, breakpoints)

    train_perc = train_counts / len(train)
    hold_perc = hold_counts / len(holdout)

    psi_values = (train_perc - hold_perc) * np.log(
        (train_perc + 1e-6) / (hold_perc + 1e-6)
    )

    return np.sum(psi_values)


# -----------------------------
# base rate drift
# -----------------------------
def base_rate_drift(train, holdout):

    r_train = train.mean()
    r_hold = holdout.mean()

    return {
        "train_pos_rate": float(r_train),
        "holdout_pos_rate": float(r_hold),
        "relative_change": float((r_hold - r_train) / r_train)
    }


# -----------------------------
# score drift
# -----------------------------
def score_drift(train, holdout):
    train = train.dropna()
    holdout = holdout.dropna()
    return {
        "mean_train": float(train.mean()),
        "mean_holdout": float(holdout.mean()),
        "std_train": float(train.std()),
        "std_holdout": float(holdout.std()),
        "p95_train": float(np.percentile(train, 95)),
        "p95_holdout": float(np.percentile(holdout, 95)),
    }


# -----------------------------
# drift by state
# -----------------------------
def drift_by_state(df):

    rows = []

    for state, g in df.groupby("state"):

        train = g[g["period"] == "train"]
        hold = g[g["period"] == "holdout"]

        if len(train) < 100 or len(hold) < 100:
            continue

        psi = calculate_psi(
            train["score"].dropna().astype(float),
            hold["score"].dropna().astype(float)
        )

        rows.append({
            "state": state,
            "psi": psi,
            "train_n": len(train),
            "holdout_n": len(hold),
        })

    return pd.DataFrame(rows)


# -----------------------------
# drift alert logic
# -----------------------------
def drift_alert(psi, base_rate_change):

    alerts = []

    if psi > 0.25:
        alerts.append("SEVERE_SCORE_DRIFT")

    elif psi > 0.10:
        alerts.append("MODERATE_SCORE_DRIFT")

    if abs(base_rate_change) > 0.15:
        alerts.append("BASE_RATE_DRIFT")

    if not alerts:
        alerts.append("STABLE")

    return alerts


# -----------------------------
# main
# -----------------------------
def main():

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True)
    parser.add_argument("--score-col", default="oof_proba_cal")
    parser.add_argument("--target-col", default="signal")
    parser.add_argument("--time-col", default="time")
    parser.add_argument("--state-col", default="state")
    parser.add_argument("--train-end", required=True)
    parser.add_argument("--holdout-start", required=True)

    args = parser.parse_args()

    df = pd.read_parquet(args.input)

    df[args.time_col] = pd.to_datetime(df[args.time_col])

    train = df[df[args.time_col] <= args.train_end].copy()
    holdout = df[df[args.time_col] >= args.holdout_start].copy()

    train["period"] = "train"
    holdout["period"] = "holdout"

    df_all = pd.concat([train, holdout])

    score_train = train[args.score_col].dropna().astype(float)
    score_hold = holdout[args.score_col].dropna().astype(float)

    # PSI
    psi = calculate_psi(score_train, score_hold)

    # base rate
    br = base_rate_drift(train[args.target_col], holdout[args.target_col])

    # score drift
    sc = score_drift(score_train, score_hold)

    # state drift
    df_states = drift_by_state(
        df_all.rename(columns={
            args.score_col: "score",
            args.state_col: "state"
        })
    )

    # alerts
    alerts = drift_alert(psi, br["relative_change"])

    report = {
        "psi": psi,
        "base_rate": br,
        "score_stats": sc,
        "alerts": alerts
    }

    print("\nMODEL DRIFT REPORT")
    print("-------------------")
    print(json.dumps(report, indent=2))

    df_states.to_csv("drift_by_state.csv", index=False)

    with open("drift_report.json", "w") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()