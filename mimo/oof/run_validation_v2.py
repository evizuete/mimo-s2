#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


def read_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")
    suf = path.suffix.lower()
    if suf == ".parquet":
        return pd.read_parquet(path)
    if suf in {".csv", ".txt"}:
        return pd.read_csv(path)
    if suf in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"Formato no soportado: {path.suffix}")


def load_gate_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"No existe gate config: {path}")

    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    if path.suffix.lower() == ".py":
        spec = importlib.util.spec_from_file_location("gate_cfg_mod", str(path))
        if spec is None or spec.loader is None:
            raise RuntimeError(f"No se pudo importar {path}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        if hasattr(mod, "gate_by_action_and_state"):
            return mod.gate_by_action_and_state
        raise AttributeError(f"{path} no contiene gate_by_action_and_state")

    raise ValueError(f"Formato de gate config no soportado: {path.suffix}")


def normalize_state_name(x: Any) -> str:
    return str(x).strip().lower()


def evaluate_subset(y_true: np.ndarray, y_score: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]

    if len(y_true) == 0:
        return {
            "n": 0,
            "base_rate": float("nan"),
            "auc_pr": float("nan"),
            "auc_roc": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "signal_rate": float("nan"),
            "tp": 0,
            "fp": 0,
            "fn": 0,
            "tn": 0,
        }

    pred = np.ones_like(y_true, dtype=int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())

    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)

    try:
        auc_pr = float(average_precision_score(y_true, y_score))
    except Exception:
        auc_pr = float("nan")

    try:
        auc_roc = float(roc_auc_score(y_true, y_score))
    except Exception:
        auc_roc = float("nan")

    return {
        "n": int(len(y_true)),
        "base_rate": float(y_true.mean()),
        "auc_pr": auc_pr,
        "auc_roc": auc_roc,
        "precision": float(precision),
        "recall": float(recall),
        "signal_rate": 1.0,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def build_threshold_lookup(
    train_df: pd.DataFrame,
    score_col: str,
    state_col: str,
    state_to_percentile: dict[str, int],
    global_percentile: int,
) -> dict[str, float]:
    out: dict[str, float] = {}

    scores_global = train_df[score_col].astype(float).to_numpy()
    scores_global = scores_global[np.isfinite(scores_global)]
    if len(scores_global) == 0:
        raise ValueError("No hay scores válidos en train para construir thresholds")

    out["_global"] = float(np.percentile(scores_global, global_percentile))

    train_df = train_df.copy()
    train_df["_state_norm"] = train_df[state_col].map(normalize_state_name)

    for state, pct in state_to_percentile.items():
        if state == "_global":
            continue
        s = train_df.loc[train_df["_state_norm"] == state, score_col].astype(float).to_numpy()
        s = s[np.isfinite(s)]
        out[state] = float(np.percentile(s, pct)) if len(s) else out["_global"]

    return out


def apply_gate(
    holdout_df: pd.DataFrame,
    train_df: pd.DataFrame,
    score_col: str,
    target_col: str,
    state_col: str,
    side_name: str,
    gate_cfg: dict[str, Any],
) -> tuple[pd.DataFrame, dict[str, float], pd.DataFrame]:
    if "production" not in gate_cfg or side_name not in gate_cfg["production"]:
        raise KeyError(f"No existe gate config para production/{side_name}")

    side_cfg_raw = gate_cfg["production"][side_name]
    side_cfg = {normalize_state_name(k): int(v) for k, v in side_cfg_raw.items()}
    global_pct = int(side_cfg.get("_global", 95))

    thresholds = build_threshold_lookup(
        train_df=train_df,
        score_col=score_col,
        state_col=state_col,
        state_to_percentile=side_cfg,
        global_percentile=global_pct,
    )

    hold = holdout_df.copy()
    hold["_state_norm"] = hold[state_col].map(normalize_state_name)

    hold["gate_threshold"] = hold["_state_norm"].map(lambda s: thresholds.get(s, thresholds["_global"]))
    hold["gate_required_percentile"] = hold["_state_norm"].map(lambda s: side_cfg.get(s, global_pct))
    #hold["accepted"] = hold[score_col].astype(float) >= hold["gate_threshold"].astype(float)

    NO_TRADE_STATES = {"low_vol", "volatile", 'range'}
    hold["accepted"] = (~hold["_state_norm"].isin(NO_TRADE_STATES)) & (hold[score_col].astype(float) >= hold["gate_threshold"].astype(float))
    accepted = hold[hold["accepted"]].copy()

    metrics = evaluate_subset(
        accepted[target_col].to_numpy(),
        accepted[score_col].to_numpy(),
    )
    metrics["accepted_n"] = int(len(accepted))
    metrics["accepted_rate"] = float(len(accepted) / max(len(hold), 1))
    metrics["input_n"] = int(len(hold))

    rows = []
    for st, g in hold.groupby("_state_norm"):
        ga = g[g["accepted"]]
        rows.append(
            {
                "state": st,
                "input_n": int(len(g)),
                "accepted_n": int(len(ga)),
                "accepted_rate": float(len(ga) / max(len(g), 1)),
                "required_percentile": int(side_cfg.get(st, global_pct)),
                "threshold_used": float(thresholds.get(st, thresholds["_global"])),
                "precision_post_gate": float(ga[target_col].mean()) if len(ga) else float("nan"),
                "mean_score_post_gate": float(ga[score_col].mean()) if len(ga) else float("nan"),
            }
        )

    by_state = pd.DataFrame(rows).sort_values(["accepted_rate", "state"], ascending=[False, True])
    return accepted, metrics, by_state


def print_report(side: str, metrics: dict[str, float], by_state: pd.DataFrame) -> None:
    print("\n" + "=" * 60)
    print(f"{side.upper()} HOLDOUT VALIDATION WITH DECISION GATES")
    print("=" * 60)
    print(f"input_n       : {metrics['input_n']:,}")
    print(f"accepted_n    : {metrics['accepted_n']:,}")
    print(f"accepted_rate : {metrics['accepted_rate']:.4f}")
    print(f"base_rate     : {metrics['base_rate']:.4f}")
    print(f"AUC-ROC       : {metrics['auc_roc']:.4f}")
    print(f"AUC-PR        : {metrics['auc_pr']:.4f}")
    print(f"precision     : {metrics['precision']:.4f}")
    print(f"recall        : {metrics['recall']:.4f}")
    print(f"TP/FP/FN/TN   : {metrics['tp']}/{metrics['fp']}/{metrics['fn']}/{metrics['tn']}")

    if not by_state.empty:
        cols = [
            "state",
            "input_n",
            "accepted_n",
            "accepted_rate",
            "required_percentile",
            "threshold_used",
            "precision_post_gate",
        ]
        print("\nBy state:")
        print(by_state[cols].to_string(index=False))


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Validación holdout con decision gates")
    ap.add_argument("--release", default="200373")
    ap.add_argument("--base-dir", default=None, help="Ruta base al deploy_full. Si se omite usa ./artifacts/<release>/oof/deploy_full")
    ap.add_argument("--gate-config", required=False, default="./decision_engine_percentiles.py", help="Path a gate_by_action_and_state.py o .json")
    ap.add_argument("--long-holdout", default=None)
    ap.add_argument("--short-holdout", default=None)
    ap.add_argument("--long-train", default=None)
    ap.add_argument("--short-train", default=None)
    ap.add_argument("--score-col", default="y_pred_cal")
    ap.add_argument("--target-col", default="y_true")
    ap.add_argument("--state-col", default="state")
    return ap.parse_args()


def main() -> None:
    args = parse_args()

    base_dir = Path(args.base_dir) if args.base_dir else Path("../../artifacts") / args.release / "oof" / "deploy_full"
    data_dir = base_dir / "data"

    long_holdout_path = Path(args.long_holdout) if args.long_holdout else data_dir / f"holdout_predictions_{args.release}_long.parquet"
    short_holdout_path = Path(args.short_holdout) if args.short_holdout else data_dir / f"holdout_predictions_{args.release}_short.parquet"
    long_train_path = Path(args.long_train) if args.long_train else base_dir / f"oof_{args.release}_long.parquet"
    short_train_path = Path(args.short_train) if args.short_train else base_dir / f"oof_{args.release}_short.parquet"

    gate_cfg = load_gate_config(Path(args.gate_config))

    print("📥 Cargando datasets...")
    df_long_hold = read_table(long_holdout_path)
    df_short_hold = read_table(short_holdout_path)
    df_long_train = read_table(long_train_path).rename(columns={"oof_proba_cal": args.score_col, "signal": args.target_col})
    df_short_train = read_table(short_train_path).rename(columns={"oof_proba_cal": args.score_col, "signal": args.target_col})

    accepted_long, metrics_long, by_state_long = apply_gate(
        holdout_df=df_long_hold,
        train_df=df_long_train,
        score_col=args.score_col,
        target_col=args.target_col,
        state_col=args.state_col,
        side_name="long",
        gate_cfg=gate_cfg,
    )

    accepted_short, metrics_short, by_state_short = apply_gate(
        holdout_df=df_short_hold,
        train_df=df_short_train,
        score_col=args.score_col,
        target_col=args.target_col,
        state_col=args.state_col,
        side_name="short",
        gate_cfg=gate_cfg,
    )

    print_report("long", metrics_long, by_state_long)
    print_report("short", metrics_short, by_state_short)

    reports_dir = base_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    accepted_long.to_parquet(reports_dir / "holdout_long_post_gate.parquet", index=False)
    accepted_short.to_parquet(reports_dir / "holdout_short_post_gate.parquet", index=False)
    by_state_long.to_csv(reports_dir / "holdout_long_post_gate_by_state.csv", index=False)
    by_state_short.to_csv(reports_dir / "holdout_short_post_gate_by_state.csv", index=False)

    summary = {
        "release": args.release,
        "gate_config": str(args.gate_config),
        "long": metrics_long,
        "short": metrics_short,
    }

    with open(reports_dir / "holdout_post_gate_validation.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\n✅ Guardado resumen: {reports_dir / 'holdout_post_gate_validation.json'}")


if __name__ == "__main__":
    main()
