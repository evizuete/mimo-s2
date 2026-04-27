from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score, precision_recall_curve


RELEASE = "200372"
BASE_DIR = Path("../../artifacts") / RELEASE / "oof" / "deploy_full"
DATA_DIR = BASE_DIR / "data"

LONG_PATH = DATA_DIR / f"holdout_predictions_{RELEASE}_long.parquet"
SHORT_PATH = DATA_DIR / f"holdout_predictions_{RELEASE}_short.parquet"


def load_preds(path: Path, side: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe: {path}")
    df = pd.read_parquet(path).copy()
    df["side"] = side
    return df


def evaluate_probs(y_true: np.ndarray, y_score: np.ndarray) -> dict:
    y_true = np.asarray(y_true).astype(int)
    y_score = np.asarray(y_score).astype(float)

    mask = np.isfinite(y_true) & np.isfinite(y_score)
    y_true = y_true[mask]
    y_score = y_score[mask]

    auc_pr = float(average_precision_score(y_true, y_score))
    auc_roc = float(roc_auc_score(y_true, y_score))

    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    f1 = 2 * precision[:-1] * recall[:-1] / (precision[:-1] + recall[:-1] + 1e-12)
    best_idx = int(np.argmax(f1))
    best_thr = float(thresholds[best_idx])

    pred = (y_score >= best_thr).astype(int)
    tp = int(((pred == 1) & (y_true == 1)).sum())
    fp = int(((pred == 1) & (y_true == 0)).sum())
    fn = int(((pred == 0) & (y_true == 1)).sum())
    tn = int(((pred == 0) & (y_true == 0)).sum())

    precision_at_thr = tp / max(tp + fp, 1)
    recall_at_thr = tp / max(tp + fn, 1)
    signal_rate = float(pred.mean())
    base_rate = float(y_true.mean())

    return {
        "n": int(len(y_true)),
        "auc_pr": auc_pr,
        "auc_roc": auc_roc,
        "base_rate": base_rate,
        "best_threshold_f1": best_thr,
        "precision": float(precision_at_thr),
        "recall": float(recall_at_thr),
        "signal_rate": signal_rate,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def print_report(side: str, metrics: dict):
    print("\n" + "=" * 50)
    print(f"{side.upper()} HOLDOUT VALIDATION")
    print("=" * 50)
    print(f"n           : {metrics['n']:,}")
    print(f"base_rate   : {metrics['base_rate']:.4f}")
    print(f"AUC-ROC     : {metrics['auc_roc']:.4f}")
    print(f"AUC-PR      : {metrics['auc_pr']:.4f}")
    print(f"best_thr_f1 : {metrics['best_threshold_f1']:.4f}")
    print(f"precision   : {metrics['precision']:.4f}")
    print(f"recall      : {metrics['recall']:.4f}")
    print(f"signal_rate : {metrics['signal_rate']:.4f}")
    print(f"TP/FP/FN/TN : {metrics['tp']}/{metrics['fp']}/{metrics['fn']}/{metrics['tn']}")


def main():
    print("📥 Cargando holdout predictions...")
    df_long = load_preds(LONG_PATH, "long")
    df_short = load_preds(SHORT_PATH, "short")

    print(f"  ✅ LONG : {len(df_long):,} filas")
    print(f"  ✅ SHORT: {len(df_short):,} filas")

    # columnas esperadas: y_true, y_pred_cal
    for name, df in [("long", df_long), ("short", df_short)]:
        missing = [c for c in ["y_true", "y_pred_cal"] if c not in df.columns]
        if missing:
            raise ValueError(f"{name}: faltan columnas {missing}")

    metrics_long = evaluate_probs(df_long["y_true"].values, df_long["y_pred_cal"].values)
    metrics_short = evaluate_probs(df_short["y_true"].values, df_short["y_pred_cal"].values)

    print_report("long", metrics_long)
    print_report("short", metrics_short)

    out = {
        "release": RELEASE,
        "long": metrics_long,
        "short": metrics_short,
    }

    out_path = BASE_DIR / "reports" / "holdout_predictions_validation.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print(f"\n✅ Guardado: {out_path}")


if __name__ == "__main__":
    main()