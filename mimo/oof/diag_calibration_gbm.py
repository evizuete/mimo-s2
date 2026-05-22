"""
diag_calibration_gbm.py
═══════════════════════════════════════════════════════════════════════════
Diagnóstico de calibración: ¿la distribución de probs sobre holdout está
alineada con OOF, o se ha desplazado?

MÉTRICAS:
  · Brier score (train, holdout)
  · Expected Calibration Error (ECE) en 10 bins quantile-based
  · Reliability diagram (predicted vs empirical TP rate por bin)
  · Histograma TEXTUAL de probs cal en train vs holdout (10 buckets)

DIAGNOSIS ESPERADAS:
  · Histograma train ≈ holdout → distribución estable, sin shift
  · Histograma holdout desplazado a la izquierda → modelo más cauto post-train
    (no produce probs altas) → thr_OOF queda demasiado alto → 0 signals
  · ECE_train bajo + ECE_holdout alto → calibrator overfit a OOF
  · Reliability: empirical TP rate por bin debería seguir la diagonal y=x

USO:
  python -m mimo.oof.diag_calibration_gbm \\
    --release 202602_GBM --inherit-config-from 202601 \\
    --best-json artifacts/202602_GBM/oof/<tag>/reports/best_per_side.json \\
    --train-from 2024-01-01 --train-to 2025-10-30 \\
    --holdout-from 2025-11-01 --holdout-to 2026-04-10 \\
    --out-json artifacts/202602_GBM/oof/<tag>/reports/calibration_diag.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import joblib
import numpy as np
import optuna
import pandas as pd

from mimo.data_managers.databases import Database
from mimo.data_managers.data_manager import DataManager
from mimo.data_managers.data_pipeline_v2 import DataPipeline
from mimo.features.feature_builder import FeatureConfig
from mimo.models.model_builder import Config, ModelConfig
from mimo.states_manager.state_detector import StateConfig
from mimo.oof.main_oof_gbm_holdout import (
    _split_trial_params, _collect_tabular_columns,
    _refit_with_internal_val,
)
from mimo.oof.main_oof_regime_weights_v7 import (
    BARRIERS_BY_RELEASE, LONG_VARIANTS, SHORT_VARIANTS,
    _VOL_INVARIANT_RELEASES, _REDUCED_FEATURES_RELEASES,
    _ULTRA_REDUCED_FEATURES_RELEASES,
    _get_barriers_for_release, _get_feature_masks_for_release, _tf_defaults,
    install_regime_weight_patch, resolve_regime_weights, set_global_seeds,
)

# Side-effect: inyectar _GBM_BARRIERS_BY_RELEASE.
from mimo.oof.main_oof_gbm import _GBM_EXCLUDE_PATTERNS  # noqa: F401


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _brier(p: np.ndarray, y: np.ndarray) -> float:
    if len(p) == 0: return float("nan")
    return float(np.mean((p - y) ** 2))


def _reliability(p: np.ndarray, y: np.ndarray, n_bins: int = 10) -> Dict[str, Any]:
    """Bins quantile-based. Devuelve mean_pred, empirical, count por bin + ECE."""
    if len(p) == 0:
        return {"bins": [], "ece": float("nan"), "n": 0}
    # Bins por quantiles para tener N comparable por bin
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    if len(edges) < 2:
        return {"bins": [], "ece": float("nan"), "n": int(len(p))}
    bin_ids = np.digitize(p, edges[1:-1], right=False)
    bins = []
    ece = 0.0
    N = len(p)
    for b in range(len(edges) - 1):
        mask = (bin_ids == b)
        n = int(mask.sum())
        if n == 0: continue
        mean_pred = float(p[mask].mean())
        empirical = float(y[mask].mean())
        bins.append({"bin": b, "n": n,
                     "edge_lo": float(edges[b]), "edge_hi": float(edges[b+1]),
                     "mean_pred": mean_pred, "empirical": empirical,
                     "gap": empirical - mean_pred})
        ece += (n / N) * abs(empirical - mean_pred)
    return {"bins": bins, "ece": float(ece), "n": int(N)}


def _hist_text(p: np.ndarray, n_buckets: int = 10, width: int = 50) -> str:
    """Histograma textual con buckets fijos [0,1]."""
    edges = np.linspace(0, 1, n_buckets + 1)
    counts, _ = np.histogram(p, bins=edges)
    max_count = max(counts.max(), 1)
    lines = []
    for i in range(n_buckets):
        bar = "█" * int(round(width * counts[i] / max_count))
        lines.append(f"  [{edges[i]:.2f}-{edges[i+1]:.2f}) | {counts[i]:>7d} | {bar}")
    return "\n".join(lines)


def _print_side_diag(side: str, p_tr: np.ndarray, y_tr: np.ndarray,
                     p_hd: np.ndarray, y_hd: np.ndarray, thr_oof: float) -> Dict[str, Any]:
    print(f"\n══ {side.upper()} ══════════════════════════════════════════════════════")
    rel_tr = _reliability(p_tr, y_tr)
    rel_hd = _reliability(p_hd, y_hd)
    brier_tr = _brier(p_tr, y_tr)
    brier_hd = _brier(p_hd, y_hd)
    print(f"  thr_OOF aplicado en holdout: {thr_oof:.4f}")
    print(f"  Brier   train={brier_tr:.4f}  holdout={brier_hd:.4f}")
    print(f"  ECE     train={rel_tr['ece']:.4f}  holdout={rel_hd['ece']:.4f}")
    print(f"  n_above_thr_OOF (train, holdout): {int((p_tr>=thr_oof).sum())}, "
          f"{int((p_hd>=thr_oof).sum())}")
    print(f"  p99 (train, holdout): {np.quantile(p_tr,0.99):.4f}, "
          f"{np.quantile(p_hd,0.99):.4f}")
    print(f"\n  Histograma TRAIN (probs cal):")
    print(_hist_text(p_tr))
    print(f"\n  Histograma HOLDOUT (probs cal):")
    print(_hist_text(p_hd))
    print(f"\n  Reliability HOLDOUT  (bin: predicted vs empirical TP rate)")
    print(f"  {'bin':>3} {'n':>6} {'mean_pred':>10} {'empirical':>10} {'gap':>9}")
    for b in rel_hd["bins"]:
        print(f"  {b['bin']:>3} {b['n']:>6} {b['mean_pred']:>10.4f} "
              f"{b['empirical']:>10.4f} {b['gap']:>+9.4f}")
    return {"brier_train": brier_tr, "brier_holdout": brier_hd,
            "ece_train": rel_tr["ece"], "ece_holdout": rel_hd["ece"],
            "reliability_train": rel_tr, "reliability_holdout": rel_hd,
            "p99_train": float(np.quantile(p_tr, 0.99)),
            "p99_holdout": float(np.quantile(p_hd, 0.99)),
            "n_above_thr_train": int((p_tr >= thr_oof).sum()),
            "n_above_thr_holdout": int((p_hd >= thr_oof).sum())}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", required=True)
    ap.add_argument("--inherit-config-from", default=None)
    ap.add_argument("--best-json", required=True)
    ap.add_argument("--base-tf", default="5min")
    ap.add_argument("--variant-long", choices=sorted(LONG_VARIANTS.keys()), default="moderate")
    ap.add_argument("--variant-short", choices=sorted(SHORT_VARIANTS.keys()), default="moderate")
    ap.add_argument("--regime-weights-long-json", default=None)
    ap.add_argument("--regime-weights-short-json", default=None)
    ap.add_argument("--label-horizon-long", type=int, default=3)
    ap.add_argument("--label-horizon-short", type=int, default=3)
    ap.add_argument("--train-from", type=_parse_date, required=True)
    ap.add_argument("--train-to", type=_parse_date, required=True)
    ap.add_argument("--holdout-from", type=_parse_date, required=True)
    ap.add_argument("--holdout-to", type=_parse_date, required=True)
    ap.add_argument("--optuna-storage", default=os.environ.get(
        "OPTUNA_STORAGE",
        "mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db",
    ))
    ap.add_argument("--study-prefix", default="oof_study_gbm")
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--out-json", default=None)
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    set_global_seeds(int(args.seed))
    release = str(args.release)
    base_tf = str(args.base_tf)

    if args.inherit_config_from:
        src, dst = str(args.inherit_config_from), release
        if dst != src:
            if src in BARRIERS_BY_RELEASE and dst not in BARRIERS_BY_RELEASE:
                BARRIERS_BY_RELEASE[dst] = BARRIERS_BY_RELEASE[src]
            if src in _VOL_INVARIANT_RELEASES:  _VOL_INVARIANT_RELEASES.add(dst)
            if src in _REDUCED_FEATURES_RELEASES: _REDUCED_FEATURES_RELEASES.add(dst)
            if src in _ULTRA_REDUCED_FEATURES_RELEASES: _ULTRA_REDUCED_FEATURES_RELEASES.add(dst)
            print(f"🧬 [INHERIT-CONFIG] '{dst}' ← '{src}'")

    # Load best trial info
    with open(args.best_json) as fh: best = json.load(fh)
    top_long, top_short = best["top_long"][0], best["top_short"][0]
    n_long, n_short = int(top_long["trial"]), int(top_short["trial"])
    thr_long  = float(top_long["ev_long"]["thr"])
    thr_short = float(top_short["ev_short"]["thr"])

    study_name = f"{args.study_prefix}_{release}_multitask"
    study = optuna.load_study(study_name=study_name, storage=args.optuna_storage)
    tr_long_obj  = next(t for t in study.trials if t.number == n_long)
    tr_short_obj = next(t for t in study.trials if t.number == n_short)
    lgbm_long,  meta_long  = _split_trial_params(tr_long_obj.params,  seed=int(args.seed))
    lgbm_short, meta_short = _split_trial_params(tr_short_obj.params, seed=int(args.seed))
    cal_long  = joblib.load(tr_long_obj.user_attrs["cal_long_path"])
    cal_short = joblib.load(tr_short_obj.user_attrs["cal_short_path"])

    regime_weights_by_side = resolve_regime_weights(args)
    install_regime_weight_patch(regime_weights_by_side, verbose=False)

    print(f"\n📊 Cargando OHLCV {args.train_from.date()} → {args.holdout_to.date()}")
    db = Database()
    resample = None if str(base_tf).lower() in ("1min", "1m") else base_tf
    dm = DataManager.from_database_historical_2(
        db, from_date=args.train_from, to_date=args.holdout_to, resample=resample
    )
    df_rates = dm.df.copy()
    df_rates["time"] = pd.to_datetime(df_rates["time"])

    barriers = _get_barriers_for_release(release)
    tf_defaults = _tf_defaults(base_tf)
    feature_config = FeatureConfig(
        ema_periods=[9, 21, 50],
        label_method="triple_barrier_dual",
        label_horizon=max(args.label_horizon_long, args.label_horizon_short),
        tp_barrier=barriers["tp_base"], sl_barrier=barriers["sl_base"],
        label_method_long="triple_barrier_dual",
        regime_barriers_long=barriers["regime_barriers_long"],
        label_method_short="triple_barrier_dual",
        regime_barriers_short=barriers["regime_barriers_short"],
        tp_barrier_short=None, sl_barrier_short=None,
        feature_masks=_get_feature_masks_for_release(release),
        price_norm_window=tf_defaults["price_norm_window"],
        use_vol_invariant_features=(release in _VOL_INVARIANT_RELEASES),
        use_reduced_features=(release in _REDUCED_FEATURES_RELEASES),
        use_ultra_reduced_features=(release in _ULTRA_REDUCED_FEATURES_RELEASES),
    )
    general_config = Config(release=release, use_oof=True, oof_splits=5)
    regime_config = StateConfig(adx_trend_threshold=25.0)
    pipeline = DataPipeline(
        general_config=general_config, feature_config=feature_config,
        model_config=ModelConfig(seq_len_short=64, seq_len_long=256, target_type="multitask"),
        regime_config=regime_config,
    )
    df_prepared = pipeline.prepare_data(df_rates, labels=True, side="both",
                                        set_market_condition=False, ensure_regime=True)
    feat_cols = [c for c in _collect_tabular_columns(pipeline) if c in df_prepared.columns]
    _exclude = _GBM_EXCLUDE_PATTERNS.get(release)
    if _exclude:
        feat_cols = [c for c in feat_cols if not any(p in c for p in _exclude)]
        print(f"🚫 [EXCLUDE] release={release} → {len(feat_cols)} features")
    needed = feat_cols + ["time", "high", "low", "close", "atr", "signal_long", "signal_short"]
    df_clean = df_prepared.loc[df_prepared[needed].notna().all(axis=1)].reset_index(drop=True)

    tr_mask = df_clean["time"] < pd.Timestamp(args.holdout_from)
    hd_mask = df_clean["time"] >= pd.Timestamp(args.holdout_from)
    df_tr, df_hd = df_clean.loc[tr_mask].reset_index(drop=True), df_clean.loc[hd_mask].reset_index(drop=True)

    X_tr = df_tr[feat_cols].astype(np.float32).values
    X_hd = df_hd[feat_cols].astype(np.float32).values
    y_long_tr  = df_tr["signal_long"].astype(np.int8).values
    y_short_tr = df_tr["signal_short"].astype(np.int8).values
    y_long_hd  = df_hd["signal_long"].astype(np.int8).values
    y_short_hd = df_hd["signal_short"].astype(np.int8).values
    w_tr = df_tr["regime_weight"].astype(np.float32).values \
        if "regime_weight" in df_tr.columns else np.ones(len(df_tr), dtype=np.float32)

    # Refit + predict
    print(f"\n🌳 Refit LONG (trial #{n_long})...")
    booster_long, _ = _refit_with_internal_val(
        X=X_tr, y=y_long_tr, w=w_tr, lgbm_params=lgbm_long,
        n_estimators=int(meta_long.get("n_estimators", 500)),
        early_stopping_rounds=int(meta_long.get("early_stopping_rounds", 50)),
    )
    print(f"🌳 Refit SHORT (trial #{n_short})...")
    booster_short, _ = _refit_with_internal_val(
        X=X_tr, y=y_short_tr, w=w_tr, lgbm_params=lgbm_short,
        n_estimators=int(meta_short.get("n_estimators", 500)),
        early_stopping_rounds=int(meta_short.get("early_stopping_rounds", 50)),
    )

    p_long_tr_cal  = cal_long.predict(booster_long.predict(X_tr).astype(np.float64)).astype(np.float32)
    p_long_hd_cal  = cal_long.predict(booster_long.predict(X_hd).astype(np.float64)).astype(np.float32)
    p_short_tr_cal = cal_short.predict(booster_short.predict(X_tr).astype(np.float64)).astype(np.float32)
    p_short_hd_cal = cal_short.predict(booster_short.predict(X_hd).astype(np.float64)).astype(np.float32)

    long_diag  = _print_side_diag("long",  p_long_tr_cal,  y_long_tr,
                                  p_long_hd_cal,  y_long_hd,  thr_long)
    short_diag = _print_side_diag("short", p_short_tr_cal, y_short_tr,
                                  p_short_hd_cal, y_short_hd, thr_short)

    # Veredicto compacto
    print("\n══ VEREDICTO ════════════════════════════════════════════════════════")
    for side, diag, thr in (("LONG ", long_diag, thr_long), ("SHORT", short_diag, thr_short)):
        ratio = diag["n_above_thr_holdout"] / max(diag["n_above_thr_train"], 1)
        train_rate = diag["n_above_thr_train"] / len(p_long_tr_cal if side == "LONG " else p_short_tr_cal)
        hold_rate  = diag["n_above_thr_holdout"] / len(p_long_hd_cal if side == "LONG " else p_short_hd_cal)
        ece_delta = diag["ece_holdout"] - diag["ece_train"]
        print(f"  {side}: thr_OOF={thr:.4f} | "
              f"sig_rate train={train_rate:.4f} → hold={hold_rate:.4f} (ratio={ratio:.2f}) | "
              f"ECE_train={diag['ece_train']:.3f} ECE_hold={diag['ece_holdout']:.3f} (Δ={ece_delta:+.3f})")
        if ratio < 0.3:
            print(f"    ⚠️  DISTRIBUTION SHIFT: holdout produce {ratio:.0%} de sig que train → calibrator no extrapola")
        elif ece_delta > 0.05:
            print(f"    ⚠️  CALIBRATOR OVERFIT: ECE empeora {ece_delta:.3f} en holdout")
        else:
            print(f"    ✅  Calibración estable train→hold; degradación es de señal genuina")

    report = {
        "release": release, "long_trial": n_long, "short_trial": n_short,
        "thr_long": thr_long, "thr_short": thr_short,
        "long":  long_diag, "short": short_diag,
    }
    out_json = args.out_json or str(Path(args.best_json).parent / "calibration_diag.json")
    with open(out_json, "w") as fh: json.dump(report, fh, indent=2, default=str)
    print(f"\n📁 Reporte: {out_json}")


if __name__ == "__main__":
    main()
