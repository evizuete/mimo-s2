"""
main_oof_gbm_ensemble.py
═══════════════════════════════════════════════════════════════════════════
FASE 4 GBM: ensemble de top-N trials para reducir varianza vs picar best.

¿POR QUÉ ensemble?
  Picar el "trial #1" sobre 10-80 muestras tiene selection bias optimista.
  Promediar los TOP-N reduce la varianza:
    · Si los top-N coinciden → señal robusta, ensemble ≈ trial #1.
    · Si los top-N divergen  → ensemble suaviza el ruido, mejor extrapolación.

ESTRATEGIA:
  Para cada side (LONG, SHORT) por separado:
    1. Pick top-N trials del Optuna study (rank por ev.score con sig>=min).
    2. Para cada trial:
        a) Refit booster con sus params sobre full train (mismo que fase 2).
        b) Carga calibrator OOF persistido.
        c) Predice probs calibradas sobre holdout.
    3. Combina: PROMEDIO de probs calibradas (igualitario, no ponderado).
    4. Escanea threshold sobre el promedio:
        - thr_ensemble: best thr en OOF aggregate (sin leak)
          → para esto necesitaríamos OOF preds del ensemble; usamos en
          su lugar el thr MEDIANO de los top-N (proxy razonable).
    5. Evalúa contra holdout con ese thr.

USO:
  python -m mimo.oof.main_oof_gbm_ensemble \\
    --release 202602_GBM --inherit-config-from 202601 \\
    --best-json artifacts/202602_GBM/oof/<tag>/reports/best_per_side.json \\
    --top-n 5 \\
    --variant-long vol_boost_td_down --variant-short vol_boost \\
    --train-from 2024-01-01 --train-to 2025-10-30 \\
    --holdout-from 2025-11-01 --holdout-to 2026-04-10 \\
    --out-json artifacts/202602_GBM/oof/<tag>/reports/ensemble_report.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd

from mimo.data_managers.databases import Database
from mimo.data_managers.data_manager import DataManager
from mimo.data_managers.data_pipeline_v2 import DataPipeline
from mimo.features.feature_builder import FeatureConfig
from mimo.models.model_builder import Config, ModelConfig
from mimo.states_manager.state_detector import StateConfig
from mimo.oof.ev_objective import compute_ev_at_best_threshold
from mimo.oof.main_oof_gbm_holdout import (
    _split_trial_params, _collect_tabular_columns,
    _refit_with_internal_val, _json_safe,
)
from mimo.oof.main_oof_regime_weights_v7 import (
    BARRIERS_BY_RELEASE, LONG_VARIANTS, SHORT_VARIANTS,
    _VOL_INVARIANT_RELEASES, _REDUCED_FEATURES_RELEASES,
    _ULTRA_REDUCED_FEATURES_RELEASES,
    _get_barriers_for_release, _get_feature_masks_for_release, _tf_defaults,
    install_regime_weight_patch, resolve_regime_weights, set_global_seeds,
)

# Side-effect: importar main_oof_gbm inyecta _GBM_BARRIERS_BY_RELEASE
# en BARRIERS_BY_RELEASE para releases 202604/5/6.
from mimo.oof.main_oof_gbm import _GBM_EXCLUDE_PATTERNS  # noqa: F401


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _select_top_trials(
    study: optuna.Study, *, side: str, top_n: int, min_signals: int,
) -> List[optuna.trial.FrozenTrial]:
    """side='long' o 'short'. Ranking por ev_<side>.score con filtro de min_signals."""
    key = f"ev_{side}"
    candidates = []
    for t in study.trials:
        if t.state.name != "COMPLETE": continue
        ev = t.user_attrs.get(key, {}) or {}
        if int(ev.get("n_signals", 0) or 0) < min_signals: continue
        score = ev.get("score", float("-inf"))
        if not isinstance(score, (int, float)) or not np.isfinite(score): continue
        candidates.append((float(score), t))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in candidates[:top_n]]


def _refit_and_predict_side(
    *, side: str, trial: optuna.trial.FrozenTrial, seed: int,
    X_tr: np.ndarray, y_tr: np.ndarray, w_tr: np.ndarray, X_hd: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Refit booster del trial + predict raw+cal sobre holdout. Devuelve
    (p_raw, p_cal, thr_oof) — thr_oof viene del user_attr ev_<side>."""
    lgbm_params, meta_params = _split_trial_params(trial.params, seed=seed)
    booster, _ = _refit_with_internal_val(
        X=X_tr, y=y_tr, w=w_tr, lgbm_params=lgbm_params,
        n_estimators=int(meta_params.get("n_estimators", 500)),
        early_stopping_rounds=int(meta_params.get("early_stopping_rounds", 50)),
    )
    p_raw = booster.predict(X_hd).astype(np.float32)
    cal_path = trial.user_attrs.get(f"cal_{side}_path")
    if cal_path and os.path.exists(cal_path):
        cal = joblib.load(cal_path)
        p_cal = cal.predict(p_raw.astype(np.float64)).astype(np.float32)
    else:
        p_cal = p_raw  # sin calibrator disponible, devolver raw
    ev = trial.user_attrs.get(f"ev_{side}", {}) or {}
    thr = float(ev.get("thr", float("nan")))
    return p_raw, p_cal, thr


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", required=True)
    ap.add_argument("--inherit-config-from", default=None)
    ap.add_argument("--best-json", required=True,
                    help="Solo se usa para localizar el reports dir por defecto. "
                         "Los top-N se sacan directo del Optuna study.")
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--min-signals-trial", type=int, default=100,
                    help="Mín signals en OOF para que un trial sea elegible.")

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

    ap.add_argument("--cost-per-signal", type=float, default=0.05)
    ap.add_argument("--ev-min-signals", type=int, default=30)
    ap.add_argument("--max-drawdown-R", type=float, default=30.0)
    ap.add_argument("--thr-strategy", choices=("median", "best_trial", "rescan"),
                    default="median",
                    help="median: mediana de thrs OOF de los top-N. "
                         "best_trial: thr del trial top-1. "
                         "rescan: re-escanear thr sobre el promedio (en holdout — info).")

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

    # Inherit config
    if args.inherit_config_from:
        src, dst = str(args.inherit_config_from), release
        if dst != src:
            if src in BARRIERS_BY_RELEASE and dst not in BARRIERS_BY_RELEASE:
                BARRIERS_BY_RELEASE[dst] = BARRIERS_BY_RELEASE[src]
            if src in _VOL_INVARIANT_RELEASES:  _VOL_INVARIANT_RELEASES.add(dst)
            if src in _REDUCED_FEATURES_RELEASES: _REDUCED_FEATURES_RELEASES.add(dst)
            if src in _ULTRA_REDUCED_FEATURES_RELEASES: _ULTRA_REDUCED_FEATURES_RELEASES.add(dst)
            print(f"🧬 [INHERIT-CONFIG] '{dst}' ← '{src}'")

    # Load study
    study_name = f"{args.study_prefix}_{release}_multitask"
    print(f"\n📂 Optuna study: {study_name}")
    study = optuna.load_study(study_name=study_name, storage=args.optuna_storage)

    top_long  = _select_top_trials(study, side="long",  top_n=args.top_n, min_signals=args.min_signals_trial)
    top_short = _select_top_trials(study, side="short", top_n=args.top_n, min_signals=args.min_signals_trial)
    print(f"   Top-{args.top_n} LONG  : trials #{[t.number for t in top_long]}")
    print(f"   Top-{args.top_n} SHORT : trials #{[t.number for t in top_short]}")
    if len(top_long) < 1 or len(top_short) < 1:
        raise SystemExit("❌ Sin trials elegibles en uno de los lados")

    # Regime weights
    regime_weights_by_side = resolve_regime_weights(args)
    install_regime_weight_patch(regime_weights_by_side, verbose=False)

    # OHLCV completo
    print(f"\n📊 Cargando OHLCV {args.train_from.date()} → {args.holdout_to.date()}")
    db = Database()
    resample = None if str(base_tf).lower() in ("1min", "1m") else base_tf
    dm = DataManager.from_database_historical_2(
        db, from_date=args.train_from, to_date=args.holdout_to, resample=resample
    )
    df_rates = dm.df.copy()
    df_rates["time"] = pd.to_datetime(df_rates["time"])

    # FeatureConfig idéntico al training
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
    general_config = Config(release=release, use_oof=True, oof_splits=5,
                            oof_epochs=1, save_oof_artifacts=True)
    regime_config = StateConfig(adx_trend_threshold=25.0)
    pipeline = DataPipeline(
        general_config=general_config, feature_config=feature_config,
        model_config=ModelConfig(seq_len_short=64, seq_len_long=256, target_type="multitask"),
        regime_config=regime_config,
    )

    print("⚙️  prepare_data...")
    df_prepared = pipeline.prepare_data(
        df_rates, labels=True, side="both",
        set_market_condition=False, ensure_regime=True,
    )
    feat_cols = [c for c in _collect_tabular_columns(pipeline) if c in df_prepared.columns]
    _exclude = _GBM_EXCLUDE_PATTERNS.get(release)
    if _exclude:
        feat_cols = [c for c in feat_cols if not any(p in c for p in _exclude)]
        print(f"🚫 [EXCLUDE] release={release} → {len(feat_cols)} features")
    needed = feat_cols + ["time", "high", "low", "close", "atr", "signal_long", "signal_short"]
    keep = df_prepared[needed].notna().all(axis=1)
    df_clean = df_prepared.loc[keep].reset_index(drop=True)

    tr_mask = df_clean["time"] < pd.Timestamp(args.holdout_from)
    hd_mask = df_clean["time"] >= pd.Timestamp(args.holdout_from)
    df_tr, df_hd = df_clean.loc[tr_mask].reset_index(drop=True), df_clean.loc[hd_mask].reset_index(drop=True)
    print(f"   train: {len(df_tr):,}  hold: {len(df_hd):,}")
    X_tr = df_tr[feat_cols].astype(np.float32).values
    y_long_tr  = df_tr["signal_long"].astype(np.int8).values
    y_short_tr = df_tr["signal_short"].astype(np.int8).values
    w_tr = df_tr["regime_weight"].astype(np.float32).values \
        if "regime_weight" in df_tr.columns else np.ones(len(df_tr), dtype=np.float32)
    X_hd = df_hd[feat_cols].astype(np.float32).values

    horizon = int(feature_config.label_horizon)
    tp_mult = float(feature_config.tp_barrier)
    sl_mult = float(feature_config.sl_barrier)

    # Refit + predict para cada trial top-N
    def _ensemble_side(side: str, trials: List[optuna.trial.FrozenTrial],
                       y_tr: np.ndarray) -> Tuple[np.ndarray, List[float], List[int]]:
        """Devuelve (p_cal_promedio_holdout, thrs_oof_individuales, trial_numbers)."""
        cal_stack = []
        thrs = []
        nums = []
        for i, t in enumerate(trials):
            print(f"   · ({i+1}/{len(trials)}) refit {side} trial #{t.number}...")
            _, p_cal, thr_oof = _refit_and_predict_side(
                side=side, trial=t, seed=int(args.seed),
                X_tr=X_tr, y_tr=y_tr, w_tr=w_tr, X_hd=X_hd,
            )
            cal_stack.append(p_cal)
            thrs.append(thr_oof)
            nums.append(t.number)
        p_avg = np.mean(np.stack(cal_stack, axis=0), axis=0).astype(np.float32)
        return p_avg, thrs, nums

    print(f"\n🌲 Ensembling LONG (top-{len(top_long)})...")
    p_long_avg, thrs_long, nums_long = _ensemble_side("long", top_long, y_long_tr)
    print(f"\n🌲 Ensembling SHORT (top-{len(top_short)})...")
    p_short_avg, thrs_short, nums_short = _ensemble_side("short", top_short, y_short_tr)

    # Decidir thr según estrategia
    def _resolve_thr(thrs: List[float], strategy: str) -> float:
        thrs_clean = [t for t in thrs if np.isfinite(t)]
        if not thrs_clean: return float("nan")
        if strategy == "best_trial": return thrs_clean[0]
        if strategy == "median":     return float(np.median(thrs_clean))
        if strategy == "rescan":     return float("nan")  # se decide post-scan
        return float(np.median(thrs_clean))

    thr_long_use  = _resolve_thr(thrs_long, args.thr_strategy)
    thr_short_use = _resolve_thr(thrs_short, args.thr_strategy)

    # Eval
    df_oof_hold = pd.DataFrame({
        "time": df_hd["time"].values,
        "high": df_hd["high"].astype(np.float64).values,
        "low":  df_hd["low"].astype(np.float64).values,
        "close":df_hd["close"].astype(np.float64).values,
        "atr":  df_hd["atr"].astype(np.float64).values,
        "signal_long":  df_hd["signal_long"].astype(np.int8).values,
        "signal_short": df_hd["signal_short"].astype(np.int8).values,
        "oof_proba_long_cal":  p_long_avg,
        "oof_proba_short_cal": p_short_avg,
    })

    def _eval(side: str, proba_col: str, side_is_long: bool, thr: float):
        if args.thr_strategy == "rescan":
            return compute_ev_at_best_threshold(
                df_oof_hold, proba_col=proba_col, side_is_long=side_is_long,
                horizon=horizon, tp_mult=tp_mult, sl_mult=sl_mult,
                cost_per_signal=args.cost_per_signal,
                n_thr=80, thr_lo=0.05, thr_hi=0.60,
                min_signals=args.ev_min_signals, max_drawdown_R=args.max_drawdown_R,
            )
        return compute_ev_at_best_threshold(
            df_oof_hold, proba_col=proba_col, side_is_long=side_is_long,
            horizon=horizon, tp_mult=tp_mult, sl_mult=sl_mult,
            cost_per_signal=args.cost_per_signal,
            n_thr=1, thr_lo=thr, thr_hi=thr,
            min_signals=args.ev_min_signals, max_drawdown_R=args.max_drawdown_R,
        )

    print(f"\n🎯 Evaluación ensemble en holdout (thr_strategy={args.thr_strategy}):")
    print(f"   thr_long_use ={thr_long_use:.4f}  (de thrs OOF {[f'{t:.3f}' for t in thrs_long]})")
    print(f"   thr_short_use={thr_short_use:.4f} (de thrs OOF {[f'{t:.3f}' for t in thrs_short]})")
    long_res  = _eval("long",  "oof_proba_long_cal",  True,  thr_long_use)
    short_res = _eval("short", "oof_proba_short_cal", False, thr_short_use)

    print("\n" + "═" * 70)
    for side, r, used_thr in (("LONG ", long_res, thr_long_use),
                               ("SHORT", short_res, thr_short_use)):
        ev = r.get("ev_net", float("nan"))
        if ev is None or not np.isfinite(ev):
            print(f"  {side}: sin señales (thr={used_thr:.4f})")
            continue
        print(f"  {side}: ev_net={ev:+.4f}R  sig={int(r.get('n_signals', 0))}  "
              f"prec_TP={r.get('prec_TP', 0):.3f}  mdd={r.get('mdd_R', 0):.1f}R  "
              f"thr_used={r.get('thr', used_thr):.4f}")

    # Total R
    def _r_contrib(res):
        n = int(res.get("n_signals", 0) or 0)
        ev = res.get("ev_net")
        return ev * n if ev is not None and np.isfinite(ev) else 0.0
    total_R = _r_contrib(long_res) + _r_contrib(short_res)
    months = (args.holdout_to - args.holdout_from).days / 30.44
    print(f"\n  💰 TOTAL R ensemble holdout: {total_R:+.2f}R en {months:.1f}m  "
          f"({total_R/max(months,0.01):+.2f}R/mes)")
    print("═" * 70)

    # Persistir
    report = {
        "release": release, "best_json": str(args.best_json),
        "top_n": args.top_n,
        "thr_strategy": args.thr_strategy,
        "long_trials":  nums_long,
        "short_trials": nums_short,
        "thrs_oof_long":  thrs_long,
        "thrs_oof_short": thrs_short,
        "thr_long_use":  thr_long_use,
        "thr_short_use": thr_short_use,
        "horizon": horizon, "tp_mult": tp_mult, "sl_mult": sl_mult,
        "holdout": {
            "long":  _json_safe(long_res),
            "short": _json_safe(short_res),
            "total_R": total_R,
            "months":  months,
        },
        "holdout_period": [str(args.holdout_from.date()), str(args.holdout_to.date())],
    }
    out_json = args.out_json or str(Path(args.best_json).parent / "ensemble_report.json")
    with open(out_json, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\n📁 Reporte: {out_json}")


if __name__ == "__main__":
    main()
