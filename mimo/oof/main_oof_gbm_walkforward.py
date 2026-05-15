"""
main_oof_gbm_walkforward.py
═══════════════════════════════════════════════════════════════════════════
FASE 3 GBM: validación walk-forward sobre ventanas deslizantes.

¿POR QUÉ walk-forward vs holdout estático?
  Un holdout único (2025-11 → 2026-04) puede caer en un régimen atípico
  por azar. Walk-forward entrena en ventanas deslizantes y testea en la
  siguiente ventana, reportando estadísticas (mediana, p10/p90) en LUGAR
  de un único número. Te dice si el modelo es robusto al drift.

ESQUEMA:
  Para cada test_start en [start_from, end - test_window]:
    train_window: [test_start - TRAIN_MONTHS, test_start)
    test_window:  [test_start, test_start + TEST_MONTHS)

  Ej con TRAIN=12, TEST=1, STEP=1:
    Window  1: train=2024-01→2024-12, test=2025-01
    Window  2: train=2024-02→2025-01, test=2025-02
    ...
    Window N: train=2025-04→2026-03, test=2026-04

CAVEAT importante sobre calibración:
  El calibrator OOF original vio TODO el train (2024-01→2025-10) por vía
  de los k-folds OOF. Aplicarlo a un walk-test que cae DENTRO de ese rango
  hay una leve contaminación (el calibrator "sabe" lo que pasó en el
  test window). Para el test window OUT-OF-TIME (post-2025-10), no hay leak.
  Esto se documenta en el reporte; usar el modo --raw-probs para evitar
  el calibrator (threshold escaneado por ventana, info pura).

REPORTE:
  · ev_net mediana, p10, p90 por lado
  · % ventanas positivas (PWR — Positive Window Rate)
  · R total acumulado
  · CSV opcional con métricas por ventana para post-análisis

USO:
  python -m mimo.oof.main_oof_gbm_walkforward \\
    --release 202602_GBM --inherit-config-from 202601 \\
    --best-json artifacts/202602_GBM/oof/<tag>/reports/best_per_side.json \\
    --train-months 12 --test-months 1 --step-months 1 \\
    --walk-from 2025-01-01 --walk-to 2026-04-10 \\
    --variant-long vol_boost_td_down --variant-short vol_boost \\
    --out-json artifacts/202602_GBM/oof/<tag>/reports/walkforward_report.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from dateutil.relativedelta import relativedelta

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


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def _generate_windows(
    *, walk_from: datetime, walk_to: datetime,
    train_months: int, test_months: int, step_months: int,
) -> List[Tuple[datetime, datetime, datetime, datetime]]:
    """Genera (train_start, train_end, test_start, test_end) por ventana."""
    out = []
    test_start = walk_from
    while True:
        test_end = test_start + relativedelta(months=test_months)
        if test_end > walk_to:
            break
        train_start = test_start - relativedelta(months=train_months)
        train_end = test_start
        out.append((train_start, train_end, test_start, test_end))
        test_start = test_start + relativedelta(months=step_months)
    return out


def _evaluate_window(
    *,
    df_clean: pd.DataFrame, feat_cols: List[str],
    train_start: datetime, train_end: datetime,
    test_start: datetime, test_end: datetime,
    lgbm_long: Dict[str, Any], meta_long: Dict[str, Any],
    lgbm_short: Dict[str, Any], meta_short: Dict[str, Any],
    cal_long, cal_short, thr_long: float, thr_short: float,
    horizon: int, tp_mult: float, sl_mult: float,
    cost_per_signal: float, max_drawdown_R: float,
    use_raw_probs: bool, min_signals_window: int,
) -> Dict[str, Any]:
    """Refit en train_window, predict en test_window, evaluar."""
    tr_mask = (df_clean["time"] >= pd.Timestamp(train_start)) & (df_clean["time"] < pd.Timestamp(train_end))
    te_mask = (df_clean["time"] >= pd.Timestamp(test_start)) & (df_clean["time"] < pd.Timestamp(test_end))
    df_tr = df_clean.loc[tr_mask].reset_index(drop=True)
    df_te = df_clean.loc[te_mask].reset_index(drop=True)
    if len(df_tr) < 20000 or len(df_te) < 1000:
        return {"window": (str(train_start.date()), str(train_end.date()),
                           str(test_start.date()), str(test_end.date())),
                "n_train": len(df_tr), "n_test": len(df_te), "skipped": True}

    X_tr = df_tr[feat_cols].astype(np.float32).values
    y_long_tr  = df_tr["signal_long"].astype(np.int8).values
    y_short_tr = df_tr["signal_short"].astype(np.int8).values
    w_tr = df_tr["regime_weight"].astype(np.float32).values \
        if "regime_weight" in df_tr.columns else np.ones(len(df_tr), dtype=np.float32)
    X_te = df_te[feat_cols].astype(np.float32).values

    # Refit por lado
    booster_long, _ = _refit_with_internal_val(
        X=X_tr, y=y_long_tr, w=w_tr, lgbm_params=lgbm_long,
        n_estimators=int(meta_long.get("n_estimators", 500)),
        early_stopping_rounds=int(meta_long.get("early_stopping_rounds", 50)),
    )
    booster_short, _ = _refit_with_internal_val(
        X=X_tr, y=y_short_tr, w=w_tr, lgbm_params=lgbm_short,
        n_estimators=int(meta_short.get("n_estimators", 500)),
        early_stopping_rounds=int(meta_short.get("early_stopping_rounds", 50)),
    )

    p_long_raw  = booster_long.predict(X_te).astype(np.float32)
    p_short_raw = booster_short.predict(X_te).astype(np.float32)

    if use_raw_probs:
        # Modo info: probs raw + threshold sweep por ventana.
        p_long_use, p_short_use = p_long_raw, p_short_raw
        thr_lo, thr_hi, n_thr = 0.05, 0.60, 60
    else:
        # Modo operativa: calibrar + thr fijo del OOF.
        p_long_use  = cal_long.predict(p_long_raw.astype(np.float64)).astype(np.float32)
        p_short_use = cal_short.predict(p_short_raw.astype(np.float64)).astype(np.float32)
        thr_lo, thr_hi, n_thr = thr_long, thr_long, 1

    df_eval = pd.DataFrame({
        "time": df_te["time"].values,
        "high": df_te["high"].astype(np.float64).values,
        "low":  df_te["low"].astype(np.float64).values,
        "close":df_te["close"].astype(np.float64).values,
        "atr":  df_te["atr"].astype(np.float64).values,
        "signal_long":  df_te["signal_long"].astype(np.int8).values,
        "signal_short": df_te["signal_short"].astype(np.int8).values,
        "oof_proba_long_cal":  p_long_use,
        "oof_proba_short_cal": p_short_use,
    })

    long_res = compute_ev_at_best_threshold(
        df_eval, proba_col="oof_proba_long_cal", side_is_long=True,
        horizon=horizon, tp_mult=tp_mult, sl_mult=sl_mult,
        cost_per_signal=cost_per_signal,
        n_thr=n_thr, thr_lo=thr_lo, thr_hi=thr_hi,
        min_signals=min_signals_window, max_drawdown_R=max_drawdown_R,
    )
    # Threshold del SHORT — si en modo operativa, usar thr_short específico
    if not use_raw_probs:
        thr_lo, thr_hi, n_thr = thr_short, thr_short, 1
    short_res = compute_ev_at_best_threshold(
        df_eval, proba_col="oof_proba_short_cal", side_is_long=False,
        horizon=horizon, tp_mult=tp_mult, sl_mult=sl_mult,
        cost_per_signal=cost_per_signal,
        n_thr=n_thr, thr_lo=thr_lo, thr_hi=thr_hi,
        min_signals=min_signals_window, max_drawdown_R=max_drawdown_R,
    )
    return {
        "window": (str(train_start.date()), str(train_end.date()),
                   str(test_start.date()), str(test_end.date())),
        "n_train": len(df_tr), "n_test": len(df_te), "skipped": False,
        "long":  _json_safe(long_res),
        "short": _json_safe(short_res),
    }


def _summarize(results: List[Dict[str, Any]], side: str) -> Dict[str, Any]:
    """Estadísticos de ev_net, n_signals y prec_TP a través de ventanas."""
    valid = [r for r in results if not r.get("skipped") and r.get(side, {}).get("ev_net") is not None]
    if not valid:
        return {"n_windows": 0}
    evs = np.array([r[side]["ev_net"] for r in valid], dtype=np.float64)
    sigs = np.array([r[side].get("n_signals", 0) or 0 for r in valid], dtype=np.int64)
    precs = np.array([r[side].get("prec_TP", float("nan")) for r in valid], dtype=np.float64)
    contrib = np.where(np.isfinite(evs), evs * sigs, 0.0)
    return {
        "n_windows": len(valid),
        "n_pos_windows": int((evs > 0).sum()),
        "pwr": float((evs > 0).mean()),  # Positive Window Rate
        "ev_median": float(np.nanmedian(evs)),
        "ev_p10":    float(np.nanpercentile(evs, 10)),
        "ev_p90":    float(np.nanpercentile(evs, 90)),
        "n_signals_total": int(sigs.sum()),
        "n_signals_median": float(np.median(sigs)),
        "prec_median": float(np.nanmedian(precs)),
        "R_total":   float(contrib.sum()),
    }


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

    # Walk-forward windowing
    ap.add_argument("--walk-from", type=_parse_date, required=True,
                    help="Inicio del primer TEST window.")
    ap.add_argument("--walk-to", type=_parse_date, required=True,
                    help="Fin del último TEST window (exclusivo).")
    ap.add_argument("--train-months", type=int, default=12)
    ap.add_argument("--test-months", type=int, default=1)
    ap.add_argument("--step-months", type=int, default=1)

    # Modo de evaluación
    ap.add_argument("--raw-probs", action="store_true",
                    help="Si se pasa, evalúa con probs RAW (sin calibrator) "
                         "y threshold escaneado por ventana. Más defensible "
                         "contra el leak del calibrator OOF.")
    ap.add_argument("--min-signals-window", type=int, default=15,
                    help="Mín signals por ventana para reportar. Más bajo "
                         "que en holdout porque cada ventana es ~1 mes.")

    ap.add_argument("--cost-per-signal", type=float, default=0.05)
    ap.add_argument("--max-drawdown-R", type=float, default=30.0)

    ap.add_argument("--optuna-storage", default=os.environ.get(
        "OPTUNA_STORAGE",
        "mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db",
    ))
    ap.add_argument("--study-prefix", default="oof_study_gbm")
    ap.add_argument("--seed", type=int, default=47)

    ap.add_argument("--out-json", default=None)
    ap.add_argument("--out-csv", default=None,
                    help="Si se pasa, persiste métricas por ventana en CSV.")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    set_global_seeds(int(args.seed))
    release = str(args.release)
    base_tf = str(args.base_tf)

    # 1) Inherit config
    if args.inherit_config_from:
        src, dst = str(args.inherit_config_from), release
        if dst != src:
            if src in BARRIERS_BY_RELEASE and dst not in BARRIERS_BY_RELEASE:
                BARRIERS_BY_RELEASE[dst] = BARRIERS_BY_RELEASE[src]
            if src in _VOL_INVARIANT_RELEASES:  _VOL_INVARIANT_RELEASES.add(dst)
            if src in _REDUCED_FEATURES_RELEASES: _REDUCED_FEATURES_RELEASES.add(dst)
            if src in _ULTRA_REDUCED_FEATURES_RELEASES: _ULTRA_REDUCED_FEATURES_RELEASES.add(dst)
            print(f"🧬 [INHERIT-CONFIG] '{dst}' ← '{src}'")

    # 2) Load best_per_side.json
    print(f"\n📂 Cargando best_per_side.json: {args.best_json}")
    with open(args.best_json) as fh:
        best = json.load(fh)
    top_long_meta = best["top_long"][0]
    top_short_meta = best["top_short"][0]
    n_long, n_short = int(top_long_meta["trial"]), int(top_short_meta["trial"])
    thr_long  = float(top_long_meta["ev_long"]["thr"])
    thr_short = float(top_short_meta["ev_short"]["thr"])
    print(f"   LONG  trial #{n_long} thr={thr_long:.4f}")
    print(f"   SHORT trial #{n_short} thr={thr_short:.4f}")

    # 3) Load Optuna trials for params + calibrator paths
    study_name = f"{args.study_prefix}_{release}_multitask"
    study = optuna.load_study(study_name=study_name, storage=args.optuna_storage)
    tr_long_obj  = next(t for t in study.trials if t.number == n_long)
    tr_short_obj = next(t for t in study.trials if t.number == n_short)
    lgbm_long,  meta_long  = _split_trial_params(tr_long_obj.params,  seed=int(args.seed))
    lgbm_short, meta_short = _split_trial_params(tr_short_obj.params, seed=int(args.seed))
    cal_long_path  = tr_long_obj.user_attrs.get("cal_long_path")
    cal_short_path = tr_short_obj.user_attrs.get("cal_short_path")
    cal_long  = joblib.load(cal_long_path)  if cal_long_path  and os.path.exists(cal_long_path)  else None
    cal_short = joblib.load(cal_short_path) if cal_short_path and os.path.exists(cal_short_path) else None
    if not args.raw_probs and (cal_long is None or cal_short is None):
        raise SystemExit("❌ Sin calibradores; pasa --raw-probs para evaluar sin ellos")

    # 4) Regime weights
    regime_weights_by_side = resolve_regime_weights(args)
    install_regime_weight_patch(regime_weights_by_side, verbose=False)

    # 5) Build configs + load OHLCV completo (ANTES del primer train_start)
    windows = _generate_windows(
        walk_from=args.walk_from, walk_to=args.walk_to,
        train_months=args.train_months, test_months=args.test_months,
        step_months=args.step_months,
    )
    if not windows:
        raise SystemExit("❌ Ninguna ventana generada — revisa walk_from/walk_to vs train_months")
    earliest_train = windows[0][0]
    latest_test    = windows[-1][3]
    # Pad para que features de ventana larga tengan historia al inicio
    earliest_load = earliest_train - relativedelta(months=2)
    print(f"\n📊 Cargando OHLCV {earliest_load.date()} → {latest_test.date()} "
          f"({len(windows)} ventanas)")
    db = Database()
    resample = None if str(base_tf).lower() in ("1min", "1m") else base_tf
    dm = DataManager.from_database_historical_2(
        db, from_date=earliest_load, to_date=latest_test, resample=resample
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
    general_config = Config(release=release, use_oof=True, oof_splits=5,
                            oof_epochs=1, save_oof_artifacts=True)
    regime_config = StateConfig(adx_trend_threshold=25.0)
    pipeline = DataPipeline(
        general_config=general_config, feature_config=feature_config,
        model_config=ModelConfig(seq_len_short=64, seq_len_long=256, target_type="multitask"),
        regime_config=regime_config,
    )

    print("⚙️  prepare_data (un solo pase sobre todo el rango)...")
    df_prepared = pipeline.prepare_data(
        df_rates, labels=True, side="both",
        set_market_condition=False, ensure_regime=True,
    )
    feat_cols = [c for c in _collect_tabular_columns(pipeline) if c in df_prepared.columns]
    needed = feat_cols + ["time", "high", "low", "close", "atr", "signal_long", "signal_short"]
    keep = df_prepared[needed].notna().all(axis=1)
    df_clean = df_prepared.loc[keep].reset_index(drop=True)
    print(f"   {len(df_clean):,} filas × {len(feat_cols)} features")

    horizon = int(feature_config.label_horizon)
    tp_mult = float(feature_config.tp_barrier)
    sl_mult = float(feature_config.sl_barrier)

    # 6) Walk-forward loop
    print(f"\n🔄 Walk-forward: {len(windows)} ventanas "
          f"(train={args.train_months}m, test={args.test_months}m, step={args.step_months}m)")
    print(f"   modo: {'RAW PROBS + thr scan' if args.raw_probs else 'CALIBRATED + thr OOF fijo'}")

    results = []
    for i, (ts, te, vs, ve) in enumerate(windows):
        print(f"\n── Window {i+1}/{len(windows)}: train={ts.date()}→{te.date()} "
              f"test={vs.date()}→{ve.date()}")
        res = _evaluate_window(
            df_clean=df_clean, feat_cols=feat_cols,
            train_start=ts, train_end=te, test_start=vs, test_end=ve,
            lgbm_long=lgbm_long, meta_long=meta_long,
            lgbm_short=lgbm_short, meta_short=meta_short,
            cal_long=cal_long, cal_short=cal_short,
            thr_long=thr_long, thr_short=thr_short,
            horizon=horizon, tp_mult=tp_mult, sl_mult=sl_mult,
            cost_per_signal=args.cost_per_signal,
            max_drawdown_R=args.max_drawdown_R,
            use_raw_probs=bool(args.raw_probs),
            min_signals_window=int(args.min_signals_window),
        )
        if res.get("skipped"):
            print(f"   ⏭️  saltada (n_train={res['n_train']}, n_test={res['n_test']})")
        else:
            l = res.get("long", {}) or {}
            s = res.get("short", {}) or {}
            print(f"   LONG  ev_net={l.get('ev_net'):+.4f}R sig={int(l.get('n_signals',0))} "
                  f"prec={l.get('prec_TP', 0):.3f}"
                  if l.get("ev_net") is not None else "   LONG  sin señales")
            print(f"   SHORT ev_net={s.get('ev_net'):+.4f}R sig={int(s.get('n_signals',0))} "
                  f"prec={s.get('prec_TP', 0):.3f}"
                  if s.get("ev_net") is not None else "   SHORT sin señales")
        results.append(res)

    # 7) Summary
    summary_long  = _summarize(results, "long")
    summary_short = _summarize(results, "short")

    print("\n" + "═" * 70)
    print("  WALK-FORWARD SUMMARY")
    print("═" * 70)
    for side, sm in (("LONG", summary_long), ("SHORT", summary_short)):
        if sm.get("n_windows", 0) == 0:
            print(f"  {side}: 0 ventanas válidas")
            continue
        print(f"  {side}:")
        print(f"    Ventanas válidas        : {sm['n_windows']}")
        print(f"    Ventanas positivas (PWR): {sm['n_pos_windows']} / {sm['n_windows']} ({100*sm['pwr']:.0f}%)")
        print(f"    EV_net  mediana        : {sm['ev_median']:+.4f}R")
        print(f"    EV_net  p10 / p90      : {sm['ev_p10']:+.4f}R / {sm['ev_p90']:+.4f}R")
        print(f"    Signals totales         : {sm['n_signals_total']}")
        print(f"    Prec_TP mediana         : {sm['prec_median']:.3f}")
        print(f"    💰 R total acumulado    : {sm['R_total']:+.2f}R")

    # 8) Persist
    report = {
        "release": release, "best_json": str(args.best_json),
        "mode": "raw_probs" if args.raw_probs else "calibrated",
        "walk_config": {
            "walk_from": str(args.walk_from.date()),
            "walk_to":   str(args.walk_to.date()),
            "train_months": args.train_months,
            "test_months":  args.test_months,
            "step_months":  args.step_months,
        },
        "long_trial":  n_long, "short_trial": n_short,
        "thr_long": thr_long, "thr_short": thr_short,
        "horizon": horizon, "tp_mult": tp_mult, "sl_mult": sl_mult,
        "summary_long":  summary_long,
        "summary_short": summary_short,
        "windows": results,
    }
    out_json = args.out_json or str(Path(args.best_json).parent / "walkforward_report.json")
    with open(out_json, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\n📁 Reporte: {out_json}")

    if args.out_csv:
        rows = []
        for r in results:
            if r.get("skipped"): continue
            row = {"train_start": r["window"][0], "train_end": r["window"][1],
                   "test_start": r["window"][2], "test_end": r["window"][3]}
            for side in ("long", "short"):
                d = r.get(side, {}) or {}
                row[f"{side}_ev_net"]  = d.get("ev_net")
                row[f"{side}_sig"]     = d.get("n_signals")
                row[f"{side}_prec"]    = d.get("prec_TP")
                row[f"{side}_thr"]     = d.get("thr")
                row[f"{side}_mdd_R"]   = d.get("mdd_R")
            rows.append(row)
        pd.DataFrame(rows).to_csv(args.out_csv, index=False)
        print(f"📊 CSV: {args.out_csv}")


if __name__ == "__main__":
    main()
