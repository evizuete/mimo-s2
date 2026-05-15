"""
diag_per_state_gbm.py
═══════════════════════════════════════════════════════════════════════════
Slicea las señales del holdout por market_state (TREND_UP, RANGE, LOW_VOL,
HIGH_VOL, BREAKOUT_WAIT_*, TRANSITION_*) y reporta EV-net por estado.

DIAGNOSIS POSIBLES:
  · Algunas señales rentables en TREND, otras en RANGE → modelo general OK
  · Solo rentable en TRANSITION_* → señal dependiente de cambio de régimen,
    operar solo cuando se detecten transitions
  · Solo rentable en HIGH_VOL → señal dependiente de volatilidad alta
  · Rentable en LOW_VOL pero negativo en HIGH_VOL → modelo confunde ruido
    de mercado, hay que excluir HIGH_VOL en producción
  · Distribución uniforme y negativa → señal genuinamente débil

ESTRATEGIA:
  Reusa el refit + cal + thr_OOF de fase 2, pero ANTES de aplicar
  compute_ev_at_best_threshold, slicea df_oof_hold por 'state'. Para cada
  estado con >= MIN_SIG_PER_STATE señales, computa ev_net usando triple
  barrier sobre solo esa porción del holdout.

USO:
  python -m mimo.oof.diag_per_state_gbm \\
    --release 202602_GBM --inherit-config-from 202601 \\
    --best-json artifacts/202602_GBM/oof/<tag>/reports/best_per_side.json \\
    --train-from 2024-01-01 --train-to 2025-10-30 \\
    --holdout-from 2025-11-01 --holdout-to 2026-04-10 \\
    --out-json artifacts/202602_GBM/oof/<tag>/reports/per_state_diag.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

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


def _eval_per_state(
    df_oof_hold: pd.DataFrame, *, proba_col: str, side_is_long: bool,
    thr_oof: float, states: List[str], horizon: int, tp_mult: float, sl_mult: float,
    cost_per_signal: float, max_drawdown_R: float, min_signals_per_state: int,
) -> Dict[str, Any]:
    """Para cada estado: filtra df_oof_hold, evalúa con thr_oof, devuelve métricas."""
    results = {}
    for state in states:
        sub = df_oof_hold[df_oof_hold["state"] == state].reset_index(drop=True)
        if len(sub) < min_signals_per_state * 5:
            results[state] = {"n_rows": int(len(sub)), "skipped": True,
                              "reason": "too_few_rows"}
            continue
        res = compute_ev_at_best_threshold(
            sub, proba_col=proba_col, side_is_long=side_is_long,
            horizon=horizon, tp_mult=tp_mult, sl_mult=sl_mult,
            cost_per_signal=cost_per_signal,
            n_thr=1, thr_lo=thr_oof, thr_hi=thr_oof,
            min_signals=min_signals_per_state, max_drawdown_R=max_drawdown_R,
        )
        results[state] = {"n_rows": int(len(sub)), "skipped": False, **_json_safe(res)}
    return results


def _print_state_table(side: str, per_state: Dict[str, Dict[str, Any]]) -> None:
    print(f"\n══ {side.upper()} — EV-net por estado en holdout ══════════════════════")
    print(f"  {'state':<20} {'n_rows':>8} {'n_sig':>6} {'ev_net':>10} {'prec_TP':>8} {'mdd_R':>8}")
    # Orden: por ev_net descendente, NaN al final
    items = list(per_state.items())
    def _sort_key(kv):
        ev = kv[1].get("ev_net")
        if ev is None: return (1, 0.0)  # NaN al final
        try: return (0, -float(ev))
        except Exception: return (1, 0.0)
    items.sort(key=_sort_key)
    for state, d in items:
        if d.get("skipped"):
            print(f"  {state:<20} {d['n_rows']:>8} {'—':>6} {'—':>10} {'—':>8} {'—':>8}  [{d.get('reason','')}]")
            continue
        ev = d.get("ev_net"); sig = d.get("n_signals", 0); prec = d.get("prec_TP")
        mdd = d.get("mdd_R")
        ev_s   = f"{float(ev):+.4f}R" if ev   is not None else "—"
        prec_s = f"{float(prec):.3f}" if prec is not None else "—"
        mdd_s  = f"{float(mdd):.1f}R" if mdd  is not None else "—"
        print(f"  {state:<20} {d['n_rows']:>8} {int(sig):>6} {ev_s:>10} {prec_s:>8} {mdd_s:>8}")


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
    ap.add_argument("--cost-per-signal", type=float, default=0.05)
    ap.add_argument("--max-drawdown-R", type=float, default=30.0)
    ap.add_argument("--min-signals-per-state", type=int, default=10,
                    help="Mín signals en un estado para reportar su EV.")
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
    needed = feat_cols + ["time", "high", "low", "close", "atr", "signal_long", "signal_short", "state"]
    df_clean = df_prepared.loc[df_prepared[needed].notna().all(axis=1)].reset_index(drop=True)

    tr_mask = df_clean["time"] < pd.Timestamp(args.holdout_from)
    hd_mask = df_clean["time"] >= pd.Timestamp(args.holdout_from)
    df_tr, df_hd = df_clean.loc[tr_mask].reset_index(drop=True), df_clean.loc[hd_mask].reset_index(drop=True)

    X_tr = df_tr[feat_cols].astype(np.float32).values
    X_hd = df_hd[feat_cols].astype(np.float32).values
    y_long_tr  = df_tr["signal_long"].astype(np.int8).values
    y_short_tr = df_tr["signal_short"].astype(np.int8).values
    w_tr = df_tr["regime_weight"].astype(np.float32).values \
        if "regime_weight" in df_tr.columns else np.ones(len(df_tr), dtype=np.float32)

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

    p_long_hd_cal  = cal_long.predict(booster_long.predict(X_hd).astype(np.float64)).astype(np.float32)
    p_short_hd_cal = cal_short.predict(booster_short.predict(X_hd).astype(np.float64)).astype(np.float32)

    df_oof_hold = pd.DataFrame({
        "time": df_hd["time"].values,
        "high": df_hd["high"].astype(np.float64).values,
        "low":  df_hd["low"].astype(np.float64).values,
        "close":df_hd["close"].astype(np.float64).values,
        "atr":  df_hd["atr"].astype(np.float64).values,
        "state":df_hd["state"].values,
        "signal_long":  df_hd["signal_long"].astype(np.int8).values,
        "signal_short": df_hd["signal_short"].astype(np.int8).values,
        "oof_proba_long_cal":  p_long_hd_cal,
        "oof_proba_short_cal": p_short_hd_cal,
    })

    states = sorted(df_oof_hold["state"].astype(str).unique())
    print(f"\n📋 Estados en holdout: {states}")

    horizon = int(feature_config.label_horizon)
    tp_mult = float(feature_config.tp_barrier)
    sl_mult = float(feature_config.sl_barrier)

    per_state_long = _eval_per_state(
        df_oof_hold, proba_col="oof_proba_long_cal", side_is_long=True,
        thr_oof=thr_long, states=states,
        horizon=horizon, tp_mult=tp_mult, sl_mult=sl_mult,
        cost_per_signal=args.cost_per_signal,
        max_drawdown_R=args.max_drawdown_R,
        min_signals_per_state=int(args.min_signals_per_state),
    )
    per_state_short = _eval_per_state(
        df_oof_hold, proba_col="oof_proba_short_cal", side_is_long=False,
        thr_oof=thr_short, states=states,
        horizon=horizon, tp_mult=tp_mult, sl_mult=sl_mult,
        cost_per_signal=args.cost_per_signal,
        max_drawdown_R=args.max_drawdown_R,
        min_signals_per_state=int(args.min_signals_per_state),
    )

    _print_state_table("long",  per_state_long)
    _print_state_table("short", per_state_short)

    print("\n══ INTERPRETACIÓN ══════════════════════════════════════════════")
    for side, ps in (("LONG ", per_state_long), ("SHORT", per_state_short)):
        valid = {k: v for k, v in ps.items() if not v.get("skipped")}
        pos_states = [k for k, v in valid.items()
                      if v.get("ev_net") is not None and float(v["ev_net"]) > 0]
        neg_states = [k for k, v in valid.items()
                      if v.get("ev_net") is not None and float(v["ev_net"]) < 0]
        print(f"  {side}: estados rentables = {pos_states}")
        print(f"         estados negativos = {neg_states}")
        if pos_states and not neg_states:
            print(f"    ✅  Modelo uniformemente positivo (raro, sospechar overfit)")
        elif len(pos_states) >= 3:
            print(f"    ✅  Señal generaliza a múltiples regímenes")
        elif len(pos_states) <= 1:
            print(f"    ⚠️  Señal concentrada en 1 régimen — considerar specialist por estado")

    report = {
        "release": release, "long_trial": n_long, "short_trial": n_short,
        "thr_long": thr_long, "thr_short": thr_short,
        "horizon": horizon, "tp_mult": tp_mult, "sl_mult": sl_mult,
        "states": states,
        "per_state_long":  per_state_long,
        "per_state_short": per_state_short,
        "holdout_period": [str(args.holdout_from.date()), str(args.holdout_to.date())],
    }
    out_json = args.out_json or str(Path(args.best_json).parent / "per_state_diag.json")
    with open(out_json, "w") as fh: json.dump(report, fh, indent=2, default=str)
    print(f"\n📁 Reporte: {out_json}")


if __name__ == "__main__":
    main()
