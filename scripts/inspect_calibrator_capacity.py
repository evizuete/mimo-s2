#!/usr/bin/env python3
"""
inspect_calibrator_capacity.py — Compara la capacidad discriminativa del
isotonic actual (21d) vs uno entrenado con 45d sobre los mismos datos.

Sin retrain del modelo, sin tocar producción. Te dice si la cola alta
gana resolución al ampliar el tail.
"""
from __future__ import annotations
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

RELEASE = "202500"
TAG = "rw_both_Lvol_boost_td_down_h3_Svol_boost_h3"
DEPLOY = Path(f"../artifacts/{RELEASE}/oof/deploy_2026_04_combined_specialists_seed47")


def _find_holdout_preds(side: str) -> Path:
    """Busca holdout_predictions_{release}_{side}.parquet del specialist dir."""
    bases = [
        Path(f"../artifacts/{RELEASE}/oof/{TAG}_{side}_specialist_seed47"),
        Path(f"../artifacts/{RELEASE}/oof/{TAG}_{side}_specialist_seed47_cutoff_mar31"),
    ]
    target_name = f"holdout_predictions_{RELEASE}_{side}.parquet"
    for base in bases:
        if not base.exists():
            continue
        # Búsqueda directa
        for sub in (base, base / "data"):
            p = sub / target_name
            if p.exists():
                return p
        # Búsqueda recursiva como último recurso (filtrando por side)
        candidates = [p for p in base.rglob("*.parquet")
                      if f"_{side}." in p.name and "holdout" in p.name.lower()]
        if candidates:
            return candidates[0]
    raise SystemExit(f"❌ No encuentro {target_name} en {bases}")

for side in ["long", "short"]:
    print(f"\n{'═'*72}\n  SIDE = {side.upper()}\n{'═'*72}")

    # 1) Tail actual (21d) tal como está en el deploy
    tail_21_path = DEPLOY / "data" / f"deploy_calibration_tail_{RELEASE}_{side}.parquet"
    if not tail_21_path.exists():
        print(f"  ⚠️  Sin {tail_21_path.name} — pasa al siguiente")
        continue
    tail_21 = pd.read_parquet(tail_21_path)
    tail_21["time"] = pd.to_datetime(tail_21["time"])

    print(f"\n  📊 Tail 21d (actual):")
    print(f"     n={len(tail_21):,}  rango={tail_21['time'].min().date()} → {tail_21['time'].max().date()}")
    cap_21 = float(tail_21["oof_proba_cal"].max())
    p99_21 = float(tail_21["oof_proba_cal"].quantile(0.99))
    p95_21 = float(tail_21["oof_proba_cal"].quantile(0.95))
    n_unique_21 = int(tail_21["oof_proba_cal"].nunique())
    saturated_21 = int((tail_21["oof_proba_cal"] >= cap_21 - 1e-9).sum())
    print(f"     cap (max cal):   {cap_21:.4f}")
    print(f"     p95 cal:         {p95_21:.4f}")
    print(f"     p99 cal:         {p99_21:.4f}")
    print(f"     valores únicos:  {n_unique_21}")
    print(f"     en cap (saturados): {saturated_21}/{len(tail_21)} ({100*saturated_21/len(tail_21):.1f}%)")

    # 2) Holdout completo → simular tail de 45d
    hp_path = _find_holdout_preds(side)
    print(f"\n  📂 Holdout predictions: {hp_path.name}")
    hp = pd.read_parquet(hp_path)
    hp["time"] = pd.to_datetime(hp["time"])

    raw_col = next((c for c in ("y_pred_raw", "oof_proba_raw", "p_raw") if c in hp.columns), None)
    sig_col = next((c for c in ("y_true", "signal", "y") if c in hp.columns), None)
    if raw_col is None or sig_col is None:
        print(f"  ❌ Cols requeridas no en {hp_path}: cols={list(hp.columns)}")
        continue

    cutoff = hp["time"].max()
    tail_45 = hp[hp["time"] >= cutoff - pd.Timedelta(days=45)].copy()
    print(f"\n  📊 Tail 45d (simulada):")
    print(f"     n={len(tail_45):,}  rango={tail_45['time'].min().date()} → {tail_45['time'].max().date()}")

    iso_45 = IsotonicRegression(out_of_bounds="clip")
    iso_45.fit(tail_45[raw_col].to_numpy(), tail_45[sig_col].to_numpy())
    cal_45_on_45 = iso_45.predict(tail_45[raw_col].to_numpy())

    cap_45 = float(cal_45_on_45.max())
    p99_45 = float(np.quantile(cal_45_on_45, 0.99))
    p95_45 = float(np.quantile(cal_45_on_45, 0.95))
    n_unique_45 = len(np.unique(cal_45_on_45))
    saturated_45 = int((cal_45_on_45 >= cap_45 - 1e-9).sum())
    print(f"     cap (max cal):   {cap_45:.4f}")
    print(f"     p95 cal:         {p95_45:.4f}")
    print(f"     p99 cal:         {p99_45:.4f}")
    print(f"     valores únicos:  {n_unique_45}")
    print(f"     en cap:          {saturated_45}/{len(tail_45)} ({100*saturated_45/len(tail_45):.1f}%)")

    # 3) Comparativa
    print(f"\n  🎯 COMPARATIVA (45d vs 21d):")
    print(f"     cap:             {cap_45:.4f} vs {cap_21:.4f}   "
          f"Δ={cap_45-cap_21:+.4f}  "
          f"{'✅ MEJORA' if cap_45 > cap_21 + 0.005 else '⚠️  similar'}")
    print(f"     valores únicos:  {n_unique_45} vs {n_unique_21}   "
          f"Δ={n_unique_45-n_unique_21:+d}  "
          f"{'✅ más resolución' if n_unique_45 > n_unique_21 * 1.5 else '⚠️  similar'}")
    print(f"     p99 cal:         {p99_45:.4f} vs {p99_21:.4f}   "
          f"Δ={p99_45-p99_21:+.4f}")

    # 4) Diagnóstico
    print(f"\n  📌 DIAGNÓSTICO:")
    if cap_45 > cap_21 + 0.02 and n_unique_45 > n_unique_21 * 1.3:
        print(f"     🟢 Tail 45d aporta MÁS RESOLUCIÓN. Cambio justificado.")
    elif cap_45 > cap_21 + 0.005:
        print(f"     🟡 Tail 45d aporta algo. Cambio marginal.")
    else:
        print(f"     🔴 Tail 45d NO mejora. El problema NO es el tamaño del tail.")
        print(f"        Considera Beta calibration (paso 2) en lugar de ampliar tail.")