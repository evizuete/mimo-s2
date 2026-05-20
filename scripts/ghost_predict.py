#!/usr/bin/env python3
"""
ghost_predict.py — Ghost simulation: aplica el modelo deployado sobre OHLCV
de un rango histórico arbitrario y emite un parquet con predicciones +
outcomes triple barrier listos para los analyzers de mimo/oof/shift_analyzer/.

Caso de uso
───────────
Tienes un deploy (p.ej. cutoff mar-31) y quieres saber si su calibración
sigue siendo válida sobre abril (datos que el deploy no vio). Este script:

  1. Construye el TradingSimulator con el deploy elegido (reusa build_simulator
     de replay_s2_202500.py → autodetecta multitask vs binary).
  2. Carga OHLCV 1min de BD para el rango [from, to].
  3. Ejecuta simulator.predict() → obtiene p_raw, p_cal y state por barra
     (alineadas a 5min, base del modelo).
  4. Resamplea OHLCV a 5min y aplica triple barrier sobre cada predicción.
  5. Emite parquet con:
       time | side | state | oof_proba_raw | oof_proba_cal |
       outcome | signal | R_multiple | period
  6. Si --include-tail, concatena `deploy_calibration_tail_<release>_<side>.parquet`
     del deploy marcado como period='train' (TRAIN ∪ HOLDOUT en un solo archivo).

El parquet resultante es directamente consumible por:
  - mimo.oof.shift_analyzer.analyze_calibration_mimo
  - mimo.oof.shift_analyzer.analyze_model_drift
  - mimo.oof.shift_analyzer.analyze_threshold_recalibration
  - mimo.oof.shift_analyzer.analyze_percentile_stability
  - mimo.oof.shift_analyzer.analyze_score_to_pnl

Uso
───
  python scripts/ghost_predict.py \
    --release 202500 \
    --deploy-subdir deploy_2026_04_combined_specialists_seed47 \
    --from 2026-04-01 --to 2026-04-30 \
    --include-tail \
    --out /tmp/ghost_apr.parquet
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

# Reuse build_simulator + load_ohlcv del replay.
# Necesitamos DOS paths en sys.path:
#   1. scripts/ para encontrar replay_s2_202500.py por nombre no-namespaced.
#   2. project root para que `from mimo...` (que replay_s2_202500 hace
#      internamente) funcione cuando este script se invoca como
#      `python3 scripts/ghost_predict.py` desde el root del repo. Sin esta
#      línea el import de mimo falla con ModuleNotFoundError porque Python
#      solo añade automáticamente el dir DEL script (scripts/) a sys.path,
#      no el proyecto root.
_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent
sys.path.insert(0, str(_PROJECT_ROOT))  # para `from mimo.*` en replay_s2_202500
sys.path.insert(0, str(_THIS_DIR))      # para `from replay_s2_202500`
from replay_s2_202500 import build_simulator, load_ohlcv  # noqa: E402

from mimo.oof.empirical_breakeven import simulate_outcomes, wilder_atr  # noqa: E402


def _resample_to_5min(df_1min: pd.DataFrame) -> pd.DataFrame:
    """Resamplea OHLCV 1min → 5min (los predicts del modelo van en 5min)."""
    df = df_1min.copy()
    df["time"] = pd.to_datetime(df["time"])
    out = (
        df.set_index("time")[["open", "high", "low", "close"]]
        .resample("5min", label="right", closed="right")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last"})
        .dropna()
        .reset_index()
    )
    return out


def _per_side_df(
    df_pred: pd.DataFrame,
    ohlcv_5m: pd.DataFrame,
    side: str,
    tp_mult: float,
    sl_mult: float,
    horizon: int,
    atr_window: int = 14,
) -> pd.DataFrame:
    """Triple barrier sobre cada predicción del modelo para un side concreto."""
    proba_raw_col = f"pred_{side}_raw"
    proba_cal_col = f"pred_{side}_cal"
    if proba_raw_col not in df_pred.columns or proba_cal_col not in df_pred.columns:
        raise SystemExit(
            f"❌ predict() no devolvió {proba_raw_col}/{proba_cal_col}. "
            f"Cols disponibles: {list(df_pred.columns)[:20]}..."
        )

    if "state" in df_pred.columns:
        state_series = df_pred["state"].astype(str).values
    elif "market_condition" in df_pred.columns:
        state_series = df_pred["market_condition"].astype(str).values
    else:
        state_series = np.array(["unknown"] * len(df_pred))

    j = pd.DataFrame({
        "time": pd.to_datetime(df_pred["time"]),
        "oof_proba_raw": df_pred[proba_raw_col].values,
        "oof_proba_cal": df_pred[proba_cal_col].values,
        "state": state_series,
    })

    ohlcv = ohlcv_5m.copy().reset_index(drop=True)
    ohlcv["__row__"] = np.arange(len(ohlcv))
    j = j.merge(ohlcv[["time", "__row__"]], on="time", how="inner")
    if j.empty:
        raise SystemExit(
            f"❌ Sin alineación temporal predict ↔ ohlcv 5min para side={side}. "
            f"Predict times: {df_pred['time'].iloc[0]} → {df_pred['time'].iloc[-1]}. "
            f"OHLCV 5min: {ohlcv['time'].iloc[0]} → {ohlcv['time'].iloc[-1]}."
        )

    atr = wilder_atr(
        ohlcv["high"].to_numpy(),
        ohlcv["low"].to_numpy(),
        ohlcv["close"].to_numpy(),
        period=atr_window,
    )
    out = simulate_outcomes(
        j["__row__"].to_numpy(),
        ohlcv["high"].to_numpy(),
        ohlcv["low"].to_numpy(),
        ohlcv["close"].to_numpy(),
        atr,
        horizon=horizon,
        tp_mult=tp_mult,
        sl_mult=sl_mult,
        side_is_long=(side == "long"),
    )
    j = pd.concat([j.reset_index(drop=True), out], axis=1)
    j["side"] = side
    j["signal"] = (j["outcome"] == "TP").astype(int)
    keep = ["time", "side", "state", "oof_proba_raw", "oof_proba_cal",
            "outcome", "signal", "R_multiple"]
    return j[keep].reset_index(drop=True)


def _load_tail_parquet(deploy_dir: Path, release: str, side: str) -> Optional[pd.DataFrame]:
    """Carga deploy_calibration_tail_<release>_<side>.parquet del deploy si existe.

    Preserva la columna 'signal' del tail original (label triple barrier del
    entrenamiento). Solo la sintetiza desde 'outcome' si NO existe.
    """
    p = deploy_dir / "data" / f"deploy_calibration_tail_{release}_{side}.parquet"
    if not p.exists():
        print(f"  ⚠️  No existe {p.name} (deploy creado sin resume_deploy_v6), tail TRAIN no incluida.")
        return None
    df = pd.read_parquet(p)
    df["time"] = pd.to_datetime(df["time"])
    df["side"] = side
    if "state" not in df.columns:
        df["state"] = "unknown"
    if "oof_proba_raw" not in df.columns:
        df["oof_proba_raw"] = np.nan
    # Preservar 'signal' si ya existe (label original del tail).
    # Solo sintetizar desde 'outcome' si signal NO existe.
    if "signal" not in df.columns:
        if "outcome" in df.columns:
            df["signal"] = (df["outcome"] == "TP").astype(int)
        else:
            df["signal"] = np.nan
    if "outcome" not in df.columns:
        df["outcome"] = np.nan
    if "R_multiple" not in df.columns:
        df["R_multiple"] = np.nan
    keep = ["time", "side", "state", "oof_proba_raw", "oof_proba_cal",
            "outcome", "signal", "R_multiple"]
    return df[[c for c in keep if c in df.columns]].reset_index(drop=True)

def main() -> None:
    ap = argparse.ArgumentParser(
        description="Ghost simulation: aplica el modelo deployado sobre OHLCV "
                    "histórico y emite parquet listo para los analyzers de drift."
    )
    ap.add_argument("--release", required=True)
    ap.add_argument("--deploy-subdir", required=True,
                    help="Subdir bajo artifacts/<release>/oof/.")
    ap.add_argument("--from", dest="from_date", required=True,
                    help="YYYY-MM-DD inicio de la ventana a predecir.")
    ap.add_argument("--to", dest="to_date", required=True,
                    help="YYYY-MM-DD fin.")
    ap.add_argument("--side", choices=["long", "short", "both"], default="both")
    ap.add_argument("--base-tf", default="1min",
                    help="TF de carga del OHLCV (el modelo internamente resamplea "
                         "a 5min para predicciones). Default: 1min.")
    ap.add_argument("--policy-config", default="decision_policies_config_202500",
                    help="Policy module (igual que en replay).")
    ap.add_argument("--include-tail", action="store_true",
                    help="Concatena el deploy_calibration_tail_*.parquet del deploy "
                         "marcado como period='train'.")
    ap.add_argument("--tp-long", type=float, default=2.0)
    ap.add_argument("--sl-long", type=float, default=0.8)
    ap.add_argument("--horizon-long", type=int, default=3)
    ap.add_argument("--tp-short", type=float, default=2.0)
    ap.add_argument("--sl-short", type=float, default=0.8)
    ap.add_argument("--horizon-short", type=int, default=3)
    ap.add_argument("--atr-window", type=int, default=14)
    ap.add_argument("--out", required=True,
                    help="Path del parquet de salida.")
    args = ap.parse_args()

    base_dir = _THIS_DIR
    repo_root = base_dir.parent
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    print("\n🔧 Construyendo TradingSimulator...")
    simulator = build_simulator(
        release=args.release,
        deploy_subdir=args.deploy_subdir,
        policy_module=f"config.{args.policy_config}",
        base_dir=base_dir,
        artifacts_root=repo_root / "artifacts",
    )

    df_ohlcv = load_ohlcv(args.from_date, args.to_date, args.base_tf)
    if len(df_ohlcv) < 8000:
        print(f"⚠️  Solo {len(df_ohlcv)} filas en {args.base_tf}. "
              "Multi-TF features (1h) requieren ≥7500 1m bars; "
              "considera ampliar el rango con un warmup previo.")

    print("\n🔮 Ejecutando simulator.predict() sobre el rango completo...")
    df_pred = simulator.predict(df_ohlcv, simulation=True)
    print(f"   predicciones: {len(df_pred):,} filas | "
          f"{df_pred['time'].iloc[0]} → {df_pred['time'].iloc[-1]}")

    print("\n📊 Resampleando OHLCV a 5min para triple barrier...")
    ohlcv_5m = _resample_to_5min(df_ohlcv)
    print(f"   5min: {len(ohlcv_5m):,} barras | "
          f"{ohlcv_5m['time'].iloc[0]} → {ohlcv_5m['time'].iloc[-1]}")

    sides = ["long", "short"] if args.side == "both" else [args.side]
    parts = []
    artifacts_root = repo_root / "artifacts"
    deploy_dir = artifacts_root / args.release / "oof" / args.deploy_subdir

    for side in sides:
        if side == "long":
            tp, sl, hz = args.tp_long, args.sl_long, args.horizon_long
        else:
            tp, sl, hz = args.tp_short, args.sl_short, args.horizon_short
        print(f"\n  ▶ side={side}  (tp={tp}, sl={sl}, h={hz})")
        side_df = _per_side_df(df_pred, ohlcv_5m, side, tp, sl, hz, args.atr_window)
        side_df["period"] = "holdout"
        counts = dict(side_df["outcome"].value_counts())
        print(f"    n={len(side_df):,}  outcomes: {counts}")
        parts.append(side_df)

        if args.include_tail:
            tail_df = _load_tail_parquet(deploy_dir, args.release, side)
            if tail_df is not None:
                tail_df["period"] = "train"
                parts.append(tail_df)
                print(f"    + tail TRAIN concatenada: n={len(tail_df):,}")

    final = (
        pd.concat(parts, axis=0, ignore_index=True)
        .sort_values(["side", "time"])
        .reset_index(drop=True)
    )

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.to_parquet(out_path, index=False)

    print("\n" + "═" * 78)
    print("  RESUMEN")
    print("═" * 78)
    print(f"  📁 Emitido → {out_path}  ({len(final):,} rows)")
    print(f"     side    : {dict(final['side'].value_counts())}")
    print(f"     period  : {dict(final['period'].value_counts())}")
    print(f"     state   : {dict(final['state'].value_counts())}")
    if "outcome" in final.columns:
        valid_outcomes = final["outcome"].dropna()
        if len(valid_outcomes) > 0:
            print(f"     outcomes: {dict(valid_outcomes.value_counts())}")

    print("\n💡 Próximo paso — analizar drift de calibración:")
    print(f"   python -m mimo.oof.shift_analyzer.analyze_calibration_mimo \\")
    print(f"     --input {out_path} \\")
    print(f"     --score-col oof_proba_cal --target-col signal \\")
    print(f"     --state-col state --time-col time \\")
    print(f"     --period-col period \\")
    print(f"     --output-dir reports/{out_path.stem}_cal")


if __name__ == "__main__":
    main()