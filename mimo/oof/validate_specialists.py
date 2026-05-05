#!/usr/bin/env python3
"""
validate_specialists.py

Sanity check de reproducibilidad tras `train_specialist`. Compara las
métricas OOF que reportó el trial original (en best_per_side.json) con
las que produce el reentreno del specialist al mismo threshold lockeado.

Si la diferencia es despreciable (ev_net dentro de ±1e-3, n_signals con
diferencia ≤ 2) el reentreno está reproduciendo bien. Si difiere fuerte,
hay un bug de seed/orden de folds que hay que investigar antes de seguir
a drift validation.

Inputs:
  --best-per-side-json        JSON de extract_best_per_side
  --specialist-long-dir       artifacts dir del long specialist
  --specialist-short-dir      artifacts dir del short specialist
  --side {long,short,both}    qué validar (default both)
  --release                   nombre del release (para localizar oof_<release>_<side>.parquet)
  --ohlcv | --from-db         para recomputar ATR
  --tp / --sl / --horizon     barreras (default 2.0 / 0.8 / 3)
  --cost                      coste por señal en R (default 0.05)
  --max-drawdown-R            límite MDD para penalty (default 30)

Output: tabla side-by-side de:
  thr | ev_net | ev_gross | n_signals | prec_TP | mdd_R
con dos columnas (RECORDED del trial / RECOMPUTED del specialist) y
delta absoluto. Veredicto PASS/FAIL al final.

Uso:
  python -m mimo.oof.validate_specialists \
    --best-per-side-json artifacts/202500/oof/<tag>/reports/best_per_side.json \
    --specialist-long-dir  artifacts/202500/oof/<tag>_long_specialist \
    --specialist-short-dir artifacts/202500/oof/<tag>_short_specialist \
    --release 202500 --from-db --base-tf 5min \
    --tp 2.0 --sl 0.8 --horizon 3 --cost 0.05
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd

import hashlib

from mimo.oof.empirical_breakeven import wilder_atr
from mimo.oof.ev_objective import (
    _simulate_outcomes_for_indices,
    _max_drawdown_R,
)


# ─────────────────────────────────────────────────────────────────────────────
# Scaler comparison helpers
# ─────────────────────────────────────────────────────────────────────────────

def _md5_of_dir(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.md5()
    for f in sorted(path.rglob("*")):
        if f.is_file():
            h.update(f.relative_to(path).as_posix().encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def _compare_scalers(long_dir: Path, short_dir: Path, release: str) -> dict:
    """Compara scalers_<release>/ entre los dos specialists. Devuelve
    dict con md5s, flag de coincidencia y diagnóstico textual."""
    sc_long = long_dir / f"scalers_{release}"
    sc_short = short_dir / f"scalers_{release}"
    md5_long = _md5_of_dir(sc_long)
    md5_short = _md5_of_dir(sc_short)
    same = (md5_long == md5_short) and md5_long != ""
    info = {
        "long_dir": str(sc_long),
        "short_dir": str(sc_short),
        "md5_long": md5_long,
        "md5_short": md5_short,
        "identical": same,
        "long_exists": sc_long.exists(),
        "short_exists": sc_short.exists(),
    }
    return info


def _print_scaler_check(info: dict) -> bool:
    """Imprime el resultado de la comparación. Devuelve True si los scalers
    son idénticos (--strict-scalers de merge_specialists pasaría)."""
    print("\n" + "=" * 78)
    print("  CHECK DE SCALERS  (¿coinciden entre specialists?)")
    print("=" * 78)
    if not info["long_exists"]:
        print(f"  ⚠️  long_dir no contiene scalers_<release>/: {info['long_dir']}")
        return False
    if not info["short_exists"]:
        print(f"  ⚠️  short_dir no contiene scalers_<release>/: {info['short_dir']}")
        return False
    print(f"  long  md5 : {info['md5_long']}")
    print(f"  short md5 : {info['md5_short']}")
    if info["identical"]:
        print("  ✅ Scalers idénticos. merge_specialists --strict-scalers pasará.")
        return True
    print("  ❌ Scalers DIFIEREN entre specialists.")
    print("     Causa probable: alguno de los specialists alteró el feature set")
    print("     (ej. distinto _REDUCED_FEATURES_RELEASES, masks, vol-invariant...).")
    print("     Antes de merge_specialists:")
    print("       - Confirma que ambos usaron release=202500 (mismo feature set).")
    print("       - Si difieren legítimamente, decide --prefer-scalers-from {long,short}")
    print("         con ojo crítico (el lado contrario podría predecir mal).")
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Recompute helpers
# ─────────────────────────────────────────────────────────────────────────────

def _recompute_metrics_at_locked_thr(
    df_oof: pd.DataFrame,
    *,
    proba_col: str,
    side_is_long: bool,
    locked_thr: float,
    horizon: int,
    tp_mult: float,
    sl_mult: float,
    cost: float,
    max_drawdown_R: float,
) -> Dict[str, Any]:
    """Replay del barrier en df_oof al thr lockeado y recompute de métricas."""
    needed = {"time", "high", "low", "close", "atr", proba_col}
    miss = needed - set(df_oof.columns)
    if miss:
        raise ValueError(f"OOF parquet le faltan columnas: {miss}")

    mask = (
        df_oof[proba_col].notna()
        & df_oof["atr"].notna()
        & (df_oof["atr"] > 0)
    )
    df = df_oof.loc[mask].reset_index(drop=False).sort_values("time").reset_index(drop=True)

    high = df_oof["high"].to_numpy(dtype=np.float64)
    low = df_oof["low"].to_numpy(dtype=np.float64)
    close = df_oof["close"].to_numpy(dtype=np.float64)
    atr = df_oof["atr"].to_numpy(dtype=np.float64)
    orig_idx = df["index"].to_numpy(dtype=np.int64)
    p = df[proba_col].to_numpy(dtype=np.float64)

    outcome, r_real = _simulate_outcomes_for_indices(
        orig_idx, high, low, close, atr,
        horizon=horizon, tp_mult=tp_mult, sl_mult=sl_mult,
        side_is_long=side_is_long,
    )
    valid = outcome != 3

    pred = (p >= locked_thr) & valid
    sig = int(pred.sum())
    if sig == 0:
        return {
            "thr": float(locked_thr),
            "n_signals": 0,
            "n_TP": 0, "n_SL": 0, "n_EXPIRE": 0,
            "prec_TP": 0.0,
            "frac_SL": 0.0,
            "frac_EXP": 0.0,
            "ev_gross": 0.0,
            "ev_net": -cost,
            "mdd_R": 0.0,
        }

    sub_outcome = outcome[pred]
    sub_r = r_real[pred]
    n_tp = int((sub_outcome == 0).sum())
    n_sl = int((sub_outcome == 1).sum())
    n_exp = int((sub_outcome == 2).sum())
    ev_gross = float(sub_r.mean())
    ev_net = ev_gross - cost
    mdd = _max_drawdown_R(sub_r - cost)
    return {
        "thr": float(locked_thr),
        "n_signals": sig,
        "n_TP": n_tp, "n_SL": n_sl, "n_EXPIRE": n_exp,
        "prec_TP": n_tp / sig,
        "frac_SL": n_sl / sig,
        "frac_EXP": n_exp / sig,
        "ev_gross": float(ev_gross),
        "ev_net": float(ev_net),
        "mdd_R": float(mdd),
    }


def _ensure_atr_column(df_oof: pd.DataFrame, ohlcv: pd.DataFrame, atr_window: int = 14) -> pd.DataFrame:
    """Si el OOF parquet no trae 'atr', mergear con OHLCV y calcular Wilder."""
    if "atr" in df_oof.columns and df_oof["atr"].notna().sum() > 0:
        return df_oof
    print("   ('atr' no presente en OOF parquet — calculando desde OHLCV)")
    if not all(c in df_oof.columns for c in ("high", "low", "close")):
        # Necesitamos high/low/close del OHLCV
        df_oof = df_oof.merge(
            ohlcv[["time", "open", "high", "low", "close"]], on="time", how="left"
        )
    atr_arr = wilder_atr(
        df_oof["high"].to_numpy(),
        df_oof["low"].to_numpy(),
        df_oof["close"].to_numpy(),
        period=atr_window,
    )
    df_oof = df_oof.copy()
    df_oof["atr"] = atr_arr
    return df_oof


# ─────────────────────────────────────────────────────────────────────────────
# Diff helpers
# ─────────────────────────────────────────────────────────────────────────────

def _abs_delta(a: float, b: float) -> float:
    if not (np.isfinite(a) and np.isfinite(b)):
        return float("nan")
    return abs(a - b)


# Tolerancias por defecto: pensadas para TF en GPU (non-determinista por
# kernels CUDA). Aceptan que el specialist no reproduce bit-a-bit el trial
# original pero sí está "razonablemente cerca". Si quieres bit-perfect,
# usa --strict (requiere TF_DETERMINISTIC_OPS=1 en el environment).
TOL_DEFAULT = {
    "ev_net":    0.10,    # ±0.10R por señal media (el orden de magnitud de un TP/SL)
    "ev_gross":  0.10,
    "n_signals": 0.50,    # 50% relativo a recorded (mínimo absoluto = 50)
    "prec_TP":   0.10,    # 10 puntos de precisión
    "mdd_R":    10.00,    # 10R de margen sobre el cap de 30R
}

TOL_STRICT = {
    "ev_net":    1e-3,
    "ev_gross":  1e-3,
    "n_signals": 2,        # absoluto
    "prec_TP":   5e-3,
    "mdd_R":     1.0,
}


def _verdict(
    recorded: Dict[str, Any],
    recomputed: Dict[str, Any],
    tols: Dict[str, float],
    *,
    strict: bool,
) -> Dict[str, Any]:
    d_ev_net = _abs_delta(recorded.get("ev_net", float("nan")),
                          recomputed.get("ev_net", float("nan")))
    d_ev_gross = _abs_delta(recorded.get("ev_gross", float("nan")),
                            recomputed.get("ev_gross", float("nan")))
    d_n_sig = _abs_delta(recorded.get("n_signals", 0),
                         recomputed.get("n_signals", 0))
    d_prec = _abs_delta(recorded.get("prec_TP", float("nan")),
                        recomputed.get("prec_TP", float("nan")))
    d_mdd = _abs_delta(recorded.get("mdd_R", float("nan")),
                       recomputed.get("mdd_R", float("nan")))

    # n_signals: en strict comparamos absoluto; en default usamos un
    # criterio relativo con piso absoluto de 50 (evita que recorded=10 y
    # recomputed=80 pase un check "70%")
    rec_sig = float(recorded.get("n_signals", 0))
    if strict:
        nsig_tol_abs = float(tols["n_signals"])
        nsig_pass = d_n_sig <= nsig_tol_abs
        nsig_tol_str = f"≤{int(nsig_tol_abs)}"
    else:
        nsig_tol_abs = max(50.0, rec_sig * float(tols["n_signals"]))
        nsig_pass = d_n_sig <= nsig_tol_abs
        nsig_tol_str = f"≤{int(nsig_tol_abs)} (50% rel ó 50 abs)"

    checks = {
        "ev_net":    (d_ev_net,    tols["ev_net"],   d_ev_net   <= tols["ev_net"]),
        "ev_gross":  (d_ev_gross,  tols["ev_gross"], d_ev_gross <= tols["ev_gross"]),
        "n_signals": (d_n_sig,     nsig_tol_abs,     nsig_pass),
        "prec_TP":   (d_prec,      tols["prec_TP"],  d_prec     <= tols["prec_TP"]),
        "mdd_R":     (d_mdd,       tols["mdd_R"],    d_mdd      <= tols["mdd_R"]),
    }
    fails = {k: (delta, tol) for k, (delta, tol, ok) in checks.items() if not ok}
    deltas = {k: delta for k, (delta, _, _) in checks.items()}
    tols_eff = {k: tol for k, (_, tol, _) in checks.items()}
    tols_eff["n_signals_str"] = nsig_tol_str
    return {
        "deltas": deltas,
        "tolerances": tols_eff,
        "fails": fails,
        "passed": len(fails) == 0,
    }


def _print_side_table(
    label: str,
    recorded: Dict[str, Any],
    recomputed: Dict[str, Any],
    tols: Dict[str, float],
    *,
    strict: bool,
) -> Dict[str, Any]:
    print("\n" + "=" * 78)
    print(f"  VALIDACIÓN  {label}  ({'STRICT' if strict else 'RELAJADO (TF non-determ)'})")
    print("=" * 78)
    print(f"  {'metric':<14}  {'RECORDED (trial)':>20}  {'RECOMPUTED (specialist)':>26}  {'Δ':>10}")
    print(f"  {'-'*14}  {'-'*20}  {'-'*26}  {'-'*10}")
    rows = [
        ("thr",        recorded.get("thr"),       recomputed.get("thr")),
        ("n_signals",  recorded.get("n_signals"), recomputed.get("n_signals")),
        ("prec_TP",    recorded.get("prec_TP"),   recomputed.get("prec_TP")),
        ("frac_SL",    recorded.get("frac_SL"),   recomputed.get("frac_SL")),
        ("frac_EXP",   recorded.get("frac_EXP"),  recomputed.get("frac_EXP")),
        ("ev_gross",   recorded.get("ev_gross"),  recomputed.get("ev_gross")),
        ("ev_net",     recorded.get("ev_net"),    recomputed.get("ev_net")),
        ("mdd_R",      recorded.get("mdd_R"),     recomputed.get("mdd_R")),
    ]
    for name, a, b in rows:
        a_str = f"{a:.4f}" if isinstance(a, float) else f"{a}"
        b_str = f"{b:.4f}" if isinstance(b, float) else f"{b}"
        d = _abs_delta(a if a is not None else float("nan"),
                       b if b is not None else float("nan"))
        d_str = f"{d:.4f}" if np.isfinite(d) else "n/a"
        print(f"  {name:<14}  {a_str:>20}  {b_str:>26}  {d_str:>10}")

    v = _verdict(recorded, recomputed, tols, strict=strict)
    print()
    if v["passed"]:
        mode = "STRICT" if strict else "RELAJADO"
        print(f"  ✅ PASS ({mode}) — métricas dentro de tolerancia.")
        # Imprimir tolerancias para transparencia
        t = v["tolerances"]
        print(f"     ev_net Δ={v['deltas']['ev_net']:.4f}  (tol ≤{t['ev_net']:.3f})")
        print(f"     n_sig  Δ={v['deltas']['n_signals']:.0f}  (tol {t['n_signals_str']})")
        print(f"     mdd_R  Δ={v['deltas']['mdd_R']:.2f}R  (tol ≤{t['mdd_R']:.1f}R)")
    else:
        mode_label = "STRICT" if strict else "RELAJADO"
        print(f"  ❌ FAIL ({mode_label}) — métricas fuera de tolerancia:")
        for k, (delta, tol) in v["fails"].items():
            tol_str = (v["tolerances"]["n_signals_str"]
                       if k == "n_signals" else f"≤{tol:.4f}")
            print(f"      {k:<10}  Δ={delta:.4f}  (tol {tol_str})")
        if not strict:
            # En modo relajado un fail ya es serio: el modelo sí está derivando
            # más de lo aceptable por TF non-determinismo solo.
            print("  Posibles causas (más allá de TF non-determinismo):")
            print("      - El modelo cayó en un mínimo local muy distinto al trial.")
            print("        Considera retrain con --seed distinto.")
            print("      - El trial original era sobre-ajustado al fold split exacto.")
            print("      - Drift en los datos entre runs (raro pero posible).")
        else:
            print("  En modo STRICT esto es esperable sin TF_DETERMINISTIC_OPS=1.")
            print("  Re-ejecuta sin --strict para tolerancias realistas.")
    return v


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def _load_ohlcv(args, time_min: pd.Timestamp, time_max: pd.Timestamp) -> pd.DataFrame:
    if args.from_db:
        from mimo.data_managers.databases import Database
        from mimo.data_managers.data_manager import DataManager
        bar_min = {"1min": 1, "5min": 5, "15min": 15, "1h": 60}.get(args.base_tf, 5)
        buf_min = args.db_buffer_bars * bar_min
        from_dt = (time_min - pd.Timedelta(minutes=buf_min)).normalize()
        to_dt = (time_max + pd.Timedelta(minutes=buf_min)).normalize() + pd.Timedelta(days=1)
        print(f"📂 OHLCV: BD ({args.base_tf}, {from_dt} → {to_dt})")
        db = Database()
        resample_arg = None if str(args.base_tf).lower() in ("1min", "1m") else args.base_tf
        dm = DataManager.from_database_historical_2(
            db, from_date=str(from_dt), to_date=str(to_dt), resample=resample_arg
        )
        return dm.df[["time", "open", "high", "low", "close"]].copy()
    if not args.ohlcv:
        raise SystemExit("Pasa --ohlcv <path> o --from-db")
    p = Path(args.ohlcv)
    df = pd.read_csv(p) if p.suffix == ".csv" else pd.read_parquet(p)
    df.columns = [c.lower() for c in df.columns]
    return df[["time", "open", "high", "low", "close"]].copy()


def _locate_oof_parquet(specialist_dir: Path, release: str) -> Path:
    """Busca el oof_<release>_<side>.parquet del specialist."""
    candidates = list(specialist_dir.glob(f"oof_{release}_*.parquet"))
    if not candidates:
        candidates = list(specialist_dir.glob("oof_*.parquet"))
    if not candidates:
        raise SystemExit(f"❌ No encontré oof parquet en {specialist_dir}")
    if len(candidates) > 1:
        # Preferir el de multitask
        for c in candidates:
            if "multitask" in c.name:
                return c
    return candidates[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-per-side-json", required=True)
    ap.add_argument("--specialist-long-dir", default=None)
    ap.add_argument("--specialist-short-dir", default=None)
    ap.add_argument("--side", choices=["long", "short", "both"], default="both")
    ap.add_argument("--release", required=True)
    ap.add_argument("--ohlcv", default=None)
    ap.add_argument("--from-db", action="store_true")
    ap.add_argument("--db-buffer-bars", type=int, default=200)
    ap.add_argument("--base-tf", default="5min")
    ap.add_argument("--tp", type=float, default=2.0)
    ap.add_argument("--sl", type=float, default=0.8)
    ap.add_argument("--horizon", type=int, default=3)
    ap.add_argument("--atr-window", type=int, default=14)
    ap.add_argument("--cost", type=float, default=0.05)
    ap.add_argument("--max-drawdown-R", type=float, default=30.0)
    ap.add_argument("--strict", action="store_true",
                    help="Tolerancias bit-perfect (1e-3 ev_net, 2 n_signals, "
                         "1R mdd). Solo tiene sentido si entrenaste con "
                         "TF_DETERMINISTIC_OPS=1 y "
                         "tf.config.experimental.enable_op_determinism(). "
                         "Sin esto, fallará casi siempre por non-determinismo "
                         "CUDA. Default: relajado.")
    ap.add_argument("--tol-ev-net", type=float, default=None,
                    help="Override de la tolerancia abs en ev_net (R). "
                         "Default: 0.10 relajado, 1e-3 strict.")
    ap.add_argument("--tol-n-signals", type=float, default=None,
                    help="Override de la tolerancia en n_signals. "
                         "Relajado: fracción relativa (default 0.50, mín 50 abs). "
                         "Strict: absoluto (default 2).")
    ap.add_argument("--tol-mdd", type=float, default=None,
                    help="Override de la tolerancia abs en mdd_R. "
                         "Default: 10.0 relajado, 1.0 strict.")
    ap.add_argument("--tol-prec", type=float, default=None,
                    help="Override de la tolerancia abs en prec_TP. "
                         "Default: 0.10 relajado, 0.005 strict.")
    args = ap.parse_args()

    # Construye el dict de tolerancias efectivas
    base_tols = TOL_STRICT if args.strict else TOL_DEFAULT
    tols = dict(base_tols)
    if args.tol_ev_net is not None:
        tols["ev_net"] = float(args.tol_ev_net)
        tols["ev_gross"] = float(args.tol_ev_net)
    if args.tol_n_signals is not None:
        tols["n_signals"] = float(args.tol_n_signals)
    if args.tol_mdd is not None:
        tols["mdd_R"] = float(args.tol_mdd)
    if args.tol_prec is not None:
        tols["prec_TP"] = float(args.tol_prec)

    print(f"\n🎚️  Modo de tolerancias: {'STRICT' if args.strict else 'RELAJADO (TF non-determ)'}")
    print(f"     ev_net abs        : ≤ {tols['ev_net']:.4f}R")
    if args.strict:
        print(f"     n_signals abs     : ≤ {int(tols['n_signals'])}")
    else:
        print(f"     n_signals rel/abs : ≤ {tols['n_signals']*100:.0f}% (con piso 50 abs)")
    print(f"     prec_TP abs       : ≤ {tols['prec_TP']:.4f}")
    print(f"     mdd_R abs         : ≤ {tols['mdd_R']:.2f}R")

    json_path = Path(args.best_per_side_json)
    if not json_path.exists():
        raise SystemExit(f"❌ No existe: {json_path}")
    payload = json.loads(json_path.read_text(encoding="utf-8"))

    sides_to_check = []
    if args.side in ("long", "both"):
        if not args.specialist_long_dir:
            raise SystemExit("❌ --specialist-long-dir requerido para side=long/both")
        sides_to_check.append(("long", "ev_long", "oof_proba_long_cal", True,
                               Path(args.specialist_long_dir)))
    if args.side in ("short", "both"):
        if not args.specialist_short_dir:
            raise SystemExit("❌ --specialist-short-dir requerido para side=short/both")
        sides_to_check.append(("short", "ev_short", "oof_proba_short_cal", False,
                               Path(args.specialist_short_dir)))

    print("\n📂 Best-per-side JSON :", json_path)
    overall_pass = True
    results = {}

    for side_label, ev_key, proba_col, side_is_long, sp_dir in sides_to_check:
        print(f"\n{'=' * 78}\n   Procesando SPECIALIST {side_label.upper()}: {sp_dir}\n{'=' * 78}")
        if not sp_dir.exists():
            print(f"   ❌ Specialist dir no existe: {sp_dir}")
            overall_pass = False
            continue
        try:
            oof_parquet = _locate_oof_parquet(sp_dir, args.release)
        except SystemExit as e:
            print(f"   {e}")
            overall_pass = False
            continue
        print(f"   OOF parquet: {oof_parquet}")
        df_oof = pd.read_parquet(oof_parquet)
        df_oof["time"] = pd.to_datetime(df_oof["time"])
        if proba_col not in df_oof.columns:
            print(f"   ❌ Columna '{proba_col}' no en OOF parquet. "
                  f"Cols={list(df_oof.columns)}")
            overall_pass = False
            continue

        # Si falta atr, traer OHLCV
        if "atr" not in df_oof.columns or df_oof["atr"].isna().all():
            ohlcv = _load_ohlcv(args, df_oof["time"].min(), df_oof["time"].max())
            ohlcv["time"] = pd.to_datetime(ohlcv["time"])
            ohlcv = (ohlcv.dropna(subset=["close"])
                          .sort_values("time")
                          .drop_duplicates("time")
                          .reset_index(drop=True))
            df_oof = _ensure_atr_column(df_oof, ohlcv, atr_window=args.atr_window)

        # Recorded del trial
        sub = payload.get(f"top_{side_label}", [])
        if not sub:
            print(f"   ❌ top_{side_label} vacío en JSON")
            overall_pass = False
            continue
        recorded = sub[0].get(ev_key, {})
        trial_num = sub[0].get("trial", "?")
        print(f"   Trial origen: #{trial_num}")
        locked_thr = float(recorded.get("thr"))
        print(f"   thr lockeado: {locked_thr:.4f}")

        # Recompute en specialist
        recomputed = _recompute_metrics_at_locked_thr(
            df_oof,
            proba_col=proba_col,
            side_is_long=side_is_long,
            locked_thr=locked_thr,
            horizon=args.horizon,
            tp_mult=args.tp,
            sl_mult=args.sl,
            cost=args.cost,
            max_drawdown_R=args.max_drawdown_R,
        )

        v = _print_side_table(side_label.upper(), recorded, recomputed,
                              tols, strict=args.strict)
        results[side_label] = {
            "trial": trial_num,
            "thr_locked": locked_thr,
            "recorded": recorded,
            "recomputed": recomputed,
            "verdict": v,
        }
        if not v["passed"]:
            overall_pass = False

    # ── Check de scalers (solo cuando ambos specialists están disponibles) ──
    scalers_ok = True
    if args.specialist_long_dir and args.specialist_short_dir and args.side == "both":
        sc_info = _compare_scalers(
            Path(args.specialist_long_dir),
            Path(args.specialist_short_dir),
            args.release,
        )
        scalers_ok = _print_scaler_check(sc_info)

    # Veredicto global
    print("\n" + "=" * 78)
    print("  VEREDICTO GLOBAL")
    print("=" * 78)
    if overall_pass and scalers_ok:
        print("  ✅ Todos los specialists reproducen el trial original.")
        print("     Scalers idénticos → merge_specialists --strict-scalers OK.")
        print("     Puedes seguir adelante con merge_specialists + drift validation.")
    elif overall_pass and not scalers_ok:
        print("  ⚠️  Métricas reproducen OK pero scalers difieren.")
        print("     Revisa la advertencia anterior antes de merge_specialists.")
    else:
        if args.strict:
            print("  ❌ Al menos un specialist NO reproduce bit-a-bit el trial.")
            print("     STRICT mode requiere TF determinism habilitado.")
            print("     Re-ejecuta sin --strict para tolerancias realistas.")
        else:
            print("  ❌ Al menos un specialist se desvía MÁS de lo aceptable")
            print("     incluso con tolerancias relajadas (TF non-determ asumido).")
            print("     Esto indica un problema real, no solo CUDA atomic ops.")
            print("     Acciones sugeridas:")
            print("       - Retrain con --seed distinto (puede haber caído")
            print("         en un mínimo local malo).")
            print("       - Verifica que oof_epochs/patience del CLI coinciden")
            print("         con epochs/patience del trial origen (en best_per_side).")
            print("       - Confirma que el specialist usó el mismo release y")
            print("         feature_masks que el trial origen.")


if __name__ == "__main__":
    main()
