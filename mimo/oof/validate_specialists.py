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
    """Validación direccional: solo falla si el specialist es PEOR que el
    recorded por más de la tolerancia. Si el specialist mejora (más EV,
    más prec, menos MDD), eso NO es un fallo aunque el delta sea grande.

    En strict mode usamos delta absoluto (cualquier desvío bit-perfect es
    sospechoso, sea hacia arriba o hacia abajo).

    Métricas y dirección:
      ev_net, ev_gross, prec_TP    → higher is better, fail si baja > tol
      mdd_R, frac_SL               → lower is better,  fail si sube > tol
      n_signals                    → fail solo si explota (>2x recorded) o
                                     colapsa (<10% recorded). Ser más
                                     selectivo o levemente más prolífico
                                     no es problema.
    """
    rec_ev_net = float(recorded.get("ev_net", float("nan")))
    rec_ev_gross = float(recorded.get("ev_gross", float("nan")))
    rec_n_sig = float(recorded.get("n_signals", 0))
    rec_prec = float(recorded.get("prec_TP", float("nan")))
    rec_mdd = float(recorded.get("mdd_R", float("nan")))

    rcm_ev_net = float(recomputed.get("ev_net", float("nan")))
    rcm_ev_gross = float(recomputed.get("ev_gross", float("nan")))
    rcm_n_sig = float(recomputed.get("n_signals", 0))
    rcm_prec = float(recomputed.get("prec_TP", float("nan")))
    rcm_mdd = float(recomputed.get("mdd_R", float("nan")))

    # Deltas firmados (recomputed - recorded). Positivo = recomputed más alto.
    s_ev_net = rcm_ev_net - rec_ev_net
    s_ev_gross = rcm_ev_gross - rec_ev_gross
    s_n_sig = rcm_n_sig - rec_n_sig
    s_prec = rcm_prec - rec_prec
    s_mdd = rcm_mdd - rec_mdd

    if strict:
        # Strict: cualquier desvío > tol es fallo (sea hacia arriba o abajo)
        nsig_tol_abs = float(tols["n_signals"])
        checks = {
            "ev_net":    (s_ev_net,   tols["ev_net"],   abs(s_ev_net)   <= tols["ev_net"],   "abs"),
            "ev_gross":  (s_ev_gross, tols["ev_gross"], abs(s_ev_gross) <= tols["ev_gross"], "abs"),
            "n_signals": (s_n_sig,    nsig_tol_abs,     abs(s_n_sig)    <= nsig_tol_abs,     "abs"),
            "prec_TP":   (s_prec,     tols["prec_TP"],  abs(s_prec)     <= tols["prec_TP"],  "abs"),
            "mdd_R":     (s_mdd,      tols["mdd_R"],    abs(s_mdd)      <= tols["mdd_R"],    "abs"),
        }
        nsig_tol_str = f"|Δ|≤{int(nsig_tol_abs)}"
    else:
        # Relajado direccional:
        #   - ev_net/ev_gross/prec_TP: fail si BAJA más de tol
        #   - mdd_R: fail si SUBE más de tol
        #   - n_signals: fail solo si explota o colapsa
        nsig_explosion = rec_n_sig * 2.0   # más de 2x recorded → mod inestable
        nsig_collapse = max(10.0, rec_n_sig * 0.1)  # < 10% → modelo casi sin señal

        checks = {
            "ev_net":    (s_ev_net,   tols["ev_net"],   s_ev_net   >= -tols["ev_net"],   "down"),
            "ev_gross":  (s_ev_gross, tols["ev_gross"], s_ev_gross >= -tols["ev_gross"], "down"),
            "prec_TP":   (s_prec,     tols["prec_TP"],  s_prec     >= -tols["prec_TP"],  "down"),
            "mdd_R":     (s_mdd,      tols["mdd_R"],    s_mdd      <=  tols["mdd_R"],    "up"),
            "n_signals": (
                s_n_sig,
                (nsig_collapse, nsig_explosion),
                (rcm_n_sig <= nsig_explosion) and (rcm_n_sig >= nsig_collapse),
                "range",
            ),
        }
        nsig_tol_str = (
            f"{int(nsig_collapse)} ≤ rcm ≤ {int(nsig_explosion)} "
            f"(0.1×–2.0× recorded)"
        )

    fails = {}
    deltas = {}
    tols_eff = {}
    for k, (signed_delta, tol, ok, mode) in checks.items():
        deltas[k] = signed_delta
        tols_eff[k] = tol
        if not ok:
            fails[k] = (signed_delta, tol, mode)
    tols_eff["n_signals_str"] = nsig_tol_str
    return {
        "deltas": deltas,                   # firmados (positivo = recomputed mayor)
        "tolerances": tols_eff,
        "fails": fails,
        "passed": len(fails) == 0,
        "strict": strict,
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
    mode = "STRICT" if strict else "RELAJADO"
    if v["passed"]:
        print(f"  ✅ PASS ({mode}) — specialist no es peor que el trial dentro de tolerancia.")
        # Resaltar mejoras notables (signed delta a favor del specialist)
        improvements = []
        if v["deltas"]["ev_net"] > tols["ev_net"]:
            improvements.append(f"ev_net +{v['deltas']['ev_net']:.4f}R")
        if v["deltas"]["prec_TP"] > tols["prec_TP"]:
            improvements.append(f"prec_TP +{v['deltas']['prec_TP']:.4f}")
        if v["deltas"]["mdd_R"] < -tols["mdd_R"]:
            improvements.append(f"MDD {v['deltas']['mdd_R']:+.2f}R")
        if improvements and not strict:
            print(f"     🎯 Specialist MEJORA al trial en: {', '.join(improvements)}")
    else:
        print(f"  ❌ FAIL ({mode}) — specialist se desvía a peor:")
        for k, (delta, tol, m) in v["fails"].items():
            if m == "down":
                # ev_net/prec/etc. baja
                print(f"      {k:<10}  recomputed {delta:+.4f} respecto a recorded "
                      f"(baja > tol {tol:.4f}) ❌")
            elif m == "up":
                # mdd_R sube
                print(f"      {k:<10}  recomputed {delta:+.4f} respecto a recorded "
                      f"(sube > tol {tol:.4f}) ❌")
            elif m == "range":
                # n_signals fuera de [collapse, explosion]
                lo, hi = tol
                print(f"      {k:<10}  recomputed={recomputed.get('n_signals')} "
                      f"fuera de [{int(lo)}, {int(hi)}] ❌")
            else:
                # strict abs
                print(f"      {k:<10}  |Δ|={abs(delta):.4f} > tol {tol:.4f}")
        if not strict:
            print("  El specialist es PEOR que el trial original en métricas críticas.")
            print("  Acciones sugeridas:")
            print("      - Retrain con --seed distinto.")
            print("      - Revisa que el trial origen no estuviera sobre-ajustado")
            print("        al fold split (mira ev_net en holdout, no solo OOF).")
        else:
            print("  En modo STRICT cualquier desvío bit-no-perfecto falla.")
            print("  Re-ejecuta sin --strict para chequeo direccional realista.")
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

    print(f"\n🎚️  Modo de tolerancias: {'STRICT' if args.strict else 'RELAJADO (TF non-determ, direccional)'}")
    if args.strict:
        print(f"     |Δ ev_net|       ≤ {tols['ev_net']:.4f}R")
        print(f"     |Δ n_signals|    ≤ {int(tols['n_signals'])}")
        print(f"     |Δ prec_TP|      ≤ {tols['prec_TP']:.4f}")
        print(f"     |Δ mdd_R|        ≤ {tols['mdd_R']:.2f}R")
        print(f"     Cualquier desvío bit-no-perfecto será fallo.")
    else:
        print(f"     ev_net   no debe BAJAR más de  {tols['ev_net']:.4f}R")
        print(f"     prec_TP  no debe BAJAR más de  {tols['prec_TP']:.4f}")
        print(f"     mdd_R    no debe SUBIR más de  {tols['mdd_R']:.2f}R")
        print(f"     n_signals debe estar en  [10% × recorded, 2× recorded]")
        print(f"     (mejorar respecto al trial NO es fallo)")

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
