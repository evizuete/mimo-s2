#!/usr/bin/env python3
"""
barrier_sweep.py

Barrido empirico de configuraciones (horizon, tp, sl) sobre los datos historicos
para identificar barriers triple-barrier economicamente viables, sin reentrenar
ningun modelo. Ataca el problema "que barriers tienen sentido en h=5 (5 minutos
en velas de 1m)" con datos en lugar de intuicion.

Que hace:
  1. Carga rates historicos (DataManager o parquet).
  2. Computa ATR si no esta presente.
  3. Para cada combinacion (side, h, tp, sl) en una grilla:
       - Simula triple barrier sobre el dataset entero.
       - Cuenta resultados: TP first, SL first, TIMEOUT.
       - Calcula pos_rate, expectancy real (con timeout=0), break-even.
       - Calcula "lift required" = BE / pos_rate (lift que necesita el modelo
         para que esa config sea rentable).
  4. Filtra por restricciones (sl >= 1.0 ATR para no caer en ruido).
  5. Imprime top N por menor lift_to_breakeven (= mas alcanzable).
  6. Opcional: desglose por estado para los top configs.

Uso:
    python -m mimo.diagnosis.barrier_sweep
    python -m mimo.diagnosis.barrier_sweep --from-date 2025-06-01 --to-date 2026-04-01
    python -m mimo.diagnosis.barrier_sweep --per-state
    python -m mimo.diagnosis.barrier_sweep --rates-parquet rates.parquet
    python -m mimo.diagnosis.barrier_sweep --horizons 3,5,8,10 --assumed-lift 1.75
    python -m mimo.diagnosis.barrier_sweep --tie-policy sl_first   # desempate conservador
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
from numba import jit


HORIZONS_DEFAULT = [3, 5, 8, 10, 12, 15]
TP_MULTS_DEFAULT = [1.0, 1.25, 1.5, 1.75, 2.0, 2.25, 2.5, 3.0, 3.5]
SL_MULTS_DEFAULT = [1.0, 1.1, 1.25, 1.5, 1.75, 2.0]
SIDES = ["long", "short"]


@jit(nopython=True, cache=True)
def triple_barrier_outcomes(close, high, low, atr, horizon, tp_mult, sl_mult, side_is_long, tie_policy):
    """
    Como triple_barrier_fixed_numba pero distingue TIMEOUT de SL.
    Devuelve outcome[i] in {1: TP first, -1: SL first, 0: timeout, 99: skip (atr inv)}.

    tie_policy:
      0 = tp_first: si TP y SL se tocan en la misma vela/minuto, gana TP.
      1 = sl_first: si TP y SL se tocan en la misma vela/minuto, gana SL.
          Esta es la opcion conservadora para OHLC de 1m, porque no conocemos
          el orden intrabar real.
    """
    n = len(close)
    outcome = np.full(n, 99, dtype=np.int8)

    for i in range(n - horizon):
        entry = close[i]
        atr_val = atr[i]
        if np.isnan(atr_val) or atr_val <= 0:
            continue

        if side_is_long:
            tp_level = entry + tp_mult * atr_val
            sl_level = entry - sl_mult * atr_val
        else:
            tp_level = entry - tp_mult * atr_val
            sl_level = entry + sl_mult * atr_val

        hit_tp = -1
        hit_sl = -1
        for k in range(1, horizon + 1):
            j = i + k
            if j >= n:
                break
            if side_is_long:
                if hit_tp == -1 and high[j] >= tp_level:
                    hit_tp = k
                if hit_sl == -1 and low[j] <= sl_level:
                    hit_sl = k
            else:
                if hit_tp == -1 and low[j] <= tp_level:
                    hit_tp = k
                if hit_sl == -1 and high[j] >= sl_level:
                    hit_sl = k
            if hit_tp != -1 and hit_sl != -1:
                break

        if hit_tp != -1 and hit_sl != -1:
            if hit_tp < hit_sl:
                outcome[i] = 1
            elif hit_sl < hit_tp:
                outcome[i] = -1
            else:
                # Empate dentro de la misma vela/minuto.
                # OHLC no permite saber que barrera se toco primero.
                if tie_policy == 0:
                    outcome[i] = 1
                else:
                    outcome[i] = -1
        elif hit_tp != -1:
            outcome[i] = 1
        elif hit_sl != -1:
            outcome[i] = -1
        else:
            outcome[i] = 0
    return outcome


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from-date", default="2025-01-01")
    ap.add_argument("--to-date", default="2026-04-26")
    ap.add_argument("--rates-parquet", default=None,
                    help="Si se pasa, lee rates desde este parquet en lugar de la DB.")
    ap.add_argument("--horizons", default=None, help="Lista CSV, e.g. '3,5,8'.")
    ap.add_argument("--tp-mults", default=None, help="Lista CSV.")
    ap.add_argument("--sl-mults", default=None, help="Lista CSV.")
    ap.add_argument("--min-sl", type=float, default=1.0,
                    help="SL minimo en ATR (default 1.0 — evita ruido sub-ATR).")
    ap.add_argument("--max-tp-sl-ratio", type=float, default=5.0)
    ap.add_argument("--min-tp-sl-ratio", type=float, default=0.8)
    ap.add_argument("--assumed-lift", type=float, default=1.75,
                    help="Lift asumido del modelo en operating region (default 1.75 ≈ LONG h=5).")
    ap.add_argument("--tie-policy", choices=["tp_first", "sl_first"], default="sl_first",
                    help=("Como resolver una vela donde TP y SL se tocan en el mismo minuto. "
                          "tp_first reproduce el criterio optimista anterior; "
                          "sl_first es conservador y recomendado para validacion."))
    ap.add_argument("--top-n", type=int, default=15)
    ap.add_argument("--per-state", action="store_true")
    ap.add_argument("--out", default="barrier_sweep_results.csv")
    return ap.parse_args()


def parse_csv_list(s: Optional[str], default: List) -> List[float]:
    if s is None:
        return default
    return [float(x) for x in s.split(",")]


def add_state_to_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la columna 'state' usando FeatureEngineer + add_mimo_state,
    reproduciendo el mismo régimen que ve el pipeline de producción.
    Caro (recomputa indicadores sobre todo el df), por eso solo se llama
    cuando --per-state está activo.
    """
    from mimo.features.feature_builder import FeatureEngineer, FeatureConfig
    from mimo.states_manager.state_detector import add_mimo_state

    print("[state] computando features e indicadores para detector de régimen...")
    fe = FeatureEngineer(FeatureConfig())
    df_feat = fe.generate_all_features(df)
    print(f"[state] features OK ({len(df_feat):,} filas tras dropna)")

    df_state = add_mimo_state(df_feat, set_market_condition=False)
    print(f"[state] estados detectados: "
          f"{df_state['state'].value_counts().to_dict()}")
    return df_state


def load_rates(args: argparse.Namespace) -> pd.DataFrame:
    if args.rates_parquet:
        print(f"[load] reading parquet {args.rates_parquet}")
        df = pd.read_parquet(args.rates_parquet)
    else:
        from mimo.data_managers.data_manager import DataManager
        from mimo.data_managers.databases import Database
        print(f"[load] reading DB from {args.from_date} to {args.to_date}")
        db = Database()
        from_dt = datetime.fromisoformat(args.from_date)
        to_dt = datetime.fromisoformat(args.to_date)
        dm = DataManager.from_database_historical_2(db, from_date=from_dt, to_date=to_dt)
        df = dm.df

    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.sort_values("time").reset_index(drop=True)

    required = {"close", "high", "low"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"faltan columnas requeridas: {missing}")

    if "atr" not in df.columns:
        print("[load] atr no encontrada — computando ATR(14)")
        df["atr"] = compute_atr(df, period=14)

    return df


def compute_atr(df: pd.DataFrame, period: int = 14) -> np.ndarray:
    high = df["high"].to_numpy(np.float64)
    low = df["low"].to_numpy(np.float64)
    close = df["close"].to_numpy(np.float64)
    prev_close = np.roll(close, 1)
    prev_close[0] = close[0]
    tr = np.maximum.reduce([
        high - low,
        np.abs(high - prev_close),
        np.abs(low - prev_close),
    ])
    return pd.Series(tr).rolling(period).mean().to_numpy()


def aggregate_outcome(outcome: np.ndarray) -> dict:
    valid = outcome != 99
    n = int(valid.sum())
    if n == 0:
        return {"n": 0, "n_tp": 0, "n_sl": 0, "n_timeout": 0,
                "pos_rate": float("nan"), "sl_rate": float("nan"),
                "timeout_rate": float("nan")}
    n_tp = int((outcome == 1).sum())
    n_sl = int((outcome == -1).sum())
    n_to = int((outcome == 0).sum())
    return {
        "n": n,
        "n_tp": n_tp,
        "n_sl": n_sl,
        "n_timeout": n_to,
        "pos_rate": n_tp / n,
        "sl_rate": n_sl / n,
        "timeout_rate": n_to / n,
    }


def sweep(df: pd.DataFrame, horizons: List[int], tp_mults: List[float],
          sl_mults: List[float], min_sl: float,
          min_ratio: float, max_ratio: float, tie_policy_name: str) -> pd.DataFrame:
    close = df["close"].to_numpy(np.float64)
    high = df["high"].to_numpy(np.float64)
    low = df["low"].to_numpy(np.float64)
    atr = df["atr"].to_numpy(np.float64)

    tie_policy = 0 if tie_policy_name == "tp_first" else 1

    rows = []
    configs = []
    for side in SIDES:
        for h in horizons:
            for tp in tp_mults:
                for sl in sl_mults:
                    if sl < min_sl:
                        continue
                    ratio = tp / sl
                    if ratio < min_ratio or ratio > max_ratio:
                        continue
                    configs.append((side, int(h), float(tp), float(sl)))

    print(f"[sweep] {len(configs)} configs en {len(close):,} filas")
    for idx, (side, h, tp, sl) in enumerate(configs, 1):
        is_long = side == "long"
        outcome = triple_barrier_outcomes(close, high, low, atr, h, tp, sl, is_long, tie_policy)
        agg = aggregate_outcome(outcome)
        rows.append({
            "side": side, "h": h, "tp": tp, "sl": sl,
            "tie_policy": tie_policy_name,
            "tp_sl_ratio": tp / sl,
            "BE": sl / (tp + sl),
            **agg,
        })
        if idx % 50 == 0:
            print(f"  [{idx}/{len(configs)}]")
    return pd.DataFrame(rows)


def add_economic_metrics(df: pd.DataFrame, assumed_lift: float) -> pd.DataFrame:
    out = df.copy()

    out["lift_to_breakeven"] = out["BE"] / out["pos_rate"].clip(lower=1e-6)

    out["assumed_precision"] = (out["pos_rate"] * assumed_lift).clip(upper=1.0)

    out["E_R_unconditional"] = (
        out["pos_rate"] * out["tp"] - out["sl_rate"] * out["sl"]
    )

    p = out["assumed_precision"]
    sl_rate_assumed = (1.0 - p) * (out["sl_rate"] / (out["sl_rate"] + out["timeout_rate"] + 1e-9))
    to_rate_assumed = (1.0 - p) * (out["timeout_rate"] / (out["sl_rate"] + out["timeout_rate"] + 1e-9))
    out["E_R_assumed_lift"] = (
        p * out["tp"] - sl_rate_assumed * out["sl"]
    )
    out["margin_to_BE"] = out["assumed_precision"] - out["BE"]

    return out


def per_state_for_top(df: pd.DataFrame, top_configs: pd.DataFrame,
                      min_n: int = 200, tie_policy_name: str = "sl_first") -> pd.DataFrame:
    if "state" not in df.columns:
        print("[per-state] columna 'state' no existe en rates — skip")
        return pd.DataFrame()

    close = df["close"].to_numpy(np.float64)
    high = df["high"].to_numpy(np.float64)
    low = df["low"].to_numpy(np.float64)
    atr = df["atr"].to_numpy(np.float64)
    states = df["state"].astype(str).to_numpy()

    tie_policy = 0 if tie_policy_name == "tp_first" else 1

    rows = []
    for _, c in top_configs.iterrows():
        is_long = c["side"] == "long"
        h = int(c["h"])
        outcome = triple_barrier_outcomes(close, high, low, atr, h, float(c["tp"]),
                                          float(c["sl"]), is_long, tie_policy)
        valid = outcome != 99
        for s in np.unique(states):
            mask = valid & (states == s)
            n = int(mask.sum())
            if n < min_n:
                continue
            sub = outcome[mask]
            n_tp = int((sub == 1).sum())
            n_sl = int((sub == -1).sum())
            rows.append({
                "side": c["side"], "h": h, "tp": float(c["tp"]), "sl": float(c["sl"]),
                "tie_policy": tie_policy_name,
                "BE": float(c["BE"]),
                "state": s, "n": n,
                "pos_rate": n_tp / n,
                "sl_rate": n_sl / n,
                "lift_to_breakeven": float(c["BE"]) / max(n_tp / n, 1e-6),
            })
    return pd.DataFrame(rows)


def print_top(out: pd.DataFrame, side: str, top_n: int, sort_by: str = "lift_to_breakeven") -> pd.DataFrame:
    sub = out[out["side"] == side].copy()
    if sub.empty:
        return sub
    if sort_by == "lift_to_breakeven":
        sub = sub.nsmallest(top_n, "lift_to_breakeven")
    else:
        sub = sub.nlargest(top_n, sort_by)
    cols = ["h", "tp", "sl", "tp_sl_ratio", "BE", "pos_rate",
            "sl_rate", "timeout_rate", "lift_to_breakeven",
            "assumed_precision", "margin_to_BE", "E_R_assumed_lift", "n"]
    print(sub[cols].round(4).to_string(index=False))
    return sub


def main() -> None:
    args = parse_args()

    horizons = [int(x) for x in parse_csv_list(args.horizons, HORIZONS_DEFAULT)]
    tp_mults = parse_csv_list(args.tp_mults, TP_MULTS_DEFAULT)
    sl_mults = parse_csv_list(args.sl_mults, SL_MULTS_DEFAULT)

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 50)
    pd.set_option("display.max_rows", 200)

    df = load_rates(args)
    nan_atr = float(df["atr"].isna().mean())
    print(f"[load] {len(df):,} filas | atr nan-rate: {nan_atr:.4f}")

    if args.per_state and "state" not in df.columns:
        df = add_state_to_df(df)

    print(f"[config] tie_policy={args.tie_policy}")
    raw = sweep(df, horizons, tp_mults, sl_mults, args.min_sl,
                args.min_tp_sl_ratio, args.max_tp_sl_ratio, args.tie_policy)
    out = add_economic_metrics(raw, args.assumed_lift)

    print("\n" + "=" * 90)
    print(f"TOP {args.top_n} LONG por menor lift_to_breakeven  (asumiendo lift={args.assumed_lift}x)")
    print("=" * 90)
    long_top = print_top(out, "long", args.top_n)

    print("\n" + "=" * 90)
    print(f"TOP {args.top_n} SHORT por menor lift_to_breakeven  (asumiendo lift={args.assumed_lift}x)")
    print("=" * 90)
    short_top = print_top(out, "short", args.top_n)

    if args.per_state:
        print("\n" + "=" * 90)
        print("PER-STATE BREAKDOWN (top 3 por (side, h) — cubre todos los horizontes)")
        print("=" * 90)
        # Tomar top-K por (side, h) en vez de top-K global, así obtenemos
        # representación de TODOS los horizontes (no solo el dominante h=60)
        # y se puede comparar el spread por estado entre horizontes cortos
        # y largos para localizar el "sweet spot" donde el estado aporta info.
        top_for_states = (
            out.sort_values("lift_to_breakeven")
               .groupby(["side", "h"], as_index=False, group_keys=False)
               .head(3)
        )
        ps = per_state_for_top(df, top_for_states, tie_policy_name=args.tie_policy)
        if not ps.empty:
            print(ps.round(4).to_string(index=False))
            ps_path = Path(args.out).with_name(Path(args.out).stem + "_per_state.csv")
            ps.to_csv(ps_path, index=False)
            print(f"[ok] per-state guardado: {ps_path}")

    out_path = Path(args.out)
    out.to_csv(out_path, index=False)
    print(f"\n[ok] sweep completo guardado: {out_path}")

    print("\n" + "=" * 90)
    print("LECTURA RAPIDA")
    print("=" * 90)
    if not long_top.empty:
        best_long = long_top.iloc[0]
        print(f"  Mejor LONG : h={int(best_long['h'])} tp={best_long['tp']} sl={best_long['sl']} tie={best_long['tie_policy']}")
        print(f"               BE={best_long['BE']:.3f}  pos_rate={best_long['pos_rate']:.3f}")
        print(f"               lift_to_breakeven={best_long['lift_to_breakeven']:.2f}x")
    if not short_top.empty:
        best_short = short_top.iloc[0]
        print(f"  Mejor SHORT: h={int(best_short['h'])} tp={best_short['tp']} sl={best_short['sl']} tie={best_short['tie_policy']}")
        print(f"               BE={best_short['BE']:.3f}  pos_rate={best_short['pos_rate']:.3f}")
        print(f"               lift_to_breakeven={best_short['lift_to_breakeven']:.2f}x")

    print(f"\n  Tu modelo actual da lift ~1.7x en LONG h=5 OOF.")
    print(f"  Configs con lift_to_breakeven < 1.7 → economicamente viables ya")
    print(f"  Configs con lift_to_breakeven en 1.7-2.5 → marginales (mejor modelo o per-state gate)")
    print(f"  Configs con lift_to_breakeven > 3 → no operables")


if __name__ == "__main__":
    main()
