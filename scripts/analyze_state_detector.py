#!/usr/bin/env python3
"""
analyze_state_detector.py
=========================

Auditoría empírica del StateDetector sobre un periodo histórico.

Diagnostica:
  1. Distribución de estados (% del tiempo en cada uno)
  2. Persistencia (longitud típica de cada bloque de estado consecutivo)
  3. Flapping (cuántas transiciones rápidas hay)
  4. Hit rate por estado (¿el precio se mueve en la dirección esperada en N ticks?)
  5. Umbrales — ¿están bien calibrados?
     · ADX hardcoded (25, 18) vs distribución empírica
     · Percentiles de ATR_norm, BB_width

Esto NO modifica nada — solo reporta para decidir qué mejoras priorizar.

Uso
---
  # Sobre marzo 2026 (OOS)
  python scripts/analyze_state_detector.py \\
    --release 202500 \\
    --deploy-subdir deploy_PROD_combined_seed47_20260517 \\
    --policy-config decision_policies_config_202500_PROD_20260517 \\
    --from 2026-03-01 --to 2026-03-31 \\
    --warmup-from 2026-02-10

  # Comparativa rápida sobre múltiples periodos
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

sys.path.insert(0, str(_PROJECT_ROOT / "scripts"))
from replay_s2_202500 import build_simulator, load_ohlcv


def predict_states(release: str, deploy_subdir: str, policy_config: str,
                   from_date: str, to_date: str, warmup_from: str) -> pd.DataFrame:
    """Ejecuta simulator.predict() y devuelve df enriquecido con state."""
    project_root = Path(__file__).resolve().parent.parent
    sim = build_simulator(
        release=release, deploy_subdir=deploy_subdir,
        policy_module=f"config.{policy_config}",
        base_dir=project_root / "main",
        artifacts_root=project_root / "artifacts",
        score_low_quantile=80,
    )
    df = load_ohlcv(warmup_from, to_date, "5min")
    df_pred = sim.predict(df, simulation=True)
    # Filtrar al rango de evaluación (sin warmup)
    df_pred = df_pred.reset_index(drop=False)
    if "time" not in df_pred.columns:
        df_pred["time"] = df_pred.index
    df_pred["time"] = pd.to_datetime(df_pred["time"])
    mask = (df_pred["time"] >= pd.Timestamp(from_date)) & (df_pred["time"] <= pd.Timestamp(to_date) + pd.Timedelta(days=1))
    return df_pred[mask].reset_index(drop=True)


def analyze_distribution(df: pd.DataFrame) -> None:
    print(f"\n{'='*70}")
    print(f"  1. DISTRIBUCIÓN DE ESTADOS")
    print(f"{'='*70}")
    if "state" not in df.columns:
        print("  ❌ No hay columna 'state'")
        return
    n = len(df)
    print(f"  Total ticks: {n}")
    counts = df["state"].value_counts()
    print(f"\n  {'Estado':<22} {'n':>6} {'%':>7}")
    print("  " + "-" * 40)
    for st, c in counts.items():
        pct = c / n * 100
        bar = "█" * int(pct / 2)
        print(f"  {st:<22} {c:>6} {pct:>6.1f}%  {bar}")


def analyze_persistence(df: pd.DataFrame) -> None:
    print(f"\n{'='*70}")
    print(f"  2. PERSISTENCIA — longitud de bloques consecutivos")
    print(f"{'='*70}")
    states = df["state"].values
    if len(states) == 0:
        return
    # Run-length encoding
    runs = []
    current_state = states[0]
    current_len = 1
    for s in states[1:]:
        if s == current_state:
            current_len += 1
        else:
            runs.append((current_state, current_len))
            current_state = s
            current_len = 1
    runs.append((current_state, current_len))

    # Stats por estado
    by_state = {}
    for st, ln in runs:
        by_state.setdefault(st, []).append(ln)

    print(f"\n  {'Estado':<22} {'n_runs':>8} {'mean':>8} {'median':>8} {'p90':>8} {'max':>6}")
    print("  " + "-" * 65)
    for st in sorted(by_state, key=lambda s: -np.mean(by_state[s])):
        lens = by_state[st]
        m = np.mean(lens)
        med = np.median(lens)
        p90 = np.percentile(lens, 90)
        mx = max(lens)
        print(f"  {st:<22} {len(lens):>8} {m:>7.1f}  {med:>7.1f}  {p90:>7.1f}  {mx:>6}")

    # Total transiciones
    n_transitions = len(runs) - 1
    rate = n_transitions / len(states) * 100
    print(f"\n  Total transiciones: {n_transitions} ({rate:.2f}% de los ticks)")
    print(f"  Run length medio global: {np.mean([ln for _,ln in runs]):.1f} barras")

    # Flapping: transiciones tipo A → B → A
    flaps = 0
    for i in range(len(runs) - 2):
        if runs[i][0] == runs[i + 2][0] and runs[i + 1][1] <= 2:
            flaps += 1
    print(f"  Flapping detectado (A→B→A con B≤2 barras): {flaps}")


def analyze_hit_rate(df: pd.DataFrame, k_forward: int = 5) -> None:
    print(f"\n{'='*70}")
    print(f"  3. HIT RATE por estado (próximas {k_forward} barras)")
    print(f"{'='*70}")
    if "close" not in df.columns or "state" not in df.columns:
        return
    closes = df["close"].values
    states = df["state"].values
    if len(closes) < k_forward + 1:
        return

    drifts = closes[k_forward:] - closes[:-k_forward]  # signed
    states_aligned = states[:-k_forward]

    by_state = {}
    for st, d in zip(states_aligned, drifts):
        by_state.setdefault(st, []).append(d)

    print(f"\n  {'Estado':<22} {'n':>6} {'drift_median':>14} {'|drift|_med':>14} {'hit_dir':>9}")
    print("  " + "-" * 75)
    for st, drifts_list in sorted(by_state.items(), key=lambda x: -len(x[1])):
        arr = np.array(drifts_list)
        med = np.median(arr)
        abs_med = np.median(np.abs(arr))
        # Hit_dir según estado
        hit = None
        st_lower = st.lower()
        if "trend_up" in st_lower or "transition_up" in st_lower or "breakout_wait_up" in st_lower:
            hit = (arr > 0).mean() * 100
        elif "trend_down" in st_lower or "transition_down" in st_lower or "breakout_wait_down" in st_lower:
            hit = (arr < 0).mean() * 100
        hit_str = f"{hit:.0f}%" if hit is not None else "n/a"
        print(f"  {st:<22} {len(arr):>6} {med:>+14.3f} {abs_med:>14.3f} {hit_str:>9}")

    print(f"\n  Interpretación:")
    print(f"    · hit_dir > 60% = state predice dirección correctamente")
    print(f"    · hit_dir ≈ 50% = random (state no es predictor)")
    print(f"    · hit_dir < 40% = state predice MAL la dirección")


def analyze_thresholds(df: pd.DataFrame) -> None:
    print(f"\n{'='*70}")
    print(f"  4. UMBRALES — ¿están bien calibrados?")
    print(f"{'='*70}")

    print(f"\n  --- ADX (umbrales hardcoded 25 trend / 18 range) ---")
    if "adx" in df.columns:
        adx = df["adx"].dropna()
        if len(adx) > 0:
            print(f"    n={len(adx)}, mean={adx.mean():.1f}, std={adx.std():.1f}")
            print(f"    Percentiles ADX:")
            for p in [10, 25, 35, 50, 65, 75, 85, 90]:
                v = np.percentile(adx, p)
                marker = ""
                if abs(v - 18) < 1:
                    marker = "  ← cerca de adx_range=18"
                if abs(v - 25) < 1:
                    marker = "  ← cerca de adx_trend=25"
                print(f"      p{p:>2}: {v:>6.1f}{marker}")
            n_below_18 = (adx < 18).sum()
            n_btw = ((adx >= 18) & (adx < 25)).sum()
            n_above_25 = (adx >= 25).sum()
            tot = len(adx)
            print(f"    Distribución:")
            print(f"      ADX < 18 (range):           {n_below_18} ({n_below_18/tot*100:.1f}%)")
            print(f"      ADX 18-25 (transition):     {n_btw} ({n_btw/tot*100:.1f}%)")
            print(f"      ADX >= 25 (trend):          {n_above_25} ({n_above_25/tot*100:.1f}%)")

    print(f"\n  --- ATR_norm ---")
    if "atr_norm" in df.columns:
        atr = df["atr_norm"].dropna()
        if len(atr) > 0:
            print(f"    n={len(atr)}, mean={atr.mean():.6f}")
            print(f"    Percentiles:")
            for p in [10, 30, 50, 70, 80, 90, 95]:
                print(f"      p{p:>2}: {np.percentile(atr, p):.6f}")

    print(f"\n  --- BB_width ---")
    if "bb_width" in df.columns:
        bbw = df["bb_width"].dropna()
        if len(bbw) > 0:
            print(f"    n={len(bbw)}, mean={bbw.mean():.6f}")
            print(f"    Percentiles:")
            for p in [10, 20, 35, 50, 70, 80, 90]:
                print(f"      p{p:>2}: {np.percentile(bbw, p):.6f}")


def analyze_transition_matrix(df: pd.DataFrame) -> None:
    """Matriz de transiciones state(t) → state(t+1)."""
    print(f"\n{'='*70}")
    print(f"  5. MATRIZ DE TRANSICIONES (% de cada A→B sobre el total de A)")
    print(f"{'='*70}")
    if "state" not in df.columns or len(df) < 2:
        return
    states = df["state"].values
    transitions = Counter()
    state_counts = Counter()
    for i in range(len(states) - 1):
        s_from = states[i]
        s_to = states[i + 1]
        state_counts[s_from] += 1
        transitions[(s_from, s_to)] += 1

    all_states = sorted(state_counts.keys())
    header_label = "from / to"
    print(f"\n  {header_label:<22}", end="")
    for s in all_states:
        print(f"{s[:8]:>10s}", end="")
    print()
    for s_from in all_states:
        total = state_counts[s_from]
        print(f"  {s_from:<22}", end="")
        for s_to in all_states:
            pct = transitions[(s_from, s_to)] / total * 100 if total else 0
            if s_from == s_to:
                print(f"{pct:>9.1f}%", end="")  # persistencia (diagonal)
            else:
                print(f"{pct:>9.1f}%", end="")
        print()
    print()
    print(f"  Lectura: diagonal alta = estados estables. Off-diagonal alta = inestable.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--release", default="202500")
    parser.add_argument("--deploy-subdir", required=True)
    parser.add_argument("--policy-config", required=True)
    parser.add_argument("--from", dest="from_date", required=True)
    parser.add_argument("--to", dest="to_date", required=True)
    parser.add_argument("--warmup-from", required=True)
    parser.add_argument("--k-forward", type=int, default=5,
                        help="N barras para hit_rate forward (default 5)")
    args = parser.parse_args()

    print(f"\n📂 Ejecutando StateDetector sobre {args.from_date} → {args.to_date}")
    df = predict_states(
        release=args.release,
        deploy_subdir=args.deploy_subdir,
        policy_config=args.policy_config,
        from_date=args.from_date,
        to_date=args.to_date,
        warmup_from=args.warmup_from,
    )
    print(f"   {len(df)} ticks analizables (post warmup)")

    analyze_distribution(df)
    analyze_persistence(df)
    analyze_hit_rate(df, k_forward=args.k_forward)
    analyze_thresholds(df)
    analyze_transition_matrix(df)

    return 0


if __name__ == "__main__":
    sys.exit(main())
