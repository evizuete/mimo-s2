#!/usr/bin/env python3
"""
analyze_lockbox_sl_hypothesis.py
================================

Análisis empírico sobre los trades de un replay LOCKBOX (n>>100) para:

  1. Validar si SL/range_diario predice PnL  (hipótesis del usuario)
  2. Diagnosticar si el StateDetector clasifica correctamente cada régimen
     (cross-check: el state X debería tener win_rate consistente, no aleatorio)
  3. Si SL/range es el problema → sugerir una fórmula de SL adaptativo

Origen de datos:
  · trades.parquet del replay (con entry, sl, tp, exit, pnl, market_condition)
  · OHLCV 1min de la tabla `rates` para calcular range diario

Uso
---
  # Sobre el replay rolling actual (~445 trades)
  python scripts/analyze_lockbox_sl_hypothesis.py \\
    --replay-dir /tmp/replay_lockbox_rolling

  # Sobre cualquier otro replay
  python scripts/analyze_lockbox_sl_hypothesis.py \\
    --replay-dir /tmp/replay_lockbox_baseline_202500_20260520_091246
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Asegurar PROJECT_ROOT en sys.path para import de mimo
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def load_daily_ranges(dates: List[str]) -> pd.DataFrame:
    """Carga OHLCV 1m por día → df con (date, open, high, low, close, range, net, tr)."""
    from mimo.data_managers.databases import Database
    from sqlalchemy import text

    db = Database()
    db.connect()

    rows = []
    for d in dates:
        with db.engine.connect() as conn:
            q = text(
                "SELECT open, high, low, close FROM rates "
                "WHERE DATE(time) = :d ORDER BY time ASC"
            )
            r = conn.execute(q, {"d": d}).fetchall()
        if not r:
            continue
        opens = [float(x[0]) for x in r]
        highs = [float(x[1]) for x in r]
        lows = [float(x[2]) for x in r]
        closes = [float(x[3]) for x in r]
        rng = max(highs) - min(lows)
        net = closes[-1] - opens[0]
        tr = abs(net) / rng if rng > 0 else 0
        rows.append({
            "date": d, "open": opens[0], "close": closes[-1],
            "high": max(highs), "low": min(lows),
            "range": rng, "net_move": net, "trend_ratio": tr,
            "character": "RANGE" if tr < 0.4 else ("TEND_DEBIL" if tr < 0.7 else "TEND_CLARA"),
            "n_bars": len(r),
        })
    return pd.DataFrame(rows)


def enrich_trades(trades: pd.DataFrame, daily_ranges: pd.DataFrame) -> pd.DataFrame:
    """Añade a cada trade: day_range, sl_distance, sl_to_range_pct."""
    t = trades.copy()
    t["entry_time"] = pd.to_datetime(t["entry_time"])
    t["date"] = t["entry_time"].dt.strftime("%Y-%m-%d")
    t = t.merge(
        daily_ranges[["date", "range", "trend_ratio", "character", "net_move"]],
        on="date", how="left",
    )
    t["sl_distance"] = (t["sl"] - t["entry"]).abs()
    t["sl_to_range_pct"] = t["sl_distance"] / t["range"] * 100
    t["winner"] = t["pnl"] > 0
    return t


def report_state_classification(t: pd.DataFrame) -> None:
    """¿Está el StateDetector clasificando correctamente?

    Para cada market_condition, mostrar: n trades, win_rate, avg PnL.
    Si win_rate está cerca de 50% en un estado direccional, está mal.
    """
    print(f"\n{'='*78}")
    print(f"  ¿EL STATEDETECTOR CLASIFICA BIEN?")
    print(f"{'='*78}")
    print(f"\nWin rate y PnL por (regime × side) — n={len(t)}:")
    print(f"{'State':<22} {'Side':<6} {'n':>5} {'wins':>6} {'wr':>7} {'avg_pnl':>10} {'tot_pnl':>10}  Lectura")
    print("-" * 110)

    rows = []
    for (state, side), grp in t.groupby(["market_condition", "side"]):
        n = len(grp)
        wins = int(grp["winner"].sum())
        wr = wins / n * 100
        avg_pnl = grp["pnl"].mean()
        tot_pnl = grp["pnl"].sum()
        rows.append((state, side, n, wins, wr, avg_pnl, tot_pnl))

    rows.sort(key=lambda r: -r[2])  # por n desc
    for state, side, n, wins, wr, avg, tot in rows:
        if n < 5:
            interp = "(muestra pequeña)"
        elif wr < 35:
            interp = "🚨 mal clasificado — losing"
        elif wr < 45:
            interp = "⚠️ marginal"
        elif wr < 55:
            interp = "≈ random — state no predictor"
        elif wr < 65:
            interp = "✓ algún edge"
        else:
            interp = "✅ buen edge"
        print(f"{state:<22} {side:<6} {n:>5} {wins:>6} {wr:>6.1f}% {avg:>+9.2f}$ {tot:>+9.2f}$  {interp}")


def report_sl_range_hypothesis(t: pd.DataFrame) -> None:
    """¿El SL/range es predictor de PnL?"""
    print(f"\n{'='*78}")
    print(f"  HIPÓTESIS: SL/RANGE PREDICE PnL?")
    print(f"{'='*78}")

    if "sl_to_range_pct" not in t.columns or t["sl_to_range_pct"].isna().all():
        print("  ❌ No hay datos de SL/range. Verifica que load_daily_ranges devuelva data.")
        return

    valid = t[t["sl_to_range_pct"].notna()].copy()
    if len(valid) == 0:
        print("  ❌ Ningún trade tiene range disponible.")
        return

    # Buckets
    buckets = [(0, 3, "muy cómodo (<3%)"),
               (3, 6, "cómodo (3-6%)"),
               (6, 9, "moderado (6-9%)"),
               (9, 12, "ajustado (9-12%)"),
               (12, 100, "apretado (>12%) ⚠️")]

    print(f"\n  Distribución por bucket de SL/range_diario:")
    print(f"  {'Bucket':<22} {'n':>5} {'win_rate':>10} {'avg_pnl':>10} {'tot_pnl':>10}")
    print("  " + "-" * 70)

    bucket_stats = []
    for lo, hi, label in buckets:
        sub = valid[(valid["sl_to_range_pct"] >= lo) & (valid["sl_to_range_pct"] < hi)]
        if len(sub) == 0:
            continue
        wr = sub["winner"].mean() * 100
        avg = sub["pnl"].mean()
        tot = sub["pnl"].sum()
        bucket_stats.append((label, len(sub), wr, avg, tot))
        print(f"  {label:<22} {len(sub):>5} {wr:>9.1f}% {avg:>+9.2f}$ {tot:>+9.2f}$")

    # Comparativa apretados vs cómodos
    tight = valid[valid["sl_to_range_pct"] > 9]
    comfy = valid[valid["sl_to_range_pct"] <= 9]
    if len(tight) > 0 and len(comfy) > 0:
        print(f"\n  Comparativa:")
        print(f"    Apretados (SL/rng > 9%):  n={len(tight):>4} | wr={tight['winner'].mean()*100:.1f}% | avg_pnl={tight['pnl'].mean():+.2f}$ | total={tight['pnl'].sum():+.2f}$")
        print(f"    Cómodos   (SL/rng ≤ 9%):  n={len(comfy):>4} | wr={comfy['winner'].mean()*100:.1f}% | avg_pnl={comfy['pnl'].mean():+.2f}$ | total={comfy['pnl'].sum():+.2f}$")
        diff = comfy["pnl"].mean() - tight["pnl"].mean()
        print(f"    Diferencia avg_pnl:       {diff:+.2f}$/trade")

        # Test de significancia simple (Welch's t-test si scipy disponible)
        try:
            from scipy.stats import ttest_ind
            stat, p = ttest_ind(comfy["pnl"], tight["pnl"], equal_var=False)
            sig = "✅ significativo" if p < 0.05 else ("⚠️ marginal" if p < 0.1 else "❌ no significativo")
            print(f"    Test (Welch's t-test):    t={stat:.2f}, p={p:.4f}  {sig}")
        except ImportError:
            pass

    # Correlación
    if len(valid) > 10:
        corr = valid["sl_to_range_pct"].corr(valid["pnl"])
        print(f"\n  Correlación SL/range_pct vs PnL: {corr:+.3f}")
        if abs(corr) < 0.05:
            interp = "muy débil"
        elif abs(corr) < 0.15:
            interp = "débil"
        elif abs(corr) < 0.3:
            interp = "moderada"
        else:
            interp = "fuerte"
        print(f"  Interpretación: correlación {interp} ({'negativa' if corr < 0 else 'positiva'})")


def report_range_distribution(t: pd.DataFrame) -> None:
    """Distribución del range diario y outcomes."""
    print(f"\n{'='*78}")
    print(f"  RANGE DIARIO — DISTRIBUCIÓN Y PnL")
    print(f"{'='*78}")
    by_day = t.groupby("date").agg(
        n=("pnl", "size"),
        wins=("winner", "sum"),
        pnl=("pnl", "sum"),
        range=("range", "first"),
        char=("character", "first"),
        tr=("trend_ratio", "first"),
        avg_sl=("sl_distance", "mean"),
    ).reset_index()
    by_day["wr"] = by_day["wins"] / by_day["n"] * 100
    by_day["sl_to_range"] = by_day["avg_sl"] / by_day["range"] * 100

    print(f"\n  {'Date':<12} {'Char':<11} {'Range':>8} {'TR':>6} {'n':>5} {'wr':>6} {'avg_sl':>8} {'sl/rng':>8} {'PnL':>10}")
    print("  " + "-" * 90)
    for _, r in by_day.sort_values("date").iterrows():
        warn = "⚠️" if r["sl_to_range"] > 12 else ""
        emoji = "🟢" if r["pnl"] > 0 else "🔴"
        print(f"  {r['date']:<12} {r['char']:<11} {r['range']:>7.2f}$ {r['tr']:>5.2f}  {r['n']:>5} {r['wr']:>5.1f}% {r['avg_sl']:>7.2f}$ {r['sl_to_range']:>7.1f}% {emoji} {r['pnl']:>+8.2f}$ {warn}")


def report_sl_recommendation(t: pd.DataFrame) -> None:
    """Si hipótesis confirmada, sugerir fórmula de SL adaptativo."""
    print(f"\n{'='*78}")
    print(f"  RECOMENDACIÓN — SL ADAPTATIVO")
    print(f"{'='*78}")

    valid = t[t["sl_to_range_pct"].notna()].copy()
    if len(valid) < 20:
        print("  Muestra insuficiente para recomendación robusta.")
        return

    # Si la hipótesis se confirma, ¿cuál sería un buen umbral?
    # Pivot point: SL ≤ X% del range → trade rentable.
    valid_sorted = valid.sort_values("sl_to_range_pct")
    cum_pnl = valid_sorted["pnl"].cumsum().values
    pcts = valid_sorted["sl_to_range_pct"].values

    if len(pcts) > 0:
        # Buscar el threshold óptimo (donde cum_pnl es máximo)
        idx_max = cum_pnl.argmax()
        optimal_threshold = pcts[idx_max] if idx_max < len(pcts) else pcts[-1]
        max_cum_pnl = cum_pnl[idx_max]
        total_pnl = cum_pnl[-1]
        print(f"\n  Umbral óptimo (operando solo SL/range ≤ X):")
        print(f"    X óptimo: {optimal_threshold:.1f}% — cum PnL máximo: {max_cum_pnl:+.2f}$")
        print(f"    vs total sin filtro: {total_pnl:+.2f}$")
        print(f"    Mejora: {max_cum_pnl - total_pnl:+.2f}$")

    # Stats para SL distance sugerida basada en ATR del día (proxy: range/4)
    print(f"\n  Stats actuales de SL distance:")
    print(f"    SL absoluto:        mean={valid['sl_distance'].mean():.2f}$ | median={valid['sl_distance'].median():.2f}$")
    print(f"    Range diario:       mean={valid['range'].mean():.2f}$ | median={valid['range'].median():.2f}$")
    print(f"    Ratio SL/range:     mean={valid['sl_to_range_pct'].mean():.1f}% | median={valid['sl_to_range_pct'].median():.1f}%")

    # Fórmula sugerida: SL = min(ATR * k_atr, range_diario * k_range)
    # Donde k_range ~5% sería un buen punto de partida
    print(f"\n  Fórmula sugerida de SL adaptativo:")
    print(f"    sl_distance = min(ATR × 0.8, day_range × 0.06)")
    print()
    print(f"  Aplicación al adaptive_sl_manager.py:")
    print(f"    Añadir cap basado en day_range (estimado del HLC del día actual)")
    print(f"    para evitar que stops absolutos consuman >6% del range en días tranquilos.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--replay-dir", required=True,
                        help="Directorio con trades.parquet (output de replay_s2_202500.py)")
    args = parser.parse_args()

    replay_dir = Path(args.replay_dir)
    trades_path = replay_dir / "trades.parquet"
    if not trades_path.exists():
        print(f"❌ No se encontró: {trades_path}", file=sys.stderr)
        return 2

    print(f"📂 Cargando trades: {trades_path}")
    trades = pd.read_parquet(trades_path)
    print(f"   n_trades: {len(trades)}")
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    dates = sorted(trades["entry_time"].dt.strftime("%Y-%m-%d").unique().tolist())
    print(f"   rango: {dates[0]} → {dates[-1]} ({len(dates)} días)")

    print(f"\n📥 Cargando OHLCV de la BD para {len(dates)} días...")
    daily = load_daily_ranges(dates)
    if daily.empty:
        print(f"❌ No se obtuvieron OHLCV. ¿Está la BD accesible?", file=sys.stderr)
        return 2
    print(f"   {len(daily)} días con datos.")

    t = enrich_trades(trades, daily)

    # Reportes
    report_state_classification(t)
    report_range_distribution(t)
    report_sl_range_hypothesis(t)
    report_sl_recommendation(t)

    return 0


if __name__ == "__main__":
    sys.exit(main())
