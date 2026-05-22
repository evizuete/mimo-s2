#!/usr/bin/env python3
"""
validate_temporal_filters.py
============================

Aplica 3 filtros temporales/contextuales sobre los trades del replay
y reporta el impacto en PnL si esos trades NO se hubieran emitido:

  · F1 (TEND_CLARA):  excluir trades en días con trend_ratio_diario >= umbral
  · F2 (HOUR_BLOCK):   excluir trades en horas estadísticamente perdedoras
  · F3 (DOW_BLOCK):    excluir trades en días de semana perdedores

Cada filtro se evalúa individualmente y en combinación (subset) para ver
contribución marginal vs solapamiento.

⚠️ ATENCIÓN — sesgo de selección
--------------------------------
Los umbrales de hora/día se eligen MIRANDO la distribución del PnL en el
LOCKBOX. Esto es entrenar en el LOCKBOX → riesgo de overfit. El resultado
acumulado es el LÍMITE SUPERIOR. Para evaluación honesta:
  1. Validar en una ventana NO usada para elegir umbrales (otro periodo).
  2. O usar criterios genéricos (ADX > X) en vez de patterns observados.

Esta validación es exploratoria — útil para responder "¿hay edge?" pero
NO para decidir si promover a producción sin validación adicional.

Uso
---
  python scripts/validate_temporal_filters.py \\
    --replay-dir /tmp/replay_lockbox_30d
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from scripts.analyze_lockbox_sl_hypothesis import load_daily_ranges, enrich_trades


# Horas y días "malos" identificados en investigation (LOCKBOX 30d):
# Estos umbrales se obtienen del análisis previo → sesgo de overfit (caveat).
BAD_HOURS = {0, 13, 16, 17, 8, 10}     # tot_pnl negativos sostenidos
BAD_DOW = {"Monday", "Wednesday"}       # peor PnL por día de semana
TEND_CLARA_TR_THR = 0.7                 # trend_ratio_diario


def apply_filters(t: pd.DataFrame, *, use_tend_clara=True, use_hours=True, use_dow=True) -> pd.DataFrame:
    """Devuelve t con columnas adicionales f1/f2/f3/passed."""
    t = t.copy()
    t["entry_time"] = pd.to_datetime(t["entry_time"])
    t["hour"] = t["entry_time"].dt.hour
    t["dow"] = t["entry_time"].dt.day_name()

    t["block_tend_clara"] = (t["trend_ratio"] >= TEND_CLARA_TR_THR) if use_tend_clara else False
    t["block_hour"] = t["hour"].isin(BAD_HOURS) if use_hours else False
    t["block_dow"] = t["dow"].isin(BAD_DOW) if use_dow else False
    t["any_block"] = t["block_tend_clara"] | t["block_hour"] | t["block_dow"]
    t["passed"] = ~t["any_block"]
    return t


def report_filter(name: str, all_t: pd.DataFrame, mask_keep: pd.Series) -> dict:
    kept = all_t[mask_keep]
    excluded = all_t[~mask_keep]
    n_all = len(all_t)
    n_keep = len(kept)
    n_excl = len(excluded)
    pnl_all = all_t["pnl"].sum()
    pnl_keep = kept["pnl"].sum()
    pnl_excl = excluded["pnl"].sum()
    wr_all = (all_t["pnl"] > 0).mean() * 100
    wr_keep = (kept["pnl"] > 0).mean() * 100 if n_keep else 0
    return {
        "name": name,
        "n_all": n_all, "n_keep": n_keep, "n_excl": n_excl,
        "pct_excl": n_excl / n_all * 100,
        "pnl_all": pnl_all, "pnl_keep": pnl_keep, "pnl_excl": pnl_excl,
        "wr_all": wr_all, "wr_keep": wr_keep,
        "improvement": pnl_keep - pnl_all,
    }


def print_filter_row(r: dict) -> None:
    print(f"  {r['name']:<35} {r['n_keep']:>5}/{r['n_all']:<5} "
          f"({r['pct_excl']:>4.0f}% excl) "
          f"PnL: {r['pnl_all']:>+8.2f}$ → {r['pnl_keep']:>+8.2f}$ "
          f"({r['improvement']:>+8.2f}$)  wr: {r['wr_all']:>4.1f}% → {r['wr_keep']:>4.1f}%")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--replay-dir", required=True)
    args = parser.parse_args()

    trades = pd.read_parquet(Path(args.replay_dir) / "trades.parquet")
    trades["entry_time"] = pd.to_datetime(trades["entry_time"])
    dates = sorted(trades["entry_time"].dt.strftime("%Y-%m-%d").unique().tolist())
    daily = load_daily_ranges(dates)
    t = enrich_trades(trades, daily)
    t["entry_time"] = pd.to_datetime(t["entry_time"])
    t["hour"] = t["entry_time"].dt.hour
    t["dow"] = t["entry_time"].dt.day_name()

    print(f"\n{'='*100}")
    print(f"  IMPACTO DE FILTROS TEMPORALES (LOCKBOX {dates[0]} → {dates[-1]}, n={len(t)})")
    print(f"{'='*100}\n")

    # 1. Sin filtros (baseline)
    base = report_filter("BASELINE (sin filtros)", t, pd.Series(True, index=t.index))
    print_filter_row(base)

    # 2. Cada filtro individual
    print(f"\n  --- Filtros individuales ---")
    f1 = report_filter(f"F1: bloquear TEND_CLARA (tr>={TEND_CLARA_TR_THR})", t, t["trend_ratio"] < TEND_CLARA_TR_THR)
    print_filter_row(f1)

    f2 = report_filter(f"F2: bloquear horas {sorted(BAD_HOURS)}", t, ~t["hour"].isin(BAD_HOURS))
    print_filter_row(f2)

    f3 = report_filter(f"F3: bloquear {BAD_DOW}", t, ~t["dow"].isin(BAD_DOW))
    print_filter_row(f3)

    # 3. Combinaciones
    print(f"\n  --- Combinaciones ---")
    mask_12 = (t["trend_ratio"] < TEND_CLARA_TR_THR) & (~t["hour"].isin(BAD_HOURS))
    print_filter_row(report_filter("F1 + F2", t, mask_12))

    mask_13 = (t["trend_ratio"] < TEND_CLARA_TR_THR) & (~t["dow"].isin(BAD_DOW))
    print_filter_row(report_filter("F1 + F3", t, mask_13))

    mask_23 = (~t["hour"].isin(BAD_HOURS)) & (~t["dow"].isin(BAD_DOW))
    print_filter_row(report_filter("F2 + F3", t, mask_23))

    mask_123 = mask_12 & (~t["dow"].isin(BAD_DOW))
    print_filter_row(report_filter("F1 + F2 + F3 (todos)", t, mask_123))

    # 4. Análisis por carácter de día (después de filtros)
    print(f"\n{'='*100}")
    print(f"  Comportamiento post-filtros (F1+F2+F3) por carácter de día")
    print(f"{'='*100}")
    filt = t[mask_123].copy()
    print(f"  n trades pasan: {len(filt)} (de {len(t)})")
    print()
    for ch, sub in filt.groupby("character"):
        wr = (sub["pnl"] > 0).mean() * 100
        print(f"  {ch:<12} n={len(sub):>4} wr={wr:>5.1f}% PnL={sub['pnl'].sum():>+8.2f}$ avg={sub['pnl'].mean():>+6.2f}$")

    # 5. Cuántos días siguen activos
    days_active_base = t["entry_time"].dt.strftime("%Y-%m-%d").nunique()
    days_active_filt = filt["entry_time"].dt.strftime("%Y-%m-%d").nunique() if len(filt) else 0
    print(f"\n  Días activos: {days_active_base} → {days_active_filt}")

    # 6. Sesgo de selección — disclaimer
    print(f"\n{'='*100}")
    print(f"  ⚠️ CAVEAT — SESGO DE SELECCIÓN")
    print(f"{'='*100}")
    print(f"  Los umbrales BAD_HOURS y BAD_DOW se eligieron observando este mismo LOCKBOX.")
    print(f"  Esto es overfitting al test set. El PnL mostrado es un LÍMITE SUPERIOR.")
    print(f"  Validación honesta requiere:")
    print(f"    1. Aplicar los mismos umbrales a un periodo distinto (e.g., otro mes)")
    print(f"    2. O elegir umbrales por criterio genérico (ADX > X) en vez de hora/día específicos")
    print(f"  Si el PnL del F1+F2+F3 es enormemente mejor que baseline, considera:")
    print(f"    · Es real y operable → validar con replay sobre otro periodo")
    print(f"    · Es overfit → encontrar criterio genérico equivalente")
    return 0


if __name__ == "__main__":
    sys.exit(main())
