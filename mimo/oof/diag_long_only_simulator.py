"""
diag_long_only_simulator.py
═══════════════════════════════════════════════════════════════════════════
Simulador detallado del rendimiento LONG-only / SHORT-only / COMBINED
basado en walkforward_report.json. Decide si el modelo va a producción.

INPUTS:
  --walkforward-json  : walkforward_report.json (output de fase 3)
  --r-per-month       : factor para anualizar (default 12, mensual)

OUTPUT:
  · Texto: tabla comparativa de las 3 estrategias + veredicto
  · JSON: todas las métricas calculadas (long_only_sim.json)

MÉTRICAS POR ESTRATEGIA:
  · R total, R/mes (mean, median, std)
  · Sharpe annualized = (mean × periods) / (std × sqrt(periods))
  · Sortino = mean × periods / (downside_std × sqrt(periods))
  · Max drawdown (peak-to-trough sobre R acumulado)
  · Calmar = (mean × periods) / |max_dd|
  · Win rate (% ventanas positivas)
  · Profit factor = sum_R_pos / |sum_R_neg|
  · Best/worst window, max consecutive losses
  · Skew, kurtosis

USO:
  python -m mimo.oof.diag_long_only_simulator \\
    --walkforward-json artifacts/202603_GBM/oof/<tag>/reports/walkforward_report_raw.json \\
    --out-json artifacts/202603_GBM/oof/<tag>/reports/long_only_sim.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _fnum(v, d=float("nan")):
    return d if v is None else float(v)


def _money_equivalence(returns: np.ndarray,
                       *,
                       initial_capital: float,
                       risk_per_trade_pct: float) -> Dict[str, Any]:
    """
    Convierte una secuencia de R/ventana a equivalentes monetarios.

    Dos modelos de position sizing:
      · LINEAL (fixed fractional sobre CAPITAL INICIAL):
        cada R vale (initial_capital × risk_pct), constante.
        capital_final = initial × (1 + risk_pct × R_total)
      · COMPOUND (fixed fractional sobre CAPITAL ACTUAL):
        cada R_i se aplica como factor (1 + risk_pct × R_i).
        capital_final = initial × ∏(1 + risk_pct × R_i)

    El método LINEAL es más conservador y suele usarse en backtests
    académicos. El COMPOUND refleja mejor la realidad de reinvertir
    profits pero amplifica drawdowns (y picos).
    """
    r = np.asarray([x for x in returns if x is not None and not math.isnan(float(x))],
                   dtype=np.float64)
    if len(r) == 0 or initial_capital <= 0 or risk_per_trade_pct <= 0:
        return {}

    risk_frac = risk_per_trade_pct / 100.0
    r_total = float(r.sum())
    r_per_eur = initial_capital * risk_frac  # 1R = X €

    # ─── LINEAL (no compound) ───
    pct_total_linear = r_total * risk_per_trade_pct
    eur_total_linear = r_total * r_per_eur
    final_capital_linear = initial_capital + eur_total_linear

    # ─── COMPOUND ───
    # Cualquier R_i ≤ -1/risk_frac arruina al trader (ej. risk=1% → R_i ≤ -100)
    # En la práctica esto no ocurre en estos walkforwards.
    growth_factors = 1.0 + risk_frac * r
    if (growth_factors <= 0).any():
        # Trader liquidado en algún punto: NaN
        pct_total_compound = float("nan")
        eur_total_compound = float("nan")
        final_capital_compound = float("nan")
    else:
        compound_factor = float(np.prod(growth_factors))
        final_capital_compound = initial_capital * compound_factor
        eur_total_compound = final_capital_compound - initial_capital
        pct_total_compound = (compound_factor - 1.0) * 100.0

    # ─── Drawdown monetario (sobre cumsum lineal) ───
    cumsum = np.cumsum(r)
    running_max = np.maximum.accumulate(np.concatenate(([0.0], cumsum)))[1:]
    max_dd_R = float((running_max - cumsum).max()) if len(cumsum) > 0 else 0.0
    max_dd_eur_linear = max_dd_R * r_per_eur
    max_dd_pct_linear = max_dd_R * risk_per_trade_pct

    return {
        "initial_capital_eur": float(initial_capital),
        "risk_per_trade_pct":  float(risk_per_trade_pct),
        "r_value_eur":         float(r_per_eur),
        "linear": {
            "pct_total":      float(pct_total_linear),
            "eur_total":      float(eur_total_linear),
            "final_capital":  float(final_capital_linear),
            "max_dd_pct":     float(max_dd_pct_linear),
            "max_dd_eur":     float(max_dd_eur_linear),
        },
        "compound": {
            "pct_total":      float(pct_total_compound) if not math.isnan(pct_total_compound) else None,
            "eur_total":      float(eur_total_compound) if not math.isnan(eur_total_compound) else None,
            "final_capital":  float(final_capital_compound) if not math.isnan(final_capital_compound) else None,
            "ruined":         bool(math.isnan(pct_total_compound)),
        },
    }


def _compute_metrics(returns: np.ndarray, *, periods_per_year: int = 12) -> Dict[str, Any]:
    """Métricas de una secuencia de retornos por ventana (no NaN)."""
    r = np.asarray([x for x in returns if x is not None and not math.isnan(float(x))],
                   dtype=np.float64)
    if len(r) == 0:
        return {"n": 0}

    pos = r[r > 0]
    neg = r[r < 0]

    cumsum = np.cumsum(r)
    running_max = np.maximum.accumulate(np.concatenate(([0.0], cumsum)))[1:]
    drawdowns = running_max - cumsum
    max_dd = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0

    # Max consecutive losses
    consec_loss = max_consec_loss = 0
    for x in r:
        if x < 0:
            consec_loss += 1
            max_consec_loss = max(max_consec_loss, consec_loss)
        else:
            consec_loss = 0

    mean = float(np.mean(r))
    std = float(np.std(r, ddof=1)) if len(r) > 1 else 0.0
    downside_std = float(np.std(neg, ddof=1)) if len(neg) > 1 else 0.0

    sharpe = (mean * periods_per_year) / (std * math.sqrt(periods_per_year)) if std > 0 else float("nan")
    sortino = (mean * periods_per_year) / (downside_std * math.sqrt(periods_per_year)) if downside_std > 0 else float("nan")
    calmar = (mean * periods_per_year) / max_dd if max_dd > 1e-9 else float("nan")

    # Skew/kurtosis via momentos (sin scipy para evitar dependencia)
    if len(r) > 2 and std > 0:
        m3 = float(np.mean((r - mean) ** 3))
        m4 = float(np.mean((r - mean) ** 4))
        skew = m3 / (std ** 3)
        kurt = m4 / (std ** 4) - 3.0
    else:
        skew = kurt = float("nan")

    return {
        "n": int(len(r)),
        "R_total": float(r.sum()),
        "R_mean": mean,
        "R_median": float(np.median(r)),
        "R_std": std,
        "R_p10": float(np.percentile(r, 10)) if len(r) >= 2 else float("nan"),
        "R_p25": float(np.percentile(r, 25)) if len(r) >= 2 else float("nan"),
        "R_p75": float(np.percentile(r, 75)) if len(r) >= 2 else float("nan"),
        "R_p90": float(np.percentile(r, 90)) if len(r) >= 2 else float("nan"),
        "R_best": float(r.max()),
        "R_worst": float(r.min()),
        "sharpe": sharpe,
        "sortino": sortino,
        "calmar": calmar,
        "max_dd": max_dd,
        "max_consec_losses": int(max_consec_loss),
        "n_pos_windows": int(len(pos)),
        "n_neg_windows": int(len(neg)),
        "win_rate": float(len(pos) / len(r)),
        "profit_factor": float(pos.sum() / abs(neg.sum())) if neg.sum() != 0 else float("inf"),
        "avg_win": float(pos.mean()) if len(pos) > 0 else 0.0,
        "avg_loss": float(neg.mean()) if len(neg) > 0 else 0.0,
        "skew": skew,
        "kurt_excess": kurt,
        "cumsum": cumsum.tolist(),
        "returns": r.tolist(),
    }


def _verdict(metrics: Dict[str, Any]) -> str:
    """Veredicto cualitativo basado en Sharpe + Calmar + win rate."""
    if metrics["n"] == 0:
        return "❓ Sin datos"
    sh = metrics["sharpe"]; ca = metrics["calmar"]; wr = metrics["win_rate"]; rt = metrics["R_total"]
    if not (np.isfinite(sh) and np.isfinite(ca)):
        return "❓ Métricas degeneradas"
    if rt < 0 and sh < 0:
        return "❌ DESTRUYE EQUITY — descartar"
    if sh > 1.0 and ca > 1.5 and wr > 0.55:
        return "✅ STRATEGY OK — desplegar"
    if sh > 0.5 and rt > 0:
        return "⚠️  MARGINAL — desplegar con tamaño chico / seguir monitoreando"
    return "❌ NO PRODUCT — refinar antes de prod"


def _print_metrics_row(label: str, metrics: Dict[str, Any], width: int = 14) -> str:
    """Devuelve una columna formateada vertical (para tabla)."""
    if metrics["n"] == 0:
        return f"{label:<24} | {'—':>{width}}"
    def f(v, fmt="+.2f"):
        if v is None or not np.isfinite(v): return "n/a"
        return format(v, fmt)
    rows = [
        ("n ventanas",      f"{metrics['n']}"),
        ("R total",         f"{metrics['R_total']:+.2f}R"),
        ("R mean / ventana", f"{metrics['R_mean']:+.4f}R"),
        ("R median / vent.", f"{metrics['R_median']:+.4f}R"),
        ("R std / ventana", f"{metrics['R_std']:.4f}R"),
        ("R p10 / p90",     f"{metrics['R_p10']:+.4f} / {metrics['R_p90']:+.4f}R"),
        ("R best",          f"{metrics['R_best']:+.2f}R"),
        ("R worst",         f"{metrics['R_worst']:+.2f}R"),
        ("Sharpe (ann.)",   f(metrics["sharpe"])),
        ("Sortino",         f(metrics["sortino"])),
        ("Calmar",          f(metrics["calmar"])),
        ("Max DD",          f"{metrics['max_dd']:.2f}R"),
        ("Win rate",        f"{100*metrics['win_rate']:.0f}%"),
        ("Profit factor",   f(metrics["profit_factor"])),
        ("Avg win / loss",  f"{metrics['avg_win']:+.2f} / {metrics['avg_loss']:+.2f}R"),
        ("Max consec losses", f"{metrics['max_consec_losses']}"),
        ("Skew / kurt",     f"{f(metrics['skew'])} / {f(metrics['kurt_excess'])}"),
    ]
    return rows


def _print_comparative_table(per_strat: Dict[str, Dict[str, Any]]) -> None:
    """3-column comparative table: LONG_ONLY, SHORT_ONLY, COMBINED."""
    strats = list(per_strat.keys())
    rows_lists = {s: _print_metrics_row(s, per_strat[s]) for s in strats}
    # Take any one for header labels
    sample = rows_lists[strats[0]]
    if not isinstance(sample, list): return  # Sin datos en alguno

    header = f"  {'Métrica':<22}"
    for s in strats:
        header += f" | {s:>16}"
    print(header)
    print("  " + "─" * (22 + 19 * len(strats)))

    for i in range(len(sample)):
        label = sample[i][0]
        line = f"  {label:<22}"
        for s in strats:
            line += f" | {rows_lists[s][i][1]:>16}"
        print(line)


def _print_equity_curve_ascii(metrics: Dict[str, Any], strat_label: str, width: int = 60) -> None:
    """ASCII equity curve."""
    if metrics["n"] == 0: return
    cum = metrics["cumsum"]
    if len(cum) < 2: return
    lo, hi = min(cum), max(cum)
    rng = max(hi - lo, 1e-9)
    height = 10
    print(f"\n  Equity curve {strat_label} (R acumulado):")
    for i in range(height, -1, -1):
        level = lo + (i / height) * rng
        line = "    "
        for v in cum:
            line += "█" if v >= level else " "
        ann = f"  {level:+8.1f}R" if i in (0, height // 2, height) else ""
        print(line + ann)
    print(f"    {'└' + '─' * (len(cum) - 1)}  (ventanas 1→{len(cum)})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--walkforward-json", required=True)
    ap.add_argument("--periods-per-year", type=int, default=12,
                    help="Anualización factor. test_months=1 → 12. Si "
                         "test_months=3, usar 4. Etc.")
    ap.add_argument("--out-json", default=None)
    ap.add_argument("--initial-capital", type=float, default=10000.0,
                    help="Capital inicial en EUR para traducción monetaria. Default 10000€.")
    ap.add_argument("--risk-per-trade-pct", type=float, default=1.0,
                    help="Porcentaje del capital arriesgado por trade (1R). Default 1%%.")
    args = ap.parse_args()

    print(f"📂 Cargando {args.walkforward_json}")
    with open(args.walkforward_json) as fh:
        wf = json.load(fh)

    windows = wf.get("windows", [])
    if not windows:
        raise SystemExit("❌ Sin ventanas en walkforward report")

    long_R   = []
    short_R  = []
    combined_R = []

    print(f"\n📊 {len(windows)} ventanas | release={wf['release']} | mode={wf.get('mode', '?')}")
    print(f"   walk={wf['walk_config']['walk_from']} → {wf['walk_config']['walk_to']}  "
          f"train={wf['walk_config']['train_months']}m  test={wf['walk_config']['test_months']}m")

    print(f"\n  {'Ventana test':<14} | {'LONG':>10} | {'SHORT':>10} | {'L+S':>10}")
    print("  " + "─" * 53)
    for w in windows:
        if w.get("skipped"): continue
        l = w.get("long")  or {}
        s = w.get("short") or {}
        n_l = int(l.get("n_signals", 0) or 0)
        n_s = int(s.get("n_signals", 0) or 0)
        ev_l = _fnum(l.get("ev_net"))
        ev_s = _fnum(s.get("ev_net"))
        # R per window = ev_net × n_signals (gross contrib)
        r_l = ev_l * n_l if (not math.isnan(ev_l) and n_l > 0) else 0.0
        r_s = ev_s * n_s if (not math.isnan(ev_s) and n_s > 0) else 0.0
        r_c = r_l + r_s
        long_R.append(r_l); short_R.append(r_s); combined_R.append(r_c)
        ts = w["window"][2]
        print(f"  {ts:<14} | {r_l:>+9.2f}R | {r_s:>+9.2f}R | {r_c:>+9.2f}R")

    # Compute metrics for each strategy
    pp = int(args.periods_per_year)
    m_long  = _compute_metrics(np.array(long_R),     periods_per_year=pp)
    m_short = _compute_metrics(np.array(short_R),    periods_per_year=pp)
    m_comb  = _compute_metrics(np.array(combined_R), periods_per_year=pp)
    per_strat = {"LONG-only": m_long, "SHORT-only": m_short, "COMBINED L+S": m_comb}

    # Equivalencia monetaria
    money_long  = _money_equivalence(np.array(long_R),     initial_capital=args.initial_capital, risk_per_trade_pct=args.risk_per_trade_pct)
    money_short = _money_equivalence(np.array(short_R),    initial_capital=args.initial_capital, risk_per_trade_pct=args.risk_per_trade_pct)
    money_comb  = _money_equivalence(np.array(combined_R), initial_capital=args.initial_capital, risk_per_trade_pct=args.risk_per_trade_pct)

    # Comparative table
    print("\n" + "═" * 75)
    print("  TABLA COMPARATIVA DE ESTRATEGIAS")
    print("═" * 75)
    _print_comparative_table(per_strat)

    # Verdicts
    print("\n" + "═" * 75)
    print("  VEREDICTO")
    print("═" * 75)
    for label, m in per_strat.items():
        print(f"  {label:<14} : {_verdict(m)}")

    # ─── Equivalencia monetaria ───
    print("\n" + "═" * 95)
    print(f"  EQUIVALENCIA MONETARIA  (capital inicial = {args.initial_capital:,.0f}€, "
          f"riesgo/trade = {args.risk_per_trade_pct:.2f}% → 1R = {args.initial_capital * args.risk_per_trade_pct / 100:,.2f}€)")
    print("═" * 95)
    print(f"  {'Estrategia':<14} | {'R total':>9} | "
          f"{'%  LIN':>8} | {'EUR LIN':>11} | {'Final LIN':>11} | "
          f"{'%  COMP':>8} | {'EUR COMP':>11} | {'Final COMP':>11} | "
          f"{'MaxDD %':>7} | {'MaxDD €':>9}")
    print("  " + "─" * 130)
    for label, met, money in (("LONG-only", m_long, money_long),
                               ("SHORT-only", m_short, money_short),
                               ("COMBINED",   m_comb,  money_comb)):
        if not money:
            print(f"  {label:<14} | sin datos")
            continue
        lin = money["linear"]; comp = money["compound"]
        comp_pct_str = "RUINED" if comp.get("ruined") else (f"{comp['pct_total']:+8.2f}%" if comp.get('pct_total') is not None else "  n/a")
        comp_eur_str = "    n/a   " if comp.get("ruined") else (f"{comp['eur_total']:+11,.0f}€" if comp.get('eur_total') is not None else "    n/a   ")
        comp_fin_str = "    n/a   " if comp.get("ruined") else (f"{comp['final_capital']:11,.0f}€" if comp.get('final_capital') is not None else "    n/a   ")
        print(f"  {label:<14} | {met['R_total']:+8.2f}R | "
              f"{lin['pct_total']:+7.2f}% | {lin['eur_total']:+10,.0f}€ | {lin['final_capital']:10,.0f}€ | "
              f"{comp_pct_str:>8} | {comp_eur_str:>11} | {comp_fin_str:>11} | "
              f"{lin['max_dd_pct']:6.2f}% | {lin['max_dd_eur']:8,.0f}€")
    print(f"\n  Nota: LIN = position sizing fijo sobre capital inicial (conservador, sin compounding)")
    print(f"        COMP = position sizing fijo sobre capital actual (compounding, refleja realidad)")

    # Equity curves (only LONG-only — the candidate)
    _print_equity_curve_ascii(m_long, "LONG-only")

    # Persist
    report = {
        "walkforward_json": str(args.walkforward_json),
        "release": wf.get("release"),
        "mode": wf.get("mode"),
        "walk_config": wf.get("walk_config"),
        "periods_per_year": pp,
        "strategies": {
            "long_only":   m_long,
            "short_only":  m_short,
            "combined":    m_comb,
        },
        "verdicts": {
            "long_only":   _verdict(m_long),
            "short_only":  _verdict(m_short),
            "combined":    _verdict(m_comb),
        },
        "money_equivalence": {
            "long_only":   money_long,
            "short_only":  money_short,
            "combined":    money_comb,
        },
    }
    out_json = args.out_json or str(Path(args.walkforward_json).parent / "long_only_sim.json")
    with open(out_json, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\n📁 Reporte: {out_json}")


if __name__ == "__main__":
    main()
