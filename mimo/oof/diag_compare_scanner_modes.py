"""
diag_compare_scanner_modes.py
═══════════════════════════════════════════════════════════════════════════
Compara los resultados de DOS walkforwards corridos en distinto modo:

  · MODO LEGACY (con look-ahead):    walkforward_report_X_legacy.json
  · MODO SIN LOOK-AHEAD:             walkforward_report_X_nolookahead.json

Reporta el delta para entender cuánto del rendimiento del modelo era
real vs cuánto era post-hoc del threshold scanner.

USO:
  python -m mimo.oof.diag_compare_scanner_modes \\
    --legacy-json   reports/walkforward_report_cnn_mlp_legacy.json \\
    --nolookahead-json reports/walkforward_report_cnn_mlp_nolookahead.json \\
    --cost-per-signal 0.10
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _annualized_sharpe(returns: List[float], periods_per_year: int = 12) -> float:
    r = np.asarray([x for x in returns if x is not None and not math.isnan(float(x))],
                   dtype=np.float64)
    if len(r) < 2: return float("nan")
    mean = float(np.mean(r))
    std  = float(np.std(r, ddof=1))
    if std <= 0: return float("nan")
    return (mean * periods_per_year) / (std * math.sqrt(periods_per_year))


def _strategy_stats(windows: List[Dict[str, Any]], side: str, cost: float) -> Dict[str, Any]:
    """Calcula R y Sharpe por lado con cost especificado.

    Re-aplica el cost (puede diferir del usado en el walkforward).
    Útil para stress test del modo nuevo.
    """
    returns = []
    n_signals_total = 0
    n_traded_windows = 0
    for w in windows:
        if w.get("skipped"): continue
        sd = w.get(side) or {}
        eg = sd.get("ev_gross")
        n  = sd.get("n_signals", 0) or 0
        if eg is None or n == 0:
            returns.append(0.0)
            continue
        ev_net = float(eg) - cost
        r = ev_net * n
        returns.append(r)
        n_signals_total += int(n)
        n_traded_windows += 1
    R = float(sum(returns))
    sh = _annualized_sharpe(returns)
    return {
        "R_total":         R,
        "sharpe":          sh,
        "n_signals":       n_signals_total,
        "n_traded_windows": n_traded_windows,
        "returns":         returns,
    }


def _fmt(v, fmt="{:+.3f}"):
    if v is None: return "    n/a"
    try:
        f = float(v)
        if math.isnan(f): return "    n/a"
        return fmt.format(f)
    except: return "    n/a"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--legacy-json", required=True)
    ap.add_argument("--nolookahead-json", required=True)
    ap.add_argument("--cost-per-signal", type=float, default=0.10,
                    help="Cost a aplicar al recalcular métricas (default 0.10 realista)")
    ap.add_argument("--initial-capital", type=float, default=10000.0)
    ap.add_argument("--risk-per-trade-pct", type=float, default=1.0)
    args = ap.parse_args()

    legacy = json.load(open(args.legacy_json))
    nolook = json.load(open(args.nolookahead_json))

    print("\n" + "═" * 100)
    print(f"  COMPARATIVA SCANNER: LEGACY (con look-ahead) vs NO-LOOKAHEAD")
    print("═" * 100)
    print(f"  Legacy report:       {args.legacy_json}")
    print(f"  No-lookahead report: {args.nolookahead_json}")
    print(f"  Cost aplicado:       {args.cost_per_signal:.2f}R")
    print(f"  Capital inicial:     {args.initial_capital:,.0f}€  (1R = {args.initial_capital*args.risk_per_trade_pct/100:,.2f}€)")

    # ─── Tabla comparativa ───
    print(f"\n  {'Strategy':<10} {'Mode':<14} | {'R_total':>9} | {'Sharpe':>7} | "
          f"{'Wins':>5} | {'Signals':>8} | {'EUR LIN':>11}")
    print(f"  {'─' * 90}")

    summary = {}
    for side in ("long", "short", "combined"):
        for mode_label, wf in (("LEGACY", legacy), ("NO-LOOKAHEAD", nolook)):
            if side == "combined":
                stats_l = _strategy_stats(wf["windows"], "long",  args.cost_per_signal)
                stats_s = _strategy_stats(wf["windows"], "short", args.cost_per_signal)
                combined_returns = [a + b for a, b in zip(stats_l["returns"], stats_s["returns"])]
                stats = {
                    "R_total":  sum(combined_returns),
                    "sharpe":   _annualized_sharpe(combined_returns),
                    "n_signals": stats_l["n_signals"] + stats_s["n_signals"],
                    "n_traded_windows": stats_l["n_traded_windows"] + stats_s["n_traded_windows"],
                    "returns":  combined_returns,
                }
            else:
                stats = _strategy_stats(wf["windows"], side, args.cost_per_signal)
            summary[(side, mode_label)] = stats
            r_per_eur = args.initial_capital * args.risk_per_trade_pct / 100
            eur = stats["R_total"] * r_per_eur
            print(f"  {side.upper():<10} {mode_label:<14} | "
                  f"{stats['R_total']:>+9.2f} | {_fmt(stats['sharpe'], '{:+7.3f}'):>7} | "
                  f"{stats['n_traded_windows']:>5} | {stats['n_signals']:>8} | "
                  f"{eur:>+10,.0f}€")
        # Delta
        leg = summary[(side, "LEGACY")]
        nlk = summary[(side, "NO-LOOKAHEAD")]
        delta_r = nlk["R_total"] - leg["R_total"]
        leg_sh = leg["sharpe"] if not (leg["sharpe"] is None or math.isnan(leg["sharpe"])) else None
        nlk_sh = nlk["sharpe"] if not (nlk["sharpe"] is None or math.isnan(nlk["sharpe"])) else None
        delta_sh = (nlk_sh - leg_sh) if (leg_sh is not None and nlk_sh is not None) else None
        delta_r_pct = (delta_r / leg["R_total"] * 100) if leg["R_total"] != 0 else 0.0
        print(f"  {side.upper():<10} {'Δ NEW vs LEG':<14} | "
              f"{delta_r:>+9.2f} | {_fmt(delta_sh, '{:+7.3f}'):>7} | "
              f"({delta_r_pct:+.1f}% R)")
        print()

    # ─── Análisis de filtros aplicables en producción usando 'scan' ───
    if any(w.get("no_lookahead") for w in nolook.get("windows", [])):
        print("\n" + "═" * 100)
        print(f"  FILTROS APLICABLES EN PRODUCCIÓN (usando métricas del scanner sobre val_internal)")
        print("═" * 100)
        print(f"\n  Si 'scan.score > 0' AND/OR 'scan.prec_TP >= 0.22' EN PRODUCCIÓN, ¿cuánto mejora?")
        print(f"\n  {'Filter':<35} {'Strategy':<10} | {'R':>8} | {'Sharpe':>7} | {'Wins':>5} | {'Skips':>5} | {'EUR LIN':>11}")
        print(f"  {'─' * 95}")

        filters = {
            "no filter":                              lambda sc: True,
            "scan.score > 0":                         lambda sc: (sc.get("score") or -999) > 0 if sc else True,
            "scan.prec_TP >= 0.22":                   lambda sc: (sc.get("prec_TP") or 0) >= 0.22 if sc else True,
            "scan.score > 0 AND scan.prec >= 0.22":   lambda sc: (sc.get("score") or -999) > 0 and (sc.get("prec_TP") or 0) >= 0.22 if sc else True,
        }

        for fname, ffn in filters.items():
            for side in ("long", "short", "combined"):
                if side == "combined":
                    sides = ("long", "short")
                else:
                    sides = (side,)
                rets = []
                n_sigs = 0; n_trad = 0; n_skip = 0
                for w in nolook["windows"]:
                    if w.get("skipped"): continue
                    win_r = 0.0
                    for s in sides:
                        sd = w.get(s) or {}
                        scan = sd.get("scan", {})
                        if not ffn(scan):
                            n_skip += 1
                            continue
                        eg = sd.get("ev_gross")
                        n  = sd.get("n_signals", 0) or 0
                        if eg is None or n == 0: continue
                        win_r += (float(eg) - args.cost_per_signal) * n
                        n_sigs += int(n)
                        n_trad += 1
                    rets.append(win_r)
                R = sum(rets)
                sh = _annualized_sharpe(rets)
                r_per_eur = args.initial_capital * args.risk_per_trade_pct / 100
                eur = R * r_per_eur
                print(f"  {fname:<35} {side.upper():<10} | "
                      f"{R:>+7.2f}R | {_fmt(sh, '{:+7.3f}'):>7} | "
                      f"{n_trad:>5} | {n_skip:>5} | {eur:>+10,.0f}€")
            print(f"  {'─' * 95}")

    # ─── Recomendaciones ───
    print("\n" + "═" * 100)
    print(f"  RECOMENDACIONES")
    print("═" * 100)

    leg_comb = summary[("combined", "LEGACY")]
    nlk_comb = summary[("combined", "NO-LOOKAHEAD")]
    if nlk_comb["sharpe"] is not None and not math.isnan(nlk_comb["sharpe"]):
        if nlk_comb["sharpe"] > 1.5:
            print(f"\n  ✅ COMBINED sin look-ahead mantiene Sharpe={nlk_comb['sharpe']:.2f} > 1.5 → DESPLEGABLE")
        elif nlk_comb["sharpe"] > 0.5:
            print(f"\n  ⚠️  COMBINED sin look-ahead: Sharpe={nlk_comb['sharpe']:.2f} → marginal, considerar filtros")
        else:
            print(f"\n  ❌ COMBINED sin look-ahead: Sharpe={nlk_comb['sharpe']:.2f} → no desplegable")
    if leg_comb["R_total"] != 0 and nlk_comb["R_total"]:
        retain_pct = nlk_comb["R_total"] / leg_comb["R_total"] * 100
        print(f"  📊 NO-LOOKAHEAD retiene {retain_pct:.1f}% del R_total LEGACY")
        if retain_pct < 50:
            print(f"     → El look-ahead inflaba MÁS DE LA MITAD del resultado. Resultado actual es el real.")
        elif retain_pct < 80:
            print(f"     → El look-ahead inflaba un 20-50% del resultado. El modelo sigue siendo bueno.")
        else:
            print(f"     → El look-ahead apenas inflaba el resultado. El modelo es robusto independiente del scanner.")


if __name__ == "__main__":
    main()
