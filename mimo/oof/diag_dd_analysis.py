"""
diag_dd_analysis.py
═══════════════════════════════════════════════════════════════════════════
Analiza la peor ventana (max DD) del walkforward y compara con las
ventanas positivas para entender qué falló.

PREGUNTA: ¿Qué tiene de distinto el mes peor vs los meses ganadores?

MÉTRICAS POR VENTANA (extraídas del walkforward_report):
  · ev_net, prec_TP, frac_SL, frac_EXPIRE
  · n_signals, sig_rate
  · mdd_R intra-ventana
  · threshold seleccionado (indica qué tan selectivo fue ese mes)

COMPARATIVA worst-vs-pos:
  · Diferencias en prec_TP (¿precision cayó dramáticamente?)
  · Diferencias en frac_SL (¿stops triggerearon más?)
  · Diferencias en threshold (¿modelo se volvió más confidente sin razón?)
  · Diferencias en n_signals (¿overtrading?)

USO:
  python -m mimo.oof.diag_dd_analysis \\
    --walkforward-json artifacts/202603_GBM/oof/<tag>/reports/walkforward_report_raw.json \\
    --side long
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, List

import numpy as np


def _fnum(v, d=float("nan")):
    return d if v is None else float(v)


def _r_window(w: Dict[str, Any], side: str) -> float:
    d = w.get(side) or {}
    n = int(d.get("n_signals", 0) or 0)
    ev = _fnum(d.get("ev_net"), 0.0)
    return ev * n if (n > 0 and math.isfinite(ev)) else 0.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--walkforward-json", required=True)
    ap.add_argument("--side", choices=("long", "short"), default="long")
    args = ap.parse_args()

    with open(args.walkforward_json) as fh: wf = json.load(fh)
    windows = [w for w in wf.get("windows", []) if not w.get("skipped")]
    if not windows:
        raise SystemExit("❌ Sin ventanas válidas")

    side = args.side
    rs = [(w, _r_window(w, side)) for w in windows]

    # Identify worst (peak-to-trough)
    cumsum = np.cumsum([r for _, r in rs])
    running_max = np.maximum.accumulate(np.concatenate(([0.0], cumsum)))[1:]
    dd_curve = running_max - cumsum
    worst_idx = int(dd_curve.argmax())
    worst_R_in_window = min((r for _, r in rs))
    worst_window_idx = int(np.argmin([r for _, r in rs]))

    print(f"📊 Walkforward {side.upper()} análisis | release={wf['release']}")
    print(f"   {len(rs)} ventanas | R_total={sum(r for _, r in rs):+.2f}R")
    print(f"\n   Max DD = {dd_curve[worst_idx]:.2f}R en ventana #{worst_idx+1}")
    print(f"   Peor ventana intra-monthly = #{worst_window_idx+1} con R={worst_R_in_window:+.2f}R")

    # Per-window table
    print("\n" + "═" * 90)
    print(f"  PER-WINDOW STATS ({side.upper()})")
    print("═" * 90)
    print(f"  {'#':>3} {'test_start':<12} {'R':>9} {'cumR':>9} {'thr':>6} "
          f"{'n_sig':>6} {'sig_rate':>9} {'prec':>6} {'frac_SL':>8} {'frac_EXP':>9} {'mdd_R':>7}")
    for i, (w, r) in enumerate(rs):
        d = w[side]
        cum = float(cumsum[i])
        marker = ""
        if i == worst_idx: marker = " ← max DD"
        elif i == worst_window_idx: marker = " ← worst R"
        print(f"  {i+1:>3} {w['window'][2]:<12} {r:>+9.2f} {cum:>+9.2f} "
              f"{_fnum(d.get('thr')):>6.3f} {int(d.get('n_signals',0)):>6} "
              f"{_fnum(d.get('sig_rate')):>9.5f} {_fnum(d.get('prec_TP')):>6.3f} "
              f"{_fnum(d.get('frac_SL')):>8.3f} {_fnum(d.get('frac_EXP')):>9.3f} "
              f"{_fnum(d.get('mdd_R')):>7.2f}{marker}")

    # Comparative worst vs positives
    pos = [(w, r) for w, r in rs if r > 0]
    neg = [(w, r) for w, r in rs if r < 0]
    worst = rs[worst_window_idx]
    if not pos:
        print("\n⚠️  Ninguna ventana positiva — no se puede comparar")
        return

    def _agg(rows, side):
        arr = [r[0][side] for r in rows]
        m = lambda k: float(np.mean([_fnum(d.get(k)) for d in arr if d.get(k) is not None]))
        return {"prec_TP": m("prec_TP"), "frac_SL": m("frac_SL"),
                "frac_EXP": m("frac_EXP"), "thr": m("thr"),
                "sig_rate": m("sig_rate"), "mdd_R": m("mdd_R"),
                "n_signals": float(np.mean([int(d.get("n_signals", 0) or 0) for d in arr])),
                "n_windows": len(rows)}

    agg_pos   = _agg(pos, side)
    agg_neg   = _agg(neg, side)
    worst_dict = worst[0][side]
    worst_stats = {"prec_TP": _fnum(worst_dict.get("prec_TP")),
                   "frac_SL": _fnum(worst_dict.get("frac_SL")),
                   "frac_EXP": _fnum(worst_dict.get("frac_EXP")),
                   "thr": _fnum(worst_dict.get("thr")),
                   "sig_rate": _fnum(worst_dict.get("sig_rate")),
                   "mdd_R": _fnum(worst_dict.get("mdd_R")),
                   "n_signals": int(worst_dict.get("n_signals", 0) or 0)}

    print("\n" + "═" * 90)
    print(f"  COMPARATIVA: peor mes vs media de positivos vs media de negativos")
    print("═" * 90)
    print(f"  {'Métrica':<14} | {'WORST':>12} | {'POS (n='+str(len(pos))+')':>14} "
          f"| {'NEG (n='+str(len(neg))+')':>14} | {'Δ worst vs pos':>16}")
    print("  " + "─" * 78)
    keys = ["prec_TP", "frac_SL", "frac_EXP", "thr", "sig_rate", "n_signals", "mdd_R"]
    for k in keys:
        ws = worst_stats[k]
        ps = agg_pos[k]
        ns = agg_neg[k]
        delta = ws - ps if math.isfinite(ws) and math.isfinite(ps) else float("nan")
        ws_str = f"{ws:>12.4f}" if isinstance(ws, float) else f"{ws:>12d}"
        ps_str = f"{ps:>14.4f}"
        ns_str = f"{ns:>14.4f}"
        d_str  = f"{delta:>+16.4f}" if math.isfinite(delta) else f"{'n/a':>16}"
        print(f"  {k:<14} | {ws_str} | {ps_str} | {ns_str} | {d_str}")

    # Interpretación
    print("\n══ INTERPRETACIÓN ══════════════════════════════════════════════")
    issues = []
    if math.isfinite(worst_stats["prec_TP"]) and math.isfinite(agg_pos["prec_TP"]):
        if worst_stats["prec_TP"] < agg_pos["prec_TP"] * 0.85:
            issues.append(f"  · prec_TP cayó {(1 - worst_stats['prec_TP']/agg_pos['prec_TP'])*100:.0f}% "
                          f"vs mes positivo promedio → modelo equivocado más a menudo")
    if math.isfinite(worst_stats["frac_SL"]) and math.isfinite(agg_pos["frac_SL"]):
        if worst_stats["frac_SL"] > agg_pos["frac_SL"] * 1.10:
            issues.append(f"  · frac_SL +{(worst_stats['frac_SL']/agg_pos['frac_SL']-1)*100:.0f}% vs pos → "
                          f"stops disparándose más (posible mecha intrabar)")
    if worst_stats["n_signals"] > agg_pos["n_signals"] * 1.3:
        issues.append(f"  · n_signals +{(worst_stats['n_signals']/agg_pos['n_signals']-1)*100:.0f}% vs pos → "
                      f"overtrading; threshold muy bajo en ese mes")
    if math.isfinite(worst_stats["thr"]) and math.isfinite(agg_pos["thr"]):
        if worst_stats["thr"] < agg_pos["thr"] * 0.7:
            issues.append(f"  · thr {worst_stats['thr']:.3f} mucho más bajo que pos avg "
                          f"{agg_pos['thr']:.3f} → ese mes el threshold scan eligió ser laxo")

    if not issues:
        print("  ✅ No hay desviaciones obvias; el peor mes parece variance random")
    else:
        for x in issues: print(x)

    print("\n══ RECOMENDACIONES ══════════════════════════════════════════════")
    if worst_stats["n_signals"] > agg_pos["n_signals"] * 1.3:
        print("  · En producción aplicar piso de threshold (no aceptar thr<X)")
    if worst_stats["frac_SL"] > agg_pos["frac_SL"] * 1.10:
        print("  · Considerar barriers más anchas (TP/SL ratio menor, ej. 1.67)")
        print("    → ya cubierto en 202604_GBM (SL=1.5)")
    if all(math.isfinite(worst_stats[k]) for k in ("prec_TP", "frac_SL")):
        if worst_stats["prec_TP"] < 0.15 and worst_stats["frac_SL"] > 0.5:
            print("  · Circuit breaker: si 2-3 días consecutivos con prec<X, pausar trading")


if __name__ == "__main__":
    main()
