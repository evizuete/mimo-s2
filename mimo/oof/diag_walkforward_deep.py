"""
diag_walkforward_deep.py
═══════════════════════════════════════════════════════════════════════════
Diagnóstico PROFUNDO de un walkforward_report.json. Va más allá del
simulator estándar y responde 5 preguntas que importan para diagnosticar
por qué una estrategia (típicamente LONG) colapsa con costos realistas:

  1) ¿Qué ventanas son tóxicas?
     · Definición: sig_rate alto + prec_TP baja + score ≤ 0
     · Suelen ser causa del colapso a cost elevado
     · Las identificamos y reportamos por separado

  2) ¿Cuánto soporta cada ventana de cost?
     · cost_breakeven = ev_gross (cost hasta el cual la ventana
       sigue siendo rentable; con cost > ev_gross la ventana pierde)
     · Reportamos cost_breakeven mediano, p10, p25 por estrategia

  3) Distribución del threshold elegido
     · Muestra si el scanner converge a thresholds estables o
       fluctúa wildly entre ventanas (señal de inestabilidad)

  4) Simulación de NO-TRADE RULE (filtro de calidad)
     · Para cada nivel de cost ∈ {0.05, 0.10, 0.15, 0.20}:
       - Calcula R_total y Sharpe SIN filtro (baseline)
       - Calcula R_total y Sharpe FILTRANDO ventanas con score ≤ 0
       - Calcula R_total y Sharpe FILTRANDO ventanas con prec_TP < 0.22
       - Muestra el delta: ¿la regla mejora o empeora?

  5) Comparativa LONG vs SHORT
     · Side-by-side de las métricas para diagnosticar dónde
       falla el modelo en una dirección y no en la otra.

USO:
  python3 -m mimo.oof.diag_walkforward_deep \\
    --walkforward-json artifacts/.../walkforward_report_cnn_mlp.json \\
    --out-json artifacts/.../diagnostic_deep.json \\
    --initial-capital 10000 --risk-per-trade-pct 1

OUTPUT:
  · stdout: tablas formateadas para inspección humana
  · --out-json: JSON con todas las métricas para consumo programático
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# Métricas atómicas
# ═══════════════════════════════════════════════════════════════════════════

def _annualized_sharpe(returns: List[float], periods_per_year: int = 12) -> float:
    """Sharpe anualizado de retornos por ventana."""
    r = np.asarray([x for x in returns if x is not None and not math.isnan(float(x))],
                   dtype=np.float64)
    if len(r) < 2: return float("nan")
    mean = float(np.mean(r))
    std = float(np.std(r, ddof=1))
    if std <= 0: return float("nan")
    return (mean * periods_per_year) / (std * math.sqrt(periods_per_year))


def _max_drawdown(returns: List[float]) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if len(r) == 0: return 0.0
    cum = np.cumsum(r)
    rmax = np.maximum.accumulate(np.concatenate(([0.0], cum)))[1:]
    return float((rmax - cum).max())


def _r_per_window(side_data: Dict[str, Any]) -> float:
    """R neto que aportó la ventana para una estrategia (long o short)."""
    if not side_data: return 0.0
    return float(side_data.get("total_R_net") or 0.0)


def _ev_gross(side_data: Dict[str, Any]) -> float:
    """EV bruto por señal (antes de cost). Sirve para cost-breakeven."""
    if not side_data: return float("nan")
    eg = side_data.get("ev_gross")
    return float(eg) if eg is not None else float("nan")


# ═══════════════════════════════════════════════════════════════════════════
# 1) Detección de ventanas tóxicas
# ═══════════════════════════════════════════════════════════════════════════

def _classify_window(side_data: Dict[str, Any],
                     *,
                     toxic_sig_rate: float = 0.04,
                     toxic_prec: float = 0.20) -> str:
    """
    Clasifica una ventana según su salud:
      · 'toxic'   : sig_rate alto + prec baja (arrastra cost)
      · 'star'    : pocas señales + prec alta + R positivo
      · 'mediocre': resto
      · 'skipped' : sin datos
    """
    if not side_data: return "skipped"
    sr = side_data.get("sig_rate", 0) or 0
    prec = side_data.get("prec_TP", 0) or 0
    n_sig = side_data.get("n_signals", 0) or 0
    total_r = side_data.get("total_R_net", 0) or 0

    if n_sig < 5:
        return "low_volume"
    if sr >= toxic_sig_rate and prec < toxic_prec:
        return "toxic"
    if prec >= 0.30 and total_r > 0:
        return "star"
    if total_r > 0:
        return "ok"
    return "mediocre"


# ═══════════════════════════════════════════════════════════════════════════
# 2) Cost-breakeven por ventana
# ═══════════════════════════════════════════════════════════════════════════

def _cost_breakeven(side_data: Dict[str, Any]) -> Optional[float]:
    """
    Cost máximo donde la ventana sigue siendo EV positivo.
    cost_breakeven = ev_gross. Con cost > ev_gross la ventana pierde.
    Si ev_gross ≤ 0, la ventana es perdedora incluso sin costos.
    """
    eg = _ev_gross(side_data)
    if math.isnan(eg): return None
    return float(eg)


# ═══════════════════════════════════════════════════════════════════════════
# 3) Simulación con filtros (no-trade rules)
# ═══════════════════════════════════════════════════════════════════════════

def _simulate_with_filter(windows: List[Dict[str, Any]],
                          side: str,
                          *,
                          cost_per_signal: float,
                          filter_fn) -> Dict[str, Any]:
    """
    Re-simula los retornos por ventana aplicando:
      · El cost variable que se pasa (en lugar del cost original del walkforward)
      · Una función de filtro: filter_fn(side_data) → True (operar) / False (skip)

    Reconstruye total_R_net por ventana como:
      ev_net_new = ev_gross - cost_per_signal
      total_R_new = ev_net_new × n_signals  (si filtro pasa) else 0
    """
    returns = []
    n_skipped = 0
    n_traded = 0
    n_signals_total = 0
    for w in windows:
        if w.get("skipped"): continue
        side_data = w.get(side) or {}
        if not side_data:
            returns.append(0.0)
            continue
        if not filter_fn(side_data):
            returns.append(0.0)
            n_skipped += 1
            continue
        eg = _ev_gross(side_data)
        n_sig = side_data.get("n_signals", 0) or 0
        if math.isnan(eg) or n_sig == 0:
            returns.append(0.0)
            continue
        ev_net = eg - cost_per_signal
        r = ev_net * n_sig
        returns.append(r)
        n_traded += 1
        n_signals_total += n_sig

    return {
        "returns":     returns,
        "R_total":     float(sum(returns)),
        "sharpe":      _annualized_sharpe(returns, periods_per_year=12),
        "max_dd":      _max_drawdown(returns),
        "n_windows_traded":  n_traded,
        "n_windows_skipped": n_skipped,
        "n_signals_total":   int(n_signals_total),
    }


# ═══════════════════════════════════════════════════════════════════════════
# 4) Equivalencia monetaria (recap del simulator estándar)
# ═══════════════════════════════════════════════════════════════════════════

def _money_eur(r_total: float, initial_capital: float, risk_pct: float, returns: Optional[List[float]] = None) -> Dict[str, float]:
    risk_frac = risk_pct / 100.0
    r_per_eur = initial_capital * risk_frac
    lin_eur = r_total * r_per_eur
    out = {
        "lin_pct":   r_total * risk_pct,
        "lin_eur":   lin_eur,
        "lin_final": initial_capital + lin_eur,
    }
    if returns is not None and len(returns) > 0:
        gf = 1.0 + risk_frac * np.asarray(returns, dtype=np.float64)
        if (gf <= 0).any():
            out.update({"comp_pct": None, "comp_eur": None, "comp_final": None, "comp_ruined": True})
        else:
            cf = float(np.prod(gf))
            out.update({
                "comp_pct":    (cf - 1.0) * 100.0,
                "comp_eur":    initial_capital * (cf - 1.0),
                "comp_final":  initial_capital * cf,
                "comp_ruined": False,
            })
    return out


# ═══════════════════════════════════════════════════════════════════════════
# Renderizado de tablas
# ═══════════════════════════════════════════════════════════════════════════

def _fmt(v, fmt="{:+.3f}"):
    if v is None: return "    n/a"
    try:
        f = float(v)
        if math.isnan(f): return "    n/a"
        return fmt.format(f)
    except: return "    n/a"


def _print_section(title: str) -> None:
    print()
    print("═" * 100)
    print(f"  {title}")
    print("═" * 100)


# ═══════════════════════════════════════════════════════════════════════════
# Análisis principal
# ═══════════════════════════════════════════════════════════════════════════

def analyze(wf: Dict[str, Any],
            *,
            initial_capital: float = 10000.0,
            risk_pct: float = 1.0) -> Dict[str, Any]:
    windows = wf.get("windows", [])
    if not windows:
        raise SystemExit("❌ Walkforward sin ventanas")

    # ─── Header ───
    walk_cfg = wf.get("walk_config", {})
    print(f"\n📂 Walkforward: {walk_cfg.get('walk_from', '?')} → {walk_cfg.get('walk_to', '?')}")
    print(f"   {len(windows)} ventanas | train={walk_cfg.get('train_months')}m, test={walk_cfg.get('test_months')}m")
    print(f"   arch={wf.get('arch', '?')}, mode={wf.get('mode', '?')}")
    print(f"   capital={initial_capital:,.0f}€, riesgo/trade={risk_pct:.2f}% → 1R={initial_capital*risk_pct/100:,.2f}€")

    # ─── 1) Clasificación de ventanas ───
    _print_section("1) CLASIFICACIÓN DE VENTANAS — tóxicas, estrella, mediocres")

    classifications = {"long": [], "short": []}
    rows_by_side = {"long": [], "short": []}
    for w in windows:
        if w.get("skipped"): continue
        ts = w["window"][2]
        for side in ("long", "short"):
            sd = w.get(side) or {}
            cls = _classify_window(sd)
            classifications[side].append(cls)
            rows_by_side[side].append({
                "ts": ts,
                "n_signals": sd.get("n_signals", 0) or 0,
                "sig_rate":  sd.get("sig_rate", 0) or 0,
                "prec_TP":   sd.get("prec_TP", 0) or 0,
                "total_R":   sd.get("total_R_net", 0) or 0,
                "thr":       sd.get("thr", 0) or 0,
                "score":     sd.get("score", 0) or 0,
                "ev_gross":  sd.get("ev_gross", 0) or 0,
                "class":     cls,
            })

    for side in ("long", "short"):
        print(f"\n  {side.upper()}")
        print(f"  {'window':<14} {'cls':<10} {'thr':>5} {'sig_rate':>9} {'n_sig':>6} {'prec':>6} {'ev_gross':>9} {'score':>8} {'R':>8}")
        print(f"  {'─' * 90}")
        for r in rows_by_side[side]:
            tag = {
                "toxic":      "❌ TOXIC",
                "star":       "⭐ STAR",
                "ok":         "✓ ok",
                "mediocre":   "· med",
                "low_volume": ". quiet",
                "skipped":    "  skip",
            }.get(r["class"], r["class"])
            print(f"  {r['ts']:<14} {tag:<10} {r['thr']:>5.2f} "
                  f"{r['sig_rate']*100:>8.2f}% {r['n_signals']:>6} "
                  f"{r['prec_TP']*100:>5.1f}% {r['ev_gross']:>+9.3f} "
                  f"{r['score']:>+8.3f} {r['total_R']:>+7.2f}R")

        # Resumen por clase
        from collections import Counter
        cnt = Counter(classifications[side])
        print(f"\n    Resumen: {dict(cnt)}")

    # ─── 2) Cost-breakeven por ventana ───
    _print_section("2) COST-BREAKEVEN — ¿qué cost soporta cada ventana antes de perder?")

    cb_stats = {}
    for side in ("long", "short"):
        cbs = []
        for w in windows:
            sd = w.get(side) or {}
            cb = _cost_breakeven(sd)
            if cb is not None: cbs.append(cb)
        if not cbs: continue
        cb_stats[side] = {
            "n":     len(cbs),
            "min":   min(cbs),
            "p25":   float(np.percentile(cbs, 25)),
            "p50":   float(np.median(cbs)),
            "p75":   float(np.percentile(cbs, 75)),
            "max":   max(cbs),
            "n_below_010": int(sum(1 for c in cbs if c < 0.10)),
            "n_below_005": int(sum(1 for c in cbs if c < 0.05)),
            "n_negative":  int(sum(1 for c in cbs if c <= 0)),
        }

    print(f"\n  {'Side':<8} | {'p25':>7} | {'p50 (med)':>10} | {'p75':>7} | {'< 0.05':>9} | {'< 0.10':>9} | {'EV≤0':>8}")
    print(f"  {'─' * 80}")
    for side, s in cb_stats.items():
        print(f"  {side.upper():<8} | {s['p25']:>+7.3f} | {s['p50']:>+10.3f} | "
              f"{s['p75']:>+7.3f} | {s['n_below_005']:>4}/{s['n']:<4} | "
              f"{s['n_below_010']:>4}/{s['n']:<4} | {s['n_negative']:>4}/{s['n']}")
    print(f"\n  Interpretación: 'p50 (med)' es el cost máximo que tolera la VENTANA MEDIANA antes de pérdida.")
    print(f"  Si p50 < 0.10, ese lado ya es marginal con costos realistas de mercado.")

    # ─── 3) Distribución de thresholds ───
    _print_section("3) DISTRIBUCIÓN DE THRESHOLDS — ¿el scanner es estable?")

    print(f"\n  {'Side':<8} | {'thr_min':>8} | {'thr_p25':>8} | {'thr_med':>8} | {'thr_p75':>8} | {'thr_max':>8} | {'std':>8}")
    print(f"  {'─' * 80}")
    thr_stats = {}
    for side in ("long", "short"):
        thrs = [r["thr"] for r in rows_by_side[side] if r["thr"] > 0]
        if not thrs: continue
        thr_stats[side] = {
            "min":   min(thrs), "max": max(thrs),
            "p25":   float(np.percentile(thrs, 25)),
            "p50":   float(np.median(thrs)),
            "p75":   float(np.percentile(thrs, 75)),
            "std":   float(np.std(thrs, ddof=1)) if len(thrs) > 1 else 0.0,
        }
        s = thr_stats[side]
        print(f"  {side.upper():<8} | {s['min']:>8.3f} | {s['p25']:>8.3f} | "
              f"{s['p50']:>8.3f} | {s['p75']:>8.3f} | {s['max']:>8.3f} | {s['std']:>8.4f}")
    print(f"\n  Interpretación: std alto → scanner inestable, elige thresholds dispares según ventana.")

    # ─── 4) Simulación con filtros ───
    _print_section("4) IMPACTO DE NO-TRADE RULES (filtros de calidad)")

    cost_levels = [0.05, 0.10, 0.15, 0.20]
    filter_results = {}

    filters = {
        "BASELINE (sin filtro)":            lambda sd: True,
        # IMPORTANTE: todos los filtros se aplican sobre sd["scan"] (métricas
        # del scanner en val_thr, ex-ante), NUNCA sobre sd top-level (que son
        # métricas del test, ex-post → look-ahead bias). El campo "scan" lo
        # guarda main_oof_cnn_walkforward._one_side para trazabilidad y para
        # poder filtrar honestamente en producción.
        "Filter: scan.score > 0":              lambda sd: (((sd or {}).get("scan") or {}).get("score") or -999) > 0,
        "Filter: scan.prec_TP >= 0.22":        lambda sd: (((sd or {}).get("scan") or {}).get("prec_TP") or 0) >= 0.22,
        "Filter: scan.score > 0 AND scan.prec >= 0.22": lambda sd: ((((sd or {}).get("scan") or {}).get("score") or -999) > 0) and ((((sd or {}).get("scan") or {}).get("prec_TP") or 0) >= 0.22),
        "Filter: scan.thr >= 0.42 (hard floor)": lambda sd: (((sd or {}).get("scan") or {}).get("thr") or 0) >= 0.42,
    }

    for side in ("long", "short", "combined"):
        print(f"\n  {side.upper()}")
        print(f"  {'Filter':<40} | {'cost':>5} | {'R_total':>9} | {'Sharpe':>7} | {'MaxDD':>7} | {'Wins':>6} | {'Skips':>6} | {'Signals':>8}")
        print(f"  {'─' * 105}")

        filter_results[side] = {}
        for fname, ffn in filters.items():
            filter_results[side][fname] = {}
            for cost in cost_levels:
                if side == "combined":
                    sim_l = _simulate_with_filter(windows, "long",  cost_per_signal=cost, filter_fn=ffn)
                    sim_s = _simulate_with_filter(windows, "short", cost_per_signal=cost, filter_fn=ffn)
                    r_combined = [a + b for a, b in zip(sim_l["returns"], sim_s["returns"])]
                    sim = {
                        "returns":     r_combined,
                        "R_total":     sum(r_combined),
                        "sharpe":      _annualized_sharpe(r_combined),
                        "max_dd":      _max_drawdown(r_combined),
                        "n_windows_traded":  sim_l["n_windows_traded"] + sim_s["n_windows_traded"],
                        "n_windows_skipped": sim_l["n_windows_skipped"] + sim_s["n_windows_skipped"],
                        "n_signals_total":   sim_l["n_signals_total"] + sim_s["n_signals_total"],
                    }
                else:
                    sim = _simulate_with_filter(windows, side, cost_per_signal=cost, filter_fn=ffn)
                filter_results[side][fname][f"cost_{cost}"] = sim

                fname_disp = fname if cost == cost_levels[0] else ""
                print(f"  {fname_disp:<40} | {cost:>5.2f} | {sim['R_total']:>+9.2f} | "
                      f"{_fmt(sim['sharpe']):>7} | {sim['max_dd']:>7.2f} | "
                      f"{sim['n_windows_traded']:>6} | {sim['n_windows_skipped']:>6} | "
                      f"{sim['n_signals_total']:>8}")
            print(f"  {'─' * 105}")

    # ─── 5) Recomendación accionable ───
    _print_section("5) RECOMENDACIONES ACCIONABLES")

    rec = []
    # LONG marginal a cost=0.10
    base_long_010 = filter_results["long"]["BASELINE (sin filtro)"]["cost_0.1"]
    best_long_010 = max(
        (filter_results["long"][f]["cost_0.1"] for f in filters),
        key=lambda s: s.get("sharpe") if s.get("sharpe") is not None and not math.isnan(s.get("sharpe", float('nan'))) else -999
    )

    if base_long_010["sharpe"] is not None and base_long_010["sharpe"] < 1.0:
        rec.append(f"⚠️  LONG es marginal a cost=0.10 (Sharpe baseline={base_long_010['sharpe']:.2f}).")
    if best_long_010["sharpe"] is not None and base_long_010["sharpe"] is not None:
        if best_long_010["sharpe"] > base_long_010["sharpe"] * 1.3:
            rec.append(f"✅ Un filtro mejora LONG significativamente: Sharpe pasa de {base_long_010['sharpe']:.2f} → {best_long_010['sharpe']:.2f}.")
            rec.append(f"   Considera integrar el mejor filtro al threshold scanner.")

    base_short_010 = filter_results["short"]["BASELINE (sin filtro)"]["cost_0.1"]
    if base_short_010["sharpe"] is not None and base_short_010["sharpe"] > 3.0:
        rec.append(f"⭐ SHORT es excelente sin necesidad de filtros (Sharpe={base_short_010['sharpe']:.2f}). Desplegar tal cual.")

    base_comb_010 = filter_results["combined"]["BASELINE (sin filtro)"]["cost_0.1"]
    if base_comb_010["sharpe"] is not None and base_comb_010["sharpe"] > 2.0 and base_long_010.get("sharpe", -999) < 1.0:
        rec.append(f"💡 COMBINED rinde por arrastre del SHORT. Considera desplegar SOLO SHORT para máximo Sharpe.")

    cb_long_p50 = cb_stats.get("long", {}).get("p50")
    if cb_long_p50 is not None and cb_long_p50 < 0.10:
        rec.append(f"⚠️  cost_breakeven mediano LONG = {cb_long_p50:.3f} (< 0.10R). LONG no robusto a costos reales.")

    cb_short_p50 = cb_stats.get("short", {}).get("p50")
    if cb_short_p50 is not None and cb_short_p50 >= 0.15:
        rec.append(f"✅ cost_breakeven mediano SHORT = {cb_short_p50:.3f} (≥ 0.15R). SHORT muy robusto.")

    if rec:
        for r in rec: print(f"\n  {r}")
    else:
        print("\n  (Sin recomendaciones automáticas — examina las tablas manualmente)")

    # ─── Equivalencia EUR del mejor filtro a cost=0.10 ───
    _print_section(f"6) MEJOR ESCENARIO REALISTA (cost=0.10) — equivalencia EUR")

    print(f"\n  capital inicial = {initial_capital:,.0f}€, riesgo/trade = {risk_pct:.2f}% → 1R = {initial_capital*risk_pct/100:,.2f}€\n")
    print(f"  {'Strategy':<12} {'Filter':<40} | {'R':>8} | {'Sharpe':>7} | {'EUR LIN':>10} | {'EUR COMP':>10} | {'Final':>10}")
    print(f"  {'─' * 110}")

    for side in ("long", "short", "combined"):
        # Best filter por Sharpe a cost=0.10
        candidates = []
        for fname, by_cost in filter_results[side].items():
            sim = by_cost["cost_0.1"]
            sh = sim.get("sharpe")
            if sh is None or math.isnan(sh): continue
            candidates.append((fname, sim))
        if not candidates: continue
        candidates.sort(key=lambda x: x[1]["sharpe"], reverse=True)
        best_name, best = candidates[0]
        money = _money_eur(best["R_total"], initial_capital, risk_pct, best["returns"])
        comp_str = "RUINED" if money.get("comp_ruined") else (f"{money.get('comp_eur'):+10,.0f}€" if money.get('comp_eur') is not None else "n/a")
        comp_fin = "n/a" if money.get("comp_ruined") else (f"{money.get('comp_final'):10,.0f}€" if money.get('comp_final') is not None else "n/a")
        print(f"  {side.upper():<12} {best_name:<40} | "
              f"{best['R_total']:>+7.2f}R | {best['sharpe']:>+7.3f} | "
              f"{money['lin_eur']:>+9,.0f}€ | {comp_str:>10} | {comp_fin:>10}")

    return {
        "walkforward_json": str(wf.get("walkforward_json", "")),
        "classifications":  classifications,
        "cost_breakeven_stats": cb_stats,
        "threshold_stats":  thr_stats,
        "filter_results":   {
            side: {
                fname: {
                    cost_key: {
                        # Exclude 'returns' from JSON output to keep it small
                        k: v for k, v in sim.items() if k != "returns"
                    } for cost_key, sim in by_cost.items()
                } for fname, by_cost in by_side.items()
            } for side, by_side in filter_results.items()
        },
        "recommendations":  rec,
    }


# ═══════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--walkforward-json", required=True,
                    help="Path al walkforward_report.json a analizar")
    ap.add_argument("--out-json", default=None,
                    help="Path para escribir el diagnóstico en JSON")
    ap.add_argument("--initial-capital", type=float, default=10000.0)
    ap.add_argument("--risk-per-trade-pct", type=float, default=1.0)
    args = ap.parse_args()

    print(f"\n📂 Cargando {args.walkforward_json}")
    wf = json.load(open(args.walkforward_json))
    wf["walkforward_json"] = args.walkforward_json

    report = analyze(wf,
                     initial_capital=args.initial_capital,
                     risk_pct=args.risk_per_trade_pct)

    if args.out_json:
        with open(args.out_json, "w") as fh:
            json.dump(report, fh, indent=2, default=str)
        print(f"\n📁 Diagnóstico persistido en: {args.out_json}")


if __name__ == "__main__":
    main()
