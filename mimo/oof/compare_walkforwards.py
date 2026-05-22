"""
compare_walkforwards.py
═══════════════════════════════════════════════════════════════════════════
Compara dos JSONs de walkforward (típicamente "lookahead" vs "honest") lado
a lado para cuantificar el bias atribuible a los fixes honest:
  · NO_LOOKAHEAD_SCANNER=1 (val_internal para threshold)
  · CAUSAL_SCALER=1        (rolling scaler causal)
  · VAL_THR_MONTHS=2       (val_internal de 2 meses)

USO:
  python3 -m mimo.oof.compare_walkforwards \\
      --a artifacts/202500/reports/walkforward_report_cnn_tcn_LOOKAHEAD.json \\
      --b artifacts/202500/reports/walkforward_report_cnn_tcn.json \\
      --label-a lookahead --label-b honest

EMITE:
  · Agregados side-by-side (R_total, pwr, ev_median, prec_median, n_signals)
    por LONG / SHORT / COMBINED.
  · Tabla per-ventana de R_net long y short con delta.
  · Diff del status de fixes honest por ventana (no_lookahead, n_val_thr).
  · Estimación cuantitativa del bias (B - A en R total).

NO toca disco: solo imprime a stdout. Si quieres exportar a CSV añade
--out-csv <path>.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from typing import Any, Dict, List, Optional, Tuple


def _load(path: str) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _fmt(v: Optional[float], prec: int = 2) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        return f"{v:.{prec}f}"
    return str(v)


def _delta(b: Optional[float], a: Optional[float], prec: int = 2) -> str:
    """Signed string b - a (positive means B > A)."""
    if a is None or b is None:
        return "—"
    d = b - a
    sign = "+" if d >= 0 else ""
    return f"{sign}{d:.{prec}f}"


def _pct_change(b: Optional[float], a: Optional[float]) -> str:
    if a is None or b is None or abs(a) < 1e-9:
        return "—"
    pct = 100.0 * (b - a) / abs(a)
    sign = "+" if pct >= 0 else ""
    return f"{sign}{pct:.1f}%"


def _bar(value: float, scale: float = 5.0, width: int = 20) -> str:
    """Tiny ASCII bar centered at 0. value in same units as scale; bar fills
    ±scale → ±width chars."""
    if value == 0 or scale <= 0:
        return " " * width + "│" + " " * width
    n = int(round(min(abs(value) / scale, 1.0) * width))
    if value > 0:
        return " " * width + "│" + ("█" * n).ljust(width)
    return (" " * (width - n) + "█" * n) + "│" + " " * width


def _summary_combined(rep: dict) -> Dict[str, float]:
    sl = rep.get("summary_long", {}) or {}
    ss = rep.get("summary_short", {}) or {}
    return {
        "R_total":          (sl.get("R_total", 0.0) or 0.0)
                          + (ss.get("R_total", 0.0) or 0.0),
        "n_signals_total":  (sl.get("n_signals_total", 0) or 0)
                          + (ss.get("n_signals_total", 0) or 0),
        "n_pos_windows":    (sl.get("n_pos_windows", 0) or 0)
                          + (ss.get("n_pos_windows", 0) or 0),
        "n_windows":        max(sl.get("n_windows", 0) or 0,
                                ss.get("n_windows", 0) or 0),
    }


def _print_section(title: str) -> None:
    print(f"\n  {'═' * 78}")
    print(f"  {title}")
    print(f"  {'═' * 78}")


def _print_row(label: str, va: Any, vb: Any, prec: int = 2,
               width_label: int = 22) -> None:
    delta = _delta(vb if isinstance(vb, (int, float)) else None,
                   va if isinstance(va, (int, float)) else None, prec=prec)
    pct = _pct_change(vb if isinstance(vb, (int, float)) else None,
                      va if isinstance(va, (int, float)) else None)
    print(f"  {label:<{width_label}} {_fmt(va, prec):>12} {_fmt(vb, prec):>12} "
          f"{delta:>10} {pct:>8}")


def _print_summary(ra: dict, rb: dict, label_a: str, label_b: str) -> None:
    _print_section(f"AGGREGATES — {label_a}  vs  {label_b}")
    print(f"  {'METRIC':<22} {label_a:>12} {label_b:>12} "
          f"{'Δ (B-A)':>10} {'%Δ':>8}")
    print(f"  {'-' * 68}")

    for side in ("long", "short"):
        sa = ra.get(f"summary_{side}", {}) or {}
        sb = rb.get(f"summary_{side}", {}) or {}
        print(f"\n  ── {side.upper()} ──")
        _print_row("R_total",         sa.get("R_total"),        sb.get("R_total"))
        _print_row("pwr (% windows>0)", sa.get("pwr"),          sb.get("pwr"), prec=3)
        _print_row("ev_median",       sa.get("ev_median"),      sb.get("ev_median"), prec=3)
        _print_row("ev_p10",          sa.get("ev_p10"),         sb.get("ev_p10"), prec=3)
        _print_row("ev_p90",          sa.get("ev_p90"),         sb.get("ev_p90"), prec=3)
        _print_row("prec_median",     sa.get("prec_median"),    sb.get("prec_median"), prec=3)
        _print_row("n_signals_total", sa.get("n_signals_total"), sb.get("n_signals_total"), prec=0)
        _print_row("n_signals_median", sa.get("n_signals_median"), sb.get("n_signals_median"), prec=0)

    print(f"\n  ── COMBINED ──")
    ca = _summary_combined(ra)
    cb = _summary_combined(rb)
    _print_row("R_total",          ca["R_total"],         cb["R_total"])
    _print_row("n_signals_total",  ca["n_signals_total"], cb["n_signals_total"], prec=0)
    _print_row("n_pos_windows",    ca["n_pos_windows"],   cb["n_pos_windows"], prec=0)


def _print_per_window(ra: dict, rb: dict, label_a: str, label_b: str) -> None:
    wa = ra.get("windows", [])
    wb = rb.get("windows", [])
    map_a = {tuple(w["window"]): w for w in wa}
    map_b = {tuple(w["window"]): w for w in wb}
    keys = sorted(set(map_a) | set(map_b))

    for side in ("long", "short"):
        _print_section(f"PER-WINDOW {side.upper()} (R_net + n_signals)")
        print(f"  {'test_from':<14} {label_a + ' R':>10} {label_b + ' R':>10} "
              f"{'Δ R':>10}   {label_a + ' n':>8} {label_b + ' n':>8} "
              f"  bar (Δ R, ±10R)")
        print(f"  {'-' * 78}")
        cum_a, cum_b = 0.0, 0.0
        for k in keys:
            wa_w = map_a.get(k, {}) or {}
            wb_w = map_b.get(k, {}) or {}
            la = (wa_w.get(side) or {})
            lb = (wb_w.get(side) or {})
            wlabel = str(k[2]) if len(k) >= 3 else "?"
            ra_r = la.get("total_R_net")
            rb_r = lb.get("total_R_net")
            na = la.get("n_signals", 0)
            nb = lb.get("n_signals", 0)
            cum_a += ra_r or 0.0
            cum_b += rb_r or 0.0
            delta_r = (rb_r - ra_r) if (ra_r is not None and rb_r is not None) else None
            bar = _bar(delta_r or 0.0, scale=10.0, width=10)
            print(f"  {wlabel:<14} {_fmt(ra_r):>10} {_fmt(rb_r):>10} "
                  f"{_delta(rb_r, ra_r):>10}   {na:>8} {nb:>8}   {bar}")
        print(f"  {'-' * 78}")
        print(f"  {'CUMUL':<14} {cum_a:>10.2f} {cum_b:>10.2f} "
              f"{cum_b - cum_a:>+10.2f}")


def _print_honest_status(ra: dict, rb: dict, label_a: str, label_b: str) -> None:
    _print_section("HONEST FIXES STATUS (per window)")
    wa = ra.get("windows", [])
    wb = rb.get("windows", [])
    map_a = {tuple(w["window"]): w for w in wa}
    map_b = {tuple(w["window"]): w for w in wb}
    keys = sorted(set(map_a) | set(map_b))

    print(f"  {'test_from':<14}  "
          f"{label_a + ' no_lookahead':>22} {label_a + ' n_val_thr':>14}  "
          f"{label_b + ' no_lookahead':>22} {label_b + ' n_val_thr':>14}")
    print(f"  {'-' * 96}")
    for k in keys:
        wa_w = map_a.get(k, {}) or {}
        wb_w = map_b.get(k, {}) or {}
        wlabel = str(k[2]) if len(k) >= 3 else "?"
        a_nl = wa_w.get("no_lookahead", "?")
        b_nl = wb_w.get("no_lookahead", "?")
        a_vt = wa_w.get("n_val_thr", "?")
        b_vt = wb_w.get("n_val_thr", "?")
        print(f"  {wlabel:<14}  "
              f"{str(a_nl):>22} {str(a_vt):>14}  "
              f"{str(b_nl):>22} {str(b_vt):>14}")


def _print_bias_verdict(ra: dict, rb: dict, label_a: str, label_b: str) -> None:
    _print_section("BIAS ESTIMATE")
    sl_a = (ra.get("summary_long", {}) or {}).get("R_total", 0.0) or 0.0
    sl_b = (rb.get("summary_long", {}) or {}).get("R_total", 0.0) or 0.0
    ss_a = (ra.get("summary_short", {}) or {}).get("R_total", 0.0) or 0.0
    ss_b = (rb.get("summary_short", {}) or {}).get("R_total", 0.0) or 0.0
    d_long  = sl_b - sl_a
    d_short = ss_b - ss_a
    d_comb  = d_long + d_short
    a_comb  = sl_a + ss_a
    b_comb  = sl_b + ss_b

    print(f"  Interpretación: B (honest) - A (lookahead) → cuánto cayó el R al")
    print(f"  aplicar los fixes. Si B < A en mucho, la diferencia es bias.")
    print()
    print(f"    ΔR LONG        : {d_long:+.2f}R")
    print(f"    ΔR SHORT       : {d_short:+.2f}R")
    print(f"    ΔR COMBINED    : {d_comb:+.2f}R")
    print()
    print(f"    {label_a} COMBINED : {a_comb:+.2f}R")
    print(f"    {label_b} COMBINED : {b_comb:+.2f}R")
    if abs(a_comb) > 1e-9:
        retention = 100.0 * b_comb / a_comb if (a_comb * b_comb) > 0 else 0.0
        if a_comb > 0 and b_comb > 0:
            print(f"    Retención alpha: {retention:.1f}% — fracción de R que sobrevive a los fixes")
        elif a_comb > 0 and b_comb <= 0:
            pct_lost = 100.0 * (a_comb - b_comb) / a_comb
            print(f"    Alpha PERDIDO al aplicar fixes: {pct_lost:.1f}% — el {label_a} estaba inflado")
        else:
            print(f"    {label_a} ya era negativo — no hay alpha que retener")

    # Verdict
    print()
    if a_comb > 50 and b_comb > 50 and (b_comb / max(a_comb, 1e-9)) > 0.5:
        print("  🟢 VEREDICTO: Alpha real. El modelo sobrevive razonablemente a los fixes.")
    elif a_comb > 50 and b_comb > 0:
        print("  🟡 VEREDICTO: Alpha parcialmente real — caída fuerte pero positivo.")
    elif a_comb > 0 and b_comb <= 0:
        print("  🔴 VEREDICTO: Alpha era principalmente lookahead bias.")
    elif a_comb <= 0 and b_comb <= 0:
        print("  ⚫ VEREDICTO: Modelo no funciona ni con ni sin fixes.")


def _maybe_export_csv(ra: dict, rb: dict, label_a: str, label_b: str,
                      out_csv: str) -> None:
    wa = ra.get("windows", [])
    wb = rb.get("windows", [])
    map_a = {tuple(w["window"]): w for w in wa}
    map_b = {tuple(w["window"]): w for w in wb}
    keys = sorted(set(map_a) | set(map_b))
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "window_test_from", "side",
            f"{label_a}_R", f"{label_b}_R", "delta_R",
            f"{label_a}_n_sig", f"{label_b}_n_sig",
            f"{label_a}_prec_TP", f"{label_b}_prec_TP",
            f"{label_a}_no_lookahead", f"{label_b}_no_lookahead",
        ])
        for k in keys:
            wa_w = map_a.get(k, {}) or {}
            wb_w = map_b.get(k, {}) or {}
            test_from = str(k[2]) if len(k) >= 3 else "?"
            for side in ("long", "short"):
                la = (wa_w.get(side) or {})
                lb = (wb_w.get(side) or {})
                ra_r = la.get("total_R_net")
                rb_r = lb.get("total_R_net")
                w.writerow([
                    test_from, side,
                    ra_r, rb_r,
                    (rb_r - ra_r) if (ra_r is not None and rb_r is not None) else None,
                    la.get("n_signals"), lb.get("n_signals"),
                    la.get("prec_TP"), lb.get("prec_TP"),
                    wa_w.get("no_lookahead"), wb_w.get("no_lookahead"),
                ])
    print(f"\n  📄 CSV exportado a: {out_csv}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", required=True,
                    help="JSON A (típicamente lookahead/raw)")
    ap.add_argument("--b", required=True,
                    help="JSON B (típicamente honest/no-lookahead)")
    ap.add_argument("--label-a", default="A")
    ap.add_argument("--label-b", default="B")
    ap.add_argument("--out-csv", default=None,
                    help="Opcional: exporta tabla per-window a CSV")
    args = ap.parse_args()

    ra = _load(args.a)
    rb = _load(args.b)

    print()
    print("  " + "═" * 78)
    print(f"  WALKFORWARD COMPARISON: {args.label_a}  ←→  {args.label_b}")
    print("  " + "═" * 78)
    print(f"  A ({args.label_a}): {args.a}")
    print(f"      arch={ra.get('arch')} study={ra.get('study_name')} "
          f"best_trial={ra.get('best_trial')}")
    print(f"  B ({args.label_b}): {args.b}")
    print(f"      arch={rb.get('arch')} study={rb.get('study_name')} "
          f"best_trial={rb.get('best_trial')}")

    _print_summary(ra, rb, args.label_a, args.label_b)
    _print_per_window(ra, rb, args.label_a, args.label_b)
    _print_honest_status(ra, rb, args.label_a, args.label_b)
    _print_bias_verdict(ra, rb, args.label_a, args.label_b)

    if args.out_csv:
        _maybe_export_csv(ra, rb, args.label_a, args.label_b, args.out_csv)

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
