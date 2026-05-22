#!/usr/bin/env python3
"""
extract_best_per_side.py

Escanea un study de Optuna (oof_study_<release>_multitask) y reporta los
top-N trials por LONG, por SHORT y por score balanceado, con sus
user_attrs (thr, ev_net, sig, MDD, prec_TP) y los params del trial.

Pensado para la opción B: dos modelos especializados.
  · Best LONG  → reentrenar con esos params, usar solo P_long
  · Best SHORT → reentrenar con esos params, usar solo P_short

Uso:
  python -m mimo.oof.extract_best_per_side --release 202500
  python -m mimo.oof.extract_best_per_side --release 202500 --top-n 5 \
    --out-json artifacts/202500/oof/<tag>/reports/best_per_side.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import optuna


def _safe(d: Dict[str, Any], key: str, default: float = float("nan")) -> float:
    v = d.get(key, default) if isinstance(d, dict) else default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def _fmt_row_side(r: Dict[str, Any], side_key: str) -> str:
    ev = r.get(side_key, {}) or {}
    return (
        f"  T{r['trial']:>3} | "
        f"thr={_safe(ev, 'thr'):.3f} | "
        f"ev_net={_safe(ev, 'ev_net'):+.4f}R | "
        f"ev_gross={_safe(ev, 'ev_gross'):+.4f}R | "
        f"sig={int(_safe(ev, 'n_signals', 0)):>4} | "
        f"prec_TP={_safe(ev, 'prec_TP'):.3f} | "
        f"mdd={_safe(ev, 'mdd_R'):.1f}R | "
        f"penalty={_safe(ev, 'penalty_mdd'):.2f} | "
        f"score={_safe(ev, 'score'):+.4f}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--study-prefix", default="oof_study")
    ap.add_argument("--side", default="multitask")
    ap.add_argument(
        "--storage",
        default="mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db",
    )
    ap.add_argument("--top-n", type=int, default=5)
    ap.add_argument("--min-signals", type=int, default=100,
                    help="Filtrar trials con menos señales que este umbral en ese lado.")
    ap.add_argument("--out-json", default=None,
                    help="Si se pasa, persiste el reporte completo a JSON.")
    args = ap.parse_args()

    study_name = f"{args.study_prefix}_{args.release}_{args.side}"
    print(f"📂 Cargando study '{study_name}'...")
    study = optuna.load_study(study_name=study_name, storage=args.storage)

    rows: List[Dict[str, Any]] = []
    for t in study.trials:
        if t.state.name != "COMPLETE":
            continue
        ev_long = t.user_attrs.get("ev_long", {}) or {}
        ev_short = t.user_attrs.get("ev_short", {}) or {}
        rows.append({
            "trial": t.number,
            "value": float(t.value) if t.value is not None else float("nan"),
            "ev_long": ev_long,
            "ev_short": ev_short,
            "params": dict(t.params),
        })

    if not rows:
        print("❌ No hay trials completados en este study.")
        return

    print(f"   {len(rows)} trials completados\n")

    # ── filtros y rankings ──────────────────────────────────────────────
    def _has_min_sig(r: Dict[str, Any], side_key: str) -> bool:
        return int(_safe(r.get(side_key, {}), "n_signals", 0)) >= args.min_signals

    rows_long = [r for r in rows if _has_min_sig(r, "ev_long")]
    rows_short = [r for r in rows if _has_min_sig(r, "ev_short")]

    by_long = sorted(
        rows_long,
        key=lambda r: _safe(r["ev_long"], "score", float("-inf")),
        reverse=True,
    )
    by_short = sorted(
        rows_short,
        key=lambda r: _safe(r["ev_short"], "score", float("-inf")),
        reverse=True,
    )
    by_combined = sorted(rows, key=lambda r: r["value"], reverse=True)

    # ── stats globales ──────────────────────────────────────────────────
    n_long_pos = sum(1 for r in rows_long if _safe(r["ev_long"], "score") > 0)
    n_short_pos = sum(1 for r in rows_short if _safe(r["ev_short"], "score") > 0)
    n_both_pos = sum(
        1 for r in rows
        if _safe(r["ev_long"], "score") > 0 and _safe(r["ev_short"], "score") > 0
    )
    n_combined_pos = sum(1 for r in rows if r["value"] > 0)

    print("=" * 100)
    print("  STATS GLOBALES")
    print("=" * 100)
    print(f"  Trials completados              : {len(rows)}")
    print(f"  Trials con LONG  sig>={args.min_signals}        : {len(rows_long)}")
    print(f"  Trials con SHORT sig>={args.min_signals}        : {len(rows_short)}")
    print(f"  Trials con ev_long score > 0    : {n_long_pos} "
          f"({100*n_long_pos/max(len(rows_long),1):.0f}%)")
    print(f"  Trials con ev_short score > 0   : {n_short_pos} "
          f"({100*n_short_pos/max(len(rows_short),1):.0f}%)")
    print(f"  Trials con AMBOS lados > 0      : {n_both_pos} "
          f"({100*n_both_pos/max(len(rows),1):.0f}%)")
    print(f"  Trials con score balanceado > 0 : {n_combined_pos} "
          f"({100*n_combined_pos/max(len(rows),1):.0f}%)")

    # ── tablas top-N ─────────────────────────────────────────────────────
    print("\n" + "=" * 100)
    print(f"  TOP {args.top_n} LONG  (rank por ev_long.score)")
    print("=" * 100)
    for r in by_long[: args.top_n]:
        print(_fmt_row_side(r, "ev_long"))

    print("\n" + "=" * 100)
    print(f"  TOP {args.top_n} SHORT (rank por ev_short.score)")
    print("=" * 100)
    for r in by_short[: args.top_n]:
        print(_fmt_row_side(r, "ev_short"))

    print("\n" + "=" * 100)
    print(f"  TOP {args.top_n} COMBINED (rank por score balanceado)")
    print("=" * 100)
    for r in by_combined[: args.top_n]:
        ev_l = r["ev_long"] or {}
        ev_s = r["ev_short"] or {}
        print(
            f"  T{r['trial']:>3} | balanced={r['value']:+.4f} | "
            f"LONG ev_net={_safe(ev_l, 'ev_net'):+.4f} sig={int(_safe(ev_l, 'n_signals', 0))} "
            f"thr={_safe(ev_l, 'thr'):.3f} | "
            f"SHORT ev_net={_safe(ev_s, 'ev_net'):+.4f} sig={int(_safe(ev_s, 'n_signals', 0))} "
            f"thr={_safe(ev_s, 'thr'):.3f}"
        )

    # ── params del #1 por lado ──────────────────────────────────────────
    if by_long:
        print("\n" + "=" * 100)
        print(f"  PARAMS DEL BEST LONG  (Trial {by_long[0]['trial']})")
        print("=" * 100)
        for k, v in sorted(by_long[0]["params"].items()):
            print(f"  {k:>22s} : {v}")
    if by_short:
        print("\n" + "=" * 100)
        print(f"  PARAMS DEL BEST SHORT (Trial {by_short[0]['trial']})")
        print("=" * 100)
        for k, v in sorted(by_short[0]["params"].items()):
            print(f"  {k:>22s} : {v}")

    # ── potencial combinado ─────────────────────────────────────────────
    if by_long and by_short:
        ev_l = by_long[0]["ev_long"]
        ev_s = by_short[0]["ev_short"]
        total_R = (
            _safe(ev_l, "ev_net") * _safe(ev_l, "n_signals", 0)
            + _safe(ev_s, "ev_net") * _safe(ev_s, "n_signals", 0)
        )
        print("\n" + "=" * 100)
        print("  POTENCIAL COMBINADO  (dos modelos especializados, opción B)")
        print("=" * 100)
        print(f"  LONG  Trial {by_long[0]['trial']}  : "
              f"+{_safe(ev_l, 'ev_net'):.4f}R × {int(_safe(ev_l, 'n_signals', 0))} sig "
              f"= {_safe(ev_l, 'ev_net') * _safe(ev_l, 'n_signals', 0):+.2f}R")
        print(f"  SHORT Trial {by_short[0]['trial']} : "
              f"+{_safe(ev_s, 'ev_net'):.4f}R × {int(_safe(ev_s, 'n_signals', 0))} sig "
              f"= {_safe(ev_s, 'ev_net') * _safe(ev_s, 'n_signals', 0):+.2f}R")
        print(f"  TOTAL R_net en holdout (~6 meses): {total_R:+.2f}R")

    # ── persistencia opcional ───────────────────────────────────────────
    if args.out_json:
        out = Path(args.out_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "study_name": study_name,
            "release": args.release,
            "n_completed": len(rows),
            "stats": {
                "trials_completed": len(rows),
                "long_score_positive": n_long_pos,
                "short_score_positive": n_short_pos,
                "both_sides_positive": n_both_pos,
                "combined_score_positive": n_combined_pos,
            },
            "top_long": by_long[: args.top_n],
            "top_short": by_short[: args.top_n],
            "top_combined": by_combined[: args.top_n],
        }
        with open(out, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, default=str)
        print(f"\n📁 Reporte completo persistido: {out}")


if __name__ == "__main__":
    main()
