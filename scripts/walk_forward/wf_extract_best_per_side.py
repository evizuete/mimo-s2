#!/usr/bin/env python3
"""
wf_extract_best_per_side.py
─────────────────────────────────────────────────────────────────────────────
Extrae top-N trials por LONG y por SHORT de un Optuna study EXISTENTE
(ej: oof_study_202500_multitask) y lo escribe como best_per_side.json
compatible con `train_specialist --best-per-side-json`.

Pensado para el bootstrap del walk-forward en modo specialists:

    # 1. Generar best_per_side.json desde el study del 202500
    python -m scripts.walk_forward.wf_extract_best_per_side \\
        --release 202500 \\
        --out artifacts/202500/best_per_side.json

    # 2. Lanzar walk-forward en modo specialists con ese bootstrap
    python -m scripts.walk_forward.wf_orchestrator \\
        --start 2026-01-05 --end 2026-05-04 \\
        --mode specialists \\
        --bootstrap-best-per-side artifacts/202500/best_per_side.json \\
        --optuna-every 4 --optuna-trials 40

Diferencia frente a `wf_extract_best_params.py`:
  - Ese extrae UN best_trial → dict plano (un solo modelo).
  - Este extrae TOP-N por side (long, short, combined) → JSON con la estructura
    que `train_specialist` necesita para entrenar dos modelos especializados,
    cada uno con los hyperparams óptimos para SU side.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_OPTUNA_STORAGE = (
    "mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db"
)


def _safe(d, key, default=float("nan")):
    v = d.get(key, default) if isinstance(d, dict) else default
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--release", required=True,
                    help="Release del study a extraer (ej: 202500).")
    ap.add_argument("--study-side", default="multitask",
                    help="Sufijo del study (default: multitask).")
    ap.add_argument("--study-prefix", default="oof_study")
    ap.add_argument("--storage", default=DEFAULT_OPTUNA_STORAGE)
    ap.add_argument("--top-n", type=int, default=5,
                    help="Cuántos trials guardar por ranking (default 5). "
                         "train_specialist consume el top-1 de cada lado.")
    ap.add_argument("--min-signals", type=int, default=100,
                    help="Filtrar trials con menos señales que este umbral en "
                         "el lado correspondiente (default 100).")
    ap.add_argument("--out", type=Path, required=True,
                    help="Path destino del best_per_side.json.")
    args = ap.parse_args()

    try:
        import optuna  # type: ignore
    except ImportError:
        sys.exit("❌ optuna no instalado")

    study_name = f"{args.study_prefix}_{args.release}_{args.study_side}"
    print(f"📂 Cargando study: {study_name}")
    print(f"   storage: {args.storage}")
    try:
        study = optuna.load_study(study_name=study_name, storage=args.storage)
    except Exception as e:
        sys.exit(f"❌ no pude cargar el study: {e}")

    rows = []
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
        sys.exit("❌ el study no tiene trials COMPLETE")

    print(f"   {len(rows)} trials completados")

    def _has_min_sig(r, side_key):
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

    if not by_long:
        sys.exit(f"❌ ningún trial con n_signals_long >= {args.min_signals}")
    if not by_short:
        sys.exit(f"❌ ningún trial con n_signals_short >= {args.min_signals}")

    n_long_pos = sum(1 for r in rows_long if _safe(r["ev_long"], "score") > 0)
    n_short_pos = sum(1 for r in rows_short if _safe(r["ev_short"], "score") > 0)
    n_both_pos = sum(
        1 for r in rows
        if _safe(r["ev_long"], "score") > 0 and _safe(r["ev_short"], "score") > 0
    )
    n_combined_pos = sum(1 for r in rows if r["value"] > 0)

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
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, default=str))

    best_long = by_long[0]
    best_short = by_short[0]
    same_trial = best_long["trial"] == best_short["trial"]

    print(f"\n✅ best_per_side → {args.out}  (top_long={len(by_long[:args.top_n])}, "
          f"top_short={len(by_short[:args.top_n])})")
    print(f"   stats: long_pos={n_long_pos}/{len(rows_long)}  "
          f"short_pos={n_short_pos}/{len(rows_short)}  both_pos={n_both_pos}/{len(rows)}")
    print(f"   best LONG  : trial #{best_long['trial']}  "
          f"score={_safe(best_long['ev_long'], 'score'):+.4f}  "
          f"ev_net={_safe(best_long['ev_long'], 'ev_net'):+.4f}R")
    print(f"   best SHORT : trial #{best_short['trial']}  "
          f"score={_safe(best_short['ev_short'], 'score'):+.4f}  "
          f"ev_net={_safe(best_short['ev_short'], 'ev_net'):+.4f}R")
    if same_trial:
        print(f"   ℹ️  ambos lados coinciden en trial #{best_long['trial']} → "
              f"un único specialist sería suficiente, pero el pipeline corre "
              f"los dos por consistencia.")
    else:
        print(f"   🎯 lados en distintos trials → specialists aporta "
              f"valor real frente a un único multitask.")


if __name__ == "__main__":
    main()
