#!/usr/bin/env python3
"""
train_specialist.py

Orquesta el reentrenamiento de modelos "especialistas" usando los hyperparams
óptimos por lado obtenidos de `extract_best_per_side`. Genera artefactos
independientes por lado (`<exp_tag>_long_specialist`, `_short_specialist`).

Bajo el capó: lanza `main_oof_regime_weights_v7.py` por cada lado solicitado,
con --locked-params-json + --locked-side-key + --exp-tag-suffix + --optuna-trials 1.
Cada side corre en su propio subprocess para no leak state TF/Optuna.

Uso:
  python -m mimo.oof.train_specialist \
    --best-per-side-json artifacts/202500/oof/<tag>/reports/best_per_side.json \
    --side both \
    --release 202500 --base-tf 5min --target-type multitask \
    --variant-long vol_boost_td_down --variant-short vol_boost \
    --label-horizon-long 3 --label-horizon-short 3 \
    --train-from 2024-01-01 --train-to 2025-10-30 \
    --holdout-from 2025-11-01 --holdout-to 2026-05-02 \
    --objective ev_net --cost-per-signal 0.05 --max-drawdown-R 30 \
    --oof-epochs 120 --oof-patience 15
"""
from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path


def _build_cmd(args, side_key: str) -> list[str]:
    """Construye el comando para entrenar el especialista de un lado."""
    # Si seed != 42, lo añadimos al sufijo del exp_tag para no pisar el run
    # default (seed=42) cuando iteramos buscando un specialist más estable.
    suffix = f"_{side_key}_specialist"
    if int(args.seed) != 42:
        suffix += f"_seed{int(args.seed)}"
    cmd = [
        sys.executable, "-m", "mimo.oof.main_oof_regime_weights_v7",
        "--release", args.release,
        "--base-tf", args.base_tf,
        "--target-type", args.target_type,
        "--side", "both",  # multitask sigue entrenando ambas heads, usamos solo la del side
        "--variant-long", args.variant_long,
        "--variant-short", args.variant_short,
        "--label-horizon-long", str(args.label_horizon_long),
        "--label-horizon-short", str(args.label_horizon_short),
        "--train-from", args.train_from,
        "--train-to", args.train_to,
        "--holdout-from", args.holdout_from,
        "--holdout-to", args.holdout_to,
        "--use-tpe",
        "--optuna-trials", "1",
        "--objective", args.objective,
        "--cost-per-signal", str(args.cost_per_signal),
        "--ev-min-signals", str(args.ev_min_signals),
        "--max-drawdown-R", str(args.max_drawdown_R),
        "--ev-thr-lo", str(args.ev_thr_lo),
        "--ev-thr-hi", str(args.ev_thr_hi),
        "--oof-epochs", str(args.oof_epochs),
        "--oof-patience", str(args.oof_patience),
        "--locked-params-json", args.best_per_side_json,
        "--locked-side-key", side_key,
        "--exp-tag-suffix", suffix,
        "--seed", str(args.seed),
        "--notes",
        f"specialist_{side_key} from {Path(args.best_per_side_json).name} (seed={args.seed})",
    ]
    return cmd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--best-per-side-json", required=True,
                    help="JSON producido por extract_best_per_side.")
    ap.add_argument("--side", choices=["long", "short", "both"], default="both",
                    help="Qué especialista(s) entrenar.")
    # Reproducimos los flags relevantes de main_oof_regime_weights_v7:
    ap.add_argument("--release", required=True)
    ap.add_argument("--base-tf", default="5min")
    ap.add_argument("--target-type", default="multitask")
    ap.add_argument("--variant-long", default="vol_boost_td_down")
    ap.add_argument("--variant-short", default="vol_boost")
    ap.add_argument("--label-horizon-long", type=int, default=3)
    ap.add_argument("--label-horizon-short", type=int, default=3)
    ap.add_argument("--train-from", required=True)
    ap.add_argument("--train-to", required=True)
    ap.add_argument("--holdout-from", required=True)
    ap.add_argument("--holdout-to", required=True)
    ap.add_argument("--objective", default="ev_net")
    ap.add_argument("--cost-per-signal", type=float, default=0.05)
    ap.add_argument("--ev-min-signals", type=int, default=100)
    ap.add_argument("--max-drawdown-R", type=float, default=30.0)
    ap.add_argument("--ev-thr-lo", type=float, default=0.10)
    ap.add_argument("--ev-thr-hi", type=float, default=0.40)
    ap.add_argument("--oof-epochs", type=int, default=120)
    ap.add_argument("--oof-patience", type=int, default=15)
    ap.add_argument("--seed", type=int, default=42,
                    help="Seed global para Python random / numpy / TF / Optuna. "
                         "Cambiar este valor produce specialists distintos. "
                         "Si seed != 42 el exp_tag lleva sufijo '_seedN' para "
                         "no pisar el specialist default. Default: 42.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Imprime los comandos pero no los ejecuta.")
    args = ap.parse_args()

    # Validación
    json_path = Path(args.best_per_side_json)
    if not json_path.exists():
        raise SystemExit(f"❌ No existe: {json_path}")
    with json_path.open("r", encoding="utf-8") as f:
        payload = json.load(f)

    sides_to_train: list[str] = []
    if args.side in ("long", "both"):
        if not payload.get("top_long"):
            print("⚠️  best_per_side.json no contiene top_long. Saltando LONG.")
        else:
            sides_to_train.append("long")
    if args.side in ("short", "both"):
        if not payload.get("top_short"):
            print("⚠️  best_per_side.json no contiene top_short. Saltando SHORT.")
        else:
            sides_to_train.append("short")
    if not sides_to_train:
        raise SystemExit("❌ Nada que entrenar.")

    print(f"\n🎯 Specialists a entrenar: {sides_to_train}")
    print(f"📂 Best-per-side JSON      : {json_path}\n")

    for side_key in sides_to_train:
        sub = payload[f"top_{side_key}"][0]
        ev = sub.get(f"ev_{side_key}", {})
        print("=" * 80)
        print(f"  ▶ Entrenando SPECIALIST {side_key.upper()}")
        print("=" * 80)
        print(f"  trial origen     : #{sub.get('trial', '?')}")
        print(f"  thr óptimo (OOF) : {ev.get('thr', float('nan')):.4f}")
        print(f"  ev_net (OOF)     : {ev.get('ev_net', float('nan')):+.4f}R")
        print(f"  sig (OOF)        : {int(ev.get('n_signals', 0))}")
        print(f"  MDD (OOF)        : {ev.get('mdd_R', float('nan')):.1f}R")
        print()

        cmd = _build_cmd(args, side_key)
        print("$ " + " ".join(shlex.quote(c) for c in cmd))
        print()

        if args.dry_run:
            print("(dry-run — no se ejecuta)\n")
            continue

        rc = subprocess.run(cmd).returncode
        if rc != 0:
            print(f"❌ Specialist {side_key} falló con rc={rc}")
            sys.exit(rc)
        print(f"\n✅ Specialist {side_key} OK\n")

    print("\n🎉 Todos los specialists entrenados.")
    print("   Artifacts en:")
    for side_key in sides_to_train:
        suffix = f"_{side_key}_specialist"
        if int(args.seed) != 42:
            suffix += f"_seed{int(args.seed)}"
        print(
            f"     artifacts/{args.release}/oof/"
            f"rw_both_L{args.variant_long}_h{args.label_horizon_long}"
            f"_S{args.variant_short}_h{args.label_horizon_short}{suffix}/"
        )


if __name__ == "__main__":
    main()
