#!/usr/bin/env python3
"""
wf_extract_best_params.py
─────────────────────────────────────────────────────────────────────────────
Extrae study.best_trial.params de un Optuna study EXISTENTE (ej: oof_study_202500_multitask)
y lo escribe como best_params_flat.json compatible con --locked-params-json.

Útil para bootstrap del walk-forward sin tener que correr Optuna desde cero:

    # 1. Extraer params del 202500
    python -m scripts.walk_forward.wf_extract_best_params \\
        --release 202500 \\
        --out artifacts/202500/best_params_flat.json

    # 2. Lanzar walk-forward con Optuna desactivado, usando esos params
    python -m scripts.walk_forward.wf_orchestrator \\
        --start 2026-01-05 --end 2026-05-04 \\
        --no-optuna \\
        --bootstrap-locked-params artifacts/202500/best_params_flat.json
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


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--release", required=True,
                    help="Release del study a extraer (ej: 202500).")
    ap.add_argument("--study-side", default="multitask",
                    help="Sufijo del study (default: multitask).")
    ap.add_argument("--study-prefix", default="oof_study")
    ap.add_argument("--storage", default=DEFAULT_OPTUNA_STORAGE)
    ap.add_argument("--out", type=Path, required=True,
                    help="Path destino del best_params_flat.json.")
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

    if study.best_trial is None:
        sys.exit("❌ el study no tiene best_trial")

    params = dict(study.best_trial.params)
    # IMPORTANTE: v7 (--locked-params-json) espera un dict PLANO de hyperparams,
    # NO un envoltorio con metadata. La metadata va a un sidecar _meta.json.
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(params, indent=2))

    meta_path = args.out.with_name(args.out.stem + "_meta.json")
    meta = {
        "release": args.release,
        "study_name": study_name,
        "best_value": float(study.best_value)
            if study.best_value is not None else None,
        "best_trial_number": study.best_trial.number,
        "params_file": args.out.name,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"\n✅ best_params_flat → {args.out}  (dict plano, {len(params)} params)")
    print(f"   metadata        → {meta_path}")
    print(f"   trial #{study.best_trial.number}, value={meta['best_value']:.6f}")
    print(f"   params:")
    for k, v in sorted(params.items()):
        print(f"     {k:>22s} : {v}")


if __name__ == "__main__":
    main()
