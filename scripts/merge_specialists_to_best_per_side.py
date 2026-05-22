"""
merge_specialists_to_best_per_side.py
═══════════════════════════════════════════════════════════════════════════
Fusiona dos JSONs de tuning per-side (output de 012_optuna_arch.sh con
SIDE=long y SIDE=short separados, p.ej. best_params_tcn_long_only.json y
best_params_tcn_short_only.json) en un best_per_side.json compatible con
Fase 2 del deploy (002_train_specialists.sh → train_specialist.py).

El formato target lo consume main_oof_regime_weights_v7.py:3066+ vía
  --locked-params-json best_per_side.json --locked-side-key {long,short}
que lee sub[0].get("params") del array top_long/top_short.

Caso de uso típico — preparar deploy del TCN v4:
  python scripts/merge_specialists_to_best_per_side.py \\
    --long-json  artifacts/202500/oof/tuning/best_params_tcn_long_only.json \\
    --short-json artifacts/202500/oof/tuning/best_params_tcn_short_only.json \\
    --out artifacts/202500/oof/rw_both_Lvol_boost_td_down_h3_Svol_boost_h3/reports/best_per_side.json

Nota: el campo ev_long/ev_short.ev_net se rellena con el value de Optuna
(AUC-PR si --objective=auc_pr) como proxy numérico no-None. Es solo para
que la validación de 002_train_specialists.sh no falle por campo faltante;
no representa el EV real (eso requeriría walkforward, no tuning).
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--long-json", required=True,
                    help="JSON de tuning side=long (formato best_params_<arch>_long_only.json).")
    ap.add_argument("--short-json", required=True,
                    help="JSON de tuning side=short (formato best_params_<arch>_short_only.json).")
    ap.add_argument("--out", required=True,
                    help="Path de salida best_per_side.json compatible con Fase 2.")
    ap.add_argument("--no-backup", action="store_true",
                    help="Por defecto, si --out existe se mueve a un .pre_merge_<ts>. "
                         "Con --no-backup se sobreescribe sin backup.")
    args = ap.parse_args()

    long_path  = Path(args.long_json)
    short_path = Path(args.short_json)
    if not long_path.exists():
        raise SystemExit(f"❌ --long-json no existe: {long_path}")
    if not short_path.exists():
        raise SystemExit(f"❌ --short-json no existe: {short_path}")

    long_data  = json.loads(long_path.read_text())
    short_data = json.loads(short_path.read_text())

    # Validación cruzada — avisar si arch o release no coinciden
    if long_data.get("arch") != short_data.get("arch"):
        print(f"⚠️  arch mismatch: long='{long_data.get('arch')}' "
              f"short='{short_data.get('arch')}'", file=sys.stderr)
    if long_data.get("release") != short_data.get("release"):
        print(f"⚠️  release mismatch: long='{long_data.get('release')}' "
              f"short='{short_data.get('release')}'", file=sys.stderr)

    # Validar que los params estén — sin esto train_specialist falla luego.
    for label, data in (("long", long_data), ("short", short_data)):
        if not data.get("best_params"):
            raise SystemExit(
                f"❌ --{label}-json no contiene 'best_params'. "
                f"¿Es realmente un output de 012_optuna_arch.sh / main_oof_arch_tuning.py?"
            )

    arch = long_data.get("arch", "unknown")
    release = long_data.get("release", "unknown")

    merged = {
        "release": release,
        "arch": arch,
        "source": "merged_from_per_side_tuning",
        "long_study":  long_data.get("study_name"),
        "short_study": short_data.get("study_name"),
        "n_completed": (long_data.get("n_trials_completed", 0)
                        + short_data.get("n_trials_completed", 0)),
        "stats": {
            "long_best_value":  long_data.get("best_value"),
            "short_best_value": short_data.get("best_value"),
            "objective":        long_data.get("objective", "?"),
        },
        "top_long": [{
            "trial": long_data.get("best_trial"),
            "value": long_data.get("best_value"),
            # ev_long sintético: el value de Optuna (AUC-PR si objective=auc_pr,
            # EV_net si objective=ev_net del tuning); train_specialist.py:117+
            # solo exige que ev_net sea numérico no-None para validar el JSON.
            "ev_long": {
                "ev_net":    long_data.get("best_value", 0.0),
                "n_signals": 0,
                "thr":       0.0,
                "prec_TP":   0.0,
                "mdd_R":     0.0,
                "_note":     "synthetic: value=Optuna trial value, no walkforward EV",
            },
            "params": dict(long_data.get("best_params", {})),
        }],
        "top_short": [{
            "trial": short_data.get("best_trial"),
            "value": short_data.get("best_value"),
            "ev_short": {
                "ev_net":    short_data.get("best_value", 0.0),
                "n_signals": 0,
                "thr":       0.0,
                "prec_TP":   0.0,
                "mdd_R":     0.0,
                "_note":     "synthetic: value=Optuna trial value, no walkforward EV",
            },
            "params": dict(short_data.get("best_params", {})),
        }],
    }

    out_path = Path(args.out)
    if out_path.exists() and not args.no_backup:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        backup = out_path.with_name(f"{out_path.stem}_pre_merge_{ts}{out_path.suffix}")
        out_path.rename(backup)
        print(f"📦 Backup del best_per_side.json previo: {backup}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2))

    print(f"✅ Merge listo: {out_path}")
    print(f"   arch={arch} | release={release}")
    print(f"   LONG  Trial #{merged['top_long'][0]['trial']:>3} "
          f"value={merged['top_long'][0]['value']:+.4f}  "
          f"(study={merged['long_study']})")
    print(f"         params: {len(merged['top_long'][0]['params'])} keys")
    print(f"   SHORT Trial #{merged['top_short'][0]['trial']:>3} "
          f"value={merged['top_short'][0]['value']:+.4f}  "
          f"(study={merged['short_study']})")
    print(f"         params: {len(merged['top_short'][0]['params'])} keys")
    print()
    print("📋 Siguiente paso — Fase 2 con TCN:")
    print("   ARCH=tcn RELEASE={r} SEED=47 bash 002_train_specialists.sh".format(r=release))


if __name__ == "__main__":
    main()
