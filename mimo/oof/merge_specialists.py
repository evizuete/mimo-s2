#!/usr/bin/env python3
"""
merge_specialists.py

Combina los artefactos de dos specialists (long y short) en un único dir
"combined" con la estructura que `Helper.load_everything` espera para la
rama single-side. Ese dir se puede pasar como `artifacts_path` al simulator
y este cargará dos modelos distintos (uno por side), cada uno con su
calibrador, percentiles y policy.

Lo que hace:
  - copia model_<release>_multitask.keras del long_specialist  → model_<release>_long.keras
  - copia model_<release>_multitask.keras del short_specialist → model_<release>_short.keras
  - extrae oof_calibrator_<release>_multitask.joblib (dict)
       - cal_dict['long']  del long_specialist  → oof_calibrator_<release>_long.joblib
       - cal_dict['short'] del short_specialist → oof_calibrator_<release>_short.joblib
  - copia scalers_<release>/ (de long_specialist; verifica que coincide con el de short)
  - copia percentiles_<release>_long.json (del long_specialist)
  - copia percentiles_<release>_short.json (del short_specialist)
  - copia inference_policy_<release>_long.json y _short.json
  - escribe meta.json con punteros a los specialists origen y un MD5
    de los scalers para auditoría.

El simulator detecta `model_<release>_long.keras` + `model_<release>_short.keras`
en lugar de `model_<release>_multitask.keras` y carga la rama single-side.
El fix aplicado al simulator (`_extract_side_proba`) sigue funcionando porque
incluso con un modelo multitask "renombrado", `model.predict()` devuelve dict
y el helper extrae la cabeza correcta.

Uso:
  python -m mimo.oof.merge_specialists \
    --release 202500 \
    --long-dir  artifacts/202500/oof/<tag>_long_specialist \
    --short-dir artifacts/202500/oof/<tag>_short_specialist \
    --out-dir   artifacts/202500/oof/<tag>_combined
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict

import joblib


def _md5_of_dir(path: Path) -> str:
    """MD5 hash recursivo del contenido del dir (orden de archivos estable)."""
    if not path.exists():
        return ""
    h = hashlib.md5()
    for f in sorted(path.rglob("*")):
        if f.is_file():
            h.update(f.relative_to(path).as_posix().encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def _copy_required(src: Path, dst: Path, label: str) -> None:
    if not src.exists():
        raise SystemExit(f"❌ {label} no existe: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"   ✓ {label}: {src.name}")


def _copy_optional(src: Path, dst: Path, label: str) -> bool:
    if not src.exists():
        print(f"   ⚠️  {label} no existe (opcional): {src.name}")
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"   ✓ {label}: {src.name}")
    return True


def _extract_calibrator(src_calibrator: Path, dst: Path, side: str) -> None:
    """oof_calibrator_<release>_multitask.joblib es un dict {long, short}.
    Extraemos el del side requerido y lo guardamos como single-side."""
    if not src_calibrator.exists():
        raise SystemExit(f"❌ Calibrator multitask no existe: {src_calibrator}")
    cal_dict = joblib.load(str(src_calibrator))
    if not isinstance(cal_dict, dict) or side not in cal_dict:
        raise SystemExit(
            f"❌ Calibrator en {src_calibrator} no es dict con key '{side}'. "
            f"Type={type(cal_dict).__name__}, keys={list(cal_dict) if isinstance(cal_dict, dict) else 'n/a'}"
        )
    dst.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(cal_dict[side], str(dst))
    print(f"   ✓ calibrator[{side}] extraído → {dst.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--long-dir", required=True,
                    help="artifacts dir del long specialist (output de train_specialist --side long).")
    ap.add_argument("--short-dir", required=True,
                    help="artifacts dir del short specialist.")
    ap.add_argument("--out-dir", required=True,
                    help="dir destino del combined deployment.")
    ap.add_argument("--strict-scalers", action="store_true",
                    help="Si los MD5 de scalers difieren entre specialists, abortar.")
    ap.add_argument("--prefer-scalers-from",
                    choices=["long", "short"], default="long",
                    help="De qué specialist tomar scalers_<release>/ "
                         "(default: long; deberían ser idénticos).")
    args = ap.parse_args()

    long_dir = Path(args.long_dir)
    short_dir = Path(args.short_dir)
    out_dir = Path(args.out_dir)
    release = args.release

    if not long_dir.exists():
        raise SystemExit(f"❌ --long-dir no existe: {long_dir}")
    if not short_dir.exists():
        raise SystemExit(f"❌ --short-dir no existe: {short_dir}")

    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "data").mkdir(parents=True, exist_ok=True)
    (out_dir / "reports").mkdir(parents=True, exist_ok=True)

    # ── 1. Modelos: multitask renombrado a single-side ────────────────
    print("\n[1/4] Modelos:")
    src_model_long = long_dir / f"model_{release}_multitask.keras"
    src_model_short = short_dir / f"model_{release}_multitask.keras"
    dst_model_long = out_dir / f"model_{release}_long.keras"
    dst_model_short = out_dir / f"model_{release}_short.keras"
    _copy_required(src_model_long, dst_model_long,
                   "model_long ← long_specialist/multitask")
    _copy_required(src_model_short, dst_model_short,
                   "model_short ← short_specialist/multitask")

    # ── 2. Calibradores: extraer dict[side] del multitask ─────────────
    print("\n[2/4] Calibradores:")
    src_cal_long_multi = long_dir / f"oof_calibrator_{release}_multitask.joblib"
    src_cal_short_multi = short_dir / f"oof_calibrator_{release}_multitask.joblib"
    dst_cal_long = out_dir / f"oof_calibrator_{release}_long.joblib"
    dst_cal_short = out_dir / f"oof_calibrator_{release}_short.joblib"
    _extract_calibrator(src_cal_long_multi, dst_cal_long, "long")
    _extract_calibrator(src_cal_short_multi, dst_cal_short, "short")

    # ── 3. Scalers: copiar uno (deberían ser idénticos en feature set fijo)
    print("\n[3/4] Scalers:")
    src_scalers_long = long_dir / f"scalers_{release}"
    src_scalers_short = short_dir / f"scalers_{release}"
    md5_long = _md5_of_dir(src_scalers_long)
    md5_short = _md5_of_dir(src_scalers_short)
    same = md5_long == md5_short and md5_long != ""
    if same:
        print(f"   ℹ️  scalers idénticos en ambos specialists (md5={md5_long[:12]})")
    else:
        msg = (f"   ⚠️  scalers DIFIEREN entre specialists "
               f"(long md5={md5_long[:12]}, short md5={md5_short[:12]})")
        print(msg)
        if args.strict_scalers:
            raise SystemExit("❌ --strict-scalers + scalers distintos → aborto.")

    src_scalers = src_scalers_long if args.prefer_scalers_from == "long" else src_scalers_short
    dst_scalers = out_dir / f"scalers_{release}"
    if dst_scalers.exists():
        shutil.rmtree(dst_scalers)
    if src_scalers.exists():
        shutil.copytree(src_scalers, dst_scalers)
        print(f"   ✓ scalers ← {args.prefer_scalers_from}_specialist  ({len(list(dst_scalers.rglob('*')))} archivos)")
    else:
        print(f"   ⚠️  scalers source dir no existe: {src_scalers}")

    # snapshots side-specific (multitask los guarda con sufijo _multitask)
    src_snap_long = long_dir / f"scalers_{release}_multitask"
    src_snap_short = short_dir / f"scalers_{release}_multitask"
    dst_snap_long = out_dir / f"scalers_{release}_long"
    dst_snap_short = out_dir / f"scalers_{release}_short"
    if src_snap_long.exists():
        if dst_snap_long.exists():
            shutil.rmtree(dst_snap_long)
        shutil.copytree(src_snap_long, dst_snap_long)
        print(f"   ✓ snapshot scalers_<release>_long ← long_specialist/_multitask")
    if src_snap_short.exists():
        if dst_snap_short.exists():
            shutil.rmtree(dst_snap_short)
        shutil.copytree(src_snap_short, dst_snap_short)
        print(f"   ✓ snapshot scalers_<release>_short ← short_specialist/_multitask")

    # ── 4. Percentiles + inference policy + thresholds régimen ────────
    print("\n[4/4] Policies & metadatos:")
    pairs = [
        # (filename, source_dir, label)
        (f"percentiles_{release}_long.json",       long_dir,  "percentiles_long"),
        (f"percentiles_{release}_short.json",      short_dir, "percentiles_short"),
        (f"data/inference_policy_{release}_long.json",  long_dir,  "inference_policy_long"),
        (f"data/inference_policy_{release}_short.json", short_dir, "inference_policy_short"),
        (f"data/calibration_dataset_{release}_long.parquet",  long_dir,  "calibration_dataset_long"),
        (f"data/calibration_dataset_{release}_short.parquet", short_dir, "calibration_dataset_short"),
        (f"data/holdout_predictions_{release}_long.parquet",  long_dir,  "holdout_predictions_long"),
        (f"data/holdout_predictions_{release}_short.parquet", short_dir, "holdout_predictions_short"),
    ]
    for rel, src_dir, label in pairs:
        src = src_dir / rel
        dst = out_dir / rel
        _copy_optional(src, dst, label)

    # State-detector thresholds (también necesarios; cualquiera de los dos sirve)
    src_thr = long_dir / "regime_thresholds.json"
    if not src_thr.exists():
        src_thr = short_dir / "regime_thresholds.json"
    if src_thr.exists():
        _copy_optional(src_thr, out_dir / "regime_thresholds.json", "regime_thresholds")

    # ── meta.json con punteros y hashes ───────────────────────────────
    meta = {
        "release": release,
        "merged_at": datetime.now(timezone.utc).isoformat(),
        "long_specialist": str(long_dir.resolve()),
        "short_specialist": str(short_dir.resolve()),
        "scalers_md5_long": md5_long,
        "scalers_md5_short": md5_short,
        "scalers_identical": same,
        "scalers_taken_from": args.prefer_scalers_from,
        "model_long_size_bytes": dst_model_long.stat().st_size,
        "model_short_size_bytes": dst_model_short.stat().st_size,
        "notes": (
            "Combined deployment from two specialists. Each model is multitask "
            "but only the corresponding head is consumed in production "
            "(simulator handles dict outputs via _extract_side_proba)."
        ),
    }
    meta_path = out_dir / "merge_specialists_meta.json"
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print("\n" + "=" * 70)
    print(f"  ✅ Combined deployment listo en: {out_dir}")
    print("=" * 70)
    print(f"  · model_{release}_long.keras")
    print(f"  · model_{release}_short.keras")
    print(f"  · oof_calibrator_{release}_long.joblib")
    print(f"  · oof_calibrator_{release}_short.joblib")
    print(f"  · scalers_{release}/")
    print(f"  · percentiles_{release}_long.json / _short.json")
    print(f"  · data/inference_policy_{release}_long.json / _short.json")
    if not same:
        print(f"\n  ⚠️  Scalers difieren entre specialists. Tomado del lado "
              f"'{args.prefer_scalers_from}'. Si el otro hubiese cambiado el "
              f"feature engineering, las predicciones del lado contrario se "
              f"corromperán. Revisar antes de paper trading.")
    print(f"\n  📁 meta: {meta_path}")
    print()
    print("  Próximo paso: pasar este dir como artifacts_path al simulator:")
    print(f"    artifacts_path = '{out_dir}'")


if __name__ == "__main__":
    main()
