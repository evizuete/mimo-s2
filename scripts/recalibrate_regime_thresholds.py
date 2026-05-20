#!/usr/bin/env python3
"""
recalibrate_regime_thresholds.py — Recalcula los thresholds del StateDetector
del deploy activo en producción sobre datos recientes y los persiste en los
meta.json de los scalers_* del deploy.

Motivación
──────────
El StateDetector clasifica el régimen de cada bar (TREND_UP, RANGE, VOLATILE,
LOW_VOL, etc.) usando 6 thresholds que se calculan como percentiles del
training period (vol_low/high son p20/p80 de atr_norm, bb_p20/35/70 son
percentiles de bb_width, rexp_p80 de range_expansion).

Cuando el mercado entra en un régimen de volatilidad estructuralmente distinto
al del training (p.ej. oro a $4,700 vs $2,500 del train), los thresholds quedan
desfasados y la clasificación de regímenes se rompe: TODO se etiqueta como
VOLATILE (o como LOW_VOL si el drift va al otro lado), independientemente del
estado real del mercado.

Detectado en producción el 2026-05-20: tras promover TCN v4 deploy, el sistema
clasificó el 68% del tiempo como VOLATILE porque vol_high persistido era
10.31 bps pero el p80 actual del mercado era 15.05 bps (+46%).

Solución: este script recompute los 6 thresholds sobre los últimos N días
(default 90) y reescribe los meta.json correspondientes. NO toca el modelo
TCN ni los calibradores — solo la clasificación de estado.

Cadencia recomendada
────────────────────
  · Diario: ejecutar con --dry-run para detectar drift sin aplicar.
  · Mensual: aplicar con --apply si el drift confirma desfase >30%.
  · Tras eventos macro extremos (rotura de tendencia, FED, etc.): ejecutar
    --dry-run antes de la próxima sesión de trading.

Cuándo NO basta con esto y toca re-Optuna (Fase 1+)
───────────────────────────────────────────────────
Si el drift es de las FEATURES en sí (atr_norm, bb_width tienen rangos
extremos) Y el PnL del modelo en producción degrada notablemente (>10%
sostenido vs lockbox baseline), entonces los thresholds recalibrados no
salvan al modelo: el TCN está viendo distribuciones de input que no vio
en train. Toca ciclo completo 001→006.

Uso
───
  # Dry-run (compute + comparativa, sin escribir):
  python scripts/recalibrate_regime_thresholds.py \\
    --release 202500 \\
    --deploy-subdir deploy_validation_combined_seed47 \\
    --days 90

  # Aplicar (con backup automático):
  python scripts/recalibrate_regime_thresholds.py \\
    --release 202500 \\
    --deploy-subdir deploy_validation_combined_seed47 \\
    --days 90 \\
    --apply

  # Auto-detectar deploy activo desde main/s2_main.py:
  python scripts/recalibrate_regime_thresholds.py --auto-detect-deploy --apply
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# Silenciar logs ruidosos de TF (se carga al importar el pipeline).
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")


# Threshold keys que persiste el StateDetector en meta.json bajo "regime_thresholds".
THRESHOLD_KEYS = ("vol_low", "vol_high", "bb_p20", "bb_p35", "bb_p70", "rexp_p80")


def auto_detect_deploy(release: str) -> str:
    """Lee main/s2_main.py y extrae el deploy_subdir activo (línea no-comentada
    que asigna artifacts_path)."""
    s2 = _PROJECT_ROOT / "main" / "s2_main.py"
    if not s2.exists():
        raise SystemExit(f"❌ Auto-detect requiere {s2}, no existe.")
    pattern = re.compile(
        r'^\s*artifacts_path\s*=\s*str\(\(base_dir\s*/\s*"\.\."\s*/\s*"artifacts"\s*'
        r'/\s*release\s*/\s*"oof"\s*/\s*"([^"]+)"',
        re.MULTILINE,
    )
    text = s2.read_text()
    m = pattern.search(text)
    if not m:
        raise SystemExit(
            f"❌ No pude extraer deploy_subdir activo de {s2}. "
            f"Pasa --deploy-subdir explícitamente."
        )
    return m.group(1)


def compute_thresholds(release: str, deploy_dir: Path, days: int) -> Dict[str, float]:
    """Carga últimos `days` días de OHLCV, computa features y devuelve los
    6 thresholds del StateDetector."""
    from datetime import datetime as _dt
    import pandas as pd

    from mimo.data_managers.databases import Database
    from mimo.data_managers.data_manager import DataManager
    from mimo.data_managers.data_pipeline_v2 import DataPipeline
    from mimo.features.feature_builder import FeatureConfig
    from mimo.models.model_builder import ModelConfig, Config
    from mimo.states_manager.state_detector import StateConfig

    end = _dt.now()
    start = end - timedelta(days=days)
    print(f"📂 Cargando OHLCV {start.date()} → {end.date()}...")
    db = Database()
    dm = DataManager.from_database_historical_2(
        db, from_date=start, to_date=end, resample="5min"
    )
    df_rates = dm.df.copy()
    df_rates["time"] = pd.to_datetime(df_rates["time"])
    print(f"   {len(df_rates):,} barras 5min")

    # Detectar si la release usa vol_invariant / reduced features (impacta
    # algunos features pero NO los que llevan a los 6 thresholds, así que
    # los marcamos True por seguridad).
    fc = FeatureConfig(
        ema_periods=[9, 21, 50],
        price_norm_window=100,
        use_vol_invariant_features=True,
        use_reduced_features=True,
    )
    gc = Config(release=release)
    mc = ModelConfig(target_type="multitask")
    rc = StateConfig(adx_trend_threshold=25.0)

    pipeline = DataPipeline(
        general_config=gc, feature_config=fc, model_config=mc, regime_config=rc
    )
    print("⚙️  prepare_data...")
    df_prep = pipeline.prepare_data(
        df_rates,
        labels=False,
        side="both",
        set_market_condition=False,
        ensure_regime=False,
    )
    print(f"   {len(df_prep):,} filas tras prepare_data")

    new_thr = pipeline.state_detector.compute_thresholds(df_prep)
    # Anexo informativo: distribución del atr_norm bajo los thresholds nuevos
    pct_high = float((df_prep["atr_norm"] > new_thr["vol_high"]).mean() * 100)
    pct_low = float((df_prep["atr_norm"] < new_thr["vol_low"]).mean() * 100)
    new_thr["_diagnostics"] = {
        "lookback_days": days,
        "n_bars": int(len(df_prep)),
        "pct_volatile_under_new": pct_high,
        "pct_low_vol_under_new": pct_low,
        "computed_at": datetime.now().isoformat(timespec="seconds"),
    }
    return new_thr


def find_meta_paths(deploy_dir: Path) -> List[Path]:
    """Devuelve los meta.json bajo scalers_* dentro del deploy_dir."""
    pat = "scalers_*/meta.json"
    paths = sorted(deploy_dir.glob(pat))
    if not paths:
        raise SystemExit(
            f"❌ No hay {pat} bajo {deploy_dir}. "
            f"¿Deploy correcto?"
        )
    return paths


def show_diff(meta_paths: List[Path], new_thr: Dict[str, float]) -> Dict[str, float]:
    """Imprime comparativa old vs new y devuelve los thresholds 'old' del
    primer meta.json encontrado (asumimos que los 3 son consistentes)."""
    old = json.loads(meta_paths[0].read_text()).get("regime_thresholds") or {}
    print()
    print("=" * 70)
    print("COMPARATIVA THRESHOLDS")
    print("=" * 70)
    print(f"{'Key':<14} {'Old (persisted)':<22} {'New (recent)':<22} {'Δ %':<10}")
    print("-" * 70)
    max_delta_pct = 0.0
    for k in THRESHOLD_KEYS:
        ov = float(old.get(k, 0.0)) if k in old else None
        nv = float(new_thr.get(k, 0.0))
        if ov is None or ov == 0.0:
            d_str = "n/a"
        else:
            d_pct = (nv - ov) / ov * 100
            max_delta_pct = max(max_delta_pct, abs(d_pct))
            d_str = f"{d_pct:+.1f}%"
        ov_s = f"{ov:.6f}" if ov is not None else "—"
        print(f"{k:<14} {ov_s:<22} {nv:<22.6f} {d_str:<10}")
    diag = new_thr.get("_diagnostics", {})
    print()
    print("Bajo NEW thresholds, distribución actual:")
    print(f"  VOLATILE = {diag.get('pct_volatile_under_new', 0):.2f}%")
    print(f"  LOW_VOL  = {diag.get('pct_low_vol_under_new', 0):.2f}%")
    print(f"  Drift máximo absoluto: {max_delta_pct:.1f}%")
    if max_delta_pct >= 30:
        print(f"  ⚠️  Drift >= 30%: RECOMENDACIÓN aplicar.")
    elif max_delta_pct >= 15:
        print(f"  ℹ️  Drift moderado (15-30%): considera aplicar.")
    else:
        print(f"  ✅ Drift bajo (<15%): no urgente.")
    return old


def apply_thresholds(meta_paths: List[Path], new_thr: Dict[str, float]) -> None:
    """Hace backup de cada meta.json y reescribe regime_thresholds con new_thr.
    Excluye el campo informativo '_diagnostics' del JSON persistido (no es
    parte del schema oficial)."""
    persisted = {k: float(new_thr[k]) for k in THRESHOLD_KEYS}
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    for p in meta_paths:
        backup = p.with_name(f"{p.name}.before_recal_{ts}")
        shutil.copy2(p, backup)
        d = json.loads(p.read_text())
        d["regime_thresholds"] = persisted
        # Anexo informativo en el meta para auditoría sin contaminar el schema:
        d.setdefault("_history", []).append({
            "ts": datetime.now().isoformat(timespec="seconds"),
            "action": "recalibrate_regime_thresholds",
            "lookback_days": new_thr.get("_diagnostics", {}).get("lookback_days"),
            "n_bars": new_thr.get("_diagnostics", {}).get("n_bars"),
        })
        p.write_text(json.dumps(d, indent=2))
        print(f"   📦 backup: {backup}")
        print(f"   ✅ updated: {p}")
    print()
    print("✅ Thresholds persistidos en todos los meta.json del deploy.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--release", default="202500",
                    help="Release del deploy. Default: 202500.")
    ap.add_argument("--deploy-subdir", default=None,
                    help="Subdir bajo artifacts/<release>/oof/. Si --auto-detect-deploy, "
                         "se lee de main/s2_main.py.")
    ap.add_argument("--auto-detect-deploy", action="store_true",
                    help="Lee el deploy_subdir activo de main/s2_main.py.")
    ap.add_argument("--days", type=int, default=90,
                    help="Días de lookback para calcular thresholds. Default: 90.")
    ap.add_argument("--apply", action="store_true",
                    help="Aplicar los nuevos thresholds (escribir meta.json + backup). "
                         "Sin esta flag solo muestra la comparativa (dry-run).")
    args = ap.parse_args()

    if args.auto_detect_deploy:
        if args.deploy_subdir:
            raise SystemExit("❌ Usa --auto-detect-deploy O --deploy-subdir, no ambos.")
        args.deploy_subdir = auto_detect_deploy(args.release)
        print(f"🔍 Deploy auto-detectado: {args.deploy_subdir}")
    if not args.deploy_subdir:
        raise SystemExit("❌ Falta --deploy-subdir (o --auto-detect-deploy).")

    deploy_dir = (
        _PROJECT_ROOT / "artifacts" / args.release / "oof" / args.deploy_subdir
    )
    if not deploy_dir.exists():
        raise SystemExit(f"❌ Deploy no existe: {deploy_dir}")

    meta_paths = find_meta_paths(deploy_dir)
    print(f"🎯 Deploy: {deploy_dir}")
    print(f"   meta.json files: {len(meta_paths)}")

    new_thr = compute_thresholds(args.release, deploy_dir, args.days)
    show_diff(meta_paths, new_thr)

    if not args.apply:
        print()
        print("ℹ️  DRY-RUN. Para aplicar añade --apply.")
        print("   Tras --apply, reinicia s2 para que load_scalers inyecte los nuevos thresholds.")
        return

    print()
    print("=" * 70)
    print("APPLY — escribiendo meta.json (con backup)")
    print("=" * 70)
    apply_thresholds(meta_paths, new_thr)
    print()
    print("📋 Próximo paso — reiniciar s2 para que load_scalers inyecte los nuevos:")
    print("   pkill -f s2_main; sleep 3")
    print("   cd main && nohup /home/evizuete/boti/bin/python3 s2_main.py \\")
    print("     > ../logs/s2_$(date +%Y%m%d_%H%M).log 2>&1 &")


if __name__ == "__main__":
    main()
