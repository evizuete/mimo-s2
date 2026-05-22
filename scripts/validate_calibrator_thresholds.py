#!/usr/bin/env python3
"""
validate_calibrator_thresholds.py
==================================

Test de regresión: tras cualquier cambio de calibradores (iso ↔ Platt, swap
por champion, refit, etc.), valida que TODOS los umbrales del runtime que
dependen de cal_probs siguen siendo alcanzables sobre los datos OOF.

Motivación
----------
El incident INC-2026-05-20 reveló que varios umbrales defensivos del runtime
(transition_min_proba_delta, min_proba_edge, expansion_proba_threshold,
compression_proba_threshold, extension_proba_threshold) estaban calibrados
sobre cal_probs del calibrador isotónico saturado. Cuando se hizo swap a
Platt scaling, varios de esos umbrales pasaron a ser inalcanzables (0% de
paso sobre datos OOF) → bloqueo total del sistema.

Este test calcula la tasa de paso empírica de cada umbral en los datos OOF
de calibración y alerta si alguno cae fuera de un rango razonable.

Rangos de validez (configurables por umbral)
--------------------------------------------
  · MIN_PASS_RATE_PCT: si < min, el umbral es "casi imposible" → ALERT
  · MAX_PASS_RATE_PCT: si > max, el umbral es "demasiado permisivo" → ALERT

Si el umbral está en un estado "filter-defensivo" (esperamos baja activación),
ajustar el rango a [1%, 15%]. Si es un filtro de "entrada" (esperamos mayor
activación), ajustar a [10%, 50%].

Uso
---
  # Validación interactiva con tabla
  python scripts/validate_calibrator_thresholds.py

  # Modo CI/script (silencioso si OK, exit 1 si algún fallo)
  python scripts/validate_calibrator_thresholds.py --quiet

  # Cambiar deploy:
  python scripts/validate_calibrator_thresholds.py \\
    --deploy artifacts/202500/oof/deploy_validation_combined_seed47

Exit codes
----------
  0: todos los umbrales OK
  1: al menos un umbral fuera del rango razonable
  2: error (no se encontraron calibradores/datos)
"""
from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Especificación de umbrales que dependen de cal_probs
# ---------------------------------------------------------------------------

@dataclass
class ThresholdSpec:
    """Especificación de un umbral runtime que depende de cal_probs."""

    name: str
    """Identificador descriptivo del umbral."""

    source_file: str
    """Archivo donde se define el valor."""

    source_attr: str
    """Atributo o nombre de variable en el archivo (para auto-discovery)."""

    description: str
    """Qué hace el umbral en runtime (1 línea)."""

    # Cómo calcular la métrica empírica sobre los datos OOF
    metric_fn: Callable[[np.ndarray, np.ndarray, np.ndarray], np.ndarray]
    """Función (cal_long, cal_short, state) -> np.ndarray de valores a comparar
    contra el umbral. Devuelve la métrica que pasa al umbral en runtime."""

    # Filtro de estado (opcional) — solo evaluar en ciertos estados
    state_filter: Optional[Callable[[np.ndarray], np.ndarray]] = None
    """Si se provee, máscara boolean sobre states. Solo se evalúa el subset."""

    # Rangos de validez en % de paso
    min_pass_pct: float = 1.0
    """Por debajo de este % → umbral inalcanzable → ALERT."""

    max_pass_pct: float = 60.0
    """Por encima de este % → umbral demasiado permisivo → ALERT."""


def _delta_metric(cl: np.ndarray, cs: np.ndarray, _state: np.ndarray) -> np.ndarray:
    """|cal_long - cal_short| (delta absoluto, usado por transition_weak filter)."""
    return np.abs(cs - cl)


def _edge_long_metric(cl: np.ndarray, cs: np.ndarray, _state: np.ndarray) -> np.ndarray:
    """cal_long - cal_short (signed edge favorable a LONG)."""
    return cl - cs


def _edge_short_metric(cl: np.ndarray, cs: np.ndarray, _state: np.ndarray) -> np.ndarray:
    """cal_short - cal_long (signed edge favorable a SHORT)."""
    return cs - cl


def _max_side_metric(cl: np.ndarray, cs: np.ndarray, _state: np.ndarray) -> np.ndarray:
    """max(cal_long, cal_short) — el lado que más vota."""
    return np.maximum(cl, cs)


def _is_transition(state: np.ndarray) -> np.ndarray:
    return np.char.find(state.astype(str), "TRANSITION") >= 0


def _is_trend_down(state: np.ndarray) -> np.ndarray:
    return np.char.find(state.astype(str), "TREND_DOWN") >= 0


def _is_trend_up(state: np.ndarray) -> np.ndarray:
    return np.char.find(state.astype(str), "TREND_UP") >= 0


# Catálogo canónico de umbrales del runtime que dependen del calibrador.
# Actualizar esta lista cuando se añada un nuevo filtro de cal en cualquier
# parte del código (s2_service*, adaptive_*, decision_engine, etc.).
THRESHOLD_SPECS = [
    ThresholdSpec(
        name="transition_min_proba_delta",
        source_file="main/s2_config.py",
        source_attr="transition_min_proba_delta",
        description="Filtro TRANSITION_WEAK_SIGNAL: |Δ cal_long-cal_short| en TRANSITION.",
        metric_fn=_delta_metric,
        state_filter=_is_transition,
        min_pass_pct=5.0,   # filtro defensivo — debe dejar pasar al menos 5%
        max_pass_pct=40.0,
    ),
    ThresholdSpec(
        name="min_proba_edge (LONG_in_TREND_DOWN)",
        source_file="main/s2_config.py",
        source_attr="min_proba_edge",
        description="Filtro reversal_guard: edge cal_long-cal_short en TREND_DOWN para BUY.",
        metric_fn=_edge_long_metric,
        state_filter=_is_trend_down,
        min_pass_pct=5.0,
        max_pass_pct=50.0,
    ),
    ThresholdSpec(
        name="min_proba_edge (SHORT_in_TREND_UP)",
        source_file="main/s2_config.py",
        source_attr="min_proba_edge",
        description="Filtro reversal_guard: edge cal_short-cal_long en TREND_UP para SELL.",
        metric_fn=_edge_short_metric,
        state_filter=_is_trend_up,
        min_pass_pct=5.0,
        max_pass_pct=70.0,
    ),
    ThresholdSpec(
        name="expansion_proba_threshold (adaptive_sl)",
        source_file="main/adaptive_sl_manager.py",
        source_attr="expansion_proba_threshold",
        description="SL Expansion: proba >= umbral añade 1 señal (de 5) para expandir SL.",
        metric_fn=_max_side_metric,
        min_pass_pct=1.0,   # P95-P99 es OK — esperamos baja activación
        max_pass_pct=20.0,
    ),
    ThresholdSpec(
        name="compression_proba_threshold (adaptive_sl)",
        source_file="main/adaptive_sl_manager.py",
        source_attr="compression_proba_threshold",
        description="SL Compression: proba lado opuesto >= umbral activa compresión defensiva.",
        metric_fn=_max_side_metric,
        min_pass_pct=0.5,   # defensiva — esperamos muy baja activación (P99+)
        max_pass_pct=10.0,
    ),
    ThresholdSpec(
        name="compression_proba_threshold (adaptive_tp)",
        source_file="main/adaptive_tp_manager.py",
        source_attr="compression_proba_threshold",
        description="TP Compression: proba lado opuesto >= umbral añade 1 señal para comprimir TP.",
        metric_fn=_max_side_metric,
        min_pass_pct=1.0,
        max_pass_pct=20.0,
    ),
    ThresholdSpec(
        name="extension_proba_threshold (adaptive_tp)",
        source_file="main/adaptive_tp_manager.py",
        source_attr="extension_proba_threshold",
        description="TP Extension: proba lado de la posición >= umbral activa extender TP.",
        metric_fn=_max_side_metric,
        min_pass_pct=0.5,
        max_pass_pct=10.0,
    ),
]


# ---------------------------------------------------------------------------
# Auto-discovery de valores actuales desde los archivos fuente
# ---------------------------------------------------------------------------

def _find_threshold_value(file_path: Path, attr_name: str) -> Optional[float]:
    """
    Lee un archivo Python y busca la primera línea con el patrón:
        attr_name: float = VALUE
        attr_name = VALUE
    Devuelve el valor o None si no se encuentra.
    """
    if not file_path.exists():
        return None
    import re
    pattern = rf"^\s*{re.escape(attr_name)}\s*(?::\s*[A-Za-z_]+)?\s*=\s*([\d\.]+)"
    for line in file_path.read_text(encoding="utf-8").splitlines():
        # Saltar líneas comentadas
        stripped = line.lstrip()
        if stripped.startswith("#"):
            continue
        match = re.match(pattern, line)
        if match:
            return float(match.group(1))
    return None


# ---------------------------------------------------------------------------
# Lógica de validación principal
# ---------------------------------------------------------------------------

def _load_oof_data(deploy: Path, release: str) -> pd.DataFrame:
    """Carga los datos OOF de calibración y aplica el calibrador actual."""
    data_dir = deploy / "data"
    long_path = data_dir / f"deploy_calibration_tail_{release}_long.parquet"
    short_path = data_dir / f"deploy_calibration_tail_{release}_short.parquet"

    if not long_path.exists() or not short_path.exists():
        raise FileNotFoundError(
            f"Datos OOF no encontrados:\n  {long_path}\n  {short_path}"
        )

    df_l = pd.read_parquet(long_path)
    df_s = pd.read_parquet(short_path)

    merged = (
        df_l[["time", "state", "oof_proba_raw"]]
        .rename(columns={"oof_proba_raw": "raw_long"})
        .merge(
            df_s[["time", "oof_proba_raw"]].rename(columns={"oof_proba_raw": "raw_short"}),
            on="time",
        )
    )

    cal_long = joblib.load(deploy / f"oof_calibrator_{release}_long.joblib")
    cal_short = joblib.load(deploy / f"oof_calibrator_{release}_short.joblib")

    merged["cal_long"] = cal_long.predict(merged["raw_long"].values)
    merged["cal_short"] = cal_short.predict(merged["raw_short"].values)

    return merged


def _evaluate_threshold(
    spec: ThresholdSpec,
    df: pd.DataFrame,
    current_value: Optional[float],
    project_root: Path,
) -> dict:
    """Evalúa un spec sobre los datos OOF y devuelve el reporte."""
    cl = df["cal_long"].values
    cs = df["cal_short"].values
    state = df["state"].astype(str).values

    metric = spec.metric_fn(cl, cs, state)
    if spec.state_filter is not None:
        mask = spec.state_filter(state)
        metric = metric[mask]
        n_eval = int(mask.sum())
    else:
        n_eval = len(metric)

    # Auto-discover valor actual si no se proporcionó
    if current_value is None:
        current_value = _find_threshold_value(project_root / spec.source_file, spec.source_attr)

    if current_value is None:
        return {
            "spec": spec,
            "status": "ERROR_NO_VALUE",
            "n_eval": n_eval,
            "current_value": None,
            "pass_pct": None,
            "p50": None,
            "p90": None,
            "p99": None,
            "metric_max": None,
        }

    pass_pct = float((metric >= current_value).mean() * 100) if n_eval > 0 else 0.0

    if pass_pct < spec.min_pass_pct:
        status = "ALERT_INALCANZABLE"
    elif pass_pct > spec.max_pass_pct:
        status = "ALERT_DEMASIADO_PERMISIVO"
    else:
        status = "OK"

    return {
        "spec": spec,
        "status": status,
        "n_eval": n_eval,
        "current_value": current_value,
        "pass_pct": pass_pct,
        "p50": float(np.percentile(metric, 50)) if n_eval else None,
        "p90": float(np.percentile(metric, 90)) if n_eval else None,
        "p99": float(np.percentile(metric, 99)) if n_eval else None,
        "metric_max": float(metric.max()) if n_eval else None,
    }


def _print_report(reports: list, deploy: Path, quiet: bool) -> int:
    """Imprime el reporte y devuelve el exit code (0 ok, 1 alerta)."""
    n_alerts = sum(1 for r in reports if r["status"].startswith("ALERT"))
    n_errors = sum(1 for r in reports if r["status"].startswith("ERROR"))

    if quiet and n_alerts == 0 and n_errors == 0:
        return 0

    print(f"\n{'='*78}")
    print(f"  VALIDACIÓN DE UMBRALES DEPENDIENTES DE CAL_PROBS")
    print(f"  Deploy: {deploy}")
    print(f"{'='*78}\n")

    for r in reports:
        s = r["spec"]
        status = r["status"]
        emoji = "✅" if status == "OK" else ("❌" if status.startswith("ALERT") else "⚠️ ")

        print(f"{emoji} {s.name}")
        print(f"   Source:      {s.source_file}::{s.source_attr}")
        print(f"   Descripción: {s.description}")

        if status == "ERROR_NO_VALUE":
            print(f"   Status:      NO SE PUDO LEER EL VALOR DEL ARCHIVO")
            print()
            continue

        # Tabla de métricas
        print(f"   Valor actual: {r['current_value']:.4f}")
        print(f"   n eval:       {r['n_eval']}")
        if r['n_eval'] > 0:
            print(f"   Métrica P50/P90/P99/max: {r['p50']:.4f} / {r['p90']:.4f} / {r['p99']:.4f} / {r['metric_max']:.4f}")
        print(f"   Pasa el umbral:  {r['pass_pct']:.2f}%  (rango razonable: [{s.min_pass_pct:.1f}%, {s.max_pass_pct:.1f}%])")
        print(f"   Status:          {status}")

        if status == "ALERT_INALCANZABLE":
            sugerido = r['p90'] if r['p90'] is not None else r['current_value']
            print(f"   💡 Sugerencia:   bajar a ~{sugerido:.4f} (P90 empírico) para recuperar ~10% paso")
        elif status == "ALERT_DEMASIADO_PERMISIVO":
            sugerido = r['p50'] if r['p50'] is not None else r['current_value']
            print(f"   💡 Sugerencia:   subir a ~{sugerido:.4f} (P50 empírico) para reducir a ~50% paso")
        print()

    # Resumen
    print(f"{'─'*78}")
    n_ok = len(reports) - n_alerts - n_errors
    print(f"  TOTAL: {len(reports)} umbrales | ✅ {n_ok} OK | ❌ {n_alerts} alertas | ⚠️  {n_errors} errores")
    print(f"{'─'*78}\n")

    return 1 if (n_alerts > 0 or n_errors > 0) else 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--deploy",
        default="artifacts/202500/oof/deploy_validation_combined_seed47",
        help="Path al deploy (con los .joblib y data/) — default: deploy_validation_combined_seed47",
    )
    parser.add_argument("--release", default="202500", help="Release tag (default: 202500)")
    parser.add_argument("--quiet", action="store_true", help="Solo emitir output si hay alertas/errores")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent
    deploy = Path(args.deploy)
    if not deploy.is_absolute():
        deploy = project_root / deploy

    try:
        df = _load_oof_data(deploy, args.release)
    except FileNotFoundError as e:
        print(f"❌ {e}", file=sys.stderr)
        return 2

    reports = [
        _evaluate_threshold(spec, df, current_value=None, project_root=project_root)
        for spec in THRESHOLD_SPECS
    ]

    return _print_report(reports, deploy, args.quiet)


if __name__ == "__main__":
    sys.exit(main())
