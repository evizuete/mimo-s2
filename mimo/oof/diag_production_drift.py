"""
diag_production_drift.py
═══════════════════════════════════════════════════════════════════════════
Detector de drift en producción para el modelo GBM LONG-only.

Lee los logs JSONL de inferencia generados por GBMTradingSimulator y
compara contra el baseline persistido en production/<release>/ para
detectar:

  · Drift en la distribución de probas raw del booster
  · Caída drástica de sig_rate (modelo emite menos señales)
  · Cambio en la composición de razones de rechazo (no signal reasons)
  · Cambio en la distribución de estados de mercado vistos

Veredicto automatico:
  ✅  OK             — métricas dentro de tolerancias
  ⚠️   WARNING       — drift moderado, monitoreo cercano
  🔴  RETUNE_NOW    — drift fuerte, relanzar 008_deploy_long_only_gbm.sh

USO:
  python -m mimo.oof.diag_production_drift \\
    --release 202603_GBM \\
    --lookback-days 30 \\
    --out-json production/202603_GBM/drift_report.json
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _fnum(v, d=float("nan")) -> float:
    if v is None: return d
    try:
        f = float(v)
        return f if math.isfinite(f) else d
    except Exception:
        return d


def _load_baseline(production_dir: Path) -> Dict[str, Any]:
    """Lee threshold.json + metadata.json para tener las stats del scan
    que sirvieron para fijar el threshold de producción."""
    thr_path = production_dir / "production_threshold.json"
    meta_path = production_dir / "production_metadata.json"
    if not thr_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Baseline no encontrado en {production_dir}. "
            f"¿Has ejecutado 008_deploy_long_only_gbm.sh?"
        )
    with open(thr_path) as fh: thr = json.load(fh)
    with open(meta_path) as fh: meta = json.load(fh)
    return {
        "threshold":            _fnum(thr.get("threshold")),
        "scan_ev_net":          _fnum(thr.get("scan_stats", {}).get("ev_net")),
        "scan_n_signals":       int(thr.get("scan_stats", {}).get("n_signals", 0) or 0),
        "scan_prec_TP":         _fnum(thr.get("scan_stats", {}).get("prec_TP")),
        "scan_window":          thr.get("scan_window"),
        "scan_months":          int(thr.get("scan_months", 0) or 0),
        "release":              meta.get("release"),
        "trial":                meta.get("trial"),
        "as_of":                meta.get("as_of"),
        "refit_months":         int(meta.get("refit_months", 0) or 0),
        "n_train_rows":         int(meta.get("n_train_rows", 0) or 0),
        "n_scan_rows":          int(meta.get("n_scan_rows", 0) or 0),
        "tp_mult":              _fnum(meta.get("tp_mult")),
        "sl_mult":              _fnum(meta.get("sl_mult")),
    }


def _load_inference_logs(log_dir: Path, lookback_days: int) -> List[Dict[str, Any]]:
    """Lee gbm_inference_*.jsonl de los últimos N días. Devuelve lista de
    entries (excluye los headers GBM_LOGGER_STARTED)."""
    cutoff = datetime.utcnow() - timedelta(days=int(lookback_days))
    entries: List[Dict[str, Any]] = []
    for path in sorted(glob.glob(str(log_dir / "gbm_inference_*.jsonl"))):
        # Detectar fecha del archivo (sufijo YYYYMMDD)
        try:
            base = os.path.basename(path)
            date_str = base.replace("gbm_inference_", "").split(".")[0][:8]
            file_date = datetime.strptime(date_str, "%Y%m%d")
            if file_date < cutoff:
                continue
        except Exception:
            pass  # si no parsea fecha, incluir igual
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line: continue
                try:
                    e = json.loads(line)
                    if e.get("event") == "GBM_LOGGER_STARTED": continue
                    entries.append(e)
                except Exception:
                    continue
    return entries


def _stats_from_entries(entries: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Calcula estadísticas agregadas del periodo de log."""
    if not entries:
        return {"n_entries": 0}

    n_total = len(entries)
    n_long  = sum(1 for e in entries if e.get("decision") == "LONG")
    n_none  = sum(1 for e in entries if e.get("decision") == "NO_SIGNAL")

    probs = [_fnum(e.get("model", {}).get("proba_long_raw")) for e in entries]
    probs = [p for p in probs if math.isfinite(p)]
    thrs  = [_fnum(e.get("model", {}).get("threshold")) for e in entries]
    thrs  = [t for t in thrs if math.isfinite(t)]
    crosses = sum(1 for e in entries
                  if e.get("model", {}).get("crosses_thr") is True)

    # Razones de rechazo
    reasons: Dict[str, int] = {}
    for e in entries:
        if e.get("decision") == "NO_SIGNAL":
            r = str(e.get("reason") or "UNKNOWN")
            reasons[r] = reasons.get(r, 0) + 1

    # Estados de mercado vistos
    states: Dict[str, int] = {}
    for e in entries:
        s = str(e.get("market", {}).get("state") or "")
        if s:
            states[s] = states.get(s, 0) + 1

    out = {
        "n_entries":           n_total,
        "n_long":              n_long,
        "n_no_signal":         n_none,
        "long_rate":           float(n_long / n_total) if n_total > 0 else 0.0,
        "crosses_thr_rate":    float(crosses / n_total) if n_total > 0 else 0.0,
        "reasons":             dict(sorted(reasons.items(),
                                           key=lambda kv: -kv[1])),
        "states":              dict(sorted(states.items(),
                                           key=lambda kv: -kv[1])),
    }
    if probs:
        arr = np.asarray(probs, dtype=np.float64)
        out["proba_long_raw"] = {
            "n":     int(len(arr)),
            "mean":  float(arr.mean()),
            "std":   float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "p10":   float(np.percentile(arr, 10)),
            "p25":   float(np.percentile(arr, 25)),
            "p50":   float(np.percentile(arr, 50)),
            "p75":   float(np.percentile(arr, 75)),
            "p90":   float(np.percentile(arr, 90)),
            "p99":   float(np.percentile(arr, 99)),
            "max":   float(arr.max()),
        }
    if thrs:
        out["threshold_observed"] = {
            "min": float(min(thrs)), "max": float(max(thrs)),
            "n_unique": len(set(thrs)),
        }
    return out


def _diagnose_drift(baseline: Dict[str, Any], obs: Dict[str, Any],
                    lookback_days: int) -> Tuple[str, List[str], List[str]]:
    """Devuelve (verdict, issues, ok_signals).
    verdict in {OK, WARNING, RETUNE_NOW}."""
    issues: List[str] = []
    ok_signals: List[str] = []

    if obs.get("n_entries", 0) == 0:
        return "❓ NO_DATA", ["Sin entradas en logs"], []

    # 1) Crosses threshold rate vs expected
    # Expected: scan_n_signals / n_scan_rows (durante el scan period)
    expected_long_rate = (
        baseline["scan_n_signals"] / baseline["n_scan_rows"]
        if baseline["n_scan_rows"] > 0 else 0.0
    )
    observed_long_rate = obs.get("long_rate", 0.0)
    if expected_long_rate > 0:
        ratio = observed_long_rate / expected_long_rate
        if ratio < 0.30:
            issues.append(f"🔴 long_rate caída SEVERA: obs={observed_long_rate:.5f} "
                          f"esperado={expected_long_rate:.5f} (ratio={ratio:.2f})")
        elif ratio < 0.60:
            issues.append(f"⚠️  long_rate caída moderada: obs={observed_long_rate:.5f} "
                          f"esperado={expected_long_rate:.5f} (ratio={ratio:.2f})")
        elif ratio > 2.5:
            issues.append(f"⚠️  long_rate aumentó mucho: obs={observed_long_rate:.5f} "
                          f"esperado={expected_long_rate:.5f} (ratio={ratio:.2f}) — "
                          f"posible threshold demasiado bajo")
        else:
            ok_signals.append(f"✓ long_rate consistente (ratio={ratio:.2f})")

    # 2) Distribution shift en proba_long_raw (p90 baja → modelo menos confiado)
    if "proba_long_raw" in obs:
        p_stats = obs["proba_long_raw"]
        thr = baseline["threshold"]
        if math.isfinite(thr):
            # ¿El p99 alcanza el threshold? Si no, el modelo nunca emitirá
            if p_stats["p99"] < thr * 0.95:
                issues.append(f"🔴 p99({p_stats['p99']:.4f}) < threshold({thr:.4f}) — "
                              f"modelo no alcanza el thr en >99% de las barras")
            elif p_stats["p90"] < thr * 0.85:
                issues.append(f"⚠️  p90({p_stats['p90']:.4f}) muy por debajo de "
                              f"threshold({thr:.4f}) — pocas señales emitibles")
            else:
                ok_signals.append(f"✓ Distribución de probas saludable "
                                  f"(p90={p_stats['p90']:.4f}, thr={thr:.4f})")

    # 3) BELOW_THRESHOLD dominancia (esperado pero diagnóstico de drift si crece)
    total_no_signal = obs.get("n_no_signal", 0)
    if total_no_signal > 0:
        below = obs.get("reasons", {}).get("BELOW_THRESHOLD", 0)
        below_pct = below / total_no_signal
        # En condiciones normales BELOW_THRESHOLD debería ser ~99% de los NO_SIGNAL
        if below_pct < 0.85:
            others = [(r, n) for r, n in obs["reasons"].items()
                      if r != "BELOW_THRESHOLD"]
            issues.append(f"⚠️  BELOW_THRESHOLD solo {below_pct*100:.0f}% de los NO_SIGNAL. "
                          f"Otros motivos importantes: {others[:3]}")
        else:
            ok_signals.append(f"✓ BELOW_THRESHOLD domina los rechazos ({below_pct*100:.0f}%)")

    # 4) Estados raros (>5% de distribución que NO estaban en scan period)
    # (No tenemos los estados del scan period en baseline, así que solo
    #  reportamos qué se ve más)
    if obs.get("states"):
        top_state = next(iter(obs["states"]))
        top_share = obs["states"][top_state] / obs["n_entries"]
        if top_share > 0.7:
            issues.append(f"⚠️  Un estado domina ({top_state}={top_share*100:.0f}%) — "
                          f"mercado en régimen muy específico, posible bias")

    # Veredicto: prioridad RED > WARNING > OK
    has_red = any("🔴" in s for s in issues)
    has_warn = any("⚠️" in s for s in issues)
    if has_red:
        return "🔴 RETUNE_NOW", issues, ok_signals
    if has_warn:
        return "⚠️  WARNING", issues, ok_signals
    return "✅ OK", issues, ok_signals


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--release", required=True,
                    help="Release identifier, ej. 202603_GBM. Usado para "
                         "encontrar production/<release>/ y logs/.")
    ap.add_argument("--production-base", default="production",
                    help="Base dir bajo el cual está <release>/.")
    ap.add_argument("--lookback-days", type=int, default=30,
                    help="Cuántos días retroceder en los logs.")
    ap.add_argument("--log-dir", default=None,
                    help="Override del log dir (default: "
                         "production/<release>/logs/).")
    ap.add_argument("--out-json", default=None)
    args = ap.parse_args()

    prod_dir = Path(args.production_base) / args.release
    log_dir = Path(args.log_dir) if args.log_dir else (prod_dir / "logs")

    print(f"📂 Production dir: {prod_dir}")
    print(f"📂 Log dir       : {log_dir}")
    print(f"📅 Lookback days : {args.lookback_days}")

    # 1) Baseline
    baseline = _load_baseline(prod_dir)
    print(f"\n📊 BASELINE (deploy {baseline['as_of']}):")
    print(f"   release={baseline['release']}  trial=#{baseline['trial']}")
    print(f"   threshold={baseline['threshold']:.4f}")
    print(f"   scan window={baseline['scan_window']}  ({baseline['scan_months']}m)")
    print(f"   scan stats: ev_net={baseline['scan_ev_net']:+.4f}R  "
          f"sig={baseline['scan_n_signals']}  prec={baseline['scan_prec_TP']:.3f}")
    expected_sig_rate = (
        baseline["scan_n_signals"] / baseline["n_scan_rows"]
        if baseline["n_scan_rows"] > 0 else 0.0
    )
    print(f"   expected long_rate ≈ {expected_sig_rate:.5f} "
          f"({baseline['scan_n_signals']}/{baseline['n_scan_rows']})")

    # 2) Logs observados
    entries = _load_inference_logs(log_dir, args.lookback_days)
    print(f"\n📥 Logs cargados: {len(entries)} entries de los últimos "
          f"{args.lookback_days} días")

    obs = _stats_from_entries(entries)
    if obs.get("n_entries", 0) == 0:
        print("\n❌ Sin datos para analizar.")
        return

    # 3) Stats observados
    print(f"\n📈 OBSERVADO:")
    print(f"   n_entries          = {obs['n_entries']}")
    print(f"   n_long             = {obs['n_long']}")
    print(f"   n_no_signal        = {obs['n_no_signal']}")
    print(f"   long_rate          = {obs['long_rate']:.5f}")
    print(f"   crosses_thr_rate   = {obs['crosses_thr_rate']:.5f}")
    if "proba_long_raw" in obs:
        ps = obs["proba_long_raw"]
        print(f"   proba_long_raw     = mean={ps['mean']:.4f}  std={ps['std']:.4f}")
        print(f"                        p10={ps['p10']:.4f}  p50={ps['p50']:.4f}  "
              f"p90={ps['p90']:.4f}  p99={ps['p99']:.4f}  max={ps['max']:.4f}")

    print(f"\n   Reasons (top):")
    for r, n in list(obs["reasons"].items())[:5]:
        print(f"     {r:<25s}  {n}  ({n/obs['n_entries']*100:.1f}%)")

    print(f"\n   States (top):")
    for s, n in list(obs["states"].items())[:5]:
        print(f"     {s:<25s}  {n}  ({n/obs['n_entries']*100:.1f}%)")

    # 4) Diagnóstico
    verdict, issues, ok_signals = _diagnose_drift(baseline, obs, args.lookback_days)
    print("\n" + "═" * 70)
    print(f"  VEREDICTO: {verdict}")
    print("═" * 70)
    if issues:
        print("  Issues detectados:")
        for it in issues: print(f"    {it}")
    if ok_signals:
        print("  OK:")
        for s in ok_signals: print(f"    {s}")

    # 5) Persistir
    report = {
        "release":       args.release,
        "ts":            datetime.utcnow().isoformat() + "Z",
        "lookback_days": int(args.lookback_days),
        "baseline":      baseline,
        "observed":      obs,
        "verdict":       verdict,
        "issues":        issues,
        "ok_signals":    ok_signals,
    }
    out_json = args.out_json or str(prod_dir / "drift_report.json")
    with open(out_json, "w") as fh:
        json.dump(report, fh, indent=2, default=str)
    print(f"\n📁 Reporte: {out_json}")


if __name__ == "__main__":
    main()
