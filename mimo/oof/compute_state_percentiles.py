#!/usr/bin/env python3
"""
compute_state_percentiles.py

Para cada estado de régimen calcula a qué percentil de la distribución OOF/tail
corresponde el `selected_threshold` actual. Sirve para construir un fichero
`config/decision_policies_config_<release>.py` cuyas gates por régimen reproduzcan
el threshold global elegido por `select_thresholds_from_tail`.

Por qué:
  El `decision_engine` consume `gate_by_action_and_state[action][state] = N`
  (un percentil entero, ej. 90/95/99). Después resuelve N → threshold raw
  contra `percentiles_<release>_<side>.json[state]['percentiles']['p{N}']`.

  Tras `select_thresholds_from_tail` el `_meta.selected_threshold` es el thr
  óptimo EV-net global, pero la distribución de probabilidades calibradas
  varía por régimen → el mismo thr raw puede caer en p70 en RANGE y p92 en
  TREND_UP, por ejemplo.

Lo que hace:
  1. Lee `percentiles_<release>_<side>.json` en --deploy-dir.
  2. Para cada estado interpola el percentil al que corresponde
     `_meta.selected_threshold` dentro de su distribución.
  3. Redondea al percentil disponible más cercano (50, 60, 70, 75, ..., 99).
  4. Imprime una tabla y opcionalmente genera un stub Python listo para
     pegar en `config/decision_policies_config_<release>.py`.

Uso:
  python -m mimo.oof.compute_state_percentiles \
    --release 202500 \
    --deploy-dir ../../artifacts/202500/oof/deploy_full \
    --emit-config-stub
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple


AVAILABLE_PERCENTILES = (50, 60, 70, 75, 80, 85, 90, 95, 96, 97, 98, 99)


def _interpolate_percentile(
    selected: float,
    pct_pairs: List[Tuple[int, float]],
) -> float:
    """Dado el threshold seleccionado y una lista ordenada [(p_key, p_val), ...],
    devuelve el percentil (interpolado) al que corresponde `selected`."""
    if not pct_pairs:
        return float("nan")
    pct_pairs = sorted(pct_pairs)
    # Clamping por debajo y por encima de la distribución conocida
    if selected <= pct_pairs[0][1]:
        return float(pct_pairs[0][0])
    if selected >= pct_pairs[-1][1]:
        return float(pct_pairs[-1][0])
    # Interpolación lineal entre los dos percentiles que rodean `selected`
    for (k_lo, v_lo), (k_hi, v_hi) in zip(pct_pairs[:-1], pct_pairs[1:]):
        if v_lo <= selected <= v_hi:
            if v_hi - v_lo < 1e-12:
                return float(k_lo)
            frac = (selected - v_lo) / (v_hi - v_lo)
            return float(k_lo) + frac * (float(k_hi) - float(k_lo))
    return float(pct_pairs[-1][0])


def _round_to_available(p: float) -> int:
    """Redondea un percentil interpolado al disponible más cercano."""
    return int(min(AVAILABLE_PERCENTILES, key=lambda x: abs(x - p)))


def _state_pct_pairs(state_payload: Dict) -> List[Tuple[int, float]]:
    """Extrae [(50, p50), (60, p60), ..., (99, p99)] válidos del payload de un estado."""
    pcts = state_payload.get("percentiles", {}) or {}
    pairs: List[Tuple[int, float]] = []
    for k, v in pcts.items():
        if not k.startswith("p"):
            continue
        try:
            key_int = int(k[1:])
            val_float = float(v)
            pairs.append((key_int, val_float))
        except (ValueError, TypeError):
            continue
    return sorted(pairs)


def _is_no_trade(state_payload: Dict) -> bool:
    return bool(state_payload.get("no_trade", False))


def _state_n(state_payload: Dict) -> Optional[int]:
    n = state_payload.get("n")
    return int(n) if isinstance(n, (int, float)) else None


def _process_side(
    side: str,
    deploy_dir: Path,
    release: str,
) -> Optional[Dict]:
    pct_path = deploy_dir / f"percentiles_{release}_{side}.json"
    if not pct_path.exists():
        print(f"  ⚠️  no existe {pct_path}")
        return None

    payload = json.loads(pct_path.read_text(encoding="utf-8"))
    meta = payload.get("_meta", {})
    selected = meta.get("selected_threshold")
    if selected is None:
        print(f"  ⚠️  {pct_path} no tiene _meta.selected_threshold")
        return None
    selected = float(selected)

    states = sorted([k for k in payload.keys() if not k.startswith("_")])
    rows: List[Dict] = []

    for state in states + ["_global"]:
        if state not in payload:
            continue
        sp = payload[state] or {}
        pairs = _state_pct_pairs(sp)
        if not pairs:
            continue
        interp = _interpolate_percentile(selected, pairs)
        rounded = _round_to_available(interp)
        rows.append({
            "state": state,
            "n": _state_n(sp),
            "no_trade": _is_no_trade(sp),
            "p_min": pairs[0],
            "p_max": pairs[-1],
            "interp": interp,
            "rounded": rounded,
        })

    return {
        "side": side,
        "selected_threshold": selected,
        "selected_threshold_old_f1": meta.get("selected_threshold_old_f1"),
        "threshold_source": meta.get("threshold_source"),
        "states": rows,
    }


def _print_side_table(info: Dict) -> None:
    print("\n" + "═" * 96)
    print(f"  SIDE = {info['side'].upper()}  selected_threshold = {info['selected_threshold']:.4f}"
          f"  (was F1 = {info.get('selected_threshold_old_f1', 'n/a')})")
    print("═" * 96)
    print(f"  {'state':<22s}  {'n':>6s}  {'p_min':>14s}  {'p_max':>14s}  "
          f"{'interp':>8s}  {'gate (avail)':>14s}  flag")
    print(f"  {'-'*22}  {'-'*6}  {'-'*14}  {'-'*14}  {'-'*8}  {'-'*14}  {'-'*8}")
    for r in info["states"]:
        nstr = str(r["n"]) if r["n"] is not None else "n/a"
        flag = "no_trade" if r["no_trade"] else ""
        print(
            f"  {r['state']:<22s}  {nstr:>6s}  "
            f"p{r['p_min'][0]:>2d}={r['p_min'][1]:.4f}  "
            f"p{r['p_max'][0]:>2d}={r['p_max'][1]:.4f}  "
            f"p{r['interp']:>5.1f}  "
            f"p{r['rounded']:>3d}          {flag}"
        )


# ─────────────────────────────────────────────────────────────────────────────
# Mapa estado canónico (en lowercase como decision_engine los espera) ──────────

_STATE_MAP_LOWER = {
    "BREAKOUT_WAIT_DOWN": "breakout_wait_down",
    "BREAKOUT_WAIT_UP":   "breakout_wait_up",
    "LOW_VOL":            "low_vol",
    "RANGE":              "range",
    "TRANSITION_DOWN":    "transition_down",
    "TRANSITION_UP":      "transition_up",
    "TREND_DOWN":         "trend_down",
    "TREND_UP":           "trend_up",
    "VOLATILE":           "volatile",
    "_global":            "_global",
}


def _emit_config_stub(long_info: Optional[Dict], short_info: Optional[Dict],
                      release: str) -> str:
    """Genera un stub Python con el dict gate_by_action_and_state listo para pegar."""

    def _block(info: Optional[Dict], side: str) -> str:
        if not info:
            return f'        "{side}": {{}},'
        lines = [f'        "{side}": {{']
        # Orden canónico
        canonical_order = [
            "trend_up", "trend_down",
            "transition_up", "transition_down",
            "range",
            "breakout_wait_up", "breakout_wait_down",
            "volatile", "low_vol",
            "_global",
        ]
        by_state: Dict[str, int] = {}
        for r in info["states"]:
            key = _STATE_MAP_LOWER.get(r["state"], r["state"].lower())
            by_state[key] = r["rounded"]

        for key in canonical_order:
            val = by_state.get(key)
            if val is None:
                # Fallback razonable
                if key in ("volatile", "low_vol"):
                    val = 99
                else:
                    val = by_state.get("_global", 95)
            comment = ""
            if key in ("volatile", "low_vol"):
                comment = "  # bloqueado por score_cap"
            lines.append(f'            "{key}": {val:>2d},{comment}')
        lines.append("        },")
        return "\n".join(lines)

    out = []
    out.append(f"# config/decision_policies_config_{release}.py")
    out.append(f"# Auto-generado por compute_state_percentiles.py")
    out.append(f"# selected_threshold LONG  = {long_info['selected_threshold']:.4f}"
               if long_info else "# (LONG no disponible)")
    out.append(f"# selected_threshold SHORT = {short_info['selected_threshold']:.4f}"
               if short_info else "# (SHORT no disponible)")
    out.append("")
    out.append("gate_by_action_and_state = {")
    out.append('    "training": {')
    # En training usamos los mismos percentiles -5 (más permisivo)
    if long_info:
        for r in long_info["states"]:
            r["rounded_training"] = max(50, r["rounded"] - 5)
    if short_info:
        for r in short_info["states"]:
            r["rounded_training"] = max(50, r["rounded"] - 5)

    def _block_training(info: Optional[Dict], side: str) -> str:
        if not info:
            return f'        "{side}": {{}},'
        lines = [f'        "{side}": {{']
        canonical_order = [
            "trend_up", "trend_down",
            "transition_up", "transition_down",
            "range",
            "breakout_wait_up", "breakout_wait_down",
            "volatile", "low_vol",
            "_global",
        ]
        by_state: Dict[str, int] = {}
        for r in info["states"]:
            key = _STATE_MAP_LOWER.get(r["state"], r["state"].lower())
            by_state[key] = r.get("rounded_training", r["rounded"])
        for key in canonical_order:
            val = by_state.get(key)
            if val is None:
                if key in ("volatile", "low_vol"):
                    val = 99
                else:
                    val = by_state.get("_global", 90)
            lines.append(f'            "{key}": {val:>2d},')
        lines.append("        },")
        return "\n".join(lines)

    out.append(_block_training(long_info, "long"))
    out.append(_block_training(short_info, "short"))
    out.append("    },")
    out.append('    "production": {')
    out.append(_block(long_info, "long"))
    out.append(_block(short_info, "short"))
    out.append("    },")
    out.append("}")
    out.append("")
    out.append("# score_cap y risk_mult: punto de partida copiado del 200383.")
    out.append("# Ajusta según riesgo aceptable per régimen.")
    out.append("score_cap_by_state = {")
    out.append('    "training": {')
    out.append('        "trend_up": 1.50, "trend_down": 1.50,')
    out.append('        "transition": 1.25, "range": 1.00, "breakout": 1.00,')
    out.append('        "volatile": 0.50, "low_vol": 0.25,')
    out.append("    },")
    out.append('    "production": {')
    out.append('        "trend_up": 1.50, "trend_down": 1.50,')
    out.append('        "transition": 1.05, "range": 0.75, "breakout": 0.90,')
    out.append('        "volatile": 0.00, "low_vol": 0.00,')
    out.append("    },")
    out.append("}")
    out.append("")
    out.append("risk_mult_by_state = {")
    out.append('    "training": {')
    out.append('        "trend_up": 1.00, "trend_down": 1.00,')
    out.append('        "transition": 0.75, "range": 0.50, "breakout": 0.50,')
    out.append('        "volatile": 0.00, "low_vol": 0.00,')
    out.append("    },")
    out.append('    "production": {')
    out.append('        "trend_up": 1.00, "trend_down": 1.00,')
    out.append('        "transition": 0.55, "range": 0.25, "breakout": 0.40,')
    out.append('        "volatile": 0.00, "low_vol": 0.00,')
    out.append("    },")
    out.append("}")
    return "\n".join(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", required=True)
    ap.add_argument("--deploy-dir", required=True,
                    help="Dir donde están los percentiles_<release>_<side>.json")
    ap.add_argument("--emit-config-stub", action="store_true",
                    help="Imprime un stub Python listo para pegar en "
                         "config/decision_policies_config_<release>.py")
    ap.add_argument("--out-stub", default=None,
                    help="Si se pasa, persiste el stub a este path "
                         "(opcionalmente, en config/decision_policies_config_<release>.py).")
    args = ap.parse_args()

    deploy_dir = Path(args.deploy_dir)
    if not deploy_dir.exists():
        raise SystemExit(f"❌ --deploy-dir no existe: {deploy_dir}")

    long_info = _process_side("long", deploy_dir, args.release)
    short_info = _process_side("short", deploy_dir, args.release)

    if long_info:
        _print_side_table(long_info)
    if short_info:
        _print_side_table(short_info)

    print("\n" + "═" * 96)
    print("  Cómo leer la tabla:")
    print("═" * 96)
    print("    interp        = percentil interpolado al que corresponde "
          "selected_threshold dentro de la distribución del estado.")
    print("    gate (avail)  = percentil más cercano dentro de los disponibles "
          f"({', '.join('p'+str(p) for p in AVAILABLE_PERCENTILES)}).")
    print("    flag=no_trade = el estado tiene no_trade=True en percentiles json "
          "(no se opera).")
    print("    Estados con n bajo (<800) suelen caer al _global; cuidado al usar")
    print("    sus interp como gate per-régimen (poca base estadística).")

    if args.emit_config_stub or args.out_stub:
        stub = _emit_config_stub(long_info, short_info, args.release)
        print("\n" + "═" * 96)
        print(f"  STUB PYTHON  (pegar en config/decision_policies_config_{args.release}.py)")
        print("═" * 96)
        print(stub)
        if args.out_stub:
            out = Path(args.out_stub)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(stub, encoding="utf-8")
            print(f"\n📁 Stub persistido en: {out}")


if __name__ == "__main__":
    main()
