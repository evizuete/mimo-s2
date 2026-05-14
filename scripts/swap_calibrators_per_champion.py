#!/usr/bin/env python3
"""
swap_calibrators_per_champion.py — Reemplaza in-place el/los calibrator(s)
de un deploy_dir aplicando el campeón recomendado por `simulate_calibrators.py`.

Nivel 1 (homogéneo por side):
  - Si todos los states de un side comparten el mismo `champion_method`,
    se sustituye el `calibrator_long.pkl` / `calibrator_short.pkl` por uno
    ÚNICO entrenado con ese método sobre los datos de holdout (filtrado por
    tail si aplica).

Nivel 2 (heterogéneo por state):
  - Si distintos states dentro de un mismo side eligen distintos métodos,
    se crea un calibrador "stateful" que enruta por state. Se serializa con
    pickle como objeto `StatefulCalibrator` con interfaz `.predict(p_raw, state)`.

Backups:
  - El fichero original se guarda como `<nombre>.before_swap_<TS>`.
  - Se escribe `<deploy_dir>/calibrator_swap_meta.json` con el detalle.

Uso:
  python3 scripts/swap_calibrators_per_champion.py \\
    --release 202500 \\
    --specialist-tag rw_both_Lvol_boost_td_down_h3_Svol_boost_h3 \\
    --seed 47 \\
    --champion-config reports/champion_config.json \\
    --deploy-dir artifacts/202500/oof/deploy_validation_combined_seed47
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import pickle
import shutil
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


# ───────────────────────────── Método → spec ─────────────────────────────

METHOD_SPECS: Dict[str, Dict] = {
    "iso_21d":           {"kind": "isotonic", "tail_days": 21,   "per_state": False},
    "iso_45d":           {"kind": "isotonic", "tail_days": 45,   "per_state": False},
    "iso_full":          {"kind": "isotonic", "tail_days": None, "per_state": False},
    "beta_21d":          {"kind": "beta",     "tail_days": 21,   "per_state": False},
    "beta_45d":          {"kind": "beta",     "tail_days": 45,   "per_state": False},
    "per_state_iso_45d": {"kind": "isotonic", "tail_days": 45,   "per_state": True},
}


# ───────────────────────────── Calibradores ─────────────────────────────

class _SimpleIsotonic:
    """Wrapper isotónico con interfaz `.predict(p_raw)`."""
    def __init__(self):
        self.model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, p_raw, y):
        self.model.fit(p_raw, y)
        return self

    def predict(self, p_raw):
        return self.model.predict(p_raw)


class _SimpleBeta:
    """Wrapper Beta calibration con interfaz `.predict(p_raw)`."""
    def __init__(self):
        try:
            from betacal import BetaCalibration
        except ImportError as e:
            raise SystemExit("❌ `betacal` no instalado: pip install betacal") from e
        self.model = BetaCalibration(parameters="abm")

    def fit(self, p_raw, y):
        self.model.fit(p_raw, y)
        return self

    def predict(self, p_raw):
        return self.model.predict(p_raw)


class StatefulCalibrator:
    """Enruta por state. Mantiene `predict(p_raw)` para retro-compat (usa fallback)."""

    def __init__(self, per_state: Dict[str, object], fallback, default_method: str):
        self.per_state = per_state
        self.fallback = fallback
        self.default_method = default_method

    # API state-aware
    def predict_with_state(self, p_raw: np.ndarray, state: np.ndarray) -> np.ndarray:
        p_raw = np.asarray(p_raw, dtype=float)
        state = np.asarray(state)
        out = np.empty_like(p_raw)
        seen = set(np.unique(state).tolist())
        for st in seen:
            mask = state == st
            cal = self.per_state.get(str(st), self.fallback)
            out[mask] = cal.predict(p_raw[mask])
        return out

    # API legacy (sin state): aplica fallback. Útil si algún consumidor llama .predict
    def predict(self, p_raw: np.ndarray) -> np.ndarray:
        return self.fallback.predict(np.asarray(p_raw, dtype=float))


# ───────────────────────────── Loaders ─────────────────────────────

def find_holdout_predictions(release: str, tag: str, side: str, seed: int) -> Path:
    bases = [
        Path(f"artifacts/{release}/oof/{tag}_{side}_specialist_seed{seed}"),
        Path(f"artifacts/{release}/oof/{tag}_{side}_specialist_seed{seed}_cutoff_mar31"),
    ]
    target = f"holdout_predictions_{release}_{side}.parquet"
    for base in bases:
        if not base.exists():
            continue
        for sub in (base, base / "data"):
            p = sub / target
            if p.exists():
                return p
    raise SystemExit(f"❌ No encuentro {target} bajo {bases}")


def _filter_by_tail(df: pd.DataFrame, tail_days: Optional[int]) -> pd.DataFrame:
    if tail_days is None:
        return df
    t_max = pd.to_datetime(df["time"]).max()
    cutoff = t_max - pd.Timedelta(days=int(tail_days))
    return df[pd.to_datetime(df["time"]) >= cutoff].copy()


def fit_calibrator(method: str, hp: pd.DataFrame, raw_col: str, y_col: str,
                   state_col: Optional[str] = None,
                   min_n_per_state: int = 100) -> object:
    spec = METHOD_SPECS.get(method)
    if spec is None:
        raise ValueError(f"método desconocido: {method}")

    df = _filter_by_tail(hp, spec["tail_days"])
    p_raw = df[raw_col].to_numpy()
    y = df[y_col].to_numpy()

    if not spec["per_state"]:
        return _SimpleIsotonic().fit(p_raw, y) if spec["kind"] == "isotonic" \
               else _SimpleBeta().fit(p_raw, y)

    if state_col is None or state_col not in df.columns:
        raise SystemExit(f"❌ {method} requiere columna state en holdout_predictions")
    states = df[state_col].astype(str).to_numpy()
    fallback = _SimpleIsotonic().fit(p_raw, y) if spec["kind"] == "isotonic" \
               else _SimpleBeta().fit(p_raw, y)
    per_state: Dict[str, object] = {}
    for st in pd.Series(states).unique():
        mask = states == st
        if int(mask.sum()) < min_n_per_state:
            continue
        try:
            cal = (_SimpleIsotonic() if spec["kind"] == "isotonic" else _SimpleBeta())
            cal.fit(p_raw[mask], y[mask])
            per_state[str(st)] = cal
        except Exception as e:
            print(f"     ⚠️  fit per-state {st} fallo: {e}")
    return StatefulCalibrator(per_state=per_state, fallback=fallback, default_method=method)


def fit_per_state_with_mixed_methods(
    hp: pd.DataFrame, state_to_method: Dict[str, str],
    raw_col: str, y_col: str, state_col: str = "state",
    min_n_per_state: int = 30,
) -> StatefulCalibrator:
    """Cuando un side tiene heterogeneidad: entrena un calibrador por state
    con SU método campeón. El fallback usa el método más frecuente."""

    # Fallback = método mayoritario, sobre TODO el tail correspondiente
    from collections import Counter
    counts = Counter(state_to_method.values())
    fallback_method = counts.most_common(1)[0][0]
    fb_spec = METHOD_SPECS[fallback_method]
    df_fb = _filter_by_tail(hp, fb_spec["tail_days"])
    p_fb, y_fb = df_fb[raw_col].to_numpy(), df_fb[y_col].to_numpy()
    fallback = (_SimpleIsotonic().fit(p_fb, y_fb) if fb_spec["kind"] == "isotonic"
                else _SimpleBeta().fit(p_fb, y_fb))

    per_state: Dict[str, object] = {}
    for st, method in state_to_method.items():
        spec = METHOD_SPECS[method]
        df_st = _filter_by_tail(hp, spec["tail_days"])
        mask = df_st[state_col].astype(str).to_numpy() == str(st)
        if int(mask.sum()) < min_n_per_state:
            print(f"     ⚠️  state={st} método={method}: n={int(mask.sum())} < {min_n_per_state}, usa fallback")
            continue
        p_raw = df_st[raw_col].to_numpy()[mask]
        y = df_st[y_col].to_numpy()[mask]
        try:
            cal = _SimpleIsotonic().fit(p_raw, y) if spec["kind"] == "isotonic" \
                  else _SimpleBeta().fit(p_raw, y)
            per_state[str(st)] = cal
        except Exception as e:
            print(f"     ⚠️  fit state={st} ({method}) fallo: {e}")
    return StatefulCalibrator(per_state=per_state, fallback=fallback,
                              default_method=fallback_method)


# ───────────────────────────── Swap ─────────────────────────────

def swap_side(side: str, side_cfg: Dict[str, str], release: str, tag: str, seed: int,
              deploy_dir: Path, ts: str, dry_run: bool = False) -> Dict:
    cal_path = deploy_dir / f"calibrator_{side}.pkl"
    if not cal_path.exists():
        # Fallback name pattern (specialists merged)
        alt = deploy_dir / f"calibrator_{side}_specialist.pkl"
        if alt.exists():
            cal_path = alt
        else:
            return {"side": side, "skipped": True, "reason": f"no existe {cal_path.name}"}

    hp_path = find_holdout_predictions(release, tag, side, seed)
    hp = pd.read_parquet(hp_path)
    hp["time"] = pd.to_datetime(hp["time"])
    raw_col = "y_pred_raw" if "y_pred_raw" in hp.columns else "oof_proba_raw"
    y_col = "y_true" if "y_true" in hp.columns else "signal"

    methods_in_side = set(side_cfg.values())
    homogeneous = len(methods_in_side) == 1

    if homogeneous:
        method = next(iter(methods_in_side))
        print(f"  ▶ {side}: HOMOGÉNEO → {method}")
        new_cal = fit_calibrator(method, hp, raw_col=raw_col, y_col=y_col, state_col="state")
        level = 1
        details = {"method": method, "states": list(side_cfg.keys())}
    else:
        print(f"  ▶ {side}: HETEROGÉNEO → {len(methods_in_side)} métodos distintos")
        for st, m in sorted(side_cfg.items()):
            print(f"       state={st:>10s} → {m}")
        new_cal = fit_per_state_with_mixed_methods(
            hp, side_cfg, raw_col=raw_col, y_col=y_col, state_col="state"
        )
        level = 2
        details = {
            "fallback_method": new_cal.default_method,
            "state_to_method": dict(side_cfg),
        }

    if dry_run:
        print(f"     🔍 DRY-RUN: no se escribe {cal_path.name}")
        return {"side": side, "skipped": False, "level": level, "details": details,
                "dry_run": True}

    # Backup + write
    backup = cal_path.with_suffix(cal_path.suffix + f".before_swap_{ts}")
    shutil.copy2(cal_path, backup)
    with cal_path.open("wb") as f:
        pickle.dump(new_cal, f)
    print(f"     💾 {cal_path.name} reemplazado  (backup: {backup.name})")
    return {"side": side, "skipped": False, "level": level, "details": details,
            "backup": backup.name, "calibrator_file": cal_path.name}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="202500")
    ap.add_argument("--specialist-tag", default="rw_both_Lvol_boost_td_down_h3_Svol_boost_h3")
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--champion-config", required=True,
                    help="JSON {side: {state: method}} producido por simulate_calibrators.py")
    ap.add_argument("--deploy-dir", required=True)
    ap.add_argument("--side", choices=["long", "short", "both"], default="both")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    cfg_path = Path(args.champion_config)
    if not cfg_path.exists():
        raise SystemExit(f"❌ No existe {cfg_path}")
    config = json.loads(cfg_path.read_text())

    deploy_dir = Path(args.deploy_dir)
    if not deploy_dir.exists():
        raise SystemExit(f"❌ No existe {deploy_dir}")

    sides = ["long", "short"] if args.side == "both" else [args.side]
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"{'═'*72}\n  SWAP CALIBRATORS — {deploy_dir}\n{'═'*72}")
    print(f"  Champion: {cfg_path}")
    print(f"  DRY-RUN: {args.dry_run}")
    print()

    results = []
    for side in sides:
        side_cfg = config.get(side)
        if not side_cfg:
            print(f"  ⚠️  {side}: sin entradas en champion-config, salto.")
            results.append({"side": side, "skipped": True, "reason": "no champion entries"})
            continue
        res = swap_side(
            side=side, side_cfg=side_cfg,
            release=args.release, tag=args.specialist_tag, seed=args.seed,
            deploy_dir=deploy_dir, ts=ts, dry_run=args.dry_run,
        )
        results.append(res)

    meta = {
        "timestamp": ts,
        "release": args.release,
        "specialist_tag": args.specialist_tag,
        "seed": args.seed,
        "champion_config_path": str(cfg_path),
        "deploy_dir": str(deploy_dir),
        "dry_run": args.dry_run,
        "results": results,
    }
    if not args.dry_run:
        meta_path = deploy_dir / "calibrator_swap_meta.json"
        meta_path.write_text(json.dumps(meta, indent=2, default=str))
        print(f"\n📁 Meta: {meta_path}")
    else:
        print(f"\n🔍 DRY-RUN — meta no escrito")
        print(json.dumps(meta, indent=2, default=str))

    print("\n✅ swap completado. Recuerda: re-correr select_thresholds_from_tail "
          "y compute_state_percentiles si quieres regenerar el operating point.")


if __name__ == "__main__":
    main()
