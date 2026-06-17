#!/usr/bin/env python3
"""
swap_calibrators_per_champion.py — Reemplaza los calibradores oof_calibrator_*
en un deploy combinado siguiendo la config campeón de simulate_calibrators.py.

Nivel 1 (mismo método para todos los estados de un side):
  - Fitea un único calibrador global con el método elegido
  - Drop-in replacement: compatible con predict_side_pack actual

Nivel 2 (métodos distintos por estado):
  - Construye un StatefulCalibrator con dispatch por estado
  - Hace backup del calibrador original (.joblib.before_swap)
  - REQUIERE modificar predict_side_pack para pasar `state` al .predict()
    (avisa por consola y en metadata)

Uso:
  python scripts/swap_calibrators_per_champion.py \\
    --release 202500 \\
    --specialist-tag rw_both_Lvol_boost_td_down_h3_Svol_boost_h3 \\
    --seed 47 \\
    --champion-config reports/champion_config.json \\
    --deploy-dir artifacts/202500/oof/deploy_PROD_combined_specialists_seed47
"""
from __future__ import annotations
import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression


METHOD_SPECS = {
    "iso_21d":           {"kind": "isotonic", "tail_days": 21,   "per_state": False},
    "iso_45d":           {"kind": "isotonic", "tail_days": 45,   "per_state": False},
    "iso_full":          {"kind": "isotonic", "tail_days": None, "per_state": False},
    "beta_21d":          {"kind": "beta",     "tail_days": 21,   "per_state": False},
    "beta_45d":          {"kind": "beta",     "tail_days": 45,   "per_state": False},
    "per_state_iso_45d": {"kind": "isotonic", "tail_days": 45,   "per_state": True},
}


def _make_calibrator(kind: str):
    if kind == "isotonic":
        return IsotonicRegression(out_of_bounds="clip")
    if kind == "beta":
        try:
            from betacal import BetaCalibration
        except ImportError:
            raise SystemExit("❌ `betacal` no instalado: pip install betacal")
        return BetaCalibration(parameters="abm")
    raise ValueError(f"Unknown calibrator kind: {kind}")


class StatefulCalibrator:
    """Dispatch por estado. predict() acepta state opcional para vectorización."""
    def __init__(self):
        self.per_state: Dict[str, Any] = {}
        self.fallback = None
        self.kind_per_state: Dict[str, str] = {}

    def predict(self, p_raw, state=None):
        p_raw = np.asarray(p_raw, dtype=float)
        if state is None:
            if self.fallback is None:
                raise RuntimeError("StatefulCalibrator sin fallback y sin state")
            return self.fallback.predict(p_raw)
        states = np.asarray(state)
        out = np.empty_like(p_raw, dtype=float)
        for st in np.unique(states):
            mask = states == st
            cal = self.per_state.get(str(st), self.fallback)
            out[mask] = cal.predict(p_raw[mask])
        return out


def find_holdout_predictions(release, tag, side, seed) -> Path:
    bases = [
        Path(f"../artifacts/{release}/oof/{tag}_{side}_specialist_seed{seed}"),
        Path(f"../artifacts/{release}/oof/{tag}_{side}_specialist_seed{seed}_cutoff_mar31"),
    ]
    target = f"holdout_predictions_{release}_{side}.parquet"
    for base in bases:
        if not base.exists():
            continue
        for sub in (base, base / "data"):
            p = sub / target
            if p.exists():
                return p
    raise SystemExit(f"❌ No encuentro {target}")


def _filter_by_tail(p_raw, y, state, hp_time, ref_time, tail_days):
    if tail_days is None:
        return p_raw, y, state
    cutoff = ref_time - pd.Timedelta(days=tail_days)
    mask = pd.to_datetime(hp_time) >= cutoff
    mask = mask.to_numpy() if hasattr(mask, "to_numpy") else np.asarray(mask)
    return (p_raw[mask], y[mask], state[mask] if state is not None else None)


def fit_calibrator(method: str, p_raw, y, state, hp_time, ref_time,
                   min_n_per_state: int = 100):
    spec = METHOD_SPECS[method]
    p_use, y_use, st_use = _filter_by_tail(p_raw, y, state, hp_time, ref_time,
                                            spec["tail_days"])
    if spec["per_state"]:
        sc = StatefulCalibrator()
        sc.fallback = _make_calibrator(spec["kind"]).fit(p_use, y_use)
        if st_use is None:
            return sc
        for st in np.unique(st_use):
            m = st_use == st
            if int(m.sum()) < min_n_per_state:
                continue
            sc.per_state[str(st)] = _make_calibrator(spec["kind"]).fit(p_use[m], y_use[m])
            sc.kind_per_state[str(st)] = spec["kind"]
        return sc
    return _make_calibrator(spec["kind"]).fit(p_use, y_use)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="202500")
    ap.add_argument("--specialist-tag", default="rw_both_Lvol_boost_td_down_h3_Svol_boost_h3")
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--champion-config", required=True)
    ap.add_argument("--deploy-dir", required=True)
    ap.add_argument("--min-n-per-state", type=int, default=100)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    deploy_dir = Path(args.deploy_dir)
    if not deploy_dir.exists():
        raise SystemExit(f"❌ deploy_dir no existe: {deploy_dir}")
    with open(args.champion_config) as f:
        champion = json.load(f)

    print(f"\n🔧 SWAP CALIBRADORES")
    print(f"   deploy_dir:   {deploy_dir}")
    print(f"   champion_cfg: {args.champion_config}")
    print(f"   dry-run:      {args.dry_run}\n")

    swap_meta = {
        "swapped_at": datetime.now(timezone.utc).isoformat(),
        "champion_config": champion,
        "sides": {},
    }

    for side, state_methods in champion.items():
        print(f"{'═'*72}\n  SIDE = {side.upper()}\n{'═'*72}")
        unique_methods = set(state_methods.values())
        is_nivel_2 = len(unique_methods) > 1

        hp_path = find_holdout_predictions(args.release, args.specialist_tag, side, args.seed)
        print(f"  📂 Historical: {hp_path.name}")
        hp = pd.read_parquet(hp_path)
        raw_col = "y_pred_raw" if "y_pred_raw" in hp.columns else "oof_proba_raw"
        sig_col = "y_true" if "y_true" in hp.columns else "signal"
        p_raw = hp[raw_col].to_numpy()
        y = hp[sig_col].to_numpy()
        state_arr = hp["state"].astype(str).to_numpy() if "state" in hp.columns else None
        hp_time = pd.to_datetime(hp["time"])
        ref_time = hp_time.max()

        target_path = deploy_dir / f"oof_calibrator_{args.release}_{side}.joblib"

        if not is_nivel_2:
            method = next(iter(unique_methods))
            print(f"  ✅ Nivel 1: todos los estados → '{method}'")
            cal = fit_calibrator(method, p_raw, y, state_arr, hp_time, ref_time,
                                 args.min_n_per_state)
            if args.dry_run:
                print(f"  (dry-run) escribiría: {target_path}")
            else:
                if target_path.exists():
                    backup = target_path.with_suffix(".joblib.before_swap")
                    if not backup.exists():
                        shutil.copy2(target_path, backup)
                        print(f"  📁 backup: {backup.name}")
                joblib.dump(cal, target_path)
                print(f"  ✅ escrito: {target_path.name}")
            swap_meta["sides"][side] = {"level": 1, "method": method,
                                        "n_train_samples": int(len(p_raw))}
        else:
            print(f"  ⚠️  Nivel 2: estados con métodos distintos")
            for st, m in state_methods.items():
                print(f"      {st}: {m}")
            stateful = StatefulCalibrator()
            # fallback: método más votado
            most_voted = pd.Series(list(state_methods.values())).mode()[0]
            print(f"  → fallback global: {most_voted}")
            stateful.fallback = fit_calibrator(most_voted, p_raw, y, state_arr,
                                               hp_time, ref_time, args.min_n_per_state)
            if state_arr is None:
                print(f"  ⚠️  sin columna state, solo fallback se entrenará")
            else:
                for st, m in state_methods.items():
                    mask = state_arr == st
                    if int(mask.sum()) < 50:
                        print(f"     ⚠️  {st}: {int(mask.sum())} samples, salto")
                        continue
                    cal_st = fit_calibrator(m, p_raw[mask], y[mask],
                                            state_arr[mask], hp_time[mask],
                                            ref_time, args.min_n_per_state)
                    if isinstance(cal_st, StatefulCalibrator):
                        cal_st = cal_st.fallback
                    stateful.per_state[str(st)] = cal_st
                    stateful.kind_per_state[str(st)] = m

            if args.dry_run:
                print(f"  (dry-run) escribiría: {target_path}")
            else:
                if target_path.exists():
                    backup = target_path.with_suffix(".joblib.before_swap")
                    if not backup.exists():
                        shutil.copy2(target_path, backup)
                joblib.dump(stateful, target_path)
                print(f"  ✅ escrito StatefulCalibrator: {target_path.name}")
                print(f"  🚨 REQUIERE: modificar predict_side_pack para llamar")
                print(f"     calibrator.predict(p_raw, state) en vez de calibrator.predict(p_raw)")
            swap_meta["sides"][side] = {
                "level": 2,
                "methods_per_state": state_methods,
                "fallback_method": most_voted,
                "n_train_samples": int(len(p_raw)),
                "REQUIRES_simulator_change": True,
            }
        print()

    if not args.dry_run:
        meta_path = deploy_dir / "calibrator_swap_meta.json"
        meta_path.write_text(json.dumps(swap_meta, indent=2))
        print(f"📁 Metadata: {meta_path}")

    print(f"\n✅ Swap completado")
    if any(s.get("REQUIRES_simulator_change") for s in swap_meta["sides"].values()):
        print(f"\n🚨 AVISO: hay sides con StatefulCalibrator (Nivel 2).")
        print(f"   Producción requiere modificar predict_side_pack:")
        print(f"   Buscar la línea que llama a `self.calibrators[side].predict(p_raw)`,")
        print(f"   y cambiarla por:")
        print(f"     state_arr = df_p['state'].astype(str).to_numpy()")
        print(f"     cal = self.calibrators[side]")
        print(f"     if hasattr(cal, 'per_state'):")
        print(f"         p_cal = cal.predict(p_raw, state_arr)")
        print(f"     else:")
        print(f"         p_cal = cal.predict(p_raw)")


if __name__ == "__main__":
    main()