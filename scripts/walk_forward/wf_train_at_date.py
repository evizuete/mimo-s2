#!/usr/bin/env python3
"""
wf_train_at_date.py
─────────────────────────────────────────────────────────────────────────────
Walk-forward step: para una fecha de corte t, entrena un release nuevo
ejecutando las 4 stages que componen el pipeline 202500:

    [1] mimo.oof.main_oof_regime_weights_v7        (Optuna OOF training)
    [2] mimo.oof.resume_deploy_full_v6_multitask   (deploy_full + tail recalib)
    [3] mimo.oof.select_thresholds_from_tail        (selected_threshold por side)
    [4] mimo.oof.compute_state_percentiles          (percentiles → policy stub)

Ventanas (defaults):
    train_from   = cutoff - (train_months + holdout_months) meses
    train_to     = cutoff - holdout_months                       (= holdout_from)
    holdout_from = cutoff - holdout_months
    holdout_to   = cutoff
    calib_tail   = últimos calib_days dentro del holdout

Idempotente por stage: si el output principal de una stage existe, la salta.

Ejemplos:
    # Modo "weight-only" (carga best_params previo, sin Optuna)
    python -m scripts.walk_forward.wf_train_at_date \\
        --cutoff 2026-01-12 \\
        --locked-params-json artifacts/wf_20260105/best_params.json

    # Modo "Optuna mensual"
    python -m scripts.walk_forward.wf_train_at_date \\
        --cutoff 2026-01-05 --run-optuna --optuna-trials 40
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Optional


REPO_ROOT = Path(__file__).resolve().parents[2]

# v7 (main_oof_regime_weights_v7.py) y otras stages escriben artifacts vía
# Path("../../artifacts") relativo al CWD. Para que "../../artifacts" resuelva
# a <REPO_ROOT>/artifacts, hay que lanzar los subprocesos desde una subcarpeta
# 2 niveles bajo REPO_ROOT. Usamos REPO_ROOT/mimo/oof (existe siempre) y
# añadimos REPO_ROOT a PYTHONPATH para que `python -m mimo.oof.X` siga funcionando.
SUBPROCESS_CWD = REPO_ROOT / "mimo" / "oof"


# ─────────────────────────────────────────────────────────────────────────────
# Configuración fija (alineada con read.me / 202500)
# ─────────────────────────────────────────────────────────────────────────────

BASE_TF = "5min"
TARGET_TYPE = "multitask"
VARIANT_LONG = "vol_boost_td_down"
VARIANT_SHORT = "vol_boost"
LABEL_HORIZON = 3
TP_BARRIER = 2.0
SL_BARRIER = 0.8
OOF_EPOCHS = 120
OOF_PATIENCE = 15

# Optuna (cuando run_optuna=True)
EV_NET_OBJECTIVE = "ev_net"
COST_PER_SIGNAL = 0.05
MAX_DRAWDOWN_R = 30.0
EV_MIN_SIGNALS = 100
EV_THR_LO = 0.10
EV_THR_HI = 0.40

# Threshold sweep (stage 3)
THR_SWEEP_LO = 0.10
THR_SWEEP_HI = 0.45
THR_SWEEP_N = 70

# Optuna storage (igual que extract_best_per_side)
DEFAULT_OPTUNA_STORAGE = (
    "mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db"
)
OPTUNA_STUDY_PREFIX = "oof_study"


@dataclass
class WindowSpec:
    cutoff: datetime
    train_from: datetime
    train_to: datetime
    holdout_from: datetime
    holdout_to: datetime
    calib_days: int

    def fmt(self, dt: datetime) -> str:
        return dt.strftime("%Y-%m-%d")


def compute_windows(cutoff: datetime, train_months: int,
                    holdout_months: int, calib_days: int) -> WindowSpec:
    holdout_to = cutoff
    holdout_from = cutoff - relativedelta_safe(months=holdout_months)
    train_to = holdout_from
    train_from = train_to - relativedelta_safe(months=train_months)
    return WindowSpec(
        cutoff=cutoff,
        train_from=train_from,
        train_to=train_to,
        holdout_from=holdout_from,
        holdout_to=holdout_to,
        calib_days=calib_days,
    )


def relativedelta_safe(years: int = 0, months: int = 0):
    """relativedelta sin importar dateutil si no está. Implementación simple
    suficiente para nuestro caso (sumar/restar años/meses preservando día,
    con clipping al último día del mes destino)."""
    try:
        from dateutil.relativedelta import relativedelta
        return relativedelta(years=years, months=months)
    except ImportError:
        # Fallback manual: convertimos a timedelta aproximado (años=365.25d).
        from datetime import timedelta
        return timedelta(days=int(years * 365.25 + months * 30.4375))


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess helper
# ─────────────────────────────────────────────────────────────────────────────

def run_stage(name: str, cmd: list[str], log_dir: Path,
              stream: bool = True) -> int:
    """Ejecuta una stage y vuelca stdout/stderr a log_dir/<name>.log.
    Si stream=True, también imprime cada línea en tiempo real al stdout
    del proceso padre (útil para seguir entrenamientos largos)."""
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{name}.log"
    started = time.time()
    print(f"\n▶ [{name}] start  (log → {log_path})")
    print(f"  cmd: {' '.join(cmd)}")

    env = os.environ.copy()
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(REPO_ROOT) + (
        os.pathsep + existing_pp if existing_pp else ""
    )

    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# {' '.join(cmd)}\n# started {datetime.now().isoformat()}\n\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            cwd=str(SUBPROCESS_CWD),
            env=env,
            bufsize=1,
            text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            if stream:
                print(f"  │ {line.rstrip()}")
        rc = proc.wait()
        elapsed = time.time() - started
        fh.write(f"\n# exit_code={rc} elapsed={elapsed:.1f}s\n")

    status = "ok" if rc == 0 else f"FAIL (rc={rc})"
    print(f"◀ [{name}] {status}  ({elapsed:.1f}s)")
    return rc


# ─────────────────────────────────────────────────────────────────────────────
# Stage builders
# ─────────────────────────────────────────────────────────────────────────────

def stage_oof_training(window: WindowSpec, release: str,
                       run_optuna: bool, optuna_trials: int,
                       locked_params_json: Optional[Path]) -> list[str]:
    cmd = [
        sys.executable, "-m", "mimo.oof.main_oof_regime_weights_v7",
        "--release", release,
        "--side", "both",
        "--target-type", TARGET_TYPE,
        "--base-tf", BASE_TF,
        "--variant-long", VARIANT_LONG,
        "--variant-short", VARIANT_SHORT,
        "--label-horizon-long", str(LABEL_HORIZON),
        "--label-horizon-short", str(LABEL_HORIZON),
        "--train-from", window.fmt(window.train_from),
        "--train-to", window.fmt(window.train_to),
        "--holdout-from", window.fmt(window.holdout_from),
        "--holdout-to", window.fmt(window.holdout_to),
        "--objective", EV_NET_OBJECTIVE,
        "--cost-per-signal", str(COST_PER_SIGNAL),
        "--max-drawdown-R", str(MAX_DRAWDOWN_R),
        "--ev-min-signals", str(EV_MIN_SIGNALS),
        "--ev-thr-lo", str(EV_THR_LO),
        "--ev-thr-hi", str(EV_THR_HI),
        "--oof-epochs", str(OOF_EPOCHS),
        "--oof-patience", str(OOF_PATIENCE),
    ]
    if run_optuna:
        cmd += ["--use-tpe", "--optuna-trials", str(optuna_trials)]
    else:
        if locked_params_json is None:
            raise SystemExit(
                "❌ Sin --run-optuna se requiere --locked-params-json apuntando "
                "a un best_params.json previo (modo weight-only)."
            )
        if not locked_params_json.exists():
            raise SystemExit(f"❌ locked_params_json no existe: {locked_params_json}")
        # Modo weight-only: pasamos solo --locked-params-json, SIN --skip-optuna.
        # v7 reduce el grid_space a un único punto (singleton) y corre 1 trial
        # con GridSampler — entrena pesos con los hyperparams fijados sin
        # necesidad de un Optuna study previo en la DB.
        cmd += ["--locked-params-json", str(locked_params_json)]
    return cmd


def stage_resume_deploy(window: WindowSpec, release: str,
                        deploy_subdir: str) -> list[str]:
    return [
        sys.executable, "-m", "mimo.oof.resume_deploy_full_v6_multitask",
        "--release", release,
        "--target-type", TARGET_TYPE,
        "--base-tf", BASE_TF,
        "--variant-long", VARIANT_LONG,
        "--variant-short", VARIANT_SHORT,
        "--label-horizon-long", str(LABEL_HORIZON),
        "--label-horizon-short", str(LABEL_HORIZON),
        "--train-from", window.fmt(window.train_from),
        "--holdout-from", window.fmt(window.holdout_from),
        "--holdout-to", window.fmt(window.holdout_to),
        "--deploy-calib-days", str(window.calib_days),
        "--train-artifacts-subdir", "auto",
        "--deploy-subdir", deploy_subdir,
    ]


def stage_select_thresholds(release: str, deploy_dir: Path) -> list[str]:
    return [
        sys.executable, "-m", "mimo.oof.select_thresholds_from_tail",
        "--release", release,
        "--deploy-dir", str(deploy_dir),
        "--side", "both",
        "--from-db",
        "--base-tf", BASE_TF,
        "--tp-long", str(TP_BARRIER),
        "--sl-long", str(SL_BARRIER),
        "--horizon-long", str(LABEL_HORIZON),
        "--tp-short", str(TP_BARRIER),
        "--sl-short", str(SL_BARRIER),
        "--horizon-short", str(LABEL_HORIZON),
        "--cost", str(COST_PER_SIGNAL),
        "--thr-lo", str(THR_SWEEP_LO),
        "--thr-hi", str(THR_SWEEP_HI),
        "--n-points", str(THR_SWEEP_N),
    ]


def stage_compute_percentiles(release: str, deploy_dir: Path,
                              out_stub: Path) -> list[str]:
    return [
        sys.executable, "-m", "mimo.oof.compute_state_percentiles",
        "--release", release,
        "--deploy-dir", str(deploy_dir),
        "--out-stub", str(out_stub),
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Idempotency probes
# ─────────────────────────────────────────────────────────────────────────────

def oof_done(release: str, artifacts_root: Path) -> bool:
    """OOF terminó si existe al menos un holdout_predictions parquet."""
    base = artifacts_root / release / "oof"
    if not base.exists():
        return False
    matches = list(base.rglob(f"holdout_predictions_{release}_*.parquet"))
    return len(matches) > 0


def deploy_done(release: str, artifacts_root: Path,
                deploy_subdir: str) -> bool:
    """Deploy terminó si existen percentiles_<release>_<side>.json."""
    deploy_dir = artifacts_root / release / "oof" / deploy_subdir
    return all(
        (deploy_dir / f"percentiles_{release}_{side}.json").exists()
        for side in ("long", "short")
    )


def thresholds_done(release: str, deploy_dir: Path) -> bool:
    """select_thresholds escribe selected_threshold dentro de percentiles_*.json."""
    for side in ("long", "short"):
        p = deploy_dir / f"percentiles_{release}_{side}.json"
        if not p.exists():
            return False
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            return False
        thr = data.get("_meta", {}).get("selected_threshold")
        if thr is None:
            return False
    return True


def policy_stub_done(stub_path: Path) -> bool:
    return stub_path.exists() and stub_path.stat().st_size > 0


def export_best_params_flat(release: str, artifacts_root: Path,
                             storage: str, study_prefix: str,
                             study_side: str = "multitask") -> Optional[Path]:
    """Tras una run Optuna, extrae study.best_trial.params como dict plano y
    persiste en artifacts/<release>/best_params_flat.json. Pensado para que
    las semanas walk-forward subsiguientes lo pasen por --locked-params-json.

    Si no se puede importar optuna o conectar al storage, devuelve None y
    logea un warning (no bloquea el step)."""
    out_path = artifacts_root / release / "best_params_flat.json"
    try:
        import optuna  # type: ignore
    except ImportError:
        print("⚠️  optuna no instalado; no se puede exportar best_params_flat")
        return None

    study_name = f"{study_prefix}_{release}_{study_side}"
    try:
        study = optuna.load_study(study_name=study_name, storage=storage)
    except Exception as e:  # KeyError, OperationalError, etc.
        print(f"⚠️  no pude cargar study {study_name}: {e}")
        return None

    try:
        params = dict(study.best_trial.params)
    except Exception as e:
        print(f"⚠️  study sin best_trial: {e}")
        return None

    # IMPORTANTE: v7 (--locked-params-json) espera un dict PLANO de hyperparams.
    # Metadata va a un sidecar para debugging.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(params, indent=2))

    meta_path = out_path.with_name(out_path.stem + "_meta.json")
    meta = {
        "release": release,
        "study_name": study_name,
        "best_value": float(study.best_value)
            if study.best_value is not None else None,
        "best_trial_number": study.best_trial.number,
        "params_file": out_path.name,
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    print(f"✅ best_params_flat → {out_path}  (dict plano, {len(params)} params)")
    print(f"   meta → {meta_path}  (trial #{study.best_trial.number}, "
          f"value={meta['best_value']:.4f})")
    return out_path


# ─────────────────────────────────────────────────────────────────────────────
# Main driver
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cutoff", required=True,
                    help="Fecha de corte (YYYY-MM-DD), final del holdout.")
    ap.add_argument("--release-tag", default=None,
                    help="Tag del release (default: wf_<YYYYMMDD> derivado del cutoff).")
    ap.add_argument("--train-months", type=int, default=24,
                    help="Tamaño de la ventana de training en meses (default 24).")
    ap.add_argument("--holdout-months", type=int, default=6)
    ap.add_argument("--calib-days", type=int, default=21)
    ap.add_argument("--run-optuna", action="store_true",
                    help="Ejecuta búsqueda Optuna completa (modo mensual).")
    ap.add_argument("--optuna-trials", type=int, default=40)
    ap.add_argument("--locked-params-json", type=Path, default=None,
                    help="Path a best_params.json (modo weight-only).")
    ap.add_argument("--artifacts-root", type=Path,
                    default=REPO_ROOT / "artifacts",
                    help="Raíz donde se escriben artifacts/<release>/...")
    ap.add_argument("--config-root", type=Path,
                    default=REPO_ROOT / "config",
                    help="Raíz donde se escribe decision_policies_config_<release>.py")
    ap.add_argument("--deploy-subdir", default="deploy_full")
    ap.add_argument("--force", action="store_true",
                    help="Ignora idempotency probes y re-corre todas las stages.")
    ap.add_argument("--skip-stages", default="",
                    help="Lista CSV de stages a saltar: oof,deploy,thresholds,percentiles")
    ap.add_argument("--optuna-storage", default=DEFAULT_OPTUNA_STORAGE,
                    help="URL del Optuna storage (para exportar best_params tras Optuna).")
    ap.add_argument("--optuna-study-prefix", default=OPTUNA_STUDY_PREFIX)
    args = ap.parse_args()

    cutoff = datetime.strptime(args.cutoff, "%Y-%m-%d")
    release = args.release_tag or f"wf_{cutoff.strftime('%Y%m%d')}"
    skip = {s.strip() for s in args.skip_stages.split(",") if s.strip()}

    window = compute_windows(cutoff, args.train_months,
                             args.holdout_months, args.calib_days)

    artifacts_root = args.artifacts_root.resolve()
    deploy_dir = artifacts_root / release / "oof" / args.deploy_subdir
    log_dir = artifacts_root / release / "wf_logs"
    config_root = args.config_root.resolve()
    stub_path = config_root / f"decision_policies_config_{release}.py"

    # ── Print plan ───────────────────────────────────────────────────────────
    print("═" * 80)
    print(f"  WALK-FORWARD STEP — release={release}")
    print("═" * 80)
    print(f"  cutoff        : {window.fmt(window.cutoff)}")
    print(f"  train         : [{window.fmt(window.train_from)} → {window.fmt(window.train_to)})")
    print(f"  holdout       : [{window.fmt(window.holdout_from)} → {window.fmt(window.holdout_to)})")
    print(f"  calib_tail    : últimos {window.calib_days} días del holdout")
    print(f"  mode          : {'OPTUNA' if args.run_optuna else 'WEIGHT-ONLY'}")
    if not args.run_optuna:
        print(f"  locked_params : {args.locked_params_json}")
    print(f"  artifacts     : {artifacts_root / release}")
    print(f"  policy stub   : {stub_path}")
    print("═" * 80)

    # ── Persist plan to disk ─────────────────────────────────────────────────
    plan_path = artifacts_root / release / "wf_step_plan.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_payload = {
        "release": release,
        "cutoff": window.fmt(window.cutoff),
        "windows": {
            "train_from": window.fmt(window.train_from),
            "train_to": window.fmt(window.train_to),
            "holdout_from": window.fmt(window.holdout_from),
            "holdout_to": window.fmt(window.holdout_to),
            "calib_days": window.calib_days,
        },
        "run_optuna": args.run_optuna,
        "optuna_trials": args.optuna_trials if args.run_optuna else None,
        "locked_params_json": str(args.locked_params_json) if args.locked_params_json else None,
        "fixed": {
            "base_tf": BASE_TF,
            "target_type": TARGET_TYPE,
            "variant_long": VARIANT_LONG,
            "variant_short": VARIANT_SHORT,
            "label_horizon": LABEL_HORIZON,
            "tp_barrier": TP_BARRIER,
            "sl_barrier": SL_BARRIER,
            "oof_epochs": OOF_EPOCHS,
            "oof_patience": OOF_PATIENCE,
        },
        "started_at": datetime.now().isoformat(),
    }
    plan_path.write_text(json.dumps(plan_payload, indent=2))

    # ── Stage 1: OOF training ────────────────────────────────────────────────
    if "oof" in skip:
        print("⏭  [oof] saltada por --skip-stages")
    elif not args.force and oof_done(release, artifacts_root):
        print(f"⏭  [oof] ya completada (holdout_predictions encontrado)")
    else:
        cmd = stage_oof_training(window, release, args.run_optuna,
                                 args.optuna_trials, args.locked_params_json)
        rc = run_stage("01_oof_training", cmd, log_dir)
        if rc != 0:
            sys.exit(rc)
        if not oof_done(release, artifacts_root):
            sys.exit("❌ stage oof terminó sin error pero no encuentro holdout_predictions")
        if args.run_optuna:
            export_best_params_flat(
                release=release,
                artifacts_root=artifacts_root,
                storage=args.optuna_storage,
                study_prefix=args.optuna_study_prefix,
                study_side="multitask",
            )

    # ── Stage 2: deploy_full + tail recalibration ────────────────────────────
    if "deploy" in skip:
        print("⏭  [deploy] saltada por --skip-stages")
    elif not args.force and deploy_done(release, artifacts_root, args.deploy_subdir):
        print(f"⏭  [deploy] ya completada (percentiles_*.json encontrado)")
    else:
        cmd = stage_resume_deploy(window, release, args.deploy_subdir)
        rc = run_stage("02_resume_deploy", cmd, log_dir)
        if rc != 0:
            sys.exit(rc)
        if not deploy_done(release, artifacts_root, args.deploy_subdir):
            sys.exit("❌ stage deploy terminó sin error pero faltan percentiles_*.json")

    # ── Stage 3: select thresholds from tail ─────────────────────────────────
    if "thresholds" in skip:
        print("⏭  [thresholds] saltada por --skip-stages")
    elif not args.force and thresholds_done(release, deploy_dir):
        print(f"⏭  [thresholds] ya completada (selected_threshold presente)")
    else:
        cmd = stage_select_thresholds(release, deploy_dir)
        rc = run_stage("03_select_thresholds", cmd, log_dir)
        if rc != 0:
            sys.exit(rc)
        if not thresholds_done(release, deploy_dir):
            sys.exit("❌ stage thresholds terminó sin error pero falta selected_threshold")

    # ── Stage 4: compute state percentiles + emit policy stub ────────────────
    if "percentiles" in skip:
        print("⏭  [percentiles] saltada por --skip-stages")
    elif not args.force and policy_stub_done(stub_path):
        print(f"⏭  [percentiles] ya completada ({stub_path.name} existe)")
    else:
        cmd = stage_compute_percentiles(release, deploy_dir, stub_path)
        rc = run_stage("04_compute_percentiles", cmd, log_dir)
        if rc != 0:
            sys.exit(rc)
        if not policy_stub_done(stub_path):
            sys.exit(f"❌ stage percentiles terminó sin error pero {stub_path.name} no existe")

    # ── Done ─────────────────────────────────────────────────────────────────
    plan_payload["finished_at"] = datetime.now().isoformat()
    plan_path.write_text(json.dumps(plan_payload, indent=2))

    print("\n" + "═" * 80)
    print(f"  ✅ WF STEP COMPLETE — release={release}")
    print("═" * 80)
    print(f"  artifacts  : {artifacts_root / release}")
    print(f"  deploy_dir : {deploy_dir}")
    print(f"  policy     : {stub_path}")
    print(f"  logs       : {log_dir}")


if __name__ == "__main__":
    main()
