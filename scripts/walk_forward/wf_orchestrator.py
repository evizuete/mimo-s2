#!/usr/bin/env python3
"""
wf_orchestrator.py
─────────────────────────────────────────────────────────────────────────────
Orquestador walk-forward semanal.

Por cada lunes en [start, end]:
    1. Define modo: OPTUNA si (week_index % optuna-every == 0) sino WEIGHT-ONLY.
    2. Llama wf_train_at_date.py con la fecha de corte (cutoff = lunes).
       Si WEIGHT-ONLY, pasa --locked-params-json apuntando al
       best_params_flat.json del último OPTUNA.
    3. Llama wf_drift_metrics.py → decide auto-promote.
       · PROMOTE     → este release es el activo a partir de t.
       · KEEP        → mantenemos el activo previo, solo registramos drift.
    4. Lanza replay [t, t+7d] usando el release ACTIVO de la semana
       (replay_s2_202500.py). Output → artifacts/<active_release>/wf_live_<YYYYMMDD>/.
    5. Persiste un wf_step_summary.json en artifacts/walk_forward/<YYYY-MM-DD>/.
    6. Continúa con la siguiente semana.

Estado y resumability:
    artifacts/walk_forward/_state.json mantiene:
        - last_completed_week: YYYY-MM-DD
        - last_optuna_release: wf_YYYYMMDD (donde vive best_params_flat)
        - active_release: wf_YYYYMMDD (modelo vigente para replay)

Si el orquestador se interrumpe, al relanzarlo resume desde
last_completed_week + 1 sin reprocesar las anteriores.

Ejemplo:
    python -m scripts.walk_forward.wf_orchestrator \\
        --start 2026-01-05 --end 2026-05-04 \\
        --optuna-every 4 --optuna-trials 40
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, Optional


REPO_ROOT = Path(__file__).resolve().parents[2]


# ─────────────────────────────────────────────────────────────────────────────
# Helpers de fechas
# ─────────────────────────────────────────────────────────────────────────────

def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def fmt_date(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d")


def to_monday(dt: datetime) -> datetime:
    """Devuelve el lunes de la semana de dt (Mon=0 ... Sun=6)."""
    return dt - timedelta(days=dt.weekday())


def weekly_cutoffs(start: datetime, end: datetime) -> list[datetime]:
    """Lunes desde el primer lunes >= start hasta el último lunes <= end."""
    first = to_monday(start)
    if first < start:
        first = first + timedelta(days=7)
    last = to_monday(end)
    out = []
    cur = first
    while cur <= last:
        out.append(cur)
        cur = cur + timedelta(days=7)
    return out


def release_tag_for(cutoff: datetime) -> str:
    return f"wf_{cutoff.strftime('%Y%m%d')}"


# ─────────────────────────────────────────────────────────────────────────────
# State persistence
# ─────────────────────────────────────────────────────────────────────────────

def load_state(state_path: Path) -> Dict[str, Any]:
    if not state_path.exists():
        return {
            "last_completed_week": None,
            "last_optuna_release": None,
            "active_release": None,
            "history": [],
        }
    try:
        return json.loads(state_path.read_text())
    except json.JSONDecodeError:
        return {"last_completed_week": None, "last_optuna_release": None,
                "active_release": None, "history": []}


def save_state(state_path: Path, state: Dict[str, Any]) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2))


# ─────────────────────────────────────────────────────────────────────────────
# Subprocess runners
# ─────────────────────────────────────────────────────────────────────────────

def run_cmd(name: str, cmd: list[str], log_path: Path,
            allowed_rc: tuple[int, ...] = (0,)) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n▶ [{name}]\n  cmd: {' '.join(cmd)}\n  log: {log_path}")
    started = time.time()
    with log_path.open("w", encoding="utf-8") as fh:
        fh.write(f"# {' '.join(cmd)}\n# {datetime.now().isoformat()}\n\n")
        fh.flush()
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=str(REPO_ROOT), bufsize=1, text=True,
        )
        assert proc.stdout is not None
        for line in proc.stdout:
            fh.write(line)
            fh.flush()
            print(f"  │ {line.rstrip()}")
        rc = proc.wait()
    elapsed = time.time() - started
    status = "ok" if rc in allowed_rc else f"FAIL(rc={rc})"
    print(f"◀ [{name}] {status} ({elapsed:.1f}s)")
    return rc


def cmd_train_at_date(cutoff: datetime, run_optuna: bool, optuna_trials: int,
                      locked_params_json: Optional[Path],
                      train_years: int, holdout_months: int, calib_days: int,
                      artifacts_root: Path) -> list[str]:
    cmd = [
        sys.executable, "-m", "scripts.walk_forward.wf_train_at_date",
        "--cutoff", fmt_date(cutoff),
        "--train-years", str(train_years),
        "--holdout-months", str(holdout_months),
        "--calib-days", str(calib_days),
        "--artifacts-root", str(artifacts_root),
    ]
    if run_optuna:
        cmd += ["--run-optuna", "--optuna-trials", str(optuna_trials)]
    else:
        if locked_params_json is None:
            raise RuntimeError("WEIGHT-ONLY pero no hay locked_params_json")
        cmd += ["--locked-params-json", str(locked_params_json)]
    return cmd


def cmd_drift_metrics(release: str, artifacts_root: Path) -> list[str]:
    return [
        sys.executable, "-m", "scripts.walk_forward.wf_drift_metrics",
        "--release", release,
        "--artifacts-root", str(artifacts_root),
    ]


def cmd_replay(release: str, deploy_subdir: str, policy_module: str,
               from_dt: datetime, to_dt: datetime,
               out_dir: Path, initial_equity: float) -> list[str]:
    return [
        sys.executable, "-m", "scripts.replay_s2_202500",
        "--release", release,
        "--deploy-subdir", deploy_subdir,
        "--policy-config", policy_module,
        "--from", fmt_date(from_dt),
        "--to", fmt_date(to_dt),
        "--out", str(out_dir),
        "--initial-equity", f"{initial_equity:.2f}",
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", required=True, help="Inicio (YYYY-MM-DD).")
    ap.add_argument("--end", required=True, help="Fin (YYYY-MM-DD).")
    ap.add_argument("--optuna-every", type=int, default=4,
                    help="Optuna se corre cada N semanas (default 4 = mensual).")
    ap.add_argument("--optuna-trials", type=int, default=40)
    ap.add_argument("--train-years", type=int, default=2)
    ap.add_argument("--holdout-months", type=int, default=6)
    ap.add_argument("--calib-days", type=int, default=21)
    ap.add_argument("--artifacts-root", type=Path,
                    default=REPO_ROOT / "artifacts")
    ap.add_argument("--wf-root", type=Path,
                    default=REPO_ROOT / "artifacts" / "walk_forward")
    ap.add_argument("--initial-equity", type=float, default=10000.0)
    ap.add_argument("--deploy-subdir", default="deploy_full")
    ap.add_argument("--bootstrap-locked-params", type=Path, default=None,
                    help="Si la primera semana NO es Optuna, "
                         "best_params_flat.json a usar como bootstrap "
                         "(p.ej. del 202500).")
    ap.add_argument("--dry-run", action="store_true",
                    help="Calcula y muestra el plan sin ejecutar nada.")
    args = ap.parse_args()

    start = parse_date(args.start)
    end = parse_date(args.end)
    cutoffs = weekly_cutoffs(start, end)

    if not cutoffs:
        sys.exit("❌ No hay lunes en el rango especificado.")

    artifacts_root = args.artifacts_root.resolve()
    wf_root = args.wf_root.resolve()
    state_path = wf_root / "_state.json"
    state = load_state(state_path)

    # ── Plan ────────────────────────────────────────────────────────────────
    print("═" * 80)
    print(f"  WALK-FORWARD ORCHESTRATOR")
    print("═" * 80)
    print(f"  rango     : {fmt_date(cutoffs[0])} … {fmt_date(cutoffs[-1])}  ({len(cutoffs)} semanas)")
    print(f"  optuna    : cada {args.optuna_every} semanas ({args.optuna_trials} trials/run)")
    print(f"  train     : {args.train_years}y  |  holdout: {args.holdout_months}m  |  calib: {args.calib_days}d")
    print(f"  artifacts : {artifacts_root}")
    print(f"  state     : {state_path}")
    print(f"  resume    : last_completed_week={state.get('last_completed_week')}")
    print("─" * 80)
    print("  plan:")
    for i, t in enumerate(cutoffs):
        is_optuna = (i % args.optuna_every == 0)
        marker = "  ⚙ OPTUNA  " if is_optuna else "    weight "
        print(f"    {i:2d}. {marker}  {fmt_date(t)}  → {release_tag_for(t)}")
    print("═" * 80)

    if args.dry_run:
        print("\n[dry-run] saliendo sin ejecutar.")
        return

    last_completed = state.get("last_completed_week")
    if last_completed:
        last_dt = parse_date(last_completed)
    else:
        last_dt = None

    for i, cutoff in enumerate(cutoffs):
        if last_dt is not None and cutoff <= last_dt:
            print(f"\n⏭  [{fmt_date(cutoff)}] ya completada — skip")
            continue

        is_optuna_week = (i % args.optuna_every == 0)
        release = release_tag_for(cutoff)
        wf_step_dir = wf_root / fmt_date(cutoff)
        wf_step_dir.mkdir(parents=True, exist_ok=True)

        # locked_params resolución para WEIGHT-ONLY
        locked_params_path: Optional[Path] = None
        if not is_optuna_week:
            if state.get("last_optuna_release"):
                locked_params_path = (
                    artifacts_root / state["last_optuna_release"]
                    / "best_params_flat.json"
                )
            elif args.bootstrap_locked_params:
                locked_params_path = args.bootstrap_locked_params.resolve()
            else:
                sys.exit(
                    f"❌ semana {fmt_date(cutoff)} es WEIGHT-ONLY pero no hay "
                    "best_params previo ni --bootstrap-locked-params."
                )
            if not locked_params_path.exists():
                sys.exit(f"❌ locked_params no existe: {locked_params_path}")

        print("\n" + "█" * 80)
        print(f"  WEEK {i+1}/{len(cutoffs)}  {fmt_date(cutoff)}  "
              f"{'OPTUNA' if is_optuna_week else 'weight-only'}")
        print("█" * 80)

        # ── Stage A: train_at_date ──────────────────────────────────────────
        rc = run_cmd(
            f"train@{fmt_date(cutoff)}",
            cmd_train_at_date(
                cutoff=cutoff,
                run_optuna=is_optuna_week,
                optuna_trials=args.optuna_trials,
                locked_params_json=locked_params_path,
                train_years=args.train_years,
                holdout_months=args.holdout_months,
                calib_days=args.calib_days,
                artifacts_root=artifacts_root,
            ),
            wf_step_dir / "01_train.log",
        )
        if rc != 0:
            sys.exit(f"❌ train falló semana {fmt_date(cutoff)} (rc={rc}); aborto orquestador")

        # Si fue Optuna, registramos best_params_flat
        if is_optuna_week:
            best_path = artifacts_root / release / "best_params_flat.json"
            if not best_path.exists():
                print(f"⚠️  Optuna terminó pero no encuentro {best_path.name}. "
                      "Las semanas siguientes weight-only podrían fallar.")
            else:
                state["last_optuna_release"] = release

        # ── Stage B: drift_metrics → promote? ───────────────────────────────
        rc = run_cmd(
            f"drift@{fmt_date(cutoff)}",
            cmd_drift_metrics(release, artifacts_root),
            wf_step_dir / "02_drift.log",
            allowed_rc=(0, 10),  # 10 = no promovido (no es error fatal)
        )
        if rc not in (0, 10):
            sys.exit(f"❌ drift_metrics falló semana {fmt_date(cutoff)} (rc={rc})")

        promote_path = artifacts_root / release / "promote_decision.json"
        promoted = False
        if promote_path.exists():
            promoted = bool(json.loads(promote_path.read_text()).get("promoted", False))

        # ── Stage C: actualizar active_release ──────────────────────────────
        if promoted:
            state["active_release"] = release
            print(f"✅ PROMOTE: active_release ← {release}")
        else:
            print(f"❌ KEEP_PREVIOUS: active_release sigue = {state.get('active_release')}")

        active_release = state.get("active_release")
        if active_release is None:
            print("⚠️  No hay active_release todavía (primera semana sin promote). "
                  "Saltamos replay esta semana.")
            replay_summary_path = None
        else:
            # ── Stage D: replay [t, t+7d] con el release activo ─────────────
            replay_from = cutoff
            replay_to = cutoff + timedelta(days=7)
            replay_out = artifacts_root / active_release / f"wf_live_{cutoff.strftime('%Y%m%d')}"
            policy_module = f"config.decision_policies_config_{active_release}"

            rc = run_cmd(
                f"replay@{fmt_date(cutoff)}",
                cmd_replay(
                    release=active_release,
                    deploy_subdir=args.deploy_subdir,
                    policy_module=policy_module,
                    from_dt=replay_from,
                    to_dt=replay_to,
                    out_dir=replay_out,
                    initial_equity=args.initial_equity,
                ),
                wf_step_dir / "03_replay.log",
            )
            if rc != 0:
                print(f"⚠️  replay falló semana {fmt_date(cutoff)} (rc={rc}); continúo")
                replay_summary_path = None
            else:
                replay_summary_path = replay_out / "summary.json"

        # ── Stage E: wf_step_summary.json ───────────────────────────────────
        wf_summary = {
            "week_index": i,
            "cutoff": fmt_date(cutoff),
            "release_trained": release,
            "is_optuna_week": is_optuna_week,
            "promoted": promoted,
            "active_release_for_replay": active_release,
            "replay_period": {
                "from": fmt_date(cutoff),
                "to": fmt_date(cutoff + timedelta(days=7)),
            },
            "paths": {
                "wf_step_dir": str(wf_step_dir),
                "drift_metrics": str(artifacts_root / release / "wf_drift_metrics.json"),
                "promote_decision": str(promote_path) if promote_path.exists() else None,
                "replay_summary": str(replay_summary_path) if replay_summary_path and replay_summary_path.exists() else None,
            },
            "completed_at": datetime.now().isoformat(),
        }
        (wf_step_dir / "wf_step_summary.json").write_text(json.dumps(wf_summary, indent=2))

        # ── Update state ────────────────────────────────────────────────────
        state["last_completed_week"] = fmt_date(cutoff)
        state.setdefault("history", []).append({
            "week": fmt_date(cutoff),
            "release": release,
            "is_optuna": is_optuna_week,
            "promoted": promoted,
            "active": active_release,
        })
        save_state(state_path, state)

    print("\n" + "═" * 80)
    print(f"  ✅ ORCHESTRATOR COMPLETO  ({len(cutoffs)} semanas)")
    print(f"  state: {state_path}")
    print(f"  next: python -m scripts.walk_forward.wf_aggregate "
          f"--wf-root {wf_root}")
    print("═" * 80)


if __name__ == "__main__":
    main()
