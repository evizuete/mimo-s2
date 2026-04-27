# optuna_rl_search.py
#
# Ejecuta Optuna llamando por subprocess a train_rl_policy.py
# con split 2 meses TRAIN + 1 mes VALID (aprox. 30 días/mes).
#
# Requisitos:
# - train_rl_policy.py debe soportar:
#   - --eval_only (para validar sin entrenar)
#   - usar rl_config (no fijar rl_config_diag)
#
# Ejemplo:
#   python optuna_rl_search.py --release 200304 --from 2025-10-25 --train_script train_rl_policy.py \
#       --n_trials 50 --debug
#
import argparse
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from pathlib import Path

import optuna


def parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d")


def run_v0(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    out = (p.stdout or "") + "\n" + (p.stderr or "")
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n{out}")
    return out

def run(cmd: list[str], cwd: str | None = None, timeout: int | None = None) -> str:
    print("[RUN]", " ".join(map(str, cmd)))

    p = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        cwd=cwd,
        bufsize=1,
        universal_newlines=True,
    )

    lines = []
    try:
        assert p.stdout is not None
        for line in p.stdout:
            print(line, end="")      # streaming
            lines.append(line)
        rc = p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        p.kill()
        raise RuntimeError(f"Timeout running: {' '.join(cmd)}")
    finally:
        try:
            if p.stdout is not None:
                p.stdout.close()
        except Exception:
            pass

    out = "".join(lines)
    if rc != 0:
        raise RuntimeError(f"Command failed ({rc}): {' '.join(cmd)}\n{out}")
    return out


def grab_float(out: str, label: str):
    # matches: "NetPnL     : 14610.56" or "MaxDD      : 2175.60"
    m = re.search(rf"{re.escape(label)}\s*:\s*([-\d\.]+)", out)
    return float(m.group(1)) if m else None


def grab_ptake_std(out: str):
    # matches: "mean/std   : 0.5000 / 0.0056"
    m = re.search(r"mean/std\s*:\s*([-\d\.]+)\s*/\s*([-\d\.]+)", out)
    return float(m.group(2)) if m else None


def grab_ptake_percentiles(out: str):
    # matches: "percentiles: {1: 0.49, 5: 0.49, ...}"
    m = re.search(r"percentiles:\s*(\{.*\})", out)
    return m.group(1) if m else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_script", default="main_rl_train.py")
    ap.add_argument("--release", required=True)
    ap.add_argument("--from", dest="from_date", required=True, help="Inicio de la ventana TRAIN (2 meses aprox.)")

    ap.add_argument("--artifacts_path", default="./artifacts")
    ap.add_argument("--study_name", default=None)
    ap.add_argument("--n_trials", type=int, default=30)

    # Split 2M + 1M (30 días/mes)
    ap.add_argument("--train_months", type=int, default=2)
    ap.add_argument("--valid_months", type=int, default=1)

    # Score (VALID)
    ap.add_argument("--alpha_dd", type=float, default=3.0, help="Penalización por drawdown (dinero)")
    ap.add_argument("--beta_trades", type=float, default=0.2, help="Penalización por número de trades")

    # Pruning por policy plana (requiere --debug en train)
    ap.add_argument("--debug", action="store_true", default=False, help="Activa --debug en TRAIN para medir std(pTAKE)")
    ap.add_argument("--min_ptake_std", type=float, default=0.02)
    ap.add_argument("--prune_flat", action="store_true", default=True)

    # Pass-through al train_rl_policy.py
    ap.add_argument("--initial_equity", type=float, default=10_000.0)
    ap.add_argument("--spread_price", type=float, default=0.07)
    ap.add_argument("--max_daily_loss_pct", type=float, default=0.0580)
    ap.add_argument("--sizing_equity_mode", default="balance")
    ap.add_argument("--compound", action="store_true", default=True)

    ap.add_argument("--use_oof", action="store_true", default=True)
    ap.add_argument("--oof_splits", type=int, default=3)
    ap.add_argument("--oof_epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=8192)
    ap.add_argument("--seq_len_short", type=int, default=64)
    ap.add_argument("--seq_len_long", type=int, default=256)

    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    os.makedirs(args.artifacts_path, exist_ok=True)
    out_dir = Path(args.artifacts_path) / "optuna_rl_subprocess"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Ventanas
    from_date = parse_date(args.from_date)
    train_to = from_date + timedelta(days=int(args.train_months) * 30)
    valid_from = train_to
    valid_to = train_to + timedelta(days=int(args.valid_months) * 30)

    def objective(trial: optuna.Trial) -> float:
        # Suggest (rangos “seguros” para evitar policies planas por entropía enorme o lr ínfimo)
        rl_lr = trial.suggest_float("rl_lr", 3e-4, 8e-3, log=True)
        rl_entropy = trial.suggest_float("rl_entropy", 0.0, 5e-4)
        rl_baseline_beta = trial.suggest_float("rl_baseline_beta", 0.80, 0.98)
        rl_max_grad_norm = trial.suggest_float("rl_max_grad_norm", 0.5, 10.0)
        rl_trade_cost = 0.0 #trial.suggest_float("rl_trade_cost", 0.05, 1.5)
        rl_batch = trial.suggest_categorical("rl_batch", [128, 256, 512, 1024])

        rl_chop_soft_thr = trial.suggest_float("rl_chop_soft_thr", 0.50, 0.80)
        rl_exhaustion_soft_thr = trial.suggest_float("rl_exhaustion_soft_thr", 0.50, 0.80)
        rl_chop_penalty_coef = trial.suggest_float("rl_chop_penalty_coef", 0.0, 0.12)
        rl_exhaustion_penalty_coef = trial.suggest_float("rl_exhaustion_penalty_coef", 0.0, 0.12)

        policy_path = str(out_dir / f"trial_{trial.number:05d}.npz")

        # -------- TRAIN (2 meses) --------
        cmd_train = [
            sys.executable, args.train_script,
            #"python", args.train_script,
            "--release", str(args.release),
            "--from", from_date.strftime("%Y-%m-%d"),
            "--to", train_to.strftime("%Y-%m-%d"),
            "--artifacts_path", args.artifacts_path,
            "--policy_out", policy_path,

            "--rl_lr", str(rl_lr),
            "--rl_entropy", str(rl_entropy),
            "--rl_baseline_beta", str(rl_baseline_beta),
            "--rl_max_grad_norm", str(rl_max_grad_norm),
            "--rl_trade_cost", str(rl_trade_cost),
            "--rl_batch", str(rl_batch),
            "--rl_chop_soft_thr", str(rl_chop_soft_thr),
            "--rl_exhaustion_soft_thr", str(rl_exhaustion_soft_thr),
            "--rl_chop_penalty_coef", str(rl_chop_penalty_coef),
            "--rl_exhaustion_penalty_coef", str(rl_exhaustion_penalty_coef),

            "--initial_equity", str(args.initial_equity),
            "--spread_price", str(args.spread_price),
            "--max_daily_loss_pct", str(args.max_daily_loss_pct),
            "--sizing_equity_mode", str(args.sizing_equity_mode),

            "--oof_splits", str(args.oof_splits),
            "--oof_epochs", str(args.oof_epochs),
            "--batch_size", str(args.batch_size),
            "--seq_len_short", str(args.seq_len_short),
            "--seq_len_long", str(args.seq_len_long),
        ]
        if args.use_oof:
            cmd_train.append("--use_oof")
        if args.compound:
            cmd_train.append("--compound")
        if args.debug:
            cmd_train.append("--debug")

        print(f"[trial {trial.number}] lr={rl_lr:.6f} ent={rl_entropy:.6f} bb={rl_baseline_beta:.6f} "
              f"gn={rl_max_grad_norm:.6f} batch={rl_batch} tc={rl_trade_cost} "
              f"ch_thr={rl_chop_soft_thr} ex_thr={rl_exhaustion_soft_thr} "
              f"ch_k={rl_chop_penalty_coef} ex_k={rl_exhaustion_penalty_coef}")

        print("[CMD TRAIN]", " ".join(cmd_train))
        out_train = run(cmd_train)

        if args.debug:
            ptake_std = grab_ptake_std(out_train)
            ptake_p = grab_ptake_percentiles(out_train)

            if ptake_std is not None:
                trial.set_user_attr("ptake_std_train", float(ptake_std))
            if ptake_p is not None:
                trial.set_user_attr("ptake_percentiles_train", ptake_p)

            # -------- PRINT PARAMS + DEBUG WHEN PRUNING --------
            if args.prune_flat and (ptake_std is not None) and (ptake_std < float(args.min_ptake_std)):

                print("\n================ PRUNED TRIAL =================")
                print(f"trial: {trial.number}")
                print(f"ptake_std: {ptake_std:.6f}  (min={args.min_ptake_std})")
                print("params:")

                for k, v in trial.params.items():
                    print(f"  {k}: {v}")

                if ptake_p is not None:
                    print("pTAKE percentiles:", ptake_p)

                # guardar log completo del TRAIN
                log_path = out_dir / f"trial_{trial.number:05d}_train.log"
                log_path.write_text(out_train, encoding="utf-8", errors="ignore")
                print(f"train log saved to: {log_path}")
                print("================================================\n")

                raise optuna.TrialPruned(f"flat policy: std={ptake_std:.4f}")

        # -------- VALID (1 mes) --------
        cmd_eval = [
            sys.executable, args.train_script,
            "--release", str(args.release),
            "--from", valid_from.strftime("%Y-%m-%d"),
            "--to", valid_to.strftime("%Y-%m-%d"),
            "--artifacts_path", args.artifacts_path,
            "--eval_only",
            "--policy_init", policy_path,
            "--policy_out", policy_path,  # no se usa, pero mantenemos firma homogénea

            "--initial_equity", str(args.initial_equity),
            "--spread_price", str(args.spread_price),
            "--max_daily_loss_pct", str(args.max_daily_loss_pct),
            "--sizing_equity_mode", str(args.sizing_equity_mode),

            "--oof_splits", str(args.oof_splits),
            "--oof_epochs", str(args.oof_epochs),
            "--batch_size", str(args.batch_size),
            "--seq_len_short", str(args.seq_len_short),
            "--seq_len_long", str(args.seq_len_long),
        ]
        if args.use_oof:
            cmd_eval.append("--use_oof")
        if args.compound:
            cmd_eval.append("--compound")

        out_eval = run(cmd_eval)

        pnl = grab_float(out_eval, "NetPnL")
        dd = grab_float(out_eval, "MaxDD")
        trades = grab_float(out_eval, "Trades")
        pf = grab_float(out_eval, "ProfitFactor")
        wr = grab_float(out_eval, "WinRate")

        if pnl is None or dd is None or trades is None:
            raise RuntimeError("No pude parsear NetPnL/MaxDD/Trades del output de VALID.\n" + out_eval)

        score = float(pnl) - float(args.alpha_dd) * float(dd) - float(args.beta_trades) * float(trades)

        trial.set_user_attr("valid_pnl", float(pnl))
        trial.set_user_attr("valid_dd", float(dd))
        trial.set_user_attr("valid_trades", float(trades))
        if pf is not None:
            trial.set_user_attr("valid_pf", float(pf))
        if wr is not None:
            trial.set_user_attr("valid_wr", float(wr))

        return score

    pruner = optuna.pruners.MedianPruner(n_warmup_steps=max(2, args.n_trials // 5))
    study = optuna.create_study(
        direction="maximize",
        study_name=args.study_name or f"rl_gate_{args.release}",
        storage='mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db',
        load_if_exists=True,
        sampler=optuna.samplers.TPESampler(seed=args.seed),
        pruner=pruner,
    )

    print(f"[WINDOWS] TRAIN {from_date.date()} -> {train_to.date()} | VALID {valid_from.date()} -> {valid_to.date()}")
    study.optimize(objective, n_trials=int(args.n_trials))

    print("\n=== BEST ===")
    print("score:", study.best_value)
    print("params:", study.best_params)
    print("attrs:", study.best_trial.user_attrs)

    # Copiar mejor policy como BEST
    best_trial_num = study.best_trial.number
    best_policy = out_dir / f"trial_{best_trial_num:05d}.npz"
    final_path = Path(args.artifacts_path) / f"rl_policy_gate_{args.release}_BEST.npz"
    if best_policy.exists():
        final_path.write_bytes(best_policy.read_bytes())
        print("Best policy saved:", str(final_path))


if __name__ == "__main__":
    main()
