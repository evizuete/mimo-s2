"""
fwd_check_staged_policy.py

Forward check directo con la policy del STAGED (sin pasar por prod_train).
Permite comparar si el problema es el fine-tuning o el gate eval_threshold_scale.

USO:
    python fwd_check_staged_policy.py

TAMBIÉN hace un segundo pase con eval_threshold_scale=1.0 (sin filtro)
para descartar que el gate sea el cuello de botella en 2026.
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# ─── CONFIGURACIÓN — ajusta si tu release o rutas cambian ──────────────────
release_prod     = '200367'
ruta_best_trial  = f'../../artifacts/{release_prod}/rl/staged/best'
ruta_deploy_full = f'../../artifacts/{release_prod}/oof/deploy_full'
out_root         = Path(f'../../artifacts/{release_prod}/rl/fwd_staged_check').resolve()
ruta_main        = './main_rl_train.py'

fwd_check_from   = "2026-01-01"
fwd_check_to     = "2026-03-17"

# ─── Mismos parámetros fijos que en freeze_prod v3 ─────────────────────────
BASE_FIXED = {
    "initial_equity":          10000.0,
    "spread_price":            0.07,
    "max_daily_loss_pct":      0.0350,
    "sizing_equity_mode":      "balance",
    "compound":                True,
    "use_oof":                 True,
    "oof_splits":              3,
    "oof_epochs":              25,
    "batch_size":              8192,
    "seq_len_short":           64,
    "seq_len_long":            256,
    "rl_trade_cost":           0.0075,
    "rl_train_threshold":      0.08,
    "rl_max_grad_norm":        2.0,
    "rl_batch":                64,
    "rl_baseline_beta":        0.92,
    "rl_vol_filter":           "none",
    "rl_vol_filter_mult":      1.5,
}


# ─── HELPERS ───────────────────────────────────────────────────────────────

def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)

def dump_json(p: Path, obj) -> None:
    ensure_dir(p.parent)
    p.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")

def build_cmd(params: dict) -> list:
    cmd = [sys.executable, ruta_main]
    for k, v in params.items():
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(v)])
    return cmd

def run(cmd: list, log_path: Path, timeout_s: int = 3600) -> subprocess.CompletedProcess:
    ensure_dir(log_path.parent)
    print(f"  CMD: {' '.join(cmd[:6])} ...")
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("=== CMD ===\n")
        f.write(" ".join(cmd) + "\n\n")
        f.write("=== STDOUT ===\n")
        f.write(res.stdout + "\n\n")
        f.write("=== STDERR ===\n")
        f.write(res.stderr + "\n\n")
        f.write("=== RET ===\n")
        f.write(str(res.returncode) + "\n")
    return res

def parse_metrics(stdout: str) -> dict:
    import re
    out = {}
    patterns = {
        "n_trades":      r"Trades\s*:\s*(\d+)",
        "net_pnl":       r"NetPnL\s*:\s*([-+]?\d+\.?\d*)",
        "profit_factor": r"ProfitFactor\s*:\s*([-+]?\d+\.?\d*)",
        "win_rate":      r"WinRate\s*:\s*([-+]?\d+\.?\d*)",
        "max_dd_pct":    r"MaxDD\s+\(%\)\s*:\s*([-+]?\d+\.?\d*)",
    }
    for k, pat in patterns.items():
        m = re.search(pat, stdout)
        if m:
            try:
                out[k] = float(m.group(1))
            except ValueError:
                pass
    return out

def print_metrics(label: str, m: dict) -> None:
    trades = m.get("n_trades", 0)
    pf     = m.get("profit_factor", 0)
    wr     = m.get("win_rate", 0)
    dd     = abs(m.get("max_dd_pct", 0))
    pnl    = m.get("net_pnl", 0)
    print(f"\n  [{label}]")
    print(f"    Trades : {trades:.0f}   {'✅' if trades >= 100 else '🔴 < 100'}")
    print(f"    PF     : {pf:.4f}   {'✅' if pf >= 1.05 else ('⚠️' if pf >= 0.95 else '🔴')}")
    print(f"    WinRate: {wr*100:.1f}%")
    print(f"    MaxDD% : {dd*100:.2f}%")
    print(f"    NetPnL : {pnl:.2f}")

def find_staged_policy(best_trial_path: Path, best_params: dict, release: str) -> Path:
    """
    Mismo algoritmo que freeze_prod v3: busca el trial dir que matchea best_params.
    Fallback a _best.npz si no lo encuentra.
    """
    staged_policy = None
    logs_dir = best_trial_path.parent / "logs"
    best_params_items = set(
        f"{k}={round(float(v),8)}" for k, v in best_params.items()
        if isinstance(v, float)
    )

    if logs_dir.exists():
        for trial_dir in sorted(logs_dir.iterdir()):
            cfg = trial_dir / "config_val.json"
            if not cfg.exists():
                continue
            try:
                cfg_data = json.loads(cfg.read_text(encoding="utf-8"))
                cfg_items = set(
                    f"{k}={round(float(v),8)}" for k, v in cfg_data.items()
                    if isinstance(v, float) and k in best_params
                )
                if best_params_items == cfg_items:
                    npz_candidates = list(trial_dir.glob(f"rl_policy_gate_{release}_trial_*.npz"))
                    if npz_candidates:
                        staged_policy = npz_candidates[0]
                        print(f"  Trial dir encontrado: {trial_dir.name}  →  {staged_policy.name}")
                        break
            except Exception:
                continue

    if staged_policy is None or not staged_policy.exists():
        staged_policy = best_trial_path / f"rl_policy_gate_{release}_best.npz"
        print(f"  WARNING: trial dir no encontrado — usando fallback _best.npz")

    if not staged_policy.exists():
        raise FileNotFoundError(f"No existe policy staged: {staged_policy}")

    return staged_policy


# ─── MAIN ──────────────────────────────────────────────────────────────────

def main():
    best_trial_path  = Path(ruta_best_trial).resolve()
    deploy_full_path = Path(ruta_deploy_full).resolve()

    if not best_trial_path.exists():
        raise FileNotFoundError(f"No existe ruta_best_trial: {best_trial_path}")
    if not deploy_full_path.exists():
        raise FileNotFoundError(f"No existe ruta_deploy_full: {deploy_full_path}")

    ensure_dir(out_root)

    # Cargar best_params
    json_path = best_trial_path / 'best_trial.json'
    best = json.loads(json_path.read_text(encoding="utf-8"))
    best_params = best.get("best_params", {})
    if not best_params:
        raise ValueError("best_trial.json no contiene 'best_params'")

    # Localizar policy del staged
    staged_policy = find_staged_policy(best_trial_path, best_params, release_prod)
    print(f"\n=== FWD CHECK — STAGED POLICY ===")
    print(f"  Policy : {staged_policy}")
    print(f"  Periodo: {fwd_check_from} → {fwd_check_to}")
    print(f"  W_sum esperado (staged): ver log — debe ser ~0.000595")

    eval_threshold_scale = best_params.get("rl_eval_threshold_scale", 0.34180529903172097)
    print(f"  eval_threshold_scale original: {eval_threshold_scale:.4f}")

    results = {}

    # ── TEST A: policy staged + eval_threshold_scale original (producción real) ──
    print(f"\n[A] Policy staged + eval_threshold_scale={eval_threshold_scale:.4f} (modo production)")
    params_a = dict(BASE_FIXED)
    params_a.update(best_params)
    params_a.update({
        "release":        release_prod,
        "from":           fwd_check_from,
        "to":             fwd_check_to,
        "artifacts_path": str(deploy_full_path),
        "eval_only":      True,
        "policy_init":    str(staged_policy),
        "rl_mode":        "production",
    })
    res_a = run(build_cmd(params_a), out_root / "fwd_staged_production.log")
    metrics_a = parse_metrics(res_a.stdout)
    dump_json(out_root / "metrics_staged_production.json", metrics_a)
    print_metrics("staged + production gate", metrics_a)
    results["staged_production"] = metrics_a

    # ── TEST B: policy staged + eval_threshold_scale=1.0 (sin filtro de gate) ──
    # Si B da muchos más trades que A, el gate es el problema.
    # Si B sigue dando pocos trades, el problema es la policy en sí.
    print(f"\n[B] Policy staged + eval_threshold_scale=1.0 (gate desactivado)")
    params_b = dict(params_a)
    params_b["rl_eval_threshold_scale"] = 1.0
    res_b = run(build_cmd(params_b), out_root / "fwd_staged_nogate.log")
    metrics_b = parse_metrics(res_b.stdout)
    dump_json(out_root / "metrics_staged_nogate.json", metrics_b)
    print_metrics("staged + gate=1.0 (sin filtro)", metrics_b)
    results["staged_nogate"] = metrics_b

    # ── TEST C: policy fine-tuned (la que ya tienes) + eval_threshold_scale=1.0 ──
    # Compara con la policy fine-tuned del prod_train para aislar el efecto del fine-tune.
    prod_policy = Path(f'../../artifacts/{release_prod}/rl/final/rl_policy_gate_{release_prod}.npz').resolve()
    if prod_policy.exists():
        print(f"\n[C] Policy fine-tuned (prod_train) + eval_threshold_scale=1.0 (gate desactivado)")
        params_c = dict(params_a)
        params_c["rl_eval_threshold_scale"] = 1.0
        params_c["policy_init"] = str(prod_policy)
        res_c = run(build_cmd(params_c), out_root / "fwd_finetuned_nogate.log")
        metrics_c = parse_metrics(res_c.stdout)
        dump_json(out_root / "metrics_finetuned_nogate.json", metrics_c)
        print_metrics("fine-tuned + gate=1.0", metrics_c)
        results["finetuned_nogate"] = metrics_c
    else:
        print(f"\n[C] SKIP — no existe policy fine-tuned en: {prod_policy}")

    # ── RESUMEN ──────────────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"RESUMEN COMPARATIVO  {fwd_check_from} → {fwd_check_to}")
    print(f"{'='*60}")
    headers = ["escenario", "trades", "PF", "WR%", "DD%"]
    rows = [
        ("A staged+gate_orig",    results.get("staged_production", {})),
        ("B staged+gate_off",     results.get("staged_nogate", {})),
        ("C finetuned+gate_off",  results.get("finetuned_nogate", {})),
    ]
    print(f"  {'escenario':<26} {'trades':>7} {'PF':>7} {'WR%':>6} {'DD%':>6}")
    print(f"  {'-'*54}")
    for label, m in rows:
        if not m:
            continue
        print(
            f"  {label:<26} "
            f"{m.get('n_trades',0):>7.0f} "
            f"{m.get('profit_factor',0):>7.4f} "
            f"{m.get('win_rate',0)*100:>6.1f} "
            f"{abs(m.get('max_dd_pct',0))*100:>6.2f}"
        )

    print(f"\n  Referencia (staged test Jul-Dic25): 695 trades | PF 1.191 | WR 44.6% | DD 3.28%")
    print(f"  Referencia (prod_train original):    60 trades | PF 0.876 | WR 38.3% | DD 1.46%")

    print(f"\n  INTERPRETACIÓN:")
    ta = results.get("staged_production", {}).get("n_trades", 0)
    tb = results.get("staged_nogate", {}).get("n_trades", 0)
    tc = results.get("finetuned_nogate", {}).get("n_trades", 0)

    if tb > ta * 2:
        print(f"  → El gate eval_threshold_scale={eval_threshold_scale:.3f} es demasiado estricto en 2026.")
        print(f"    B tiene {tb:.0f} trades vs {ta:.0f} de A. Considera subir eval_threshold_scale.")
    else:
        print(f"  → El gate NO es el cuello de botella principal (A={ta:.0f}, B={tb:.0f}).")

    if tc > 0 and tb > 0:
        if tb > tc * 1.2:
            print(f"  → El fine-tuning degradó la policy: staged ({tb:.0f}) > finetuned ({tc:.0f}) sin gate.")
        elif tc > tb * 1.2:
            print(f"  → El fine-tuning mejoró la policy: finetuned ({tc:.0f}) > staged ({tb:.0f}) sin gate.")
        else:
            print(f"  → Fine-tuning tuvo efecto neutro en actividad ({tb:.0f} vs {tc:.0f} sin gate).")

    dump_json(out_root / "summary_staged_check.json", {
        "timestamp": datetime.now().isoformat(),
        "staged_policy": str(staged_policy),
        "period": [fwd_check_from, fwd_check_to],
        "results": results,
    })
    print(f"\n  Logs y métricas en: {out_root}")


if __name__ == "__main__":
    main()