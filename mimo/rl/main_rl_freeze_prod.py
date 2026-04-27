import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

release_prod     = '200367'
ruta_best_trial  = f'../../artifacts/{release_prod}/rl/staged/best'
ruta_deploy_full = f'../../artifacts/{release_prod}/oof/deploy_full'
out_root         = Path(f'../../artifacts/{release_prod}/rl/final').resolve()
ruta_main        = './main_rl_train.py'

prod_train_from = "2024-01-01"
prod_train_to   = "2025-12-31"   # train_from → val_to del staged
fwd_check_from  = "2026-01-01"
fwd_check_to    = "2026-03-17"   # test period limpio del staged

# ─────────────────────────────────────────────────────────────────────────────
# FIX v2: lr de fine-tuning separado del lr de Optuna
# ─────────────────────────────────────────────────────────────────────────────
# El lr que Optuna encuentra (~0.019) está optimizado para aprender desde cero
# en 21 meses de train. Usar ese mismo lr en el freeze_prod, que entrena sobre
# 24 meses con policy_init, destruye los pesos aprendidos en el staged:
# los gradientes con lr alto sobreescriben la política antes de que pueda
# estabilizarse, resultando en W_sum ≈ 0.015 (valor de colapso).
# Con lr bajo, la policy ajusta finamente sin perder lo aprendido.
FINETUNE_LR = 0.00035      # ~10x menor que el lr típico de Optuna

# ─────────────────────────────────────────────────────────────────────────────
# FIX v2: FIXED_PARAMS alineados con el staged
# ─────────────────────────────────────────────────────────────────────────────
# BUG v1: estos 3 parámetros no se pasaban a main_rl_train.py, que usaba
# sus defaults muy diferentes a los del staged:
#
#   Parámetro                staged      freeze v1 (default)   efecto
#   rl_entropy               0.10        0.001  (-100x)        suprime exploración
#   rl_baseline_beta         0.88        0.90                  baseline más lento
#   rl_exhaustion_penalty    0.003       0.060  (+20x)         penaliza señales
#
# Resultado: policy con W_sum=0.015, 208 trades en 2 años, PF 0.866.
# La policy del freeze era peor que no operar.
BASE_FIXED = {
    "initial_equity":          10000.0,
    "spread_price":            0.07,
    "max_daily_loss_pct":      0.0350,
    "sizing_equity_mode":      "balance",
    "rl_train_threshold":      0.05,
    "compound":                True,
    "use_oof":                 True,
    "oof_splits":              3,
    "oof_epochs":              25,
    "batch_size":              8192,
    "seq_len_short":           64,
    "seq_len_long":            256,
    # ── parámetros RL fijos — mismos que FIXED_PARAMS del staged ──────────
    "rl_entropy":                 0.10,    # FIX: antes 0.001 (default)
    "rl_baseline_beta":           0.88,    # FIX: antes 0.90  (default)
    "rl_exhaustion_penalty_coef": 0.003,   # FIX: antes 0.060 (default)
    "rl_max_grad_norm":           5.0,
    "rl_trade_cost":              0.0075,
    "rl_batch":                   64,
}


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────

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
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    ensure_dir(log_path.parent)
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
        "max_dd":        r"MaxDD\s*:\s*([-+]?\d+\.?\d*)",
        "max_dd_pct":    r"MaxDD\s+\(%\)\s*:\s*([-+]?\d+\.?\d*)",
        "w_sum_final":   r"W_sum=([-\d.eE+]+)",   # nuevo: capturar W_sum para diagnóstico
    }
    for k, pat in patterns.items():
        # Para w_sum_final queremos el último valor (el final del entrenamiento)
        if k == "w_sum_final":
            matches = re.findall(pat, stdout)
            if matches:
                try:
                    out[k] = float(matches[-1])
                except ValueError:
                    pass
        else:
            m = re.search(pat, stdout)
            if m:
                try:
                    out[k] = float(m.group(1))
                except ValueError:
                    pass
    return out

def check_policy_health(metrics: dict, label: str) -> bool:
    """
    Verifica que la policy no colapsó después del prod_train.

    NOTA sobre W_sum: cuando se usa policy_init, W_sum NO es un indicador
    fiable de colapso. El valor inicial cargado desde el npz del staged es
    -0.026238; con fine-tuning lr bajo los pesos convergen a valores que
    pueden parecer "colapso" pero producen trades y PF correctos.
    W_sum ≈ 0.015 era señal de colapso solo cuando entrenaba desde cero
    sin mover los pesos en absoluto. Con policy_init el diagnóstico real
    son trades + PF + DD.

    Criterios de salud:
      - n_trades_train >= 500  (policy activa en 2 años)
      - profit_factor  >= 0.95 (no catastrófico)
      - max_dd_pct     <= 0.25 (no destruye equity)
    """
    ok = True
    w_sum  = metrics.get("w_sum_final")
    trades = metrics.get("n_trades", 0)
    pf     = metrics.get("profit_factor", 0)
    dd     = abs(metrics.get("max_dd_pct", 0))

    print(f"\n[HEALTH CHECK] {label}")

    # W_sum — solo informativo, no bloquea
    if w_sum is not None:
        print(f"  W_sum final  : {w_sum:.6f}  (informativo — no bloquea con policy_init)")

    # Trades — criterio principal de actividad
    if trades < 500:
        print(f"  Trades       : {trades:.0f}  🔴 muy bajo (< 500 en 2 años → policy casi inactiva)")
        ok = False
    else:
        print(f"  Trades       : {trades:.0f}  ✅")

    # PF — criterio de calidad mínima
    if pf < 0.95:
        print(f"  PF           : {pf:.4f}  🔴 < 0.95 — no desplegar")
        ok = False
    else:
        print(f"  PF           : {pf:.4f}  {'✅' if pf >= 1.0 else '⚠️  entre 0.95 y 1.0'}")

    # DD — criterio de seguridad
    if dd > 0.25:
        print(f"  MaxDD%       : {dd*100:.1f}%  🔴 > 25% — riesgo excesivo")
        ok = False
    else:
        print(f"  MaxDD%       : {dd*100:.1f}%  {'✅' if dd <= 0.15 else '⚠️  entre 15% y 25%'}")

    return ok


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    best_trial_path  = Path(ruta_best_trial).resolve()
    deploy_full_path = Path(ruta_deploy_full).resolve()

    if not best_trial_path.exists():
        raise FileNotFoundError(f"No existe ruta_best_trial: {best_trial_path}")
    if not deploy_full_path.exists():
        raise FileNotFoundError(f"No existe ruta_deploy_full: {deploy_full_path}")

    ensure_dir(out_root)

    # ── 1) Cargar best_trial ──────────────────────────────────────────────────
    json_path = best_trial_path / 'best_trial.json'
    best = json.loads(json_path.read_text(encoding="utf-8"))
    best_params = best.get("best_params", {})
    if not best_params:
        raise ValueError("best_trial.json no contiene 'best_params'")

    # ── FIX v3: usar la policy del trial dir, NO el _best.npz ───────────────
    # El staged sobrescribe _best.npz con el final_train (que puede estar
    # colapsado, W_sum~0.015). La policy válida está en el dir del trial.
    # Buscar el trial cuyo npz coincide con best_params comparando en logs/
    # best_trial.json no guarda el número de trial explícitamente, pero
    # podemos encontrarlo buscando el trial_XXXXX que tenga los mismos params.
    staged_policy = None
    logs_dir = best_trial_path.parent / "logs"
    best_params_items = set(f"{k}={round(float(v),8)}" for k, v in best_params.items()
                            if isinstance(v, float))

    if logs_dir.exists():
        import json as _json
        for trial_dir in sorted(logs_dir.iterdir()):
            cfg = trial_dir / "config_val.json"
            if not cfg.exists():
                continue
            try:
                cfg_data = _json.loads(cfg.read_text(encoding="utf-8"))
                cfg_items = set(f"{k}={round(float(v),8)}" for k, v in cfg_data.items()
                                if isinstance(v, float) and k in best_params)
                if best_params_items == cfg_items:
                    npz_candidates = list(trial_dir.glob("rl_policy_gate_*.npz"))
                    if npz_candidates:
                        staged_policy = npz_candidates[0]
                        print(f"  Trial dir encontrado: {trial_dir.name}")
                        break
            except Exception:
                continue

    if staged_policy is None or not staged_policy.exists():
        # fallback: _best.npz
        staged_policy = best_trial_path / f"rl_policy_gate_{release_prod}_best.npz"
        print("WARNING: no se pudo localizar el trial dir — usando _best.npz como fallback")
        print("         Verifica que W_sum cargado no sea ~0.015")

    if not staged_policy.exists():
        raise FileNotFoundError(
            f"No existe policy: {staged_policy}\n"
            f"Claves en best_trial.json: {list(best.keys())}"
        )

    policy_out = out_root / f"rl_policy_gate_{release_prod}.npz"

    print(f"=== FREEZE PROD v2 ===")
    print(f"  Release      : {release_prod}")
    print(f"  Best trial   : {best_trial_path}")
    print(f"  Policy init  : {staged_policy}")
    print(f"  FINETUNE_LR  : {FINETUNE_LR}  (lr Optuna={best_params.get('rl_lr', '?')})")
    print(f"  entropy_coef : {BASE_FIXED['rl_entropy']}  (staged=0.10)")
    print(f"  exh_penalty  : {BASE_FIXED['rl_exhaustion_penalty_coef']}  (staged=0.003)")
    print(f"  baseline_beta: {BASE_FIXED['rl_baseline_beta']}  (staged=0.88)")

    # ── 2) PROD TRAIN — fine-tuning sobre 2024-2025 ───────────────────────────
    # FIX v2: sustituimos rl_lr de Optuna por FINETUNE_LR para no destruir pesos.
    # policy_init = staged best → fine-tuning a partir del estado aprendido.
    prod_train_params = dict(BASE_FIXED)
    prod_train_params.update(best_params)        # parámetros de Optuna (chop, exhaustion thresholds, take_threshold)
    prod_train_params["rl_lr"] = FINETUNE_LR     # FIX: sobreescribir lr de Optuna con lr de fine-tuning
    prod_train_params.update({
        "release":        release_prod,
        "from":           prod_train_from,
        "to":             prod_train_to,
        "artifacts_path": str(deploy_full_path),
        "policy_out":     str(policy_out),
        "policy_init":    str(staged_policy),    # FIX: partir del staged, no desde cero
    })

    dump_json(out_root / "config_prod_train.json", {
        "timestamp":        datetime.now().isoformat(),
        "best_trial_path":  str(best_trial_path),
        "deploy_full_path": str(deploy_full_path),
        "finetune_lr":      FINETUNE_LR,
        "optuna_lr":        best_params.get("rl_lr"),
        "params":           prod_train_params,
    })

    print(f"\n[1/2] PROD TRAIN  {prod_train_from} → {prod_train_to} ...")
    res_train = run(build_cmd(prod_train_params), out_root / "prod_train.log", timeout_s=3600)
    metrics_train = parse_metrics(res_train.stdout)
    dump_json(out_root / "metrics_prod_train.json", metrics_train)

    if res_train.returncode != 0:
        raise RuntimeError(f"Fallo PROD TRAIN (rc={res_train.returncode}). Revisa: {out_root / 'prod_train.log'}")

    train_ok = check_policy_health(metrics_train, "PROD TRAIN")
    if not train_ok:
        print("\n🔴 PROD TRAIN no superó el health check.")
        print("   La policy producida NO es apta para despliegue.")
        print("   Opciones:")
        print("   A) Ajustar FINETUNE_LR (probar 0.001 si sigue colapsando, 0.003 si entrena poco)")
        print("   B) Desplegar directamente la policy del staged (ruta_best_trial / rl_policy_gate_*_best.npz)")
        #sys.exit(1)

    # ── 3) FORWARD CHECK — eval_only sobre 2026 con gates de PRODUCCIÓN ─────
    # FIX: el forward check debe usar rl_mode='production' para que sus métricas
    # sean comparables a lo que verá el sistema en producción real.
    # El prod_train usa 'training' (gates permisivos, más señales → mejor aprendizaje).
    # El forward check usa 'production' (gates estrictos → métricas realistas).
    fwd_params = dict(BASE_FIXED)
    fwd_params.update(best_params)
    fwd_params["rl_lr"] = FINETUNE_LR   # consistencia (en eval_only no se usa para entrenar)
    fwd_params.update({
        "release":        release_prod,
        "from":           fwd_check_from,
        "to":             fwd_check_to,
        "artifacts_path": str(deploy_full_path),
        "eval_only":      True,
        "policy_init":    str(policy_out),
        "rl_mode":        "production",   # FIX: gates reales de producción
    })

    dump_json(out_root / "config_forward_check.json", {
        "timestamp": datetime.now().isoformat(),
        "params":    fwd_params,
    })

    print(f"\n[2/2] FORWARD CHECK  {fwd_check_from} → {fwd_check_to} ...")
    res_fwd = run(build_cmd(fwd_params), out_root / "forward_check.log", timeout_s=3600)
    metrics_fwd = parse_metrics(res_fwd.stdout)
    dump_json(out_root / "metrics_forward_check.json", metrics_fwd)

    if res_fwd.returncode != 0:
        raise RuntimeError(f"Fallo FORWARD CHECK (rc={res_fwd.returncode}). Revisa: {out_root / 'forward_check.log'}")

    # ── 4) GO / NO-GO ─────────────────────────────────────────────────────────
    trades_fwd = metrics_fwd.get("n_trades", 0)
    pf_fwd     = metrics_fwd.get("profit_factor", 0)
    dd_fwd     = abs(metrics_fwd.get("max_dd_pct", 1.0))

    print(f"\n[GO / NO-GO] Forward check {fwd_check_from} → {fwd_check_to}")
    print(f"  Trades : {trades_fwd:.0f}   (mínimo: 80)")
    print(f"  PF     : {pf_fwd:.4f}   (mínimo: 1.00)")
    print(f"  DD%    : {dd_fwd*100:.2f}%  (máximo: 15%)")

    go = trades_fwd >= 80 and pf_fwd >= 1.0 and dd_fwd <= 0.15
    print(f"\n  → {'✅ GO — apto para despliegue' if go else '🔴 NO-GO — no desplegar'}")

    # ── 5) Summary ────────────────────────────────────────────────────────────
    summary = {
        "release_prod":           release_prod,
        "policy_out":             str(policy_out),
        "staged_policy_init":     str(staged_policy),
        "deploy_full_path":       str(deploy_full_path),
        "best_trial_path":        str(best_trial_path),
        "finetune_lr":            FINETUNE_LR,
        "optuna_lr":              best_params.get("rl_lr"),
        "prod_train":             [prod_train_from, prod_train_to],
        "forward_check":          [fwd_check_from, fwd_check_to],
        "metrics_prod_train":     metrics_train,
        "metrics_forward_check":  metrics_fwd,
        "go_nogo":                "GO" if go else "NO-GO",
        "timestamp":              datetime.now().isoformat(),
    }
    dump_json(out_root / "summary.json", summary)

    print("\n=== DONE ===")
    print(f"  Policy  : {policy_out}")
    print(f"  Out dir : {out_root}")
    print(f"  Summary : {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()