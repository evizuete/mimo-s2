"""
sweep_eval_threshold.py

Barre eval_threshold_scale en el rango [0.05, 0.50] para encontrar
el valor que maximiza PF en el forward check 2026 con la policy staged.

CONTEXTO:
  effective_take_threshold = rl_take_threshold * eval_threshold_scale
  rl_take_threshold (Optuna) = 0.2810

  Con escala=0.341 → umbral=0.0959 → 60 trades, PF=0.876
  Con escala=1.0   → umbral=0.2810 → 0 trades (señales 2026 demasiado débiles)

  Buscamos la escala que maximiza el balance trades/PF.

USO:
    python sweep_eval_threshold.py

Genera:
  artifacts/200367/rl/sweep_threshold/
    sweep_results.json    — tabla completa
    sweep_summary.txt     — resumen imprimible
    fwd_scale_XXXX.log    — log de cada run
"""

import json
import subprocess
import sys
from pathlib import Path
from datetime import datetime

# ─── CONFIG ────────────────────────────────────────────────────────────────
release_prod     = '200367'
ruta_best_trial  = f'../../artifacts/{release_prod}/rl/staged/best'
ruta_deploy_full = f'../../artifacts/{release_prod}/oof/deploy_full'
out_root         = Path(f'../../artifacts/{release_prod}/rl/sweep_threshold').resolve()
ruta_main        = './main_rl_train.py'

fwd_check_from  = "2026-01-01"
fwd_check_to    = "2026-03-17"

# Policy a usar: staged (trial_00029) — sin fine-tuning
# Cambia a la ruta de prod_train si quieres comparar con la fine-tuned
POLICY_PATH = f'../../artifacts/{release_prod}/rl/staged/logs/trial_00029/rl_policy_gate_{release_prod}_trial_00029.npz'

# Escalas a probar — de más permisivo a más restrictivo
# Con 0.341 hay 60 trades; con 1.0 hay 0. Exploramos el espacio intermedio
# y también por debajo de 0.341 para ver si hay más trades con mejor PF.
SCALES = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.23, 0.25,
          0.28, 0.30, 0.34, 0.38, 0.42, 0.46, 0.50]

BASE_FIXED = {
    "initial_equity":      10000.0,
    "spread_price":        0.07,
    "max_daily_loss_pct":  0.0350,
    "sizing_equity_mode":  "balance",
    "compound":            True,
    "use_oof":             True,
    "oof_splits":          3,
    "oof_epochs":          25,
    "batch_size":          8192,
    "seq_len_short":       64,
    "seq_len_long":        256,
    "rl_trade_cost":       0.0075,
    "rl_train_threshold":  0.08,
    "rl_max_grad_norm":    2.0,
    "rl_batch":            64,
    "rl_baseline_beta":    0.92,
    "rl_vol_filter":       "none",
    "rl_vol_filter_mult":  1.5,
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

def run(cmd: list, log_path: Path, timeout_s: int = 1800) -> subprocess.CompletedProcess:
    ensure_dir(log_path.parent)
    res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    with log_path.open("w", encoding="utf-8") as f:
        f.write("=== CMD ===\n" + " ".join(cmd) + "\n\n")
        f.write("=== STDOUT ===\n" + res.stdout + "\n\n")
        f.write("=== STDERR ===\n" + res.stderr + "\n\n")
        f.write("=== RET ===\n" + str(res.returncode) + "\n")
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
        import re as _re
        m = _re.search(pat, stdout)
        if m:
            try:
                out[k] = float(m.group(1))
            except ValueError:
                pass
    return out


# ─── MAIN ──────────────────────────────────────────────────────────────────

def main():
    best_trial_path  = Path(ruta_best_trial).resolve()
    deploy_full_path = Path(ruta_deploy_full).resolve()
    policy_path      = Path(POLICY_PATH).resolve()

    if not best_trial_path.exists():
        raise FileNotFoundError(f"No existe: {best_trial_path}")
    if not deploy_full_path.exists():
        raise FileNotFoundError(f"No existe: {deploy_full_path}")
    if not policy_path.exists():
        raise FileNotFoundError(f"No existe policy: {policy_path}")

    ensure_dir(out_root)

    # Cargar best_params
    json_path = best_trial_path / 'best_trial.json'
    best = json.loads(json_path.read_text(encoding="utf-8"))
    best_params = best.get("best_params", {})
    rl_take_threshold = best_params.get("rl_take_threshold", 0.2810)

    print(f"=== SWEEP eval_threshold_scale ===")
    print(f"  Policy          : {policy_path.name}")
    print(f"  rl_take_threshold: {rl_take_threshold:.4f}")
    print(f"  Periodo          : {fwd_check_from} → {fwd_check_to}")
    print(f"  Escalas a probar : {SCALES}")
    print(f"  Out dir          : {out_root}")
    print(f"  Referencia (staged test Jul-Dic25): 695 trades | PF 1.191 | WR 44.6%")
    print()

    results = []

    for scale in SCALES:
        effective = rl_take_threshold * scale
        print(f"  [{scale:.2f}] umbral efectivo={effective:.4f} ...", end=" ", flush=True)

        params = dict(BASE_FIXED)
        params.update(best_params)
        params.update({
            "release":               release_prod,
            "from":                  fwd_check_from,
            "to":                    fwd_check_to,
            "artifacts_path":        str(deploy_full_path),
            "eval_only":             True,
            "policy_init":           str(policy_path),
            "rl_mode":               "production",
            "rl_eval_threshold_scale": scale,
        })

        log_name = f"fwd_scale_{str(scale).replace('.', '')}.log"
        res = run(build_cmd(params), out_root / log_name)

        m = parse_metrics(res.stdout)
        trades = m.get("n_trades", 0)
        pf     = m.get("profit_factor", 0.0)
        wr     = m.get("win_rate", 0.0)
        dd     = abs(m.get("max_dd_pct", 0.0))
        pnl    = m.get("net_pnl", 0.0)

        result = {
            "scale":             scale,
            "effective_threshold": round(effective, 5),
            "n_trades":          trades,
            "profit_factor":     pf,
            "win_rate":          wr,
            "max_dd_pct":        dd,
            "net_pnl":           pnl,
            "go_nogo": (
                "GO ✅" if trades >= 100 and pf >= 1.05 and dd <= 0.12
                else "NO-GO ❌"
            ),
        }
        results.append(result)

        status = result["go_nogo"]
        print(f"trades={trades:4.0f}  PF={pf:.3f}  WR={wr*100:.1f}%  DD={dd*100:.2f}%  {status}")

    # ── Guardar resultados ──────────────────────────────────────────────────
    dump_json(out_root / "sweep_results.json", {
        "timestamp":        datetime.now().isoformat(),
        "policy":           str(policy_path),
        "rl_take_threshold": rl_take_threshold,
        "period":           [fwd_check_from, fwd_check_to],
        "results":          results,
    })

    # ── Resumen imprimible ──────────────────────────────────────────────────
    go_results   = [r for r in results if "GO ✅" in r["go_nogo"]]
    best_pf      = max(results, key=lambda r: r["profit_factor"])
    best_trades  = max(results, key=lambda r: r["n_trades"])
    # Score compuesto: penaliza pocas trades y premia PF
    def score(r):
        if r["n_trades"] < 30:
            return -999
        return r["profit_factor"] * min(1.0, r["n_trades"] / 100)
    best_combined = max(results, key=score)

    lines = []
    lines.append(f"\n{'='*65}")
    lines.append(f"RESUMEN SWEEP — {fwd_check_from} → {fwd_check_to}")
    lines.append(f"{'='*65}")
    lines.append(f"  {'scale':>6}  {'umbral_ef':>9}  {'trades':>7}  {'PF':>7}  {'WR%':>5}  {'DD%':>5}  {'go/no-go'}")
    lines.append(f"  {'-'*63}")
    for r in results:
        marker = " ◄" if r["scale"] == best_combined["scale"] else ""
        lines.append(
            f"  {r['scale']:>6.2f}  {r['effective_threshold']:>9.5f}"
            f"  {r['n_trades']:>7.0f}  {r['profit_factor']:>7.4f}"
            f"  {r['win_rate']*100:>5.1f}  {r['max_dd_pct']*100:>5.2f}"
            f"  {r['go_nogo']}{marker}"
        )
    lines.append(f"\n  Mejor PF       : scale={best_pf['scale']:.2f}  PF={best_pf['profit_factor']:.4f}  trades={best_pf['n_trades']:.0f}")
    lines.append(f"  Más trades     : scale={best_trades['scale']:.2f}  trades={best_trades['n_trades']:.0f}  PF={best_trades['profit_factor']:.4f}")
    lines.append(f"  Mejor combinado: scale={best_combined['scale']:.2f}  PF={best_combined['profit_factor']:.4f}  trades={best_combined['n_trades']:.0f}  ◄ RECOMENDADO")

    if go_results:
        lines.append(f"\n  Escalas que pasan GO/NO-GO (trades≥100, PF≥1.05, DD≤12%):")
        for r in go_results:
            lines.append(f"    scale={r['scale']:.2f}  trades={r['n_trades']:.0f}  PF={r['profit_factor']:.4f}")
    else:
        lines.append(f"\n  ⚠️  Ninguna escala supera los criterios GO/NO-GO actuales.")
        lines.append(f"     Considera relajar los criterios o revisar la policy.")

    summary_text = "\n".join(lines)
    print(summary_text)
    (out_root / "sweep_summary.txt").write_text(summary_text, encoding="utf-8")
    print(f"\n  Resultados guardados en: {out_root}")


if __name__ == "__main__":
    main()