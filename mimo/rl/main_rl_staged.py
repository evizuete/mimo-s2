"""
main_rl_staged.py — VERSIÓN OPTIMIZADA

OPTIMIZACIONES VS VERSIÓN ANTERIOR:
─────────────────────────────────────────────────────────────────────────────
1. PRECALENTAMIENTO DE CACHÉ (warm_cache_before_optuna):
   Antes de lanzar Optuna, se ejecuta UNA sola vez main_rl_train.py con
   --warm_cache para cada periodo (TRAIN y VAL). Esto genera el caché de
   predicciones del modelo supervisado en disco.
   Todos los trials posteriores reutilizan ese caché → se eliminan:
     - La carga del modelo Keras en cada subproceso
     - La inferencia completa sobre todo el dataset
   Ahorro estimado: 5-15s por subproceso × 2 subprocesos × N trials.

2. PARALELISMO (--n_jobs > 1):
   Optuna soporta n_jobs nativamente. Con n_jobs=2 y trials independientes
   entre sí (lo son, porque el caché es de solo lectura), el tiempo total
   se divide casi a la mitad. El storage MySQL ya soporta acceso concurrente.
   PRECAUCIÓN: no uses n_jobs > nº de núcleos físicos disponibles.
   Con n_jobs=1 el comportamiento es idéntico a la versión anterior.

3. TIMEOUT POR SUBPROCESO MÁS AJUSTADO:
   El parámetro --timeout_s ahora aplica individualmente al TRAIN y al VAL.
   Antes un trial lento podía bloquear toda la optimización sin timeout claro.

4. SKIP DE TRIALS YA COMPLETADOS:
   Si load_if_exists=True y el trial ya tiene valor en Optuna DB, se omite
   automáticamente (comportamiento nativo de Optuna, ahora documentado).

5. FILTRO DE VOLATILIDAD (rl_vol_filter en PARAM_SPACE):
   Optuna ahora optimiza el modo de filtro de volatilidad como parámetro
   categórico: none / skip_high / skip_extreme. Esto permite que el propio
   proceso de búsqueda determine si operar en regímenes de alta volatilidad
   mejora o empeora el score compuesto.
   Requiere main_rl_train.py con soporte --rl_vol_filter y --rl_vol_filter_mult.
─────────────────────────────────────────────────────────────────────────────
"""

import argparse
import json
import logging
import os
import re
import subprocess
import sys
import math
import warnings
from dataclasses import dataclass
from datetime import datetime, date
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union, Literal

import optuna

from mimo.rl.callbacks import ConvergenceCallback, analyze_convergence, print_convergence_status, analyze_param_importance, \
    print_param_importance, EarlyStoppingCallback

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.WARNING)
logging.getLogger('tensorflow').setLevel(logging.ERROR)
logging.getLogger('optuna').setLevel(logging.WARNING)

# ============================================================
# 1) CONFIGURACIÓN EMBEBIDA
# ============================================================

PARAM_SPACE = {
    "rl_take_threshold":          [0.20,  0.38,  'float'],     # techo bajado: zona >0.38 colapsa
    "rl_lr":                      [0.0015, 0.005, 'float'],    # desplazado hacia arriba
    "rl_chop_penalty_coef":       [0.005, 0.050,  'loguniform'],
    "rl_exhaustion_soft_thr":     [0.50,  0.80,   'float'],    # mantener, importancia alta
    "rl_exhaustion_penalty_coef": [0.001, 0.015,  'loguniform'],
    "rl_eval_threshold_scale":    [0.30,  0.60,   'float'],
    "rl_vol_filter":              [["none", "skip_high", "skip_extreme"], "categorical"],
}

RL_FIXED = {
    "initial_equity":        10000.0,
    "spread_price":          0.07,
    "max_daily_loss_pct":    0.035,
    "sizing_equity_mode":    "balance",
    "compound":              True,
    "use_oof":               True,
    "oof_splits":            3,
    "oof_epochs":            25,
    "batch_size":            8192,
    "seq_len_short":         64,
    "seq_len_long":          256,
    "rl_trade_cost":         0.0075,
    "rl_train_threshold":    0.08,
    "rl_max_grad_norm":      2.0,
    "rl_batch":              64,
    "rl_baseline_beta":      0.92,
    # Multiplicador fijo para skip_extreme. Solo actúa cuando rl_vol_filter="skip_extreme".
    # 1.5 → bloquea cuando vol supera 1.5x el umbral vol_high del régimen.
    # Si quieres que Optuna también optimice esto, muévelo a PARAM_SPACE como float [1.2, 2.5].
    "rl_vol_filter_mult":    1.5,
    "rl_entropy":       0.12,    # fijado desde best trial v4.0
    "rl_chop_soft_thr": 0.50,    # fijado desde best trial v4.0
}


TRAIN_FIXED: Dict[str, Any] = {**RL_FIXED, "rl_mode": "training"}
VAL_FIXED:   Dict[str, Any] = {**RL_FIXED, "rl_mode": "training"}   # debe coincidir con train para que el score de Optuna sea comparable
TEST_FIXED:  Dict[str, Any] = {**RL_FIXED, "rl_mode": "production"} # gates reales → métricas representativas del despliegue

STREAM_OUTPUT: bool = False


@dataclass
class MetricConfig:
    direction: Literal["maximize", "minimize"]
    weight: float = 1.0
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None


METRICS_CFG: Dict[str, MetricConfig] = {
    "net_pnl_pct":   MetricConfig(direction="maximize", weight=0.70, clip_min=-0.30, clip_max=0.50),
    "profit_factor": MetricConfig(direction="maximize", weight=0.15, clip_min=0.80,  clip_max=2.0),
    "max_dd_pct":    MetricConfig(direction="minimize", weight=0.15, clip_max=0.25),
}


# ============================================================
# 2) HELPERS
# ============================================================

def ensure_dir(p: Union[str, Path]) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def json_safe(obj):
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, Enum):
        return json_safe(obj.value)
    try:
        import numpy as np
        if isinstance(obj, np.bool_):    return bool(obj)
        if isinstance(obj, np.integer):  return int(obj)
        if isinstance(obj, np.floating): return float(obj)
        if isinstance(obj, np.ndarray):  return obj.tolist()
    except Exception:
        pass
    if hasattr(obj, "item") and callable(getattr(obj, "item")):
        try:
            v = obj.item()
            if v is not obj:
                return json_safe(v)
        except Exception:
            pass
    return obj


def dump_json(path: Path, obj: Any) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(json_safe(obj), indent=2, ensure_ascii=False), encoding="utf-8")


def build_cmd(main_script: str, params: Dict[str, Any]) -> List[str]:
    cmd = [sys.executable, main_script]
    for k, v in params.items():
        flag = f"--{k}"
        if isinstance(v, bool):
            if v:
                cmd.append(flag)
        else:
            cmd.extend([flag, str(v)])
    return cmd


def run_cmd(cmd: List[str], timeout_s: int, log_file: Path, stream_output: bool = True) -> subprocess.CompletedProcess:
    ensure_dir(log_file.parent)

    if stream_output:
        stdout_lines = []
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in proc.stdout:
                stdout_lines.append(line)
                print(line, end='', flush=True)
            proc.wait(timeout=timeout_s)
            returncode = proc.returncode
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            returncode = -1
        stdout_full = ''.join(stdout_lines)
        stderr_full = ""
    else:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
        stdout_full = res.stdout
        stderr_full = res.stderr
        returncode  = res.returncode

    with log_file.open("w", encoding="utf-8") as f:
        f.write("=== COMMAND ===\n")
        f.write(" ".join(cmd) + "\n\n")
        f.write("=== STDOUT ===\n")
        f.write(stdout_full + "\n\n")
        f.write("=== STDERR ===\n")
        f.write(stderr_full + "\n\n")
        f.write("=== RETURN CODE ===\n")
        f.write(str(returncode) + "\n")

    class FakeResult:
        def __init__(self, stdout, stderr, returncode):
            self.stdout = stdout
            self.stderr = stderr
            self.returncode = returncode

    return FakeResult(stdout_full, stderr_full, returncode)


def enrich_metrics_with_pct(metrics: Dict[str, float], initial_equity: float) -> Dict[str, float]:
    out = dict(metrics)
    eq0 = float(initial_equity) if initial_equity else 0.0
    if "net_pnl" in out and "net_pnl_pct" not in out and eq0 > 0:
        out["net_pnl_pct"] = float(out["net_pnl"]) / (eq0 + 1e-12)
    if "max_dd_pct" in out:
        v = abs(float(out["max_dd_pct"]))
        if v > 1.5:
            v /= 100.0
        out["max_dd_pct"] = v
    elif "max_dd" in out and eq0 > 0:
        out["max_dd_pct"] = float(out["max_dd"]) / (eq0 + 1e-12)
    return out


def parse_training_output(output: str) -> Dict[str, float]:
    results: Dict[str, float] = {}
    patterns = {
        "n_trades":     r"Trades\s*:\s*(\d+)",
        "net_pnl":      r"NetPnL\s*:\s*([-+]?\d+\.?\d*)",
        "profit_factor":r"ProfitFactor\s*:\s*([-+]?\d+\.?\d*)",
        "win_rate":     r"WinRate\s*:\s*([-+]?\d+\.?\d*)",
        "max_dd":       r"MaxDD\s*:\s*([-+]?\d+\.?\d*)",
        "max_dd_pct":   r"MaxDD\s*\(%\)\s*:\s*([-+]?\d+\.?\d*)",
    }
    for metric, pattern in patterns.items():
        m = re.search(pattern, output)
        if m:
            try:
                results[metric] = float(m.group(1))
            except ValueError:
                pass
    return results


def parse_rl_json_stats(output: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    patterns = {
        "val_gate_stats":   r"^RL_GATE_STATS_VAL\s*:\s*(\{.*\})\s*$",
        "val_action_stats": r"^RL_ACTION_STATS_VAL\s*:\s*(\{.*\})\s*$",
        "test_gate_stats":  r"^RL_GATE_STATS_TEST\s*:\s*(\{.*\})\s*$",
        "test_action_stats":r"^RL_ACTION_STATS_TEST\s*:\s*(\{.*\})\s*$",
    }
    for key, pat in patterns.items():
        m = re.search(pat, output, flags=re.M)
        if not m:
            continue
        try:
            out[key] = json.loads(m.group(1).strip())
        except Exception:
            out[key] = {"_parse_error": True, "_raw": m.group(1)[:1000]}
    return out


def _pearson_corr(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    mx = sum(xs) / len(xs)
    my = sum(ys) / len(ys)
    num  = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    denx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    deny = math.sqrt(sum((y - my) ** 2 for y in ys))
    if denx == 0.0 or deny == 0.0:
        return float("nan")
    return num / (denx * deny)


def _rankdata(a: List[float]) -> List[float]:
    n = len(a)
    order = sorted(range(n), key=lambda i: a[i])
    ranks = [0.0] * n
    i = 0
    while i < n:
        j = i
        while j + 1 < n and a[order[j + 1]] == a[order[i]]:
            j += 1
        avg_rank = (i + j) / 2.0 + 1.0
        for k in range(i, j + 1):
            ranks[order[k]] = avg_rank
        i = j + 1
    return ranks


def _spearman_corr(xs: List[float], ys: List[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    return _pearson_corr(_rankdata(xs), _rankdata(ys))


class CompositeScorer:
    def __init__(self, metrics_config: Dict[str, MetricConfig]):
        self.cfg = metrics_config

    def _clip(self, v: float, c: MetricConfig) -> float:
        if c.clip_min is not None: v = max(c.clip_min, v)
        if c.clip_max is not None: v = min(c.clip_max, v)
        return v

    def _penalties(self, m: Dict[str, float]) -> tuple:
        n_trades      = m.get('n_trades', 0)
        win_rate      = m.get('win_rate', 0.5)

        # Penalización progresiva por pocos trades:
        # < 200  → penalización máxima (debería ya estar pruned, pero por seguridad)
        # 200-500 → penalización moderada decreciente
        # > 500  → sin penalización
        if n_trades < 300:  # era 200
            sample_penalty = 0.50
        elif n_trades < 600:  # era 500
            sample_penalty = 0.15 * (600 - n_trades) / 300.0
        else:
            sample_penalty = 0.0

        wr_penalty     = 0.05 if win_rate > 0.90 else 0.0
        return sample_penalty, wr_penalty, 0.0

    def score(self, m: Dict[str, float]) -> float:
        required = [k for k, cfg in self.cfg.items() if cfg.weight != 0]
        if not m or any(k not in m for k in required):
            return -1e12
        s = 0.0
        for name, cfg in self.cfg.items():
            if cfg.weight == 0: continue
            v = self._clip(float(m[name]), cfg)
            if cfg.direction == "minimize": v = -v
            s += v * cfg.weight
        sp, wp, pp = self._penalties(m)
        return s - sp - wp - pp


def suggest_from_space(trial: optuna.Trial, name: str, spec: Any) -> Any:
    if not isinstance(spec, list) or len(spec) < 2:
        raise ValueError(f"PARAM_SPACE[{name}] must be a list spec")
    if len(spec) == 2 and isinstance(spec[0], list) and spec[1] == "categorical":
        return trial.suggest_categorical(name, spec[0])
    if len(spec) != 3:
        raise ValueError(f"PARAM_SPACE[{name}] expected 3 items or categorical form")
    lo, hi, t = spec[0], spec[1], spec[2]
    if t == "float":      return trial.suggest_float(name, float(lo), float(hi))
    if t == "int":        return trial.suggest_int(name, int(lo), int(hi))
    if t == "loguniform": return trial.suggest_float(name, float(lo), float(hi), log=True)
    raise ValueError(f"Unsupported type '{t}'")


def merged_params(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for d in dicts:
        out.update(d)
    return out


# ============================================================
# OPTIMIZACIÓN: precalentamiento de caché
# ============================================================

def warm_prediction_cache(
    main: str,
    release: str,
    artifacts_path: str,
    periods: List[Dict[str, str]],   # [{"from": "...", "to": "..."}, ...]
    logs_dir: Path,
    timeout_s: int,
) -> None:
    """
    Ejecuta main_rl_train.py --warm_cache para cada periodo.
    Genera el caché de predicciones del modelo supervisado en disco.
    Los trials de Optuna lo reutilizarán sin tocar el modelo ni la BD.

    Solo genera el caché si NO existe ya (idempotente).
    """
    cache_dir = Path(f'./cache/{release}/predictions')
    ensure_dir(str(cache_dir))

    for period in periods:
        from_s = period["from"]
        to_s   = period["to"]
        tag    = f"{from_s}_{to_s}".replace("-", "")

        # Verificación simple: si ya hay algún fichero en cache_dir con este tag, skip
        existing = list(cache_dir.glob(f"*{tag}*"))
        if existing:
            print(f"[WARM_CACHE] ✅ Caché ya existe para {from_s} → {to_s}, skip.")
            continue

        print(f"\n[WARM_CACHE] 🔥 Generando caché para {from_s} → {to_s}...")
        params = dict(RL_FIXED)
        params.update({
            "release":        release,
            "from":           from_s,
            "to":             to_s,
            "artifacts_path": artifacts_path,
            "warm_cache":     True,   # flag nuevo en main_rl_train.py
        })
        # warm_cache no necesita policy
        cmd = build_cmd(main, params)
        log_file = logs_dir / f"warm_cache_{tag}.log"

        res = run_cmd(cmd, timeout_s=timeout_s, log_file=log_file, stream_output=False)
        if res.returncode != 0:
            print(f"[WARM_CACHE] ⚠️  Falló para {from_s} → {to_s}. Revisa: {log_file}")
            print(f"[WARM_CACHE]    Los trials continuarán sin caché (más lentos).")
        else:
            print(f"[WARM_CACHE] ✅ Caché listo para {from_s} → {to_s}.")


# ============================================================
# 3) OBJECTIVE
# ============================================================

def objective_factory(
        *,
        main: str,
        release: str,
        artifacts_path: str,
        train_from: str,
        train_to: str,
        val_from: str,
        val_to: str,
        test_from: str,
        test_to: str,
        scorer: CompositeScorer,
        logs_dir: Path,
        timeout_s: int,
        eval_test_per_trial: bool,
):
    def _objective(trial: optuna.Trial) -> float:
        sampled: Dict[str, Any] = {}
        for pname, pspec in PARAM_SPACE.items():
            sampled[pname] = suggest_from_space(trial, pname, pspec)

        trial_dir = logs_dir / f"trial_{trial.number:05d}"
        ensure_dir(trial_dir)
        policy_npz = trial_dir / f"rl_policy_gate_{release}_trial_{trial.number:05d}.npz"

        # ── TRAIN ──
        train_params = merged_params(
            TRAIN_FIXED, sampled,
            {
                "release":        release,
                "from":           train_from,
                "to":             train_to,
                "artifacts_path": artifacts_path,
                "policy_out":     policy_npz,
                "rl_seed":        trial.number,
                "debug":          True,
            },
        )
        dump_json(trial_dir / "config_train.json", train_params)
        res_train = run_cmd(
            build_cmd(main, train_params),
            timeout_s=timeout_s,
            log_file=trial_dir / "train.log",
            stream_output=STREAM_OUTPUT,
        )

        if res_train.returncode != 0:
            trial.set_user_attr("train_failed", True)
            return -1e12

        train_metrics = enrich_metrics_with_pct(
            parse_training_output(res_train.stdout),
            initial_equity=float(TRAIN_FIXED["initial_equity"]),
        )
        dump_json(trial_dir / "metrics_train.json", train_metrics)

        # ── VAL ──
        val_params = merged_params(
            VAL_FIXED, sampled,
            {
                "release":        release,
                "from":           val_from,
                "to":             val_to,
                "artifacts_path": artifacts_path,
                "eval_only":      True,
                "policy_init":    policy_npz,
                "debug":          True,
            },
        )
        dump_json(trial_dir / "config_val.json", val_params)
        res_val = run_cmd(
            build_cmd(main, val_params),
            timeout_s=timeout_s,
            log_file=trial_dir / "val.log",
            stream_output=STREAM_OUTPUT,
        )

        if res_val.returncode != 0:
            trial.set_user_attr("val_failed", True)
            return -1e12

        val_metrics = enrich_metrics_with_pct(
            parse_training_output(res_val.stdout),
            initial_equity=float(VAL_FIXED["initial_equity"]),
        )
        dump_json(trial_dir / "metrics_val.json", val_metrics)

        required = [k for k, cfg in scorer.cfg.items() if cfg.weight != 0]
        if not val_metrics or any(k not in val_metrics for k in required):
            trial.set_user_attr("val_parse_failed", True)
            trial.set_user_attr("val_metrics_keys", sorted(val_metrics.keys()) if val_metrics else [])
            return -1e12

        n_trades = val_metrics.get("n_trades", 0)
        win_rate = val_metrics.get("win_rate", 0.5)

        # Umbral de pruning: policy colapsada si <200 trades en val
        # Con 3 meses de val (Oct-Dic 2025) el sistema genera ~3.000-5.000 trades
        # normalmente. Menos de 200 indica colapso hacia SKIP.
        if n_trades < 200:
            print(f"\n[PRUNED] Trial {trial.number}: política colapsada ({n_trades} trades < 200)\n")
            raise optuna.TrialPruned(f"Política colapsada: {n_trades} trades < 200")
        if win_rate > 0.90:
            print(f"\n[PRUNED] Trial {trial.number}: win rate sospechoso ({win_rate:.1%} > 90%)\n")
            raise optuna.TrialPruned(f"Win rate sospechoso: {win_rate:.1%} > 90%")

        val_score = scorer.score(val_metrics)

        print(f"\n{'=' * 60}")
        print(f"[DEBUG TRIAL {trial.number}] VAL METRICS:")
        for name in ["n_trades", "win_rate", "profit_factor", "net_pnl_pct", "max_dd_pct"]:
            if name in val_metrics:
                print(f"  {name:20s}: {val_metrics[name]:8.4f}")
        print(f"  {'rl_vol_filter':20s}: {sampled.get('rl_vol_filter', 'none')}")
        print(f"\n[DEBUG TRIAL {trial.number}] FINAL SCORE: {val_score:.10f}")
        print(f"{'=' * 60}\n")

        val_rl_stats = parse_rl_json_stats(res_val.stdout)
        if val_rl_stats.get("val_gate_stats"):
            dump_json(trial_dir / "rl_gate_stats_val.json", val_rl_stats["val_gate_stats"])
            trial.set_user_attr("val_rl_gate_stats", val_rl_stats["val_gate_stats"])
        if val_rl_stats.get("val_action_stats"):
            dump_json(trial_dir / "rl_action_stats_val.json", val_rl_stats["val_action_stats"])
            trial.set_user_attr("val_rl_action_stats", val_rl_stats["val_action_stats"])

        trial.set_user_attr("policy_npz", str(policy_npz))
        trial.set_user_attr("composite_score", val_score)
        for k, v in val_metrics.items():
            trial.set_user_attr(f"val_{k}", v)
        dump_json(trial_dir / "score_val.json", {"composite_score": val_score})

        if eval_test_per_trial:
            test_params = merged_params(
                TEST_FIXED, sampled,
                {
                    "release":        release,
                    "from":           test_from,
                    "to":             test_to,
                    "artifacts_path": artifacts_path,
                    "eval_only":      True,
                    "policy_init":    policy_npz,
                    "debug":          True,
                },
            )
            dump_json(trial_dir / "config_test.json", test_params)
            res_test = run_cmd(
                build_cmd(main, test_params),
                timeout_s=timeout_s,
                log_file=trial_dir / "test.log",
                stream_output=STREAM_OUTPUT,
            )
            if res_test.returncode == 0:
                test_metrics = enrich_metrics_with_pct(
                    parse_training_output(res_test.stdout),
                    initial_equity=float(TEST_FIXED["initial_equity"]),
                )
                dump_json(trial_dir / "metrics_test.json", test_metrics)
                test_rl_stats = parse_rl_json_stats(res_test.stdout)
                if test_rl_stats.get("test_gate_stats"):
                    dump_json(trial_dir / "rl_gate_stats_test.json", test_rl_stats["test_gate_stats"])
                    trial.set_user_attr("test_rl_gate_stats", test_rl_stats["test_gate_stats"])
                for k, v in test_metrics.items():
                    trial.set_user_attr(f"test_{k}", v)
            else:
                trial.set_user_attr("test_failed", True)

        return val_score

    return _objective


# ============================================================
# 4) MAIN
# ============================================================

def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--release", required=True)
    ap.add_argument("--artifacts_path", required=True)
    ap.add_argument("--main", required=True)

    ap.add_argument("--train_from", required=True)
    ap.add_argument("--train_to", required=True)
    ap.add_argument("--val_from", required=True)
    ap.add_argument("--val_to", required=True)
    ap.add_argument("--test_from", required=True)
    ap.add_argument("--test_to", required=True)

    ap.add_argument("--eval_test_per_trial", action="store_true")

    ap.add_argument("--storage", default='mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db')
    ap.add_argument("--study_name", default=None)
    ap.add_argument("--n_trials", type=int, default=40)
    ap.add_argument("--timeout_s", type=int, default=7200)
    # ── OPTIMIZACIÓN: paralelismo ──────────────────────────────────────────
    ap.add_argument("--n_jobs", type=int, default=1,
                    help="Número de trials en paralelo. "
                         "Requiere storage MySQL. Máx recomendado: nº nucleos físicos / 2.")
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--out_dir", default=f"./artifacts/rl")
    ap.add_argument('--conv_window', type=int, default=20)
    ap.add_argument('--conv_threshold', type=float, default=0.01)
    ap.add_argument('--conv_check_every', type=int, default=5)

    # ── OPTIMIZACIÓN: control de precalentamiento ──────────────────────────
    ap.add_argument("--skip_warm_cache", action="store_true",
                    help="No precalentar el caché antes de Optuna (útil si ya existe).")

    args = ap.parse_args()

    out_dir  = Path(args.out_dir) / 'staged'
    logs_dir = out_dir / "logs"
    ensure_dir(logs_dir)

    scorer = CompositeScorer(METRICS_CFG)

    # ── OPTIMIZACIÓN: precalentamiento del caché ───────────────────────────
    if not args.skip_warm_cache:
        print("\n" + "=" * 60)
        print("PRECALENTAMIENTO DE CACHÉ DE PREDICCIONES")
        print("=" * 60)
        warm_prediction_cache(
            main          = args.main,
            release       = args.release,
            artifacts_path= args.artifacts_path,
            periods       = [
                {"from": args.train_from, "to": args.train_to},
                {"from": args.val_from,   "to": args.val_to},
                # Incluir TEST solo si se evaluará por trial
                *([{"from": args.test_from, "to": args.test_to}] if args.eval_test_per_trial else []),
            ],
            logs_dir      = logs_dir,
            timeout_s     = args.timeout_s,
        )
        print("=" * 60 + "\n")
    else:
        print("[WARM_CACHE] skip_warm_cache=True, omitiendo precalentamiento.")
    # ──────────────────────────────────────────────────────────────────────

    study_name = args.study_name or f"rl_study_{args.release}_v4.1"
    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        n_startup_trials=8,
        multivariate=True,
    )
    pruner = optuna.pruners.MedianPruner(n_warmup_steps=max(5, args.n_trials // 5))

    study = optuna.create_study(
        direction="maximize",
        study_name=study_name,
        storage=args.storage if args.storage else None,
        load_if_exists=True,
        sampler=sampler,
        pruner=pruner,
    )

    dump_json(
        out_dir / "experiment_config.json",
        {
            "release":       args.release,
            "artifacts_path":args.artifacts_path,
            "main":          args.main,
            "train_range":   [args.train_from, args.train_to],
            "val_range":     [args.val_from, args.val_to],
            "test_range":    [args.test_from, args.test_to],
            "param_space":   PARAM_SPACE,
            "train_fixed":   TRAIN_FIXED,
            "val_fixed":     VAL_FIXED,
            "test_fixed":    TEST_FIXED,
            "metrics_cfg":   {k: METRICS_CFG[k].__dict__ for k in METRICS_CFG},
            "n_jobs":        args.n_jobs,
            "timestamp":     datetime.now().isoformat(),
        },
    )

    objective = objective_factory(
        main          = args.main,
        release       = args.release,
        artifacts_path= args.artifacts_path,
        train_from    = args.train_from,
        train_to      = args.train_to,
        val_from      = args.val_from,
        val_to        = args.val_to,
        test_from     = args.test_from,
        test_to       = args.test_to,
        scorer        = scorer,
        logs_dir      = logs_dir,
        timeout_s     = args.timeout_s,
        eval_test_per_trial=bool(args.eval_test_per_trial),
    )

    print("\n=== RL OPT STAGED ===")
    print("Study    :", study.study_name)
    print("Release  :", args.release)
    print("Artifacts:", args.artifacts_path)
    print("TRAIN    :", args.train_from, "->", args.train_to)
    print("VAL      :", args.val_from, "->", args.val_to)
    print("TEST     :", args.test_from, "->", args.test_to)
    print("Trials   :", args.n_trials)
    print("n_jobs   :", args.n_jobs, ("(paralelo)" if args.n_jobs > 1 else "(secuencial)"))
    print("VolFilter: optimizable [none / skip_high / skip_extreme]  mult_fixed=",
          RL_FIXED.get("rl_vol_filter_mult", 1.5))
    print()

    convergence_callback = ConvergenceCallback(
        window=args.conv_window,
        threshold=args.conv_threshold,
        check_every=args.conv_check_every,
    )
    early_stopping_callback = EarlyStoppingCallback(patience=50, threshold=0.01, min_trials=20)

    # ── OPTIMIZACIÓN: n_jobs pasado a optimize() ──────────────────────────
    study.optimize(
        objective,
        n_trials=args.n_trials,
        n_jobs=args.n_jobs,       # ← antes siempre era 1 (hardcoded)
        show_progress_bar=True,
        callbacks=[convergence_callback, early_stopping_callback],
    )
    # ──────────────────────────────────────────────────────────────────────

    # ── Post-optimización ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("POST-OPTIMIZATION ANALYSIS")
    print("=" * 70)

    convergence_final = analyze_convergence(study, window=20, threshold=0.01)
    print_convergence_status(convergence_final)
    dump_json(out_dir / "convergence_analysis.json", convergence_final)

    importance_analysis = analyze_param_importance(study, n_top=10)
    print_param_importance(importance_analysis)
    dump_json(out_dir / "param_importance.json", importance_analysis)

    if "top_params" in importance_analysis:
        print("\n💡 RECOMMENDATIONS:")
        for i, (param, score) in enumerate(list(importance_analysis["top_params"].items())[:5], 1):
            print(f"    {i}. {param} (importance: {score:.4f})")
        for param, score in importance_analysis.get("least_important", {}).items():
            print(f"    - {param} (importance: {score:.4f}) — considera fijar o eliminar")
        print()

    evolution_data = []
    best_so_far = float('-inf')
    for trial in study.trials:
        if trial.value is not None:
            if trial.value > best_so_far:
                best_so_far = trial.value
            evolution_data.append({
                "trial": trial.number,
                "value": float(trial.value),
                "best_so_far": float(best_so_far),
                "datetime": trial.datetime_complete.isoformat() if trial.datetime_complete else None,
            })
    dump_json(out_dir / "optimization_evolution.json", {"evolution": evolution_data})

    if args.eval_test_per_trial:
        rows = []
        for t in study.trials:
            if t.value is None: continue
            ua  = t.user_attrs
            row = {"trial": t.number, "val_score": float(t.value)}
            for k in ["net_pnl_pct", "profit_factor", "max_dd_pct", "win_rate", "n_trades"]:
                row[f"val_{k}"]  = ua.get(f"val_{k}")
                row[f"test_{k}"] = ua.get(f"test_{k}")
            rows.append(row)
        dump_json(out_dir / "trials_metrics.json", {"rows": rows})

        def _get_xy(key_y):
            xs, ys = [], []
            for r in rows:
                x = r.get("val_score")
                y = r.get(key_y)
                if x is not None and y is not None:
                    xs.append(float(x)); ys.append(float(y))
            return xs, ys

        corr_report = {"n_trials_with_test": len([r for r in rows if r.get("test_net_pnl_pct") is not None])}
        for key in ["test_net_pnl_pct", "test_profit_factor", "test_max_dd_pct", "test_win_rate"]:
            xs, ys = _get_xy(key)
            corr_report[key] = {"n": len(xs), "pearson": _pearson_corr(xs, ys), "spearman": _spearman_corr(xs, ys)}
        dump_json(out_dir / "val_test_correlation.json", corr_report)

    best        = study.best_trial
    best_params = best.params
    best_dir    = out_dir / "best"
    ensure_dir(best_dir)

    dump_json(best_dir / "best_trial.json", {
        "study":          study.study_name,
        "release":        args.release,
        "best_value":     study.best_value,
        "best_params":    best_params,
        "best_metrics_val": {k: best.user_attrs.get(k) for k in
                             ["val_net_pnl_pct", "val_profit_factor", "val_win_rate", "val_max_dd_pct",
                              "val_n_trades", "composite_score"]},
        "vol_filter":     best_params.get("rl_vol_filter", "none"),
        "timestamp":      datetime.now().isoformat(),
    })

    # ── FINAL TRAIN ───────────────────────────────────────────────────────
    final_policy = best_dir / f"rl_policy_gate_{args.release}_best.npz"

    final_train_params = merged_params(
        TRAIN_FIXED, best_params,
        {
            "release":        args.release,
            "from":           args.train_from,
            "to":             args.val_to,
            "artifacts_path": args.artifacts_path,
            "policy_out":     final_policy,
        },
    )
    dump_json(best_dir / "config_final_train.json", final_train_params)
    res_final_train = run_cmd(
        build_cmd(args.main, final_train_params),
        timeout_s=args.timeout_s,
        log_file=best_dir / "final_train.log",
        stream_output=STREAM_OUTPUT,
    )
    if res_final_train.returncode != 0:
        print("❌ ERROR final train. Revisa:", best_dir / "final_train.log")
        sys.exit(1)

    final_train_metrics = enrich_metrics_with_pct(
        parse_training_output(res_final_train.stdout),
        initial_equity=float(TRAIN_FIXED["initial_equity"]),
    )
    dump_json(best_dir / "metrics_final_train.json", final_train_metrics)

    # ── FINAL TEST ────────────────────────────────────────────────────────
    final_test_params = merged_params(
        TEST_FIXED, best_params,
        {
            "release":        args.release,
            "from":           args.test_from,
            "to":             args.test_to,
            "artifacts_path": args.artifacts_path,
            "eval_only":      True,
            "policy_init":    final_policy,
        },
    )
    dump_json(best_dir / "config_final_test.json", final_test_params)
    res_final_test = run_cmd(
        build_cmd(args.main, final_test_params),
        timeout_s=args.timeout_s,
        log_file=best_dir / "final_test.log",
        stream_output=STREAM_OUTPUT,
    )
    if res_final_test.returncode != 0:
        print("❌ ERROR final test. Revisa:", best_dir / "final_test.log")
        sys.exit(1)

    final_test_metrics = enrich_metrics_with_pct(
        parse_training_output(res_final_test.stdout),
        initial_equity=float(TEST_FIXED["initial_equity"]),
    )
    dump_json(best_dir / "metrics_final_test.json", final_test_metrics)

    print("\n=== DONE ===")
    print("Best params :", best_params)
    print("Final policy:", final_policy)
    print("Test metrics:", final_test_metrics)
    print("Outputs in  :", best_dir)


if __name__ == "__main__":
    main()