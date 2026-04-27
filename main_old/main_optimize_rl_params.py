# optimize_rl_params.py
from __future__ import annotations

import json
import os
import random
from datetime import datetime
from typing import Any, Dict, Optional

from mimo_old.decision_engine import DecisionPolicy, RiskConfig
from mimo_old.feature_builder import FeatureConfig
from mimo_old.model_builder import Config, ModelConfig
from mimo.strategies.regime_detector import RegimeConfig
from reinforcement_learning.rl_experiment import RLExperiment, RLRunConfig
from reinforcement_learning.rl_policy_evaluator import RLPolicyEvaluator, EvalObjective


#from rl_experiment import RLExperiment, RLRunConfig
#from rl_policy_evaluator import RLPolicyEvaluator, EvalObjective

#from reinforcement_learning import rl_experiment


# -----------------------------
# 1) Construcción del experimento
# -----------------------------
def build_experiment(release: str) -> RLExperiment:
    """
    IMPORTANTE:
    Copia aquí EXACTAMENTE el bloque que ya usas para construir RLExperiment
    (general_config, model_config, feature_config, regime_config, decision_policy).

    Ejemplo:
        from mimo_old.config import GeneralConfig, ModelConfig, FeatureConfig, RegimeConfig
        from mimo_old.decision_policy import DecisionPolicy
        ...
        exp = RLExperiment(
            general_config=general_config,
            model_config=model_config,
            feature_config=feature_config,
            regime_config=regime_config,
            decision_policy=decision_policy,
        )
        return exp
    """

    general_config = Config(
        release=release,
        use_oof=True,
        oof_splits=8,
        oof_epochs=25,
        save_oof_artifacts=True,
    )

    model_config = ModelConfig(
        seq_len_short=64,
        seq_len_long=256,
        batch_size=4096,
    )

    feature_config = FeatureConfig(
        ema_periods=[9, 21, 50],
        label_horizon=5,
        label_method='adaptative'
    )

    regime_config = RegimeConfig(
        adx_trend_threshold=25.0
    )

    decision_policy = DecisionPolicy(
        gate_by_action_and_state={
            "long": {
                'trend_up': 95,
                'transition': 97,
                'range': 99,
                'breakout': 97,
                'volatile': 99,
                "_global": 98,
            },
            "short": {
                'trend_down': 95,
                'transition': 97,
                'range': 99,
                'breakout': 97,
                'volatile': 99,
                "_global": 98,
            },
        },
        score_cap_by_state={
            'trend_up': 1.5,
            'trend_down': 1.5,
            'transition': 1.25,
            'range': 1.0,
            'volatile': 0.75,
        },
        risk_mult_by_state={
            'trend_up': 1.0,
            'trend_down': 1.0,
            'transition': 0.75,
            'range': 0.50,
            'breakout': 0.50,
            'volatile': 0.25
        },
        score_low_quantile=50,
        score_high_quantile=99,
        require_delta_rel=True,
        min_delta_rel=0.15,
        allow_volatile=False,
    )

    risk_config = RiskConfig(
        base_risk_pct=0.005,
        min_score_to_trade=0.25,
        max_risk_pct=0.02,
        max_positions=1,
    )

    rl_config = {
        'lr': 5e-3, #5e-4,
        'entropy_coef': 1e-3,
        'baseline_beta': 0.90, #0.95,
        'max_grad_norm': 10.0,
        'trade_cost_money': 0.5,
        'batch_size': 32, #256
        'chop_soft_thr': 0.60,
        'exhaustion_soft_thr': 0.60,
        'chop_penalty_coef': 0.08,
        'exhaustion_penalty_coef': 0.06,
    }

    exp = RLExperiment(
        general_config=general_config,
        model_config=model_config,
        feature_config=feature_config,
        regime_config=regime_config,
        decision_policy=decision_policy,
        risk_config=risk_config,
        artifacts_path='../artifacts',
        rl_config=rl_config
    )

    return exp

# -----------------------------
# 2) Espacio de búsqueda
# -----------------------------
def suggest_params_optuna(trial) -> Dict[str, Any]:
    """
    Ajusta aquí lo que de verdad quieres optimizar.
    Con lo que expone RLRunConfig ahora mismo, lo más útil suele ser:
      - max_daily_loss_pct (firewall diario)
      - rl_take_threshold (umbral de acción de la policy)
      - sizing_equity_mode (balance/mtm/fixed si lo soportas)
      - compound (impacto del interés compuesto en el sizing)
    """
    return {
        "max_daily_loss_pct": trial.suggest_float("max_daily_loss_pct", 0.0550, 0.0800),   # 1%..10%
        "rl_take_threshold": trial.suggest_float("rl_take_threshold", 0.5980, 0.61),
        "compound": trial.suggest_categorical("compound", [True]),
        "sizing_equity_mode": trial.suggest_categorical("sizing_equity_mode", ["balance"]), # balance y luego mtm
    }


def suggest_params_random(rng: random.Random) -> Dict[str, Any]:
    return {
        "max_daily_loss_pct": rng.uniform(0.01, 0.10),
        "rl_take_threshold": rng.uniform(0.40, 0.70),
        "compound": rng.choice([True, False]),
        "sizing_equity_mode": rng.choice(["balance", "mtm", "fixed"]),
    }


# -----------------------------
# 3) Main optimizer
# -----------------------------
def main():
    # Periodos: entrena en una ventana y evalúa en otra (OOS)
    train_from = datetime.fromisoformat("2024-01-01")
    train_to = datetime.fromisoformat("2025-08-31")

    eval_from = datetime.fromisoformat("2025-09-01")
    eval_to = datetime.fromisoformat("2025-10-31")

    release = '200292'
    n_trials = 40
    seed = 42
    optuna_db = 'mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db'

    outdir = f"./artifacts/rl_opt_{release}"
    os.makedirs(outdir, exist_ok=True)

    exp = build_experiment(release=release)

    evaluator = RLPolicyEvaluator(
        experiment=exp,
        train_from=train_from,
        train_to=train_to,
        eval_from=eval_from,
        eval_to=eval_to,
        outdir=outdir,
        objective=EvalObjective(lambda_dd=2.0),
    )

    base_run = RLRunConfig(
        policy_path="rl_policy.npz",   # el evaluator lo reubica por trial
        use_rl=True,
        rl_train=True,                 # se fuerza por mode="train"/"eval"
        rl_eval_deterministic=False,
        rl_take_threshold=0.50,
        initial_equity=10_000.0,
        sizing_equity_mode="balance",
        compound=True,
        max_daily_loss_pct=0.03,
        max_daily_profit_pct=None,
    )

    # Intenta Optuna; si no está, hace random search.
    try:
        import optuna

        def objective(trial):
            params = suggest_params_optuna(trial)
            run = RLRunConfig(**{**base_run.__dict__, **params})

            tag = f"trial_{trial.number:04d}"
            payload = evaluator.train_and_eval(run=run, tag=tag, export_trade_log=False)

            score = float(payload["metrics"]["score"])
            # Optuna minimiza o maximiza; aquí MAXIMIZAMOS score
            return score

        study_name = f'RL_{release}'
        sampler = optuna.samplers.TPESampler(seed=42)
        pruner = optuna.pruners.MedianPruner(n_warmup_steps=max(2, n_trials // 5))
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=optuna_db,
            load_if_exists=True,
            sampler=sampler,
            pruner=pruner,
        )

        study.optimize(objective, n_trials=n_trials)

        best = {
            "best_value": study.best_value,
            "best_params": study.best_params,
        }
        with open(os.path.join(outdir, "best_optuna.json"), "w", encoding="utf-8") as f:
            json.dump(best, f, indent=2, ensure_ascii=False)

        print("\n=== BEST (Optuna) ===")
        print(best)

    except Exception as e:
        print(f"[warn] Optuna no disponible o falló ({e}). Usando random search…")

        rng = random.Random(1337)
        best_score = -1e18
        best_payload: Optional[Dict[str, Any]] = None

        for i in range(30):
            params = suggest_params_random(rng)
            run = RLRunConfig(**{**base_run.__dict__, **params})

            tag = f"rand_{i:04d}"
            payload = evaluator.train_and_eval(run=run, tag=tag, export_trade_log=False)
            score = float(payload["metrics"]["score"])

            if score > best_score:
                best_score = score
                best_payload = payload

            print(f"[rand] {tag} score={score:.2f} net={payload['metrics']['net_pnl']:.2f} mdd%={payload['metrics']['max_dd_pct']:.4f}")

        if best_payload:
            with open(os.path.join(outdir, "best_random.json"), "w", encoding="utf-8") as f:
                json.dump(best_payload, f, indent=2, ensure_ascii=False)

            print("\n=== BEST (Random) ===")
            print({
                "score": best_payload["metrics"]["score"],
                "params": best_payload["run"],
                "metrics": best_payload["metrics"],
                "policy_path": best_payload["policy_path"],
            })


if __name__ == "__main__":
    main()
