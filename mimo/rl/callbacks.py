from typing import Dict, Any

import numpy as np
import optuna


def analyze_convergence(study: optuna.Study, window: int = 20, threshold: float = 0.01) -> Dict[str, Any]:
    """
    Analiza si el estudio ha convergido verificando la varianza de los últimos N trials.

    Args:
        study: Estudio de Optuna
        window: Número de trials a considerar (default: 20)
        threshold: Threshold de std para considerar convergencia (default: 0.01)

    Returns:
        Dict con análisis de convergencia
    """
    if len(study.trials) < window:
        return {
            "converged": False,
            "reason": f"Not enough trials ({len(study.trials)} < {window})",
            "n_trials": len(study.trials),
            "window": window,
        }

    # Obtener valores de los últimos N trials
    recent_values = []
    for trial in study.trials[-window:]:
        if trial.value is not None:
            recent_values.append(trial.value)

    if len(recent_values) < window // 2:
        return {
            "converged": False,
            "reason": f"Too many failed trials in window ({len(recent_values)} valid)",
            "n_trials": len(study.trials),
            "window": window,
        }

    # Calcular estadísticas
    mean_value = np.mean(recent_values)
    std_value = np.std(recent_values)
    relative_std = std_value / abs(mean_value) if mean_value != 0 else float('inf')

    converged = relative_std < threshold

    return {
        "converged": converged,
        "n_trials": len(study.trials),
        "window": window,
        "mean_last_n": float(mean_value),
        "std_last_n": float(std_value),
        "relative_std": float(relative_std),
        "threshold": threshold,
        "reason": f"Relative std {relative_std:.4f} {'<' if converged else '>='} threshold {threshold}"
    }


def analyze_param_importance(study: optuna.Study, n_top: int = 10) -> Dict[str, Any]:
    """
    Analiza la importancia de cada parámetro en el estudio.

    Args:
        study: Estudio de Optuna
        n_top: Número de parámetros top a retornar

    Returns:
        Dict con análisis de importancia
    """
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    if len(completed_trials) < 2:
        return {
            "error": f"Need at least 2 completed trials, got {len(completed_trials)}",
            "n_completed": len(completed_trials),
        }

    try:
        # Calcular importancia usando el evaluator por defecto (FanovaImportanceEvaluator)
        importance = optuna.importance.get_param_importances(
            study,
            evaluator=None,  # Usa FanovaImportanceEvaluator por defecto
        )

        # Ordenar por importancia
        sorted_importance = sorted(importance.items(), key=lambda x: x[1], reverse=True)

        return {
            "n_completed_trials": len(completed_trials),
            "all_params": {k: float(v) for k, v in sorted_importance},
            "top_params": {k: float(v) for k, v in sorted_importance[:n_top]},
            "least_important": {k: float(v) for k, v in sorted_importance[-3:]} if len(sorted_importance) > 3 else {},
        }

    except Exception as e:
        # Fallback: calcular importancia simple basada en correlación
        return analyze_param_importance_simple(study, n_top)


def analyze_param_importance_simple(study: optuna.Study, n_top: int = 10) -> Dict[str, Any]:
    """
    Versión simplificada de análisis de importancia usando correlación.
    Fallback cuando Fanova falla.
    """
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

    # Obtener todos los parámetros
    all_params = set()
    for trial in completed_trials:
        all_params.update(trial.params.keys())

    # Calcular correlación simple para cada parámetro
    importance_scores = {}
    for param in all_params:
        param_values = []
        trial_values = []

        for trial in completed_trials:
            if param in trial.params and trial.value is not None:
                param_values.append(trial.params[param])
                trial_values.append(trial.value)

        if len(param_values) > 1:
            # Normalizar valores categóricos a numéricos
            if isinstance(param_values[0], (int, float)):
                correlation = abs(np.corrcoef(param_values, trial_values)[0, 1])
            else:
                # Para categóricos, usar varianza de medias por categoría
                from collections import defaultdict
                category_values = defaultdict(list)
                for pv, tv in zip(param_values, trial_values):
                    category_values[pv].append(tv)

                category_means = [np.mean(vals) for vals in category_values.values()]
                correlation = np.std(category_means) if len(category_means) > 1 else 0.0

            importance_scores[param] = float(correlation)

    sorted_importance = sorted(importance_scores.items(), key=lambda x: x[1], reverse=True)

    return {
        "n_completed_trials": len(completed_trials),
        "method": "simple_correlation",
        "all_params": {k: float(v) for k, v in sorted_importance},
        "top_params": {k: float(v) for k, v in sorted_importance[:n_top]},
    }


def print_convergence_status(convergence_info: Dict[str, Any]) -> None:
    """Imprime el status de convergencia de forma legible"""
    print("\n" + "=" * 70)
    print("📊 CONVERGENCE ANALYSIS")
    print("=" * 70)

    if convergence_info.get("converged"):
        print("✅ CONVERGED")
    else:
        print("⚠️  NOT CONVERGED")

    print(f"  Total trials: {convergence_info.get('n_trials', 'N/A')}")
    print(f"  Window size: {convergence_info.get('window', 'N/A')}")

    if "mean_last_n" in convergence_info:
        print(f"  Mean (last {convergence_info['window']} trials): {convergence_info['mean_last_n']:.6f}")
        print(f"  Std (last {convergence_info['window']} trials): {convergence_info['std_last_n']:.6f}")
        print(f"  Relative Std: {convergence_info['relative_std']:.6f}")
        print(f"  Threshold: {convergence_info['threshold']:.6f}")

    print(f"  Reason: {convergence_info.get('reason', 'N/A')}")
    print("=" * 70 + "\n")


def print_param_importance(importance_info: Dict[str, Any]) -> None:
    """Imprime la importancia de parámetros de forma legible"""
    print("\n" + "=" * 70)
    print("🔍 PARAMETER IMPORTANCE ANALYSIS")
    print("=" * 70)

    if "error" in importance_info:
        print(f"❌ ERROR: {importance_info['error']}")
        print("=" * 70 + "\n")
        return

    print(f"  Completed trials: {importance_info.get('n_completed_trials', 'N/A')}")

    if "method" in importance_info:
        print(f"  Method: {importance_info['method']}")

    print("\n  📈 TOP PARAMETERS (most important):")
    for i, (param, score) in enumerate(importance_info.get("top_params", {}).items(), 1):
        bar_length = int(score * 50)  # Escala a 50 caracteres
        bar = "█" * bar_length
        print(f"    {i:2d}. {param:25s}: {score:6.4f} {bar}")

    if "least_important" in importance_info and importance_info["least_important"]:
        print("\n  📉 LEAST IMPORTANT PARAMETERS:")
        for param, score in importance_info["least_important"].items():
            print(f"      {param:25s}: {score:6.4f}")

    print("=" * 70 + "\n")


class EarlyStoppingCallback:
    """Stop optimization si ya convergió"""

    def __init__(self,
                 patience: int = 10,
                 threshold: float = 0.01,
                 min_trials: int = 20):
        self.patience = patience
        self.threshold = threshold
        self.min_trials = min_trials
        self.trials_since_improvement = 0
        self.best_value = float('-inf')

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial):
        """Check si debemos parar"""

        # Esperar mínimo de trials
        if len(study.trials) < self.min_trials:
            return

        # Check convergencia
        conv = analyze_convergence(study, window=15, threshold=self.threshold)

        if conv.get('converged'):
            print(f"\n🎯 Early stopping: Convergencia detectada después de {len(study.trials)} trials")
            print(f"   Rel_std: {conv['relative_std']:.6f} < {self.threshold}")
            study.stop()
            return

        # Check patience (sin mejora)
        if trial.value is not None:
            if trial.value > self.best_value:
                self.best_value = trial.value
                self.trials_since_improvement = 0
            else:
                self.trials_since_improvement += 1

        if self.trials_since_improvement >= self.patience:
            print(f"\n⏰ Early stopping: {self.patience} trials sin mejora")
            study.stop()

class ConvergenceCallback:
    """
    Callback para monitorear convergencia durante la optimización.
    Se ejecuta después de cada trial.
    """

    def __init__(self, window: int = 20, threshold: float = 0.01, check_every: int = 5):
        """
        Args:
            window: Número de trials para calcular convergencia
            threshold: Threshold de relative std para considerar convergencia
            check_every: Verificar convergencia cada N trials
        """
        self.window = window
        self.threshold = threshold
        self.check_every = check_every
        self.last_check = 0

    def __call__(self, study: optuna.Study, trial: optuna.trial.FrozenTrial) -> None:
        """Callback ejecutado después de cada trial"""

        # Solo verificar cada check_every trials
        if len(study.trials) - self.last_check < self.check_every:
            return

        self.last_check = len(study.trials)

        # Analizar convergencia
        conv_info = analyze_convergence(study, window=self.window, threshold=self.threshold)

        # Imprimir status
        print(f"\n[Trial {trial.number}] Convergence check:")
        print(f"  Last {self.window} trials: mean={conv_info.get('mean_last_n', 0):.6f}, "
              f"std={conv_info.get('std_last_n', 0):.6f}, "
              f"rel_std={conv_info.get('relative_std', 0):.6f}")

        if conv_info.get('converged'):
            print(f"  ✅ Converged! (rel_std < {self.threshold})")
            # Nota: No detenemos automáticamente, solo informamos
        else:
            print(f"  ⚠️  Not converged (rel_std >= {self.threshold})")


'''

# Uso
early_stopping = EarlyStoppingCallback(
    patience=10,
    threshold=0.01,
    min_trials=20
)

study.optimize(
    objective,
    n_trials=args.n_trials,
    n_jobs=1,
    show_progress_bar=True,
    callbacks=[convergence_callback, early_stopping]
)

'''
