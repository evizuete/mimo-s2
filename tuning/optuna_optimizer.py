import optuna
from optuna.storages import RDBStorage
from typing import Dict, List, Callable, Any, Optional, Union
import logging
from datetime import datetime
import subprocess
import json
import re
import pandas as pd


class RLOptunaOptimizer:
    """
    Clase especializada para optimizar hiperparámetros de RL en trading usando Optuna.

    Ejemplo de uso:
        # Definir espacio de búsqueda para RL
        param_space = {
            'rl_lr': [1e-4, 1e-2, 'loguniform'],
            'rl_entropy': [1e-4, 1e-2, 'loguniform'],
            'rl_baseline_beta': [0.80, 0.99, 'float'],
            'rl_max_grad_norm': [1.0, 20.0, 'float'],
            'rl_trade_cost': [0.1, 2.0, 'float'],
            'rl_batch': [[16, 32, 64, 128, 256], 'categorical'],
            'rl_chop_soft_thr': [0.50, 0.80, 'float'],
            'rl_exhaustion_soft_thr': [0.50, 0.80, 'float'],
            'rl_chop_penalty_coef': [0.0, 0.15, 'float'],
            'rl_exhaustion_penalty_coef': [0.0, 0.15, 'float'],
        }

        # Función que ejecuta el entrenamiento
        def train_rl_model(params):
            # Construir comando con parámetros
            cmd = build_training_command(params)
            # Ejecutar entrenamiento
            result = run_training(cmd)
            # Extraer métricas
            return {
                'net_pnl': result['net_pnl'],
                'profit_factor': result['profit_factor'],
                'win_rate': result['win_rate'],
                'n_trades': result['n_trades']
            }

        # Crear optimizador
        optimizer = RLOptunaOptimizer(
            param_space=param_space,
            objective_function=train_rl_model,
            storage_url='sqlite:///rl_optuna_optimization.db',
            study_name='rl_hyperopt_200285',
            direction='maximize',
            metric_name='net_pnl'
        )

        # Ejecutar optimización
        best_params = optimizer.optimize(n_trials=100)
    """

    def __init__(
            self,
            param_space: Dict[str, Union[List, tuple]],
            objective_function: Callable,
            storage_url: str = 'mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db',
            study_prefix: Optional[str] = None,
            direction: str = 'maximize',
            metric_name: str = 'net_pnl',
            load_if_exists: bool = True,

            # Parámetros fijos del script
            fixed_params: Optional[Dict[str, Any]] = None,
            # Para multi-objetivo
            metric_weights: Optional[Dict[str, float]] = None
    ):
        """
        Inicializa el optimizador para RL trading.

        Args:
            param_space: Diccionario con parámetros y sus configuraciones
            objective_function: Función que ejecuta el entrenamiento y devuelve métricas
            storage_url: URL de la base de datos
            study_prefix: Prefijo a aplicar al nombre del estudio
            direction: 'maximize' o 'minimize'
            metric_name: Nombre de la métrica principal a optimizar
            load_if_exists: Cargar estudio existente si existe
            fixed_params: Parámetros que no se optimizan (ej: release, fechas)
            metric_weights: Pesos para combinar múltiples métricas
                Ej: {'net_pnl': 0.5, 'profit_factor': 0.3, 'win_rate': 0.2}
        """
        self.param_space = param_space
        self.objective_function = objective_function
        self.metric_name = metric_name
        self.direction = direction
        self.fixed_params = fixed_params or {}
        self.metric_weights = metric_weights

        # Configurar almacenamiento
        self.storage = RDBStorage(url=storage_url)

        # Generar nombre de estudio si no se proporciona
        if study_name is None:
            study_name = f"rl_study_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Crear o cargar estudio
        self.study = optuna.create_study(
            study_name=study_name,
            storage=self.storage,
            direction=direction,
            load_if_exists=load_if_exists
        )

        # Configurar logging
        optuna.logging.set_verbosity(optuna.logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Para tracking de mejores resultados
        self.best_results_history = []

    def _suggest_parameter(self, trial: optuna.Trial, param_name: str, config: List) -> Any:
        """Sugiere un valor para un parámetro."""

        if len(config) == 3:  # Numérico: [min, max, tipo]
            min_val, max_val, param_type = config

            if param_type == 'int':
                return trial.suggest_int(param_name, min_val, max_val)
            elif param_type == 'float':
                return trial.suggest_float(param_name, min_val, max_val)
            elif param_type == 'loguniform':
                return trial.suggest_float(param_name, min_val, max_val, log=True)
            else:
                raise ValueError(f"Tipo no válido: {param_type}")

        elif len(config) == 2:  # Categórico: [[opciones], 'categorical']
            choices, param_type = config

            if param_type == 'categorical':
                return trial.suggest_categorical(param_name, choices)
            else:
                raise ValueError(f"Tipo no válido: {param_type}")
        else:
            raise ValueError(f"Configuración inválida para {param_name}: {config}")

    def _calculate_composite_score(self, results: Dict[str, float]) -> float:
        """
        Calcula un score compuesto si se usan múltiples métricas.
        """
        if not self.metric_weights:
            return results[self.metric_name]

        score = 0.0
        for metric, weight in self.metric_weights.items():
            if metric in results:
                score += results[metric] * weight

        return score

    def _objective(self, trial: optuna.Trial) -> float:
        """Función objetivo que Optuna optimizará."""

        # Generar parámetros sugeridos
        params = {}
        for param_name, config in self.param_space.items():
            params[param_name] = self._suggest_parameter(trial, param_name, config)

        # Añadir parámetros fijos
        params.update(self.fixed_params)

        # Log de parámetros del trial
        self.logger.info(f"\n{'=' * 120}")
        self.logger.info(f"Trial {trial.number}")
        self.logger.info(f"{'=' * 120}")
        for k, v in params.items():
            self.logger.info(f"  {k}: {v}")

        # Ejecutar función objetivo
        try:
            results = self.objective_function(params)

            # Almacenar todas las métricas como atributos del trial
            if isinstance(results, dict):
                for key, value in results.items():
                    trial.set_user_attr(key, value)

                # Calcular score (simple o compuesto)
                score = self._calculate_composite_score(results)

                # Log de resultados
                self.logger.info(f"\nResultados:")
                for k, v in results.items():
                    self.logger.info(f"  {k}: {v}")
                self.logger.info(f"  score: {score:.4f}")
                self.logger.info(f"{'=' * 120}\n")

                return score
            else:
                return results

        except Exception as e:
            self.logger.error(f"Error en trial {trial.number}: {str(e)}")
            # Retornar un valor muy malo en vez de fallar
            return float('-inf') if self.direction == 'maximize' else float('inf')

    def optimize(
            self,
            n_trials: int = 100,
            timeout: Optional[int] = None,
            n_jobs: int = 1,
            show_progress_bar: bool = True,
            callbacks: Optional[List[Callable]] = None,
            # Callbacks personalizados para RL
            save_best_policy: bool = True,
            early_stopping_rounds: Optional[int] = None,
            min_improvement: float = 0.0
    ) -> Dict[str, Any]:
        """
        Ejecuta la optimización.

        Args:
            n_trials: Número de trials
            timeout: Tiempo máximo en segundos
            n_jobs: Procesos paralelos
            show_progress_bar: Mostrar barra de progreso
            callbacks: Callbacks adicionales
            save_best_policy: Guardar la policy del mejor trial
            early_stopping_rounds: Parar si no hay mejora en N trials
            min_improvement: Mejora mínima para considerar progreso
        """
        # Early stopping callback
        if early_stopping_rounds:
            callbacks = callbacks or []
            callbacks.append(
                self._create_early_stopping_callback(
                    early_stopping_rounds,
                    min_improvement
                )
            )

        self.study.optimize(
            self._objective,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            show_progress_bar=show_progress_bar,
            callbacks=callbacks
        )

        return self.get_best_params()

    def _create_early_stopping_callback(
            self,
            patience: int,
            min_improvement: float
    ) -> Callable:
        """Crea callback para early stopping."""

        def callback(study, trial):
            if len(study.trials) < patience:
                return

            # Obtener mejores valores de últimos N trials
            recent_trials = study.trials[-patience:]
            best_recent = max([t.value for t in recent_trials if t.value is not None])

            # Comparar con mejor valor global
            if self.direction == 'maximize':
                improvement = best_recent - study.best_value
                if improvement < min_improvement:
                    study.stop()
                    self.logger.info(f"Early stopping: sin mejora > {min_improvement} en {patience} trials")
            else:
                improvement = study.best_value - best_recent
                if improvement < min_improvement:
                    study.stop()
                    self.logger.info(f"Early stopping: sin mejora > {min_improvement} en {patience} trials")

        return callback

    def get_best_params(self) -> Dict[str, Any]:
        """Obtiene los mejores parámetros encontrados."""
        return self.study.best_params

    def get_best_value(self) -> float:
        """Obtiene el mejor valor de la métrica."""
        return self.study.best_value

    def get_best_trial(self) -> optuna.Trial:
        """Obtiene el mejor trial completo."""
        return self.study.best_trial

    def get_trials_dataframe(self) -> pd.DataFrame:
        """Obtiene un DataFrame con todos los trials."""
        df = self.study.trials_dataframe()
        return df

    def get_pareto_front(self) -> pd.DataFrame:
        """
        Para optimización multi-objetivo, obtiene el frente de Pareto.
        Útil cuando quieres balancear múltiples métricas.
        """
        df = self.get_trials_dataframe()

        if not self.metric_weights:
            return df.nlargest(10, 'value')

        # Identificar trials en el frente de Pareto
        metrics = list(self.metric_weights.keys())
        pareto_trials = []

        for idx, row in df.iterrows():
            is_dominated = False
            for _, other_row in df.iterrows():
                if self._dominates(other_row, row, metrics):
                    is_dominated = True
                    break
            if not is_dominated:
                pareto_trials.append(idx)

        return df.loc[pareto_trials]

    def _dominates(self, a, b, metrics):
        """Verifica si trial 'a' domina a trial 'b' en todas las métricas."""
        better_in_any = False
        for metric in metrics:
            col = f'user_attrs_{metric}'
            if col not in a or col not in b:
                continue
            if a[col] < b[col]:
                return False
            if a[col] > b[col]:
                better_in_any = True
        return better_in_any

    def print_study_summary(self, top_n: int = 5):
        """Imprime un resumen del estudio."""
        print(f"\n{'=' * 70}")
        print(f"Resumen del Estudio: {self.study.study_name}")
        print(f"{'=' * 70}")
        print(f"Trials completados: {len(self.study.trials)}")
        print(f"Mejor {self.metric_name}: {self.study.best_value:.4f}")

        print(f"\nMejores parámetros:")
        for param, value in self.study.best_params.items():
            print(f"  {param}: {value}")

        # Métricas adicionales del mejor trial
        best_trial = self.study.best_trial
        if best_trial.user_attrs:
            print(f"\nTodas las métricas del mejor trial:")
            for key, value in best_trial.user_attrs.items():
                print(f"  {key}: {value}")

        # Top N trials
        print(f"\nTop {top_n} trials:")
        df = self.get_trials_dataframe()
        top_trials = df.nlargest(top_n, 'value')

        for idx, (_, row) in enumerate(top_trials.iterrows(), 1):
            print(f"\n  #{idx} (Trial {int(row['number'])}): {row['value']:.4f}")
            for param in self.param_space.keys():
                col = f'params_{param}'
                if col in row:
                    print(f"    {param}: {row[col]}")

        print(f"{'=' * 70}\n")

    def export_best_config(self, filepath: str):
        """Exporta la mejor configuración a un archivo JSON."""
        config = {
            'study_name': self.study.study_name,
            'best_value': self.study.best_value,
            'best_params': self.study.best_params,
            'best_metrics': self.study.best_trial.user_attrs,
            'timestamp': datetime.now().isoformat()
        }

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

        print(f"Configuración exportada a: {filepath}")

    def plot_optimization_history(self):
        """Genera gráfico del historial."""
        try:
            from optuna.visualization import plot_optimization_history
            return plot_optimization_history(self.study)
        except ImportError:
            self.logger.warning("Instala plotly: pip install plotly")
            return None

    def plot_param_importances(self):
        """Genera gráfico de importancia de parámetros."""
        try:
            from optuna.visualization import plot_param_importances
            return plot_param_importances(self.study)
        except ImportError:
            self.logger.warning("Instala plotly: pip install plotly")
            return None

    def plot_parallel_coordinate(self, params: Optional[List[str]] = None):
        """Genera gráfico de coordenadas paralelas."""
        try:
            from optuna.visualization import plot_parallel_coordinate
            return plot_parallel_coordinate(self.study, params=params)
        except ImportError:
            self.logger.warning("Instala plotly: pip install plotly")
            return None

    def plot_slice(self, params: Optional[List[str]] = None):
        """Genera gráfico de slice para ver relación param-valor."""
        try:
            from optuna.visualization import plot_slice
            return plot_slice(self.study, params=params)
        except ImportError:
            self.logger.warning("Instala plotly: pip install plotly")
            return None


# ============================================================================
# FUNCIONES AUXILIARES PARA TU CASO ESPECÍFICO
# ============================================================================

def build_training_command(params: Dict[str, Any]) -> List[str]:
    """
    Construye el comando para ejecutar main_rl_train.py con los parámetros dados.
    """
    cmd = ["python", "main_rl_train.py"]

    for key, value in params.items():
        # Convertir nombre de parámetro a formato CLI
        param_name = f"--{key}"

        # Manejar booleanos
        if isinstance(value, bool):
            if value:
                cmd.append(param_name)
        else:
            cmd.extend([param_name, str(value)])

    return cmd


def parse_training_output(output: str) -> Dict[str, float]:
    """
    Parsea la salida del script de entrenamiento para extraer métricas.

    Busca líneas como:
        Trades     : 150
        NetPnL     : 1234.56
        ProfitFactor: 1.45
        WinRate    : 0.65
    """
    results = {}

    patterns = {
        'n_trades': r'Trades\s*:\s*(\d+)',
        'net_pnl': r'NetPnL\s*:\s*([-+]?\d+\.?\d*)',
        'profit_factor': r'ProfitFactor\s*:\s*(\d+\.?\d*)',
        'win_rate': r'WinRate\s*:\s*(\d+\.?\d*)',
        'max_dd': r'MaxDD\s*:\s*([-+]?\d+\.?\d*)',
        'max_dd_pct': r'MaxDD \(%\)\s*:\s*([-+]?\d+\.?\d*)',
    }

    for metric, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            results[metric] = float(match.group(1))

    return results


def run_rl_training(params: Dict[str, Any]) -> Dict[str, float]:
    """
    Ejecuta el entrenamiento RL y devuelve las métricas.

    Esta función:
    1. Construye el comando
    2. Ejecuta el script
    3. Parsea la salida
    4. Devuelve métricas
    """
    cmd = build_training_command(params)

    try:
        # Ejecutar comando
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=3600  # 1 hora timeout
        )

        # Parsear salida
        metrics = parse_training_output(result.stdout)

        # Si no se pudo parsear, usar valores por defecto muy malos
        if not metrics:
            print(f"WARNING: No se pudieron parsear métricas. Output:\n{result.stdout}")
            return {
                'net_pnl': -999999.0,
                'profit_factor': 0.0,
                'win_rate': 0.0,
                'n_trades': 0
            }

        return metrics

    except subprocess.TimeoutExpired:
        print("ERROR: Timeout en entrenamiento")
        return {
            'net_pnl': -999999.0,
            'profit_factor': 0.0,
            'win_rate': 0.0,
            'n_trades': 0
        }
    except Exception as e:
        print(f"ERROR: {e}")
        return {
            'net_pnl': -999999.0,
            'profit_factor': 0.0,
            'win_rate': 0.0,
            'n_trades': 0
        }


# ============================================================================
# EJEMPLO DE USO COMPLETO PARA TU CASO
# ============================================================================

if __name__ == "__main__":
    # Definir espacio de búsqueda para hiperparámetros RL
    param_space = {
        'rl_lr': [1e-4, 1e-2, 'loguniform'],
        'rl_entropy': [1e-4, 1e-2, 'loguniform'],
        'rl_baseline_beta': [0.80, 0.99, 'float'],
        'rl_max_grad_norm': [1.0, 20.0, 'float'],
        'rl_trade_cost': [0.1, 2.0, 'float'],
        'rl_batch': [[16, 32, 64, 128], 'categorical'],
        'rl_chop_soft_thr': [0.50, 0.80, 'float'],
        'rl_exhaustion_soft_thr': [0.50, 0.80, 'float'],
        'rl_chop_penalty_coef': [0.0, 0.15, 'float'],
        'rl_exhaustion_penalty_coef': [0.0, 0.15, 'float'],
    }

    # Parámetros fijos (que no se optimizan)
    fixed_params = {
        'release': '200285',
        'from': '2025-03-01',
        'to': '2025-11-01',
        'artifacts_path': './artifacts',
        'initial_equity': 10000.0,
        'spread_price': 0.07,
    }

    # Pesos para métricas múltiples (opcional)
    metric_weights = {
        'net_pnl': 0.50,  # 50% peso en PnL
        'profit_factor': 0.30,  # 30% peso en profit factor
        'win_rate': 0.20,  # 20% peso en win rate
    }

    # Crear optimizador
    optimizer = RLOptunaOptimizer(
        param_space=param_space,
        objective_function=run_rl_training,
        storage_url='sqlite:///rl_hyperopt_200285.db',
        study_name='rl_optimization_v1',
        direction='maximize',
        metric_name='net_pnl',  # Métrica principal
        fixed_params=fixed_params,
        metric_weights=metric_weights  # Para score compuesto
    )

    # Ejecutar optimización
    print("Iniciando optimización de hiperparámetros RL...")
    best_params = optimizer.optimize(
        n_trials=100,
        n_jobs=1,  # Cambiar a >1 para paralelizar
        show_progress_bar=True,
        early_stopping_rounds=15,  # Parar si no mejora en 15 trials
        min_improvement=10.0  # Mejora mínima de 10 en PnL
    )

    # Mostrar resultados
    optimizer.print_study_summary(top_n=10)

    # Exportar mejor configuración
    optimizer.export_best_config('./artifacts/best_rl_config.json')

    # Generar visualizaciones
    try:
        optimizer.plot_optimization_history().write_html('./artifacts/optimization_history.html')
        optimizer.plot_param_importances().write_html('./artifacts/param_importances.html')
        optimizer.plot_parallel_coordinate().write_html('./artifacts/parallel_coordinate.html')
        print("Visualizaciones guardadas en ./artifacts/")
    except Exception as e:
        print(f"No se pudieron generar visualizaciones: {e}")

    # Obtener DataFrame con todos los trials
    df = optimizer.get_trials_dataframe()
    df.to_csv('./artifacts/all_trials.csv', index=False)
    print("Todos los trials guardados en ./artifacts/all_trials.csv")