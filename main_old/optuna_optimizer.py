import json
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import optuna
import pandas as pd
from optuna.storages import RDBStorage
from typing import Dict, List, Callable, Any, Optional, Union, Literal
import logging
from datetime import datetime

from tensorflow.python.autograph.pyct import cfg

from mimo_old.model_builder import Config


@dataclass
class MetricConfig:
    direction: Literal['maximize', 'minimize']
    weight: float = 1.0
    normalize: bool = False
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None

    def __post_init__(self):
        if not 0 <= self.weight <= 1:
            raise ValueError(f'Weight must be between 0 and 1, got {self.weight}')

class RLOptunaOptimizer:
    def __init__(
            self,
            release: str,
            side: str,
            param_space: Dict[str, Union[List, tuple]],
            objective_function: Callable,
            metrics_config: Dict[str, Union[MetricConfig, Dict]],
            storage_url: str = '',
            study_name: Optional[str] = None,
            fixed_params: Optional[Dict[str, Any]] = None,
            seed: int = 42
    ):

        self.release = release
        self.side = side
        self.seed = seed
        self.param_space = param_space
        self.objective_function = objective_function
        self.fixed_params = fixed_params
        self.study = None

        self.metrics_config = {}
        for name, config in metrics_config.items():
            if isinstance(config, dict):
                self.metrics_config[name] = MetricConfig(**config)
            else:
                self.metrics_config[name] = config

        total_weight = sum(m.weight for m in self.metrics_config.values())
        if total_weight >= 1.0:
            raise ValueError(f'Total weight must be lower or equal than 1.0, got {total_weight}')

        self.primary_metric = max(
            self.metrics_config.items(),
            key=lambda x: x[1].weight
        )[0]

        self.direction = self.metrics_config[self.primary_metric].direction
        self.storage = storage_url
        if study_name is None:
            self.study_name = f'rl_study_{self.release}_{self.side}'

        optuna.logging.set_verbosity(optuna.logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.metric_stats = {
            name: {'min': float('inf'), 'max': float('-inf')} for name in self.metrics_config.keys()
        }

    def _suggest_parameter(self, trial: optuna.Trial, param_name: str, config: List):
        if len(config) == 3:    # [min, max, type]
            min_val, max_val, param_type = config

            if param_type == 'int':
                return trial.suggest_int(param_name, min_val, max_val)
            elif param_type == 'float':
                return trial.suggest_float(param_name, min_val, max_val)
            elif param_type == 'loguniform':
                return trial.suggest_float(param_name, min_val, max_val, log=True)
            else:
                raise ValueError(f'Param type is no valid: {param_type}')
        elif len(config) == 2: # Categorical: [[options], 'categorical']
            choices, param_type = config

            if param_type == 'categorical':
                return trial.suggest_categorical(param_name, choices)
            else:
                raise ValueError(f'Param type is no valid: {param_type}')
        else:
            raise ValueError(f'Invalid configuration for {param_name}: {config}')

    def _normalize_metric(self, value: float, metric_name: str) -> float:
        stats = self.metric_stats[metric_name]
        if stats['max'] == stats['min']:
            return 0.5

        normalized = (value - stats['min']) / (stats['max'] - stats['min'])
        return max(0.0, min(1.0, normalized))

    def _update_metric_stats(self, metric_name: str, value: float):
        stats = self.metric_stats[metric_name]
        stats['min'] = min(stats['min'], value)
        stats['max'] = max(stats['max'], value)

    def _process_metric_value(self, value: float, config: MetricConfig) -> float:
        if config.clip_min is not None:
            value = max(config.clip_min, value)
        if config.clip_max is not None:
            value = min(config.clip_max, value)

        return value

    def _calculate_composite_score(self, results: Dict[str, float]) -> float:
        score = 0.0

        for metric_name, config in self.metrics_config.items():
            if metric_name not in results:
                self.logger.warning(f'Metric {metric_name} not found in results')
                continue

            value = results[metric_name]
            value = self._process_metric_value(value, config)
            self._update_metric_stats(metric_name, value)

            if config.weight == 0:
                continue

            if config.normalize:
                value = self._normalize_metric(value, metric_name)

            if config.direction == 'minimize':
                value = -value

            score += value * config.weight

        return score

    def _objective(self, trial: optuna.Trial) -> float:
        params = {}
        for param_name, config in self.param_space.items():
            params[param_name] = self._suggest_parameter(trial, param_name, config)

        params.update(self.fixed_params)

        self.logger.info(f"\n{'=' * 70}")
        self.logger.info(f"Trial {trial.number}")
        self.logger.info(f"{'=' * 70}")
        self.logger.info("Parameters:")
        for k, v in params.items():
            if k in self.param_space:  # Only parameters to be optimized
                self.logger.info(f"  {k}: {v}")

        try:
            results = self.objective_function(params)
            if isinstance(results, dict):
                for key, value in results.items():
                    trial.set_user_attr(key, value)

                score = self._calculate_composite_score(results)
                self.logger.info(f'\nResults: ')
                for metric_name in self.metrics_config.keys():
                    if metric_name in results:
                        config = self.metrics_config[metric_name]
                        direction_symbol = '↑' if config.direction == 'maximize' else '↓'
                        weight_str = f'(w={config.weight:.2f})' if config.weight > 0 else '(tracking)'
                        self.logger.info(f'\t{direction_symbol} {metric_name}: {results[metric_name]:.4f} {weight_str}')

                self.logger.info(f'\n\tCOMPOSITE SCORING: {score:.4f}')
                self.logger.info(f'{"="*70}\n')

                return score
            else:
                return results

        except Exception as e:
            self.logger.error(f'Error on trial {trial.number}: {str(e)}')
            import traceback
            self.logger.error(traceback.format_exc())
            return float('-inf') if self.direction == 'maximize' else float('inf')

    def optimize(
            self,
            n_trials: int = 100,
            use_grid: bool = False,
            load_if_exists: bool = True,
            timeout: Optional[int] = None,
            n_jobs: int = 1,
            show_progress_bar: bool = True,
            callbacks: Optional[List[Callable]] = None,
            early_stopping_rounds: Optional[int] = None,
            min_improvement: float = 0.0
    ) -> Dict[str, Any]:

        if use_grid:
            sampler = optuna.samplers.GridSampler(self.param_space, seed=self.seed)
            n_trials = min(n_trials, len(sampler._all_grids)) if n_trials is not None else len(sampler._all_grids)
        else:
            sampler = optuna.samplers.TPESampler(seed=self.seed)
            if n_trials is None:
                raise ValueError('use_grid=False requires to specify number of trials')

        pruner = optuna.pruners.MedianPruner(n_warmup_steps=max(2, n_trials // 5))
        self.study = optuna.create_study(
            direction=self.direction,
            study_name=self.study_name,
            storage=self.storage,
            load_if_exists=load_if_exists,
            sampler=sampler,
            pruner=pruner,
        )

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
            callbacks=callbacks,
        )

        return self.get_best_params()

    def _create_early_stopping_callback(self, patience: int, min_improvement: float) -> Callable:
        def callback(study, trial):
            if len(study.trials) < patience:
                return

            recent_trials = study.trials[-patience:]
            recent_values = [t.value for t in recent_trials if t.value is not None]
            if not recent_values:
                return

            best_recent = max(recent_values) if self.direction == 'maximize' else min(recent_values)
            if self.direction == 'maximize':
                improvement = best_recent - study.best_value
            else:
                improvement = study.best_value - best_recent

            if improvement < min_improvement:
                study.stop()
                self.logger.info(f'\nEarly stopping: no improvement greater than {min_improvement}.')

        return callback

    def get_best_params(self) -> Dict[str, Any]:
        return self.study.best_params

    def get_best_value(self) -> float:
        return self.study.best_value

    def get_best_trial(self) -> optuna.Trial:
        return self.study.best_trial

    def get_best_metrics(self) -> Dict[str, float]:
        return self.study.best_trial.user_attrs

    def get_trials_dataframe(self) -> pd.DataFrame:
        df = self.study.trials_dataframe()
        return df

    def get_metric_summary(self) -> pd.DataFrame:
        df = self.get_trials_dataframe()

        summary_data = []
        for metric_name, config in self.metrics_config.items():
            col = f'user_attrs_{metric_name}'
            if col in df.columns:
                summary_data.append({
                    'metric': metric_name,
                    'direction': config.direction,
                    'weight': config.weight,
                    'mean': df[col].mean(),
                    'std': df[col].std(),
                    'min': df[col].min(),
                    'max': df[col].max(),
                    'best': df[col].max() if config.direction == 'maximize' else df[col].min(),
                })

        return pd.DataFrame(summary_data)

    def get_pareto_front(self, metrics: Optional[List[str]] = None) -> pd.DataFrame:
        if metrics is None:
            metrics = [name for name, config in self.metrics_config.items() if config.weight > 0]

        df = self.get_trials_dataframe()

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

    def _dominates(self, a, b, metrics: List[str]) -> bool:
        better_in_any = False

        for metric in metrics:
            col = f'user_attrs_{metric}'
            if col not in a or col not in b:
                continue

            config = self.metrics_config[metric]
            if config.direction == 'maximize':
                if a[col] < b[col]:
                    return False
                if a[col] > b[col]:
                    better_in_any = True
            else:
                if a[col] > b[col]:
                    return False
                if a[col] < b[col]:
                    better_in_any = True

        return better_in_any

    def print_study_summary(self, top_n: int = 5):
        print(f'\n{"="*80}')
        print(f'OPTIMIZATION REPORT: {self.study.study_name}')
        print(f'\n{"="*80}')
        print(f'Number of trials completed: {len(self.study.trials)}')
        print(f'Best composite scoring: {self.study.best_value:.4f}')

        print(f'\n{"-"*80}')
        print(f'METRICS CONFIGURATION:')
        print(f'\n{"-"*80}')
        for metric_name, config in self.metrics_config.items():
            direction_symbol = '↑' if config.direction == 'maximize' else '↓'
            weight_info = f'weight={config.weight:.2f}' if config.weight > 0 else 'tracking'
            primary = ' [PRINCIPAL]' if metric_name == self.primary_metric else ''
            print(f'\t{direction_symbol} {metric_name} {weight_info}{primary}')

        print(f'\n{"-"*80}')
        print('BEST PARAMETERS:')
        print(f'\n{"-"*80}')
        for param, value in self.study.best_params.items():
            print(f'\t{param}: {value:.4f}')

        best_trial = self.study.best_trial
        if best_trial.user_attrs:
            print(f'\n{"-"*80}')
            print('BEST TRIAL METRICS:')
            print(f'\n{"-"*80}')
            for metric_name, config in self.metrics_config.items():
                if metric_name in best_trial.user_attrs:
                    value = best_trial.user_attrs[metric_name]
                    direction_symbol = '↑' if config.direction == 'maximize' else '↓'
                    print(f'\t{direction_symbol} {metric_name} {value:.4f}')

        print(f'\n{"-"*80}')
        print(f'TOP {top_n} TRIALS:')
        print(f'\n{"-"*80}')
        df = self.get_trials_dataframe()

        if self.direction == 'maximize':
            top_trials = df.nlargest(top_n, 'value')
        else:
            top_trials = df.nsmallest(top_n, 'value')

        for idx, (_, row) in enumerate(top_trials.iterrows(), 1):
            print(f'\n\t#{idx} Trial {int(row["number"])}: score={row["value"]:.4f}')
            print(f'\tParameters:')
            for param in self.param_space.keys():
                col = f'params_{param}'
                if col in row:
                    print(f'\t\t{param}: {row[col]}')

            print('\tMetrics:')
            for metric_name in self.metrics_config.keys():
                col = f'user_attrs_{metric_name}'
                if col in row:
                    config = self.metrics_config[metric_name]
                    direction_symbol = '↑' if config.direction == 'maximize' else '↓'
                    print(f'\t{direction_symbol} {metric_name} {row[col]:.4f}')

        print(f'\n{"-"*80}')
        print(f'SUMMARY OF METRIC STATISTICS:')
        print(f'\n{"-"*80}')
        metric_summary = self.get_metric_summary()
        print(metric_summary.to_string(index=False))

        print(f'\n{"="*80}')

    def export_best_config(self, filepath: str):
        config = {
            'study_name': self.study.study_name,
            'optimization_info': {
                'n_trials': len(self.study.trials),
                'best_score': self.study.best_value,
                'direction': self.direction,
            },
            'metrics_config': {
                name: {
                    'direction': cfg.direction,
                    'weight': cfg.weight,
                    'normalize': cfg.normalize,
                    'clip_min': cfg.clip_min,
                    'clip_max': cfg.clip_max,
                } for name, cfg in self.metrics_config.items()
            },
            'best_params': self.study.best_params,
            'best_metrics': self.study.best_trial.user_attrs,
            'creation_date': datetime.now().isoformat()
        }

        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, 'w') as f:
            json.dump(config, f, indent=2)

        print(f'Configuration saved to: {filepath}')

    def export_all_trials(self, filepath: str):
        df = self.get_trials_dataframe()
        df.to_csv(filepath, index=False)
        print(f'All trials were saved to: {filepath}')

    def plot_optimization_history(self):
        try:
            from optuna.visualization import plot_optimization_history
            return plot_optimization_history(self.study)
        except ImportError:
            self.logger.warning(f'Install plotly: pip install plotly')
            return None

    def plot_param_importances(self):
        try:
            from optuna.visualization import plot_param_importances
            return plot_param_importances(self.study)
        except ImportError:
            self.logger.warning(f'Install plotly: pip install plotly')
            return None

    def plot_parallel_coordinate(self, params: Optional[List[str]] = None):
        try:
            from optuna.visualization import plot_parallel_coordinate
            return plot_parallel_coordinate(self.study, params=params)
        except ImportError:
            self.logger.warning(f'Install plotly: pip install plotly')
            return None

    def plot_slice(self, params: Optional[List[str]] = None):
        try:
            from optuna.visualization import plot_slice
            return plot_slice(self.study, params=params)
        except ImportError:
            self.logger.warning(f'Install plotly: pip install plotly')
            return None

    def plot_contour(self, params: Optional[List[str]] = None):
        try:
            from optuna.visualization import plot_contour
            return plot_contour(self.study, params=params)
        except ImportError:
            self.logger.warning(f'Install plotly: pip install plotly')
            return None


def build_training_command(params: Dict[str, Any]) -> List[str]:
    cmd = [sys.executable, 'main_rl_train.py']

    for key, value in params.items():
        param_name = f'--{key}'
        if isinstance(value, bool):
            if value:
                cmd.append(param_name)
        else:
            cmd.extend([param_name, str(value)])

    return cmd

def parse_training_output(output: str) -> Dict[str, float]:
    results = {}
    patterns = {
        'n_trades': r'Trades\s*:\s*(\d+)',
        'net_pnl': r'NetPnL\s*:\s*([-+]?\d+\.?\d*)',
        'profit_factor': r'ProfitFactor\s*:\s*(\d+\.?\d*)',
        'win_rate': r'WinRate\s*:\s*(\d+\.?\d*)',
        'max_dd': r'MaxDD\s*:\s*([-+]?\d+\.?\d*)',
        'max_dd_pct': r'MaxDD\s+\(%\)\s*:\s*([-+]?\d+\.?\d*)',
    }

    for metric, pattern in patterns.items():
        match = re.search(pattern, output)
        if match:
            try:
                results[metric] = float(match.group(1))
            except ValueError:
                pass

    return results

def run_rl_training(params: Dict[str, Any]) -> Dict[str, float]:
    cmd = build_training_command(params)
    print(f'Running: {" ".join(cmd)}')

    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=3600
        )

        metrics = parse_training_output(result.stdout)
        if not metrics or 'net_pnl' not in metrics:
            print(f'WARNING: Metrics were not parsed. Output: \n{result.stdout[:500]}')
            return {
                'net_pnl': -999999.0,
                'profit_factor': 0.0,
                'win_rate': 0.0,
                'n_trades': 0,
                'max_dd': 999999.0,
            }

        return metrics
    except subprocess.TimeoutExpired:
        print("❌ ERROR: Training timed out.")
        return {
            'net_pnl': -999999.0,
            'profit_factor': 0.0,
            'win_rate': 0.0,
            'n_trades': 0,
            'max_dd': 999999.0,
        }
    except Exception as e:
        print(f"❌ ERROR: {e}")
        return {
            'net_pnl': -999999.0,
            'profit_factor': 0.0,
            'win_rate': 0.0,
            'n_trades': 0,
            'max_dd': 999999.0,
        }


if __name__ == '__main__':
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

    fixed_params = {
        'release': '200285',
        'from': '2025-03-01',
        'to': '2025-11-01',
        'artifacts_path': './artifacts',
        'initial_equity': 10000.0,
        'spread_price': 0.07,
    }

    metrics_config = {
        # Métricas a optimizar (con peso > 0)
        'net_pnl': MetricConfig(
            direction='maximize',
            weight=0.40,  # 40% del score
        ),
        'profit_factor': MetricConfig(
            direction='maximize',
            weight=0.30,  # 30% del score
            clip_min=0.0,
            clip_max=3.0,  # Evitar outliers extremos
        ),
        'win_rate': MetricConfig(
            direction='maximize',
            weight=0.20,  # 20% del score
            clip_min=0.0,
            clip_max=1.0,
        ),
        'max_dd': MetricConfig(
            direction='minimize',
            weight=0.10,  # 10% del score (penalizar drawdown)
        ),

        # Métricas de tracking (weight=0, solo para análisis)
        'n_trades': MetricConfig(
            direction='maximize',
            weight=0.0,  # No contribuye al score, solo tracking
        ),
        'max_dd_pct': MetricConfig(
            direction='minimize',
            weight=0.0,  # Solo tracking
        ),
    }

    release = '200304'
    side = 'long'
    dir_prefix = 'tuning_rl'

    # Crear optimizador
    optimizer = RLOptunaOptimizer(
        release='200304',
        side='long',
        param_space=param_space,
        fixed_params=fixed_params,
        objective_function=run_rl_training,
        metrics_config=metrics_config,
        storage_url='mysql+pymysql://evizuete:Ev1z43t3.00@10.1.21.25:3306/optuna_db',
    )

    print("\n" + "=" * 80)
    print("INICIANDO OPTIMIZACIÓN DE HIPERPARÁMETROS RL")
    print("=" * 80)

    # Ejecutar optimización
    best_params = optimizer.optimize(
        n_trials=100,
        n_jobs=1,  # Cambiar a >1 para paralelizar
        show_progress_bar=True,
        early_stopping_rounds=15,
        min_improvement=10.0  # Mejora mínima en score compuesto
    )

    # Mostrar resultados
    optimizer.print_study_summary(top_n=10)

    # Exportar resultados
    optimizer.export_best_config(f'./artifacts/{dir_prefix}_{release}_{side}/best_rl_config.json')
    optimizer.export_all_trials(f'./artifacts/{dir_prefix}_{release}_{side}/all_trials.csv')

    # Generar visualizaciones
    print("\n" + "=" * 80)
    print("GENERANDO VISUALIZACIONES")
    print("=" * 80)

    try:
        optimizer.plot_optimization_history().write_html(f'./artifacts/{dir_prefix}_{release}_{side}/optimization_history.html')
        optimizer.plot_param_importances().write_html(f'./artifacts/{dir_prefix}_{release}_{side}/param_importances.html')
        optimizer.plot_parallel_coordinate().write_html(f'./artifacts/{dir_prefix}_{release}_{side}/parallel_coordinate.html')
        optimizer.plot_contour().write_html(f'./artifacts/{dir_prefix}_{release}_{side}/contour.html')
        print(f"✓ Plots saved on ./artifacts/{dir_prefix}_{release}_{side}")

    except Exception as e:
        print(f"⚠ No plots were generated: {e}")

    # Análisis multi-objetivo (Pareto front)
    print("\n" + "=" * 80)
    print("PARETO FRONT (No dominated solutions)")
    print("=" * 80)
    pareto_df = optimizer.get_pareto_front()
    print(f"Found {len(pareto_df)} solutions on Pareto Front")
    pareto_df.to_csv(f'./artifacts/{dir_prefix}_{release}_{side}/pareto_front.csv', index=False)
    print(f"✓ Pareto front saved on ./artifacts/{dir_prefix}_{release}_{side}/pareto_front.csv")

    print("\n✓ Process successfully finalised!\n")

