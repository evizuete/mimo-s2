import logging
from dataclasses import dataclass
from typing import Literal, Optional, Dict, Union, List, Callable, Any

import optuna
from sklearn.neighbors import radius_neighbors_graph


@dataclass
class MetricConfig:
    direction: Literal['maximize', 'minimize']
    weight: float = 1.0
    normalize: bool = False
    clip_min: Optional[float] = None
    clip_max: Optional[float] = None

    def __post_init__(self):
        if not 0 <= self.weight <= 1:
            raise ValueError(f"weight must be between 0 and 1, got {self.weight}")


class RLOptunaOptimizer:
    def __init__(self,
                 param_space: Dict[str, Union[List, tuple]],
                 objective_function: Callable,
                 metrics_config: Dict[str, Union[MetricConfig, Dict]],
                 optuna_db: str = '',
                 study_prefix: str = 'opt',
                 load_if_exists: bool = True,
                 fixed_params: Optional[Dict[str, Any]] = None
    ):

        self.param_space = param_space
        self.optuna_db = optuna_db
        self.objective_function = objective_function
        self.fixed_params = fixed_params or {}

        self.metrics_config = {}
        for name, config in metrics_config.items():
            if isinstance(config, dict):
                self.metrics_config[name] = MetricConfig(**config)
            else:
                self.metrics_config[name] = config

        total_weight = sum(m.weight for m in self.metrics_config.values())
        if total_weight > 1.0:
            raise ValueError(f"total_weight must be <= 1.0, got {total_weight}")

        self.primary_metric = max(
            self.metrics_config.items(),
            key=lambda x: x[1].weight
        )[0]

        self.direction = self.metrics_config[self.primary_metric].direction
        self.study_name = f'{study_prefix}_{self.fixed_params["release"]}_{self.fixed_params["side"]}'
        self.study = optuna.create_study(
            study_name = self.study_name,
            storage = self.optuna_db,
            direction=self.direction,
            load_if_exists = load_if_exists,
        )

        optuna.logging.set_verbosity(optuna.logging.INFO)
        self.logger = logging.getLogger(__name__)
        self.metric_stats = {
            name: {'min': float('inf'), 'max': float('-inf')} for name in self.metrics_config.keys()
        }

    def _suggest_parameter(self, trial:optuna.Trial, param_name: str, config: List):
        if len(config) == 3:
            min_val, max_val, param_type = config

            if param_type == 'int':
                return trial.suggest_int(param_name, min_val, max_val)

            elif param_type == 'float':
                return trial.suggest_float(param_name, min_val, max_val)

            elif param_type == 'loguniform':
                return trial.suggest_loguniform(param_name, min_val, max_val, log=True)

            else:
                raise ValueError(f'Invalid type')

        elif len(config) == 2:
            choices, param_type = config

            if param_type == 'categorical':
                return trial.suggest_categorical(param_name, choices)
            else:
                raise ValueError(f'Invalid type')

        else:
            raise ValueError(f'Invalid configuration for {param_name}: {config}')


    def _normalize_metric(self, value: float, metric_name: str) -> float:
        stats = self.metric_stats[metric_name]
        if stats['max'] == stats['min']:
            return 0.5

        normalized = (value - stats['min']) / (stats['max'] - stats['min'])
        return max(0.0, min(1.0, normalized))




