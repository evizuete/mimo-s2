import os
from typing import Optional, Dict, Any

import optuna
import pandas as pd


class OptunaOptimizer:





def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

class Optimizer:
    def __init__(self,
                 release: str = '',
                 side: str = '',
                 optuna_db: Optional[str] = None,
                 study_prefix: str = 'optimizer',
                 out_dir: str = '.',
                 seed: int = 42
    ):
        self.release = release
        self.side = side

        self.optuna_db = optuna_db
        self.study_prefix = study_prefix

        ensure_dir(out_dir)
        self.out_dir = out_dir

        self.seed = seed
        self.grid_space = None

    def _suggest_config(self, trial: optuna.Trial) -> Dict[str, Any]:
        trial.suggest_categorical()

    def optimize(self,
                 df: pd.DataFrame,
                 use_grid: bool = False,
                 n_trials: Optional[int] = None,
                 grid_space: Optional[Dict[str, list]] = None
    ):
        self.grid_space = grid_space

        if use_grid:
            if not grid_space:
                raise ValueError('use_grid=True requires grid_space with potential values per parameter')

            sampler = optuna.samplers.GridSampler(grid_space, seed=self.seed)
            n_trials = min(n_trials, len(sampler._all_grids)) if n_trials is not None else len(sampler._all_grids)
        else:
            sampler = optuna.samplers.TPESampler(seed=self.seed)
            if n_trials is None:
                raise ValueError('use_grid=False requires to specify number of trials')

        pruner = optuna.pruners.MedianPruner(n_warmup_steps=max(2, n_trials // 5))
        study_name = f'{self.study_prefix}_opt_{self.release}_{self.side}'
        study = optuna.create_study(
            direction='maximize',
            study_name=study_name,
            storage=self.optuna_db,
            load_if_exists=True,
            sampler=sampler,
            pruner=pruner
        )

    def _choices(self, name: str):
        if not self.grid_space or name not in self.grid_space:
            raise RuntimeError(f'No grid space for {name}')

        return self.grid_space[name]



