import hashlib
import json
import os
import shutil
from dataclasses import dataclass
from typing import Dict, Any, Optional

import joblib
import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score

from mimo.data_managers.data_pipeline_v2 import DataPipeline
from mimo.features.feature_builder import FeatureConfig
from mimo.helpers.helper import Helper
from mimo.helpers.scaler_loader import load_pipeline_scalers_for_side
from mimo.models.model_builder import Config, ModelConfig, TradingModel
from mimo.models.model_evaluator import ModelEvaluator
from mimo.oof.probs_calibration import ProbsCalibration
from mimo.states_manager.state_detector import StateConfig


def params_signature(params: Dict[str, Any]) -> str:
    """
    Firma estable de hiperparámetros para validar reutilización de OOF.
    """
    dumped = json.dumps(params, sort_keys=True)
    return hashlib.md5(dumped.encode("utf-8")).hexdigest()

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def safe_float(x) -> float:
    try:
        return float(x)
    except Exception:
        return float("nan")

def save_json(obj: Dict[str, Any], path: str):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _extract_feature_schema(pipeline: DataPipeline, side: Optional[str] = None) -> Dict[str, Any]:
    """Extrae el esquema efectivo de features para un side concreto."""
    out: Dict[str, Any] = {}
    fe = getattr(pipeline, "feature_engineer", None)
    if fe is None:
        return out

    try:
        if side is not None and hasattr(fe, "set_side"):
            fe.set_side(side)
        if hasattr(fe, "_assign_features_to_inputs"):
            fe._assign_features_to_inputs()
        feature_columns = getattr(fe, "feature_columns", None) or {}
        for key, cols in feature_columns.items():
            out[key] = list(cols)
    except Exception as ex:
        out["_error"] = str(ex)
    return out


def _snapshot_side_specific_scalers(pipeline: DataPipeline, generic_scaler_path: str, out_dir: str, release: str, side: str):
    """
    Crea una copia side-specific del directorio de scalers para evitar sobrescrituras
    cuando LONG y SHORT usan máscaras distintas.
    """
    if not generic_scaler_path or not os.path.isdir(generic_scaler_path):
        return

    side_dir = os.path.join(out_dir, f"scalers_{release}_{side}")
    if os.path.exists(side_dir):
        shutil.rmtree(side_dir)
    shutil.copytree(generic_scaler_path, side_dir)

    schema = {
        "release": release,
        "side": side,
        "saved_at": pd.Timestamp.utcnow().isoformat(),
        "feature_columns": _extract_feature_schema(pipeline, side=side),
    }
    save_json(schema, os.path.join(side_dir, "feature_schema.json"))
    print(f"[SCALERS] Snapshot side-specific guardado en {side_dir}")

def load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def mask_oof(df: pd.DataFrame, proba_col: str = "oof_proba_cal") -> np.ndarray:
    return np.isfinite(df[proba_col].to_numpy()) & np.isfinite(df["signal"].to_numpy())


@dataclass
class TrainerArtifacts:
    best_params: Dict[str, Any]
    best_value: float
    study_name: str
    side: str
    oof_metrics: Dict[str, Any]
    percentiles: Dict[str, Any]
    calibrator_path: str
    percentiles_path: str
    model_path: str
    oof_meta_path: str
    oof_df_path: Optional[str]

class OptunaOOFTrainer:
    def __init__(
        self,
        *,
        general_config: Config,
        feature_config: FeatureConfig = FeatureConfig(),
        regime_config: StateConfig = StateConfig(),
        base_model_config: ModelConfig = ModelConfig(),
        out_dir: str = ".",
        optuna_db: Optional[str] = None,   # ej: "sqlite:///optuna.db"
        study_prefix: str = "mimo_old",
        seed: int = 42,
        reload: bool = False,
        calibration_temperature: float = 1.0,  # deprecated: usar temperature_long/temperature_short
        temperature_long: float = 1.0,
        temperature_short: float = 1.2
    ):
        self.grid_space = None
        self.general_config = general_config
        self.feature_config = feature_config
        self.regime_config = regime_config  # StateConfig
        self.base_model_config = base_model_config
        self.out_dir = out_dir
        self.optuna_db = optuna_db
        self.study_prefix = study_prefix
        self.seed = seed
        # Compatibilidad hacia atrás: si los nuevos son 1.0 pero el legacy != 1.0,
        # usar el legacy para ambos lados.
        _t_long  = float(temperature_long)
        _t_short = float(temperature_short)
        if abs(_t_long - 1.0) < 1e-9 and abs(_t_short - 1.0) < 1e-9 \
                and abs(float(calibration_temperature) - 1.0) > 1e-9:
            _t_long = _t_short = float(calibration_temperature)
        self.calibration_temperature = float(calibration_temperature)  # legacy
        self.temperature_long  = _t_long
        self.temperature_short = _t_short

        ensure_dir(self.out_dir)

        self.evaluator = ModelEvaluator()

        self.best_params_by_side: Dict[str, Dict[str, Any]] = {}
        self.best_model_config_by_side: Dict[str, ModelConfig] = {}
        self.best_oof_df_by_side: Dict[str, pd.DataFrame] = {}
        self.best_calibrator_by_side: Dict[str, Any] = {}
        self.best_percentiles_by_side: Dict[str, Any] = {}

        if reload:
            for side in ['long', 'short']:
                self._load_best_from_storage(side=side)


    def _load_best_from_storage(self, side: str) -> ModelConfig:
        if not self.optuna_db:
            raise ValueError('Optuna DB is not configured and study cannot be loaded from storage')

        study_name = f'{self.study_prefix}_{self.general_config.release}_{side}'
        study = optuna.load_study(study_name=study_name, storage=self.optuna_db)

        self.best_params_by_side[side] = dict(study.best_trial.params)
        best_mc = self._model_config_from_params(self.best_params_by_side[side])
        best_epochs = study.best_trial.user_attrs.get('best_epochs')
        if best_epochs:
            best_mc.epochs = self.aggregate_best_epochs(best_epochs, method='median')

        self.best_model_config_by_side[side] = best_mc
        return best_mc

    @staticmethod
    def aggregate_best_epochs(best_epochs: dict[int, float], method: str = 'median') -> int:
        """
            best_epochs: dict[fold] -> {"epoch": int, "val": float}
            method: "median" | "mean" | "wmean"
            """
        items = list(best_epochs.values())
        epochs = np.array([int(d["epoch"]) for d in items], dtype=float)

        if method == "median":
            e = int(np.round(np.median(epochs)))

        elif method == "mean":
            e = int(np.round(np.mean(epochs)))

        elif method == "wmean":
            vals = np.array([float(d.get("val", np.nan)) for d in items], dtype=float)
            # si val tiene NaN o pesos raros, fallback a median
            if not np.all(np.isfinite(vals)) or np.all(vals == vals[0]):
                e = int(np.round(np.median(epochs)))
            else:
                # Normaliza pesos positivos
                w = vals - np.min(vals) + 1e-9
                e = int(np.round(np.average(epochs, weights=w)))
        else:
            raise ValueError(f"Unknown method={method}")

        return max(1, e)

    # --------
    # 1) Search space -> ModelConfig
    # --------
    def _suggest_model_config(self, trial: optuna.Trial) -> ModelConfig:
        model_config = ModelConfig(**vars(self.base_model_config))

        # Arquitectura
        model_config.conv1d_filters = trial.suggest_categorical('conv1d_filters', self._choices('conv1d_filters'))
        model_config.lstm_units = trial.suggest_categorical('lstm_units', self._choices('lstm_units'))
        model_config.context_units = trial.suggest_categorical('context_units', self._choices('context_units'))
        model_config.time_units = trial.suggest_categorical('time_units', self._choices('time_units'))
        model_config.head_units = trial.suggest_categorical('head_units', self._choices('head_units'))

        # Regularización
        model_config.dropout_seq = trial.suggest_categorical('dropout_seq', self._choices('dropout_seq'))
        model_config.dropout_dense = trial.suggest_categorical('dropout_dense', self._choices('dropout_dense'))
        model_config.dropout_lstm = trial.suggest_categorical('dropout_lstm', self._choices('dropout_lstm'))
        model_config.l2_reg = trial.suggest_categorical('l2_reg', self._choices('l2_reg'))

        # Entrenamiento
        model_config.learning_rate = trial.suggest_categorical('learning_rate', self._choices('learning_rate'))
        model_config.batch_size = trial.suggest_categorical('batch_size', self._choices('batch_size'))

        # Focal loss
        model_config.focal_alpha = trial.suggest_categorical('focal_alpha', self._choices('focal_alpha'))
        model_config.focal_gamma = trial.suggest_categorical('focal_gamma', self._choices('focal_gamma'))

        # Flags
        model_config.use_attention = trial.suggest_categorical('use_attention', self._choices('use_attention'))
        model_config.use_gate = trial.suggest_categorical('use_gate', self._choices('use_gate'))

        # OOF epochs por fold (ligero para Optuna)
        model_config.epochs = trial.suggest_categorical('epochs', self._choices('epochs'))
        model_config.patience = trial.suggest_categorical('patience', self._choices('patience'))

        return model_config

    def _trial_oof_cache_paths(self, *, study_name: str, side: str, trial_number: int) -> Dict[str, str]:
        base = os.path.join(self.out_dir, "_optuna_cache", study_name, side, f"trial_{trial_number:05d}")
        ensure_dir(base)
        return {
            "dir": base,
            "oof_df": os.path.join(base, "oof_df.parquet"),
            "cal": os.path.join(base, "calibrator.joblib"),
            "best_epochs": os.path.join(base, "best_epochs.json"),
            'pct': os.path.join(base, 'percentiles.json'),
            "meta": os.path.join(base, "oof_meta.json"),
        }

    # --------
    # 2) Objective OOF
    # --------
    def _objective(
        self,
        trial: optuna.Trial,
        df_rates: pd.DataFrame,
        side: str,
        *,
        n_splits: int,
        epochs_per_fold: int,
        quantiles=(50, 60, 70, 75, 80, 85, 90, 95, 96, 97, 98, 99),
        max_signal_rate_penalty: float = 0.20,
    ) -> float:

        # 1) Config trial
        model_config = self._suggest_model_config(trial)
        print(f'Side: {side}. Parameters: {model_config}')

        # 2) Pipeline (usa tus clases)
        pipeline = DataPipeline(
            general_config=self.general_config,
            feature_config=self.feature_config,
            model_config=model_config,
            regime_config=self.regime_config,
        )

        # 3) Preparar DF + labels según side
        df_prepared = pipeline.prepare_data(df_rates, labels=True, side=side, set_market_condition=False, ensure_regime=True)
        pos_rate = float(np.nanmean(df_prepared['signal'], axis=0))
        print(f'pos_rate: {pos_rate:.4f}')

        # 4) OOF + calibración isotónica (tu clase)
        _cal_temp = self.temperature_long if side == 'long' else self.temperature_short
        probs_calibrator = ProbsCalibration(self.general_config, model_config, temperature=_cal_temp)
        df_oof, calibrator, best_epochs = probs_calibrator.generate_oof_predictions(
            df_prepared=df_prepared,
            pipeline=pipeline,
            side=side,
            n_splits=n_splits,
            epochs_per_fold=epochs_per_fold,
            verbose=1
        )

        # 4b) Persistir cache OOF por trial (para reusar en prepare_production_model)
        study_name = trial.study.study_name  # nombre real del estudio
        paths = self._trial_oof_cache_paths(study_name=study_name, side=side, trial_number=trial.number)

        # guarda OOF df + calibrador + best_epochs
        df_oof.to_parquet(paths["oof_df"], index=False)
        joblib.dump(calibrator, paths["cal"])
        save_json(best_epochs, paths["best_epochs"])

        # meta mínima (opcional, pero útil para auditoría)
        oof_meta = {
            "release": self.general_config.release,
            "side": side,
            "trial_number": trial.number,
            "params": dict(trial.params),
            "saved_at": pd.Timestamp.utcnow().isoformat(),
        }
        save_json(oof_meta, paths["meta"])

        # registra rutas en el trial para recuperarlas luego sin recalcular
        trial.set_user_attr("oof_df_path", paths["oof_df"])
        trial.set_user_attr("cal_path", paths["cal"])
        trial.set_user_attr("best_epochs_path", paths["best_epochs"])
        trial.set_user_attr('pct_path', paths['pct'])
        trial.set_user_attr("oof_meta_path", paths["meta"])

        trial.set_user_attr('best_epochs', best_epochs)
        trial.set_user_attr('best_epoch_agg', self.aggregate_best_epochs(best_epochs, method='median'))

        # 5) Métricas sobre oof_proba_cal
        m = mask_oof(df_oof, "oof_proba_cal")
        if m.sum() < 2000:
            # muy poca muestra útil => descarta trial
            raise optuna.exceptions.TrialPruned()

        y = df_oof.loc[m, "signal"].to_numpy().astype(int)
        p_cal = df_oof.loc[m, "oof_proba_cal"].to_numpy().astype(float)

        # AUC-PR como core
        auc_pr = float(average_precision_score(y, p_cal))

        # Métrica operativa (umbral) con tu evaluator
        eval_metrics = self.evaluator.evaluate_predictions(
            y_true=y,
            y_pred_proba=p_cal,
            verbose=False,
            beta_primary=0.25,
            min_precision=0.45,
            max_signal_rate=0.15,
        )

        sr = float(eval_metrics.get("selected_signal_rate", np.nan))
        prec = float(eval_metrics.get("selected_precision", 0.0))

        # Penalización si se va de señal
        penalty = 0.0
        if np.isfinite(sr) and sr > max_signal_rate_penalty:
            penalty = (sr - max_signal_rate_penalty) * 0.5

        # objetivo final
        # - prioriza AUC-PR
        # - añade “precision” operativa
        # score = auc_pr + 0.10 * prec - penalty
        score = (
                auc_pr
                + 0.30 * prec
                - 0.80 * max(0.0, sr - 0.12)
        )

        ''' Probar esto 
        score = (
                auc_pr
                + 0.30 * prec
                - 0.50 * max(0.0, sr - 0.12)
        )
        '''

        # 6) Guardar percentiles por régimen (para debug/inspección del trial)
        percentiles = probs_calibrator.compute_percentiles_by_regime(
            df_with_oof=df_oof,
            proba_col="oof_proba_cal",
            regime_col="state",
            quantiles=quantiles,
            min_n=800,
        )

        save_json(percentiles, paths['pct'])

        # log a Optuna
        trial.set_user_attr("auc_pr", auc_pr)
        trial.set_user_attr("selected_precision", prec)
        trial.set_user_attr("selected_signal_rate", sr)
        trial.set_user_attr("percentiles_meta", percentiles.get("_meta", {}))

        return float(score)

    def _choices(self, name: str):
        if not self.grid_space or name not in self.grid_space:
            raise RuntimeError(f'No grid space for {name}')

        return self.grid_space[name]

    # --------
    # 3) Optimize
    # --------
    def optimize(
        self,
        *,
        df_rates: pd.DataFrame,
        side: str,
        n_trials: Optional[int] = None,
        n_splits: Optional[int] = None,
        epochs_per_fold: Optional[int] = None,
        use_grid: bool = False,
        grid_space: Optional[Dict[str, list]] = None,
        load_if_exists: bool = True
    ) -> optuna.Study:

        self.grid_space = grid_space

        if n_splits is None:
            n_splits = int(self.general_config.oof_splits)
        if epochs_per_fold is None:
            epochs_per_fold = int(self.general_config.oof_epochs)

        if use_grid:
            if not grid_space:
                raise ValueError("use_grid=True requiere grid_space con listas por parámetro.")

            sampler = optuna.samplers.GridSampler(grid_space, seed=self.seed)
            n_trials = min(n_trials, len(sampler._all_grids)) if n_trials is not None else len(sampler._all_grids)
        else:
            sampler = optuna.samplers.TPESampler(seed=self.seed)
            if n_trials is None:
                raise ValueError('use_grid=False requiere especificar n_trials')

        print(f'Se ejecutarán un total de {n_trials} trials')

        pruner = optuna.pruners.MedianPruner(n_warmup_steps=max(2, n_trials // 5))

        study_name = f"{self.study_prefix}_{self.general_config.release}_{side}"
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=self.optuna_db,
            load_if_exists=load_if_exists,
            sampler=sampler,
            pruner=pruner,
        )

        def obj(trial: optuna.Trial):
            return self._objective(
                trial,
                df_rates=df_rates,
                side=side,
                n_splits=n_splits,
                epochs_per_fold=epochs_per_fold,
            )

        study.optimize(obj, n_trials=n_trials, gc_after_trial=True)

        # guardar best params
        self.best_params_by_side[side] = dict(study.best_trial.params)
        best_mc = self._model_config_from_params(self.best_params_by_side[side])
        best_epochs = study.best_trial.user_attrs.get('best_epochs')
        if best_epochs:
            best_mc.epochs = self.aggregate_best_epochs(best_epochs, method='median')

        self.best_model_config_by_side[side] = best_mc

        return study

    def _model_config_from_params(self, params: Dict[str, Any]) -> ModelConfig:
        mc = ModelConfig(**vars(self.base_model_config))
        for k, v in params.items():
            if hasattr(mc, k):
                setattr(mc, k, v)
        return mc

    # --------
    # 4) Prepare production artifacts (final train + export)
    # --------
    def prepare_production_model(
        self,
        *,
        df_rates: pd.DataFrame,
        side: str,
        quantiles=(50, 60, 70, 75, 80, 85, 90, 95, 96, 97, 98, 99),
        save_oof_df: bool = True,
        reuse_best_trial_oof: bool = True,
        inference_policy: str = "transform",
    ) -> TrainerArtifacts:

        if side not in self.best_model_config_by_side:
            raise ValueError(f"No hay best params para side={side}. Ejecuta optimize() primero.")

        if inference_policy not in ("transform", "transform_then_update"):
            raise ValueError(
                f"inference_policy inválida: {inference_policy}. "
                f"Valores permitidos: 'transform', 'transform_then_update'"
            )

        release = self.general_config.release
        best_params = self.best_params_by_side[side]
        params_sig = params_signature(best_params)

        model_config = ModelConfig(**vars(self.best_model_config_by_side[side]))
        model_config.epochs = int(model_config.epochs * 1.05)

        oof_meta_path = os.path.join(self.out_dir, f'oof_meta_{release}_{side}.json')
        oof_df_path = os.path.join(self.out_dir, f'oof_{release}_{side}.parquet')
        cal_path = os.path.join(self.out_dir, f'oof_calibrator_{release}_{side}.joblib')
        pct_path = os.path.join(self.out_dir, f'percentiles_{release}_{side}.json')
        model_path = os.path.join(self.out_dir, f'model_{release}_{side}.keras')
        scalers_path = os.path.join(self.out_dir, f'scalers_{release}.joblib')

        reuse_ok = False
        meta_existing: Optional[Dict[str, Any]] = None

        if reuse_best_trial_oof and (not (os.path.exists(oof_df_path) and os.path.exists(cal_path))):
            # OJO: aquí necesitas acceder al best_trial del study.
            # Si no lo guardas, crea de nuevo el study con load_if_exists=True y lee best_trial.
            study_name = f"{self.study_prefix}_{release}_{side}"
            study = optuna.create_study(
                study_name=study_name,
                storage=self.optuna_db,
                load_if_exists=True,
                direction="maximize",
            )
            bt = study.best_trial

            src_oof = bt.user_attrs.get("oof_df_path")
            src_cal = bt.user_attrs.get("cal_path")
            src_pct = bt.user_attrs.get('pct_path')
            src_meta = bt.user_attrs.get("oof_meta_path")

            ensure_dir(os.path.dirname(oof_df_path))

            if src_oof and os.path.exists(src_oof) and not os.path.exists(oof_df_path):
                pd.read_parquet(src_oof).to_parquet(oof_df_path, index=False)

            if src_cal and os.path.exists(src_cal) and not os.path.exists(cal_path):
                joblib.dump(joblib.load(src_cal), cal_path)

            if src_pct and os.path.exists(src_pct) and not os.path.exists(pct_path):
                with open(src_pct, 'r', encoding='utf8') as f:
                    pct_obj = json.load(f)
                save_json(pct_obj, pct_path)

            if src_meta and os.path.exists(src_meta) and not os.path.exists(oof_meta_path):
                with open(src_meta, 'r', encoding='utf8') as f:
                    meta_obj = json.load(f)
                save_json(meta_obj, oof_meta_path)

        if reuse_best_trial_oof and all(os.path.exists(p) for p in [oof_meta_path, oof_df_path, cal_path, pct_path]):
            reuse_ok = True
        else:
            reuse_ok = False

        if reuse_ok:
            print(f"[OOF] Reusing cached OOF artifacts for release={release} side={side}")
            df_oof = pd.read_parquet(oof_df_path)
            calibrator = joblib.load(cal_path)
            percentiles = load_json(pct_path)
        else:
            print(f"[OOF] Computing OOF artifacts for release={release} side={side}")
            pipeline_oof = DataPipeline(
                general_config=self.general_config,
                feature_config=self.feature_config,
                model_config=model_config,
                regime_config=self.regime_config,
            )

            df_prepared = pipeline_oof.prepare_data(df_rates, labels=True, side=side, set_market_condition=False, ensure_regime=True)

            _cal_temp = self.temperature_long if side == 'long' else self.temperature_short
            probs_calibrator = ProbsCalibration(self.general_config, model_config, temperature=_cal_temp)
            df_oof, calibrator, best_epochs = probs_calibrator.generate_oof_predictions(
                df_prepared=df_prepared,
                pipeline=pipeline_oof,
                side=side,
                n_splits=int(self.general_config.oof_splits),
                epochs_per_fold=int(self.general_config.oof_epochs),
                verbose=1,
            )

            percentiles = probs_calibrator.compute_percentiles_by_regime(
                df_with_oof=df_oof,
                proba_col="oof_proba_cal",
                regime_col="state",
                quantiles=quantiles,
                min_n=800,
            )

            # persistir artefactos OOF
            df_oof.to_parquet(oof_df_path, index=False)
            joblib.dump(calibrator, cal_path)
            save_json(percentiles, pct_path)

            oof_meta = {
                'release': release,
                'side': side,
                'params_sig': params_sig,
                'quantiles': list(quantiles),
                'saved_at': pd.Timestamp.utcnow().isoformat(),
            }
            save_json(oof_meta, oof_meta_path)

        # 3) Evaluación OOF SIEMPRE (auditoría)
        m = mask_oof(df_oof, "oof_proba_cal")
        y_oof = df_oof.loc[m, "signal"].to_numpy().astype(int)
        p_cal = df_oof.loc[m, "oof_proba_cal"].to_numpy().astype(float)

        oof_eval = self.evaluator.evaluate_predictions(
            y_true=y_oof,
            y_pred_proba=p_cal,
            verbose=True,
            beta_primary=0.25,
            min_precision=0.45,
            max_signal_rate=0.15,
        )

        # opcional: añade resumen AUC-PR explícito por claridad
        oof_eval["auc_pr"] = float(average_precision_score(y_oof, p_cal))

        # 4) Entrenar modelo final en TODO (producción)
        pipeline_prod = DataPipeline(
            general_config=self.general_config,
            feature_config=self.feature_config,
            model_config=model_config,
            regime_config=self.regime_config,
        )

        df_prepared_prod = pipeline_prod.prepare_data(df_rates, labels=True, side=side, set_market_condition=False, ensure_regime=True)
        if pipeline_prod.feature_config.feature_masks is not None:
            sequences_all = pipeline_prod.create_sequences_by_side(df_prepared_prod, sides=(side,), fit_scalers=True, train=True)
            data_all = sequences_all[side]
        else:
            data_all = pipeline_prod.create_sequences(df_prepared_prod, fit_scalers=True, train=True)

        X_all = {k: v for k, v in data_all.items() if k not in ["labels", "weights"]}
        y_all = data_all["labels"]
        w_all = data_all["weights"]

        tm = TradingModel(self.general_config, model_config, side=side)

        pos_rate = float(np.nanmean(y_all))
        pos_rate = np.clip(pos_rate, 0.05, 0.50)
        init_bias = float(np.log(pos_rate / (1 - pos_rate)))
        init_bias = np.clip(init_bias, -2.0, 2.0)
        print(f'[BIAS] pos_rate={pos_rate:.4f}, init_bias={init_bias:.4f}')

        # Seleccionar arquitectura según flag del ModelConfig
        # use_hierarchical_fusion=True → v3 (fusión jerárquica market/entry)
        # use_hierarchical_fusion=False → v2 (fusión plana, default)
        _build_fn = (
            tm.build_model_v3
            if getattr(model_config, 'use_hierarchical_fusion', False)
            else tm.build_model_v2
        )
        _build_fn(
            shape_short=(model_config.seq_len_short, X_all["seq_short"].shape[-1]),
            shape_long=(model_config.seq_len_long, X_all["seq_long"].shape[-1]),
            n_context=X_all["context"].shape[-1],
            n_time=X_all["time"].shape[-1],
            init_bias=init_bias,
        )
        tm.compile_model()

        tm.train(
            X_train=X_all,
            y_train=y_all,
            X_val=None,
            y_val=None,
            sample_weight=w_all,
            verbose=1,
            for_production=True,
        )

        # 5) Export artefactos
        tm.save(self.out_dir)

        # calibrator y percentiles ya están guardados (reusado o recalculado), pero los re-aseguramos:
        joblib.dump(calibrator, cal_path)
        save_json(percentiles, pct_path)

        # ── UMBRALES DE RÉGIMEN + SCALERS ──────────────────────────────────
        # compute_and_store_regime_thresholds() DEBE llamarse ANTES de save_scalers()
        # para que los umbrales queden en meta.json y load_scalers() los inyecte.
        # Guardamos siempre (sobreescribiendo si ya existía) para garantizar que
        # los thresholds están presentes — un meta.json sin thresholds es incorrecto.
        try:
            pipeline_prod.compute_and_store_regime_thresholds(df_prepared_prod)
            print(f"[REGIME] Umbrales de régimen persistidos en {self.out_dir}")
        except Exception as _e_regime:
            print(f"[REGIME] WARNING: No se pudieron calcular umbrales de régimen: {_e_regime}")

        pipeline_prod.rolling_infer_policy = inference_policy
        print(f"[INFER] rolling_infer_policy persistida para {side.upper()}: {pipeline_prod.rolling_infer_policy}")

        saved_scalers_path = pipeline_prod.save_scalers(self.out_dir, include_buffer=True)
        _snapshot_side_specific_scalers(pipeline_prod, saved_scalers_path, self.out_dir, release, side)
        # ─────────────────────────────────────────────────────────────────────

        # guardar oof df si se pidió (si reusamos ya existía igualmente)
        if save_oof_df and not os.path.exists(oof_df_path):
            df_oof.to_parquet(oof_df_path, index=False)

        return TrainerArtifacts(
            best_params=best_params,
            best_value=float("nan"),
            study_name=f"{self.study_prefix}_opt_{release}_{side}",
            side=side,
            oof_metrics=oof_eval,
            percentiles=percentiles,
            calibrator_path=cal_path,
            percentiles_path=pct_path,
            model_path=model_path,
            oof_meta_path=oof_meta_path,
            oof_df_path=oof_df_path if save_oof_df else None,
        )

    def evaluate_holdout(self, artifacts, df_hold, side: str):
        import joblib
        import numpy as np
        import tensorflow as tf
        from sklearn.metrics import average_precision_score

        pipeline = self._build_eval_pipeline(artifacts, side, inference_policy='transform')
        df_prep = pipeline.prepare_data(
            df_hold.copy(),
            labels=True,
            side=side,
            set_market_condition=False,
            ensure_regime=True
        )

        if pipeline.feature_config.feature_masks is not None:
            sequences = pipeline.create_sequences_by_side(
                df_prep, sides=(side,), fit_scalers=False, train=True
            )
            data = sequences[side]
        else:
            data = pipeline.create_sequences(
                df_prep, fit_scalers=False, train=True
            )

        X = {k: v for k, v in data.items() if k not in ['labels', 'weights']}
        y_true = data['labels'].astype(int)

        keras_model = tf.keras.models.load_model(artifacts.model_path)

        x_list = [X['seq_short'], X['seq_long'], X['context'], X['time']]
        y_pred = keras_model.predict(x_list, verbose=0, batch_size=4096).reshape(-1)

        calibrator = joblib.load(artifacts.calibrator_path)
        y_cal = calibrator.predict(y_pred) if hasattr(calibrator, 'predict') else calibrator.transform(y_pred)

        m = np.isfinite(y_cal) & np.isfinite(y_true)
        yv = y_true[m]
        pv = y_cal[m]

        out = self.evaluator.evaluate_predictions(
            y_true=yv,
            y_pred_proba=pv,
            verbose=True,
            beta_primary=0.25,
            min_precision=0.45,
            max_signal_rate=0.15
        )

        out['auc_pr'] = float(average_precision_score(yv, pv))
        return out

    def evaluate_holdout_v0(self, artifacts, df_hold, side: str):

        model_config = self.best_model_config_by_side.get(side, self.base_model_config)
        pipeline = DataPipeline(
            general_config=self.general_config,
            feature_config=self.feature_config,
            model_config=model_config,
            regime_config=self.regime_config
        )

        path = self.out_dir
        pipeline.load_scalers(base_path=path)

        df_prep = pipeline.prepare_data(df_hold, labels=True, side=side, set_market_condition=False, ensure_regime=True)
        if pipeline.feature_config.feature_masks is not None:
            sequences = pipeline.create_sequences_by_side(df_prep, sides=(side, ), fit_scalers=False, train=True)
            data = sequences[side]
        else:
            data = pipeline.create_sequences(df_prep, fit_scalers=False, train=True)

        X = {k: v for k, v in data.items() if k not in ['labels', 'weights']}
        y_true = data['labels'].astype(int)

        helper = Helper(general_config=self.general_config, path=path)
        model = helper.load_model(side=side)
        # Acceder al modelo Keras directamente para poder pasar batch_size y evitar
        # retracing de tf.function por shapes distintas entre llamadas.
        # helper.load_model() puede devolver un TradingModel wrapper o un Keras model.
        _keras_model = getattr(model, 'model', model)  # TradingModel.model o el modelo mismo
        _x_list = [X['seq_short'], X['seq_long'], X['context'], X['time']]
        y_pred = _keras_model.predict(_x_list, verbose=0, batch_size=4096).reshape(-1)

        cal_path = os.path.join(path, f'oof_calibrator_{self.general_config.release}_{side}.joblib')
        calibrator = joblib.load(cal_path)

        y_cal = calibrator.predict(y_pred) if hasattr(calibrator, 'predict') else calibrator.transform(y_pred)

        evaluator = ModelEvaluator()
        m = np.isfinite(y_cal) & np.isfinite(y_true)
        yv = y_true[m]
        pv = y_cal[m]

        out = evaluator.evaluate_predictions(
            y_true=yv, y_pred_proba=pv, verbose=True,
            beta_primary=0.25, min_precision=0.45, max_signal_rate=0.15
        )

        out['auc_pr'] = float(average_precision_score(yv, pv))
        return out

    def evaluate_holdout_walkforward(
            self,
            artifacts,
            df_hold,
            side: str,
            *,
            min_rows: int | None = None,
            return_predictions: bool = False,
    ):
        """
        Walk-forward más eficiente:
          - sin df_slice por iteración
          - sin keras_model.predict() por iteración
          - usando create_last_sample_by_side_from_index(...)
        """
        import joblib
        import numpy as np
        import tensorflow as tf
        from sklearn.metrics import average_precision_score, roc_auc_score

        if artifacts is None:
            raise ValueError("artifacts no puede ser None")

        if side not in ("long", "short"):
            raise ValueError(f"side inválido: {side}")

        model_config = self.best_model_config_by_side.get(side, self.base_model_config)
        pipeline = self._build_eval_pipeline(
            artifacts,
            side,
            inference_policy="transform_then_update"
        )

        df_prep = pipeline.prepare_data(
            df_hold.copy(),
            labels=True,
            side=side,
            set_market_condition=False,
            ensure_regime=True,
        )

        if min_rows is None:
            min_rows = int(model_config.seq_len_long)

        if len(df_prep) < min_rows:
            raise ValueError(
                f"Holdout insuficiente tras prepare_data(): {len(df_prep)} filas < min_rows={min_rows}"
            )

        keras_model = tf.keras.models.load_model(artifacts.model_path)
        calibrator = joblib.load(artifacts.calibrator_path)

        y_true_all = []
        y_pred_all = []
        pred_rows = []

        for end_idx in range(min_rows, len(df_prep) + 1):
            data = pipeline.create_last_sample_by_side_from_index(
                df_prep,
                end_idx=end_idx,
                side=side,
                fit_scalers=False,
                train=True,
            )

            labels_last = data.get("labels", None)
            if labels_last is None or len(labels_last) == 0:
                continue

            X_last = [
                data["seq_short"],
                data["seq_long"],
                data["context"],
                data["time"],
            ]

            y_true = int(labels_last[0])

            # Más ligero que predict() en loops largos
            y_raw_tensor = keras_model(X_last, training=False)
            y_raw = float(np.asarray(y_raw_tensor).reshape(-1)[0])

            if hasattr(calibrator, "predict"):
                y_cal = float(calibrator.predict(np.array([y_raw]))[0])
            else:
                y_cal = float(calibrator.transform(np.array([y_raw]))[0])

            y_true_all.append(y_true)
            y_pred_all.append(y_cal)

            if return_predictions:
                row_last = df_prep.iloc[end_idx - 1]
                pred_rows.append({
                    "time": (
                        row_last["time"].isoformat()
                        if hasattr(row_last["time"], "isoformat")
                        else str(row_last["time"])
                    ),
                    "state": row_last["state"] if "state" in row_last else None,
                    "y_true": y_true,
                    "y_pred_raw": y_raw,
                    "y_pred_cal": y_cal,
                })

        if len(y_true_all) == 0:
            raise RuntimeError("No se generaron predicciones walk-forward válidas")

        y_true_arr = np.asarray(y_true_all, dtype=np.int32)
        y_pred_arr = np.asarray(y_pred_all, dtype=np.float32)

        mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
        y_true_arr = y_true_arr[mask]
        y_pred_arr = y_pred_arr[mask]

        if len(y_true_arr) == 0:
            raise RuntimeError("Todas las predicciones walk-forward resultaron inválidas (NaN/Inf)")

        eval_dict = self.evaluator.evaluate_predictions(
            y_true=y_true_arr,
            y_pred_proba=y_pred_arr,
            verbose=True,
            beta_primary=0.25,
            min_precision=0.45,
            max_signal_rate=0.15,
        )

        try:
            eval_dict["auc_pr"] = float(average_precision_score(y_true_arr, y_pred_arr))
        except Exception:
            eval_dict["auc_pr"] = float("nan")

        try:
            eval_dict["auc_roc"] = float(roc_auc_score(y_true_arr, y_pred_arr))
        except Exception:
            eval_dict["auc_roc"] = float("nan")

        out = {
            "side": side,
            "mode": "walkforward_transform_then_update_v2",
            "n_samples": int(len(y_true_arr)),
            "metrics": eval_dict,
        }

        if return_predictions:
            out["predictions"] = pred_rows

        return out

    def evaluate_holdout_walkforward_v0(
            self,
            artifacts,
            df_hold,
            side: str,
            *,
            min_rows: int | None = None,
            batch_size: int = 1,
    ):
        import os
        import joblib
        import numpy as np
        import pandas as pd
        from sklearn.metrics import average_precision_score, roc_auc_score

        model_config = self.best_model_config_by_side.get(side, self.base_model_config)

        pipeline = DataPipeline(
            general_config=self.general_config,
            feature_config=self.feature_config,
            model_config=model_config,
            regime_config=self.regime_config
        )

        path = self.out_dir

        # Igual que en validación final, mejor cargar scalers side-specific si existen
        side_dir = os.path.join(path, f"scalers_{self.general_config.release}_{side}")
        generic_dir = os.path.join(path, f"scalers_{self.general_config.release}")

        if os.path.isdir(side_dir):
            import shutil
            import tempfile
            tmp_root = tempfile.mkdtemp(prefix=f"scalers_{side}_")
            shutil.copytree(side_dir, os.path.join(tmp_root, f"scalers_{self.general_config.release}"))
            pipeline.load_scalers(base_path=tmp_root)
        else:
            pipeline.load_scalers(base_path=path if os.path.isdir(generic_dir) else path)

        # clave
        pipeline.rolling_infer_policy = "transform_then_update"
        df_prep = pipeline.prepare_data(
            df_hold.copy(),
            labels=True,
            side=side,
            set_market_condition=False,
            ensure_regime=True
        )

        helper = Helper(general_config=self.general_config, path=path)
        model = helper.load_model(side=side)
        _keras_model = getattr(model, "model", model)

        cal_path = os.path.join(path, f"oof_calibrator_{self.general_config.release}_{side}.joblib")
        calibrator = joblib.load(cal_path)

        if min_rows is None:
            min_rows = int(model_config.seq_len_long) + 5

        rows = []
        y_true_all = []
        y_pred_all = []

        for end_idx in range(min_rows, len(df_prep) + 1):
            df_slice = df_prep.iloc[:end_idx].copy()

            if pipeline.feature_config.feature_masks is not None:
                sequences = pipeline.create_sequences_by_side(
                    df_slice,
                    sides=(side,),
                    fit_scalers=False,
                    train=True
                )
                data = sequences[side]
            else:
                data = pipeline.create_sequences(
                    df_slice,
                    fit_scalers=False,
                    train=True
                )

            X_last = [
                data["seq_short"][-1:],
                data["seq_long"][-1:],
                data["context"][-1:],
                data["time"][-1:]
            ]
            y_true = int(data["labels"][-1])

            y_raw = float(_keras_model.predict(X_last, verbose=0, batch_size=batch_size).reshape(-1)[0])
            y_cal = float(calibrator.predict(np.array([y_raw]))[0])

            y_true_all.append(y_true)
            y_pred_all.append(y_cal)

            row_last = df_slice.iloc[-1]
            rows.append({
                "time": row_last["time"] if "time" in row_last else None,
                "state": row_last["state"] if "state" in row_last else None,
                "y_true": y_true,
                "y_pred_raw": y_raw,
                "y_pred_cal": y_cal,
            })

        y_true_arr = np.asarray(y_true_all, dtype=np.int32)
        y_pred_arr = np.asarray(y_pred_all, dtype=np.float32)

        eval_dict = self.evaluator.evaluate_predictions(
            y_true=y_true_arr,
            y_pred_proba=y_pred_arr,
            verbose=True,
            beta_primary=0.25,
            min_precision=0.45,
            max_signal_rate=0.15
        )

        try:
            eval_dict["auc_pr"] = float(average_precision_score(y_true_arr, y_pred_arr))
        except Exception:
            eval_dict["auc_pr"] = float("nan")

        try:
            eval_dict["auc_roc"] = float(roc_auc_score(y_true_arr, y_pred_arr))
        except Exception:
            eval_dict["auc_roc"] = float("nan")

        return {
            "side": side,
            "mode": "walkforward_transform_then_update",
            "n_samples": int(len(y_true_arr)),
            "metrics": eval_dict,
            "predictions": pd.DataFrame(rows).to_dict(orient="records"),
        }

    def _build_eval_pipeline(self, artifacts, side: str, inference_policy: str):
        model_config = self.best_model_config_by_side.get(side, self.base_model_config)

        pipeline = DataPipeline(
            general_config=self.general_config,
            feature_config=self.feature_config,
            model_config=model_config,
            regime_config=self.regime_config,
        )

        load_pipeline_scalers_for_side(pipeline, artifacts.model_path, side)
        pipeline.rolling_infer_policy = inference_policy
        return pipeline

    def choose_inference_policy(
            self,
            holdout_static: dict,
            holdout_walk: dict,
            *,
            min_auc_pr_gain: float = 0.01,
            min_precision_gain: float = 0.00,
            max_signal_rate_ratio: float = 1.20,
    ) -> dict:
        """
        Decide la policy de inferencia comparando holdout estático vs walk-forward.

        Convención:
          - "transform"              => modo estático / frozen scalers
          - "transform_then_update"  => modo walk-forward / causal

        Retorna:
          {
            "selected_policy": "transform" | "transform_then_update",
            "selected_mode_label": "static" | "walkforward",
            "is_walkforward_selected": bool,
            "reason": {...}
          }
        """

        s_metrics = holdout_static.get("metrics", holdout_static)
        w_metrics = holdout_walk.get("metrics", holdout_walk)

        s_auc_pr = float(s_metrics.get("auc_pr", float("nan")))
        w_auc_pr = float(w_metrics.get("auc_pr", float("nan")))

        s_prec = float(
            s_metrics.get("selected_precision", s_metrics.get("precision", float("nan")))
        )
        w_prec = float(
            w_metrics.get("selected_precision", w_metrics.get("precision", float("nan")))
        )

        s_sr = float(
            s_metrics.get("selected_signal_rate", s_metrics.get("signal_rate", float("nan")))
        )
        w_sr = float(
            w_metrics.get("selected_signal_rate", w_metrics.get("signal_rate", float("nan")))
        )

        delta_auc_pr = w_auc_pr - s_auc_pr
        delta_prec = w_prec - s_prec

        if s_sr <= 1e-12:
            sr_ratio = float("inf") if w_sr > 0 else 1.0
        else:
            sr_ratio = w_sr / s_sr

        choose_walk = (
                np.isfinite(delta_auc_pr)
                and np.isfinite(delta_prec)
                and delta_auc_pr >= min_auc_pr_gain
                and delta_prec >= min_precision_gain
                and sr_ratio <= max_signal_rate_ratio
        )

        selected_policy = "transform_then_update" if choose_walk else "transform"
        selected_mode_label = "walkforward" if choose_walk else "static"

        return {
            "selected_policy": selected_policy,
            "selected_mode_label": selected_mode_label,
            "is_walkforward_selected": bool(choose_walk),
            "reason": {
                "static_auc_pr": s_auc_pr,
                "walk_auc_pr": w_auc_pr,
                "delta_auc_pr": delta_auc_pr,
                "static_precision": s_prec,
                "walk_precision": w_prec,
                "delta_precision": delta_prec,
                "static_signal_rate": s_sr,
                "walk_signal_rate": w_sr,
                "signal_rate_ratio": sr_ratio,
                "thresholds": {
                    "min_auc_pr_gain": min_auc_pr_gain,
                    "min_precision_gain": min_precision_gain,
                    "max_signal_rate_ratio": max_signal_rate_ratio,
                },
            },
        }

    def persist_selected_inference_policy(
            self,
            *,
            side: str,
            selected_policy: str,
            decision: Optional[dict] = None,
    ) -> Dict[str, Any]:
        """
        Persiste la policy final elegida tras holdout.

        Escribe:
          - out_dir/inference_policy_<release>_<side>.json
          - out_dir/oof_meta_<release>_<side>.json   (merge)
          - out_dir/scalers_<release>_<side>/meta.json (merge, si existe)

        Nota:
          No toca scalers_<release>/meta.json genérico para no pisar el otro side
          cuando LONG y SHORT comparten carpeta base.
        """
        if selected_policy not in ("transform", "transform_then_update"):
            raise ValueError(
                f"selected_policy inválida: {selected_policy}. "
                f"Valores permitidos: 'transform', 'transform_then_update'"
            )

        release = self.general_config.release
        selected_mode_label = "walkforward" if selected_policy == "transform_then_update" else "static"

        payload = {
            "release": release,
            "side": side,
            "selected_policy": selected_policy,
            "selected_mode_label": selected_mode_label,
            "is_walkforward_selected": bool(selected_policy == "transform_then_update"),
            "saved_at": pd.Timestamp.utcnow().isoformat(),
        }

        if decision is not None:
            payload["decision"] = decision

        policy_path = os.path.join(self.out_dir, f"inference_policy_{release}_{side}.json")
        save_json(payload, policy_path)

        patched_paths = []

        def _merge_json(path: str, patch: Dict[str, Any]):
            if not os.path.exists(path):
                return
            try:
                with open(path, "r", encoding="utf-8") as f:
                    obj = json.load(f)
            except Exception:
                obj = {}

            if not isinstance(obj, dict):
                obj = {"_previous_value": obj}

            obj["selected_inference_policy"] = selected_policy
            obj["selected_mode_label"] = selected_mode_label
            obj["is_walkforward_selected"] = bool(selected_policy == "transform_then_update")
            obj["policy_saved_at"] = payload["saved_at"]

            if decision is not None:
                obj["holdout_policy"] = decision

            with open(path, "w", encoding="utf-8") as f:
                json.dump(obj, f, ensure_ascii=False, indent=2)

            patched_paths.append(path)

        # meta OOF del side
        _merge_json(
            os.path.join(self.out_dir, f"oof_meta_{release}_{side}.json"),
            payload,
        )

        # meta side-specific scalers
        _merge_json(
            os.path.join(self.out_dir, f"scalers_{release}_{side}", "meta.json"),
            payload,
        )

        return {
            "policy_path": policy_path,
            "patched_paths": patched_paths,
            "selected_policy": selected_policy,
            "selected_mode_label": selected_mode_label,
        }

    def evaluate_holdout_walkforward_fast(
            self,
            artifacts,
            df_hold,
            side: str,
            *,
            min_rows: int | None = None,
            inference_batch_size: int = 64,
            return_predictions: bool = False,
    ):
        """
        Walk-forward rápido:
          - pandas fuera del loop
          - selección de columnas fuera del loop
          - scaler causal paso a paso
          - inferencia en micro-lotes
        """
        import joblib
        import numpy as np
        import tensorflow as tf
        from sklearn.metrics import average_precision_score, roc_auc_score

        if artifacts is None:
            raise ValueError("artifacts no puede ser None")

        if side not in ("long", "short"):
            raise ValueError(f"side inválido: {side}")

        model_config = self.best_model_config_by_side.get(side, self.base_model_config)
        pipeline = self._build_eval_pipeline(
            artifacts,
            side,
            inference_policy="transform_then_update"
        )

        df_prep = pipeline.prepare_data(
            df_hold.copy(),
            labels=True,
            side=side,
            set_market_condition=False,
            ensure_regime=True,
        )

        if min_rows is None:
            min_rows = int(model_config.seq_len_long)

        if len(df_prep) < min_rows:
            raise ValueError(
                f"Holdout insuficiente tras prepare_data(): {len(df_prep)} filas < min_rows={min_rows}"
            )

        state = pipeline.prepare_walkforward_state(
            df_prep,
            side=side,
            train=True,
        )

        keras_model = tf.keras.models.load_model(artifacts.model_path)
        calibrator = joblib.load(artifacts.calibrator_path)

        y_true_all = []
        y_pred_all = []
        pred_rows = []

        batch_seq_short = []
        batch_seq_long = []
        batch_context = []
        batch_time = []
        batch_meta = []

        def flush_batch():
            nonlocal batch_seq_short, batch_seq_long, batch_context, batch_time, batch_meta
            nonlocal y_true_all, y_pred_all, pred_rows

            if not batch_meta:
                return

            X_batch = [
                np.concatenate(batch_seq_short, axis=0),
                np.concatenate(batch_seq_long, axis=0),
                np.concatenate(batch_context, axis=0),
                np.concatenate(batch_time, axis=0),
            ]

            y_raw_batch = keras_model(X_batch, training=False)
            y_raw_batch = np.asarray(y_raw_batch).reshape(-1)

            if hasattr(calibrator, "predict"):
                y_cal_batch = calibrator.predict(y_raw_batch)
            else:
                y_cal_batch = calibrator.transform(y_raw_batch)

            for meta, y_raw, y_cal in zip(batch_meta, y_raw_batch, y_cal_batch):
                y_true_all.append(int(meta["y_true"]))
                y_pred_all.append(float(y_cal))

                if return_predictions:
                    pred_rows.append({
                        "time": meta["time"],
                        "state": meta["state"],
                        "y_true": int(meta["y_true"]),
                        "y_pred_raw": float(y_raw),
                        "y_pred_cal": float(y_cal),
                    })

            batch_seq_short = []
            batch_seq_long = []
            batch_context = []
            batch_time = []
            batch_meta = []

        for end_idx in range(min_rows, len(df_prep) + 1):
            data = pipeline.create_last_sample_from_state(
                state,
                end_idx=end_idx,
                fit_scalers=False,
                train=True,
            )

            labels_last = data.get("labels", None)
            if labels_last is None or len(labels_last) == 0:
                continue

            row_last = df_prep.iloc[end_idx - 1]
            meta = {
                "time": row_last["time"] if "time" in row_last else None,
                "state": row_last["state"] if "state" in row_last else None,
                "y_true": int(labels_last[0]),
            }

            batch_seq_short.append(data["seq_short"])
            batch_seq_long.append(data["seq_long"])
            batch_context.append(data["context"])
            batch_time.append(data["time"])
            batch_meta.append(meta)

            if len(batch_meta) >= inference_batch_size:
                flush_batch()

        flush_batch()

        if len(y_true_all) == 0:
            raise RuntimeError("No se generaron predicciones walk-forward válidas")

        y_true_arr = np.asarray(y_true_all, dtype=np.int32)
        y_pred_arr = np.asarray(y_pred_all, dtype=np.float32)

        mask = np.isfinite(y_true_arr) & np.isfinite(y_pred_arr)
        y_true_arr = y_true_arr[mask]
        y_pred_arr = y_pred_arr[mask]

        if len(y_true_arr) == 0:
            raise RuntimeError("Todas las predicciones walk-forward resultaron inválidas (NaN/Inf)")

        eval_dict = self.evaluator.evaluate_predictions(
            y_true=y_true_arr,
            y_pred_proba=y_pred_arr,
            verbose=True,
            beta_primary=0.25,
            min_precision=0.45,
            max_signal_rate=0.15,
        )

        try:
            eval_dict["auc_pr"] = float(average_precision_score(y_true_arr, y_pred_arr))
        except Exception:
            eval_dict["auc_pr"] = float("nan")

        try:
            eval_dict["auc_roc"] = float(roc_auc_score(y_true_arr, y_pred_arr))
        except Exception:
            eval_dict["auc_roc"] = float("nan")

        out = {
            "side": side,
            "mode": "walkforward_transform_then_update_fast",
            "n_samples": int(len(y_true_arr)),
            "metrics": eval_dict,
        }

        if return_predictions:
            out["predictions"] = pred_rows

        return out