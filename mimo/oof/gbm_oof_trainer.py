"""
gbm_oof_trainer.py
═══════════════════════════════════════════════════════════════════════════
Drop-in alternative al CNN-LSTM trainer (optuna_oof_trainer_v2.py) usando
LightGBM. Reusa la pipeline de features y labels al 100%, solo cambia el
modelo que se entrena por fold.

DESIGN PRINCIPLES:
  - 100% aislado de optuna_oof_trainer_v2 / probs_calibration / model_builder.
  - Reusa todo lo MODEL-AGNOSTIC del v7: BARRIERS_BY_RELEASE, _tf_defaults
    (para horizon labels), FeatureConfig, DataPipeline, regime weights,
    feature_masks.
  - Output compatible con extract_best_per_side: produce un Optuna study
    en MySQL con la misma estructura de user_attrs (ev_long, ev_short,
    score) → fase 2+ del pipeline funcionan sin modificación.
  - Sin secuencias: el GBM consume features tabulares en cada timestamp.
    El feature engineering ya agrega contexto multi-timescale (1h, 15m, etc.)
    así que la red no añadía valor para este problema (GBM sanity check
    confirmó AUC=0.66 con esta tabularización).

ARTIFACTS PERSISTIDOS POR TRIAL (cache en _optuna_cache/):
  - oof_df.parquet              ← oof_proba_raw, oof_proba_cal, labels
  - calibrator.joblib           ← IsotonicRegression
  - best_model.joblib           ← lgb.Booster del fold final
  - percentiles.json            ← percentiles por estado
  - oof_meta.json               ← n_signals, threshold óptimo, etc.
"""

from __future__ import annotations

import gc
import hashlib
import json
import os
import time
from copy import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import joblib
import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score
from sklearn.model_selection import TimeSeriesSplit

from mimo.data_managers.data_pipeline_v2 import DataPipeline
from mimo.models.model_builder import Config
from mimo.oof.ev_objective import compute_balanced_objective


# ─── utilidades comunes ──────────────────────────────────────────────────

def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _params_signature(params: Dict[str, Any]) -> str:
    """Hash corto de un dict de params para cache keying."""
    blob = json.dumps(params, sort_keys=True, default=str).encode()
    return hashlib.md5(blob).hexdigest()[:8]


# ─── Trainer ────────────────────────────────────────────────────────────

@dataclass
class GBMTrainerArtifacts:
    """Paths a los artifacts de un trial completado."""
    oof_df: str
    calibrator: str
    booster: str
    percentiles: str
    meta: str


class GBMOOFTrainer:
    """Optuna + OOF + LightGBM, paralelo al CNN trainer pero radicalmente más
    simple: sin secuencias, sin TF, sin focal loss compleja.

    El trainer asume target_type='multitask': entrena DOS boosters por trial
    (uno por cabeza: long y short) compartiendo features y splits OOF.
    """

    def __init__(
        self,
        *,
        general_config: Config,
        feature_config,                # FeatureConfig (mismo que CNN, con label_method='triple_barrier_dual')
        regime_config,                 # StateConfig (mismo que CNN)
        out_dir: str,
        optuna_db: str,
        study_prefix: str = "oof_study_gbm",
        seed: int = 47,
        grid_space: Optional[Dict[str, Any]] = None,
        # Hiperparámetros del objetivo EV-net (mismos defaults que CNN)
        cost_per_signal: float = 0.05,
        ev_min_signals: int = 100,
        max_drawdown_R: float = 30.0,
        ev_thr_lo: float = 0.10,
        ev_thr_hi: float = 0.40,
        ev_n_thr: int = 60,
        exclude_feature_patterns: Optional[List[str]] = None,
    ):
        self.general_config = general_config
        self.feature_config = feature_config
        self.regime_config = regime_config
        self.out_dir = out_dir
        self.optuna_db = optuna_db
        self.study_prefix = study_prefix
        self.seed = int(seed)
        self.grid_space = grid_space
        self.cost_per_signal = float(cost_per_signal)
        self.ev_min_signals = int(ev_min_signals)
        self.exclude_feature_patterns = list(exclude_feature_patterns or [])
        self.max_drawdown_R = float(max_drawdown_R)
        self.ev_thr_lo = float(ev_thr_lo)
        self.ev_thr_hi = float(ev_thr_hi)
        self.ev_n_thr = int(ev_n_thr)
        _ensure_dir(self.out_dir)

    # -------- helpers de sugerencia (mismo patrón que CNN trainer) -----

    def _choices(self, name: str):
        if not self.grid_space or name not in self.grid_space:
            raise RuntimeError(f"No grid space for {name}")
        return self.grid_space[name]

    def _suggest(self, trial: optuna.Trial, name: str):
        spec = self._choices(name)
        if isinstance(spec, dict):
            low = spec["low"]
            high = spec["high"]
            log = bool(spec.get("log", False))
            step = spec.get("step", None)
            is_int_range = (
                isinstance(low, int) and isinstance(high, int)
                and not log and (step is None or isinstance(step, int))
            )
            if is_int_range:
                return trial.suggest_int(name, int(low), int(high), step=int(step) if step else 1)
            return trial.suggest_float(name, float(low), float(high), step=step, log=log)
        return trial.suggest_categorical(name, spec)

    def _suggest_lgbm_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Construye el dict de params para LightGBM.

        Hiperparámetros core de LGBM expuestos al tuning:
          - num_leaves          (capacidad)
          - learning_rate       (LR del boosting)
          - min_data_in_leaf    (regularización por hoja)
          - feature_fraction    (col subsampling)
          - bagging_fraction    (row subsampling)
          - lambda_l1, lambda_l2 (regularización L1/L2)
          - max_depth           (-1 sin límite por defecto)
          - n_estimators        (boosting rounds, con early stopping)
        """
        params = {
            "objective": "binary",
            "metric": "average_precision",
            "verbose": -1,
            "seed": self.seed,
            "n_jobs": -1,
            "boosting_type": "gbdt",
            # Tuneables
            "num_leaves":         self._suggest(trial, "num_leaves"),
            "learning_rate":      self._suggest(trial, "learning_rate"),
            "min_data_in_leaf":   self._suggest(trial, "min_data_in_leaf"),
            "feature_fraction":   self._suggest(trial, "feature_fraction"),
            "bagging_fraction":   self._suggest(trial, "bagging_fraction"),
            "bagging_freq":       self._suggest(trial, "bagging_freq"),
            "lambda_l1":          self._suggest(trial, "lambda_l1"),
            "lambda_l2":          self._suggest(trial, "lambda_l2"),
            "max_depth":          self._suggest(trial, "max_depth"),
        }
        # Estos van fuera del dict de params LGBM (los usa el bucle de train)
        meta = {
            "n_estimators":       self._suggest(trial, "n_estimators"),
            "early_stopping":     self._suggest(trial, "early_stopping_rounds"),
        }
        return {"lgbm": params, "meta": meta}

    # -------- pipeline / features tabulares ----------------------------

    def _build_pipeline(self) -> DataPipeline:
        """Pipeline mínimo para que prepare_data funcione. seq_len_* da igual
        porque no se usan secuencias en GBM, pero el pipeline necesita un
        ModelConfig válido para inicializar."""
        from mimo.models.model_builder import ModelConfig
        mc = ModelConfig(
            seq_len_short=64,   # cualquier valor; no se usa
            seq_len_long=256,
            target_type="multitask",
        )
        return DataPipeline(
            general_config=self.general_config,
            feature_config=self.feature_config,
            model_config=mc,
            regime_config=self.regime_config,
        )

    def _collect_tabular_columns(self, pipeline: DataPipeline) -> List[str]:
        """Unión plana de columnas (sequence_short + sequence_long + context
        + time) — las usamos como features tabulares para LGBM."""
        all_cols = pipeline._get_all_feature_columns()
        flat: List[str] = []
        for key in ("sequence_short", "sequence_long", "context", "time"):
            for c in all_cols.get(key, []):
                if c not in flat:
                    flat.append(c)
        return flat

    # -------- objetivo Optuna ------------------------------------------

    def _trial_cache_paths(self, *, study_name: str, trial_number: int) -> Dict[str, str]:
        base = os.path.join(self.out_dir, "_optuna_cache_gbm", study_name, f"trial_{trial_number:05d}")
        _ensure_dir(base)
        return {
            "dir": base,
            "oof_df":      os.path.join(base, "oof_df.parquet"),
            "cal_long":    os.path.join(base, "calibrator_long.joblib"),
            "cal_short":   os.path.join(base, "calibrator_short.joblib"),
            "booster_long":  os.path.join(base, "best_booster_long.joblib"),
            "booster_short": os.path.join(base, "best_booster_short.joblib"),
            "percentiles": os.path.join(base, "percentiles.json"),
            "meta":        os.path.join(base, "oof_meta.json"),
        }

    def _train_lgbm_oof(
        self,
        *,
        X: np.ndarray,
        y: np.ndarray,
        w: np.ndarray,
        lgbm_params: Dict[str, Any],
        meta_params: Dict[str, Any],
        n_splits: int,
        side_label: str,
        fold_callback: Optional[Callable[[int, float, int], None]] = None,
    ) -> Tuple[np.ndarray, lgb.Booster, IsotonicRegression]:
        """Entrena LGBM en TimeSeriesSplit y devuelve (oof_raw, last_booster, calibrator).

        Devuelve siempre el booster del último fold como "best" (sirve para
        inferencia rápida sobre holdout; en deploy_full se reentrena con todo).
        """
        oof_raw = np.full(len(y), np.nan, dtype=np.float32)
        tscv = TimeSeriesSplit(n_splits=n_splits)
        last_booster: Optional[lgb.Booster] = None
        n_estimators = int(meta_params["n_estimators"])
        early_stop = int(meta_params["early_stopping"])

        for fold, (tr_idx, va_idx) in enumerate(tscv.split(X)):
            X_tr, y_tr, w_tr = X[tr_idx], y[tr_idx], w[tr_idx]
            X_va, y_va = X[va_idx], y[va_idx]

            if y_tr.sum() < 50 or y_va.sum() < 20:
                # Fold inviable, no entrenamos pero rellenamos con base rate
                oof_raw[va_idx] = float(y_tr.mean()) if y_tr.size else 0.0
                continue

            # scale_pos_weight adicional sobre y_tr (sin tocar w_tr, que tiene
            # regime weights). Solo balanceo binario aproximado.
            n_pos = int(y_tr.sum()); n_neg = len(y_tr) - n_pos
            params = dict(lgbm_params)
            params["scale_pos_weight"] = float(n_neg / max(n_pos, 1))

            train_ds = lgb.Dataset(X_tr, label=y_tr, weight=w_tr)
            valid_ds = lgb.Dataset(X_va, label=y_va, reference=train_ds)

            booster = lgb.train(
                params,
                train_ds,
                num_boost_round=n_estimators,
                valid_sets=[valid_ds],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=early_stop, verbose=False),
                ],
            )
            p_va = booster.predict(X_va, num_iteration=booster.best_iteration)
            oof_raw[va_idx] = p_va.astype(np.float32)
            last_booster = booster

            if fold_callback is not None:
                mask = np.isfinite(oof_raw)
                if mask.sum() >= 200 and len(np.unique(y[mask])) >= 2:
                    partial = float(average_precision_score(y[mask], oof_raw[mask]))
                    fold_callback(fold, partial, n_splits)

            # liberar memoria
            del train_ds, valid_ds
            gc.collect()

        # Calibración isotónica sobre OOF
        mask = np.isfinite(oof_raw)
        cal = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
        if mask.sum() < 1000 or len(np.unique(y[mask])) < 2:
            # No hay datos suficientes para calibrar; identidad
            cal.fit([0.0, 1.0], [0.0, 1.0])
        else:
            cal.fit(oof_raw[mask].astype(np.float64), y[mask].astype(np.float64))

        return oof_raw, last_booster, cal

    def _objective(
        self,
        trial: optuna.Trial,
        *,
        df_rates: pd.DataFrame,
        n_splits: int,
    ) -> float:
        # 1) Sugerir params
        suggestion = self._suggest_lgbm_params(trial)
        lgbm_params = suggestion["lgbm"]
        meta_params = suggestion["meta"]
        print(f"[Trial {trial.number}] lgbm_params={lgbm_params}")
        print(f"[Trial {trial.number}] meta_params={meta_params}")

        # 2) Pipeline + datos preparados (multitask: signal_long, signal_short)
        pipeline = self._build_pipeline()
        df_prepared = pipeline.prepare_data(
            df_rates, labels=True, side="both",
            set_market_condition=False, ensure_regime=True,
        )
        if "signal_long" not in df_prepared.columns or "signal_short" not in df_prepared.columns:
            raise RuntimeError(
                "El pipeline no produjo signal_long/signal_short — verifica que "
                "feature_config.label_method='triple_barrier_dual' (multitask)."
            )

        # 3) Feature matrix tabular
        feat_cols = self._collect_tabular_columns(pipeline)
        feat_cols = [c for c in feat_cols if c in df_prepared.columns]
        # Exclude por patrón (ej. release 202605 quita features time-of-day)
        if self.exclude_feature_patterns:
            n_before = len(feat_cols)
            feat_cols = [c for c in feat_cols
                         if not any(p in c for p in self.exclude_feature_patterns)]
            n_excluded = n_before - len(feat_cols)
            if n_excluded > 0:
                print(f"[Trial {trial.number}] EXCLUDED {n_excluded} features por patterns "
                      f"{self.exclude_feature_patterns}")
        if not feat_cols:
            raise RuntimeError("No hay feature columns disponibles para GBM")

        # Necesitamos también OHLCV+atr para que compute_balanced_objective
        # pueda simular triple-barrier en el threshold sweep.
        needed_ohlc = ["time", "high", "low", "close", "atr"]
        needed_labels = ["signal_long", "signal_short"]
        keep = df_prepared[feat_cols + needed_ohlc + needed_labels].notna().all(axis=1)
        df_clean = df_prepared.loc[keep].reset_index(drop=True)
        print(f"[Trial {trial.number}] {len(df_clean):,} rows × {len(feat_cols)} features")

        X = df_clean[feat_cols].astype(np.float32).values
        y_long = df_clean["signal_long"].astype(np.int8).values
        y_short = df_clean["signal_short"].astype(np.int8).values
        w = df_clean["regime_weight"].astype(np.float32).values \
            if "regime_weight" in df_clean.columns else np.ones(len(df_clean), dtype=np.float32)

        # 4) OOF por lado con reporting per-fold sobre LONG (para Hyperband)
        def _fold_cb_long(fold_i: int, partial: float, total: int):
            trial.report(partial, step=fold_i)
            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        print(f"[Trial {trial.number}] entrenando LONG...")
        oof_long_raw, booster_long, cal_long = self._train_lgbm_oof(
            X=X, y=y_long, w=w,
            lgbm_params=lgbm_params, meta_params=meta_params,
            n_splits=n_splits, side_label="long",
            fold_callback=_fold_cb_long,
        )
        print(f"[Trial {trial.number}] entrenando SHORT...")
        oof_short_raw, booster_short, cal_short = self._train_lgbm_oof(
            X=X, y=y_short, w=w,
            lgbm_params=lgbm_params, meta_params=meta_params,
            n_splits=n_splits, side_label="short",
            fold_callback=None,  # reporting solo en long; short es el segundo entreno
        )

        # 5) Calibrar y persistir
        oof_long_cal = cal_long.predict(np.nan_to_num(oof_long_raw, nan=0.0)).astype(np.float32)
        oof_short_cal = cal_short.predict(np.nan_to_num(oof_short_raw, nan=0.0)).astype(np.float32)

        # df_oof: nombres de columnas COMPATIBLES con el CNN trainer
        # (oof_proba_long_cal / oof_proba_short_cal) para que extract_best_per_side
        # y compute_balanced_objective consuman idéntico.
        df_oof = pd.DataFrame({
            "time":  df_clean["time"].values,
            "high":  df_clean["high"].astype(np.float32).values,
            "low":   df_clean["low"].astype(np.float32).values,
            "close": df_clean["close"].astype(np.float32).values,
            "atr":   df_clean["atr"].astype(np.float32).values,
            "state": df_clean["state"].values if "state" in df_clean.columns else "UNKNOWN",
            "signal_long":  y_long,
            "signal_short": y_short,
            "oof_proba_long_raw":  oof_long_raw,
            "oof_proba_long_cal":  oof_long_cal,
            "oof_proba_short_raw": oof_short_raw,
            "oof_proba_short_cal": oof_short_cal,
        })

        paths = self._trial_cache_paths(study_name=trial.study.study_name,
                                        trial_number=trial.number)
        df_oof.to_parquet(paths["oof_df"], index=False)
        joblib.dump(cal_long,  paths["cal_long"])
        joblib.dump(cal_short, paths["cal_short"])
        if booster_long  is not None: joblib.dump(booster_long,  paths["booster_long"])
        if booster_short is not None: joblib.dump(booster_short, paths["booster_short"])

        trial.set_user_attr("oof_df_path",        paths["oof_df"])
        trial.set_user_attr("cal_long_path",      paths["cal_long"])
        trial.set_user_attr("cal_short_path",     paths["cal_short"])
        trial.set_user_attr("booster_long_path",  paths["booster_long"])
        trial.set_user_attr("booster_short_path", paths["booster_short"])

        # 6) EV-net balanceado long+short (mismo que CNN trainer)
        tp_mult = float(self.feature_config.tp_barrier)
        sl_mult = float(self.feature_config.sl_barrier)
        horizon = int(self.feature_config.label_horizon)
        ev_res = compute_balanced_objective(
            df_oof,
            long_proba_col="oof_proba_long_cal",
            short_proba_col="oof_proba_short_cal",
            horizon=horizon,
            tp_mult=tp_mult, sl_mult=sl_mult,
            cost_per_signal=self.cost_per_signal,
            n_thr=self.ev_n_thr,
            thr_lo=self.ev_thr_lo, thr_hi=self.ev_thr_hi,
            min_signals=self.ev_min_signals,
            max_drawdown_R=self.max_drawdown_R,
        )
        score = float(ev_res.get("score", -1.0))
        if not np.isfinite(score):
            score = -1.0
        long_d = ev_res.get("long", {})
        short_d = ev_res.get("short", {})

        trial.set_user_attr("ev_score", score)
        trial.set_user_attr("ev_long",  _json_safe(long_d))
        trial.set_user_attr("ev_short", _json_safe(short_d))

        # Meta para extract_best_per_side
        meta = {
            "score": score,
            "ev_long":  _json_safe(long_d),
            "ev_short": _json_safe(short_d),
            "lgbm_params": lgbm_params,
            "meta_params": meta_params,
            "n_features": len(feat_cols),
            "n_rows": len(df_clean),
            "horizon": horizon,
            "tp_mult": tp_mult,
            "sl_mult": sl_mult,
        }
        with open(paths["meta"], "w") as fh:
            json.dump(meta, fh, indent=2, default=str)

        print(f"[Trial {trial.number}] DONE | score={score:+.4f}  "
              f"LONG  thr={long_d.get('thr', float('nan')):.3f} "
              f"ev_net={long_d.get('ev_net', float('nan')):+.4f}R "
              f"sig={long_d.get('n_signals', 0)} "
              f"mdd={long_d.get('mdd_R', float('nan')):.1f}R | "
              f"SHORT thr={short_d.get('thr', float('nan')):.3f} "
              f"ev_net={short_d.get('ev_net', float('nan')):+.4f}R "
              f"sig={short_d.get('n_signals', 0)} "
              f"mdd={short_d.get('mdd_R', float('nan')):.1f}R")

        return score

    # -------- API pública ----------------------------------------------

    def optimize(
        self,
        *,
        df_rates: pd.DataFrame,
        n_trials: int,
        n_splits: int = 5,
        load_if_exists: bool = True,
    ) -> optuna.Study:
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        pruner = optuna.pruners.HyperbandPruner(
            min_resource=1, max_resource=n_splits, reduction_factor=3,
        )
        study_name = f"{self.study_prefix}_{self.general_config.release}_multitask"
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=self.optuna_db,
            load_if_exists=load_if_exists,
            sampler=sampler, pruner=pruner,
        )
        print(f"🚀 GBM Optuna study: {study_name}  |  trials={n_trials}")

        def obj(trial: optuna.Trial) -> float:
            return self._objective(trial, df_rates=df_rates, n_splits=n_splits)

        t0 = time.time()
        study.optimize(obj, n_trials=n_trials, gc_after_trial=True)
        print(f"✅ Optuna done in {time.time() - t0:.1f}s — {len(study.trials)} trials")
        return study


def _json_safe(d: Dict[str, Any]) -> Dict[str, Any]:
    """Convierte dict con numpy floats / NaN / Inf en JSON-serializable."""
    out = {}
    for k, v in d.items():
        if isinstance(v, (np.floating, float)):
            fv = float(v)
            if np.isnan(fv): out[k] = "nan"
            elif np.isinf(fv): out[k] = "inf" if fv > 0 else "-inf"
            else: out[k] = fv
        elif isinstance(v, (np.integer, int)):
            out[k] = int(v)
        elif isinstance(v, np.ndarray):
            out[k] = v.tolist()
        else:
            out[k] = v
    return out
