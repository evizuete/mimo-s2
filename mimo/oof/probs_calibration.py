"""
probs_calibration.py
═══════════════════════════════════════════════════════════════════════════
Calibración isotónica de probabilidades OOF + cálculo de percentiles
por estado de mercado.

Cambios respecto a versión anterior:
  - Eliminado REGIME_5_TO_3 y map_to_3_regimes: los percentiles se calculan
    siempre sobre los estados completos (TREND_UP, TREND_DOWN, RANGE, etc.)
  - Los estados en NO_TRADE_STATES (LOW_VOL) se incluyen en el output
    marcados con 'no_trade': True para que el DecisionEngine los ignore.
  - Los percentiles globales (_global) se calculan EXCLUYENDO LOW_VOL.
"""

import ctypes
import gc
from typing import Dict, Any, Iterable, Tuple

import numpy as np
import pandas as pd
import tensorflow as tf
from pandas import DataFrame
from sklearn.isotonic import IsotonicRegression
from sklearn.model_selection import TimeSeriesSplit
from scipy.special import expit, logit

from mimo.data_managers.data_pipeline_v2 import DataPipeline
from mimo.models.model_builder import ModelConfig, TradingModel, Config
from mimo.models.scale_augmentation import augment_train_batch, ScaleAugmentConfig, _build_aug_skip_map

# Estados que no deben generar señales operativas
NO_TRADE_STATES = frozenset({"LOW_VOL"})


def make_fold_slices_from_samples(
    df: pd.DataFrame,
    L: int,
    train_samp_idx: np.ndarray,
    val_samp_idx: np.ndarray,
    *,
    require_full_context: bool = True,
):
    """
    Genera df_train_rows y df_val_rows a partir de índices de muestra (sample_idx),
    asumiendo que cada muestra i termina en la fila global:
        row_final = (L - 1) + i
    """
    if L <= 1:
        raise ValueError("L debe ser >= 2 para secuencias con contexto.")

    train_end = int(train_samp_idx[-1]) + 1
    val_start = int(val_samp_idx[0])
    val_end   = int(val_samp_idx[-1]) + 1
    val_count = val_end - val_start

    row_final_train_last = (L - 1) + (train_end - 1)
    row_final_val_last   = (L - 1) + (val_end - 1)
    row_final_val_first  = (L - 1) + val_start
    row_start_context    = row_final_val_first - (L - 1)

    if require_full_context and row_start_context < 0:
        raise RuntimeError(
            f"No hay contexto suficiente para val_start={val_start} con L={L}. "
            f"row_start_context={row_start_context}."
        )

    row_start_context = max(0, row_start_context)

    train_rows_end_excl = row_final_train_last + 1
    val_rows_start      = row_start_context
    val_rows_end_excl   = row_final_val_last + 1

    df_train_rows = df.iloc[:train_rows_end_excl].copy()
    df_val_rows   = df.iloc[val_rows_start:val_rows_end_excl].copy()

    debug = {
        "train_end_samples_excl": train_end,
        "val_start_sample": val_start,
        "val_end_samples_excl": val_end,
        "val_count": val_count,
        "row_final_train_last": row_final_train_last,
        "row_final_val_first": row_final_val_first,
        "row_final_val_last": row_final_val_last,
        "val_rows_start": val_rows_start,
        "val_rows_end_excl": val_rows_end_excl,
        "df_train_rows_shape": df_train_rows.shape,
        "df_val_rows_shape": df_val_rows.shape,
    }

    return df_train_rows, df_val_rows, debug


class ProbsCalibration:
    def __init__(self, general_config: Config, model_config: ModelConfig, temperature: float = 1.0):
        """
        Args:
            temperature: Factor de escala aplicado en espacio logit tras calibración isotónica.
                         > 1.0 → más dispersión (percentiles más separados)
                         < 1.0 → más conservador (comprime hacia 0.5)
                         = 1.0 → sin cambio (comportamiento estándar)
                         Rango recomendado: 1.0 – 2.0
        """
        self.general_config = general_config
        self.model_config = model_config
        self.temperature = float(temperature)

    @staticmethod
    def _apply_temperature(p: np.ndarray, temperature: float) -> np.ndarray:
        """
        Aplica temperature scaling en espacio logit:
            p_out = sigmoid(logit(p) * temperature)

        temperature > 1 → empuja probabilidades hacia los extremos (0 y 1)
        temperature < 1 → comprime probabilidades hacia 0.5
        temperature = 1 → identidad
        """
        if abs(temperature - 1.0) < 1e-6:
            return p
        p_clipped = np.clip(p, 1e-4, 1.0 - 1e-4)
        return expit(logit(p_clipped) * temperature).astype(np.float32)

    @staticmethod
    def _compute_sample_row_index(seq_len_long: int, n_rows: int) -> np.ndarray:
        context_offset = seq_len_long - 1
        n_samples = n_rows - seq_len_long + 1
        if n_samples <= 0:
            return np.array([], dtype=np.int64)
        return np.arange(context_offset, context_offset + n_samples, dtype=np.int64)

    def generate_oof_predictions(
            self,
            df_prepared: pd.DataFrame,
            pipeline: DataPipeline,
            side: str,
            n_splits: int = 5,
            epochs_per_fold: int = 25,
            verbose: int = 1,
    ) -> Tuple[DataFrame, IsotonicRegression, Dict[int, Any]]:
        """
        Genera predicciones OOF y calibración isotónica sin leakage.

        Devuelve df con columnas:
          - oof_proba_raw : predicción sigmoide sin calibrar (OOF)
          - oof_proba_cal : probabilidad calibrada (isotónica) entrenada SOLO con OOF
        """
        if "oof_proba_raw" in df_prepared.columns or "oof_proba_cal" in df_prepared.columns:
            df = df_prepared.reset_index(drop=True)
        else:
            df = df_prepared.copy().reset_index(drop=True)

        L = int(self.model_config.seq_len_long)
        row_index = self._compute_sample_row_index(L, len(df))
        if row_index.size == 0:
            raise ValueError("No hay suficientes filas para crear secuencias con seq_len_long.")

        n_samples = row_index.size
        oof_raw = np.full(len(df), np.nan, dtype=np.float32)
        oof_y   = np.full(len(df), -1,    dtype=np.int8)

        best_epochs: Dict[int, Any] = {}

        tscv = TimeSeriesSplit(n_splits=n_splits)

        for fold, (train_samp_idx, val_samp_idx) in enumerate(tscv.split(np.arange(n_samples))):
            train_end = int(train_samp_idx[-1]) + 1
            val_start = int(val_samp_idx[0])
            val_end   = int(val_samp_idx[-1]) + 1
            val_count = val_end - val_start

            df_train_rows, df_val_rows, debug = make_fold_slices_from_samples(
                df, L, train_samp_idx, val_samp_idx, require_full_context=True
            )

            if verbose:
                print(f"\n[OOF] Fold {fold + 1}/{n_splits}")
                print("[OOF] slice dbg:", debug)

            from copy import copy
            pipeline_fold = copy(pipeline)
            pipeline_fold.scalers = {}
            pipeline_fold.is_fitted = False

            if pipeline_fold.feature_config.feature_masks is not None:
                _side = "long" if side == "long" else "short"
                sequences = pipeline_fold.create_sequences_by_side(
                    df_train_rows, sides=(_side,), fit_scalers=True, train=True
                )
                data_train = sequences[_side]
                sequences_val = pipeline_fold.create_sequences_by_side(
                    df_val_rows, sides=(_side,), fit_scalers=False, train=True
                )
                data_val = sequences_val[_side]
            else:
                data_train = pipeline_fold.create_sequences(df_train_rows, fit_scalers=True,  train=True)
                data_val   = pipeline_fold.create_sequences(df_val_rows,   fit_scalers=False, train=True)

            X_train = {k: v for k, v in data_train.items() if k not in ["labels", "weights"]}
            y_train = data_train["labels"].astype(int)
            w_train = data_train["weights"]

            X_val = {k: v for k, v in data_val.items() if k not in ["labels", "weights"]}
            y_val = data_val["labels"].astype(int)

            # ── Entrenar modelo del fold ───────────────────────────────
            fold_model_config = ModelConfig(**vars(self.model_config))
            fold_model_config.epochs = int(epochs_per_fold)

            tm = TradingModel(self.general_config, fold_model_config, side=side)

            pos_rate = float(np.clip(np.nanmean(y_train), 1e-4, 1 - 1e-4))
            init_bias = float(np.log(pos_rate / (1 - pos_rate)))

            # Seleccionar arquitectura según flag del ModelConfig
            # use_hierarchical_fusion=True → v3 (fusión jerárquica market/entry)
            # use_hierarchical_fusion=False → v2 (fusión plana, default)
            _build_fn = (
                tm.build_model_v3
                if getattr(fold_model_config, 'use_hierarchical_fusion', False)
                else tm.build_model_v2
            )
            _build_fn(
                shape_short=(fold_model_config.seq_len_short,
                             len(pipeline_fold.feature_engineer.feature_columns["sequence_short"])),
                shape_long=(fold_model_config.seq_len_long,
                            len(pipeline_fold.feature_engineer.feature_columns["sequence_long"])),
                n_context=len(pipeline_fold.feature_engineer.feature_columns["context"]),
                n_time=len(pipeline_fold.feature_engineer.feature_columns["time"]),
                init_bias=init_bias,
            )
            tm.compile_model()

            aug_cfg = ScaleAugmentConfig(
                enabled=True,
                prob=0.60,
                mult_low=0.98,
                mult_high=1.02,
                noise_std=0.008,
                context_noise_std=0.004,
                seed=42 + fold,
            )

            skip_map = _build_aug_skip_map(pipeline_fold, side)
            X_train_aug = augment_train_batch(
                X_train,
                cfg=aug_cfg,
                skip_map=skip_map
            )

            history = tm.train(
                X_train_aug, y_train,
                X_val,   y_val,
                sample_weight=w_train,
                verbose=0 if verbose == 0 else 1,
                for_production=False,
            )

            vals = history["val_auc_pr"]
            best_epoch = int(np.argmax(vals)) + 1
            best_val   = float(np.max(vals))
            best_epochs[fold] = {"epoch": best_epoch, "val": best_val}

            if verbose:
                print(f"[OOF] Fold {fold + 1}: best epoch={best_epoch}, best val={best_val:.4f}")

            # ── Predicción OOF ─────────────────────────────────────────
            # Pasar batch_size explícito al modelo Keras para que compile el graph
            # con shapes dinámicas (None, seq_len, features) en lugar de la shape
            # concreta de cada fold. Sin esto cada fold produce un retracing porque
            # X_val tiene un tamaño distinto → "5 out of last 5 calls retracing".
            # TradingModel.predict() no expone batch_size, accedemos al modelo Keras
            # directamente si está disponible; si no, caemos al wrapper normal.
            _keras_model = getattr(tm, 'model', None)
            if _keras_model is not None and hasattr(_keras_model, 'predict'):
                _x_list = [X_val['seq_short'], X_val['seq_long'],
                           X_val['context'], X_val['time']]
                y_pred_val = _keras_model.predict(
                    _x_list, batch_size=4096, verbose=0
                ).astype(np.float32).reshape(-1)
            else:
                y_pred_val = tm.predict(X_val).astype(np.float32).reshape(-1)

            if len(y_pred_val) != val_count:
                raise RuntimeError(
                    f"Desalineación fold {fold + 1}: val_count={val_count} != len(pred)={len(y_pred_val)}"
                )

            val_row_positions = (L - 1) + np.arange(val_start, val_end, dtype=np.int32)
            oof_raw[val_row_positions] = y_pred_val
            oof_y[val_row_positions]   = y_val

            # Liberar memoria entre folds: el modelo + tensores del fold
            # anterior + buffers del scaler suman varios GB y disparaban
            # OOM (exit 137) en folds tardíos con datasets más grandes.
            del tm, history
            del data_train, data_val
            del X_train, X_val, X_train_aug
            del y_train, y_val, w_train, y_pred_val
            del pipeline_fold
            tf.keras.backend.clear_session()
            gc.collect()
            try:
                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass

        # ── Calibración isotónica ──────────────────────────────────────
        mask = np.isfinite(oof_raw) & np.isfinite(oof_y) & (oof_y >= 0)
        if mask.sum() < 1000:
            raise ValueError(f"No hay suficientes muestras OOF para calibrar. mask.sum()={mask.sum()}")

        y_true = oof_y[mask].astype(int)
        p_raw  = oof_raw[mask].astype(float)

        uniq = np.unique(y_true)
        if not set(uniq.tolist()).issubset({0, 1}):
            raise ValueError(f"Labels inválidos para calibración: unique={uniq}. Deben ser 0/1.")

        print(f"OOF mask.sum()={int(mask.sum())}  "
              f"Pos={int(y_true.sum())}  Neg={int(len(y_true) - y_true.sum())}")
        print(f"p_raw min/mean/max = {float(np.min(p_raw)):.4f} / "
              f"{float(np.mean(p_raw)):.4f} / {float(np.max(p_raw)):.4f}")

        # Clip antes de la isotónica para evitar que valores espurios cercanos a 0/1
        # (NaNs convertidos a 0.0 que pasan isfinite, o extremos del padding) causen
        # que la regresión isotónica extrapole a 0 o 1 en los bordes, lo que después
        # del temperature scaling produce p_cal_min=0.000 aunque p_raw_min>0.29.
        p_raw = np.clip(p_raw, 1e-4, 1.0 - 1e-4)

        # Entrenar la isotónica con TODOS los datos OOF (no subsampling).
        # El subsampling previo causaba que cal.predict() extrapolara a 0.0
        # para valores fuera del rango del subconjunto, produciendo p_cal_min=0.000
        # aunque p_raw_min > 0.30. IsotonicRegression es O(n log n) y con ~450k
        # muestras tarda < 1s, por lo que el subsampling no es necesario.
        cal = IsotonicRegression(out_of_bounds="clip")
        cal.fit(p_raw, y_true)

        p_cal_isotonic = cal.predict(p_raw).astype(np.float32)

        # Temperature scaling post-calibración
        if abs(self.temperature - 1.0) > 1e-6:
            p_cal_isotonic = self._apply_temperature(p_cal_isotonic, self.temperature)
            print(f"[Temperature] aplicado t={self.temperature:.3f} | "
                  f"p_cal min/mean/max = {float(p_cal_isotonic.min()):.4f} / "
                  f"{float(p_cal_isotonic.mean()):.4f} / {float(p_cal_isotonic.max()):.4f}")

        oof_cal = np.full(len(df), np.nan, dtype=np.float32)
        oof_cal[mask] = p_cal_isotonic

        df["oof_proba_raw"] = oof_raw
        df["oof_proba_cal"] = oof_cal

        return df, cal, best_epochs

    def compute_percentiles_by_regime(
            self,
            df_with_oof: pd.DataFrame,
            proba_col: str = "oof_proba_cal",
            regime_col: str = "state",
            quantiles: Iterable[int] = (50, 60, 70, 75, 80, 85, 90, 95, 96, 97, 98, 99),
            map_to_3_regimes: bool = False,   # mantenido por compatibilidad, ignorado
            min_n: int = 800,
    ) -> Dict[str, Any]:
        """
        Calcula percentiles por estado de mercado sobre probabilidades OOF calibradas.

        Siempre usa los estados completos (regime_col="state" por defecto).
        El parámetro map_to_3_regimes se mantiene por compatibilidad pero se ignora.

        Los estados en NO_TRADE_STATES (LOW_VOL) se incluyen marcados con
        'no_trade': True — el DecisionEngine debe ignorarlos para operar.

        Los percentiles globales (_global) se calculan EXCLUYENDO LOW_VOL
        para que no contaminen los umbrales operativos.
        """
        if map_to_3_regimes:
            print("[WARNING] compute_percentiles_by_regime: map_to_3_regimes=True ignorado. "
                  "Usando estados completos.")

        df = df_with_oof[np.isfinite(df_with_oof[proba_col])].copy()

        reg_col = regime_col
        all_states = sorted(df[reg_col].dropna().unique().tolist())

        quantiles = list(quantiles)

        # Percentiles globales excluyendo NO_TRADE_STATES
        df_trade = df[~df[reg_col].isin(NO_TRADE_STATES)]
        if len(df_trade) == 0:
            raise ValueError("No hay muestras fuera de NO_TRADE_STATES para calcular percentiles globales.")

        global_percentiles = np.percentile(df_trade[proba_col].values, quantiles, method="linear")

        out: Dict[str, Any] = {}

        for state in all_states:
            g = df[df[reg_col] == state]
            is_no_trade = state in NO_TRADE_STATES

            if len(g) >= min_n:
                pcts = np.percentile(g[proba_col].values, quantiles, method="linear")
            else:
                # Fallback a global (solo para estados operativos)
                # Para LOW_VOL, usar global igualmente (son informativos, no operativos)
                pcts = global_percentiles
                if verbose_fallback := (len(g) > 0):
                    print(f"  [percentiles] {state}: n={len(g)} < min_n={min_n} → usando global")

            entry = {
                "percentiles": {f"p{q}": float(v) for q, v in zip(quantiles, pcts)},
                "n": int(len(g)),
            }
            if is_no_trade:
                entry["no_trade"] = True

            out[str(state)] = entry

        out["_global"] = {
            "percentiles": {f"p{q}": float(v) for q, v in zip(quantiles, global_percentiles)},
            "n": int(len(df_trade)),
            "note": "Excludes NO_TRADE_STATES",
        }

        out["_meta"] = {
            "proba_col":       proba_col,
            "regime_col":      reg_col,
            "quantiles":       quantiles,
            "no_trade_states": list(NO_TRADE_STATES),
            "states_found":    all_states,
        }

        return out