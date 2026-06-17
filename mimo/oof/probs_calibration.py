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

        # ── Detección de modo (quantile / triple_class / multitask / binary) ──
        target_type = getattr(self.model_config, 'target_type', 'binary')
        is_quantile = (target_type == 'quantile')
        is_triple_class = (target_type == 'triple_class')
        is_multitask = (target_type == 'multitask')
        if is_quantile:
            quantile_levels = tuple(self.model_config.quantile_levels)
            n_q = len(quantile_levels)
            oof_raw = np.full((len(df), n_q), np.nan, dtype=np.float32)
            oof_y = np.full(len(df), np.nan, dtype=np.float32)
        elif is_multitask:
            # Dos columnas: [P_long_oof, P_short_oof] y [is_long_TP, is_short_TP].
            oof_raw = np.full((len(df), 2), np.nan, dtype=np.float32)
            oof_y = np.full((len(df), 2), -1, dtype=np.int8)
        else:
            # binary y triple_class comparten storage: oof_raw=P(TP), oof_y=is_TP.
            oof_raw = np.full(len(df), np.nan, dtype=np.float32)
            oof_y = np.full(len(df), -1, dtype=np.int8)

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
                # Multitask usa 'both' como vista canónica para el pipeline:
                # feature_engineer con side='both' devuelve la UNIÓN de
                # columnas long+short, así el trunk multitask ve features
                # direccionales de ambos lados a la vez.
                _side = side if side in ("long", "short") else "both"
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
            X_val = {k: v for k, v in data_val.items() if k not in ["labels", "weights"]}
            w_train = data_train["weights"]

            if is_quantile:
                # Target continuo (forward return / atr). NaN posibles en las
                # últimas h filas; el dropna previo en generate_all_features +
                # el filtro de mask_oof se encargan downstream.
                y_train = data_train["labels"].astype(np.float32)
                y_val = data_val["labels"].astype(np.float32)
            elif is_multitask:
                # Labels shape (N, 2) con [is_long_TP, is_short_TP] desde
                # data_pipeline._extract_labels_weights. Se pasan tal cual a
                # tm.train, que internamente los splittea a dict para Keras.
                y_train = data_train["labels"].astype(np.float32)
                y_val = data_val["labels"].astype(np.float32)
            else:
                # binary: labels son {0, 1}.
                # triple_class: labels son {0=SL, 1=TIMEOUT, 2=TP}. Se pasan
                # tal cual a SparseCategoricalCrossentropy en TradingModel.
                y_train = data_train["labels"].astype(int)
                y_val = data_val["labels"].astype(int)

            # ── Entrenar modelo del fold ───────────────────────────────
            fold_model_config = ModelConfig(**vars(self.model_config))
            fold_model_config.epochs = int(epochs_per_fold)

            tm = TradingModel(self.general_config, fold_model_config, side=side)

            if is_quantile:
                # En quantile mode no hay prior-bias informativo (la cabeza es
                # lineal y debe aprender la mediana del return desde los datos).
                init_bias = 0.0
            elif is_triple_class:
                # Cabeza softmax(3); el bias en la última Dense es zero por
                # diseño. El init_bias del flujo binario no aplica.
                init_bias = 0.0
            elif is_multitask:
                # Dos cabezas binarias → init_bias por lado vía dict.
                pr_long = float(np.clip(np.nanmean(y_train[:, 0]), 1e-4, 1 - 1e-4))
                pr_short = float(np.clip(np.nanmean(y_train[:, 1]), 1e-4, 1 - 1e-4))
                init_bias = {
                    'long': float(np.log(pr_long / (1 - pr_long))),
                    'short': float(np.log(pr_short / (1 - pr_short))),
                }
            else:
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

            # Para multitask el aug_skip_map se construye con la vista
            # 'both' (UNIÓN) — coherente con la unión de features que ve
            # el modelo en multitask.
            _aug_side = side if side in ("long", "short") else "both"
            skip_map = _build_aug_skip_map(pipeline_fold, _aug_side)
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

            if is_quantile:
                # En quantile mode la métrica que se reporta es val_loss
                # (pinball, a minimizar). Guardamos su negativo en 'val' para
                # mantener la convención "más alto = mejor" del best_epochs dict.
                vals_loss = history["val_loss"]
                best_epoch = int(np.argmin(vals_loss)) + 1
                best_val = -float(np.min(vals_loss))
            elif is_multitask:
                # Métricas Keras multi-output: '<output>_<metric>'.
                # Tomamos la media de val_auc_pr_long y val_auc_pr_short por época.
                vals_long = np.asarray(history.get("val_signal_long_auc_pr", []))
                vals_short = np.asarray(history.get("val_signal_short_auc_pr", []))
                if len(vals_long) == 0 or len(vals_short) == 0:
                    raise RuntimeError(
                        f"multitask fold {fold + 1}: no se encontraron las métricas "
                        f"val_signal_long_auc_pr / val_signal_short_auc_pr en history. "
                        f"Claves disponibles: {list(history.keys())}"
                    )
                vals_mean = 0.5 * (vals_long + vals_short)
                best_epoch = int(np.argmax(vals_mean)) + 1
                best_val = float(np.max(vals_mean))
            else:
                # binary → val_auc_pr; triple_class → val_auc_pr_tp (P(TP) vs is_TP).
                auc_key = "val_auc_pr_tp" if is_triple_class else "val_auc_pr"
                vals = history[auc_key]
                best_epoch = int(np.argmax(vals)) + 1
                best_val = float(np.max(vals))
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
                _raw = _keras_model.predict(_x_list, batch_size=4096, verbose=0)
                # Multitask: predict devuelve list/dict (named outputs).
                if is_multitask:
                    if isinstance(_raw, dict):
                        p_long = np.asarray(_raw['signal_long']).reshape(-1)
                        p_short = np.asarray(_raw['signal_short']).reshape(-1)
                    elif isinstance(_raw, (list, tuple)):
                        p_long = np.asarray(_raw[0]).reshape(-1)
                        p_short = np.asarray(_raw[1]).reshape(-1)
                    else:
                        raise RuntimeError(
                            f"multitask predict shape inesperado: "
                            f"type={type(_raw)}"
                        )
                    y_pred_val = np.stack([p_long, p_short], axis=-1).astype(np.float32)
                else:
                    y_pred_val = np.asarray(_raw).astype(np.float32)
            else:
                y_pred_val = tm.predict(X_val).astype(np.float32)

            if is_quantile:
                # Shape esperado: (val_count, n_q). Mantener 2D.
                if y_pred_val.ndim == 1:
                    y_pred_val = y_pred_val.reshape(-1, 1)
                if y_pred_val.shape[0] != val_count:
                    raise RuntimeError(
                        f"Desalineación fold {fold + 1}: val_count={val_count} "
                        f"!= pred.shape[0]={y_pred_val.shape[0]}"
                    )
            elif is_multitask:
                if y_pred_val.ndim != 2 or y_pred_val.shape[-1] != 2:
                    raise RuntimeError(
                        f"multitask fold {fold + 1}: pred shape {y_pred_val.shape} "
                        f"inesperado (se esperaba (N, 2))."
                    )
                if y_pred_val.shape[0] != val_count:
                    raise RuntimeError(
                        f"Desalineación fold {fold + 1}: val_count={val_count} "
                        f"!= pred.shape[0]={y_pred_val.shape[0]}"
                    )
            elif is_triple_class:
                # Shape esperado: (val_count, 3) softmax. Extraemos P(TP)=col 2.
                if y_pred_val.ndim != 2 or y_pred_val.shape[-1] != 3:
                    raise RuntimeError(
                        f"triple_class fold {fold + 1}: pred shape {y_pred_val.shape} "
                        f"inesperado (se esperaba (N, 3))."
                    )
                y_pred_val = y_pred_val[:, 2].astype(np.float32)
                if len(y_pred_val) != val_count:
                    raise RuntimeError(
                        f"Desalineación fold {fold + 1}: val_count={val_count} != len(pred)={len(y_pred_val)}"
                    )
            else:
                y_pred_val = y_pred_val.reshape(-1)
                if len(y_pred_val) != val_count:
                    raise RuntimeError(
                        f"Desalineación fold {fold + 1}: val_count={val_count} != len(pred)={len(y_pred_val)}"
                    )

            val_row_positions = (L - 1) + np.arange(val_start, val_end, dtype=np.int32)
            oof_raw[val_row_positions] = y_pred_val
            if is_triple_class:
                # Binarizamos label para el calibrador y métricas binarias
                # downstream: 1=TP (clase 2), 0=no-TP (clases 0 y 1).
                oof_y[val_row_positions] = (np.asarray(y_val) == 2).astype(np.int8)
            elif is_multitask:
                # y_val shape (N, 2) ya. Guardamos como int.
                oof_y[val_row_positions] = np.asarray(y_val).astype(np.int8)
            else:
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

        if is_quantile:
            # ── Calibración conformal por cuantil ──────────────────────────
            # Para cada cuantil predicho q, calculamos el residuo (y - q_pred)
            # y aplicamos un shift constante igual al q-cuantil de los residuos
            # para que la cobertura empírica iguale q.
            mask = np.isfinite(oof_raw).all(axis=1) & np.isfinite(oof_y)
            if mask.sum() < 1000:
                raise ValueError(
                    f"No hay suficientes muestras OOF para calibrar. mask.sum()={mask.sum()}"
                )

            y_true = oof_y[mask].astype(np.float32)
            q_raw = oof_raw[mask].astype(np.float32)  # (M, n_q)
            quantile_levels = list(self.model_config.quantile_levels)

            print(f"OOF mask.sum()={int(mask.sum())}")
            print(f"y_true (returns/atr) min/mean/median/std/max = "
                  f"{float(y_true.min()):.4f} / {float(y_true.mean()):.4f} / "
                  f"{float(np.median(y_true)):.4f} / {float(y_true.std()):.4f} / "
                  f"{float(y_true.max()):.4f}")

            shifts = []
            q_cal = q_raw.copy()
            for i, q in enumerate(quantile_levels):
                residuals = y_true - q_raw[:, i]
                shift = float(np.quantile(residuals, q))
                q_cal[:, i] = q_raw[:, i] + shift
                shifts.append(shift)
                pre_cov = float(np.mean(y_true <= q_raw[:, i]))
                post_cov = float(np.mean(y_true <= q_cal[:, i]))
                print(f"[Conformal] q={q:.2f} shift={shift:+.4f} "
                      f"coverage pre={pre_cov:.3f} post={post_cov:.3f}")

            # Calibrator es un dict con shifts (no un IsotonicRegression).
            cal = {
                "type": "conformal_quantile_shifts",
                "quantiles": quantile_levels,
                "shifts": shifts,
            }

            # Persistir TODAS las columnas (raw y calibradas) en el df.
            # Para compat downstream, oof_proba_raw/cal apuntan al cuantil más
            # bajo (q25 por defecto): es la salida de decisión natural — opera
            # cuando el q25 supera el umbral económico/percentil.
            oof_raw_all = np.full((len(df), len(quantile_levels)), np.nan, dtype=np.float32)
            oof_cal_all = np.full((len(df), len(quantile_levels)), np.nan, dtype=np.float32)
            oof_raw_all[mask] = q_raw
            oof_cal_all[mask] = q_cal

            for i, q in enumerate(quantile_levels):
                qi = int(round(q * 100))
                df[f"oof_q{qi}_raw"] = oof_raw_all[:, i]
                df[f"oof_q{qi}_cal"] = oof_cal_all[:, i]

            # Aliases para que el resto del pipeline (compute_percentiles_by_regime,
            # holdout eval, decision engine) siga funcionando sin cambios. La
            # decisión se toma sobre el cuantil inferior (q25) calibrado.
            df["oof_proba_raw"] = oof_raw_all[:, 0]
            df["oof_proba_cal"] = oof_cal_all[:, 0]

            return df, cal, best_epochs

        if is_multitask:
            # ── Calibración isotónica DUAL (multitask) ─────────────────────
            # oof_raw shape (N, 2) [P_long, P_short], oof_y shape (N, 2).
            # Entrenamos UN calibrador isotónico por lado y los devolvemos
            # como dict {'long': cal_long, 'short': cal_short}.
            mask = (
                np.isfinite(oof_raw).all(axis=1)
                & (oof_y[:, 0] >= 0) & (oof_y[:, 1] >= 0)
            )
            if mask.sum() < 1000:
                raise ValueError(
                    f"multitask: muestras OOF insuficientes mask.sum()={int(mask.sum())}"
                )

            print(f"OOF mask.sum()={int(mask.sum())} (multitask)")
            print(f"  P_long  min/mean/max = "
                  f"{float(oof_raw[mask, 0].min()):.4f} / "
                  f"{float(oof_raw[mask, 0].mean()):.4f} / "
                  f"{float(oof_raw[mask, 0].max()):.4f}")
            print(f"  P_short min/mean/max = "
                  f"{float(oof_raw[mask, 1].min()):.4f} / "
                  f"{float(oof_raw[mask, 1].mean()):.4f} / "
                  f"{float(oof_raw[mask, 1].max()):.4f}")

            cal_dict = {}
            oof_cal = np.full((len(df), 2), np.nan, dtype=np.float32)
            for k, name in enumerate(['long', 'short']):
                p_raw_k = np.clip(oof_raw[mask, k].astype(float), 1e-4, 1.0 - 1e-4)
                y_true_k = oof_y[mask, k].astype(int)
                pos_k = int(y_true_k.sum())
                print(f"  [cal-{name}] Pos={pos_k:,}  Neg={len(y_true_k) - pos_k:,}")
                cal_k = IsotonicRegression(out_of_bounds="clip")
                cal_k.fit(p_raw_k, y_true_k)
                p_cal_k = cal_k.predict(p_raw_k).astype(np.float32)
                if abs(self.temperature - 1.0) > 1e-6:
                    p_cal_k = self._apply_temperature(p_cal_k, self.temperature)
                oof_cal[mask, k] = p_cal_k
                cal_dict[name] = cal_k

            # Guardar en df: dos columnas cada (raw, cal). Aliases:
            #   oof_proba_raw / oof_proba_cal apuntan al LADO LONG
            #   (downstream que aún espera columna única).
            df["oof_proba_long_raw"] = oof_raw[:, 0]
            df["oof_proba_long_cal"] = oof_cal[:, 0]
            df["oof_proba_short_raw"] = oof_raw[:, 1]
            df["oof_proba_short_cal"] = oof_cal[:, 1]
            df["oof_proba_raw"] = oof_raw[:, 0]
            df["oof_proba_cal"] = oof_cal[:, 0]

            return df, {"type": "multitask_isotonic", **cal_dict}, best_epochs

        # ── Calibración isotónica (binary, comportamiento original) ────────
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