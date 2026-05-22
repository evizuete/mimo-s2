from dataclasses import dataclass
from typing import Tuple, Dict

import numpy as np
import tensorflow as tf
from keras import Model, Input, layers, regularizers
from keras.src.callbacks import EarlyStopping, ReduceLROnPlateau, Callback, TerminateOnNaN
from keras.src.optimizers import Adam
from keras.src.saving import load_model, register_keras_serializable


@dataclass
class Config:
    release: str | None = None
    period: int = 120
    val_size: float = 0.20
    use_oof: bool = True
    oof_splits: int = 6
    oof_epochs: int = 25
    save_oof_artifacts: bool = True

@dataclass
class ModelConfig:
    """Configuración del modelo single-output optimizado"""
    # Arquitectura multi-scale
    seq_len_short: int = 64
    seq_len_long: int = 256

    # Capas
    conv1d_filters: int = 96
    lstm_units: int = 96
    dense_units: int = 64
    context_units: int = 48
    time_units: int = 32
    head_units: int = 64

    # Regularización
    dropout_seq: float = 0.15
    dropout_lstm: float = 0.15
    dropout_dense: float = 0.20
    l2_reg: float = 1e-5

    # Entrenamiento
    learning_rate: float = 1e-3  # 1e-4
    batch_size: int = 4096  # 256
    epochs: int = 100
    patience: int = 15

    # Focal loss para desbalance
    focal_alpha: float = 0.25
    focal_gamma: float = 1.25  # 1.5 #2.0

    # Features
    use_attention: bool = True
    use_gate: bool = True

    # v3: fusión jerárquica — separa "contexto de mercado" de "trigger de entrada"
    # False → comportamiento idéntico a v2 (retrocompatible)
    # True  → build_model_v3: market_repr = [x_long, x_context]
    #                          entry_repr  = [x_short, x_time]
    #                          combined    = [market_repr, entry_repr]
    use_hierarchical_fusion: bool = False

    # v3: hybrid loss = focal + ranking
    # 0.0 → solo focal (comportamiento actual, retrocompatible)
    # 0.1-0.3 → término de ranking ListNet ponderado
    # El ranking loss empuja al modelo a ORDENAR bien las señales dentro del batch,
    # no solo a clasificarlas — mejora directamente la calidad de los percentiles.
    ranking_loss_weight: float = 0.0

    # Tipo de target/output:
    #   "binary"   → clasificación binaria con sigmoid + focal/hybrid loss (default)
    #   "quantile" → regresión cuantílica con cabeza lineal y pinball loss.
    # En modo quantile el modelo predice len(quantile_levels) cuantiles del
    # forward return normalizado por ATR; ranking_loss y focal_alpha/gamma se
    # ignoran. La calibración en probs_calibration aplica un shift conformal.
    target_type: str = "binary"
    quantile_levels: tuple = (0.25, 0.50, 0.75)

    # Multi-task LONG+SHORT (target_type='multitask'):
    #   Dos cabezas binarias compartiendo el trunk. Cada una con su loss focal,
    #   sus métricas (auc_pr, auc_roc) y sus sample_weights independientes.
    #   loss_weight_long/short escalan la contribución de cada cabeza al loss
    #   total. Útil para compensar asimetría: si SHORT pos_rate >> LONG, subir
    #   loss_weight_long para que la cabeza débil reciba más gradient.
    loss_weight_long: float = 1.0
    loss_weight_short: float = 1.0

    # Activación de las capas Dense del MLP (gelu | relu | swish | elu).
    # Solo lo leen las archs que invocan get_activation() del model_config
    # (actualmente: mlp_flatten). Las archs legacy (mlp, hybrid, etc.)
    # ignoran este campo y usan 'gelu' hardcoded para mantener compat.
    activation: str = 'gelu'

    # ── HPs estructurales v4 (CNN-LSTM build_model_v3 v4) ──
    # Reales fields del dataclass (no setattr) para que ModelConfig(**vars(mc))
    # —patrón usado en optuna_oof_trainer y probs_calibration— no rompa.
    # Semántica:
    #   · valores > 0 → se usan tal cual.
    #   · 0           → "auto / derivar" (legacy v3 default).
    #   build_model_v3 v4 honra la sentinel 0 explícitamente.
    # Backward-compat: si la release no toca estos campos en su grid_space,
    # los defaults producen exactamente el modelo v3 legacy.
    kernel_size_short: int = 3      # Conv1D rama corta (legacy 3)
    kernel_size_long:  int = 5      # Conv1D rama larga (legacy 5)
    gru_units:         int = 0      # 0 → lstm_units // 2 (legacy derivado)
    attn_num_heads:    int = 4      # MultiHeadAttention heads (legacy 4)
    attn_key_dim:      int = 0      # 0 → max(8, ch_short // 4) (legacy)

    # ── HPs estructurales TCN (build_tcn_mlp en model_alternatives.py) ──
    # Mismo patrón: fields reales para que ModelConfig(**vars(mc)) no rompa
    # cuando viene de un trial TCN (commit a99ef61 hizo condicional el suggest,
    # pero los attributes se persisten al ModelConfig y la reconstrucción
    # vía **vars() en probs_calibration.py:292 fallaba con TypeError).
    # Defaults: matchean los legacy de build_tcn_mlp (cero impacto si la
    # release no los declara en grid_space).
    kernel_size:              int   = 3       # TCN: kernel único (no split)
    n_tcn_blocks_long:        int   = 5       # bloques rama larga (dilations 2^i)
    n_tcn_blocks_short:       int   = 3       # bloques rama corta (legacy=3)
    tcn_filters_short_ratio:  float = 0.5     # filters_short = filters_long * ratio
    ctx_dense_units:          int   = 64      # ancho de la Dense(ctx+time)
    tcn_pooling:              str   = "gap"   # gap | gmp | gap_gmp | attention
    use_se:                   int   = 0       # 0=off, 1=Squeeze-and-Excite por bloque
    se_ratio:                 int   = 8       # ratio reducción canal en SE

    # ── Arquitectura del modelo ──
    # 'original_v3' (default) → CNN-LSTM nativo (build_model_v2/v3 según
    # use_hierarchical_fusion). Otras opciones delegan en build_model_by_arch
    # (model_alternatives.py): 'mlp', 'mlp_flatten', 'hybrid', 'transformer',
    # 'tcn'. Permite construir modelos no-CNN-LSTM desde el pipeline de deploy
    # (Fase 2/3). El walkforward 010 sigue usando _walkforward_arch attribute
    # como mecanismo paralelo legacy.
    arch: str = "original_v3"


class TradingModel:
    """Modelo de deep learning con arquitectura multi-scale"""

    def __init__(self, general_config: Config, model_config: ModelConfig = ModelConfig(), side = None):
        self.general_config = general_config
        self.model_config = model_config
        self.side = side
        self.model = None
        self.history = None

    def _build_output_head(self, x, init_bias):
        """
        Construye la cabeza de salida según self.model_config.target_type.

        target_type='binary' (default):
            Dense(1) → sigmoid → 'signal' (probabilidad).
            init_bias (float) se aplica al bias del Dense para acelerar
            convergencia con clases desbalanceadas.

        target_type='quantile':
            Dense(N_q, linear) → 'signal' (vector de cuantiles del return).

        target_type='triple_class':
            Dense(3, softmax) → 'signal' sobre {0=SL, 1=TIMEOUT, 2=TP}.

        target_type='multitask':
            Dos cabezas binarias compartiendo el trunk:
              Dense(1, sigmoid) → 'signal_long'
              Dense(1, sigmoid) → 'signal_short'
            init_bias debe ser dict {'long': bias_l, 'short': bias_s} o
            un float (se aplica a ambas cabezas).
            Devuelve LISTA [p_long, p_short]; el resto de la API de Keras
            (Model(outputs=...), compile con loss dict, fit con y dict)
            se encarga del wiring.
        """
        if self.model_config.target_type == "quantile":
            n_q = len(self.model_config.quantile_levels)
            return layers.Dense(
                n_q,
                activation='linear',
                name='signal',
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                bias_initializer='zeros',
            )(x)

        if self.model_config.target_type == "triple_class":
            # Softmax(3) sobre clases {0=SL, 1=TIMEOUT, 2=TP}.
            # Downstream usa signal[..., 2] como P(TP) para calibración y umbrales.
            return layers.Dense(
                3,
                activation='softmax',
                name='signal',
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                bias_initializer='zeros',
            )(x)

        if self.model_config.target_type == "multitask":
            if isinstance(init_bias, dict):
                bias_long = float(init_bias.get('long', 0.0))
                bias_short = float(init_bias.get('short', 0.0))
            else:
                bias_long = bias_short = float(init_bias)
            logit_long = layers.Dense(
                1, name='logit_long',
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                bias_initializer=tf.keras.initializers.Constant(bias_long),
            )(x)
            logit_short = layers.Dense(
                1, name='logit_short',
                kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
                bias_initializer=tf.keras.initializers.Constant(bias_short),
            )(x)
            p_long = layers.Activation('sigmoid', name='signal_long')(logit_long)
            p_short = layers.Activation('sigmoid', name='signal_short')(logit_short)
            # Devolvemos un dict — Keras 3 usa estructura dict-keyed para
            # match loss/metrics/y/sample_weight por nombre. Devolverlo como
            # list provoca KeyError(0) en compile_utils.resolve_path al
            # intentar indexar un dict de losses con un int de la lista.
            return {'signal_long': p_long, 'signal_short': p_short}

        # Camino original (binario): logits + sigmoid
        signal_logit = layers.Dense(
            1,
            name='signal_logit',
            kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
            bias_initializer=tf.keras.initializers.Constant(float(init_bias)),
        )(x)
        return layers.Activation('sigmoid', name='signal')(signal_logit)

    def build_model(self,
                    shape_short: Tuple[int, int],
                    shape_long: Tuple[int, int],
                    n_context: int,
                    n_time: int,
                    init_bias: float = 0.0) -> Model:
        """
        Construye el modelo con múltiples inputs y single output

        Args:
            shape_short: (seq_len_short, n_features_short)
            shape_long: (seq_len_long, n_features_long)
            n_context: número de features de contexto
            n_time: número de features temporales
            init_bias: bias inicial para la capa de salida
        """
        config = self.model_config

        # === INPUTS ===
        input_short = Input(shape=shape_short, name='seq_short')
        input_long = Input(shape=shape_long, name='seq_long')
        input_context = Input(shape=(n_context,), name='context')
        input_time = Input(shape=(n_time,), name='time')

        # === PROCESAMIENTO SECUENCIA CORTA ===
        # Conv1D para capturar patrones locales
        x_short = layers.Conv1D(
            config.conv1d_filters,
            kernel_size=3,
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(config.l2_reg)
        )(input_short)
        x_short = layers.BatchNormalization()(x_short)
        x_short = layers.Dropout(config.dropout_seq)(x_short)

        # LSTM para dependencias temporales
        x_short = layers.LSTM(
            config.lstm_units,
            return_sequences=False,
            dropout=config.dropout_lstm,
            recurrent_dropout=config.dropout_lstm,
            kernel_regularizer=regularizers.l2(config.l2_reg)
        )(x_short)

        # === PROCESAMIENTO SECUENCIA LARGA ===
        # Reducir dimensionalidad con Conv1D strided
        x_long = layers.Conv1D(
            config.conv1d_filters // 2,
            kernel_size=5,
            strides=2,
            padding='same',
            activation='relu'
        )(input_long)
        x_long = layers.BatchNormalization()(x_long)

        # GRU más eficiente para secuencias largas
        x_long = layers.GRU(
            config.lstm_units // 2,
            return_sequences=False,
            dropout=config.dropout_lstm,
            kernel_regularizer=regularizers.l2(config.l2_reg)
        )(x_long)

        # === PROCESAMIENTO CONTEXTO ===
        x_context = layers.Dense(
            config.context_units,
            activation='relu',
            kernel_regularizer=regularizers.l2(config.l2_reg)
        )(input_context)
        x_context = layers.Dropout(config.dropout_dense)(x_context)

        # === PROCESAMIENTO TIEMPO ===
        x_time = layers.Dense(
            config.time_units,
            activation='relu',
            kernel_regularizer=regularizers.l2(config.l2_reg)
        )(input_time)

        # === ATENCIÓN (opcional) ===
        if config.use_attention:
            # Self-attention en la secuencia corta
            attention = layers.MultiHeadAttention(
                num_heads=4,
                key_dim=config.lstm_units // 4
            )(x_short[..., tf.newaxis], x_short[..., tf.newaxis])
            attention = layers.Flatten()(attention)
            x_short = layers.Add()([x_short, attention])
            x_short = layers.LayerNormalization()(x_short)

        # === FUSIÓN ===
        combined = layers.Concatenate()([x_short, x_long, x_context, x_time])

        # Gated fusion (opcional)
        if config.use_gate:
            gate = layers.Dense(
                combined.shape[-1],
                activation='sigmoid',
                kernel_regularizer=regularizers.l2(config.l2_reg * 0.1)
            )(combined)
            combined = layers.Multiply()([combined, gate])

        # === HEAD FINAL ===
        x = layers.Dense(
            config.head_units,
            activation='relu',
            kernel_regularizer=regularizers.l2(config.l2_reg)
        )(combined)
        x = layers.Dropout(config.dropout_dense)(x)

        x = layers.Dense(
            config.head_units // 2,
            activation='relu',
            kernel_regularizer=regularizers.l2(config.l2_reg)
        )(x)
        x = layers.Dropout(config.dropout_dense)(x)

        # === OUTPUT ===
        signal_output = self._build_output_head(x, init_bias)

        # === CREAR MODELO ===
        model = Model(
            inputs=[input_short, input_long, input_context, input_time],
            outputs=signal_output,
            name='TradingModel'
        )

        self.model = model
        return model

    def build_model_v2(self,
                    shape_short: Tuple[int, int],
                    shape_long: Tuple[int, int],
                    n_context: int,
                    n_time: int,
                    init_bias: float = 0.0) -> Model:
        """
        Construye el modelo con múltiples inputs y single output

        Args:
            shape_short: (seq_len_short, n_features_short)
            shape_long: (seq_len_long, n_features_long)
            n_context: número de features de contexto
            n_time: número de features temporales
            init_bias: bias inicial para la capa de salida
        """
        config = self.model_config

        # === INPUTS ===
        input_short = Input(shape=shape_short, name='seq_short')
        input_long = Input(shape=shape_long, name='seq_long')
        input_context = Input(shape=(n_context,), name='context')
        input_time = Input(shape=(n_time,), name='time')

        # === PROCESAMIENTO SECUENCIA CORTA ===
        # Conv1D para capturar patrones locales
        x_short = layers.Conv1D(
            config.conv1d_filters,
            kernel_size=3,
            padding='same',
            activation='relu',
            kernel_regularizer=regularizers.l2(config.l2_reg)
        )(input_short)
        x_short = layers.BatchNormalization()(x_short)
        x_short = layers.Dropout(config.dropout_seq)(x_short)

        if config.use_attention:
            att = layers.MultiHeadAttention(
                num_heads=4,
                key_dim=max(8, x_short.shape[-1] // 4),
            )(x_short, x_short)  # (B, T, C)

            x_short = layers.Add()([x_short, att])
            x_short = layers.LayerNormalization()(x_short)

        # ahora ya sí reduces a vector
        x_short = layers.LSTM(
            config.lstm_units,
            return_sequences=False,
            dropout=config.dropout_lstm,
            recurrent_dropout=config.dropout_lstm,
            kernel_regularizer=regularizers.l2(config.l2_reg)
        )(x_short)

        # === PROCESAMIENTO SECUENCIA LARGA ===
        # Reducir dimensionalidad con Conv1D strided
        x_long = layers.Conv1D(
            config.conv1d_filters // 2,
            kernel_size=5,
            strides=2,
            padding='same',
            activation='relu'
        )(input_long)
        x_long = layers.BatchNormalization()(x_long)

        # GRU más eficiente para secuencias largas
        x_long = layers.GRU(
            config.lstm_units // 2,
            return_sequences=False,
            dropout=config.dropout_lstm,
            kernel_regularizer=regularizers.l2(config.l2_reg)
        )(x_long)

        # === PROCESAMIENTO CONTEXTO ===
        x_context = layers.Dense(
            config.context_units,
            activation='relu',
            kernel_regularizer=regularizers.l2(config.l2_reg)
        )(input_context)
        x_context = layers.Dropout(config.dropout_dense)(x_context)

        # === PROCESAMIENTO TIEMPO ===
        x_time = layers.Dense(
            config.time_units,
            activation='relu',
            kernel_regularizer=regularizers.l2(config.l2_reg)
        )(input_time)

        # === FUSIÓN ===
        combined = layers.Concatenate()([x_short, x_long, x_context, x_time])

        # Gated fusion (opcional)
        if config.use_gate:
            gate = layers.Dense(
                combined.shape[-1],
                activation='sigmoid',
                kernel_regularizer=regularizers.l2(config.l2_reg * 0.1)
            )(combined)
            combined = layers.Multiply()([combined, gate])

        # === HEAD FINAL ===
        x = layers.Dense(
            config.head_units,
            activation='relu',
            kernel_regularizer=regularizers.l2(config.l2_reg)
        )(combined)
        x = layers.Dropout(config.dropout_dense)(x)

        x = layers.Dense(
            config.head_units // 2,
            activation='relu',
            kernel_regularizer=regularizers.l2(config.l2_reg)
        )(x)
        x = layers.Dropout(config.dropout_dense)(x)

        # === OUTPUT ===
        signal_output = self._build_output_head(x, init_bias)

        # === CREAR MODELO ===
        model = Model(
            inputs=[input_short, input_long, input_context, input_time],
            outputs=signal_output,
            name='TradingModel'
        )

        self.model = model
        return model

    def build_model_v3(self,
                    shape_short: Tuple[int, int],
                    shape_long: Tuple[int, int],
                    n_context: int,
                    n_time: int,
                    init_bias: float = 0.0) -> Model:
        """
        v3: Fusión jerárquica en dos pasos.

        En v2 todo se concatena de golpe:
            combined = Concat([x_short, x_long, x_context, x_time])

        En v3 se separa en dos representaciones semánticas antes de fusionar:
            market_repr = Concat([x_long, x_context])   → "¿hay contexto favorable?"
            entry_repr  = Concat([x_short, x_time])     → "¿hay trigger ejecutable?"
            combined    = Concat([market_repr, entry_repr])

        Ventajas:
        - El modelo aprende explícitamente a separar calidad de contexto vs timing.
        - El score final es más ordenable (mejora percentiles por estado).
        - No requiere cambios en pipeline ni en feature_builder.
        - Compatible con los mismos 4 inputs que v2.

        Activar con: ModelConfig(use_hierarchical_fusion=True)

        HPs estructurales expuestos en v4 (backward-compat vía getattr):
          · activation        ∈ {relu, gelu, swish, elu}    default 'relu' (legacy)
          · kernel_size_short ∈ {3, 5, 7}                   default 3
          · kernel_size_long  ∈ {3, 5, 7}                   default 5
          · gru_units         int                           default lstm_units // 2
          · attn_num_heads    ∈ {2, 4, 8}                   default 4
          · attn_key_dim      int                           default max(8, ch_short // 4)

        Antes (v3 legacy) activation estaba hard-coded a 'relu' en TODAS las
        Dense/Conv (el campo model_config.activation existía pero se ignoraba
        silenciosamente — bug). v4 lo respeta.
        """
        config = self.model_config

        # HPs estructurales (campos reales del dataclass + getattr para
        # retrocompat con tests/llamadas que pasan model_configs sintéticos sin
        # estos campos). Sentinel 0 → "auto / derivar" (defaults legacy v3).
        act        = str(getattr(config, 'activation', 'relu')).lower()
        ksize_s    = int(getattr(config, 'kernel_size_short', 3) or 3)
        ksize_l    = int(getattr(config, 'kernel_size_long', 5) or 5)
        gru_units_raw = int(getattr(config, 'gru_units', 0) or 0)
        gru_units  = gru_units_raw if gru_units_raw > 0 else (config.lstm_units // 2)
        attn_heads = int(getattr(config, 'attn_num_heads', 4) or 4)

        # === INPUTS (idénticos a v2) ===
        input_short   = Input(shape=shape_short, name='seq_short')
        input_long    = Input(shape=shape_long,  name='seq_long')
        input_context = Input(shape=(n_context,), name='context')
        input_time    = Input(shape=(n_time,),    name='time')

        # === RAMA CORTA — microestructura y trigger local ===
        x_short = layers.Conv1D(
            config.conv1d_filters,
            kernel_size=ksize_s,
            padding='same',
            activation=act,
            kernel_regularizer=regularizers.l2(config.l2_reg)
        )(input_short)
        x_short = layers.BatchNormalization()(x_short)
        x_short = layers.Dropout(config.dropout_seq)(x_short)

        if config.use_attention:
            # attn_key_dim: 0 → auto-derivar de la dimensión de canal (legacy).
            attn_key_dim_raw = int(getattr(config, 'attn_key_dim', 0) or 0)
            attn_key_dim = (
                attn_key_dim_raw if attn_key_dim_raw > 0
                else max(8, int(x_short.shape[-1]) // 4)
            )
            att = layers.MultiHeadAttention(
                num_heads=attn_heads,
                key_dim=attn_key_dim,
            )(x_short, x_short)
            x_short = layers.Add()([x_short, att])
            x_short = layers.LayerNormalization()(x_short)

        x_short = layers.LSTM(
            config.lstm_units,
            return_sequences=False,
            dropout=config.dropout_lstm,
            recurrent_dropout=0.0,
            kernel_regularizer=regularizers.l2(config.l2_reg)
        )(x_short)

        # === RAMA LARGA — régimen y estructura macro ===
        x_long = layers.Conv1D(
            config.conv1d_filters // 2,
            kernel_size=ksize_l,
            strides=2,
            padding='same',
            activation=act
        )(input_long)
        x_long = layers.BatchNormalization()(x_long)

        x_long = layers.GRU(
            gru_units,
            return_sequences=False,
            dropout=config.dropout_lstm,
            kernel_regularizer=regularizers.l2(config.l2_reg)
        )(x_long)

        # === CONTEXTO ===
        x_context = layers.Dense(
            config.context_units,
            activation=act,
            kernel_regularizer=regularizers.l2(config.l2_reg)
        )(input_context)
        x_context = layers.LayerNormalization()(x_context)
        x_context = layers.Dropout(config.dropout_dense)(x_context)

        # === TIEMPO ===
        x_time = layers.Dense(
            config.time_units,
            activation=act,
            kernel_regularizer=regularizers.l2(config.l2_reg)
        )(input_time)

        # === FUSIÓN JERÁRQUICA ===
        # Paso A: representación de mercado (contexto estructural)
        # Responde: "¿el régimen y la estructura de fondo son favorables?"
        # NB: market_units sigue derivado de gru_units (no de lstm_units // 2).
        market_repr = layers.Concatenate()([x_long, x_context])
        market_units = gru_units + config.context_units
        market_repr = layers.Dense(
            market_units,
            activation=act,
            kernel_regularizer=regularizers.l2(config.l2_reg)
        )(market_repr)
        market_repr = layers.Dropout(config.dropout_dense)(market_repr)

        # Paso B: representación de entrada (timing y microestructura)
        # Responde: "¿hay un trigger ejecutable ahora mismo?"
        entry_repr = layers.Concatenate()([x_short, x_time])
        entry_units = config.lstm_units + config.time_units
        entry_repr = layers.Dense(
            entry_units,
            activation=act,
            kernel_regularizer=regularizers.l2(config.l2_reg)
        )(entry_repr)
        entry_repr = layers.Dropout(config.dropout_dense)(entry_repr)

        # Paso C: decisión final = contexto × trigger
        combined = layers.Concatenate()([market_repr, entry_repr])

        # Gate opcional sobre la fusión final
        if config.use_gate:
            gate = layers.Dense(
                combined.shape[-1],
                activation='sigmoid',  # gate semantics: sigmoid es estructural
                kernel_regularizer=regularizers.l2(config.l2_reg * 0.1)
            )(combined)
            combined = layers.Multiply()([combined, gate])

        # === HEAD FINAL (idéntico a v2) ===
        x = layers.Dense(
            config.head_units,
            activation=act,
            kernel_regularizer=regularizers.l2(config.l2_reg)
        )(combined)
        x = layers.Dropout(config.dropout_dense)(x)

        x = layers.Dense(
            config.head_units // 2,
            activation=act,
            kernel_regularizer=regularizers.l2(config.l2_reg)
        )(x)
        x = layers.Dropout(config.dropout_dense)(x)

        # === OUTPUT ===
        signal_output = self._build_output_head(x, init_bias)

        model = Model(
            inputs=[input_short, input_long, input_context, input_time],
            outputs=signal_output,
            name='TradingModel_v3'
        )

        self.model = model
        return model

    def compile_model(self, class_weight: Dict[int, float] = None):
        optimizer = Adam(
            learning_rate=self.model_config.learning_rate,
            clipnorm=1.0,
            beta_1=0.9,
            beta_2=0.999,
            epsilon=1e-8
        )

        if self.model_config.target_type == "quantile":
            # ── Quantile regression ───────────────────────────────────────────
            # Pinball loss multi-output. focal_alpha/gamma y ranking_loss_weight
            # se ignoran (no aplican). Métricas: MAE sobre el cuantil mediano y
            # cobertura empírica del intervalo predicho.
            qs = self.model_config.quantile_levels
            loss = PinballLoss(quantiles=qs)

            mid_idx = len(qs) // 2  # típicamente 1 para (0.25, 0.50, 0.75)
            metrics = [
                QuantileMAE(quantile_idx=mid_idx, name=f"mae_q{int(qs[mid_idx]*100)}"),
                QuantileCoverage(low_idx=0, high_idx=len(qs) - 1, name="coverage"),
            ]
            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            return

        if self.model_config.target_type == "triple_class":
            # ── Triple barrier 3-class ────────────────────────────────────────
            # SparseCategoricalCrossentropy sobre {0=SL, 1=TIMEOUT, 2=TP}.
            # focal_alpha/gamma y ranking_loss_weight se ignoran. La métrica
            # AUC se computa sobre P(TP) vs no-TP (binarización implícita).
            loss = tf.keras.losses.SparseCategoricalCrossentropy()
            metrics = [
                tf.keras.metrics.SparseCategoricalAccuracy(name='acc'),
                TripleClassTPAUC(name='auc_pr_tp', curve='PR'),
                TripleClassTPAUC(name='auc_roc_tp', curve='ROC'),
            ]
            self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
            return

        if self.model_config.target_type == "multitask":
            # ── Multi-task LONG + SHORT (dos cabezas binarias) ────────────────
            # Cada cabeza tiene su propio focal loss. focal_alpha puede venir
            # como float (mismo para ambas) o como dict {'long': ..., 'short': ...}.
            # ranking_loss_weight se ignora (la diversidad ya viene de las dos
            # cabezas con regimen weights independientes).
            fa = self.model_config.focal_alpha
            if isinstance(fa, dict):
                alpha_long = float(fa.get('long', 0.35))
                alpha_short = float(fa.get('short', 0.40))
            else:
                alpha_long = alpha_short = float(fa)
            gamma = float(self.model_config.focal_gamma)

            losses = {
                'signal_long': ClippedBinaryFocalCrossentropy(
                    alpha=alpha_long, gamma=gamma, from_logits=False,
                ),
                'signal_short': ClippedBinaryFocalCrossentropy(
                    alpha=alpha_short, gamma=gamma, from_logits=False,
                ),
            }
            metrics = {
                'signal_long': [
                    tf.keras.metrics.AUC(name='auc_roc', curve='ROC'),
                    tf.keras.metrics.AUC(name='auc_pr', curve='PR'),
                ],
                'signal_short': [
                    tf.keras.metrics.AUC(name='auc_roc', curve='ROC'),
                    tf.keras.metrics.AUC(name='auc_pr', curve='PR'),
                ],
            }
            # loss_weights configurables vía atributos en model_config; defaults 1.0.
            lw_long = float(getattr(self.model_config, 'loss_weight_long', 1.0))
            lw_short = float(getattr(self.model_config, 'loss_weight_short', 1.0))
            self.model.compile(
                optimizer=optimizer,
                loss=losses,
                loss_weights={'signal_long': lw_long, 'signal_short': lw_short},
                metrics=metrics,
            )
            return

        # ── Hybrid loss: focal + ranking (ListNet) ────────────────────────────
        # Si ranking_loss_weight=0.0 → solo focal (retrocompatible con v1/v2).
        # Si ranking_loss_weight>0.0 → focal + ListNet ponderado.
        #
        # ListNet ranking loss:
        #   Entrena al modelo a ORDENAR bien las señales dentro del batch,
        #   no solo a clasificarlas. Esto mejora directamente la calidad de
        #   los percentiles por estado: el top 10% de señales debería tener
        #   un win rate significativamente mayor que el top 50%.
        #
        #   Implementación: softmax sobre scores vs softmax sobre labels →
        #   cross-entropy entre las dos distribuciones de ranking.
        #   Solo se aplica cuando hay suficiente variedad de labels en el batch
        #   (evita NaN cuando todos los labels son 0 o todos son 1).
        #
        ranking_w = float(self.model_config.ranking_loss_weight)

        focal_loss_fn = ClippedBinaryFocalCrossentropy(
            alpha=self.model_config.focal_alpha,
            gamma=self.model_config.focal_gamma,
            from_logits=False,
        )

        if ranking_w > 0.0:
            # HybridFocalRankingLoss definida a nivel de módulo para que Keras
            # pueda localizarla correctamente al deserializar el modelo desde disco.
            loss = HybridFocalRankingLoss(
                alpha=self.model_config.focal_alpha,
                gamma=self.model_config.focal_gamma,
                ranking_weight=ranking_w,
            )
        else:
            loss = focal_loss_fn

        metrics = [
            tf.keras.metrics.AUC(name='auc_roc', curve='ROC'),
            tf.keras.metrics.AUC(name='auc_pr', curve='PR'),
            # precision_30/recall_30 eliminados: con pos_rate~0.274 el threshold 0.30
            # está solo 0.026 sobre la base rate → recall_30≈1.0 siempre, no discrimina.
            # precision_35 añadido: threshold 0.076 sobre la base rate, zona intermedia
            # donde ranking_loss debería mostrar mejora consistente.
            tf.keras.metrics.Precision(name='precision_35', thresholds=0.35),
            tf.keras.metrics.Precision(name='precision_40', thresholds=0.40),
            tf.keras.metrics.Recall(name='recall_40', thresholds=0.40),
        ]

        # loss ya definida arriba: focal puro (ranking_loss_weight=0) o hybrid (>0)
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=metrics
        )

    def get_logits(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        """Extrae los logits antes de la activación sigmoid (solo modo binary)."""
        if self.model_config.target_type == "quantile":
            raise NotImplementedError(
                "get_logits() no aplica en modo quantile: la cabeza es lineal "
                "y no hay layer 'signal_logit'. Usa model.predict() directamente "
                "para obtener los cuantiles."
            )
        logit_model = Model(
            inputs=self.model.inputs,
            outputs=self.model.get_layer('signal_logit').output
        )

        logits = logit_model.predict(
            [X['seq_short'], X['seq_long'], X['context'], X['time']],
            batch_size=self.model_config.batch_size,
            verbose=0
        )

        return logits.ravel()

    def train(self,
              X_train: Dict[str, np.ndarray],
              y_train: np.ndarray,
              X_val: Dict[str, np.ndarray] = None,
              y_val: np.ndarray = None,
              sample_weight: np.ndarray = None,
              verbose: int = 1,
              for_production: bool = False) -> Dict:
        """
        Entrena el modelo

        Args:
            X_train: Dict con keys ['seq_short', 'seq_long', 'context', 'time']
            y_train: Etiquetas de entrenamiento
            X_val: Datos de validación (opcional)
            y_val: Etiquetas de validación (opcional)
            sample_weight: Pesos para las muestras (opcional)
            verbose: Nivel de verbosidad
        """

        # Callbacks
        is_quantile_callbacks = (self.model_config.target_type == "quantile")
        is_triple_class_callbacks = (self.model_config.target_type == "triple_class")
        is_multitask_callbacks = (self.model_config.target_type == "multitask")
        # Para triple_class, las métricas binarias usan sufijo '_tp'.
        # Para multitask, vigilamos el AUC-PR de la cabeza LONG (suele ser
        # el lado con menos pos_rate y por tanto más informativo); el SHORT
        # entrena en paralelo con su propia loss.
        if is_triple_class_callbacks:
            train_auc_metric = "auc_pr_tp"
        elif is_multitask_callbacks:
            train_auc_metric = "signal_long_auc_pr"
        else:
            train_auc_metric = "auc_pr"
        val_auc_metric = f"val_{train_auc_metric}"

        if for_production:
            callbacks = [
                # LR schedule: reduce a mitad cada N épocas fijas si la loss no mejora
                ReduceLROnPlateau(
                    monitor='loss',  # train loss — lo único disponible
                    factor=0.5,
                    patience=max(3, self.model_config.patience // 4),
                    min_lr=1e-6,
                    mode='min',
                    verbose=verbose
                ),
                TerminateOnNaN(),
            ]
        elif is_quantile_callbacks:
            # En quantile mode no hay val_auc_pr; usamos val_loss (pinball) a minimizar.
            # GeneralizationGapStopping se desactiva: val_loss es continuo y la noción
            # de "gap" entre auc_pr de train y val no aplica directamente (la pinball
            # de train baja monotónicamente y la "gap" es ruidosa para regresión).
            mon = 'val_loss' if X_val is not None else 'loss'
            callbacks = [
                EarlyStopping(
                    monitor=mon,
                    patience=self.model_config.patience,
                    restore_best_weights=True,
                    mode='min',
                    verbose=verbose,
                ),
                ReduceLROnPlateau(
                    monitor=mon,
                    factor=0.5,
                    patience=self.model_config.patience // 3,
                    min_lr=1e-6,
                    mode='min',
                    verbose=verbose,
                ),
                TerminateOnNaN(),
            ]
        else:
            callbacks = [
                EarlyStopping(
                    monitor=val_auc_metric if X_val is not None else train_auc_metric,
                    patience=self.model_config.patience,
                    restore_best_weights=True,
                    mode='max',
                    verbose=verbose
                ),
                ReduceLROnPlateau(monitor=val_auc_metric if X_val is not None else train_auc_metric,
                                  factor=0.5,
                                  patience=self.model_config.patience // 3,  # más reactivo que ES
                                  min_lr=1e-6,
                                  mode='max',  # ← obligatorio cuando monitor es AUC
                                  verbose=verbose),
                GeneralizationGapStopping(
                    monitor_train=train_auc_metric,
                    monitor_val=val_auc_metric,
                    mode='max',
                    max_gap=0.03,
                    min_epochs=8,
                    patience=2,
                    restore_best_weights=True
                ), TerminateOnNaN(),
            ]

        # Multitask: y_train/y_val esperados como np.ndarray (N, 2) con columnas
        # [is_long_TP, is_short_TP]. sample_weight esperado como (N, 2) con la
        # misma convención. Convertimos a dict para Keras multi-output.
        if is_multitask_callbacks:
            def _split_dual(arr, name=''):
                if arr is None:
                    return None
                a = np.asarray(arr)
                if a.ndim != 2 or a.shape[-1] != 2:
                    raise ValueError(
                        f"multitask {name}: shape {a.shape} inesperado, "
                        f"se requiere (N, 2) con columnas [long, short]."
                    )
                return {
                    'signal_long': a[:, 0].astype(np.float32),
                    'signal_short': a[:, 1].astype(np.float32),
                }

            y_train_in = _split_dual(y_train, 'y_train')
            y_val_in = _split_dual(y_val, 'y_val')
            sw_in = _split_dual(sample_weight, 'sample_weight')
        else:
            y_train_in = y_train
            y_val_in = y_val
            sw_in = sample_weight

        # Datos de validación
        validation_data = None
        if X_val is not None and y_val_in is not None:
            validation_data = (
                [X_val['seq_short'], X_val['seq_long'], X_val['context'], X_val['time']],
                y_val_in
            )
        else:
            validation_data = None

        # Entrenar
        self.history = self.model.fit(
            x=[X_train['seq_short'], X_train['seq_long'], X_train['context'], X_train['time']],
            y=y_train_in,
            batch_size=self.model_config.batch_size,
            epochs=self.model_config.epochs,
            validation_data=validation_data,
            sample_weight=sw_in,
            callbacks=callbacks,
            verbose=verbose,
            shuffle=False  # Importante para series temporales
        )

        return self.history.history

    def predict(self, X: Dict[str, np.ndarray]) -> np.ndarray:
        """Realiza predicciones"""
        return self.model.predict(
            [X['seq_short'], X['seq_long'], X['context'], X['time']],
            batch_size=self.model_config.batch_size,
            verbose=0
        ).ravel()

    def evaluate(self, X: Dict[str, np.ndarray], y: np.ndarray) -> Dict:
        """Evalúa el modelo"""
        results = self.model.evaluate(
            [X['seq_short'], X['seq_long'], X['context'], X['time']],
            y,
            batch_size=self.model_config.batch_size,
            verbose=0,
            return_dict=True
        )
        return results

    def save(self, path: str = None):
        filename = f'model_{self.general_config.release}_{self.side}.keras'
        if path is None:
            path = f'./{filename}'
        else:
            path = f'{path}/{filename}'

        self.model.save(path)

    def load(self, path: str = None):
        filename = f'model_{self.general_config.release}_{self.side}.keras'
        if path is None:
            path = f'./{filename}'
        else:
            path = f'{path}/{filename}'

        # safe_mode=False permite cargar Lambda layers usadas en arch alternativas
        # (TCN attention pooling, p.ej.). El .keras file lo produce nuestro
        # pipeline (no es untrusted input), así que aceptable.
        self.model = load_model(path, safe_mode=False)

class GeneralizationGapStopping(Callback):
    def __init__(self,
                 monitor_train='auc_pr',  # métrica en entrenamiento
                 monitor_val='val_auc_pr',  # métrica en validación
                 mode='max',  # 'max' si mayor es mejor (AUC/accuracy), 'min' si menor es mejor (loss)
                 max_gap=0.08,  # umbral de gap permitido
                 min_epochs=10,  # no parar antes de X epochs
                 patience=2,  # nº de epochs consecutivos tolerados con gap>max_gap
                 restore_best_weights=True):  # restaurar mejor val
        super().__init__()
        self.monitor_train = monitor_train
        self.monitor_val = monitor_val
        self.mode = mode
        self.max_gap = max_gap
        self.min_epochs = min_epochs
        self.patience = patience
        self.restore_best_weights = restore_best_weights
        self._bad_epochs = 0
        self._best_val = -np.inf if mode == 'max' else np.inf
        self._best_weights = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        tr = logs.get(self.monitor_train)
        va = logs.get(self.monitor_val)
        if tr is None or va is None:
            return  # no están las métricas aún

        # gap definido de forma que "gap grande" = peor generalización
        if self.mode == 'max':  # ej. PR-AUC/accuracy
            gap = tr - va
            is_better = va > self._best_val
        else:  # ej. loss
            gap = va - tr
            is_better = va < self._best_val

        # guarda mejores pesos por métrica de validación
        if is_better:
            self._best_val = va
            if self.restore_best_weights:
                self._best_weights = self.model.get_weights()

        # no evaluar muy pronto
        if (epoch + 1) < self.min_epochs:
            return

        # cuenta epochs “malos” por gap
        if gap > self.max_gap:
            self._bad_epochs += 1
        else:
            self._bad_epochs = 0

        if self._bad_epochs > self.patience:
            if self.restore_best_weights and self._best_weights is not None:
                self.model.set_weights(self._best_weights)
            print(f"\n⛔ Parando por generalization gap: {gap:.4f} > {self.max_gap} "
                  f"(patience {self.patience}) en epoch {epoch + 1}. Mejor {self.monitor_val} = {self._best_val:.4f}")
            self.model.stop_training = True


@register_keras_serializable(package='mimo_old')
class HybridFocalRankingLoss(tf.keras.losses.Loss):
    """
    Hybrid loss = (1 - w) * focal + w * ListNet_ranking

    Definida a nivel de módulo (no dentro de compile_model) para que Keras
    pueda localizarla correctamente al deserializar modelos guardados en disco.

    ListNet:
      P_pred = softmax(scores)
      P_true = softmax(labels * scale)  # scale=10 para separar 0/1
      loss   = -sum(P_true * log(P_pred))

    Solo activa el ranking term cuando el batch tiene ambas clases
    (evita NaN con batches homogéneos).
    """
    def __init__(self, alpha, gamma, ranking_weight,
                 ranking_scale=10.0,
                 reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                 name='hybrid_focal_ranking'):
        super().__init__(reduction=reduction, name=name)
        self.alpha          = float(alpha)
        self.gamma          = float(gamma)
        self.ranking_weight = float(ranking_weight)
        self.ranking_scale  = float(ranking_scale)
        self._focal = ClippedBinaryFocalCrossentropy(
            alpha=alpha, gamma=gamma, from_logits=False
        )

    def call(self, y_true, y_pred):
        focal = self._focal(y_true, y_pred)

        y_true_f = tf.cast(tf.reshape(y_true, [-1]), tf.float32)
        y_pred_f = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

        has_pos = tf.reduce_sum(y_true_f) > 0.5
        has_neg = tf.reduce_sum(1.0 - y_true_f) > 0.5

        def _ranking_loss():
            eps = tf.keras.backend.epsilon()
            p_pred = tf.nn.softmax(y_pred_f)
            p_true = tf.nn.softmax(y_true_f * self.ranking_scale)
            return -tf.reduce_sum(p_true * tf.math.log(p_pred + eps))

        ranking = tf.cond(
            has_pos & has_neg,
            _ranking_loss,
            lambda: tf.constant(0.0)
        )

        return (1.0 - self.ranking_weight) * focal + self.ranking_weight * ranking

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            'alpha':          self.alpha,
            'gamma':          self.gamma,
            'ranking_weight': self.ranking_weight,
            'ranking_scale':  self.ranking_scale,
        })
        return cfg


@register_keras_serializable(package="mimo_old")
class ClippedBinaryFocalCrossentropy(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False,
                 reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                 name="clipped_focal"):
        super().__init__(reduction=reduction, name=name)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.from_logits = bool(from_logits)
        self._focal = tf.keras.losses.BinaryFocalCrossentropy(
            alpha=self.alpha,
            gamma=self.gamma,
            from_logits=self.from_logits,
            reduction=tf.keras.losses.Reduction.NONE,
        )

    def call(self, y_true, y_pred):
        eps = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, eps, 1.0 - eps)
        loss = self._focal(y_true, y_pred)
        return tf.reduce_mean(loss)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({
            "alpha": self.alpha,
            "gamma": self.gamma,
            "from_logits": self.from_logits,
        })
        return cfg


@register_keras_serializable(package="mimo_old")
class PinballLoss(tf.keras.losses.Loss):
    """
    Quantile regression loss (pinball / check loss) para regresión cuantílica
    multi-output.

    y_true: shape (batch, 1) o (batch,) — target continuo (forward return).
    y_pred: shape (batch, n_quantiles) — predicciones para cada cuantil.

    pinball_q(e) = max(q*e, (q-1)*e),  donde e = y_true - y_pred

    Promedia sobre cuantiles y batch. Si todos los cuantiles fueran q=0.5
    coincide con MAE (Mean Absolute Error).
    """

    def __init__(self, quantiles=(0.25, 0.50, 0.75),
                 reduction=tf.keras.losses.Reduction.SUM_OVER_BATCH_SIZE,
                 name="pinball"):
        super().__init__(reduction=reduction, name=name)
        self.quantiles = tuple(float(q) for q in quantiles)

    def call(self, y_true, y_pred):
        # y_true puede venir (batch,) o (batch, 1); broadcast a (batch, 1)
        y_true = tf.cast(tf.reshape(y_true, (-1, 1)), y_pred.dtype)
        q = tf.constant(self.quantiles, dtype=y_pred.dtype)
        diff = y_true - y_pred  # (batch, n_q)
        loss = tf.maximum(q * diff, (q - 1.0) * diff)
        return tf.reduce_mean(loss)

    def get_config(self):
        cfg = super().get_config()
        cfg.update({"quantiles": list(self.quantiles)})
        return cfg


@register_keras_serializable(package="mimo_old")
class QuantileMAE(tf.keras.metrics.Metric):
    """MAE sobre uno de los cuantiles predichos (típicamente q50)."""

    def __init__(self, quantile_idx: int = 1, name: str = "mae_q50", **kwargs):
        super().__init__(name=name, **kwargs)
        self.quantile_idx = int(quantile_idx)
        self.total = self.add_weight(name="total", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, (-1,)), y_pred.dtype)
        y_pred_q = y_pred[:, self.quantile_idx]
        ae = tf.abs(y_true - y_pred_q)
        self.total.assign_add(tf.reduce_sum(ae))
        self.count.assign_add(tf.cast(tf.size(ae), self.total.dtype))

    def result(self):
        return tf.math.divide_no_nan(self.total, self.count)

    def reset_state(self):
        self.total.assign(0.0)
        self.count.assign(0.0)


@register_keras_serializable(package="mimo_old")
class QuantileCoverage(tf.keras.metrics.Metric):
    """
    Cobertura empírica del intervalo [q_low, q_high]. Para q=(0.25,0.75) la
    cobertura ideal es 0.50; valores menores indican intervalos demasiado
    estrechos (overconfidence), mayores indican demasiado anchos.
    """

    def __init__(self, low_idx: int = 0, high_idx: int = 2,
                 name: str = "coverage", **kwargs):
        super().__init__(name=name, **kwargs)
        self.low_idx = int(low_idx)
        self.high_idx = int(high_idx)
        self.inside = self.add_weight(name="inside", initializer="zeros")
        self.count = self.add_weight(name="count", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(tf.reshape(y_true, (-1,)), y_pred.dtype)
        lo = y_pred[:, self.low_idx]
        hi = y_pred[:, self.high_idx]
        inside = tf.cast((y_true >= lo) & (y_true <= hi), self.inside.dtype)
        self.inside.assign_add(tf.reduce_sum(inside))
        self.count.assign_add(tf.cast(tf.size(inside), self.inside.dtype))

    def result(self):
        return tf.math.divide_no_nan(self.inside, self.count)

    def reset_state(self):
        self.inside.assign(0.0)
        self.count.assign(0.0)


@register_keras_serializable(package="mimo_old")
class TripleClassTPAUC(tf.keras.metrics.AUC):
    """
    AUC sobre P(TP) = softmax[..., 2] vs binarización (clase==2) para
    target_type='triple_class'. Permite trackear discriminación TP vs no-TP
    durante el entrenamiento 3-class.
    """

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true_int = tf.cast(tf.reshape(y_true, (-1,)), tf.int32)
        y_true_tp = tf.cast(tf.equal(y_true_int, 2), tf.float32)
        p_tp = y_pred[:, 2]
        return super().update_state(y_true_tp, p_tp, sample_weight=sample_weight)


# Side-effect import al final del módulo: forzar la carga de model_alternatives
# para que el decorator @register_keras_serializable de sus custom layers
# (WeightedSumPooling1D del TCN, etc.) ejecute y registre las clases.
# Esto garantiza que TradingModel.load() pueda deserializar modelos TCN sin
# pasar custom_objects manualmente. Se hace al final para evitar cualquier
# orden de carga conflictivo dentro del propio model_builder.
from mimo.models import model_alternatives as _register_custom_layers  # noqa: F401, E402