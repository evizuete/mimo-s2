"""
model_alternatives.py
═══════════════════════════════════════════════════════════════════════════
Arquitecturas alternativas al CNN-LSTM del proyecto, todas drop-in
compatibles con el pipeline existente:
  · Mismos 4 inputs (seq_short, seq_long, context, time)
  · Mismos output names (signal_long, signal_short) para multitask
  · Mismo bias inicial pattern del original

ARQUITECTURAS DISPONIBLES:
  · 'mlp'         → MLP-tabular puro (sequences colapsadas a stats)
  · 'mlp_flatten' → MLP sobre sequences APLANADAS (conserva historial completo)
  · 'hybrid'      → Conv1D + GAP + MLP fuerte (sin LSTM)
  · 'transformer' → Transformer encoder ligero sobre seq_long + MLP
  · 'tcn'         → Temporal Convolutional Network (dilated convs)

USO desde walkforward script:
  from mimo.models.model_alternatives import build_model_by_arch
  model = build_model_by_arch(
      'mlp',                    # nombre del arch
      shape_short=(24, 30),
      shape_long=(96, 30),
      n_context=20, n_time=8,
      model_config=mc,
      init_bias={'long': -2.3, 'short': -2.3},
  )
  # model es Keras Model con outputs dict {'signal_long', 'signal_short'}
"""

from __future__ import annotations

from typing import Dict, Tuple, Union

import tensorflow as tf
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Model

# Para registrar custom layers serializables (ver WeightedSumPooling1D abajo).
try:
    from tensorflow.keras.saving import register_keras_serializable
except Exception:
    # Fallback para versiones de Keras donde el path es distinto.
    from keras.saving import register_keras_serializable  # type: ignore


# ─── helpers comunes ────────────────────────────────────────────────────


@register_keras_serializable(package="MimoTCN")
class WeightedSumPooling1D(layers.Layer):
    """Suma sobre el axis temporal de un tensor (batch, T, C) → (batch, C).

    Pensado para usarse al final de un attention pooling: tras
    Multiply([seq, softmax_weights]) el resultado es una secuencia ponderada
    de la que queremos extraer el "weighted sum" (axis=1).

    Reemplaza el patrón anterior `layers.Lambda(lambda t: tf.reduce_sum(t,
    axis=1))` que tenía DOS problemas en Keras 3 al cargar el .keras:
      1) Sin output_shape explícito, la inferencia de shape fallaba.
      2) Aún con output_shape, al EJECUTAR la lambda durante predict el
         body `tf.reduce_sum(...)` lanzaba `NameError: name 'tf' is not
         defined` porque Keras deserializa el lambda code en un namespace
         que NO incluye el módulo `tf` capturado por closure.

    Custom Layer registered se serializa por nombre+package en el .keras
    config, se deserializa sin custom_objects, y no depende de variables
    externas → bulletproof para load_model().
    """

    def call(self, inputs):
        # tf.reduce_sum funciona dentro de Layer.call() porque tf SÍ está
        # disponible cuando este módulo se importa (a diferencia del lambda
        # body que se ejecuta tras serializar a string).
        return tf.reduce_sum(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        # (batch, T, C) → (batch, C)
        return (input_shape[0], input_shape[-1])

    def get_config(self):
        # No state extra → solo nombre. super().get_config() devuelve {name, ...}.
        return super().get_config()

def _make_inputs(shape_short: Tuple[int, int], shape_long: Tuple[int, int],
                 n_context: int, n_time: int):
    """Inputs idénticos al CNN-LSTM v3 original. Garantiza compatibilidad
    con pipeline.create_sequences_by_side."""
    return (
        layers.Input(shape=shape_short, name='seq_short'),
        layers.Input(shape=shape_long,  name='seq_long'),
        layers.Input(shape=(n_context,), name='context'),
        layers.Input(shape=(n_time,),    name='time'),
    )


def _parse_init_bias(init_bias) -> Tuple[float, float]:
    """init_bias puede ser float (mismo para ambas heads) o dict
    {'long': ..., 'short': ...}."""
    if isinstance(init_bias, dict):
        return float(init_bias.get('long', 0.0)), float(init_bias.get('short', 0.0))
    v = float(init_bias)
    return v, v


def _multitask_head(x, bias_long: float, bias_short: float) -> Dict[str, tf.Tensor]:
    """Cabezas binarias separadas con bias inicial. Devuelve dict con
    nombres 'signal_long' y 'signal_short' (compatible con compile/loss
    dict-keyed del CNN-LSTM original)."""
    logit_l = layers.Dense(
        1, name='logit_long',
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer=tf.keras.initializers.Constant(bias_long),
    )(x)
    logit_s = layers.Dense(
        1, name='logit_short',
        kernel_initializer=tf.keras.initializers.RandomNormal(stddev=0.01),
        bias_initializer=tf.keras.initializers.Constant(bias_short),
    )(x)
    p_long  = layers.Activation('sigmoid', name='signal_long')(logit_l)
    p_short = layers.Activation('sigmoid', name='signal_short')(logit_s)
    return {'signal_long': p_long, 'signal_short': p_short}


class PositionalEmbedding(layers.Layer):
    """Positional embedding aprendida para Transformer."""
    def __init__(self, seq_len: int, d_model: int, **kwargs):
        super().__init__(**kwargs)
        self.seq_len = int(seq_len)
        self.d_model = int(d_model)
        self.pos_emb = layers.Embedding(self.seq_len, self.d_model, name='pos_emb_lookup')

    def call(self, x):
        positions = tf.range(self.seq_len, dtype=tf.int32)
        emb = self.pos_emb(positions)  # (seq_len, d_model)
        return x + emb[tf.newaxis, :, :]

    def get_config(self):
        c = super().get_config()
        c.update({'seq_len': self.seq_len, 'd_model': self.d_model})
        return c


class SequenceStdPool(layers.Layer):
    """Calcula std a lo largo del tiempo: shape (B, T, F) → (B, F)."""
    def call(self, x):
        return tf.math.reduce_std(x, axis=1)


class SequenceLastPool(layers.Layer):
    """Toma el último timestep: shape (B, T, F) → (B, F)."""
    def call(self, x):
        return x[:, -1, :]


# ─── A: MLP-tabular puro ────────────────────────────────────────────────

def build_mlp_tabular(
    shape_short, shape_long, n_context, n_time, model_config, init_bias=0.0,
):
    """Sequences → stats (mean/std/last) concatenadas con context+time → MLP.
    Sin LSTM, sin Conv. La hipótesis es que la señal es esencialmente tabular
    (validado por el GBM superando al CNN-LSTM en este problema)."""
    inp_s, inp_l, inp_ctx, inp_t = _make_inputs(
        shape_short, shape_long, n_context, n_time)

    # Stats por secuencia (cero parámetros añadidos)
    def _stats(seq, name):
        m = layers.GlobalAveragePooling1D(name=f'{name}_mean')(seq)
        s = SequenceStdPool(name=f'{name}_std')(seq)
        last = SequenceLastPool(name=f'{name}_last')(seq)
        return layers.Concatenate(name=f'{name}_stats')([m, s, last])

    feat_short = _stats(inp_s, 'short')
    feat_long  = _stats(inp_l, 'long')
    all_feat   = layers.Concatenate(name='all_features')([
        feat_short, feat_long, inp_ctx, inp_t,
    ])

    l2 = float(getattr(model_config, 'l2_reg', 1e-5))
    drop_d = float(getattr(model_config, 'dropout_dense', 0.2))
    units = int(getattr(model_config, 'head_units', 128))

    x = layers.Dense(units, activation='gelu',
                     kernel_regularizer=regularizers.l2(l2),
                     name='mlp_dense1')(all_feat)
    x = layers.LayerNormalization(name='mlp_ln1')(x)
    x = layers.Dropout(drop_d * 1.5, name='mlp_drop1')(x)
    x = layers.Dense(units // 2, activation='gelu',
                     kernel_regularizer=regularizers.l2(l2),
                     name='mlp_dense2')(x)
    x = layers.Dropout(drop_d, name='mlp_drop2')(x)
    x = layers.Dense(max(units // 4, 16), activation='gelu',
                     name='mlp_dense3')(x)

    bias_l, bias_s = _parse_init_bias(init_bias)
    outputs = _multitask_head(x, bias_l, bias_s)

    return Model(
        inputs=[inp_s, inp_l, inp_ctx, inp_t],
        outputs=outputs, name='mlp_tabular',
    )


# ─── A2: MLP-flatten — aplana secuencias en lugar de colapsar a stats ───

def build_mlp_flatten(
    shape_short, shape_long, n_context, n_time, model_config, init_bias=0.0,
):
    """Variante de build_mlp_tabular que CONSERVA la información temporal:
    aplana seq_short (T_s, F_s) → (T_s*F_s) y seq_long análogamente, en lugar
    de colapsar a mean/std/last. El MLP recibe el historial completo como
    vector tabular plano y puede aprender qué timesteps usar.

    Motivación: build_mlp_tabular destruye la dinámica temporal al reducir
    cada secuencia a 3 stats; el GBM en cambio ve features tabulares planas
    (último snapshot) y consigue +103R Sharpe 1.15 walkforward. Esta variante
    da al MLP TODA la historia, no sólo stats agregadas ni el último valor.

    Coste: dimensión del input aumenta de (3*F_s + 3*F_l) a (T_s*F_s + T_l*F_l)
    — para shapes típicas (24,30)+(96,30) pasa de 180 a 3600 features. La
    primera Dense con head_units=128 ⇒ ~460k parámetros, regularizable con
    dropout/L2 estándar del config.
    """
    inp_s, inp_l, inp_ctx, inp_t = _make_inputs(
        shape_short, shape_long, n_context, n_time)

    flat_short = layers.Flatten(name='short_flat')(inp_s)
    flat_long  = layers.Flatten(name='long_flat')(inp_l)
    all_feat   = layers.Concatenate(name='all_features')([
        flat_short, flat_long, inp_ctx, inp_t,
    ])

    l2 = float(getattr(model_config, 'l2_reg', 1e-5))
    drop_d = float(getattr(model_config, 'dropout_dense', 0.2))
    units = int(getattr(model_config, 'head_units', 128))
    act = str(getattr(model_config, 'activation', 'gelu')).lower()
    # swish viene como nombre 'swish' (tf.keras.activations.swish), válido en
    # Keras 2.4+. Mapeo defensivo por si alguien pasa 'silu' (alias PyTorch).
    if act == 'silu':
        act = 'swish'

    x = layers.Dense(units, activation=act,
                     kernel_regularizer=regularizers.l2(l2),
                     name='mlp_dense1')(all_feat)
    x = layers.LayerNormalization(name='mlp_ln1')(x)
    x = layers.Dropout(drop_d * 1.5, name='mlp_drop1')(x)
    x = layers.Dense(units // 2, activation=act,
                     kernel_regularizer=regularizers.l2(l2),
                     name='mlp_dense2')(x)
    x = layers.Dropout(drop_d, name='mlp_drop2')(x)
    x = layers.Dense(max(units // 4, 16), activation=act,
                     name='mlp_dense3')(x)

    bias_l, bias_s = _parse_init_bias(init_bias)
    outputs = _multitask_head(x, bias_l, bias_s)

    return Model(
        inputs=[inp_s, inp_l, inp_ctx, inp_t],
        outputs=outputs, name='mlp_flatten',
    )


# ─── B: CNN-MLP híbrido reducido ────────────────────────────────────────

def build_hybrid_cnn_mlp(
    shape_short, shape_long, n_context, n_time, model_config, init_bias=0.0,
):
    """Conv1D pequeño + GAP por sequence + MLP fuerte por context+time."""
    inp_s, inp_l, inp_ctx, inp_t = _make_inputs(
        shape_short, shape_long, n_context, n_time)

    l2 = float(getattr(model_config, 'l2_reg', 1e-5))
    drop_seq = float(getattr(model_config, 'dropout_seq', 0.05))
    drop_d   = float(getattr(model_config, 'dropout_dense', 0.2))
    conv_f   = max(int(getattr(model_config, 'conv1d_filters', 32)) // 2, 16)

    def _conv_branch(x, name):
        x = layers.Conv1D(conv_f, 5, activation='gelu', padding='causal',
                          kernel_regularizer=regularizers.l2(l2),
                          name=f'{name}_conv1')(x)
        x = layers.Dropout(drop_seq, name=f'{name}_drop1')(x)
        x = layers.Conv1D(conv_f, 3, activation='gelu', padding='causal',
                          kernel_regularizer=regularizers.l2(l2),
                          name=f'{name}_conv2')(x)
        return layers.GlobalAveragePooling1D(name=f'{name}_gap')(x)

    s_short = _conv_branch(inp_s, 'short')
    s_long  = _conv_branch(inp_l, 'long')

    units_ctx = int(getattr(model_config, 'context_units', 64))
    c = layers.Concatenate(name='ctx_time')([inp_ctx, inp_t])
    c = layers.Dense(units_ctx, activation='gelu',
                     kernel_regularizer=regularizers.l2(l2),
                     name='ctx_dense1')(c)
    c = layers.LayerNormalization(name='ctx_ln')(c)
    c = layers.Dropout(drop_d * 1.5, name='ctx_drop1')(c)
    c = layers.Dense(units_ctx // 2, activation='gelu', name='ctx_dense2')(c)
    c = layers.Dropout(drop_d, name='ctx_drop2')(c)

    head_units = int(getattr(model_config, 'head_units', 64))
    x = layers.Concatenate(name='fusion')([s_short, s_long, c])
    x = layers.Dense(head_units, activation='gelu',
                     kernel_regularizer=regularizers.l2(l2),
                     name='head_dense')(x)
    x = layers.Dropout(drop_d, name='head_drop')(x)

    bias_l, bias_s = _parse_init_bias(init_bias)
    outputs = _multitask_head(x, bias_l, bias_s)
    return Model(
        inputs=[inp_s, inp_l, inp_ctx, inp_t],
        outputs=outputs, name='hybrid_cnn_mlp',
    )


# ─── C: Transformer encoder ligero ──────────────────────────────────────

def build_transformer_lite(
    shape_short, shape_long, n_context, n_time, model_config, init_bias=0.0,
):
    """Transformer encoder (2 bloques, 4 heads) sobre seq_long + Conv1D sobre
    seq_short + MLP sobre context/time."""
    inp_s, inp_l, inp_ctx, inp_t = _make_inputs(
        shape_short, shape_long, n_context, n_time)

    l2 = float(getattr(model_config, 'l2_reg', 1e-5))
    drop_seq = float(getattr(model_config, 'dropout_seq', 0.1))
    drop_d   = float(getattr(model_config, 'dropout_dense', 0.2))

    seq_long_len = int(shape_long[0])
    n_feat_long  = int(shape_long[1])
    d_model = max(n_feat_long, 32)
    n_heads = 4
    n_blocks = 2

    # Project to d_model si difiere
    if n_feat_long != d_model:
        s = layers.Dense(d_model, name='long_proj')(inp_l)
    else:
        s = inp_l

    # Positional encoding
    s = PositionalEmbedding(seq_long_len, d_model, name='pos_emb')(s)

    # Transformer blocks (pre-norm style)
    for i in range(n_blocks):
        n1 = layers.LayerNormalization(name=f'pre_norm_{i}')(s)
        attn = layers.MultiHeadAttention(
            num_heads=n_heads, key_dim=d_model // n_heads,
            dropout=drop_seq, name=f'mha_{i}',
        )(n1, n1)
        s = layers.Add(name=f'add_attn_{i}')([s, attn])
        n2 = layers.LayerNormalization(name=f'pre_norm_ff_{i}')(s)
        ff = layers.Dense(d_model * 2, activation='gelu',
                          kernel_regularizer=regularizers.l2(l2),
                          name=f'ff1_{i}')(n2)
        ff = layers.Dropout(drop_seq, name=f'ff_drop_{i}')(ff)
        ff = layers.Dense(d_model, name=f'ff2_{i}')(ff)
        s = layers.Add(name=f'add_ff_{i}')([s, ff])

    s_long = layers.GlobalAveragePooling1D(name='long_gap')(s)

    # Sequence short: conv simple (es seq corta, transformer es overkill)
    s_short = layers.Conv1D(16, 3, activation='gelu', padding='causal',
                            kernel_regularizer=regularizers.l2(l2),
                            name='short_conv')(inp_s)
    s_short = layers.GlobalAveragePooling1D(name='short_gap')(s_short)

    # Tabular
    c = layers.Concatenate(name='ctx_time')([inp_ctx, inp_t])
    c = layers.Dense(64, activation='gelu',
                     kernel_regularizer=regularizers.l2(l2),
                     name='ctx_dense')(c)
    c = layers.LayerNormalization(name='ctx_ln')(c)
    c = layers.Dropout(drop_d, name='ctx_drop')(c)

    head_units = int(getattr(model_config, 'head_units', 64))
    x = layers.Concatenate(name='fusion')([s_short, s_long, c])
    x = layers.Dense(head_units, activation='gelu', name='head_dense')(x)
    x = layers.Dropout(drop_d, name='head_drop')(x)

    bias_l, bias_s = _parse_init_bias(init_bias)
    outputs = _multitask_head(x, bias_l, bias_s)
    return Model(
        inputs=[inp_s, inp_l, inp_ctx, inp_t],
        outputs=outputs, name='transformer_lite',
    )


# ─── D: TCN + MLP ────────────────────────────────────────────────────────

def build_tcn_mlp(
    shape_short, shape_long, n_context, n_time, model_config, init_bias=0.0,
):
    """Temporal Convolutional Network (dilated convs con residual blocks)
    sobre ambas secuencias + MLP sobre context+time.

    HPs estructurales expuestos (vía model_config / setattr extras):
      · activation              ∈ {gelu, relu, swish, elu}        default 'gelu'
      · kernel_size             ∈ {2, 3, 5, 7}                    default 3
      · n_tcn_blocks_long       ∈ {2..6} (dilations 2^i)          default 5
      · n_tcn_blocks_short      ∈ {2..4}                          default derivado
      · tcn_filters_short_ratio float (filters_short = ratio*long) default 0.5
      · tcn_pooling             ∈ {gap, gmp, gap_gmp, attention}  default 'gap'
      · conv1d_filters          categórico                         default 32
      · ctx_dense_units         int (capacidad rama context+time)  default 64
      · use_se                  bool (Squeeze-and-Excite por bloque) default False
      · se_ratio                int (reducción canal SE)            default 8
      · attn_units              int (proyección attention pooling)  default 32
      · dropout_seq/dense, l2_reg, head_units                       ya existentes
    """
    inp_s, inp_l, inp_ctx, inp_t = _make_inputs(
        shape_short, shape_long, n_context, n_time)

    l2       = float(getattr(model_config, 'l2_reg', 1e-5))
    drop_seq = float(getattr(model_config, 'dropout_seq', 0.1))
    drop_d   = float(getattr(model_config, 'dropout_dense', 0.2))
    filters  = max(int(getattr(model_config, 'conv1d_filters', 32)), 16)
    act      = str(getattr(model_config, 'activation', 'gelu')).lower()
    ksize    = int(getattr(model_config, 'kernel_size', 3))
    n_long   = int(getattr(model_config, 'n_tcn_blocks_long', 5))
    # n_short: si se expone vía HP (v4), úsalo; si no, deriva como v3.
    _n_short_default = min(3, max(2, n_long - 2))
    n_short  = int(getattr(model_config, 'n_tcn_blocks_short', _n_short_default))
    pooling  = str(getattr(model_config, 'tcn_pooling', 'gap')).lower()
    fs_ratio = float(getattr(model_config, 'tcn_filters_short_ratio', 0.5))
    filters_short = max(int(round(filters * fs_ratio)), 16)
    use_se   = bool(int(getattr(model_config, 'use_se', 0)))
    se_ratio = max(int(getattr(model_config, 'se_ratio', 8)), 2)
    ctx_units = int(getattr(model_config, 'ctx_dense_units', 64))
    attn_units = max(int(getattr(model_config, 'attn_units', 32)), 8)

    def _se_block(x, name):
        """Squeeze-and-Excite por canal: GAP → Dense(C/r,relu) → Dense(C,sigmoid)
        → reescala canales. Coste ~2·C²/r params, muy bajo vs conv blocks."""
        c = int(x.shape[-1])
        r = max(c // se_ratio, 4)
        s = layers.GlobalAveragePooling1D(name=f'{name}_gap')(x)
        s = layers.Dense(r, activation='relu',
                         kernel_regularizer=regularizers.l2(l2),
                         name=f'{name}_d1')(s)
        s = layers.Dense(c, activation='sigmoid',
                         kernel_regularizer=regularizers.l2(l2),
                         name=f'{name}_d2')(s)
        s = layers.Reshape((1, c), name=f'{name}_rs')(s)
        return layers.Multiply(name=f'{name}_mul')([x, s])

    def _tcn_block(x, dilation, filt, name):
        residual = x
        x = layers.Conv1D(filt, ksize, dilation_rate=dilation, padding='causal',
                          activation=act, kernel_regularizer=regularizers.l2(l2),
                          name=f'{name}_c1')(x)
        x = layers.Dropout(drop_seq, name=f'{name}_d1')(x)
        x = layers.Conv1D(filt, ksize, dilation_rate=dilation, padding='causal',
                          activation=act, kernel_regularizer=regularizers.l2(l2),
                          name=f'{name}_c2')(x)
        x = layers.Dropout(drop_seq, name=f'{name}_d2')(x)
        if int(residual.shape[-1]) != filt:
            residual = layers.Conv1D(filt, 1, padding='same',
                                     name=f'{name}_res_proj')(residual)
        x = layers.Add(name=f'{name}_add')([x, residual])
        x = layers.LayerNormalization(name=f'{name}_norm')(x)
        if use_se:
            x = _se_block(x, f'{name}_se')
        return x

    def _attn_pool(seq, name_prefix: str):
        """Attention pooling aprendible: score por timestep → softmax → weighted sum.
        Captura "qué momento de la ventana importa" en lugar de promediar todo.

        Implementación: Dense+softmax para los scores, Multiply para aplicar,
        WeightedSumPooling1D (custom Layer registered) para reducir axis temporal.
        El custom Layer reemplaza un Lambda que daba problemas de serialización
        en Keras 3 (output_shape implícito + closure de `tf` no disponible).
        """
        h = layers.Dense(attn_units, activation='tanh',
                         kernel_regularizer=regularizers.l2(l2),
                         name=f'{name_prefix}_attn_proj')(seq)
        scores = layers.Dense(1, name=f'{name_prefix}_attn_score')(h)
        scores = layers.Reshape((-1,), name=f'{name_prefix}_attn_sq')(scores)
        w = layers.Softmax(axis=1, name=f'{name_prefix}_attn_sm')(scores)
        w = layers.Reshape((-1, 1), name=f'{name_prefix}_attn_rs')(w)
        weighted = layers.Multiply(name=f'{name_prefix}_attn_mul')([seq, w])
        return WeightedSumPooling1D(name=f'{name_prefix}_attn_sum')(weighted)

    def _pool(seq, name_prefix: str):
        if pooling == 'gmp':
            return layers.GlobalMaxPooling1D(name=f'{name_prefix}_gmp')(seq)
        if pooling == 'gap_gmp':
            gap = layers.GlobalAveragePooling1D(name=f'{name_prefix}_gap')(seq)
            gmp = layers.GlobalMaxPooling1D(name=f'{name_prefix}_gmp')(seq)
            return layers.Concatenate(name=f'{name_prefix}_pool')([gap, gmp])
        if pooling == 'attention':
            return _attn_pool(seq, name_prefix)
        return layers.GlobalAveragePooling1D(name=f'{name_prefix}_gap')(seq)

    # TCN seq_long: dilations [1, 2, 4, ..., 2^(n_long-1)]
    s = inp_l
    for i in range(n_long):
        s = _tcn_block(s, 2 ** i, filters, f'tcn_l{i}')
    s_long = _pool(s, 'long')

    # TCN seq_short: dilations [1, 2, 4, ..., 2^(n_short-1)]
    s = inp_s
    for i in range(n_short):
        s = _tcn_block(s, 2 ** i, filters_short, f'tcn_s{i}')
    s_short = _pool(s, 'short')

    c = layers.Concatenate(name='ctx_time')([inp_ctx, inp_t])
    c = layers.Dense(ctx_units, activation=act,
                     kernel_regularizer=regularizers.l2(l2),
                     name='ctx_dense')(c)
    c = layers.LayerNormalization(name='ctx_ln')(c)
    c = layers.Dropout(drop_d, name='ctx_drop')(c)

    head_units = int(getattr(model_config, 'head_units', 64))
    x = layers.Concatenate(name='fusion')([s_short, s_long, c])
    x = layers.Dense(head_units, activation=act, name='head_dense')(x)
    x = layers.Dropout(drop_d, name='head_drop')(x)

    bias_l, bias_s = _parse_init_bias(init_bias)
    outputs = _multitask_head(x, bias_l, bias_s)
    return Model(
        inputs=[inp_s, inp_l, inp_ctx, inp_t],
        outputs=outputs, name='tcn_mlp',
    )


# ─── Registry ────────────────────────────────────────────────────────────

ARCH_REGISTRY = {
    'mlp':         build_mlp_tabular,
    'mlp_flatten': build_mlp_flatten,
    'hybrid':      build_hybrid_cnn_mlp,
    'transformer': build_transformer_lite,
    'tcn':         build_tcn_mlp,
}


def build_model_by_arch(
    arch_name: str,
    shape_short: Tuple[int, int], shape_long: Tuple[int, int],
    n_context: int, n_time: int,
    model_config, init_bias: Union[float, dict] = 0.0,
) -> Model:
    """Factory: dado un nombre de arch, construye y devuelve Keras Model.
    Targeta multitask (output dict signal_long+signal_short).
    Raises si arch no está en ARCH_REGISTRY."""
    if arch_name not in ARCH_REGISTRY:
        raise ValueError(
            f"arch '{arch_name}' no soportada. Opciones: {list(ARCH_REGISTRY)}"
        )
    return ARCH_REGISTRY[arch_name](
        shape_short, shape_long, n_context, n_time, model_config, init_bias,
    )
