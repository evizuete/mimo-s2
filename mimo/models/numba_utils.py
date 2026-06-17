"""
Utilidades optimizadas con Numba para acelerar operaciones críticas.
Versión completa con todas las funciones necesarias.
"""
import numpy as np
from numba import jit, prange

# Configuración global de Numba
NUMBA_CACHE = True
NUMBA_PARALLEL = True
NUMBA_FASTMATH = True

# ============================================================================
# ROLLING STATISTICS
# ============================================================================

@jit(nopython=True, cache=NUMBA_CACHE)
def fast_rolling_mean(arr, window):
    """Rolling mean optimizado con cumsum."""
    n = len(arr)
    result = np.empty(n)
    result[:window - 1] = np.nan

    cumsum = np.empty(n + 1)
    cumsum[0] = 0
    for i in range(n):
        cumsum[i + 1] = cumsum[i] + arr[i]

    for i in range(window - 1, n):
        result[i] = (cumsum[i + 1] - cumsum[i - window + 1]) / window

    return result


@jit(nopython=True, cache=NUMBA_CACHE)
def fast_rolling_std(arr, window):
    """Rolling std optimizado."""
    n = len(arr)
    result = np.empty(n)
    result[:window - 1] = np.nan

    for i in range(window - 1, n):
        window_data = arr[i - window + 1:i + 1]
        result[i] = np.std(window_data)

    return result


@jit(nopython=True, cache=NUMBA_CACHE)
def fast_rolling_max(arr, window):
    """Rolling max optimizado."""
    n = len(arr)
    result = np.empty(n)
    result[:window - 1] = np.nan

    for i in range(window - 1, n):
        result[i] = np.max(arr[i - window + 1:i + 1])

    return result


@jit(nopython=True, cache=NUMBA_CACHE)
def fast_rolling_min(arr, window):
    """Rolling min optimizado."""
    n = len(arr)
    result = np.empty(n)
    result[:window - 1] = np.nan

    for i in range(window - 1, n):
        result[i] = np.min(arr[i - window + 1:i + 1])

    return result


@jit(nopython=True, cache=NUMBA_CACHE)
def fast_ema(arr, span):
    """Exponential Moving Average optimizado."""
    n = len(arr)
    result = np.empty(n)
    alpha = 2.0 / (span + 1.0)

    result[0] = arr[0]
    for i in range(1, n):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]

    return result


@jit(nopython=True, cache=NUMBA_CACHE)
def fast_rolling_quantile(arr, window, q):
    """Rolling quantile optimizado."""
    n = len(arr)
    result = np.empty(n)
    result[:window - 1] = np.nan

    for i in range(window - 1, n):
        window_data = arr[i - window + 1:i + 1]
        sorted_data = np.sort(window_data)
        idx = int(q * (len(sorted_data) - 1))
        result[i] = sorted_data[idx]

    return result


# ============================================================================
# TRIPLE BARRIER LABELING (OPTIMIZACIÓN CRÍTICA #1)
# ============================================================================

@jit(nopython=True, cache=NUMBA_CACHE)
def triple_barrier_fixed_numba(close, high, low, atr, horizon, tp_mult, sl_mult, side_is_long):
    """
    Triple barrier labeling optimizado con Numba.
    Respeta el orden temporal: label=1 solo si el TP se toca ANTES que el SL.

    Args:
        close, high, low: arrays de precios
        atr: Average True Range
        horizon: horizonte temporal
        tp_mult: multiplicador de TP en ATRs
        sl_mult: multiplicador de SL en ATRs
        side_is_long: True para long, False para short

    Returns:
        signal: array de señales (1 si TP antes que SL, 0 si SL primero o timeout)
    """
    n = len(close)
    signal = np.zeros(n, dtype=np.int8)

    for i in range(n - horizon):
        entry   = close[i]
        atr_val = atr[i]

        if np.isnan(atr_val) or atr_val <= 0:
            continue

        if side_is_long:
            tp_level = entry + tp_mult * atr_val
            sl_level = entry - sl_mult * atr_val
        else:
            tp_level = entry - tp_mult * atr_val
            sl_level = entry + sl_mult * atr_val

        hit_tp = -1
        hit_sl = -1

        for k in range(1, horizon + 1):
            j = i + k
            if j >= n:
                break

            if side_is_long:
                if hit_tp == -1 and high[j] >= tp_level:
                    hit_tp = k
                if hit_sl == -1 and low[j] <= sl_level:
                    hit_sl = k
            else:
                if hit_tp == -1 and low[j] <= tp_level:
                    hit_tp = k
                if hit_sl == -1 and high[j] >= sl_level:
                    hit_sl = k

            # Parar en cuanto ambos están determinados
            if hit_tp != -1 and hit_sl != -1:
                break

        # label=1 solo si TP se tocó antes que SL (o SL nunca se tocó)
        if hit_tp != -1 and (hit_sl == -1 or hit_tp <= hit_sl):
            signal[i] = 1

    return signal


@jit(nopython=True, cache=NUMBA_CACHE)
def triple_barrier_3class_numba(close, high, low, atr, horizon, tp_mult, sl_mult, side_is_long):
    """
    Triple barrier 3-class labeling. Devuelve {0=SL, 1=TIMEOUT, 2=TP}.

    Filas con horizonte incompleto o ATR inválido quedan como 1 (TIMEOUT)
    por convención — el pipeline filtrará via dropna donde aplique.
    """
    n = len(close)
    label = np.ones(n, dtype=np.int8)  # default TIMEOUT (1)

    for i in range(n - horizon):
        entry   = close[i]
        atr_val = atr[i]

        if np.isnan(atr_val) or atr_val <= 0:
            continue

        if side_is_long:
            tp_level = entry + tp_mult * atr_val
            sl_level = entry - sl_mult * atr_val
        else:
            tp_level = entry - tp_mult * atr_val
            sl_level = entry + sl_mult * atr_val

        hit_tp = -1
        hit_sl = -1

        for k in range(1, horizon + 1):
            j = i + k
            if j >= n:
                break

            if side_is_long:
                if hit_tp == -1 and high[j] >= tp_level:
                    hit_tp = k
                if hit_sl == -1 and low[j] <= sl_level:
                    hit_sl = k
            else:
                if hit_tp == -1 and low[j] <= tp_level:
                    hit_tp = k
                if hit_sl == -1 and high[j] >= sl_level:
                    hit_sl = k

            if hit_tp != -1 and hit_sl != -1:
                break

        if hit_tp != -1 and (hit_sl == -1 or hit_tp <= hit_sl):
            label[i] = 2  # TP first
        elif hit_sl != -1:
            label[i] = 0  # SL first
        # else: queda como 1 (TIMEOUT)

    return label


@jit(nopython=True, cache=NUMBA_CACHE)
def triple_barrier_adaptive_numba(close, high, low, atr, horizon, side_is_long):
    """
    Triple barrier adaptativo - calcula potencial sin umbrales fijos.

    Returns:
        potential: potencial de la operación en ATRs
    """
    n = len(close)
    potential = np.empty(n)
    potential[:] = np.nan

    for i in range(n - horizon):
        entry = close[i]
        atr_val = atr[i]

        if atr_val < 1e-10:
            continue

        if side_is_long:
            # Buscar máximo futuro
            future_high = np.max(high[i + 1:i + horizon + 1])
            potential[i] = (future_high - entry) / atr_val
        else:
            # Buscar mínimo futuro
            future_low = np.min(low[i + 1:i + horizon + 1])
            potential[i] = (entry - future_low) / atr_val

    return potential


# ============================================================================
# PATTERN DETECTION
# ============================================================================

@jit(nopython=True, cache=NUMBA_CACHE)
def consecutive_runs_numba(arr):
    """
    Cuenta rachas consecutivas (OPTIMIZACIÓN CRÍTICA #2).

    Ejemplo: [1,1,1,2,2,3,3,3,3,1] -> [3,3,3,2,2,4,4,4,4,1]
    """
    n = len(arr)
    runs = np.empty(n, dtype=np.int32)
    current_run = 1

    for i in range(1, n):
        if arr[i] == arr[i - 1]:
            current_run += 1
        else:
            runs[i - 1] = current_run
            current_run = 1

    runs[n - 1] = current_run

    # Propagar hacia atrás
    for i in range(n - 2, -1, -1):
        if arr[i] == arr[i + 1]:
            runs[i] = runs[i + 1]

    return runs


@jit(nopython=True, cache=NUMBA_CACHE)
def autocorr_numba(arr, lag):
    """
    Autocorrelación con lag específico.
    """
    n = len(arr)
    if n <= lag:
        return np.nan

    mean = np.mean(arr)
    arr_centered = arr - mean

    c0 = np.sum(arr_centered ** 2)
    if c0 == 0:
        return np.nan

    c_lag = np.sum(arr_centered[:n - lag] * arr_centered[lag:])

    return c_lag / c0


@jit(nopython=True, cache=NUMBA_CACHE)
def rolling_autocorr_numba(arr, window, lag):
    """
    Rolling autocorrelation (OPTIMIZACIÓN CRÍTICA #3).
    """
    n = len(arr)
    result = np.empty(n)
    result[:window - 1] = np.nan

    for i in range(window - 1, n):
        window_data = arr[i - window + 1:i + 1]
        result[i] = autocorr_numba(window_data, lag)

    return result


# ============================================================================
# INDICATORS
# ============================================================================

@jit(nopython=True, cache=NUMBA_CACHE)
def rsi_numba(prices, period=14):
    """
    RSI optimizado con Numba.
    """
    n = len(prices)
    rsi = np.empty(n)
    rsi[:period] = np.nan

    # Calcular cambios
    deltas = np.diff(prices)

    # Separar ganancias y pérdidas
    gains = np.where(deltas > 0, deltas, 0.0)
    losses = np.where(deltas < 0, -deltas, 0.0)

    # Primera avg gain/loss
    avg_gain = np.mean(gains[:period])
    avg_loss = np.mean(losses[:period])

    # RSI inicial
    if avg_loss == 0:
        rsi[period] = 100.0
    else:
        rs = avg_gain / avg_loss
        rsi[period] = 100.0 - (100.0 / (1.0 + rs))

    # Calcular resto
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period

        if avg_loss == 0:
            rsi[i] = 100.0
        else:
            rs = avg_gain / avg_loss
            rsi[i] = 100.0 - (100.0 / (1.0 + rs))

    return rsi


@jit(nopython=True, cache=NUMBA_CACHE)
def atr_numba(high, low, close, period=14):
    """
    Average True Range optimizado.
    """
    n = len(high)
    tr = np.empty(n)
    atr = np.empty(n)

    # True Range
    tr[0] = high[0] - low[0]
    for i in range(1, n):
        hl = high[i] - low[i]
        hc = abs(high[i] - close[i - 1])
        lc = abs(low[i] - close[i - 1])
        tr[i] = max(hl, hc, lc)

    # ATR
    atr[:period] = np.nan
    atr[period] = np.mean(tr[:period + 1])

    alpha = 1.0 / period
    for i in range(period + 1, n):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]

    return atr


# ============================================================================
# BACKTESTING - EXIT CHECKS (OPTIMIZACIÓN CRÍTICA #4)
# ============================================================================

@jit(nopython=True, cache=NUMBA_CACHE)
def check_exit_single_numba(entry_price, current_price, side_is_long,
                            stop_loss, take_profit, trailing_stop,
                            max_ret, bars_held, max_holding):
    """
    Check exit conditions para una posición individual.

    Args:
        entry_price: precio de entrada
        current_price: precio actual
        side_is_long: 1 para long, 0 para short
        stop_loss: stop loss (fracción, ej. 0.02 = 2%)
        take_profit: take profit (fracción)
        trailing_stop: trailing stop (fracción)
        max_ret: máximo retorno alcanzado
        bars_held: barras en posición
        max_holding: máximo período de tenencia

    Returns:
        should_exit: bool
        exit_type: 0=no exit, 1=stop_loss, 2=take_profit, 3=trailing, 4=timeout
        ret: retorno actual
    """
    # Calcular retorno
    if side_is_long:
        ret = (current_price - entry_price) / entry_price
    else:
        ret = (entry_price - current_price) / entry_price

    # Check stop loss
    if ret <= -stop_loss:
        return True, 1, ret

    # Check take profit
    if ret >= take_profit:
        return True, 2, ret

    # Check trailing stop
    if trailing_stop > 0 and ret < max_ret - trailing_stop:
        return True, 3, ret

    # Check timeout
    if bars_held >= max_holding:
        return True, 4, ret

    return False, 0, ret


@jit(nopython=True, cache=NUMBA_CACHE, parallel=True)
def batch_check_exits_numba(entry_prices, current_prices, sides_long,
                            stop_losses, take_profits, trailing_stops,
                            max_rets, bars_held, max_holdings):
    """
    Batch exit checking en paralelo para múltiples posiciones.
    """
    n = len(entry_prices)
    should_exits = np.empty(n, dtype=np.bool_)
    exit_types = np.empty(n, dtype=np.int8)
    returns = np.empty(n, dtype=np.float64)

    for i in prange(n):
        should_exits[i], exit_types[i], returns[i] = check_exit_single_numba(
            entry_prices[i], current_prices[i], sides_long[i],
            stop_losses[i], take_profits[i], trailing_stops[i],
            max_rets[i], bars_held[i], max_holdings[i]
        )

    return should_exits, exit_types, returns


# ============================================================================
# UTILITIES
# ============================================================================

@jit(nopython=True, cache=NUMBA_CACHE)
def normalize_array(arr):
    """Normalizar array (z-score)."""
    mean = np.mean(arr)
    std = np.std(arr)
    if std == 0:
        return np.zeros_like(arr)
    return (arr - mean) / std


@jit(nopython=True, cache=NUMBA_CACHE)
def winsorize_numba(arr, lower_pct=0.01, upper_pct=0.99):
    """
    Winsorize array (clip outliers).
    """
    sorted_arr = np.sort(arr[~np.isnan(arr)])
    n = len(sorted_arr)

    if n == 0:
        return arr

    lower_idx = int(n * lower_pct)
    upper_idx = int(n * upper_pct)

    lower_val = sorted_arr[lower_idx]
    upper_val = sorted_arr[upper_idx]

    return np.clip(arr, lower_val, upper_val)


# ============================================================================
# BATCH FEATURE CALCULATION
# ============================================================================

@jit(nopython=True, cache=NUMBA_CACHE, parallel=True)
def batch_rolling_stats(close, windows):
    """
    Calcular múltiples rolling stats en paralelo.

    Args:
        close: array de precios
        windows: lista de ventanas

    Returns:
        sma: matriz [n_samples, n_windows]
        std: matriz [n_samples, n_windows]
    """
    n = len(close)
    n_windows = len(windows)

    sma = np.empty((n, n_windows))
    std = np.empty((n, n_windows))

    for w_idx in prange(n_windows):
        window = windows[w_idx]

        for i in range(n):
            if i < window - 1:
                sma[i, w_idx] = np.nan
                std[i, w_idx] = np.nan
            else:
                window_data = close[i - window + 1:i + 1]
                sma[i, w_idx] = np.mean(window_data)
                std[i, w_idx] = np.std(window_data)

    return sma, std


# ============================================================================
# TRADING SIMULATOR - EXIT CHECKS OPTIMIZADOS
# ============================================================================

@jit(nopython=True, cache=NUMBA_CACHE)
def check_exit_simple_numba(side_is_long, high, low, tp, sl):
    """
    Chequeo simple de SL/TP por mecha (versión de _check_exit_in_bar).

    Args:
        side_is_long: 1 para long, 0 para short
        high, low: mecha de la barra
        tp, sl: niveles de take profit y stop loss

    Returns:
        exit_type: 0=no exit, 1=SL, 2=TP, 3=SL and TP same bar
        exit_price: precio de salida
    """
    if side_is_long == 1:  # long
        hit_tp = high >= tp
        hit_sl = low <= sl

        if hit_tp and hit_sl:
            return 3, sl  # "SL and TP on same bar"
        if hit_sl:
            return 1, sl  # "SL"
        if hit_tp:
            return 2, tp  # "TP"
    else:  # short
        hit_tp = low <= tp
        hit_sl = high >= sl

        if hit_tp and hit_sl:
            return 3, sl  # "SL and TP on same bar"
        if hit_sl:
            return 1, sl  # "SL"
        if hit_tp:
            return 2, tp  # "TP"

    return 0, 0.0  # No exit


@jit(nopython=True, cache=NUMBA_CACHE)
def check_exit_advanced_numba(side_is_long, o, h, l, c, tp, sl, atr,
                              tp_mode_is_wick, exit_mode_is_hard,
                              firewall_mult, confirm_count, confirm_bars,
                              entry_price, last_sweep_was_sl):
    """
    Chequeo avanzado de exit (versión de _check_exit_in_bar_v2).

    Args:
        side_is_long: 1 para long, 0 para short
        o, h, l, c: OHLC de la barra
        tp, sl: niveles
        atr: Average True Range
        tp_mode_is_wick: 1 para wick, 0 para close
        exit_mode_is_hard: 1 para hard, 0 para close_confirm
        firewall_mult: multiplicador de firewall (0 para desactivar)
        confirm_count: contador de confirmaciones
        confirm_bars: barras necesarias para confirmar
        entry_price: precio de entrada
        last_sweep_was_sl: si hubo sweep del SL

    Returns:
        exit_type: 0=no exit, 1=TP, 2=SL, 3=SL_CLOSE_CONFIRM, 4=FIREWALL
        exit_price: precio de salida
        new_confirm_count: nuevo contador de confirmaciones
        swept_sl: si hubo sweep en esta barra
    """
    # 1) TP (configurable por wick o close)
    if side_is_long == 1:  # long
        hit_tp = (h >= tp) if tp_mode_is_wick == 1 else (c >= tp)
        if hit_tp:
            return 1, tp, 0, 0  # TP
    else:  # short
        hit_tp = (l <= tp) if tp_mode_is_wick == 1 else (c <= tp)
        if hit_tp:
            return 1, tp, 0, 0  # TP

    # 2) SL
    swept_sl = 0
    new_confirm = confirm_count

    if exit_mode_is_hard == 1:  # Hard SL
        if side_is_long == 1:
            if l <= sl:
                return 2, sl, 0, 0  # SL
        else:
            if h >= sl:
                return 2, sl, 0, 0  # SL
    else:  # Close confirm
        if side_is_long == 1:
            breached_on_close = (c <= sl)
            swept_intrabar = (l <= sl) and (c > sl)
        else:
            breached_on_close = (c >= sl)
            swept_intrabar = (h >= sl) and (c < sl)

        if swept_intrabar:
            swept_sl = 1

        if breached_on_close:
            new_confirm = confirm_count + 1
        else:
            new_confirm = 0

        if new_confirm >= confirm_bars:
            return 3, c, new_confirm, swept_sl  # SL_CLOSE_CONFIRM

    # 3) Firewall
    if firewall_mult > 0 and atr > 0:
        fw = firewall_mult * atr
        if side_is_long == 1:
            if (entry_price - c) >= fw:
                return 4, c, new_confirm, swept_sl  # FIREWALL
        else:
            if (c - entry_price) >= fw:
                return 4, c, new_confirm, swept_sl  # FIREWALL

    return 0, 0.0, new_confirm, swept_sl  # No exit


@jit(nopython=True, cache=NUMBA_CACHE, parallel=True)
def batch_unrealized_pnl_numba(entry_prices, sides_long, qtys, value_per_lots,
                               bid_price, ask_price):
    """
    Calcular PnL no realizado para múltiples posiciones en paralelo.

    Args:
        entry_prices: array de precios de entrada
        sides_long: array (1=long, 0=short)
        qtys: array de cantidades
        value_per_lots: array de valores por lote
        bid_price, ask_price: precios actuales

    Returns:
        pnl: PnL total
        individual_pnls: array de PnL por posición
    """
    n = len(entry_prices)
    individual_pnls = np.empty(n)

    for i in prange(n):
        if sides_long[i] == 1:  # long
            exit_px = bid_price
            delta = exit_px - entry_prices[i]
        else:  # short
            exit_px = ask_price
            delta = entry_prices[i] - exit_px

        individual_pnls[i] = delta * qtys[i] * value_per_lots[i]

    return np.sum(individual_pnls), individual_pnls


@jit(nopython=True, cache=NUMBA_CACHE, parallel=True)
def batch_check_exits_simple_numba(sides_long, highs, lows, tps, sls):
    """
    Batch check de exits simples para múltiples posiciones.

    Returns:
        exit_types: array de tipos de salida
        exit_prices: array de precios de salida
    """
    n = len(sides_long)
    exit_types = np.empty(n, dtype=np.int8)
    exit_prices = np.empty(n)

    for i in prange(n):
        exit_types[i], exit_prices[i] = check_exit_simple_numba(
            sides_long[i], highs[i], lows[i], tps[i], sls[i]
        )

    return exit_types, exit_prices