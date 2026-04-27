"""
Cálculo de Métricas Históricas para Calibración Adaptativa

Este módulo calcula:
- historical_max: Máximo precio histórico
- historical_min: Mínimo precio histórico
- historical_atr_mean: Promedio histórico de ATR
- regime: Detección de régimen (ranging/trending)
"""
from datetime import datetime, timedelta

import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional

from mimo_old.data_manager import DataManager
from mimo_old.databases import Database


class HistoricalMetricsCalculator:
    """
    Calcula métricas históricas para calibración adaptativa

    Uso:
        calculator = HistoricalMetricsCalculator(lookback_days=365)

        # Con DataFrame
        metrics = calculator.calculate_from_dataframe(df)

        # Con datos actuales
        metrics = calculator.calculate_from_current(
            price=4891.89,
            atr=15.50,
            historical_prices=[...],
            historical_atrs=[...]
        )
    """

    def __init__(
            self,
            lookback_days: int = 365,
            min_lookback_days: int = 30,
            regime_sma_fast: int = 20,
            regime_sma_slow: int = 50,
            regime_atr_period: int = 14
    ):
        """
        Args:
            lookback_days: Días de historia para calcular máx/mín (default: 1 año)
            min_lookback_days: Mínimo de días requeridos (default: 1 mes)
            regime_sma_fast: Período SMA rápida para régimen (default: 20)
            regime_sma_slow: Período SMA lenta para régimen (default: 50)
            regime_atr_period: Período ATR para régimen (default: 14)
        """
        self.lookback_days = lookback_days
        self.min_lookback_days = min_lookback_days
        self.regime_sma_fast = regime_sma_fast
        self.regime_sma_slow = regime_sma_slow
        self.regime_atr_period = regime_atr_period

    # ========================================================================
    # MÉTODO 1: DESDE DATAFRAME (RECOMENDADO)
    # ========================================================================

    def calculate_from_dataframe(
            self,
            df: pd.DataFrame,
            price_col: str = 'close',
            high_col: str = 'high',
            low_col: str = 'low',
            atr_col: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calcula métricas desde un DataFrame histórico

        Args:
            df: DataFrame con datos OHLC
            price_col: Nombre de columna de precio (default: 'close')
            high_col: Nombre de columna de high (default: 'high')
            low_col: Nombre de columna de low (default: 'low')
            atr_col: Nombre de columna de ATR (opcional, se calcula si no existe)

        Returns:
            Dict con métricas calculadas
        """

        last_date = df['time'].max()
        cutoff_date = last_date - timedelta(days=self.lookback_days)
        df_window = df[df['time'] >= cutoff_date].copy()

        # 1. Historical Max/Min
        hist_max = float(df_window[price_col].max())
        hist_min = float(df_window[price_col].min())

        # 2. ATR
        if atr_col and atr_col in df_window.columns:
            atr_values = df_window[atr_col]
        else:
            # Calcular ATR si no existe
            atr_values = self._calculate_atr(
                df_window[high_col],
                df_window[low_col],
                df_window[price_col],
                period=self.regime_atr_period
            )

        hist_atr_mean = float(atr_values.mean())

        # 3. Régimen
        regime = self._detect_regime_from_df(
            df_window,
            price_col=price_col,
            atr_values=atr_values
        )

        # 4. Precio actual
        current_price = float(df_window[price_col].iloc[-1])
        current_atr = float(atr_values.iloc[-1])

        return {
            'price': current_price,
            'hist_max': hist_max,
            'hist_min': hist_min,
            'atr': current_atr,
            'hist_atr_avg': hist_atr_mean,
            'macro_regime': regime,

            # Metadata adicional
            'pct_of_max': (current_price / hist_max) * 100,
            'pct_of_min': (current_price / hist_min) * 100,
            'atr_ratio': current_atr / hist_atr_mean,
            'lookback_days': len(df_window)
        }

    # ========================================================================
    # MÉTODO 2: DESDE LISTAS/ARRAYS (STREAMING)
    # ========================================================================

    def calculate_from_arrays(
            self,
            prices: np.ndarray,
            highs: Optional[np.ndarray] = None,
            lows: Optional[np.ndarray] = None,
            atrs: Optional[np.ndarray] = None
    ) -> Dict[str, Any]:
        """
        Calcula métricas desde arrays/listas

        Args:
            prices: Array de precios (close)
            highs: Array de highs (opcional, para calcular ATR)
            lows: Array de lows (opcional, para calcular ATR)
            atrs: Array de ATRs (opcional, se calcula si no se provee)

        Returns:
            Dict con métricas calculadas
        """

        prices = np.array(prices)

        if len(prices) < self.min_lookback_days:
            raise ValueError(
                f"Array tiene {len(prices)} elementos, "
                f"necesita al menos {self.min_lookback_days}"
            )

        # Limitar a lookback
        prices_window = prices[-self.lookback_days:]

        # 1. Historical Max/Min
        hist_max = float(np.max(prices_window))
        hist_min = float(np.min(prices_window))

        # 2. ATR
        if atrs is not None:
            atrs = np.array(atrs)
            atrs_window = atrs[-self.lookback_days:]
            hist_atr_mean = float(np.mean(atrs_window))
            current_atr = float(atrs_window[-1])
        elif highs is not None and lows is not None:
            highs = np.array(highs)
            lows = np.array(lows)

            highs_window = highs[-self.lookback_days:]
            lows_window = lows[-self.lookback_days:]

            atr_values = self._calculate_atr_from_arrays(
                highs_window,
                lows_window,
                prices_window,
                period=self.regime_atr_period
            )

            hist_atr_mean = float(np.mean(atr_values))
            current_atr = float(atr_values[-1])
        else:
            # Usar volatilidad de precios como proxy
            returns = np.diff(prices_window) / prices_window[:-1]
            hist_atr_mean = float(np.std(returns) * prices_window[-1])
            current_atr = hist_atr_mean

        # 3. Régimen
        regime = self._detect_regime_from_arrays(
            prices_window,
            current_atr,
            hist_atr_mean
        )

        # 4. Precio actual
        current_price = float(prices_window[-1])

        return {
            'price': current_price,
            'hist_max': hist_max,
            'hist_min': hist_min,
            'atr': current_atr,
            'hist_atr_avg': hist_atr_mean,
            'macro_regime': regime,

            # Metadata
            'pct_of_max': (current_price / hist_max) * 100,
            'pct_of_min': (current_price / hist_min) * 100,
            'atr_ratio': current_atr / hist_atr_mean,
            'lookback_days': len(prices_window)
        }

    # ========================================================================
    # MÉTODO 3: ACTUALIZACIÓN INCREMENTAL (MÁS EFICIENTE)
    # ========================================================================

    def create_rolling_calculator(self) -> 'RollingMetricsCalculator':
        """
        Crea un calculador incremental que mantiene estado

        Returns:
            RollingMetricsCalculator para updates incrementales
        """
        return RollingMetricsCalculator(
            lookback_days=self.lookback_days,
            regime_sma_fast=self.regime_sma_fast,
            regime_sma_slow=self.regime_sma_slow
        )

    # ========================================================================
    # FUNCIONES AUXILIARES
    # ========================================================================

    def _calculate_atr(
            self,
            high: pd.Series,
            low: pd.Series,
            close: pd.Series,
            period: int = 14
    ) -> pd.Series:
        """Calcula ATR desde Series de pandas"""

        # True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # ATR como media móvil del TR
        atr = tr.rolling(window=period).mean()

        return atr

    def _calculate_atr_from_arrays(
            self,
            high: np.ndarray,
            low: np.ndarray,
            close: np.ndarray,
            period: int = 14
    ) -> np.ndarray:
        """Calcula ATR desde arrays numpy"""

        # True Range
        tr1 = high - low
        tr2 = np.abs(high[1:] - close[:-1])
        tr3 = np.abs(low[1:] - close[:-1])

        # Pad first element
        tr2 = np.concatenate([[tr1[0]], tr2])
        tr3 = np.concatenate([[tr1[0]], tr3])

        tr = np.maximum(tr1, np.maximum(tr2, tr3))

        # ATR como media móvil
        atr = np.convolve(tr, np.ones(period) / period, mode='same')

        return atr

    def _detect_regime_from_df(
            self,
            df: pd.DataFrame,
            price_col: str,
            atr_values: pd.Series
    ) -> str:
        """
        Detecta régimen desde DataFrame

        Lógica:
        - TRENDING: Precio lejos de SMAs, dirección clara
        - RANGING: Precio oscilando alrededor de SMAs
        """

        prices = df[price_col]

        # Calcular SMAs
        sma_fast = prices.rolling(window=self.regime_sma_fast).mean()
        sma_slow = prices.rolling(window=self.regime_sma_slow).mean()

        current_price = prices.iloc[-1]
        current_sma_fast = sma_fast.iloc[-1]
        current_sma_slow = sma_slow.iloc[-1]
        current_atr = atr_values.iloc[-1]

        # Distancia a SMAs en términos de ATR
        dist_to_fast = abs(current_price - current_sma_fast) / current_atr
        dist_to_slow = abs(current_price - current_sma_slow) / current_atr

        # SMA fast vs slow (tendencia)
        sma_diff = abs(current_sma_fast - current_sma_slow) / current_atr

        # Lógica de detección
        if dist_to_fast > 1.5 or dist_to_slow > 2.0 or sma_diff > 1.0:
            return 'trending'
        else:
            return 'ranging'

    def _detect_regime_from_arrays(
            self,
            prices: np.ndarray,
            current_atr: float,
            hist_atr_mean: float
    ) -> str:
        """Detecta régimen desde arrays"""

        if len(prices) < self.regime_sma_slow:
            return 'unknown'

        # SMAs simples
        sma_fast = np.mean(prices[-self.regime_sma_fast:])
        sma_slow = np.mean(prices[-self.regime_sma_slow:])

        current_price = prices[-1]

        # Distancias
        dist_to_fast = abs(current_price - sma_fast) / current_atr
        dist_to_slow = abs(current_price - sma_slow) / current_atr
        sma_diff = abs(sma_fast - sma_slow) / current_atr

        # Detección
        if dist_to_fast > 1.5 or dist_to_slow > 2.0 or sma_diff > 1.0:
            return 'trending'
        else:
            return 'ranging'


# ============================================================================
# CALCULADOR INCREMENTAL (ROLLING)
# ============================================================================

class RollingMetricsCalculator:
    """
    Calculador incremental que mantiene estado en memoria
    Más eficiente para sistemas en tiempo real
    """

    def __init__(
            self,
            lookback_days: int = 365,
            regime_sma_fast: int = 20,
            regime_sma_slow: int = 50
    ):
        self.lookback_days = lookback_days
        self.regime_sma_fast = regime_sma_fast
        self.regime_sma_slow = regime_sma_slow

        # Buffers circulares
        self.prices = []
        self.atrs = []

        # Caches
        self._hist_max = None
        self._hist_min = None
        self._hist_atr_mean = None

    def update(
            self,
            price: float,
            atr: float
    ) -> Dict[str, Any]:
        """
        Actualiza con nuevo precio/ATR y retorna métricas

        Args:
            price: Precio actual
            atr: ATR actual

        Returns:
            Dict con métricas actualizadas
        """

        # Añadir a buffers
        self.prices.append(price)
        self.atrs.append(atr)

        # Mantener solo lookback_days
        if len(self.prices) > self.lookback_days:
            self.prices.pop(0)
            self.atrs.pop(0)

        # Calcular métricas
        self._hist_max = max(self.prices)
        self._hist_min = min(self.prices)
        self._hist_atr_mean = np.mean(self.atrs)

        # Régimen
        regime = self._detect_regime_rolling()

        return {
            'price': price,
            'hist_max': self._hist_max,
            'hist_min': self._hist_min,
            'atr': atr,
            'hist_atr_avg': self._hist_atr_mean,
            'macro_regime': regime,

            # Metadata
            'pct_of_max': (price / self._hist_max) * 100,
            'pct_of_min': (price / self._hist_min) * 100,
            'atr_ratio': atr / self._hist_atr_mean,
            'buffer_size': len(self.prices)
        }

    def _detect_regime_rolling(self) -> str:
        """Detecta régimen con datos en buffer"""

        if len(self.prices) < self.regime_sma_slow:
            return 'unknown'

        prices_arr = np.array(self.prices)

        sma_fast = np.mean(prices_arr[-self.regime_sma_fast:])
        sma_slow = np.mean(prices_arr[-self.regime_sma_slow:])

        current_price = prices_arr[-1]
        current_atr = self.atrs[-1]

        dist_to_fast = abs(current_price - sma_fast) / current_atr
        dist_to_slow = abs(current_price - sma_slow) / current_atr
        sma_diff = abs(sma_fast - sma_slow) / current_atr

        if dist_to_fast > 1.5 or dist_to_slow > 2.0 or sma_diff > 1.0:
            return 'trending'
        else:
            return 'ranging'

    def get_current_metrics(self) -> Dict[str, Any]:
        """Retorna métricas actuales sin actualizar"""

        if not self.prices:
            return None

        price = self.prices[-1]
        atr = self.atrs[-1]

        return {
            'price': price,
            'hist_max': self._hist_max,
            'hist_min': self._hist_min,
            'atr': atr,
            'hist_atr_avg': self._hist_atr_mean,
            'macro_regime': self._detect_regime_rolling(),
            'pct_of_max': (price / self._hist_max) * 100 if self._hist_max else 0,
            'pct_of_min': (price / self._hist_min) * 100 if self._hist_min else 0,
            'atr_ratio': atr / self._hist_atr_mean if self._hist_atr_mean else 1.0,
            'buffer_size': len(self.prices)
        }


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TEST: CÁLCULO DE MÉTRICAS HISTÓRICAS")
    print("=" * 70)

    # ========================================================================
    # OPCIÓN 1: DESDE DATAFRAME
    # ========================================================================

    print("\n1️⃣  OPCIÓN 1: Desde DataFrame")
    print("-" * 70)

    from_date = datetime(2024, 1, 1)
    to_date = datetime(2026, 1, 23)

    db = Database()
    dm = DataManager.from_database_historical_2(db, from_date, to_date)
    df = dm.df

    calculator = HistoricalMetricsCalculator(lookback_days=365)
    metrics = calculator.calculate_from_dataframe(df)

    print(f"Precio actual: {metrics['price']:.2f}")
    print(f"Máximo histórico: {metrics['hist_max']:.2f}")
    print(f"Mínimo histórico: {metrics['hist_min']:.2f}")
    print(f"% del máximo: {metrics['pct_of_max']:.2f}%")
    print(f"ATR actual: {metrics['atr']:.2f}")
    print(f"ATR promedio: {metrics['hist_atr_avg']:.2f}")
    print(f"Ratio ATR: {metrics['atr_ratio']:.2f}x")
    print(f"Régimen: {metrics['macro_regime']}")

    '''

    # ========================================================================
    # OPCIÓN 2: DESDE ARRAYS
    # ========================================================================

    print("\n2️⃣  OPCIÓN 2: Desde Arrays/Listas")
    print("-" * 70)

    prices_list = list(df['close'].values)
    highs_list = list(df['high'].values)
    lows_list = list(df['low'].values)

    metrics2 = calculator.calculate_from_arrays(
        prices=np.array(prices_list),
        highs=np.array(highs_list),
        lows=np.array(lows_list)
    )

    print(f"Precio actual: {metrics2['price']:.2f}")
    print(f"Máximo histórico: {metrics2['hist_max']:.2f}")
    print(f"Régimen: {metrics2['macro_regime']}")

    # ========================================================================
    # OPCIÓN 3: ROLLING (INCREMENTAL)
    # ========================================================================

    print("\n3️⃣  OPCIÓN 3: Rolling (Incremental)")
    print("-" * 70)

    rolling = calculator.create_rolling_calculator()

    # Simular streaming
    print("\nSimulando updates incrementales...")
    for i in range(len(df)):
        price = df['close'].iloc[i]

        # Calcular ATR simple (high-low)
        atr = df['high'].iloc[i] - df['low'].iloc[i]

        metrics_rolling = rolling.update(price, atr)

        # Mostrar solo cada 100 días
        if i % 100 == 0:
            print(f"Día {i}: precio={price:.2f}, max={metrics_rolling['hist_max']:.2f}, "
                  f"regime={metrics_rolling['macro_regime']}")

    print("\nÚltimas métricas:")
    final_metrics = rolling.get_current_metrics()
    print(f"Precio: {final_metrics['price']:.2f}")
    print(f"% del máximo: {final_metrics['pct_of_max']:.2f}%")
    print(f"Régimen: {final_metrics['macro_regime']}")

    print("\n" + "=" * 70)
    print("✅ Todas las opciones funcionando correctamente")
    print("=" * 70)
    
    '''
