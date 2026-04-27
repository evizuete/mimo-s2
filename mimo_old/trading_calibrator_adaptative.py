"""
Integración de Calibración Adaptativa en el Pipeline de Trading

Este módulo se integra DESPUÉS de tu modelo predictivo pero ANTES de
la decisión de trading, ajustando las probabilidades calibradas según
el régimen de mercado.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional


class TradingCalibratorAdaptive:
    """
    Calibrador adaptativo integrado con el pipeline de trading

    Uso:
        calibrator = TradingCalibratorAdaptive()

        # Después de obtener predicciones
        adjusted = calibrator.adjust_probabilities(
            raw_p_buy=model.predict_raw_buy(),
            raw_p_sell=model.predict_raw_sell(),
            cal_p_buy=calibrator_model.predict_buy(),
            cal_p_sell=calibrator_model.predict_sell(),
            market_data={
                'price': current_price,
                'hist_max': historical_max,
                'hist_min': historical_min,
                'atr': current_atr,
                'hist_atr_avg': historical_atr_avg,
                'macro_regime': 'ranging/trending'
            }
        )

        # Usar adjusted['p_buy'] y adjusted['p_sell']
    """

    def __init__(
            self,
            min_calibration_ratio: float = 0.2,
            extreme_threshold: float = 0.98,
            extreme_discount: float = 0.7,
            high_vol_threshold: float = 1.5,
            low_vol_threshold: float = 0.7,
            ranging_boost: float = 1.2,
            enable_logging: bool = True
    ):
        """
        Args:
            min_calibration_ratio: Ratio mínimo cal/raw (0.2 = máximo 80% reducción)
            extreme_threshold: % de máximo/mínimo para considerar extremo (0.98 = 98%)
            extreme_discount: Factor de descuento en extremos (0.7 = -30%)
            high_vol_threshold: Ratio ATR para considerar alta volatilidad (1.5 = +50%)
            low_vol_threshold: Ratio ATR para considerar baja volatilidad (0.7 = -30%)
            ranging_boost: Factor de boost en ranging (1.2 = +20%)
            enable_logging: Habilitar logging de ajustes
        """
        self.min_ratio = min_calibration_ratio
        self.extreme_threshold = extreme_threshold
        self.extreme_discount = extreme_discount
        self.high_vol_threshold = high_vol_threshold
        self.low_vol_threshold = low_vol_threshold
        self.ranging_boost = ranging_boost
        self.enable_logging = enable_logging

        # Estadísticas
        self.stats = {
            'total_adjustments': 0,
            'near_max_count': 0,
            'near_min_count': 0,
            'high_vol_count': 0,
            'low_vol_count': 0,
            'ranging_boost_count': 0,
            'floor_applied_count': 0,
            'normal_count': 0
        }

    def adjust_probabilities(
            self,
            raw_p_buy: float,
            raw_p_sell: float,
            cal_p_buy: float,
            cal_p_sell: float,
            market_data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Ajusta probabilidades calibradas según contexto de mercado

        Args:
            raw_p_buy: Probabilidad raw de compra
            raw_p_sell: Probabilidad raw de venta
            cal_p_buy: Probabilidad calibrada de compra
            cal_p_sell: Probabilidad calibrada de venta
            market_data: Dict con datos de mercado

        Returns:
            Dict con probabilidades ajustadas y metadata
        """

        self.stats['total_adjustments'] += 1

        # Ajustar BUY
        adj_p_buy, reason_buy = self._calibrate_single(
            raw_prob=raw_p_buy,
            cal_prob=cal_p_buy,
            market_data=market_data
        )

        # Ajustar SELL
        adj_p_sell, reason_sell = self._calibrate_single(
            raw_prob=raw_p_sell,
            cal_prob=cal_p_sell,
            market_data=market_data
        )

        # Actualizar estadísticas
        self._update_stats(reason_buy)
        self._update_stats(reason_sell)

        # Calcular ratios
        buy_ratio = adj_p_buy / raw_p_buy if raw_p_buy > 0 else 0
        sell_ratio = adj_p_sell / raw_p_sell if raw_p_sell > 0 else 0

        result = {
            # Probabilidades ajustadas (USAR ESTAS)
            'p_buy': adj_p_buy,
            'p_sell': adj_p_sell,

            # Metadata
            'reason_buy': reason_buy,
            'reason_sell': reason_sell,

            # Probabilidades originales (referencia)
            'raw_buy': raw_p_buy,
            'raw_sell': raw_p_sell,
            'cal_buy': cal_p_buy,
            'cal_sell': cal_p_sell,

            # Ratios de ajuste
            'buy_ratio': buy_ratio,
            'sell_ratio': sell_ratio,

            # Factor de mejora
            'buy_improvement': (adj_p_buy / cal_p_buy) if cal_p_buy > 0 else 0,
            'sell_improvement': (adj_p_sell / cal_p_sell) if cal_p_sell > 0 else 0
        }

        if self.enable_logging:
            self._log_adjustment(result, market_data)

        return result

    def _calibrate_single(
            self,
            raw_prob: float,
            cal_prob: float,
            market_data: Dict[str, Any]
    ) -> Tuple[float, str]:
        """
        Calibra una probabilidad individual

        Returns:
            (adjusted_prob, reason)
        """

        # Extraer datos de mercado
        price = market_data.get('price', 0)
        hist_max = market_data.get('hist_max', price)
        hist_min = market_data.get('hist_min', price)
        atr = market_data.get('atr', 0)
        hist_atr_avg = market_data.get('hist_atr_avg', atr)
        macro_regime = market_data.get('macro_regime', 'unknown')

        # Validaciones básicas
        if raw_prob < 0.01:
            return cal_prob, "low_raw_prob"

        if price <= 0 or hist_max <= 0:
            return cal_prob, "invalid_price"

        # Calcular contexto
        pct_of_max = price / hist_max
        pct_of_min = price / hist_min if hist_min > 0 else 0.5
        vol_ratio = atr / hist_atr_avg if hist_atr_avg > 0 else 1.0
        actual_ratio = cal_prob / raw_prob if raw_prob > 0.01 else 1.0

        # =====================================================================
        # CASO 1: CERCA DE MÁXIMO HISTÓRICO (PROBLEMA PRINCIPAL)
        # =====================================================================
        if pct_of_max >= self.extreme_threshold:
            # Usar raw con descuento conservador
            adjusted = raw_prob * self.extreme_discount
            return adjusted, f"near_max_{pct_of_max:.1%}"

        # =====================================================================
        # CASO 2: CERCA DE MÍNIMO HISTÓRICO
        # =====================================================================
        if pct_of_min <= (2 - self.extreme_threshold):
            adjusted = raw_prob * self.extreme_discount
            return adjusted, f"near_min_{pct_of_min:.1%}"

        # =====================================================================
        # CASO 3: VOLATILIDAD EXTREMA
        # =====================================================================
        if vol_ratio > self.high_vol_threshold:
            # Alta volatilidad → Interpolar con raw
            adjusted = 0.6 * (raw_prob * 0.8) + 0.4 * cal_prob
            return adjusted, f"high_vol_{vol_ratio:.2f}x"

        # =====================================================================
        # CASO 4: VOLATILIDAD MUY BAJA
        # =====================================================================
        if vol_ratio < self.low_vol_threshold:
            # Baja volatilidad → Usar calibradas sin mucho ajuste
            adjusted = cal_prob * 1.1  # Pequeño boost
            return min(adjusted, raw_prob * 0.9), f"low_vol_{vol_ratio:.2f}x"

        # =====================================================================
        # CASO 5: RANGING (MENOS RESTRICTIVO)
        # =====================================================================
        if macro_regime == 'ranging':
            adjusted = cal_prob * self.ranging_boost
            # Cap al 90% del raw
            adjusted = min(adjusted, raw_prob * 0.9)
            return adjusted, "ranging_boost"

        # =====================================================================
        # CASO 6: COLAPSO EXCESIVO (RATIO MUY BAJO)
        # =====================================================================
        if actual_ratio < self.min_ratio:
            # Suavizar: 70% cal + 30% raw con floor
            floor_prob = raw_prob * self.min_ratio
            adjusted = 0.7 * cal_prob + 0.3 * floor_prob
            return adjusted, f"floor_applied_ratio_{actual_ratio:.3f}"

        # =====================================================================
        # CASO 7: NORMAL - USAR CALIBRADO
        # =====================================================================
        return cal_prob, "normal"

    def _update_stats(self, reason: str):
        """Actualiza estadísticas de ajustes"""
        if 'near_max' in reason:
            self.stats['near_max_count'] += 1
        elif 'near_min' in reason:
            self.stats['near_min_count'] += 1
        elif 'high_vol' in reason:
            self.stats['high_vol_count'] += 1
        elif 'low_vol' in reason:
            self.stats['low_vol_count'] += 1
        elif 'ranging' in reason:
            self.stats['ranging_boost_count'] += 1
        elif 'floor' in reason:
            self.stats['floor_applied_count'] += 1
        elif 'normal' in reason:
            self.stats['normal_count'] += 1

    def _log_adjustment(self, result: Dict[str, Any], market_data: Dict[str, Any]):
        """Log de ajustes (solo si hay cambio significativo)"""

        # Solo loguear si hay mejora significativa
        buy_improvement = result['buy_improvement']
        sell_improvement = result['sell_improvement']

        if buy_improvement > 2.0 or sell_improvement > 2.0:
            price = market_data.get('price', 0)
            hist_max = market_data.get('hist_max', 0)
            pct_max = (price / hist_max * 100) if hist_max > 0 else 0

            print(f"\n🔧 Calibración Adaptativa:")
            print(f"   Precio: {price:.2f} ({pct_max:.1f}% del máx)")
            print(f"   BUY: {result['cal_buy']:.4f} → {result['p_buy']:.4f} "
                  f"(x{buy_improvement:.1f}, {result['reason_buy']})")
            print(f"   SELL: {result['cal_sell']:.4f} → {result['p_sell']:.4f} "
                  f"(x{sell_improvement:.1f}, {result['reason_sell']})")

    def get_stats(self) -> Dict[str, Any]:
        """Retorna estadísticas de ajustes"""
        total = self.stats['total_adjustments']
        if total == 0:
            return self.stats

        return {
            **self.stats,
            'near_max_pct': (self.stats['near_max_count'] / total) * 100,
            'near_min_pct': (self.stats['near_min_count'] / total) * 100,
            'high_vol_pct': (self.stats['high_vol_count'] / total) * 100,
            'ranging_boost_pct': (self.stats['ranging_boost_count'] / total) * 100,
            'floor_applied_pct': (self.stats['floor_applied_count'] / total) * 100,
            'normal_pct': (self.stats['normal_count'] / total) * 100
        }

    def reset_stats(self):
        """Resetea estadísticas"""
        for key in self.stats:
            self.stats[key] = 0


# ============================================================================
# FUNCIÓN DE CONVENIENCIA
# ============================================================================

def create_adaptive_calibrator(
        conservative: bool = False,
        aggressive: bool = False
) -> TradingCalibratorAdaptive:
    """
    Crea calibrador con configuración predefinida

    Args:
        conservative: Configuración conservadora (menos ajustes)
        aggressive: Configuración agresiva (más ajustes)

    Returns:
        TradingCalibratorAdaptive configurado
    """

    if aggressive:
        return TradingCalibratorAdaptive(
            min_calibration_ratio=0.3,  # Permite hasta 70% reducción
            extreme_threshold=0.95,  # 95% del máximo
            extreme_discount=0.8,  # -20% en extremos
            high_vol_threshold=1.3,  # Threshold más bajo
            ranging_boost=1.3,  # +30% en ranging
            enable_logging=True
        )

    elif conservative:
        return TradingCalibratorAdaptive(
            min_calibration_ratio=0.15,  # Permite hasta 85% reducción
            extreme_threshold=0.99,  # 99% del máximo
            extreme_discount=0.6,  # -40% en extremos
            high_vol_threshold=1.7,  # Threshold más alto
            ranging_boost=1.1,  # +10% en ranging
            enable_logging=True
        )

    else:
        # Balanceado (default)
        return TradingCalibratorAdaptive(
            min_calibration_ratio=0.2,
            extreme_threshold=0.98,
            extreme_discount=0.7,
            high_vol_threshold=1.5,
            ranging_boost=1.2,
            enable_logging=True
        )


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":

    print("=" * 70)
    print("TEST: CALIBRACIÓN ADAPTATIVA EN MÁXIMOS HISTÓRICOS")
    print("=" * 70)

    # Crear calibrador
    calibrator = create_adaptive_calibrator(conservative=False)

    # Simular datos en máximos históricos
    market_data = {
        'price': 4891.89,
        'hist_max': 4900.00,
        'hist_min': 2000.00,
        'atr': 15.50,
        'hist_atr_avg': 12.00,
        'macro_regime': 'ranging'
    }

    # Probabilidades del modelo
    raw_p_buy = 0.2564
    raw_p_sell = 0.2607
    cal_p_buy = 0.0030
    cal_p_sell = 0.0039

    print("\n📊 SITUACIÓN:")
    print(f"   Precio: {market_data['price']:.2f}")
    print(f"   Máximo histórico: {market_data['hist_max']:.2f}")
    print(f"   % del máximo: {(market_data['price'] / market_data['hist_max']) * 100:.2f}%")
    print()

    print("📈 PROBABILIDADES ORIGINALES:")
    print(f"   Raw  BUY:  {raw_p_buy:.4f} ({raw_p_buy * 100:.2f}%)")
    print(f"   Cal  BUY:  {cal_p_buy:.4f} ({cal_p_buy * 100:.2f}%)")
    print(f"   Ratio: {(cal_p_buy / raw_p_buy) * 100:.1f}%")
    print()

    # Aplicar calibración adaptativa
    result = calibrator.adjust_probabilities(
        raw_p_buy=raw_p_buy,
        raw_p_sell=raw_p_sell,
        cal_p_buy=cal_p_buy,
        cal_p_sell=cal_p_sell,
        market_data=market_data
    )

    print("=" * 70)
    print("✅ PROBABILIDADES AJUSTADAS")
    print("=" * 70)

    print(f"\n🔵 BUY:")
    print(f"   Raw:      {result['raw_buy']:.4f} ({result['raw_buy'] * 100:.2f}%)")
    print(f"   Cal:      {result['cal_buy']:.4f} ({result['cal_buy'] * 100:.2f}%)")
    print(f"   Adjusted: {result['p_buy']:.4f} ({result['p_buy'] * 100:.2f}%) ← USAR")
    print(f"   Razón:    {result['reason_buy']}")
    print(f"   Mejora:   {result['buy_improvement']:.1f}x vs calibrado")

    print(f"\n🔴 SELL:")
    print(f"   Raw:      {result['raw_sell']:.4f} ({result['raw_sell'] * 100:.2f}%)")
    print(f"   Cal:      {result['cal_sell']:.4f} ({result['cal_sell'] * 100:.2f}%)")
    print(f"   Adjusted: {result['p_sell']:.4f} ({result['p_sell'] * 100:.2f}%) ← USAR")
    print(f"   Razón:    {result['reason_sell']}")
    print(f"   Mejora:   {result['sell_improvement']:.1f}x vs calibrado")

    print("\n" + "=" * 70)
    print("💡 INTERPRETACIÓN")
    print("=" * 70)

    # Evaluar con gate de 15%
    gate = 15.0

    if result['p_buy'] >= gate:
        print(f"\n✅ BUY VIABLE ({result['p_buy'] * 100:.1f}% >= {gate}%)")
        print(f"   Recomendación: OPERAR")
    else:
        print(f"\n⚠️  BUY NO VIABLE ({result['p_buy'] * 100:.1f}% < {gate}%)")
        print(f"   Diferencia: {gate - result['p_buy'] * 100:.1f}% faltante")

    print("\n" + "=" * 70)
