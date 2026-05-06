# -*- coding: utf-8 -*-
"""
adaptive_sl_manager.py
======================
Gestión dinámica del Stop Loss virtual para posiciones abiertas en S3.

Cambios v24.0:
  FIX-1: sync_sl(new_sl) añadido como método público (simetría con AdaptiveTPManager).
          S3 lo llama antes de on_bar() para garantizar que el manager opera
          siempre con el virtual_sl vigente (trailing, BE y profit_locks pueden
          haberlo movido desde el último ciclo).
  FIX-2: _bars_open ahora cuenta velas M1 reales (cambio de minuto en ts),
          no llamadas al monitor. Corrige compression_min_hold_bars y
          expansion_max_bars que expiraban en 0.6s con monitor a 5Hz.
          Nuevo atributo _last_bar_minute para tracking de vela actual.

Implementa dos mecanismos complementarios:

  A) EXPANSIÓN (anti-sweep)
     Cuando el precio toca el virtual SL pero hay señales de reversión,
     amplía el SL una vez y da un tiempo limitado para que el rebote se confirme.
     Si no se confirma, cierra sin excepciones.

  B) COMPRESIÓN proactiva
     Cuando la tesis de entrada se invalida (el modelo gira, MACD cruza en contra,
     volumen creciente adverso), acerca el SL al precio actual para limitar el daño,
     sin esperar a que el precio llegue al SL original.

Reglas de seguridad invariantes:
  - El hard SL del broker nunca se toca.
  - El virtual SL expandido nunca puede superar hard_sl - hard_sl_margin_pts.
  - La compresión solo mueve el SL HACIA el precio, nunca lo aleja.
  - Expansión y compresión son mutuamente excluyentes por estado.
  - Una expansión solo puede activarse UNA vez por posición.

Integración con S3:
  Ver sección "INTEGRACIÓN EN S3" al final del fichero.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Enumerados de estado
# ---------------------------------------------------------------------------

class SLState(Enum):
    NORMAL      = auto()   # Monitorización estándar, compresión activa
    IN_SL_ZONE  = auto()   # Precio ha tocado el virtual SL, evaluando expansión
    EXPANDED    = auto()   # SL ampliado, timer activo
    CLOSED      = auto()   # Posición cerrada, objeto inerte


class CloseReason(Enum):
    VIRTUAL_SL          = "VIRTUAL_SL"
    VIRTUAL_TP          = "VIRTUAL_TP"
    EXPANSION_TIMEOUT   = "EXPANSION_TIMEOUT"
    EXPANSION_SL        = "EXPANSION_SL"
    COMPRESSION_SL      = "COMPRESSION_SL"
    HARD_SL             = "HARD_SL"


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveSLConfig:
    """
    Parámetros del gestor adaptativo. Todos los valores en puntos del broker
    (misma unidad que virtual_sl_points / emergency_sl_points en S3).

    Ejemplo para XAUUSD M1 (1 punto = 0.1 precio):
        expansion_pts = 50      →  5.0 USD de margen extra
        hard_sl_margin_pts = 50 →  5.0 USD de distancia mínima al hard SL
    """

    # ── Expansión ──────────────────────────────────────────────────────────
    expansion_enabled: bool = True

    # Puntos extra que se añaden al virtual SL cuando se detecta reversión
    expansion_pts: int = 50

    # Timer: N velas M1 máximo en modo expandido (si no rebota, cierra)
    expansion_max_bars: int = 3

    # Segundos máximos en modo expandido (red de seguridad independiente de barras)
    expansion_max_seconds: float = 210.0   # 3.5 min ≈ 3-4 velas M1

    # Número mínimo de señales de reversión que deben coincidir para expandir
    expansion_min_signals: int = 2

    # Umbral mínimo de proba_reversal del modelo para contar como señal
    expansion_proba_threshold: float = 0.45

    # RSI: zonas extremas para considerar señal de reversión
    expansion_rsi_oversold: float  = 35.0   # para BUY (precio bajó mucho)
    expansion_rsi_overbought: float = 65.0  # para SELL (precio subió mucho)

    # Ratio máximo cuerpo/rango de la vela actual para considerar "indecisión"
    expansion_indecision_body_ratio: float = 0.35

    # ── Compresión ─────────────────────────────────────────────────────────
    compression_enabled: bool = True

    # El modelo debe tener proba de reversión >= este umbral para comprimir
    compression_proba_threshold: float = 0.60

    # Offset desde el precio actual donde se pone el nuevo SL comprimido
    # (en puntos, en dirección favorable a la posición)
    compression_offset_pts: int = 30

    # Velas mínimas desde la apertura antes de permitir compresión
    # (evita comprimir en el ruido inicial de la entrada)
    compression_min_hold_bars: int = 3

    # Cooldown en velas tras una compresión (evita comprimir repetidamente en chop)
    compression_cooldown_bars: int = 5

    # MACD: umbral del histograma para considerar cruce relevante
    compression_macd_hist_threshold: float = 0.0

    # Volumen: factor mínimo sobre la media para considerar volumen adverso creciente
    compression_volume_factor: float = 1.3

    # ── Seguridad ──────────────────────────────────────────────────────────
    # Distancia mínima que debe haber siempre entre el virtual SL
    # (expandido o comprimido) y el hard SL del broker.
    hard_sl_margin_pts: int = 35

    # Factor de conversión puntos → precio
    # XAUUSD con broker en décimas: 1 punto = 0.1 precio → pts_to_price = 0.1
    pts_to_price: float = 0.1


# ---------------------------------------------------------------------------
# Señales de reversión (resultado intermedio del evaluador)
# ---------------------------------------------------------------------------

@dataclass
class ReversalSignals:
    """Resultado de la evaluación de señales de reversión en un bar."""
    proba_ok: bool = False        # modelo dice que hay reversión
    rsi_ok: bool = False          # RSI en zona extrema
    indecision_ok: bool = False   # vela de indecisión (body/range pequeño)
    macd_ok: bool = False         # MACD apoya la reversión
    volume_ok: bool = False       # volumen decreciente (agotamiento del movimiento)

    @property
    def count(self) -> int:
        return sum([self.proba_ok, self.rsi_ok, self.indecision_ok,
                    self.macd_ok, self.volume_ok])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proba_ok": self.proba_ok,
            "rsi_ok": self.rsi_ok,
            "indecision_ok": self.indecision_ok,
            "macd_ok": self.macd_ok,
            "volume_ok": self.volume_ok,
            "count": self.count,
        }


# ---------------------------------------------------------------------------
# Resultado de on_bar
# ---------------------------------------------------------------------------

@dataclass
class BarResult:
    """
    Lo que S3 debe hacer tras procesar un bar con el AdaptiveSLManager.

    action:
        "none"      → no hacer nada, seguir esperando
        "close"     → cerrar la posición ahora al precio de mercado
        "update_sl" → actualizar el virtual SL al valor new_virtual_sl

    new_virtual_sl  nuevo precio del virtual SL (solo si action="update_sl")
    reason          razón del cierre o del movimiento
    debug           dict con info de diagnóstico para el log de S3
    """
    action: str
    new_virtual_sl: Optional[float] = None
    reason: Optional[str] = None
    debug: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# AdaptiveSLManager
# ---------------------------------------------------------------------------

class AdaptiveSLManager:
    """
    Gestor de SL adaptativo para UNA posición abierta.

    Ciclo de vida:
        1. Instanciar al abrir la posición (ver integración en S3 al final).
        2. Llamar on_bar() en cada vela/tick mientras la posición esté abierta.
        3. Actuar según el BarResult devuelto (close / update_sl / none).
        4. Llamar on_closed() cuando la posición se cierra por cualquier motivo.

    Una instancia = una posición. No reutilizar entre posiciones.
    """

    def __init__(
        self,
        ticket: int,
        side: str,             # "BUY" o "SELL"
        entry_price: float,
        virtual_sl: float,     # SL virtual original (en precio)
        virtual_tp: float,     # TP virtual (en precio)
        hard_sl: float,        # SL de emergencia del broker (en precio)
        config: AdaptiveSLConfig = AdaptiveSLConfig(),
    ):
        self.ticket = ticket
        self.side = side.upper()
        self.entry_price = entry_price
        self.config = config

        # SL/TP actuales (pueden modificarse durante la vida de la posición)
        self.virtual_sl = virtual_sl
        self.virtual_sl_original = virtual_sl
        self.virtual_tp = virtual_tp
        self.hard_sl = hard_sl

        # Estado de la máquina de estados
        self._state: SLState = SLState.NORMAL

        # Contadores
        self._bars_open: int = 0
        self._compression_cooldown: int = 0

        # FIX v24.0: tracking de barra actual para que _bars_open cuente velas M1
        # reales y no llamadas al monitor (que corre a 5Hz).
        # Se actualiza en on_bar() cuando cambia el minuto de `ts`.
        self._last_bar_minute: Optional[int] = None

        # Control de expansión (solo una por posición)
        self._expansion_used: bool = False
        self._expansion_bar_count: int = 0
        self._expansion_ts_start: Optional[float] = None

        logger.info(
            f"[AdaptiveSL] ticket={ticket} side={side} "
            f"entry={entry_price:.5f} vsl={virtual_sl:.5f} "
            f"vtp={virtual_tp:.5f} hard_sl={hard_sl:.5f}"
        )

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def state(self) -> SLState:
        return self._state

    # ------------------------------------------------------------------
    # API pública
    # ------------------------------------------------------------------

    def on_bar(
        self,
        *,
        bid: float,
        ask: float,
        bar_open: float,
        bar_high: float,
        bar_low: float,
        bar_close: float,
        atr: float,
        rsi: Optional[float] = None,
        macd_hist: Optional[float] = None,
        macd_hist_prev: Optional[float] = None,
        volume: Optional[float] = None,
        volume_ma: Optional[float] = None,
        proba_long: Optional[float] = None,
        proba_short: Optional[float] = None,
        ts: Optional[float] = None,
    ) -> BarResult:
        """
        Procesa un nuevo bar/tick. Devuelve la acción que S3 debe ejecutar.

        Parámetros
        ----------
        bid / ask           Precio de venta/compra actual
        bar_*               OHLC del bar actual (o último bar cerrado)
        atr                 ATR actual
        rsi                 RSI del bar actual (opcional pero recomendado)
        macd_hist           Histograma MACD actual (opcional)
        macd_hist_prev      Histograma MACD del bar anterior (opcional)
        volume              Volumen del bar actual (opcional)
        volume_ma           Media de volumen (N barras) (opcional)
        proba_long/short    Probabilidades calibradas del modelo S2 (opcional)
        ts                  Timestamp Unix del bar (usa time.time() si None)
        """
        if self._state == SLState.CLOSED:
            return BarResult(action="none", debug={"state": "CLOSED"})

        ts = ts or time.time()

        # FIX v24.0: _bars_open cuenta velas M1 reales, no llamadas al monitor.
        # El monitor corre a 5Hz; sin este guard, compression_min_hold_bars=3
        # expiraba en 0.6s en lugar de 3 minutos.
        # Detectamos nueva barra por cambio de minuto en el timestamp.
        _current_minute = int(ts // 60)
        if self._last_bar_minute is None or _current_minute != self._last_bar_minute:
            self._bars_open += 1
            self._last_bar_minute = _current_minute
            if self._compression_cooldown > 0:
                self._compression_cooldown -= 1

        # Precio de ejecución relevante según el lado
        # BUY cierra vendiendo → bid; SELL cierra comprando → ask
        exec_price = bid if self.side == "BUY" else ask

        # ── 1. Hard SL del broker ─────────────────────────────────────────
        hard_hit = (
            (self.side == "BUY"  and exec_price <= self.hard_sl) or
            (self.side == "SELL" and exec_price >= self.hard_sl)
        )
        if hard_hit:
            return self._close(CloseReason.HARD_SL, exec_price, {
                "hard_sl": self.hard_sl,
            })

        # ── 2. TP virtual ─────────────────────────────────────────────────
        tp_hit = (
            (self.side == "BUY"  and exec_price >= self.virtual_tp) or
            (self.side == "SELL" and exec_price <= self.virtual_tp)
        )
        if tp_hit:
            return self._close(CloseReason.VIRTUAL_TP, exec_price, {
                "virtual_tp": self.virtual_tp,
            })

        # ── 3. Máquina de estados ─────────────────────────────────────────
        kwargs = dict(
            exec_price=exec_price,
            bar_open=bar_open, bar_high=bar_high,
            bar_low=bar_low, bar_close=bar_close,
            atr=atr, rsi=rsi,
            macd_hist=macd_hist, macd_hist_prev=macd_hist_prev,
            volume=volume, volume_ma=volume_ma,
            proba_long=proba_long, proba_short=proba_short,
            ts=ts,
        )

        if self._state == SLState.NORMAL:
            return self._handle_normal(**kwargs)
        elif self._state == SLState.IN_SL_ZONE:
            return self._handle_in_sl_zone(**kwargs)
        elif self._state == SLState.EXPANDED:
            return self._handle_expanded(exec_price=exec_price, ts=ts)

        return BarResult(action="none")

    def on_closed(self) -> None:
        """Notificar que la posición fue cerrada externamente (manual, BE, etc.)."""
        self._state = SLState.CLOSED
        logger.info(f"[AdaptiveSL] ticket={self.ticket} cerrado externamente")

    def sync_sl(self, new_sl: float) -> None:
        """
        Sincronizar el virtual SL actual desde S3 (trailing, BE, compresión).

        v24.0: añadido por simetría con AdaptiveTPManager.sync_sl(). Permite que
        S3 use la misma API en ambos managers sin acceder a atributos internos.
        También resuelve el bug donde on_bar() evaluaba un SL stale si el trailing
        o el BE habían movido tr.virtual_sl_price en ciclos anteriores.
        """
        self.virtual_sl = new_sl

    def snapshot(self) -> Dict[str, Any]:
        """Estado interno completo para logging y diagnóstico."""
        return {
            "ticket": self.ticket,
            "side": self.side,
            "state": self._state.name,
            "virtual_sl": self.virtual_sl,
            "virtual_sl_original": self.virtual_sl_original,
            "virtual_tp": self.virtual_tp,
            "hard_sl": self.hard_sl,
            "bars_open": self._bars_open,
            "expansion_used": self._expansion_used,
            "expansion_bar_count": self._expansion_bar_count,
            "compression_cooldown": self._compression_cooldown,
        }

    # ------------------------------------------------------------------
    # Handlers de estado
    # ------------------------------------------------------------------

    def _handle_normal(self, *, exec_price, bar_open, bar_high, bar_low,
                       bar_close, atr, rsi, macd_hist, macd_hist_prev,
                       volume, volume_ma, proba_long, proba_short, ts) -> BarResult:

        sl_touched = (
            (self.side == "BUY"  and exec_price <= self.virtual_sl) or
            (self.side == "SELL" and exec_price >= self.virtual_sl)
        )

        if sl_touched:
            if self.config.expansion_enabled and not self._expansion_used:
                # Transicionar a IN_SL_ZONE y evaluar en el mismo bar
                self._state = SLState.IN_SL_ZONE
                return self._handle_in_sl_zone(
                    exec_price=exec_price,
                    bar_open=bar_open, bar_high=bar_high,
                    bar_low=bar_low, bar_close=bar_close,
                    atr=atr, rsi=rsi,
                    macd_hist=macd_hist, macd_hist_prev=macd_hist_prev,
                    volume=volume, volume_ma=volume_ma,
                    proba_long=proba_long, proba_short=proba_short,
                    ts=ts,
                )
            # Sin expansión disponible → cerrar directamente
            return self._close(CloseReason.VIRTUAL_SL, exec_price, {
                "virtual_sl": self.virtual_sl,
                "expansion_available": False,
            })

        # SL no tocado → evaluar compresión proactiva
        if self.config.compression_enabled:
            compression_result = self._try_compress(
                exec_price=exec_price,
                proba_long=proba_long, proba_short=proba_short,
                macd_hist=macd_hist, macd_hist_prev=macd_hist_prev,
                volume=volume, volume_ma=volume_ma,
            )
            if compression_result is not None:
                return compression_result

        return BarResult(action="none", debug={
            "state": "NORMAL",
            "bars_open": self._bars_open,
            "virtual_sl": self.virtual_sl,
            "exec_price": exec_price,
        })

    def _handle_in_sl_zone(self, *, exec_price, bar_open, bar_high, bar_low,
                           bar_close, atr, rsi, macd_hist, macd_hist_prev,
                           volume, volume_ma, proba_long, proba_short, ts) -> BarResult:
        """
        El precio ha tocado el SL. Evalúa señales de reversión para decidir
        si expandir el SL o cerrar.
        """
        signals = self._evaluate_reversal_signals(
            exec_price=exec_price,
            bar_open=bar_open, bar_high=bar_high,
            bar_low=bar_low, bar_close=bar_close,
            atr=atr, rsi=rsi,
            macd_hist=macd_hist, macd_hist_prev=macd_hist_prev,
            volume=volume, volume_ma=volume_ma,
            proba_long=proba_long, proba_short=proba_short,
        )

        debug_base = {
            "state": "IN_SL_ZONE",
            "exec_price": exec_price,
            "virtual_sl": self.virtual_sl,
            "reversal_signals": signals.to_dict(),
        }

        if signals.count >= self.config.expansion_min_signals:
            new_sl = self._compute_expanded_sl()

            if new_sl is None:
                # No hay margen suficiente hasta el hard SL → cerrar
                logger.warning(
                    f"[AdaptiveSL] ticket={self.ticket} expansión bloqueada: "
                    f"sin margen hasta hard_sl ({self.hard_sl:.5f})"
                )
                return self._close(CloseReason.VIRTUAL_SL, exec_price, {
                    **debug_base, "expansion_blocked": "hard_sl_margin_insuficiente",
                })

            old_sl = self.virtual_sl
            self.virtual_sl = new_sl
            self._state = SLState.EXPANDED
            self._expansion_used = True
            self._expansion_bar_count = 0
            self._expansion_ts_start = ts

            logger.info(
                f"[AdaptiveSL] ticket={self.ticket} EXPANSIÓN activada "
                f"sl {old_sl:.5f} → {new_sl:.5f} "
                f"(signals={signals.count}/{self.config.expansion_min_signals})"
            )
            return BarResult(
                action="update_sl",
                new_virtual_sl=new_sl,
                reason="ADAPTIVE_EXPANSION",
                debug={
                    **debug_base,
                    "old_virtual_sl": old_sl,
                    "new_virtual_sl": new_sl,
                },
            )

        # Señales insuficientes → cerrar en el SL
        return self._close(CloseReason.VIRTUAL_SL, exec_price, {
            **debug_base,
            "expansion_rejected": (
                f"signals={signals.count} < min={self.config.expansion_min_signals}"
            ),
        })

    def _handle_expanded(self, *, exec_price, ts) -> BarResult:
        """
        SL expandido activo. Monitoriza si el precio rebota o se acaba el tiempo.
        La compresión queda suspendida en este estado.
        """
        self._expansion_bar_count += 1
        elapsed = ts - (self._expansion_ts_start or ts)

        debug_base = {
            "state": "EXPANDED",
            "expansion_bar_count": self._expansion_bar_count,
            "elapsed_seconds": round(elapsed, 1),
            "virtual_sl": self.virtual_sl,
            "exec_price": exec_price,
        }

        # ¿El precio tocó el SL expandido?
        sl_hit = (
            (self.side == "BUY"  and exec_price <= self.virtual_sl) or
            (self.side == "SELL" and exec_price >= self.virtual_sl)
        )
        if sl_hit:
            return self._close(CloseReason.EXPANSION_SL, exec_price, debug_base)

        # ¿Se agotó el timer (barras o segundos)?
        bars_timeout = self._expansion_bar_count >= self.config.expansion_max_bars
        time_timeout = elapsed >= self.config.expansion_max_seconds

        if bars_timeout or time_timeout:
            return self._close(CloseReason.EXPANSION_TIMEOUT, exec_price, {
                **debug_base,
                "bars_timeout": bars_timeout,
                "time_timeout": time_timeout,
            })

        # ¿El precio rebotó con margen suficiente (10 puntos de colchón)?
        rebound_margin = self._pts(10)
        rebounded = (
            (self.side == "BUY"  and exec_price > self.virtual_sl + rebound_margin) or
            (self.side == "SELL" and exec_price < self.virtual_sl - rebound_margin)
        )
        if rebounded:
            self._state = SLState.NORMAL
            logger.info(
                f"[AdaptiveSL] ticket={self.ticket} rebote confirmado → "
                f"volviendo a NORMAL con vsl={self.virtual_sl:.5f}"
            )
            return BarResult(action="none", debug={**debug_base, "rebote_confirmado": True})

        # Esperando dentro del timer
        return BarResult(action="none", debug=debug_base)

    # ------------------------------------------------------------------
    # Compresión proactiva
    # ------------------------------------------------------------------

    def _try_compress(
        self, *, exec_price,
        proba_long, proba_short,
        macd_hist, macd_hist_prev,
        volume, volume_ma,
    ) -> Optional[BarResult]:
        """
        Intenta comprimir el SL si la tesis se ha invalidado.
        Devuelve BarResult(action="update_sl") si comprime, None si no.
        """
        if self._compression_cooldown > 0:
            return None
        if self._bars_open < self.config.compression_min_hold_bars:
            return None

        signals_against = self._evaluate_compression_signals(
            exec_price=exec_price,
            proba_long=proba_long, proba_short=proba_short,
            macd_hist=macd_hist, macd_hist_prev=macd_hist_prev,
            volume=volume, volume_ma=volume_ma,
        )
        if not signals_against:
            return None

        new_sl = self._compute_compressed_sl(exec_price)
        if new_sl is None:
            return None

        old_sl = self.virtual_sl
        self.virtual_sl = new_sl
        self._compression_cooldown = self.config.compression_cooldown_bars

        logger.info(
            f"[AdaptiveSL] ticket={self.ticket} COMPRESIÓN "
            f"sl {old_sl:.5f} → {new_sl:.5f} "
            f"signals={signals_against}"
        )
        return BarResult(
            action="update_sl",
            new_virtual_sl=new_sl,
            reason="ADAPTIVE_COMPRESSION",
            debug={
                "state": "NORMAL",
                "compression": True,
                "old_virtual_sl": old_sl,
                "new_virtual_sl": new_sl,
                "exec_price": exec_price,
                "signals_against": signals_against,
            },
        )

    # ------------------------------------------------------------------
    # Evaluadores de señales
    # ------------------------------------------------------------------

    def _evaluate_reversal_signals(
        self, *, exec_price, bar_open, bar_high, bar_low, bar_close,
        atr, rsi, macd_hist, macd_hist_prev,
        volume, volume_ma, proba_long, proba_short,
    ) -> ReversalSignals:
        """Evalúa qué señales de reversión están presentes en este bar."""
        s = ReversalSignals()

        # 1. Probabilidad de reversión del modelo
        proba_rev = proba_long if self.side == "BUY" else proba_short
        if proba_rev is not None:
            s.proba_ok = proba_rev >= self.config.expansion_proba_threshold

        # 2. RSI en zona extrema
        if rsi is not None:
            s.rsi_ok = (
                rsi <= self.config.expansion_rsi_oversold  if self.side == "BUY"
                else rsi >= self.config.expansion_rsi_overbought
            )

        # 3. Vela de indecisión (cuerpo pequeño relativo al rango)
        bar_range = bar_high - bar_low
        bar_body  = abs(bar_close - bar_open)
        if bar_range > 1e-9:
            s.indecision_ok = (
                (bar_body / bar_range) <= self.config.expansion_indecision_body_ratio
            )

        # 4. MACD apoya la reversión (histograma gira hacia el lado favorable)
        if macd_hist is not None and macd_hist_prev is not None:
            if self.side == "BUY":
                # antes negativo, ahora mejorando
                s.macd_ok = (macd_hist_prev < 0) and (macd_hist > macd_hist_prev)
            else:
                # antes positivo, ahora deteriorando
                s.macd_ok = (macd_hist_prev > 0) and (macd_hist < macd_hist_prev)

        # 5. Volumen decreciente (agotamiento del movimiento adverso)
        if volume is not None and volume_ma is not None and volume_ma > 1e-9:
            s.volume_ok = volume < volume_ma

        return s

    def _evaluate_compression_signals(
        self, *, exec_price,
        proba_long, proba_short,
        macd_hist, macd_hist_prev,
        volume, volume_ma,
    ) -> List[str]:
        """
        Devuelve lista de señales de invalidación de tesis.
        Lista vacía → sin señales → no comprimir.
        """
        signals: List[str] = []

        # 1. El modelo invierte su opinión con alta convicción
        if proba_long is not None and proba_short is not None:
            if self.side == "BUY":
                if proba_short >= self.config.compression_proba_threshold:
                    signals.append(f"model_flip_short({proba_short:.3f})")
            else:
                if proba_long >= self.config.compression_proba_threshold:
                    signals.append(f"model_flip_long({proba_long:.3f})")

        # 2. MACD cruza en contra
        if macd_hist is not None and macd_hist_prev is not None:
            thr = self.config.compression_macd_hist_threshold
            if self.side == "BUY":
                if macd_hist_prev >= thr and macd_hist < thr:
                    signals.append(f"macd_cross_bearish({macd_hist:.5f})")
            else:
                if macd_hist_prev <= thr and macd_hist > thr:
                    signals.append(f"macd_cross_bullish({macd_hist:.5f})")

        # 3. Volumen creciente adverso (confirma el movimiento en contra)
        if volume is not None and volume_ma is not None and volume_ma > 1e-9:
            price_adverse = (
                (self.side == "BUY"  and exec_price < self.entry_price) or
                (self.side == "SELL" and exec_price > self.entry_price)
            )
            if price_adverse and volume >= volume_ma * self.config.compression_volume_factor:
                signals.append(f"volume_adverse({volume / volume_ma:.2f}x)")

        return signals

    # ------------------------------------------------------------------
    # Cálculo de nuevos SL
    # ------------------------------------------------------------------

    def _compute_expanded_sl(self) -> Optional[float]:
        """
        Calcula el nuevo precio del SL expandido.
        Devuelve None si no hay margen suficiente hasta el hard SL.
        """
        expansion  = self._pts(self.config.expansion_pts)
        min_margin = self._pts(self.config.hard_sl_margin_pts)

        if self.side == "BUY":
            new_sl = self.virtual_sl - expansion
            limit  = self.hard_sl + min_margin
            return new_sl if new_sl >= limit else None
        else:
            new_sl = self.virtual_sl + expansion
            limit  = self.hard_sl - min_margin
            return new_sl if new_sl <= limit else None

    def _compute_compressed_sl(self, exec_price: float) -> Optional[float]:
        """
        Calcula el nuevo SL comprimido: precio actual con offset favorable.
        Siempre respeta la distancia mínima al hard SL.
        Devuelve None si el resultado no mejoraría el SL actual.
        """
        offset     = self._pts(self.config.compression_offset_pts)
        min_margin = self._pts(self.config.hard_sl_margin_pts)
        # Mínimo tick de mejora para garantizar que el SL efectivamente se mueve
        min_improvement = self._pts(1)

        if self.side == "BUY":
            # Candidato: precio - offset (SL más cercano al precio, i.e. más alto)
            candidate = exec_price - offset
            floor     = self.hard_sl + min_margin
            new_sl    = max(candidate, floor)
            # Solo tiene sentido si mejora (sube) el SL actual
            if new_sl <= self.virtual_sl + min_improvement:
                return None
            return new_sl
        else:
            # Candidato: precio + offset (SL más cercano al precio, i.e. más bajo)
            candidate = exec_price + offset
            ceil_     = self.hard_sl - min_margin
            new_sl    = min(candidate, ceil_)
            # Solo tiene sentido si mejora (baja) el SL actual
            if new_sl >= self.virtual_sl - min_improvement:
                return None
            return new_sl

    def _pts(self, points: int) -> float:
        """Convierte puntos del broker a unidades de precio."""
        return points * self.config.pts_to_price

    # ------------------------------------------------------------------
    # Cierre interno
    # ------------------------------------------------------------------

    def _close(self, reason: CloseReason, exec_price: float,
               debug: Dict[str, Any]) -> BarResult:
        self._state = SLState.CLOSED
        logger.info(
            f"[AdaptiveSL] ticket={self.ticket} "
            f"CLOSE reason={reason.value} exec_price={exec_price:.5f}"
        )
        return BarResult(
            action="close",
            reason=reason.value,
            debug={"close_reason": reason.value, "exec_price": exec_price, **debug},
        )


# ---------------------------------------------------------------------------
# INTEGRACIÓN EN S3 — código listo para pegar
# ---------------------------------------------------------------------------
#
# ── 1. Imports y config (una sola vez, al inicializar S3) ─────────────────
#
#   from adaptive_sl_manager import AdaptiveSLManager, AdaptiveSLConfig
#
#   ADAPTIVE_SL_CONFIG = AdaptiveSLConfig(
#       expansion_pts=50,
#       expansion_max_bars=3,
#       expansion_max_seconds=210,
#       expansion_min_signals=2,
#       expansion_proba_threshold=0.45,
#       expansion_rsi_oversold=35.0,
#       expansion_rsi_overbought=65.0,
#       expansion_indecision_body_ratio=0.35,
#       compression_proba_threshold=0.60,
#       compression_offset_pts=30,
#       compression_min_hold_bars=3,
#       compression_cooldown_bars=5,
#       compression_volume_factor=1.3,
#       hard_sl_margin_pts=50,
#       pts_to_price=0.1,
#   )
#
#   # Diccionario ticket → manager (vive en el objeto S3)
#   self.adaptive_managers: Dict[int, AdaptiveSLManager] = {}
#
#
# ── 2. Al abrir una posición ──────────────────────────────────────────────
#
#   manager = AdaptiveSLManager(
#       ticket=result.ticket,
#       side=req_side,                        # "BUY" o "SELL"
#       entry_price=result.price,
#       virtual_sl=virtual_sl_price,
#       virtual_tp=virtual_tp_price,
#       hard_sl=broker_emergency_sl,
#       config=ADAPTIVE_SL_CONFIG,
#   )
#   self.adaptive_managers[result.ticket] = manager
#
#
# ── 3. En el loop de cada bar/tick ────────────────────────────────────────
#
#   for ticket, manager in list(self.adaptive_managers.items()):
#
#       result = manager.on_bar(
#           bid=current_bid,
#           ask=current_ask,
#           bar_open=bar.open,
#           bar_high=bar.high,
#           bar_low=bar.low,
#           bar_close=bar.close,
#           atr=indicators.atr,
#           rsi=indicators.rsi,                      # None si no disponible
#           macd_hist=indicators.macd_hist,           # None si no disponible
#           macd_hist_prev=indicators.macd_hist_prev, # None si no disponible
#           volume=bar.volume,                        # None si no disponible
#           volume_ma=indicators.volume_ma,           # None si no disponible
#           proba_long=last_model_output.p_buy_cal,
#           proba_short=last_model_output.p_sell_cal,
#           ts=bar.timestamp,
#       )
#
#       if result.action == "close":
#           self._execute_close(ticket)
#           self.log_event("ADAPTIVE_SL_CLOSE",
#                          ticket=ticket,
#                          reason=result.reason,
#                          **result.debug)
#           del self.adaptive_managers[ticket]
#
#       elif result.action == "update_sl":
#           # Actualizar el SL virtual en tu estructura de posición
#           self.positions[ticket].virtual_sl_price = result.new_virtual_sl
#           self.log_event("ADAPTIVE_SL_UPDATE",
#                          ticket=ticket,
#                          new_virtual_sl=result.new_virtual_sl,
#                          reason=result.reason,
#                          **result.debug)
#
#
# ── 4. Al cerrar una posición por cualquier motivo externo ────────────────
#      (manual, BE, trailing, partial close, max_hold_seconds, etc.)
#
#   if ticket in self.adaptive_managers:
#       self.adaptive_managers[ticket].on_closed()
#       del self.adaptive_managers[ticket]
#
# ---------------------------------------------------------------------------