# -*- coding: utf-8 -*-
"""
adaptive_tp_manager.py
======================
Gestión dinámica del Take Profit virtual para posiciones abiertas en S3.

Cambios v24.0:
  FIX-1: _bars_open ahora cuenta velas M1 reales (cambio de minuto en ts),
          no llamadas al monitor. Corrige compression_min_hold_bars que
          expiraba en 0.6s con monitor a 5Hz.
          Nuevo atributo _last_bar_minute para tracking de vela actual.

Simétrico al AdaptiveSLManager. Implementa dos mecanismos complementarios:

  A) COMPRESIÓN del TP (capturar antes de que rebote)
     Cuando el precio se acerca al TP pero el modelo pierde convicción,
     acerca el TP al precio actual para asegurar la ganancia disponible.
     Útil en régimen range/transition donde los TP raramente se alcanzan limpio.

  B) EXTENSIÓN del TP (dejar correr con momentum)
     Cuando el precio toca el TP con señales de continuación fuertes,
     extiende el objetivo en lugar de cerrar. Solo una extensión por posición.
     Al extender, comprime simultáneamente el SL al nivel del trailing actual
     para no devolver lo ganado.
     Útil en régimen trend donde el primer TP suele ser un punto de aceleración.

Reglas de seguridad invariantes:
  - El TP comprimido nunca puede alejarse del precio (solo se acerca).
  - El TP extendido nunca puede superar hard_tp_max (si se configura).
  - La extensión solo puede activarse UNA vez por posición.
  - Compresión y extensión son mutuamente excluyentes por estado.
  - Al extender el TP se sincroniza un nuevo SL sugerido al llamador.
  - Si no hay señales de continuación en el TP, el cierre se delega
    al sistema estándar de S3 (este manager devuelve action="none").

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

class TPState(Enum):
    NORMAL      = auto()   # Monitorización estándar, compresión activa
    IN_TP_ZONE  = auto()   # Precio ha entrado en la zona de proximidad al TP
    EXTENDED    = auto()   # TP ampliado, nueva meta activa
    CLOSED      = auto()   # Posición cerrada, objeto inerte


class TPCloseReason(Enum):
    VIRTUAL_TP            = "VIRTUAL_TP"             # TP tocado, sin extensión
    VIRTUAL_TP_COMPRESSED = "VIRTUAL_TP_COMPRESSED"  # TP comprimido tocado
    EXTENSION_TP          = "EXTENSION_TP"           # TP extendido alcanzado
    EXTENSION_SL          = "EXTENSION_SL"           # SL comprimido tras extensión
    VIRTUAL_SL            = "VIRTUAL_SL"             # SL tocado (fallback)


# ---------------------------------------------------------------------------
# Configuración
# ---------------------------------------------------------------------------

@dataclass
class AdaptiveTPConfig:
    """
    Parámetros del gestor adaptativo de TP. Todos los valores en puntos del
    broker (misma unidad que virtual_sl_points / virtual_tp_points en S3).

    Ejemplo para XAUUSD M1 (1 punto = 0.1 precio):
        tp_proximity_zone_pts = 30   →  3.0 USD de zona de alerta antes del TP
        compression_offset_pts = 15  →  1.5 USD de margen al comprimir
        extension_pts = 50           →  5.0 USD de extensión del objetivo
    """

    # ── Compresión (capturar antes del rebote) ─────────────────────────────
    compression_enabled: bool = True

    # Distancia al TP (en puntos) a partir de la cual se evalúan señales
    # de agotamiento para comprimir el TP hacia el precio actual.
    tp_proximity_zone_pts: int = 30

    # Número mínimo de señales de agotamiento para activar la compresión
    compression_min_signals: int = 2

    # El modelo debe tener proba del lado OPUESTO >= este umbral (convicción
    # de que el movimiento se va a girar) para contar como señal.
    # 2026-05-20: bajado de 0.45 a 0.15 (P95 LONG con calibrador Platt).
    # El valor original 0.45 era inalcanzable (cal_max LONG con Platt=0.35,
    # SHORT=0.23). Ver INC-2026-05-20 anexo A.
    compression_proba_threshold: float = 0.15

    # RSI en zona extrema para el lado de la posición (agotamiento)
    compression_rsi_overbought: float = 70.0   # para BUY (subida agotada)
    compression_rsi_oversold: float   = 30.0   # para SELL (bajada agotada)

    # Ratio máximo cuerpo/rango de la vela para considerar "indecisión"
    compression_indecision_body_ratio: float = 0.35

    # Volumen decreciente: factor sobre la media para contar como agotamiento
    # (< 1.0 → volumen por debajo de la media)
    compression_volume_exhaustion_factor: float = 0.80

    # Offset desde el precio actual donde se pone el TP comprimido
    # (en puntos, en dirección favorable: entre el precio y el TP original)
    compression_offset_pts: int = 15

    # Velas mínimas desde la apertura antes de evaluar compresión
    compression_min_hold_bars: int = 3

    # Cooldown en velas tras una compresión (evita comprimir en cada tick)
    compression_cooldown_bars: int = 3

    # ── Extensión (dejar correr) ───────────────────────────────────────────
    extension_enabled: bool = True

    # Puntos extra añadidos al TP cuando se detecta continuación
    extension_pts: int = 50

    # Número mínimo de señales de continuación para activar la extensión
    extension_min_signals: int = 2

    # El modelo debe tener proba del lado de la posición >= este umbral
    # 2026-05-20: bajado de 0.60 a 0.20 (P99 LONG con calibrador Platt).
    # El valor original 0.60 era inalcanzable. Extensión es "dejar correr"
    # cuando el momentum persiste — mantenemos P99 (~1% activación) en vez
    # de P95 porque extender TP es decisión de mayor compromiso (más
    # exposición). Ver INC-2026-05-20 anexo A.
    extension_proba_threshold: float = 0.20

    # RSI: zona "caliente" que confirma que el movimiento tiene fuerza
    extension_rsi_hot_buy:  float = 55.0   # RSI > umbral para BUY (fuerza alcista)
    extension_rsi_hot_sell: float = 45.0   # RSI < umbral para SELL (fuerza bajista)

    # Volumen creciente: factor sobre la media para confirmar momentum
    extension_volume_factor: float = 1.20

    # Offset del SL sugerido tras extender: cuántos puntos por detrás del
    # precio actual se pone el nuevo SL (protege lo ganado sin cortarlo)
    extension_sl_trail_pts: int = 30

    # ── Seguridad ──────────────────────────────────────────────────────────
    # TP máximo absoluto (en precio). 0 = sin límite.
    # Útil para evitar objetivos absurdos en extensiones encadenadas.
    hard_tp_max: float = 0.0

    # Factor de conversión puntos → precio
    # XAUUSD con broker en décimas: 1 punto = 0.1 precio → pts_to_price = 0.1
    pts_to_price: float = 0.1


# ---------------------------------------------------------------------------
# Señales de agotamiento (para compresión)
# ---------------------------------------------------------------------------

@dataclass
class ExhaustionSignals:
    """Resultado de la evaluación de señales de agotamiento del movimiento."""
    proba_ok:      bool = False   # modelo pierde convicción en el lado actual
    rsi_ok:        bool = False   # RSI en zona extrema (agotamiento)
    indecision_ok: bool = False   # vela de indecisión
    volume_ok:     bool = False   # volumen decreciente
    macd_ok:       bool = False   # MACD aplanándose / girando

    @property
    def count(self) -> int:
        return sum([self.proba_ok, self.rsi_ok, self.indecision_ok,
                    self.volume_ok, self.macd_ok])

    def to_dict(self) -> Dict[str, Any]:
        return {
            "proba_ok":      self.proba_ok,
            "rsi_ok":        self.rsi_ok,
            "indecision_ok": self.indecision_ok,
            "volume_ok":     self.volume_ok,
            "macd_ok":       self.macd_ok,
            "count":         self.count,
        }


# ---------------------------------------------------------------------------
# Resultado de on_bar
# ---------------------------------------------------------------------------

@dataclass
class TPBarResult:
    """
    Lo que S3 debe hacer tras procesar un bar con el AdaptiveTPManager.

    action:
        "none"        → no hacer nada, seguir esperando
        "close"       → cerrar la posición ahora (TP comprimido o extendido alcanzado)
        "update_tp"   → actualizar el virtual TP al valor new_virtual_tp
        "update_both" → actualizar TP Y SL simultáneamente (tras extensión)

    new_virtual_tp    nuevo precio del TP (si action in ["update_tp", "update_both"])
    new_virtual_sl    nuevo precio del SL sugerido (solo si action="update_both")
    reason            razón del cierre o del movimiento
    debug             dict con info de diagnóstico para el log de S3
    """
    action: str
    new_virtual_tp: Optional[float] = None
    new_virtual_sl: Optional[float] = None
    reason: Optional[str] = None
    debug: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# AdaptiveTPManager
# ---------------------------------------------------------------------------

class AdaptiveTPManager:
    """
    Gestor de TP adaptativo para UNA posición abierta.

    Ciclo de vida:
        1. Instanciar al abrir la posición.
        2. Llamar on_bar() en cada vela/tick mientras la posición esté abierta.
        3. Actuar según el TPBarResult devuelto.
        4. Llamar on_closed() cuando la posición se cierra por cualquier motivo.

    Una instancia = una posición. No reutilizar entre posiciones.

    Relación con el sistema estándar de S3:
        - Si on_bar() devuelve action="none" cuando el precio está en el TP,
          el sistema estándar de S3 debe proceder con el cierre normal.
          El AdaptiveTPManager solo intercepta cuando hay señales claras.
        - Si devuelve action="close", S3 debe cerrar inmediatamente.
        - Si devuelve action="update_tp" o "update_both", S3 actualiza los
          niveles y continúa monitorizando.
    """

    def __init__(
        self,
        ticket: int,
        side: str,             # "BUY" o "SELL"
        entry_price: float,
        virtual_tp: float,     # TP virtual original (en precio)
        virtual_sl: float,     # SL virtual actual (en precio), para cálculos de R
        config: AdaptiveTPConfig = AdaptiveTPConfig(),
    ):
        self.ticket = ticket
        self.side = side.upper()
        self.entry_price = entry_price
        self.config = config

        # TP/SL actuales (pueden modificarse durante la vida de la posición)
        self.virtual_tp = virtual_tp
        self.virtual_tp_original = virtual_tp
        self.virtual_sl = virtual_sl   # se actualiza desde S3 si trailing/BE lo mueven

        # Estado de la máquina de estados
        self._state: TPState = TPState.NORMAL

        # Contadores
        self._bars_open: int = 0
        self._compression_cooldown: int = 0

        # FIX v24.0: tracking de barra actual para que _bars_open cuente velas M1
        # reales y no llamadas al monitor (que corre a 5Hz).
        self._last_bar_minute: Optional[int] = None

        # Control de extensión (solo una por posición)
        self._extension_used: bool = False

        # SL sugerido tras extensión (para que S3 lo aplique)
        self._extension_sl_suggested: Optional[float] = None

        logger.info(
            f"[AdaptiveTP] ticket={ticket} side={side} "
            f"entry={entry_price:.5f} vtp={virtual_tp:.5f} vsl={virtual_sl:.5f}"
        )

    # ------------------------------------------------------------------
    # Propiedades
    # ------------------------------------------------------------------

    @property
    def state(self) -> TPState:
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
    ) -> TPBarResult:
        """
        Procesa un nuevo bar/tick. Devuelve la acción que S3 debe ejecutar.

        Importante: cuando este método devuelve action="none" y el precio
        ha tocado el TP, S3 debe proceder con el cierre estándar. El manager
        solo devuelve "close" cuando hay una razón adaptativa específica
        (TP comprimido alcanzado, SL post-extensión alcanzado).

        Parámetros
        ----------
        bid / ask           Precio de venta/compra actual
        bar_*               OHLC del bar actual (o último bar cerrado)
        atr                 ATR actual
        rsi                 RSI del bar actual (opcional pero recomendado)
        macd_hist           Histograma MACD actual (opcional)
        macd_hist_prev      Histograma MACD del bar anterior (opcional)
        volume              Volumen del bar actual (opcional)
        volume_ma           Media de volumen N barras (opcional)
        proba_long/short    Probabilidades calibradas del modelo S2 (opcional)
        ts                  Timestamp Unix del bar (usa time.time() si None)
        """
        if self._state == TPState.CLOSED:
            return TPBarResult(action="none", debug={"state": "CLOSED"})

        ts = ts or time.time()

        # FIX v24.0: _bars_open cuenta velas M1 reales, no llamadas al monitor.
        # El monitor corre a 5Hz; sin este guard, compression_min_hold_bars=3
        # expiraba en 0.6s en lugar de 3 minutos.
        _current_minute = int(ts // 60)
        if self._last_bar_minute is None or _current_minute != self._last_bar_minute:
            self._bars_open += 1
            self._last_bar_minute = _current_minute
            if self._compression_cooldown > 0:
                self._compression_cooldown -= 1

        # Precio de ejecución relevante según el lado
        # BUY cierra vendiendo → bid; SELL cierra comprando → ask
        exec_price = bid if self.side == "BUY" else ask

        # ── Máquina de estados ────────────────────────────────────────────
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

        if self._state == TPState.NORMAL:
            return self._handle_normal(**kwargs)
        elif self._state == TPState.IN_TP_ZONE:
            return self._handle_in_tp_zone(**kwargs)
        elif self._state == TPState.EXTENDED:
            return self._handle_extended(**kwargs)

        return TPBarResult(action="none")

    def on_closed(self) -> None:
        """Notificar que la posición fue cerrada externamente."""
        self._state = TPState.CLOSED
        logger.info(f"[AdaptiveTP] ticket={self.ticket} cerrado externamente")

    def sync_sl(self, new_sl: float) -> None:
        """
        Sincronizar el SL virtual actual desde S3 (BE, trailing, compresión SL).
        Necesario para que los cálculos de extensión usen el SL real vigente.
        """
        self.virtual_sl = new_sl

    def snapshot(self) -> Dict[str, Any]:
        """Estado interno completo para logging y diagnóstico."""
        return {
            "ticket":               self.ticket,
            "side":                 self.side,
            "state":                self._state.name,
            "virtual_tp":           self.virtual_tp,
            "virtual_tp_original":  self.virtual_tp_original,
            "virtual_sl":           self.virtual_sl,
            "bars_open":            self._bars_open,
            "extension_used":       self._extension_used,
            "compression_cooldown": self._compression_cooldown,
            "extension_sl_suggested": self._extension_sl_suggested,
        }

    # ------------------------------------------------------------------
    # Handlers de estado
    # ------------------------------------------------------------------

    def _handle_normal(self, *, exec_price, bar_open, bar_high, bar_low,
                       bar_close, atr, rsi, macd_hist, macd_hist_prev,
                       volume, volume_ma, proba_long, proba_short, ts) -> TPBarResult:
        """
        Estado normal: monitoriza si el precio entra en la zona de proximidad
        al TP y evalúa señales de agotamiento para comprimir.
        """

        # ── ¿El precio tocó el TP? ────────────────────────────────────────
        tp_hit = (
            (self.side == "BUY"  and exec_price >= self.virtual_tp) or
            (self.side == "SELL" and exec_price <= self.virtual_tp)
        )
        if tp_hit:
            if self.config.extension_enabled and not self._extension_used:
                # Transicionar a IN_TP_ZONE y evaluar extensión en el mismo bar
                self._state = TPState.IN_TP_ZONE
                return self._handle_in_tp_zone(
                    exec_price=exec_price,
                    bar_open=bar_open, bar_high=bar_high,
                    bar_low=bar_low, bar_close=bar_close,
                    atr=atr, rsi=rsi,
                    macd_hist=macd_hist, macd_hist_prev=macd_hist_prev,
                    volume=volume, volume_ma=volume_ma,
                    proba_long=proba_long, proba_short=proba_short,
                    ts=ts,
                )
            # Sin extensión disponible → el sistema estándar cierra
            # Devolvemos "none" para que S3 lo maneje con su lógica normal
            return TPBarResult(action="none", debug={
                "state": "NORMAL",
                "tp_hit": True,
                "extension_available": False,
                "delegate_to_standard": True,
            })

        # ── ¿El precio está en la zona de proximidad al TP? ───────────────
        in_zone = self._in_proximity_zone(exec_price)
        if in_zone and self.config.compression_enabled:
            compression_result = self._try_compress(
                exec_price=exec_price,
                bar_open=bar_open, bar_high=bar_high,
                bar_low=bar_low, bar_close=bar_close,
                rsi=rsi, macd_hist=macd_hist, macd_hist_prev=macd_hist_prev,
                volume=volume, volume_ma=volume_ma,
                proba_long=proba_long, proba_short=proba_short,
            )
            if compression_result is not None:
                return compression_result

        return TPBarResult(action="none", debug={
            "state": "NORMAL",
            "bars_open": self._bars_open,
            "virtual_tp": self.virtual_tp,
            "exec_price": exec_price,
            "in_zone": in_zone,
        })

    def _handle_in_tp_zone(self, *, exec_price, bar_open, bar_high, bar_low,
                           bar_close, atr, rsi, macd_hist, macd_hist_prev,
                           volume, volume_ma, proba_long, proba_short, ts) -> TPBarResult:
        """
        El precio ha tocado el TP. Evalúa señales de continuación para decidir
        si extender el TP o dejar que el sistema estándar cierre.
        """
        signals = self._evaluate_continuation_signals(
            exec_price=exec_price,
            bar_open=bar_open, bar_high=bar_high,
            bar_low=bar_low, bar_close=bar_close,
            rsi=rsi, macd_hist=macd_hist, macd_hist_prev=macd_hist_prev,
            volume=volume, volume_ma=volume_ma,
            proba_long=proba_long, proba_short=proba_short,
        )

        debug_base = {
            "state": "IN_TP_ZONE",
            "exec_price": exec_price,
            "virtual_tp": self.virtual_tp,
            "continuation_signals": signals,
            "signals_count": len(signals),
        }

        if len(signals) >= self.config.extension_min_signals:
            new_tp = self._compute_extended_tp()

            if new_tp is None:
                # hard_tp_max bloqueó la extensión → dejar que el estándar cierre
                logger.info(
                    f"[AdaptiveTP] ticket={self.ticket} extensión bloqueada: "
                    f"hard_tp_max ({self.config.hard_tp_max:.5f})"
                )
                self._state = TPState.NORMAL
                return TPBarResult(action="none", debug={
                    **debug_base,
                    "extension_blocked": "hard_tp_max",
                    "delegate_to_standard": True,
                })

            # Calcular el SL sugerido para proteger lo ganado
            new_sl = self._compute_post_extension_sl(exec_price)

            old_tp = self.virtual_tp
            self.virtual_tp = new_tp
            self._state = TPState.EXTENDED
            self._extension_used = True
            self._extension_sl_suggested = new_sl

            logger.info(
                f"[AdaptiveTP] ticket={self.ticket} EXTENSIÓN activada "
                f"tp {old_tp:.5f} → {new_tp:.5f}  "
                f"sl_sugerido={new_sl:.5f}  "
                f"(signals={len(signals)}/{self.config.extension_min_signals})"
            )

            return TPBarResult(
                action="update_both",
                new_virtual_tp=new_tp,
                new_virtual_sl=new_sl,
                reason="ADAPTIVE_EXTENSION",
                debug={
                    **debug_base,
                    "old_virtual_tp": old_tp,
                    "new_virtual_tp": new_tp,
                    "new_virtual_sl": new_sl,
                },
            )

        # Señales insuficientes → dejar que el sistema estándar cierre en el TP
        self._state = TPState.NORMAL
        return TPBarResult(action="none", debug={
            **debug_base,
            "extension_rejected": (
                f"signals={len(signals)} < min={self.config.extension_min_signals}"
            ),
            "delegate_to_standard": True,
        })

    def _handle_extended(self, *, exec_price, bar_open, bar_high, bar_low,
                         bar_close, atr, rsi, macd_hist, macd_hist_prev,
                         volume, volume_ma, proba_long, proba_short, ts) -> TPBarResult:
        """
        TP extendido activo. Monitoriza el nuevo objetivo y el SL comprimido
        post-extensión. La compresión de TP queda suspendida en este estado.
        """
        debug_base = {
            "state": "EXTENDED",
            "virtual_tp": self.virtual_tp,
            "virtual_tp_original": self.virtual_tp_original,
            "virtual_sl": self.virtual_sl,
            "exec_price": exec_price,
        }

        # ── ¿Se alcanzó el SL comprimido post-extensión? ─────────────────
        # S3 actualiza self.virtual_sl vía sync_sl() cuando mueve el SL,
        # así que aquí siempre tenemos el SL vigente.
        sl_hit = (
            (self.side == "BUY"  and exec_price <= self.virtual_sl) or
            (self.side == "SELL" and exec_price >= self.virtual_sl)
        )
        if sl_hit:
            return self._close(TPCloseReason.EXTENSION_SL, exec_price, debug_base)

        # ── ¿Se alcanzó el TP extendido? ─────────────────────────────────
        tp_hit = (
            (self.side == "BUY"  and exec_price >= self.virtual_tp) or
            (self.side == "SELL" and exec_price <= self.virtual_tp)
        )
        if tp_hit:
            return self._close(TPCloseReason.EXTENSION_TP, exec_price, debug_base)

        return TPBarResult(action="none", debug=debug_base)

    # ------------------------------------------------------------------
    # Compresión del TP
    # ------------------------------------------------------------------

    def _in_proximity_zone(self, exec_price: float) -> bool:
        """¿El precio está dentro de la zona de alerta antes del TP?"""
        zone = self._pts(self.config.tp_proximity_zone_pts)
        if self.side == "BUY":
            return exec_price >= (self.virtual_tp - zone)
        else:
            return exec_price <= (self.virtual_tp + zone)

    def _try_compress(
        self, *, exec_price,
        bar_open, bar_high, bar_low, bar_close,
        rsi, macd_hist, macd_hist_prev,
        volume, volume_ma, proba_long, proba_short,
    ) -> Optional[TPBarResult]:
        """
        Intenta comprimir el TP si hay señales de agotamiento.
        Devuelve TPBarResult(action="update_tp") si comprime, None si no.
        """
        if self._compression_cooldown > 0:
            return None
        if self._bars_open < self.config.compression_min_hold_bars:
            return None

        signals = self._evaluate_exhaustion_signals(
            exec_price=exec_price,
            bar_open=bar_open, bar_high=bar_high,
            bar_low=bar_low, bar_close=bar_close,
            rsi=rsi, macd_hist=macd_hist, macd_hist_prev=macd_hist_prev,
            volume=volume, volume_ma=volume_ma,
            proba_long=proba_long, proba_short=proba_short,
        )

        if signals.count < self.config.compression_min_signals:
            return None

        new_tp = self._compute_compressed_tp(exec_price)
        if new_tp is None:
            return None

        old_tp = self.virtual_tp
        self.virtual_tp = new_tp
        self._compression_cooldown = self.config.compression_cooldown_bars

        logger.info(
            f"[AdaptiveTP] ticket={self.ticket} COMPRESIÓN "
            f"tp {old_tp:.5f} → {new_tp:.5f} "
            f"(signals={signals.count})"
        )
        return TPBarResult(
            action="update_tp",
            new_virtual_tp=new_tp,
            reason="ADAPTIVE_TP_COMPRESSION",
            debug={
                "state": "NORMAL",
                "compression": True,
                "old_virtual_tp": old_tp,
                "new_virtual_tp": new_tp,
                "exec_price": exec_price,
                "exhaustion_signals": signals.to_dict(),
            },
        )

    # ------------------------------------------------------------------
    # Evaluadores de señales
    # ------------------------------------------------------------------

    def _evaluate_exhaustion_signals(
        self, *, exec_price,
        bar_open, bar_high, bar_low, bar_close,
        rsi, macd_hist, macd_hist_prev,
        volume, volume_ma, proba_long, proba_short,
    ) -> ExhaustionSignals:
        """Evalúa señales de agotamiento del movimiento (para compresión)."""
        s = ExhaustionSignals()

        # 1. Modelo pierde convicción en el lado actual
        # (proba del lado opuesto empieza a subir)
        if proba_long is not None and proba_short is not None:
            if self.side == "BUY":
                # Si proba_short sube por encima del umbral, el modelo empieza a
                # ver reversal. No necesitamos que supere 0.5, solo el umbral config.
                s.proba_ok = proba_short >= self.config.compression_proba_threshold
            else:
                s.proba_ok = proba_long >= self.config.compression_proba_threshold

        # 2. RSI en zona de agotamiento
        if rsi is not None:
            if self.side == "BUY":
                s.rsi_ok = rsi >= self.config.compression_rsi_overbought
            else:
                s.rsi_ok = rsi <= self.config.compression_rsi_oversold

        # 3. Vela de indecisión (cuerpo pequeño vs rango)
        bar_range = bar_high - bar_low
        bar_body  = abs(bar_close - bar_open)
        if bar_range > 1e-9:
            s.indecision_ok = (
                (bar_body / bar_range) <= self.config.compression_indecision_body_ratio
            )

        # 4. Volumen decreciente (momentum se agota)
        if volume is not None and volume_ma is not None and volume_ma > 1e-9:
            s.volume_ok = (
                volume <= volume_ma * self.config.compression_volume_exhaustion_factor
            )

        # 5. MACD aplanándose o girando contra la posición
        if macd_hist is not None and macd_hist_prev is not None:
            if self.side == "BUY":
                # histograma positivo pero decreciendo → pérdida de momentum
                s.macd_ok = (
                    macd_hist > 0 and
                    macd_hist < macd_hist_prev
                )
            else:
                # histograma negativo pero aumentando → pérdida de momentum bajista
                s.macd_ok = (
                    macd_hist < 0 and
                    macd_hist > macd_hist_prev
                )

        return s

    def _evaluate_continuation_signals(
        self, *, exec_price,
        bar_open, bar_high, bar_low, bar_close,
        rsi, macd_hist, macd_hist_prev,
        volume, volume_ma, proba_long, proba_short,
    ) -> List[str]:
        """
        Evalúa señales de continuación del movimiento (para extensión).
        Devuelve lista de señales presentes. Lista vacía = sin momentum.
        """
        signals: List[str] = []

        # 1. Modelo con alta convicción en el lado actual
        if proba_long is not None and proba_short is not None:
            if self.side == "BUY":
                if proba_long >= self.config.extension_proba_threshold:
                    signals.append(f"model_long_strong({proba_long:.3f})")
            else:
                if proba_short >= self.config.extension_proba_threshold:
                    signals.append(f"model_short_strong({proba_short:.3f})")

        # 2. RSI en zona "caliente" (fuerza en la dirección)
        if rsi is not None:
            if self.side == "BUY":
                if rsi >= self.config.extension_rsi_hot_buy:
                    signals.append(f"rsi_bullish({rsi:.1f})")
            else:
                if rsi <= self.config.extension_rsi_hot_sell:
                    signals.append(f"rsi_bearish({rsi:.1f})")

        # 3. Volumen creciente (confirma el breakout del TP)
        if volume is not None and volume_ma is not None and volume_ma > 1e-9:
            if volume >= volume_ma * self.config.extension_volume_factor:
                signals.append(f"volume_breakout({volume / volume_ma:.2f}x)")

        # 4. MACD acelerando en la dirección de la posición
        if macd_hist is not None and macd_hist_prev is not None:
            if self.side == "BUY":
                # histograma positivo y creciendo → momentum alcista acelerando
                if macd_hist > 0 and macd_hist > macd_hist_prev:
                    signals.append(f"macd_accelerating_bull({macd_hist:.5f})")
            else:
                # histograma negativo y haciéndose más negativo
                if macd_hist < 0 and macd_hist < macd_hist_prev:
                    signals.append(f"macd_accelerating_bear({macd_hist:.5f})")

        # 5. Vela de breakout (cuerpo grande, no indecisión)
        bar_range = bar_high - bar_low
        bar_body  = abs(bar_close - bar_open)
        if bar_range > 1e-9:
            body_ratio = bar_body / bar_range
            # Cuerpo > 60% del rango = vela de decisión (opuesto a indecisión)
            if body_ratio >= 0.60:
                # Además el cierre debe ser en la dirección correcta
                if self.side == "BUY" and bar_close >= bar_open:
                    signals.append(f"breakout_candle_bull(body_ratio={body_ratio:.2f})")
                elif self.side == "SELL" and bar_close <= bar_open:
                    signals.append(f"breakout_candle_bear(body_ratio={body_ratio:.2f})")

        return signals

    # ------------------------------------------------------------------
    # Cálculo de nuevos niveles
    # ------------------------------------------------------------------

    def _compute_compressed_tp(self, exec_price: float) -> Optional[float]:
        """
        Calcula el nuevo TP comprimido: precio actual + offset hacia el TP.

        El TP comprimido debe estar:
          - Entre el precio actual y el TP original (no lo puede superar)
          - Por encima del precio actual + offset mínimo (para que tenga sentido)
          - Mejorando (acercándose al precio) respecto al TP vigente
        """
        offset      = self._pts(self.config.compression_offset_pts)
        min_tick    = self._pts(1)

        if self.side == "BUY":
            # TP comprimido: precio + offset (entre precio y TP original)
            candidate = exec_price + offset
            # Debe mejorar el TP actual (bajarlo, acercarlo al precio)
            if candidate >= self.virtual_tp - min_tick:
                return None
            # No puede ser inferior al precio actual (sin ganancia)
            if candidate <= exec_price:
                return None
            return candidate
        else:
            # TP comprimido: precio - offset
            candidate = exec_price - offset
            if candidate <= self.virtual_tp + min_tick:
                return None
            if candidate >= exec_price:
                return None
            return candidate

    def _compute_extended_tp(self) -> Optional[float]:
        """
        Calcula el nuevo TP extendido: TP original + extension_pts.
        Devuelve None si hard_tp_max lo bloquea.
        """
        extension = self._pts(self.config.extension_pts)

        if self.side == "BUY":
            new_tp = self.virtual_tp + extension
            if self.config.hard_tp_max > 0 and new_tp > self.config.hard_tp_max:
                return None
            return new_tp
        else:
            new_tp = self.virtual_tp - extension
            if self.config.hard_tp_max > 0 and new_tp < self.config.hard_tp_max:
                return None
            return new_tp

    def _compute_post_extension_sl(self, exec_price: float) -> float:
        """
        Calcula el SL sugerido tras extender el TP.

        Lógica: ponemos el SL a 'extension_sl_trail_pts' detrás del precio
        actual (en dirección desfavorable), garantizando que está por delante
        del SL virtual actual (no lo empeoramos si el trailing ya lo movió más).

        Para BUY: new_sl = max(virtual_sl_actual, exec_price - trail)
        Para SELL: new_sl = min(virtual_sl_actual, exec_price + trail)
        """
        trail = self._pts(self.config.extension_sl_trail_pts)

        if self.side == "BUY":
            candidate = exec_price - trail
            return max(self.virtual_sl, candidate)
        else:
            candidate = exec_price + trail
            return min(self.virtual_sl, candidate)

    def _pts(self, points: int) -> float:
        """Convierte puntos del broker a unidades de precio."""
        return points * self.config.pts_to_price

    # ------------------------------------------------------------------
    # Cierre interno
    # ------------------------------------------------------------------

    def _close(self, reason: TPCloseReason, exec_price: float,
               debug: Dict[str, Any]) -> TPBarResult:
        self._state = TPState.CLOSED
        logger.info(
            f"[AdaptiveTP] ticket={self.ticket} "
            f"CLOSE reason={reason.value} exec_price={exec_price:.5f}"
        )
        return TPBarResult(
            action="close",
            reason=reason.value,
            debug={"close_reason": reason.value, "exec_price": exec_price, **debug},
        )


# ---------------------------------------------------------------------------
# INTEGRACIÓN EN S3 — código listo para pegar en s3_service_v8.py
# ---------------------------------------------------------------------------
#
# ── 1. Import y config (junto al AdaptiveSLManager, en __init__) ──────────
#
#   from adaptive_tp_manager import AdaptiveTPManager, AdaptiveTPConfig
#
#   if _ADAPTIVE_SL_AVAILABLE:   # o un flag propio
#       self.ADAPTIVE_TP_CONFIG = AdaptiveTPConfig(
#           # Compresión
#           tp_proximity_zone_pts=30,        # empezar a evaluar a 3.0 USD del TP
#           compression_min_signals=2,
#           compression_proba_threshold=0.45,
#           compression_rsi_overbought=70.0,
#           compression_rsi_oversold=30.0,
#           compression_indecision_body_ratio=0.35,
#           compression_volume_exhaustion_factor=0.80,
#           compression_offset_pts=15,       # TP comprimido a 1.5 USD del precio
#           compression_min_hold_bars=3,
#           compression_cooldown_bars=3,
#           # Extensión
#           extension_pts=50,                # extender 5.0 USD más allá del TP
#           extension_min_signals=2,
#           extension_proba_threshold=0.60,
#           extension_rsi_hot_buy=55.0,
#           extension_rsi_hot_sell=45.0,
#           extension_volume_factor=1.20,
#           extension_sl_trail_pts=30,       # SL post-extensión a 3.0 USD del precio
#           hard_tp_max=0.0,                 # sin límite absoluto
#           pts_to_price=0.1,
#       )
#       self._adaptive_tp_managers: Dict[int, Any] = {}
#
#
# ── 2. Al abrir posición (en _handle_open, junto al AdaptiveSLManager) ────
#
#   if _ADAPTIVE_TP_AVAILABLE and tr.virtual_tp > 0:
#       tp_manager = AdaptiveTPManager(
#           ticket=ticket,
#           side=side,
#           entry_price=entry_price,
#           virtual_tp=tr.virtual_tp,
#           virtual_sl=tr.virtual_sl_price,
#           config=self.ADAPTIVE_TP_CONFIG,
#       )
#       self._adaptive_tp_managers[ticket] = tp_manager
#
#
# ── 3. En _monitor_loop (ANTES del sistema de confirmación estándar) ───────
#
#   # Llamar DESPUÉS de _run_adaptive_sl para tener el SL actualizado
#   tp_action = self._run_adaptive_tp(tr, bid, ask)
#   if tp_action == "closed":
#       continue    # el adaptativo cerró → saltar al siguiente ticket
#   elif tp_action == "delegated":
#       pass        # sin señales en el TP → dejar que el sistema estándar cierre
#   # En cualquier caso, el sistema estándar sigue evaluando TP/SL después
#
#
# ── 4. Implementar _run_adaptive_tp en S3 ────────────────────────────────
#
#   def _run_adaptive_tp(self, tr, bid, ask) -> str:
#       """
#       Devuelve:
#           "closed"    → posición cerrada por el adaptativo
#           "delegated" → TP tocado pero sin señales; el estándar debe cerrar
#           "none"      → nada que hacer, seguir monitorizando
#       """
#       manager = self._adaptive_tp_managers.get(tr.ticket)
#       if manager is None:
#           return "none"
#
#       # Sincronizar SL actual (trailing, BE, compresión SL pueden haberlo movido)
#       manager.sync_sl(tr.virtual_sl_price)
#
#       # Si el TP cambió desde S3 (partial close ajustó R), sincronizarlo también
#       if tr.virtual_tp > 0 and tr.virtual_tp != manager.virtual_tp:
#           manager.virtual_tp = tr.virtual_tp
#
#       atr = tr.ind_atr if tr.ind_atr and tr.ind_atr > 0 else (tr.point * 100)
#       last_bar = self._get_last_bar_close(tr.symbol)
#       bar_price = bid if tr.side == "BUY" else ask
#       bar_close = last_bar if last_bar else bar_price
#
#       try:
#           result = manager.on_bar(
#               bid=bid, ask=ask,
#               bar_open=bar_close, bar_high=bar_close,
#               bar_low=bar_close,  bar_close=bar_close,
#               atr=atr,
#               rsi=tr.ind_rsi, macd_hist=tr.ind_macd_hist,
#               macd_hist_prev=tr.ind_macd_hist_prev,
#               volume=tr.ind_volume, volume_ma=tr.ind_volume_ma,
#               proba_long=tr.ind_proba_long, proba_short=tr.ind_proba_short,
#               ts=now(),
#           )
#       except Exception as e:
#           self._send({"event": "ADAPTIVE_TP_ERROR", "ticket": tr.ticket, "error": str(e)})
#           return "none"
#
#       if result.action == "close":
#           self._send({
#               "event": "ADAPTIVE_TP_CLOSE",
#               "ticket": tr.ticket, "symbol": tr.symbol,
#               "side": tr.side, "reason": result.reason,
#               "virtual_tp": tr.virtual_tp,
#               **result.debug,
#           })
#           self._close_position_full(tr, bid, ask, f"ADAPTIVE_{result.reason}")
#           return "closed"
#
#       elif result.action == "update_tp":
#           old_tp = tr.virtual_tp
#           tr.virtual_tp = result.new_virtual_tp
#           self._send({
#               "event": "ADAPTIVE_TP_UPDATE",
#               "ticket": tr.ticket, "symbol": tr.symbol,
#               "old_virtual_tp": old_tp,
#               "new_virtual_tp": tr.virtual_tp,
#               "reason": result.reason,
#               **result.debug,
#           })
#           return "none"
#
#       elif result.action == "update_both":
#           # Extensión: actualizar TP y comprimir SL simultáneamente
#           old_tp = tr.virtual_tp
#           old_sl = tr.virtual_sl_price
#           tr.virtual_tp = result.new_virtual_tp
#           # El SL sugerido solo se aplica si mejora el SL actual
#           if result.new_virtual_sl and self._is_improvement(
#                   tr, old_sl, result.new_virtual_sl):
#               tr.virtual_sl_price = result.new_virtual_sl
#           self._send({
#               "event": "ADAPTIVE_TP_EXTENSION",
#               "ticket": tr.ticket, "symbol": tr.symbol,
#               "old_virtual_tp": old_tp,
#               "new_virtual_tp": tr.virtual_tp,
#               "old_virtual_sl": old_sl,
#               "new_virtual_sl": tr.virtual_sl_price,
#               "reason": result.reason,
#               **result.debug,
#           })
#           return "none"
#
#       # result.action == "none": verificar si el TP fue tocado sin señales
#       debug = result.debug or {}
#       if debug.get("delegate_to_standard"):
#           return "delegated"   # señal al llamador: cierra tú con tu lógica
#
#       return "none"
#
#
# ── 5. Al cerrar posición (junto al _remove_adaptive_manager del SL) ──────
#
#   tp_mgr = self._adaptive_tp_managers.pop(ticket, None)
#   if tp_mgr:
#       tp_mgr.on_closed()
#
#
# ── 6. En S3State.on_event de S2 (main_trading_s2_v8.py) ─────────────────
#
#   elif event == 'ADAPTIVE_TP_UPDATE':
#       ticket = int(msg.get('ticket', -1))
#       if ticket in self.positions:
#           new_tp = msg.get('new_virtual_tp', 0)
#           if new_tp:
#               self.positions[ticket].virtual_tp = float(new_tp)
#
#   elif event == 'ADAPTIVE_TP_EXTENSION':
#       ticket = int(msg.get('ticket', -1))
#       if ticket in self.positions:
#           if msg.get('new_virtual_tp'):
#               self.positions[ticket].virtual_tp = float(msg['new_virtual_tp'])
#           if msg.get('new_virtual_sl'):
#               self.positions[ticket].virtual_sl = float(msg['new_virtual_sl'])
#
#   elif event == 'ADAPTIVE_TP_CLOSE':
#       ticket = int(msg.get('ticket', -1))
#       self.positions.pop(ticket, None)
#
# ---------------------------------------------------------------------------