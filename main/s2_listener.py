import json
import logging
import time
import zmq


class S3EventListener:
    """Listener de eventos publicados por S3 via ZMQ SUB.

    Responsabilidades:
      - recibir eventos de S3 y actualizarlos en s3_state (posiciones abiertas/cerradas),
      - mantener contadores de observabilidad por tipo de evento,
      - emitir diagnósticos periódicos al system_logger para depuración.

    Diagnóstico (cada DIAG_INTERVAL_SECS):
      - total de eventos recibidos y desglose por tipo desde el último diagnóstico,
      - tiempo sin recibir eventos (silencio de S3),
      - número de posiciones activas según s3_state,
      - errores consecutivos (si los hay).
    """

    DIAG_INTERVAL_SECS = 120       # emitir diagnóstico cada 2 minutos
    SILENCE_WARN_SECS = 60         # alertar si S3 no publica eventos en 60s

    def __init__(self, sub_socket: zmq.Socket, s3_state, logger=None):
        self.sub_socket = sub_socket
        self.s3_state = s3_state
        self.logger = logger or logging.getLogger("s2_system")

        # ── Contadores acumulados (vida del proceso) ────────────────────────
        self.total_received: int = 0
        self.by_event: dict = {}
        self.last_event_ts: float = 0.0

        # ── Contadores del periodo de diagnóstico ───────────────────────────
        self._diag_last_ts: float = time.time()
        self._diag_period_received: int = 0
        self._diag_period_by_event: dict = {}
        self._diag_period_errors: int = 0

    def diagnostics(self) -> dict:
        """Devuelve un snapshot de observabilidad para consumo externo."""
        now = time.time()
        silence = (now - self.last_event_ts) if self.last_event_ts > 0 else None
        return {
            "total_received": self.total_received,
            "by_event": dict(self.by_event),
            "last_event_ts": self.last_event_ts,
            "silence_secs": round(silence, 1) if silence is not None else None,
            "s3_positions": self.s3_state.n_positions(),
            "s3_sides_open": self.s3_state.sides_open(),
        }

    def _maybe_emit_diag(self, consecutive_errors: int) -> None:
        """Emite diagnóstico periódico si ha pasado DIAG_INTERVAL_SECS."""
        now = time.time()
        elapsed = now - self._diag_last_ts
        if elapsed < self.DIAG_INTERVAL_SECS:
            return

        silence = (now - self.last_event_ts) if self.last_event_ts > 0 else None

        self.logger.info(json.dumps({
            "event": "S3_LISTENER_DIAG",
            "period_secs": round(elapsed, 1),
            "period_events": self._diag_period_received,
            "period_by_event": self._diag_period_by_event,
            "period_errors": self._diag_period_errors,
            "total_received": self.total_received,
            "silence_secs": round(silence, 1) if silence is not None else None,
            "s3_positions": self.s3_state.n_positions(),
            "s3_sides_open": self.s3_state.sides_open(),
            "consecutive_errors": consecutive_errors,
            "ts": now,
        }))

        # Reset contadores del periodo
        self._diag_last_ts = now
        self._diag_period_received = 0
        self._diag_period_by_event = {}
        self._diag_period_errors = 0

    def _maybe_warn_silence(self) -> None:
        """Emite alerta si S3 lleva demasiado tiempo sin publicar eventos."""
        if self.last_event_ts <= 0:
            return
        silence = time.time() - self.last_event_ts
        if silence >= self.SILENCE_WARN_SECS:
            # Solo alertar una vez por periodo de silencio (evitar spam).
            # Usamos un flag que se resetea cuando llega un evento.
            if not getattr(self, "_silence_warned", False):
                self._silence_warned = True
                self.logger.warning(json.dumps({
                    "event": "S3_LISTENER_SILENCE",
                    "silence_secs": round(silence, 1),
                    "last_event_ts": self.last_event_ts,
                    "s3_positions": self.s3_state.n_positions(),
            "s3_sides_open": self.s3_state.sides_open(),
                    "ts": time.time(),
                }))

    def run_forever(self):
        consecutive_errors = 0
        while True:
            try:
                raw = self.sub_socket.recv_string()
                msg = json.loads(raw)

                # ── Contadores acumulados ────────────────────────────────────
                self.total_received += 1
                ev = msg.get("event", "UNKNOWN")
                self.by_event[ev] = self.by_event.get(ev, 0) + 1
                self.last_event_ts = time.time()

                # ── Contadores del periodo ───────────────────────────────────
                self._diag_period_received += 1
                self._diag_period_by_event[ev] = self._diag_period_by_event.get(ev, 0) + 1

                # ── Reset flag de silencio ───────────────────────────────────
                self._silence_warned = False

                # ── Actualizar estado ────────────────────────────────────────
                self.s3_state.on_event(msg)
                consecutive_errors = 0

            except zmq.Again:
                # Timeout de recepción — sin mensaje disponible.
                # Aprovechar para comprobar silencio y emitir diagnóstico.
                self._maybe_warn_silence()
                self._maybe_emit_diag(consecutive_errors)
                continue

            except Exception as e:
                consecutive_errors += 1
                self._diag_period_errors += 1
                self.logger.error(json.dumps({
                    "event": "S3_LISTENER_ERROR",
                    "error": str(e),
                    "consecutive_errors": consecutive_errors,
                    "ts": time.time(),
                }))
                time.sleep(0.1)

            # Emitir diagnóstico periódico también tras procesar evento
            self._maybe_emit_diag(consecutive_errors)