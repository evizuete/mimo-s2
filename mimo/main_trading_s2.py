"""
main_trading_s2_v50_1.py — S2 Trading Signal Service
Versión: 50.1

Cambios v50.1 (sobre v50.0) — hardening ATR fallback + auditoría:

  FIX-1 — endurecer inyección de _atr_fallback desde el pipeline:
    El loop principal ya inyectaba _atr_fallback con el ATR del pipeline, pero
    dependía de truthiness (`if _pip_atr:`). Se sustituye por conversión defensiva
    a float y umbral explícito `> 0`, evitando ambigüedades con tipos raros y
    dejando el valor listo para create_order_request().

  FIX-2 — cascada ATR robusta en create_order_request() + trazabilidad:
    Se sustituye la lógica basada en `or` por una cascada explícita y auditada:
      atr_at_entry → order.atr → _atr_fallback → sintético mínimo seguro.
    Se añaden metadatos `atr_used`, `atr_source`, `atr_min_valid` y
    `atr_fallback_used` para diagnosticar si una orden salió con ATR real,
    fallback del pipeline o ATR sintético.

Cambios v50.0 (sobre v49.2) — RR alineado con el etiquetado de entrenamiento:

  CONTEXTO:
    El entrenamiento se realizó con SL_Barrier=1.5 y TP_Barrier=2.5, lo que implica
    un ratio objetivo teórico de 2.5/1.5 = 1.67R. En v49.2 el cap operativo en
    trend_up/trend_down se dejó en 1.5, ligeramente por debajo del RR implícito en
    el etiquetado. Eso introduce una desalineación semántica entre lo que el modelo
    aprendió como evento positivo y lo que la ejecución en tendencia permite capturar.

  FIX-1 — MAX_RR_BY_REGIME en tendencias confirmado a 1.67:
    Se sube el techo de trend_up y trend_down de 1.5 → 1.67 para alinear el TP
    máximo en regímenes tendenciales con el RR implícito del entrenamiento.
    No se tocan range/low_vol/volatile ni transition_* para conservar el sesgo
    operativo conservador en contextos menos limpios.

  MEJORA-2 — Constantes explícitas del etiquetado de entrenamiento:
    Se añaden TRAINING_SL_BARRIER_R, TRAINING_TP_BARRIER_R y TRAINING_LABEL_RR
    para que el 1.67 no quede como “número mágico” aislado del contexto de
    entrenamiento. Facilita auditoría, trazabilidad y cambios futuros.

Cambios v49.2 (sobre v49.1) — FIX NOT_TRACKED + MAX_RR por régimen:

  FIX-1 — Limpieza de _keepalive_last_sent con _s1_open_tickets como fuente de verdad:
    Causa raíz: cuando S3 cierra un ticket (vSL/MaxHold) y lo elimina de _tracked,
    el listener ZMQ de S2 puede tardar hasta 60s en procesar el POSITION_CLOSED_EXTERNAL
    correspondiente y actualizar S3State. Durante ese intervalo, _active_tickets (derivado
    de s3_state.get_positions()) sigue incluyendo el ticket cerrado, por lo que el bloque
    de limpieza no lo elimina de _keepalive_last_sent y el BLOQUE A sigue enviando MODIFYs
    a ese ticket. S3 responde MODIFY_FAILED NOT_TRACKED porque ya lo eliminó de _tracked.
    Confirmado log 08/04/2026: ticket 332864832 generó 16 MODIFY_FAILED NOT_TRACKED entre
    20:30 y 20:45 pese a haber cerrado por vSL a las 20:29:04.

    Fix: usar _s1_open_tickets (lista directa de MT5, actualizada cada tick desde S1) como
    fuente de verdad adicional en el bloque de limpieza. Un ticket ausente de _s1_open_tickets
    ya no existe en el broker → eliminarlo inmediatamente de _keepalive_last_sent sin esperar
    al listener ZMQ. La condición de limpieza pasa de comparar solo contra _active_tickets
    a comparar contra (_active_tickets | _s1_open_tickets).
    Reducción esperada: de ~60s de lag a ~1-2s (ciclo de polling de S1).

  FIX-2 — MAX_RR_BY_REGIME diferenciado por régimen:
    Causa: todos los regímenes tenían MAX_RR=1.0 (v44.0), lo que recorta el TP en todos
    los casos. Con avg_win=16.6pts y avg_loss=16.1pts (ratio 1.03), el sistema necesita
    WR>49% para ser rentable. El WR observado es 47.4%, generando expectativa ligeramente
    negativa. En sesiones con TREND_DOWN confirmado el precio tiene mayor recorrido y el
    modelo produce sus scores más altos (0.8-1.0), pero el TP capped a RR 1.0 fuerza
    salidas prematuras. Confirmado log 08/04/2026: 10 señales SELL en TREND_DOWN,
    tp_rr_capped=True en todas, RR real ~1.67 recortado a ~1.0.

    Fix: subir MAX_RR a 1.5 en trend_down y trend_up (tendencias confirmadas con mayor
    recorrido esperado), 1.2 en transition_down/transition_up (tendencias en formación),
    reducir a 0.8 en volatile (mayor riesgo de reversión). Range y low_vol mantienen 1.0.
    AVISO: validar sobre ≥30 días de historial antes de confirmar en producción. Si el
    WR en trend_down es <40%, revertir a 1.0. Si es >55%, se puede subir a 1.8.


Cambios v49.1 (sobre v49.0) — BUGFIX: int(ticket) falla si ticket es float string:

  Causa raíz confirmada 18/03/2026: el listener recibe DEBUG_POSITION_OPENED_PUBLISHED
  (emitido por S3 justo después de POSITION_OPENED) pero no ejecuta el bloque
  POSITION_OPENED en on_event. Esto es imposible si fuera un handshake ZMQ tardío
  (perdería ambos mensajes). La única explicación: on_event recibe el POSITION_OPENED
  pero lanza una excepción en la línea 2221 antes de loguear nada.

  La excepción más probable: int(msg['ticket']) falla si ticket viene como float.
  json.loads() convierte números JSON sin decimales a int, pero si el resultado
  del broker en MT5 devuelve el ticket como float (ej: 319547958.0), json_dump
  en S3 lo serializa como "319547958.0" y json.loads en S2 lo parsea como float.
  int(319547958.0) funciona, pero int("319547958.0") lanza ValueError.
  Alternativamente, el campo puede ser None o ausente si result['order'] es None
  en un edge case del broker.

  FIX — Conversión defensiva de ticket en on_event POSITION_OPENED:
    Sustituir int(msg['ticket']) por int(float(msg['ticket'])) para tolerar
    floats serializados como string. Añadir try/except específico que loguea
    el valor exacto de msg['ticket'] y todos los campos del msg si falla,
    sin silenciar el error (a diferencia del except genérico del listener).
    Esto permite identificar el tipo exacto y valor del campo problemático
    en el log de mimo_signals con evento POSITION_OPENED_PARSE_ERROR.


Cambios v49.0 (sobre v48.0) — Fix duplicados BLOQUE A/B + Fix-3 real via listener + debug POSITION_OPENED:

  Contexto: análisis log 18/03/2026 sesión v48. Tres problemas identificados y corregidos.

  FIX-1 — MODIFYs duplicados (POSITION_MODIFIED ×2 por ticket por vela):
    Causa: en el primer tick donde un ticket recibe MODIFY, _keepalive_last_bar_time
    no tiene entrada para ese ticket. BLOQUE A envía MODIFY y registra
    _keepalive_last_bar_time[ticket] = bar_time. BLOQUE B evalúa _new_bar_trigger
    ANTES de que BLOQUE A lo haya actualizado (el set fue calculado en pre-BLOQUE A):
    _last_bar = None → None != bar_time → True → envía un segundo MODIFY idéntico.
    Confirmado: todos los POSITION_MODIFIED en S3 aparecen duplicados con Δ < 1ms.
    Fix: introducir _modified_in_this_tick = set() justo antes del BLOQUE A.
    BLOQUE A añade cada ticket procesado a ese set. BLOQUE B salta cualquier
    ticket que ya esté en _modified_in_this_tick → cero duplicados.

  FIX-2 — Fix-3 real: registrar ticket nuevo desde POSITION_OPENED en el listener:
    El Fix-3 de v48 intentaba MODIFY inmediato via request.get('ticket'), pero
    request={} (el ticket lo asigna el broker post-fill, nunca está en request).
    La forma correcta: cuando el listener recibe POSITION_OPENED de S3, ese
    evento contiene el ticket real. Añadir callback _on_position_opened_cb en
    S3State, invocado desde on_event tras registrar la posición. El loop principal
    asigna el callback tras crear s3_state; el callback popula _keepalive_last_sent
    con ts=0 para ese ticket (→ MODIFY en el siguiente tick vía BLOQUE B).
    Eliminar el bloque _new_ticket de v48 (inoperante).

  FIX-3 — Debug log en on_event para diagnosticar n_positions_tracked=0 permanente:
    n_positions_tracked=0 en todos los S3_LISTENER_DIAG pese a que el listener
    recibe cientos de eventos. on_event no lanza excepciones (Fix-1 de v48 lo
    confirma: cero S3_LISTENER_ERROR). Añadir evento DEBUG_POSITION_OPENED_REGISTERED
    emitido en mimo_signals cada vez que on_event registra exitosamente un
    POSITION_OPENED en self.positions. Permite distinguir dos hipótesis:
      A) El evento nunca llega al listener (race condition ZMQ handshake al arrancar).
      B) Llega y se registra, pero algo limpia self.positions justo después.
    Si aparece DEBUG_POSITION_OPENED_REGISTERED → hipótesis B (bug en limpieza).
    Si no aparece → hipótesis A (ZMQ handshake) → solución: ZMQ_CONNECT_DELAY.
    Complementario: S3 v28.4 añade DEBUG_POSITION_OPENED_PUBLISHED justo después
    del _send(POSITION_OPENED), para confirmar que S3 lo publica en absoluto.


Cambios v48.0 (sobre v47.2) — MODIFY inmediato post-OPEN + fixes keepalive + diagnóstico listener:

  Contexto: análisis log 18/03/2026. Ticket 319351168 recibió ADAPTIVE_INDICATORS_STALE
  a los 120s. Causa raíz: el bloque de MODIFYs se ejecuta ANTES del bloque de señales
  en el loop principal, por lo que el ticket recién abierto nunca recibe MODIFY en su
  primer tick. s3_state vacío (n_positions_tracked=0 en todos los S3_LISTENER_DIAG)
  impide que el fallback nivel-1 funcione. El fallback nivel-2 (_keepalive_last_sent)
  tarda hasta KEEPALIVE_INTERVAL_SECS=30s en enviar el primer MODIFY al nuevo ticket.
  Adicionalmente, la condición de limpieza de _keepalive_last_sent permitía que tickets
  ya cerrados permanecieran activos (MODIFY_FAILED NOT_TRACKED en 319350164).

  FIX-1 — Diagnóstico msg_preview en S3_LISTENER_ERROR:
    El except Exception del s3_event_listener loguea en s2_system pero no incluía
    el mensaje que causó el error. Imposible saber qué campo de POSITION_OPENED
    falla silenciosamente y deja s3_state vacío.
    Fix: añadir 'msg_preview': str(msg)[:300] al log de S3_LISTENER_ERROR.

  FIX-2 — Condición de limpieza de _keepalive_last_sent:
    Antes: `if _s3_positions or effective_positions == 0` — no limpiaba cuando
    s3_state estaba vacío pero MT5 reportaba posiciones abiertas (effective_positions>0).
    Resultado: tickets cerrados (ej. 319350164) permanecían en _keepalive_last_sent
    y recibían MODIFYs → MODIFY_FAILED NOT_TRACKED en S3.
    Fix: añadir `or _active_tickets` a la condición. Si el fallback nivel-2 ha
    poblado _active_tickets, sabemos exactamente qué tickets son válidos y podemos
    limpiar los demás sin riesgo.

  FIX-3 — MODIFY inmediato al nuevo ticket tras OPEN (causa raíz del STALE):
    El BLOQUE A/B de MODIFYs se ejecuta antes del bloque de señales. El ticket
    recién abierto no existe en _keepalive_last_sent cuando el BLOQUE A itera,
    por lo que no recibe MODIFY en su primer tick. En el siguiente tick, el
    fallback nivel-2 lo incluye — pero si s3_state sigue vacío y la limpieza
    (Fix-2) elimina el ticket equivocado, el primer MODIFY puede retrasarse
    hasta KEEPALIVE_INTERVAL_SECS=30s o más.
    Fix: inmediatamente después de send_order(OPEN), si request contiene el
    ticket y hay indicadores disponibles, enviar un MODIFY al nuevo ticket y
    registrarlo en _keepalive_last_sent. Garantiza que S3 recibe indicadores
    frescos en el mismo tick del OPEN, eliminando el gap de hasta 30s.
    Nota: si request.get('ticket') es None (broker asigna ticket post-fill),
    la condición `if _new_ticket` protege el bloque y no se envía nada —
    el comportamiento cae al del tick siguiente (ya mejorado por Fix-2).


Cambios v47.0 (sobre v46.0) — sent_ts en ORDER_REQUEST + diagnóstico listener ZMQ:

  FIX-1 — sent_ts en ORDER_REQUEST OPEN:
    S2 añade sent_ts=time.time() justo antes de enviar cada ORDER_REQUEST OPEN.
    S3 v28.3 lo usa para rechazar mensajes del buffer ZMQ con > 30s de antigüedad
    (OPEN_REJECTED_STALE). Esto invalida automáticamente todo el backlog al
    arrancar S3 sin depender de cuántos mensajes haya acumulados.
    El STARTUP_GRACE_BARS de v46 sigue activo como primera capa, pero ya no es
    suficiente por sí solo — sent_ts es la solución definitiva.

  FIX-2 — Diagnóstico profundo del listener ZMQ de eventos S3:
    v47.2: BUGFIX nombre del logger — el listener usaba getLogger("signal")
    pero el logger correcto es getLogger("mimo_signals") (L2578 de setup_signal_logger).
    Todos los S3_LISTENER_DIAG y LOSS_STREAK_SL_REGISTERED se perdían silenciosamente
    porque "signal" no tiene handlers configurados. Confirmado: 0 eventos en el log
    pese a que el listener recibía eventos y el timer de 60s debía haber disparado.

    v47.1: S3_LISTENER_DIAG ahora dispara por CONTEO (cada 200 eventos)
    además de por tiempo (cada 60s). El trigger anterior (zmq.Again=timeout)
    nunca ocurría porque S3 publica HEARTBEAT cada 10s + MODIFYs cada barra,
    manteniendo el listener ocupado permanentemente sin llegar al timeout.
    El campo "trigger" indica si disparó por "time", "count" o "timeout".

    Confirmado 18/03/2026: LOSS_STREAK_SL_REGISTERED=0 pese a 32 SL closes en S3.
    streak_count=0 en todos los LOSS_STREAK_STATE → el listener no recibe eventos.
    Nuevo evento S3_LISTENER_DIAG (cada 60s en el signal log):
      - total_received: nº total de eventos recibidos desde arranque
      - secs_since_last_event: tiempo desde el último evento recibido
      - top_events: top 10 tipos de evento más frecuentes recibidos
      - sl_in_state: tamaño de last_sl_close_by_side por side
      - n_positions_tracked: posiciones que tiene s3_state en este momento
    Si total_received=0 después de varios minutos → el socket SUB no se conecta.
    Si total_received>0 pero sl_in_state=0 → on_event falla silenciosamente.
    Si sl_in_state>0 pero LOSS_STREAK no dispara → bug en la lógica del cooldown.


Cambios v46.0 (sobre v45.1) — startup buffer fix + LOSS_STREAK diagnóstico:

  FIX-1 — STARTUP_OPEN_BLOCKED: gracia de arranque de 1 barra.
    Causa: al iniciar S2, el buffer ZMQ PUSH de S3 contiene ORDER_REQUESTs
    de la sesión anterior con el vTP del config que estaba activo entonces.
    S3 los procesa en los primeros ms de la sesión. Si S2 envía un OPEN en
    la primera barra, S3 lo procesa junto a esos mensajes del buffer — el
    primero que pasa el DEDUP guard de 5s usa el vTP antiguo.
    Confirmado log 18/03/2026: primeras 3 aperturas con RR=2.5 a pesar de
    S2 v45.1 con MAX_RR=1.0. Señales 07:24+ correctas con RR≈1.0.
    Fix: nueva constante STARTUP_GRACE_BARS=1. En las primeras N barras
    vistas (bar_times únicos), cualquier OPEN queda bloqueado con el evento
    STARTUP_OPEN_BLOCKED. Los MODIFYs y keepalives no se ven afectados.
    La 2ª barra (bar_time distinto) ya opera con el buffer vacío.
    Implementación: set _startup_bars_seen acumula bar_times; la guarda se
    activa mientras len(set) <= STARTUP_GRACE_BARS.

  FIX-2 — LOSS_STREAK diagnóstico: dos eventos nuevos en el log.
    Causa conocida: LOSS_STREAK_COOLDOWN nunca dispara pese a decenas de
    SL closes consecutivos. Se desconoce si el listener ZMQ recibe los
    eventos de S3 y alimenta last_sl_close_by_side, o si hay un fallo
    silencioso en on_event().
    Nuevo evento LOSS_STREAK_SL_REGISTERED: se emite en el signal_logger
    cada vez que un cierre SL se registra en last_sl_close_by_side.
    Incluye: ticket, side, close_event, streak_count acumulado, be_armed,
    is_ext_loss. Permite verificar que el listener llena el buffer.
    Nuevo evento LOSS_STREAK_STATE: se emite en el signal_logger en cada
    señal evaluada. Incluye: side, streak_count total, recent_in_window,
    would_trigger, oldest/newest SL ts. Permite ver el estado exacto del
    contador en el momento de cada señal y entender por qué no dispara.


Cambios v45.0 (sobre v44.0) — runner_tight_trail_pts: partial_close + SL ajustado:

  Alternativa a v44 (cierre total al primer TP).
  Mantiene el partial_close (50% al TP) pero elimina el riesgo del runner
  usando un trailing muy ajustado en lugar del trailing normal (~160pts).

  MECANISMO:
    Cuando partial_done=True, S3 sustituye trail_distance por
    RUNNER_TIGHT_TRAIL_PTS=20pts. El SL del runner queda a 20pts del máximo
    favorable — prácticamente breakeven del runner.
    Si el precio sigue moviéndose a favor: el tight trail lo sigue y cierra
    el runner con ganancia adicional.
    Si revierte: pierde solo 20pts en el 50% del runner = 10pts netos.
    Resultado esperado: conserva la ganancia del partial y convierte el runner
    de una fuente de pérdidas en una opción con riesgo casi nulo.

  CAMBIOS vs v44:
    1. RUNNER_TIGHT_TRAIL_PTS = 20  (nueva constante)
    2. partial_close.enabled = True  (reactivado)
    3. partial_close.runner_tight_trail_pts = 20  (enviado a S3)
    4. trail_modes restaurados (runner/trend_soft/range/transition)
       El modo runner vuelve a tener sentido porque hay partial_close activo.
  Requiere: s3_service_v28.py (soporta runner_tight_trail_pts en PartialCloseCfg).

v45.1 — partial_trigger_pct 60%→90% (análisis 17/03/2026):
  Con RR=1.0, TP=200pts. Al 60% el runner necesita 80pts más para llegar al TP
  → alto riesgo de revertir antes con tight_trail=20pts.
  Al 90% el runner necesita solo 20pts más = exactamente el tight_trail margin.
  Si el precio avanza 20pts más: TP completo. Si revierte: pierde 0.05R neto.
  Confirmado con datos: 58 tickets alcanzaron profit_R>=0.9 en la sesión.


Cambios v44.0 (sobre v43.0) — RR=1.0: TP a 1×ATR, sin partial close ni runner:

  Análisis sesión 17/03/2026: 152 trades, WR 4.6%, −128R.
  vSL = 1.0×ATR exacto en todos los trades (sin varianza). El ATR es el rango
  normal de una barra M1 — el stop estaba dentro del ruido habitual del mercado.
  Con RR=2.5 el TP requería 2.5×ATR, raramente alcanzable. WR teórico máximo
  en paseo aleatorio: 28.6%. WR real: 4.6% — por debajo del aleatorio.
  Evidencia clave: 94/152 trades (62%) llegaron a +1R antes de revertir
  (confirmado por PARTIAL_CLOSE_TRIGGERED). De esos 94, solo 7 cerraron en TP
  completo. Los 86 restantes: el runner se cerró en pérdida tras reversión normal.
  Con RR=1.0 y cierre total al primer TP, WR simulado ≈62%, R/trade ≈+0.24.

  FIX-1 — MIN_RR_BY_REGIME y MAX_RR_BY_REGIME: todos los regímenes a 1.0.
    El modelo propone siempre RR=2.5; el cap lo recorta a 1.0. El floor
    garantiza que ningún TP quede por debajo de 1.0×vSL.

  FIX-2 — partial_close desactivado (enabled: False).
    Con TP a 1×ATR no tiene sentido cerrar el 50% y dejar runner.
    86 trades con partial close perdieron el runner — se cierra el 100% al TP.

  FIX-3 — trailing unificado en modo trend_soft (sin runner).
    El modo runner implicaba dejar media posición correr post partial-close.
    Con partial_close desactivado, el runner no tiene sentido.
    trail_mult=0.80, trail_step_mult=0.15 para todos los regímenes.

Cambios v43.0 (sobre v42.0) — FIX case-sensitivity en COUNTER_TREND:

  BUG: el filtro 4 usaba state raw (uppercase) vs sets lowercase -> nunca bloqueaba.
  Fix: _state_lower = str(state).lower() antes del check _is_counter.
  Confirmado log 17/03/2026: 4 señales SHORT en TREND_UP pasaron con v42.


Cambios v42.0 (sobre v41.0) — FIX pérdidas sesión nocturna 16-17/03/2026:

  Contexto: sesión 23:00-07:20 UTC, oro +3745 pts (5002→5038). El modelo generó
  57/76 señales en SHORT (75%) en un mercado fundamentalmente alcista. Resultado:
  WR 6.2%, pérdidas masivas. Tres causas identificadas y corregidas:

  FIX-1 — Bloqueo total de señales contra-tendencia (COUNTER_TREND_BLOCK_TOTAL=True):
    Causa: el filtro v29 penalizaba +0.15 al umbral de score para señales contra-tendencia
    (SELL en TREND_UP, BUY en TREND_DOWN). Con scores de 0.95-1.0 confirmados en log,
    una penalización de 0.15 nunca bloqueaba nada. 12 señales SHORT en TREND_UP pasaron
    el filtro con scores hasta 1.0 mientras el oro subía +2300pts en 4 horas.
    Fix: nueva constante COUNTER_TREND_BLOCK_TOTAL=True — cuando está activa, cualquier
    señal en dirección contraria al régimen queda bloqueada independientemente del score,
    emitiendo COUNTER_TREND_BLOCKED en el log. Se mantiene la lógica de penalización
    anterior como fallback (COUNTER_TREND_BLOCK_TOTAL=False) para facilitar comparación.

  FIX-2 — Filtro de sesión horaria (SESSION_START_UTC=7, SESSION_END_UTC=22):
    Causa: sesión asiática (23:00-07:00 UTC) con volumen bajo, spreads amplios y
    tendencias prolongadas. El modelo tiene escasa representación de estas condiciones
    en su entrenamiento. 8h nocturnas generaron 76 señales con WR 6.2% frente al
    promedio diurno esperado >40%.
    Fix: nuevo filtro 4b OUT_OF_SESSION. Si la hora UTC actual está fuera del rango
    [SESSION_START_UTC, SESSION_END_UTC), la señal queda bloqueada. Horario por defecto:
    07:00-22:00 UTC (sesiones London overlap con NY). Configurable con dos constantes.

  FIX-3 — LOSS_STREAK_COOLDOWN: contabilizar POSITION_CLOSED_EXTERNAL como pérdida:
    Causa: con posiciones duplicadas, cuando el vSL dispara en el ticket A, el ticket B
    (duplicado) cierra como POSITION_CLOSED_EXTERNAL — no como VIRTUAL_SL_TRIGGERED.
    S2's S3State solo registraba en last_sl_close_by_side los cierres VIRTUAL_SL_*,
    por lo que solo la mitad de las pérdidas reales se contabilizaban. Con MAX=2 y
    conteo efectivo de 1 por par, el cooldown nunca llegaba a disparar.
    Fix: POSITION_CLOSED_EXTERNAL también se registra como pérdida si pos.be_armed=False
    (la posición nunca armó BE → nunca fue ganadora → casi seguro perdió).

Cambios v40.0 (sobre v39.0) — FIX case-sensitivity en RR y diagnóstico volume_ma:

  BUG-1 (crítico) — MIN_RR_BY_REGIME y MAX_RR_BY_REGIME nunca se aplicaban:
    Causa: los dicts tienen claves en minúsculas ('range', 'transition_down', etc.)
    pero _regime se construía como str(order.get('state') ...) sin normalizar.
    El modelo devuelve estados en mayúsculas ('RANGE', 'TRANSITION_DOWN', 'TREND_UP').
    'RANGE' != 'range' → .get() siempre caía al _default=2.5/1.0 → ningún cap ni
    mínimo se aplicaba nunca, todos los trades salían con el RR del modelo (2.5).
    Confirmado log 16/03/2026: 5 señales con RANGE/TRANSITION_DOWN, tp_rr_capped=False
    en todas, RR 2.49-2.50 — exactamente el default. El cap MAX_RR existía desde
    v37 pero era inefectivo.
    Fix: añadir .strip().lower() a la construcción de _regime. Una línea, afecta
    tanto MIN_RR como MAX_RR (ambos usan _regime).

  BUG-2 (diagnóstico) — volume_ma ausente en 6/6 señales pese al prefill:
    Causa: _prefill_volume_sma fallaba silenciosamente con Exception capturada
    que no mostraba el error real. Además filtraba _fv > 0, descartando ticks
    con tick_volume=0 (válidos en XAUUSD fuera de sesión activa).
    Fix: mensaje de error completo (antes solo mostraba repr), incluye columnas
    disponibles para diagnóstico, y acepta _fv >= 0 (solo excluye NaN).
    La próxima sesión mostrará en stdout qué falla exactamente.

Cambios v39.0 (sobre v38.0) — FIX variable global en guard anti-duplicado:

  BUG-1 (crítico) — _open_guard_bar_time y _open_guard_sent_ts nunca se actualizaban:
    Causa: en Python, asignar una variable dentro de una función o bloque sin declararla
    con `global` crea una variable LOCAL, sin tocar la variable de módulo del mismo nombre.
    El guard de v38 asignaba _open_guard_bar_time y _open_guard_sent_ts dentro del bloque
    `else:` del `if live_order:`, que está a su vez dentro del `while True:` principal.
    Sin `global`, cada tick creaba variables locales que desaparecían al final del bloque.
    La siguiente iteración leía la variable de módulo que seguía siendo 0 → el guard nunca
    se activaba → las órdenes duplicadas seguían enviándose exactamente igual que en v37.
    Confirmado en producción: tickets 318379717 y 318379718, misma hora (22:27:00),
    mismo precio (5011.06), mismo volumen (0.35) — v38 no los habría bloqueado.
    Fix: añadir `global _open_guard_bar_time, _open_guard_sent_ts` justo antes de las
    asignaciones. Esto garantiza que la variable de módulo se actualiza correctamente
    y el guard funciona en la siguiente iteración del loop.

Cambios v38.0 (sobre v37.0) — FIX órdenes duplicadas:

  BUG-1 (crítico) — Race condition ZMQ: dos órdenes idénticas en ticks consecutivos:
    Causa: S2 corre a 5Hz (200ms/tick). Al enviar una orden de apertura, S3 tarda
    ~50-200ms en abrir la posición y publicar POSITION_OPENED. El s3_event_listener
    (hilo separado) actualiza s3_state con otros ~50-200ms de latencia ZMQ. En esa
    ventana de ~100-400ms llegan 1-2 ticks más con effective_positions todavía en
    su valor anterior, decide_live devuelve la misma señal, y S2 envía una segunda
    orden idéntica. Con MAX_POSITIONS=2, esto produce dos trades simultáneos iguales
    con las mismas características — confirmado en producción.
    Fix: nuevas variables de módulo _open_guard_bar_time (int) y _open_guard_sent_ts
    (float). Al enviar una orden, se registran el bar_time del tick y el timestamp
    actual. Los siguientes ticks del mismo bar_time O dentro de OPEN_GUARD_SECS=5.0s
    quedan bloqueados con razón OPEN_GUARD(...) en el filtro 0-pre, antes de cualquier
    otro filtro y completamente independiente de effective_positions.
    OPEN_GUARD_SECS=5.0 es conservador: el ciclo normal de apertura (S2→S3→MT5→S3→
    s3_event_listener) tarda <500ms en condiciones normales. 5s cubre incluso latencias
    extremas sin bloquear señales legítimas en barras diferentes.

  BUG-2 (menor) — Clave 'state' duplicada en log_signal:
    La clave 'state' aparecía dos veces en el dict de record del logger (L2261-2262).
    Python los acepta silenciosamente (la segunda sobreescribe la primera — mismo valor,
    sin crash ni pérdida de datos) pero es código muerto. Eliminada la línea duplicada.

Cambios v37.0 (sobre v36.0) — RR máximo por régimen:

  MEJORA-1 — Nuevo MAX_RR_BY_REGIME: techo de RR por estado de mercado:
    Causa: todos los trades salían con RR=2.5 fijo porque MIN_RR_BY_REGIME
    actúa solo como suelo (eleva TPs demasiado cortos) pero no había ningún
    techo. El modelo siempre superaba el mínimo y el TP nunca se recortaba.
    En RANGE esto es especialmente dañino: el precio rara vez recorre 2.5R
    antes de rebotar en la banda contraria, y el TP queda inalcanzable.
    Fix: nuevo dict MAX_RR_BY_REGIME y bloque de cap en create_order_request.
    Si tp_points > max_tp_pts, el TP se recalcula al máximo del régimen.
    El cap se aplica DESPUÉS del suelo mínimo — si min > max (configuración
    errónea) el suelo prevalece y el cap no actúa (_max_tp_pts = 0 guard).
    Valores provisionales (revisar con ≥ 100 trades):
      RANGE:         1.5  (era 2.5 fijo — mayor WR esperado con target cercano)
      TRANSITION_*:  2.0  (recorte moderado en estado incierto)
      TREND_*:       3.0  (dejar correr tendencias con alta convicción)
      VOLATILE:      1.5  (objetivos conservadores)
      _default:      2.5  (sin cambio para regímenes no listados)
    Nuevo campo de auditoría en SIGNAL_SENT:
      'tp_rr_capped':     True si el TP fue recortado por MAX_RR
      'tp_max_rr_regime': valor MAX_RR aplicado en ese régimen

Cambios v36.0 (sobre v35.0) — FIX ruta de logs:

  FIX-1 — Logs escritos fuera del directorio de la aplicación:
    Causa: setup_signal_logger usaba log_dir='../logs' relativo al CWD en el
    momento de ejecución. Si S2 se lanzaba desde un directorio distinto al de
    la aplicación (p.ej. desde ~/ o con un script de arranque que cambia el CWD),
    los ficheros signals_*.jsonl y s2_system_*.jsonl acababan en ubicaciones
    inesperadas y difíciles de encontrar.
    Fix: log_dir se resuelve ahora como Path(__file__).parent / 'logs', es decir,
    siempre dentro del directorio donde está el propio script main_trading_s2.py,
    independientemente de desde dónde se lance el proceso.
    El call site ya no pasa log_dir explícitamente — usa el default.

Cambios v35.0 (sobre v34.0) — FIXES post-análisis log 16/03/2026:

  FIX-1 — KEEPALIVE_INTERVAL_SECS reducido de 60 → 30s:
    Causa: con KEEPALIVE_ON_NEW_BAR=True (v33), el fallback de tiempo solo actúa
    si no ha habido cambio de barra. Para scalping con holds medianos de 30-90s,
    el 69% de los trades (38/55 en log 16/03) abrían y cerraban dentro de la misma
    vela M1 sin ver ningún cambio de bar_time. Consecuencia: 38 trades recibieron
    0 MODIFYs → AdaptiveSL operó completamente ciego en indicadores.
    Fix: reducir KEEPALIVE_INTERVAL_SECS de 60 a 30s. El trigger por vela sigue
    siendo el mecanismo principal; el fallback de 30s cubre trades de ≥30s que
    no llegan a ver la siguiente barra. Sigue siendo < IND_STALE_WARN_SECS=120s.

  FIX-2 — _volume_sma pre-rellena desde df_rates al arrancar y tras reconexión:
    Causa: tras cada restart de S2, _volume_sma queda con el buffer vacío y
    volume_ma=None durante las primeras 20 barras (~20 minutos). En log 16/03:
    S2 se reinició 2 veces (12:56 y 13:04), dejando volume_ma=None en 16/55
    señales (todos los trades entre 12:57 y 13:20).
    Fix: nueva función _prefill_volume_sma() que carga las últimas VOLUME_MA_PERIOD
    filas de df_rates y alimenta el buffer antes del primer tick. Se llama:
      1. Tras engine.load_artifacts() en el arranque inicial.
      2. Tras _volume_sma.reset() en cada reconexión ZMQ.
    Si falla la carga de BD, continúa sin error (non-critical — degradación silenciosa
    era el comportamiento anterior, ahora al menos se intenta).

Cambios v34.0 (sobre v33.0) — BUGFIXES extracción de indicadores:

  BUG-1 (crítico) — `or` silenciaba valores 0.0 válidos en current_indicators y _open_indicators:
    Causa: todas las cadenas de fallback usaban el operador `or` de Python.
    En Python, `0.0 or fallback` evalúa 0.0 como False y salta al fallback,
    descartando el valor original aunque sea completamente válido.
    Campos afectados con riesgo real de cero:
      - macd_hist / macd_hist_prev: el cruce de cero del histograma MACD (macd_hist=0.0)
        es la señal de compresión más importante del AdaptiveSL. Con `or`, S3 recibía
        macd_hist=None exactamente en ese bar → la señal de compresión nunca disparaba.
      - proba_long / proba_short: 0.0 es posible cuando el modelo tiene certeza en
        dirección contraria.
      - volume: 0 en ticks sin actividad.
    Fix: nuevo helper _first_not_none(*vals) que usa `is not None` en lugar de
    truthiness. Sustituye todos los `A or B` en current_indicators y _open_indicators.

  BUG-2 (importante) — data.get('atr') siempre None en current_indicators:
    Causa: ATR no es un campo del tick MT5 — S1 no lo incluye en el payload ZMQ.
    La primera fuente de la cadena de ATR era siempre None, lo que obligaba a caer
    a _ind_source.get('atr_at_entry') (solo disponible con señal activa) o _diag.get('atr').
    Fix: eliminado data.get('atr') de la cadena. Nuevo orden: pipeline (_pip_last, 'atr')
    → señal activa (atr_at_entry) → _diag.get('atr').

  BUG-3 (importante) — proba_long/short ausentes entre señales:
    Causa: entre señales (live_order=None), _ind_source = _diag = engine._last_diag.
    Si _last_diag no expone 'proba_long'/'proba_short' (depende de TradingSimulator),
    ambos son None en todos los ticks sin señal activa. Consecuencia: el AdaptiveSL
    nunca puede disparar la compresión model_flip entre señales, precisamente cuando
    el modelo está girando de convicción sin haber emitido señal nueva todavía.
    Fix: añadido _pip(_pip_last, 'proba_long/short') como fuente intermedia,
    por si el pipeline expone esas columnas en df_prepared.



Cambios v32.0 (sobre v31.0) — BUGFIXES post-análisis log 13/03/2026:

  FIX-1 — be_offset_points siempre salía en 13pt por inversión de min/max:
    Causa: la fórmula calculaba _be_offset_cap = max(floor, 0.12*vSL) y luego
    be_offset_points = min(floor, cap). Con cap >= floor siempre, min() devolvía
    floor (13pt con spread=10pt) en el 100% de los trades. El BE quedaba a solo
    13pt del entry en todos los casos — prácticamente sin protección en XAUUSD
    donde el spread es ~10pt.
    Confirmado 13/03/2026: ticket 317672859 SELL llegó a +113pt, BE se armó con
    offset=13pt, precio rebotó y cerró en -49pt con 62pt de slippage.
    Fix: cambiar min() → max() para que be_offset_points escale con el vSL.
    Con vSL=270pt: round(0.12*270)=32pt → be_offset=32pt (era 13pt).

  FIX-2 — Umbral LOW_SCORE_WITH_OPEN_POS subido de 0.20 → 0.35 para segundas entradas:
    Causa: con umbral 0.20, los trades de score [0.20-0.25) podían acumularse
    con posición ya abierta. Análisis 13/03/2026: esos 6 trades generaron
    avg_R=-0.84. Con min_score>=0.35 para segundas entradas se hubieran excluido
    4 de los BUYs perdedores de la racha de las 18:35h (-1R cada uno).
    El umbral 0.35 cubre la zona de pérdidas estructurales sin bloquear señales
    con convicción real (score>=0.35 en segunda entrada es razonable).

  FIX-3 — Cooldown por racha de pérdidas consecutivas (nuevo filtro 2b):
    Causa: el modelo generó 9 BUYs consecutivos entre 18:35 y 18:44h del
    13/03/2026 con scores entre 0.21 y 0.64 mientras el precio bajaba. Ningún
    filtro existente los detuvo: el modelo clasificaba el régimen como
    TRANSITION_UP o RANGE localmente aunque el contexto macro era bajista.
    Fix: nuevo filtro 2b en el bloque pre-envío. Si los últimos LOSS_STREAK_MAX
    (2) cierres del mismo side ocurrieron por VIRTUAL_SL dentro de
    LOSS_STREAK_WINDOW_SECS (300s), bloquear ese side durante
    LOSS_STREAK_COOLDOWN_SECS (300s). Actúa independientemente del score.
    El historial se mantiene en S3State.last_sl_close_by_side (actualizado por
    el s3_event_listener) y el cooldown activo en _loss_streak_cooldown_until.
    Los cierres por TP o TIME_FORCE_CLOSE resetean la racha del side.

Cambios v30.8 (sobre v30.7) — BUGFIX: ADAPTIVE_INDICATORS_STALE por cero MODIFYs:

  FIX-10 — _active_tickets vacío tras restart de S3 (fallback 3 niveles):
    Causa: cuando S3 reinicia, el socket ZMQ PUB/SUB pierde los primeros
    eventos (POSITION_OPENED). s3_state queda vacío → _active_tickets vacío
    → S2 no envía MODIFYs → ADAPTIVE_INDICATORS_STALE en 100% de posiciones.
    Confirmado log 13/03/2026: 28 posiciones, 0 POSITION_MODIFIED en todo el día.
    Fix: fallback 3 niveles para _active_tickets:
      1. s3_state.get_positions()         (fuente canónica — ZMQ events S3)
      2. _keepalive_last_sent.keys()       (tickets de OPENs enviados en sesión)
      3. _s1_open_tickets                  (lista MT5 directa enviada por S1)
    El nivel 3 es el paracaídas: S1 ahora incluye open_tickets=[int] en cada
    tick (lista de tickets de posiciones abiertas según MT5). Si los niveles
    1 y 2 están vacíos, S2 usa la lista de S1 y puebla _keepalive_last_sent
    con ts=0 (→ keepalive inmediato en el mismo tick).

Cambios v30.7 (sobre v30.6) — BUGFIX: margen hardSL fijo 50pts no escalaba con ATR:

  FIX-9 — Emergency SL = vSL + max(3×ATR, 50pts) en lugar de vSL + 50pts fijo:
    Causa: con HARD_SL_MARGIN_PTS=50 fijo, el margen entre vSL y hardSL no
    escalaba con la volatilidad. Con ATR=329pts (sesión 13/03/2026), el margen
    era 50/329 = 0.15×ATR → hardSL a 50pts del vSL. Cualquier sweep de 50pts
    activaba el hardSL del broker antes de que S3 pudiera reaccionar.
    Diseño corregido:
      vSL  = 1×ATR  (nivel de gestión S3, sin cambio)
      hardSL = vSL + max(3×ATR, 50pts)
    Con ATR=329: margen = 987pts → hardSL a 4×ATR del entry (paracaídas real).
    Con ATR=20:  margen = 50pts  (suelo absoluto, comportamiento pre-fix).
    Semántica: si S3 falla/crashea, el broker cierra a 4×ATR; en condiciones
    normales S3 siempre actúa antes de que el precio llegue al hardSL.

Cambios v30.6 (sobre v30.5) — BUGFIX: error 1054 "Unknown column bid" al guardar en BD:

  FIX-8 — Columnas bid/ask/tick_ok incompatibles con schema antiguo de rates:
    Causa: S1 añadió bid, ask, tick_ok al payload ZMQ. S2 los recibe y los mete
    en el df → INSERT falla con MySQL 1054 si la tabla rates no tiene esas columnas.
    Fix en dos partes:
      1. Script SQL: migrate_rates_add_tick_columns.sql — ALTER TABLE para añadir
         las columnas. Ejecutar una vez contra la BD de producción.
      2. Fallback defensivo en S2: al arrancar, inspecciona el schema real de rates.
         Si las columnas no existen (BD antigua / pre-migración), las dropea del df
         antes del INSERT. El check se cachea en db._rates_has_tick_cols para no
         hacer DESCRIBE en cada tick. Log al arranque: "[DB] rates tick columns: True/False".

Cambios v30.5 (sobre v30.4) — BUGFIX: _db_save_async tragaba excepciones silenciosamente:

  FIX-7 — _db_save_async re-raise + continue en fallo de BD:
    Causa raíz del bug "entry_time=14:06 durante 19 barras":
    _db_save_async tenía try/except que imprimía el error pero retornaba None.
    .result() devolvía None sin propagar la excepción → el loop continuaba con
    load_from_database_real → modelo veía siempre la última barra guardada con
    éxito (14:06) → entry_time fija durante tantos ticks como durase el fallo.
    FIX-4 (.result() bloqueante) era necesario pero no suficiente: el save
    fallaba silenciosamente y .result() lo ignoraba.
    Fix en dos partes:
      1. _db_save_async elimina try/except → propaga la excepción al Future.
      2. Loop principal: try/except alrededor de .result() → si falla, imprime
         [ERROR][db-save] y hace `continue` (salta el tick sin llamar al modelo).
    Resultado: si la BD falla, S2 salta el tick y lo logea claramente en lugar
    de generar señales obsoletas durante minutos sin advertencia.

Cambios v30.4 (sobre v30.3) — BUGFIX: SIGNAL_TOO_OLD reportaba age=132 por timezone mismatch:

  FIX-6 — Timezone mismatch en filtro SIGNAL_TOO_OLD:
    Causa: entry_time viene del modelo (BD), donde S2 almacena broker_unix - 7200s
    (UTC). bar_time viene de data["time"] = raw broker unix (UTC+2). Al restar
    ambos unix timestamps directamente, la diferencia incluía 7200s = 120 barras
    de offset fijo. Resultado: age siempre ~120 + barras_reales → siempre bloqueado
    con ages absurdos (132 barras en el log cuando la señal tenía solo 12 reales).
    Fix: ajustar bar_time a UTC antes de comparar: bar_ts = raw_bar_time - 7200.
    Con el fix, ages correctos: 2-6 barras para señales zombie del 13/03/2026.

Cambios v30.3 (sobre v30.2) — Umbral ENTRY_GAP_TOO_LARGE dinámico (1×ATR):

  FIX-5 — MAX_ENTRY_GAP_PTS 100→300 + umbral dinámico 1×ATR:
    Contexto: con FIX-4 (db.save síncrono), el modelo ve la barra recién
    cerrada y bid/ask de S1 es el tick del momento de cierre → gap esperado
    < 5pts en condiciones normales. MAX_ENTRY_GAP_PTS=100 ya no bloquea
    señales legítimas, pero tampoco protege bien contra eventos extremos.
    Cambio: umbral dinámico = max(300, atr_pts) donde atr_pts = ATR de la
    señal en puntos (disponible en live_order["atr"]).
    Semántica: "si el precio se movió más de 1 ATR desde que el modelo
    decidió, la señal caducó" — se adapta automáticamente al régimen de
    volatilidad. MAX_ENTRY_GAP_PTS=300 actúa como fallback si atr no viene.
    El campo block_reason incluye ahora threshold y atr_pts para auditoría.

Cambios v30.2 (sobre v30.1) — BUGFIX RAÍZ: señal repetida por race condition BD:

  FIX-4 — db.save() asíncrono causaba que decide_live() viera barras antiguas:
    Causa: _db_executor.submit() lanzaba el save en un hilo de background y
    load_from_database_real() se ejecutaba inmediatamente a continuación en el
    mismo tick. El save (background) casi nunca terminaba antes que la lectura
    → el modelo veía las mismas 800 barras que el tick anterior → devolvía la
    misma señal con entry/sl/tp idénticos tick tras tick.
    Confirmado como causa raíz de los 5 SIGNAL_BLOCKED ENTRY_GAP_TOO_LARGE
    del 13/03/2026: señal de 14:06 repetida hasta 14:12 (5 barras).
    Fix: guardar el Future de submit() y llamar .result() antes de
    load_from_database_real(). El save tarda <5ms → impacto mínimo.
    Nota: el filtro SIGNAL_TOO_OLD (v30.1) sigue activo como segunda línea
    de defensa para señales repetidas que pudieran escapar por otros motivos.

Cambios v30.1 (sobre v30.0) — BUGFIX: señales rancias bloqueadas múltiples barras:

  FIX-3 — Filtro 5: SIGNAL_TOO_OLD:
    Causa: decide_live() devuelve la misma señal (mismo entry_time/entry/sl/tp)
    mientras el modelo no genere una nueva. En M1, S2 procesa la señal de la
    barra N en la barra N+1 (antigüedad=1, normal). Si en N+2, N+3... el modelo
    sigue sin actualizar, S2 reintentaba la misma señal con geometría obsoleta.
    Confirmado 13/03/2026: señal de las 14:06 reenviada hasta las 14:12
    (5 barras, gaps 234-779pts) mientras el oro bajaba 800pts. El filtro
    ENTRY_GAP_TOO_LARGE la bloqueaba en cada intento, pero el problema raíz
    era que la señal nunca debió reintentarse más de 1 barra después.
    Fix: nuevo filtro 5 (SIGNAL_TOO_OLD). Calcula antigüedad en barras como
    round((bar_time - entry_time) / 60). Si > MAX_SIGNAL_AGE_BARS=1 → BLOCKED.
    El antiguo filtro 5 (ENTRY_GAP_TOO_LARGE) pasa a ser filtro 6.

Cambios v30.0 (sobre v29.0) — Geometría del trade anclada al precio real de fill (bid/ask de S1):

  Problema estructural identificado 13/03/2026:
    S2 calculaba toda la geometría del trade (vSL, emergency SL, BE, trailing)
    anclada a order["entry"] — el precio que el modelo vio en la última barra
    cerrada. MT5 ejecuta al bid/ask real del momento, que puede diferir
    significativamente (caso confirmado: gap=332pts → OPEN_FAILED 10016).
    El problema ocurre cuando gap > min_vsl + hard_sl_margin: el emergency SL
    calculado sobre model_entry queda al lado incorrecto del fill real.

  FIX-1 — Filtro 5: ENTRY_GAP_TOO_LARGE (pre-send block):
    Bloquea señales donde |bid/ask - model_entry| > max(300, atr_pts).
    Referencia: ask para BUY, bid para SELL (precio real de fill esperado).
    Fallback a data["close"] si S1 no envía bid/ask (S1 antiguo).
    En condiciones normales gap < 5pts; nunca bloquea señales legítimas.

  FIX-2 — create_order_request: geometría re-anclada a fill_ref (bid/ask):
    Nuevos params opcionales: bid=, ask= (enviados desde el loop principal).
    fill_ref = ask (BUY) | bid (SELL) | model_entry (fallback S1 sin tick).
    El virtual_sl_price se recalcula como fill_ref ± vsl_pts*point,
    conservando la distancia en puntos del modelo pero anclando el precio
    absoluto al fill esperado. El emergency SL hereda este ancla.
    Resultado: toda la geometría (vSL, emergency SL, BE, trailing) queda
    correctamente posicionada respecto al fill real, eliminando el bug
    estructural independientemente del gap model_entry → real_fill.
    Metadatos nuevos: fill_ref_price, fill_ref_src, model_entry, fill_ref_gap_pts.

  Nota: S3 v20.3 FIX-8 sigue activo como segunda línea de defensa para el
    pequeño gap residual entre envío de S2 y fill real en MT5 (latencia ZMQ).

Cambios v29.0 (sobre v28.0) — BUGFIXES calidad de señal: vSL demasiado ajustado + sesgo contra-tendencia:

  FIX-1 — 9 trades cerrados en <10s: señales con vSL ~ entry enviadas a S3:
    Causa: create_order_request aplica la guardia MIN_VSL_ATR_RATIO, pero en
    casos de ATR casi-cero o señales degeneradas del modelo, el vSL puede
    quedar en MIN_VSL_POINTS=20pts. Con 20pts el AdaptiveSL no tiene margen
    operativo (necesita expansion_pts=25 + hard_sl_margin=20 = 45pts) y el
    precio cruza el vSL por ruido o spread en los primeros segundos.
    Confirmado 13/03/2026: 9/37 trades cerraron en 0-5s del OPEN;
    todos tenían virtual_sl_price ≈ entry.
    Fix: nuevo filtro 3 en pre-send block. Si price_to_points(entry, sl)
    < MIN_VIABLE_TRADE_SL_PTS=50pts → SIGNAL_BLOCKED con reason=VSL_TOO_TIGHT.
    El umbral de 50pts cubre: expansion_pts(25) + hard_sl_margin(20) + 5pts margen.

  FIX-2 — Sesgo SELL en mercado alcista: 30 SELLs vs 7 BUYs (sesión 13/03/2026):
    Causa: gate_by_action_and_state no penaliza suficiente las señales en
    contra-tendencia. El modelo tiene sesgo histórico hacia SELLs en XAUUSD
    (confirmado 13/03/2026: XAUUSD subió 5085→5114 (+29pts) pero el 81% de
    las señales fueron SELLs). Las señales contra-tendencia pueden ser válidas
    (reversiones reales) pero las marginales destruyen valor.
    Fix: nuevo filtro 4 en pre-send block. Si regime ∈ COUNTER_TREND_REGIMES_LONG
    y side=SELL (o viceversa), exigir score >= MIN_SCORE_TO_TRADE + 0.15 = 0.25.
    Emite SIGNAL_BLOCKED con reason=COUNTER_TREND_LOW_SCORE para diagnóstico.
    Constantes modificables: COUNTER_TREND_SCORE_PENALTY, COUNTER_TREND_REGIMES_*.

Cambios v28.0 (sobre v27.0) — BUGFIX: Hard SL calculado desde entry en lugar de desde vSL:

  BUG-A (IMPORTANTE) — Hard SL al lado incorrecto del vSL en ~27% de los trades:
    Causa: emergency_sl_price = entry ± (emergency_atr_mult * ATR). Cuando el
    vSL calculado desde el modelo es mayor que emergency_atr_mult*ATR (posible
    porque MIN_VSL_ATR_RATIO=0.5 y emergency_atr_mult=1.5 comparten el mismo ATR
    base, y los vSLs del modelo pueden ser más amplios que 0.5*ATR), la guardia
     expandía emergency_points
    correctamente en puntos, pero emergency_sl_price ya estaba calculado como precio
    desde el entry — quedando entre el entry y el vSL.
    Confirmado en log 12/03/2026: tickets 316980946, 316982379, 317022688,
    317024839 con hard SL al lado incorrecto (márgenes de -8 a -43 pts reales).
    En estos trades el broker cerraba antes de que S3 pudiera actuar.

  BUG-B (IMPORTANTE) — Margen vSL→hard SL siempre mínimo (1-47 pts reales):
    Causa: incluso en trades sin BUG-A, la combinación de emergency_atr_mult=1.5
    y la guardia a 1.5x producía hard SLs a apenas 1-47 pts del vSL en precio.
    S3 necesita ≥30 pts de margen para activar la expansión anti-sweep
    (AdaptiveSL). Con 1-47 pts: expansion_blocked=hard_sl_margin_insuficiente
    en el 100% de los 5 cierres AdaptiveSL de la sesión 12/03/2026.

  Fix: calcular emergency_sl_price DESDE el vSL añadiendo HARD_SL_MARGIN_PTS
    (50 pts = 5.0 USD) de margen garantizado más allá del nivel virtual.
    Consecuencias:
    - Hard SL siempre más allá del vSL en la dirección correcta (BUG-A resuelto)
    - Margen vSL→hard SL siempre exactamente 50 pts (BUG-B resuelto)
    - S3 AdaptiveSL puede activar expansión anti-sweep: 50 pts > threshold 30 pts
    - emergency_points calculado desde el precio resultante (consistente con S3)


Cambios v26.0 (sobre v25.0) — BUGFIX: Tickets zombie en S3State bloquean MAX_POSITIONS:

  BUG-1 (IMPORTANTE) — S3State.on_event() lista incompleta de eventos de cierre:
    Causa: el método on_event() de S3State elimina un ticket de self.positions
    cuando recibe ciertos eventos de cierre de S3. Sin embargo, la lista de eventos
    manejados era incompleta — faltaban cuatro tipos de cierre que S3 emite:

      - TIME_FORCE_CLOSE_TRIGGERED      (7 ocurrencias en sesión 12/03/2026)
      - ADAPTIVE_VIRTUAL_SL_TRIGGERED   (9 ocurrencias)
      - ADAPTIVE_EXTENSION_SL_TRIGGERED (2 ocurrencias)
      - ADAPTIVE_HARD_SL_TRIGGERED      (1 ocurrencia)

    Cuando S3 cierra una posición por alguno de estos eventos, S2 no recibe
    ningún otro evento de cierre alternativo de forma garantizada. El ticket
    queda como "abierto" en S3State indefinidamente — un "ticket zombie" —
    hasta que casualmente llega un POSITION_CLOSED_EXTERNAL (que solo ocurre
    al abrir el siguiente trade) o hasta fin de sesión.

    Consecuencias medidas en sesión 12/03/2026:
    a) 7 tickets zombie por TIME_FORCE_CLOSE, con duraciones de 60s a 1318s
       (22 minutos el más largo).
    b) effective_positions inflado artificialmente → MAX_POSITIONS(2/2)
       alcanzado sin haber 2 posiciones reales → 39% de ticks bloqueados
       en el log de señales de S2 (21/54 ticks con NO_SIGNAL por MAX_POSITIONS).
    c) Bloque A/B de MODIFYs envía comandos a tickets ya cerrados → S3 los
       ignora, consumiendo capacidad de envío innecesariamente.
    d) Logs continuos de [S3_SYNC] (s3 != mt5) mientras el zombie está activo.

    Fix: añadir los cuatro eventos faltantes al bloque elif de positions.pop()
    en S3State.on_event(). Los eventos ADAPTIVE_VIRTUAL_SL_TRIGGERED,
    ADAPTIVE_EXTENSION_SL_TRIGGERED y ADAPTIVE_HARD_SL_TRIGGERED también se
    añaden aunque no generaron zombies persistentes en el log analizado (porque
    en esos casos llegaba un POSITION_CLOSED_EXTERNAL antes de que se abriera
    el siguiente trade), para garantizar la corrección en todos los escenarios
    posibles, especialmente si max_positions > 1 o el intervalo entre trades
    es largo.

    Nota: TIME_FORCE_CLOSE_TRIGGERED es el único que genera zombies persistentes
    porque S3 lo emite al expirar el tiempo máximo de posición, y no hay garantía
    de que llegue un POSITION_CLOSED_EXTERNAL antes de que S2 quiera abrir otra.

Cambios v25.0 (sobre v24.0) — BUGFIX CRÍTICO: MODIFY/KEEPALIVE nunca llegaban a S3:

  BUG-1 (CRÍTICO) — 0 POSITION_MODIFIED en toda la sesión 12/03/2026:
    Causa: el bloque de envío de MODIFY (normal + keepalive) usa
      _active_tickets = set(s3_state.get_positions())
    para determinar a qué tickets enviar los indicadores. S3State.get_positions()
    devuelve self.positions, que se rellena en on_event() cuando llega un evento
    POSITION_OPENED desde S3 a través del s3_event_listener.

    El problema: el s3_event_listener suscribe al socket events_sub con topic ''
    (todos los eventos). Sin embargo, el socket SUB de ZMQ con suscripción vacía
    ('') solo recibe mensajes que NO tienen prefijo de topic — es decir, mensajes
    crudos. S3 publica con events_socket.send_string(json_dump(msg)), sin ningún
    prefijo de topic. Con ZMQ SUB, suscribirse a b'' significa "recibir todo" en
    ZMQ ≥ 4.x, pero esto depende de cómo el PUB envía los mensajes:
    - Si PUB hace send_string(data) → el mensaje llega como un único frame.
    - Si PUB hace send_multipart([topic, data]) → requiere filtrar por topic.

    En este caso S3 usa send_string() (un frame), y el SUB con b'' debería
    recibirlo. El verdadero problema es otro: la condición de envío del MODIFY
    normal es:

      if current_indicators and (_has_pipeline_data or len(current_indicators) > 1):

    Esta condición evalúa current_indicators ANTES de comprobar _active_tickets.
    Si current_indicators está vacío (no hay datos del pipeline en ese tick),
    se cae al bloque else (keepalive). En el bloque keepalive, el bucle sobre
    _active_tickets sí se ejecuta, pero _keepalive_last_sent.get(ticket, 0)
    devuelve 0 para tickets nuevos, y la condición

      (_now_ts - _last_sent) >= KEEPALIVE_INTERVAL_SECS

    es True desde el primer tick. Sin embargo, _ka_indicators se inicializa con:
      _ka_indicators = _keepalive_last_indicators.get(ticket) or current_indicators

    Si _keepalive_last_indicators[ticket] no existe Y current_indicators está
    vacío, _ka_indicators queda vacío → la condición `if _ka_indicators:` es
    False y el MODIFY keepalive NO se envía.

    Confirmado con el log del 12/03/2026:
    - 0 POSITION_MODIFIED en 90 posiciones abiertas.
    - 29/90 posiciones con ADAPTIVE_INDICATORS_STALE a los 120s exactos.
    - seconds_since_update=120 en todos (≠ None): el OPEN sí llegó con
      indicadores (ind_last_update_ts inicializado por el fix de v23), pero
      después nunca llegó ningún MODIFY.
    - null_fields=[] en todos los stale: los indicadores del OPEN eran completos;
      el AdaptiveSL tenía datos correctos al abrir pero quedó ciego 120s después.

    Fix: separar la lógica en dos bloques independientes con orden explícito:
    1. Si hay datos del pipeline (_has_pipeline_data), enviar MODIFY normal a
       todos los tickets activos, independientemente de current_indicators.
    2. Siempre (después del bloque 1) evaluar keepalive para todos los tickets
       cuyo último MODIFY tiene más de KEEPALIVE_INTERVAL_SECS. Así el keepalive
       actúa como garantía de último recurso aunque el bloque 1 ya haya enviado.
    3. El MODIFY normal actualiza _keepalive_last_sent, por lo que el keepalive
       no enviará un segundo MODIFY redundante en el mismo ciclo.
    4. Si current_indicators está vacío pero _keepalive_last_indicators tiene
       datos previos, el keepalive usa esos datos (comportamiento ya existente).
       Se añade además un fallback que intenta construir indicators mínimos desde
       _pip_last aunque current_indicators esté vacío, para no llegar al
       keepalive con un dict completamente vacío.

  BUG-2 (IMPORTANTE) — Hard SL demasiado amplio: ratio hard_sl/virtual_sl = 4x:
    Análisis log 12/03/2026: emergency_sl_points mediano=771pts, virtual_sl_points
    mediano=193pts → ratio=4.0x en el 100% de los trades.
    Causa: emergency_atr_mult=4.0 en create_order_request. Con ATR típico de
    350-450pts en XAUUSD, el hard SL queda a 1.400-1.800pts del entry. Esto
    equivale a ~$140-180 de exposición máxima por trade, muy superior al riesgo
    virtual de 1R (~193pts = ~$19).
    El hard SL a 4x ATR tiene dos consecuencias negativas:
    a) En gaps o movimientos rápidos (news), el precio puede cruzar el vSL sin
       que S3 lo detecte a tiempo (loop 0.2s) y continuar hasta el hard SL,
       generando pérdidas de 4x el riesgo nominal. Confirma los 96 cierres
       externos observados en el log.
    b) La función de AdaptiveSL _compute_expanded_sl exige margen entre vSL y
       hard SL para poder expandir (anti-sweep). Con hard_sl = 4x ATR y vSL =
       0.5x ATR, el margen disponible es 3.5x ATR ≈ 700-1000pts — mucho más
       espacio del necesario para la expansión (25pts), pero implica que si el
       precio cae más allá del vSL el hard SL no para la pérdida hasta 771pts
       adicionales.

    Fix: reducir emergency_atr_mult de 4.0 → 1.5. Con ATR=350pts:
      emergency_sl = 1.5 * 350 = 525pts (era 1400pts)
      Ratio new: 525 / 193 = 2.7x (era 4.0x)
    El hard SL sigue siendo más amplio que el vSL (garantizado por la guardia
    `if emergency_points <= virtual_sl_points: emergency_points = virtual_sl_points * 1.5`),
    pero actúa como paracaídas real para gaps de 1-2x ATR, no para movimientos
    de 4x ATR que prácticamente nunca ocurren en condiciones normales.
    La reducción de exposición máxima: $140 → $52 por trade en XAUUSD.
    El AdaptiveSL mantiene margen suficiente para expansión anti-sweep (los 25pts
    de expansion_pts siguen siendo < 525-193=332pts de margen disponible).

Cambios v24.0 (sobre v23.0) — CONTEXTO DE MERCADO EN MODIFY/KEEPALIVE:

  MEJORA — Nuevo campo 'context' en todos los comandos MODIFY (normales y keepalive):
    Motivación: S3 gestionaba el ciclo de vida de cada posición completamente ciego
    al contexto de mercado post-apertura. Sabía qué régimen había en el momento del
    OPEN (porque S2 lo incluye en los indicadores iniciales), pero si el régimen
    cambiaba durante la vida del trade, S3 seguía aplicando la misma lógica de
    gestión sin saberlo.

    Tres campos añadidos:

    1. 'regime' (str): régimen actual del modelo (trend_up, trend_down, range,
       transition_up, transition_down, volatile, low_vol, breakout_wait_up/down).
       Fuente: live_order.get('state') si hay señal, _diag.get('state') si no.
       Uso en S3: ajuste dinámico de TIME_FORCE_CLOSE_SECS por régimen (ver S3 v18);
       inhibición de compresión de TP en trend genuino; enriquecimiento del log de
       cierre (permite análisis post-sesión por régimen sin join con log de S2).
       TrackedPos: ind_regime (Optional[str]).

    2. 'score' (float): puntuación del modelo en el tick actual (0-1).
       Fuente: live_order.get('score') si hay señal activa; None si no hay señal.
       El score refleja la "convicción" actual del modelo. Si cae significativamente
       respecto al score de entrada, la tesis original se ha debilitado.
       Uso en S3: enriquecimiento del log de cierre (correlación score-at-close vs
       PnL para calibración); futura señal de exit anticipado si score < umbral.
       TrackedPos: ind_score (Optional[float]).

    3. 'spread' (int): spread actual en puntos (dato del tick MT5).
       Fuente: data.get('spread') — disponible en todos los ticks.
       Uso en S3: inhibición de TIME_FORCE_CLOSE cuando spread > umbral configurable
       (MAX_SPREAD_FOR_CLOSE_PTS). Si el modelo va a ejecutar un TIME_FORCE_CLOSE
       con spread alto, espera hasta que el spread baje o hasta que se cumpla un
       timeout de seguridad (SPREAD_INHIBIT_MAX_WAIT_SECS) para no dejar la posición
       sin gestión indefinidamente. En sesión 11/03/2026, los TIME_FORCE_CLOSE
       tuvieron PnL medio de +260pts — proteger esos cierres del spread tiene impacto
       directo en PnL.
       TrackedPos: ind_spread (Optional[int]).

    Implementación: 'context' es un sub-dict del MODIFY, procesado por
    _handle_modify en S3 via _update_tracked_context() (análogo a
    _update_tracked_indicators). Se mantiene separado de 'indicators' para
    claridad semántica: indicators = señales técnicas del mercado;
    context = metadatos del modelo y del broker en el tick actual.

    En el keepalive forzado se incluye también 'context' con los últimos valores
    conocidos, ya que el régimen y el spread son igual de relevantes cuando el
    pipeline no tiene datos frescos — de hecho, especialmente entonces.

Cambios v23.0 (sobre v22.0) — BUGFIXES POST-ANÁLISIS LOG 11/03/2026 (continuación):

  BUG-1 (CRÍTICO) — ADAPTIVE_INDICATORS_STALE residual: posiciones abren sin
    indicadores aunque el pipeline tenga datos frescos:
    Causa: el dict 'indicators' del comando OPEN (create_order_request líneas
    936-945) extrae los indicadores de order.get('rsi'), order.get('macd_hist'),
    etc. — es decir, de live_order. Si decide_live() no expone esos campos en el
    orden (lo que ocurre frecuentemente cuando el pipeline tiene datos pero
    live_order no los copia todos), la posición abre con ind_last_update_ts=0.
    En S3, ind_last_update_ts==0 activa la segunda rama del stale check
    (línea 2822) exactamente a los IND_STALE_MIN_HOLD+IND_STALE_WARN_SECS = 150s,
    emitiendo ADAPTIVE_INDICATORS_STALE con todos los campos null.
    Confirmado sesión 11/03/2026: 33/70 posiciones con stale, siempre a los 120s
    exactos — patrón sistemático, no aleatorio.

    Fix: fusionar los indicadores de _pip_last en el OPEN, con prioridad
    live_order > _pip_last. Nuevo parámetro 'indicators_override' en
    create_order_request que, si se provee, reemplaza el bloque de indicadores
    interno. En el loop, se construye _open_indicators fusionando ambas fuentes
    antes de cada llamada. Así la posición siempre abre con ind_last_update_ts
    inicializado y los 3 campos clave (rsi, macd_hist, atr) presentes.

  BUG-2 (IMPORTANTE) — WRONG_SIDE_SL_CORRECTED residual con ATR casi-cero:
    Causa: la guardia de v22.0 (`if not _raw_atr or float(_raw_atr) <= 0`)
    solo descarta ATR=0 o None. Si atr_at_entry llega como un float muy pequeño
    (p.ej. 0.0001 — posible en primera barra tras reconexión) la condición es
    False y se usa ese valor. Con atr=0.0001, atr_pts = int(round(0.0001/0.01))
    = 0 → _min_vsl = MIN_VSL_POINTS = 20 → la guardia de mínimo no actúa.
    Resultado: vSLs de 20-33 pts llegan a S3 causando WRONG_SIDE_SL_CORRECTED
    y posterior slippage masivo (ticket 316211980: 524 pts).

    Fix: cambiar umbral de <= 0 a <= MIN_VSL_POINTS * point. Cualquier ATR que
    produzca un _min_vsl < MIN_VSL_POINTS es tratado como inválido y se activa
    el fallback (cascada: _atr_fallback del pipeline → sintético). Esto garantiza
    que _min_vsl sea siempre >= MIN_VSL_POINTS independientemente del ATR recibido.

  MEJORA — KEEPALIVE de indicadores: MODIFY forzado cada KEEPALIVE_INTERVAL_SECS
    aunque el pipeline no haya producido señal nueva:
    Causa raíz del stale: el MODIFY de v22 se envía en cada tick del pipeline,
    pero si el pipeline deja de ejecutarse (reconexión, excepción silenciosa,
    warmup incompleto), los MODIFYs dejan de llegar y S3 entra en degradación
    sin alertas hasta IND_STALE_WARN_SECS=120s. El BUG-1 soluciona el caso de
    apertura; este fix garantiza la renovación continua durante la vida de la posición.

    Fix: nuevo dict _keepalive_last_sent (ticket → ts) que registra el último
    MODIFY enviado por cada posición. En cada tick, ADEMÁS del MODIFY normal, si
    han pasado más de KEEPALIVE_INTERVAL_SECS desde el último MODIFY para algún
    ticket, se envía un MODIFY forzado con los indicadores más frescos disponibles
    (_pip_last con fallback a _last_sent_indicators). Así el gap máximo entre
    MODIFYs es siempre <= KEEPALIVE_INTERVAL_SECS (defecto: 60s), bien por debajo
    del IND_STALE_WARN_SECS=120s de S3.

    Implementación sin estado externo: _keepalive_last_sent se gestiona dentro
    del mismo while True, sin threads adicionales. Coste: O(n_posiciones) por tick
    — insignificante para las 1-5 posiciones típicas del perfil scalping.

Cambios v22.0 (sobre v21.0) — BUGFIXES POST-ANÁLISIS LOG 11/03/2026:

  BUG-1 (CRÍTICO) — BE trigger activado a 1R en lugar de 0.5R:
    Causa: be_trigger_points = int(round(be_trigger_ratio * tp_points)).
    Con tp_points del modelo (RR medio 2.58) y ratio=0.40, el resultado es
    be_trigger ≈ 1R = virtual_sl_points. El trade necesitaba recorrer TODO
    el riesgo en su favor antes de quedar protegido. Confirmado en log
    11/03/2026: be_trigger_points == virtual_sl_points en 46/50 trades;
    solo 10/50 llegaron a BE_ARMED; 31 cerraron por vSL completamente
    desprotegidos (-6,276 pts).

    Fix: be_trigger_points = int(round(be_trigger_ratio * virtual_sl_points)).
    El BE se activa cuando el trade lleva be_trigger_ratio * 1R ganado.
    Nuevo defecto: be_trigger_ratio=0.50 → BE a los 0.5R (mitad del riesgo).
    be_offset_points también calculado sobre virtual_sl_points (no TP).

    Impacto esperado: con RR mínimo=1.0, el BE se activa a mitad de camino
    hacia el TP. Con WR=31% (sesión 11/03), los trades ganadores quedarán
    protegidos desde mucho antes, reduciendo la pérdida de trades revertidos.

  BUG-2 (IMPORTANTE) — ATR=0/None en create_order_request cuando la señal
    llega sin atr_at_entry válido:
    Causa: order['atr_at_entry'] puede ser 0 o None en la primera barra tras
    reconexión o cuando el pipeline no tiene warmup suficiente. En ese caso
    atr_pts=0 → _min_vsl=MIN_VSL_POINTS=20. La guardia de mínimo vSL (v17)
    no actúa. Los 6 casos WRONG_SIDE_SL_CORRECTED del 11/03 tenían atr_pts=null
    y vSL corregido de 29-33 pts, generando slippage extremo al cierre.

    Fix en create_order_request():
      1. Fallback en cascada: atr_at_entry → order.get('atr') → order['_atr_fallback'].
      2. Si todo es 0/None, ATR sintético = MIN_VSL_POINTS * 4 * point (mínimo seguro).
    Fix en loop principal:
      3. Si live_order.atr_at_entry es falsy, inyectar '_atr_fallback' con el ATR
         del pipeline (_pip(_pip_last, 'atr')) antes de llamar a create_order_request.

  BUG-3 (IMPORTANTE) — ADAPTIVE_INDICATORS_STALE en 44% de posiciones:
    Causa: en ticks sin señal activa (live_order=None), _ind_source = _last_diag,
    que no expone rsi/macd_hist. current_indicators solo tenía volume/proba_long/short.
    Los MODIFYs parciales dejaban S3 sin RSI ni MACD. Tras IND_STALE_WARN_SECS=120s
    sin recibir esos campos, S3 emitía ADAPTIVE_INDICATORS_STALE.
    El caso extremo (ticket 316211980, slip=524 pts) fue directo consecuencia:
    AdaptiveSL ciego desde apertura, no pudo comprimir el SL antes del breakout.

    Fix: añadir guardia _has_pipeline_data (comprueba si _pip_last tiene al menos
    rsi/macd_hist/atr) antes del envío de MODIFY. Si el pipeline tiene datos frescos
    (casi siempre: se ejecuta en cada tick), el MODIFY se envía aunque live_order=None.
    current_indicators ya extraía rsi/macd_hist de _pip_last — el fix garantiza que
    el MODIFY se envía también en ticks sin señal.


  MEJORA-9 — TP ajustado a la banda de Bollinger inferior/superior en régimen 'range':
    Contexto: en mercado en rango, el precio raramente tiene recorrido suficiente
    para alcanzar el TP calculado por el modelo. La banda inferior/superior de
    Bollinger actúa como soporte/resistencia natural y representa un objetivo
    más realista que el TP del modelo.

    Lógica:
      - Solo se aplica cuando el régimen detectado es 'range' (o 'low_vol' si
        se desea incluir, configurable con BOLLINGER_TP_REGIMES).
      - Para SELL: si bb_lower está disponible y está entre entry y virtual_tp
        (es decir, bb_lower > virtual_tp y bb_lower < entry), se usa bb_lower
        como nuevo TP siempre que el RR resultante supere BOLLINGER_TP_MIN_RR.
      - Para BUY: análogamente con bb_upper.
      - Si el TP de Bollinger resulta PEOR que el modelo (menos recorrido) o
        no supera el RR mínimo configurado, se mantiene el TP original.
      - El ajuste se aplica DESPUÉS del fix de RR mínimo (P5 v16.0) para no
        entrar en conflicto: si el modelo ya tenía un TP ajustado por RR, el
        ajuste Bollinger solo sobreescribe si mejora el objetivo.

    Parámetros nuevos en create_order_request():
      bb_upper: float | None  — banda superior de Bollinger del último bar
      bb_lower: float | None  — banda inferior de Bollinger del último bar

    Constante nueva:
      BOLLINGER_TP_REGIMES: set  — regímenes donde se aplica el ajuste
      BOLLINGER_TP_MIN_RR:  float — RR mínimo que debe cumplir el TP de Bollinger

    Campos nuevos en metadata (auditoría):
      'bb_tp_applied':     bool  — True si se usó la banda como TP
      'bb_tp_original':    float — TP original antes del ajuste (si aplica)
      'bb_tp_band_value':  float — valor de la banda usada

    Fuente de las bandas: create_order_request() recibe bb_upper/bb_lower
    desde el loop principal de S2, que los extrae de engine._last_df_prepared
    (columnas 'bb_upper' / 'bb_lower' calculadas por el pipeline). Si el
    pipeline no produce esas columnas, los valores son None y el ajuste no
    se aplica (comportamiento idéntico a v20.0).



Cambios v20.0 (sobre v19.0) — OPTIMIZACIÓN DE LATENCIA (cuello de botella principal):

  MEJORA-5 — Inferencia del modelo: de ~7s a <100ms por tick:
    Causa: decide_live() llamaba a predict(df_rates, simulation=False) que construye
    secuencias para TODAS las filas del historial (2048 filas → ~1793 secuencias con
    seq_len_long=256) y luego pasa el array completo a model.predict(). Con batch_size=8192
    eso genera 42 steps × ~61ms = ~2.5s por modelo × 2 modelos = ~7s por tick.
    Confirmado en traza de sesión 11/03/2026:
      42/42 ━━━━━━━━━━━━━━━━━━━━ 4s 61ms/step   ← long
      42/42 ━━━━━━━━━━━━━━━━━━━━ 3s 53ms/step   ← short
    El motor solo usa df_pred.iloc[-1] (la última fila) para decide_at(). Las 1792
    secuencias anteriores se calculan, se pasan al modelo y se descartan sin usarse.

    Fix (en trading_simulator.py): nuevo método predict_live() que:
    1. Ejecuta pipeline.prepare_data() sobre el historial (necesario para warmup de
       indicadores técnicos: EMA-50 necesita 50 barras, ADX/RSI necesitan ~14).
    2. Llama pipeline.create_sequences_by_side() igual que predict().
    3. Sliceea únicamente la ÚLTIMA secuencia de cada array: X[-1:] → shape (1, seq_len, features).
    4. Llama model.predict() con batch_size=1 → 1 único forward pass en GPU: <50ms.
    5. Devuelve un DataFrame de 1 fila con las mismas columnas que predict() para
       total compatibilidad con decide_at() y print_last().
    6. Expone engine._last_df_prepared para que S2 pueda leer RSI/MACD/ATR del
       último bar sin recalcular (ver MEJORA-6).
    decide_live() ahora llama self.predict_live() en lugar de self.predict().
    Impacto estimado: ~7s → <150ms por tick (50× más rápido).

  MEJORA-6 — Eliminación de pipeline.prepare_data() duplicado en S2:
    Causa: desde v14.0 el loop de S2 llamaba explícitamente a pipeline.prepare_data(df_rates)
    DESPUÉS de decide_live() para extraer RSI/MACD/ATR del último bar. Pero decide_live()
    ya ejecutaba prepare_data() internamente dentro de predict(). En v19.0 esto suponía
    dos ejecuciones completas de prepare_data() sobre 2048 filas por tick (~200-400ms total).
    Fix: predict_live() expone el df_prepared ya calculado en engine._last_df_prepared.
    S2 v20 lee engine._last_df_prepared directamente; solo hace fallback a prepare_data()
    si el atributo no existe (compatibilidad hacia atrás con versiones antiguas del engine).
    Impacto: elimina ~100-200ms de prepare_data() redundante por tick.

  MEJORA-7 — Reducción de last_n_rates: 2048 → 800 filas:
    Causa: el mínimo real lo impone el RollingScaler del pipeline (scaler_warmup_size=390,
    confirmado en data_pipeline.py), no los indicadores técnicos. A eso se suma
    seq_len_long=256 para construir la última secuencia → mínimo real = 390 + 256 = 646.
    Cargar 2048 barras era 3× más de lo necesario.
    Nota: se intentó primero con 400 filas pero el df quedaba vacío tras dropna()
    (solo ~10 filas post-warmup del scaler) causando IndexError en live_update_scalers_from_df.
    Fix: last_n_rates=800 (margen de seguridad ~25% sobre el mínimo de 646).
    Impacto: carga BD y prepare_data ~2.5× más rápidos.

  Impacto total estimado v20.0 vs v19.0:
    Antes: ~7.0s por tick (7s inferencia + 0.3s prepare_data×2 + 0.1s BD)
    Después: ~0.15s por tick (<0.1s inferencia + 0.05s prepare_data×1 + 0.01s BD)
    Mejora: ~45× más rápido → señal disponible en <200ms tras el cierre de vela.

  MEJORA-8 — Filtro de spread en S2 (primera línea de defensa):
    Causa: S3 rechazaba órdenes con OPEN_REJECTED_SPREAD cuando el spread superaba
    max_spread_points=15 del perfil scalping. Confirmado sesión 11/03/2026: spread=16pts
    con límite=15 → OPEN_REJECTED_SPREAD. La señal había viajado a S3 innecesariamente.
    S2 no tenía filtro de spread propio, dependiendo únicamente del rechazo de S3.

    Fix: nueva constante MAX_SPREAD_POINTS=20 y chequeo (filtro 0) antes de los filtros
    existentes en el bloque pre-envío. El tick MT5 incluye spread en puntos directamente
    en data['spread'], por lo que no requiere ninguna llamada adicional.
    Si spread > MAX_SPREAD_POINTS: SIGNAL_BLOCKED(SPREAD_TOO_HIGH) en el log de S2
    y la orden no se envía a S3. Trazabilidad completa en signals_YYYYMMDD.jsonl.
    También corregido: filtros 1 y 2 ahora tienen `and block_reason is None` para
    garantizar que no sobreescriben un bloqueo ya establecido por el filtro 0.
    Coordinado con S3 v15.1: max_spread_points subido de 15 → 20 en perfil scalping
    para que ambas capas usen el mismo umbral.

Cambios v19.0 (sobre v18.0) — OPTIMIZACIÓN:

  MEJORA-4 — SMA de volumen recalculada entera en cada tick (O(period) → O(1)):
    Causa: _volume_ma_from_rates() hacía df[col].iloc[-period:].mean() en cada tick,
    recorriendo siempre las 20 barras del periodo completo aunque solo haya cambiado
    una. Con df_rates de 2048 filas, el slice Pandas tiene además overhead de indexado.
    Además, la función se definía DENTRO del while True, recreando el objeto función
    en cada iteración sin necesidad.

    Fix: clase IncrementalSMA con buffer circular (collections.deque, maxlen=period).
    - push(value): añade el nuevo valor, actualiza la suma acumulada en O(1).
      Al expulsar el valor más antiguo (deque lleno) lo resta de la suma;
      al añadir el nuevo lo suma. Sin recorrido del buffer.
    - value: propiedad que devuelve sum/len en O(1), o None si el buffer
      no está lleno todavía (comportamiento idéntico al anterior: espera
      a tener `period` muestras antes de emitir un valor).
    - reset(): vacía el buffer — útil en reconexión o reinicio del engine.
    Instancia global _volume_sma creada ANTES del while True, persiste entre ticks.
    En cada tick se alimenta con la última barra de df_rates (no el df completo).
    Complejidad: O(1) por tick vs O(period) anterior.

    También movidos al bloque de imports del módulo los `import math` e
    `import math as _math` que estaban dentro de _pip() y _volume_ma_from_rates()
    respectivamente (se ejecutaban en cada tick).

Cambios v18.0 (sobre v17.0) — MEJORAS DE OBSERVABILIDAD:

  MEJORA-1 — Excepciones silenciosas en s3_event_listener:
    Causa: el bloque `except Exception: time.sleep(0.1)` descartaba cualquier
    error sin dejar rastro — errores de parseo JSON de mensajes S3 corruptos,
    bugs en on_event() al actualizar s3_state, y posibles errores de socket
    quedaban completamente invisibles. Un bug silencioso aquí puede desincronizar
    s3_state (el contador de posiciones locales) sin que ningún operador lo note.

    Fix: diferenciar entre json.JSONDecodeError (datos corruptos de S3) y
    Exception genérica (bug en lógica de estado). Ambos se registran en el
    nuevo logger 's2_system' con campos: event, ts, error, detail,
    consecutive_errors. Tras _MAX_CONSECUTIVE_ERRORS (10) errores seguidos
    se emite además un evento S3_LISTENER_DEGRADED como alerta.
    Logger dedicado: s2_system_YYYYMMDD.jsonl (misma rotación que signals_*.jsonl).
    Inicializado en setup_signal_logger() para evitar duplicar handlers.

  MEJORA-2 — Campo 'macro_regime' siempre vacío en el log:
    Causa: engine._last_diag incluye 'macro_regime' en su output pero el
    detector de régimen macro no está integrado en el pipeline actual.
    El campo llega vacío en el 100% de los eventos (confirmado log 10/03/2026:
    82/82 eventos con macro_regime="") y genera ruido en el JSONL sin aportar
    información útil para análisis o alertas.

    Fix: en log_signal(), al copiar model_diag al record, se excluye
    explícitamente la clave 'macro_regime' mediante dict comprehension.
    No se elimina del engine ni del _last_diag para no romper otros consumidores.

  MEJORA-3 — Función check_calibration() para diagnóstico de calibración:
    Contexto: análisis del log 10/03/2026 muestra proba_cal siempre < 0.55
    (media ~0.31 long, ~0.33 short) con señales raw más diferenciadas.
    Indicativo de posible sobre-compresión del calibrador isotónico.

    Añadida función check_calibration(signals_jsonl_path) que:
    1. Calcula estadísticos de distribución de proba_cal vs proba_raw.
    2. Mide ratio de compresión raw/cal (alerta si media > 2x).
    3. Construye reliability diagram (calibration curve) con señales SIGNAL_SENT.
    4. Calcula ECE (Expected Calibration Error) — aceptable si < 0.05.
    5. Emite alerta si proba_cal nunca supera 0.55 en toda la sesión.
    Usable desde notebook o CLI sin necesidad de arrancar el engine completo.

Cambios v17.0 (sobre v16.0) — BUGFIXES:

  BUG-1 — WRONG_SIDE_SL persiste en S3 a pesar del fix de S2 v16:
    Causa: El fix de v16 (P4) detecta el wrong-side comparando raw_sl_price vs
    entry, pero el recálculo posterior `entry ± raw_sl_pts * point` puede
    generar otro wrong-side cuando raw_sl_pts es muy pequeño (0, 1 o 2) debido
    a errores de representación de coma flotante. Ejemplo:
      entry=5133.50, raw_sl_pts=1, point=0.01
      → virtual_sl_price = 5133.50 + 1 * 0.01 = 5133.51  (correcto)
      pero si raw_sl_pts=0 y atr/point redondeó a 0:
      → virtual_sl_price = 5133.50 + 0 * 0.01 = 5133.50  (igual al entry → S3 wrong-side)
    Confirmado en log 10/03/2026: 9 WRONG_SIDE_SL_CORRECTED con tickets que sí
    aparecen en POSITION_OPENED con vsl_en_open ya incorrecto.

    Fix (FIX v17.0 — P1): añadir guardia final post-recálculo con mínimo
    absoluto de MIN_VSL_POINTS (20pts) y mínimo proporcional al ATR
    (MIN_VSL_ATR_RATIO=0.5 → 0.5 ATR). Se aplica DESPUÉS del recálculo de
    virtual_sl_price para garantizar que nunca llegue un wrong-side a S3.
    Emite log VSL_MIN_ENFORCED para auditoría cuando se activa el mínimo.

  BUG-2 — Slippage excesivo en VIRTUAL_SL (media 27pts, máx 129pts):
    Causa: El modelo genera vSLs muy ajustados (casos observados: 21-72pts)
    en condiciones de volatilidad alta. Con ATR actual ~0.00086 * 5150 ≈ 443pts
    (post-recalibración de umbrales), un vSL de 21pts es < 5% del ATR:
    cualquier spike lo sobrepasa con slippage masivo antes de que S3 ejecute.
    Confirmado: tickets con slip=129pts tenían vSL de solo 21pts de distancia.

    Fix (FIX v17.0 — P2): el mismo mínimo ATR-based del BUG-1 resuelve también
    este problema. Con MIN_VSL_ATR_RATIO=0.5 y ATR típico de ~350-450pts:
      min_vsl ≈ 175-225pts → elimina los vSL de 21-72pts que generaban slippage.
    El mínimo absoluto de 20pts actúa como fallback cuando ATR no está disponible.
    Añadido campo 'vsl_min_enforced' en metadata para auditoría.

Cambios v16.0 (sobre v15.0) — BUGFIXES:

Cambios v15.0 (sobre v14.0) — BUGFIX:

  BUG-1 — volume_ma siempre None: la columna no existe en el pipeline:
    Causa: v14.0 asumía que feature_builder producía 'volume_ma'. Tras
    inspección de feature_builder.py, data_pipeline.py y trading_simulator.py
    se confirma que el pipeline opera SIN volumen (línea 124 feature_builder:
    '# 3. Microestructura (sin volumen)'). _pip(_pip_last, 'volume_ma')
    devuelve siempre None → volume_ok=False en 100% de cierres AdaptiveSL
    (confirmado log 09/03/2026: 0/8 cierres con volume_ok=True).

    Fix: helper _volume_ma_from_rates(df_rates, period=VOLUME_MA_PERIOD=20)
    calcula la SMA de 'volume' (=tick_volume renombrado) de las últimas 20
    barras directamente desde df_rates, sin depender del pipeline.
    Si df_rates no tiene la columna o hay < 20 filas, devuelve None.

    Columnas confirmadas en feature_builder.py tras inspección directa:
      'rsi'       → _add_rsi()   ✅ existe
      'macd_hist' → _add_macd()  ✅ existe
      'volume_ma' → NO EXISTE    ❌ calculado aquí desde df_rates

Cambios v14.0 (sobre v13.0) — BUGFIXES:

  BUG-1 — RSI, MACD y volume_ma nunca llegaban al AdaptiveSL:
    Causa: current_indicators intentaba leer rsi/macd_hist/volume_ma desde
    data{} (tick MT5 — no los tiene) o desde _ind_source (live_order / _last_diag
    — decide_live no los expone en su output ni en _last_diag).
    El pipeline sí los calcula internamente, pero solo para features del modelo
    sin exponerlos hacia fuera.

    Fix: pipeline.prepare_data(df_rates) ya se llamaba en línea 751 para
    live_update_scalers_from_df. Ahora se garantiza que SIEMPRE se ejecuta
    (sacado del guard 'if hasattr'), y se extrae la última fila para obtener:
      - rsi:            columna 'rsi' del pipeline (RSI-14 estándar)  ✅
      - macd_hist:      columna 'macd_hist' (MACD histogram actual)   ✅
      - macd_hist_prev: penúltima fila de 'macd_hist' (cruce hist.)   ✅
      - volume_ma:      columna 'volume_ma' — NO EXISTE en pipeline   ❌→fix v15

    Se inyectan en current_indicators con prioridad sobre los fallbacks
    existentes. Sin cambios en TradingSimulator ni en _last_diag.

    Impacto: rsi_ok y macd_ok activos desde v14. volume_ok requería fuente
    alternativa — corregido en v15.0.

  BUG-2 — S2 no recupera la conexión ZMQ cuando el mercado reabre:
    Causa: el socket SUB usa RCVTIMEO=30s. Cuando MT5 cierra/reinicia su socket
    PUB (al cerrar el mercado o reiniciar el EA), ZMQ mantiene el socket SUB
    "conectado" internamente pero el canal de mensajes queda muerto. Como
    zmq.Again no distingue "mercado cerrado" de "publisher caído", S2 nunca
    sabe que necesita reconectar y el loop queda sordo para siempre aunque
    MT5 reabra la sesión horas después.

    Fix: contador de timeouts consecutivos (_no_tick_count). Si se acumulan
    _ZMQ_RECONNECT_AFTER_TIMEOUTS timeouts sin ningún tick (por defecto 40,
    equivale a 40 * 30s = 20 min), S2 descarta el socket viejo, crea uno nuevo
    y reconecta a la misma dirección. El contador se resetea a 0 en cuanto llega
    cualquier tick válido. Incluye backoff exponencial (2^intento, cap 300s) para
    no martillar el broker si MT5 tarda en reabrir, y log explícito de cada
    intento de reconexión para auditoría.

Cambios v13.0 (sobre v12.0) — BUGFIXES:

  BUG-1 — volume siempre None en current_indicators → AdaptiveSL ciego de volumen:
    Causa: El tick MT5 llega con clave 'tick_volume' en data{}. La línea
      df = df.rename(columns={"tick_volume": "volume"})
    renombra solo el DataFrame, no el dict data{}. Por tanto data.get('volume')
    devolvía siempre None aunque el dato existía en el tick.
    Fix: data.get('volume') or data.get('tick_volume') en current_indicators.
    Impacto: volume_ok en reversal_signals ahora puede activarse para el
    AdaptiveSL expansion/compression anti-sweep.

  BUG-2 — partial_trigger_pct=60 en create_order_request nunca llega a S3:
    Causa: create_order_request calcula partial_trigger_pct=60.0 y close_fraction=0.50
    pero no los incluye en el dict 'risk' que se envía a S3. S3 usa su perfil
    hardcoded scalping con trigger_profit_pct=100 (necesita recorrer 1R completo
    antes de hacer el parcial, en lugar del 60% configurado en S2).
    Fix: añadir sub-dict 'partial_close' dentro de 'risk' con los valores
    calculados en create_order_request, de modo que S3 los sobreescriba.

  BUG-3 — _ind_source usa live_order OR _last_diag, pero si no hay señal
    (live_order=None) _ind_source = _diag. En ese caso atr_at_entry no existe
    en _last_diag → atr en current_indicators queda None en ticks sin señal.
    Los MODIFY enviados a posiciones abiertas en ticks sin señal enviaban
    {proba_long, proba_short} únicamente, sin atr.
    Fix: extraer atr siempre desde _last_diag.get('atr') como fallback adicional.

Cambios v12.0 (sobre v11.0) — BUGFIXES:

  BUG-1 (CRÍTICO) — vSL en dirección incorrecta y mismatch vsl_price/vsl_pts:
    Causa: create_order_request() usaba order['sl'] directamente tal como lo
    devuelve decide_live() sin validar dirección ni consistencia. Dos problemas:
    a) En ~15% de las posiciones sl < entry en SELL o sl > entry en BUY →
       S3 detecta la posición en zona SL desde el primer tick y cierra en <1s
       con P&L aleatorio.
    b) En el 88% de las posiciones |vsl_price - entry| ≠ vsl_pts * point →
       S3 calcula mal el R realizado, el BE trigger y el trailing distance porque
       vsl_pts_log y vsl_price no son consistentes entre sí.
    Detectado: análisis del log s3_events_20260306.jsonl — 10 cierres en <5s,
    overshoot medio +55pts, P&L del día $-1531.
    Fix (en create_order_request):
    1. Detectar SL al lado incorrecto y reflejarlo simétricamente.
    2. Recalcular siempre virtual_sl_price = entry ± vsl_pts * point para
       garantizar consistencia exacta entre precio y puntos.
    3. Añadir raw_sl_from_model y sl_direction_corrected al metadata para
       auditoría en signals_YYYYMMDD.jsonl.
"""
import json
import math
import os
import time
from collections import deque
from typing import Dict, Any, Optional

import pandas as pd
import zmq

from config.decision_policies_config import score_cap_by_state, risk_mult_by_state, gate_by_action_and_state
from mimo.data_managers.databases import Database
from mimo.features.feature_builder import FeatureConfig
from mimo.models.model_builder import ModelConfig, Config
from mimo.strategies.decision_engine import DecisionPolicy, RiskConfig
from mimo.strategies.regime_detector import RegimeConfig
from mimo.strategies.trading_simulator_v2 import TradingSimulator, load_rl_policy_npz

# v15.0: periodo SMA para volume_ma calculado desde df_rates (no del pipeline)
VOLUME_MA_PERIOD = 20

# ── RR mínimo garantizado por régimen (P5 — v16.0) ─────────────────────────
# Si el modelo devuelve un TP con RR inferior a este umbral, S2 lo recalcula
# en create_order_request antes de enviar a S3. Calibrado sobre análisis
# 09/03/2026: RR actual mediano = 0.27, WR necesaria = 79%. Con RR ≥ 1.0
# el WR necesario baja a 50%, alcanzable con el modelo actual.
# RR diferenciado por régimen: en trend se permite más margen al TP (1.5);
# en range y transition se exige al menos 1:1 (igual riesgo que recompensa).
MIN_RR_BY_REGIME: dict = {
    # v44.0: todos los regímenes fijados a 1.0.
    # Análisis sesión 17/03/2026: 94/152 trades (62%) alcanzaron 1R antes de revertir.
    # Con RR=2.5 el TP requería 3.75×ATR — raramente alcanzable en M1.
    # Con RR=1.0 el TP = 1×ATR, alineado con la movilidad real del mercado.
    'trend_up':           1.0,
    'trend_down':         1.0,
    'transition_up':      1.0,
    'transition_down':    1.0,
    'range':              1.0,
    'breakout_wait_up':   1.0,
    'breakout_wait_down': 1.0,
    'volatile':           1.0,
    'low_vol':            1.0,
    '_default':           1.0,
}

# ── RR máximo por régimen (v37.0) ───────────────────────────────────────────
# Actúa como TECHO del RR: si el modelo devuelve un TP con RR superior al
# máximo del régimen, se recorta. Complementa MIN_RR_BY_REGIME (suelo).
#
# Calibración provisional (sesión 16/03/2026, 17 trades):
#   - Todos los trades salieron con RR=2.5 fijo → el modelo siempre supera
#     el MIN y no había cap superior. Esto es subóptimo porque en RANGE
#     el TP a 2.5R casi nunca se alcanza limpio (el precio rebota antes).
#
#   RANGE (1.5): precio oscila entre bandas. TP natural = banda contraria.
#     Bollinger ya recorta cuando está disponible; este MAX actúa como red
#     de seguridad cuando Bollinger no aplica.
#   TRANSITION_* (2.0): estado incierto, recorte moderado.
#   TREND_* (3.0): con score ≥ 0.40 requerido y tendencia confirmada,
#     el precio puede recorrer más de 2.5R — se sube el techo.
#   LOW_VOL / VOLATILE (1.5): objetivos conservadores.
#   _default (2.5): comportamiento anterior sin cambios para régimenes
#     no listados explícitamente.
#
# Revisar con ≥ 100 trades antes de ajustar definitivamente.
TRAINING_SL_BARRIER_R = 1.5
TRAINING_TP_BARRIER_R = 2.5
TRAINING_LABEL_RR = TRAINING_TP_BARRIER_R / TRAINING_SL_BARRIER_R   # = 1.6667

MAX_RR_BY_REGIME: dict = {
    # v49.2: techo diferenciado por régimen.
    # En tendencias confirmadas (trend_*) el precio tiene mayor recorrido y el
    # modelo produce sus scores más altos → permitir TP más amplio.
    # En transición el recorrido es menor pero ya hay dirección → RR intermedio.
    # En range el precio revierte rápido → mantener RR ajustado (1.0).
    # En volatile el riesgo de reversión brusca es máximo → TP conservador (0.8).
    # AVISO: validar con ≥30 días de historial. Si WR en trend < 40% → revertir a 1.0.
    # v44.0 anterior: todos a 1.0 (techo igualado por cautela post-recalibración).
    'trend_up':           TRAINING_LABEL_RR,   # alineado con el RR implícito del entrenamiento (2.5/1.5)
    'trend_down':         TRAINING_LABEL_RR,   # alineado con el RR implícito del entrenamiento (2.5/1.5)
    'transition_up':      1.2,   # tendencia en formación — recorrido moderado
    'transition_down':    1.2,   # tendencia en formación — recorrido moderado
    'range':              1.0,   # rango: reversión rápida — TP ajustado
    'breakout_wait_up':   1.2,   # potencial breakout — algo de margen
    'breakout_wait_down': 1.2,   # potencial breakout — algo de margen
    'volatile':           0.8,   # volatilidad extrema — TP conservador
    'low_vol':            1.0,   # baja volatilidad — igual que range
    '_default':           1.0,
}

# ── Mínimo de distancia del vSL (FIX v17.0) ────────────────────────────────
# MIN_VSL_POINTS: mínimo absoluto en puntos (fallback cuando ATR no disponible
#   o cuando ATR * ratio resulte menor). 20pts en XAUUSD = $0.20, suficiente
#   para absorber el spread pero no tan grande que cambie la lógica de riesgo.
# MIN_VSL_ATR_RATIO: fracción del ATR actual usada como mínimo dinámico.
#   0.5 ATR garantiza que el vSL tenga margen suficiente para absorber
#   el ruido normal del mercado sin generar slippage masivo.
#   Con ATR post-recalibración ≈ 350-450pts → mínimo ≈ 175-225pts.
#   Referencia: análisis log 10/03/2026 — slippage medio 27pts, máx 129pts
#   en trades con vSL de 21-72pts.
MIN_VSL_POINTS:    int   = 20    # mínimo absoluto (pts)
MIN_VSL_ATR_RATIO: float = 0.50  # mínimo dinámico = 0.5 * ATR

# ── Spread máximo para enviar orden (primera línea de defensa en S2) ────────
# S3 tiene su propio filtro en el perfil scalping (max_spread_points=20, v15.0).
# Este filtro en S2 evita que la orden viaje a S3 cuando el spread ya es excesivo,
# manteniendo el log de S2 limpio (SIGNAL_BLOCKED en lugar de que S3 emita
# OPEN_REJECTED_SPREAD). Ambos límites deben coincidir.
# XAUUSD: spread normal ~10pts, picos típicos ~15-18pts en noticias.
# 20pts cubre los picos normales sin dejar pasar spreads patológicos.
MAX_SPREAD_POINTS: int = 20

# ── vSL mínimo viable por trade (v29.0) ────────────────────────────────────
# Señales con virtual_sl_points < MIN_VIABLE_TRADE_SL_PTS se rechazan antes
# de enviar a S3. Con vSL < 50pts el AdaptiveSL no tiene margen operativo
# (expansion_pts=25 + hard_sl_margin=20 = 45pts requeridos) y el precio puede
# cruzar el vSL en <10s por ruido o spread.
# Confirmado 13/03/2026: 9 trades cerraron en 0-5s; todos tenían vSL ~ entry.
# El guardia se aplica DESPUÉS de calcular el vSL final (post-guardia min ATR),
# por lo que no puede ser eludida por vSLs artificialmente corregidos.
MIN_VIABLE_TRADE_SL_PTS: int = 50

# ── Score mínimo para SELL en régimen alcista (v29.0) ──────────────────────
# Cuando el régimen detectado es trend_up, penalizar señales SELL requiriendo
# un score mínimo mayor. Mitiga el sesgo direccional del modelo en mercados
# alcistas (confirmado 13/03/2026: 30 SELLs vs 7 BUYs con oro subiendo).
# El threshold adicional (+0.15) no bloquea SELLs fuertes pero filtra los
# marginales que el modelo emite por sesgo de entrenamiento.
# ── Filtro contra-tendencia (v42.0: bloqueo total, antes penalización) ─────
# Confirmado 16-17/03/2026: el modelo generó 12 señales SHORT en TREND_UP con
# scores 0.44-1.0 mientras el oro subía de 5015 a 5038 (+2300pts). La penalización
# anterior de +0.15 era insuficiente — con score=1.0 nunca bloquea nada.
# Nuevo comportamiento: BLOQUEO TOTAL de señales contra-tendencia fuerte.
# SELL en TREND_UP → siempre bloqueado (nunca tiene sentido hacer short en tendencia alcista)
# BUY en TREND_DOWN → siempre bloqueado
# Excepción: si COUNTER_TREND_ALLOW_WITH_SCORE=True y score > umbral muy alto
# (por defecto desactivado — el histórico muestra que incluso con score=1.0 fallan).
COUNTER_TREND_SCORE_PENALTY: float = 0.15  # mantenido por compatibilidad (no usado con bloqueo total)
COUNTER_TREND_REGIMES_LONG:  set   = {'trend_up', 'trending_up', 'bull'}
COUNTER_TREND_REGIMES_SHORT: set   = {'trend_down', 'trending_down', 'bear'}
COUNTER_TREND_BLOCK_TOTAL:   bool  = True   # v42.0: bloqueo total (False = penalización como antes)

# ── Filtro horario: bloquear operativa en sesión nocturna (v42.0) ───────────
# Confirmado 16-17/03/2026: sesión 23:00-07:00 UTC (Asian session).
# Volumen bajo, spreads más amplios, tendencias prolongadas sin corrección.
# El modelo fue entrenado principalmente en sesión europea/americana y tiende
# a sobregenerar señales durante la noche con sesgo direccional incorrecto.
# En 8h nocturnas: 76 señales, WR 6.2%, pérdidas acumuladas masivas.
# Horario permitido: 07:00-22:00 UTC (sesiones London + NY + overlap).
# Se puede ajustar con SESSION_START_UTC / SESSION_END_UTC.
SESSION_START_UTC: int = 7    # hora UTC de apertura (inclusive)
SESSION_END_UTC:   int = 22   # hora UTC de cierre (exclusive: hasta las 21:59)

# Debe coincidir con min_score_to_trade en RiskConfig dentro de trading_engine().
# Se define aquí para ser accesible en el loop principal (fuera del scope de trading_engine).
MIN_SCORE_TO_TRADE: float = 0.10

# ── Gap máximo entre model_entry y precio de mercado actual (v30.0) ─────────
# S2 calcula vSL y emergency SL anclados a order["entry"] (precio del modelo
# en la última barra). Si el precio de mercado actual ya se alejó demasiado
# de ese entry, la geometría completa del trade queda desplazada: el vSL y el
# emergency SL calculados sobre model_entry pueden quedar al lado incorrecto
# del fill real → MT5 retcode 10016 "Invalid stops".
# Caso confirmado: gap=332pts entre model_entry=5114.85 y close=5118.17;
# emergency_sl = model_vsl + 50pts = 5116.29 < bid(5118) → SELL inválido.
# El problema ocurre cuando gap > min_vsl + hard_sl_margin.
# Con ATR=188pts: min_vsl=94, margin=50 → threshold=144pts < gap(332) → fallo.
# Tras FIX-4 (db.save síncrono), gap esperado en condiciones normales < 5pts.
# El filtro ya no previene OPEN_FAILED (cubierto por FIX-4 + re-ancla v30 +
# S3 FIX-8), pero sí bloquea eventos extremos: news spikes (NFP/Fed:
# 300-500pts en <1s), gaps de sesión, flash crashes.
# Umbral preferido: DINÁMICO = 1×ATR (ver filtro 6). Este valor se usa
# como fallback si atr no está disponible en live_order.
MAX_ENTRY_GAP_PTS: int = 300

# ── Antigüedad máxima de señal en barras (v30.1) ─────────────────────────────
# decide_live() devuelve la misma señal (mismo entry_time) mientras el modelo
# no genere una nueva. En M1 el modelo genera señal en la barra N; S2 la
# procesa en N+1 (normal, antigüedad=1). Si el modelo no actualiza, la misma
# señal se reenvía en N+2, N+3... con el precio ya muy alejado del entry.
# Confirmado 13/03/2026: señal de 14:06 bloqueada 5 veces (14:08-14:12)
# con gaps de 234-779pts porque el precio bajó 800pts sin nueva señal.
# MAX_SIGNAL_AGE_BARS=1: permite la barra inmediatamente siguiente (normal);
# bloquea en la barra N+2 en adelante (señal rancia).
MAX_SIGNAL_AGE_BARS: int = 1

# ── Ajuste de TP a banda de Bollinger en mercado en rango (v21.0) ────────────
# BOLLINGER_TP_REGIMES: regímenes donde se sustituye el TP del modelo por la
#   banda de Bollinger inferior (SELL) o superior (BUY). Solo tiene sentido en
#   mercados donde el precio rebota entre bandas sin tendencia clara.
# BOLLINGER_TP_MIN_RR: el TP de Bollinger solo se aplica si el RR resultante
#   es al menos este valor. Evita aceptar un TP demasiado cercano al entry.
#   Valor por defecto: 0.5 (la mitad del riesgo como mínimo).
BOLLINGER_TP_REGIMES: set = {'range', 'low_vol'}
BOLLINGER_TP_MIN_RR:  float = 0.5

# v45.0: distancia del trailing del runner tras el partial close.
# Con partial_close activo (50% al TP), el runner restante tiene un trailing
# muy ajustado para no devolver la ganancia si el precio revierte.
# 20pts ≈ spread típico * 1.5 — suficiente para no saltar por el spread
# pero prácticamente en breakeven del runner. 0 = trailing normal (v27 behavior).
RUNNER_TIGHT_TRAIL_PTS: int = 20

# ── Cooldown por pérdidas consecutivas en el mismo side (v32.0) ─────────────
# Si los últimos LOSS_STREAK_MAX trades del mismo side cerraron por VIRTUAL_SL
# en menos de LOSS_STREAK_WINDOW_SECS, bloquear nuevas entradas en ese side
# durante LOSS_STREAK_COOLDOWN_SECS. Evita la acumulación de posiciones cuando
# el mercado se mueve sistemáticamente en contra (racha de 9 BUYs perdedores
# del 13/03/2026 entre 18:35 y 18:44h).
# Parámetros calibrados sobre ese evento: 2 pérdidas en 5 min → cooldown 5 min.
# El cooldown es por side (BUY/SELL independientes) y se resetea si hay un win.
LOSS_STREAK_MAX:          int = 2      # pérdidas consecutivas que activan el cooldown
LOSS_STREAK_WINDOW_SECS:  int = 300    # ventana temporal para contar la racha (5 min)
LOSS_STREAK_COOLDOWN_SECS: int = 300   # tiempo de bloqueo tras activar el cooldown (5 min)
# v35.0: reducido de 60 → 30s.
# Con el trigger por vela (KEEPALIVE_ON_NEW_BAR=True, v33), los trades que abren y
# cierran dentro de la misma vela M1 (hold < 60s) nunca ven un cambio de bar_time
# y nunca reciben un MODIFY. Análisis log 16/03: 38/55 trades sin ningún MODIFY,
# mediana de hold de los trades sin MODIFY = 62s. Para scalping con trades de 30-90s,
# el fallback de tiempo necesita ser suficientemente corto para cubrir al menos los
# trades más largos dentro de la misma vela.
# 30s: cubre trades de ≥30s con el fallback de tiempo, manteniendo el trigger por
# vela como mecanismo principal. Sigue siendo < IND_STALE_WARN_SECS=120s de S3.
KEEPALIVE_INTERVAL_SECS: int = 30

# v33.0: forzar MODIFY en cada vela M1 nueva, independientemente del tiempo.
# Con KEEPALIVE_ON_NEW_BAR=True, el bloque B dispara siempre que bar_time del
# tick actual difiere del último bar_time registrado para ese ticket. Como S1
# emite un tick por vela M1 cerrada, esto garantiza un MODIFY cada ~60s aunque
# los ticks lleguen muy seguidos (mercado activo) o muy espaciados (quietud).
# KEEPALIVE_INTERVAL_SECS sigue activo como red de seguridad para reconnect,
# bar_time=None o gaps de S1.
KEEPALIVE_ON_NEW_BAR: bool = True

TF_ENABLE_ONEDNN_OPTS = 0

def price_to_points(price_a: float, price_b: float, point: float = 0.01) -> int:
    return int(round(abs(price_a - price_b) / point))


class IncrementalSMA:
    """
    Media móvil simple con actualización incremental O(1) por tick.

    Usa un buffer circular (deque, maxlen=period) y mantiene la suma acumulada.
    Cuando el buffer está lleno, cada push() resta el valor expulsado y suma el
    nuevo — sin recorrer nunca el buffer completo.

    Reemplaza a _volume_ma_from_rates() (v15.0), que recalculaba la SMA entera
    (df[col].iloc[-period:].mean()) en cada tick: O(period) por tick, además de
    overhead de slice/indexado Pandas sobre df_rates de 2048 filas.

    v19.0: instancia global _volume_sma, alimentada con la última barra de
    df_rates en cada tick. Complejidad: O(1) por tick.

    Uso:
        sma = IncrementalSMA(period=20)
        sma.push(volume_value)   # llamar una vez por tick con el volumen del bar actual
        sma.value                # devuelve float o None si aún no hay `period` muestras
        sma.reset()              # vaciar buffer (reconexión, reinicio engine)
    """

    def __init__(self, period: int = 20):
        self._period = period
        self._buf: deque = deque(maxlen=period)
        self._total: float = 0.0

    def push(self, value: float) -> None:
        """Añade un nuevo valor al buffer y actualiza la suma acumulada en O(1)."""
        if len(self._buf) == self._period:
            # El deque va a expulsar el valor más antiguo — restarlo antes
            self._total -= self._buf[0]
        self._buf.append(value)
        self._total += value

    @property
    def value(self) -> Optional[float]:
        """
        Devuelve la SMA actual, o None si el buffer aún no tiene `period` muestras.
        Comportamiento idéntico al de _volume_ma_from_rates (requiere periodo completo).
        """
        if len(self._buf) < self._period:
            return None
        result = self._total / self._period
        return None if (math.isnan(result) or result <= 0) else result

    def reset(self) -> None:
        """Vacía el buffer. Llamar al reconectar ZMQ o reiniciar el engine."""
        self._buf.clear()
        self._total = 0.0

def send_order(push_socket, payload: Dict[str, Any],):
    push_socket.send_json(payload)
    print(f"\t[ZMQ->S3] ORDER_REQUEST sent: {payload}\n")


def check_calibration(signals_jsonl_path: str, n_bins: int = 10, min_samples: int = 5) -> None:
    """
    Diagnóstico de calibración isotónica a partir del log de señales (JSONL).

    Carga todos los eventos SIGNAL_SENT y NO_SIGNAL con model_diag, y compara
    las probabilidades calibradas (proba_long_cal / proba_short_cal) con la
    frecuencia real de que el siguiente movimiento fuese favorable.

    IMPORTANTE: este diagnóstico es ESTADÍSTICO y requiere un volumen mínimo de
    señales enviadas (idealmente ≥ 100 SIGNAL_SENT) para ser fiable. Con logs de
    poco volumen como el de 10/03 (0 SIGNAL_SENT) solo puede evaluar la
    distribución de probas, no la calibración real.

    Métricas que genera:
      1. Distribución de proba_long_cal y proba_short_cal (media, std, percentiles).
         Si la media está por debajo de 0.35 en ambas, el calibrador puede estar
         comprimiendo demasiado la señal del modelo raw.
      2. Ratio raw/cal: diferencia entre proba_raw y proba_cal. Un ratio muy alto
         (>2x de media) indica sobre-compresión del calibrador.
      3. Calibration curve (reliability diagram): agrupa señales en bins de proba_cal
         y calcula la tasa de éxito real en cada bin. Una calibración perfecta daría
         puntos sobre la diagonal y=x.
      4. Expected Calibration Error (ECE): media ponderada del error absoluto entre
         proba predicha y frecuencia real. ECE < 0.05 es aceptable en trading.
      5. Alerta de sobre-compresión: si proba_cal_max < 0.55 en todos los eventos,
         el calibrador nunca supera el 55% de confianza — señal de posible degradación.

    Uso desde línea de comandos (o desde un notebook de análisis):
        from main_trading_s2_v18 import check_calibration
        check_calibration('./logs/signals_20260310.jsonl')

    O directamente:
        python -c "from main_trading_s2_v19 import check_calibration; check_calibration('./logs/signals_20260310.jsonl')"
    """
    # math importado a nivel de módulo (v19.0)

    events = []
    with open(signals_jsonl_path, encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"\n{'='*60}")
    print(f"DIAGNÓSTICO DE CALIBRACIÓN ISOTÓNICA")
    print(f"Fuente: {signals_jsonl_path}  ({len(events)} eventos totales)")
    print(f"{'='*60}")

    # ── 1. Distribución de probas (todos los eventos con model_diag) ─────────
    all_long_cal  = [e['model_diag']['proba_long_cal']  for e in events
                     if 'model_diag' in e and 'proba_long_cal'  in e['model_diag']]
    all_short_cal = [e['model_diag']['proba_short_cal'] for e in events
                     if 'model_diag' in e and 'proba_short_cal' in e['model_diag']]
    all_long_raw  = [e['model_diag']['proba_long_raw']  for e in events
                     if 'model_diag' in e and 'proba_long_raw'  in e['model_diag']]
    all_short_raw = [e['model_diag']['proba_short_raw'] for e in events
                     if 'model_diag' in e and 'proba_short_raw' in e['model_diag']]

    def _stats(vals):
        if not vals:
            return {}
        n = len(vals)
        mu = sum(vals) / n
        std = (math.sqrt(sum((v - mu)**2 for v in vals) / n)) if n > 1 else 0.0
        s = sorted(vals)
        return {'n': n, 'mean': round(mu, 4), 'std': round(std, 4),
                'min': round(s[0], 4), 'p25': round(s[n//4], 4),
                'median': round(s[n//2], 4), 'p75': round(s[3*n//4], 4),
                'max': round(s[-1], 4)}

    print("\n── 1. Distribución de probabilidades calibradas ──")
    for label, vals in [('proba_long_cal', all_long_cal), ('proba_short_cal', all_short_cal)]:
        st = _stats(vals)
        if st:
            print(f"  {label}: n={st['n']}  mean={st['mean']}  std={st['std']}  "
                  f"[{st['min']} .. {st['p25']} .. {st['median']} .. {st['p75']} .. {st['max']}]")

    print("\n── 2. Ratio raw / cal (compresión del calibrador) ──")
    ratios_long  = [r/c for r, c in zip(all_long_raw,  all_long_cal)  if c > 0]
    ratios_short = [r/c for r, c in zip(all_short_raw, all_short_cal) if c > 0]
    for label, ratios in [('long raw/cal', ratios_long), ('short raw/cal', ratios_short)]:
        if ratios:
            mean_r = sum(ratios) / len(ratios)
            max_r  = max(ratios)
            flag = ' ⚠️  SOBRE-COMPRESIÓN (>2x)' if mean_r > 2.0 else ''
            print(f"  {label}: mean={mean_r:.2f}  max={max_r:.2f}{flag}")

    # Alerta de sobre-compresión global
    if all_long_cal and max(all_long_cal) < 0.55 and max(all_short_cal) < 0.55:
        print("\n  ⚠️  ALERTA: proba_cal nunca supera 0.55 en ningún evento.")
        print("      El calibrador puede estar en degradación o el modelo no tiene señal.")

    # ── 2. Calibration curve (solo con SIGNAL_SENT que tienen resultado) ─────
    sent = [e for e in events if e['event'] == 'SIGNAL_SENT']
    print(f"\n── 3. Calibration curve (reliability diagram) ──")
    print(f"  SIGNAL_SENT disponibles: {len(sent)}")

    if len(sent) < min_samples:
        print(f"  ⚠️  Insuficientes señales enviadas (mínimo recomendado: {min_samples}).")
        print("      Para evaluar la calibración real se necesita el log de resultados")
        print("      de S3 (signals + s3_events correlacionados por ticket).")
        print("      Con el log actual solo es posible evaluar la distribución (sección 1-2).")
    else:
        # Intentar correlacionar proba_cal con éxito real usando req_virtual_sl / req_virtual_tp
        # Esta aproximación usa la dirección de la señal y el close del siguiente bar
        # (solo válido si los eventos están ordenados por tiempo)
        closes = [e.get('close') for e in events if 'close' in e and e.get('close') is not None]
        if len(closes) < 2:
            print("  No hay suficientes datos de precio para construir la curva.")
        else:
            samples = []
            sent_idx = [(i, e) for i, e in enumerate(events) if e['event'] == 'SIGNAL_SENT']
            for idx, ev in sent_idx:
                # Buscar el siguiente close disponible tras la señal
                next_closes = [events[j].get('close') for j in range(idx+1, min(idx+6, len(events)))
                               if events[j].get('close') is not None]
                if not next_closes:
                    continue
                entry = ev.get('entry') or (ev.get('req_virtual_sl') and ev.get('close'))
                if entry is None:
                    entry = ev.get('close')
                side  = ev.get('req_side') or ev.get('side')
                proba = ev.get('proba_long') if side == 'BUY' else ev.get('proba_short')
                if proba is None or entry is None:
                    continue
                next_close = next_closes[0]
                success = 1 if (side == 'BUY' and next_close > entry) else \
                          1 if (side == 'SELL' and next_close < entry) else 0
                samples.append((proba, success))

            if len(samples) < min_samples:
                print(f"  Solo {len(samples)} señales correlacionables — curva no fiable.")
            else:
                bin_size = 1.0 / n_bins
                ece = 0.0
                total = len(samples)
                print(f"  {'Bin':>12}  {'N':>5}  {'Proba media':>12}  {'Éxito real':>12}  {'Error':>8}")
                for b in range(n_bins):
                    lo, hi = b * bin_size, (b + 1) * bin_size
                    in_bin = [(p, s) for p, s in samples if lo <= p < hi]
                    if len(in_bin) < 2:
                        continue
                    mean_p = sum(p for p, _ in in_bin) / len(in_bin)
                    mean_s = sum(s for _, s in in_bin) / len(in_bin)
                    err    = abs(mean_p - mean_s)
                    ece   += err * len(in_bin) / total
                    flag   = ' ⚠️' if err > 0.10 else ''
                    print(f"  [{lo:.2f}-{hi:.2f}]  {len(in_bin):>5}  {mean_p:>12.4f}  {mean_s:>12.4f}  {err:>8.4f}{flag}")
                print(f"\n  ECE (Expected Calibration Error): {ece:.4f}  "
                      f"{'✅ aceptable' if ece < 0.05 else '⚠️  revisar calibración'}")

    print(f"\n{'='*60}\n")

def create_order_request(
        order: Dict[str, Any],
        # FIX v25.0 (BUG-2): reducido de 4.0 → 1.5.
        # Con 4.0*ATR el hard SL quedaba a ~1400pts del entry (mediana sesión
        # 12/03/2026: 771pts) → ratio hard/virtual = 4x. Los 96 cierres externos
        # observados confirman que el precio cruzaba el vSL y continuaba hasta el
        # hard SL sin que S3 lo detectara a tiempo (loop 0.2s). Exposición máxima
        # por trade: ~$140 con 4.0x vs ~$52 con 1.5x. El hard SL sigue siendo al
        # menos 1.5x el virtual_sl (garantizado por la guardia posterior), actuando
        # como paracaídas real para gaps de 1-2 ATR en lugar de ser prácticamente
        # inaccesible (4 ATR = movimiento extremadamente improbable en scalping).
        emergency_atr_mult: float = 1.5,
        be_trigger_ratio: float = 0.50,   # v22.0: 50% del vSL (antes: 40% del TP → daba ≈1R)
        be_offset_ratio: float = 0.02,    # v22.0: 2% del vSL (antes: 2% del TP)
        trail_sl_ratio: float = 1.0,  # trailing como ratio del virtual_sl_points (1.0 = 1R)
        trail_step_ratio: float = 0.50,  # paso mínimo como ratio del virtual_sl_points (era 0.15, subido a 0.50 para reducir ruido)
        max_hold_seconds: int = 900,
        close_fraction: float = 0.50,
        partial_trigger_pct: float = 90.0,  # v45.1: subido de 60→90% del TP (ver análisis).
        point: float = 0.01,
        comment: str = '',
        bb_upper: Optional[float] = None,  # v21.0: banda superior Bollinger del último bar
        bb_lower: Optional[float] = None,  # v21.0: banda inferior Bollinger del último bar
        # v30.0: bid/ask del tick actual (enviados por S1). Cuando están presentes,
        # se usan como referencia de precio real para recalcular la geometría del
        # trade (vSL, emergency SL) anclada al fill esperado en lugar de al
        # model_entry (precio de la barra del modelo, que puede diferir cientos
        # de puntos del fill real en mercados rápidos).
        bid: Optional[float] = None,
        ask: Optional[float] = None,
) -> Dict[str, Any]:
    entry = float(order['entry'])
    virtual_tp_price = float(order['tp'])
    # ── FIX v23.0 (BUG-2): guardia ATR casi-cero ────────────────────────────
    # v22.0 descartaba solo ATR=0/None. Un ATR casi-cero (p.ej. 0.0001 — posible
    # en la primera barra tras reconexión) pasaba la guarda anterior y producía
    # atr_pts = int(round(0.0001/0.01)) = 0 → _min_vsl = MIN_VSL_POINTS = 20,
    # dejando pasar vSLs de 20-33pts que causaban WRONG_SIDE_SL_CORRECTED y
    # slippage masivo (ticket 316211980 sesión 11/03: 524pts con vSL de ~28pts).
    # Fix v23.0: el umbral de invalidez sube de <= 0 a <= MIN_VSL_POINTS * point.
    # Cualquier ATR que produzca _min_vsl < MIN_VSL_POINTS se trata como inválido
    # y activa la cascada de fallback (pipeline → sintético). Esto garantiza que
    # _min_vsl sea siempre >= MIN_VSL_POINTS con independencia del ATR recibido.
    _ATR_MIN_VALID = MIN_VSL_POINTS * point   # ATR mínimo para ser útil (0.20 en XAUUSD)

    def _safe_float(x):
        try:
            return float(x)
        except Exception:
            return 0.0

    _atr_signal = order.get('atr_at_entry')
    _atr_order  = order.get('atr')
    _atr_pipe   = order.get('_atr_fallback')

    _raw_atr = _safe_float(_atr_signal)
    _atr_source = 'atr_at_entry'

    if _raw_atr <= _ATR_MIN_VALID:
        _raw_atr = _safe_float(_atr_order)
        _atr_source = 'order.atr'

    if _raw_atr <= _ATR_MIN_VALID:
        _raw_atr = _safe_float(_atr_pipe)
        _atr_source = '_atr_fallback'

    if _raw_atr <= _ATR_MIN_VALID:
        _raw_atr = MIN_VSL_POINTS * 4 * point   # sintético: 80pts en XAUUSD
        _atr_source = 'synthetic_min_safe'

    atr = _raw_atr
    # ─────────────────────────────────────────────────────────────────────────
    side_str = order['side']
    qty = float(order['qty'])

    mt5_side = 'BUY' if side_str == 'long' else 'SELL'

    # ── v30.0: precio de referencia para geometría anclada al fill esperado ──
    # Si S1 envió bid/ask del tick actual, usarlos como referencia de precio
    # real (BUY ejecuta al ask, SELL al bid). Si no están disponibles (S1
    # antiguo), usar model_entry como siempre (comportamiento idéntico a v29).
    if mt5_side == 'BUY' and ask is not None and ask > 0:
        fill_ref = float(ask)
        _fill_ref_src = 'ask'
    elif mt5_side == 'SELL' and bid is not None and bid > 0:
        fill_ref = float(bid)
        _fill_ref_src = 'bid'
    else:
        fill_ref = entry   # fallback: model_entry (comportamiento pre-v30)
        _fill_ref_src = 'model_entry'
    # ─────────────────────────────────────────────────────────────────────────

    # Calculamos los dos niveles de SL
    # 1. SL de emergencias - Va al broker
    emergency_sl_distance = emergency_atr_mult * atr
    if side_str == 'long':
        emergency_sl_price = entry - emergency_sl_distance
    else:
        emergency_sl_price = entry + emergency_sl_distance

    emergency_points = price_to_points(entry, emergency_sl_price, point)

    # 2. SL Virtual (señal). Para calcular R y ratios
    # ── FIX v12.0: validar dirección y consistencia del vSL ─────────────────
    # decide_live() puede devolver un sl que esté al lado incorrecto del entry
    # (SELL con sl < entry, BUY con sl > entry) cuando el modelo genera señales
    # con ATR muy bajo o en condiciones de volatilidad extrema. Esto causa que
    # S3 detecte la posición en zona SL desde el primer tick y la cierre en <1s.
    # También puede ocurrir que vsl_price no coincida con entry ± vsl_pts * point
    # porque decide_live usa ATR * multiplier como offset de precio pero
    # price_to_points calcula los puntos con floor(), generando mismatch.
    # Fix: recalcular siempre vsl_price desde entry ± vsl_pts * point
    # para garantizar consistencia total entre precio y puntos.
    raw_sl_price = float(order['sl'])
    raw_sl_pts   = price_to_points(entry, raw_sl_price, point)
    _s2_wrong_sl_corrected = False  # flag para auditoría

    # ── FIX v16.0 (P4): Detectar SL wrong-side y corregirlo en S2 ──────────
    # Idéntica lógica al fix de S3 v11 pero aplicada en origen para que el
    # AdaptiveSLManager reciba el vSL correcto desde el primer tick.
    # Añadido: fallback cuando raw_sl_pts==0 (sl == entry exacto) usando ATR.
    if mt5_side == 'SELL':
        sl_correct_side = raw_sl_price > entry   # SELL: SL debe estar POR ENCIMA
        if not sl_correct_side:
            if raw_sl_pts <= 0:  # sl == entry o pts negativos
                raw_sl_pts = max(10, int(round(atr / point)))  # fallback: 1 ATR
            raw_sl_price = entry + raw_sl_pts * point
            _s2_wrong_sl_corrected = True
    else:  # BUY
        sl_correct_side = raw_sl_price < entry   # BUY: SL debe estar POR DEBAJO
        if not sl_correct_side:
            if raw_sl_pts <= 0:
                raw_sl_pts = max(10, int(round(atr / point)))
            raw_sl_price = entry - raw_sl_pts * point
            _s2_wrong_sl_corrected = True

    # Recalcular vsl_price desde entry ± vsl_pts * point para garantizar
    # consistencia exacta entre precio y puntos (evita mismatch por redondeo ATR)
    virtual_sl_points = raw_sl_pts
    if mt5_side == 'SELL':
        virtual_sl_price = entry + virtual_sl_points * point
    else:
        virtual_sl_price = entry - virtual_sl_points * point

    # ── FIX v17.0 (P1 + P2): Guardia de mínimo vSL post-recálculo ───────────
    # P1: Previene wrong-side residual por errores de coma flotante cuando
    #     raw_sl_pts es muy pequeño (0-2). El recálculo anterior puede generar
    #     virtual_sl_price == entry o al lado incorrecto.
    # P2: Previene slippage masivo por vSL demasiado ajustado respecto al ATR.
    #     Con ATR post-recalibración ≈ 350-450pts, vSLs de 21-72pts generaban
    #     slippages de hasta 129pts (análisis log 10/03/2026).
    # Se aplica DESPUÉS del recálculo para ser la última línea de defensa antes
    # de enviar a S3. El mínimo dinámico (ATR-based) tiene prioridad sobre el
    # mínimo absoluto; el absoluto actúa solo como fallback si ATR no está disponible.
    atr_pts = int(round(atr / point)) if atr > 0 and point > 0 else 0
    _min_vsl = max(MIN_VSL_POINTS, int(atr_pts * MIN_VSL_ATR_RATIO))
    _vsl_min_enforced = False
    if virtual_sl_points < _min_vsl:
        virtual_sl_points = _min_vsl
        if mt5_side == 'SELL':
            virtual_sl_price = entry + virtual_sl_points * point
        else:
            virtual_sl_price = entry - virtual_sl_points * point
        _vsl_min_enforced = True
    # Guardia final de dirección: por si acaso algún path numérico llegó aquí
    # con virtual_sl_price al lado incorrecto (doble seguro tras todo lo anterior)
    if mt5_side == 'SELL' and virtual_sl_price <= entry:
        virtual_sl_points = max(virtual_sl_points, _min_vsl)
        virtual_sl_price  = entry + virtual_sl_points * point
        _vsl_min_enforced  = True
    elif mt5_side == 'BUY' and virtual_sl_price >= entry:
        virtual_sl_points = max(virtual_sl_points, _min_vsl)
        virtual_sl_price  = entry - virtual_sl_points * point
        _vsl_min_enforced  = True
    # ─────────────────────────────────────────────────────────────────────────

    # ── FIX v28.0: Recalcular hard SL desde el vSL, no desde el entry ──────────
    # v30.0: re-anclar vSL al fill_ref (bid/ask real) antes de calcular hard SL.
    # v30.6: emergency SL = vSL + max(3×ATR, MIN_HARD_SL_MARGIN_PTS).
    #
    # Diseño semántico:
    #   vSL  = 1×ATR  → nivel de gestión de S3 (cierre virtual)
    #   hardSL = vSL + 3×ATR  → paracaídas real en broker si S3 falla/crashea
    #   Margen vSL→hardSL = 3×ATR: suficiente para que S3 reaccione incluso
    #   en mercado rápido, y escala correctamente con la volatilidad.
    #
    # Con margen fijo de 50pts (v28): con ATR=329pts el margen era 50/329 = 0.15×ATR
    # → el hard SL quedaba a 50pts del vSL, prácticamente en la misma zona.
    # Cualquier sweep de 50pts activaba el hard SL del broker antes de que S3
    # pudiera reaccionar (confirmado 13/03/2026: trades 317456318/317456813).
    #
    # MIN_HARD_SL_MARGIN_PTS actúa como suelo absoluto para ATRs pequeños
    # (garantiza que S3 AdaptiveSL tenga siempre margen mínimo para operar).
    EMERGENCY_ATR_MULT    = 3.0   # margen vSL→hardSL en múltiplos de ATR
    MIN_HARD_SL_MARGIN_PTS = 50   # suelo absoluto (si ATR pequeño)

    # ── v30.0: re-anclar vSL al fill_ref (bid/ask real) si está disponible ──
    # Si S1 envió bid/ask del tick actual, el vSL se ancla al fill esperado
    # (no al model_entry). La distancia en puntos se conserva.
    # Si fill_ref == entry (S1 antiguo), comportamiento idéntico a v29.
    if _fill_ref_src != 'model_entry':
        if mt5_side == 'SELL':
            virtual_sl_price = fill_ref + virtual_sl_points * point
        else:
            virtual_sl_price = fill_ref - virtual_sl_points * point
    # ─────────────────────────────────────────────────────────────────────────

    # emergency SL = vSL + max(3×ATR, MIN_HARD_SL_MARGIN_PTS)
    _hard_margin_pts = max(MIN_HARD_SL_MARGIN_PTS, int(round(EMERGENCY_ATR_MULT * atr_pts)))
    if mt5_side == 'SELL':
        emergency_sl_price = virtual_sl_price + _hard_margin_pts * point
    else:
        emergency_sl_price = virtual_sl_price - _hard_margin_pts * point
    emergency_points = price_to_points(entry, emergency_sl_price, point)

    # Calcular ratios basados en virtual_sl (no en emergency_sl)
    # Distancia al TP en puntos
    tp_points = price_to_points(entry, virtual_tp_price, point)

    # ── FIX v16.0 (P5): garantizar RR mínimo por régimen ────────────────────
    # Si el modelo devuelve un TP con RR < MIN_RR_BY_REGIME, recalculamos el
    # TP para alcanzar exactamente el RR mínimo. Esto resuelve el RR mediano
    # de 0.27 que exigía WR ≥ 79% para ser rentable (análisis 09/03/2026).
    _regime     = str(order.get('state') or order.get('market_condition') or '_default').strip().lower()
    _min_rr     = MIN_RR_BY_REGIME.get(_regime, MIN_RR_BY_REGIME['_default'])
    _min_tp_pts = int(round(virtual_sl_points * _min_rr))
    _tp_corrected = False
    if tp_points < _min_tp_pts:
        # Recalcular TP para cumplir el RR mínimo
        if mt5_side == 'BUY':
            virtual_tp_price = entry + _min_tp_pts * point
        else:
            virtual_tp_price = entry - _min_tp_pts * point
        tp_points    = _min_tp_pts
        _tp_corrected = True

    # v37.0: cap de RR máximo por régimen ────────────────────────────────────
    # Si el modelo devuelve un TP con RR superior al techo del régimen,
    # se recorta. Ejemplo: en RANGE el modelo puede proponer RR=2.5 pero
    # el precio raramente llega tan lejos antes de rebotar en la banda.
    _max_rr     = MAX_RR_BY_REGIME.get(_regime, MAX_RR_BY_REGIME['_default'])
    _max_tp_pts = int(round(virtual_sl_points * _max_rr))
    _tp_capped  = False
    if tp_points > _max_tp_pts > 0:
        if mt5_side == 'BUY':
            virtual_tp_price = entry + _max_tp_pts * point
        else:
            virtual_tp_price = entry - _max_tp_pts * point
        tp_points   = _max_tp_pts
        _tp_capped  = True
    # ─────────────────────────────────────────────────────────────────────────

    # ── v21.0: Ajuste de TP a banda de Bollinger en régimen 'range' ──────────
    # En mercado en rango las bandas de Bollinger actúan como soporte/resistencia
    # natural. Si el régimen es 'range' (o cualquier régimen en BOLLINGER_TP_REGIMES)
    # y la banda está entre el entry y el TP calculado, la usamos como nuevo TP:
    # es el nivel donde el mercado "naturalmente" para, más realista que el TP del modelo.
    #
    # Condiciones para aplicar el ajuste:
    #   1. Régimen en BOLLINGER_TP_REGIMES.
    #   2. Banda disponible (no None).
    #   3. La banda está entre entry y virtual_tp (hay recorrido real hacia ella).
    #   4. El RR resultante supera BOLLINGER_TP_MIN_RR (no acepta TPs demasiado cercanos).
    #
    # Si alguna condición no se cumple, virtual_tp_price no se toca.
    _bb_tp_applied    = False
    _bb_tp_original   = virtual_tp_price
    _bb_tp_band_value = None

    _bb_regime_match = _regime in BOLLINGER_TP_REGIMES
    if _bb_regime_match:
        # Seleccionar la banda relevante según el lado de la operación
        _bb_target = bb_lower if mt5_side == 'SELL' else bb_upper

        if _bb_target is not None:
            # Verificar que la banda esté en la dirección correcta respecto al entry
            # SELL: necesitamos bb_lower < entry (banda por debajo del precio de entrada)
            # BUY:  necesitamos bb_upper > entry (banda por encima del precio de entrada)
            _bb_correct_side = (
                (mt5_side == 'SELL' and _bb_target < entry) or
                (mt5_side == 'BUY'  and _bb_target > entry)
            )

            if _bb_correct_side:
                # Calcular el RR que tendría el TP de Bollinger
                _bb_tp_pts = price_to_points(entry, _bb_target, point)
                _bb_rr     = _bb_tp_pts / virtual_sl_points if virtual_sl_points > 0 else 0.0

                # Solo aplicar si el RR supera el mínimo configurado
                if _bb_rr >= BOLLINGER_TP_MIN_RR:
                    _bb_tp_band_value = _bb_target
                    virtual_tp_price  = _bb_target
                    tp_points         = _bb_tp_pts
                    _bb_tp_applied    = True
                    print(f"\t[BB_TP] régimen={_regime}  lado={mt5_side}  "
                          f"banda={_bb_target:.2f}  RR={_bb_rr:.2f}  "
                          f"(TP original={_bb_tp_original:.2f} → ajustado a banda)")
                else:
                    print(f"\t[BB_TP] régimen={_regime} — banda disponible ({_bb_target:.2f}) "
                          f"pero RR={_bb_rr:.2f} < mínimo={BOLLINGER_TP_MIN_RR} → TP sin cambios")
    # ─────────────────────────────────────────────────────────────────────────

    # ── v31.0: BE y trailing contextual por state ────────────────────────────
    # La autoridad de la geometría vive en S2: S3 ejecuta, pero no debe tener que
    # reinterpretar un BE demasiado corto o un trailing demasiado genérico.
    state = str(order.get('state') or order.get('market_condition') or '_default').strip().lower()
    score = float(order.get('score', 0) or 0.0)
    _bid = float(order.get('bid', 0) or 0.0)
    _ask = float(order.get('ask', 0) or 0.0)
    spread_pts = int(round(abs(_ask - _bid) / point)) if (_bid > 0 and _ask > 0 and point > 0) else 10

    if state in ('range',):
        be_trigger_mult = 0.40
    elif state in ('transition_up', 'transition_down'):
        be_trigger_mult = 0.45
    elif state in ('trend_up', 'trend_down', 'breakout_wait_up', 'breakout_wait_down') and score >= 0.55:
        be_trigger_mult = 0.60
    else:
        be_trigger_mult = max(0.50, float(be_trigger_ratio))
    be_trigger_points = max(8, int(round(be_trigger_mult * virtual_sl_points)))

    # v32.0 FIX-1 — be_offset_points corregido: min() → max()
    # El cálculo anterior siempre devolvía el floor (13pt con spread=10pt) porque
    # min(floor, cap) = floor cuando cap >= floor. Esto dejaba el BE a solo 13pt del
    # entry en todos los trades — prácticamente sin protección en XAUUSD (spread ~10pt).
    # Confirmado 13/03/2026: ticket 317672859 SELL llegó a +113pt, BE se armó con
    # offset=13pt, precio rebotó y cerró en -49pt con 62pt de slippage.
    # Fix: usar max(floor, cap) para que el offset escale con el tamaño del vSL.
    # Con vSL=270pt: round(0.12*270)=32pt → be_offset=32pt (era 13pt).
    # Floor sigue siendo max(12, ceil(spread*1.25)) como suelo absoluto.
    _be_offset_floor = max(12, int(math.ceil(spread_pts * 1.25)), int(round(0.03 * virtual_sl_points)))
    _be_offset_cap = max(_be_offset_floor, int(round(0.12 * virtual_sl_points))) if virtual_sl_points > 0 else _be_offset_floor
    be_offset_points = max(_be_offset_floor, _be_offset_cap)

    # v45.0: modos de trailing restaurados. El runner vuelve a tener sentido
    # porque partial_close está activo — pero ahora el runner usa tight trail
    # (RUNNER_TIGHT_TRAIL_PTS=20pts) en lugar del trailing normal, así no
    # devuelve la ganancia si el precio revierte.
    if state in ('trend_up', 'trend_down', 'breakout_wait_up', 'breakout_wait_down'):
        if score >= 0.55:
            trail_mult = 0.95
            trail_step_mult = 0.12
            trail_mode = 'runner'
        else:
            trail_mult = 0.80
            trail_step_mult = 0.15
            trail_mode = 'trend_soft'
    elif state in ('range',):
        trail_mult = 0.55
        trail_step_mult = 0.10
        trail_mode = 'range'
    else:
        trail_mult = 0.70
        trail_step_mult = 0.12
        trail_mode = 'transition'

    trail_points = max(20, int(round(trail_mult * virtual_sl_points)))
    trail_step_points = max(6, int(round(trail_step_mult * virtual_sl_points)))

    # Construir solicitud
    request = {
        'action': 'OPEN',
        'symbol': 'XAUUSD.r',
        'side': mt5_side,
        'volume': qty,
        'magic': int(release),
        'comment': comment,
        'deviation': 10,

        # TP virtual (NO se envía al broker, solo gestión interna)
        'virtual_tp': virtual_tp_price,
        'virtual_sl': virtual_sl_price,

        # Metadatos para logging / analisis
        'metadata': {
            's2_script': S2_SCRIPT_NAME,
            's2_version': S2_VERSION,
            'state': state,
            'trail_mode': trail_mode,
            'atr': atr,
            'entry': entry,
            'emergency_sl_price': emergency_sl_price,
            'emergency_sl_points': emergency_points,
            'virtual_sl_price': virtual_sl_price,
            'virtual_sl_points': virtual_sl_points,
            'tp_points': tp_points,
            'risk_R': virtual_sl_points,
            'trail_points': trail_points,
            'trail_step_points': trail_step_points,
            'be_trigger_points': be_trigger_points,
            'be_offset_points': be_offset_points,
            # FIX v12.0 / v16.0: auditoría de corrección SL
            'raw_sl_from_model': float(order['sl']),
            'sl_direction_corrected': not sl_correct_side,
            'sl_corrected_s2': _s2_wrong_sl_corrected,  # P4 v16.0
            # FIX v17.0 (P1+P2): auditoría de mínimo vSL
            'vsl_min_enforced': _vsl_min_enforced,
            'vsl_min_used': _min_vsl,
            'atr_pts': atr_pts,
            'atr_used': atr,
            'atr_source': _atr_source,
            'atr_min_valid': _ATR_MIN_VALID,
            'atr_fallback_used': _atr_source != 'atr_at_entry',
            # FIX v16.0 (P5): auditoría de corrección TP
            'raw_tp_from_model': float(order['tp']),
            'tp_rr_corrected': _tp_corrected,
            'tp_min_rr_regime': _min_rr,
            'tp_max_rr_regime': _max_rr,       # v37.0
            'tp_rr_capped':     _tp_capped,     # v37.0: True si el TP fue recortado por MAX_RR
            'tp_state': _regime,
            # v21.0: auditoría de ajuste TP a banda Bollinger
            'bb_tp_applied':    _bb_tp_applied,
            'bb_tp_original':   _bb_tp_original   if _bb_tp_applied else None,
            'bb_tp_band_value': _bb_tp_band_value if _bb_tp_applied else None,
            # v30.0: auditoría de anclaje de geometría al precio real
            'fill_ref_price':   round(fill_ref, 5),
            'fill_ref_src':     _fill_ref_src,   # "ask"|"bid"|"model_entry"
            'model_entry':      round(entry, 5),
            'fill_ref_gap_pts': round(abs(fill_ref - entry) / point, 1),
        },

        # ── Indicadores para AdaptiveSLManager (v8.0) ─────────────────────
        # S3 los usa para evaluar señales de expansión/compresión del SL.
        # FIX v23.0 (BUG-1): si el caller provee 'indicators_override', se usa
        # ese dict directamente. Permite al loop fusionar live_order + _pip_last
        # para garantizar que el OPEN llega con ind_last_update_ts inicializado.
        # Sin override, comportamiento idéntico a v22 (extrae de order).
        'indicators': order.get('_indicators_override') or {k: v for k, v in {
            'atr':            order.get('atr_at_entry'),
            'rsi':            order.get('rsi'),
            'macd_hist':      order.get('macd_hist'),
            'macd_hist_prev': order.get('macd_hist_prev'),
            'volume':         order.get('volume'),
            'volume_ma':      order.get('volume_ma'),
            'proba_long':     order.get('proba_long'),
            'proba_short':    order.get('proba_short'),
        }.items() if v is not None},

        # ── v24.0: contexto de mercado en el OPEN ─────────────────────────
        # Régimen, score y spread en el momento de la apertura. Permite a S3
        # conocer el contexto desde el primer ciclo sin esperar al primer MODIFY.
        # Si el loop ha inyectado '_context_override', se usa ese dict directamente.
        **(({'context': order['_context_override']}
            if order.get('_context_override') else {})),

        'risk': {
            'profile': 'scalping',
            'trailing_mode': trail_mode,
            'emergency_points': emergency_points,  # sobreescribe el fijo del perfil con el valor ATR-based
            'trail_points': trail_points,  # contextual por state/score
            'trail_step_points': trail_step_points,  # contextual por state/score
            'be_trigger_points': be_trigger_points,  # contextual por state
            'be_offset_points': be_offset_points,  # suelo absoluto + spread + %R
            # FIX v13.0 (BUG-2): incluir partial_close para sobreescribir el
            # trigger_profit_pct=100 hardcoded en el perfil scalping de S3.
            # S2 calcula partial_trigger_pct=60 (60% del camino hacia TP) pero
            # nunca lo enviaba, de modo que S3 usaba su propio 100% siempre.
            'partial_close': {
                # v45.0: reactivado con runner_tight_trail_pts.
                # En lugar de cerrar el 100% al TP (v44), se mantiene el partial
                # close (50% al TP) pero el runner usa un trailing muy ajustado
                # (RUNNER_TIGHT_TRAIL_PTS=20pts) en lugar del trailing normal (~160pts).
                # Efecto: si el precio llega al TP, se cierra el 50% con ganancia
                # y el runner queda con SL a solo 20pts del máximo — si el precio
                # revierte inmediatamente pierde casi nada; si sigue: gana más.
                'enabled': True,
                'trigger_profit_pct': partial_trigger_pct,
                'close_fraction': close_fraction,
                'basis': 'R',
                'runner_tight_trail_pts': RUNNER_TIGHT_TRAIL_PTS,  # v45.0
            },
            # FIX v6.0: sl_mode='touch' en todos los regímenes para evitar slippage
            # El modo 'close' añadía hasta 60s de retraso en scalping (ver S3 v7.0)
            'close_confirmation': {
                'exit_mode': 'hard',
                'tp_mode': 'touch',
                'tp_confirm_count': 0,
                'sl_mode': 'touch',
                'sl_confirm_count': 0,
            }
        }
    }

    return request

# ============================================================================
# S3 STATE SYNC  (v6.0)
# Mantiene un estado local sincronizado con los eventos que publica S3.
# Permite a S2 tomar decisiones de riesgo basadas en la realidad de S3
# (posiciones cerradas, parciales ejecutados) en lugar de confiar solo en
# el n_positions que llega con cada tick de MT5.
# ============================================================================

import threading
from dataclasses import dataclass


@dataclass
class S3Position:
    ticket: int
    side: str           # 'BUY' | 'SELL'
    volume: float
    entry_price: float
    partial_done: bool = False
    be_armed: bool = False
    virtual_sl: float = 0.0
    virtual_tp: float = 0.0


class S3State:
    """
    Estado local sincronizado con eventos de S3.

    Thread-safe: todas las lecturas/escrituras usan self._lock.
    Se actualiza en background por s3_event_listener.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self.positions: Dict[int, S3Position] = {}
        self.last_event: Optional[Dict] = None
        self.last_slippage_pts: Optional[float] = None
        # v32.0: historial de cierres por SL por side, para el cooldown de pérdidas.
        # side ('BUY'/'SELL') → list de (ts_close: float, reason: str).
        # El listener lo actualiza; el loop principal lo lee para el filtro 2b.
        self.last_sl_close_by_side: Dict[str, list] = {'BUY': [], 'SELL': []}
        # v49.0: callback invocado tras registrar un POSITION_OPENED en self.positions.
        # Permite al loop principal registrar el ticket en _keepalive_last_sent
        # en el momento en que el ticket real está disponible (post-fill del broker),
        # sin esperar al siguiente tick. Firma: fn(ticket: int) -> None.
        # Se asigna desde el loop principal justo después de crear s3_state.
        self._on_position_opened_cb = None

    # ── Lecturas ─────────────────────────────────────────────────────────────

    def n_positions(self) -> int:
        with self._lock:
            return len(self.positions)

    def get_positions(self) -> Dict[int, S3Position]:
        with self._lock:
            return dict(self.positions)

    def total_volume(self) -> float:
        with self._lock:
            return sum(p.volume for p in self.positions.values())

    def sides_open(self) -> list:
        """Devuelve lista de sides ('BUY'/'SELL') de posiciones abiertas."""
        with self._lock:
            return [p.side for p in self.positions.values()]

    # ── Escrituras (llamadas desde el listener) ───────────────────────────────

    def on_event(self, msg: Dict) -> None:
        event = msg.get('event', '')
        with self._lock:
            self.last_event = msg

            if event == 'POSITION_OPENED':
                # v49.1 FIX: conversión defensiva de ticket.
                # int(msg['ticket']) falla si ticket es float serializado como string
                # (ej: "319547958.0") → ValueError. int(float(...)) lo tolera.
                # El try/except específico loguea el valor exacto si aún falla,
                # sin depender del except genérico del listener que no incluía msg.
                try:
                    ticket = int(float(msg['ticket']))
                except Exception as _parse_err:
                    import logging as _log491, json as _json491, time as _time491
                    _log491.getLogger('mimo_signals').error(_json491.dumps({
                        'event': 'POSITION_OPENED_PARSE_ERROR',
                        'error': type(_parse_err).__name__,
                        'detail': str(_parse_err),
                        'ticket_raw': repr(msg.get('ticket')),
                        'ticket_type': type(msg.get('ticket')).__name__,
                        'msg_keys': list(msg.keys()),
                        'ts': _time491.time(),
                    }))
                    return  # no registrar posición con ticket inválido
                self.positions[ticket] = S3Position(
                    ticket=ticket,
                    side=msg.get('side', ''),
                    volume=float(msg.get('volume', 0)),
                    entry_price=float(msg.get('entry_price', 0)),
                    virtual_sl=float(msg.get('virtual_sl_price', 0)),
                    virtual_tp=float(msg.get('virtual_tp', 0)),
                )
                # v49.0 FIX-3: debug log para confirmar que el evento llega y se registra.
                import logging as _log49
                _log49.getLogger('mimo_signals').info(__import__('json').dumps({
                    'event': 'DEBUG_POSITION_OPENED_REGISTERED',
                    'ticket': ticket,
                    'ticket_raw': repr(msg.get('ticket')),
                    'positions_count': len(self.positions),
                    'ts': __import__('time').time(),
                }))
                # v49.0 FIX-2: invocar callback para registrar ticket en _keepalive_last_sent.
                _cb = self._on_position_opened_cb
                if _cb is not None:
                    try:
                        _cb(ticket)
                    except Exception:
                        pass  # nunca interrumpir on_event por un fallo del callback

            elif event in (
                # Cierres estándar
                'VIRTUAL_SL_TRIGGERED',
                'VIRTUAL_TP_TRIGGERED',
                'MAX_HOLD_TIME_TRIGGERED',
                'PARTIAL_CLOSE_TO_FULL',
                'POSITION_CLOSED_EXTERNAL',
                'MANUAL_CLOSE_TRIGGERED',
                # AdaptiveSL / AdaptiveTP (tenían handlers separados en v25,
                # consolidados aquí para mantener la lista completa en un solo lugar)
                'ADAPTIVE_SL_CLOSE',
                'ADAPTIVE_TP_CLOSE',
                # v26.0 — cierres adaptativos y forzados que faltaban:
                # Su ausencia generaba tickets "zombie" en S3State que inflaban
                # effective_positions y bloqueaban nuevas aperturas por MAX_POSITIONS.
                'TIME_FORCE_CLOSE_TRIGGERED',       # expiración de tiempo máximo
                'ADAPTIVE_VIRTUAL_SL_TRIGGERED',    # AdaptiveSL comprimió hasta entry
                'ADAPTIVE_EXTENSION_SL_TRIGGERED',  # SL post-extensión de TP alcanzado
                'ADAPTIVE_HARD_SL_TRIGGERED',       # hard SL adaptativo activado
            ):
                ticket = int(msg.get('ticket', -1))
                pos = self.positions.pop(ticket, None)
                # Guardar último slippage observado
                if 'slippage_pts' in msg:
                    self.last_slippage_pts = float(msg['slippage_pts'])
                # v32.0: registrar cierre para el cooldown de pérdidas consecutivas.
                # Solo cierres por SL (no TP ni forzados por tiempo) computan como pérdida.
                # v42.0: añadir POSITION_CLOSED_EXTERNAL cuando ocurre inmediatamente
                # después de un VIRTUAL_SL_TRIGGERED en el mismo ticket. Con duplicados,
                # cuando un ticket cierra por vSL, el duplicado cierra como EXTERNAL —
                # antes esto hacía que solo la mitad de los SL se contabilizaran en la
                # racha, impidiendo que LOSS_STREAK_COOLDOWN disparara. Ahora EXTERNAL
                # se trata como SL si el ticket tenía una pérdida (pos.be_armed=False
                # indica que nunca llegó a BE, i.e. casi seguro perdió).
                _sl_close_reasons = {
                    'VIRTUAL_SL_TRIGGERED', 'ADAPTIVE_VIRTUAL_SL_TRIGGERED',
                    'ADAPTIVE_HARD_SL_TRIGGERED', 'ADAPTIVE_EXTENSION_SL_TRIGGERED',
                    'ADAPTIVE_SL_CLOSE',
                }
                _is_sl_close = event in _sl_close_reasons
                # POSITION_CLOSED_EXTERNAL contabiliza como SL si la posición
                # nunca llegó a armar BE (nunca fue ganadora → probable pérdida)
                _is_ext_loss = (
                    event == 'POSITION_CLOSED_EXTERNAL'
                    and pos is not None
                    and not pos.be_armed  # sin BE = nunca fue positiva
                )
                if (_is_sl_close or _is_ext_loss) and pos is not None:
                    _side = pos.side  # 'BUY' o 'SELL'
                    _ts_close = float(msg.get('ts', time.time()))
                    self.last_sl_close_by_side[_side].append((_ts_close, event))
                    # v46.0: diagnóstico LOSS_STREAK — loguear cada registro para
                    # confirmar que el listener recibe los cierres SL y los acumula.
                    import logging as _logging
                    _sl_diag_logger = _logging.getLogger('mimo_signals')  # v47.2
                    _sl_diag_logger.info(__import__('json').dumps({
                        'event': 'LOSS_STREAK_SL_REGISTERED',
                        'ts': _ts_close,
                        'ticket': ticket,
                        'side': _side,
                        'close_event': event,
                        'streak_count': len(self.last_sl_close_by_side[_side]),
                        'be_armed': pos.be_armed,
                        'is_ext_loss': bool(_is_ext_loss),
                    }))
                # Cierres por TP o TIME_FORCE_CLOSE se registran como "win/neutral"
                # para resetear la racha del side correspondiente.
                _tp_close_reasons = {
                    'VIRTUAL_TP_TRIGGERED', 'ADAPTIVE_TP_CLOSE',
                    'ADAPTIVE_EXTENSION_TP_TRIGGERED', 'TIME_FORCE_CLOSE_TRIGGERED',
                }
                if event in _tp_close_reasons and pos is not None:
                    _side = pos.side
                    self.last_sl_close_by_side[_side].clear()  # reset racha en ese side

            elif event == 'PARTIAL_CLOSE_TRIGGERED':
                ticket = int(msg.get('ticket', -1))
                if ticket in self.positions and msg.get('volume_updated'):
                    self.positions[ticket].volume = float(msg.get('remaining_volume', 0))
                    self.positions[ticket].partial_done = True

            elif event == 'BE_ARMED':
                ticket = int(msg.get('ticket', -1))
                if ticket in self.positions:
                    self.positions[ticket].be_armed = True
                    self.positions[ticket].virtual_sl = float(msg.get('new_virtual_sl', 0))

            elif event == 'TRAILING_UPDATED':
                ticket = int(msg.get('ticket', -1))
                if ticket in self.positions:
                    self.positions[ticket].virtual_sl = float(msg.get('new_virtual_sl', 0))

            # ── Eventos AdaptiveSLManager (v8.0) ─────────────────────────────
            elif event == 'ADAPTIVE_SL_UPDATE':
                # Compresión o expansión del SL virtual por el adaptativo
                ticket = int(msg.get('ticket', -1))
                if ticket in self.positions:
                    new_sl = msg.get('new_virtual_sl', 0)
                    if new_sl:
                        self.positions[ticket].virtual_sl = float(new_sl)

            # ── Eventos AdaptiveTPManager (v9.0) ─────────────────────────────
            # Nota: ADAPTIVE_SL_CLOSE consolidado en el bloque de cierres (v26.0)
            elif event == 'ADAPTIVE_TP_COMPRESSION':
                # TP comprimido: acercado al precio por señales de agotamiento
                ticket = int(msg.get('ticket', -1))
                if ticket in self.positions:
                    new_tp = msg.get('new_virtual_tp', 0)
                    if new_tp:
                        self.positions[ticket].virtual_tp = float(new_tp)

            elif event == 'ADAPTIVE_TP_EXTENSION':
                # TP extendido: nuevo objetivo + SL comprimido simultáneo
                ticket = int(msg.get('ticket', -1))
                if ticket in self.positions:
                    new_tp = msg.get('new_virtual_tp', 0)
                    new_sl = msg.get('new_virtual_sl', 0)
                    if new_tp:
                        self.positions[ticket].virtual_tp = float(new_tp)
                    if new_sl:
                        self.positions[ticket].virtual_sl = float(new_sl)

            # Nota: ADAPTIVE_TP_CLOSE consolidado en el bloque de cierres (v26.0)


def s3_event_listener(sub_socket: zmq.Socket, s3_state: S3State) -> None:
    """
    Hilo background que consume eventos publicados por S3 y actualiza S3State.
    No bloquea el loop principal de S2.
    v11.0: RCVTIMEO en el socket evita que el thread se cuelgue si S3 deja de publicar.
    v18.0: Excepciones loguean el error completo (antes se silenciaban con bare except).
           Se diferencia entre errores de parseo JSON (probablemente datos corruptos de S3)
           y errores en on_event (bug en la lógica de estado local). Ambos se registran
           en el logger de sistema para diagnóstico post-sesión.
    """
    _s2_logger = logging.getLogger('s2_system')
    _sig_logger = logging.getLogger('mimo_signals')  # v47.2: nombre correcto (era 'signal')
    _consecutive_errors = 0
    _MAX_CONSECUTIVE_ERRORS = 10

    # v47.0: contadores de diagnóstico del listener.
    # Permiten verificar si el listener recibe eventos de S3 en absoluto.
    # Se loguea un resumen cada _DIAG_INTERVAL_SECS segundos.
    _diag_total_received  = 0   # total de eventos recibidos desde arranque
    _diag_by_event: dict  = {}  # event → count
    _diag_last_log_ts     = time.time()
    _DIAG_INTERVAL_SECS   = 60  # loguear resumen cada 60s
    _diag_last_event_ts   = 0.0 # ts del último evento recibido (cualquiera)

    while True:
        try:
            raw = sub_socket.recv_string()
            msg = json.loads(raw)
            # v47.0: contabilizar antes de on_event para capturar aunque falle
            _diag_total_received += 1
            _ev = msg.get('event', 'UNKNOWN')
            _diag_by_event[_ev] = _diag_by_event.get(_ev, 0) + 1
            _diag_last_event_ts = time.time()
            s3_state.on_event(msg)
            _consecutive_errors = 0

            # v47.1: emitir S3_LISTENER_DIAG cada _DIAG_INTERVAL_EVENTS eventos
            # o cada _DIAG_INTERVAL_SECS segundos, lo que ocurra antes.
            # Antes solo se emitía en zmq.Again — pero con S3 activo (heartbeats
            # cada 10s, MODIFYs cada barra) el listener nunca llega al timeout
            # y el diagnóstico nunca se emitía. Ahora se emite también por conteo.
            _DIAG_INTERVAL_EVENTS = 200  # ~2 min con heartbeat+modify normales
            _now_event = time.time()
            _diag_by_time  = _now_event - _diag_last_log_ts >= _DIAG_INTERVAL_SECS
            _diag_by_count = _diag_total_received % _DIAG_INTERVAL_EVENTS == 0
            if _diag_by_time or _diag_by_count:
                _diag_last_log_ts = _now_event
                _sig_logger.info(json.dumps({
                    'event': 'S3_LISTENER_DIAG',
                    'ts': _now_event,
                    'total_received': _diag_total_received,
                    'trigger': 'time' if _diag_by_time else 'count',
                    'secs_since_last_event': round(_now_event - _diag_last_event_ts, 1)
                                             if _diag_last_event_ts > 0 else None,
                    'top_events': dict(sorted(
                        _diag_by_event.items(), key=lambda x: -x[1])[:10]),
                    'sl_in_state': {
                        'BUY':  len(s3_state.last_sl_close_by_side.get('BUY', [])),
                        'SELL': len(s3_state.last_sl_close_by_side.get('SELL', [])),
                    },
                    'n_positions_tracked': len(s3_state.get_positions()),
                }))
        except zmq.Again:
            # Timeout — S3 sin eventos durante RCVTIMEO=100ms.
            # v47.1: el diagnóstico principal ya se emite en el try block
            # por conteo/tiempo. Aquí solo forzamos si llevan > DIAG_INTERVAL sin emitir.
            _now_diag = time.time()
            if _now_diag - _diag_last_log_ts >= _DIAG_INTERVAL_SECS:
                _diag_last_log_ts = _now_diag
                _sig_logger.info(json.dumps({
                    'event': 'S3_LISTENER_DIAG',
                    'ts': _now_diag,
                    'total_received': _diag_total_received,
                    'trigger': 'timeout',
                    'secs_since_last_event': round(_now_diag - _diag_last_event_ts, 1)
                                             if _diag_last_event_ts > 0 else None,
                    'top_events': dict(sorted(
                        _diag_by_event.items(), key=lambda x: -x[1])[:10]),
                    'sl_in_state': {
                        'BUY':  len(s3_state.last_sl_close_by_side.get('BUY', [])),
                        'SELL': len(s3_state.last_sl_close_by_side.get('SELL', [])),
                    },
                    'n_positions_tracked': len(s3_state.get_positions()),
                }))
        except json.JSONDecodeError as e:
            _consecutive_errors += 1
            _s2_logger.error(json.dumps({
                'event': 'S3_LISTENER_JSON_ERROR',
                'ts': time.time(),
                'error': str(e),
                'raw_preview': raw[:200] if isinstance(raw, str) else repr(raw)[:200],
                'consecutive_errors': _consecutive_errors,
            }))
            if _consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                _s2_logger.error(json.dumps({
                    'event': 'S3_LISTENER_DEGRADED',
                    'ts': time.time(),
                    'consecutive_errors': _consecutive_errors,
                    'detail': 'Demasiados errores consecutivos en s3_event_listener — posible corrupción de mensajes S3',
                }))
            time.sleep(0.1)
        except Exception as e:
            _consecutive_errors += 1
            _s2_logger.error(json.dumps({
                'event': 'S3_LISTENER_ERROR',
                'ts': time.time(),
                'error': type(e).__name__,
                'detail': str(e),
                'msg_preview': str(msg)[:300] if 'msg' in dir() else 'unavailable',  # v48.0
                'consecutive_errors': _consecutive_errors,
            }))
            if _consecutive_errors >= _MAX_CONSECUTIVE_ERRORS:
                _s2_logger.error(json.dumps({
                    'event': 'S3_LISTENER_DEGRADED',
                    'ts': time.time(),
                    'consecutive_errors': _consecutive_errors,
                    'detail': 'Demasiados errores consecutivos en s3_event_listener — s3_state puede estar desincronizado',
                }))
            time.sleep(0.1)


def trading_engine(release: str):
    general_config = Config(
        release=release,
        oof_splits=5,
        oof_epochs=80
    )

    model_config = ModelConfig(
        seq_len_short=64,
        seq_len_long=256,
        epochs=80,
        patience=12,
        use_hierarchical_fusion=True,
        ranking_loss_weight=0.2,
    )

    feature_config = FeatureConfig(
        ema_periods=[9, 21, 50],
        label_horizon=10,
        tp_barrier=2.5,
        sl_barrier=1.5, # Para evitar que en aquellos casos en que el Virtual SL es muy justo, cierre a perdidas
        label_method='adaptive',
        feature_masks={
            'long': {'ema_bull': True, 'rsi_oversold': True, 'macd_positive': True,
                     'ema_bear': False, 'rsi_overbought': False, 'macd_negative': False},
            'short': {'ema_bear': True, 'rsi_overbought': True, 'macd_negative': True,
                      'ema_bull': False, 'rsi_oversold': False, 'macd_positive': False},
        },
    )

    regime_config = RegimeConfig(
        adx_trend_threshold=25.0
    )

    mode = 'production'
    decision_policy = DecisionPolicy(
        gate_by_action_and_state=gate_by_action_and_state[mode],
        score_cap_by_state=score_cap_by_state[mode],
        risk_mult_by_state=risk_mult_by_state[mode],
        score_low_quantile=75,
        score_high_quantile=99,
        require_delta_rel=True,
        min_delta_rel=0.20,
        allow_volatile=False,
    )

    risk_config = RiskConfig(
        base_risk_pct=0.0035,  # 0.5% equity
        min_score_to_trade=0.0, #0.2898742140685939
        max_risk_pct=0.02,
        max_positions=2
    )

    artifacts_path = f'../../../artifacts/{release}/oof/deploy_full'
    policy_path = f'../../../artifacts/{release}/rl/final/rl_policy_gate_{release}.npz'
    rl_config = {
        "lr": 0.002,  # FINETUNE_LR del freeze
        "entropy_coef": 0.1,  # igual que staged
        "baseline_beta": 0.88,  # igual que staged
        "chop_soft_thr": 0.3670136046832346,  # trial 121
        "exhaustion_soft_thr": 0.4805561001350015,  # trial 121
        "rl_take_threshold": 0.036, #0.1444771151557441,  # trial 121
        "chop_penalty_coef": 0.004458284077367999,  # trial 121
        "exhaustion_penalty_coef": 0.003,  # fijo staged
        "max_grad_norm": 5.0,
        "trade_cost_money": 0.0075,
        "batch_size": 64,
    }

    engine = TradingSimulator(
        general_config=general_config,
        model_config=model_config,
        feature_config=feature_config,
        regime_config=regime_config,
        decision_policy=decision_policy,
        risk_config=risk_config,
        artifacts_path=artifacts_path,
        use_rl=False,
        rl_config=rl_config,
        rl_train=False,
        rl_eval_deterministic=True,
        rl_take_threshold=rl_config['rl_take_threshold'],
        rl_policy_path=policy_path,
        spread_price=0.07,
        mtm_use_bid_ask=True,
        mtm_price_col='close',
        sizing_equity_mode='balance',
        max_daily_loss_pct=0.035,
        max_daily_profit_pct=None,
        compound=True,
        enable_live_scaler_updates=True,
        anomaly_block_threshold=1.2,  # FIX v10.1: subido de 0.8 → 1.2 (0.8 bloqueaba 3h en sesión europea XAUUSD)
        signal_cooldown_bars=3
    )

    engine.load_artifacts()
    if engine.use_rl:
        ok = load_rl_policy_npz(engine.rl_wrapper, policy_path)
        if not ok:
            raise RuntimeError(f'WARNING! RL Policy was not loaded from {policy_path}')

    return engine

# ── AÑADIR al bloque de imports (arriba del todo) ──────────────────────────
import logging
from pathlib import Path

S2_SCRIPT_NAME = "main_trading_s2_v50.py"
S2_VERSION = "50.0"


# ── AÑADIR como función, junto a price_to_points / send_order ──────────────

def setup_signal_logger(log_dir: str = None) -> logging.Logger:
    """
    Crea un logger dedicado a señales del modelo.
    Escribe en JSONL rotando automáticamente a medianoche: signals_YYYYMMDD.jsonl

    v36.0: log_dir ahora se resuelve relativo al directorio del propio script
    (Path(__file__).parent / 'logs') en lugar de '../logs' relativo al CWD.

    v41.0: FIX rotación de log.
    Problema anterior: TimedRotatingFileHandler abre el fichero con la fecha del
    DÍA DE ARRANQUE y al rotar a medianoche lo renombra con sufijo pero el fichero
    activo sigue siendo signals_20260316.jsonl. Toda la sesión de noche se escribe
    en el fichero del día anterior.
    Fix: usar un namer() callback que nombra el fichero ACTIVO con la fecha actual
    en lugar de la fecha de arranque. El fichero rotado recibe el sufijo de la fecha
    anterior. Así signals_YYYYMMDD.jsonl corresponde siempre al día correcto.

    FIX v6.0: sustituido FileHandler (fecha fija en arranque) por
    TimedRotatingFileHandler (rotación real a medianoche sin reiniciar el servicio).
    """
    from logging.handlers import TimedRotatingFileHandler

    # v36.0: ruta anclada al directorio del script, independiente del CWD
    if log_dir is None:
        log_dir = str(Path(__file__).parent / 'logs')

    Path(log_dir).mkdir(parents=True, exist_ok=True)

    def _make_logger(name: str, base_stem: str) -> logging.Logger:
        """
        Crea (o reutiliza) un logger con TimedRotatingFileHandler correcto.
        El fichero activo siempre tiene la fecha de HOY: {base_stem}_YYYYMMDD.jsonl
        Al rotar, el fichero anterior queda como {base_stem}_YYYYMMDD.jsonl.old
        (sin importar cuándo arrancó S2).
        """
        logger = logging.getLogger(name)
        logger.setLevel(logging.DEBUG)

        if logger.handlers:
            return logger  # ya configurado (reinicio en el mismo proceso)

        date_str = pd.Timestamp.now().strftime('%Y%m%d')
        log_path = Path(log_dir) / f'{base_stem}_{date_str}.jsonl'

        fh = TimedRotatingFileHandler(
            filename=str(log_path),
            when='midnight',
            interval=1,
            backupCount=30,
            encoding='utf-8',
            utc=False,
        )
        # v41.0: namer callback — el fichero ACTIVO usa la fecha de HOY.
        # TimedRotatingFileHandler por defecto renombra el fichero viejo con un sufijo
        # y el nuevo fichero hereda el nombre base (signals_20260316.jsonl sin importar
        # la fecha real). Con namer(), el nuevo fichero se abre con la fecha actual.
        def _namer(default_name: str) -> str:
            # default_name = base_path + '.' + suffix (e.g. signals_20260316.jsonl.20260317)
            # Extraemos la fecha del sufijo para construir el nombre correcto del backup.
            # El fichero ACTIVO se reabre con la fecha actual (ya está en el filename del handler).
            return default_name  # el backup queda con sufijo de fecha (comportamiento estándar)

        # El truco real: rotator que al crear el nuevo fichero lo nombra con la fecha actual
        _log_dir_path = Path(log_dir)
        _base_stem_ref = base_stem
        def _rotator(source: str, dest: str) -> None:
            import os, shutil
            # Renombrar el fichero actual (source) al destino de backup (dest)
            if os.path.exists(source):
                shutil.move(source, dest)
            # Abrir el nuevo fichero con la fecha actual
            new_date = pd.Timestamp.now().strftime('%Y%m%d')
            new_path = str(_log_dir_path / f'{_base_stem_ref}_{new_date}.jsonl')
            # TimedRotatingFileHandler usará self.baseFilename para el stream nuevo.
            # Actualizamos el baseFilename del handler para que apunte a la fecha de hoy.
            fh.baseFilename = new_path

        fh.namer   = _namer
        fh.rotator = _rotator
        fh.suffix  = '%Y%m%d'
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(fh)
        return logger

    signal_logger = _make_logger('mimo_signals', 'signals')
    sys_logger    = _make_logger('s2_system',    's2_system')

    date_str = pd.Timestamp.now().strftime('%Y%m%d')
    log_path     = Path(log_dir) / f'signals_{date_str}.jsonl'
    sys_log_path = Path(log_dir) / f's2_system_{date_str}.jsonl'

    signal_logger.info(json.dumps({'event': 'LOGGER_STARTED', 'log_file': str(log_path),
                                   'ts': time.time()}))
    sys_logger.info(json.dumps({'event': 'SYSTEM_LOGGER_STARTED',
                                'log_file': str(sys_log_path), 'ts': time.time()}))

    return signal_logger

def log_signal(
    logger: logging.Logger,
    event: str,                        # 'SIGNAL_SENT' | 'SIGNAL_BLOCKED' | 'NO_SIGNAL'
    bar_data: dict,                    # datos del tick/vela recibidos
    live_order: Optional[Dict] = None, # la señal del modelo (puede ser None)
    block_reason: Optional[str] = None,# motivo de bloqueo si aplica
    request: Optional[Dict] = None,    # el request construido, si se envió
    n_positions: int = 0,
    equity: float = 0.0,
    balance: float = 0.0,
    model_diag: Optional[Dict] = None, # diagnóstico del modelo aunque no haya señal
    indicators: Optional[Dict] = None, # indicadores enviados al AdaptiveSLManager (v8.0)
) -> None:
    """
    Registra en JSONL cada decisión del motor: señal enviada, bloqueada o ausente.

    Estructura del registro:
      - ts / bar_time / event
      - regime / side / score / proba_long / proba_short / delta_rel
      - entry / sl / tp / qty / atr
      - sl_points / tp_points / rr_ratio
      - n_positions / equity / balance
      - block_reason (si SIGNAL_BLOCKED)
      - model_diag (en NO_SIGNAL: score/regime/side del modelo aunque se haya filtrado)
      - request (si SIGNAL_SENT)
    """
    record: Dict[str, Any] = {
        'event':       event,
        'ts':          time.time(),
        'bar_time':    str(bar_data.get('time', '')),
        'open':        bar_data.get('open'),
        'high':        bar_data.get('high'),
        'low':         bar_data.get('low'),
        'close':       bar_data.get('close'),
        'n_positions': n_positions,
        'equity':      round(equity, 2),
        'balance':     round(balance, 2),
    }

    if live_order:
        entry = float(live_order.get('entry', 0) or 0)
        sl    = float(live_order.get('sl',    0) or 0)
        tp    = float(live_order.get('tp',    0) or 0)
        atr   = float(live_order.get('atr_at_entry', 0) or 0)
        qty   = float(live_order.get('qty',   0) or 0)

        sl_pts = price_to_points(entry, sl) if entry and sl else None
        tp_pts = price_to_points(entry, tp) if entry and tp else None
        rr     = round(tp_pts / sl_pts, 2) if sl_pts and tp_pts and sl_pts > 0 else None

        record.update({
            's2_script':   S2_SCRIPT_NAME,
            's2_version':  S2_VERSION,
            'side':        live_order.get('side'),
            # 'state' es la clave canónica en live_order (market_condition es alias deprecado)
            'state':       live_order.get('state') or live_order.get('market_condition'),
            'score':       round(float(live_order.get('score', 0) or 0), 6),
            # proba_long / proba_short: ahora expuestos directamente por _build_order
            'proba_long':  round(float(live_order.get('proba_long',  live_order.get('proba_cal', 0)) or 0), 6),
            'proba_short': round(float(live_order.get('proba_short', 0) or 0), 6),
            'delta_rel':   round(float(live_order.get('delta_rel',   0) or 0), 6),
            'score_long':  round(float(live_order.get('score_long',  0) or 0), 6),
            'score_short': round(float(live_order.get('score_short', 0) or 0), 6),
            'entry':       entry,
            'sl':          sl,
            'tp':          tp,
            'qty':         qty,
            'atr':         round(atr, 4),
            'sl_points':   sl_pts,
            'tp_points':   tp_pts,
            'rr_ratio':    rr,
        })

    if block_reason:
        record['block_reason'] = block_reason

    # Diagnóstico del modelo en ticks sin señal: permite ver qué estaba viendo
    # el modelo (score, regime, side candidato) aunque la señal se haya filtrado.
    # v18.0: se elimina 'macro_regime' del registro — el campo siempre llega vacío
    # (el detector de régimen macro no está integrado en el pipeline actual) y
    # genera ruido en el log sin aportar información útil.
    if model_diag:
        _diag_clean = {k: v for k, v in model_diag.items() if k != 'macro_regime'}
        record['model_diag'] = _diag_clean

    if request:
        record['req_side']          = request.get('side')
        record['req_volume']        = request.get('volume')
        record['req_virtual_sl']    = request.get('virtual_sl')
        record['req_virtual_tp']    = request.get('virtual_tp')
        record['req_emergency_pts'] = request.get('risk', {}).get('emergency_points')
        # v41.0: extraer tp_rr_capped y rr real del request (antes solo se logeaba
        # rr_ratio del modelo, que no refleja el cap aplicado por MAX_RR_BY_REGIME)
        _req_meta = request.get('metadata', {})
        if _req_meta.get('tp_rr_capped') is not None:
            record['tp_rr_capped']     = _req_meta['tp_rr_capped']
            record['tp_max_rr_regime'] = _req_meta.get('tp_max_rr_regime')
            record['tp_rr_corrected']  = _req_meta.get('tp_rr_corrected')
        # RR real enviado a S3 (distinto de rr_ratio que viene del modelo)
        _rvtp = request.get('virtual_tp')
        _rvsl = request.get('virtual_sl')
        _rent = _req_meta.get('entry') or record.get('entry')
        if _rvtp and _rvsl and _rent:
            try:
                record['req_rr_ratio'] = round(abs(float(_rvtp) - float(_rent)) /
                                               abs(float(_rvsl) - float(_rent)), 2)
            except (ZeroDivisionError, TypeError):
                pass

    if indicators:
        record['indicators'] = indicators

    logger.info(json.dumps(record, ensure_ascii=False))

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = 'true'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce logs de TensorFlow
os.environ['TF_GPU_THREAD_MODE'] = 'gpu_private'
os.environ['TF_GPU_THREAD_COUNT'] = '1'

import tensorflow as tf

gpus = tf.config.list_physical_devices('GPU')

context = zmq.Context.instance()
subscription = context.socket(zmq.SUB)
subscription.setsockopt(zmq.SUBSCRIBE, b'XAUUSD.r')
subscription.setsockopt(zmq.RCVHWM, 10000)
subscription.setsockopt(zmq.RCVTIMEO, 30_000)   # v11.0: 30s timeout — evita cuelgue si MT5 se desconecta
subscription.connect(f'tcp://10.1.21.25:5555')
time.sleep(1.0)

# --- S3 (Execution & Monitoring) via ZeroMQ ---
S3_ORDERS_ADDR = "tcp://10.1.21.25:5557"  # s3_executor PULL bind
S3_EVENTS_ADDR = "tcp://10.1.21.25:5558"  # s3_executor PUB bind

orders_push = context.socket(zmq.PUSH)
orders_push.setsockopt(zmq.SNDHWM, 10000)
orders_push.setsockopt(zmq.LINGER, 0)
orders_push.connect(S3_ORDERS_ADDR)

# v6.0: suscribirse a eventos S3 para mantener estado local sincronizado
events_sub = context.socket(zmq.SUB)
events_sub.setsockopt(zmq.SUBSCRIBE, b'')   # todos los eventos (no hay topic prefix en S3)
events_sub.setsockopt(zmq.RCVHWM, 10000)
events_sub.setsockopt(zmq.LINGER, 0)
events_sub.setsockopt(zmq.RCVTIMEO, 100)    # v27.0: 100ms — procesa eventos S3 casi en tiempo real
events_sub.connect(S3_EVENTS_ADDR)

# Estado local sincronizado con S3
s3_state = S3State()

# Lanzar listener en background
_s3_listener_thread = threading.Thread(
    target=s3_event_listener,
    args=(events_sub, s3_state),
    daemon=True,
    name='s3-event-listener'
)
_s3_listener_thread.start()
print(f"[S2] S3 event listener started → {S3_EVENTS_ADDR}")

db = Database()
db.connect()

release='200372'

engine = trading_engine(release=release)
signal_logger = setup_signal_logger()  # v36.0: log_dir resuelto al directorio del script

# v11.0: watchdog — registra el último tick para detectar desconexiones
_last_tick_ts: float = time.time()
_NO_TICK_WARN_SECS = 120   # avisa si no llega ningún tick en 2 min

# v19.0: SMA incremental de volumen — instancia persistente entre ticks.
# Reemplaza _volume_ma_from_rates() que recalculaba las 20 barras en cada tick.
# Se alimenta con una sola barra por tick en lugar del df_rates completo.
_volume_sma = IncrementalSMA(period=VOLUME_MA_PERIOD)

def _prefill_volume_sma(sma: IncrementalSMA, db_instance, n_bars: int = VOLUME_MA_PERIOD) -> None:
    """
    Pre-rellena el buffer de _volume_sma con las últimas n_bars de df_rates.

    v35.0: sin esto, tras cada restart de S2 el buffer queda vacío y volume_ma=None
    durante ~20 minutos mientras se acumulan las 20 barras necesarias.
    v40.0: mejor diagnóstico del fallo + acepta tick_volume=0 como dato válido
    (antes se filtraba _fv > 0, descartando ticks quietos con volumen=0 real).
    """
    try:
        _df = engine.helper.load_from_database_real(db_instance, last_n_rates=n_bars + 5)
        if _df is None or len(_df) == 0:
            print(f"[S2] volume_sma pre-fill: load_from_database_real returned empty/None")
            return
        # Mostrar columnas disponibles para diagnóstico
        _vol_col = next((c for c in ('volume', 'tick_volume', 'ticks_volume', 'real_volume') if c in _df.columns), None)
        if _vol_col is None:
            print(f"[S2] volume_sma pre-fill: no volume column in df. Columns: {list(_df.columns)}")
            return
        _series = _df[_vol_col].dropna().tail(n_bars)
        _pushed = 0
        for _v in _series:
            try:
                _fv = float(_v)
                if not math.isnan(_fv):   # v40: acepta 0.0 — excluir NaN basta
                    sma.push(_fv)
                    _pushed += 1
            except (TypeError, ValueError):
                pass
        if _pushed > 0:
            print(f"[S2] volume_sma pre-filled with {_pushed}/{len(_series)} bars"
                  f" (col={_vol_col}) → value={sma.value:.1f}")
        else:
            print(f"[S2] volume_sma pre-fill: 0 valid rows in {len(_series)} bars"
                  f" (col={_vol_col}, all NaN or empty)")
    except Exception as _e:
        print(f"[S2] volume_sma pre-fill failed: {_e}")

# v35.0: pre-rellenar _volume_sma desde df_rates para que volume_ma esté disponible
# desde el primer tick tras un restart, sin esperar 20 barras (~20 min).
_prefill_volume_sma(_volume_sma, db)

# v23.0: keepalive de indicadores — garantiza que S3 recibe MODIFY al menos cada
# KEEPALIVE_INTERVAL_SECS por posición, independientemente de la actividad del
# pipeline. Evita que S3 emita ADAPTIVE_INDICATORS_STALE cuando el pipeline
# tarda en procesar (reconexión, warmup, ticks lentos).
# _keepalive_last_sent: ticket → timestamp del último MODIFY enviado (cualquier causa)
# _keepalive_last_indicators: ticket → último dict de indicadores enviado (fallback
#   si el pipeline no tiene datos frescos en el tick del keepalive)
_keepalive_last_sent:       dict = {}   # ticket(int) → float (ts)
_keepalive_last_indicators: dict = {}   # ticket(int) → dict
_keepalive_last_context:    dict = {}   # ticket(int) → dict  — v24.0: regime/score/spread
_keepalive_last_bar_time:   dict = {}   # ticket(int) → int (bar_time del último MODIFY) — v33.0

# v49.0 FIX-2: callback asignado a s3_state para registrar el ticket real en
# _keepalive_last_sent en cuanto el listener recibe POSITION_OPENED de S3.
# ts=0 garantiza que BLOQUE B envíe MODIFY en el siguiente tick (timeout_expired=True).
# Definido aquí porque referencia _keepalive_last_sent, que debe existir antes.
def _on_position_opened_keepalive(ticket: int) -> None:
    _keepalive_last_sent.setdefault(ticket, 0)

s3_state._on_position_opened_cb = _on_position_opened_keepalive

# v38.0: guardia anti-duplicado de órdenes abiertas.
# Problema: S2 corre a 5Hz (200ms/tick). Al enviar una orden, S3 tarda ~50-200ms
# en abrir la posición y publicar POSITION_OPENED. El s3_event_listener (hilo
# separado) actualiza s3_state con otros ~50-100ms de latencia ZMQ.
# En esa ventana de ~100-400ms llegan 1-2 ticks más: effective_positions sigue
# siendo 0, decide_live devuelve la misma señal y S2 envía una segunda orden
# idéntica. Con MAX_POSITIONS=2, esto produce 2 trades simultáneos iguales.
#
# Fix: al enviar una orden, registrar el bar_time del tick actual.
# En los siguientes ticks del mismo bar_time: bloquear nuevas órdenes durante
# OPEN_GUARD_SECS, independientemente de lo que diga effective_positions.
# El guard se limpia automáticamente cuando llega una nueva barra (bar_time cambia).
_open_guard = {
    'bar_time': 0,    # bar_time (raw broker unix) del último OPEN enviado
    'sent_ts':  0.0,  # time.time() del último OPEN enviado
}
OPEN_GUARD_SECS: float = 5.0        # tiempo mínimo entre dos órdenes de apertura

# v46.0: gracia de arranque — bloquear OPENs durante el primer ciclo completo.
# Causa: al arrancar S2, el buffer ZMQ PUSH de S3 puede contener ORDER_REQUESTs
# de la sesión anterior (con vTP del config antiguo). S3 los procesa en los
# primeros ciclos. Si S2 envía un OPEN inmediatamente, S3 lo encola junto con
# los mensajes del buffer — el primero que pasa el DEDUP guard usa el vTP viejo.
# Fix: en el primer ciclo (primer bar_time visto), bloquear OPEN con
# STARTUP_OPEN_BLOCKED. El buffer ZMQ se vaciará en ese ciclo (OPEN_DEDUP
# de S3 rechaza todos los del buffer) y el siguiente ciclo ya opera limpio.
# No afecta a MODIFYs ni a ninguna otra lógica.
_startup_bars_seen: set = set()   # bar_times procesados desde el arranque
STARTUP_GRACE_BARS: int = 1       # nº de barras iniciales donde se bloquean OPENs

# v32.0: estado de cooldown por pérdidas consecutivas.
# _loss_streak_closes: side → list[(ts_close, reason)] — historial reciente de cierres
# _loss_streak_cooldown_until: side → float (ts hasta el que está bloqueado, 0=libre)
# Se actualiza en el s3_event_listener a través de s3_state, y se lee en el
# bloque de filtros pre-envío. El estado es por side ('BUY'/'SELL') e independiente.
_loss_streak_closes:          dict = {'BUY': [], 'SELL': []}   # side → [(ts, reason)]
_loss_streak_cooldown_until:  dict = {'BUY': 0.0, 'SELL': 0.0} # side → ts de fin de cooldown

# v14.0 (BUG-2): reconexión automática al socket MT5 cuando el mercado reabre.
# Cuando MT5 cierra el mercado su socket PUB desaparece o se reinicia. ZMQ
# mantiene el SUB "conectado" internamente pero sin recibir mensajes, y como
# zmq.Again es idéntico en ambos casos (mercado cerrado vs publisher caído),
# S2 nunca detectaba que necesitaba reconectar.
# Solución: contar timeouts consecutivos. Tras _ZMQ_RECONNECT_AFTER_TIMEOUTS
# sin ningún tick (= 40 * 30s = 20 min por defecto), recrear el socket.
_MT5_SUB_ADDR  = 'tcp://10.1.21.25:5555'
_MT5_SUB_TOPIC = b'XAUUSD.r'
_ZMQ_RECONNECT_AFTER_TIMEOUTS = 40   # 40 × 30s = 20 min sin ticks → reconectar
_no_tick_count:   int   = 0          # timeouts consecutivos desde el último tick
_reconnect_tries: int   = 0          # intentos de reconexión en la racha actual

def _reconnect_subscription() -> zmq.Socket:
    """
    Descarta el socket SUB viejo y crea uno nuevo conectado a MT5.
    Devuelve el nuevo socket listo para usar.
    """
    global subscription
    try:
        subscription.setsockopt(zmq.LINGER, 0)
        subscription.close()
    except Exception:
        pass
    s = context.socket(zmq.SUB)
    s.setsockopt(zmq.SUBSCRIBE, _MT5_SUB_TOPIC)
    s.setsockopt(zmq.RCVHWM, 10000)
    s.setsockopt(zmq.RCVTIMEO, 30_000)
    s.connect(_MT5_SUB_ADDR)
    return s

# v11.0: pool para db.save() asíncrono (no bloquea el loop principal)
import concurrent.futures
_db_executor = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix='db-save')

def _db_save_async(df_to_save):
    """Guarda en BD. Lanza la excepción para que .result() la propague al loop principal.
    Si se traga silenciosamente, .result() devuelve None aunque el save haya fallado,
    y load_from_database_real lee la última barra guardada con éxito → modelo ve datos
    obsoletos → misma señal repetida indefinidamente (bug confirmado 13/03/2026).
    """
    db.save(df_to_save, table_name='rates')

while True:
    # v11.0: recv con timeout — si MT5 se desconecta no se queda colgado para siempre
    try:
        topic, raw = subscription.recv_multipart()
    except zmq.Again:
        elapsed = time.time() - _last_tick_ts
        if elapsed > _NO_TICK_WARN_SECS:
            print(f"[WARN][S2] Sin ticks desde hace {elapsed:.0f}s — MT5 desconectado o mercado cerrado")

        # v14.0 (BUG-2): reconexión automática.
        # Cada zmq.Again incrementa el contador. Si supera el umbral, el socket
        # PUB de MT5 probablemente desapareció (cierre de mercado + reinicio EA).
        # Recreamos el socket con backoff exponencial para no saturar la red.
        _no_tick_count += 1
        if _no_tick_count >= _ZMQ_RECONNECT_AFTER_TIMEOUTS:
            _no_tick_count = 0
            _reconnect_tries += 1
            backoff = min(2 ** (_reconnect_tries - 1) * 5, 300)  # 5s, 10s, 20s … cap 300s
            print(f"[WARN][S2] Reconectando socket MT5 (intento #{_reconnect_tries}, backoff={backoff}s)…")
            time.sleep(backoff)
            subscription = _reconnect_subscription()
            print(f"[INFO][S2] Socket MT5 recreado → {_MT5_SUB_ADDR}")
            _volume_sma.reset()  # vaciar primero
            _prefill_volume_sma(_volume_sma, db)  # v35.0: re-rellenar desde BD tras reconexión
        continue

    # Tick válido recibido — resetear contadores de reconexión
    _last_tick_ts   = time.time()
    _no_tick_count  = 0
    _reconnect_tries = 0
    data = json.loads(raw.decode("utf-8"))

    print(f'Data from MT5: {data}')
    symbol = data.pop('symbol')
    free_margin = data.pop('free_margin')
    balance = data.pop('balance')
    equity = data.pop('equity')
    n_positions   = data.pop('n_positions')
    # FIX v30.8: tickets de posiciones abiertas enviados por S1 (lista de ints).
    # Fallback para cuando s3_state no ha recibido POSITION_OPENED tras restart de S3.
    _s1_open_tickets = set(data.pop('open_tickets', []) or [])

    df = pd.DataFrame.from_dict([data])
    df['time'] = pd.to_datetime(df['time'] - 2 * 3600, unit='s', utc=True)
    df = df.drop(columns=['real_volume'], errors='ignore')
    df = df.rename(columns={"tick_volume": 'volume', "ticks_volume": 'volume'})
    # v30.5: bid, ask, tick_ok se incluyen si la BD tiene las columnas (post-migración).
    # Si la BD es antigua (pre-migración), dropeamos para no romper el INSERT.
    # Ejecutar migrate_rates_add_tick_columns.sql para habilitar el guardado de tick data.
    _TICK_COLS = ['bid', 'ask', 'tick_ok']
    _has_tick_cols = getattr(db, '_rates_has_tick_cols', None)
    if _has_tick_cols is None:
        try:
            import sqlalchemy
            with db.engine.connect() as _conn:
                _insp = sqlalchemy.inspect(db.engine)
                _cols = {c['name'] for c in _insp.get_columns('rates')}
                _has_tick_cols = all(c in _cols for c in _TICK_COLS)
        except Exception:
            _has_tick_cols = False
        db._rates_has_tick_cols = _has_tick_cols
        print(f"[DB] rates tick columns (bid/ask/tick_ok) present: {_has_tick_cols}")
    if not _has_tick_cols:
        df = df.drop(columns=_TICK_COLS, errors='ignore')
    # v30.1 (FIX-4): esperar a que el save termine ANTES de load_from_database_real.
    # v30.4 (FIX-7): _db_save_async ya no traga excepciones silenciosamente.
    # Si db.save() falla, .result() lanza la excepción aquí y saltamos el tick.
    # Antes: _db_save_async tenía try/except que devolvía None en fallo → el loop
    # continuaba con load_from_database_real → modelo veía la barra 14:06 en todos
    # los ticks siguientes → entry_time siempre = 14:06 durante el fallo.
    _db_save_future = _db_executor.submit(_db_save_async, df.copy())
    try:
        _db_save_future.result()  # espera bloqueante; lanza si db.save() falló
    except Exception as _db_err:
        print(f"[ERROR][db-save] Fallo al guardar barra en BD — saltando tick: {_db_err}")
        continue  # próximo tick del while True

    bar_snapshot = {
        'time':   data.get('time'),
        'open':   data.get('open'),
        'high':   data.get('high'),
        'low':    data.get('low'),
        'close':  data.get('close'),
        'spread': data.get('spread'),   # v20.0: incluido para SIGNAL_BLOCKED spread
    }

    # v20.0: reducido de 2048 a 800 filas.
    # El cuello de botella no son los indicadores técnicos sino el RollingScaler
    # del pipeline: scaler_warmup_size=390 (confirmado en data_pipeline.py línea 49).
    # El scaler necesita 390 filas de warmup antes de producir valores escalados
    # válidos. A eso se suma seq_len_long=256 para construir la última secuencia.
    # Mínimo real: 390 + 256 = 646 filas. Se usa 800 como margen de seguridad (~25%).
    # Con 400 filas el df quedaba vacío tras dropna() (solo ~10 filas post-warmup),
    # causando IndexError en live_update_scalers_from_df (time.iloc[-1] sobre Serie vacía).
    # Impacto vs 2048: carga BD y prepare_data ~2.5× más rápidos.
    df_rates = engine.helper.load_from_database_real(db, last_n_rates=800)

    # v27.0 (FIX-1): calcular effective_positions ANTES de decide_live.
    # Antes: decide_live recibía n_positions crudo de MT5, que podía incluir
    # zombies o ir retrasado respecto a S3. Resultado: MAX_POSITIONS(2/2) incluso
    # con solo 1 posición real, bloqueando señales y el bloque MODIFY.
    # Ahora: se usa max(mt5, s3) como fuente de verdad para el engine también.
    _s3_n_pre = s3_state.n_positions()
    effective_positions = max(n_positions, _s3_n_pre)
    if _s3_n_pre != n_positions:
        print(f"\t[S3_SYNC] s3={_s3_n_pre} vs mt5={n_positions} → using effective={effective_positions}")

    live_order = engine.decide_live(df_rates=df_rates, equity=equity, current_positions=effective_positions, max_positions=2)

    # v20.0: se elimina la segunda llamada a pipeline.prepare_data() que existía
    # desde v14.0. predict_live() (nuevo método en TradingSimulator) ya ejecuta
    # prepare_data() internamente y expone el df_prepared a través de
    # engine._last_df_prepared — reutilizamos ese resultado en lugar de recalcular.
    # Esto elimina una ejecución completa de prepare_data por tick (~100-200ms).
    _df_prepared = getattr(engine, '_last_df_prepared', None)
    if _df_prepared is None and hasattr(engine, 'pipeline'):
        # Fallback por si predict_live no está disponible (versión antigua del engine)
        try:
            _df_prepared = engine.pipeline.prepare_data(df_rates)
        except Exception as _e:
            print(f'[WARN][v20] pipeline.prepare_data fallback failed: {_e}')

    # Extraer indicadores del pipeline (última fila y penúltima para macd_hist_prev)
    _pip_last  = _df_prepared.iloc[-1]  if _df_prepared is not None and len(_df_prepared) >= 1 else None
    _pip_prev  = _df_prepared.iloc[-2]  if _df_prepared is not None and len(_df_prepared) >= 2 else None

    def _pip(row, col):
        """Extrae un valor float del df_prepared; devuelve None si no existe o es NaN."""
        if row is None: return None
        v = row.get(col) if hasattr(row, 'get') else (row[col] if col in row.index else None)
        if v is None: return None
        # v19.0: math importado a nivel de módulo (antes: import math dentro de esta función)
        return None if (isinstance(v, float) and math.isnan(v)) else float(v)

    # v21.0: extraer bandas de Bollinger del pipeline para ajuste de TP en rango.
    # El pipeline debe producir columnas 'bb_upper' y 'bb_lower'. Si no existen,
    # _pip devuelve None y create_order_request mantiene el TP original (sin cambios).
    _bb_upper = _pip(_pip_last, 'bb_upper')
    _bb_lower = _pip(_pip_last, 'bb_lower')

    # v19.0: actualizar SMA incremental con el volumen de la barra actual.
    # Una sola operación O(1) en lugar de recalcular df[col].iloc[-period:].mean()
    # sobre el df_rates completo (O(period) + overhead de slice Pandas).
    # El volumen actual viene de data (tick MT5 con tick_volume renombrado a volume
    # tras el df.rename en líneas anteriores), o como fallback desde la última fila
    # de df_rates si el tick no lo trae directamente.
    _tick_volume = data.get('volume') or data.get('tick_volume')
    if _tick_volume is None and df_rates is not None and len(df_rates) > 0:
        _vol_col = 'volume' if 'volume' in df_rates.columns else (
                   'tick_volume' if 'tick_volume' in df_rates.columns else (
                   'ticks_volume' if 'ticks_volume' in df_rates.columns else None))
        if _vol_col:
            _tick_volume = df_rates[_vol_col].iloc[-1]
    if _tick_volume is not None:
        try:
            _tv = float(_tick_volume)
            if not math.isnan(_tv) and _tv > 0:
                _volume_sma.push(_tv)
        except (TypeError, ValueError):
            pass

    floating = equity - balance

    # ── v34.0: extracción de indicadores para AdaptiveSL/TP ──────────────────
    # Tres bugs corregidos respecto a v33:
    #
    # BUG-1 (crítico) — `or` silenciaba valores 0.0 válidos:
    #   `_pip(..., 'macd_hist') or fallback` → si macd_hist=0.0 (cruce de cero),
    #   Python evalúa 0.0 como False y cae al siguiente fallback → None → la clave
    #   queda ausente en el dict. S3 recibe macd_hist=None exactamente en el bar
    #   más relevante para la compresión del SL. Mismo problema en rsi, proba_long,
    #   proba_short, volume cuando sus valores son cero.
    #   Fix: helper _first_not_none() que usa `is not None` en lugar de truthiness.
    #
    # BUG-2 (importante) — data.get('atr') siempre None:
    #   ATR no es un campo del tick MT5. La primera fuente era siempre None.
    #   Fix: reordenar → pipeline primero, _diag.get('atr') como fallback final.
    #   Eliminado data.get('atr') de la cadena.
    #
    # BUG-3 (importante) — proba_long/short ausentes entre señales:
    #   Entre señales (live_order=None) _ind_source = _diag. Si _last_diag no
    #   expone probas, son None en todos los ticks sin señal activa → el AdaptiveSL
    #   nunca puede disparar model_flip compression entre señales.
    #   Fix: añadir _pip(_pip_last, 'proba_long/short') como fuente intermedia,
    #   por si el pipeline expone esas columnas en df_prepared.

    _diag = getattr(engine, '_last_diag', None) or {}
    _ind_source = live_order or _diag

    def _first_not_none(*vals):
        """Primer valor no-None. Seguro para 0.0, 0, False — a diferencia de `or`."""
        for v in vals:
            if v is not None:
                return v
        return None

    current_indicators = {k: v for k, v in {
        # ATR: pipeline > señal activa > _diag. data.get('atr') eliminado (no es campo MT5).
        'atr': _first_not_none(
            _pip(_pip_last, 'atr'),
            _ind_source.get('atr_at_entry'),
            _diag.get('atr'),
        ),
        # RSI: pipeline > tick > _ind_source. Seguro para rsi=0.0.
        'rsi': _first_not_none(
            _pip(_pip_last, 'rsi'),
            data.get('rsi'),
            _ind_source.get('rsi'),
        ),
        # macd_hist: pipeline primero — CRÍTICO para cruces de cero (macd_hist=0.0).
        'macd_hist': _first_not_none(
            _pip(_pip_last, 'macd_hist'),
            data.get('macd_hist'),
            _ind_source.get('macd_hist'),
        ),
        # macd_hist_prev: penúltima fila del pipeline. Seguro para 0.0.
        'macd_hist_prev': _first_not_none(
            _pip(_pip_prev, 'macd_hist'),
            _ind_source.get('macd_hist_prev'),
        ),
        # volume: tick MT5. Puede ser 0 en ticks quietos — conservar con _first_not_none.
        'volume': _first_not_none(
            data.get('volume'),
            data.get('tick_volume'),
        ),
        # volume_ma: SMA incremental O(1). Sin cambio.
        'volume_ma': _volume_sma.value,
        # proba_long/short: señal activa > pipeline (BUG-3) > _diag.
        'proba_long': _first_not_none(
            _ind_source.get('proba_long'),
            _ind_source.get('proba_cal'),
            _pip(_pip_last, 'proba_long'),
        ),
        'proba_short': _first_not_none(
            _ind_source.get('proba_short'),
            _pip(_pip_last, 'proba_short'),
        ),
    }.items() if v is not None}

    # ── v24.0: contexto de mercado para MODIFY ───────────────────────────────
    # Construido en cada tick independientemente de si hay señal activa.
    # Fuente: live_order (señal activa) > _diag (último diagnóstico) > None.
    # 'spread' viene directamente del tick MT5 (disponible en todos los ticks).
    _ctx_source = live_order or _diag
    current_context = {k: v for k, v in {
        'state': _ctx_source.get('state'),
        'score':  live_order.get('score') if live_order else None,  # solo si hay señal activa
        'spread': data.get('spread'),
    }.items() if v is not None}

    # ── FIX v25.0 (BUG-1) + v24.0 (CONTEXT): MODIFY enriquecido ──────────────
    # v25: REESCRITURA del bloque de MODIFY para corregir el bug que causó
    #   0 POSITION_MODIFIED en toda la sesión 12/03/2026.
    #
    # CAUSA DEL BUG (v23/v24):
    #   El bloque original tenía estructura if/else:
    #     if current_indicators and (_has_pipeline_data or len > 1):
    #         # MODIFY normal
    #     else:
    #         # keepalive (solo si timeout por ticket)
    #   Si current_indicators estaba vacío en un tick, se caía al else.
    #   En el else, _ka_indicators = _keepalive_last_indicators.get(ticket) or current_indicators.
    #   Si _keepalive_last_indicators[ticket] no existía Y current_indicators
    #   estaba vacío → _ka_indicators = {} → `if _ka_indicators:` era False
    #   → el MODIFY keepalive NO se enviaba. Resultado: 0 MODIFYs en sesión.
    #
    # FIX v25: dos bloques INDEPENDIENTES (no if/else):
    #   BLOQUE A — MODIFY normal: se ejecuta si hay datos del pipeline.
    #     Envía a todos los tickets activos. Actualiza _keepalive_last_sent.
    #   BLOQUE B — Keepalive: se ejecuta SIEMPRE después del bloque A.
    #     Para cada ticket sin MODIFY reciente (>= KEEPALIVE_INTERVAL_SECS),
    #     envía un MODIFY forzado con los mejores datos disponibles.
    #     El bloque A ya actualizó _keepalive_last_sent, por lo que si el
    #     bloque A envió para un ticket, el bloque B lo saltará (ts fresco).
    #   Además: fallback de indicators en el keepalive — si current_indicators
    #     y _keepalive_last_indicators están vacíos, intentar construir
    #     indicators mínimos directamente desde _pip_last para no enviar vacío.
    _has_pipeline_data = any(
        _pip(_pip_last, col) is not None for col in ('rsi', 'macd_hist', 'atr')
    )
    _now_ts = time.time()
    _s3_positions = s3_state.get_positions()   # dict ticket→S3Position
    _active_tickets = set(_s3_positions.keys())

    # ── FIX v30.8: fallback 3 niveles cuando s3_state vacío ──────────────────
    # Si s3_state está vacío pero MT5 reporta posiciones abiertas, el evento
    # POSITION_OPENED de S3 no llegó al s3_event_listener (problema clásico de
    # ZMQ PUB/SUB tras restart de S3: los primeros mensajes se pierden antes de
    # que el SUB complete el handshake).
    # Consecuencia: _active_tickets vacío → cero MODIFYs → ADAPTIVE_INDICATORS_STALE
    # en el 100% de las posiciones. Confirmado log 13/03/2026 (28 posiciones, 0 MODIFYs).
    #
    # Fallback en 3 niveles de prioridad descendente:
    #   1. s3_state (fuente canónica — eventos ZMQ de S3)
    #   2. _keepalive_last_sent (tickets a los que S2 envió OPEN en este arranque)
    #   3. _s1_open_tickets (lista directa de MT5 enviada por S1 en cada tick)
    #
    # El nivel 3 es el paracaídas: funciona incluso si S3 reinició y S2 perdió
    # todos los eventos, y sin necesidad de importar MetaTrader5 en S2.
    if not _active_tickets and effective_positions > 0:
        _active_tickets = set(_keepalive_last_sent.keys())  # nivel 2
    if not _active_tickets and _s1_open_tickets:
        _active_tickets = _s1_open_tickets                  # nivel 3
        # Poblar _keepalive_last_sent para que los próximos ciclos usen nivel 2
        for _t in _s1_open_tickets:
            _keepalive_last_sent.setdefault(_t, 0)          # 0 → keepalive inmediato
    # ─────────────────────────────────────────────────────────────────────────

    # Limpiar tickets cerrados del estado keepalive.
    # v48.0: limpiar si s3_state tiene datos, si no hay posiciones abiertas,
    # o si _active_tickets está poblado por el fallback — en ese caso el fallback
    # ya sabe qué tickets son válidos y podemos eliminar los que no estén.
    # Antes: solo limpiaba con _s3_positions o effective_positions==0, dejando
    # tickets cerrados en _keepalive_last_sent cuando s3_state=={} y MT5 > 0.
    # Resultado: MODIFY_FAILED NOT_TRACKED en tickets ya cerrados (ej. 319350164).
    #
    # v49.2 FIX-1: usar _s1_open_tickets como fuente de verdad adicional.
    # El listener ZMQ puede tardar hasta 60s en propagar el cierre desde S3 a
    # S3State. Durante ese intervalo _active_tickets sigue incluyendo el ticket
    # cerrado. _s1_open_tickets se actualiza desde MT5 cada tick (~1-2s), por lo
    # que un ticket ausente ahí ya no existe en el broker → eliminarlo de inmediato.
    # La unión de ambas fuentes cubre todos los casos: tickets nuevos que S3 ya
    # registró (en _active_tickets) y tickets vivos en MT5 pendientes de evento
    # S3 (en _s1_open_tickets). Un ticket fuera de AMBAS fuentes está cerrado.
    _valid_tickets = _active_tickets | _s1_open_tickets  # v49.2: doble fuente
    if _s3_positions or effective_positions == 0 or _active_tickets:
        for _t in list(_keepalive_last_sent.keys()):
            if _t not in _valid_tickets:
                _keepalive_last_sent.pop(_t, None)
                _keepalive_last_indicators.pop(_t, None)
                _keepalive_last_context.pop(_t, None)
                _keepalive_last_bar_time.pop(_t, None)   # v33.0

    # ── BLOQUE A: MODIFY normal (pipeline con datos frescos) ─────────────────
    # v49.0 FIX-1: _modified_in_this_tick registra los tickets ya enviados por
    # BLOQUE A en este tick. BLOQUE B lo consulta para no reenviar duplicados.
    # Causa del bug: en el primer tick de un ticket, _keepalive_last_bar_time[ticket]
    # no existe → BLOQUE B calcula _new_bar_trigger=True aunque BLOQUE A ya envió.
    _modified_in_this_tick: set = set()
    if _has_pipeline_data and _active_tickets:
        for ticket in _active_tickets:
            _modify_cmd = {
                'action':     'MODIFY',
                'ticket':     ticket,
                'indicators': current_indicators,
            }
            if current_context:
                _modify_cmd['context'] = current_context
            send_order(orders_push, _modify_cmd)
            _keepalive_last_sent[ticket]       = _now_ts
            _keepalive_last_indicators[ticket] = current_indicators
            if current_context:
                _keepalive_last_context[ticket] = current_context
            # v33.0: registrar bar_time del MODIFY normal para el trigger por vela.
            # Evita que el bloque B reenvíe un keepalive redundante en el mismo tick.
            _bar_time_raw = data.get('time')
            if _bar_time_raw is not None:
                _keepalive_last_bar_time[ticket] = int(_bar_time_raw)
            _modified_in_this_tick.add(ticket)  # v49.0 FIX-1

    # ── BLOQUE B: Keepalive (garantía de último recurso) ─────────────────────
    # Se ejecuta SIEMPRE, independientemente del bloque A.
    # Si el bloque A ya envió para un ticket, _keepalive_last_sent[ticket] = _now_ts
    # → la condición de timeout es False → el bloque B lo saltará sin enviar duplicado.
    # Si el bloque A NO envió (sin pipeline), el bloque B actúa como fallback.
    #
    # v33.0: trigger DOBLE — dispara si ha expirado el timeout de tiempo O si
    # la vela M1 es nueva respecto a la última registrada para el ticket.
    # El trigger por vela garantiza exactamente un MODIFY por vela cerrada,
    # que equivale a un MODIFY cada ~60s sin depender del intervalo real entre
    # ticks. Resuelve el caso donde los ticks llegan muy seguidos (mercado activo)
    # y el timeout de 60s no había expirado aunque ya había pasado una vela entera.
    _current_bar_time_raw = data.get('time')
    _current_bar_time_int = int(_current_bar_time_raw) if _current_bar_time_raw is not None else None

    for ticket in _active_tickets:
        # v49.0 FIX-1: saltar tickets ya procesados por BLOQUE A en este tick.
        # Evita el duplicado donde _new_bar_trigger=True porque _keepalive_last_bar_time
        # no tenía entrada para el ticket (primer tick) aunque BLOQUE A ya envió.
        if ticket in _modified_in_this_tick:
            continue
        _last_sent = _keepalive_last_sent.get(ticket, 0)
        _timeout_expired = (_now_ts - _last_sent) >= KEEPALIVE_INTERVAL_SECS

        # v33.0: nueva vela = bar_time cambió desde el último MODIFY para este ticket
        _new_bar_trigger = False
        if KEEPALIVE_ON_NEW_BAR and _current_bar_time_int is not None:
            _last_bar = _keepalive_last_bar_time.get(ticket)
            _new_bar_trigger = (_last_bar is None or _current_bar_time_int != _last_bar)

        if not _timeout_expired and not _new_bar_trigger:
            continue  # MODIFY reciente y misma vela → no enviar

        # FIX v25: fallback de indicators más robusto.
        # Prioridad: last_known > current (puede estar vacío) > pip_last directo.
        # Esto evita que el keepalive se cancele por _ka_indicators vacío, que era
        # la causa raíz del bug: con ambos vacíos, `if _ka_indicators:` era False.
        _ka_indicators = (
            _keepalive_last_indicators.get(ticket)
            or (current_indicators if current_indicators else None)
            or {k: v for k, v in {
                'atr':       _pip(_pip_last, 'atr'),
                'rsi':       _pip(_pip_last, 'rsi'),
                'macd_hist': _pip(_pip_last, 'macd_hist'),
                'volume_ma': _volume_sma.value,
            }.items() if v is not None}
        )
        if not _ka_indicators:
            # No hay absolutamente ningún dato de indicadores disponible.
            # Enviar de todas formas sin indicators para al menos actualizar
            # el contexto (spread, regime) y reiniciar el timer keepalive.
            _ka_indicators = {}

        _ka_cmd = {
            'action':    'MODIFY',
            'ticket':    ticket,
            'keepalive': True,
        }
        if _ka_indicators:
            _ka_cmd['indicators'] = _ka_indicators

        # Context del keepalive: spread fresco + state del último conocido.
        # El spread cambia tick a tick → siempre usar el más reciente disponible.
        # El regime cambia lentamente → usar el último conocido si no hay uno fresco.
        _ka_context = {}
        _last_ctx = _keepalive_last_context.get(ticket, {})
        if current_context.get('spread') is not None:
            _ka_context['spread'] = current_context['spread']
        elif _last_ctx.get('spread') is not None:
            _ka_context['spread'] = _last_ctx['spread']
        if current_context.get('regime'):
            _ka_context['regime'] = current_context['regime']
        elif _last_ctx.get('regime'):
            _ka_context['regime'] = _last_ctx['regime']
        if _ka_context:
            _ka_cmd['context'] = _ka_context

        send_order(orders_push, _ka_cmd)
        _keepalive_last_sent[ticket] = _now_ts
        # v33.0: registrar bar_time del keepalive para el trigger por vela.
        # Evita re-envío en el mismo bar_time si el bloque B se evaluara dos veces.
        if _current_bar_time_int is not None:
            _keepalive_last_bar_time[ticket] = _current_bar_time_int
        # No actualizamos _keepalive_last_indicators con datos vacíos:
        # preservamos el último dict completo para el próximo keepalive.

    # ── Posiciones: effective_positions calculado antes de decide_live (v27.0) ──
    # s3_n y effective_positions ya están calculados al inicio del tick (FIX-1).
    # Re-leemos s3_n aquí solo para el diagnóstico LOW_SCORE_WITH_OPEN_POS.
    s3_n = s3_state.n_positions()
    # No recalcular effective_positions: el valor ya fue usado en decide_live
    # y debe ser consistente durante todo el tick.

    if live_order:
        # v39.0: guard anti-duplicado — usa dict mutable (_open_guard) para
        # evitar el problema de scope: en Python, reasignar una variable simple
        # (x = ...) dentro de un while a nivel de módulo requiere `global`, pero
        # `global` no está permitido aquí porque el nombre tiene annotación de tipo
        # o porque el while está a nivel de módulo (no dentro de función). Mutar
        # una clave de un dict (d['k'] = ...) sí funciona sin global.

        score = live_order.get('score', 0)
        state = live_order.get('state', '?')
        side  = live_order['side']
        mt5_side = 'BUY' if side == 'long' else 'SELL'

        print(f"\t[SCORE] {score:.4f}  side={side}  regime={state}")

        # ── FILTROS DE SEGURIDAD PRE-ENVÍO ───────────────────────────────────
        block_reason = None

        # 0-pre. Anti-duplicado de apertura (v38.0).
        #   Si ya enviamos una orden en este mismo bar_time O hace menos de
        #   OPEN_GUARD_SECS, bloquear. Evita el race condition ZMQ donde S3
        #   tarda 100-400ms en publicar POSITION_OPENED y en ese intervalo S2
        #   procesa 1-2 ticks más con effective_positions todavía en el valor
        #   anterior, enviando una segunda orden idéntica.
        # 0-startup: gracia de arranque — bloquear OPENs en las primeras STARTUP_GRACE_BARS barras.
        # Deja que el buffer ZMQ de S3 se vacíe antes de enviar nuevas órdenes.
        _bar_time_for_startup = data.get('time')
        if _bar_time_for_startup is not None:
            _startup_bars_seen.add(int(_bar_time_for_startup))
        if len(_startup_bars_seen) <= STARTUP_GRACE_BARS:
            block_reason = (
                f'STARTUP_OPEN_BLOCKED('
                f'bars_seen={len(_startup_bars_seen)}, '
                f'grace={STARTUP_GRACE_BARS})'
            )

        _now_guard = time.time()
        _guard_bar = data.get('time')
        _same_bar  = (_guard_bar is not None and int(_guard_bar) == _open_guard['bar_time'])
        _too_soon  = (_now_guard - _open_guard['sent_ts']) < OPEN_GUARD_SECS
        if block_reason is None and (_same_bar or _too_soon):
            block_reason = (
                f'OPEN_GUARD(same_bar={_same_bar}, '
                f'elapsed={_now_guard - _open_guard["sent_ts"]:.1f}s < {OPEN_GUARD_SECS}s)'
            )

        # 0. Spread excesivo: no enviar si el spread supera el límite configurado.
        #    v20.0: primera línea de defensa — evita que la orden llegue a S3 y
        #    genere OPEN_REJECTED_SPREAD. El tick MT5 trae spread en puntos directamente.
        #    S3 tiene el mismo límite (max_spread_points=20 en perfil scalping) como
        #    segunda línea de defensa por si el spread sube entre el envío y la apertura.
        _current_spread = data.get('spread')
        if _current_spread is not None and int(_current_spread) > MAX_SPREAD_POINTS:
            block_reason = (
                f'SPREAD_TOO_HIGH(spread={int(_current_spread)}pts, max={MAX_SPREAD_POINTS}pts)'
            )

        # 1. No acumular contra la dirección con drawdown significativo.
        #    v6.0: usamos sides_open() de S3 para saber el lado REAL abierto,
        #    en lugar de inferirlo del último side enviado (que podía ya estar cerrado).
        if effective_positions > 0 and floating < -50 and block_reason is None:
            sides = s3_state.sides_open()
            if sides and mt5_side not in sides:
                # La señal actual va en dirección opuesta a lo que S3 tiene abierto
                block_reason = (
                    f'OPPOSITE_SIDE_IN_DRAWDOWN('
                    f'floating={floating:.2f}, open_sides={sides}, new={mt5_side})'
                )

        # 2. Score insuficiente con posición ya abierta: no acumular con señales débiles.
        #    v6.0: usamos effective_positions en lugar de n_positions.
        #    v32.0 FIX-2: umbral subido de 0.20 → 0.35 para segundas entradas.
        #    Análisis 13/03/2026: los 6 trades con score [0.20-0.25) generaron avg_R=-0.84
        #    mientras que con min_score>=0.40 para segundas entradas sum_R pasaba de
        #    -17.68 a +1.84 (excluía 8 trades con sum_R=-19.53). El umbral 0.35 es
        #    más conservador que 0.40 pero cubre la zona de más pérdidas estructurales
        #    (scores 0.21-0.25 en la racha de BUYs del 13/03/2026, 4 trades × ~-1R).
        if effective_positions > 0 and score < 0.35 and block_reason is None:
            block_reason = f'LOW_SCORE_WITH_OPEN_POS(score={score:.4f}, s3_n={s3_n}, threshold=0.35)'

        # 2b. Cooldown por racha de pérdidas consecutivas en el mismo side.
        #     v32.0 FIX-3: si los últimos LOSS_STREAK_MAX cierres del mismo side
        #     ocurrieron por VIRTUAL_SL dentro de LOSS_STREAK_WINDOW_SECS, bloquear
        #     ese side durante LOSS_STREAK_COOLDOWN_SECS.
        #     Origen: racha de 9 BUYs perdedores del 13/03/2026 entre 18:35–18:44h.
        #     El modelo generaba TRANSITION_UP/RANGE repetidamente mientras el precio
        #     bajaba. Ningún filtro de score lo detuvo porque algunos scores eran >0.40.
        #     El cooldown actúa independientemente del score: si el mercado está
        #     rechazando ese side de forma sistemática, esperamos antes de reintentar.
        if block_reason is None:
            _now_for_cooldown = time.time()
            _cooldown_until = _loss_streak_cooldown_until.get(mt5_side, 0.0)
            if _now_for_cooldown < _cooldown_until:
                _remaining = round(_cooldown_until - _now_for_cooldown)
                block_reason = (
                    f'LOSS_STREAK_COOLDOWN('
                    f'side={mt5_side}, '
                    f'remaining={_remaining}s, '
                    f'streak={LOSS_STREAK_MAX})'
                )
            else:
                # Comprobar si con este tick se completa una nueva racha
                _sl_closes = s3_state.last_sl_close_by_side.get(mt5_side, [])
                _window_start = _now_for_cooldown - LOSS_STREAK_WINDOW_SECS
                _recent = [ts for ts, _ in _sl_closes if ts >= _window_start]
                # v46.0: diagnóstico — loguear estado del streak en cada señal.
                # Permite verificar si el listener está alimentando last_sl_close_by_side.
                signal_logger.info(json.dumps({
                    'event': 'LOSS_STREAK_STATE',
                    'ts': _now_for_cooldown,
                    'side': mt5_side,
                    'streak_count': len(_sl_closes),
                    'recent_in_window': len(_recent),
                    'window_secs': LOSS_STREAK_WINDOW_SECS,
                    'streak_max': LOSS_STREAK_MAX,
                    'would_trigger': len(_recent) >= LOSS_STREAK_MAX,
                    'oldest_sl_ts': _sl_closes[0][0] if _sl_closes else None,
                    'newest_sl_ts': _sl_closes[-1][0] if _sl_closes else None,
                }))
                if len(_recent) >= LOSS_STREAK_MAX:
                    _loss_streak_cooldown_until[mt5_side] = _now_for_cooldown + LOSS_STREAK_COOLDOWN_SECS
                    block_reason = (
                        f'LOSS_STREAK_COOLDOWN('
                        f'side={mt5_side}, '
                        f'streak={len(_recent)}_in_{LOSS_STREAK_WINDOW_SECS}s, '
                        f'cooldown={LOSS_STREAK_COOLDOWN_SECS}s)'
                    )

        # 3. vSL demasiado ajustado: cierre casi garantizado en <10s.
        #    v29.0: create_order_request ya aplicó guardia MIN_VSL_ATR_RATIO, pero
        #    en casos de ATR casi-cero el vSL puede quedar en MIN_VSL_POINTS=20pts.
        #    Rechazar aquí (antes de enviar) evita que trades sin margen operativo
        #    consuman slippage y deterioren el PnL sin posibilidad de defensa.
        #    Confirmado 13/03/2026: 9 trades en <10s; todos con vSL ~ entry.
        if block_reason is None:
            _sl_pts_check = price_to_points(
                float(live_order.get('entry', 0) or 0),
                float(live_order.get('sl', 0) or 0),
            )
            if _sl_pts_check is not None and _sl_pts_check < MIN_VIABLE_TRADE_SL_PTS:
                block_reason = (
                    f'VSL_TOO_TIGHT(vsl_pts={_sl_pts_check}, '
                    f'min={MIN_VIABLE_TRADE_SL_PTS}pts)'
                )

        # 4. Señal contra-tendencia: bloqueo total en v42.0.
        #    v29.0: penalización de +0.15 al score mínimo — insuficiente con scores altos.
        #    v42.0: BLOQUEO TOTAL cuando COUNTER_TREND_BLOCK_TOTAL=True.
        #    Confirmado 16-17/03/2026: 12 señales SHORT en TREND_UP con scores 0.44-1.0
        #    mientras el oro subía +2300pts. Con penalización=0.15 y score=1.0 nunca bloquea.
        #    SELL en TREND_UP o BUY en TREND_DOWN son señales estructuralmente incorrectas
        #    para este sistema de scalping — el modelo tiene sesgo histórico pero el mercado
        #    no respeta esas señales en tendencia fuerte.
        if block_reason is None:
            _state_lower = str(state).lower()   # v43: modelo devuelve uppercase ('TREND_UP')
            _is_counter = (
                (_state_lower in COUNTER_TREND_REGIMES_LONG  and mt5_side == 'SELL')
                or (_state_lower in COUNTER_TREND_REGIMES_SHORT and mt5_side == 'BUY')
            )
            if _is_counter:
                if COUNTER_TREND_BLOCK_TOTAL:
                    block_reason = (
                        f'COUNTER_TREND_BLOCKED('
                        f'regime={state}, side={mt5_side}, score={score:.4f})'
                    )
                else:
                    # Comportamiento v29: penalización de score
                    _min_score_counter = MIN_SCORE_TO_TRADE + COUNTER_TREND_SCORE_PENALTY
                    if score < _min_score_counter:
                        block_reason = (
                            f'COUNTER_TREND_LOW_SCORE('
                            f'regime={state}, side={mt5_side}, '
                            f'score={score:.4f}, min={_min_score_counter:.4f})'
                        )

        # 4b. Filtro de sesión horaria: bloquear operativa nocturna (v42.0).
        #     Sesión asiática (23:00-07:00 UTC): volumen bajo, spreads altos,
        #     tendencias prolongadas en una dirección. El modelo no tiene suficiente
        #     representación de esas condiciones en su entrenamiento.
        #     Confirmado 16-17/03/2026: 8h nocturnas con WR 6.2%, 75% señales SHORT
        #     en mercado fundamentalmente alcista (+3745 pts).
        #     Horario permitido: SESSION_START_UTC (07:00) a SESSION_END_UTC (22:00) UTC.
        if block_reason is None:
            import datetime as _dt
            _utc_hour = _dt.datetime.utcnow().hour
            _in_session = SESSION_START_UTC <= _utc_hour < SESSION_END_UTC
            if not _in_session:
                block_reason = (
                    f'OUT_OF_SESSION('
                    f'utc_hour={_utc_hour}, '
                    f'allowed={SESSION_START_UTC}:00-{SESSION_END_UTC}:00 UTC)'
                )

        # 5. Señal rancia: el modelo no ha generado una nueva señal en las últimas
        #    MAX_SIGNAL_AGE_BARS barras. En M1, antigüedad=1 es normal (el modelo
        #    genera en la barra N, S2 procesa en N+1). Antigüedad >= 2 indica que
        #    el modelo no actualizó: la señal tiene entry/sl/tp del pasado y la
        #    geometría ya no corresponde al mercado actual.
        #    Confirmado 13/03/2026: señal de 14:06 reenviada hasta 14:12 (5 barras)
        #    con gaps de 234-779pts mientras el oro bajaba 800pts sin nueva señal.
        if block_reason is None:
            _signal_entry_time = live_order.get('entry_time')
            _current_bar_time  = data.get('time')
            if _signal_entry_time is not None and _current_bar_time is not None:
                try:
                    import pandas as _pd
                    # entry_time viene del modelo (BD), donde S2 almacena
                    # broker_unix - 7200s (UTC). Puede ser timezone-aware (UTC)
                    # o naive (se interpreta como UTC en la máquina del servidor).
                    _sig_ts = int(_pd.Timestamp(_signal_entry_time).timestamp())
                    # bar_time viene de data['time'] = raw broker unix (UTC+2).
                    # Para comparar en la misma referencia, ajustar a UTC: -7200s.
                    _bar_ts = int(_current_bar_time) - 7200
                    # Cada barra M1 = 60s. Antigüedad en barras (redondeado).
                    _age_bars = max(0, round((_bar_ts - _sig_ts) / 60))
                    if _age_bars > MAX_SIGNAL_AGE_BARS:
                        block_reason = (
                            f'SIGNAL_TOO_OLD('
                            f'entry_time={_signal_entry_time}, '
                            f'bar_time={_current_bar_time}, '
                            f'age={_age_bars}bars, '
                            f'max={MAX_SIGNAL_AGE_BARS}bars)'
                        )
                except Exception:
                    pass  # si no se puede parsear, no bloquear

        # 6. Gap excesivo entre model_entry y precio actual (umbral dinámico 1×ATR).
        #    v30.0: filtro original con MAX_ENTRY_GAP_PTS fijo.
        #    v30.2: umbral dinámico = max(MAX_ENTRY_GAP_PTS, atr_pts) donde
        #    atr_pts = ATR de la señal en puntos. Semántica: "si el precio ya
        #    se movió más de 1 ATR desde que el modelo decidió, la señal caducó".
        #    Con FIX-4, gap normal < 5pts → este filtro solo dispara en eventos
        #    extremos (news spike, gap de sesión, flash crash).
        #    Referencia: ask para BUY, bid para SELL. Fallback a close si S1 antiguo.
        if block_reason is None:
            _model_entry  = float(live_order.get('entry', 0) or 0)
            _mt5_side_gap = live_order.get('side', '')
            _bid_s1 = data.get('bid')
            _ask_s1 = data.get('ask')
            if _mt5_side_gap == 'long' and _ask_s1:
                _ref_price = float(_ask_s1)
            elif _mt5_side_gap == 'short' and _bid_s1:
                _ref_price = float(_bid_s1)
            else:
                _ref_price = float(data.get('close', 0) or 0)
            if _model_entry > 0 and _ref_price > 0:
                _entry_gap = abs(_ref_price - _model_entry) / 0.01
                # Umbral dinámico: 1×ATR de la señal, mínimo MAX_ENTRY_GAP_PTS
                _atr_pts_signal = float(live_order.get('atr', 0) or 0) / 0.01
                _gap_threshold = max(MAX_ENTRY_GAP_PTS,
                                     int(_atr_pts_signal) if _atr_pts_signal > 0 else 0)
                if _entry_gap > _gap_threshold:
                    _ref_label = 'ask' if _mt5_side_gap == 'long' else ('bid' if _mt5_side_gap == 'short' else 'close')
                    block_reason = (
                        f'ENTRY_GAP_TOO_LARGE('
                        f'model_entry={_model_entry:.2f}, '
                        f'{_ref_label}={_ref_price:.2f}, '
                        f'gap={_entry_gap:.0f}pts, '
                        f'threshold={_gap_threshold}pts, '
                        f'atr={_atr_pts_signal:.0f}pts)'
                    )
        if block_reason:
            # FIX v10.1: añadir model_diag a SIGNAL_BLOCKED para ver engine_debug
            # en el log (antes aparecía reason=? porque model_diag era None aquí).
            _blocked_diag = getattr(engine, '_last_diag', None)
            log_signal(signal_logger, event='SIGNAL_BLOCKED',
                       bar_data=bar_snapshot, live_order=live_order,
                       block_reason=block_reason,
                       model_diag=_blocked_diag,
                       n_positions=effective_positions, equity=equity, balance=balance)
            print(f"\t[BLOCKED] {block_reason}")
        else:
            # ── FIX v23.0 (BUG-1): indicadores completos en el OPEN ──────────
            # Fusionar live_order + _pip_last para garantizar que la posición
            # abre con ind_last_update_ts inicializado y los campos clave presentes.
            # Prioridad: live_order (más específico) > _pip_last (siempre disponible).
            # v34.0: _first_not_none en lugar de `or` — mismo BUG-1 que en current_indicators.
            # Si live_order.macd_hist=0.0 (cruce de cero en el bar de entrada), `or`
            # lo descartaba y la posición abría con macd_hist=None en ind_last_update_ts.
            _open_indicators = {k: v for k, v in {
                'atr':            _first_not_none(live_order.get('atr_at_entry'), _pip(_pip_last, 'atr')),
                'rsi':            _first_not_none(live_order.get('rsi'),           _pip(_pip_last, 'rsi')),
                'macd_hist':      _first_not_none(live_order.get('macd_hist'),     _pip(_pip_last, 'macd_hist')),
                'macd_hist_prev': _first_not_none(live_order.get('macd_hist_prev'), _pip(_pip_prev, 'macd_hist')),
                'volume':         live_order.get('volume'),
                'volume_ma':      _volume_sma.value,
                'proba_long':     live_order.get('proba_long'),
                'proba_short':    live_order.get('proba_short'),
            }.items() if v is not None}
            if _open_indicators:
                live_order['_indicators_override'] = _open_indicators

            # ── v24.0: contexto de mercado en el OPEN ────────────────────────
            # S3 recibe el régimen y spread desde el primer momento, sin esperar
            # al primer MODIFY. Permite correlación régimen→PnL en el log de cierre.
            if current_context:
                live_order['_context_override'] = current_context

            # ── FIX v23.0 (BUG-2) + v22.0: inyectar ATR del pipeline ────────
            # v23: umbral sube de <= 0 a <= MIN_VSL_POINTS*point para descartar
            # también ATRs casi-cero que producen atr_pts=0 y _min_vsl=20.
            _pip_atr = _pip(_pip_last, 'atr')
            try:
                _pip_atr_f = float(_pip_atr) if _pip_atr is not None else 0.0
            except Exception:
                _pip_atr_f = 0.0

            if _pip_atr_f > 0:
                live_order['_atr_fallback'] = _pip_atr_f

            # v30.0: pasar bid/ask del tick actual para anclar geometría al
            # precio real de fill en lugar del model_entry.
            _tick_bid = data.get('bid')
            _tick_ask = data.get('ask')
            request = create_order_request(live_order, comment='production',
                                           bb_upper=_bb_upper, bb_lower=_bb_lower,
                                           bid=_tick_bid, ask=_tick_ask)
            # v47.0: añadir sent_ts al request justo antes de enviarlo.
            # S3 usa este timestamp para rechazar mensajes del buffer ZMQ
            # de sesiones anteriores (OPEN_REJECTED_STALE si age > 30s).
            request['sent_ts'] = time.time()
            send_order(orders_push, request)
            # v38/v39: actualizar guardia anti-duplicado via dict mutable
            _open_guard['bar_time'] = int(data.get('time') or 0)
            _open_guard['sent_ts']  = time.time()
            # v39.1: segunda capa — incrementar effective_positions localmente
            # para que los filtros 1-2 también bloqueen el tick siguiente si el
            # guard fallara. effective_positions se recalcula desde s3_state al
            # inicio del siguiente tick, por lo que este incremento solo dura
            # los ~200ms hasta el próximo recv().
            effective_positions += 1
            # v49.0: el MODIFY inmediato post-OPEN se gestiona ahora via callback
            # _on_position_opened_cb asignado a s3_state. Cuando el listener recibe
            # POSITION_OPENED de S3, on_event invoca el callback con el ticket real
            # (asignado por el broker post-fill) y lo registra en _keepalive_last_sent
            # con ts=0 → MODIFY en el siguiente tick via BLOQUE B.
            # El bloque _new_ticket de v48 (request.get('ticket')) se elimina porque
            # request={} siempre — el ticket no está disponible en el momento del OPEN.
            log_signal(signal_logger, event='SIGNAL_SENT',
                       bar_data=bar_snapshot, live_order=live_order,
                       request=request, n_positions=effective_positions, equity=equity, balance=balance,
                       indicators=current_indicators)

    else:
        # Sin señal: leer el diagnóstico completo que decide_live dejó en _last_diag
        model_diag = getattr(engine, '_last_diag', None)
        log_signal(signal_logger, event='NO_SIGNAL',
                   bar_data=bar_snapshot, n_positions=effective_positions,
                   equity=equity, balance=balance, model_diag=model_diag)

    print('=' * 220 + '\n')

print('')