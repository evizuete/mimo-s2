#!/usr/bin/env python3
"""
s3_service.py — S3 Trading Risk Management Service
Versión: 30.0

Cambios v28.5 (sobre v28.4) — ENRIQUECIMIENTO DE CIERRES + FIXES 13/04/2026:

  FIX-1 — _close_position_full: profit_pts, hold_seconds, volume en TODOS los cierres:
    Antes solo TIME_FORCE_CLOSE los incluía via extra_payload. Ahora se calculan
    automáticamente en _close_position_full para VIRTUAL_SL, VIRTUAL_TP, ADAPTIVE_*,
    MAX_HOLD_TIME y cualquier razón futura. extra_payload sigue funcionando y
    sobreescribe los valores automáticos si son más precisos.

  FIX-2 — POSITION_CLOSED_EXTERNAL enriquecido con datos del trade:
    Cuando tr_closed está disponible (posición trackeada por S3), el evento incluye
    side, open_price, volume, hold_seconds, virtual_sl, virtual_tp. Antes era solo
    ticket+symbol+reason, lo que impedía el análisis post-sesión de cierres externos.

  FIX-3 — be_trigger_points scalping 40→100, trail_points 40→60:
    Con be_trigger=40 (~13% del vSL de ~313pts) el BE se armaba demasiado pronto
    cortando winners a avg +0.22R mientras los losers llegaban a -1.05R (EV=-0.25R).
    Análisis log 13/04/2026: 30 cierres por vSL, 0 VIRTUAL_TP_TRIGGERED en 34 trades.

  FIX-4 — TRAINING_LABEL_RR definido a nivel módulo:
    Evita NameError en POSITION_RECOVERED cuando el broker no tiene TP registrado.
    Constantes: TRAINING_SL_BARRIER_R=1.5, TRAINING_TP_BARRIER_R=2.5, TRAINING_LABEL_RR=1.6667.

  FIX-5 — ADAPTIVE_SL_MIN_HOLD_SKIP: respetar min_hold_seconds en AdaptiveSL:
    Tickets 334164903 (-23pts) y 334242422 (-16pts) cerraban a 0-1s porque
    ADAPTIVE_EXPANSION_SL_TRIGGERED no comprobaba min_hold_seconds. Ahora emite
    ADAPTIVE_SL_MIN_HOLD_SKIP y pospone el cierre hasta cumplir el mínimo.

  FIX-6 — Reintento automático de cierre por slippage extremo (MAX_SLIPPAGE_RETRY_PTS=400):
    Si slippage_pts > 400 tras un cierre por vSL, S3 espera 300ms, verifica si queda
    volumen residual en MT5 y lanza un segundo close_position_market con deviation=50.
    Emite SLIPPAGE_RETRY_CLOSE (reintento ejecutado), SLIPPAGE_RETRY_SKIP (sin residual)
    o SLIPPAGE_RETRY_ERROR (fallo inesperado).
    Caso motivador: ticket 334460414 (13/04/2026), slippage=514pts, loss=-0.26R.

Cambios v28.4 (sobre v28.3) — DEBUG_POSITION_OPENED_PUBLISHED: diagnóstico ZMQ handshake:

  Contexto: análisis log 18/03/2026. n_positions_tracked=0 permanente en S2 pese a
  que el listener recibe cientos de eventos de S3. S2 v49 añade debug log en on_event
  para saber si POSITION_OPENED llega al listener. S3 v28.4 añade el debug
  complementario: confirmar que S3 publica el evento en absoluto.

  DEBUG — DEBUG_POSITION_OPENED_PUBLISHED emitido justo después de _send(POSITION_OPENED):
    Si aparece en s3_events pero NO aparece DEBUG_POSITION_OPENED_REGISTERED en S2
    → S3 publica el evento pero S2 no lo recibe → race condition ZMQ handshake.
      Solución: añadir time.sleep(0.5) tras events_sub.connect() en S2 para dar
      tiempo al SUB a completar el handshake antes de que S3 emita los primeros eventos.
    Si aparece en s3_events Y aparece en S2 → el evento llega y se registra, pero
      algo limpia s3_state.positions después → bug en la lógica de limpieza de S2.
    Si NO aparece en s3_events → _handle_open falla antes de llegar al _send →
      revisar excepciones en OPEN_ERROR o POST_FILL_GAP_ABORT.
    Este evento se puede eliminar una vez confirmada la causa raíz.


Cambios v28.3 (sobre v28.2) — OPEN_REJECTED_STALE: invalidar buffer ZMQ antiguo:

  BUG: el buffer ZMQ PUSH de S3 acumula ORDER_REQUESTs de sesiones anteriores
  indefinidamente. Al arrancar S3, procesa esos mensajes viejos abriendo
  posiciones con el config del momento en que se generaron (vTP, RR antiguos).
  Confirmado 18/03/2026: 35 mensajes en buffer con RR=2.5 (config pre-v44)
  procesados durante 35 min intercalados con señales reales de v45 (RR=1.0).
  El DEDUP guard de 5s no puede rechazarlos porque son barras distintas.
  Fix: S2 incluye sent_ts=time.time() en cada ORDER_REQUEST OPEN.
  S3 comprueba la antigüedad: si now()-sent_ts > ORDER_STALE_SECS (30s),
  rechaza con OPEN_REJECTED_STALE. Los mensajes del buffer son siempre más
  viejos que cualquier señal nueva → se invalidan automáticamente al arrancar.
  Si sent_ts no viene (versión antigua de S2), el check se salta (retrocompat).

Cambios v28.2 (sobre v28.1) — BUGFIX UnboundLocalError en _effective_trailing_params:

  BUG: meta['runner_tight'] = True se ejecutaba ANTES de que meta fuera definido.
  La variable meta se construía al FINAL del método, pero el bloque del tight trail
  la referenciaba en medio → UnboundLocalError: cannot access local variable 'meta'.
  Impacto: 5526 MONITOR_ERROR durante toda la sesión 18/03/2026. Cada ciclo del
  monitor (0.2s) fallaba → trailing no se actualizaba → posiciones gestionadas a ciegas.
  Fix: mover la construcción de meta ANTES del bloque tight trail. También actualizar
  meta['trail_distance_pts'] y meta['trail_step_pts'] tras aplicar el override.

Cambios v28.1 (sobre v28.0) — RUNNER_VTP_DISABLED: vTP desactivado tras partial:

  FIX: tras el partial close exitoso, el vTP original (ej: 200pts con RR=1.0)
  cerraba el runner a solo 20pts del trigger (partial al 90% → runner a 20pts del TP).
  Con tight trail activo, el runner podría capturar un movimiento mucho mayor
  si el precio sigue tendencia — pero el vTP lo cerraba prematuramente.
  Fix: en _check_partial_close, si el cierre parcial es exitoso y queda volumen,
  se pone tr.virtual_tp = 0.0 → el AdaptiveSL no lo sincroniza (condición > 0)
  → el runner corre guiado únicamente por el tight trail y max_hold_seconds.
  Evento nuevo: RUNNER_VTP_DISABLED con old_virtual_tp, remaining_volume, tight_trail_pts.

Cambios v28.0 (sobre v27.0) — runner_tight_trail_pts: trailing ajustado post-partial:

  CONTEXTO:
    Con partial_close activo (50% al TP), el runner restante usa el trailing
    normal (~160pts con vSL=200). Una reversión de 160pts desde el máximo
    favorable lo cierra — movimiento trivial en M1. Resultado sesión 17/03:
    86 trades cerraron el partial en TP pero el runner perdió, generando
    pérdida neta a pesar de haber llegado al TP.

  FIX: runner_tight_trail_pts en PartialCloseCfg.
    Si runner_tight_trail_pts > 0 y partial_done=True, _effective_trailing_params
    sustituye el trail_distance normal por este valor pequeño (ej: 20pts).
    Efecto: en cuanto se ejecuta el partial, el SL del runner se ajusta a
    precio_actual - 20pts (SELL) o precio_actual + 20pts (BUY).
    Si el precio sigue moviéndose a favor: el tight trail lo acompaña y
    el runner gana más. Si el precio revierte: cierra con pérdida de solo
    20pts en el runner (vs los ~160pts actuales).
    El parámetro viene de S2 en risk.partial_close.runner_tight_trail_pts.
    Valor 0 = comportamiento anterior (trailing normal, sin cambios).

  CAMBIOS:
    1. PartialCloseCfg: campo runner_tight_trail_pts: int = 0
    2. _parse_risk_cfg: lee runner_tight_trail_pts del dict partial_close
    3. _effective_trailing_params: si tight > 0 y partial_done, override
       trail_distance_pts = tight, trail_step_pts = tight // 4
       Loguea runner_tight=True en trail_meta para auditoría.

Cambios v27.0 (sobre v26.0) — FIX duplicados: MAX_POSITIONS guard en _handle_open:

  BUG-1 (CRÍTICO) — S3 abría 2 posiciones por cada ORDER_REQUEST de S2:
    Causa: el buffer ZMQ PUSH de S2 puede acumular señales de barras anteriores
    que S3 no ha procesado todavía (S3 ocupado en _monitor_loop o _executor_loop
    tardó un ciclo extra). En el tick siguiente S3 procesa AMBAS — la señal antigua
    y la nueva — llamando _handle_open dos veces y abriendo 2 posiciones idénticas.
    Confirmado sesión 16-17/03/2026:
      - S2 envió 76 señales → S3 abrió 161 posiciones (ratio 2.1x)
      - 59 pares duplicados con Δ=39-42ms, mismo entry y side
      - 152 SL closes, 10 TP closes → WR 6.2% → pérdidas masivas
    El fix de S2 v39 (OPEN_GUARD) previene que S2 envíe dos mensajes en el mismo
    bar_time, pero no puede prevenir el retraso de procesamiento en S3.
    _handle_open no tenía ningún guard contra el número de posiciones abiertas.
    Fix: al inicio de _handle_open, ANTES de cualquier procesamiento, consultar
    MT5 via positions_get() para contar las posiciones reales abiertas con el mismo
    magic. Si ya hay max_positions o más, rechazar con OPEN_REJECTED_MAX_POSITIONS
    sin tocar el broker. max_positions viene del risk_cfg del perfil (scalping=2).
    Fallback a len(_tracked) si positions_get() falla. Techo absoluto hard limit=4.

Cambios v26.0 (sobre v25.0) — FIX ruta de logs:

  FIX-1 — Logs escritos fuera del directorio de la aplicación:
    Causa: _resolve_log_path usaba Path('../logs') relativo al CWD en el momento
    de ejecución del proceso. Si S3 se lanzaba desde un directorio distinto al
    del script (p.ej. con un script de arranque que cambia el CWD), el fichero
    s3_events_YYYYMMDD.jsonl se creaba en una ubicación inesperada.
    Fix: nueva constante de módulo _S3_DIR = Path(__file__).parent.
    _resolve_log_path usa _S3_DIR / 'logs', siempre relativo al directorio
    donde vive el propio script s3_service.py, independientemente del CWD.
    El parámetro log_file del constructor no se ve afectado (permite override
    explícito de la ruta si se necesita).

Cambios v25.0 (sobre v24.0) — FIX post-análisis log 16/03/2026:

  FIX-1 — ADAPTIVE_TP_EXTENSION_OVERRIDDEN emitido 12935 veces (1 por ciclo 0.2s):
    Causa: el evento se emitía dentro del bloque
      `if _override_extension != manager.config.extension_enabled`
    El bloque `finally` de _run_adaptive_tp restaura manager.config.extension_enabled
    al valor original al final de cada llamada. En el siguiente ciclo del monitor
    (0.2s después), el manager vuelve a tener extension_enabled=True, la condición
    vuelve a ser True, y el evento se re-emite. Con trades en estado TRANSITION_*
    de duración típica 60-120s, esto generaba 300-600 eventos por trade.
    Confirmado log 16/03/2026: 12935 ADAPTIVE_TP_EXTENSION_OVERRIDDEN para 27 tickets
    en 4h, uno cada 0.209s (exactamente el intervalo del monitor_loop).
    Fix: nuevo campo TrackedPos._tp_override_last_state (Optional[str], default None).
    El evento solo se emite cuando ind_state cambia respecto al último estado registrado.
    Permite auditar cuándo el estado entra o sale de una zona de override, sin el ruido
    de las 300 emisiones intermedias.

Cambios v24.0 (sobre v23.0) — BUGFIXES post-revisión código 16/03/2026:

  FIX-1 — AdaptiveSLManager.on_bar() llamado con virtual_sl desincronizado:
    Causa: _run_adaptive_sl sincronizaba manager.virtual_sl = tr.virtual_sl_price
    SOLO cuando el propio manager devolvía un update_sl, no al inicio de cada ciclo.
    Si el trailing, BE o profit_locks habían movido tr.virtual_sl_price en ciclos
    anteriores, on_bar() evaluaba sl_touched, expansión y compresión contra el
    nivel antiguo. Podía expandir desde un SL ya superado por el trailing, o
    comprimir desde un nivel incorrecto.
    Fix: llamar manager.sync_sl(tr.virtual_sl_price) ANTES de on_bar(), igual
    que ya se hace con virtual_tp. Se usa sync_sl() en lugar de asignación
    directa para consistencia con el API del AdaptiveTPManager.

  FIX-2 — mt5_bridge: comment=reason sin truncar en close_position_market/partial:
    Causa: MT5 acepta máximo 31 chars en el campo comment. El bridge pasaba
    el parámetro reason directamente sin ningún límite. Con strings largos
    como "ADAPTIVE_EXPANSION_TIMEOUT" el broker devuelve retcode -2.
    Fix: comment="" en close_position_market y close_position_partial.
    La razón de cierre ya está en el evento ZMQ de S3 — no necesita estar
    también en el comment de MT5.

  FIX-3 — _bars_open en AdaptiveSL/TP manager conta ticks del monitor (5Hz),
           no velas M1:
    Causa: on_bar() se incrementaba en cada llamada del monitor_loop (cada 0.2s).
    Con compression_min_hold_bars=3, la guardia de "mínimo 3 velas antes de
    comprimir" expiraba en 0.6s en lugar de 3 minutos. expansion_max_bars=3
    en el AdaptiveSL expirada en 0.6s en lugar de dar 3 velas reales de rebote.
    Fix: ambos managers detectan nueva barra por cambio de minuto en el timestamp
    (int(ts // 60) != last_bar_minute). _bars_open y compression_cooldown solo
    se incrementan al cambiar de vela. expansion_max_seconds sigue activo como
    red de seguridad independiente del contador de barras.

  FIX-4 — BE_SKIPPED emitido en cada ciclo del monitor (>1000 eventos/sesión):
    Causa: BE_SKIPPED se emitía en cada ciclo del monitor mientras profit >=
    be_trigger_points pero el trailing ya había mejorado el vSL más allá del
    nivel BE. Con 5Hz y trades de 1-2 min, esto generaba 17 eventos por trade
    de media (1052 en la sesión del 16/03/2026). El ruido enterraba eventos útiles.
    Fix: nuevo flag TrackedPos.be_skip_warned (bool, default False). BE_SKIPPED
    solo se emite la primera vez que se detecta la condición por posición.

  FIX-5 — sync_sl() ausente en AdaptiveSLManager (asimetría de API con TP manager):
    Causa: AdaptiveTPManager expone sync_sl() como método público. AdaptiveSL
    no lo tenía; S3 accedía a manager.virtual_sl directamente. Riesgo de
    AttributeError si se intercambian referencias en una refactorización.
    Fix: añadido sync_sl(new_sl) a AdaptiveSLManager con la misma semántica
    que el TP manager. S3 ahora usa sync_sl() en ambos managers de forma uniforme.

Cambios v22.0 (sobre v21.2) — BUGFIX post-análisis log 13/03/2026:

  FIX-1 — Validación post-fill: cierre inmediato si gap entry→mercado > 2×vSL:
    Causa: el filtro ENTRY_GAP_TOO_LARGE de S2 compara model_entry vs bid/ask
    en el momento de ENVIAR la orden. Si el precio se mueve bruscamente durante
    la latencia ZMQ→MT5 (fill real), S3 recibe un POSITION_OPENED con entry
    reportado válido pero el precio de mercado ya está muy alejado. La geometría
    completa del trade (vSL, hardSL) queda incoherente con la realidad desde
    el primer milisegundo.
    Confirmado 13/03/2026: ticket 317706824 SELL entry=5013.99, precio real
    al primer ciclo del monitor=5045.69 → gap=3170pt=14.2×vSL. ADAPTIVE_HARD_SL
    se activó instantáneamente → pérdida de -14.22R en un solo trade.
    Fix: en _handle_open, justo después de capturar entry_price del fill result,
    leer tick_bid_ask(symbol) y calcular gap = |current_price - entry_price| / point.
    Si gap > POST_FILL_GAP_MAX_MULT (2.0) × virtual_sl_points → cerrar
    inmediatamente con bridge.close_position_market() y emitir POST_FILL_GAP_ABORT.
    La posición NO se añade a _tracked. En condiciones normales gap < 5pt y el
    filtro nunca dispara. Solo actúa en eventos extremos (news spike, flash crash,
    latencia de ejecución anómala).
    Nueva constante: self.POST_FILL_GAP_MAX_MULT = 2.0 (configurable en __init__).

Cambios v20.1 (sobre v20.0) — BUGFIX OPEN GRACE PERIOD:

  FIX-4 — Falso POSITION_CLOSED_EXTERNAL a los ~155ms del OPEN:
    Causa: MT5 tarda entre 100-500ms en registrar una posición nueva en su API
    interna (positions_get()). Durante ese intervalos, el ticket ya está en
    _known_open pero no aparece en open_tickets del monitor. La diferencia
    "closed = _known_open - open_tickets" incluye el ticket → falso
    DETECTED_CLOSED → S3 elimina el ticket de _tracked y deja la posición
    sin supervisión (trailing, BE, vSL, TIME_FORCE_CLOSE).
    Confirmado 13/03/2026: ticket 317234422 BUY @ 5082.22 marcado cerrado a
    los 155ms; abierto 3h+ sin gestión de S3. Llegó a +140pts sin trailing.
    Fix: grace period por ticket (OPEN_GRACE_SECS=5.0s). Si opened_ts es más
    reciente que el umbral, se emite OPEN_GRACE_SKIP y se salta el ticket en
    ese ciclo. 5s = 10x el peor caso observado; no interfiere con cierres
    manuales reales (>10s en ejecutarse).

Cambios v20.0 (sobre v19.0) — BUGFIXES post-análisis log 13/03/2026:

  FIX-1 — virtual_sl_points no se actualizaba en risk_cfg tras WRONG_SIDE_SL_CORRECTED:
    Causa: _validate_and_correct_tp_sl() puede modificar tr.virtual_sl_price y
    tr.virtual_sl_points (por WRONG_SIDE_SL_CORRECTED o VSL_EXPANDED_MIN_ATR).
    Sin embargo, el risk_cfg ya había sido construido antes con los valores originales
    de S2, de modo que be_trigger_points, trail_points y trail_step_points quedaban
    desincronizados con la geometría real del trade.
    Ejemplo sesión 13/03/2026:
      ticket 317115024: vSL corregido de 259pts → 7pts por WRONG_SIDE.
      risk_cfg tenía be_trigger_points=130, trail_points=259 (calculados sobre 259pts).
      Con vSL real=7pts el AdaptiveSL entraba en expansión desde el primer tick
      y cerraba por EXPANSION_TIMEOUT en ~0.8s.
    Fix: tras _validate_and_correct_tp_sl, si virtual_sl_points cambió, recalcular
    los campos de risk_cfg proporcionales (be_trigger_points, trail_points,
    trail_step_points) manteniendo la misma ratio que usó S2. Emite nuevo evento
    RISK_CFG_RECALCULATED_AFTER_VSL_CORRECTION para auditoría.
    Además: POSITION_OPENED ahora emite los valores POST-corrección de
    virtual_sl_price/virtual_sl_points (en v19 emitía los valores de S2 pre-corrección
    aunque tr ya los tuviera actualizados).

  FIX-2 — VSL_EXPANDED_MIN_ATR nunca disparaba para posiciones SELL:
    Causa: la guardia de vSL mínimo usaba _is_improvement(tr, current_sl, new_sl)
    para decidir si aplicar la expansión. Para SELL, _is_improvement retorna True
    cuando new_sl < current_sl (SL más bajo = más protector del profit en una SELL).
    Pero la guardia de mínimo necesita ampliar el vSL hacia afuera del entry (más alto
    para SELL), lo que produce new_sl > current_sl → _is_improvement = False → no aplica.
    Confirmado en log 13/03/2026: 0 eventos VSL_EXPANDED_MIN_ATR pese a 14 tickets con
    WRONG_SIDE_SL_CORRECTED a distancias de 7-73pts (fallback=100pts debería disparar).
    Fix: reemplazar _is_improvement por comparación directa de distancia al entry:
    si abs(new_vsl - entry) > abs(current_vsl - entry) → expandir. Correcto para
    BUY y SELL independientemente.

  FIX-3 — S3 ignoraba emergency_sl_price de S2 y recalculaba desde bid/ask:
    Causa: _handle_open calculaba emergency_sl = bid/ask ± emergency_points*point
    ignorando el campo metadata['emergency_sl_price'] enviado por S2. Esto deshacía
    el fix de S2 v28 (HARD_SL_MARGIN_PTS=50): S3 enviaba al broker un hard SL propio
    que podía quedar entre entry y vSL por la misma geometría de v27.
    Confirmado: en sesión 13/03/2026 el 29% de trades tuvieron hard SL entre entry
    y vSL; 7 cerraron por ADAPTIVE_HARD_SL_TRIGGERED con hard SL incorrecto.
    Fix: si metadata['emergency_sl_price'] > 0 (S2 v28+), usarlo directamente.
    Si no viene (S2 < v28 o fallback), continuar con el cálculo desde bid/ask.

Cambios v19.0 (sobre v18.0) — BUGFIXES TIME_FORCE_CLOSE:

  FIX-1 — spread_inhibit_since se reseteaba en cada ciclo con spread normal:
    Causa: el bloque `else` del spread_inhibit (spread OK) ejecutaba
    `tr.spread_inhibit_since = 0.0` incondicionalmente. En XAUUSD, el spread
    fluctúa continuamente: si durante la inhibición el spread bajaba aunque
    fuera un ciclo (200ms), el contador se reseteaba y los 30s volvían a
    empezar. Con picos alternantes de spread (sube → baja → sube) el cierre
    podía postergarse indefinidamente mientras el profit se evaporaba.
    Fix: el `else` ya no resetea spread_inhibit_since una vez iniciado.
    El contador solo se inicia cuando spread_inhibit_since == 0.0 (primera
    detección) y ya no puede ser reseteado hasta que el ticket se cierra
    (y desaparece de _tracked). Si el spread normaliza antes de los 30s,
    el cierre se ejecuta igualmente en el siguiente ciclo — comportamiento
    correcto y más conservador.

  FIX-3 — Expansión anti-sweep (AdaptiveSL) estructuralmente bloqueada:
    Causa: hard_sl_margin_pts=50 exigía un mínimo de 60 pts de margen entre
    vSL y hard SL del broker para poder activar la expansión anti-sweep.
    S2 coloca el hard SL muy cerca del vSL — margen real de 0 a 42 pts en
    todos los tickets del log 12/03/2026 (mediana ~15 pts). Resultado: el
    100% de los cierres con ADAPTIVE_SL_CLOSE registraron
    expansion_blocked=hard_sl_margin_insuficiente. La función anti-sweep
    nunca pudo activarse en toda la sesión, pese a que en 4 de 5 casos
    había señales de reversión (reversal_count: 1-2).
    Fix: hard_sl_margin_pts 50 → 20. El threshold mínimo baja de 60 a 30 pts.
    Con los datos del 12/03: 4 de 5 tickets bloqueados habrían podido expandir.
    Colchón residual de 2.0 USD es suficiente para XAUUSD M1 scalping.
    ACCIÓN PARALELA EN S2: aumentar separación vSL/hard SL al abrir (≥50 pts).

  FIX-2 — TIME_FORCE_CLOSE no disparaba si el profit retrocedía después de
           haber tenido beneficio significativo:
    Causa: la condición `_profit_pts_now >= 0.3R` solo evaluaba el profit
    actual en ese tick. Si el precio devolvía parte de la ganancia (sin llegar
    al vSL), el profit bajaba por debajo del umbral y el TIME_FORCE_CLOSE
    nunca disparaba — la orden permanecía abierta indefinidamente. Escenario
    real observado 12/03/2026: SELL XAUUSD con beneficio máximo de +297pts
    que retrocedió; el TIME_FORCE_CLOSE nunca cerró.
    Fix: evaluar también el profit máximo histórico (peak) calculado desde
    `tr.max_favorable_price` (ya disponible en TrackedPos). Si el peak superó
    2× el umbral mínimo (2 × 0.3R = 0.6R), el TIME_FORCE_CLOSE se activa
    aunque el profit actual sea menor. El evento de cierre incluye ahora
    `peak_profit_points` y `triggered_by_peak` para trazabilidad en el log.
    El multiplicador 2× evita falsos positivos en trades que simplemente
    oscilan alrededor del umbral sin haber tenido beneficio real.

Cambios v18.0 (sobre v17.0) — CONTEXTO DE MERCADO EN MODIFY/KEEPALIVE:

  MEJORA-1 — TrackedPos: tres nuevos campos de contexto de mercado:
    - ind_regime  (Optional[str]):   último régimen del modelo recibido via MODIFY.
    - ind_score   (Optional[float]): score del modelo en el último tick con señal.
    - ind_spread  (Optional[int]):   spread en puntos del último MODIFY recibido.

    Procesados por _update_tracked_context(), análogo a _update_tracked_indicators().
    Llamado desde _handle_open (con context del OPEN si lo incluye S2) y desde
    _handle_modify (con context del MODIFY). Se mantiene separado de indicators
    por claridad semántica: indicators = señales técnicas; context = metadatos
    del modelo y del broker en ese tick.

  MEJORA-2 — Inhibición de TIME_FORCE_CLOSE por spread elevado:
    Si ind_spread > MAX_SPREAD_FOR_CLOSE_PTS (configurable, defecto 18pts) en el
    momento en que se va a ejecutar un TIME_FORCE_CLOSE, S3 espera hasta que el
    spread baje o hasta que hayan pasado SPREAD_INHIBIT_MAX_WAIT_SECS (defecto 30s)
    desde que se superó el timeout. Así se evita ejecutar cierres rentables pagando
    un spread anómalo. Si el spread sigue alto tras el wait, el cierre se ejecuta
    igualmente para no dejar la posición sin gestión.

    Motivación empírica: sesión 11/03/2026, los 10 TIME_FORCE_CLOSE tuvieron PnL
    medio de +260pts. Un spread de 20pts en el cierre representa un coste de ~7.7%
    del profit medio — no es despreciable.

    Implementación: nuevo flag tr.spread_inhibit_since (float, 0.0 = sin inhibición).
    En el bloque TIME_FORCE_CLOSE, antes de llamar a _close_position_full:
      1. Si spread OK → cerrar normalmente.
      2. Si spread > umbral Y inhibit_since == 0 → registrar ts y continuar (no cerrar aún).
      3. Si spread > umbral Y inhibit_since > 0 Y wait < SPREAD_INHIBIT_MAX_WAIT_SECS → esperar.
      4. Si spread > umbral Y wait >= max → cerrar igualmente + emitir SPREAD_INHIBIT_TIMEOUT.
    Evento de auditoría: SPREAD_INHIBIT_ACTIVE (cuando empieza la espera) y
    SPREAD_INHIBIT_TIMEOUT (cuando se ejecuta el cierre a pesar del spread alto).

  MEJORA-3 — state y score en eventos de cierre:
    Todos los eventos de cierre (_close_position_full) incluyen ahora:
      "state": (getattr(tr, "ind_state", None) or getattr(tr, "ind_regime", None)),   # régimen en el momento del cierre
      "score":  tr.ind_score,    # score del modelo en el último tick con señal
    Permite análisis post-sesión directo desde el log de S3 sin join con S2.

  MEJORA-4 — keepalive: propagar campo 'keepalive' y loguear context en POSITION_MODIFIED:
    Ya incluido en v17 para keepalive. Se extiende para incluir también el context
    en POSITION_MODIFIED, permitiendo auditar la evolución del régimen y del spread
    durante la vida de cada posición.

Cambios v17.0 (sobre v16.0) — BUGFIXES POST-ANÁLISIS LOG 11/03/2026:

  FIX-1 — POSITION_CLOSED_EXTERNAL con symbol=null en 88% de los casos:
    Causa: _close_position_full() elimina el ticket de self._tracked ANTES de
    ejecutar el cierre en el broker (fix v10.0, correcto para evitar dobles
    cierres). Cuando el monitor detecta el ticket como "cerrado externamente"
    en el ciclo siguiente, self._tracked.pop(t, None) devuelve None porque ya
    fue eliminado, y tr_closed.symbol no es accesible → symbol=null en el evento.
    Confirmado sesión 11/03/2026: 68/77 POSITION_CLOSED_EXTERNAL con symbol=null.

    Fix: nuevo dict self._ticket_symbols (ticket → symbol) mantenido en paralelo
    a self._tracked. Se rellena en _handle_open y _recover_open_positions, y se
    consulta en el bloque de cierres externos como respaldo cuando tr_closed=None.
    Se limpia en el mismo bloque tras emitir el evento. Coste: O(1) por operación,
    memoria despreciable (1 string por posición abierta).

  FIX-2 — SLIPPAGE_ALERT para slippages extremos (> MAX_SLIPPAGE_ALERT_PTS):
    Causa: el ticket 316211980 (sesión 11/03) cerró con 524pts de slippage sin
    ningún evento de alerta específico. El slippage ya aparecía en el campo
    slippage_pts del evento de cierre (VIRTUAL_SL_TRIGGERED, etc.), pero no
    había forma de filtrar alertas automáticamente por umbral.

    Fix: en _close_position_full(), si slippage_pts > MAX_SLIPPAGE_ALERT_PTS
    (defecto: 200pts), se emite un evento SLIPPAGE_ALERT adicional con ticket,
    symbol, side, slippage_pts, threshold, bid, ask y close_reason para
    correlación inmediata. El umbral es configurable en __init__.
    El evento se emite DESPUÉS del evento de cierre principal para no interferir
    con el flujo existente.

  FIX-3 — MODIFY con keepalive: loguear campo 'keepalive' en POSITION_MODIFIED:
    Causa: S2 v23 añade 'keepalive: True' a los MODIFYs forzados para distinguirlos
    de los MODIFYs normales en el log. S3 ignoraba cualquier campo extra en el
    comando MODIFY y no lo propagaba al evento POSITION_MODIFIED, imposibilitando
    auditar cuántos MODIFYs eran keepalives vs datos frescos del pipeline.

    Fix: en _handle_modify(), si el comando incluye 'keepalive: True', añadir
    el campo al evento POSITION_MODIFIED. Sin cambios funcionales: los indicadores
    se actualizan exactamente igual, solo cambia la trazabilidad en el log.

Cambios v16.0 (sobre v15.0) — BUGFIX CRÍTICO: VSL_EXPANDED_MIN_ATR nunca actuaba:

  BUG (CRÍTICO) — Lógica invertida en guardia de vSL mínimo ATR (_validate_and_correct_tp_sl):
    Causa: la condición para aplicar la expansión de vSL era:
      if self._is_improvement(tr, old_sl, new_sl) is False:
    Es decir, solo expandía el vSL cuando el nuevo precio NO mejoraba el actual
    (cuando el nuevo SL era más cercano al precio que el original). Exactamente
    lo contrario de lo que se necesitaba: la guardia ampliaba vSLs ya suficientemente
    grandes y dejaba intactos los vSLs ajustados de 20-78 pts que debía corregir.
    Evidencia en log 11/03/2026: 0 eventos VSL_EXPANDED_MIN_ATR en toda la sesión
    con vSLs de 20-78 pts y ATR de 300-450 pts. La guardia nunca disparó.
    El FIX-2 de v15.0 (guardia S3 independiente de S2) quedó totalmente inoperante.

    Fix: cambiar `is False` → sin negación. La expansión se aplica exactamente
    cuando _is_improvement() devuelve True, es decir, cuando el nuevo SL coloca
    el nivel de protección más lejos del precio (más amplio → mejora).
    Impacto: con ATR=350 pts y MIN_VSL_ATR_RATIO_S3=0.40 → _min_vsl=140 pts.
    Cualquier vSL < 140 pts (como los 20-78 pts del log 11/03) será expandido
    automáticamente en S3 como segunda línea de defensa, independientemente de
    si S2 ya lo corrigió o no.

Cambios v15.0 (sobre v14.0) — MEJORAS Y BUGFIXES (análisis log 10/03/2026):

  FIX-1 — TIME_FORCE_CLOSE destruye valor en trades con momentum fuerte:
    Causa: el nivel 2 de TIME_FORCE_CLOSE (hold >= 300s AND profit > 0) cerraba
    cualquier trade en profit sin importar la magnitud. Análisis log 10/03/2026:
    14 de 36 cierres forzados tenían > 200 pts de beneficio flotante; casos
    extremos de 786 pts, 842 pts y 324 pts cerrados exactamente en el segundo 300.
    El umbral "profit > 0" (incluso 1 punto) es demasiado permisivo.

    Fix: el nivel 2 ahora exige profit_pts >= TIME_FORCE_CLOSE_MIN_PROFIT_R * vsl_pts
    (por defecto 0.3R, es decir al menos el 30% del riesgo inicial en ganancia)
    antes de forzar cierre. Esto evita que se cierre un trade que apenas está 1pt
    en verde y que potencialmente puede rebotar.
    Adicionalmente, si profit_pts > TIME_FORCE_CLOSE_EXTEND_R * vsl_pts (1.5R por
    defecto), el timeout se extiende TIME_FORCE_CLOSE_EXTEND_SECS (120s) una sola
    vez por posición, permitiendo que los trades con momentum real corran más.
    La extensión se registra con TIME_FORCE_CLOSE_EXTENDED para auditoría.
    La flag tr.time_force_close_extended evita extensiones múltiples.

  FIX-2 — Slippage masivo en vSL muy ajustados (max 275 pts en log 10/03):
    Causa: S3 aceptaba cualquier vSL sin validar su distancia mínima respecto al
    ATR. Con vSLs de 20-78 pts y ATR de ~200 pts, el precio saltaba el nivel
    completo en un solo bar generando slippage extremo (275, 160, 129 pts).
    El fix de S2 v17 (MIN_VSL_ATR_RATIO) no era suficiente por sí solo: S3 debe
    tener su propia guardia independiente de S2.

    Fix: en _validate_and_correct_tp_sl(), nueva guardia VSL_TOO_TIGHT que
    comprueba si virtual_sl_points < MIN_VSL_ATR_RATIO_S3 * atr_pts. Si el vSL es
    demasiado ajustado, lo amplia al mínimo ATR-based y emite VSL_EXPANDED_MIN_ATR
    para auditoría. Usa tr.ind_atr si disponible; si no, usa un fallback conservador
    de MIN_VSL_FALLBACK_POINTS (100 pts). Se aplica solo si el nuevo SL mejora
    (no sobreescribe un SL ya corregido y más amplio).

  FIX-3 — WRONG_SIDE_SL_CORRECTED: campos original_vsl/corrected_vsl siempre None:
    Causa: el evento WRONG_SIDE_SL_CORRECTED usaba las claves "original_vsl" y
    "corrected_vsl" en _send(), pero el análisis del log muestra que ambos campos
    llegaban siempre como None. Inspeccionando el código, los valores sí están
    disponibles (tr.virtual_sl_price antes y corrected después) pero el evento
    los publicaba con las claves incorrectas (no coincidían con los nombres
    reales de la variable).

    Fix: renombrar a "original_vsl" → valor real de tr.virtual_sl_price capturado
    ANTES de la corrección, y "corrected_vsl" → valor de corrected calculado.
    Añadido además "atr_pts" al evento para correlacionar la magnitud del error
    con el ATR del momento.

  FIX-4 — POSITION_CLOSED_EXTERNAL duplicado en 12 tickets (condición de carrera):
    Causa: el cache de posiciones (TTL=1s) puede devolver un ticket como "ausente"
    en dos ciclos consecutivos antes de que _known_open se actualice. En esos dos
    ciclos el bloque "closed = self._known_open - open_tickets" genera el mismo
    evento POSITION_CLOSED_EXTERNAL dos veces para el mismo ticket.

    Fix: en el bloque de detección de cierres externos, añadir un set
    _external_close_in_progress que registra los tickets ya procesados en el
    ciclo actual del monitor. Si un ticket ya fue procesado como cierre externo,
    se omite. El set se vacía al inicio de cada iteración del while loop.
    Adicionalmente, _known_open se actualiza a open_tickets solo después de
    procesar todos los cierres, garantizando que la diferencia solo se calcula
    una vez por ciclo.

  FIX-5 — AdaptiveSL/TP ciego sin indicadores: sin alerta proactiva:
    Causa: cuando S2 está bloqueado (MAX_POSITIONS) o desconectado, los comandos
    MODIFY con indicadores frescos no llegan. Los managers adaptativos operan
    solo con precio/vela sin RSI, MACD ni probabilidades. Este estado era invisible:
    ningún evento advertía de la degradación. Análisis 10/03: 0 POSITION_MODIFIED
    en toda la sesión de 81 minutos.

    Fix: nuevo campo tr.ind_last_update_ts en TrackedPos. Se actualiza cada vez
    que _update_tracked_indicators recibe datos. En _run_adaptive_sl y
    _run_adaptive_tp, si now() - tr.ind_last_update_ts > IND_STALE_WARN_SECS
    (120s) y la posición lleva abierta más de IND_STALE_MIN_HOLD (30s), se emite
    ADAPTIVE_INDICATORS_STALE una sola vez por posición (flag tr.ind_stale_warned).
    El evento incluye seconds_since_update y los campos que están en None para
    diagnóstico inmediato.

  FIX-6 — Doble evento por TIME_FORCE_CLOSE (TIME_FORCE_CLOSE_TRIGGERED +
           PROFIT_FLOOR_CLOSE_TRIGGERED para el mismo cierre):
    Causa: al activarse el nivel 2, el código emitía TIME_FORCE_CLOSE_TRIGGERED
    con los datos del trade y luego llamaba a _close_position_full() que emite
    internamente f"{reason}_TRIGGERED" → "PROFIT_FLOOR_CLOSE_TRIGGERED". Resultado:
    dos eventos distintos para un mismo cierre, conteo duplicado en análisis.

    Fix: se elimina el evento TIME_FORCE_CLOSE_TRIGGERED separado. En su lugar,
    se llama directamente a _close_position_full() con reason="TIME_FORCE_CLOSE",
    que emite "TIME_FORCE_CLOSE_TRIGGERED" como único evento de cierre con todos
    los campos (profit_points, hold_seconds, entry_price, close_price, slippage).
    Para ello _close_position_full() incluye ahora profit_pts y hold_seconds en
    el payload cuando reason contiene "TIME_FORCE_CLOSE" o "PROFIT_FLOOR".

Cambios v14.0 (sobre v13.0) — MEJORAS:

  P1 — Profit-lock dinámico por tiempo (time-based floor):
    Dos umbrales en _monitor_loop (antes del trailing, paso 4.8):
    · hold >= TIME_PROFIT_FLOOR_SECS (120s) Y profit_pts >= vsl_pts (>=1R)
      -> mover vSL a entry+be_offset si mejora el SL actual. Evento:
      TIME_PROFIT_FLOOR_ARMED. Evita que trades con 1R flotante vuelvan a BE.
    · hold >= TIME_FORCE_CLOSE_SECS (300s) Y profit_pts > 0
      -> cierre forzado (PROFIT_FLOOR_CLOSE). Evita que el precio elimine
      completamente un profit que existio. Evento: TIME_FORCE_CLOSE_TRIGGERED.

  P2 — Expansion AdaptiveSL dinamica por trade:
    _create_adaptive_manager calcula expansion_pts disponible antes de
    instanciar el AdaptiveSLManager, creando un config POR TRADE:
      available_pts = abs(virtual_sl - hard_sl) / 0.1
      expansion_pts = clamp(available_pts - hard_sl_margin_pts, 10, 25)
    Cualquier trade con margen > 35 pts puede expandir (antes umbral=75).
    Emite expansion_pts_computed y available_margin_pts en ADAPTIVE_SL_CREATED.

  P6 — Trailing step reducido (0.50R -> 0.20R):
    _calc_trailing_sl: dynamic_step = trail_distance * 0.20 (era 0.50).
    Con trail_points=40 pts: step 8 pts en vez de 20 pts, el trailing
    reacciona cada ~8 pts de avance en vez de ~20 pts.

Cambios v13.0 (sobre v12.0) — BUGFIXES:

  BUG-1 — expansion_pts=50 bloquea el 100% de las expansiones AdaptiveSL:
    Causa: _compute_expanded_sl() exige que exista un margen de
    (expansion_pts + hard_sl_margin_pts) puntos entre el virtual_sl y el
    hard_sl del broker. Con ambos a 50 pts se necesitan 100 pts de margen.
    El análisis del log 09/03/2026 (8 cierres AdaptiveSL) muestra que el
    margen real al abrir es mediana=99 pts (p25=74, min=46, max=199), de
    modo que expansion_blocked=hard_sl_margin_insuficiente en el 100%
    de los casos — la función anti-sweep nunca puede activarse.
    Fix: expansion_pts 50 → 25 (2.5 USD). El mínimo requerido baja a
    75 pts → la expansión se activa cuando el margen es ≥ 75 pts.
    Con los datos del 09/03: 2 de 8 cierres habrían podido expandir.

  BUG-2 — _validate_and_correct_tp_sl() no valida la dirección del vTP:
    Causa: el fix v11.0 añadió validación de dirección solo para el vSL
    (WRONG_SIDE_SL_CORRECTED). El vTP no se validaba: si S2 enviaba un
    virtual_tp al mismo lado que el drawdown (SELL con vTP > entry, o
    BUY con vTP < entry), el AdaptiveTPManager calculaba zonas erróneas
    y el VIRTUAL_TP nunca podía dispararse (el precio nunca cruza un TP
    que está en la dirección contraria al profit).
    Detectado en log 09/03/2026: ticket 314215545 SELL con vTP > entry.
    Fix: bloque idéntico al del SL — reflejo simétrico respecto al entry,
    emite WRONG_SIDE_TP_CORRECTED para auditoría.

Cambios v12.0 (sobre v11.0) — BUGFIXES:

  BUG-1 — bar OHLC plano en on_bar() del AdaptiveSL → indecision_ok siempre False:
    Causa: _run_adaptive_sl pasaba bar_open=bar_high=bar_low=bar_close al
    AdaptiveSLManager (todos igual al precio tick). Con OHLC plano, cualquier
    cálculo de body_ratio = |close-open|/|high-low| resulta en 0/0 → indecision_ok
    bloqueado estructuralmente, independientemente del mercado.
    Fix: extender _get_last_bar_ohlc() (nueva función, reemplaza _get_last_bar_close)
    para devolver el OHLC completo de la última vela M1 cerrada vía MT5
    copy_rates_from_pos(). Se pasan bar_open/high/low/close reales a on_bar().
    Fallback: si MT5 no disponible o falla, usar bid/ask como antes.

  BUG-2 — expansion_proba_threshold=0.45 por encima del rango real del modelo:
    Causa: El modelo calibrado produce proba_long/short en rango 0.23–0.41
    (máximo absoluto observado en producción). El umbral 0.45 nunca se alcanza,
    por lo que proba_ok es estructuralmente False aunque los indicadores lleguen.
    Fix: expansion_proba_threshold bajado a 0.35 (percentil 75 aprox. del modelo).
    Con esto proba_ok podrá activarse en ~25% de los ticks, aportando señal real
    al sistema de expansión anti-sweep.

  BUG-3 — partial_close.trigger_profit_pct del perfil scalping sobreescribible:
    S2 v13.0 ahora envía 'partial_close' dentro del dict 'risk'. El perfil
    scalping hardcodeaba trigger_profit_pct=100; S2 calculaba 60 pero no lo
    enviaba. El método _resolve_risk_profile ya soporta sobreescritura por clave,
    pero 'partial_close' es un subdict y no se fusionaba correctamente.
    Fix: en _resolve_risk_profile, si 'partial_close' llega en el risk override,
    fusionar sus campos sobre los del perfil base (merge en lugar de replace).

Cambios v11.0 (sobre v10.1) — BUGFIXES:

  BUG-1 (CRÍTICO) — vSL calculado en dirección incorrecta:
    Causa: En algunos casos S2 enviaba un virtual_sl_price que quedaba al lado
    incorrecto del entry (SELL con vSL < entry, BUY con vSL > entry). Esto
    provocaba que la posición se abriese ya "en zona SL" y se cerrase en el
    primer tick (0-2 segundos), con P&L aleatorio.
    Detectado: 6 de 43 posiciones el 06/03/2026 con cierre en +0.0s / +0.2s.
    Fix: _validate_and_correct_tp_sl() detecta vSL en dirección incorrecta y
    lo refleja simétricamente: corrected = entry ± |entry - vSL|.
    Emite WRONG_SIDE_SL_CORRECTED para auditoría.

  BUG-2 — AdaptiveSL con overshoot sistemático (+55 pts media, +473 pts max):
    Causa: expansion_min_signals=2 exigía 2 señales de reversión para cerrar
    al tocar el vSL. reversal_signals.count era 0 en el 100% de los cierres,
    el manager nunca cerraba en el tick exacto y el precio seguía en contra
    hasta agotar max_bars/max_seconds.
    Fix: expansion_min_signals=0 → cierre inmediato al tocar el vSL.
    La expansión anti-sweep sigue activa cuando hay señales reales (count > 0).

Cambios v10.1 (sobre v10.0) — BUGFIXES:

  BUG A — TypeError: MT5Bridge.close_position_market() got an unexpected
  keyword argument 'comment':
    Causa: En v10.0 se añadió `comment=` como kwarg separado a las llamadas
    de close_position_market(). MT5Bridge no define ese parámetro.
    Fix: Se elimina el kwarg `comment` y se sanitiza el string directamente
    en el campo `reason` ("mimo_close:<reason[:20]>") en los tres puntos de
    cierre: _close_position_full, MANUAL_CLOSE y PARTIAL_TO_FULL.

  BUG B — ADAPTIVE_SL_CREATED con virtual_tp absurdo (~5447) en recovery:
    Causa: _create_adaptive_manager usaba entry_price * 1.05/0.95 como
    fallback cuando tr.virtual_tp == 0. Para XAUUSD (~5180) esto genera un
    TP ficticio de ~5447 (5% lejos), completamente fuera de cualquier
    objetivo de scalping. El AdaptiveSL calculaba proximity zones erróneas
    y nunca activaba compresión ni expansión correctamente.
    Fix: Si no hay TP real, se estima 1.67R desde la entrada usando
    virtual_sl_points (ratio RR típico del perfil scalping). El log incluye
    el campo tp_source="real"|"estimated_1.67R"|"none" para auditoría.

  BUG C — ADAPTIVE_TP_SKIPPED en posiciones recuperadas con virtual_tp=0:
    Causa: Derivado del BUG B. tr.virtual_tp seguía siendo 0.0 (el fallback
    solo existía dentro de _create_adaptive_manager para el SLManager, no se
    propagaba a tr). El bloque AdaptiveTPManager veía tr.virtual_tp==0 y
    lo saltaba, dejando los dos managers desincronizados desde el arranque.
    Fix: El nuevo fallback en _create_adaptive_manager es solo para el
    AdaptiveSLManager (que sí necesita un TP reference). El AdaptiveTPManager
    se sigue saltando si no hay TP real — comportamiento correcto, pero ahora
    explícito y documentado.

  BUG D — inferred_profile="default" con comment "aggressive;scalp":
    Causa: La inferencia de perfil comparaba el candidato exacto contra la
    lista de nombres completos. "scalp" != "scalping" → fallback a "default".
    Consecuencia: la posición recuperada usaba emergency_points=100 en lugar
    de los 730 del perfil scalping, con BE, trail y partial_close incorrectos.
    Fix: Se añade un diccionario de aliases que resuelve abreviaciones comunes:
    scalp→scalping, agg→aggressive, cons→conservative.


Cambios v10.0 (sobre v9.0) — BUGFIXES:

  BUG 1 (CRÍTICO) — Cierres duplicados / VIRTUAL_SL_TRIGGERED con ok=False:
    Causa: _close_position_full enviaba la orden de cierre al broker pero NO
    eliminaba la posición de self._tracked ni de self._known_open. En el mismo
    ciclo del monitor_loop (o el siguiente, antes de que el price feed detectase
    el cierre externo) la posición volvía a evaluarse, el virtual SL seguía
    tocado y se lanzaban nuevos intentos de cierre sobre un ticket ya cerrado,
    todos fallando con retcode 10013 / 'Invalid request'.
    Fix: self._tracked.pop() + self._known_open.discard() + _remove_adaptive_manager()
    se ejecutan ANTES de llamar a bridge.close_position_market(), garantizando
    que la posición desaparece del tracking en el mismo instante en que se decide
    el cierre, independientemente de si la orden llega a ejecutarse o no.

  BUG 2 — ADAPTIVE_SL_CLOSE falla con 'Invalid "comment" argument' (retcode -2):
    Causa: close_position_market recibía el campo `reason` (p.ej. "ADAPTIVE_VIRTUAL_SL",
    "ADAPTIVE_SL", etc.) y el bridge lo usaba directamente como campo `comment`
    de la request MT5. MT5 acepta máx 31 chars ASCII imprimibles en ese campo;
    strings con guiones bajos o mayúsculas largas superan ese límite o producen
    el error -2.
    Fix: se añade el parámetro `comment` explícito con el formato compacto
    "mimo_close:<reason[:20]>" en todos los cierres de posición full.

  BUG 3 (derivado de BUG 1) — Ticket 307956525 cerrado externamente de inmediato:
    Una vez que BUG 1 queda resuelto, los cierres externos inmediatos pasan a ser
    genuinos (intervención manual o broker) y se registran como POSITION_CLOSED_EXTERNAL
    sin intentos adicionales de cierre.


Cambios v9.0 (sobre v8.0):
- ADAPTIVE TP: Integración del AdaptiveTPManager (adaptive_tp_manager.py).
  Simétrico al AdaptiveSLManager, cada posición tiene su gestor de TP:

    A) COMPRESIÓN del TP: cuando el precio se acerca al TP pero hay señales
       de agotamiento (RSI extremo, volumen decreciente, modelo pierde
       convicción), acerca el TP al precio actual para capturar la ganancia
       disponible antes de que el precio rebote.
       Se evalúa dentro de la zona de proximidad (tp_proximity_zone_pts).

    B) EXTENSIÓN del TP: cuando el precio toca el TP con señales de
       continuación fuertes (modelo convicto, volumen creciente, MACD
       acelerando, vela de breakout), extiende el objetivo y simultáneamente
       comprime el SL para proteger lo ganado.
       Solo una extensión por posición.

  FIX v9.0: El AdaptiveSLManager ya no cierra cuando la razón es VIRTUAL_TP
  (ese cierre lo gestiona el sistema estándar). Esto elimina el bug de doble
  cierre observado en el log del 25/02 donde ADAPTIVE_SL_CLOSE fallaba y
  VIRTUAL_TP_TRIGGERED cerraba la posición igualmente (con slippage añadido).

Cambios v8.0 (sobre v7.0):
- ADAPTIVE SL: Integración del AdaptiveSLManager (adaptive_sl_manager.py).
  Cada posición tiene su propio gestor de SL adaptativo que actúa ANTES
  de que el sistema de confirmación standard decida el cierre.

    A) EXPANSIÓN (anti-sweep): cuando el precio toca el virtual SL pero hay
       señales de reversión (modelo, RSI, MACD, volumen), amplía el SL una
       sola vez y da hasta expansion_max_bars/expansion_max_seconds para
       confirmar el rebote. Si no rebota → cierra sin más esperas.

    B) COMPRESIÓN proactiva: si la tesis se invalida (modelo gira, MACD cruza
       en contra, volumen adverso), acerca el SL al precio actual antes de
       que el precio llegue al SL original.

  El hard SL del broker (broker_emergency_sl) permanece siempre intacto.

  Indicadores opcionales que S2 puede enviar en el campo "indicators" del
  comando OPEN (o en cada ciclo via MODIFY) para potenciar el adaptativo:
      rsi, macd_hist, macd_hist_prev, volume, volume_ma, atr,
      proba_long, proba_short

  Si no se envían indicadores, el adaptativo opera con las señales
  disponibles (precio/vela), omitiendo las que faltan.

  Ver ADAPTIVE_SL_CONFIG en __init__ para ajustar los parámetros.

Cambios v7.0 (sobre v4.0):
- FIX SLIPPAGE: sl_mode del perfil scalping cambiado de 'close' a 'touch' y
  sl_confirm_count a 0 — esperar cierre de vela M1 antes de ejecutar el SL
  añadía hasta 60s de retraso en scalping, provocando slippages de 90+ puntos
- PRICE FEED THREAD: nuevo hilo _price_feed_loop dedicado exclusivamente a
  mantener precios bid/ask actualizados en self._prices (dict en memoria).
  El monitor lee de ese dict en lugar de llamar a MT5 en cada ciclo, reduciendo
  la latencia de reacción y las llamadas concurrentes a MT5
- POSITIONS CACHE: positions_get() ahora se cachea 1 segundo. La llamada es
  pesada y no necesita ejecutarse 5 veces/segundo; los precios siguen
  actualizándose a 20Hz desde el price feed thread
- LOG ROTATION FIX: el nombre del fichero de log se recalcula en cada escritura
  para rotar automáticamente a medianoche sin necesidad de reiniciar el servicio

Cambios v4.0 (sobre v3.0):
- FIX CRÍTICO: close_position_partial en mt5_bridge construye su propia request
  (elimina el bug 'Invalid comment argument' que bloqueaba todos los cierres parciales)
- trail_step_points del perfil scalping subido a 20 pts (~50% del trail_points)
- _calc_trailing_sl: step dinámico = max(configurado, 50% trail_distance) para
  perfiles ATR que reciben trail_points grande pero tenían step pequeño del perfil base
- Loop de monitorización: partial_close se ejecuta ANTES del break-even para evitar
  race condition donde BE se calcula sobre volumen no reducido
- _check_partial_close: vol_to_close pasa por quantize_volume para evitar
  floating-point acumulado; log explícito PARTIAL_CLOSE_VOLUME_NOT_UPDATED si falla
- _close_position_full: campo slippage_pts añadido a VIRTUAL_SL_TRIGGERED
"""
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from json import dumps as json_dump, loads as json_load
from pathlib import Path

S3_SCRIPT_NAME = "s3_service.py"
S3_VERSION = "30.0"

# v26.0: directorio base del script, independiente del CWD del proceso.
# Usado en _resolve_log_path para que los logs siempre se escriban en
# <directorio_del_script>/logs/ sin importar desde dónde se lance S3.
_S3_DIR = Path(__file__).parent

"""
Cambios v23.0 (sobre v22.0) — FIXES post-análisis log 16/03/2026:

  FIX-1 (cambio 1) — AdaptiveTP extension deshabilitada en estados ruidosos:
    Causa: en el log del 16/03/2026, los 2 cierres ADAPT_TP ocurrieron en estado
    EXTENDED con extensión de TP activada. El runner post-extensión revirtió
    completamente: PnL total = -60.1 pts·lot (-30.0 de media). La extensión
    apuesta por continuación en un estado donde el precio ya superó el TP original
    y tiene alta probabilidad de revertir.
    Fix: nuevo dict STATE_OVERRIDES en __init__ que mapea grupos de estados a
    overrides de comportamiento. Para EXTENDED, TRANSITION_* y RANGE se desactiva
    extension_enabled en el AdaptiveTPManager antes de llamar a on_bar. El config
    se restaura siempre en un bloque finally, evitando mutación persistente.
    Nuevo evento ADAPTIVE_TP_EXTENSION_OVERRIDDEN para trazabilidad en el log.
    El dict es completamente configurable sin tocar el código del manager.

  FIX-2 (cambio 2) — tp_confirm_count dinámico por estado para el VTP:
    Causa: ticket 317873970 (16/03/2026), SELL 0.96 lotes en estado TRANSITION_DOWN.
    tp_confirm_count=0 (touch) ejecutó el VTP sobre el primer toque del precio,
    que resultó ser ruido. El partial close previo había reducido el volumen del
    runner, que cerró en pérdida (-29.3 pts·lot).
    Fix: nuevo método _effective_tp_confirm_count(tr) que consulta STATE_OVERRIDES
    y devuelve max(base_del_perfil, override). Para TRANSITION_* y RANGE devuelve 1,
    exigiendo que el precio toque el TP al menos 2 ciclos consecutivos antes de
    ejecutar el cierre. Solo puede AUMENTAR el valor base del perfil, nunca reducirlo.
    El campo 'override_applied' en el evento TP_CONFIRMED permite auditar cuántas
    veces actuó el override sin buscar en ADAPTIVE_TP_EXTENSION_OVERRIDDEN.

  FIX-3 (cambio 3, aplicado en S2 v33.0) — Keepalive por vela M1:
    Ver main_trading_s2_v33.py — KEEPALIVE_ON_NEW_BAR=True.
    En S3 no hay cambios de código: el beneficio se recibe automáticamente
    porque S2 enviará MODIFY en cada vela en lugar de cada 60s reales.
    El IND_STALE_WARN_SECS=120 de S3 sigue sin tocar; con el fix de S2 el
    gap máximo entre MODIFYs será 1 vela (~60s), la mitad del umbral de alerta.
"""

from typing import Dict, Any, List, Optional

import dataclasses
import json
import zmq
try:
    from adaptive_sl_manager import AdaptiveSLManager, AdaptiveSLConfig
    from adaptive_tp_manager import AdaptiveTPManager, AdaptiveTPConfig
    _ADAPTIVE_SL_AVAILABLE = True
except ImportError:
    AdaptiveSLManager = None
    AdaptiveSLConfig = None
    AdaptiveTPManager = None
    AdaptiveTPConfig = None
    _ADAPTIVE_SL_AVAILABLE = False
try:
    import MetaTrader5 as _mt5
except ImportError:
    _mt5 = None  # fallback: sl_mode=close usará precio bid/ask

import mt5_bridge as _mt5b
print(f"[DEBUG] mt5_bridge cargado desde: {_mt5b.__file__}")

# ============================================================================
# CONFIGURACIÓN
# ============================================================================

# Constantes de training — deben coincidir con s2_config.py y main_oof.py.
# Usadas en recovery (POSITION_RECOVERED) para reconstruir virtual_tp cuando
# el broker no tiene TP registrado (tp=0). BUG preexistente v28.6: si estas
# constantes no están definidas aquí, POSITION_RECOVERED lanza NameError.
TRAINING_SL_BARRIER_R: float = 1.5
TRAINING_TP_BARRIER_R: float = 2.5
TRAINING_LABEL_RR: float = TRAINING_TP_BARRIER_R / TRAINING_SL_BARRIER_R  # 1.6667

def now() -> float:
    """Timestamp actual en segundos"""
    return time.time()


class _LogEncoder(json.JSONEncoder):
    """Encoder JSON que serializa dataclasses y conjuntos (set)"""
    def default(self, obj):
        if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
            return dataclasses.asdict(obj)
        if isinstance(obj, set):
            return list(obj)
        return super().default(obj)


def norm_action(a: str) -> str:
    """Normaliza acción"""
    return (a or "").strip().upper()


def safe_side(s: str) -> str:
    """Normaliza side a BUY/SELL"""
    s = (s or "").strip().upper()
    if s in ("LONG", "BUY", "B", "L"):
        return "BUY"
    if s in ("SHORT", "SELL", "S"):
        return "SELL"
    return s


# ============================================================================
# DATACLASSES
# ============================================================================

@dataclass
class CloseConfirmationConfig:
    """
    Configuración de confirmación para cierres

    Modos de salida:
    - 'hard': Cierra al tocar nivel (sin confirmación)
    - 'soft': Requiere que vela cierre más allá del nivel
    - 'close_confirm': Requiere N barras consecutivas confirmando
    """

    exit_mode: str = "hard"  # 'hard', 'soft', 'close_confirm'
    close_confirm_bars: int = 1  # Barras necesarias para confirmación

    tp_mode: str = "touch"  # 'touch' o 'close'
    tp_confirm_count: int = 0  # Confirmaciones adicionales para TP

    sl_mode: str = "touch"  # 'touch' o 'close' (generalmente 'touch')
    sl_confirm_count: int = 0  # Confirmaciones para SL (generalmente 0)

    use_close_price: bool = False  # Usar precio de cierre de vela
    min_bars_beyond: int = 0  # Barras mínimas más allá del nivel


@dataclass
class ConfirmationTracker:
    """Tracking de confirmaciones para niveles TP/SL"""

    tp_confirm_count: int = 0
    sl_confirm_count: int = 0

    tp_close_history: List[bool] = field(default_factory=list)
    sl_close_history: List[bool] = field(default_factory=list)

    last_close_price: float = 0.0
    last_close_ts: float = 0.0

    tp_touched: bool = False
    sl_touched: bool = False


@dataclass
class PartialCloseCfg:
    enabled: bool = False
    trigger_profit_pct: float = 0.0
    close_fraction: float = 0.0
    basis: str = 'R'  # R or EQUITY
    runner_tight_trail_pts: int = 0  # v28.0: si > 0, tras el partial close el trailing
                                     # del runner usa esta distancia en lugar del normal.
                                     # Ej: 20pts -> SL del runner a 20pts del máximo favorable,
                                     # casi breakeven. 0 = comportamiento previo (trailing normal).


@dataclass
class RiskCfg:
    profile: str = ''
    emergency_points: int = 0
    be_trigger_points: int = 0
    be_offset_points: int = 0
    trail_points: int = 0
    trail_step_points: int = 0
    max_hold_seconds: int = 0
    min_hold_seconds: int = 0   # Tiempo minimo antes de permitir cierre virtual
    max_spread_points: int = 0  # Spread maximo para aceptar la entrada (0 = sin filtro)
    partial_close: PartialCloseCfg = field(default_factory=PartialCloseCfg)
    close_confirmation: CloseConfirmationConfig = field(default_factory=CloseConfirmationConfig)


@dataclass
class TrackedPos:
    ticket: int
    symbol: str
    side: str  # BUY/SELL
    volume: float
    entry_price: float
    point: float
    magic: int
    opened_ts: float
    risk: RiskCfg

    # Sistema DUAL de Stop Loss
    broker_emergency_sl: float = 0.0
    emergency_sl_points: int = 0

    virtual_sl_price: float = 0.0
    virtual_sl_points: int = 0

    # Control interno
    be_armed: bool = False
    be_armed_ts: float = 0.0        # v29.0: instante exacto en que se armó el BE
    trailing_wait_logged: bool = False  # v29.0: evita spam de TRAILING_DELAY_ACTIVE
    trailing_unlocked: bool = False     # v30.0: estado latched con histéresis para evitar parpadeo
    be_skip_warned: bool = False    # v24.0: suprime BE_SKIPPED duplicados (se emite solo 1 vez)
    # v25.0: último estado para el que se emitió ADAPTIVE_TP_EXTENSION_OVERRIDDEN.
    # Permite emitir el evento solo cuando el estado cambia, no en cada ciclo del monitor.
    _tp_override_last_state: Optional[str] = None
    virtual_tp: float = 0.0
    partial_done: bool = False

    max_favorable_price: float = 0.0

    # Sistema de confirmación
    confirmation: ConfirmationTracker = field(default_factory=ConfirmationTracker)

    # Indicadores para el AdaptiveSLManager (actualizados en cada ciclo si vienen de S2)
    # Todos opcionales; si son None el adaptativo ignora esa señal.
    ind_atr: Optional[float] = None
    ind_rsi: Optional[float] = None
    ind_macd_hist: Optional[float] = None
    ind_macd_hist_prev: Optional[float] = None
    ind_volume: Optional[float] = None
    ind_volume_ma: Optional[float] = None
    ind_proba_long: Optional[float] = None
    ind_proba_short: Optional[float] = None

    # FIX v15.0 (FIX-5): timestamp de la última actualización de indicadores.
    # Permite detectar cuando S2 lleva mucho tiempo sin enviar MODIFY (degradación silenciosa).
    ind_last_update_ts: float = 0.0
    ind_stale_warned: bool = False  # True una vez emitido ADAPTIVE_INDICATORS_STALE

    # FIX v15.0 (FIX-1): flag para la extensión única del TIME_FORCE_CLOSE.
    # Evita que un trade con profit > 1.5R extienda el timeout más de una vez.
    time_force_close_extended: bool = False

    # v18.0: contexto de mercado — actualizados en cada MODIFY junto con indicators.
    # Mantenidos separados de los indicadores técnicos por claridad semántica:
    # indicators = señales técnicas del mercado (RSI, MACD, ATR…)
    # context    = metadatos del modelo y del broker en ese tick
    ind_regime: Optional[str]   = None   # compatibilidad retro con logs/código anterior
    ind_state:  Optional[str]   = None   # estado del modelo (state) — campo canónico v21+
    ind_score:  Optional[float] = None   # score del modelo (solo cuando hay señal activa)
    ind_spread: Optional[int]   = None   # spread en puntos del último tick recibido

    # v18.0: inhibición de TIME_FORCE_CLOSE por spread elevado.
    # 0.0 = no hay inhibición activa. > 0 = ts desde el que se está esperando.
    spread_inhibit_since: float = 0.0


# ============================================================================
# CONFIGURACIÓN POR DEFECTO
# ============================================================================

# Perfiles de riesgo predefinidos
RISK_PROFILES = {
    "default": {
        "emergency_points": 100,
        "be_trigger_points": 50,
        "be_offset_points": 5,
        "trail_points": 50,
        "trail_step_points": 10,  # NUEVO v3.0
        "max_hold_seconds": 3600,
        'min_hold_seconds': 0,      # Sin restriccion por defecto
        'max_spread_points': 0,     # Sin filtro de spread por defecto
        "partial_close": {
            "enabled": True,
            "trigger_profit_pct": 150,
            "close_fraction": 0.5,
            "basis": "R",
            "runner_tight_trail_pts": 20
        },
        "close_confirmation": {
            "exit_mode": "hard",
            "tp_mode": "touch",
            "tp_confirm_count": 0,
            "sl_mode": "touch",
            "sl_confirm_count": 0
        }
    },

    "scalping": {
        "emergency_points": 50,         # sobreescrito por el main via ATR; este es el fallback
        # FIX 13/04/2026: be_trigger 40→100, trail 40→60.
        # Con be_trigger=40 (~13% del vSL de 313pts) el BE se armaba demasiado pronto
        # cortando winners a avg +0.22R mientras los losers llegaban a -1.05R (EV=-0.25R).
        # Análisis log 13/04/2026: 30 cierres por vSL, 0 VIRTUAL_TP_TRIGGERED en 34 trades.
        # Subir be_trigger a 100pts (~32% del vSL) da margen para que el precio respire
        # sin disparar el BE prematuramente. trail=60 (vs 40) amplía el corredor de trailing.
        "be_trigger_points": 100,       # armar BE cuando precio avanza 100pts (era 40)
        "be_offset_points": 15,         # BE más holgado para cubrir spread/slippage sin volver a negativo
        "trail_points": 60,             # distancia del trailing desde el máximo favorable (era 40)
        "trail_step_points": 20,        # mover SL solo si mejora al menos 20pts (~33% de trail_points)
        # 1100s = label_horizon (3×5min=15min) + ~3min buffer.
        # 900s coincidía exacto con el horizonte de entrenamiento, sin margen para:
        #   (a) latencia OPEN entre cierre 5m en S2 y ejecución en MT5 (1-5min típico),
        #   (b) que triple-barrier evalúa EXPIRE en el cierre del bar 3, no a los 900s netos,
        #   (c) volatilidad de ticks/spread cerca del límite.
        # Resultado anterior: cierres MAX_HOLD_TIME prematuros que perdían la cola TP que
        # el etiquetado del modelo sí capturaba como TP-first.
        "max_hold_seconds": 1100,
        'min_hold_seconds': 15,         # mínimo 15s antes de permitir cierre virtual
        'max_spread_points': 20,        # v15.1: subido de 15 → 20pts. Con spread normal
                                        # ~10pts y picos típicos de 15-18pts en noticias,
                                        # 15 rechazaba entradas válidas (confirmado sesión
                                        # 11/03/2026: OPEN_REJECTED_SPREAD con spread=16pts).
                                        # S2 v20 tiene el mismo límite como primera defensa.
        "partial_close": {
            "enabled": True,
            "trigger_profit_pct": 100,
            "close_fraction": 0.5,
            "basis": "R"
        },
        "close_confirmation": {
            "exit_mode": "hard",
            "tp_mode": "touch",         # coger el TP rápido
            "tp_confirm_count": 0,
            "sl_mode": "touch",         # FIX v7.0: touch en lugar de close para evitar slippage en scalping
            "sl_confirm_count": 0       # FIX v7.0: sin confirmación adicional para reacción inmediata
        }
    },

    "swing": {
        "emergency_points": 200,
        "be_trigger_points": 100,
        "be_offset_points": 10,
        "trail_points": 100,
        "trail_step_points": 30,
        "max_hold_seconds": 432000,
        'min_hold_seconds': 60,         # mínimo 1 minuto para swing
        'max_spread_points': 30,        # spread más tolerante en swing
        "partial_close": {
            "enabled": True,
            "trigger_profit_pct": 200,
            "close_fraction": 0.5,
            "basis": "R"
        },
        "close_confirmation": {
            "exit_mode": "soft",
            "tp_mode": "close",
            "tp_confirm_count": 1,
            "sl_mode": "touch",
            "sl_confirm_count": 0
        }
    },

    "conservative": {
        "emergency_points": 80,
        "be_trigger_points": 30,
        "be_offset_points": 5,
        "trail_points": 40,
        "trail_step_points": 8,
        "max_hold_seconds": 1800,
        'min_hold_seconds': 10,         # mínimo 10s: protección básica
        'max_spread_points': 20,        # spread estricto acorde al perfil
        "partial_close": {
            "enabled": True,
            "trigger_profit_pct": 80,
            "close_fraction": 0.7,
            "basis": "R"
        },
        "close_confirmation": {
            "exit_mode": "hard",
            "tp_mode": "touch",
            "tp_confirm_count": 0,
            "sl_mode": "touch",
            "sl_confirm_count": 0
        }
    },

    "aggressive": {
        "emergency_points": 150,
        "be_trigger_points": 80,
        "be_offset_points": 5,
        "trail_points": 80,
        "trail_step_points": 20,
        "max_hold_seconds": 10800,
        'min_hold_seconds': 0,          # sin restricción: el perfil acepta entradas rápidas
        'max_spread_points': 40,        # spread más tolerante para no perder entradas
        "partial_close": {
            "enabled": True,
            "trigger_profit_pct": 250,
            "close_fraction": 0.4,
            "basis": "R"
        },
        "close_confirmation": {
            "exit_mode": "soft",
            "tp_mode": "close",
            "tp_confirm_count": 0,
            "sl_mode": "touch",
            "sl_confirm_count": 0
        }
    }
}


def get_risk_profile(profile_name: str = "default") -> Dict[str, Any]:
    """
    Obtiene un perfil de riesgo por nombre

    Args:
        profile_name: Nombre del perfil (default, scalping, swing, conservative, aggressive)

    Returns:
        Diccionario con la configuración de riesgo
    """
    profile = RISK_PROFILES.get(profile_name)

    if profile is None:
        print(f"[WARNING] Perfil '{profile_name}' no encontrado, usando 'default'")
        return RISK_PROFILES["default"].copy()

    return profile.copy()


# ============================================================================
# SERVICIO S3
# ============================================================================

class S3Service:
    """
    Servicio de gestión de riesgo con confirmación de cierre
    """

    def __init__(
            self,
            orders_pull_addr: str = "tcp://10.1.21.25:5557",
            events_pub_addr: str = "tcp://10.1.21.25:5558",
            monitor_interval: float = 0.2,
            log_file: Optional[str] = None
    ):
        """
        Args:
            orders_pull_addr: Dirección ZMQ para recibir órdenes
            events_pub_addr: Dirección ZMQ para publicar eventos
            monitor_interval: Intervalo de monitorización en segundos
            log_file: Ruta del fichero de log (por defecto: s3_events_YYYYMMDD.jsonl)
        """
        self.orders_pull_addr = orders_pull_addr
        self.events_pub_addr = events_pub_addr
        self.monitor_interval = monitor_interval

        # ── P1 v14.0: umbrales para profit-lock dinamico por tiempo ───────────
        self.TIME_PROFIT_FLOOR_SECS: int = 120   # hold >= X s Y profit >= 1R -> floor BE
        self.TIME_FORCE_CLOSE_SECS:  int = 300   # hold >= X s Y profit > 0 -> cierre forzado

        # ── FIX v15.0 (FIX-1): refinamiento TIME_FORCE_CLOSE ────────────────
        # Profit mínimo requerido para que el nivel 2 fuerce el cierre.
        # 0.3R = al menos el 30% del riesgo inicial en ganancia (antes: profit > 0,
        # lo que incluía incluso 1pt y cerraba trades que podían rebotar).
        self.TIME_FORCE_CLOSE_MIN_PROFIT_R: float = 0.30
        # Si profit > EXTEND_R * vsl_pts, extender el timeout EXTEND_SECS una sola vez.
        # Permite que trades con momentum real (> 1.5R a los 300s) corran algo más.
        self.TIME_FORCE_CLOSE_EXTEND_R:    float = 1.50  # 1.5R de beneficio → extender
        self.TIME_FORCE_CLOSE_EXTEND_SECS: int   = 120   # extensión única de 2 min

        # ── FIX v15.0 (FIX-2): guardia de vSL mínimo en S3 ─────────────────
        # MIN_VSL_ATR_RATIO_S3: fracción del ATR usada como distancia mínima del vSL.
        # 0.4 ATR = guardia ligeramente más conservadora que la de S2 (0.5 ATR)
        # para actuar como segunda línea de defensa independiente.
        # Con ATR típico 200 pts → mínimo 80 pts; con ATR 300 pts → 120 pts.
        self.MIN_VSL_ATR_RATIO_S3:    float = 0.40
        # Fallback cuando ind_atr no está disponible (sin MODIFY de S2).
        self.MIN_VSL_FALLBACK_POINTS: int   = 100

        # ── FIX v15.0 (FIX-5): alerta de indicadores obsoletos ──────────────
        # Segundos sin recibir MODIFY antes de emitir ADAPTIVE_INDICATORS_STALE.
        self.IND_STALE_WARN_SECS:  int = 120  # 2 min sin actualización → alerta
        # Tiempo mínimo de hold antes de evaluar staleness (evitar falsas alarmas
        # en los primeros ticks de vida de la posición antes del primer MODIFY).
        self.IND_STALE_MIN_HOLD:   int = 30   # al menos 30s de posición abierta

        # ── FIX v17.0 (FIX-2): umbral de alerta de slippage extremo ─────────
        # Si el slippage de un cierre supera este umbral, se emite un evento
        # SLIPPAGE_ALERT adicional para facilitar alertas y correlación en el log.
        # 200pts en XAUUSD = $20 — cubre slippages normales (<50pts habituales)
        # con margen suficiente para no generar ruido en condiciones normales.
        self.MAX_SLIPPAGE_ALERT_PTS: int = 200

        # Si el slippage supera este umbral, S3 intenta un segundo cierre de
        # mercado inmediato (el primero cerró parcialmente o no ejecutó al precio
        # esperado por un gap de precio). Valor conservador: 400pts en XAUUSD = $40.
        # El caso del 13/04/2026 (ticket 334460414, slippage=514pts) hubiera
        # activado este reintento para proteger el capital residual.
        # IMPORTANTE: el reintento opera sobre el volumen RESIDUAL visible en MT5
        # tras el primer cierre — si el primer cierre fue total, el reintento
        # simplemente no encontrará volumen y abortará limpiamente.
        self.MAX_SLIPPAGE_RETRY_PTS: int = 400

        # ── FIX v20.1 (FIX-4): grace period anti-falso-DETECTED_CLOSED ──────
        # MT5 tarda entre 100-500ms en registrar una posición nueva en su API
        # interna. En ese intervalo, _monitor_loop llama a positions_get() y no
        # encuentra el ticket recién abierto. Como el ticket ya está en _known_open
        # (añadido en _handle_open justo antes), la diferencia
        # "closed = _known_open - open_tickets" incluye el ticket nuevo →
        # falso POSITION_CLOSED_EXTERNAL a los ~155ms del OPEN.
        # Consecuencia: S3 elimina el ticket de _tracked y deja la posición real
        # sin supervisión (sin trailing, BE, TIME_FORCE_CLOSE ni vSL).
        # Confirmado 13/03/2026: ticket 317234422 cerrado como DETECTED_CLOSED a
        # los 155ms; el trade continuó abierto más de 3h sin gestión de S3.
        # Fix: en el bloque de detección de cierres externos, ignorar tickets cuyo
        # opened_ts sea más reciente que OPEN_GRACE_SECS. Solo tras ese periodo
        # se considera que MT5 ya ha confirmado el estado de la posición.
        # 5s: amplio margen sobre el peor caso observado (500ms); no interfiere
        # con cierres manuales rápidos reales que tardarían >10s en ejecutarse.
        self.OPEN_GRACE_SECS: float = 5.0

        # v27.0: cooldown anti-duplicado de aperturas.
        # Si se procesó un OPEN hace menos de OPEN_DEDUP_SECS, rechazar el siguiente.
        # La frecuencia de señales legítimas es ~1/min (M1); cualquier segundo OPEN
        # en menos de OPEN_DEDUP_SECS es un duplicado del buffer ZMQ.
        self.OPEN_DEDUP_SECS: float = 5.0   # ventana de bloqueo tras cada OPEN
        self._last_open_sent_ts: float = 0.0  # time.time() del último OPEN procesado

        # ── v22.0: validación post-fill en _handle_open ───────────────────────
        # Si el precio de mercado en el instante posterior al fill está a más de
        # POST_FILL_GAP_MAX_MULT × vSL del entry reportado por MT5, la geometría
        # del trade (vSL, hardSL) es incoherente con la realidad y S3 cierra
        # inmediatamente con reason POST_FILL_GAP_ABORT antes de añadir el ticket
        # a _tracked.
        # Origen: ticket 317706824 (13/03/2026) SELL abierto en 5013.99 pero el
        # precio real al primer ciclo del monitor era 5045.69 → gap = 3170pt =
        # 14.2× el vSL. ADAPTIVE_HARD_SL se activó instantáneamente: -14.22R.
        # El filtro ENTRY_GAP_TOO_LARGE de S2 no lo capturó porque el movimiento
        # ocurrió durante la latencia entre envío y fill (no antes del envío).
        # Umbral 2× vSL: cubre el gap post-fill esperado en condiciones normales
        # (<5pt) con amplísimo margen, y bloquea movimientos extremos de news/flash
        # crash. Con vSL típico de 250pt → umbral = 500pt (5 USD).
        self.POST_FILL_GAP_MAX_MULT: float = 2.0

        # ── v23.0: overrides de comportamiento por grupo de estados ──────────
        #
        # STATE_OVERRIDES: mapea grupos de estados (frozenset) a un sub-dict de
        # ajustes que S3 aplica en tiempo de ejecución cuando tr.ind_state
        # pertenece al grupo. El override se evalúa en cada ciclo del monitor;
        # si el estado cambia, el comportamiento vuelve al por defecto automáticamente.
        #
        # Claves soportadas:
        #   'extension_enabled' (bool): si False, desactiva la extensión del TP
        #       en _run_adaptive_tp aunque AdaptiveTPConfig.extension_enabled=True.
        #       Motivación: en EXTENDED/TRANSITION el runner extendido revirtió
        #       en 2/2 casos (log 16/03/2026, PnL=-60.1 pts·lot).
        #
        #   'tp_confirm_count' (int): sobreescribe tp_confirm_count del perfil para
        #       el VTP mientras el trade esté en ese estado. Solo puede AUMENTAR
        #       el valor base (nunca reducirlo), garantizando más confirmaciones en
        #       estados ruidosos. Motivación: ticket 317873970 (16/03/2026), VTP
        #       touch en TRANSITION_DOWN cerró en pérdida por ruido.
        #
        # Para deshabilitar: self.STATE_OVERRIDES = {}
        #
        self.STATE_OVERRIDES: dict = {
            # Transiciones: más ruido, TP más conservador y sin extensión.
            frozenset({
                'TRANSITION_UP', 'TRANSITION_DOWN', 'TRANSITION',
                'transition_up', 'transition_down', 'transition',
            }): {
                'extension_enabled': False,  # (1) no extender en transición
                'tp_confirm_count':  1,       # (2) confirmar antes de cerrar VTP
            },
            # EXTENDED: runner post-extensión en zona de reversión frecuente.
            frozenset({
                'EXTENDED', 'extended',
            }): {
                'extension_enabled': False,  # (1) no re-extender
                'tp_confirm_count':  0,       # VTP ya fue extendido: touch OK
            },
            # Rango y baja volatilidad: precio rebota entre niveles.
            frozenset({
                'RANGE', 'range', 'LOW_VOL', 'low_vol',
            }): {
                'extension_enabled': False,
                'tp_confirm_count':  1,
            },
            # Tendencias confirmadas: comportamiento por defecto (extensión ON, touch).
            # No hace falta listarlo: ausencia de override = defaults del perfil.
        }

        # Si el spread supera MAX_SPREAD_FOR_CLOSE_PTS en el momento del cierre,
        # S3 espera hasta SPREAD_INHIBIT_MAX_WAIT_SECS antes de ejecutar.
        # Protege cierres rentables de pagar spreads anómalos.
        # 18pts: cubre el spread normal de XAUUSD (~10pts) con margen; por encima
        # de 18pts ya estamos en spread de noticia o spread patológico.
        # 30s: ventana suficiente para que el spread normalice en picos de news;
        # más tiempo dejaría la posición sin gestión en momentos críticos.
        self.MAX_SPREAD_FOR_CLOSE_PTS:    int = 18
        self.SPREAD_INHIBIT_MAX_WAIT_SECS: int = 30

        # ── v29.0: desacoplar BREAK-EVEN y TRAILING ───────────────────────
        # Causa raíz observada el 13/04/2026: el trailing podía activarse en
        # el mismo ciclo en que se armaba el BE, provocando cierres prematuros
        # en el primer retroceso normal tras alcanzar +100pts.
        # Nuevo criterio: el trailing solo se habilita cuando han pasado al
        # menos TRAILING_AFTER_BE_MIN_SECS desde BE_ARMED y el precio además
        # ha avanzado una ganancia adicional mínima sobre el trigger del BE.
        self.TRAILING_AFTER_BE_MIN_SECS: float = 12.0
        self.TRAILING_AFTER_BE_EXTRA_PTS: int = 40
        self.TRAILING_AFTER_BE_EXTRA_R: float = 0.15
        self.TRAILING_UNLOCK_HYSTERESIS_PTS: int = 8   # v30.0: evita ON/OFF cerca del umbral
        self.INTENTIONAL_CLOSE_GRACE_SECS: float = 5.0  # v30.0: suprime POSITION_CLOSED_EXTERNAL tras cierres propios

        # ── Fichero de log (con rotación automática a medianoche) ───────────
        self._log_custom = log_file          # None = rotar por fecha; str = ruta fija
        self._log_path   = self._resolve_log_path()
        self._log_lock   = threading.Lock()
        print(f"[S3Service] Mensajes guardados en: {self._log_path.resolve()}")

        # ZMQ
        self.context = zmq.Context()

        # Socket para recibir comandos
        self.orders_socket = self.context.socket(zmq.PULL)
        self.orders_socket.setsockopt(zmq.RCVHWM, 10000)
        self.orders_socket.setsockopt(zmq.LINGER, 0)

        self.orders_socket.bind(orders_pull_addr)

        # Socket para publicar eventos
        self.events_socket = self.context.socket(zmq.PUB)
        self.events_socket.setsockopt(zmq.SNDHWM, 20000)
        self.events_socket.setsockopt(zmq.LINGER, 1000)
        self.events_socket.bind(events_pub_addr)

        time.sleep(0.1)

        # Bridge MT5
        try:
            from mt5_bridge import MT5Bridge
            self.bridge = MT5Bridge()
            self.bridge.initialize_or_raise()

        except Exception as e:
            raise RuntimeError(f"Failed to initialize MT5Bridge: {e}")

        # Estado
        self._tracked: Dict[int, TrackedPos] = {}
        self._known_open = set()
        self._stop = threading.Event() if 'threading' in dir() else None

        # FIX v17.0 (FIX-1): respaldo ticket → symbol para POSITION_CLOSED_EXTERNAL.
        # _close_position_full elimina de _tracked antes del cierre (fix v10.0),
        # por lo que cuando el monitor detecta el cierre externo tr_closed ya es None.
        # Este dict persiste el symbol hasta que se confirma el cierre externo.
        self._ticket_symbols: Dict[int, str] = {}
        self._intentional_close_tickets: Dict[int, float] = {}  # v30.0
        self._recent_closed_context: Dict[int, Dict[str, Any]] = {}  # v30.0

        # ── Price feed: precios en memoria, actualizados por hilo dedicado ──
        self._prices: Dict[str, tuple] = {}       # symbol -> (bid, ask)
        self._prices_lock = threading.Lock()

        # ── Positions cache: evitar llamadas excesivas a positions_get() ────
        self._positions_cache: list = []
        self._positions_cache_ts: float = 0.0
        self._positions_cache_ttl: float = 1.0   # refrescar cada 1 segundo

        # v23.0: serializar STATE_OVERRIDES para auditoría en el log de arranque.
        # frozenset no es serializable por defecto: convertir las claves a listas ordenadas.
        _so_serializable = {
            str(sorted(k)): v
            for k, v in self.STATE_OVERRIDES.items()
        }
        self._send({
            "event":           "SERVICE_STARTED",
            "service":         S3_SCRIPT_NAME,
            "version":         S3_VERSION,
            "state_overrides": _so_serializable,
        })

        # ── AdaptiveSLManager ────────────────────────────────────────────────
        # Un manager por posición abierta. Se crea en _handle_open/_recover,
        # se consulta en _monitor_loop y se destruye al cerrar la posición.
        self._adaptive_managers: Dict[int, Any] = {}   # ticket → AdaptiveSLManager
        self._adaptive_tp_managers: Dict[int, Any] = {}  # ticket → AdaptiveTPManager

        if _ADAPTIVE_SL_AVAILABLE:
            self.ADAPTIVE_SL_CONFIG = AdaptiveSLConfig(
                # ── Expansión (anti-sweep) ──────────────────────────────────
                expansion_enabled=True,
                # FIX v13.0 (BUG-1): expansion_pts bajado de 50 → 25.
                # Con expansion_pts=50 y hard_sl_margin_pts=50, _compute_expanded_sl
                # exige un mínimo de 100 pts entre virtual_sl y hard_sl para poder
                # expandir. El análisis del log 09/03/2026 muestra que el margen
                # disponible al abrir posiciones es mediana=99 pts (p25=74, max=199),
                # por lo que el 100% de los 8 cierres AdaptiveSL fueron bloqueados.
                # Con 25 pts, el mínimo requerido baja a 75 pts → 2 de esos 8
                # cierres habrían podido expandir (márgenes de 83 y 93 pts).
                # La expansión sigue siendo 2.5 USD (XAUUSD): suficiente anti-sweep.
                expansion_pts=25,             # 2.5 USD de margen extra (XAUUSD M1)
                expansion_max_bars=3,         # máximo 3 velas M1 en modo expandido
                expansion_max_seconds=210.0,  # red de seguridad: 3.5 min
                # FIX v11.0 (BUG-2): expansion_min_signals=0 → el AdaptiveSL cierra
                # inmediatamente al tocar el vSL sin esperar señales de reversión.
                # Con min_signals=2 (v10) el manager retrasaba el cierre esperando
                # confirmación que nunca llegaba (todos los indicadores en False),
                # generando un overshoot medio de +55 pts y máximo de +473 pts.
                # La expansión (anti-sweep) sigue funcionando correctamente: si hay
                # señales de reversión reales, expansion_enabled=True las aprovecha.
                # Sin señales → cierre inmediato al tocar el SL.
                expansion_min_signals=0,
                # FIX v12.0 (BUG-2): umbral bajado de 0.45 → 0.35.
                # El modelo calibrado produce proba_long/short en rango 0.23–0.41;
                # con 0.45 proba_ok era estructuralmente inalcanzable (nunca True).
                # 0.35 corresponde aproximadamente al percentil 75 del modelo,
                # permitiendo que proba_ok aporte señal real en ~25% de los ticks.
                expansion_proba_threshold=0.35,
                expansion_rsi_oversold=35.0,
                expansion_rsi_overbought=65.0,
                expansion_indecision_body_ratio=0.35,
                # ── Compresión proactiva ────────────────────────────────────
                compression_enabled=True,
                compression_proba_threshold=0.60,
                compression_offset_pts=30,    # 3.0 USD de offset desde precio
                compression_min_hold_bars=3,
                compression_cooldown_bars=5,
                compression_macd_hist_threshold=0.0,
                compression_volume_factor=1.3,
                # ── Seguridad ───────────────────────────────────────────────
                # FIX v19.0 (FIX-3): hard_sl_margin_pts bajado de 50 → 20.
                # Problema detectado en log 12/03/2026: la expansión anti-sweep
                # (expansion_blocked=hard_sl_margin_insuficiente) nunca se activaba
                # porque S2 coloca el hard SL muy cerca del vSL — margen real de
                # 0 a 42 pts en todos los tickets de la sesión (mediana ~15 pts).
                # Con hard_sl_margin_pts=50, el threshold mínimo era 10+50=60 pts,
                # nunca alcanzado. Con 20 pts el threshold baja a 30 pts → se activa
                # con márgenes ≥30. En log 12/03: 4 de 5 tickets bloqueados habrían
                # podido expandir. Seguridad residual de 2.0 USD es suficiente para
                # XAUUSD M1 scalping como colchón entre vSL expandido y hard SL.
                # ACCIÓN PARALELA RECOMENDADA EN S2: aumentar separación entre vSL
                # y hard SL al abrir (objetivo ≥50 pts). Este fix es segunda línea
                # de defensa; la primera es que S2 envíe márgenes más amplios.
                hard_sl_margin_pts=20,        # 2.0 USD mínimo hasta el hard SL (era 50)
                pts_to_price=0.01,            # XAUUSD: 1 punto broker = 0.01 precio
            )
            self.ADAPTIVE_TP_CONFIG = AdaptiveTPConfig(
                # ── Compresión del TP (capturar antes del rebote) ───────────
                compression_enabled=True,
                tp_proximity_zone_pts=30,     # evaluar señales a 3.0 USD del TP
                compression_min_signals=2,
                compression_proba_threshold=0.45,
                compression_rsi_overbought=70.0,
                compression_rsi_oversold=30.0,
                compression_indecision_body_ratio=0.35,
                compression_volume_exhaustion_factor=0.80,
                compression_offset_pts=15,    # TP comprimido a 1.5 USD del precio
                compression_min_hold_bars=3,
                compression_cooldown_bars=3,
                # ── Extensión del TP (dejar correr) ─────────────────────────
                extension_enabled=True,
                extension_pts=50,             # extender 5.0 USD más allá del TP
                extension_min_signals=2,
                extension_proba_threshold=0.60,
                extension_rsi_hot_buy=55.0,
                extension_rsi_hot_sell=45.0,
                extension_volume_factor=1.20,
                extension_sl_trail_pts=30,    # SL post-extensión a 3.0 USD del precio
                hard_tp_max=0.0,              # sin límite absoluto de TP
                pts_to_price=0.01,
            )
            self._send({"event": "ADAPTIVE_SL_ENABLED", "config": {
                "expansion_pts": self.ADAPTIVE_SL_CONFIG.expansion_pts,
                "expansion_max_bars": self.ADAPTIVE_SL_CONFIG.expansion_max_bars,
                "expansion_min_signals": self.ADAPTIVE_SL_CONFIG.expansion_min_signals,
                "compression_offset_pts": self.ADAPTIVE_SL_CONFIG.compression_offset_pts,
                "hard_sl_margin_pts": self.ADAPTIVE_SL_CONFIG.hard_sl_margin_pts,
            }})
            self._send({"event": "ADAPTIVE_TP_ENABLED", "config": {
                "tp_proximity_zone_pts": self.ADAPTIVE_TP_CONFIG.tp_proximity_zone_pts,
                "compression_min_signals": self.ADAPTIVE_TP_CONFIG.compression_min_signals,
                "compression_offset_pts": self.ADAPTIVE_TP_CONFIG.compression_offset_pts,
                "extension_pts": self.ADAPTIVE_TP_CONFIG.extension_pts,
                "extension_min_signals": self.ADAPTIVE_TP_CONFIG.extension_min_signals,
                "extension_sl_trail_pts": self.ADAPTIVE_TP_CONFIG.extension_sl_trail_pts,
            }})
        else:
            self.ADAPTIVE_SL_CONFIG = None
            self.ADAPTIVE_TP_CONFIG = None
            self._send({"event": "ADAPTIVE_SL_DISABLED",
                        "reason": "adaptive_sl_manager.py / adaptive_tp_manager.py no encontrados en el path"})

    def _get_state_overrides(self, tr: "TrackedPos") -> dict:
        """
        Devuelve el dict de overrides aplicable al estado actual de la posición.

        Busca en STATE_OVERRIDES el grupo (frozenset) que contiene tr.ind_state.
        Si el estado es None o no está en ningún grupo, devuelve {} (sin overrides).
        El resultado se evalúa en cada ciclo del monitor: si el estado cambia
        entre ciclos, los overrides se actualizan automáticamente.

        Uso:
            overrides = self._get_state_overrides(tr)
            extension_ok = overrides.get('extension_enabled', True)
            tp_confirms  = overrides.get('tp_confirm_count',
                               tr.risk.close_confirmation.tp_confirm_count)
        """
        if not self.STATE_OVERRIDES:
            return {}
        current_state = getattr(tr, 'ind_state', None) or getattr(tr, 'ind_regime', None)
        if not current_state:
            return {}
        for state_group, overrides in self.STATE_OVERRIDES.items():
            if current_state in state_group:
                return overrides
        return {}

    def _effective_tp_confirm_count(self, tr: "TrackedPos") -> int:
        """
        Devuelve el tp_confirm_count efectivo para la posición, aplicando el
        override de STATE_OVERRIDES si el estado actual lo requiere.

        Solo puede AUMENTAR el valor base del perfil (nunca reducirlo), para
        no relajar protecciones configuradas explícitamente.

        v23.0: permite confirmar el VTP con más ciclos en estados ruidosos
        (TRANSITION_*, RANGE) sin cambiar el perfil de riesgo global.
        """
        base = tr.risk.close_confirmation.tp_confirm_count
        overrides = self._get_state_overrides(tr)
        override_val = overrides.get('tp_confirm_count')
        if override_val is None:
            return base
        return max(base, int(override_val))

    def _resolve_log_path(self) -> Path:
        """Devuelve la ruta de log correspondiente a hoy (rotación automática).

        v26.0: el directorio de logs se resuelve relativo al directorio del script
        (_S3_DIR / 'logs') en lugar de Path('../logs') relativo al CWD.
        """
        if self._log_custom:
            return Path(self._log_custom)
        log_dir = _S3_DIR / 'logs'
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / f"s3_events_{datetime.now().strftime('%Y%m%d')}.jsonl"

    def _send(self, msg: Dict[str, Any]) -> None:
        """Envía evento via ZMQ y lo persiste en fichero de log (JSONL)"""
        msg.setdefault("service", S3_SCRIPT_NAME)
        msg.setdefault("version", S3_VERSION)
        msg["ts"] = now()
        try:
            self.events_socket.send_string(json_dump(msg))
        except zmq.Again:
            pass
        except Exception:
            pass

        # ── Persistencia en fichero con rotación automática ─────────────────
        try:
            line = json.dumps(msg, cls=_LogEncoder, ensure_ascii=False)
            with self._log_lock:
                # Recalcular ruta en cada escritura: si cambia el día, rotar
                current_path = self._resolve_log_path()
                if current_path != self._log_path:
                    print(f"[S3Service] Rotando log → {current_path.resolve()}")
                    self._log_path = current_path
                with self._log_path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
        except Exception as exc:
            # No interrumpir el servicio si falla la escritura
            print(f"[S3Service] Error escribiendo log: {exc}")

    def _parse_risk_cfg(self, risk_dict: Dict[str, Any]) -> RiskCfg:
        """
        Parsea configuración de riesgo desde dict

        NUEVO v3.0: Si risk_dict está vacío o incompleto, usa perfil por defecto
        También soporta usar perfiles predefinidos con: {"profile": "scalping"}
        """

        # Si viene un perfil, usarlo como base y aplicar encima los campos extra enviados
        if "profile" in risk_dict:
            profile_name = risk_dict["profile"]
            overrides = {k: v for k, v in risk_dict.items() if k != "profile"}
            risk_dict = get_risk_profile(profile_name)
            # FIX v12.0 (BUG-3): partial_close es un subdict — fusionar sus campos
            # sobre los del perfil base en lugar de reemplazar el subdict completo.
            # Antes: risk_dict.update(overrides) reemplazaba partial_close completo,
            # lo que funcionaba si el caller enviaba todos los campos. Si el caller
            # solo enviaba trigger_profit_pct, los demás campos (enabled, fraction)
            # se perdían. Ahora se fusiona campo a campo.
            pc_override = overrides.pop("partial_close", None)
            risk_dict.update(overrides)   # los campos escalares sobreescriben el perfil
            if pc_override and isinstance(pc_override, dict):
                # Merge selectivo: solo sobreescribir los campos que el caller envió
                risk_dict.setdefault("partial_close", {}).update(pc_override)
            risk_dict["profile"] = profile_name  # preservar el nombre correcto
            self._send({
                "event": "USING_RISK_PROFILE",
                "profile": profile_name,
                "config": risk_dict
            })

        # Si risk_dict está vacío, usar perfil default
        if not risk_dict:
            risk_dict = get_risk_profile("default")
            self._send({
                "event": "USING_DEFAULT_RISK_PROFILE",
                "config": risk_dict
            })

        # Aplicar defaults para campos faltantes
        defaults = get_risk_profile("default")
        for key in ["emergency_points", "be_trigger_points", "be_offset_points",
                    "trail_points", "trail_step_points", "max_hold_seconds",
                    'min_hold_seconds', 'max_spread_points']:
            if key not in risk_dict:
                risk_dict[key] = defaults[key]

        # Partial close
        pc_dict = risk_dict.get("partial_close", {})
        pc = PartialCloseCfg(
            enabled=pc_dict.get("enabled", False),
            trigger_profit_pct=float(pc_dict.get("trigger_profit_pct", 0)),
            close_fraction=float(pc_dict.get("close_fraction", 0)),
            basis=pc_dict.get("basis", "R"),
            runner_tight_trail_pts=int(pc_dict.get("runner_tight_trail_pts", 0)),  # v28.0
        )

        # Close confirmation
        cc_dict = risk_dict.get("close_confirmation", {})
        cc = CloseConfirmationConfig(
            exit_mode=cc_dict.get("exit_mode", "hard"),
            close_confirm_bars=int(cc_dict.get("close_confirm_bars", 1)),
            tp_mode=cc_dict.get("tp_mode", "touch"),
            tp_confirm_count=int(cc_dict.get("tp_confirm_count", 0)),
            sl_mode=cc_dict.get("sl_mode", "touch"),
            sl_confirm_count=int(cc_dict.get("sl_confirm_count", 0)),
            use_close_price=cc_dict.get("use_close_price", False),
            min_bars_beyond=int(cc_dict.get("min_bars_beyond", 0))
        )

        return RiskCfg(
            profile=str(risk_dict.get('profile', 'default')),
            emergency_points=int(risk_dict.get("emergency_points", 0)),
            be_trigger_points=int(risk_dict.get("be_trigger_points", 0)),
            be_offset_points=int(risk_dict.get("be_offset_points", 0)),
            trail_points=int(risk_dict.get("trail_points", 0)),
            trail_step_points=int(risk_dict.get("trail_step_points", 0)),
            max_hold_seconds=int(risk_dict.get("max_hold_seconds", 0)),
            min_hold_seconds=int(risk_dict.get('min_hold_seconds', 0)),
            max_spread_points=int(risk_dict.get('max_spread_points', 0)),
            partial_close=pc,
            close_confirmation=cc
        )


    # ========================================================================
    # RECUPERACIÓN AL ARRANQUE
    # ========================================================================

    def _recover_open_positions(self, magic: int = 0) -> None:
        """
        Al arrancar, detecta posiciones ya abiertas en MT5 y las reincorpora
        al tracking con configuración de riesgo por defecto.

        Se llama desde run() antes de lanzar los threads, por lo que no hay
        condiciones de carrera con el monitor.

        Args:
            magic: Si > 0, solo recupera posiciones con ese magic number.
                   Si == 0, recupera todas las posiciones abiertas.
        """
        try:
            open_positions = self.bridge.positions_get()
            if not open_positions:
                return

            recovered = 0
            for p in open_positions:
                ticket = int(p.ticket)

                # Filtrar por magic si se especifica
                if magic > 0 and int(getattr(p, "magic", 0)) != magic:
                    continue

                # Ya estaba en tracking (no debería pasar en arranque limpio)
                if ticket in self._tracked:
                    continue

                symbol  = str(p.symbol)
                side    = "BUY" if int(p.type) == 0 else "SELL"
                volume  = float(p.volume)
                entry_price = float(p.price_open)
                pos_magic   = int(getattr(p, "magic", 0))
                point   = self.bridge.point(symbol)

                # SL del broker como punto de partida para el virtual SL
                broker_sl = float(getattr(p, "sl", 0.0) or 0.0)
                virtual_sl_price = broker_sl  # mejor que nada; se actualizará con trailing

                virtual_sl_points = 0
                if broker_sl > 0 and point > 0:
                    virtual_sl_points = int(abs(entry_price - broker_sl) / point)

                # TP del broker si existe
                virtual_tp = float(getattr(p, "tp", 0.0) or 0.0)

                # Perfil de riesgo por defecto — no tenemos el original
                # Intentar inferir el perfil del comment (formato "strategy;profile")
                comment = str(getattr(p, "comment", "") or "")
                inferred_profile = "default"
                if ";" in comment:
                    parts = comment.split(";")
                    candidate = parts[-1].strip().lower()
                    # FIX v10.1: tolerar abreviaciones habituales además del
                    # nombre completo (p.ej. "scalp" en lugar de "scalping",
                    # "agg" en lugar de "aggressive", "cons" por "conservative").
                    _PROFILE_ALIASES = {
                        "scalping": "scalping", "scalp": "scalping",
                        "swing": "swing",
                        "conservative": "conservative", "cons": "conservative",
                        "aggressive": "aggressive", "agg": "aggressive",
                        "default": "default",
                    }
                    resolved = _PROFILE_ALIASES.get(candidate)
                    if resolved:
                        inferred_profile = resolved

                risk_cfg = self._parse_risk_cfg({"profile": inferred_profile})

                tr = TrackedPos(
                    ticket=ticket,
                    symbol=symbol,
                    side=side,
                    volume=volume,
                    entry_price=entry_price,
                    point=point,
                    magic=pos_magic,
                    opened_ts=now(),          # no conocemos el ts real de apertura
                    risk=risk_cfg,
                    broker_emergency_sl=broker_sl,
                    emergency_sl_points=virtual_sl_points,
                    virtual_sl_price=virtual_sl_price,
                    virtual_sl_points=virtual_sl_points,
                    virtual_tp=virtual_tp,
                    max_favorable_price=entry_price,
                )

                self._tracked[ticket] = tr
                self._known_open.add(ticket)
                self._ticket_symbols[ticket] = symbol  # FIX v17.0 (FIX-1)
                recovered += 1

                # ── Crear AdaptiveSLManager para posición recuperada (v8.0) ─
                self._create_adaptive_manager(tr)

                self._send({
                    "event": "POSITION_RECOVERED",
                    "ticket": ticket,
                    "symbol": symbol,
                    "side": side,
                    "volume": volume,
                    "entry_price": entry_price,
                    "virtual_sl_price": virtual_sl_price,
                    "virtual_sl_points": virtual_sl_points,
                    "virtual_tp": virtual_tp,
                    "inferred_profile": inferred_profile,
                    "broker_sl": broker_sl,
                    "comment": comment,
                    "warning": "Posicion recuperada tras reinicio. virtual_sl/tp basados en SL broker."
                })

            if recovered > 0:
                self._send({
                    "event": "RECOVERY_COMPLETE",
                    "recovered": recovered,
                    "tracked": len(self._tracked)
                })

        except Exception as e:
            self._send({
                "event": "RECOVERY_ERROR",
                "error": f"{e}",
                "traceback": traceback.format_exc()[-2000:]
            })

    def run(self, recovery_magic: int = 0):
        """
        Ejecuta el servicio.

        Args:
            recovery_magic: Magic number para filtrar posiciones a recuperar al arranque.
                            0 = recuperar todas las posiciones abiertas.
        """
        import threading

        # Recuperar posiciones huerfanas ANTES de lanzar threads
        self._recover_open_positions(magic=recovery_magic)

        # Thread de precio (20Hz, dedicado exclusivamente a bid/ask en memoria)
        price_feed_thread = threading.Thread(target=self._price_feed_loop, daemon=True)
        price_feed_thread.start()

        # Thread de ejecución de comandos
        executor_thread = threading.Thread(target=self._executor_loop, daemon=True)
        executor_thread.start()

        # Thread de monitorización
        monitor_thread = threading.Thread(target=self._monitor_loop, daemon=True)
        monitor_thread.start()

        # Thread de heartbeat
        heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        heartbeat_thread.start()

        self._send({"event": "SERVICE_RUNNING"})

        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            self._send({"event": "SERVICE_STOPPING"})
            if hasattr(self, '_stop'):
                self._stop.set()

    def _executor_loop(self) -> None:
        """Loop que ejecuta comandos recibidos"""
        while True:
            try:
                msg_str = self.orders_socket.recv_string(zmq.NOBLOCK)
                cmd = json_load(msg_str)
                self._handle_command(cmd)
            except zmq.Again:
                time.sleep(0.1)
            except Exception as e:
                self._send({"event": "EXECUTOR_ERROR", "error": f"{e}", "traceback": traceback.format_exc()[-2000:]})
                time.sleep(0.1)

    def _handle_command(self, cmd: Dict[str, Any]) -> None:
        """Procesa comandos de entrada"""
        action = norm_action(cmd.get("action"))

        if action == "OPEN":
            self._handle_open(cmd)
        elif action == "CLOSE":
            self._handle_close(cmd)
        elif action == "MODIFY":
            self._handle_modify(cmd)
        elif action == "PING":
            self._send({"event": "PONG", "request_id": cmd.get("request_id")})
        else:
            self._send({"event": "UNKNOWN_ACTION", "action": action, "cmd": cmd})

    def _handle_open(self, cmd: Dict[str, Any]) -> None:
        """
        Abre posición con sistema DUAL de SL y confirmación
        """
        try:
            symbol = cmd["symbol"]
            side = safe_side(cmd["side"])
            volume = float(cmd["volume"])
            magic = int(cmd.get("magic", 0))

            # =====================================================================
            # GUARD ANTI-DUPLICADO TEMPORAL (v27.0) — primera línea de defensa
            # =====================================================================
            # Causa raíz de los tickets duplicados (confirmado 16-17/03/2026):
            # S2 envía 1 ORDER_REQUEST por barra (OPEN_GUARD en S2 correcto), pero
            # el buffer ZMQ PUSH acumula mensajes de barras anteriores que S3 no
            # ha procesado todavía. En el ciclo siguiente S3 procesa AMBOS — la señal
            # antigua y la nueva — llamando _handle_open dos veces y abriendo 2
            # posiciones idénticas con Δ=39-51ms entre sí.
            # Resultado sesión 16-17/03/2026: 76 señales → 161 aperturas (2.1×),
            # 59 pares duplicados, WR 6.2% → pérdidas masivas.
            #
            # Fix: cooldown temporal — si se procesó un OPEN hace menos de
            # OPEN_DEDUP_SECS (5s), rechazar el siguiente con OPEN_REJECTED_DUPLICATE.
            # La frecuencia de señales legítimas es ~1/minuto (barras M1). Cualquier
            # segundo OPEN dentro de 5s es un duplicado del buffer ZMQ, nunca una
            # señal genuinamente nueva. No depende del número de posiciones abiertas:
            # funciona igualmente cuando hay 0, 1 o 2 posiciones.
            _now_open = now()
            _elapsed  = _now_open - self._last_open_sent_ts
            if _elapsed < self.OPEN_DEDUP_SECS:
                self._send({
                    "event": "OPEN_REJECTED_DUPLICATE",
                    "symbol": symbol,
                    "side": side,
                    "elapsed_ms": round(_elapsed * 1000),
                    "dedup_secs": self.OPEN_DEDUP_SECS,
                    "last_open_ts": self._last_open_sent_ts,
                    "reason": (
                        f"OPEN recibido {_elapsed*1000:.0f}ms después del anterior "
                        f"(< {self.OPEN_DEDUP_SECS}s). Duplicado descartado — "
                        f"probable ORDER_REQUEST retrasado en buffer ZMQ."
                    )
                })
                return
            # Registrar este OPEN — lo actualizaremos de nuevo justo antes de
            # enviar la orden al broker (post-validaciones) para no bloquear
            # si alguna validación previa lo rechaza (spread, etc.).
            # =====================================================================

            # =====================================================================
            # GUARD SENT_TS STALE (v28.3) — segunda línea de defensa contra buffer
            # =====================================================================
            # El buffer ZMQ PUSH acumula ORDER_REQUESTs de sesiones anteriores
            # indefinidamente. El DEDUP de 5s solo rechaza duplicados del mismo
            # par (mismo lado, mismo entry); no puede rechazar mensajes viejos
            # de barras distintas. Confirmado 18/03/2026: 35 mensajes del buffer
            # con RR=2.5 (config antigua) procesados durante 35 minutos mezclados
            # con las señales reales de v45.
            # Fix: S2 incluye 'sent_ts' en cada ORDER_REQUEST con time.time() del
            # momento de envío. S3 rechaza cualquier OPEN con sent_ts > STALE_SECS
            # segundos de antigüedad. Los mensajes del buffer son siempre más viejos
            # que cualquier señal nueva — se invalidan solos al arrancar S3.
            ORDER_STALE_SECS = 30  # mensajes > 30s son del buffer anterior
            _sent_ts = float(cmd.get('sent_ts', 0) or 0)
            if _sent_ts > 0:
                _order_age = now() - _sent_ts
                if _order_age > ORDER_STALE_SECS:
                    self._send({
                        'event': 'OPEN_REJECTED_STALE',
                        'symbol': symbol,
                        'side': side,
                        'sent_ts': _sent_ts,
                        'age_secs': round(_order_age, 1),
                        'stale_secs': ORDER_STALE_SECS,
                        'reason': (
                            f'ORDER_REQUEST tiene {_order_age:.0f}s de antigüedad '
                            f'(> {ORDER_STALE_SECS}s) — descartado como mensaje '
                            f'del buffer ZMQ de sesión anterior.'
                        )
                    })
                    return
            # =====================================================================

            # Configuración de riesgo
            risk_cfg = self._parse_risk_cfg(cmd.get("risk", {}))

            # Extraer metadata
            metadata = cmd.get("metadata", {})

            # =====================================================================
            # FILTRO DE SPREAD (v4.0)
            # =====================================================================
            if risk_cfg.max_spread_points > 0:
                ok_spread, bid_spread, ask_spread = self.bridge.tick_bid_ask(symbol)
                if ok_spread:
                    point_spread = self.bridge.point(symbol)
                    current_spread = (ask_spread - bid_spread) / point_spread
                    if current_spread > risk_cfg.max_spread_points:
                        self._send({
                            'event': 'OPEN_REJECTED_SPREAD',
                            'symbol': symbol,
                            'side': side,
                            'spread_pts': round(current_spread, 1),
                            'max_allowed': risk_cfg.max_spread_points,
                            'profile': risk_cfg.profile
                        })

                        return

            # =====================================================================
            # SISTEMA DUAL DE SL
            # =====================================================================

            # 1. Emergency SL (paracaídas)
            # ── FIX v20.0 (FIX-3): usar emergency_sl_price de S2 si viene en metadata ──
            # Antes S3 ignoraba completamente el emergency_sl_price calculado por S2
            # (que a partir de S2 v28 está a exactamente 50pts más allá del vSL) y
            # recalculaba desde bid/ask + emergency_points*point en el instante de apertura.
            # Esto deshacía el fix de S2 v28 (HARD_SL_MARGIN_PTS=50): S3 enviaba al broker
            # un hard SL propio, diferente al de S2, que podía quedar entre entry y vSL
            # por la misma geometría que corregimos ayer.
            # Fix: si metadata['emergency_sl_price'] > 0, usar ese precio directamente.
            # S3 solo recalcula si no viene precio de S2 (compatibilidad retroactiva).
            emergency_sl = 0.0
            emergency_sl_points = int(risk_cfg.emergency_points or 0)

            _s2_emergency_sl_price = float(metadata.get("emergency_sl_price", 0.0))
            if _s2_emergency_sl_price > 0:
                # S2 v28+: usar el precio ya calculado (50pts más allá del vSL)
                emergency_sl = _s2_emergency_sl_price
                # emergency_sl_points se preserva del risk (para auditoría/logging)
            elif emergency_sl_points > 0:
                # Fallback para S2 < v28 o si no viene el campo: recalcular desde bid/ask
                ok, bid, ask = self.bridge.tick_bid_ask(symbol)
                if ok:
                    point = self.bridge.point(symbol)
                    if side == "BUY":
                        emergency_sl = ask - (emergency_sl_points * point)
                    else:
                        emergency_sl = bid + (emergency_sl_points * point)

            # 2. Virtual SL (señal)
            virtual_sl_price = float(cmd.get("virtual_sl", 0.0))
            if virtual_sl_price == 0.0:
                virtual_sl_price = float(metadata.get("virtual_sl_price", 0.0))

            virtual_sl_points = int(metadata.get("virtual_sl_points", 0))

            # Fallback
            if virtual_sl_points == 0:
                virtual_sl_points = emergency_sl_points
                if virtual_sl_price == 0.0:
                    virtual_sl_price = emergency_sl


            # =====================================================================
            # ABRIR POSICIÓN
            # =====================================================================

            profile1 = cmd.get('comment', '')
            profile2 = risk_cfg.profile
            # v27.0: registrar timestamp ANTES de enviar al broker.
            # Si el broker falla (result.ok=False), el cooldown sigue activo
            # para evitar reintentos inmediatos del mismo duplicado en cola.
            self._last_open_sent_ts = now()
            result = self.bridge.send_market_order(
                symbol=symbol,
                side=side,
                volume=volume,
                magic=magic,
                comment=f'{profile1};{profile2}',
                deviation=20,
                sl=emergency_sl if emergency_sl > 0 else 0.0,
                tp=0.0
            )

            if result.get("ok"):
                ticket = result.get("order")
                entry_price = result.get("price", 0.0)
                point = self.bridge.point(symbol)

                # =================================================================
                # v22.0: VALIDACIÓN POST-FILL — gap entre entry y precio actual
                # =================================================================
                # Si el precio de mercado en este instante está a más de
                # POST_FILL_GAP_MAX_MULT × vSL_pts del entry reportado, la geometría
                # del trade (vSL, hardSL) es incoherente con el precio real desde
                # el primer milisegundo → abortar y cerrar antes de registrar en _tracked.
                # Confirmado 13/03/2026: ticket 317706824 SELL entry=5013.99 pero
                # precio real=5045.69 → gap=3170pt=14.2×vSL → ADAPTIVE_HARD_SL en <1s.
                _pfg_ok, _pfg_bid, _pfg_ask = self.bridge.tick_bid_ask(symbol)
                if _pfg_ok and entry_price > 0 and virtual_sl_points > 0 and point > 0:
                    _pfg_current = _pfg_bid if side == "BUY" else _pfg_ask
                    _pfg_gap_pts = abs(_pfg_current - entry_price) / point
                    _pfg_threshold = self.POST_FILL_GAP_MAX_MULT * virtual_sl_points
                    if _pfg_gap_pts > _pfg_threshold:
                        # Cerrar inmediatamente la posición recién abierta
                        _pfg_close = self.bridge.close_position_market(
                            ticket=ticket,
                            symbol=symbol,
                            side=side,
                            volume=volume,
                            magic=magic,
                            deviation=20,
                            reason="POST_FILL_GAP"
                        )
                        self._send({
                            "event": "POST_FILL_GAP_ABORT",
                            "ticket": ticket,
                            "symbol": symbol,
                            "side": side,
                            "entry_price": entry_price,
                            "current_price": _pfg_current,
                            "gap_pts": round(_pfg_gap_pts),
                            "threshold_pts": round(_pfg_threshold),
                            "virtual_sl_points": virtual_sl_points,
                            "mult": self.POST_FILL_GAP_MAX_MULT,
                            "close_result": _pfg_close,
                        })
                        return  # no registrar en _tracked
                # =================================================================

                virtual_tp_raw = cmd.get("virtual_tp", 0.0)
                virtual_sl_raw = cmd.get("virtual_sl", 0.0)

                # TP
                if 0 < virtual_tp_raw < 100:
                    self._send({
                        "event": "WARNING",
                        "ticket": ticket,
                        "message": f"virtual_tp parece ser PUNTOS ({virtual_tp_raw}) en lugar de PRECIO",
                        "expected": f"{entry_price + virtual_tp_raw * point:.2f}",
                        "received": virtual_tp_raw
                    })

                # SL
                if 0 < virtual_sl_raw < 100:
                    self._send({
                        "event": "WARNING",
                        "ticket": ticket,
                        "message": f"virtual_sl parece ser PUNTOS ({virtual_sl_raw}) en lugar de PRECIO",
                        "expected": f"{entry_price - virtual_sl_raw * point:.2f}",
                        "received": virtual_sl_raw
                    })

                # =================================================================
                # CREAR TRACKED POSITION
                # =================================================================

                tr = TrackedPos(
                    ticket=ticket,
                    symbol=symbol,
                    side=side,
                    volume=volume,
                    entry_price=entry_price,
                    point=point,
                    magic=magic,
                    opened_ts=now(),
                    risk=risk_cfg,
                    broker_emergency_sl=emergency_sl,
                    emergency_sl_points=emergency_sl_points,
                    virtual_sl_price=virtual_sl_price,
                    virtual_sl_points=virtual_sl_points,
                    virtual_tp=virtual_tp_raw,
                    max_favorable_price=entry_price
                )

                # Validar y auto-corregir TP/SL (v3.0)
                self._validate_and_correct_tp_sl(tr, entry_price)

                # ── FIX v20.0 (FIX-1): Recalcular risk_cfg con vSL corregido ─
                # _validate_and_correct_tp_sl puede cambiar tr.virtual_sl_points
                # (WRONG_SIDE_SL_CORRECTED o VSL_EXPANDED_MIN_ATR). Si eso ocurre,
                # los campos de risk_cfg basados en virtual_sl_points (be_trigger_points,
                # trail_points, trail_step_points, partial_close.trigger_profit_pct) que
                # S2 envió en el OPEN quedan desincronizados con la geometría real del trade.
                # Ejemplo sesión 13/03/2026: ticket 317115024 vSL corregido de 259pts→7pts;
                # risk_cfg tenía be_trigger_points=130, trail_points=259 — irrelevantes
                # para un vSL de 7pts. El AdaptiveSL entraba en expansión inmediatamente
                # y cerraba por EXPANSION_TIMEOUT en <1s.
                # Fix: si tr.virtual_sl_points cambió tras la corrección, reconstruir
                # risk_cfg propagando los nuevos valores. Solo se recalculan los campos
                # que S2 derivó de virtual_sl_points (los que tienen 'points' en el nombre);
                # los campos fijos del perfil (max_hold_seconds, emergency_points, etc.)
                # se preservan intactos.
                if tr.virtual_sl_points != virtual_sl_points and tr.virtual_sl_points > 0:
                    _corrected_vsl_pts = tr.virtual_sl_points
                    _orig_vsl_pts      = virtual_sl_points  # pre-corrección
                    # Recalcular campos proporcionales: usar la misma ratio que S2 usó
                    # (risk_cfg.campo / virtual_sl_points_original = ratio)
                    # y aplicarla al vSL corregido. Preservar si el campo original era 0.
                    def _rescale(orig_val: int) -> int:
                        if _orig_vsl_pts <= 0 or orig_val <= 0:
                            return orig_val
                        return max(1, int(round(orig_val * _corrected_vsl_pts / _orig_vsl_pts)))
                    risk_cfg.be_trigger_points  = _rescale(risk_cfg.be_trigger_points)
                    risk_cfg.trail_points        = _rescale(risk_cfg.trail_points)
                    risk_cfg.trail_step_points   = _rescale(risk_cfg.trail_step_points)
                    risk_cfg.be_offset_points    = risk_cfg.be_offset_points  # absoluto, no escalar
                    # partial_close.trigger_profit_pct es % de R, se preserva (no depende de pts)
                    tr.risk = risk_cfg  # sincronizar la TrackedPos con el risk_cfg recalculado
                    self._send({
                        "event": "RISK_CFG_RECALCULATED_AFTER_VSL_CORRECTION",
                        "ticket": ticket,
                        "symbol": symbol,
                        "original_vsl_pts": _orig_vsl_pts,
                        "corrected_vsl_pts": _corrected_vsl_pts,
                        "be_trigger_points": risk_cfg.be_trigger_points,
                        "trail_points":       risk_cfg.trail_points,
                        "trail_step_points":  risk_cfg.trail_step_points,
                    })
                # ─────────────────────────────────────────────────────────────

                # ── Guardar indicadores iniciales si vienen con el comando ──
                indicators = cmd.get("indicators", {})
                if indicators:
                    self._update_tracked_indicators(tr, indicators)

                # ── v18.0: guardar contexto inicial si viene con el comando ──
                context = cmd.get("context", {})
                if context:
                    self._update_tracked_context(tr, context)

                self._tracked[ticket] = tr
                self._known_open.add(ticket)
                self._ticket_symbols[ticket] = symbol  # FIX v17.0 (FIX-1)

                # ── Crear AdaptiveSLManager para esta posición (v8.0) ────────
                self._create_adaptive_manager(tr)

                self._send({
                    "event": "POSITION_OPENED",
                    "ticket": ticket,
                    "symbol": symbol,
                    "side": side,
                    "volume": volume,
                    "entry_price": entry_price,
                    "broker_emergency_sl": emergency_sl,
                    "emergency_sl_points": emergency_sl_points,
                    # FIX v20.0: emitir valores POST-corrección (tr.virtual_sl_*)
                    # En v19 se emitían virtual_sl_price/virtual_sl_points con los
                    # valores originales de S2 aunque _validate_and_correct_tp_sl
                    # los hubiera modificado en tr. Esto causaba que el log mostrara
                    # un vSL incorrecto para diagnóstico, y que risk_R (usado externamente)
                    # no reflejara el riesgo real del trade.
                    "virtual_sl_price": tr.virtual_sl_price,
                    "virtual_sl_points": tr.virtual_sl_points,
                    "virtual_tp": tr.virtual_tp,
                    "risk_R": tr.virtual_sl_points,
                    "risk_cfg": risk_cfg.__dict__,
                    # FIX v28.5: regime y score en la apertura para correlación
                    # directa en el log de S3 sin join con signals log.
                    "regime": (getattr(tr, "ind_state", None) or getattr(tr, "ind_regime", None)),
                    "score":  tr.ind_score,
                    "result": result
                })
                # v28.4: debug log para confirmar que S3 publica el POSITION_OPENED.
                # Si aparece aquí pero no en DEBUG_POSITION_OPENED_REGISTERED de S2
                # → race condition ZMQ handshake: S2 SUB no estaba listo al conectar.
                # Solución en ese caso: añadir time.sleep(0.5) tras events_sub.connect() en S2.
                self._send({
                    "event": "DEBUG_POSITION_OPENED_PUBLISHED",
                    "ticket": ticket,
                    "symbol": symbol,
                })
            else:
                self._send({
                    "event": "OPEN_FAILED",
                    "symbol": symbol,
                    "side": side,
                    "result": result
                })

        except Exception as e:
            self._send({
                "event": "OPEN_ERROR",
                "error": f"{e}",
                "traceback": traceback.format_exc()[-2000:]
            })

    def _handle_close(self, cmd: Dict[str, Any]) -> None:
        """Cierra posición"""
        try:
            ticket = int(cmd["ticket"])
            tr = self._tracked.get(ticket)

            if not tr:
                self._send({"event": "CLOSE_FAILED", "ticket": ticket, "reason": "NOT_TRACKED"})
                return

            result = self.bridge.close_position_market(
                ticket=tr.ticket,
                symbol=tr.symbol,
                side=tr.side,
                volume=tr.volume,
                magic=tr.magic,
                deviation=20,
                reason="MANUAL"
            )

            self._tracked.pop(ticket, None)
            self._known_open.discard(ticket)
            self._remove_adaptive_manager(ticket, "MANUAL_CLOSE")

            self._send({
                "event": "POSITION_CLOSED",
                "ticket": ticket,
                "result": result
            })

        except Exception as e:
            self._send({
                "event": "CLOSE_ERROR",
                "error": f"{e}",
                "traceback": traceback.format_exc()[-2000:]
            })

    def _handle_modify(self, cmd: Dict[str, Any]) -> None:
        """Modifica parámetros virtuales"""
        try:
            ticket = int(cmd["ticket"])
            tr = self._tracked.get(ticket)

            if not tr:
                self._send({"event": "MODIFY_FAILED", "ticket": ticket, "reason": "NOT_TRACKED"})
                return

            if "virtual_tp" in cmd:
                tr.virtual_tp = float(cmd["virtual_tp"])
            if "virtual_sl" in cmd:
                tr.virtual_sl_price = float(cmd["virtual_sl"])

            # ── Actualizar indicadores si vienen (v8.0) ──────────────────────
            if "indicators" in cmd:
                self._update_tracked_indicators(tr, cmd["indicators"])

            # ── v18.0: actualizar contexto si viene ──────────────────────────
            if "context" in cmd:
                self._update_tracked_context(tr, cmd["context"])

            _is_keepalive = bool(cmd.get("keepalive"))
            _modified_event = {
                "event":      "POSITION_MODIFIED",
                "ticket":     ticket,
                "virtual_tp": tr.virtual_tp,
                "virtual_sl": tr.virtual_sl_price,
            }
            # FIX v17.0 (FIX-3) + v18.0: propagar keepalive y contexto al evento.
            if _is_keepalive:
                _modified_event["keepalive"] = True
            state_value = getattr(tr, "ind_state", None) or getattr(tr, "ind_regime", None)
            if state_value is not None:
                _modified_event["state"] = state_value
            if tr.ind_spread is not None:
                _modified_event["spread"] = tr.ind_spread
            self._send(_modified_event)

        except Exception as e:
            self._send({
                "event": "MODIFY_ERROR",
                "error": f"{e}",
                "traceback": traceback.format_exc()[-2000:]
            })

    # ========================================================================
    # PRICE FEED (hilo dedicado a precios, v7.0)
    # ========================================================================

    def _price_feed_loop(self) -> None:
        """
        Hilo dedicado exclusivamente a mantener precios bid/ask actualizados
        en self._prices (dict en memoria).

        Ventajas frente al enfoque anterior (tick_bid_ask en cada ciclo del monitor):
        - El monitor lee de memoria: latencia ~0ms en lugar de ~5-20ms por llamada MT5
        - Las llamadas a MT5 están aisladas en un hilo propio, sin bloquear la lógica
        - Frecuencia configurable independientemente del monitor_interval
        """
        while True:
            try:
                # Obtener símbolos activos en este momento
                with threading.Lock():
                    symbols = list({tr.symbol for tr in self._tracked.values()})

                for symbol in symbols:
                    try:
                        ok, bid, ask = self.bridge.tick_bid_ask(symbol)
                        if ok and bid > 0 and ask > 0:
                            with self._prices_lock:
                                self._prices[symbol] = (bid, ask)
                    except Exception:
                        pass

                time.sleep(0.05)   # 20 actualizaciones/segundo por símbolo

            except Exception as e:
                self._send({
                    "event": "PRICE_FEED_ERROR",
                    "error": f"{e}"
                })
                time.sleep(0.1)

    def _get_bid_ask(self, symbol: str):
        """
        Devuelve (ok, bid, ask) desde el cache en memoria del price feed.
        Fallback a llamada directa MT5 si el símbolo aún no está en cache.
        """
        with self._prices_lock:
            prices = self._prices.get(symbol)

        if prices:
            return True, prices[0], prices[1]

        # Fallback: primera vez que se pide este símbolo
        return self.bridge.tick_bid_ask(symbol)

    def _get_positions_cached(self) -> list:
        """
        Devuelve posiciones abiertas desde cache (TTL = 1s).

        positions_get() es una llamada pesada a MT5. No necesita ejecutarse
        5 veces/segundo: los precios se actualizan en el price feed thread,
        y detectar que una posición se cerró con 1s de retraso es aceptable.
        """
        if now() - self._positions_cache_ts >= self._positions_cache_ttl:
            result = self.bridge.positions_get()
            self._positions_cache    = result if result is not None else []
            self._positions_cache_ts = now()
        return self._positions_cache

    # ========================================================================
    # MONITORIZACIÓN
    # ========================================================================

    def _get_last_bar_ohlc(self, symbol: str, timeframe_minutes: int = 1) -> Optional[tuple]:
        """
        Obtiene el OHLC completo de la última vela M1 cerrada vía MT5.
        Devuelve (open, high, low, close) o None si no está disponible.

        FIX v12.0 (BUG-1): reemplaza _get_last_bar_close() que solo devolvía
        el cierre. Con OHLC plano (open=high=low=close=bid) el AdaptiveSL no
        podía calcular body_ratio → indecision_ok siempre False, bloqueando la
        señal de reversión por patrón de vela incluso cuando el mercado mostraba
        doji o vela de indecisión clara.
        """
        if _mt5 is None:
            return None
        try:
            tf_map = {1: 1, 5: 5, 15: 15, 30: 30, 60: 16385}
            tf = tf_map.get(timeframe_minutes, 1)
            # pos=1 => la vela anterior (ya cerrada), pos=0 => vela en curso
            rates = _mt5.copy_rates_from_pos(symbol, tf, 1, 1)
            if rates is not None and len(rates) > 0:
                r = rates[0]
                return (float(r['open']), float(r['high']), float(r['low']), float(r['close']))
        except Exception:
            pass
        return None

    def _get_last_bar_close(self, symbol: str, timeframe_minutes: int = 1) -> Optional[float]:
        """
        Compatibilidad: devuelve solo el cierre de la última vela cerrada.
        Internamente llama a _get_last_bar_ohlc.
        """
        ohlc = self._get_last_bar_ohlc(symbol, timeframe_minutes)
        return ohlc[3] if ohlc else None

    def _monitor_loop(self) -> None:
        """Loop de monitorización de posiciones"""
        while True:
            try:
                time.sleep(self.monitor_interval)

                if not self._tracked:
                    continue

                # ── Posiciones desde cache (TTL 1s) — evita llamadas excesivas ──
                open_positions = self._get_positions_cached()
                if open_positions is None:
                    continue

                open_tickets = {int(p.ticket) for p in open_positions}

                for p in open_positions:
                    ticket = int(p.ticket)
                    tr = self._tracked.get(ticket)

                    if not tr:
                        continue

                    # ── Precios desde memoria (price feed thread) ────────────
                    ok, bid, ask = self._get_bid_ask(tr.symbol)
                    if not ok:
                        continue

                    # Precio de cierre de ultima vela (para sl_mode/tp_mode = "close")
                    last_bar_close = self._get_last_bar_close(tr.symbol)

                    # Actualizar máximo favorable
                    self._update_max_favorable_price(tr, bid, ask)

                    # ── 0. ADAPTIVE SL (v8.0) ──────────────────────────────
                    # Se ejecuta ANTES del sistema de confirmación standard.
                    # FIX v9.0: el AdaptiveSL ya NO cierra cuando la razón es
                    # VIRTUAL_TP — ese cierre lo gestiona el sistema estándar.
                    # Esto elimina el bug de doble cierre del 25/02.
                    if self._run_adaptive_sl(tr, bid, ask):
                        continue

                    # ── 0.5. ADAPTIVE TP (v9.0) ────────────────────────────
                    # Compresión (capturar antes del rebote) y extensión
                    # (dejar correr). Se ejecuta ANTES del sistema estándar.
                    # Si devuelve "closed" la posición ya está cerrada.
                    # Si devuelve "delegated" el TP fue tocado pero sin señales:
                    # el sistema estándar (_apply_exit_mode) cerrará normalmente.
                    tp_action = self._run_adaptive_tp(tr, bid, ask)
                    if tp_action == "closed":
                        continue
                    # "delegated" o "none" → continúa al sistema estándar

                    # 1. MAX HOLD TIME
                    if tr.risk.max_hold_seconds > 0:
                        hold_time = now() - tr.opened_ts
                        if hold_time >= tr.risk.max_hold_seconds:
                            self._close_position_full(tr, bid, ask, "MAX_HOLD_TIME")
                            continue

                    # 2. VERIFICAR TP/SL CON CONFIRMACIÓN
                    should_close, reason = self._apply_exit_mode(tr, bid, ask, close_price=last_bar_close)
                    if should_close:
                        self._close_position_full(tr, bid, ask, reason)
                        continue

                    # 3. PARTIAL CLOSE — debe ejecutarse ANTES del BE para que
                    #    tr.volume esté actualizado cuando se calcule el riesgo residual
                    if not tr.partial_done:
                        if self._check_partial_close(tr, p, bid, ask):
                            continue

                    # 4. BREAK-EVEN (v3.0: con validación mejorada)
                    #    Se evalúa después del partial close para usar el volumen correcto
                    self._handle_break_even(tr, bid, ask)

                    # 4.5. PROFIT LOCKS PROGRESIVOS (v3.0)
                    if tr.be_armed:
                        profit_pts = self._calc_profit_points(tr, bid, ask)
                        self._apply_profit_lock_levels(tr, profit_pts)

                    # 4.8. TIME-BASED PROFIT FLOOR (P1 - v14.0)
                    _hold_secs = now() - tr.opened_ts
                    _profit_pts_now = self._calc_profit_points(tr, bid, ask)

                    # Nivel 1: hold >= 120s Y profit >= 1R -> armar BE como floor
                    if (not tr.be_armed
                            and _hold_secs >= self.TIME_PROFIT_FLOOR_SECS
                            and tr.virtual_sl_points > 0
                            and _profit_pts_now >= tr.virtual_sl_points):
                        _floor_sl = self._calc_be_sl(tr)
                        if self._is_improvement(tr, tr.virtual_sl_price, _floor_sl):
                            _old_sl = tr.virtual_sl_price
                            tr.virtual_sl_price = self._improve_virtual_sl(
                                tr, tr.virtual_sl_price, _floor_sl)
                            tr.be_armed = True
                            tr.be_armed_ts = now()
                            tr.trailing_wait_logged = False
                            tr.trailing_unlocked = False
                            self._send({
                                "event": "TIME_PROFIT_FLOOR_ARMED",
                                "ticket": tr.ticket,
                                "symbol": tr.symbol,
                                "hold_seconds": round(_hold_secs),
                                "profit_points": round(_profit_pts_now, 1),
                                "threshold_1R": tr.virtual_sl_points,
                                "old_virtual_sl": _old_sl,
                                "new_virtual_sl": tr.virtual_sl_price,
                            })

                    # Nivel 2: cierre forzado por tiempo
                    # FIX v15.0 (FIX-1): profit mínimo, extensión por momentum,
                    # evento único (ver historial en v15).
                    # v18.0 (MEJORA-2): inhibición por spread elevado — si el spread
                    # actual supera MAX_SPREAD_FOR_CLOSE_PTS, el cierre se pospone
                    # hasta que el spread baje o hasta SPREAD_INHIBIT_MAX_WAIT_SECS.
                    # FIX v19.0 (FIX-2): evaluar profit MÁXIMO histórico además del
                    # profit actual. El problema: si el precio devuelve parcialmente
                    # la ganancia, _profit_pts_now puede caer por debajo de 0.3R aunque
                    # la posición llegó a tener beneficio real significativo. Con la
                    # condición original (solo _profit_pts_now), el TIME_FORCE_CLOSE
                    # nunca dispara y la orden permanece abierta indefinidamente.
                    # Solución: si el profit máximo alcanzado superó 2× el umbral
                    # (0.6R), el cierre se activa aunque el profit actual sea menor.
                    # Esto captura el escenario real observado: trade que llegó a
                    # +297pts pero el precio retrocedió y el umbral ya no se cumplía.
                    _min_profit_threshold = (
                        self.TIME_FORCE_CLOSE_MIN_PROFIT_R * tr.virtual_sl_points
                        if tr.virtual_sl_points > 0 else 1.0
                    )
                    # Calcular profit máximo histórico desde max_favorable_price
                    if tr.max_favorable_price > 0:
                        if tr.side == "BUY":
                            _peak_profit_pts = (tr.max_favorable_price - tr.entry_price) / tr.point
                        else:
                            _peak_profit_pts = (tr.entry_price - tr.max_favorable_price) / tr.point
                    else:
                        _peak_profit_pts = 0.0
                    # Disparar si: profit actual >= umbral  O  peak >= 2× umbral
                    _profit_qualifies = (
                        _profit_pts_now >= _min_profit_threshold
                        or _peak_profit_pts >= _min_profit_threshold * 2.0
                    )
                    if (_hold_secs >= self.TIME_FORCE_CLOSE_SECS
                            and _profit_qualifies):

                        # ── Extensión única para trades con fuerte momentum ──────
                        _extend_threshold = (
                            self.TIME_FORCE_CLOSE_EXTEND_R * tr.virtual_sl_points
                            if tr.virtual_sl_points > 0 else float('inf')
                        )
                        _effective_timeout = self._effective_time_force_close_secs(tr)
                        if (not tr.time_force_close_extended
                                and _profit_pts_now >= _extend_threshold):
                            tr.time_force_close_extended = True
                            _effective_timeout += self.TIME_FORCE_CLOSE_EXTEND_SECS
                            self._send({
                                "event": "TIME_FORCE_CLOSE_EXTENDED",
                                "ticket": tr.ticket,
                                "symbol": tr.symbol,
                                "hold_seconds": round(_hold_secs),
                                "profit_points": round(_profit_pts_now, 1),
                                "extend_threshold_R": self.TIME_FORCE_CLOSE_EXTEND_R,
                                "new_timeout_secs": _effective_timeout,
                            })

                        # Reevaluar con el timeout efectivo (puede haber crecido)
                        if _hold_secs >= _effective_timeout:
                            # ── v18.0 (MEJORA-2): inhibición por spread ──────────
                            # Si ind_spread > umbral, posponer el cierre para no
                            # pagar un spread anómalo en un cierre rentable.
                            # Lógica de estados via tr.spread_inhibit_since:
                            #   0.0  → sin inhibición previa
                            #   >0.0 → inhibición activa desde ese ts
                            _cur_spread = tr.ind_spread  # None si S2 no lo ha enviado
                            _spread_high = (
                                _cur_spread is not None
                                and _cur_spread > self.MAX_SPREAD_FOR_CLOSE_PTS
                            )
                            if _spread_high:
                                if tr.spread_inhibit_since == 0.0:
                                    # Primera vez que detectamos spread alto en este cierre
                                    tr.spread_inhibit_since = now()
                                    self._send({
                                        "event":   "SPREAD_INHIBIT_ACTIVE",
                                        "ticket":  tr.ticket,
                                        "symbol":  tr.symbol,
                                        "spread":  _cur_spread,
                                        "threshold": self.MAX_SPREAD_FOR_CLOSE_PTS,
                                        "max_wait_secs": self.SPREAD_INHIBIT_MAX_WAIT_SECS,
                                        "profit_points": round(_profit_pts_now, 1),
                                    })
                                    continue  # posponer este ciclo

                                _wait_secs = now() - tr.spread_inhibit_since
                                if _wait_secs < self.SPREAD_INHIBIT_MAX_WAIT_SECS:
                                    continue  # seguir esperando

                                # Timeout de inhibición agotado: cerrar igualmente
                                self._send({
                                    "event":      "SPREAD_INHIBIT_TIMEOUT",
                                    "ticket":     tr.ticket,
                                    "symbol":     tr.symbol,
                                    "spread":     _cur_spread,
                                    "waited_secs": round(_wait_secs),
                                    "profit_points": round(_profit_pts_now, 1),
                                })
                            else:
                                # FIX v19.0 (FIX-1): NO resetear spread_inhibit_since una vez
                                # iniciado. El bug original reseteaba el contador cada vez que
                                # el spread bajaba brevemente, permitiendo que picos alternantes
                                # de spread (sube → baja → sube) reiniciaran el temporizador
                                # indefinidamente. Resultado: el cierre nunca llegaba a ejecutarse
                                # y la posición permanecía abierta mientras el profit se evaporaba.
                                # Ahora: si la inhibición ya fue iniciada (spread_inhibit_since > 0),
                                # el contador sigue corriendo aunque el spread baje temporalmente.
                                # El reset solo ocurre DESPUÉS del cierre (implícito: el ticket
                                # desaparece de _tracked).
                                if tr.spread_inhibit_since == 0.0:
                                    pass  # sin inhibición previa: todo OK, proceder al cierre

                            self._close_position_full(
                                tr, bid, ask, "TIME_FORCE_CLOSE",
                                extra_payload={
                                    "hold_seconds":              round(_hold_secs),
                                    "profit_points":             round(_profit_pts_now, 1),
                                    "peak_profit_points":        round(_peak_profit_pts, 1),
                                    "min_profit_threshold_pts":  round(_min_profit_threshold, 1),
                                    "triggered_by_peak":         _profit_pts_now < _min_profit_threshold,
                                    "spread_at_close":           tr.ind_spread,
                                }
                            )
                            continue

                    # 5. TRAILING STOP
                    if tr.be_armed and tr.risk.trail_points > 0:
                        trailing_unlocked, trailing_gate_meta = self._is_trailing_unlocked(tr, bid, ask)

                        if not trailing_unlocked:
                            if not tr.trailing_wait_logged:
                                tr.trailing_wait_logged = True
                                self._send({
                                    "event": "TRAILING_DELAY_ACTIVE",
                                    "ticket": tr.ticket,
                                    "symbol": tr.symbol,
                                    **trailing_gate_meta,
                                })
                        else:
                            if tr.trailing_wait_logged:
                                self._send({
                                    "event": "TRAILING_DELAY_RELEASED",
                                    "ticket": tr.ticket,
                                    "symbol": tr.symbol,
                                    **trailing_gate_meta,
                                })
                                tr.trailing_wait_logged = False

                            trail_sl, trail_meta = self._calc_trailing_sl(tr, bid, ask)
                            if trail_sl > 0:
                                old_sl = tr.virtual_sl_price
                                tr.virtual_sl_price = self._improve_virtual_sl(tr, tr.virtual_sl_price, trail_sl)

                                if tr.virtual_sl_price != old_sl and tr.virtual_sl_price > 0:
                                    self._send({
                                        "event": "TRAILING_UPDATED",
                                        "ticket": tr.ticket,
                                        "symbol": tr.symbol,
                                        "old_virtual_sl": old_sl,
                                        "new_virtual_sl": tr.virtual_sl_price,
                                        "current_price": bid if tr.side == "BUY" else ask,
                                        **trail_meta,
                                        **trailing_gate_meta,
                                    })

                # Detectar cierres externos
                # FIX v15.0 (FIX-4): el cache de posiciones (TTL=1s) puede devolver
                # un ticket como "ausente" en dos ciclos consecutivos antes de que
                # _known_open se actualice, generando POSITION_CLOSED_EXTERNAL doble.
                # Solución: _known_open se actualiza con open_tickets al final del ciclo,
                # por lo que la diferencia "closed" solo existe en el primer ciclo en que
                # el ticket desaparece. La actualización se hace FUERA del for sobre
                # open_positions (donde estaba antes) y DESPUÉS de procesar todos los
                # cierres, garantizando que cada ticket solo aparece una vez.
                #
                # FIX v20.1 (FIX-4): grace period por ticket para evitar falso
                # DETECTED_CLOSED en apertura. MT5 tarda 100-500ms en registrar
                # una posición nueva en positions_get(). Durante ese intervalo el
                # ticket ya está en _known_open pero no en open_tickets → falso
                # DETECTED_CLOSED → S3 deja la posición sin supervisión.
                # Confirmado 13/03/2026: ticket 317234422 supervisión perdida 3h.
                # Fix: saltar tickets en grace period (opened_ts < OPEN_GRACE_SECS).
                _now = now()
                # v30.0: limpiar marcas antiguas de cierres intencionados y contexto reciente
                self._intentional_close_tickets = {
                    _t: _ts for _t, _ts in self._intentional_close_tickets.items()
                    if (_now - _ts) <= self.INTENTIONAL_CLOSE_GRACE_SECS
                }
                self._recent_closed_context = {
                    _t: _ctx for _t, _ctx in self._recent_closed_context.items()
                    if (_now - float(_ctx.get("closed_ts", _now))) <= 600.0
                }

                closed = self._known_open - open_tickets
                for t in list(closed):
                    # Grace period: si el ticket fue abierto hace menos de
                    # OPEN_GRACE_SECS, MT5 puede aún no haberlo registrado en la
                    # API. Ignorarlo este ciclo — el próximo ciclo del monitor lo
                    # encontrará en open_tickets si sigue abierto, o habrá pasado
                    # el grace y se procesará como cierre externo genuino.
                    tr_grace = self._tracked.get(t)
                    if tr_grace is not None:
                        age = _now - tr_grace.opened_ts
                        if age < self.OPEN_GRACE_SECS:
                            self._send({
                                "event": "OPEN_GRACE_SKIP",
                                "ticket": int(t),
                                "symbol": tr_grace.symbol,
                                "age_ms": round(age * 1000),
                                "grace_secs": self.OPEN_GRACE_SECS,
                                "reason": "ticket_not_yet_visible_in_mt5"
                            })
                            continue  # no tocar _tracked ni _known_open

                    # v30.0: si el cierre fue iniciado por S3, NO emitir external close.
                    intentional_ts = self._intentional_close_tickets.get(t)
                    if intentional_ts is not None and (_now - intentional_ts) <= self.INTENTIONAL_CLOSE_GRACE_SECS:
                        self._known_open.discard(t)
                        self._intentional_close_tickets.pop(t, None)
                        continue

                    tr_closed = self._tracked.pop(t, None)
                    self._known_open.discard(t)      # actualizar aquí, ticket a ticket
                    _ctx = self._recent_closed_context.get(t, {})
                    # FIX v17.0 (FIX-1): usar _ticket_symbols como respaldo cuando
                    # tr_closed=None (ticket ya eliminado por _close_position_full).
                    _sym = (tr_closed.symbol if tr_closed
                            else _ctx.get("symbol")
                            or self._ticket_symbols.get(t))
                    self._ticket_symbols.pop(t, None)  # limpiar respaldo
                    self._remove_adaptive_manager(t, "EXTERNAL_CLOSE")
                    self._send({
                        "event": "POSITION_CLOSED_EXTERNAL",
                        "ticket": int(t),
                        "symbol": _sym,
                        "reason": "DETECTED_CLOSED",
                        # v30.0: enriquecer con cache reciente si ya no existe tr_closed
                        "side":       tr_closed.side if tr_closed else _ctx.get("side"),
                        "open_price": tr_closed.entry_price if tr_closed else _ctx.get("open_price"),
                        "volume":     tr_closed.volume if tr_closed else _ctx.get("volume"),
                        "hold_seconds": round(now() - tr_closed.opened_ts, 1) if tr_closed and tr_closed.opened_ts > 0 else _ctx.get("hold_seconds"),
                        "virtual_sl": tr_closed.virtual_sl_price if tr_closed else _ctx.get("virtual_sl"),
                        "virtual_tp": tr_closed.virtual_tp if tr_closed else _ctx.get("virtual_tp"),
                    })

                # Sincronizar _known_open con el estado real de MT5 para el próximo ciclo.
                # v30.0: excluir tickets que S3 ya ha decidido cerrar y que pueden seguir
                # apareciendo temporalmente por cache/latencia del bridge.
                _open_effective = {t for t in open_tickets if t not in self._intentional_close_tickets}
                self._known_open = (self._known_open & _open_effective) | _open_effective

            except Exception as e:
                self._send({
                    "event": "MONITOR_ERROR",
                    "error": f"{e}",
                    "traceback": traceback.format_exc()[-2000:]
                })
                time.sleep(1)

    # ========================================================================
    # SISTEMA DE CONFIRMACIÓN
    # ========================================================================

    def _check_tp_with_confirmation(
            self,
            tr: TrackedPos,
            bid: float,
            ask: float,
            close_price: Optional[float] = None
    ) -> bool:
        """Verifica TP con sistema de confirmación"""

        if tr.virtual_tp <= 0:
            return False

        # VALIDACIÓN: Detectar si TP es PUNTOS
        if tr.virtual_tp < (tr.entry_price / 10):
            self._send({
                "event": "ERROR",
                "ticket": tr.ticket,
                "message": f"virtual_tp={tr.virtual_tp} parece ser PUNTOS, no PRECIO",
                "entry_price": tr.entry_price,
                "action": "Deshabilitando TP"
            })
            tr.virtual_tp = 0.0
            return False

        cfg = tr.risk.close_confirmation
        price = bid if tr.side == "BUY" else ask

        # Verificar si tocado
        if tr.side == "BUY":
            touched = price >= tr.virtual_tp
        else:
            touched = price <= tr.virtual_tp

        if not touched:
            tr.confirmation.tp_confirm_count = 0
            tr.confirmation.tp_close_history.clear()
            tr.confirmation.tp_touched = False
            return False

        tr.confirmation.tp_touched = True

        # MODO: TOUCH
        if cfg.tp_mode == "touch":
            # v23.0: usar _effective_tp_confirm_count para aplicar override por estado.
            # En estados ruidosos (TRANSITION_*, RANGE), el override puede exigir
            # confirmación aunque el perfil base tenga tp_confirm_count=0 (touch puro).
            _eff_tp_confirms = self._effective_tp_confirm_count(tr)
            if _eff_tp_confirms == 0:
                return True

            tr.confirmation.tp_confirm_count += 1

            if tr.confirmation.tp_confirm_count > _eff_tp_confirms:
                self._send({
                    "event":         "TP_CONFIRMED",
                    "ticket":        tr.ticket,
                    "confirmations": tr.confirmation.tp_confirm_count,
                    "state":         getattr(tr, "ind_state", None),   # v23.0: trazabilidad
                    "override_applied": _eff_tp_confirms != cfg.tp_confirm_count,
                })
                return True

            return False

        # MODO: CLOSE
        elif cfg.tp_mode == "close":
            if close_price is None:
                close_price = price

            if tr.side == "BUY":
                closed_beyond = close_price > tr.virtual_tp
            else:
                closed_beyond = close_price < tr.virtual_tp

            if not closed_beyond:
                return False

            # v23.0: mismo override aplicado al modo close
            _eff_tp_confirms = self._effective_tp_confirm_count(tr)
            if _eff_tp_confirms == 0:
                return True

            tr.confirmation.tp_confirm_count += 1

            if tr.confirmation.tp_confirm_count > _eff_tp_confirms:
                return True

            return False

        return False

    def _check_sl_with_confirmation(
            self,
            tr: TrackedPos,
            bid: float,
            ask: float,
            close_price: Optional[float] = None
    ) -> bool:
        """Verifica SL con sistema de confirmación"""

        if tr.virtual_sl_price <= 0:
            return False

        cfg = tr.risk.close_confirmation
        price = bid if tr.side == "BUY" else ask

        # Verificar si tocado
        if tr.side == "BUY":
            touched = price <= tr.virtual_sl_price
        else:
            touched = price >= tr.virtual_sl_price

        if not touched:
            tr.confirmation.sl_confirm_count = 0
            tr.confirmation.sl_close_history.clear()
            tr.confirmation.sl_touched = False
            return False

        tr.confirmation.sl_touched = True

        # MODO: TOUCH
        if cfg.sl_mode == "touch":
            if cfg.sl_confirm_count == 0:
                return True

            tr.confirmation.sl_confirm_count += 1

            if tr.confirmation.sl_confirm_count > cfg.sl_confirm_count:
                return True

            return False

        # MODO: CLOSE
        elif cfg.sl_mode == "close":
            # Usar close_price real si está disponible; si no, caer a precio tick
            if close_price is None:
                close_price = price

            if tr.side == "BUY":
                closed_beyond = close_price < tr.virtual_sl_price
            else:
                closed_beyond = close_price > tr.virtual_sl_price

            if not closed_beyond:
                return False

            if cfg.sl_confirm_count == 0:
                return True

            tr.confirmation.sl_confirm_count += 1

            if tr.confirmation.sl_confirm_count > cfg.sl_confirm_count:
                return True

            return False

        return False

    def _check_close_confirm_bars(
            self,
            tr: TrackedPos,
            bid: float,
            ask: float,
            close_price: Optional[float] = None
    ) -> tuple:
        """Verifica confirmación por múltiples barras"""

        cfg = tr.risk.close_confirmation

        if cfg.exit_mode != "close_confirm":
            return False, ""

        if close_price is None:
            close_price = bid if tr.side == "BUY" else ask

        current_time = now()

        # Solo actualizar si es nueva vela (>1 min)
        if current_time - tr.confirmation.last_close_ts > 60:
            tr.confirmation.last_close_price = close_price
            tr.confirmation.last_close_ts = current_time

            # Verificar TP
            if tr.virtual_tp > 0:
                if tr.side == "BUY":
                    tp_closed = close_price > tr.virtual_tp
                else:
                    tp_closed = close_price < tr.virtual_tp

                tr.confirmation.tp_close_history.append(tp_closed)

                max_history = cfg.close_confirm_bars + 2
                if len(tr.confirmation.tp_close_history) > max_history:
                    tr.confirmation.tp_close_history = tr.confirmation.tp_close_history[-max_history:]

                if len(tr.confirmation.tp_close_history) >= cfg.close_confirm_bars:
                    last_n = tr.confirmation.tp_close_history[-cfg.close_confirm_bars:]

                    if all(last_n):
                        if cfg.min_bars_beyond > 0:
                            beyond_count = sum(last_n)
                            if beyond_count >= cfg.min_bars_beyond:
                                return True, "TP_MULTI_BAR_CONFIRMED"
                        else:
                            return True, "TP_MULTI_BAR_CONFIRMED"

            # Verificar SL
            if tr.virtual_sl_price > 0:
                if tr.side == "BUY":
                    sl_closed = close_price < tr.virtual_sl_price
                else:
                    sl_closed = close_price > tr.virtual_sl_price

                tr.confirmation.sl_close_history.append(sl_closed)

                if len(tr.confirmation.sl_close_history) > max_history:
                    tr.confirmation.sl_close_history = tr.confirmation.sl_close_history[-max_history:]

                if len(tr.confirmation.sl_close_history) >= cfg.close_confirm_bars:
                    last_n = tr.confirmation.sl_close_history[-cfg.close_confirm_bars:]

                    if all(last_n):
                        return True, "SL_MULTI_BAR_CONFIRMED"

        return False, ""

    def _apply_exit_mode(
            self,
            tr: TrackedPos,
            bid: float,
            ask: float,
            close_price: Optional[float] = None
    ) -> tuple:
        """Aplica el modo de salida y retorna decisión"""

        if tr.risk.min_hold_seconds > 0:
            hold_time = now() - tr.opened_ts
            if hold_time < tr.risk.min_hold_seconds:
                return False, ''

        cfg = tr.risk.close_confirmation

        # MODO: HARD
        if cfg.exit_mode == "hard":
            if self._check_tp_with_confirmation(tr, bid, ask, close_price):
                return True, "VIRTUAL_TP"

            if self._check_sl_with_confirmation(tr, bid, ask, close_price):
                return True, "VIRTUAL_SL"

        # MODO: SOFT
        elif cfg.exit_mode == "soft":
            original_tp = cfg.tp_mode
            original_sl = cfg.sl_mode

            cfg.tp_mode = "close"
            cfg.sl_mode = "close"

            if self._check_tp_with_confirmation(tr, bid, ask, close_price):
                cfg.tp_mode = original_tp
                cfg.sl_mode = original_sl
                return True, "VIRTUAL_TP_SOFT"

            if self._check_sl_with_confirmation(tr, bid, ask, close_price):
                cfg.tp_mode = original_tp
                cfg.sl_mode = original_sl
                return True, "VIRTUAL_SL_SOFT"

            cfg.tp_mode = original_tp
            cfg.sl_mode = original_sl

        # MODO: CLOSE_CONFIRM
        elif cfg.exit_mode == "close_confirm":
            should_close, reason = self._check_close_confirm_bars(tr, bid, ask, close_price)
            if should_close:
                return True, reason

        return False, ""

    # ========================================================================
    # UTILIDADES
    # ========================================================================

    def _update_max_favorable_price(self, tr: TrackedPos, bid: float, ask: float) -> None:
        """Actualiza precio máximo favorable para trailing"""
        if tr.side == "BUY":
            tr.max_favorable_price = max(tr.max_favorable_price, bid)
        else:
            if tr.max_favorable_price == 0:
                tr.max_favorable_price = ask
            else:
                tr.max_favorable_price = min(tr.max_favorable_price, ask)

    def _calc_profit_points(self, tr: TrackedPos, bid: float, ask: float) -> float:
        """Calcula ganancia en puntos"""
        if tr.side == "BUY":
            return (bid - tr.entry_price) / tr.point
        else:
            return (tr.entry_price - ask) / tr.point

    def _calc_be_sl(self, tr: TrackedPos) -> float:
        """Calcula SL de break-even"""
        offset_price = tr.risk.be_offset_points * tr.point
        if tr.side == "BUY":
            return tr.entry_price + offset_price
        else:
            return tr.entry_price - offset_price

    def _is_improvement(self, tr: TrackedPos, current_sl: float, new_sl: float) -> bool:
        """
        Verifica si new_sl es mejor que current_sl

        Returns:
            True si new_sl mejora current_sl, False otherwise
        """
        if current_sl == 0:
            return True

        if tr.side == "BUY":
            return new_sl > current_sl  # Para BUY, SL más alto es mejor
        else:
            return new_sl < current_sl  # Para SELL, SL más bajo es mejor

    def _validate_and_correct_tp_sl(self, tr: TrackedPos, entry_price: float) -> None:
        """
        Valida y auto-corrige TP/SL si parecen ser PUNTOS en lugar de PRECIO.
        También detecta y corrige SL en la dirección incorrecta respecto al entry.

        NUEVO en v3.0: Previene errores de configuración
        FIX v11.0 (BUG-1): Detecta vSL en dirección incorrecta (p.ej. SELL con
          vSL < entry, o BUY con vSL > entry) y lo refleja simétricamente respecto
          al entry. Esto evita que la posición se cierre en el primer tick porque
          el precio ya está "en zona SL" desde la apertura.
        """
        point = tr.point

        # ========== VALIDAR Y CORREGIR TP ==========
        if 0 < tr.virtual_tp < 100:
            # Parece ser puntos, convertir a precio
            if tr.side == "BUY":
                virtual_tp_corrected = entry_price + (tr.virtual_tp * point)
            else:
                virtual_tp_corrected = entry_price - (tr.virtual_tp * point)

            self._send({
                "event": "AUTO_CORRECTED_TP",
                "ticket": tr.ticket,
                "symbol": tr.symbol,
                "side": tr.side,
                "field": "virtual_tp",
                "original_value": tr.virtual_tp,
                "corrected_value": virtual_tp_corrected,
                "entry_price": entry_price,
                "message": f"Detected POINTS ({tr.virtual_tp}) instead of PRICE, auto-corrected to {virtual_tp_corrected:.5f}"
            })

            tr.virtual_tp = virtual_tp_corrected

        # ========== VALIDAR Y CORREGIR SL (puntos → precio) ==========
        if 0 < tr.virtual_sl_price < 100:
            # Parece ser puntos, convertir a precio
            if tr.side == "BUY":
                virtual_sl_corrected = entry_price - (tr.virtual_sl_price * point)
            else:
                virtual_sl_corrected = entry_price + (tr.virtual_sl_price * point)

            self._send({
                "event": "AUTO_CORRECTED_SL",
                "ticket": tr.ticket,
                "symbol": tr.symbol,
                "side": tr.side,
                "field": "virtual_sl",
                "original_value": tr.virtual_sl_price,
                "corrected_value": virtual_sl_corrected,
                "entry_price": entry_price,
                "message": f"Detected POINTS ({tr.virtual_sl_price}) instead of PRICE, auto-corrected to {virtual_sl_corrected:.5f}"
            })

            tr.virtual_sl_price = virtual_sl_corrected

            # Recalcular virtual_sl_points si fue corregido
            if tr.virtual_sl_points == 0:
                tr.virtual_sl_points = int(abs(entry_price - tr.virtual_sl_price) / point)

        # ========== FIX v11.0: VALIDAR DIRECCIÓN DEL vSL ==========
        # Para SELL: vSL debe estar POR ENCIMA del entry (entry + dist)
        # Para BUY:  vSL debe estar POR DEBAJO del entry (entry - dist)
        # Si está al revés, lo reflejamos simétricamente respecto al entry.
        if tr.virtual_sl_price > 0 and entry_price > 0:
            sl_on_wrong_side = (
                (tr.side == "SELL" and tr.virtual_sl_price < entry_price) or
                (tr.side == "BUY"  and tr.virtual_sl_price > entry_price)
            )
            if sl_on_wrong_side:
                dist = abs(entry_price - tr.virtual_sl_price)
                original_vsl = tr.virtual_sl_price          # FIX v15.0: capturar ANTES
                if tr.side == "SELL":
                    corrected = entry_price + dist   # colocar por encima
                else:
                    corrected = entry_price - dist   # colocar por debajo
                # FIX v15.0 (FIX-3): campos original_vsl/corrected_vsl ahora tienen
                # valores reales. Antes se publicaban como None porque se leían
                # después de la asignación o con nombre de variable incorrecto.
                # Añadido atr_pts para correlacionar la magnitud del error con el ATR.
                _atr_pts_for_log = (
                    round(tr.ind_atr / point) if (tr.ind_atr and point > 0) else None
                )
                self._send({
                    "event": "WRONG_SIDE_SL_CORRECTED",
                    "ticket": tr.ticket,
                    "symbol": tr.symbol,
                    "side": tr.side,
                    "entry_price": entry_price,
                    "original_vsl": round(original_vsl, 5),   # FIX v15.0: valor real
                    "corrected_vsl": round(corrected, 5),      # FIX v15.0: valor real
                    "dist_pts": round(dist / point),
                    "atr_pts": _atr_pts_for_log,               # FIX v15.0: nuevo campo
                    "message": (
                        f"vSL {original_vsl:.2f} estaba en dirección incorrecta "
                        f"para {tr.side} con entry {entry_price:.2f}. "
                        f"Corregido a {corrected:.2f} (simétrico)"
                    )
                })
                tr.virtual_sl_price = corrected
                tr.virtual_sl_points = int(dist / point)

        # ========== FIX v15.0 (FIX-2): GUARDIA DE vSL MÍNIMO EN S3 ==========
        # Segunda línea de defensa independiente de S2 v17.
        # Si el vSL es demasiado ajustado respecto al ATR, genera slippage extremo
        # porque el precio puede saltar el nivel completo en un solo bar.
        # Análisis log 10/03/2026: slippages de 275, 160, 129 pts en vSLs de 20-78 pts
        # con ATR de ~200 pts. El fix de S2 v17 (MIN_VSL_ATR_RATIO=0.5) no era
        # suficiente por sí solo: S3 debe validar de forma autónoma.
        # Solo se aplica si el nuevo SL amplía el actual (no sobreescribe uno ya amplio).
        #
        # FIX v20.0 (FIX-2): La guardia nunca disparaba para posiciones SELL.
        # Causa: usaba _is_improvement(tr, current_sl, new_sl) para decidir si aplicar
        # la expansión. Para SELL, _is_improvement retorna True cuando new_sl < current_sl
        # (SL más bajo = más protector del profit). Pero aquí necesitamos exactamente lo
        # contrario: ampliar el vSL hacia afuera del entry (más alto para SELL). La condición
        # correcta es "¿es new_sl más lejos del entry que current_sl?" — independiente de la
        # dirección del trade. Confirmado en log 13/03/2026: 14 tickets con WRONG_SIDE_SL
        # corregidos a distancias de 7-73pts; guardia nunca emitió VSL_EXPANDED_MIN_ATR.
        # Fix: reemplazar _is_improvement por comparación directa de distancia al entry.
        if tr.virtual_sl_price > 0 and entry_price > 0 and tr.virtual_sl_points > 0:
            _atr = tr.ind_atr if (tr.ind_atr and tr.ind_atr > 0) else None
            if _atr and point > 0:
                _atr_pts = _atr / point
                _min_vsl_pts = int(_atr_pts * self.MIN_VSL_ATR_RATIO_S3)
            else:
                _min_vsl_pts = self.MIN_VSL_FALLBACK_POINTS
            _min_vsl_pts = max(_min_vsl_pts, 20)  # nunca menos de 20 pts como suelo absoluto

            if tr.virtual_sl_points < _min_vsl_pts:
                _old_vsl_price = tr.virtual_sl_price
                _old_vsl_pts   = tr.virtual_sl_points
                if tr.side == "SELL":
                    _new_vsl_price = entry_price + _min_vsl_pts * point
                else:
                    _new_vsl_price = entry_price - _min_vsl_pts * point
                # FIX v20.0: usar distancia al entry para verificar que la expansión
                # amplía el vSL. _is_improvement era incorrecto aquí: para SELL devuelve
                # True cuando new_sl < current_sl (más protector), que es lo opuesto de
                # lo que necesitamos (ampliar hacia afuera = new_sl > current_sl para SELL).
                _new_dist = abs(_new_vsl_price - entry_price)
                _cur_dist = abs(tr.virtual_sl_price - entry_price)
                if _new_dist > _cur_dist:  # el nuevo vSL es más amplio que el actual
                    tr.virtual_sl_price  = _new_vsl_price
                    tr.virtual_sl_points = _min_vsl_pts
                    self._send({
                        "event": "VSL_EXPANDED_MIN_ATR",
                        "ticket": tr.ticket,
                        "symbol": tr.symbol,
                        "side": tr.side,
                        "entry_price": entry_price,
                        "old_vsl_price": round(_old_vsl_price, 5),
                        "new_vsl_price": round(tr.virtual_sl_price, 5),
                        "old_vsl_pts": _old_vsl_pts,
                        "new_vsl_pts": _min_vsl_pts,
                        "atr_pts": round(_atr / point) if _atr else None,
                        "min_ratio_used": self.MIN_VSL_ATR_RATIO_S3,
                        "source": "atr_based" if _atr else "fallback",
                    })

        # ========== FIX v13.0: VALIDAR DIRECCIÓN DEL vTP ==========
        # Para SELL: vTP debe estar POR DEBAJO del entry (entry - dist → profit)
        # Para BUY:  vTP debe estar POR ENCIMA del entry (entry + dist → profit)
        # Si está al revés, lo reflejamos simétricamente (idéntico al fix del SL).
        # Detectado en log 09/03/2026: ticket 314215545 SELL con vTP > entry
        # (virtual_tp_points negativo → tp al lado del drawdown, no del profit).
        if tr.virtual_tp > 0 and entry_price > 0:
            tp_on_wrong_side = (
                (tr.side == "SELL" and tr.virtual_tp > entry_price) or
                (tr.side == "BUY"  and tr.virtual_tp < entry_price)
            )
            if tp_on_wrong_side:
                dist_tp = abs(entry_price - tr.virtual_tp)
                if tr.side == "SELL":
                    corrected_tp = entry_price - dist_tp   # colocar por debajo
                else:
                    corrected_tp = entry_price + dist_tp   # colocar por encima
                self._send({
                    "event": "WRONG_SIDE_TP_CORRECTED",
                    "ticket": tr.ticket,
                    "symbol": tr.symbol,
                    "side": tr.side,
                    "entry_price": entry_price,
                    "original_vtp": tr.virtual_tp,
                    "corrected_vtp": corrected_tp,
                    "dist_pts": round(dist_tp / point),
                    "message": (
                        f"vTP {tr.virtual_tp:.2f} estaba en dirección incorrecta "
                        f"para {tr.side} con entry {entry_price:.2f}. "
                        f"Corregido a {corrected_tp:.2f} (simétrico)"
                    )
                })
                tr.virtual_tp = corrected_tp

    def _state_family(self, tr: "TrackedPos") -> str:
        state = str(getattr(tr, "ind_state", "") or "").strip().lower()
        if state in ("trend_up", "trend_down", "breakout_wait_up", "breakout_wait_down"):
            return "trend"
        if state in ("range",):
            return "range"
        if state in ("transition_up", "transition_down"):
            return "transition"
        return "default"

    def _effective_time_force_close_secs(self, tr: "TrackedPos") -> int:
        timeout = int(self.TIME_FORCE_CLOSE_SECS)
        family = self._state_family(tr)
        score = float(getattr(tr, "ind_score", 0.0) or 0.0)
        peak_profit = float(getattr(tr, "peak_profit_points", 0.0) or 0.0)
        vsl = max(1.0, float(getattr(tr, "virtual_sl_points", 0.0) or 0.0))

        if family == "trend":
            timeout += 120
        elif family == "transition":
            timeout += 60
        elif family == "range":
            timeout -= 30

        if score >= 0.55:
            timeout += 60
        elif score <= 0.45:
            timeout -= 30

        if peak_profit >= 2.0 * vsl:
            timeout += 120
        elif peak_profit >= 1.5 * vsl:
            timeout += 60

        if getattr(tr, "partial_done", False) and getattr(tr, "be_armed", False):
            timeout += 60

        return max(180, timeout)

    def _get_trailing_unlock_profit_points(self, tr: "TrackedPos") -> int:
        """Profit mínimo necesario para habilitar el trailing tras el BE."""
        be_trigger = max(0, int(getattr(tr.risk, 'be_trigger_points', 0) or 0))
        vsl_pts = max(0, int(getattr(tr, 'virtual_sl_points', 0) or 0))
        extra_pts = max(
            int(self.TRAILING_AFTER_BE_EXTRA_PTS),
            int(round(vsl_pts * self.TRAILING_AFTER_BE_EXTRA_R)),
        )
        return be_trigger + extra_pts

    def _is_trailing_unlocked(self, tr: "TrackedPos", bid: float, ask: float) -> tuple[bool, dict]:
        """
        Determina si el trailing ya puede activarse tras el armado del BE.

        v30.0: añade histéresis para evitar parpadeo ON/OFF cuando el profit
        oscila alrededor del umbral de desbloqueo.
        """
        profit_pts = self._calc_profit_points(tr, bid, ask)

        # Tras un partial, el runner puede necesitar protección más rápida.
        if getattr(tr, 'partial_done', False):
            tr.trailing_unlocked = True
            return True, {
                'trailing_unlock_reason': 'partial_done',
                'since_be_secs': round(max(0.0, now() - float(getattr(tr, 'be_armed_ts', 0.0) or 0.0)), 3),
                'profit_points': round(profit_pts, 1),
                'unlock_profit_points': self._get_trailing_unlock_profit_points(tr),
                'relock_profit_points': max(0, self._get_trailing_unlock_profit_points(tr) - int(self.TRAILING_UNLOCK_HYSTERESIS_PTS)),
                'min_delay_secs': self.TRAILING_AFTER_BE_MIN_SECS,
                'latched': True,
            }

        be_ts = float(getattr(tr, 'be_armed_ts', 0.0) or 0.0)
        since_be_secs = max(0.0, now() - be_ts) if be_ts > 0 else 0.0
        unlock_profit_points = self._get_trailing_unlock_profit_points(tr)
        relock_profit_points = max(0, unlock_profit_points - int(self.TRAILING_UNLOCK_HYSTERESIS_PTS))
        prev_unlocked = bool(getattr(tr, 'trailing_unlocked', False))

        enough_delay = since_be_secs >= self.TRAILING_AFTER_BE_MIN_SECS
        if prev_unlocked:
            unlocked = enough_delay and (profit_pts >= relock_profit_points)
        else:
            unlocked = enough_delay and (profit_pts >= unlock_profit_points)

        tr.trailing_unlocked = bool(unlocked)
        meta = {
            'trailing_unlock_reason': 'delay_profit_hysteresis_gate',
            'since_be_secs': round(since_be_secs, 3),
            'profit_points': round(profit_pts, 1),
            'unlock_profit_points': unlock_profit_points,
            'relock_profit_points': relock_profit_points,
            'hysteresis_pts': int(self.TRAILING_UNLOCK_HYSTERESIS_PTS),
            'min_delay_secs': self.TRAILING_AFTER_BE_MIN_SECS,
            'latched_before': prev_unlocked,
            'latched_after': bool(unlocked),
        }
        return bool(unlocked), meta

    def _effective_trailing_params(self, tr: "TrackedPos", profit_points: float) -> tuple[int, int, dict]:
        trail_distance_pts = max(1, int(tr.risk.trail_points))
        trail_step_pts = max(1, int(tr.risk.trail_step_points))
        family = self._state_family(tr)
        score = float(getattr(tr, "ind_score", 0.0) or 0.0)
        vsl = max(1.0, float(getattr(tr, "virtual_sl_points", 0.0) or 0.0))
        profit_R = float(profit_points) / vsl

        if family == "trend":
            if score >= 0.55:
                trail_distance_pts = int(round(trail_distance_pts * 1.05))
                trail_step_pts = int(round(trail_step_pts * 0.90))
            else:
                trail_distance_pts = int(round(trail_distance_pts * 0.95))
        elif family == "range":
            trail_distance_pts = int(round(trail_distance_pts * 0.85))
            trail_step_pts = int(round(trail_step_pts * 0.85))
        elif family == "transition":
            trail_distance_pts = int(round(trail_distance_pts * 0.92))
            trail_step_pts = int(round(trail_step_pts * 0.90))

        if getattr(tr, "be_armed", False):
            if getattr(tr, "partial_done", False) and profit_R >= 2.0:
                trail_distance_pts = int(round(trail_distance_pts * 0.70))
                trail_step_pts = int(round(trail_step_pts * 0.70))
            elif profit_R >= 1.2:
                trail_distance_pts = int(round(trail_distance_pts * 0.85))
                trail_step_pts = int(round(trail_step_pts * 0.85))

        # v28.0: runner tight trail — si el partial ya se ejecutó y
        # runner_tight_trail_pts > 0, sustituir el trailing normal por
        # una distancia muy pequeña (ej: 20pts) que prácticamente cierra
        # el runner al breakeven. El objetivo: conservar la ganancia del
        # partial (50% cerrado al TP) sin dejar que el runner pierda mucho
        # si el precio revierte. Si el precio sigue moviéndose a favor,
        # el tight trail lo acompañará y cerrará con ganancia adicional.
        trail_distance_pts = max(12, trail_distance_pts)
        trail_step_pts = max(4, min(trail_step_pts, trail_distance_pts))

        # v28.0/v28.1: runner tight trail — se aplica DESPUÉS del max() para
        # que el tight sobreescriba el floor de 12pts si es mayor.
        # BUGFIX v28.1: meta se construye ANTES de este bloque para que
        # meta['runner_tight'] no genere UnboundLocalError.
        meta = {
            "state": getattr(tr, "ind_state", ""),
            "state_family": family,
            "score": round(score, 6),
            "profit_R": round(profit_R, 3),
            "trail_distance_pts": trail_distance_pts,
            "trail_step_pts": trail_step_pts,
        }

        tight = int(getattr(tr.risk.partial_close, 'runner_tight_trail_pts', 0) or 0)
        if tight > 0 and getattr(tr, 'partial_done', False):
            trail_distance_pts = tight
            trail_step_pts = max(4, tight // 4)  # paso = 25% de la distancia
            meta['runner_tight'] = True
            meta['trail_distance_pts'] = trail_distance_pts  # actualizar meta
            meta['trail_step_pts']     = trail_step_pts

        return trail_distance_pts, trail_step_pts, meta

    def _handle_break_even(self, tr: TrackedPos, bid: float, ask: float) -> None:
        """
        Activa break-even CON validación de mejora

        MEJORA v3.0: Valida que BE realmente mejora el SL actual
        """
        if tr.be_armed:
            return

        if tr.risk.be_trigger_points <= 0:
            return

        profit_pts = self._calc_profit_points(tr, bid, ask)

        if profit_pts >= tr.risk.be_trigger_points:
            if tr.risk.be_offset_points < 15:
                tr.risk.be_offset_points = 15
            be_sl = self._calc_be_sl(tr)

            # VALIDACIÓN: Solo armar si BE mejora el SL actual
            if tr.virtual_sl_price == 0 or self._is_improvement(tr, tr.virtual_sl_price, be_sl):
                tr.be_armed = True
                tr.be_armed_ts = now()
                tr.trailing_wait_logged = False
                tr.trailing_unlocked = False
                old_sl = tr.virtual_sl_price
                tr.virtual_sl_price = self._improve_virtual_sl(tr, tr.virtual_sl_price, be_sl)

                self._send({
                    "event": "BE_ARMED",
                    "ticket": tr.ticket,
                    "symbol": tr.symbol,
                    "profit_points": profit_pts,
                    "old_virtual_sl": old_sl,
                    "new_virtual_sl": tr.virtual_sl_price,
                    "entry_price": tr.entry_price
                })
            else:
                # BE no mejora SL actual (trailing ya lo superó): no aplicar.
                # v24.0: emitir BE_SKIPPED solo la primera vez — el evento se disparaba
                # en cada ciclo del monitor (5Hz) mientras el profit siguiera por encima
                # del trigger, generando >1000 eventos redundantes por sesión.
                if not tr.be_skip_warned:
                    tr.be_skip_warned = True
                    self._send({
                        "event": "BE_SKIPPED",
                        "ticket": tr.ticket,
                        "symbol": tr.symbol,
                        "profit_points": profit_pts,
                        "current_sl": tr.virtual_sl_price,
                        "be_level": be_sl,
                        "reason": "Current SL already better than BE level"
                    })

    def _apply_profit_lock_levels(self, tr: TrackedPos, profit_points: float) -> bool:
        """
        Aplica profit locks progresivos según nivel de ganancia

        NUEVO en v3.0: Garantiza ganancias en trades ganadores

        Estrategia:
        - 2R → Lock 1R (garantizar 1R mínimo)
        - 3R → Lock 2R (garantizar 2R mínimo)
        - 4R+ → Lock 3R (garantizar 3R mínimo)

        Returns:
            True si se aplicó un profit lock, False otherwise
        """
        if tr.virtual_sl_points <= 0:
            return False

        risk_distance = tr.virtual_sl_points * tr.point
        old_sl = tr.virtual_sl_price
        lock_applied = False
        lock_level_R = 0

        # Nivel 4: 4R+ → Lock 3R
        if profit_points >= 4 * tr.virtual_sl_points:
            if tr.side == "BUY":
                lock_level = tr.entry_price + (3 * risk_distance)
            else:
                lock_level = tr.entry_price - (3 * risk_distance)

            lock_level_R = 3
            lock_applied = True

        # Nivel 3: 3R → Lock 2R
        elif profit_points >= 3 * tr.virtual_sl_points:
            if tr.side == "BUY":
                lock_level = tr.entry_price + (2 * risk_distance)
            else:
                lock_level = tr.entry_price - (2 * risk_distance)

            lock_level_R = 2
            lock_applied = True

        # Nivel 2: 2R → Lock 1R
        elif profit_points >= 2 * tr.virtual_sl_points:
            if tr.side == "BUY":
                lock_level = tr.entry_price + (1 * risk_distance)
            else:
                lock_level = tr.entry_price - (1 * risk_distance)

            lock_level_R = 1
            lock_applied = True

        if lock_applied:
            # Solo aplicar si mejora el SL actual
            if self._is_improvement(tr, tr.virtual_sl_price, lock_level):
                tr.virtual_sl_price = self._improve_virtual_sl(tr, tr.virtual_sl_price, lock_level)

                self._send({
                    "event": "PROFIT_LOCK_APPLIED",
                    "ticket": tr.ticket,
                    "symbol": tr.symbol,
                    "lock_level_R": lock_level_R,
                    "profit_points": profit_points,
                    "profit_R": profit_points / tr.virtual_sl_points,
                    "old_virtual_sl": old_sl,
                    "new_virtual_sl": tr.virtual_sl_price
                })

                return True

        return False

    def _calc_trailing_sl(self, tr: TrackedPos, bid: float, ask: float) -> tuple[float, dict]:
        """Calcula trailing SL contextual por state, score y profit_R usando el mismo snapshot bid/ask del monitor."""
        profit_pts = self._calc_profit_points(tr, bid, ask)
        trail_distance_pts, trail_step_pts, trail_meta = self._effective_trailing_params(tr, profit_pts)
        trail_distance = trail_distance_pts * tr.point
        trail_step = max(trail_step_pts * tr.point, trail_distance * 0.20)

        if tr.side == "BUY":
            ref = tr.max_favorable_price or tr.entry_price
            new_sl = ref - trail_distance
            if tr.virtual_sl_price > 0:
                improvement = new_sl - tr.virtual_sl_price
                if improvement < trail_step:
                    return tr.virtual_sl_price, trail_meta
            return new_sl, trail_meta

        # SELL
        ref = tr.max_favorable_price or tr.entry_price
        new_sl = ref + trail_distance
        if tr.virtual_sl_price > 0:
            improvement = tr.virtual_sl_price - new_sl
            if improvement < trail_step:
                return tr.virtual_sl_price, trail_meta
        return new_sl, trail_meta

    def _improve_virtual_sl(self, tr: TrackedPos, current_sl: float, new_sl: float) -> float:
        """Mejora SL virtual (solo en dirección favorable)"""
        if current_sl == 0:
            return new_sl

        if tr.side == "BUY":
            return max(current_sl, new_sl)
        else:
            return min(current_sl, new_sl)

    def _close_position_full(
            self,
            tr: TrackedPos,
            bid: float,
            ask: float,
            reason: str,
            extra_payload: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Cierra posición completamente.

        FIX v15.0 (FIX-6): acepta extra_payload opcional para enriquecer el
        evento de cierre con campos adicionales (profit_points, hold_seconds…)
        sin necesidad de emitir un evento previo separado. Elimina el doble
        evento TIME_FORCE_CLOSE_TRIGGERED + PROFIT_FLOOR_CLOSE_TRIGGERED que
        se observaba en el log 10/03/2026 (36 casos).
        """
        # FIX v10.0: Eliminar de tracking ANTES de enviar la orden al broker.
        # Esto evita que el monitor_loop detecte la posición como aún abierta
        # en el mismo ciclo o en el siguiente y lance cierres duplicados
        # (bug observado en logs: múltiples VIRTUAL_SL_TRIGGERED con ok=False
        # tras el primer cierre exitoso).
        _close_ts = now()
        self._intentional_close_tickets[tr.ticket] = _close_ts
        self._recent_closed_context[tr.ticket] = {
            'closed_ts': _close_ts,
            'symbol': tr.symbol,
            'side': tr.side,
            'open_price': tr.entry_price,
            'volume': tr.volume,
            'hold_seconds': round(_close_ts - tr.opened_ts, 1) if tr.opened_ts > 0 else None,
            'virtual_sl': tr.virtual_sl_price,
            'virtual_tp': tr.virtual_tp,
            'close_reason': reason,
        }
        self._tracked.pop(tr.ticket, None)
        self._known_open.discard(tr.ticket)
        self._remove_adaptive_manager(tr.ticket, reason)

        # El bridge construye internamente comment=f"mimo_close:{reason}".
        # MT5 limita el campo comment a 31 chars -> reason debe tener <= 20 chars.
        # Se trunca aqui para cubrir razones largas como "ADAPTIVE_VIRTUAL_SL".
        result = self.bridge.close_position_market(
            ticket=tr.ticket,
            symbol=tr.symbol,
            side=tr.side,
            volume=tr.volume,
            magic=tr.magic,
            deviation=20,
            reason=reason[:20]
        )

        close_price = bid if tr.side == "BUY" else ask

        # Slippage respecto al virtual SL (solo relevante en cierres por SL)
        slippage_pts = None
        if reason == "VIRTUAL_SL" and tr.virtual_sl_price > 0 and tr.point > 0:
            if tr.side == "BUY":
                slippage_pts = round((tr.virtual_sl_price - close_price) / tr.point, 1)
            else:
                slippage_pts = round((close_price - tr.virtual_sl_price) / tr.point, 1)

        # FIX 13/04/2026: calcular profit_pts y hold_seconds aquí para que
        # aparezcan en TODOS los tipos de cierre (vSL, vTP, adaptive, max_hold…)
        # y no solo en TIME_FORCE_CLOSE (que los pasaba via extra_payload).
        # Análisis log 13/04/2026: profit_pts=None en VIRTUAL_SL_TRIGGERED dificultaba
        # el diagnóstico post-sesión sin hacer join con signals log.
        _hold_secs_auto = round(now() - tr.opened_ts, 1) if tr.opened_ts > 0 else None
        _profit_pts_auto: Optional[float] = None
        if tr.point > 0:
            if tr.side == "BUY":
                _profit_pts_auto = round((close_price - tr.entry_price) / tr.point, 1)
            else:
                _profit_pts_auto = round((tr.entry_price - close_price) / tr.point, 1)

        event_payload = {
            "event": f"{reason}_TRIGGERED",
            "ticket": tr.ticket,
            "symbol": tr.symbol,
            "side": tr.side,
            "entry_price": tr.entry_price,
            "close_price": close_price,
            "virtual_sl": tr.virtual_sl_price,
            "virtual_tp": tr.virtual_tp,
            "volume": tr.volume,
            "profit_pts": _profit_pts_auto,
            "hold_seconds": _hold_secs_auto,
            "result": result,
            # v18.0 (MEJORA-3): régimen y score en el momento del cierre.
            # Permite análisis post-sesión por state directamente desde el log
            # de S3, sin necesidad de join con el log de S2.
            "state": (getattr(tr, "ind_state", None) or getattr(tr, "ind_regime", None)),
            "score":  tr.ind_score,
        }
        if slippage_pts is not None:
            event_payload["slippage_pts"] = slippage_pts
        # FIX v15.0 (FIX-6): enriquecer con campos extra (p.ej. profit_points,
        # hold_seconds cuando la razón es TIME_FORCE_CLOSE). Los campos de
        # extra_payload sobreescriben los calculados automáticamente si vienen
        # más precisos (p.ej. TIME_FORCE_CLOSE pasa peak_profit_points).
        if extra_payload:
            event_payload.update(extra_payload)

        self._send(event_payload)

        # FIX v17.0 (FIX-2): alerta de slippage extremo
        # Se emite DESPUÉS del evento de cierre principal para no interferir.
        # Útil para alertas automáticas y correlación post-sesión.
        if (slippage_pts is not None
                and slippage_pts > self.MAX_SLIPPAGE_ALERT_PTS):
            self._send({
                "event": "SLIPPAGE_ALERT",
                "ticket": tr.ticket,
                "symbol": tr.symbol,
                "side": tr.side,
                "slippage_pts": slippage_pts,
                "threshold_pts": self.MAX_SLIPPAGE_ALERT_PTS,
                "virtual_sl": tr.virtual_sl_price,
                "close_price": close_price,
                "bid": bid,
                "ask": ask,
                "close_reason": reason,
            })

        # FIX v28.5: reintento de cierre si slippage supera MAX_SLIPPAGE_RETRY_PTS.
        # Caso típico: gap de precio brusco que deja la posición parcialmente abierta
        # o cierra a un precio muy desfavorable. El reintento verifica en MT5 si
        # queda volumen residual y lo cierra a mercado inmediatamente.
        # Ticket 334460414 (13/04/2026): slippage=514pts → este reintento lo habría
        # capturado y reducido la pérdida residual.
        if (slippage_pts is not None
                and slippage_pts > self.MAX_SLIPPAGE_RETRY_PTS):
            try:
                # Esperar brevemente para que MT5 registre el estado tras el primer cierre
                time.sleep(0.3)
                residual_positions = self.bridge.positions_get(ticket=tr.ticket)
                if residual_positions:
                    # Hay volumen residual — cerrar inmediatamente
                    residual = residual_positions[0]
                    residual_vol = float(getattr(residual, "volume", 0.0) or 0.0)
                    if residual_vol > 0:
                        ok2, bid2, ask2 = self.bridge.tick_bid_ask(tr.symbol)
                        result2 = self.bridge.close_position_market(
                            ticket=tr.ticket,
                            symbol=tr.symbol,
                            side=tr.side,
                            volume=residual_vol,
                            magic=tr.magic,
                            deviation=50,   # deviation ampliado: mercado en gap
                            reason="SLIPPAGE_RETRY"
                        )
                        close2 = bid2 if tr.side == "BUY" else ask2
                        self._send({
                            "event": "SLIPPAGE_RETRY_CLOSE",
                            "ticket": tr.ticket,
                            "symbol": tr.symbol,
                            "side": tr.side,
                            "original_slippage_pts": slippage_pts,
                            "residual_volume": residual_vol,
                            "retry_close_price": close2,
                            "retry_ok": result2.get("ok", False),
                            "retry_retcode": result2.get("retcode"),
                            "warning": (
                                f"Slippage de {slippage_pts:.0f}pts superó umbral "
                                f"de {self.MAX_SLIPPAGE_RETRY_PTS}pts. "
                                "Reintento de cierre automático ejecutado."
                            ),
                        })
                    else:
                        # Posición ya cerrada completamente — solo informar
                        self._send({
                            "event": "SLIPPAGE_RETRY_SKIP",
                            "ticket": tr.ticket,
                            "symbol": tr.symbol,
                            "reason": "no_residual_volume",
                            "original_slippage_pts": slippage_pts,
                        })
                else:
                    # No hay posición residual en MT5
                    self._send({
                        "event": "SLIPPAGE_RETRY_SKIP",
                        "ticket": tr.ticket,
                        "symbol": tr.symbol,
                        "reason": "position_not_found_in_mt5",
                        "original_slippage_pts": slippage_pts,
                    })
            except Exception as _retry_exc:
                self._send({
                    "event": "SLIPPAGE_RETRY_ERROR",
                    "ticket": tr.ticket,
                    "symbol": tr.symbol,
                    "error": str(_retry_exc),
                    "original_slippage_pts": slippage_pts,
                })

    def _check_partial_close(self, tr: TrackedPos, p_obj: Any, bid: float, ask: float) -> bool:
        """Verifica y ejecuta cierre parcial.

        Devuelve True cuando se ejecutó un partial/full con éxito y conviene
        re-evaluar la posición en el siguiente ciclo con precios frescos.
        """
        pc = tr.risk.partial_close

        if not pc.enabled or pc.trigger_profit_pct <= 0:
            return False

        pct = self._calc_profit_pct(tr, p_obj, bid, ask)
        if pct < pc.trigger_profit_pct:
            return False

        price_side_used = "bid" if tr.side == "BUY" else "ask"
        close_price_live = bid if tr.side == "BUY" else ask
        price_source = "monitor_snapshot"
        profit_pts_live = self._calc_profit_points(tr, bid, ask)

        risk_points = float(tr.virtual_sl_points or 0.0)
        if risk_points <= 0:
            broker_sl = float(getattr(p_obj, "sl", 0.0) or 0.0)
            if broker_sl > 0 and tr.point > 0:
                risk_points = abs(tr.entry_price - broker_sl) / tr.point

        r_multiple_live = None
        if risk_points > 0:
            r_multiple_live = float(profit_pts_live) / max(1e-9, risk_points)

        if profit_pts_live <= 0:
            self._send({
                "event": "PARTIAL_CLOSE_BLOCKED_NEGATIVE_PNL",
                "ticket": tr.ticket,
                "symbol": tr.symbol,
                "side": tr.side,
                "pct_basis": pc.basis,
                "pct": pct,
                "trigger_profit_pct": pc.trigger_profit_pct,
                "profit_pts_live": round(profit_pts_live, 1),
                "risk_points": round(risk_points, 1),
                "r_multiple_live": round(r_multiple_live, 4) if r_multiple_live is not None else None,
                "price_side_used": price_side_used,
                "price_source": price_source,
                "bid": bid,
                "ask": ask,
                "profit_money_mt5": float(getattr(p_obj, "profit", 0.0) or 0.0),
                "state": (getattr(tr, "ind_state", None) or getattr(tr, "ind_regime", None)),
                "score": tr.ind_score,
            })
            return False

        vol_to_close = self.bridge.quantize_volume(tr.symbol, tr.volume * pc.close_fraction)
        vmin = 0.01
        remaining = round(tr.volume - vol_to_close, 10)

        if remaining < vmin:
            result = self.bridge.close_position_market(
                ticket=tr.ticket,
                symbol=tr.symbol,
                side=tr.side,
                volume=tr.volume,
                magic=tr.magic,
                deviation=20,
                reason="PARTIAL_FULL"
            )
            self._send({
                "event": "PARTIAL_CLOSE_TO_FULL",
                "ticket": tr.ticket,
                "symbol": tr.symbol,
                "side": tr.side,
                "entry_price": tr.entry_price,
                "close_price": close_price_live,
                "pct": pct,
                "trigger_profit_pct": pc.trigger_profit_pct,
                "closed_volume": tr.volume,
                "basis": pc.basis,
                "profit_pts_live_at_trigger": round(profit_pts_live, 1),
                "risk_points": round(risk_points, 1),
                "r_multiple": round(r_multiple_live, 4) if r_multiple_live is not None else None,
                "price_side_used": price_side_used,
                "price_source": price_source,
                "result": result
            })
            return bool(result.get("ok", False))

        result = self.bridge.close_position_partial(
            ticket=tr.ticket,
            symbol=tr.symbol,
            side=tr.side,
            volume_to_close=vol_to_close,
            magic=tr.magic,
            deviation=20,
            reason="PARTIAL_TARGET"
        )

        bridge_price_sent = result.get("price_sent")
        if bridge_price_sent is not None:
            try:
                close_price_live = float(bridge_price_sent)
                price_source = "bridge_result.price_sent"
            except Exception:
                pass
        else:
            bridge_bid = result.get("bid_at_send")
            bridge_ask = result.get("ask_at_send")
            ref_price = bridge_bid if tr.side == "BUY" else bridge_ask
            if ref_price is not None:
                try:
                    close_price_live = float(ref_price)
                    price_source = "bridge_result.bid_ask"
                except Exception:
                    pass

        profit_pts_partial = None
        if tr.point > 0:
            if tr.side == "BUY":
                profit_pts_partial = round((close_price_live - tr.entry_price) / tr.point, 1)
            else:
                profit_pts_partial = round((tr.entry_price - close_price_live) / tr.point, 1)

        hold_secs_partial = round(now() - tr.opened_ts, 1) if tr.opened_ts > 0 else None

        if result.get("ok"):
            tr.partial_done = True
            tr.volume = remaining
        else:
            self._send({
                "event": "PARTIAL_CLOSE_VOLUME_NOT_UPDATED",
                "ticket": tr.ticket,
                "symbol": tr.symbol,
                "volume_unchanged": tr.volume,
                "attempted_close": vol_to_close,
                "error": result.get("error", "unknown"),
                "warning": "tr.volume NO se redujo; BE y trailing usaran volumen completo"
            })

        self._send({
            "event": "PARTIAL_CLOSE_TRIGGERED",
            "ticket": tr.ticket,
            "symbol": tr.symbol,
            "side": tr.side,
            "entry_price": tr.entry_price,
            "close_price": close_price_live,
            "pct": pct,
            "trigger_profit_pct": pc.trigger_profit_pct,
            "closed_volume": vol_to_close,
            "remaining_volume": remaining,
            "volume_updated": result.get("ok", False),
            "basis": pc.basis,
            "profit_pts": profit_pts_partial,
            "profit_pts_live_at_trigger": round(profit_pts_live, 1),
            "risk_points": round(risk_points, 1),
            "r_multiple": round(r_multiple_live, 4) if r_multiple_live is not None else None,
            "price_side_used": price_side_used,
            "price_source": price_source,
            "hold_seconds": hold_secs_partial,
            "state": (getattr(tr, "ind_state", None) or getattr(tr, "ind_regime", None)),
            "score": tr.ind_score,
            "result": result
        })

        if result.get("ok", False) and remaining >= 0.01:
            old_vtp = tr.virtual_tp
            tr.virtual_tp = 0.0
            self._send({
                "event": "RUNNER_VTP_DISABLED",
                "ticket": tr.ticket,
                "symbol": tr.symbol,
                "old_virtual_tp": old_vtp,
                "reason": "runner guiado por tight_trail — vTP desactivado",
                "remaining_volume": remaining,
                "tight_trail_pts": getattr(tr.risk.partial_close, 'runner_tight_trail_pts', 0),
            })
            return True

        return bool(result.get("ok", False))

    def _calc_profit_pct(self, tr: TrackedPos, p_obj: Any, bid: float, ask: float) -> float:
        """Calcula porcentaje de ganancia"""
        basis = (tr.risk.partial_close.basis or "R").upper()

        if basis == "EQUITY":
            try:
                acc = self.bridge.info()
                equity = float(acc.equity) if acc else 0.0
            except Exception:
                equity = 0.0
            profit_money = float(getattr(p_obj, "profit", 0.0) or 0.0)
            return 100.0 * (profit_money / max(1e-9, equity))

        # Basis = R (usar virtual_sl_points)
        profit_points = self._calc_profit_points(tr, bid, ask)
        risk_points = float(tr.virtual_sl_points or 0)

        if risk_points <= 0:
            broker_sl = float(getattr(p_obj, "sl", 0.0) or 0.0)
            if broker_sl > 0 and tr.point > 0:
                risk_points = abs(tr.entry_price - broker_sl) / tr.point

        return 100.0 * (profit_points / max(1e-9, risk_points))

    # ========================================================================
    # ADAPTIVE SL — métodos de integración (v8.0)
    # ========================================================================

    def _create_adaptive_manager(self, tr: "TrackedPos") -> None:
        """
        Crea un AdaptiveSLManager y un AdaptiveTPManager para la posición.
        No hace nada si el módulo no está disponible.
        """
        if not _ADAPTIVE_SL_AVAILABLE or self.ADAPTIVE_SL_CONFIG is None:
            return

        # ── AdaptiveSLManager ────────────────────────────────────────────────
        if tr.virtual_sl_price <= 0 or tr.broker_emergency_sl <= 0:
            self._send({
                "event": "ADAPTIVE_SL_SKIPPED",
                "ticket": tr.ticket,
                "reason": "virtual_sl_price o broker_emergency_sl no definidos",
                "virtual_sl_price": tr.virtual_sl_price,
                "broker_emergency_sl": tr.broker_emergency_sl,
            })
        else:
            try:
                # FIX v10.1: El fallback anterior usaba entry_price * 1.05/0.95
                # que para XAUUSD genera un TP ficticio de ~5447 (5% lejos del
                # precio), completamente fuera de cualquier objetivo realista de
                # scalping. El AdaptiveSLManager lo usaba para calcular proximity
                # zones, provocando que nunca entrara en modo compresión/extensión.
                # Nuevo fallback: si no hay TP real, estimamos 1.67R desde entrada
                # (ratio RR típico del perfil scalping). Esto es mucho más cercano
                # a la realidad y mantiene coherencia con los demás managers.
                if tr.virtual_tp > 0:
                    sl_virtual_tp = tr.virtual_tp
                elif tr.virtual_sl_points > 0:
                    rr = 1.67
                    tp_pts = tr.virtual_sl_points * rr * tr.point
                    sl_virtual_tp = (tr.entry_price + tp_pts) if tr.side == "BUY" else (tr.entry_price - tp_pts)
                else:
                    sl_virtual_tp = 0.0  # manager opera sin TP reference

                # FIX v14.0 (P2): config AdaptiveSL por trade con expansion_pts dinamico
                # En vez de usar ADAPTIVE_SL_CONFIG global (expansion_pts=25 fijo),
                # calculamos cuantos pts hay disponibles para expandir en ESTE trade.
                _pts_to_price_val = self.ADAPTIVE_SL_CONFIG.pts_to_price
                _hard_sl_margin   = self.ADAPTIVE_SL_CONFIG.hard_sl_margin_pts
                _avail_pts = int(
                    abs(tr.virtual_sl_price - tr.broker_emergency_sl) / _pts_to_price_val
                )
                # expansion_pts: entre 10 y 25 pts, nunca mas que (margen - hard_sl_margin)
                _dyn_exp = max(10, min(25, _avail_pts - _hard_sl_margin))
                # Crear config especifica para este trade
                import dataclasses as _dc
                _trade_sl_cfg = _dc.replace(self.ADAPTIVE_SL_CONFIG, expansion_pts=_dyn_exp)

                sl_manager = AdaptiveSLManager(
                    ticket=tr.ticket,
                    side=tr.side,
                    entry_price=tr.entry_price,
                    virtual_sl=tr.virtual_sl_price,
                    virtual_tp=sl_virtual_tp,
                    hard_sl=tr.broker_emergency_sl,
                    config=_trade_sl_cfg,  # config por trade con expansion_pts dinamico
                )
                self._adaptive_managers[tr.ticket] = sl_manager
                self._send({
                    "event": "ADAPTIVE_SL_CREATED",
                    "ticket": tr.ticket,
                    "side": tr.side,
                    "virtual_sl": tr.virtual_sl_price,
                    "virtual_tp": sl_virtual_tp,
                    "hard_sl": tr.broker_emergency_sl,
                    "tp_source": "real" if tr.virtual_tp > 0 else ("estimated_1.67R" if tr.virtual_sl_points > 0 else "none"),
                    "expansion_pts_computed": _dyn_exp,   # P2 auditoria
                    "available_margin_pts": _avail_pts,
                    "expansion_threshold_pts": _dyn_exp + _hard_sl_margin,
                })
            except Exception as e:
                self._send({
                    "event": "ADAPTIVE_SL_CREATE_ERROR",
                    "ticket": tr.ticket,
                    "error": str(e),
                })

        # ── AdaptiveTPManager ────────────────────────────────────────────────
        if tr.virtual_tp <= 0:
            self._send({
                "event": "ADAPTIVE_TP_SKIPPED",
                "ticket": tr.ticket,
                "reason": "virtual_tp no definido",
            })
        else:
            try:
                tp_manager = AdaptiveTPManager(
                    ticket=tr.ticket,
                    side=tr.side,
                    entry_price=tr.entry_price,
                    virtual_tp=tr.virtual_tp,
                    virtual_sl=tr.virtual_sl_price,
                    config=self.ADAPTIVE_TP_CONFIG,
                )
                self._adaptive_tp_managers[tr.ticket] = tp_manager
                self._send({
                    "event": "ADAPTIVE_TP_CREATED",
                    "ticket": tr.ticket,
                    "side": tr.side,
                    "virtual_tp": tr.virtual_tp,
                    "virtual_sl": tr.virtual_sl_price,
                })
            except Exception as e:
                self._send({
                    "event": "ADAPTIVE_TP_CREATE_ERROR",
                    "ticket": tr.ticket,
                    "error": str(e),
                })

    def _remove_adaptive_manager(self, ticket: int, reason: str) -> None:
        """Elimina el AdaptiveSLManager y AdaptiveTPManager de una posición cerrada."""
        sl_manager = self._adaptive_managers.pop(ticket, None)
        if sl_manager is not None:
            try:
                sl_manager.on_closed()
            except Exception:
                pass

        tp_manager = self._adaptive_tp_managers.pop(ticket, None)
        if tp_manager is not None:
            try:
                tp_manager.on_closed()
            except Exception:
                pass

    def _update_tracked_indicators(self, tr: "TrackedPos", indicators: dict) -> None:
        """
        Actualiza los indicadores almacenados en TrackedPos desde un dict.
        También sincroniza el virtual_tp del manager si cambió.

        Se llama desde _handle_open (con indicators del OPEN) y desde
        _handle_modify (con indicators del MODIFY).

        FIX v15.0 (FIX-5): registra el timestamp de la última actualización
        para permitir detectar cuando S2 lleva mucho tiempo sin enviar MODIFY.
        """
        if not indicators:
            return
        if "atr"            in indicators: tr.ind_atr            = float(indicators["atr"])
        if "rsi"            in indicators: tr.ind_rsi            = float(indicators["rsi"])
        if "macd_hist"      in indicators: tr.ind_macd_hist      = float(indicators["macd_hist"])
        if "macd_hist_prev" in indicators: tr.ind_macd_hist_prev = float(indicators["macd_hist_prev"])
        if "volume"         in indicators: tr.ind_volume         = float(indicators["volume"])
        if "volume_ma"      in indicators: tr.ind_volume_ma      = float(indicators["volume_ma"])
        if "proba_long"     in indicators: tr.ind_proba_long     = float(indicators["proba_long"])
        if "proba_short"    in indicators: tr.ind_proba_short    = float(indicators["proba_short"])
        # FIX v15.0 (FIX-5): actualizar timestamp y resetear flag de alerta
        # si llegan indicadores frescos (S2 se ha recuperado del estado de bloqueo).
        tr.ind_last_update_ts = now()
        tr.ind_stale_warned   = False   # permitir re-alertar si vuelve a quedarse stale

    def _update_tracked_context(self, tr: "TrackedPos", context: dict) -> None:
        """
        Actualiza los campos de contexto de mercado en TrackedPos desde un dict.
        Análogo a _update_tracked_indicators pero para metadatos del modelo/broker.

        Se llama desde _handle_open (con context del OPEN si lo incluye S2 v24)
        y desde _handle_modify (con context del MODIFY/keepalive).

        Campos actualizados (todos opcionales — solo si están presentes):
          state   → ind_state   (str):   estado del modelo en ese tick
          score   → ind_score   (float): score del modelo (solo cuando hay señal)
          spread  → ind_spread  (int):   spread en puntos del tick MT5

        v18.0: el spread se usa directamente en la inhibición de TIME_FORCE_CLOSE
        (ver bloque TIME_FORCE_CLOSE en _monitor_loop). El state y score enriquecen
        el log de cierre para análisis post-sesión por state.
        """
        if not context:
            return
        _state_value = context.get("state", context.get("regime"))
        if _state_value:
            _state_value = str(_state_value)
            tr.ind_state = _state_value
            tr.ind_regime = _state_value  # compatibilidad retro
        if "score"  in context and context["score"] is not None:
            tr.ind_score  = float(context["score"])
        if "spread" in context and context["spread"] is not None:
            tr.ind_spread = int(context["spread"])

    def _run_adaptive_sl(self, tr: "TrackedPos", bid: float, ask: float) -> bool:
        """
        Ejecuta un ciclo del AdaptiveSLManager para la posición.

        Devuelve True si la posición fue CERRADA por el adaptativo
        (para que el monitor_loop haga `continue` y no evalúe más).
        Devuelve False si no se hizo nada o solo se actualizó el SL.

        Lógica:
          - Si el adaptativo pide CLOSE  → cierra y devuelve True
          - Si pide UPDATE_SL            → actualiza tr.virtual_sl_price y
                                           sincroniza el TP si cambió; devuelve False
          - Si devuelve NONE             → nada; devuelve False
        """
        manager = self._adaptive_managers.get(tr.ticket)
        if manager is None:
            return False

        # FIX v15.0 (FIX-5): alerta de indicadores obsoletos.
        # Si S2 lleva más de IND_STALE_WARN_SECS sin enviar MODIFY (p.ej. por estar
        # bloqueado en MAX_POSITIONS), los managers adaptativos operan "ciegos" sin
        # RSI, MACD ni probabilidades. Emitir ADAPTIVE_INDICATORS_STALE una sola vez
        # por posición para que el operador detecte el estado de degradación.
        _hold_for_stale = now() - tr.opened_ts
        if (not tr.ind_stale_warned
                and _hold_for_stale >= self.IND_STALE_MIN_HOLD
                and tr.ind_last_update_ts > 0
                and (now() - tr.ind_last_update_ts) >= self.IND_STALE_WARN_SECS):
            tr.ind_stale_warned = True
            _null_fields = [
                f for f, v in [
                    ("atr", tr.ind_atr), ("rsi", tr.ind_rsi),
                    ("macd_hist", tr.ind_macd_hist), ("volume_ma", tr.ind_volume_ma),
                    ("proba_long", tr.ind_proba_long), ("proba_short", tr.ind_proba_short),
                ] if v is None
            ]
            self._send({
                "event": "ADAPTIVE_INDICATORS_STALE",
                "ticket": tr.ticket,
                "symbol": tr.symbol,
                "seconds_since_update": round(now() - tr.ind_last_update_ts),
                "hold_seconds": round(_hold_for_stale),
                "null_fields": _null_fields,
                "warning": (
                    "S2 no ha enviado MODIFY en más de "
                    f"{self.IND_STALE_WARN_SECS}s. AdaptiveSL operando sin "
                    "indicadores frescos."
                ),
            })
        elif (not tr.ind_stale_warned
                and _hold_for_stale >= self.IND_STALE_MIN_HOLD
                and tr.ind_last_update_ts == 0
                and _hold_for_stale >= self.IND_STALE_WARN_SECS):
            # Caso: nunca recibió ningún MODIFY desde la apertura
            tr.ind_stale_warned = True
            self._send({
                "event": "ADAPTIVE_INDICATORS_STALE",
                "ticket": tr.ticket,
                "symbol": tr.symbol,
                "seconds_since_update": None,
                "hold_seconds": round(_hold_for_stale),
                "null_fields": ["atr", "rsi", "macd_hist", "macd_hist_prev",
                                 "volume", "volume_ma", "proba_long", "proba_short"],
                "warning": "Posición nunca recibió MODIFY con indicadores desde la apertura.",
            })
        # (puede ocurrir si BE o profit_locks lo modificaron en ciclos anteriores)
        if tr.virtual_tp > 0 and tr.virtual_tp != manager.virtual_tp:
            manager.virtual_tp = tr.virtual_tp

        # FIX v24.0: sincronizar virtual_sl ANTES de on_bar via sync_sl().
        # El trailing, BE y profit_locks pueden haber movido tr.virtual_sl_price
        # en ciclos anteriores sin que el manager lo supiera. Sin esta sincronización,
        # on_bar evalúa SL_touched, expansión y compresión contra un nivel stale.
        manager.sync_sl(tr.virtual_sl_price)

        # Si no tenemos ATR en los indicadores, usamos un fallback mínimo
        atr = tr.ind_atr if tr.ind_atr and tr.ind_atr > 0 else (tr.point * 100)
        # Antes se pasaba bar_open=bar_high=bar_low=bar_close (todos iguales al
        # precio tick), lo que hacía body_ratio=0/0 → indecision_ok siempre False.
        # Ahora se usa el OHLC real de MT5, con fallback al precio tick si falla.
        bar_price = bid if tr.side == "BUY" else ask
        ohlc = self._get_last_bar_ohlc(tr.symbol)
        if ohlc:
            bar_open, bar_high, bar_low, bar_close = ohlc
        else:
            bar_open = bar_high = bar_low = bar_close = bar_price

        try:
            result = manager.on_bar(
                bid=bid,
                ask=ask,
                bar_open=bar_open,
                bar_high=bar_high,
                bar_low=bar_low,
                bar_close=bar_close,
                atr=atr,
                rsi=tr.ind_rsi,
                macd_hist=tr.ind_macd_hist,
                macd_hist_prev=tr.ind_macd_hist_prev,
                volume=tr.ind_volume,
                volume_ma=tr.ind_volume_ma,
                proba_long=tr.ind_proba_long,
                proba_short=tr.ind_proba_short,
                ts=now(),
            )
        except Exception as e:
            self._send({
                "event": "ADAPTIVE_SL_ERROR",
                "ticket": tr.ticket,
                "error": str(e),
            })
            return False

        if result.action == "close":
            # FIX v9.0: ignorar cierres por TP — los gestiona el sistema estándar.
            # El AdaptiveSL solo debe cerrar por razones de SL (expansión agotada,
            # SL comprimido alcanzado, hard SL). Devolver False hace que el flujo
            # continúe al sistema de confirmación estándar, que cerrará el TP.
            if result.reason == "VIRTUAL_TP":
                return False

            # FIX 13/04/2026: respetar min_hold_seconds antes de ejecutar cierre
            # adaptativo. Tickets 334164903 (-23pts) y 334242422 (-16pts) cerraron
            # a 0-1s de la apertura porque ADAPTIVE_EXPANSION_SL_TRIGGERED no
            # comprobaba el tiempo mínimo de hold. Con min_hold_seconds=15 en
            # scalping, el cierre se pospone hasta que la posición tiene al menos
            # ese tiempo de vida — protege contra cierres instantáneos por ruido.
            if tr.risk.min_hold_seconds > 0:
                _adaptive_hold = now() - tr.opened_ts
                if _adaptive_hold < tr.risk.min_hold_seconds:
                    self._send({
                        "event": "ADAPTIVE_SL_MIN_HOLD_SKIP",
                        "ticket": tr.ticket,
                        "symbol": tr.symbol,
                        "side": tr.side,
                        "reason": result.reason,
                        "hold_seconds": round(_adaptive_hold, 1),
                        "min_hold_seconds": tr.risk.min_hold_seconds,
                        "virtual_sl": tr.virtual_sl_price,
                        "warning": (
                            f"AdaptiveSL quería cerrar ({result.reason}) pero "
                            f"hold={_adaptive_hold:.1f}s < min_hold={tr.risk.min_hold_seconds}s. "
                            "Cierre pospuesto."
                        ),
                    })
                    return False

            # El adaptativo decide cerrar por SL: ejecutar y limpiar
            self._send({
                "event": "ADAPTIVE_SL_CLOSE",
                "ticket": tr.ticket,
                "symbol": tr.symbol,
                "side": tr.side,
                "reason": result.reason,
                "virtual_sl": tr.virtual_sl_price,
                "bid": bid,
                "ask": ask,
                **result.debug,
            })
            self._close_position_full(tr, bid, ask, f"ADAPTIVE_{result.reason}")
            return True

        elif result.action == "update_sl":
            # El adaptativo mueve el SL (expansión o compresión)
            old_sl = tr.virtual_sl_price
            new_sl = result.new_virtual_sl

            # Aplicar solo si mejora el SL actual (respeta trailing y BE ya armados)
            if new_sl and self._is_improvement(tr, old_sl, new_sl):
                tr.virtual_sl_price = new_sl
                # Sincronizar con el manager (por si _improve_virtual_sl lo hubiera
                # ajustado; aquí lo forzamos a que ambos estén alineados)
                manager.virtual_sl = tr.virtual_sl_price
                self._send({
                    "event": "ADAPTIVE_SL_UPDATE",
                    "ticket": tr.ticket,
                    "symbol": tr.symbol,
                    "side": tr.side,
                    "reason": result.reason,
                    "old_virtual_sl": old_sl,
                    "new_virtual_sl": tr.virtual_sl_price,
                    "bid": bid,
                    "ask": ask,
                    **result.debug,
                })

        return False

    def _run_adaptive_tp(self, tr: "TrackedPos", bid: float, ask: float) -> str:
        """
        Ejecuta un ciclo del AdaptiveTPManager para la posición.

        Devuelve:
            "closed"    → posición cerrada por el adaptativo (TP extendido o
                          SL post-extensión alcanzado). El monitor hace continue.
            "delegated" → TP tocado sin señales de extensión; el sistema
                          estándar (_apply_exit_mode) debe cerrar normalmente.
            "none"      → nada relevante, seguir monitorizando.

        Lógica:
            - "close"       → cierra y devuelve "closed"
            - "update_tp"   → actualiza tr.virtual_tp (compresión)
            - "update_both" → actualiza tr.virtual_tp y tr.virtual_sl_price
                              (extensión: nuevo TP + SL comprimido)
            - "none" con delegate_to_standard=True → devuelve "delegated"
            - "none" sin delegate               → devuelve "none"
        """
        manager = self._adaptive_tp_managers.get(tr.ticket)
        if manager is None:
            return "none"

        # v23.0: aplicar override de extension_enabled según estado actual.
        # Si STATE_OVERRIDES define extension_enabled=False para el estado del trade,
        # se desactiva temporalmente la extensión en el manager para este ciclo.
        # El config se restaura siempre en el bloque finally, evitando que una
        # excepción deje el manager en estado mutado para ciclos posteriores.
        _state_overrides = self._get_state_overrides(tr)
        _override_extension = _state_overrides.get('extension_enabled')
        _manager_extension_backup = None
        if (_override_extension is not None
                and hasattr(manager, 'config')
                and _override_extension != manager.config.extension_enabled):
            _manager_extension_backup = manager.config.extension_enabled
            manager.config.extension_enabled = _override_extension
            # v25.0: emitir solo cuando el estado cambia, no en cada ciclo del monitor.
            # Bug anterior: el finally restaura extension_enabled=True al final de cada
            # llamada → el siguiente ciclo ve la misma "diferencia" y re-emite el evento
            # → 12935 eventos en una sesión de 4h (1 por ciclo de 0.2s).
            # Fix: comparar con el último estado registrado en tr._tp_override_last_state.
            _current_state = getattr(tr, 'ind_state', None)
            if _current_state != tr._tp_override_last_state:
                tr._tp_override_last_state = _current_state
                self._send({
                    "event":                      "ADAPTIVE_TP_EXTENSION_OVERRIDDEN",
                    "ticket":                     tr.ticket,
                    "symbol":                     tr.symbol,
                    "state":                      _current_state,
                    "extension_enabled_override": _override_extension,
                    "original":                   _manager_extension_backup,
                })

        # Sincronizar SL actual desde TrackedPos (BE, trailing o compresión SL
        # pueden haberlo movido entre ciclos)
        manager.sync_sl(tr.virtual_sl_price)

        # Sincronizar TP si cambió desde S3 (partial close puede haberlo ajustado)
        if tr.virtual_tp > 0 and tr.virtual_tp != manager.virtual_tp:
            manager.virtual_tp = tr.virtual_tp

        last_bar = self._get_last_bar_close(tr.symbol)
        atr = tr.ind_atr if tr.ind_atr and tr.ind_atr > 0 else (tr.point * 100)
        bar_price = bid if tr.side == "BUY" else ask
        bar_close = last_bar if last_bar else bar_price

        try:
            result = manager.on_bar(
                bid=bid,
                ask=ask,
                bar_open=bar_close,
                bar_high=bar_close,
                bar_low=bar_close,
                bar_close=bar_close,
                atr=atr,
                rsi=tr.ind_rsi,
                macd_hist=tr.ind_macd_hist,
                macd_hist_prev=tr.ind_macd_hist_prev,
                volume=tr.ind_volume,
                volume_ma=tr.ind_volume_ma,
                proba_long=tr.ind_proba_long,
                proba_short=tr.ind_proba_short,
                ts=now(),
            )
        except Exception as e:
            self._send({
                "event": "ADAPTIVE_TP_ERROR",
                "ticket": tr.ticket,
                "error": str(e),
            })
            # v23.0: restaurar extension_enabled antes de salir por excepción
            if _manager_extension_backup is not None:
                manager.config.extension_enabled = _manager_extension_backup
            return "none"
        finally:
            # v23.0: garantía de restauración — se ejecuta siempre, incluso si
            # el bloque except ya restauró (idempotente: asignar el mismo valor).
            # Evita que el manager quede con extension_enabled mutado entre ciclos
            # cuando el estado del trade cambia en el siguiente tick.
            if _manager_extension_backup is not None:
                manager.config.extension_enabled = _manager_extension_backup

        if result.action == "close":
            # TP extendido o SL post-extensión alcanzado
            self._send({
                "event": "ADAPTIVE_TP_CLOSE",
                "ticket": tr.ticket,
                "symbol": tr.symbol,
                "side": tr.side,
                "reason": result.reason,
                "virtual_tp": tr.virtual_tp,
                "virtual_sl": tr.virtual_sl_price,
                "bid": bid,
                "ask": ask,
                **result.debug,
            })
            self._close_position_full(tr, bid, ask, f"ADAPTIVE_{result.reason}")
            return "closed"

        elif result.action == "update_tp":
            # Compresión del TP: acercar el objetivo al precio
            old_tp = tr.virtual_tp
            tr.virtual_tp = result.new_virtual_tp
            # Sincronizar el SL manager con el nuevo TP
            sl_mgr = self._adaptive_managers.get(tr.ticket)
            if sl_mgr:
                sl_mgr.virtual_tp = tr.virtual_tp
            self._send({
                "event": "ADAPTIVE_TP_COMPRESSION",
                "ticket": tr.ticket,
                "symbol": tr.symbol,
                "side": tr.side,
                "reason": result.reason,
                "old_virtual_tp": old_tp,
                "new_virtual_tp": tr.virtual_tp,
                "bid": bid,
                "ask": ask,
                **result.debug,
            })
            return "none"

        elif result.action == "update_both":
            # Extensión del TP: nuevo objetivo + SL comprimido simultáneo
            old_tp = tr.virtual_tp
            old_sl = tr.virtual_sl_price
            tr.virtual_tp = result.new_virtual_tp
            # SL sugerido solo se aplica si mejora el actual
            if result.new_virtual_sl and self._is_improvement(
                    tr, old_sl, result.new_virtual_sl):
                tr.virtual_sl_price = result.new_virtual_sl
                # Sincronizar el SL manager
                sl_mgr = self._adaptive_managers.get(tr.ticket)
                if sl_mgr:
                    sl_mgr.virtual_sl = tr.virtual_sl_price
                    sl_mgr.virtual_tp = tr.virtual_tp
            self._send({
                "event": "ADAPTIVE_TP_EXTENSION",
                "ticket": tr.ticket,
                "symbol": tr.symbol,
                "side": tr.side,
                "reason": result.reason,
                "old_virtual_tp": old_tp,
                "new_virtual_tp": tr.virtual_tp,
                "old_virtual_sl": old_sl,
                "new_virtual_sl": tr.virtual_sl_price,
                "bid": bid,
                "ask": ask,
                **result.debug,
            })
            return "none"

        # action == "none": comprobar si el TP fue tocado sin señales
        if (result.debug or {}).get("delegate_to_standard"):
            return "delegated"

        return "none"

    def _heartbeat_loop(self) -> None:
        """
        Envía heartbeat periódico.
        - Con posiciones abiertas: cada 10 segundos (monitorización activa)
        - En idle (sin posiciones):  cada 60 segundos (reduce ruido en log)
        """
        while True:
            has_positions = len(self._tracked) > 0
            try:
                self._send({
                    "event": "HEARTBEAT",
                    "tracked": len(self._tracked),
                    "open": len(self._known_open)
                })
            except Exception:
                pass
            time.sleep(10 if has_positions else 60)


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    import argparse
    import signal
    import sys

    parser = argparse.ArgumentParser(description="S3 Risk Management Service")
    parser.add_argument(
        "--recovery-magic", type=int, default=0,
        help="Magic number para recuperar posiciones huerfanas al arrancar (0=todas)"
    )
    args = parser.parse_args()

    service = S3Service()


    def signal_handler(sig, frame):
        print("\nShutting down S3 Service...")
        sys.exit(0)


    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    print(f"S3 Service v{S3_VERSION} started")
    print("Orders: tcp://10.1.21.25:5557")
    print("Events: tcp://10.1.21.25:5558")
    if args.recovery_magic:
        print(f"Recovery magic: {args.recovery_magic}")
    print("Press Ctrl+C to exit")

    service.run(recovery_magic=args.recovery_magic)