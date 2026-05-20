# INC-2026-05-20 — Cero trades en producción: 4 bugs en cascada + auditoría calibrador

**Severidad**: P1 grave (sistema sin operar varios días — sin pérdidas pero sin oportunidades)
**Detectado por**: observación humana (operador notó 0 trades + warnings repetidos en log)
**Detectado en**: 2026-05-19 (síntomas previos varios días)
**Resuelto en**: 2026-05-20 15:00 CEST (operativa restaurada)
**Tiempo total**: ~4h de diagnóstico activo
**Operador on-call**: Emilio Vizuete

---

## 1. Resumen ejecutivo

El sistema llevaba varios días en producción sin emitir trades pese a recibir ticks normales del mercado. Análisis reveló **cuatro problemas independientes en cascada**, cada uno enmascarando al siguiente:

1. Bug en el runtime que descartaba los thresholds del StateDetector inyectados desde `meta.json` y los recalculaba dinámicamente en cada tick → 55-68% del tiempo el sistema creía estar en VOLATILE.
2. Detector "chop" bloqueando el 15% de los ticks de forma sistemática por hardcoding `chop_block=True, chop_size_mult=0.0`.
3. Calibrador isotónico con plateaus masivos que aplastaban la señal del modelo (raw [0.20, 0.34] colapsando a un único cal en SHORT).
4. Filtro defensivo `transition_min_proba_delta=0.10` puesto por encima del P99 empírico de la distribución → bloqueo del 100% de señales en TRANSITION desde su instalación (FIX-BUG-3 del 17/04/2026).

Adicionalmente, auditoría empírica reveló que **5 umbrales más** dependen del rango del calibrador. Uno (`min_proba_edge`) se ajustó. Los otros 4 (`expansion_proba_threshold`, `compression_proba_threshold` ×2, `extension_proba_threshold`) **siempre fueron inalcanzables incluso con isotónico** y están documentados como deuda técnica.

Impacto: 0 trades durante ~7 días en sesión de mercado. **Sin pérdidas financieras**, pero pérdida de oportunidades y degradación de la confianza en el sistema.

---

## 2. Cronología

| Hora | Evento |
|---|---|
| 2026-05-13 | Deploy de TCN v4 a producción tras Lockbox OK |
| 2026-05-13 a 19 | Operativa sin trades, atribuido a "mercado en VOLATILE" |
| 2026-05-19 | Operador observa que el StateDetector marca VOLATILE 55-68% del tiempo de forma sospechosa |
| 2026-05-19 noche | Recalibración manual de thresholds en `meta.json` vía `008_recalibrate_thresholds.sh` |
| 2026-05-20 mañana | Tras restart, sigue clasificando VOLATILE en exceso. Investigación: warning `Computing thresholds DYNAMICALLY` en cada tick |
| 11:30 | **Fix #1**: bug en `pipeline_v2.py:456-463` — los thresholds inyectados no se usaban (commit `971458b`) |
| 12:45 | Tras restart aparece nuevo blocker: `strategy_gate_reason=BLOCK_CHOP` |
| 13:30 | **Fix #2**: `chop_block=False, chop_size_mult=0.5` en `trading_simulator_v3.py` (commit `0a0e394`) |
| 13:35 | Primer trade aparece como `Order:{}` en log pero cal probs idénticas (0.0758/0.1047) entre ticks |
| 13:41 | **Fix #3**: identificado calibrador isotónico saturado con plateaus enormes (SHORT: raw [0.20-0.34] → cal=0.1047 todos). Reentrenamos con Platt scaling |
| 13:48 | Primer restart con Platt falla por wrapper `PlattCalibrator` no importable. Cambiamos a `_SigmoidCalibration` directo de sklearn |
| 14:00 | Sistema procesa ticks normalmente con cal probs variables |
| 14:15 | Trades siguen sin llegar a MT5/S3. Identificado bloqueador final |
| 14:20 | **Fix #4**: `transition_min_proba_delta=0.10` mal calibrado (por encima del P99 empírico). Bajado a 0.04 (commit `5a53a77`) |
| 14:30 | Auditoría empírica del resto de umbrales que dependen del calibrador |
| 14:45 | **Fix #5**: `min_proba_edge=0.02 → 0.015` para compensar compresión del rango Platt |
| 15:00 | Operativa restaurada. Trades empezando a llegar a S3 |

---

## 3. Síntomas observados

- 162 eventos por día, **0 trades**.
- StateDetector clasificaba VOLATILE 55-68% del tiempo (precio gold ~$4500, atr_norm ~14bps, debería ser ~20-30%).
- Trazas con `Order:{...}` calculado pero S3 sin recibir nada (`tracked: 0, open: 0`).
- En el log JSONL: `block_reason=TRANSITION_WEAK_SIGNAL(delta=0.019<0.1)`, `BLOCK_CHOP`, etc.
- Calibradores isotónicos: SHORT mapeaba **raw [0.20, 0.34] → cal=0.1047 (plateau de 14 centésimas)**.

---

## 4. Investigación

### Hilo 1 — Threshold injection del StateDetector

Verificación de `meta.json` mostró thresholds correctos persistidos. Pero el warning de runtime decía "Computing thresholds DYNAMICALLY" en cada tick. Inspección de `data_pipeline_v2.py:456-463`:

```python
state_cfg_overrides = getattr(self, '_state_config_overrides', {})
state_cfg = StateConfig(**state_cfg_overrides) if state_cfg_overrides else StateConfig()
df = add_mimo_state(df, cfg=state_cfg, ...)
```

`add_mimo_state` crea internamente un **StateDetector nuevo** cada llamada. El cfg llegaba vía dict intermedio que no estaba poblado en la instancia que ejecutaba `prepare_data`.

### Hilo 2 — Chop detector

Tras fix #1, aparecía `strategy_gate_reason=BLOCK_CHOP` con alta frecuencia. Investigación: el detector `is_chop` (en `feature_builder.py:983-985`) usa percentil 85 móvil → **por construcción ~15% de las barras siempre se marcan como chop**. Y `StrategyGate` tenía hardcoded `chop_block=True, chop_size_mult=0.0` desde el initial commit (Apr 27 2026).

### Hilo 3 — Calibrador isotónico

Operador observó cal probs idénticas (0.0758/0.1047) entre ticks distintos. Diagnóstico del `IsotonicRegression`:

- LONG: solo 26 puntos de quiebre, plateau en raw [0.27, 0.32] → cal=0.124
- SHORT: solo 18 puntos de quiebre, plateau en raw [0.20, 0.34] → cal=0.1047 (¡14 centésimas!)
- SHORT cal_max global = 0.231 (nunca podía emitir cal > 23%)

Causa: `IsotonicRegression` no-paramétrica aprende escalones de los datos OOF. Si una zona tiene pocas muestras, genera plateaus inservibles. Con datos OOF n=3251 distribuidos heterogéneamente, varios plateaus enormes.

### Hilo 4 — Filtro TRANSITION_WEAK_SIGNAL

Tras fix #3, sistema construía `Order:{}` con score=0.54 pero seguía sin enviar a MT5. Investigación: en `s2_service_v2.py:910-935`, filtro post-decision exige `delta=cal_lado-cal_contra >= transition_min_proba_delta (0.10)`.

Cálculo empírico sobre datos OOF (n=72 muestras en TRANSITION):

| Percentil | iso (antiguo) | Platt (nuevo) |
|---|---|---|
| P50 | 0.029 | 0.021 |
| P90 | 0.055 | 0.040 |
| P99 | **0.090** | **0.056** |

**El umbral 0.10 estaba por encima del P99 de ambos calibradores** → bloqueaba el 100% desde su instalación. El caso histórico que justificó el filtro (3 SELL con delta≈0.088 que perdieron -675pts) era P99 puro — outlier, no la masa.

### Hilo 5 — Auditoría de otros umbrales

Tras swap iso→Platt, audité todos los thresholds que dependen de `proba_long_cal/proba_short_cal`:

| Umbral | Default | % paso Iso | % paso Platt | Status |
|---|---|---|---|---|
| `min_proba_edge` (reversal_guard) | 0.02 | LONG 27.7% / SHORT 53.3% | LONG 13.1% / SHORT 51.9% | Bajado a 0.015 |
| `expansion_proba_threshold` (adaptive_sl) | 0.45 | 0.06% | **0.00%** | Código muerto histórico |
| `compression_proba_threshold` (adaptive_sl) | 0.60 | 0.06% | **0.00%** | Código muerto histórico |
| `compression_proba_threshold` (adaptive_tp) | 0.45 | 0.06% | **0.00%** | Código muerto histórico |
| `extension_proba_threshold` (adaptive_tp) | 0.60 | 0.06% | **0.00%** | Código muerto histórico |

Los adaptive thresholds (0.45-0.60) son inalcanzables con cal_max(Platt LONG)=0.35 y cal_max(SHORT)=0.23. Ya eran inalcanzables con isotónico (0.06% paso). **Las lógicas adaptive de SL/TP basadas en proba nunca se activaron en producción**.

---

## 5. Causa raíz

**Cuatro causas independientes**, ninguna detectada en pre-producción porque cada una requería el fix de la anterior para revelarse:

### CR1 — Plumbing frágil del threshold injection
El pipeline guardaba los thresholds en TRES sitios paralelos (`self.state_detector.config.fixed_*`, `self._state_config_overrides` dict, y persistencia en meta.json). El método `prepare_data` leía el dict intermedio en lugar de la fuente fiable (`self.state_detector.config`). Cualquier desincronización entre los tres lugares producía silent failure.

### CR2 — Parámetros operativos hardcodeados sin pasar por config
`chop_block` y `chop_size_mult` estaban hardcoded en `trading_simulator_v3.py` desde el initial commit. No expuestos en `s2_config.py`, no parametrizables sin editar código. Operador no podía ajustarlos.

### CR3 — Calibrador isotónico inadecuado para señal débil
`IsotonicRegression` genera escalones grandes cuando los datos son escasos en cierta región. El SHORT con cal_max=0.231 (estructuralmente débil) tenía solo 18 puntos de quiebre, con plateaus inservibles. La elección de isotónico vs Platt no se documentó como decisión técnica; vino "por defecto" del pipeline.

### CR4 — Filtros defensivos calibrados sin verificación empírica
El umbral `transition_min_proba_delta=0.10` se introdujo tras un incident (FIX-BUG-3, 17/04/2026) que documentó la pérdida pero NO la distribución empírica. Se asumió que "delta=0.10 = señal fuerte" sin medir. El P99 empírico era 0.090 — el umbral fue puesto por encima del rango natural.

Patrón común: **defensa puesta sin medición empírica → bloqueo total cuando el comportamiento del sistema se modifica aguas arriba (ej. swap calibrador)**.

---

## 6. Resolución

### Commits aplicados

| Commit | Fix | Archivo principal |
|---|---|---|
| `971458b` | Pipeline usa `state_detector.config` directo | `mimo/data_managers/data_pipeline_v2.py` |
| `0a0e394` | Chop relajado (entrada con 0.5x size) | `mimo/strategies/trading_simulator_v3.py` |
| (sin commit, joblib) | Calibrador iso→Platt con backup | `oof_calibrator_*.joblib` |
| `5a53a77` | `transition_min_proba_delta` 0.10→0.04 | `main/s2_config.py` |
| (este commit) | `min_proba_edge` 0.02→0.015 + auditoría | `main/s2_config.py` + este doc |

### Backups creados

- `oof_calibrator_*_isotonic.bak.20260520_*.joblib` — calibradores originales (revertir si Platt en runtime resulta peor).

### Restarts necesarios

3 restarts de s2 durante el día (uno por fix). El último a las ~15:00 deja el sistema con todos los fixes activos.

---

## 7. Impacto

- **Operativa**: 0 trades durante ~7 días. Sin pérdidas pero sin captura de oportunidades.
- **Tiempo**: ~4h de diagnóstico activo del operador.
- **Confianza**: degradación moderada (sistema parecía "completamente roto" hasta entender los layers).
- **Sin posiciones abiertas** durante el incidente → sin riesgo financiero directo.

---

## 8. Acciones preventivas

### Inmediatas (este incidente)

- ✅ Audit completo de umbrales que dependen del calibrador (5 umbrales identificados, 2 ajustados, 4 documentados como deuda técnica).
- ✅ Backup explícito de calibradores antes de swap.
- ✅ Documentación del incidente (este archivo).

### Pendientes a corto plazo

- [x] **Exponer `chop_block`/`chop_size_mult` en `s2_config.py`** en lugar de hardcoded. Permite ajustar sin tocar `trading_simulator_v3.py`. → `StrategyGateConfig` añadido (commit `835abb6`).
- [x] **Test de regresión automático**: al swap calibrador, ejecutar tests que verifiquen que los umbrales aguas abajo siguen siendo alcanzables (P90 de cal_probs >= umbral). → `scripts/validate_calibrator_thresholds.py` con catálogo de 7 umbrales, exit code 0/1/2 para CI. Documentado en RUNBOOK Apéndice J.
- [x] **Smoke test post-deploy**: verificar que el sistema emite >0 trades en las primeras horas tras restart. Si no, alertar. → `scripts/smoke_test_post_deploy.py` con veredicto PASS/WAITING/FAIL, exit codes para cron, sugerencias automáticas de causa raíz. Documentado en RUNBOOK Apéndice K.
- [x] **Documentar elección de calibrador como decisión de diseño**: ¿usamos isotónico (no-paramétrico) o Platt (sigmoid)? Con qué cantidad mínima de datos OOF? → [DEC-2026-05-20](../DESIGN/DEC-2026-05-20-calibrator-choice.md) — ADR con alternativas evaluadas, rationale, criterios de elección por tamaño de OOF, y cuándo re-evaluar.

### Pendientes a medio plazo

- [ ] **Auditar las lógicas adaptive_sl/adaptive_tp**: 4 thresholds inalcanzables sugieren que esas lógicas nunca se ejecutaron. Decidir: (a) corregir thresholds para activarlas, (b) eliminar el código muerto.
- [ ] **Unificar fuente de verdad de StateDetector config**: eliminar `_state_config_overrides` y dejar `self.state_detector.config` como única fuente.
- [ ] **Dashboard de "filtros bloqueadores"**: monitor que cuente cuántos eventos bloquea cada filtro (TRANSITION_WEAK_SIGNAL, BLOCK_CHOP, REVERSAL_GUARD_*, etc.) y alerte si alguno bloquea > 95% sostenido.

---

## 9. Lecciones aprendidas

1. **Cada filtro defensivo debe documentar su distribución empírica al instalarse**. No "delta=0.10 me parece fuerte" sino "0.10 es el P90 de mi distribución observada de delta en TRANSITION (n=72, periodo X)".

2. **El cambio de calibrador es un cambio de "moneda" del sistema**. Cualquier umbral expresado en cal_probs debe revisarse. Crear checklist explícito para swap-calibrator que recorra todos los umbrales.

3. **Las defensas no usadas son las más peligrosas**. Los 4 adaptive thresholds (0.45-0.60) nunca se activaron, pero nadie lo notó porque no se loguea "este filtro nunca ha disparado". Resultado: deuda técnica acumulada que se cree activa.

4. **Tres síntomas del mismo problema parecen tres problemas distintos hasta que pelas la cebolla**. El operador inicialmente atribuía el "0 trades" a "mercado en VOLATILE". El VOLATILE era a su vez bug. Pelando, aparecen 4 layers.

5. **El JSONL estructurado fue la fuente fiable; el stdout fue engañoso**. La traza por stdout mostraba `Order:{...}` calculado, dando falsa impresión de éxito. El JSONL tenía `block_reason` correcto desde el principio. **Priorizar fuentes estructuradas para diagnóstico**.

6. **La intuición del operador fue determinante en 2 momentos**:
   - "El StateDetector marca VOLATILE demasiado" → revela CR1.
   - "Cal probs aparecen idénticas entre trazas" → revela CR3.

   Sin esas dos observaciones, el debugging hubiera tomado mucho más tiempo. **Confiar en patrones extraños observados por el operador**.

---

## 10. Anexos

### A. Distribuciones empíricas relevantes

**Distribución |Δ| en TRANSITION (n=72 OOF):**

```
percentil   iso      Platt
   P50      0.029    0.021
   P85      0.046    0.036
   P90      0.055    0.040
   P95      0.057    0.050
   P99      0.090    0.056
```

**Rangos absolutos de calibradores:**

```
Iso:    LONG [0.00, 1.00] (overfit en cola)  SHORT [0.00, 0.23]
Platt:  LONG [0.018, 0.350]                  SHORT [0.067, 0.226]
```

### B. Comandos clave de diagnóstico

```bash
# Comprobar threshold injection en runtime tras restart
grep -c "Computing thresholds DYNAMICALLY" logs/s2_*.log    # debe ser 0

# Distribución de estados clasificados
grep -oE "State: [A-Z_]+" logs/s2_*.log | sort | uniq -c | sort -rn

# Distribución de bloqueos
/path/to/python -c "
import json, collections
counts = collections.Counter()
with open('main/logs/signals_YYYYMMDD.jsonl') as f:
    for line in f:
        d = json.loads(line)
        if br := d.get('block_reason'):
            counts[br.split('(')[0]] += 1
for k,v in counts.most_common(): print(f'{v:5d}  {k}')
"

# Verificación calibradores cargados correctamente
python -c "import joblib; c=joblib.load('artifacts/.../oof_calibrator_*_long.joblib'); print(type(c).__name__)"

# Validar umbrales runtime que dependen del calibrador (acción preventiva #4)
python scripts/validate_calibrator_thresholds.py            # tabla detallada
python scripts/validate_calibrator_thresholds.py --quiet    # CI mode, exit 1 si alertas

# Smoke test post-deploy: ¿está el sistema emitiendo trades? (acción preventiva #5)
python scripts/smoke_test_post_deploy.py                    # snapshot interactivo
python scripts/smoke_test_post_deploy.py --quiet            # cron mode, exit 1 si FAIL
python scripts/smoke_test_post_deploy.py --fail-after-hours 2   # alerta tras 2h
```

### C. Referencias

- Commits del incidente: `971458b`, `0a0e394`, `5a53a77`, este commit.
- Backups calibrador: `artifacts/202500/oof/deploy_validation_combined_seed47/oof_calibrator_202500_{long,short}.isotonic.bak.20260520_*.joblib`.
- Sesión de diagnóstico: log de transcripción del 20/05/2026 con el asistente.
