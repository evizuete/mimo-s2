# DEC-2026-05-20 — Elección de calibrador de probabilidades para producción

**Estado**: Accepted
**Decisión**: Usar **Platt scaling** (`sklearn.calibration._SigmoidCalibration`) por defecto en lugar de **IsotonicRegression** para los calibradores `oof_calibrator_*.joblib`.
**Fecha**: 2026-05-20
**Autor**: Emilio Vizuete (operador) + análisis post-incident
**Contexto**: triggered por el incident [INC-2026-05-20](../INCIDENTS/INC-2026-05-20-runtime-bottlenecks.md)
**Re-evaluación**: cada 6 meses o tras cambio estructural en la distribución del modelo.

---

## 1. Contexto

### Qué hace un calibrador

El TCN emite **raw probabilities** (`oof_proba_raw`) que NO son probabilidades calibradas en el sentido frecuentista. Una raw=0.40 no significa "el evento ocurrirá el 40% de las veces" — significa "el modelo emite 0.40 para este input". El calibrador mapea raw → cal donde cal SÍ aproxima la frecuencia empírica observada en los datos de entrenamiento.

En producción:
- El calibrador se aplica tras cada predicción del modelo: `cal = calibrator.predict(raw)`.
- Los downstream filters defensivos (`transition_min_proba_delta`, `min_proba_edge`, `expansion_proba_threshold`, etc.) operan sobre `cal`, NO sobre `raw`.
- Los gates por estado del policy (`gate_by_action_and_state[*][state] = percentile`) se evalúan sobre el score derivado de `cal`.

### Por qué importa la elección de calibrador

Cambiar de calibrador cambia el rango y la distribución de `cal_probs`. Los thresholds calibrados sobre un calibrador NO son trasladables a otro sin recalibrarlos. El incident del 20/05/2026 demostró que el sistema puede quedar en estado de bloqueo total si se cambia el calibrador sin revisar todos los umbrales aguas abajo.

---

## 2. Decisión

**Usar Platt scaling** (`sklearn.calibration._SigmoidCalibration`) para los calibradores `oof_calibrator_{release}_{long,short}.joblib` en producción.

**Persistir directamente** la instancia de `_SigmoidCalibration` (no envolverla en wrappers personalizados — la deserialización desde runtime requiere clases importables, ver lección del incident).

**Re-entrenar el calibrador** con `signal` (label binario triple-barrier) sobre `oof_proba_raw` del tail de OOF (`deploy_calibration_tail_*.parquet`).

---

## 3. Alternativas consideradas

### 3.1. IsotonicRegression (no-paramétrico)

**Cómo funciona**: aprende una función monotónica escalonada que minimiza el error cuadrático entre raw y label sobre los datos OOF. Sin asunción funcional.

**Ventajas**:
- Flexible: captura no-linealidades arbitrarias en la relación raw↔frecuencia.
- Sin asunción paramétrica: ideal si el modelo está mal calibrado de formas extrañas.
- Mejor log_loss in-sample que Platt cuando hay muchos datos.

**Inconvenientes** (los que sufrimos):
- **Plateaus enormes con datos OOF escasos**. Si una región del raw tiene pocas muestras, isotónico genera un escalón único que aplasta toda esa región. En nuestro caso (n=3251 OOF, base rate ~8%):
  - LONG: 26 puntos de quiebre. Plateau en raw [0.27, 0.32] → cal=0.124 constante.
  - SHORT: 18 puntos de quiebre. Plateau en raw [0.20, 0.34] → cal=0.1047 constante (14 centésimas de raw aplastadas).
- **Overfit en colas**: con muy pocos ejemplos extremos, los puntos finales pueden saturar a 0 o 1, distorsionando la cola.
- **Rango efectivo impredecible**: cal_max puede ser <0.25 o llegar a 1.0 según los datos. Esto dificulta calibrar umbrales aguas abajo de forma robusta.

### 3.2. Platt scaling / sigmoide (paramétrico) — **ELEGIDO**

**Cómo funciona**: ajusta una sigmoide `cal = sigmoid(a*raw + b)` con dos parámetros (`a`, `b`) por máxima verosimilitud sobre los datos OOF.

**Ventajas**:
- **Suave**: monotónico continuo, sin plateaus. Cada raw distinta da una cal distinta.
- **Rango acotado** y predecible: cal siempre en (0, 1), nunca exactamente 0 o 1.
- **Solo 2 parámetros**: muy robusto a pocas muestras, no sobreajusta.
- **Deserialización trivial**: `_SigmoidCalibration` es clase pública-ish de sklearn, importable desde cualquier proceso (la primera implementación con un wrapper `PlattCalibrator` casero falló por no-importabilidad).

**Inconvenientes** (asumidos):
- **Asume forma sigmoide**: si la relación real raw↔frecuencia es muy no-sigmoide, Platt no la capturará bien. En nuestro caso parece OK (raw y label ambos en cola estrecha).
- **Comprime extremos**: cal_max para Platt LONG ≈ 0.35, para SHORT ≈ 0.23. Esto **no es un bug** — refleja la realidad estadística (SHORT casi nunca acierta con confianza alta en OOF), pero invalida umbrales calibrados asumiendo cal puede llegar a 0.6-0.8.
- **Log_loss in-sample ligeramente peor** que isotónico (≈+1.7% en nuestros datos). Esperable porque isotónico sobreajusta in-sample.

### 3.3. Beta calibration

Hubiera sido la siguiente opción si Platt no funcionase. Más flexible que Platt (3 parámetros) pero más restrictivo que isotónico. Útil cuando la curva calibrada tiene asimetría que Platt no captura.

**No evaluada** en este ciclo. Considerar para futuro si Platt resulta inadecuado.

### 3.4. Temperature scaling

Solo escala una temperatura `cal = sigmoid(raw / T)`. Demasiado restrictivo para nuestro caso (NO calibra el sesgo, solo la confianza). Útil principalmente en clasificadores softmax multi-clase, no en binario aislado.

**Descartada** sin evaluar.

---

## 4. Rationale (por qué Platt sobre isotónico)

### Razón 1: nuestros datos OOF son escasos relativamente

n=3251 muestras con base rate ~8% significa ~260 positivos. Para un calibrador no-paramétrico de 18-26 escalones, la masa de datos en cada escalón es muy pequeña (≈10-15 positivos por bin). Esto produce los plateaus que el incident reveló.

Heurística: **isotónico es defendible con >10,000 muestras OOF y base rate >15%**. Por debajo, Platt es más robusto.

### Razón 2: la calibración alimenta umbrales defensivos

Si los umbrales aguas abajo asumen cierto rango/distribución de cal, los plateaus del isotónico crean discontinuidades problemáticas. Un raw=0.265 y raw=0.270 dan cal idénticas en un plateau, pero los downstream filters NO distinguen — la información del modelo se pierde silenciosamente.

Con Platt, cada raw distinta da cal distinta (suave). Los umbrales operan sobre información continua del modelo, no sobre cuantos.

### Razón 3: SHORT es estructuralmente débil — quieres rangos honestos

Con isotónico, SHORT podía emitir cal=1.0 en algún outlier (por overfit a un único ejemplo extremo). Eso engaña a los umbrales aguas abajo que asumen cal=1.0 → señal muy fuerte. Con Platt, cal_max(SHORT)=0.23 — **la realidad estadística está respetada**: el modelo nunca emite SHORT con confianza alta, y los umbrales reflejan esto.

### Razón 4: deserialización robusta

Trying con un wrapper `PlattCalibrator` casero falló porque la clase no era importable desde `s2_main.py`. Lección: **usar clases de sklearn directamente** simplifica la persistencia/carga entre procesos. `_SigmoidCalibration` cumple, `IsotonicRegression` también, pero cualquier wrapper personalizado complica.

---

## 5. Consecuencias

### Pros (asumidos al elegir Platt)

- ✅ Sin plateaus → información del modelo preservada hasta los umbrales.
- ✅ Rango acotado y predecible → más fácil diseñar umbrales aguas abajo.
- ✅ Robusto a pocas muestras OOF (típico en quant trading).
- ✅ Deserialización trivial entre procesos.
- ✅ Refleja honestamente la debilidad estructural del modelo en SHORT.

### Cons (asumidos al elegir Platt)

- ⚠️ Log_loss in-sample ligeramente peor (~+1.7% relativo) — aceptable.
- ⚠️ Asume forma sigmoide — si futura distribución del modelo cambia (e.g., bimodal en raw), considerar beta calibration.
- ⚠️ Comprime extremos — los umbrales del runtime ahora viven en rango cal ∈ [0.07, 0.35] aprox. Ajustar todos los thresholds aguas abajo (ver acción #4 del incident).

### Acciones implementadas para gestionar los cons

1. Script `scripts/validate_calibrator_thresholds.py` ([Apéndice J del RUNBOOK](../RUNBOOK.md)) que valida automáticamente todos los umbrales aguas abajo tras cualquier cambio de calibrador. Catálogo de 7 umbrales actualmente.

2. Backup del calibrador anterior (isotónico) en cada swap: `oof_calibrator_*.isotonic.bak.YYYYMMDD_HHMMSS.joblib`. Rollback trivial si Platt resulta peor en producción.

3. Documentación de los rangos empíricos de cal_probs en este documento y en el incident report.

---

## 6. Criterios para elegir un calibrador (futura referencia)

### Cuándo usar Platt scaling (default)

- n_OOF < 10,000 muestras útiles.
- base rate < 15%.
- distribución de raw concentrada en una banda estrecha (no cola larga).
- prioridad en estabilidad y suavidad sobre la flexibilidad máxima.
- queremos persistir/cargar el calibrador entre procesos sin wrappers.

### Cuándo considerar IsotonicRegression

- n_OOF > 10,000 muestras.
- base rate > 15%.
- distribución de raw cubre todo [0, 1] uniformemente.
- la métrica clave es log_loss / Brier, no la suavidad/predictibilidad.
- aceptamos plateaus locales (los downstream filters trabajan sobre raw o sobre score post-procesado, NO directamente sobre cal).

### Cuándo considerar Beta calibration

- Platt da un mal log_loss y la curva calibrada se ve claramente asimétrica.
- Buen balance entre robustez (3 params) y flexibilidad.

### Cuándo *no* usar ningún calibrador

- El modelo ya emite probabilidades bien calibradas naturalmente (raro).
- El downstream usa los scores como ranking, no como probabilidades (orden importa, magnitud no).

---

## 7. Cuándo re-evaluar esta decisión

- **Cada 6 meses** como revisión rutinaria.
- **Tras cambio mayor en la arquitectura del modelo** (e.g., transición de CNN-LSTM a TCN, cambio de loss).
- **Tras un walk-forward** que muestre drift significativo en la distribución de raw.
- **Si el sistema entra en estado de bloqueo total** sin causa aparente, revisar primero el rango de cal_probs (síntoma del incident actual).

---

## 8. Cantidad mínima de datos OOF para calibrar

Empíricamente, sobre los datos de este deploy (n=3251):

| n_OOF | Apropiado para | Recomendación |
|---|---|---|
| < 500 | nada serio | demasiado poco — recalibrar con tail más largo |
| 500-2000 | Platt | aceptable para Platt si base rate > 5% |
| 2000-10000 | **Platt (recomendado)** | nuestro caso actual |
| 10000-50000 | Platt o isotónico (verificar plateaus) | comparar log_loss out-of-time |
| > 50000 | isotónico (si log_loss mejora >5%) | considerar para escalado masivo |

Para nuestro setup actual: **mantener n_OOF ≈ 21-45 días de tail** (3000-7000 muestras), Platt como default.

---

## 9. Referencias

### Código

- `mimo/oof/probs_calibration.py` — pipeline de fit del calibrador en training/OOF
- `mimo/helpers/helper.py:32-34` — carga del calibrador en runtime
- `mimo/strategies/trading_simulator_v3.py:1121` — invocación `.predict(raw)` en runtime
- `scripts/validate_calibrator_thresholds.py` — test de regresión de umbrales
- `artifacts/{release}/oof/{deploy}/oof_calibrator_{release}_{side}.joblib` — calibradores persistidos

### Documentación relacionada

- [INC-2026-05-20](../INCIDENTS/INC-2026-05-20-runtime-bottlenecks.md) — incident que motivó esta decisión
- [RUNBOOK Apéndice J](../RUNBOOK.md) — validación de umbrales tras swap de calibrador
- [RUNBOOK Apéndice K](../RUNBOOK.md) — smoke test post-deploy

### Bibliografía

- Platt, J. (1999). "Probabilistic Outputs for Support Vector Machines and Comparisons to Regularized Likelihood Methods".
- Niculescu-Mizil, A., & Caruana, R. (2005). "Predicting good probabilities with supervised learning". ICML 2005. — el paper canónico comparando Platt vs isotónico en distintos tamaños de dataset.
- [sklearn.calibration._SigmoidCalibration](https://github.com/scikit-learn/scikit-learn/blob/main/sklearn/calibration.py) — implementación de referencia.
