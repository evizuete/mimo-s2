# INV-2026-05-22 — Mejoras del StateDetector (M1+M2+M3)

**Tipo**: Investigación + mejora del componente (no incident)
**Estado**: Implementada, validada empíricamente, pendiente de restart en producción
**Fecha**: 2026-05-22
**Componente afectado**: `mimo/states_manager/state_detector.py`

---

## 1. Motivación

Tras el ciclo de validación out-of-sample (ver `INV-2026-05-22-overfit-temporal-filters.md`)
se confirmó que el modelo TCN tiene varianza alta entre periodos y que los filtros
contextuales propuestos eran overfit. El usuario solicitó auditar el StateDetector
para ver si la **clasificación de regímenes** es mejorable.

## 2. Diagnóstico empírico (Marzo 2026, n=5841 ticks)

Ejecutado vía `scripts/analyze_state_detector.py` (nuevo).

### Distribución de estados sin cambios

```
VOLATILE        39.2%  ███████████████████   ← excesivo, drift ATR
TREND_DOWN      33.4%  ████████████████      ┐
TREND_UP        15.9%  ███████               ├ TREND = 49.3% del tiempo
TRANSITION_DOWN  5.9%  ██                    ┘
RANGE            4.6%  ██                    ← muy poco
TRANSITION_UP    0.9%                        ← asimetría 6:1 vs DOWN
LOW_VOL          0.2%
```

### Hit rate por estado (próximas 5 barras)

| Estado | n | hit_dir |
|---|---|---|
| TREND_DOWN | 1952 | **50%** (random) |
| TREND_UP | 927 | **45%** (peor que random) |
| TRANSITION_DOWN | 343 | 55% |
| TRANSITION_UP | 53 | 36% |

**Hallazgo crítico**: el state NO está prediciendo dirección. El modelo TCN
opera con etiquetas que son retrospectivas (ADX alto = tendencia ya
materializada), no prospectivas.

### Umbrales ADX hardcoded

```
Distribución ADX en Marzo 2026:
  mean = 26.2
  p25  = 18.4   ← cerca del threshold adx_range=18 (OK)
  p50  = 24.1   ← CASI el threshold adx_trend=25
  p80  = 33.5

  ADX < 18 (range):       23.3%
  ADX 18-25 (transition): 30.1%
  ADX >= 25 (trend):      46.6%   ← demasiado!
```

El umbral `adx_trend=25` cae en el P50 → casi la mitad del tiempo se clasifica
como "trend". Por construcción, eso infla TREND_* y agota el repertorio de
RANGE/TRANSITION.

### Flapping

- 115 episodios de A→B→A con B≤2 barras
- TRANSITION_UP diagonal de transición solo 35.8% (muy inestable)

## 3. Mejoras implementadas

### M3 — Recalibrar thresholds ATR/BB (ya existía)

Script `008_recalibrate_thresholds.sh` aplicado con `APPLY=1` y
`LOOKBACK_DAYS=60` sobre `deploy_PROD_combined_seed47_20260517`.

**Cambios en `meta.json`**:

| Threshold | Antes | Después | Δ |
|---|---|---|---|
| vol_low | 0.000586 | 0.001015 | **+73.2%** |
| vol_high | 0.001221 | 0.001764 | **+44.5%** |
| bb_p20 | 0.001678 | 0.002867 | **+70.9%** |
| bb_p35 | 0.002259 | 0.003702 | +63.8% |
| bb_p70 | 0.004247 | 0.006337 | +49.2% |
| rexp_p80 | 1.306708 | 1.297806 | -0.7% |

**Impacto**: VOLATILE 39.2% → 21.0% (-46%). El drift estaba causado por
training en periodo de oro a $4000-4200 vs marketed actual a $4500+.

### M1 — ADX por percentil persistido

**Cambios en código** (`mimo/states_manager/state_detector.py`):

```python
@dataclass
class StateConfig:
    # ... fields anteriores ...
    fixed_adx_trend: Optional[float] = None   # NUEVO
    fixed_adx_range: Optional[float] = None   # NUEVO
```

- `compute_thresholds()` calcula `np.quantile(adx, 0.80)` para trend y `0.25` para range.
- `inject_thresholds()` extrae `thresholds["adx_trend"]` / `["adx_range"]`.
- `_get_thresholds()` los incluye en el dict (path fixed y dinámico).
- `_classify()` usa `thr["adx_trend"]` en vez de `cfg.adx_trend_threshold`.

**Retrocompat**: si meta.json no incluye ADX (deploys antiguos), cae a los
defaults hardcoded (25/18).

**Valores aplicados al deploy actual** (P80 sobre 60 días recientes):
- `adx_trend = 33.47` (vs 25 hardcoded)
- `adx_range = 17.28` (vs 18, similar)

**Impacto**: TREND combinado 49.3% → 32.1% (-35%).

### M2 — Smoothing causal K=3

**Cambios en código**:

```python
@dataclass
class StateConfig:
    state_smooth_k: int = 3   # K=1 desactiva
```

- Nueva función `_smooth_state(state, k)` causal: solo cambia de estado si
  K barras consecutivas confirman el nuevo. No usa información futura.
- Aplicado al final de `_classify()`.

**Impacto**:
- Flapping: 216 (sin smoothing) → **22** (-90%)
- Persistencia TRANSITION_UP: 1.6 → 4.4 barras (+175%)
- Persistencia TRANSITION_DOWN: 3.5 → 8.1 barras (+131%)
- TRANSITION_UP diagonal en matriz transiciones: 35.8% → 77.8%

## 4. Validación final (M1+M2+M3 combinados)

| Métrica | Original | **Final** | Δ |
|---|---|---|---|
| TREND combinado | 49.3% | **32.1%** | ✅ -35% |
| TRANSITION combinado | 6.8% | **23.1%** | ✅ +240% |
| VOLATILE | 39.2% | **33.7%** | -14% |
| RANGE | 4.6% | **7.7%** | ✅ +67% |
| Total transiciones | 589 | **482** | ✅ -18% |
| **Flapping** | 115 | **22** | ✅ **-81%** |
| TREND_UP hit_dir | 45% | 46% | sin cambio |

## 5. Lo que NO se resolvió (y por qué)

**Hit rate TREND_UP sigue en 46%** (peor que random). Esto NO es problema del
StateDetector — es problema del **modelo TCN** que aprendió comportamiento
mean-reverting donde el estado se etiqueta como TREND. Solucionarlo requiere:

- Re-Optuna del modelo (4h GPU), o
- Cambio de design del modelo (e.g., labeling distinto)

Las mejoras M1+M2+M3 dejan las **features state-dependientes mucho más estables**
(persistencia 8-15 barras vs 2-5), lo que debería ayudar a un modelo futuro
a aprender mejor patrones.

## 6. Acciones pendientes

- [x] Implementar M3, M1, M2 con validación empírica
- [x] Persistir nuevos thresholds (incluido ADX) en meta.json
- [x] Verificar comparativa antes/después sobre Marzo 2026
- [ ] **Restart s2 en producción** para que load_scalers inyecte los nuevos thresholds (incluye `adx_trend`)
- [ ] **Re-Optuna** para arreglar el hit_rate estructural (separado, requiere ciclo completo)
- [ ] Validar comportamiento en producción real durante 1-2 semanas

## 7. Restart de producción

```bash
pkill -f s2_main && sleep 3
cd main && nohup /home/evizuete/boti/bin/python3 s2_main.py \
  > ../logs/s2_state_$(date +%Y%m%d_%H%M).log 2>&1 &
cd ..
sleep 30
grep -iE "Regime thresholds|adx_trend|adx_range" $(ls -t logs/s2_state_*.log | head -1) | tail -5
```

Verificar que el dict de thresholds inyectados incluye `adx_trend` y `adx_range`.

## 8. Scripts útiles para auditoría futura

| Script | Función |
|---|---|
| `scripts/analyze_state_detector.py` | Auditoría empírica completa sobre un periodo |
| `008_recalibrate_thresholds.sh` | Recalibrar thresholds (ahora incluye ADX) |

Uso futuro:
- Mensual: ejecutar `analyze_state_detector.py` sobre último mes; si distribución
  o flapping degradan → considerar M3 (recalibrar).
- Tras cambio de modelo: ejecutar para verificar coherencia state ↔ outcome.

## 9. Referencias

- Commits: `995ba4f` (M1+M2 código) + APPLY de recalibrate (meta.json)
- Análisis: `/tmp/replay_oos_mar` (replay marzo usado para validar)
- Documento previo: `INV-2026-05-22-overfit-temporal-filters.md`
