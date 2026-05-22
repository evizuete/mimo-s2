# INV-2026-05-22 — Investigación: hipótesis TEND_CLARA/horas/días resultó overfit

**Tipo**: Investigación (no incident — sistema operó correctamente)
**Estado**: Cerrada — decisión documentada
**Fecha**: 2026-05-22

---

## 1. Contexto

Tras observar que el 22-may-2026 fue un día perdedor (-143$) en producción y que el día 21-may fue ganador (+249$), surgió la hipótesis de que **el modelo TCN tenía problema con días TEND_CLARA o con ciertas horas/días de la semana**.

## 2. Investigación realizada

### Hipótesis A: SL/range_diario predice PnL
- **Refutada** sobre 445 trades del LOCKBOX rolling (5 días)
- Correlación SL/range vs PnL: -0.035 (esencialmente cero)
- Welch's t-test p=0.31 (no significativo)

### Hipótesis B: TEND_CLARA es estructuralmente malo
- **Soporte aparente** en LOCKBOX abr-may (n=1816 trades, 24 días):
  - 6/6 días TEND_CLARA perdieron, -1452$ acumulado
  - Aplicar filtro F1 (bloquear TEND_CLARA) mejoraba +1452$ → PnL final +1188$
- **Refutada out-of-sample** sobre marzo 2026 (n=2417 trades, 26 días):
  - Baseline marzo: **+3856$ (+30.3%)** sin filtros
  - Aplicar F1 reduce PnL en -660$ → **TEND_CLARA en marzo era RENTABLE**

### Hipótesis C: ciertas horas/días son malas (F2/F3)
- **Refutadas out-of-sample**:
  - F2 (BAD_HOURS) mejoraba +1617$ en LOCKBOX → empeora -2422$ en marzo
  - F3 (BAD_DOW) mejoraba +1352$ en LOCKBOX → empeora -416$ en marzo
  - Combinaciones F1+F2+F3 mejoraban +2549$ → empeoran -1485$ OOS

## 3. Causa real del patrón observado en LOCKBOX

El comportamiento "TEND_CLARA = pierde" en abr-may NO era estructural — fue **una mala racha estadística** de 6 sesiones específicas. La validación out-of-sample (criterio de oro para distinguir patrón real vs overfit) demostró que en otro periodo (marzo) el mismo modelo y la misma configuración generaron +30% PnL sin filtros, incluso en sesiones TEND_CLARA.

## 4. Métricas reales del modelo TCN (3 períodos)

| Periodo | Trades | PnL | wr | Comentario |
|---|---|---|---|---|
| Marzo 2026 | 2417 | **+3856$ (+30.3%)** | 34.3% | Mes excelente |
| LOCKBOX abr-may | 1816 | -264$ (-3.3%) | 31.3% | Mes regular |
| Producción real 20-22 may | 19 | +254$ (+0.4%) | 68.4% | Operativa con filtros service activos |

**Conclusión empírica**: el modelo tiene edge real positivo en agregado, con varianza significativa entre periodos.

## 5. Decisión

**NO implementar ningún filtro temporal o de carácter de día.**

Rationale:
1. Validación OOS demuestra que los patrones identificados son overfit al LOCKBOX.
2. El modelo tiene edge demostrado en marzo (+30%) sin filtros.
3. Los filtros service ya activos (TRANSITION_WEAK_SIGNAL, POST_CLOSE_COOLDOWN, etc.) ya proporcionan defensa adecuada.
4. Acumular más evidencia 4-6 semanas en producción real antes de cualquier cambio.

## 6. Lecciones aprendidas

1. **La validación out-of-sample es obligatoria** antes de promover cualquier filtro retrospectivamente descubierto.
2. **Una sola ventana temporal (LOCKBOX) puede ser engañosa**. La hipótesis "TEND_CLARA = pierde" parecía robusta con 6/6 días → era ruido estadístico.
3. **Los filtros que mejoran +2500$ en backtest pueden destruir el mismo PnL en otro periodo**. Sin OOS, hubiéramos implementado un filtro que reduciría PnL en marzo en -2658$ (F1+F2 sobre marzo OOS).
4. **El edge del modelo es heterogéneo por periodo**, no estructural. La varianza mes-a-mes (-3% a +30%) es alta pero el agregado es positivo.

## 7. Acciones de seguimiento

- [x] Acumular evidencia diaria con `000_daily_pnl_summary.sh`
- [x] Monitor de filtros bloqueadores cada hora vía cron
- [x] Smoke test post-deploy
- [ ] Tras 4-6 semanas de producción real, re-evaluar si el edge se mantiene
- [ ] Si en 4-6 semanas el PnL agregado es negativo, considerar re-Optuna ligera

## 8. Scripts creados durante la investigación

| Script | Función |
|---|---|
| `scripts/analyze_lockbox_sl_hypothesis.py` | Análisis trades.parquet × OHLCV diario |
| `scripts/validate_temporal_filters.py` | Aplicar y validar filtros F1/F2/F3 |
| `scripts/daily_pnl_summary.py` | Reporte diario/agregado con SL/range |

Estos scripts quedan disponibles para futuras investigaciones similares — la metodología (backtest → OOS validation → decisión) es reutilizable.

## 9. Referencias

- Commits clave: `9be3d5e` (daily_pnl), `6bdbb67` (analyze_lockbox), `0db1355` (validate_temporal_filters)
- Replay LOCKBOX abr-may: `/tmp/replay_lockbox_30d/`
- Replay OOS marzo: `/tmp/replay_oos_mar/`
