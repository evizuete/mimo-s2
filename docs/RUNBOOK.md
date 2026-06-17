# S2 Release 202500 — Runbook operativo

**Versión**: 1.0  
**Última actualización**: 2026-05-12  
**Scope**: ciclo completo desde Optuna hasta producción + monitorización  
**Audiencia**: tú mismo + cualquier operador futuro

---

## Tabla de contenidos

1. [Glosario y conceptos](#glosario-y-conceptos)
2. [Fase 0 — Setup y prerrequisitos](#fase-0--setup-y-prerrequisitos)
3. [Fase 1 — Optimización de hyperparams (Optuna)](#fase-1--optimización-de-hyperparams-optuna)
4. [Fase 2 — Entrenamiento (specialists)](#fase-2--entrenamiento-specialists)
5. [Fase 3 — Calibración y policy tuning](#fase-3--calibración-y-policy-tuning)
6. [Fase 4 — Validación en LOCKBOX](#fase-4--validación-en-lockbox)
7. [Fase 5 — Production refit (datos completos)](#fase-5--production-refit-datos-completos)
8. [Fase 6 — Despliegue a producción](#fase-6--despliegue-a-producción)
9. [Fase 7 — Monitorización continua](#fase-7--monitorización-continua)
10. [Fase 8 — Checklist pre-dinero real](#fase-8--checklist-pre-dinero-real)
11. [Apéndice A — Rollback](#apéndice-a--rollback)
12. [Apéndice B — Lecciones aprendidas](#apéndice-b--lecciones-aprendidas)

---

## Glosario y conceptos

| Término | Significado |
|---|---|
| **TRAIN** | Periodo de datos usado para entrenar el modelo neural |
| **HOLDOUT** | Periodo usado para evaluar (OOF metrics) y calibrar (tail isotonic) — NO entrena el modelo |
| **TAIL** | Subset final del HOLDOUT (21d-45d) usado para isotonic calibration |
| **LOCKBOX** | Periodo INTOCABLE hasta validación final. La prueba definitiva antes de producción |
| **Specialist** | Modelo multitask entrenado con hyperparams específicos para un side (LONG o SHORT) |
| **Deploy** | Conjunto de artefactos (modelo + scalers + calibradores + policy) listos para producción |
| **ECE** | Expected Calibration Error. <0.05 buena calibración, >0.10 problemática |
| **Champion config** | Mapping JSON (side, state) → calibrador óptimo, salida de `simulate_calibrators` |

### Arquitectura temporal completa
DATOS TOTALES (ej: 2024-01-01 → 2026-05-10)
│
├── TRAIN (2024-01-01 → 2025-10-30, 22 meses)
│ └─ entrena el modelo neural
│
├── HOLDOUT (2025-11-01 → 2026-04-10, ~5 meses)
│ ├─ OOF metrics para hyperparam search
│ └─ TAIL (últimos 21-45d) → isotonic calibration
│
└── LOCKBOX (2026-04-11 → 2026-05-10, 30d)
└─ validación FINAL antes de producción (intocable hasta el final)

Tras validación exitosa: refit con datos completos hasta el último día → producción.
---
## Fase 0 — Setup y prerrequisitos
**Objetivo**: dejar el entorno limpio antes de empezar un ciclo de re-entrenamiento.
### Checklist
- [ ] Repo actualizado: `git status` clean, `git pull origin multitask`
- [ ] Branch nueva para el experimento: `git checkout -b retrain-YYYYMMDD`
- [ ] Decidir split temporal (revisar `--from` y `--to` en cada comando)
- [ ] Verificar disponibilidad de datos:
  ```bash
  python -c "
  from mimo.data_managers.databases import Database
  from mimo.data_managers.data_manager import DataManager
  db = Database()
  dm = DataManager.from_database_historical_2(db, from_date='2024-01-01', to_date='2026-05-10')
  print(f'Filas: {len(dm.df):,}')
  print(f'Rango: {dm.df.time.min()} → {dm.df.time.max()}')
  "
sin completar
Espacio en disco: mínimo 5GB libres en artifacts/
sin completar
GPU disponible: nvidia-smi
sin completar
Backup del estado de producción actual:
cp main/s2_main.py main/s2_main.py.backup_$(date +%Y%m%d)
cp config/decision_policies_config_202500.py \
   config/decision_policies_config_202500_backup_$(date +%Y%m%d).py
sin completar
Documentar versión de Python y deps clave:
pip list | grep -E "(tensorflow|sklearn|pandas|numpy|optuna|joblib)" > docs/env_$(date +%Y%m%d).txt
Variables del experimento
Define al inicio y reusa en todos los comandos:

export RELEASE=202500
export TAG=rw_both_Lvol_boost_td_down_h3_Svol_boost_h3
export SEED=47
export TRAIN_FROM=2024-01-01
export TRAIN_TO=2025-10-30
export HOLDOUT_FROM=2025-11-01
export HOLDOUT_TO=2026-04-10      # ← validación
export PROD_HOLDOUT_TO=2026-05-10  # ← refit final
export LOCKBOX_FROM=2026-04-11
export LOCKBOX_TO=2026-05-10
export TAIL_DAYS=21                # 21 o 45 según experimentación
Fase 1 — Optimización de hyperparams (Optuna)
Cuándo ejecutar:

Primera vez que entrenas el release
Trimestralmente o tras drift estructural sostenido
Cuando la arquitectura del modelo cambia
Cuándo SALTAR:

Si tienes un best_per_side.json reciente y los resultados de monitorización son sanos
Re-Optuna NO es gratis: 4-12h GPU
Comandos
# Optuna OOF search (50-100 trials para buen muestreo)
python -m mimo.oof.main_oof_regime_weights_v7 \
  --release ${RELEASE} \
  --base-tf 5min \
  --target-type multitask \
  --side both \
  --variant-long vol_boost_td_down \
  --variant-short vol_boost \
  --label-horizon-long 3 \
  --label-horizon-short 3 \
  --train-from ${TRAIN_FROM} \
  --train-to ${TRAIN_TO} \
  --holdout-from ${HOLDOUT_FROM} \
  --holdout-to ${HOLDOUT_TO} \
  --use-tpe \
  --optuna-trials 100 \
  --objective ev_net \
  --cost-per-signal 0.05 \
  --ev-min-signals 100 \
  --max-drawdown-R 30 \
  --ev-thr-lo 0.10 \
  --ev-thr-hi 0.40 \
  --oof-epochs 120 \
  --oof-patience 15

# Extracción del mejor trial por side
python -m mimo.oof.extract_best_per_side \
  --release ${RELEASE} \
  --study-tag ${TAG} \
  --out artifacts/${RELEASE}/oof/${TAG}/reports/best_per_side.json
Outputs esperados
sin completar
artifacts/${RELEASE}/oof/${TAG}/reports/best_per_side.json existe
sin completar
Contenido del JSON: estructura {top_long: [...], top_short: [...]} con métricas EV_net positivas para ambos sides
sin completar
OOF metrics razonables (AUC-PR > base_rate × 1.2, ECE < 0.05)
Tiempo estimado
Trials	Tiempo (1× RTX 6000)
30	~3h
50	~5h
100	~10h
Decisión gate
sin completar
EV_net (R) del top trial del side LONG: ¿es positivo y > +0.05R?
sin completar
EV_net (R) del top trial del side SHORT: ¿es positivo y > +0.05R?
sin completar
Si no → repetir con más trials o cambiar grid de hyperparams
sin completar
Si sí → continuar a Fase 2
Fase 2 — Entrenamiento (specialists)
Objetivo: entrenar dos modelos multitask, cada uno con los hyperparams óptimos de su side.

Pre-requisito
sin completar
best_per_side.json existe y validado en Fase 1
Preservar artefactos previos
# Si ya existen dirs de seed=47, archívalos para no perderlos
for SIDE in long short; do
  OLD=artifacts/${RELEASE}/oof/${TAG}_${SIDE}_specialist_seed${SEED}
  if [ -d "$OLD" ]; then
    mv "$OLD" "${OLD}_$(date +%Y%m%d)"
  fi
done
Comando
python -m mimo.oof.train_specialist \
  --best-per-side-json artifacts/${RELEASE}/oof/${TAG}/reports/best_per_side.json \
  --side both \
  --release ${RELEASE} \
  --base-tf 5min \
  --target-type multitask \
  --variant-long vol_boost_td_down \
  --variant-short vol_boost \
  --label-horizon-long 3 \
  --label-horizon-short 3 \
  --train-from ${TRAIN_FROM} \
  --train-to ${TRAIN_TO} \
  --holdout-from ${HOLDOUT_FROM} \
  --holdout-to ${HOLDOUT_TO} \
  --objective ev_net \
  --cost-per-signal 0.05 \
  --max-drawdown-R 30 \
  --oof-epochs 120 \
  --oof-patience 15 \
  --seed ${SEED}
Outputs esperados
sin completar
artifacts/${RELEASE}/oof/${TAG}_long_specialist_seed${SEED}/
sin completar
artifacts/${RELEASE}/oof/${TAG}_short_specialist_seed${SEED}/
sin completar
Cada dir contiene: model_${RELEASE}_multitask.keras, oof_calibrator_*.joblib, scalers, holdout_predictions, percentiles, reports/
Tiempo estimado: ~15-17 min
Verificación
for SIDE in long short; do
  SD=artifacts/${RELEASE}/oof/${TAG}_${SIDE}_specialist_seed${SEED}
  echo "── ${SIDE} ──"
  ls "$SD/model_${RELEASE}_multitask.keras" 2>&1 | head -1
  ls "$SD/data/holdout_predictions_${RELEASE}_${SIDE}.parquet" 2>&1 | head -1
done
Fase 3 — Calibración y policy tuning
Objetivo: producir un deploy combinado listo para validar, con thresholds calibrados sobre la tail.

3.1 — Production deploy training × 2
for SIDE in long short; do
  python -m mimo.oof.resume_deploy_full_v6_multitask \
    --release ${RELEASE} \
    --target-type multitask \
    --base-tf 5min \
    --variant-long vol_boost_td_down --variant-short vol_boost \
    --label-horizon-long 3 --label-horizon-short 3 \
    --train-from ${TRAIN_FROM} \
    --holdout-from ${HOLDOUT_FROM} \
    --holdout-to ${HOLDOUT_TO} \
    --deploy-calib-days ${TAIL_DAYS} \
    --train-artifacts-subdir ${TAG}_${SIDE}_specialist_seed${SEED} \
    --deploy-subdir deploy_validation_${SIDE}_specialist_seed${SEED} \
    --locked-params-json artifacts/${RELEASE}/oof/${TAG}/reports/best_per_side.json \
    --locked-side-key ${SIDE} \
    --objective ev_net --cost-per-signal 0.05 --max-drawdown-R 30
done
Tiempo: ~10 min por side, ~20 min total.

3.2 — Merge
python -m mimo.oof.merge_specialists \
  --release ${RELEASE} \
  --long-dir  artifacts/${RELEASE}/oof/deploy_validation_long_specialist_seed${SEED} \
  --short-dir artifacts/${RELEASE}/oof/deploy_validation_short_specialist_seed${SEED} \
  --out-dir   artifacts/${RELEASE}/oof/deploy_validation_combined_seed${SEED}
3.3 — (OPCIONAL) Exploración de calibradores alternativos
⚠️ Lección aprendida importante: cambiar el calibrador puede romper el operating point del policy. NO ADOPTAR un calibrador alternativo sin validar PnL end-to-end PRIMERO.

Si quieres explorar:

# Genera ghost predict sobre HOLDOUT
python scripts/ghost_predict.py \
  --release ${RELEASE} \
  --deploy-subdir deploy_validation_combined_seed${SEED} \
  --from ${HOLDOUT_FROM} --to ${HOLDOUT_TO} \
  --include-tail \
  --out /tmp/ghost_holdout.parquet

# Simula distintos calibradores
python scripts/simulate_calibrators.py \
  --release ${RELEASE} \
  --specialist-tag ${TAG} \
  --seed ${SEED} \
  --test-parquet /tmp/ghost_holdout.parquet \
  --out-report reports/cal_sim.csv
Decisión gate:

ECE del baseline (iso_21d) < 0.05 en ambos sides → NO cambies el calibrador. La calibración es ya razonable.
ECE del baseline > 0.10 en algún side → considera cambio, pero validar PnL antes de adoptar.
Si decides cambiar:

sin completar
Crea config Nivel 1 hybrid (mismo método uniforme por side)
sin completar
Aplica swap: python scripts/swap_calibrators_per_champion.py --champion-config ... --deploy-dir ...
sin completar
CRÍTICO: re-genera el policy stub tras el swap (el operating point cambió)
3.4 — Select thresholds & policy
python -m mimo.oof.select_thresholds_from_tail \
  --release ${RELEASE} \
  --deploy-dir artifacts/${RELEASE}/oof/deploy_validation_combined_seed${SEED} \
  --side both \
  --tp-long 2.0 --sl-long 0.8 --horizon-long 3 \
  --tp-short 2.0 --sl-short 0.8 --horizon-short 3 \
  --cost 0.05 --thr-lo 0.10 --thr-hi 0.45 --n-points 70 --min-signals 30 \
  --from-db

python -m mimo.oof.compute_state_percentiles \
  --release ${RELEASE} \
  --deploy-dir artifacts/${RELEASE}/oof/deploy_validation_combined_seed${SEED} \
  --emit-config-stub \
  --out-stub config/decision_policies_config_${RELEASE}_validation.py
Outputs Fase 3
sin completar
artifacts/${RELEASE}/oof/deploy_validation_combined_seed${SEED}/ con todos los artefactos
sin completar
percentiles_${RELEASE}_long.json y _short.json con _meta.selected_threshold mutado por EV-net
sin completar
config/decision_policies_config_${RELEASE}_validation.py generado
Fase 4 — Validación en LOCKBOX
Objetivo: comprobar en datos JAMÁS vistos por el deploy que el sistema funciona.

⚠️ Esta es la fase más importante. NO promuevas a producción si no pasa.

4.1 — Replay sobre LOCKBOX
python scripts/replay_s2_202500.py \
  --release ${RELEASE} \
  --deploy-subdir deploy_validation_combined_seed${SEED} \
  --policy-config decision_policies_config_${RELEASE}_validation \
  --warmup-from 2026-03-25 \
  --from ${LOCKBOX_FROM} --to ${LOCKBOX_TO} \
  --out /tmp/replay_lockbox
Output esperado (/tmp/replay_lockbox/):

summary.json con métricas agregadas
trades.parquet con cada trade
equity_curve.parquet
daily_breakdown.csv ← útil para identificar días anómalos
4.2 — Diagnóstico de drift
python scripts/ghost_predict.py \
  --release ${RELEASE} \
  --deploy-subdir deploy_validation_combined_seed${SEED} \
  --from ${LOCKBOX_FROM} --to ${LOCKBOX_TO} \
  --include-tail \
  --out /tmp/ghost_lockbox.parquet

python -m mimo.oof.shift_analyzer.analyze_calibration_mimo \
  --input /tmp/ghost_lockbox.parquet \
  --score-col oof_proba_cal --target-col signal \
  --state-col state --time-col time \
  --period-col period \
  --output-dir reports/lockbox_cal

python -m mimo.oof.shift_analyzer.analyze_score_to_pnl \
  --input /tmp/ghost_lockbox.parquet \
  --score-col oof_proba_cal --target-col signal \
  --pnl-col R_multiple --state-col state --time-col time \
  --period-col period \
  --output-dir reports/lockbox_score_pnl
4.3 — Control: baseline anterior sobre el mismo LOCKBOX
Replay del deploy anterior (en producción) sobre la misma ventana para tener comparativa:

python scripts/replay_s2_202500.py \
  --release ${RELEASE} \
  --deploy-subdir deploy_PROD_combined_specialists_seed${SEED}  # ← deploy actual prod
  --policy-config decision_policies_config_${RELEASE}_PROD \
  --warmup-from 2026-03-25 \
  --from ${LOCKBOX_FROM} --to ${LOCKBOX_TO} \
  --out /tmp/replay_lockbox_baseline
Decisión gate
Criterio	OK	Marginal	KO
PnL% LOCKBOX	> 0%	> -5%	< -5%
MDD% LOCKBOX	> -10%	> -15%	< -15%
ECE en holdout	< 0.05	< 0.08	> 0.08
AUC-PR vs base rate	> 1.2×	> 1.0×	< 1.0×
Weeks positive	≥75%	≥50%	<50%
PnL nuevo vs baseline	≥ baseline	dentro ±2pp	< baseline -5pp
Algún estado tóxico	No	1 estado	≥2 estados con PnL/trade < -0.10R
Acción:

Todo OK → continúa a Fase 5
Mayoría OK + 1-2 Marginal → continúa pero documenta áreas de vigilancia
Algún KO → NO promover. Investiga raíz, itera (Fase 2 con diferente seed, o re-Optuna)
Verificación de coherencia
sin completar
El número de trades por side está dentro de rangos esperados (no 95% un side, no 5% otro)
sin completar
El daily_breakdown.csv no muestra todo el PnL viniendo de 1-2 días outlier
sin completar
El TREND_DOWN no es masivamente negativo (estado tóxico recurrente)
Fase 5 — Production refit (datos completos)
Objetivo: re-entrenar el MISMO modelo (misma arquitectura, mismos hyperparams) usando TODOS los datos disponibles, incluyendo el LOCKBOX.

⚠️ Solo ejecutar si Fase 4 pasó la decisión gate. Si Fase 4 falla, NO incorporar LOCKBOX al entrenamiento.

Por qué este paso
El LOCKBOX cumplió su función (validar la arquitectura). El modelo de producción debe estar lo más actualizado posible. Usar todos los datos disponibles maximiza información.

Comandos
for SIDE in long short; do
  python -m mimo.oof.resume_deploy_full_v6_multitask \
    --release ${RELEASE} \
    --target-type multitask \
    --base-tf 5min \
    --variant-long vol_boost_td_down --variant-short vol_boost \
    --label-horizon-long 3 --label-horizon-short 3 \
    --train-from ${TRAIN_FROM} \
    --holdout-from ${HOLDOUT_FROM} \
    --holdout-to ${PROD_HOLDOUT_TO} \
    --deploy-calib-days ${TAIL_DAYS} \
    --train-artifacts-subdir ${TAG}_${SIDE}_specialist_seed${SEED} \
    --deploy-subdir deploy_PROD_${SIDE}_specialist_seed${SEED} \
    --locked-params-json artifacts/${RELEASE}/oof/${TAG}/reports/best_per_side.json \
    --locked-side-key ${SIDE} \
    --objective ev_net --cost-per-signal 0.05 --max-drawdown-R 30
done

python -m mimo.oof.merge_specialists \
  --release ${RELEASE} \
  --long-dir  artifacts/${RELEASE}/oof/deploy_PROD_long_specialist_seed${SEED} \
  --short-dir artifacts/${RELEASE}/oof/deploy_PROD_short_specialist_seed${SEED} \
  --out-dir   artifacts/${RELEASE}/oof/deploy_PROD_combined_seed${SEED}

python -m mimo.oof.select_thresholds_from_tail \
  --release ${RELEASE} \
  --deploy-dir artifacts/${RELEASE}/oof/deploy_PROD_combined_seed${SEED} \
  --side both \
  --tp-long 2.0 --sl-long 0.8 --horizon-long 3 \
  --tp-short 2.0 --sl-short 0.8 --horizon-short 3 \
  --cost 0.05 --thr-lo 0.10 --thr-hi 0.45 --n-points 70 --min-signals 30 \
  --from-db

python -m mimo.oof.compute_state_percentiles \
  --release ${RELEASE} \
  --deploy-dir artifacts/${RELEASE}/oof/deploy_PROD_combined_seed${SEED} \
  --emit-config-stub \
  --out-stub config/decision_policies_config_${RELEASE}_PROD.py
Sanity check (NO es validación pura)
⚠️ El sanity check NO es validación. El deploy PROD ha visto estos datos. Es solo para detectar bugs:

python scripts/replay_s2_202500.py \
  --release ${RELEASE} \
  --deploy-subdir deploy_PROD_combined_seed${SEED} \
  --policy-config decision_policies_config_${RELEASE}_PROD \
  --warmup-from 2026-04-15 \
  --from 2026-05-03 --to 2026-05-10 \
  --out /tmp/replay_prod_sanity
Decisión:

PnL ≥ baseline × 0.8 → continúa
PnL < baseline × 0.5 → STOP, revisar bugs
Lineage
Crea artifacts/${RELEASE}/oof/deploy_PROD_combined_seed${SEED}/LINEAGE.md:

# LINEAGE

- **Creado**: YYYY-MM-DD
- **Best params from**: best_per_side.json (Optuna YYYY-MM-DD, N trials)
- **Architecture**: multitask specialists, vol_boost_td_down/vol_boost, h=3
- **Train data**: 2024-01-01 → 2025-10-30
- **Holdout used**: 2025-11-01 → 2026-05-10
- **Tail calibration**: last X days
- **Validated against LOCKBOX**: 2026-04-11 → 2026-05-10
  - PnL: +X%
  - MDD: -Y%
  - ECE: Z
- **Calibrator type**: isotonic [o beta/per_state si aplica]
- **Promoted to production**: YYYY-MM-DD
Fase 6 — Despliegue a producción
Objetivo: que el servicio s2 cargue el nuevo deploy y opere con él.

Pre-deployment checklist
sin completar
Fase 4 pasó decisión gate
sin completar
Fase 5 refit completado, sanity check OK
sin completar
LINEAGE.md escrito
sin completar
Backup del estado actual existe
Cambios en main/s2_main.py
Buscar y aplicar:

# 1. Línea ~10: import policy nueva
- from config.decision_policies_config import gate_by_action_and_state, ...
+ from config.decision_policies_config_202500_PROD import (
+     gate_by_action_and_state, score_cap_by_state, risk_mult_by_state)

# 2. Líneas ~192-201: feature_masks UNION (necesario para multitask renombrado)
  feature_masks={
+     "long":  {"ema_bull": True, "rsi_oversold": True, "macd_positive": True,
+               "ema_bear": True, "rsi_overbought": True, "macd_negative": True},
+     "short": {"ema_bull": True, "rsi_oversold": True, "macd_positive": True,
+               "ema_bear": True, "rsi_overbought": True, "macd_negative": True},
  },

# 3. Línea ~227: artifacts_path
- artifacts_path = str((base_dir / ".." / "artifacts" / release / "oof" / "deploy_full").resolve())
+ artifacts_path = str((base_dir / ".." / "artifacts" / release / "oof" /
+                       "deploy_PROD_combined_seed47").resolve())

# 4. (opcional) Desactivar RL si no lo has reentrenado:
- use_rl=True, rl_policy_path=policy_path,
+ use_rl=False, rl_policy_path=None,
Despliegue
# Backup + commit
cp main/s2_main.py main/s2_main.py.pre_PROD_$(date +%Y%m%d)
git add main/s2_main.py
git commit -m "s2_main: deploy ${RELEASE}_PROD seed${SEED}"
git push origin multitask  # o tu branch de producción

# Restart del servicio (ajusta a tu infra)
sudo systemctl restart s2-service
# o si arrancas a mano:
pkill -f s2_main.py && nohup python main/s2_main.py > logs/s2_$(date +%F).log 2>&1 &
Smoke test post-restart
sin completar
Log dice 🧠 Deploy specialists-merged detectado
sin completar
Log dice Scaler context: 25 cols (multitask)
sin completar
No hay tracebacks
sin completar
Primer [DECIDE_LIVE] aparece dentro de los primeros 30s
sin completar
El Decision: action=... muestra valores razonables
tail -f logs/s2_$(date +%F).log
Fase 7 — Monitorización continua
Objetivo: detectar drift antes de que cueste dinero significativo.

Cadence
Frecuencia	Acción	Comando	Tiempo
Diaria (post-NY close)	health check rápido	monitor_health.py últimos 7d	5 min
Semanal (lunes)	health check completo	monitor_health.py últimos 30d	5 min
Semanal (lunes)	regenerar dashboard	dashboard_health.py	1 min
Mensual	revisión humana del dashboard	manual	30 min
Trimestral	re-evaluar hyperparams	re-Optuna ligera (30 trials)	4h
Comandos
# Diaria — semana corta
python scripts/monitor_health.py \
  --release ${RELEASE} \
  --deploy-subdir deploy_PROD_combined_seed${SEED} \
  --policy-config decision_policies_config_${RELEASE}_PROD \
  --from $(date -d "7 days ago" +%F) --to $(date +%F)

# Semanal — 30 días
python scripts/monitor_health.py \
  --release ${RELEASE} \
  --deploy-subdir deploy_PROD_combined_seed${SEED} \
  --policy-config decision_policies_config_${RELEASE}_PROD

# Regenerar dashboard
python scripts/dashboard_health.py \
  --reports-dir reports/monitor \
  --out reports/dashboard.html
Triggers de acción
Si rolling 5-day PnL < -3% Y >3 días consecutivos negativos
  → HALT MANUAL: pausa producción, análisis humano
Si ECE > 0.08 sostenido durante 2 semanas
  → RECALIBRACIÓN LIGERA:
    select_thresholds_from_tail con tail reciente
    compute_state_percentiles
    deploy nueva policy SIN tocar el modelo
Si AUC-PR < pos_rate × 1.0 sostenido durante 2 semanas
  → REDEPLOY COMPLETO: Fases 2-6 con cutoff actual
Si algún estado con PnL/trade < -0.10R durante 3 semanas consecutivas
  → ENDURECER policy de ese estado (sube quantile o reduce risk_mult)
Si MDD diario > 5%
  → HALT INMEDIATO + revisión
Webapp
Si prefieres operar gráficamente:

# Lanza el panel en localhost:5050
python scripts/webapp.py
Allí lanzas operaciones, ves logs en vivo, browseas reports.

Fase 8 — Checklist pre-dinero real
Objetivo: validaciones finales antes de operar con dinero real significativo.

Pre-flight checklist
Validación operativa (requerido antes de cualquier dinero real):

sin completar
Paper trading mínimo 4 semanas continuas sobre la misma config exacta de producción
sin completar
PnL de paper ≥ 80% del PnL esperado del replay (factor por slippage)
sin completar
Drawdown de paper ≤ MDD del replay × 1.5
sin completar
Latencias de ejecución medidas: ¿broker responde en <500ms? Si no, ajustar signal_cooldown_bars
sin completar
Spread real medido vs spread_price del simulator: si real > simulator × 2 → reducir tamaño
Validación de riesgo:

sin completar
RiskConfig.max_risk_pct ≤ 0.02 (no más de 2% por trade)
sin completar
RiskConfig.max_positions ≤ 3 (no más de 3 posiciones simultáneas)
sin completar
max_daily_loss_pct configurado (recomendado: 3-5%)
sin completar
Killswitch testeado: ¿qué hace el sistema si daily DD > halt limit? ¿Cierra posiciones? ¿Pausa nuevas órdenes?
sin completar
Validar que base_risk_pct produce posiciones de tamaño esperado (calcular manualmente vs broker)
Validación de infraestructura:

sin completar
El servicio se reinicia automáticamente si crashea (systemd con Restart=always)
sin completar
Logs rotan automáticamente (no llenan disco)
sin completar
BD tiene backup diario
sin completar
Conexión a broker reconecta tras desconexión (>5min downtime sin reconexión = problema)
sin completar
El sistema NO opera fuera de horarios definidos (fines de semana, festivos relevantes)
sin completar
Alertas por email/SMS configuradas para: crash del servicio, MDD diario > X%, posición abierta > Y horas
Validación contable:

sin completar
Saldo del broker matchea con equity reportado por el sistema (auditoría mensual)
sin completar
Cada trade del sistema corresponde a una orden ejecutada en broker (cross-check)
sin completar
Comisiones y swap considerados en el cálculo de PnL realizado
Validación humana:

sin completar
Operador disponible 24/5 (mercado FX) o turno de guardia rotado
sin completar
Procedimiento de halt manual documentado y conocido por el operador
sin completar
Procedimiento de rollback documentado (ver Apéndice A)
sin completar
Procedimiento de "emergencia: cerrar todo" probado
Plan de despliegue gradual de capital
Recomendación:

Semanas	% capital nominal	Criterio para subir
1-2	1%	PnL ≥ 80% de esperado, sin crashes
3-4	5%	Pre-flight checklist cumplido, monitorización OK
5-8	25%	PnL sostenido ≥ esperado × 0.7, MDD < esperado × 1.2
9-12	50%	2 meses consecutivos sanos
13+	100%	Validación de ≥3 meses con datos reales
⚠️ Si en algún tramo el PnL real < 50% del esperado → bajar al tramo anterior, investigar.

Checklist diaria de operativa
Cada mañana antes de la apertura del mercado:

sin completar
Servicio running: systemctl status s2-service (o ps aux | grep s2_main)
sin completar
Sin errores recientes en log: grep -i "error\|exception" logs/s2_$(date -d yesterday +%F).log | tail
sin completar
Saldo broker matchea con sistema (anote diff si > 0.1%)
sin completar
Posiciones abiertas razonables (no más de max_positions)
sin completar
Monitor health del día anterior: revisar JSON
# Auto-check matinal
python scripts/monitor_health.py \
  --release ${RELEASE} \
  --deploy-subdir deploy_PROD_combined_seed${SEED} \
  --from $(date -d "1 day ago" +%F) --to $(date +%F)
Apéndice A — Rollback
Cuándo hacer rollback
Rollback inmediato si:

Drawdown diario > 5%
3 días consecutivos con PnL < -2% cada uno
Crash repetido del servicio
Calidad de datos comprometida (gaps, valores anómalos del broker)
Procedimiento
Opción 1 — Rollback automático con script
# Listar deploys disponibles
python scripts/rollback_deploy.py --list

# Dry-run primero
python scripts/rollback_deploy.py \
  --target-deploy <subdir_previo> \
  --target-policy <policy_previa>

# Aplicar
python scripts/rollback_deploy.py \
  --target-deploy <subdir_previo> \
  --target-policy <policy_previa> \
  --apply

# Restart
sudo systemctl restart s2-service
Opción 2 — Rollback manual
# Restaurar s2_main.py desde backup
cp main/s2_main.py.pre_PROD_<fecha> main/s2_main.py
git diff main/s2_main.py  # verifica

# Restart
sudo systemctl restart s2-service

# Verifica que carga el deploy anterior
tail -50 logs/s2_$(date +%F).log
Tras rollback
sin completar
Verificar que el deploy anterior está cargado (mirar log de arranque)
sin completar
Verificar que opera normalmente durante 1h
sin completar
Documentar en docs/INCIDENTS.md la razón del rollback
sin completar
Análisis post-mortem antes de re-intentar despliegue
Apéndice B — Lecciones aprendidas
Lessons learned acumulados
ECE bajo ≠ PnL alto
Un calibrador con ECE excelente puede tener peor PnL si su distribución de scores cambia el operating point del policy. Siempre validar PnL end-to-end antes de adoptar cambios de calibrador.
Cambiar el calibrador requiere recalibrar todo
Al cambiar de iso_21d a iso_full, los percentiles por estado se recalculan. El alpha vivía en una zona específica del calibrador anterior. No asumir continuidad del operating point.
El alpha viene del decision engine, no solo del modelo
La calibración + el filtrado por percentil-por-estado + min_delta_rel juntos hacen el sistema. Tocar uno sin entender el efecto en los otros es peligroso.
Score saturado en el cap = pérdida masiva de resolución
Si pct_at_max > 5%, el isotonic no distingue confianzas altas. Pero ojo: a veces esa saturación ES donde vive el alpha del replay. Cambiar la saturación cambia el régimen operativo.
LOCKBOX es sagrado
Una vez que miras el LOCKBOX, deja de ser unbiased. Si vas a iterar Fase 4 con distintas configs, considera mantener un sub-LOCKBOX adicional.
State leak entre iteraciones de sweep
Cuando hagas sweeps con --policy-grid, asegúrate de que el estado del DecisionEngine (signal_cooldown, etc.) se resetea entre iteraciones. Lo hicimos en replay_s2_202500.py:reset_simulator_runtime_state.
TREND_DOWN es estructuralmente más débil
Ese estado ha mostrado PnL negativo recurrentemente. Considera: gate más alto o risk_mult reducido para ese estado específico.
El daily_breakdown es revelador
Un PnL mensual de +X% puede ser 3-4 días enormes ganando + muchos días perdiendo poco. El daily_breakdown distingue alpha real de suerte. Mira siempre las semanas individuales además del agregado.
Re-Optuna no es siempre necesario
Si tu best_per_side.json es de hace 3-6 meses y la performance es estable, no hace falta refrescarlo. Re-Optuna solo cuando hay drift sostenido en métricas.
No comparar replays sobre ventanas distintas
Tentación: comparar el +41% de abril original con un nuevo replay sobre LOCKBOX (apr-11 → may-10). NO ES COMPARABLE. Para comparar, replay del baseline EXACTAMENTE sobre la misma ventana que el experimento.
Apéndice C — Resolución de problemas comunes
"ModuleNotFoundError: No module named 'config.X'"
--policy-config debe ser el nombre del módulo Python, no un path. Solo el nombre sin extensión.

"Input 2 of layer 'TradingModel_v3' is incompatible: expected shape=(None, 25), found shape=(32, 24)"
El deploy es multitask (calibrador heredado de specialists merged) pero el replay/s2_main está cargando feature_masks asimétricas (binary, 24 cols). Solución: feature_masks UNION (25 cols).

"Deploy binary detectado" cuando debería ser multitask
merge_specialists guarda merge_specialists_meta.json (no meta.json). Verifica que el deploy tiene ese archivo. Si no, los modelos NO vienen de specialists merged.

Score percentiles JSON tiene selected_threshold muy bajo (~0.10)
recalibrate_deploy_multitask por defecto elige threshold por F1 que colapsa con base rates bajas. Solución: ejecutar select_thresholds_from_tail que mueve a EV-net.

Replay da MUCHOS más SHORTS que LONGS (o viceversa)
Probablemente la calibración o policy quedó descompensada por side. Revisa el output de select_thresholds_from_tail: ¿los thresholds tienen sentido (similar magnitud entre sides)? Si no, posible bug en el deploy.

"❌ Tail parquet no existe"
select_thresholds_from_tail busca data/deploy_calibration_tail_<release>_<side>.parquet que produce resume_deploy_v6. Si vienes del fast path (train_specialist + merge_specialists), el parquet puede no estar. Usa --from-db para que reconstruya desde DB.

Apéndice D — Archivos clave a tener identificados
Repo root
├── main/s2_main.py                              ← entrypoint del servicio prod
├── config/decision_policies_config_*.py         ← policies (una por experimento)
├── artifacts/202500/oof/
│   ├── {tag}_long_specialist_seed{N}/           ← outputs Fase 2 LONG
│   ├── {tag}_short_specialist_seed{N}/          ← outputs Fase 2 SHORT
│   ├── deploy_validation_combined_seed{N}/      ← outputs Fase 3 (validación)
│   └── deploy_PROD_combined_seed{N}/            ← outputs Fase 5 (producción)
├── scripts/
│   ├── replay_s2_202500.py                      ← replay histórico
│   ├── ghost_predict.py                         ← predicciones offline
│   ├── simulate_calibrators.py                  ← comparativa calibradores
│   ├── swap_calibrators_per_champion.py         ← swap calibradores
│   ├── monitor_health.py                        ← health check operativo
│   ├── dashboard_health.py                      ← dashboard HTML
│   ├── rollback_deploy.py                       ← rollback seguro
│   ├── webapp.py                                ← UI Flask
│   └── inspect_calibrator_capacity.py           ← diagnóstico isotonic
├── reports/
│   ├── monitor/health_<YYYY-MM-DD>/             ← outputs monitor_health
│   ├── dashboard.html                           ← dashboard renderizado
│   └── *.csv, *.json                            ← outputs varios
└── logs/
    ├── jobs/                                    ← logs del webapp
    └── s2_<YYYY-MM-DD>.log                      ← logs del servicio
Apéndice E — Glosario de cutoffs de fechas
Para un release dado:

TRAIN_FROM           = 2024-01-01    (inicio absoluto)
TRAIN_TO             = 2025-10-30    (fin del periodo TRAIN)
HOLDOUT_FROM         = 2025-11-01    (inicio HOLDOUT/validación)
HOLDOUT_TO           = 2026-04-10    (cutoff del deploy de VALIDACIÓN)
LOCKBOX_FROM         = 2026-04-11    (inicio LOCKBOX)
LOCKBOX_TO           = 2026-05-10    (fin LOCKBOX = última fecha disponible)
PROD_HOLDOUT_TO      = 2026-05-10    (cutoff del deploy de PRODUCCIÓN)
TAIL_DAYS            = 21 o 45       (longitud del tail para isotonic cal)
Para validación:
  Modelo trained on: TRAIN_FROM → TRAIN_TO
  OOF eval on:       HOLDOUT_FROM → HOLDOUT_TO
  Tail cal on:       (HOLDOUT_TO - TAIL_DAYS) → HOLDOUT_TO
  Replay LOCKBOX:    LOCKBOX_FROM → LOCKBOX_TO  ← out-of-sample puro
Para producción (tras validación exitosa):
  Modelo trained on: TRAIN_FROM → TRAIN_TO
  OOF eval on:       HOLDOUT_FROM → PROD_HOLDOUT_TO
  Tail cal on:       (PROD_HOLDOUT_TO - TAIL_DAYS) → PROD_HOLDOUT_TO  ← incluye LOCKBOX
Apéndice F — Comandos de emergencia
Cerrar TODAS las posiciones inmediatamente
# Desde el bot
python -c "
from mimo.data_managers.databases import Database
# ... código broker-specific para cerrar todo
"

# O directamente en MT5 / broker UI
Pausar el servicio sin matarlo (cierre limpio)
# Si tienes systemd con SIGTERM handling
sudo systemctl stop s2-service

# Si lo arrancas a mano
kill -TERM $(pgrep -f s2_main.py)
# Espera 30s a que cierre posiciones limpias
sleep 30
# Si sigue vivo, forzar
kill -9 $(pgrep -f s2_main.py)
Volcar estado actual a un fichero (para análisis post-mortem)
DUMPDIR=docs/emergency_dump_$(date +%Y%m%d_%H%M)
mkdir -p $DUMPDIR
cp main/s2_main.py $DUMPDIR/
cp -r config/decision_policies_config_*.py $DUMPDIR/
cp -r logs/s2_$(date +%F).log $DUMPDIR/
echo "Open positions:" > $DUMPDIR/status.txt
# ... añadir snapshot de posiciones del broker ...
tar -czf $DUMPDIR.tar.gz $DUMPDIR/
echo "Dump completo: $DUMPDIR.tar.gz"
Fin del runbook.

Este documento debe actualizarse tras cada incidente significativo (sección Lessons learned), cada cambio mayor de arquitectura o tras cada re-Optuna.

## Cómo usarlo
Guarda como `docs/RUNBOOK.md`:
```bash
mkdir -p docs
# Pega el contenido en docs/RUNBOOK.md
git add docs/RUNBOOK.md
git commit -m "docs: runbook operativo completo"
git push origin multitask
Recomendación de uso del runbook
Para tu próximo ciclo completo (Fases 1-6):

Imprime o ten abierto el documento mientras ejecutas
Marca con [x] los checklists al ir avanzando (puedes mantener una copia "live" del runbook en cada experimento)
Documenta variaciones en Apéndice B — Lecciones aprendidas
Para monitorización diaria (Fase 7):

Configura cron con el comando diario de monitor_health
Revisa el dashboard semanalmente
Usa el webapp si prefieres UI sobre CLI
Para emergencias:

Apéndice A (rollback) y Apéndice F (comandos de emergencia) en lugar visible/accesible
Ten el procedimiento de halt manual memorizado o impreso
Mejoras futuras
Si quieres extender el runbook:

Apéndice G — Procedimiento walk-forward (rolling cutoffs cada 2 semanas para validación más rigurosa)
Apéndice H — Procedimiento de re-Optuna (cuándo, cómo, criterios de aceptación)
Apéndice I — Plantilla de incident report (formato estándar para documentar problemas)


## Apéndice G — Walk-forward validation

**Objetivo**: validar la robustez temporal del sistema más allá de un único LOCKBOX. Walk-forward genera múltiples cutoffs históricos rolling y evalúa cada uno sobre el periodo inmediatamente posterior.

### Por qué hacerlo

Un único LOCKBOX (30 días) puede ser ruido. Si el sistema funciona consistentemente sobre **5+ ventanas históricas distintas**, la robustez está mucho mejor demostrada.

### Cuándo hacerlo

- **Una vez antes del primer despliegue real con dinero significativo**
- **Tras un cambio mayor de arquitectura** (variante features, target type, etc.)
- **Tras un drift estructural** cuando se sospecha que el alpha es régimen-específico
- **NO** hacer en cada ciclo mensual — es caro (varias horas)

### Diseño del walk-forward para 202500

Ejemplo de 5 cutoffs mensuales (ajustar fechas según tu disponibilidad de datos):

| Cutoff | Train hasta | Tail (21d) | Replay window | LOCKBOX para este | 
|---|---|---|---|---|
| C1: 2025-11-30 | 2025-09-30 | 2025-11-10 → 2025-11-30 | 2025-12-01 → 2025-12-31 | dic-2025 |
| C2: 2025-12-31 | 2025-10-31 | 2025-12-11 → 2025-12-31 | 2026-01-01 → 2026-01-31 | ene-2026 |
| C3: 2026-01-31 | 2025-11-30 | 2026-01-11 → 2026-01-31 | 2026-02-01 → 2026-02-28 | feb-2026 |
| C4: 2026-02-28 | 2025-12-31 | 2026-02-08 → 2026-02-28 | 2026-03-01 → 2026-03-31 | mar-2026 |
| C5: 2026-03-31 | 2026-01-31 | 2026-03-11 → 2026-03-31 | 2026-04-01 → 2026-04-30 | abr-2026 |

### Procedimiento por cutoff

Para cada cutoff `Cn` con fechas `CUTOFF_DATE`, `PRECUTOFF_DATE` (cutoff - 2 meses), `EVAL_FROM`, `EVAL_TO`:

```bash
# Variables del cutoff (ejemplo C3: cutoff jan-31, eval feb-2026)
export CUTOFF_NAME=C3_2026_01_31
export TRAIN_TO_LOCAL=2025-11-30
export HOLDOUT_TO_LOCAL=2026-01-31
export EVAL_FROM=2026-02-01
export EVAL_TO=2026-02-28
export WARMUP_FROM_LOCAL=2026-01-15

# 1. train_specialist (reusa best_per_side.json)
python -m mimo.oof.train_specialist \
  --best-per-side-json artifacts/${RELEASE}/oof/${TAG}/reports/best_per_side.json \
  --side both --release ${RELEASE} --base-tf 5min --target-type multitask \
  --variant-long vol_boost_td_down --variant-short vol_boost \
  --label-horizon-long 3 --label-horizon-short 3 \
  --train-from ${TRAIN_FROM} \
  --train-to ${TRAIN_TO_LOCAL} \
  --holdout-from 2025-11-01 \
  --holdout-to ${HOLDOUT_TO_LOCAL} \
  --objective ev_net --cost-per-signal 0.05 --max-drawdown-R 30 \
  --oof-epochs 120 --oof-patience 15 \
  --seed ${SEED}

# Renombrar para no colisionar con otros cutoffs
mv artifacts/${RELEASE}/oof/${TAG}_long_specialist_seed${SEED} \
   artifacts/${RELEASE}/oof/${TAG}_long_specialist_seed${SEED}_${CUTOFF_NAME}
mv artifacts/${RELEASE}/oof/${TAG}_short_specialist_seed${SEED} \
   artifacts/${RELEASE}/oof/${TAG}_short_specialist_seed${SEED}_${CUTOFF_NAME}

# 2. resume_deploy_v6 × 2
for SIDE in long short; do
  python -m mimo.oof.resume_deploy_full_v6_multitask \
    --release ${RELEASE} --target-type multitask --base-tf 5min \
    --variant-long vol_boost_td_down --variant-short vol_boost \
    --label-horizon-long 3 --label-horizon-short 3 \
    --train-from ${TRAIN_FROM} --holdout-from 2025-11-01 \
    --holdout-to ${HOLDOUT_TO_LOCAL} \
    --deploy-calib-days ${TAIL_DAYS} \
    --train-artifacts-subdir ${TAG}_${SIDE}_specialist_seed${SEED}_${CUTOFF_NAME} \
    --deploy-subdir deploy_${CUTOFF_NAME}_${SIDE}_specialist_seed${SEED} \
    --locked-params-json artifacts/${RELEASE}/oof/${TAG}/reports/best_per_side.json \
    --locked-side-key ${SIDE} \
    --objective ev_net --cost-per-signal 0.05 --max-drawdown-R 30
done

# 3. Merge
python -m mimo.oof.merge_specialists \
  --release ${RELEASE} \
  --long-dir  artifacts/${RELEASE}/oof/deploy_${CUTOFF_NAME}_long_specialist_seed${SEED} \
  --short-dir artifacts/${RELEASE}/oof/deploy_${CUTOFF_NAME}_short_specialist_seed${SEED} \
  --out-dir   artifacts/${RELEASE}/oof/deploy_${CUTOFF_NAME}_combined_seed${SEED}

# 4. select_thresholds + compute_state_percentiles
python -m mimo.oof.select_thresholds_from_tail \
  --release ${RELEASE} \
  --deploy-dir artifacts/${RELEASE}/oof/deploy_${CUTOFF_NAME}_combined_seed${SEED} \
  --side both \
  --tp-long 2.0 --sl-long 0.8 --horizon-long 3 \
  --tp-short 2.0 --sl-short 0.8 --horizon-short 3 \
  --cost 0.05 --thr-lo 0.10 --thr-hi 0.45 --n-points 70 --min-signals 30 \
  --from-db

python -m mimo.oof.compute_state_percentiles \
  --release ${RELEASE} \
  --deploy-dir artifacts/${RELEASE}/oof/deploy_${CUTOFF_NAME}_combined_seed${SEED} \
  --emit-config-stub \
  --out-stub config/decision_policies_config_${RELEASE}_${CUTOFF_NAME}.py

# 5. Replay sobre el periodo eval
python scripts/replay_s2_202500.py \
  --release ${RELEASE} \
  --deploy-subdir deploy_${CUTOFF_NAME}_combined_seed${SEED} \
  --policy-config decision_policies_config_${RELEASE}_${CUTOFF_NAME} \
  --warmup-from ${WARMUP_FROM_LOCAL} \
  --from ${EVAL_FROM} --to ${EVAL_TO} \
  --out /tmp/walkforward_${CUTOFF_NAME}
Repite para C1 a C5 cambiando las fechas.

Tiempo total: ~5 cutoffs × ~30 min cada uno = ~2.5h.

Agregación de resultados
Tras los 5 cutoffs, agrega manualmente o con un script:

Cutoff	Eval window	PnL%	MDD%	Trades	Win%
C1	dic-2025	+X%	-Y%	N	W%
C2	ene-2026	...	...	...	...
C3	feb-2026	...	...	...	...
C4	mar-2026	...	...	...	...
C5	abr-2026	...	...	...	...
Criterios de aceptación walk-forward
Métrica	OK	Marginal	KO
% cutoffs con PnL > 0	≥ 80% (4/5)	≥ 60% (3/5)	< 60%
MDD peor de los 5	< 15%	< 20%	> 20%
Spread PnL entre cutoffs	< 30pp	< 50pp	> 50pp
Trades por mes (consistencia)	rango ±30%	±50%	mucho mayor
Acción:

OK → sistema robusto, proceder a despliegue con confianza
Marginal → identificar qué cutoff(s) fallan, posible patrón regimen-específico, vigilar más
KO → el alpha no es robusto, no desplegar con dinero significativo. Re-Optuna o cambio arquitectural
Lessons walk-forward
Si un cutoff específico falla, mira el contexto de mercado en ese periodo (¿fue un mes muy volátil/lateral?). Quizás el sistema necesita un filtro adicional para ese régimen.
Si todos fallan en el mismo estado (p.ej. TREND_DOWN siempre pierde), refuerza la policy para ese estado en concreto.
NO ajustar la arquitectura iterativamente al walk-forward — eso es overfitting al pasado y se pierde la propiedad de generalización.
Apéndice H — Re-Optuna refresh
Objetivo: refrescar los hyperparams cuando hay evidencia de drift estructural que no se resuelve con recalibración ligera.

Cuándo NO hacer re-Optuna
Si el sistema lleva <3 meses con best_per_side.json actual
Si solo hay una semana mala
Si la causa es operativa (slippage, latency, broker issues)
Si solo una calibración pequeña arregla el problema
Cuándo SÍ hacer re-Optuna
best_per_side.json tiene >3 meses Y métricas de monitor han ido empeorando
ECE > 0.10 sostenido 2+ semanas tras intentar recalibración ligera
Algún estado pasa de positivo a tóxico consistentemente
AUC-PR < base rate sostenido (modelo pierde poder predictivo)
Tras cambio mayor de mercado (regime shift macro evidente)
Modos de re-Optuna
Modo A — Refresh ligero (30 trials, ~3h)

Reutiliza el grid de variantes/horizons existente, busca dentro del espacio actual:

python -m mimo.oof.main_oof_regime_weights_v7 \
  --release ${RELEASE} \
  --base-tf 5min --target-type multitask --side both \
  --variant-long vol_boost_td_down --variant-short vol_boost \
  --label-horizon-long 3 --label-horizon-short 3 \
  --train-from ${TRAIN_FROM} --train-to ${TRAIN_TO} \
  --holdout-from ${HOLDOUT_FROM} --holdout-to ${HOLDOUT_TO} \
  --use-tpe --optuna-trials 30 \
  --objective ev_net --cost-per-signal 0.05 \
  --ev-min-signals 100 --max-drawdown-R 30 \
  --ev-thr-lo 0.10 --ev-thr-hi 0.40 \
  --oof-epochs 120 --oof-patience 15
Usar para refresco preventivo o tras drift mediano.

Modo B — Búsqueda completa (100+ trials, ~10h)

Para drift severo o tras cambio arquitectural significativo. Mismo comando con --optuna-trials 100.

Modo C — Cambio de espacio de búsqueda

Si sospechas que el problema es estructural (variantes de features actuales no funcionan), considera:

Cambiar --variant-long / --variant-short
Cambiar --label-horizon-*
Cambiar --target-type
Esto sale del scope del refresh — es una iteración de R&D nueva.

Criterios de adopción de los nuevos hyperparams
Tras re-Optuna, comparar trial nuevo vs locked params anterior:

Métrica	Criterio mínimo
EV_net (OOF)	Nuevo ≥ antiguo + 0.02R
AUC-PR (OOF)	Nuevo ≥ antiguo × 0.95 (no degradar mucho discriminación)
n_signals (OOF)	Nuevo ≥ antiguo × 0.7 (no colapsar volumen)
MDD (OOF)	Nuevo ≤ antiguo × 1.2 (no empeorar drawdown)
Si TODAS las métricas pasan → adopta best_per_side.json nuevo.

Si solo algunas → NO adoptar. Posible overfit a los datos recientes del retrain.

Tras adoptar nuevos hyperparams
Vuelve a Fase 2 con el best_per_side.json nuevo:

train_specialist con nuevos params
resume_deploy_v6 × 2
merge_specialists
select_thresholds + compute_state_percentiles
Validación obligatoria en LOCKBOX (Fase 4) antes de promover
Production refit (Fase 5)
Despliegue (Fase 6)
⚠️ Especialmente importante: tras re-Optuna, el LOCKBOX puede estar contaminado si lo usaste para informar la decisión "hay drift, necesito Optuna". Considera reservar una ventana FUTURA como nuevo LOCKBOX antes de promover.

Apéndice I — Plantilla de incident report
Cuando ocurra un incidente operativo (PnL anómalo, crash, datos sospechosos, etc.), documéntalo siguiendo esta plantilla. Guardar en docs/INCIDENTS/INC-YYYY-MM-DD-<tag>.md.

Plantilla
# INC-2026-MM-DD — <título corto del incidente>

**Severidad**: [P0 crítico / P1 grave / P2 moderado / P3 menor]  
**Detectado por**: [monitor_health automático / observación humana / alerta broker]  
**Detectado en**: 2026-MM-DD HH:MM (zona horaria)  
**Resuelto en**: 2026-MM-DD HH:MM  
**Tiempo total**: X horas Y minutos  
**Operador on-call**: <nombre>

---

## 1. Resumen ejecutivo (1-2 párrafos)

Qué pasó, qué impacto tuvo (PnL, posiciones, downtime), cómo se resolvió.

---

## 2. Cronología

| Hora | Evento |
|---|---|
| HH:MM | Primera anomalía detectada |
| HH:MM | Investigación iniciada |
| HH:MM | Causa identificada (probable) |
| HH:MM | Acción correctiva aplicada |
| HH:MM | Servicio normalizado |
| HH:MM | Post-mortem iniciado |

---

## 3. Síntomas observados

- Métrica X cambió de A a B
- Log muestra error Y
- Trader humano observó Z

---

## 4. Investigación

### Hipótesis consideradas

1. **H1**: ... → descartada porque ...
2. **H2**: ... → confirmada por ...
3. **H3**: ... → no concluyente

### Evidencia

- Output de `monitor_health` en X fecha
- Trades del log (extraer trades.parquet)
- Logs del servicio (extracto relevante)
- Estado del broker en el momento

---

## 5. Causa raíz

Descripción detallada de la causa identificada. Si es multi-factor, enumerar.

### Categoría

- [ ] Bug de código
- [ ] Drift del modelo
- [ ] Drift de mercado (no controlable)
- [ ] Operativo (broker, conectividad, hardware)
- [ ] Configuración incorrecta
- [ ] Otro: ___

---

## 6. Resolución

Qué se hizo concretamente:
- Comandos ejecutados
- Cambios de config aplicados
- Rollback aplicado (si procede)

---

## 7. Impacto

- **PnL afectado**: +/- X% (calculo: ...)
- **Posiciones afectadas**: N trades
- **Downtime**: X minutos
- **Capital perdido / no ganado**: Z€

---

## 8. Acciones preventivas

| # | Acción | Owner | Deadline | Estado |
|---|---|---|---|---|
| 1 | Añadir alerta para X | Yo | YYYY-MM-DD | [ ] |
| 2 | Refactorizar Y para evitar bug | Yo | YYYY-MM-DD | [ ] |
| 3 | Documentar caso en runbook | Yo | YYYY-MM-DD | [ ] |

---

## 9. Lecciones aprendidas

Lo que se aprende de este incidente. Va al Apéndice B del runbook.

---

## 10. Anexos

- Link al PR/commit con la fix
- Screenshots relevantes
- Output del monitor_health del periodo afectado
- Dump del state si se hizo (ver Apéndice F del runbook)
Cómo usar la plantilla
Durante el incidente: NO escribir el informe. Resolver primero. Si hay tiempo, tomar notas de tiempos.
Inmediatamente tras resolver: capturar timestamps y evidencia (logs, dumps)
En las siguientes 24-48h: escribir el informe con calma
Revisar tras 1 semana: añadir lo que descubras después
Tracking de acciones: revisar deadlines en la revisión semanal de monitorización
Niveles de severidad
Nivel	Definición	Tiempo de respuesta
P0 Crítico	Pérdida de capital activa / servicio caído	< 15 min
P1 Grave	Datos comprometidos / decisiones erróneas	< 1h
P2 Moderado	Métricas degradadas pero operando	< 24h
P3 Menor	Curiosidad o mejora preventiva	Sin SLA
## Script para convertir a PDF/Word
Crea `scripts/build_docs.sh`:
```bash
#!/bin/bash
# build_docs.sh — Convierte docs/RUNBOOK.md a PDF + DOCX usando pandoc.
#
# Pre-requisitos:
#   - pandoc:   sudo apt install pandoc  (Linux/WSL)
#               brew install pandoc      (Mac)
#               winget install pandoc    (Windows)
#   - LaTeX para PDF:
#               sudo apt install texlive-xetex texlive-fonts-recommended (Linux/WSL)
#               brew install --cask basictex                              (Mac)
#               winget install MiKTeX.MiKTeX                              (Windows)
#
# Uso:
#   bash scripts/build_docs.sh
#   # genera: docs/build/RUNBOOK.pdf y docs/build/RUNBOOK.docx
set -e
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SRC="$REPO_ROOT/docs/RUNBOOK.md"
OUT_DIR="$REPO_ROOT/docs/build"
if [ ! -f "$SRC" ]; then
  echo "❌ No existe $SRC. Asegúrate de tener el runbook en docs/RUNBOOK.md"
  exit 1
fi
mkdir -p "$OUT_DIR"
# Detectar pandoc
if ! command -v pandoc &> /dev/null; then
  echo "❌ pandoc no instalado."
  echo "   Linux/WSL:  sudo apt install pandoc texlive-xetex"
  echo "   macOS:      brew install pandoc && brew install --cask basictex"
  echo "   Windows:    winget install pandoc"
  exit 1
fi
PANDOC_VERSION=$(pandoc --version | head -1)
echo "📚 Usando: $PANDOC_VERSION"
# ─── DOCX ─────────────────────────────────────────────────────────────
echo ""
echo "🔧 Generando DOCX..."
pandoc "$SRC" \
  --from gfm \
  --to docx \
  --output "$OUT_DIR/RUNBOOK.docx" \
  --toc \
  --toc-depth=3 \
  --highlight-style=tango \
  --metadata title="S2 Release 202500 — Runbook operativo" \
  --metadata author="Equipo S2" \
  --metadata date="$(date +%Y-%m-%d)"
if [ -f "$OUT_DIR/RUNBOOK.docx" ]; then
  SIZE=$(du -h "$OUT_DIR/RUNBOOK.docx" | cut -f1)
  echo "   ✅ $OUT_DIR/RUNBOOK.docx ($SIZE)"
else
  echo "   ❌ DOCX no generado"
fi
# ─── PDF (con LaTeX) ──────────────────────────────────────────────────
echo ""
echo "🔧 Generando PDF..."
# Detectar engine de LaTeX
PDF_ENGINE=""
for engine in xelatex pdflatex lualatex tectonic; do
  if command -v "$engine" &> /dev/null; then
    PDF_ENGINE="$engine"
    break
  fi
done
if [ -z "$PDF_ENGINE" ]; then
  echo "   ⚠️  Sin LaTeX disponible. Saltando PDF."
  echo "   Para PDF instala texlive-xetex (Linux), basictex (Mac), MiKTeX (Windows)"
else
  echo "   Usando engine: $PDF_ENGINE"
  
  # Opciones extra si es xelatex/lualatex (mejor soporte de Unicode/emojis)
  EXTRA_OPTS=""
  if [ "$PDF_ENGINE" = "xelatex" ] || [ "$PDF_ENGINE" = "lualatex" ]; then
    EXTRA_OPTS="-V mainfont=DejaVuSans -V monofont=DejaVuSansMono"
  fi
  
  pandoc "$SRC" \
    --from gfm \
    --to pdf \
    --output "$OUT_DIR/RUNBOOK.pdf" \
    --toc \
    --toc-depth=3 \
    --highlight-style=tango \
    --pdf-engine="$PDF_ENGINE" \
    -V geometry:"margin=2cm" \
    -V documentclass=report \
    -V colorlinks=true \
    -V linkcolor=blue \
    -V urlcolor=blue \
    --metadata title="S2 Release 202500 — Runbook operativo" \
    --metadata author="Equipo S2" \
    --metadata date="$(date +%Y-%m-%d)" \
    $EXTRA_OPTS 2>&1 | grep -v "^$" || true
  
  if [ -f "$OUT_DIR/RUNBOOK.pdf" ]; then
    SIZE=$(du -h "$OUT_DIR/RUNBOOK.pdf" | cut -f1)
    echo "   ✅ $OUT_DIR/RUNBOOK.pdf ($SIZE)"
  else
    echo "   ❌ PDF no generado (revisa errores arriba)"
  fi
fi
echo ""
echo "🎉 Documentos generados en $OUT_DIR/"
ls -la "$OUT_DIR/" 2>/dev/null | grep -E "(RUNBOOK|\.pdf|\.docx)" || true
Cómo usarlo
1. Asegúrate de tener los apéndices en docs/RUNBOOK.md
Añade los apéndices G, H, I de arriba al runbook existente.

2. Instala pandoc + LaTeX
WSL/Ubuntu:

sudo apt update
sudo apt install -y pandoc texlive-xetex texlive-fonts-recommended texlive-fonts-extra
Tamaño total: ~700MB. Después funciona offline.

macOS:

brew install pandoc
brew install --cask basictex
# refresca PATH
eval "$(/usr/libexec/path_helper)"
Windows:

winget install pandoc
winget install MiKTeX.MiKTeX
# o, más ligero, solo para PDF básico:
winget install pandoc
# y exportar solo DOCX (sin LaTeX)
3. Hazlo ejecutable y lánzalo
chmod +x scripts/build_docs.sh
bash scripts/build_docs.sh
Output esperado:

📚 Usando: pandoc 3.1.x
🔧 Generando DOCX...
   ✅ /home/user/repo/docs/build/RUNBOOK.docx (180K)
🔧 Generando PDF...
   Usando engine: xelatex
   ✅ /home/user/repo/docs/build/RUNBOOK.pdf (350K)
🎉 Documentos generados en /home/user/repo/docs/build/
4. Descarga / Abre
Desde WSL:

explorer.exe docs/build  # abre el dir en Windows
# o copia a tu Desktop:
cp docs/build/RUNBOOK.* /mnt/c/Users/$USER/Desktop/
Linux:

xdg-open docs/build/RUNBOOK.pdf
Mac:

open docs/build/RUNBOOK.pdf
Si no quieres instalar LaTeX (solo DOCX)
El DOCX se genera sin LaTeX. Solo necesitas pandoc, y luego Word te lo abre directamente (o LibreOffice, Google Docs, etc.). Desde Word puedes exportar a PDF con "Guardar como → PDF" si lo prefieres.

Eso es la opción más ligera (~80MB de pandoc, sin LaTeX).

Alternativa Python (DOCX puro, sin pandoc)
Si no quieres ni siquiera pandoc, puedes usar Python. Crea scripts/md_to_docx.py:

#!/usr/bin/env python3
"""
md_to_docx.py — Conversión Markdown → DOCX usando python-docx.

Pre-requisitos:
    pip install python-docx markdown

Uso:
    python scripts/md_to_docx.py docs/RUNBOOK.md docs/build/RUNBOOK.docx

Limitaciones vs pandoc:
    - Tablas básicas (sin merge celdas avanzado)
    - Highlighting de código simplificado
    - Sin TOC automático
Aún así produce un DOCX legible y editable.
"""
import re
import sys
from pathlib import Path

try:
    from docx import Document
    from docx.shared import Pt, Inches, RGBColor
    from docx.enum.style import WD_STYLE_TYPE
except ImportError:
    raise SystemExit("❌ Instala: pip install python-docx")


def convert(md_path: Path, docx_path: Path) -> None:
    text = md_path.read_text(encoding="utf-8")
    doc = Document()

    # Estilos base
    style_normal = doc.styles["Normal"]
    style_normal.font.name = "Calibri"
    style_normal.font.size = Pt(11)

    in_code = False
    code_buf: list[str] = []
    in_table = False
    table_rows: list[list[str]] = []

    for line in text.splitlines():
        # Code fences
        if line.startswith("```"):
            if in_code:
                # cerrar bloque de código
                p = doc.add_paragraph()
                run = p.add_run("\n".join(code_buf))
                run.font.name = "Consolas"
                run.font.size = Pt(9)
                p.paragraph_format.left_indent = Inches(0.3)
                code_buf = []
                in_code = False
            else:
                in_code = True
            continue
        if in_code:
            code_buf.append(line)
            continue

        # Table rows
        if "|" in line and line.strip().startswith("|"):
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if all(c.replace("-", "").replace(":", "").strip() == "" for c in cells):
                continue  # separator
            if not in_table:
                in_table = True
                table_rows = []
            table_rows.append(cells)
            continue
        elif in_table:
            # cerrar tabla
            if table_rows:
                tbl = doc.add_table(rows=len(table_rows), cols=len(table_rows[0]))
                tbl.style = "Light Grid Accent 1"
                for i, row_data in enumerate(table_rows):
                    for j, cell_text in enumerate(row_data):
                        if j < len(tbl.rows[i].cells):
                            tbl.rows[i].cells[j].text = cell_text
            in_table = False
            table_rows = []

        # Headers
        if line.startswith("# "):
            doc.add_heading(line[2:].strip(), level=0)
        elif line.startswith("## "):
            doc.add_heading(line[3:].strip(), level=1)
        elif line.startswith("### "):
            doc.add_heading(line[4:].strip(), level=2)
        elif line.startswith("#### "):
            doc.add_heading(line[5:].strip(), level=3)
        elif line.startswith("---"):
            doc.add_paragraph("─" * 80)
        elif line.startswith("- ") or line.startswith("* "):
            doc.add_paragraph(line[2:].strip(), style="List Bullet")
        elif re.match(r"^\d+\. ", line):
            doc.add_paragraph(re.sub(r"^\d+\. ", "", line).strip(), style="List Number")
        elif line.strip() == "":
            doc.add_paragraph("")
        else:
            # texto plano (con limpieza básica de markdown inline)
            cleaned = re.sub(r"\*\*([^*]+)\*\*", r"\1", line)  # bold
            cleaned = re.sub(r"\*([^*]+)\*", r"\1", cleaned)   # italic
            cleaned = re.sub(r"`([^`]+)`", r"\1", cleaned)     # code inline
            cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)  # links
            doc.add_paragraph(cleaned)

    doc.save(str(docx_path))
    print(f"✅ Generado: {docx_path}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python scripts/md_to_docx.py <input.md> <output.docx>")
        sys.exit(1)
    md = Path(sys.argv[1])
    docx = Path(sys.argv[2])
    docx.parent.mkdir(parents=True, exist_ok=True)
    convert(md, docx)
Uso:

pip install python-docx
mkdir -p docs/build
python scripts/md_to_docx.py docs/RUNBOOK.md docs/build/RUNBOOK.docx