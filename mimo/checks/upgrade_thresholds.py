import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

from mimo.data_managers.data_manager import DataManager
from mimo.data_managers.databases import Database
from mimo.features.feature_builder import FeatureConfig, FeatureEngineer
from mimo.models.model_builder import Config, ModelConfig
from mimo.data_managers.data_pipeline_v2 import DataPipeline
from mimo.states_manager.state_detector import StateDetector, StateConfig

# ── CONFIG ────────────────────────────────────────────────────────────
RELEASE        = '200372'
ARTIFACTS_PATH = Path(f'../../artifacts/{RELEASE}/oof/deploy_full')
META_PATH      = ARTIFACTS_PATH / f'scalers_{RELEASE}' / 'meta.json'

# Ventana de recalibración: últimos 6 meses
# Lo suficientemente larga para capturar el nuevo régimen estructural
# sin sobre-ajustar a un spike puntual
RECALIB_FROM = datetime(2025, 12, 1)
RECALIB_TO = datetime(2026, 3, 31)

# ── 1. Verificar que existe meta.json ─────────────────────────────────
if not META_PATH.exists():
    raise FileNotFoundError(f"No encontrado: {META_PATH}")

meta = json.loads(META_PATH.read_text(encoding='utf-8'))
old_thresholds = meta.get('regime_thresholds', {})
print(f"Umbrales actuales: {old_thresholds}")

# ── 2. Cargar datos del periodo de recalibración ──────────────────────
db = Database()
db.connect()
df_raw = DataManager.from_database_historical_2(db, from_date=RECALIB_FROM, to_date=RECALIB_TO).df
df_raw = df_raw.sort_values('time').reset_index(drop=True)
print(f"Filas cargadas: {len(df_raw):,} ({RECALIB_FROM} → {RECALIB_TO})")

# ── 3. Preparar features (igual que en entrenamiento) ─────────────────
# Necesitamos atr_norm, bb_width y range_expansion calculados
# exactamente igual que cuando se entrenó el OOF
general_config = Config(release=RELEASE, oof_splits=3, oof_epochs=25)
model_config   = ModelConfig(seq_len_short=64, seq_len_long=256, batch_size=8192)
feature_config = FeatureConfig(
    ema_periods=[9, 21, 50],
    label_horizon=10,
    tp_barrier=2.5,
    sl_barrier=1.5,
    label_method='adaptive',
)

pipeline = DataPipeline(
    general_config=general_config,
    feature_config=feature_config,
    model_config=model_config,
)

df_prepared = pipeline.prepare_data(df_raw, labels=False, side='long')
print(f"Filas tras prepare_data: {len(df_prepared):,}")

# ── 4. Calcular nuevos umbrales ───────────────────────────────────────
detector = StateDetector(StateConfig())
new_thresholds = detector.compute_thresholds(df_prepared)

print(f"\n=== Comparativa de umbrales ===")
for k in new_thresholds:
    old = old_thresholds.get(k, 'N/A')
    new = new_thresholds[k]
    if isinstance(old, float):
        pct = (new / old - 1) * 100
        print(f"  {k:20s}: {old:.6f} → {new:.6f}  ({pct:+.1f}%)")
    else:
        print(f"  {k:20s}: {old} → {new:.6f}")

# Verificación: % VOLATILE con umbrales viejos vs nuevos
atr_norm = df_prepared['atr_norm']
if old_thresholds.get('vol_high'):
    pct_old = (atr_norm > old_thresholds['vol_high']).mean()
    print(f"\n  % VOLATILE umbral viejo : {pct_old*100:.1f}%  (esperado ~20%)")
pct_new = (atr_norm > new_thresholds['vol_high']).mean()
print(f"  % VOLATILE umbral nuevo : {pct_new*100:.1f}%  (esperado ~20%)")

# ── 5. Confirmar antes de escribir ───────────────────────────────────
confirm = input("\n¿Actualizar meta.json con los nuevos umbrales? (s/n): ")
if confirm.lower() != 's':
    print("Cancelado.")
    exit(0)

# ── 6. Backup + actualización ─────────────────────────────────────────
backup_path = META_PATH.parent / f'meta_backup_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
backup_path.write_text(META_PATH.read_text(encoding='utf-8'))
print(f"\n✅ Backup guardado: {backup_path}")

meta['regime_thresholds'] = new_thresholds
meta['regime_thresholds_recalib_date'] = datetime.now().isoformat()
meta['regime_thresholds_recalib_from'] = RECALIB_FROM
meta['regime_thresholds_recalib_to']   = RECALIB_TO
meta['regime_thresholds_old']          = old_thresholds

import json
from datetime import datetime, date
from pathlib import Path

def json_serializable(obj):
    """Convierte tipos no serializables a string."""
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')

meta['regime_thresholds']              = new_thresholds
meta['regime_thresholds_recalib_date'] = datetime.now().isoformat()
meta['regime_thresholds_recalib_from'] = RECALIB_FROM
meta['regime_thresholds_recalib_to']   = RECALIB_TO
meta['regime_thresholds_old']          = old_thresholds

META_PATH.write_text(
    json.dumps(meta, indent=2, default=json_serializable),
    encoding='utf-8'
)
print(f"✅ meta.json actualizado: {META_PATH}")
print("\nReinicia el engine de producción para cargar los nuevos umbrales.")
print("Monitoriza signals_YYYYMMDD.jsonl durante 30-60 min.")
print("El % de barras VOLATILE debería volver a ~20%.")