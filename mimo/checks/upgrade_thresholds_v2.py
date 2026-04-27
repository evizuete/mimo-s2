import json
from pathlib import Path
from datetime import datetime, date
import pandas as pd

from mimo.data_managers.data_manager import DataManager
from mimo.data_managers.databases import Database
from mimo.features.feature_builder import FeatureConfig
from mimo.models.model_builder import Config, ModelConfig
from mimo.data_managers.data_pipeline_v2 import DataPipeline
from mimo.states_manager.state_detector import StateDetector, StateConfig

RELEASE = '200373'
ARTIFACTS_PATH = Path(f'../../artifacts/{RELEASE}/oof/deploy_full')
META_PATH = ARTIFACTS_PATH / f'scalers_{RELEASE}' / 'meta.json'

WINDOWS = [
    ('4m_recent', datetime(2025, 12, 1)),
    ('3m_recent', datetime(2026, 1, 1)),
]

FORCE_WINDOW = None
EXPECTED_VOLATILE = 0.20
EXPECTED_LOW_VOL = 0.30


def json_serializable(obj):
    if isinstance(obj, (datetime, date, pd.Timestamp)):
        return obj.isoformat()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


def load_meta():
    if not META_PATH.exists():
        raise FileNotFoundError(f"No encontrado: {META_PATH}")
    return json.loads(META_PATH.read_text(encoding='utf-8'))


def get_last_db_timestamp(db: Database) -> pd.Timestamp:
    q = "SELECT MAX(time) AS max_time FROM rates"
    df = pd.read_sql(q, db.engine)
    ts = df["max_time"].iloc[0]
    if ts is None or pd.isna(ts):
        raise RuntimeError("No se pudo obtener MAX(time) de la tabla rates")
    return pd.Timestamp(ts)


def prepare_pipeline():
    general_config = Config(release=RELEASE, oof_splits=3, oof_epochs=25)
    model_config = ModelConfig(seq_len_short=64, seq_len_long=256, batch_size=8192)
    feature_config = FeatureConfig(
        ema_periods=[9, 21, 50],
        label_horizon=10,
        tp_barrier=2.5,
        sl_barrier=1.5,
        label_method='adaptive',
    )
    return DataPipeline(
        general_config=general_config,
        feature_config=feature_config,
        model_config=model_config,
    )


def evaluate_window(db, pipeline, from_dt: datetime, to_dt: datetime):
    df_raw = DataManager.from_database_historical_2(
        db, from_date=from_dt, to_date=to_dt
    ).df
    df_raw = df_raw.sort_values('time').reset_index(drop=True)

    if df_raw.empty:
        raise RuntimeError(f"Sin datos en ventana {from_dt} → {to_dt}")

    df_prepared = pipeline.prepare_data(df_raw, labels=False, side='long')
    if df_prepared.empty:
        raise RuntimeError(f"Sin filas tras prepare_data en ventana {from_dt} → {to_dt}")

    detector = StateDetector(StateConfig())
    new_thresholds = detector.compute_thresholds(df_prepared)

    atr_norm = df_prepared['atr_norm']
    pct_low = float((atr_norm < new_thresholds['vol_low']).mean())
    pct_high = float((atr_norm > new_thresholds['vol_high']).mean())

    score = abs(pct_low - EXPECTED_LOW_VOL) + abs(pct_high - EXPECTED_VOLATILE)

    return {
        'from': from_dt,
        'to': to_dt,
        'raw_rows': len(df_raw),
        'prepared_rows': len(df_prepared),
        'thresholds': new_thresholds,
        'pct_low': pct_low,
        'pct_high': pct_high,
        'score': score,
    }


def print_old_vs_new(old_thresholds, new_thresholds):
    print("\n=== Comparativa de umbrales ===")
    for k in new_thresholds:
        old = old_thresholds.get(k, 'N/A')
        new = new_thresholds[k]
        if isinstance(old, float):
            pct = (new / old - 1) * 100 if old != 0 else float('inf')
            print(f"  {k:20s}: {old:.6f} → {new:.6f}  ({pct:+.1f}%)")
        else:
            print(f"  {k:20s}: {old} → {new:.6f}")


def main():
    meta = load_meta()
    old_thresholds = meta.get('regime_thresholds', {})
    print(f"Umbrales actuales: {old_thresholds}")

    db = Database()
    db.connect()

    last_ts = get_last_db_timestamp(db)
    recalib_to = last_ts.to_pydatetime()
    print(f"Última fecha disponible en BD: {last_ts}")

    pipeline = prepare_pipeline()

    results = []
    for label, from_dt in WINDOWS:
        try:
            res = evaluate_window(db, pipeline, from_dt, recalib_to)
            res['label'] = label
            results.append(res)
        except Exception as e:
            print(f"\n⚠️  Ventana {label} falló: {e}")

    if not results:
        raise RuntimeError("No se pudo evaluar ninguna ventana de recalibración")

    print("\n=== Evaluación de ventanas candidatas ===")
    for r in results:
        print(
            f"  {r['label']:10s} | {r['from']} → {r['to']} | "
            f"raw={r['raw_rows']:,} prep={r['prepared_rows']:,} | "
            f"low_vol={r['pct_low']*100:5.1f}% | volatile={r['pct_high']*100:5.1f}% | "
            f"score={r['score']:.4f}"
        )

    if FORCE_WINDOW:
        chosen = next((r for r in results if r['label'] == FORCE_WINDOW), None)
        if chosen is None:
            raise ValueError(f"FORCE_WINDOW={FORCE_WINDOW} no coincide con ninguna ventana")
    else:
        chosen = min(results, key=lambda x: x['score'])

    print(f"\nVentana seleccionada: {chosen['label']}")
    print(f"  Desde: {chosen['from']}")
    print(f"  Hasta: {chosen['to']}")
    print(f"  low_vol  esperado ~{EXPECTED_LOW_VOL*100:.0f}%  →  real {chosen['pct_low']*100:.1f}%")
    print(f"  volatile esperado ~{EXPECTED_VOLATILE*100:.0f}%  →  real {chosen['pct_high']*100:.1f}%")

    new_thresholds = chosen['thresholds']
    print_old_vs_new(old_thresholds, new_thresholds)

    if old_thresholds.get('vol_high'):
        df_old = DataManager.from_database_historical_2(
            db, from_date=chosen['from'], to_date=chosen['to']
        ).df.sort_values('time').reset_index(drop=True)
        df_old_prepared = pipeline.prepare_data(df_old, labels=False, side='long')
        atr_norm = df_old_prepared['atr_norm']
        pct_old = float((atr_norm > old_thresholds['vol_high']).mean())
        pct_new = float((atr_norm > new_thresholds['vol_high']).mean())
        print(f"\n  % VOLATILE umbral viejo : {pct_old*100:.1f}%  (esperado ~20%)")
        print(f"  % VOLATILE umbral nuevo : {pct_new*100:.1f}%  (esperado ~20%)")

    confirm = input("\n¿Actualizar meta.json con los nuevos umbrales? (s/n): ")
    if confirm.lower() != 's':
        print("Cancelado.")
        return

    backup_path = META_PATH.parent / f"meta_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    backup_path.write_text(META_PATH.read_text(encoding='utf-8'))
    print(f"\n✅ Backup guardado: {backup_path}")

    meta['regime_thresholds'] = new_thresholds
    meta['regime_thresholds_recalib_date'] = datetime.now().isoformat()
    meta['regime_thresholds_recalib_from'] = chosen['from']
    meta['regime_thresholds_recalib_to'] = chosen['to']
    meta['regime_thresholds_old'] = old_thresholds
    meta['regime_thresholds_recalib_window_label'] = chosen['label']
    meta['regime_thresholds_candidate_windows'] = [
        {
            'label': r['label'],
            'from': r['from'],
            'to': r['to'],
            'raw_rows': r['raw_rows'],
            'prepared_rows': r['prepared_rows'],
            'pct_low': r['pct_low'],
            'pct_high': r['pct_high'],
            'score': r['score'],
        }
        for r in results
    ]

    META_PATH.write_text(
        json.dumps(meta, indent=2, default=json_serializable),
        encoding='utf-8'
    )
    print(f"✅ meta.json actualizado: {META_PATH}")
    print("\nReinicia el engine de producción para cargar los nuevos umbrales.")
    print("Monitoriza signals_YYYYMMDD.jsonl durante 30-60 min.")
    print("El % de barras VOLATILE debería volver a ~20%.")


if __name__ == '__main__':
    main()
