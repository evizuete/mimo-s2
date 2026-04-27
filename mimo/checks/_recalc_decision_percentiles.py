
import json
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import joblib

from mimo.states_manager.state_detector import StateDetector, StateConfig

RELEASE = '200372'
ARTIFACTS_PATH = Path(f'../../artifacts/{RELEASE}/oof/deploy_full')
DATA_PATH = ARTIFACTS_PATH / 'data'

META_PATH = ARTIFACTS_PATH / f'scalers_{RELEASE}' / 'meta.json'
LONG_JSON_PATH = ARTIFACTS_PATH / f'percentiles_{RELEASE}_long.json'
SHORT_JSON_PATH = ARTIFACTS_PATH / f'percentiles_{RELEASE}_short.json'

OOF_LONG_PATH = ARTIFACTS_PATH / f'oof_{RELEASE}_long.parquet'
OOF_SHORT_PATH = ARTIFACTS_PATH / f'oof_{RELEASE}_short.parquet'
CAL_LONG_PATH = ARTIFACTS_PATH / f'oof_calibrator_{RELEASE}_long.joblib'
CAL_SHORT_PATH = ARTIFACTS_PATH / f'oof_calibrator_{RELEASE}_short.joblib'

HOLDOUT_LONG_PATH = DATA_PATH / f'holdout_predictions_{RELEASE}_long.parquet'
HOLDOUT_SHORT_PATH = DATA_PATH / f'holdout_predictions_{RELEASE}_short.parquet'

QUANTILES = [50, 60, 70, 75, 80, 85, 90, 95, 96, 97, 98, 99]
NO_TRADE_STATES = {'LOW_VOL'}

ABS_DELTA_WARN = 0.015
REL_DELTA_WARN = 0.05


def json_serializable(obj):
    if isinstance(obj, (datetime, date, pd.Timestamp)):
        return obj.isoformat()
    raise TypeError(f'Object of type {obj.__class__.__name__} is not JSON serializable')


def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f'No encontrado: {path}')
    return json.loads(path.read_text(encoding='utf-8'))


def save_json(path: Path, payload: dict):
    path.write_text(json.dumps(payload, indent=2, default=json_serializable), encoding='utf-8')


def backup_path_for(path: Path) -> Path:
    return path.parent / f"{path.stem}_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}{path.suffix}"


def pick_label_col(df: pd.DataFrame):
    for c in ['label', 'y_true', 'target', 'y', 'label_bin']:
        if c in df.columns:
            return c
    return None


def pick_raw_score_col(df: pd.DataFrame, side: str) -> str:
    for c in ['oof_proba', 'proba', 'proba_raw', 'y_prob', 'prob', 'pred_proba', 'pred', 'y_pred_raw', 'oof_proba_raw']:
        if c in df.columns:
            return c
    raise KeyError(f'No se encontró score/probabilidad raw para side={side}. Columnas: {list(df.columns)[:100]} ...')


def pick_calibrated_col(df: pd.DataFrame, side: str):
    for c in ['oof_proba_cal', 'proba_cal', 'prob_cal', 'pred_cal', 'y_pred_cal']:
        if c in df.columns:
            return c
    return None


def ensure_calibrated_score(df: pd.DataFrame, side: str, calibrator_path: Path):
    df = df.copy()
    cal_col = pick_calibrated_col(df, side)
    if cal_col is not None:
        return df, cal_col

    raw_col = pick_raw_score_col(df, side)
    if not calibrator_path.exists():
        raise FileNotFoundError(f'No se encontró calibrador para {side}: {calibrator_path}')

    calibrator = joblib.load(calibrator_path)
    raw = pd.to_numeric(df[raw_col], errors='coerce').fillna(0.0).to_numpy().reshape(-1, 1)

    if hasattr(calibrator, 'predict_proba'):
        out = calibrator.predict_proba(raw)
        cal = out[:, 1] if getattr(out, 'ndim', 1) == 2 and out.shape[1] >= 2 else out.reshape(-1)
    elif hasattr(calibrator, 'predict'):
        cal = calibrator.predict(raw).reshape(-1)
    elif hasattr(calibrator, 'transform'):
        cal = calibrator.transform(raw).reshape(-1)
    else:
        raise TypeError(f'Calibrador no soportado: {type(calibrator)}')

    df['oof_proba_cal'] = cal
    return df, 'oof_proba_cal'


def add_recalculated_states(df: pd.DataFrame, old_thresholds: dict, new_thresholds: dict) -> pd.DataFrame:
    df = df.copy()

    if 'state' in df.columns:
        df = df.rename(columns={'state': 'state_saved'})

    det_old = StateDetector(StateConfig())
    det_old.inject_thresholds(old_thresholds)
    df_old = det_old.detect(df.copy())
    df['state_old_recalc'] = df_old['state']

    det_new = StateDetector(StateConfig())
    det_new.inject_thresholds(new_thresholds)
    df_new = det_new.detect(df.copy())
    df['state_new_recalc'] = df_new['state']

    return df


def compute_percentiles_by_state(df: pd.DataFrame, state_col: str, score_col: str):
    out = {}
    states_found = sorted({str(s).upper() for s in df[state_col].dropna().astype(str).tolist()})

    for state in states_found:
        mask = df[state_col].astype(str).str.upper() == state
        series = pd.to_numeric(df.loc[mask, score_col], errors='coerce').dropna()
        if len(series) == 0:
            continue

        pcts = {}
        for q in QUANTILES:
            pcts[f'p{q}'] = float(series.quantile(q / 100.0))

        entry = {'percentiles': pcts, 'n': int(len(series))}
        if state in NO_TRADE_STATES:
            entry['no_trade'] = True
        out[state] = entry

    mask_global = ~df[state_col].astype(str).str.upper().isin(NO_TRADE_STATES)
    series = pd.to_numeric(df.loc[mask_global, score_col], errors='coerce').dropna()
    global_pcts = {}
    for q in QUANTILES:
        global_pcts[f'p{q}'] = float(series.quantile(q / 100.0))

    out['_global'] = {
        'percentiles': global_pcts,
        'n': int(len(series)),
        'note': 'Excludes NO_TRADE_STATES'
    }
    out['_meta'] = {
        'proba_col': score_col,
        'regime_col': state_col,
        'quantiles': QUANTILES,
        'no_trade_states': sorted(NO_TRADE_STATES),
        'states_found': states_found,
    }
    return out


def compare_percentile_jsons(old_payload: dict, new_payload: dict, label: str):
    print(f'\n=== Comparativa {label} ===')
    states = sorted(
        set(k for k in old_payload.keys() if not k.startswith('_')) |
        set(k for k in new_payload.keys() if not k.startswith('_'))
    )
    flagged = []

    for state in states:
        old_entry = old_payload.get(state, {})
        new_entry = new_payload.get(state, {})
        print(f"\n[{state}] n_old={old_entry.get('n')}  n_new={new_entry.get('n')}")

        old_p = old_entry.get('percentiles', {})
        new_p = new_entry.get('percentiles', {})

        for q in (75, 80, 90, 95, 97, 99):
            key = f'p{q}'
            if key not in old_p or key not in new_p:
                continue
            ov = float(old_p[key])
            nv = float(new_p[key])
            delta_abs = nv - ov
            delta_rel = (nv / ov - 1.0) if ov else 0.0
            mark = ''
            if abs(delta_abs) >= ABS_DELTA_WARN or abs(delta_rel) >= REL_DELTA_WARN:
                mark = '  <-- REVIEW'
                flagged.append((state, key, ov, nv, delta_abs, delta_rel))
            print(f"  {key:>3s}: {ov:.6f} → {nv:.6f}  (Δabs={delta_abs:+.6f}, Δrel={delta_rel*100:+.1f}%)" + mark)

    return flagged


def compare_state_distribution(df: pd.DataFrame, old_col: str, new_col: str, label: str):
    print(f'\n=== Distribución de estados {label} ===')
    old_counts = df[old_col].astype(str).str.upper().value_counts(normalize=True)
    new_counts = df[new_col].astype(str).str.upper().value_counts(normalize=True)
    states = sorted(set(old_counts.index) | set(new_counts.index))
    for st in states:
        old_v = float(old_counts.get(st, 0.0))
        new_v = float(new_counts.get(st, 0.0))
        print(f"  {st:18s}: old={old_v*100:5.1f}%  new={new_v*100:5.1f}%  Δ={(new_v-old_v)*100:+.1f}pp")


def validate_on_holdout(holdout_df: pd.DataFrame, old_state_col: str, new_state_col: str, score_col: str, old_payload: dict, new_payload: dict, label: str):
    y_col = pick_label_col(holdout_df)
    if y_col is None:
        print(f'\n[WARN] {label}: holdout sin label; se omite validación de hit-rate.')
        return

    print(f'\n=== Validación holdout {label} ===')
    states = sorted(set(k for k in old_payload.keys() if not k.startswith('_')) | set(k for k in new_payload.keys() if not k.startswith('_')))
    for state in states:
        if state not in old_payload or state not in new_payload:
            continue

        old_pcts = old_payload[state].get('percentiles', {})
        new_pcts = new_payload[state].get('percentiles', {})
        ref_key = 'p97' if 'p97' in old_pcts and 'p97' in new_pcts else ('p95' if 'p95' in old_pcts and 'p95' in new_pcts else None)
        if ref_key is None:
            continue

        old_thr = float(old_pcts[ref_key])
        new_thr = float(new_pcts[ref_key])

        scores = pd.to_numeric(holdout_df[score_col], errors='coerce')
        labels = pd.to_numeric(holdout_df[y_col], errors='coerce')

        mask_old = holdout_df[old_state_col].astype(str).str.upper() == state
        mask_new = holdout_df[new_state_col].astype(str).str.upper() == state

        sel_old = mask_old & (scores >= old_thr)
        sel_new = mask_new & (scores >= new_thr)

        n_old = int(sel_old.sum())
        n_new = int(sel_new.sum())
        hr_old = float(labels[sel_old].mean()) if n_old > 0 else float('nan')
        hr_new = float(labels[sel_new].mean()) if n_new > 0 else float('nan')

        print(f"  {state:18s} {ref_key}: thr_old={old_thr:.6f} thr_new={new_thr:.6f} | n_old={n_old:5d} n_new={n_new:5d} | hit_old={hr_old:.4f} hit_new={hr_new:.4f}")


def main():
    meta = load_json(META_PATH)
    old_thresholds = meta.get('regime_thresholds_old')
    new_thresholds = meta.get('regime_thresholds')
    if not old_thresholds or not new_thresholds:
        raise ValueError('meta.json debe contener regime_thresholds_old y regime_thresholds')

    long_old_json = load_json(LONG_JSON_PATH)
    short_old_json = load_json(SHORT_JSON_PATH)

    print('Usando thresholds viejos:')
    print(json.dumps(old_thresholds, indent=2))
    print('\nUsando thresholds nuevos:')
    print(json.dumps(new_thresholds, indent=2))

    long_oof = pd.read_parquet(OOF_LONG_PATH)
    short_oof = pd.read_parquet(OOF_SHORT_PATH)
    long_hold = pd.read_parquet(HOLDOUT_LONG_PATH)
    short_hold = pd.read_parquet(HOLDOUT_SHORT_PATH)

    long_oof, long_score_col = ensure_calibrated_score(long_oof, 'long', CAL_LONG_PATH)
    short_oof, short_score_col = ensure_calibrated_score(short_oof, 'short', CAL_SHORT_PATH)
    long_hold, long_hold_score_col = ensure_calibrated_score(long_hold, 'long', CAL_LONG_PATH)
    short_hold, short_hold_score_col = ensure_calibrated_score(short_hold, 'short', CAL_SHORT_PATH)

    long_oof = add_recalculated_states(long_oof, old_thresholds, new_thresholds)
    short_oof = add_recalculated_states(short_oof, old_thresholds, new_thresholds)

    # holdout_predictions no trae features suficientes para recalcular estado
    # usamos el state ya guardado como referencia operativa
    if 'state' in long_hold.columns:
        long_hold = long_hold.rename(columns={'state': 'state_saved'})
        long_hold['state_old_recalc'] = long_hold['state_saved']
        long_hold['state_new_recalc'] = long_hold['state_saved']

    if 'state' in short_hold.columns:
        short_hold = short_hold.rename(columns={'state': 'state_saved'})
        short_hold['state_old_recalc'] = short_hold['state_saved']
        short_hold['state_new_recalc'] = short_hold['state_saved']

    print(f"\nOOF long rows   : {len(long_oof):,} | score_col={long_score_col}")
    print(f"OOF short rows  : {len(short_oof):,} | score_col={short_score_col}")
    print(f"Holdout long rows  : {len(long_hold):,} | score_col={long_hold_score_col}")
    print(f"Holdout short rows : {len(short_hold):,} | score_col={short_hold_score_col}")

    compare_state_distribution(long_oof, 'state_old_recalc', 'state_new_recalc', 'OOF LONG')
    compare_state_distribution(short_oof, 'state_old_recalc', 'state_new_recalc', 'OOF SHORT')

    long_new_json = compute_percentiles_by_state(long_oof, 'state_new_recalc', long_score_col)
    short_new_json = compute_percentiles_by_state(short_oof, 'state_new_recalc', short_score_col)

    flagged_long = compare_percentile_jsons(long_old_json, long_new_json, 'LONG')
    flagged_short = compare_percentile_jsons(short_old_json, short_new_json, 'SHORT')

    validate_on_holdout(long_hold, 'state_old_recalc', 'state_new_recalc', long_hold_score_col, long_old_json, long_new_json, 'LONG')
    validate_on_holdout(short_hold, 'state_old_recalc', 'state_new_recalc', short_hold_score_col, short_old_json, short_new_json, 'SHORT')

    print('\n=== Resumen ===')
    print(f'Estados/percentiles LONG a revisar : {len(flagged_long)}')
    print(f'Estados/percentiles SHORT a revisar: {len(flagged_short)}')

    if flagged_long:
        print('\nLONG flagged:')
        for state, key, ov, nv, da, dr in flagged_long[:40]:
            print(f"  {state:18s} {key:>3s}: {ov:.6f} → {nv:.6f}  (Δabs={da:+.6f}, Δrel={dr*100:+.1f}%)")
    if flagged_short:
        print('\nSHORT flagged:')
        for state, key, ov, nv, da, dr in flagged_short[:40]:
            print(f"  {state:18s} {key:>3s}: {ov:.6f} → {nv:.6f}  (Δabs={da:+.6f}, Δrel={dr*100:+.1f}%)")

    confirm = input('\n¿Actualizar los JSON de percentiles con los nuevos valores? (s/n): ')
    if confirm.lower() != 's':
        print('Cancelado.')
        return

    long_backup = backup_path_for(LONG_JSON_PATH)
    short_backup = backup_path_for(SHORT_JSON_PATH)
    save_json(long_backup, long_old_json)
    save_json(short_backup, short_old_json)
    print(f'\n✅ Backup long : {long_backup}')
    print(f'✅ Backup short: {short_backup}')

    save_json(LONG_JSON_PATH, long_new_json)
    save_json(SHORT_JSON_PATH, short_new_json)
    print(f'✅ Actualizado: {LONG_JSON_PATH}')
    print(f'✅ Actualizado: {SHORT_JSON_PATH}')

    meta['decision_percentiles_recalib_date'] = datetime.now().isoformat()
    meta['decision_percentiles_long_source'] = str(LONG_JSON_PATH)
    meta['decision_percentiles_short_source'] = str(SHORT_JSON_PATH)
    meta['decision_percentiles_flagged_long_n'] = len(flagged_long)
    meta['decision_percentiles_flagged_short_n'] = len(flagged_short)

    save_json(META_PATH, meta)
    print(f'✅ meta.json actualizado: {META_PATH}')

    print('\nReinicia el engine de producción y monitoriza 30–60 min:')
    print('  - SIGNAL_SENT / NO_SIGNAL')
    print('  - aceptación por estado')
    print('  - distribución de estados')
    print('  - MODIFY_FAILED / ADAPTIVE_INDICATORS_STALE')


if __name__ == '__main__':
    main()
