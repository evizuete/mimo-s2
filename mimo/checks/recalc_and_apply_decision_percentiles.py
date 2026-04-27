import json
from pathlib import Path
from datetime import datetime, date
import pandas as pd
import joblib

from mimo.states_manager.state_detector import StateDetector, StateConfig

RELEASE = '200373'
ARTIFACTS_PATH = Path(f'../../artifacts/{RELEASE}/oof/deploy_full')
DATA_PATH = ARTIFACTS_PATH / 'data'

META_PATH = ARTIFACTS_PATH / f'scalers_{RELEASE}' / 'meta.json'
LONG_JSON_PATH = ARTIFACTS_PATH / f'percentiles_{RELEASE}_long.json'
SHORT_JSON_PATH = ARTIFACTS_PATH / f'percentiles_{RELEASE}_short.json'
LONG_RECALC_PATH = ARTIFACTS_PATH / f'percentiles_{RELEASE}_long_recalc.json'
SHORT_RECALC_PATH = ARTIFACTS_PATH / f'percentiles_{RELEASE}_short_recalc.json'

OOF_LONG_PATH = ARTIFACTS_PATH / f'oof_{RELEASE}_long.parquet'
OOF_SHORT_PATH = ARTIFACTS_PATH / f'oof_{RELEASE}_short.parquet'
CAL_LONG_PATH = ARTIFACTS_PATH / f'oof_calibrator_{RELEASE}_long.joblib'
CAL_SHORT_PATH = ARTIFACTS_PATH / f'oof_calibrator_{RELEASE}_short.joblib'

HOLDOUT_LONG_PATH = DATA_PATH / f'holdout_predictions_{RELEASE}_long.parquet'
HOLDOUT_SHORT_PATH = DATA_PATH / f'holdout_predictions_{RELEASE}_short.parquet'

QUANTILES = [50, 60, 70, 75, 80, 85, 90, 95, 96, 97, 98, 99]

# Regla objetiva de actualización
ABS_DELTA_RULE = 0.015
REL_DELTA_RULE = 0.05   # 5%
TARGET_PERCENTILES = ['p75', 'p80', 'p90', 'p95', 'p97', 'p99']

# Si False, excluye estados no operativos de la actualización automática
INCLUDE_NON_TRADE_STATES = True
NON_TRADE_STATES = {'LOW_VOL', 'VOLATILE'}


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
        if state == 'LOW_VOL':
            entry['no_trade'] = True
        out[state] = entry

    mask_global = ~df[state_col].astype(str).str.upper().isin({'LOW_VOL'})
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
        'no_trade_states': ['LOW_VOL'],
        'states_found': states_found,
    }
    return out


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


def should_consider_state(state: str) -> bool:
    if INCLUDE_NON_TRADE_STATES:
        return True
    return state not in NON_TRADE_STATES


def build_update_plan(base_payload: dict, recalc_payload: dict, side: str):
    plan = {}
    rows = []

    states = sorted(
        set(k for k in base_payload.keys() if not k.startswith('_')) |
        set(k for k in recalc_payload.keys() if not k.startswith('_'))
    )

    for state in states:
        if not should_consider_state(state):
            continue
        if state not in base_payload or state not in recalc_payload:
            continue

        base_p = base_payload[state].get('percentiles', {})
        recalc_p = recalc_payload[state].get('percentiles', {})
        changed_keys = []

        for key in TARGET_PERCENTILES:
            if key not in base_p or key not in recalc_p:
                continue

            old_v = float(base_p[key])
            new_v = float(recalc_p[key])
            delta_abs = new_v - old_v
            delta_rel = (new_v / old_v - 1.0) if old_v else 0.0

            flag = abs(delta_abs) >= ABS_DELTA_RULE or abs(delta_rel) >= REL_DELTA_RULE
            rows.append((side, state, key, old_v, new_v, delta_abs, delta_rel, flag))

            if flag:
                changed_keys.append(key)

        if changed_keys:
            plan[state] = changed_keys

    return plan, rows


def apply_updates(base_payload: dict, recalc_payload: dict, plan: dict, side: str):
    updated = json.loads(json.dumps(base_payload))
    changes = []

    for state, keys in plan.items():
        if state not in updated or state not in recalc_payload:
            continue

        target_p = updated[state].setdefault('percentiles', {})
        recalc_p = recalc_payload[state].get('percentiles', {})

        for key in keys:
            if key not in target_p or key not in recalc_p:
                continue
            old_v = float(target_p[key])
            new_v = float(recalc_p[key])
            target_p[key] = new_v
            changes.append((side, state, key, old_v, new_v, new_v - old_v))

    return updated, changes


def main():
    meta = load_json(META_PATH)
    old_thresholds = meta.get('regime_thresholds_old')
    new_thresholds = meta.get('regime_thresholds')
    if not old_thresholds or not new_thresholds:
        raise ValueError('meta.json debe contener regime_thresholds_old y regime_thresholds')

    long_base = load_json(LONG_JSON_PATH)
    short_base = load_json(SHORT_JSON_PATH)

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

    long_recalc = compute_percentiles_by_state(long_oof, 'state_new_recalc', long_score_col)
    short_recalc = compute_percentiles_by_state(short_oof, 'state_new_recalc', short_score_col)

    save_json(LONG_RECALC_PATH, long_recalc)
    save_json(SHORT_RECALC_PATH, short_recalc)
    print(f'\n✅ Recalculado long guardado en : {LONG_RECALC_PATH}')
    print(f'✅ Recalculado short guardado en: {SHORT_RECALC_PATH}')

    validate_on_holdout(long_hold, 'state_old_recalc', 'state_new_recalc', long_hold_score_col, long_base, long_recalc, 'LONG')
    validate_on_holdout(short_hold, 'state_old_recalc', 'state_new_recalc', short_hold_score_col, short_base, short_recalc, 'SHORT')

    long_plan, long_rows = build_update_plan(long_base, long_recalc, 'long')
    short_plan, short_rows = build_update_plan(short_base, short_recalc, 'short')

    print('\n=== Regla de actualización ===')
    print(f'  |Δabs| >= {ABS_DELTA_RULE}')
    print(f'  |Δrel| >= {REL_DELTA_RULE*100:.1f}%')
    print(f'  INCLUDE_NON_TRADE_STATES = {INCLUDE_NON_TRADE_STATES}')
    print(f'  TARGET_PERCENTILES = {TARGET_PERCENTILES}')

    print('\n=== Evaluación LONG ===')
    for side, state, key, old_v, new_v, delta_abs, delta_rel, flag in long_rows:
        mark = '  <-- UPDATE' if flag else ''
        print(f'  {state:18s} {key:>3s}: {old_v:.6f} -> {new_v:.6f}  (Δabs={delta_abs:+.6f}, Δrel={delta_rel*100:+.1f}%)' + mark)

    print('\n=== Evaluación SHORT ===')
    for side, state, key, old_v, new_v, delta_abs, delta_rel, flag in short_rows:
        mark = '  <-- UPDATE' if flag else ''
        print(f'  {state:18s} {key:>3s}: {old_v:.6f} -> {new_v:.6f}  (Δabs={delta_abs:+.6f}, Δrel={delta_rel*100:+.1f}%)' + mark)

    print('\n=== Plan LONG ===')
    if long_plan:
        for state, keys in long_plan.items():
            print(f'  {state:18s}: {keys}')
    else:
        print('  Sin cambios.')

    print('\n=== Plan SHORT ===')
    if short_plan:
        for state, keys in short_plan.items():
            print(f'  {state:18s}: {keys}')
    else:
        print('  Sin cambios.')

    long_updated, long_changes = apply_updates(long_base, long_recalc, long_plan, 'long')
    short_updated, short_changes = apply_updates(short_base, short_recalc, short_plan, 'short')
    all_changes = long_changes + short_changes

    print('\n=== Cambios que se aplicarían ===')
    if not all_changes:
        print('  No hay cambios a aplicar con la regla actual.')
        return

    for side, state, key, old_v, new_v, delta in all_changes:
        print(f'  {side:5s} {state:18s} {key:>3s}: {old_v:.6f} -> {new_v:.6f}  (Δ={delta:+.6f})')

    confirm = input('\n¿Aplicar actualización automática por regla a los JSON de percentiles? (s/n): ')
    if confirm.lower() != 's':
        print('Recalculados guardados. No se tocan los JSON definitivos.')
        return

    long_backup = backup_path_for(LONG_JSON_PATH)
    short_backup = backup_path_for(SHORT_JSON_PATH)
    save_json(long_backup, long_base)
    save_json(short_backup, short_base)

    save_json(LONG_JSON_PATH, long_updated)
    save_json(SHORT_JSON_PATH, short_updated)

    print(f'\n✅ Backup long : {long_backup}')
    print(f'✅ Backup short: {short_backup}')
    print(f'✅ Actualizado : {LONG_JSON_PATH}')
    print(f'✅ Actualizado : {SHORT_JSON_PATH}')

    meta['decision_percentiles_rule_update_date'] = datetime.now().isoformat()
    meta['decision_percentiles_rule_update_release'] = RELEASE
    meta['decision_percentiles_rule_update_long_source'] = str(LONG_RECALC_PATH)
    meta['decision_percentiles_rule_update_short_source'] = str(SHORT_RECALC_PATH)
    meta['decision_percentiles_rule_update_abs_delta'] = ABS_DELTA_RULE
    meta['decision_percentiles_rule_update_rel_delta'] = REL_DELTA_RULE
    meta['decision_percentiles_rule_update_include_non_trade_states'] = INCLUDE_NON_TRADE_STATES
    meta['decision_percentiles_rule_update_target_percentiles'] = TARGET_PERCENTILES
    meta['decision_percentiles_rule_update_long_plan'] = long_plan
    meta['decision_percentiles_rule_update_short_plan'] = short_plan
    meta['decision_percentiles_rule_update_n_changes'] = len(all_changes)

    save_json(META_PATH, meta)
    print(f'✅ meta.json actualizado: {META_PATH}')

    print('\nSiguiente paso recomendado:')
    print('  1. Reiniciar S2/S3')
    print('  2. Monitorizar 30–60 min')
    print('  3. Revisar SIGNAL_SENT / NO_SIGNAL por estado')
    print('  4. Ver si RANGE / TRANSITION quedan mejor equilibrados')


if __name__ == '__main__':
    main()
