import json
from datetime import datetime, timedelta
from pathlib import Path

import joblib
import numpy as np

from mimo_old.data_manager import DataManager
from mimo_old.data_pipeline import DataPipeline
from mimo_old.databases import Database
from mimo_old.feature_builder import FeatureConfig
from mimo_old.model_builder import Config, ModelConfig, TradingModel
from mimo_old.model_evaluator import ModelEvaluator
from mimo_old.probs_calibration import ProbsCalibration
from mimo.strategies.regime_detector import RegimeConfig


def main(general_config: Config = None, side: str = None, path: str = None):
    feature_config = FeatureConfig(
        ema_periods=[9, 21, 50],
        label_horizon=5,
        label_method='adaptive'
    )

    model_config = ModelConfig(
        seq_len_short=64,
        seq_len_long=256,
        epochs=25,
        patience=10
    )

    regime_config = RegimeConfig(
        adx_trend_threshold=25.0
    )

    pipeline = DataPipeline(general_config=general_config, feature_config=feature_config, model_config=model_config, regime_config=regime_config)

    to_date = datetime(2025, 12, 18)
    from_date = to_date - timedelta(days=general_config.period)

    db = Database()
    dm = DataManager.from_database_historical_2(db, from_date, to_date)
    df = dm.df

    print('Preparando datos...')
    df = pipeline.prepare_data(df, side=side)
    oof_calibrator = None
    percentiles = None
    best_epochs = None
    probs_calibrator = ProbsCalibration(general_config=general_config, model_config=model_config)

    if getattr(general_config, 'use_oof', False):
        print('Generando predicciones OOF...')
        df_oof, oof_calibrator, best_epochs = probs_calibrator.generate_oof_predictions(
            df_prepared=df,
            pipeline=pipeline,
            side=side,
            n_splits=int(getattr(general_config, 'oof_splits', 5)),
            epochs_per_fold=int(getattr(general_config, 'oof_epochs', 25)),
            verbose=1
        )

        print('Calculando percentiles por régimen (OOF calibrado)...')
        percentiles = probs_calibrator.compute_percentiles_by_regime(
            df_with_oof=df_oof,
            proba_col='oof_proba_cal',
            regime_col='regime',
            quantiles=(50, 75, 90, 95, 97, 98, 99),
            map_to_3_regimes=True,
            min_n=1000
        )

        if getattr(general_config, 'save_oof_artifacts', False):
            Path(path).mkdir(parents=True, exist_ok=True)

            df_oof.to_parquet(f'{path}/df_oof_{general_config.release}_{side}.parquet', index=False)
            with open(f'{path}/percentiles_{general_config.release}_{side}.json', 'w', encoding='utf-8') as f:
                json.dump(percentiles, f, indent=2)

            with open(f'{path}/best_epochs_{general_config.release}_{side}.json', 'w', encoding='utf-8') as f:
                json.dump(best_epochs, f, indent=2)

            joblib.dump(oof_calibrator, f'{path}/oof_calibrator_{general_config.release}_{side}.joblib')
            print('Artefactos OOF guardados en ./models')

    # Split temporal
    split_idx = int(len(df) * (1 - general_config.val_size))

    df_train = df.iloc[:split_idx]
    df_val = df.iloc[split_idx - model_config.seq_len_long:]

    # Crear secuencias
    print("Creando secuencias...")
    data_train = pipeline.create_sequences(df_train, fit_scalers=True, train=True)
    data_val = pipeline.create_sequences(df_val, fit_scalers=False, train=True)

    X_train = {k: v for k, v in data_train.items() if k not in ['labels', 'weights']}
    y_train = data_train['labels']
    w_train = data_train['weights']

    X_val = {k: v for k, v in data_val.items() if k not in ['labels', 'weights']}
    y_val = data_val['labels']

    # Crear y entrenar modelo
    print("Creando modelo...")
    model = TradingModel(general_config=general_config, model_config=model_config, side=side)

    # Calcular bias inicial
    pos_rate = y_train.mean()
    init_bias = np.log(pos_rate / (1 - pos_rate))

    # Construir modelo
    model.build_model_v2(
        shape_short=(model_config.seq_len_short, len(pipeline.feature_engineer.feature_columns['sequence_short'])),
        shape_long=(model_config.seq_len_long, len(pipeline.feature_engineer.feature_columns['sequence_long'])),
        n_context=len(pipeline.feature_engineer.feature_columns['context']),
        n_time=len(pipeline.feature_engineer.feature_columns['time']),
        init_bias=init_bias
    )

    # Compilar
    model.compile_model()

    # Entrenar
    print("Entrenando modelo...")
    history = model.train(
        X_train, y_train,
        X_val, y_val,
        sample_weight=w_train,
        verbose=1
    )

    vals = history['val_auc_pr']
    best_epoch = int(np.argmax(vals)) + 1
    best_val = float(np.max(vals))

    # Evaluar
    print("Evaluando modelo...")
    y_pred = model.predict(X_val)
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_predictions(y_true=y_val, y_pred_proba=y_pred, beta_primary=0.25,
                                             min_precision=0.45, max_signal_rate=0.10, verbose=True)

    return model, pipeline, metrics, X_train, percentiles, oof_calibrator, best_epoch, best_val, best_epochs

if __name__ == '__main__':
    sides = ['long', 'short']
    scaler_saved = False
    models = {}
    release = '200269'
    for side in sides:
        config = Config(
            release=release,
            period=365,
            oof_epochs=100,
            oof_splits=12
        )

        model, pipeline, metrics, X_train, percentiles, oof_cal, best_epoch, best_val, best_epochs = (
            main(general_config=config, side=side, path='./models'))

        model.save(path='./models')
        models[side] = model

        if not scaler_saved:
            pipeline.save_scalers(path='./models')
            scaler_saved = True

    print('Sistema completado exitosamente!')