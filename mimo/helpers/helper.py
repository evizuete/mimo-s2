import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from keras import Model
from keras.src.saving import load_model

from mimo.data_managers.data_manager import DataManager
from mimo.data_managers.data_pipeline_v2 import DataPipeline
from mimo.data_managers.databases import Database
from mimo.models.model_builder import Config

class Helper:
    def __init__(self, general_config: Config, path: str = None):
        self.general_config = general_config
        self.path = '.' if path is None else path
        self.database = Database()

    def load_calibrator(self, side: str):
        filename = f'oof_calibrator_{self.general_config.release}_{side}.joblib'
        #print(f'Calibrator file: {filename}')
        calibrator = joblib.load(f'{self.path}/{filename}')
        return calibrator

    def predict_proba_keras(self, model: Model, X: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        raw = model.predict(X, batch_size=batch_size, verbose=0)
        # multitask: output dict/list con [signal_long, signal_short].
        # Devolvemos shape (N, 2) — col 0=P_long, col 1=P_short.
        if isinstance(raw, dict):
            if 'signal_long' in raw and 'signal_short' in raw:
                p_l = np.asarray(raw['signal_long']).reshape(-1)
                p_s = np.asarray(raw['signal_short']).reshape(-1)
                return np.stack([p_l, p_s], axis=-1)
        if isinstance(raw, (list, tuple)) and len(raw) == 2:
            p_l = np.asarray(raw[0]).reshape(-1)
            p_s = np.asarray(raw[1]).reshape(-1)
            return np.stack([p_l, p_s], axis=-1)
        # triple_class: output (N, 3) softmax → P(TP) = col 2.
        if raw.ndim == 2 and raw.shape[-1] == 3:
            return raw[:, 2]
        return raw.reshape(-1)

    def load_model(self, side: str):
        model = load_model(f'{self.path}/model_{self.general_config.release}_{side}.keras')
        return model

    def load_everything(self, pipeline: DataPipeline):
        # Multitask: si existe `model_{release}_multitask.keras` cargamos UN
        # solo modelo y un calibrador único (dict {'long': ..., 'short': ...}).
        # Devolvemos models como {'long': model, 'short': model} apuntando a
        # la misma instancia, y calibradores expandidos por lado.
        release = self.general_config.release
        multitask_model_path = Path(f'{self.path}/model_{release}_multitask.keras')
        multitask_cal_path = Path(f'{self.path}/oof_calibrator_{release}_multitask.joblib')

        if multitask_model_path.exists() and multitask_cal_path.exists():
            multi_model = load_model(str(multitask_model_path))
            cal_dict = joblib.load(str(multitask_cal_path))
            if not isinstance(cal_dict, dict) or 'long' not in cal_dict or 'short' not in cal_dict:
                raise ValueError(
                    f"Calibrator multitask inválido en {multitask_cal_path}: "
                    f"se esperaba dict con keys 'long'/'short'."
                )
            models = {'long': multi_model, 'short': multi_model}
            calibrators = {'long': cal_dict['long'], 'short': cal_dict['short']}
        else:
            models = {}
            calibrators = {}
            for side in ['long', 'short']:
                models[side] = self.load_model(side)
                calibrators[side] = self.load_calibrator(side)

        scalers = pipeline.load_scalers(self.path)
        return models, calibrators, scalers

    @staticmethod
    def load_from_dataframe(df_rates: pd.DataFrame) -> pd.DataFrame:
        """Normaliza tipos mínimos; DataPipeline espera OHLC + time."""
        df = df_rates.copy()
        if "time" in df.columns:
            df["time"] = pd.to_datetime(df["time"])
        return df

    def load_from_database_historical(self, from_date: str, to_date: str) -> pd.DataFrame:
        dm = DataManager.from_database_historical_2(self.database, from_date, to_date)
        if dm is None:
            raise ValueError("No se han cargado datos desde la base de datos.")
        return dm.df

    @staticmethod
    def load_from_database_real(database, last_n_rates: int = 256) -> pd.DataFrame:
        dm = DataManager.from_database_real_2(database, last_n_rates=last_n_rates)
        if dm is None:
            raise ValueError("No se han cargado datos desde la base de datos.")
        return dm.df

    @staticmethod
    def save_meta(meta:dict, path: str):
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open('w', encoding='utf-8') as f:
            json.dump(meta, f, ensure_ascii=False, indent=2, default=str)

    @staticmethod
    def save_holdout_predictions(report: dict, out_path: Path, side: str):
        """
        Guarda a parquet las predicciones walk-forward del holdout.
        Espera report["walkforward"]["predictions"] con:
          time, state, y_true, y_pred_raw, y_pred_cal
        """
        preds = (
            report.get("walkforward", {}).get("predictions", None)
            if isinstance(report, dict) else None
        )

        if not preds:
            print(f"⚠️  No hay predicciones holdout para {side} en {out_path.name}")
            return

        df_preds = pd.DataFrame(preds).copy()
        df_preds["side"] = side

        if "time" in df_preds.columns:
            df_preds["time"] = pd.to_datetime(df_preds["time"], errors="coerce")

        out_path.parent.mkdir(parents=True, exist_ok=True)
        df_preds.to_parquet(out_path, index=False)

        print(f"✅ Holdout predictions guardadas: {out_path} ({len(df_preds):,} filas)")

    import numpy as np

    @staticmethod
    def _to_numpy(x, dtype=np.float32):
        if isinstance(x, (pd.Series, pd.Index)):
            return x.to_numpy(dtype=dtype)
        return np.asarray(x, dtype=dtype)

    @staticmethod
    def build_nonzero_final_weights(
            base_weight,
            states,
            y=None,
            *,
            min_base_weight=0.10,
            min_final_weight=0.05,
            regime_weight_map=None,
            auto_from_pos_rate=False,
            auto_strength=0.35,
            auto_clip=(0.85, 1.20),
            renorm_to_base_sum=True,
            return_debug=False,
    ):
        """
        Garantiza:
          1) base_weight >= min_base_weight
          2) regime_weight > 0
          3) final_weight >= min_final_weight

        Parámetros
        ----------
        base_weight : array-like
            Peso base por muestra. Puede contener ceros.
        states : array-like
            Régimen por muestra (TREND_UP, LOW_VOL, etc.).
        y : array-like, opcional
            Labels binarias. Solo se usa si auto_from_pos_rate=True.
        min_base_weight : float
            Suelo mínimo para el peso base.
        min_final_weight : float
            Suelo mínimo para el peso final.
        regime_weight_map : dict[str, float], opcional
            Pesos por régimen. Ej:
            {
                "TREND_UP": 0.95,
                "TREND_DOWN": 0.90,
                "RANGE": 1.05,
                "TRANSITION_UP": 1.08,
                "TRANSITION_DOWN": 1.08,
                "BREAKOUT_WAIT_UP": 1.10,
                "BREAKOUT_WAIT_DOWN": 1.10,
                "VOLATILE": 0.98,
                "LOW_VOL": 1.18,
            }
        auto_from_pos_rate : bool
            Si True, calcula automáticamente los pesos de régimen a partir de la
            tasa positiva por estado.
        auto_strength : float
            Intensidad del ajuste automático.
        auto_clip : tuple(float, float)
            Recorte inferior/superior de los pesos automáticos.
        renorm_to_base_sum : bool
            Reescala final_weight para que su suma quede similar a la suma del
            base_weight original.
        return_debug : bool
            Si True, devuelve también un dict con resumen por estado.

        Returns
        -------
        final_weight : np.ndarray
        debug : dict (opcional)
        """
        base_weight = Helper._to_numpy(base_weight, dtype=np.float32)
        states = np.asarray(states)
        n = len(base_weight)

        if len(states) != n:
            raise ValueError("states y base_weight deben tener la misma longitud")

        if auto_from_pos_rate:
            if y is None:
                raise ValueError("Si auto_from_pos_rate=True, debes pasar y")
            y = Helper._to_numpy(y, dtype=np.float32)
            if len(y) != n:
                raise ValueError("y y base_weight deben tener la misma longitud")

            tmp = pd.DataFrame({"state": states, "y": y})
            stats = (
                tmp.groupby("state", dropna=False)["y"]
                .agg(["mean", "count"])
                .rename(columns={"mean": "pos_rate", "count": "n"})
            )

            global_pos_rate = float(np.mean(y))
            eps = 1e-8

            auto_map = {}
            for state, row in stats.iterrows():
                pos_rate = float(row["pos_rate"])
                ratio = (pos_rate + eps) / (global_pos_rate + eps)

                # Ajuste suave alrededor de 1.0
                w = 1.0 + auto_strength * (ratio - 1.0)
                w = float(np.clip(w, auto_clip[0], auto_clip[1]))

                auto_map[state] = w

            regime_weight_map = auto_map if regime_weight_map is None else {
                **auto_map,
                **regime_weight_map,
            }

        if regime_weight_map is None:
            regime_weight_map = {}

        # 1) suelo del peso base
        safe_base = np.maximum(base_weight, float(min_base_weight)).astype(np.float32)

        # 2) peso de régimen siempre positivo
        regime_weight = np.array(
            [max(float(regime_weight_map.get(s, 1.0)), 1e-6) for s in states],
            dtype=np.float32,
        )

        # 3) combinación + suelo final
        final_weight = safe_base * regime_weight
        final_weight = np.maximum(final_weight, float(min_final_weight)).astype(np.float32)

        # 4) renormalización opcional para no inflar demasiado la masa total
        if renorm_to_base_sum:
            original_sum = float(np.sum(base_weight))
            final_sum = float(np.sum(final_weight))
            if final_sum > 0 and original_sum > 0:
                scale = original_sum / final_sum
                final_weight = final_weight * scale
                final_weight = np.maximum(final_weight, float(min_final_weight)).astype(np.float32)

        if not return_debug:
            return final_weight

        dbg_df = pd.DataFrame({
            "state": states,
            "base_weight": base_weight,
            "safe_base": safe_base,
            "regime_weight": regime_weight,
            "final_weight": final_weight,
        })

        if y is not None:
            dbg_df["y"] = y

        agg = {
            "base_weight": "sum",
            "safe_base": "sum",
            "regime_weight": "mean",
            "final_weight": ["sum", "min", "max"],
        }
        if y is not None:
            agg["y"] = ["mean", "sum", "count"]

        summary = dbg_df.groupby("state", dropna=False).agg(agg)

        debug = {
            "original_sum": float(np.sum(base_weight)),
            "safe_base_sum": float(np.sum(safe_base)),
            "final_sum": float(np.sum(final_weight)),
            "final_min": float(np.min(final_weight)),
            "final_max": float(np.max(final_weight)),
            "zero_count_final": int(np.sum(final_weight <= 0.0)),
            "summary_by_state": summary,
        }
        return final_weight, debug