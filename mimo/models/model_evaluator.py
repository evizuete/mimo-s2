from typing import Dict, Optional

import numpy as np
from sklearn.metrics import precision_recall_curve, roc_auc_score, average_precision_score, confusion_matrix


class ModelEvaluator:
    """
    Evalúa el modelo con métricas de ML y trading - VERSIÓN OPTIMIZADA

    OPTIMIZACIONES:
    1. Cache de resultados intermedios
    2. Vectorización de cálculos
    3. Reducción de conversiones de tipos
    """

    def __init__(self):
        # ═══ OPTIMIZACIÓN 1: Cache para métricas ═══
        self._metrics_cache = {}

    @staticmethod
    def _fbeta(precision: np.ndarray, recall: np.ndarray, beta: float) -> np.ndarray:
        """Cálculo vectorizado de F-beta"""
        b2 = beta * beta
        return (1 + b2) * precision * recall / (b2 * precision + recall + 1e-12)

    @staticmethod
    def _metrics_at_threshold_fast(y_true: np.ndarray, y_proba: np.ndarray, thr: float) -> Dict:
        """
        Versión optimizada de _metrics_at_threshold

        OPTIMIZACIONES:
        - Una sola llamada a confusion_matrix
        - Conversiones de tipos minimizadas
        - Cálculos vectorizados
        """
        y_pred = (y_proba >= thr)  # bool, no int (más rápido)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

        # ═══ Pre-calcular denominadores comunes ═══
        tp_fp = tp + fp
        tp_fn = tp + fn
        total = tp + tn + fp + fn

        return {
            'threshold': float(thr),
            'precision': float(tp / (tp_fp + 1e-12)),
            'recall': float(tp / (tp_fn + 1e-12)),
            'f1': float(2 * tp / (2 * tp + fp + fn + 1e-12)),
            'accuracy': float((tp + tn) / (total + 1e-12)),
            'tp': int(tp), 'fp': int(fp), 'tn': int(tn), 'fn': int(fn),
            'signal_rate': float(y_pred.sum() / len(y_pred)),
            'base_rate': float(y_true.mean()),
        }

    def evaluate_predictions(self,
                             y_true: np.ndarray,
                             y_pred_proba: np.ndarray,
                             verbose: bool = True,
                             beta_primary: float = 0.25,
                             min_precision: Optional[float] = 0.45,
                             max_signal_rate: Optional[float] = 0.15) -> Dict:
        """
        Evalúa predicciones - VERSIÓN OPTIMIZADA

        OPTIMIZACIONES:
        1. Conversión de tipos una sola vez
        2. Cálculos vectorizados
        3. Búsqueda eficiente de thresholds
        """

        # ═══ OPTIMIZACIÓN 1: Conversión de tipos una vez ═══
        y_true = y_true.astype(np.int8, copy=False)  # int8 suficiente para 0/1
        y_pred_proba = y_pred_proba.astype(np.float32, copy=False)

        # ═══ OPTIMIZACIÓN 2: Precision-recall curve (una vez) ═══
        prec, rec, thr = precision_recall_curve(y_true, y_pred_proba)

        # Ajustar longitudes
        prec_t = prec[:-1]
        rec_t = rec[:-1]
        thr_t = thr

        # ═══ OPTIMIZACIÓN 3: Calcular todos los F-beta de una vez ═══
        f_primary = self._fbeta(prec_t, rec_t, beta_primary)
        f05 = self._fbeta(prec_t, rec_t, 0.5)
        f1 = self._fbeta(prec_t, rec_t, 1.0)

        # ═══ OPTIMIZACIÓN 4: Encontrar óptimos (argmax vectorizado) ═══
        idx_primary = int(np.nanargmax(f_primary))
        idx_f05 = int(np.nanargmax(f05))
        idx_f1 = int(np.nanargmax(f1))

        thr_primary = float(thr_t[idx_primary])
        thr_f05 = float(thr_t[idx_f05])
        thr_f1 = float(thr_t[idx_f1])

        # ═══ OPTIMIZACIÓN 5: Threshold con precision mínima ═══
        thr_minprec = None
        if min_precision is not None:
            mask = prec_t >= min_precision
            if mask.any():
                # Encontrar el de mayor recall
                valid_indices = np.where(mask)[0]
                best_idx = valid_indices[np.argmax(rec_t[valid_indices])]
                thr_minprec = float(thr_t[best_idx])

        # ═══ OPTIMIZACIÓN 6: Threshold con signal_rate máximo ═══
        # Búsqueda más eficiente usando percentiles pre-calculados
        thr_maxsr = None
        if max_signal_rate is not None:
            # Generar candidatos (menos puntos, más eficiente)
            cand_percentiles = np.linspace(50, 99.9, 200)
            candidates = np.unique(np.percentile(y_pred_proba, cand_percentiles))

            for t in candidates:
                signal_rate = (y_pred_proba >= t).mean()
                if signal_rate <= max_signal_rate:
                    thr_maxsr = float(t)
                    break

        # ═══ OPTIMIZACIÓN 7: Métricas globales (una vez) ═══
        metrics = {
            "auc_roc": float(roc_auc_score(y_true, y_pred_proba)),
            "auc_pr": float(average_precision_score(y_true, y_pred_proba)),
            "base_rate": float(y_true.mean()),
            "thresholds": {
                f"best_f{beta_primary}": thr_primary,
                "best_f0.5": thr_f05,
                "best_f1": thr_f1,
                "min_precision": thr_minprec,
                "max_signal_rate": thr_maxsr,
            },
            "selected_threshold": thr_primary,
            "selected_objective": f"F{beta_primary}",
        }

        # ═══ OPTIMIZACIÓN 8: Calcular métricas para threshold seleccionado ═══
        selected = self._metrics_at_threshold_fast(y_true, y_pred_proba, metrics["selected_threshold"])
        metrics.update({f"selected_{k}": v for k, v in selected.items()})

        # ═══ OPTIMIZACIÓN 9: Alternativas (solo para thresholds válidos) ═══
        alternatives = {}
        for name, t in metrics["thresholds"].items():
            if t is not None:
                alternatives[name] = self._metrics_at_threshold_fast(y_true, y_pred_proba, float(t))

        metrics["alternatives"] = alternatives

        # ═══ Verbose output (sin cambios) ═══
        if verbose:
            print("=" * 50)
            print("MÉTRICAS GLOBALES")
            print("=" * 50)
            print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
            print(f"AUC-PR:  {metrics['auc_pr']:.4f}")
            print(f"Base rate: {metrics['base_rate']:.4f}")

            print("\n" + "=" * 50)
            print("UMBRALES CANDIDATOS")
            print("=" * 50)
            for k, v in alternatives.items():
                print(
                    f"{k:>14} | thr={v['threshold']:.4f} | prec={v['precision']:.3f} "
                    f"| rec={v['recall']:.3f} | f1={v['f1']:.3f} "
                    f"| sig={v['signal_rate']:.3f} | TP={v['tp']} FP={v['fp']}"
                )

            print("\n" + "=" * 50)
            print(f"SELECCIONADO ({metrics['selected_objective']})")
            print("=" * 50)
            print(
                f"thr={selected['threshold']:.4f} | precision={selected['precision']:.4f} "
                f"| recall={selected['recall']:.4f} | f1={selected['f1']:.4f} "
                f"| signal_rate={selected['signal_rate']:.4f}"
            )
            print(f"TP: {selected['tp']}, FP: {selected['fp']}, FN: {selected['fn']}, TN: {selected['tn']}")

        return metrics