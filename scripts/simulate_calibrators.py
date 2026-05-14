#!/usr/bin/env python3
"""
simulate_calibrators.py — Compara offline distintas estrategias de calibración
sobre los mismos datos, SIN retrain del modelo. Adicionalmente, identifica el
calibrador "campeón" por cada combinación (side, state).

Métodos comparados (configurables vía --methods):
  - iso_21d / iso_45d / iso_full: isotonic con distintas ventanas de tail
  - beta_21d / beta_45d: Beta calibration (requiere `pip install betacal`)
  - per_state_iso_45d: un isotonic por estado, fallback global

Outputs:
  - Tabla agregada por método/side
  - Tabla de hit-rate por bucket (D1, D5, D8, D9, D10)
  - **Tabla campeón por (side, state)** con mejora ECE vs baseline iso_21d
  - JSON con el mapping (side, state) → champion_method

Uso:
  python scripts/simulate_calibrators.py \\
    --release 202500 \\
    --specialist-tag rw_both_Lvol_boost_td_down_h3_Svol_boost_h3 \\
    --seed 47 \\
    --test-parquet /tmp/ghost_apr.parquet \\
    --out-report reports/calibration_simulation.csv
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, roc_auc_score


# ───────────────────────────── Calibrators ─────────────────────────────

class GlobalCalibrator:
    """Wrapper para un único calibrador global (isotonic o beta)."""
    def __init__(self, kind: str = "isotonic"):
        self.kind = kind
        self.model = None

    def fit(self, p_raw: np.ndarray, y: np.ndarray) -> "GlobalCalibrator":
        if self.kind == "isotonic":
            self.model = IsotonicRegression(out_of_bounds="clip")
            self.model.fit(p_raw, y)
        elif self.kind == "beta":
            try:
                from betacal import BetaCalibration
            except ImportError:
                raise SystemExit("❌ `betacal` no instalado: `pip install betacal`")
            self.model = BetaCalibration(parameters="abm")
            self.model.fit(p_raw, y)
        else:
            raise ValueError(f"Unknown calibrator kind: {self.kind}")
        return self

    def predict(self, p_raw: np.ndarray) -> np.ndarray:
        return self.model.predict(p_raw)


class PerStateCalibrator:
    """Un calibrador por estado, con fallback global para estados raros."""
    def __init__(self, kind: str = "isotonic", min_n_per_state: int = 100):
        self.kind = kind
        self.min_n = min_n_per_state
        self.per_state: Dict[str, GlobalCalibrator] = {}
        self.fallback: Optional[GlobalCalibrator] = None

    def fit(self, p_raw: np.ndarray, y: np.ndarray, states: np.ndarray) -> "PerStateCalibrator":
        self.fallback = GlobalCalibrator(self.kind).fit(p_raw, y)
        for state in pd.Series(states).unique():
            mask = states == state
            if int(mask.sum()) < self.min_n:
                continue
            try:
                self.per_state[state] = GlobalCalibrator(self.kind).fit(p_raw[mask], y[mask])
            except Exception as e:
                print(f"  ⚠️  fit per-state {state} fallo: {e}")
        return self

    def predict(self, p_raw: np.ndarray, states: np.ndarray) -> np.ndarray:
        out = np.empty_like(p_raw, dtype=float)
        for state in np.unique(states):
            mask = states == state
            cal = self.per_state.get(state, self.fallback)
            out[mask] = cal.predict(p_raw[mask])
        return out


# ───────────────────────────── Métricas ─────────────────────────────

def compute_ece(p_cal: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    if len(p_cal) == 0:
        return float("nan")
    bins = np.linspace(0, max(p_cal.max(), 1e-6), n_bins + 1)
    indices = np.clip(np.digitize(p_cal, bins) - 1, 0, n_bins - 1)
    ece, n = 0.0, len(p_cal)
    for b in range(n_bins):
        mask = indices == b
        if mask.sum() == 0:
            continue
        ece += (mask.sum() / n) * abs(p_cal[mask].mean() - y[mask].mean())
    return ece


def compute_bucket_hit_rates(p_cal: np.ndarray, y: np.ndarray, n_buckets: int = 10) -> Dict[str, float]:
    if len(p_cal) == 0:
        return {}
    quantiles = np.quantile(p_cal, np.linspace(0, 1, n_buckets + 1))
    quantiles[-1] += 1e-9
    indices = np.clip(np.digitize(p_cal, quantiles) - 1, 0, n_buckets - 1)
    result = {}
    for b in range(n_buckets):
        mask = indices == b
        result[f"D{b+1}_hit"] = float(y[mask].mean()) if mask.any() else np.nan
        result[f"D{b+1}_n"] = int(mask.sum())
    return result


def evaluate_method(name: str, p_cal: np.ndarray, y: np.ndarray) -> Dict:
    res = {"method": name, "n_test": len(p_cal)}
    if len(p_cal) == 0:
        return res
    res.update({
        "score_p50": float(np.quantile(p_cal, 0.50)),
        "score_p90": float(np.quantile(p_cal, 0.90)),
        "score_p95": float(np.quantile(p_cal, 0.95)),
        "score_p99": float(np.quantile(p_cal, 0.99)),
        "score_max": float(p_cal.max()),
        "n_unique": int(len(np.unique(np.round(p_cal, 4)))),
        "pct_at_max": float((p_cal >= p_cal.max() - 1e-6).mean()) * 100,
        "mean_score": float(p_cal.mean()),
        "pos_rate": float(y.mean()),
        "ece": compute_ece(p_cal, y),
    })
    try:
        res["auc_roc"] = float(roc_auc_score(y, p_cal))
        res["auc_pr"] = float(average_precision_score(y, p_cal))
    except Exception:
        res["auc_roc"] = np.nan
        res["auc_pr"] = np.nan
    res.update(compute_bucket_hit_rates(p_cal, y))
    return res


# ───────────────────────────── Loaders ─────────────────────────────

def find_holdout_predictions(release: str, tag: str, side: str, seed: int) -> Path:
    bases = [
        Path(f"../artifacts/{release}/oof/{tag}_{side}_specialist_seed{seed}"),
        Path(f"../artifacts/{release}/oof/{tag}_{side}_specialist_seed{seed}_cutoff_mar31"),
    ]
    target = f"holdout_predictions_{release}_{side}.parquet"
    for base in bases:
        if not base.exists():
            continue
        for sub in (base, base / "data"):
            p = sub / target
            if p.exists():
                return p
    raise SystemExit(f"❌ No encuentro {target}")


def load_test(path: Path, side: str) -> pd.DataFrame:
    df = pd.read_parquet(path)
    df = df[df["side"] == side].copy()
    df["time"] = pd.to_datetime(df["time"])
    for c in ("oof_proba_raw", "signal", "state"):
        if c not in df.columns:
            raise SystemExit(f"❌ {path} sin columna '{c}'")
    if "period" in df.columns:
        df = df[df["period"] == "holdout"].copy()
    return df.reset_index(drop=True)


# ───────────────────────────── Main ─────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--release", default="202500")
    ap.add_argument("--specialist-tag", default="rw_both_Lvol_boost_td_down_h3_Svol_boost_h3")
    ap.add_argument("--seed", type=int, default=47)
    ap.add_argument("--test-parquet", required=True,
                    help="Parquet de ghost_predict sobre ventana de test (LOCKBOX).")
    ap.add_argument("--side", choices=["long", "short", "both"], default="both")
    ap.add_argument("--methods",
                    default="iso_21d,iso_45d,iso_full,beta_21d,beta_45d,per_state_iso_45d")
    ap.add_argument("--baseline-method", default="iso_21d",
                    help="Método contra el que se mide mejora del campeón.")
    ap.add_argument("--min-n-per-state", type=int, default=30,
                    help="Min filas por (side, state) en LOCKBOX para considerar el combo.")
    ap.add_argument("--out-report", default="reports/calibration_simulation.csv")
    args = ap.parse_args()

    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    sides = ["long", "short"] if args.side == "both" else [args.side]
    test_path = Path(args.test_parquet)

    all_rows = []
    # Predicciones acumuladas para análisis posterior por (side, state)
    preds_by_side_method: Dict[str, Dict[str, Dict[str, np.ndarray]]] = {}

    for side in sides:
        print(f"\n{'═'*72}\n  SIDE = {side.upper()}\n{'═'*72}")
        hp_path = find_holdout_predictions(args.release, args.specialist_tag, side, args.seed)
        print(f"  📂 Historical: {hp_path.name}")
        hp = pd.read_parquet(hp_path)
        hp["time"] = pd.to_datetime(hp["time"])
        raw_col = "y_pred_raw" if "y_pred_raw" in hp.columns else "oof_proba_raw"
        sig_col = "y_true" if "y_true" in hp.columns else "signal"
        hp_raw = hp[raw_col].to_numpy()
        hp_y = hp[sig_col].to_numpy()
        hp_state = hp["state"].astype(str).to_numpy() if "state" in hp.columns else None
        hp_time = pd.to_datetime(hp["time"]).to_numpy()
        hp_max = pd.to_datetime(hp_time).max()

        test_df = load_test(test_path, side)
        print(f"  📂 Test: n={len(test_df)} | {test_df['time'].min()} → {test_df['time'].max()}")
        test_raw = test_df["oof_proba_raw"].to_numpy()
        test_y = test_df["signal"].to_numpy()
        test_state = test_df["state"].astype(str).to_numpy()

        cutoff_21 = hp_max - pd.Timedelta(days=21)
        cutoff_45 = hp_max - pd.Timedelta(days=45)

        preds_by_side_method.setdefault(side, {})

        for method in methods:
            print(f"\n  ▶ {method}...")
            try:
                if method == "iso_21d":
                    mask = pd.Series(hp_time) >= cutoff_21
                    p_cal = GlobalCalibrator("isotonic").fit(hp_raw[mask], hp_y[mask]).predict(test_raw)
                elif method == "iso_45d":
                    mask = pd.Series(hp_time) >= cutoff_45
                    p_cal = GlobalCalibrator("isotonic").fit(hp_raw[mask], hp_y[mask]).predict(test_raw)
                elif method == "iso_full":
                    p_cal = GlobalCalibrator("isotonic").fit(hp_raw, hp_y).predict(test_raw)
                elif method == "beta_21d":
                    mask = pd.Series(hp_time) >= cutoff_21
                    p_cal = GlobalCalibrator("beta").fit(hp_raw[mask], hp_y[mask]).predict(test_raw)
                elif method == "beta_45d":
                    mask = pd.Series(hp_time) >= cutoff_45
                    p_cal = GlobalCalibrator("beta").fit(hp_raw[mask], hp_y[mask]).predict(test_raw)
                elif method == "per_state_iso_45d":
                    if hp_state is None:
                        print(f"     ⚠️  sin columna state en historical, salto")
                        continue
                    mask = pd.Series(hp_time) >= cutoff_45
                    cal = PerStateCalibrator("isotonic", min_n_per_state=100).fit(
                        hp_raw[mask], hp_y[mask], hp_state[mask]
                    )
                    p_cal = cal.predict(test_raw, test_state)
                else:
                    print(f"     ⚠️  método desconocido: {method}")
                    continue
            except SystemExit:
                raise
            except Exception as e:
                print(f"     ❌ {method} falló: {e}")
                continue

            row = evaluate_method(method, p_cal, test_y)
            row["side"] = side
            all_rows.append(row)
            preds_by_side_method[side][method] = {
                "p_cal": p_cal, "y": test_y, "state": test_state,
            }
            print(f"     ECE={row['ece']:.4f}  AUC-PR={row['auc_pr']:.4f}  "
                  f"p99={row['score_p99']:.4f}  n_unique={row['n_unique']}  "
                  f"D9_hit={row.get('D9_hit', np.nan):.3f}  D10_hit={row.get('D10_hit', np.nan):.3f}")

    df_out = pd.DataFrame(all_rows)
    print(f"\n\n{'═'*110}\n  TABLA COMPARATIVA — métricas principales por método/side (ordenada por ECE asc)\n{'═'*110}")
    summary_cols = ["side", "method", "n_test", "ece", "auc_pr", "auc_roc",
                    "score_p95", "score_p99", "score_max", "n_unique", "pct_at_max"]
    print(df_out[summary_cols].sort_values(["side", "ece"]).to_string(index=False, float_format="%.4f"))

    print(f"\n{'─'*110}\n  HIT-RATE EN BUCKETS CLAVE — D9 = penúltimo, D10 = top (cap)\n{'─'*110}")
    bucket_cols = ["side", "method", "D5_hit", "D8_hit", "D9_hit", "D10_hit"]
    print(df_out[bucket_cols].to_string(index=False, float_format="%.3f"))

    # ──────────────── Campeón por (side, state) ────────────────
    print(f"\n{'═'*110}\n  CAMPEÓN POR (side, state) — minimizando ECE, desempate por AUC-PR desc\n{'═'*110}")
    champion_rows = []
    champion_config: Dict[str, Dict[str, str]] = {}

    for side, methods_results in preds_by_side_method.items():
        if not methods_results:
            continue
        # Cualquier método tiene el mismo y/state (ordenamiento idéntico)
        first_method = next(iter(methods_results.values()))
        states_all = first_method["state"]
        ys_all = first_method["y"]
        unique_states = pd.Series(states_all).unique()

        for state in unique_states:
            mask = states_all == state
            n = int(mask.sum())
            if n < args.min_n_per_state:
                continue
            y_state = ys_all[mask]

            scores_by_method = {}
            for method, data in methods_results.items():
                p_cal_state = data["p_cal"][mask]
                ece = compute_ece(p_cal_state, y_state)
                try:
                    auc_pr = (float(average_precision_score(y_state, p_cal_state))
                              if y_state.sum() > 0 else float("nan"))
                except Exception:
                    auc_pr = float("nan")
                scores_by_method[method] = {
                    "ece": ece,
                    "auc_pr": auc_pr,
                    "n_unique": int(len(np.unique(np.round(p_cal_state, 4)))),
                    "score_p99": float(np.quantile(p_cal_state, 0.99)),
                }

            # Campeón: ECE más bajo, desempate por AUC-PR más alto
            def _key(item):
                _m, s = item
                auc = s["auc_pr"] if pd.notna(s["auc_pr"]) else -1.0
                return (s["ece"], -auc)

            champion_method, champion_metrics = min(scores_by_method.items(), key=_key)
            baseline_metrics = scores_by_method.get(args.baseline_method, {})

            champion_rows.append({
                "side": side,
                "state": state,
                "n": n,
                "pos_rate": float(y_state.mean()),
                "champion": champion_method,
                "champ_ece": champion_metrics["ece"],
                "champ_auc_pr": champion_metrics["auc_pr"],
                "champ_p99": champion_metrics["score_p99"],
                "baseline": args.baseline_method,
                "base_ece": baseline_metrics.get("ece", float("nan")),
                "base_auc_pr": baseline_metrics.get("auc_pr", float("nan")),
                "Δ_ece": (baseline_metrics.get("ece", float("nan")) - champion_metrics["ece"])
                         if pd.notna(baseline_metrics.get("ece", float("nan"))) else float("nan"),
            })
            champion_config.setdefault(side, {})[state] = champion_method

    if champion_rows:
        ch_df = pd.DataFrame(champion_rows)
        ch_df_sorted = ch_df.sort_values(["side", "Δ_ece"], ascending=[True, False])
        cols_show = ["side", "state", "n", "pos_rate", "champion", "champ_ece",
                     "champ_auc_pr", "champ_p99", "base_ece", "base_auc_pr", "Δ_ece"]
        print(ch_df_sorted[cols_show].to_string(index=False, float_format="%.4f"))

        # Resumen: cuántas veces gana cada método
        print(f"\n  📊 Cuántas combinaciones (side, state) gana cada método:")
        winner_counts = ch_df["champion"].value_counts()
        for method, count in winner_counts.items():
            pct = 100 * count / len(ch_df)
            print(f"     {method:25s} → {count}/{len(ch_df)} combos ({pct:.0f}%)")

        # Diagnóstico de homogeneidad
        n_combos = len(ch_df)
        n_iso_21d = int((ch_df["champion"] == args.baseline_method).sum())
        n_unique_winners = ch_df["champion"].nunique()
        print(f"\n  🎯 DIAGNÓSTICO:")
        if n_unique_winners == 1 and ch_df["champion"].iloc[0] == args.baseline_method:
            print(f"     🟡 Baseline {args.baseline_method} gana en TODOS los combos. NO cambiar.")
        elif n_unique_winners == 1:
            winner = ch_df["champion"].iloc[0]
            print(f"     🟢 Un solo método ({winner}) gana en TODOS los combos. "
                  f"Adoptar Nivel 1 (heterogéneo por side) basta — ambos sides usarían {winner}.")
        elif n_unique_winners == 2 and len(ch_df["side"].unique()) > 1:
            sides_winners = ch_df.groupby("side")["champion"].nunique()
            if (sides_winners == 1).all():
                print(f"     🟢 Cada side prefiere UN solo método. **Nivel 1 (heterogéneo por side) es suficiente**.")
                for side in ch_df["side"].unique():
                    method = ch_df[ch_df["side"] == side]["champion"].iloc[0]
                    print(f"        {side} → {method}")
            else:
                print(f"     🟠 Hay heterogeneidad dentro de algún side. Considerar Nivel 2 (per-state).")
        else:
            print(f"     🔴 Heterogeneidad alta ({n_unique_winners} métodos campeones). "
                  f"Nivel 2 (per-state) recomendado pero ojo al overfitting.")
    else:
        print("  (sin filas: revisa --min-n-per-state o métricas)")

    # ──────────────── Persistencia ────────────────
    out = Path(args.out_report)
    out.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out, index=False)
    print(f"\n📁 Reporte por método: {out}")

    if champion_rows:
        ch_out = out.parent / "champion_per_combo.csv"
        pd.DataFrame(champion_rows).to_csv(ch_out, index=False)
        print(f"📁 Reporte campeón:   {ch_out}")

        ch_json = out.parent / "champion_config.json"
        with ch_json.open("w") as f:
            json.dump(champion_config, f, indent=2)
        print(f"📁 Config campeón:    {ch_json}")
        print(f"     ↳ Mapping (side, state) → método para consumir en producción")


if __name__ == "__main__":
    main()