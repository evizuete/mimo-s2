"""
analyze_distribution_shift.py
==============================
Diagnóstico de shift distribucional entre periodo de entrenamiento y holdout.

Métricas calculadas:
  - Estadísticas básicas (media, std, percentiles) por periodo
  - PSI (Population Stability Index) por feature
  - KS-test (Kolmogorov-Smirnov) por feature
  - Distribución de regímenes (state) por periodo
  - Correlaciones entre features: cambio entre periodos
  - Distribución de labels (signal) por periodo y por régimen
  - Rolling stats para ver drift temporal dentro del train

Uso:
    python analyze_distribution_shift.py
    python analyze_distribution_shift.py --output_dir ./shift_report
    python analyze_distribution_shift.py --top_features 20 --no_plots
"""

import argparse
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN — ajusta estas fechas a tu entorno
# ─────────────────────────────────────────────────────────────────────────────

TRAIN_FROM   = datetime(2025,  10,  1)
TRAIN_TO     = datetime(2026,  1, 31)
HOLDOUT_FROM = datetime(2026, 2, 1)
HOLDOUT_TO   = datetime(2026,  4, 10)

# Thresholds de alerta PSI
PSI_WARNING  = 0.10   # cambio moderado
PSI_CRITICAL = 0.25   # cambio severo
KS_ALPHA     = 0.05   # p-value KS test


# ─────────────────────────────────────────────────────────────────────────────
# CARGA DE DATOS
# ─────────────────────────────────────────────────────────────────────────────

def load_data(from_date: datetime, to_date: datetime) -> pd.DataFrame:
    """Carga datos OHLCV desde tu DataManager. Adapta si usas otro origen."""
    try:
        from mimo.data_managers.data_manager import DataManager
        from mimo.data_managers.databases import Database
        db = Database()
        dm = DataManager.from_database_historical_2(db, from_date=from_date, to_date=to_date)
        return dm.df
    except Exception as e:
        print(f"[ERROR] No se pudo cargar datos via DataManager: {e}")
        sys.exit(1)


def build_features(df: pd.DataFrame, general_config, feature_config,
                   model_config, regime_config) -> pd.DataFrame:
    """Genera features usando el pipeline estándar."""
    from mimo.data_managers.data_pipeline_v2 import DataPipeline
    pipeline = DataPipeline(general_config, feature_config, model_config, regime_config)
    return pipeline.prepare_data(
        df.copy(), labels=True, side="long",
        set_market_condition=False, ensure_regime=True
    )


# ─────────────────────────────────────────────────────────────────────────────
# PSI
# ─────────────────────────────────────────────────────────────────────────────

def compute_psi(expected: np.ndarray, actual: np.ndarray, n_bins: int = 10) -> float:
    """
    Population Stability Index.
      PSI < 0.10  → estable
      PSI 0.10-0.25 → cambio moderado
      PSI > 0.25  → cambio severo
    """
    exp_f = expected[np.isfinite(expected)]
    act_f = actual[np.isfinite(actual)]
    if len(exp_f) < 10 or len(act_f) < 10:
        return np.nan

    quantiles  = np.linspace(0, 100, n_bins + 1)
    bin_edges  = np.unique(np.nanpercentile(exp_f, quantiles))
    if len(bin_edges) < 2:
        return np.nan

    eps = 1e-6
    exp_counts, _ = np.histogram(exp_f, bins=bin_edges)
    act_counts, _ = np.histogram(act_f, bins=bin_edges)
    exp_pct = (exp_counts / (exp_counts.sum() + eps)) + eps
    act_pct = (act_counts / (act_counts.sum() + eps)) + eps

    return float(np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct)))


# ─────────────────────────────────────────────────────────────────────────────
# KS TEST
# ─────────────────────────────────────────────────────────────────────────────

def compute_ks(train_vals: np.ndarray, hold_vals: np.ndarray):
    """KS test de dos muestras. Retorna (statistic, p_value)."""
    t = train_vals[np.isfinite(train_vals)]
    h = hold_vals[np.isfinite(hold_vals)]
    if len(t) < 5 or len(h) < 5:
        return np.nan, np.nan
    ks_stat, p_val = stats.ks_2samp(t, h)
    return float(ks_stat), float(p_val)


# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS DE FEATURES
# ─────────────────────────────────────────────────────────────────────────────

def analyze_features(df_train: pd.DataFrame, df_hold: pd.DataFrame,
                     feature_cols: list) -> pd.DataFrame:
    """Calcula PSI, KS y estadísticas básicas para cada feature."""
    rows = []
    for col in feature_cols:
        if col not in df_train.columns or col not in df_hold.columns:
            continue

        t = df_train[col].values.astype(float)
        h = df_hold[col].values.astype(float)

        t_mean, t_std = np.nanmean(t), np.nanstd(t)
        h_mean, h_std = np.nanmean(h), np.nanstd(h)
        t_p25, t_p50, t_p75 = np.nanpercentile(t, [25, 50, 75])
        h_p25, h_p50, h_p75 = np.nanpercentile(h, [25, 50, 75])

        psi        = compute_psi(t, h)
        ks_s, ks_p = compute_ks(t, h)

        rows.append({
            "feature":        col,
            "psi":            psi,
            "ks_stat":        ks_s,
            "ks_pvalue":      ks_p,
            "ks_significant": (ks_p < KS_ALPHA) if np.isfinite(ks_p) else None,
            "train_mean":     t_mean,
            "hold_mean":      h_mean,
            "mean_shift_pct": (h_mean - t_mean) / (abs(t_mean) + 1e-9) * 100,
            "train_std":      t_std,
            "hold_std":       h_std,
            "std_ratio":      h_std / (t_std + 1e-9),
            "train_p50":      t_p50,
            "hold_p50":       h_p50,
            "train_p25":      t_p25,
            "hold_p25":       h_p25,
            "train_p75":      t_p75,
            "hold_p75":       h_p75,
            "train_nan_pct":  np.isnan(t).mean() * 100,
            "hold_nan_pct":   np.isnan(h).mean() * 100,
        })

    df_res = pd.DataFrame(rows)
    if not df_res.empty:
        df_res = df_res.sort_values("psi", ascending=False).reset_index(drop=True)
    return df_res


def get_model_feature_groups(feature_config, side: str = "long") -> dict:
    """
    Devuelve las features reales que pasan al modelo, agrupadas por bloque
    (sequence_short, sequence_long, context, time), usando el mismo
    FeatureEngineer del pipeline.
    """
    try:
        from mimo.features.feature_builder import FeatureEngineer
        fe = FeatureEngineer(feature_config)
        fe.set_side(side)
        groups = {}
        for group, cols in fe.feature_columns.items():
            groups[group] = list(dict.fromkeys(cols))
        return groups
    except Exception as e:
        print(f"  ⚠️  No se pudo resolver el set real de features del modelo: {e}")
        return {}


def flatten_feature_groups(feature_groups: dict) -> list:
    seen = set()
    ordered = []
    for _, cols in feature_groups.items():
        for c in cols:
            if c not in seen:
                seen.add(c)
                ordered.append(c)
    return ordered


def annotate_feature_groups(feature_results: pd.DataFrame, feature_groups: dict) -> pd.DataFrame:
    if feature_results.empty:
        return feature_results.copy()

    feature_to_group = {}
    for group, cols in feature_groups.items():
        for c in cols:
            feature_to_group.setdefault(c, group)

    out = feature_results.copy()
    out["feature_group"] = out["feature"].map(feature_to_group).fillna("other")
    return out


def summarize_alerts_by_group(feature_results: pd.DataFrame) -> pd.DataFrame:
    if feature_results.empty or "feature_group" not in feature_results.columns:
        return pd.DataFrame()

    rows = []
    for group, df_g in feature_results.groupby("feature_group", dropna=False):
        rows.append({
            "feature_group": group,
            "n_features": len(df_g),
            "critical": int((df_g["psi"] >= PSI_CRITICAL).sum()),
            "warning": int(((df_g["psi"] >= PSI_WARNING) & (df_g["psi"] < PSI_CRITICAL)).sum()),
            "ok": int((df_g["psi"] < PSI_WARNING).sum()),
            "ks_significant": int(df_g["ks_significant"].sum()) if "ks_significant" in df_g else 0,
            "mean_psi": float(df_g["psi"].mean()),
            "median_psi": float(df_g["psi"].median()),
            "max_psi": float(df_g["psi"].max()),
        })
    return pd.DataFrame(rows).sort_values(["max_psi", "mean_psi"], ascending=False).reset_index(drop=True)


def print_group_alert_summary(group_summary: pd.DataFrame):
    if group_summary.empty:
        return

    print("\n  RESUMEN POR BLOQUE DEL MODELO:")
    print(f"    {'Bloque':<18} {'n':>4} {'Crit':>6} {'Warn':>6} {'OK':>6} {'KS':>6} {'PSI_med':>10} {'PSI_max':>10}")
    print(f"    {'-' * 72}")
    for _, row in group_summary.iterrows():
        print(
            f"    {row['feature_group']:<18} "
            f"{int(row['n_features']):>4} "
            f"{int(row['critical']):>6} "
            f"{int(row['warning']):>6} "
            f"{int(row['ok']):>6} "
            f"{int(row['ks_significant']):>6} "
            f"{row['median_psi']:>10.3f} "
            f"{row['max_psi']:>10.3f}"
        )


def print_top_features_by_group(feature_results: pd.DataFrame, top_n_per_group: int = 5):
    if feature_results.empty or "feature_group" not in feature_results.columns:
        return

    print("\n  TOP FEATURES POR BLOQUE (ordenadas por PSI):")
    for group in ["sequence_short", "sequence_long", "context", "time"]:
        df_g = feature_results[feature_results["feature_group"] == group].sort_values("psi", ascending=False)
        if df_g.empty:
            continue
        print(f"\n    [{group}]")
        for _, row in df_g.head(top_n_per_group).iterrows():
            alert = ("🔴 CRIT" if row["psi"] >= PSI_CRITICAL else
                     "🟠 WARN" if row["psi"] >= PSI_WARNING else "🟢 OK")
            print(
                f"      • {row['feature']:<28} PSI={row['psi']:.3f}  "
                f"{alert:<7}  Δμ={row['mean_shift_pct']:+.1f}%  σ_ratio={row['std_ratio']:.2f}"
            )


# ─────────────────────────────────────────────────────────────────────────────
# ANÁLISIS DE RÉGIMEN Y LABELS
# ─────────────────────────────────────────────────────────────────────────────

def analyze_regime_distribution(df_train: pd.DataFrame,
                                df_hold: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for period, df in [("train", df_train), ("holdout", df_hold)]:
        if "state" not in df.columns:
            continue
        counts = df["state"].value_counts(normalize=True) * 100
        for state, pct in counts.items():
            rows.append({"period": period, "state": state, "pct": pct,
                         "n": int(len(df) * pct / 100)})
    return pd.DataFrame(rows)


def analyze_label_distribution(df_train: pd.DataFrame,
                               df_hold: pd.DataFrame) -> dict:
    result = {}
    for period, df in [("train", df_train), ("holdout", df_hold)]:
        if "signal" not in df.columns:
            continue
        sig = df["signal"].dropna()
        result[period] = {
            "pos_rate": float(sig.mean()),
            "n_pos":    int(sig.sum()),
            "n_total":  int(len(sig)),
        }
        if "state" in df.columns:
            by_state = df.groupby("state")["signal"].agg(["mean", "count"])
            by_state.columns = ["pos_rate", "n"]
            result[f"{period}_by_state"] = by_state.to_dict(orient="index")
    return result


# ─────────────────────────────────────────────────────────────────────────────
# DRIFT TEMPORAL
# ─────────────────────────────────────────────────────────────────────────────

def analyze_rolling_drift(df: pd.DataFrame, feature_cols: list,
                          freq: str = "W") -> pd.DataFrame:
    """Media semanal de cada feature para visualizar drift intra-periodo."""
    if "time" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    df["time"] = pd.to_datetime(df["time"])
    df = df.set_index("time").sort_index()
    cols = [c for c in feature_cols if c in df.columns]
    return df[cols].resample(freq).mean()


# ─────────────────────────────────────────────────────────────────────────────
# CORRELACIONES
# ─────────────────────────────────────────────────────────────────────────────

def analyze_correlation_shift(df_train: pd.DataFrame, df_hold: pd.DataFrame,
                              feature_cols: list, top_n: int = 10) -> pd.DataFrame:
    cols = [c for c in feature_cols
            if c in df_train.columns and c in df_hold.columns][:50]
    if len(cols) < 2:
        return pd.DataFrame()

    corr_t = df_train[cols].corr()
    corr_h = df_hold[cols].corr()
    diff   = (corr_h - corr_t).abs()
    mask   = np.triu(np.ones(diff.shape, dtype=bool), k=1)
    pairs  = diff.where(mask).stack().sort_values(ascending=False).head(top_n)

    rows = []
    for (f1, f2), delta in pairs.items():
        rows.append({
            "feature_1":    f1,
            "feature_2":    f2,
            "corr_train":   float(corr_t.loc[f1, f2]),
            "corr_holdout": float(corr_h.loc[f1, f2]),
            "delta_corr":   float(delta),
        })
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# PLOTS
# ─────────────────────────────────────────────────────────────────────────────

def _get_plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        return plt
    except ImportError:
        print("[PLOT] matplotlib no disponible. Instala con: pip install matplotlib")
        return None


def plot_psi_summary(feature_results: pd.DataFrame, top_n: int, output_dir: str):
    plt = _get_plt()
    if plt is None:
        return

    df = feature_results.head(top_n).copy()
    colors = ["#e53935" if p >= PSI_CRITICAL else
              "#fb8c00" if p >= PSI_WARNING  else
              "#43a047" for p in df["psi"]]

    fig, ax = plt.subplots(figsize=(10, max(4, top_n * 0.38)))
    ax.barh(df["feature"][::-1], df["psi"][::-1],
            color=colors[::-1], edgecolor="white", height=0.7)
    ax.axvline(PSI_WARNING,  color="#fb8c00", linestyle="--", linewidth=1.2,
               label=f"Warning  ({PSI_WARNING})")
    ax.axvline(PSI_CRITICAL, color="#e53935", linestyle="--", linewidth=1.2,
               label=f"Critical ({PSI_CRITICAL})")
    ax.set_xlabel("PSI")
    ax.set_title(f"Population Stability Index — Top {top_n} features\n"
                 f"Train {TRAIN_FROM.date()}→{TRAIN_TO.date()}  vs  "
                 f"Holdout {HOLDOUT_FROM.date()}→{HOLDOUT_TO.date()}", fontsize=10)
    ax.legend(fontsize=9)
    plt.tight_layout()

    path = os.path.join(output_dir, "psi_summary.png")
    plt.savefig(path, dpi=130, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] PSI summary → {path}")


def plot_top_psi_features(df_train: pd.DataFrame, df_hold: pd.DataFrame,
                           feature_results: pd.DataFrame, top_n: int,
                           output_dir: str):
    plt = _get_plt()
    if plt is None:
        return

    features = feature_results.head(top_n)["feature"].tolist()
    n_cols = 3
    n_rows = (len(features) + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3.5))
    axes = axes.flatten()

    for idx, feat in enumerate(features):
        ax = axes[idx]
        t_vals = df_train[feat].dropna().values
        h_vals = df_hold[feat].dropna().values

        combined = np.concatenate([t_vals, h_vals])
        lo = np.nanpercentile(combined, 1)
        hi = np.nanpercentile(combined, 99)
        bins = np.linspace(lo, hi, 40)

        ax.hist(t_vals, bins=bins, alpha=0.55, density=True,
                label="Train", color="#2196F3")
        ax.hist(h_vals, bins=bins, alpha=0.55, density=True,
                label="Holdout", color="#F44336")

        row  = feature_results[feature_results["feature"] == feat].iloc[0]
        psi  = row["psi"]
        col  = ("red" if psi >= PSI_CRITICAL else
                "darkorange" if psi >= PSI_WARNING else "black")
        ax.set_title(f"{feat}\nPSI={psi:.3f}  Δμ={row['mean_shift_pct']:+.1f}%",
                     color=col, fontsize=8)
        ax.legend(fontsize=6)
        ax.tick_params(labelsize=7)

    for idx in range(len(features), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle(f"Distribuciones Train vs Holdout — Top {top_n} por PSI",
                 fontsize=12, y=1.01)
    plt.tight_layout()

    path = os.path.join(output_dir, "top_psi_histograms.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Histogramas PSI → {path}")


def plot_regime_distribution(regime_df: pd.DataFrame, output_dir: str):
    plt = _get_plt()
    if plt is None or regime_df.empty:
        return

    pivot = regime_df.pivot(index="state", columns="period", values="pct").fillna(0)
    ax = pivot.plot(kind="bar", figsize=(8, 4), color=["#2196F3", "#F44336"],
                    edgecolor="white", alpha=0.85)
    ax.set_title("Distribución de Régimen (state) — Train vs Holdout", fontsize=11)
    ax.set_xlabel("Régimen")
    ax.set_ylabel("% muestras")
    ax.legend(title="Periodo")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()

    path = os.path.join(output_dir, "regime_distribution.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Régimen → {path}")


def plot_rolling_drift(weekly_df: pd.DataFrame, top_features: list,
                       output_dir: str):
    plt = _get_plt()
    if plt is None or weekly_df.empty:
        return

    features = [f for f in top_features if f in weekly_df.columns][:9]
    if not features:
        return

    n_cols = 3
    n_rows = (len(features) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, n_rows * 3))
    axes = axes.flatten()
    holdout_ts = pd.Timestamp(HOLDOUT_FROM)

    for idx, feat in enumerate(features):
        ax = axes[idx]
        series = weekly_df[feat].dropna()
        train_s = series[series.index <  holdout_ts]
        hold_s  = series[series.index >= holdout_ts]

        ax.plot(train_s.index, train_s.values, color="#2196F3", lw=1.5, label="Train")
        ax.plot(hold_s.index,  hold_s.values,  color="#F44336", lw=1.5, label="Holdout")
        ax.axvline(holdout_ts, color="gray", ls="--", lw=1)
        if len(train_s) > 0:
            ax.axhline(train_s.mean(), color="#2196F3", ls=":", lw=1, alpha=0.6)

        ax.set_title(feat, fontsize=8)
        ax.tick_params(labelsize=6)
        ax.legend(fontsize=6)

    for idx in range(len(features), len(axes)):
        axes[idx].set_visible(False)

    fig.suptitle("Drift temporal — media semanal por feature", fontsize=11, y=1.01)
    plt.tight_layout()

    path = os.path.join(output_dir, "rolling_drift.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Drift temporal → {path}")


def plot_label_rate_over_time(df_train: pd.DataFrame, df_hold: pd.DataFrame,
                              output_dir: str):
    """Tasa de positivos (signal) semana a semana."""
    plt = _get_plt()
    if plt is None:
        return

    dfs = []
    for period, df in [("train", df_train), ("holdout", df_hold)]:
        if "time" not in df.columns or "signal" not in df.columns:
            continue
        tmp = df[["time", "signal"]].copy()
        tmp["time"] = pd.to_datetime(tmp["time"])
        tmp = tmp.set_index("time").resample("W")["signal"].mean()
        dfs.append((period, tmp))

    if not dfs:
        return

    fig, ax = plt.subplots(figsize=(12, 4))
    colors = {"train": "#2196F3", "holdout": "#F44336"}
    for period, series in dfs:
        ax.plot(series.index, series.values, color=colors[period],
                lw=1.5, label=period)

    ax.axvline(pd.Timestamp(HOLDOUT_FROM), color="gray", ls="--", lw=1)
    ax.set_ylabel("Tasa positivos (signal=1)")
    ax.set_title("Evolución temporal de la tasa de positivos — Train vs Holdout")
    ax.legend()
    plt.tight_layout()

    path = os.path.join(output_dir, "label_rate_over_time.png")
    plt.savefig(path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"[PLOT] Tasa de positivos → {path}")


# ─────────────────────────────────────────────────────────────────────────────
# HELPERS DE REPORTE
# ─────────────────────────────────────────────────────────────────────────────

def print_section(title: str):
    print(f"\n{'═' * 70}")
    print(f"  {title}")
    print('═' * 70)


def print_feature_table(df: pd.DataFrame, top_n: int):
    hdr = (f"{'Feature':<35} {'PSI':>7} {'Alerta':>10} {'KS-p':>8} "
           f"{'μ_train':>9} {'μ_hold':>9} {'Δμ%':>8} {'σ_ratio':>8}")
    print(f"\n{hdr}")
    print("-" * len(hdr))

    for _, row in df.head(top_n).iterrows():
        psi   = row["psi"]
        ks_p  = row["ks_pvalue"]
        alert = ("🔴 CRIT" if psi >= PSI_CRITICAL else
                 "🟠 WARN" if psi >= PSI_WARNING  else "🟢 OK")
        print(
            f"  {row['feature']:<33} "
            f"{psi:>7.3f} "
            f"{alert:>10} "
            f"{ks_p:>8.3f} "
            f"{row['train_mean']:>9.4f} "
            f"{row['hold_mean']:>9.4f} "
            f"{row['mean_shift_pct']:>7.1f}% "
            f"{row['std_ratio']:>8.2f}"
        )


def summarize_alerts(df: pd.DataFrame) -> dict:
    if df.empty:
        return {"critical": 0, "warning": 0, "ok": 0, "ks_significant": 0, "total": 0}
    return {
        "critical":       int((df["psi"] >= PSI_CRITICAL).sum()),
        "warning":        int(((df["psi"] >= PSI_WARNING) & (df["psi"] < PSI_CRITICAL)).sum()),
        "ok":             int((df["psi"] <  PSI_WARNING).sum()),
        "ks_significant": int(df["ks_significant"].sum()) if "ks_significant" in df else 0,
        "total":          len(df),
    }


def build_diagnosis(alerts: dict, label_stats: dict, regime_df: pd.DataFrame,
                    scope_label: str = "features del modelo") -> tuple:
    """Retorna (veredicto, recomendación, accion_prioritaria)."""
    n_crit = alerts.get("critical", 0)
    n_warn = alerts.get("warning",  0)

    delta_pr = 0.0
    if "train" in label_stats and "holdout" in label_stats:
        delta_pr = (label_stats["holdout"]["pos_rate"]
                    - label_stats["train"]["pos_rate"])

    if n_crit >= 5 or abs(delta_pr) > 0.08:
        verdict = "🔴 SHIFT SEVERO"
        rec = (
            f"Las {scope_label} muestran un cambio distribucional material entre train y holdout. "
            "La degradación observada en walk-forward es compatible con este shift."
        )
        action = (
            "Prioriza revisar las features de volatilidad/contexto con mayor PSI, "
            "comparar su drift por bloque (short/long/context/time) y reentrenar "
            "solo después de validar que el set de inputs sea estable."
        )
    elif n_crit >= 2 or n_warn >= 8:
        verdict = "🟠 SHIFT MODERADO"
        rec = (
            f"Hay cambios relevantes en varias {scope_label}. "
            "El shift puede explicar parte de la degradación, aunque no parece un colapso de régimen."
        )
        action = (
            "Revisa especialmente las normalizaciones rolling sensibles a volatilidad "
            "y compara variantes más invariantes al régimen antes de tocar el scaler global."
        )
    else:
        verdict = "🟢 SHIFT LEVE"
        rec = (
            f"Las {scope_label} son razonablemente estables entre train y holdout. "
            "La degradación probablemente venga de calibración, optimización o umbrales, más que de drift fuerte."
        )
        action = (
            "Mantén la ventana de train y centra el análisis en calibración, early stopping, "
            "thresholds por contexto y robustez de la política de decisión."
        )

    return verdict, rec, action


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Análisis de shift distribucional — Train vs Holdout"
    )
    parser.add_argument("--output_dir",   default="./shift_report",
                        help="Directorio de salida (default: ./shift_report)")
    parser.add_argument("--top_features", type=int, default=25,
                        help="Nº de features a mostrar en tablas y plots (default: 25)")
    parser.add_argument("--no_plots",    action="store_true",
                        help="No generar gráficos PNG")
    parser.add_argument("--no_features", action="store_true",
                        help="Saltar análisis de features (solo régimen y labels)")
    parser.add_argument("--feature_scope", choices=["all", "model", "both"], default="model",
                        help="Qué features analizar: all=todas las numéricas, model=solo inputs reales, both=ambas (default: model)")
    parser.add_argument("--top_per_group", type=int, default=5,
                        help="Top N por bloque del modelo a mostrar (default: 5)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print("╔══════════════════════════════════════════════════════════════════════╗")
    print("║      ANÁLISIS DE SHIFT DISTRIBUCIONAL — Train vs Holdout            ║")
    print("╚══════════════════════════════════════════════════════════════════════╝")
    print(f"\n  Train:   {TRAIN_FROM.date()} → {TRAIN_TO.date()}")
    print(f"  Holdout: {HOLDOUT_FROM.date()} → {HOLDOUT_TO.date()}")
    print(f"  Output:  {args.output_dir}\n")

    # ── 1. Carga ──────────────────────────────────────────────────────────────
    print("📥 Cargando datos brutos...")
    df_full = load_data(TRAIN_FROM, HOLDOUT_TO)

    if "time" in df_full.columns:
        df_full["time"] = pd.to_datetime(df_full["time"])
        if not df_full["time"].is_monotonic_increasing:
            df_full = df_full.sort_values("time").reset_index(drop=True)
        df_raw_train = df_full[df_full["time"] <  pd.Timestamp(HOLDOUT_FROM)].copy()
        df_raw_hold  = df_full[df_full["time"] >= pd.Timestamp(HOLDOUT_FROM)].copy()
    else:
        split = int(len(df_full) * 0.75)
        df_raw_train = df_full.iloc[:split].copy()
        df_raw_hold  = df_full.iloc[split:].copy()

    print(f"  ✅ Train bruto:   {len(df_raw_train):,} filas")
    print(f"  ✅ Holdout bruto: {len(df_raw_hold):,} filas")

    # ── 2. Features ───────────────────────────────────────────────────────────
    print("\n⚙️  Generando features...")
    features_ok = False
    try:
        from mimo.models.model_builder import Config, ModelConfig
        from mimo.features.feature_builder import FeatureConfig
        from mimo.states_manager.state_detector import StateConfig

        general_config = Config(release="shift_analysis", use_oof=False)
        feature_config = FeatureConfig(
            ema_periods=[9, 21, 50],
            label_horizon=10,
            tp_barrier=2.5,
            sl_barrier=1.5,
            feature_masks={
                "long":  {"ema_bull": True, "rsi_oversold": True, "macd_positive": True,
                          "ema_bear": False, "rsi_overbought": False, "macd_negative": False},
                "short": {"ema_bear": True, "rsi_overbought": True, "macd_negative": True,
                          "ema_bull": False, "rsi_oversold": False, "macd_positive": False},
            },
            label_method="adaptive"
        )
        model_config  = ModelConfig(seq_len_short=64, seq_len_long=256)
        regime_config = StateConfig(adx_trend_threshold=25.0)

        df_train = build_features(df_raw_train, general_config, feature_config,
                                  model_config, regime_config)
        df_hold  = build_features(df_raw_hold,  general_config, feature_config,
                                  model_config, regime_config)

        features_ok = True
        print(f"  ✅ Features: train={len(df_train):,}  holdout={len(df_hold):,}")

    except Exception as e:
        print(f"  ⚠️  Error en pipeline ({e}). Usando columnas OHLCV brutos.")
        df_train = df_raw_train.copy()
        df_hold  = df_raw_hold.copy()

    # Columnas a analizar
    exclude = {"time", "signal", "state", "regime_weight",
               "open", "high", "low", "close", "volume",
               "spread", "real_volume", "tick_volume"}
    numeric_dtypes = [np.float32, np.float64, float, int, np.int32, np.int64]

    all_feature_cols = [
        c for c in df_train.columns
        if c not in exclude and df_train[c].dtype in numeric_dtypes
    ]

    model_feature_groups = {}
    model_feature_cols = []
    if features_ok:
        model_feature_groups = get_model_feature_groups(feature_config, side="long")
        model_feature_cols = [
            c for c in flatten_feature_groups(model_feature_groups)
            if c in df_train.columns and c in df_hold.columns
        ]

    if args.feature_scope == "all":
        feature_cols = all_feature_cols
        feature_scope_label = "todas las features numéricas del pipeline"
    elif args.feature_scope == "model":
        feature_cols = model_feature_cols if model_feature_cols else all_feature_cols
        feature_scope_label = "solo las features reales del modelo"
    else:
        feature_cols = all_feature_cols
        feature_scope_label = "todas las features numéricas del pipeline (con foco adicional en las del modelo)"

    print(f"  📊 Features numéricas disponibles: {len(all_feature_cols)}")
    if model_feature_cols:
        print(f"  🎯 Features reales del modelo:    {len(model_feature_cols)}")
        for group in ['sequence_short', 'sequence_long', 'context', 'time']:
            cols = model_feature_groups.get(group, [])
            if cols:
                print(f"     - {group:<14}: {len(cols):>2} cols")
    print(f"  🔎 Scope activo: {feature_scope_label}")

    # ── 3. Régimen ────────────────────────────────────────────────────────────
    print_section("DISTRIBUCIÓN DE RÉGIMEN (state)")
    regime_df = analyze_regime_distribution(df_train, df_hold)

    if not regime_df.empty:
        print(f"\n  {'Régimen':<20} {'Train %':>10} {'Holdout %':>11} {'Δ pp':>8}")
        print(f"  {'-' * 52}")
        for s in sorted(regime_df["state"].unique()):
            t_row = regime_df[(regime_df["state"] == s) & (regime_df["period"] == "train")]
            h_row = regime_df[(regime_df["state"] == s) & (regime_df["period"] == "holdout")]
            t_pct = t_row["pct"].values[0] if len(t_row) else 0.0
            h_pct = h_row["pct"].values[0] if len(h_row) else 0.0
            delta = h_pct - t_pct
            flag  = "  ⚠️" if abs(delta) > 10 else ""
            print(f"  {str(s):<20} {t_pct:>9.1f}% {h_pct:>10.1f}% {delta:>+7.1f}pp{flag}")

        # PSI de regímenes
        states = sorted(regime_df["state"].unique())
        eps = 1e-6
        t_dist = np.array([regime_df[(regime_df["state"]==s)&(regime_df["period"]=="train")]["pct"].values[0]
                           if len(regime_df[(regime_df["state"]==s)&(regime_df["period"]=="train")]) else 0.0
                           for s in states]) / 100 + eps
        h_dist = np.array([regime_df[(regime_df["state"]==s)&(regime_df["period"]=="holdout")]["pct"].values[0]
                           if len(regime_df[(regime_df["state"]==s)&(regime_df["period"]=="holdout")]) else 0.0
                           for s in states]) / 100 + eps
        t_dist /= t_dist.sum()
        h_dist /= h_dist.sum()
        regime_psi = float(np.sum((h_dist - t_dist) * np.log(h_dist / t_dist)))
        level = ("🔴 CRÍTICO" if regime_psi >= PSI_CRITICAL else
                 "🟠 MODERADO" if regime_psi >= PSI_WARNING else "🟢 ESTABLE")
        print(f"\n  PSI de distribución de regímenes: {regime_psi:.4f}  {level}")
        regime_df.to_csv(os.path.join(args.output_dir, "regime_distribution.csv"), index=False)
    else:
        print("  ⚠️  Columna 'state' no disponible.")

    # ── 4. Labels ─────────────────────────────────────────────────────────────
    print_section("DISTRIBUCIÓN DE LABELS (signal)")
    label_stats = analyze_label_distribution(df_train, df_hold)

    for period in ["train", "holdout"]:
        if period not in label_stats:
            continue
        s = label_stats[period]
        print(f"\n  {period.upper()}: pos_rate={s['pos_rate']:.4f} "
              f"({s['pos_rate']*100:.1f}%)   n_pos={s['n_pos']:,} / {s['n_total']:,}")

        by_key = f"{period}_by_state"
        if by_key in label_stats:
            print(f"    {'Régimen':<20} {'pos_rate':>10} {'n':>8}")
            for state, vals in sorted(label_stats[by_key].items()):
                print(f"    {str(state):<20} {vals['pos_rate']:>10.4f} {vals['n']:>8,}")

    if "train" in label_stats and "holdout" in label_stats:
        delta_pr = label_stats["holdout"]["pos_rate"] - label_stats["train"]["pos_rate"]
        flag = "  ⚠️  Cambio >5pp — puede afectar calibración" if abs(delta_pr) > 0.05 else ""
        print(f"\n  Δ pos_rate: {delta_pr:+.4f}  ({delta_pr*100:+.1f} pp){flag}")

    # ── 5. Features ───────────────────────────────────────────────────────────
    feature_results = pd.DataFrame()
    feature_results_model = pd.DataFrame()
    group_summary = pd.DataFrame()

    if not args.no_features and len(feature_cols) > 0:
        print_section(f"ANÁLISIS DE FEATURES ({len(feature_cols)} features) — PSI + KS")
        print("  Calculando... (1-3 min según nº de features)")

        feature_results = analyze_features(df_train, df_hold, feature_cols)

        if not feature_results.empty:
            alerts = summarize_alerts(feature_results)
            print(f"\n  RESUMEN DE ALERTAS [{args.feature_scope}]:")
            print(f"    🔴 Crítico  (PSI ≥ {PSI_CRITICAL}): {alerts['critical']:>3} features")
            print(f"    🟠 Warning  (PSI ≥ {PSI_WARNING}):  {alerts['warning']:>3} features")
            print(f"    🟢 OK       (PSI <  {PSI_WARNING}):  {alerts['ok']:>3} features")
            print(f"    📊 KS sign. (p < {KS_ALPHA}):       {alerts['ks_significant']:>3} features")

            print(f"\n  TOP {args.top_features} FEATURES POR PSI:")
            print_feature_table(feature_results, args.top_features)

            csv_path = os.path.join(args.output_dir, "feature_shift_analysis.csv")
            feature_results.to_csv(csv_path, index=False)
            print(f"\n  💾 CSV completo → {csv_path}")
        else:
            alerts = {}
    else:
        alerts = {}

    if not args.no_features and model_feature_cols:
        print_section(f"ANÁLISIS DE FEATURES DEL MODELO ({len(model_feature_cols)} inputs reales)")
        feature_results_model = analyze_features(df_train, df_hold, model_feature_cols)
        feature_results_model = annotate_feature_groups(feature_results_model, model_feature_groups)
        group_summary = summarize_alerts_by_group(feature_results_model)

        if not feature_results_model.empty:
            alerts_model = summarize_alerts(feature_results_model)
            print(f"\n  RESUMEN DE ALERTAS [model]:")
            print(f"    🔴 Crítico  (PSI ≥ {PSI_CRITICAL}): {alerts_model['critical']:>3} features")
            print(f"    🟠 Warning  (PSI ≥ {PSI_WARNING}):  {alerts_model['warning']:>3} features")
            print(f"    🟢 OK       (PSI <  {PSI_WARNING}):  {alerts_model['ok']:>3} features")
            print(f"    📊 KS sign. (p < {KS_ALPHA}):       {alerts_model['ks_significant']:>3} features")

            print_group_alert_summary(group_summary)
            print_top_features_by_group(feature_results_model, top_n_per_group=args.top_per_group)

            csv_model = os.path.join(args.output_dir, "feature_shift_analysis_model_only.csv")
            feature_results_model.to_csv(csv_model, index=False)
            print(f"\n  💾 CSV model-only → {csv_model}")

            if not group_summary.empty:
                csv_groups = os.path.join(args.output_dir, "feature_shift_group_summary.csv")
                group_summary.to_csv(csv_groups, index=False)
                print(f"  💾 CSV por bloque → {csv_groups}")
        else:
            alerts_model = {}
    else:
        alerts_model = alerts

    # ── 6. Correlaciones ──────────────────────────────────────────────────────
    print_section("CAMBIO EN CORRELACIONES (Top 10 pares)")
    corr_df = analyze_correlation_shift(df_train, df_hold, feature_cols)

    if not corr_df.empty:
        print(f"\n  {'Feature 1':<28} {'Feature 2':<28} "
              f"{'r_train':>9} {'r_hold':>8} {'Δ|r|':>7}")
        print(f"  {'-' * 84}")
        for _, row in corr_df.iterrows():
            print(f"  {row['feature_1']:<28} {row['feature_2']:<28} "
                  f"{row['corr_train']:>9.3f} {row['corr_holdout']:>8.3f} "
                  f"{row['delta_corr']:>7.3f}")
        corr_df.to_csv(os.path.join(args.output_dir, "correlation_shift.csv"), index=False)
    else:
        print("  (insuficientes features numéricas para comparar correlaciones)")

    # ── 7. Drift temporal ─────────────────────────────────────────────────────
    print_section("DRIFT TEMPORAL (media semanal)")
    drift_source = feature_results_model if not feature_results_model.empty else feature_results
    drift_base_cols = model_feature_cols if model_feature_cols else feature_cols
    top_drift = (drift_source.head(9)["feature"].tolist()
                 if not drift_source.empty else drift_base_cols[:9])

    df_all = pd.concat([df_train, df_hold], ignore_index=True)
    weekly_df = analyze_rolling_drift(df_all, top_drift)

    if not weekly_df.empty:
        weekly_df.to_csv(os.path.join(args.output_dir, "rolling_drift.csv"))
        print(f"  💾 Drift semanal → {args.output_dir}/rolling_drift.csv")
    else:
        print("  ⚠️  No hay columna 'time' — análisis temporal omitido.")

    # ── 8. Plots ──────────────────────────────────────────────────────────────
    if not args.no_plots:
        print_section("GENERANDO GRÁFICOS")
        if not feature_results.empty:
            plot_psi_summary(feature_results, args.top_features, args.output_dir)
            plot_top_psi_features(df_train, df_hold, feature_results,
                                  min(12, args.top_features), args.output_dir)
        if not regime_df.empty:
            plot_regime_distribution(regime_df, args.output_dir)
        if not weekly_df.empty:
            plot_rolling_drift(weekly_df, top_drift, args.output_dir)
        plot_label_rate_over_time(df_train, df_hold, args.output_dir)

    # ── 9. Diagnóstico final ──────────────────────────────────────────────────
    print_section("DIAGNÓSTICO FINAL")

    diagnosis_alerts = alerts_model if model_feature_cols else alerts
    diagnosis_scope = "features reales del modelo" if model_feature_cols else feature_scope_label
    verdict, rec, action = build_diagnosis(diagnosis_alerts, label_stats, regime_df, scope_label=diagnosis_scope)

    print(f"\n  Veredicto: {verdict}")
    print(f"\n  Qué está pasando:")
    print(f"    {rec}")
    print(f"\n  Acción recomendada:")
    print(f"    {action}")

    crit_source = feature_results_model if not feature_results_model.empty else feature_results
    if not crit_source.empty:
        crit = crit_source[crit_source["psi"] >= PSI_CRITICAL]
        crit_title = "Features críticas del modelo" if not feature_results_model.empty else "Features críticas"
        print(f"\n  {crit_title} ({len(crit)}):")
        if crit.empty:
            print("    Ninguna. ✅")
        else:
            for _, row in crit.head(10).iterrows():
                extra = f"  grupo={row['feature_group']}" if "feature_group" in row else ""
                print(f"    • {row['feature']:<35} PSI={row['psi']:.3f}  "
                      f"Δμ={row['mean_shift_pct']:+.1f}%  σ_ratio={row['std_ratio']:.2f}{extra}")

    print(f"\n  📁 Todos los resultados en: {os.path.abspath(args.output_dir)}/")

    # Archivos generados
    generated = [f for f in os.listdir(args.output_dir)
                 if f.endswith((".csv", ".png"))]
    print(f"\n  Archivos generados ({len(generated)}):")
    for f in sorted(generated):
        size = os.path.getsize(os.path.join(args.output_dir, f))
        print(f"    • {f:<45} ({size/1024:.1f} KB)")

    print("\n" + "═" * 70)
    print("  ✅ ANÁLISIS COMPLETADO")
    print("═" * 70 + "\n")


if __name__ == "__main__":
    main()