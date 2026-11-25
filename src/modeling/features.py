"""
src/modeling/features.py

Módulo de ingeniería de características espacio-temporales (Feature Engineering)
para SmartRebalance ZGZ. Compartido entre entrenamiento (train.py) e inferencia
batch (predict.py) para garantizar consistencia y ausencia de data leakage.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

# Constantes del dominio y del experimento
FREQ_MINUTES: int = 10
HORIZONS_HOURS: list[int] = [1, 2, 3]
OCCUPANCY_LOW_THRESHOLD: float = 0.20
OCCUPANCY_HIGH_THRESHOLD: float = 0.80  # Umbral real de negocio (ocupación observada > 80 %)

# Umbrales de alerta de saturación por defecto sobre pred_lgbm (calibrados en df_val
# sobre el snapshot de 23 días en EDA_prediccion_saturacion.ipynb). En entrenamiento
# se recalibran dinámicamente sobre df_val y sirven como respaldo si no hay casos > 80 %.
SATURATION_ALERT_THRESHOLDS: dict[int, float] = {
    1: 0.74,
    2: 0.70,
    3: 0.65,
}

CATEGORICAL_COLS: list[str] = ["station_id_cat"]

BASE_FEATURES: list[str] = [
    # Estado actual y capacidad física
    "occupancy_pct",
    "bikes_available",
    "anchors_free",
    "capacity",
    "disabled_units",
    # Ubicación e identidad espacial
    "station_id_cat",
    "latitude",
    "longitude",
    # Retardos, velocidad y volatilidad
    "occ_lag_10m",
    "occ_lag_30m",
    "occ_lag_1h",
    "occ_lag_24h",
    "occ_lag_7d",
    "delta_occ_30m",
    "delta_occ_1h",
    "rolling_std_1h",
    # Perfil basal causal
    "causal_hist_occ_mean",
    "occ_diff_vs_hist_mean",
    # Huella logística (furgonetas)
    "delta_bikes_10m",
    "is_logistic_jump",
    "logistic_jumps_last_1h",
    # Calendario actual
    "hour_sin",
    "hour_cos",
    "dow_sin",
    "dow_cos",
    "is_weekend_flag",
    "is_holiday",
]


def get_features_for_horizon(horizon_h: int) -> list[str]:
    """Devuelve la lista ordenada de predictores para un horizonte h concreto."""
    horizon_specific = [
        f"target_hour_sin_{horizon_h}h",
        f"target_hour_cos_{horizon_h}h",
        f"target_is_weekend_{horizon_h}h",
    ]
    return BASE_FEATURES + horizon_specific


def get_saturation_alert_threshold(horizon_h: int) -> float:
    """Devuelve el umbral de respaldo de alerta de saturación para el horizonte h."""
    return SATURATION_ALERT_THRESHOLDS.get(horizon_h, OCCUPANCY_HIGH_THRESHOLD)


def prepare_observed_base(
    df_gold: pd.DataFrame,
    df_stations: pd.DataFrame,
) -> pd.DataFrame:
    """
    Filtra exclusivamente registros observados reales (is_estimated == 0),
    incorpora atributos estáticos faltantes (capacity, latitude, longitude)
    desde dim_stations y calcula unidades averiadas.
    """
    df = df_gold[df_gold["is_estimated"] == 0].copy()

    station_attrs = ["capacity", "latitude", "longitude"]
    missing_attrs = [c for c in station_attrs if c not in df.columns and c in df_stations.columns]

    if missing_attrs:
        df = df.merge(
            df_stations[["station_id"] + missing_attrs],
            on="station_id",
            how="left",
            validate="m:1",
        )

    df["disabled_units"] = (
        df["capacity"] - df["bikes_available"] - df["anchors_free"]
    ).clip(lower=0)

    df = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True)
    return df


def add_multihorizon_targets(
    df: pd.DataFrame,
    horizons: Sequence[int] = HORIZONS_HOURS,
    tolerance_minutes: int = FREQ_MINUTES,
) -> pd.DataFrame:
    """
    Construye las variables objetivo futuras (target_occ_{h}h, target_bikes_{h}h)
    mediante merge_asof con tolerancia estricta para evitar saltos por huecos.
    Uso exclusivo en entrenamiento y evaluación.
    """
    out = df.sort_values(["timestamp", "station_id"]).reset_index(drop=True).copy()
    future_truth = out[
        ["station_id", "timestamp", "occupancy_pct", "bikes_available"]
    ].copy()

    for h in horizons:
        target_ts_col = f"_target_ts_{h}h"
        out[target_ts_col] = out["timestamp"] + pd.Timedelta(hours=h)

        out = pd.merge_asof(
            out.sort_values(target_ts_col),
            future_truth.rename(
                columns={
                    "timestamp": "_future_ts",
                    "occupancy_pct": f"target_occ_{h}h",
                    "bikes_available": f"target_bikes_{h}h",
                }
            ),
            left_on=target_ts_col,
            right_on="_future_ts",
            by="station_id",
            direction="nearest",
            tolerance=pd.Timedelta(minutes=tolerance_minutes),
        )
        out.drop(columns=[target_ts_col, "_future_ts"], inplace=True)

    return out.sort_values(["station_id", "timestamp"]).reset_index(drop=True)


def add_lag_and_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula retardos temporales estrictos, velocidad, volatilidad y proxy de furgonetas."""
    out = df.sort_values(["station_id", "timestamp"]).reset_index(drop=True).copy()

    ref = (
        out[["station_id", "timestamp", "occupancy_pct"]]
        .sort_values("timestamp")
        .reset_index(drop=True)
        .copy()
    )

    lag_specs = [
        ("occ_lag_10m", pd.Timedelta(minutes=10), pd.Timedelta(minutes=5)),
        ("occ_lag_30m", pd.Timedelta(minutes=30), pd.Timedelta(minutes=10)),
        ("occ_lag_1h", pd.Timedelta(hours=1), pd.Timedelta(minutes=10)),
        ("occ_lag_24h", pd.Timedelta(hours=24), pd.Timedelta(minutes=15)),
        ("occ_lag_7d", pd.Timedelta(days=7), pd.Timedelta(minutes=15)),
    ]

    for col_name, offset, tol in lag_specs:
        out["_lookup_ts"] = out["timestamp"] - offset
        out = pd.merge_asof(
            out.sort_values("_lookup_ts"),
            ref.rename(columns={"timestamp": "_ref_ts", "occupancy_pct": col_name}),
            left_on="_lookup_ts",
            right_on="_ref_ts",
            by="station_id",
            direction="nearest",
            tolerance=tol,
        )
        out.drop(columns=["_lookup_ts", "_ref_ts"], inplace=True)

    out = out.sort_values(["station_id", "timestamp"]).reset_index(drop=True)

    # Velocidad de cambio reciente
    out["delta_occ_30m"] = out["occupancy_pct"] - out["occ_lag_30m"]
    out["delta_occ_1h"] = out["occupancy_pct"] - out["occ_lag_1h"]

    # Volatilidad móvil de la última hora (6 capturas)
    out["rolling_std_1h"] = (
        out.groupby("station_id")["occupancy_pct"]
        .transform(lambda s: s.rolling(window=6, min_periods=2).std())
        .fillna(0.0)
    )

    # Proxy de intervención logística de furgonetas (saltos bruscos en ~10 min)
    dt_min = out.groupby("station_id")["timestamp"].diff().dt.total_seconds() / 60.0
    delta_bikes = out.groupby("station_id")["bikes_available"].diff()

    is_step_10m = dt_min.between(8, 12)
    out["delta_bikes_10m"] = np.where(is_step_10m, delta_bikes, 0.0)
    out["is_logistic_jump"] = (out["delta_bikes_10m"].abs() >= 8).astype(int)
    out["logistic_jumps_last_1h"] = (
        out.groupby("station_id")["is_logistic_jump"]
        .transform(lambda s: s.rolling(window=6, min_periods=1).sum())
    )

    return out


def add_causal_historical_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Calcula la media histórica causal por (estación, tipo de día, hora) usando solo días previos."""
    out = df.copy()
    out["_date_only"] = out["timestamp"].dt.floor("D")

    daily_slot = (
        out.groupby(["station_id", "day_type_effective", "hour", "_date_only"], as_index=False)
        .agg(
            _day_occ_sum=("occupancy_pct", "sum"),
            _day_occ_cnt=("occupancy_pct", "count"),
        )
        .sort_values(["station_id", "day_type_effective", "hour", "_date_only"])
    )

    grp_cols = ["station_id", "day_type_effective", "hour"]
    daily_slot["_cum_sum_prev"] = (
        daily_slot.groupby(grp_cols)["_day_occ_sum"].cumsum() - daily_slot["_day_occ_sum"]
    )
    daily_slot["_cum_cnt_prev"] = (
        daily_slot.groupby(grp_cols)["_day_occ_cnt"].cumsum() - daily_slot["_day_occ_cnt"]
    )

    daily_slot["causal_hist_occ_mean"] = np.where(
        daily_slot["_cum_cnt_prev"] > 0,
        daily_slot["_cum_sum_prev"] / daily_slot["_cum_cnt_prev"],
        np.nan,
    )

    out = out.merge(
        daily_slot[grp_cols + ["_date_only", "causal_hist_occ_mean"]],
        on=grp_cols + ["_date_only"],
        how="left",
        validate="m:1",
    )

    out["occ_diff_vs_hist_mean"] = out["occupancy_pct"] - out["causal_hist_occ_mean"]
    out.drop(columns=["_date_only"], inplace=True)

    return out


def add_calendar_and_horizon_features(
    df: pd.DataFrame,
    horizons: Sequence[int] = HORIZONS_HOURS,
    df_cal: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Genera variables de calendario y codificación cíclica para t y cada horizonte t+h."""
    out = df.copy()

    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24.0)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24.0)

    dow = out["timestamp"].dt.dayofweek
    out["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
    out["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)
    out["is_weekend_flag"] = (out["day_type_effective"] == "WEEKEND").astype(int)

    weekend_date_keys = None
    if df_cal is not None and {"date_key", "day_type_effective"}.issubset(df_cal.columns):
        weekend_date_keys = set(
            df_cal.loc[df_cal["day_type_effective"] == "WEEKEND", "date_key"].astype(int)
        )

    for h in horizons:
        future_ts = out["timestamp"] + pd.Timedelta(hours=h)
        f_hour = future_ts.dt.hour
        f_dow = future_ts.dt.dayofweek

        out[f"target_hour_sin_{h}h"] = np.sin(2 * np.pi * f_hour / 24.0)
        out[f"target_hour_cos_{h}h"] = np.cos(2 * np.pi * f_hour / 24.0)

        if weekend_date_keys is not None:
            f_date_key = future_ts.dt.strftime("%Y%m%d").astype(int)
            out[f"target_is_weekend_{h}h"] = np.where(
                f_date_key.isin(weekend_date_keys) | (f_dow >= 5), 1, 0
            )
        else:
            out[f"target_is_weekend_{h}h"] = (f_dow >= 5).astype(int)

    return out


def build_feature_matrix(
    df: pd.DataFrame,
    horizons: Sequence[int] = HORIZONS_HOURS,
    df_cal: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Pipeline completo de Feature Engineering listo para entrenamiento e inferencia."""
    df_feat = add_lag_and_rolling_features(df)
    df_feat = add_causal_historical_profile(df_feat)
    df_feat = add_calendar_and_horizon_features(df_feat, horizons=horizons, df_cal=df_cal)
    df_feat["station_id_cat"] = df_feat["station_id"].astype("category")
    return df_feat


def align_tree_features(
    df: pd.DataFrame,
    feature_cols: list[str],
    station_categories: list[int] | None = None,
) -> tuple[pd.DataFrame, list[int]]:
    """
    Extrae las columnas predictoras y alinea los niveles categóricos de station_id_cat.
    Si station_categories es None, toma las categorías presentes en df (uso en Train).
    """
    X = df[feature_cols].copy()
    if station_categories is None:
        categories = [int(c) for c in X["station_id_cat"].cat.categories]
    else:
        categories = [int(c) for c in station_categories]

    X["station_id_cat"] = pd.Categorical(X["station_id_cat"], categories=categories)
    return X, categories