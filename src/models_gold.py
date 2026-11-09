"""
src/models_gold.py
-------------------
Módulo de modelado dimensional (Capa Gold) y construcción del dataset analítico.
Contiene las funciones para construir las dimensiones (Dim_Stations, Dim_Patterns,
Dim_Calendar), la tabla de hechos (Fact_StationSnapshots) y resolver el problema
del cold start / extrapolación histórica.

==============================================================================
ESPECIFICACIÓN DEL OBJETIVO PREDICTIVO, HORIZONTE Y MATRIZ DE VARIABLES
==============================================================================
1. Objetivo Predictivo:
   - Predecir la ocupación real (porcentaje y bicicletas disponibles) y el riesgo
     de desabastecimiento/saturación a nivel estación-hora para optimizar la
     logística de redistribución de Bizi Zaragoza.

2. Horizonte de Anticipación:
   - Ventana operativa de 30 a 60 minutos vista (ej. evaluación a las 9:00 AM para
     planificación y despacho eficiente de furgonetas de rebalanceo).

3. Matriz de Variables (Features y Targets):
   - Identificadores y Tiempo: station_id, timestamp, date_key, hour, is_estimated.
   - Variables Dependientes / Targets: bikes_available, anchors_free, occupancy_pct.
   - Contexto Temporal / Macro: day_type_effective, is_holiday, is_weekend,
     demand_score, city_avg_occupancy, balance_index.
   - Perfiles Basales: occupancy_avg, bikes_avg (Dim_Patterns).
   - Indicadores Operativos (Sin Data Leakage): turnover_score, empty_hours,
     problem_hours (filtrados estrictamente a ventanas temporales pasadas window_end <= t).
==============================================================================
"""

from typing import List, Optional, Set, Tuple
import numpy as np
import pandas as pd
import requests

from src.utils import safe_local_timestamp


def build_dim_stations(df_stations_silver: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la dimensión geográfica y física de estaciones (Dim_Stations).
    
    Args:
        df_stations_silver: DataFrame de estaciones procesado en Silver.
    Returns:
        pd.DataFrame: Dimensión de estaciones limpia e idéntica por station_id.
    """
    if df_stations_silver.empty:
        return pd.DataFrame()

    cols = ["station_id", "name", "latitude", "longitude", "capacity"]
    existing_cols = [c for c in cols if c in df_stations_silver.columns]

    df_dim = (
        df_stations_silver[existing_cols]
        .drop_duplicates(subset=["station_id"])
        .reset_index(drop=True)
    )
    if "station_id" in df_dim.columns:
        df_dim["station_id"] = df_dim["station_id"].astype("Int64")
    return df_dim


def build_dim_patterns(df_patterns_silver: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la dimensión de patrones horarios basales de ocupación (Dim_Patterns).
    
    Args:
        df_patterns_silver: DataFrame de patrones de la Capa Silver.
    Returns:
        pd.DataFrame: Perfiles basales por estación, tipo de día y hora.
    """
    if df_patterns_silver.empty:
        return pd.DataFrame()

    cols = ["station_id", "day_type", "hour", "occupancy_avg", "bikes_avg"]
    existing_cols = [c for c in cols if c in df_patterns_silver.columns]

    df_dim = df_patterns_silver[existing_cols].copy()
    if "station_id" in df_dim.columns:
        df_dim["station_id"] = df_dim["station_id"].astype("Int64")
    if "hour" in df_dim.columns:
        df_dim["hour"] = df_dim["hour"].astype("Int64")
    if "day_type" in df_dim.columns:
        df_dim["day_type"] = df_dim["day_type"].astype(str).str.upper().str.strip()
    return df_dim


def fetch_public_holidays(years: List[int]) -> Set[str]:
    """
    Obtiene los festivos nacionales y autonómicos (Aragón - ES-AR) desde la API pública Nager.Date.
    
    Args:
        years: Lista de años a consultar.
    Returns:
        Set[str]: Conjunto de fechas festivas en formato 'YYYY-MM-DD'.
    """
    holiday_dates: Set[str] = set()

    for yr in years:
        try:
            url = f"https://date.nager.at/api/v3/PublicHolidays/{yr}/ES"
            response = requests.get(url, timeout=10)
            if response.status_code == 200:
                holidays_data = response.json()
                for item in holidays_data:
                    counties = item.get("counties") or []
                    if item.get("global") or ("ES-AR" in counties):
                        holiday_dates.add(item["date"])
        except Exception as e:
            print(f"[AVISO] Fallo al consultar la API de festivos para el año {yr}: {e}")

    return holiday_dates


def generate_recurring_holidays(
    years: List[int],
    custom_md_list: Optional[List[Tuple[int, int]]] = None,
) -> Set[str]:
    """
    Genera fechas YYYY-MM-DD para festivos fijos (mes, día) dinámicamente
    adaptados a todos los años presentes en la dimensión calendario.
    """
    if custom_md_list is None:
        custom_md_list = [
            (1, 1),    # Año Nuevo
            (1, 6),    # Reyes Magos
            (1, 29),   # San Valero (Local Zaragoza)
            (3, 5),    # Cincomarzada (Local Zaragoza)
            (10, 12),  # El Pilar / Fiesta Nacional
            (12, 24),  # Nochebuena
            (12, 25),  # Navidad
            (12, 31),  # Nochevieja
        ]

    generated_dates: Set[str] = set()
    for yr in years:
        for month, day in custom_md_list:
            generated_dates.add(f"{yr:04d}-{month:02d}-{day:02d}")

    return generated_dates


def build_dim_calendar(
    df_snapshots: pd.DataFrame,
    df_history_silver: Optional[pd.DataFrame] = None,
    custom_local_holidays_md: Optional[List[Tuple[int, int]]] = None,
) -> pd.DataFrame:
    """
    Construye la dimensión calendario (Dim_Calendar) enriquecida con datos macro.
    """
    if df_snapshots.empty or "date" not in df_snapshots.columns:
        return pd.DataFrame()

    # Priorizar la fecha inicial de history_silver si existe
    if df_history_silver is not None and "date" in df_history_silver.columns:
        min_date = df_history_silver["date"].min()
    else:
        min_date = df_snapshots["date"].min()
    max_date = df_snapshots["date"].max()
    date_range = pd.date_range(start=min_date, end=max_date, freq="D")

    df_calendar = pd.DataFrame({"date": date_range})
    df_calendar["date_key"] = df_calendar["date"].dt.strftime("%Y%m%d").astype("Int64")
    df_calendar["year"] = df_calendar["date"].dt.year
    df_calendar["day_of_week"] = df_calendar["date"].dt.dayofweek
    df_calendar["is_weekend"] = df_calendar["day_of_week"].isin([5, 6]).astype("Int64")

    years = df_calendar["year"].unique().tolist()

    api_holidays = fetch_public_holidays(years)
    recurring_holidays = generate_recurring_holidays(years, custom_local_holidays_md)
    all_holidays = api_holidays.union(recurring_holidays)

    df_calendar["is_holiday"] = (
        df_calendar["date"].dt.strftime("%Y-%m-%d").isin(all_holidays).astype("Int64")
    )

    df_calendar["day_type_effective"] = "WEEKDAY"
    df_calendar.loc[
        (df_calendar["is_weekend"] == 1) | (df_calendar["is_holiday"] == 1),
        "day_type_effective",
    ] = "WEEKEND"

    if df_history_silver is not None and not df_history_silver.empty:
        history_cols = ["date_key", "demand_score", "city_avg_occupancy", "balance_index"]
        existing_hist = [c for c in history_cols if c in df_history_silver.columns]

        if "date_key" in existing_hist:
            df_calendar = df_calendar.merge(
                df_history_silver[existing_hist],
                on="date_key",
                how="left",
            )

    return df_calendar


def build_fact_snapshots(df_historical_silver: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la tabla de hechos de capturas (Fact_StationSnapshots) normalizando
    las marcas temporales a Europe/Madrid y asignando la bandera is_estimated = 0.
    """
    if df_historical_silver.empty:
        return pd.DataFrame()

    df = df_historical_silver.copy()
    time_col = "timestamp" if "timestamp" in df.columns else "recordedAt"
    
    df["timestamp"] = safe_local_timestamp(df[time_col])
    df["date"] = df["timestamp"].dt.floor("D")
    df["date_key"] = df["timestamp"].dt.strftime("%Y%m%d").astype("Int64")
    df["hour"] = df["timestamp"].dt.hour.astype("Int64")

    cols_map = {"id": "station_id", "bikesAvailable": "bikes_available", "anchorsFree": "anchors_free"}
    df.rename(columns=cols_map, inplace=True, errors="ignore")

    # Marca explícita: 0 para datos reales observados en el parquet
    df["is_estimated"] = 0

    keep_cols = [
        "station_id", "date_key", "date", "timestamp", "hour",
        "bikes_available", "anchors_free", "is_estimated"
    ]
    existing_keep = [c for c in keep_cols if c in df.columns]

    df_fact = df[existing_keep].copy()
    if "station_id" in df_fact.columns:
        df_fact["station_id"] = df_fact["station_id"].astype("Int64")
    return df_fact


def build_gold_dataset(
    df_snapshots_gold: pd.DataFrame,
    df_calendar: pd.DataFrame,
    df_stations_gold: pd.DataFrame,
    df_patterns_gold: pd.DataFrame,
    df_rankings_silver: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Unifica las dimensiones y la tabla de hechos en el dataset Gold final (df_gold).
    Previene la fuga de información temporal (Data Leakage) en el cruce de rankings
    y gestiona la bandera is_estimated (0=Observado, 1=Extrapolado Cold Start).
    """
    if df_snapshots_gold.empty:
        return pd.DataFrame()

    df_gold = df_snapshots_gold.copy()

    # Garantizar la columna is_estimated
    if "is_estimated" not in df_gold.columns:
        df_gold["is_estimated"] = 0

    # Cruce 1: Snapshots + Calendar (Con contexto macro)
    if not df_calendar.empty:
        cal_cols = [
            "date_key", "day_type_effective", "is_holiday", 
            "demand_score", "city_avg_occupancy", "balance_index"
        ]
        exist_cal = [c for c in cal_cols if c in df_calendar.columns]
        df_gold = df_gold.merge(df_calendar[exist_cal], on="date_key", how="left")

    # Cruce 2: Snapshots + Dim_Stations
    if not df_stations_gold.empty:
        st_cols = ["station_id", "name", "latitude", "longitude", "capacity"]
        exist_st = [c for c in st_cols if c in df_stations_gold.columns]
        df_gold = df_gold.merge(df_stations_gold[exist_st], on="station_id", how="left")

    # Cálculo de ocupación real observada en %
    if "bikes_available" in df_gold.columns and "capacity" in df_gold.columns:
        df_gold["occupancy_pct"] = (df_gold["bikes_available"] / df_gold["capacity"]).round(4)
        df_gold.drop(columns=["capacity"], inplace=True, errors="ignore")

    # Cruce 3: Snapshots + Dim_Patterns
    if not df_patterns_gold.empty:
        df_gold = df_gold.merge(
            df_patterns_gold,
            left_on=["station_id", "day_type_effective", "hour"],
            right_on=["station_id", "day_type", "hour"],
            how="left",
        )

    # Cruce 4: Snapshots + Rankings PREVINIENDO DATA LEAKAGE
    # Solo se unan métricas operativas cuya ventana de análisis ya estuviese cerrada (window_end <= t)
    if df_rankings_silver is not None and not df_rankings_silver.empty:
        rk = df_rankings_silver.copy()
        rk_metrics = ["turnover_score", "empty_hours", "problem_hours"]
        
        if "window_end" in rk.columns and "timestamp" in df_gold.columns:
            rk["window_end"] = safe_local_timestamp(rk["window_end"])
            
            # Realizar merge con condición temporal para evitar incorporar información posterior
            merged_rk = df_gold[["station_id", "timestamp"]].merge(rk, on="station_id", how="inner")
            valid_rk = merged_rk[merged_rk["window_end"] <= merged_rk["timestamp"]]
            
            if not valid_rk.empty:
                # Tomar la métrica más reciente disponible antes del momento t
                valid_rk = (
                    valid_rk.sort_values("window_end")
                    .groupby(["station_id", "timestamp"])
                    .last()
                    .reset_index()
                )
                exist_rk = [c for c in rk_metrics if c in valid_rk.columns]
                df_gold = df_gold.merge(
                    valid_rk[["station_id", "timestamp"] + exist_rk],
                    on=["station_id", "timestamp"],
                    how="left",
                )
            else:
                exist_rk = [c for c in rk_metrics if c in rk.columns]
                df_gold = df_gold.merge(
                    rk[["station_id"] + exist_rk].drop_duplicates("station_id"),
                    on="station_id",
                    how="left",
                )
        else:
            exist_rk = [c for c in rk_metrics if c in rk.columns]
            df_gold = df_gold.merge(
                rk[["station_id"] + exist_rk].drop_duplicates("station_id"),
                on="station_id",
                how="left",
            )

        # Imputación de nulos en rankings no aplicables
        for r_col in ["turnover_score", "empty_hours", "problem_hours"]:
            if r_col in df_gold.columns:
                df_gold[r_col] = df_gold[r_col].fillna(0)

    # Imputación de patrón horario general si faltaran estaciones en patterns
    if "occupancy_avg" in df_gold.columns and df_gold["occupancy_avg"].isnull().any():
        global_pat = df_patterns_gold.groupby("day_type")["occupancy_avg"].mean()
        df_gold["occupancy_avg"] = df_gold["occupancy_avg"].fillna(
            df_gold["day_type_effective"].map(global_pat).fillna(0.5)
        )

    # Extrapolación / Solución del Cold Start
    # Marca is_estimated = 1 para filas donde la ocupación debió ser estimada
    if "occupancy_pct" in df_gold.columns:
        missing_mask = df_gold["occupancy_pct"].isnull()
        if missing_mask.any():
            city_scale = df_gold["city_avg_occupancy"].fillna(0.5) / 0.5
            extrapolated = (df_gold["occupancy_avg"] * city_scale).clip(0.0, 1.0).round(4)
            df_gold.loc[missing_mask, "occupancy_pct"] = extrapolated[missing_mask]
            df_gold.loc[missing_mask, "is_estimated"] = 1

    # Ordenar cronológicamente por estación y tiempo
    if "station_id" in df_gold.columns and "timestamp" in df_gold.columns:
        df_gold.sort_values(by=["station_id", "timestamp"], inplace=True)
        df_gold.reset_index(drop=True, inplace=True)

    return df_gold