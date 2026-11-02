"""
src/models_gold.py
-------------------
Módulo de modelado dimensional (Capa Gold) y construcción del dataset analítico.
Contiene las funciones para construir las dimensiones (Dim_Stations, Dim_Patterns,
Dim_Calendar), la tabla de hechos (Fact_StationSnapshots) y resolver el problema
del cold start / extrapolación histórica.
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
    df_dim["station_id"] = df_dim["station_id"].astype("int64")
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
    df_dim["station_id"] = df_dim["station_id"].astype("int64")
    df_dim["hour"] = df_dim["hour"].astype("int64")
    df_dim["day_type"] = df_dim["day_type"].astype(str).str.upper().str.strip()
    return df_dim


def fetch_public_holidays(years: List[int]) -> Set[str]:
    """
    Obtiene los festivos nacionales y autonómicos (Aragón - ES-AR) desde la API pública Nager.Date.
    Si la API falla o no responde, devuelve un conjunto vacío (el fallback manual lo asumirá).
    
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
    
    Args:
        years: Lista de años a cubrir.
        custom_md_list: Lista opcional de tuplas (mes, día).
    Returns:
        Set[str]: Conjunto de cadenas 'YYYY-MM-DD'.
    """
    if custom_md_list is None:
        # Días festivos y vísperas clave (Mes, Día):
        # - Locales Zaragoza: San Valero (29 ene), Cincomarzada (5 mar), Pilar (12 oct)
        # - Vísperas/Especiales: Nochebuena (24 dic), Nochevieja (31 dic)
        # - Fijos Nacionales / Fallback API: Año Nuevo (1 ene), Reyes (6 ene), Navidad (25 dic)
        custom_md_list = [
            (1, 1),    # Año Nuevo
            (1, 6),    # Reyes Magos
            (1, 29),   # San Valero (Local Zaragoza)
            (3, 5),    # Cincomarzada (Local Zaragoza)
            (10, 12),  # El Pilar / Fiesta Nacional
            (12, 24),  # Nochebuena (Comportamiento festivo en Bizi)
            (12, 25),  # Navidad
            (12, 31),  # Nochevieja (Comportamiento festivo en Bizi)
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
    Construye la dimensión calendario (Dim_Calendar) a partir de las marcas temporales
    del histórico, descargando festivos vía API pública y aplicando el generador 
    recurrente multi-año para locales/vísperas, enriqueciendo con métricas macro.
    
    Args:
        df_snapshots: DataFrame de hechos/capturas con columna 'date'.
        df_history_silver: DataFrame de bizi_history procesado en Silver.
        custom_local_holidays_md: Lista opcional de tuplas (mes, día) para festivos fijos.
    Returns:
        pd.DataFrame: Dimensión calendario con claves temporales, festivos y métricas macro.
    """
    if df_snapshots.empty or "date" not in df_snapshots.columns:
        return pd.DataFrame()

    min_date = df_snapshots["date"].min()
    max_date = df_snapshots["date"].max()
    date_range = pd.date_range(start=min_date, end=max_date, freq="D")

    df_calendar = pd.DataFrame({"date": date_range})
    df_calendar["date_key"] = df_calendar["date"].dt.strftime("%Y%m%d").astype("int64")
    df_calendar["year"] = df_calendar["date"].dt.year
    df_calendar["day_of_week"] = df_calendar["date"].dt.dayofweek
    df_calendar["is_weekend"] = df_calendar["day_of_week"].isin([5, 6]).astype("int64")

    # Extraer años únicos cubiertos en el rango temporal
    years = df_calendar["year"].unique().tolist()

    # 1. Consulta a la API pública de festivos Nager.Date
    api_holidays = fetch_public_holidays(years)

    # 2. Generación dinámica de festivos recurrentes para todos los años
    recurring_holidays = generate_recurring_holidays(years, custom_local_holidays_md)

    # 3. Unión de conjuntos (API + Recurrentes manuales)
    all_holidays = api_holidays.union(recurring_holidays)

    # 4. Asignación de is_holiday y tipo de día efectivo
    df_calendar["is_holiday"] = (
        df_calendar["date"].dt.strftime("%Y-%m-%d").isin(all_holidays).astype("int64")
    )

    df_calendar["day_type_effective"] = "WEEKDAY"
    df_calendar.loc[
        (df_calendar["is_weekend"] == 1) | (df_calendar["is_holiday"] == 1),
        "day_type_effective",
    ] = "WEEKEND"

    # Enriquecer con variables macro de history_silver si está disponible
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
    las marcas temporales a Europe/Madrid y generando las claves de cruce.
    
    Args:
        df_historical_silver: DataFrame histórico de parquets procesado en Silver.
    Returns:
        pd.DataFrame: Tabla de hechos estructurada con date_key, hour, bikes_available, etc.
    """
    if df_historical_silver.empty:
        return pd.DataFrame()

    df = df_historical_silver.copy()
    time_col = "timestamp" if "timestamp" in df.columns else "recordedAt"
    
    # Uso de la función segura para evitar errores de doble conversión
    df["timestamp"] = safe_local_timestamp(df[time_col])

    df["date"] = df["timestamp"].dt.floor("D")
    df["date_key"] = df["timestamp"].dt.strftime("%Y%m%d").astype("int64")
    df["hour"] = df["timestamp"].dt.hour.astype("int64")

    cols_map = {"id": "station_id", "bikesAvailable": "bikes_available", "anchorsFree": "anchors_free"}
    df.rename(columns=cols_map, inplace=True, errors="ignore")

    keep_cols = ["station_id", "date_key", "date", "timestamp", "hour", "bikes_available", "anchors_free"]
    existing_keep = [c for c in keep_cols if c in df.columns]

    df_fact = df[existing_keep].copy()
    if "station_id" in df_fact.columns:
        df_fact["station_id"] = df_fact["station_id"].astype("int64")
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
    Realiza la extrapolación para resolver el cold start si existen vacíos.
    
    Args:
        df_snapshots_gold: Tabla de hechos Fact_StationSnapshots.
        df_calendar: Dimensión Dim_Calendar enriquecida con contexto macro.
        df_stations_gold: Dimensión Dim_Stations.
        df_patterns_gold: Dimensión Dim_Patterns.
        df_rankings_silver: Métricas de fricción/rotación de rankings en Silver.
    Returns:
        pd.DataFrame: Dataset integrado listo para consumo analítico.
    """
    if df_snapshots_gold.empty:
        return pd.DataFrame()

    df_gold = df_snapshots_gold.copy()

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

    # Cálculo de ocupación real en %
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

    # Cruce 4: Snapshots + Rankings
    if df_rankings_silver is not None and not df_rankings_silver.empty:
        rk_cols = ["station_id", "turnover_score", "empty_hours", "problem_hours"]
        exist_rk = [c for c in rk_cols if c in df_rankings_silver.columns]
        df_gold = df_gold.merge(df_rankings_silver[exist_rk], on="station_id", how="left")

        # Imputación de nulos en rankings
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
    if "occupancy_pct" in df_gold.columns and df_gold["occupancy_pct"].isnull().any():
        city_scale = df_gold["city_avg_occupancy"].fillna(0.5) / 0.5
        extrapolated = (df_gold["occupancy_avg"] * city_scale).clip(0.0, 1.0).round(4)
        df_gold["occupancy_pct"] = df_gold["occupancy_pct"].fillna(extrapolated)

    # Ordenar por estación y tiempo
    if "station_id" in df_gold.columns and "timestamp" in df_gold.columns:
        df_gold.sort_values(by=["station_id", "timestamp"], inplace=True)
        df_gold.reset_index(drop=True, inplace=True)

    return df_gold