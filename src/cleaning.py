"""
src/cleaning.py
----------------
Módulo de limpieza, estandarización de esquemas y transformaciones (Capa Silver).
Contiene funciones puras que reciben DataFrames en crudo (Bronze) y devuelven
DataFrames estandarizados y tipados para la Capa Silver.
"""

from typing import Any, Dict
import json
import numpy as np
import pandas as pd

from src.utils import parse_dict_field


def clean_stations(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y estandariza el maestro de estaciones (stations_raw.json).
    - Renombra atributos a snake_case.
    - Castea tipos numéricos y timestamps con zona horaria Europe/Madrid.
    """
    if df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()
    
    # 1. Renombrar columnas
    rename_dict = {
        "id": "station_id",
        "lat": "latitude",
        "lon": "longitude",
        "bikesAvailable": "bikes_available",
        "anchorsFree": "anchors_free",
    }
    df.rename(columns=rename_dict, inplace=True)

    # 2. Convertir timestamp a Datetime con zona horaria Europe/Madrid
    if "recordedAt" in df.columns:
        df["recordedAt"] = pd.to_datetime(df["recordedAt"], errors="coerce", utc=True)
        df["recordedAt"] = df["recordedAt"].dt.tz_convert("Europe/Madrid")

    # 3. Formatear string
    if "name" in df.columns:
        df["name"] = df["name"].astype(str).str.strip()

    # 4. Forzar tipos enteros y flotantes
    int_cols = ["station_id", "capacity", "bikes_available", "anchors_free"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")

    float_cols = ["latitude", "longitude"]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    return df


def clean_rebalancing(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el reporte de transferencias logísticas (rebalancing_raw.json).
    - Renombra identificadores de origen/destino.
    - Desanida timeWindow, expectedImpact y ajusta motivos usando src.utils.
    """
    if df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()
    
    rename_dict = {
        "originStationId": "origin_station_id",
        "originStationName": "origin_station_name",
        "destinationStationId": "destination_station_id",
        "destinationStationName": "destination_station_name",
        "bikesToMove": "bikes_to_move",
        "matchScore": "match_score",
        "logisticsScore": "logistics_score",
    }
    df.rename(columns=rename_dict, inplace=True)

    # Desanidar ventanas de tiempo e impacto estimado mediante src.utils.parse_dict_field
    if "timeWindow" in df.columns:
        tw_parsed = df["timeWindow"].apply(parse_dict_field)
        df["time_window_start"] = tw_parsed.apply(lambda x: x.get("start", None))
        df["time_window_end"] = tw_parsed.apply(lambda x: x.get("end", None))
        df.drop(columns=["timeWindow"], errors="ignore", inplace=True)

    if "expectedImpact" in df.columns:
        ei_parsed = df["expectedImpact"].apply(parse_dict_field)
        df["empties_avoided"] = ei_parsed.apply(lambda x: x.get("emptiesAvoided", 0)).fillna(0).astype("int64")
        df["fulls_avoided"] = ei_parsed.apply(lambda x: x.get("fullsAvoided", 0)).fillna(0).astype("int64")
        df.drop(columns=["expectedImpact"], errors="ignore", inplace=True)

    if "reasons" in df.columns:
        df["reasons_clean"] = df["reasons"].astype(str).str.replace(r"[\[\]']", "", regex=True).str.strip()
        df.drop(columns=["reasons"], errors="ignore", inplace=True)

    # Tipado explícito de IDs
    for col in ["origin_station_id", "destination_station_id", "bikes_to_move"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")

    return df


def clean_rankings(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia métricas operativas y horas de estrés (rankings_raw.json).
    - Normaliza nombres de columnas y rangos de fechas.
    - Calcula proporciones normalizadas de fricción (full_hour_share, problem_hour_share).
    """
    if df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()
    
    rename_dict = {
        "id": "ranking_id",
        "stationId": "station_id",
        "stationName": "station_name",
        "districtName": "district_name",
        "turnoverScore": "turnover_score",
        "emptyHours": "empty_hours",
        "fullHours": "full_hours",
        "totalHours": "total_hours",
        "problemHours": "problem_hours",
        "emptyHourShare": "empty_hour_share",
        "windowStart": "window_start",
        "windowEnd": "window_end",
        "demandVsStressedHint": "demand_vs_stressed_hint",
    }
    df.rename(columns=rename_dict, inplace=True)

    # Castear station_id
    if "station_id" in df.columns:
        df["station_id"] = pd.to_numeric(df["station_id"], errors="coerce").fillna(0).astype("int64")

    # Castear métricas numéricas
    num_cols = ["turnover_score", "empty_hours", "full_hours", "total_hours", "problem_hours"]
    for col in num_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")

    # Fechas
    for date_col in ["window_start", "window_end"]:
        if date_col in df.columns:
            clean_date_str = df[date_col].astype(str).str.replace(r"\s*\([^)]*\)", "", regex=True)
            df[date_col] = pd.to_datetime(clean_date_str, errors="coerce")

    # Derivar proporciones si existen métricas necesarias
    if "full_hours" in df.columns and "total_hours" in df.columns:
        df["full_hour_share"] = np.where(df["total_hours"] > 0, (df["full_hours"] / df["total_hours"]).round(4), 0.0)
    if "problem_hours" in df.columns and "total_hours" in df.columns:
        df["problem_hour_share"] = np.where(df["total_hours"] > 0, (df["problem_hours"] / df["total_hours"]).round(4), 0.0)

    return df


def clean_patterns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia los perfiles basales de ocupación horaria (patterns_raw.json).
    - Estandariza day_type (WEEKDAY / WEEKEND).
    - Estandariza tipos numéricos.
    """
    if df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()
    
    rename_dict = {
        "stationId": "station_id",
        "dayType": "day_type",
        "occupancyAvg": "occupancy_avg",
        "bikesAvg": "bikes_avg",
        "anchorsAvg": "anchors_avg",
        "sampleCount": "sample_count",
    }
    df.rename(columns=rename_dict, inplace=True)

    if "station_id" in df.columns:
        df["station_id"] = pd.to_numeric(df["station_id"], errors="coerce").fillna(0).astype("int64")
    if "hour" in df.columns:
        df["hour"] = pd.to_numeric(df["hour"], errors="coerce").fillna(0).astype("int64")

    for col in ["occupancy_avg", "bikes_avg", "anchors_avg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    if "day_type" in df.columns:
        df["day_type"] = df["day_type"].astype(str).str.strip().str.upper()

    return df


def clean_history(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el histórico macro diario de la ciudad (bizi_history_raw.csv).
    - Renombra variables y convierte 'day'/'date' a datetime y date_key.
    - Estandariza indicadores globales de la red.
    """
    if df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()
    
    rename_dict = {
        "day": "date_str",
        "demandScore": "demand_score",
        "avgOccupancy": "city_avg_occupancy",
        "balanceIndex": "balance_index",
        "sampleCount": "sample_count",
    }
    df.rename(columns=rename_dict, inplace=True)

    # Manejo de la columna fecha
    time_col = "date_str" if "date_str" in df.columns else df.columns[0]
    df["date"] = pd.to_datetime(df[time_col], errors="coerce")
    df["date_key"] = df["date"].dt.strftime("%Y%m%d").astype("int64")

    for col in ["city_avg_occupancy", "balance_index"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(4).astype("float64")

    for col in ["demand_score", "sample_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")

    df.sort_values(by="date", inplace=True)
    df.drop_duplicates(subset=["date_key"], keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def clean_historical_snapshots(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y estandariza los snapshots históricos de estaciones (.parquet).
    - Renombra atributos a snake_case (id/stationId -> station_id, lat/lon -> latitude/longitude, etc.).
    - Castea marcas temporales a Datetime con zona horaria Europe/Madrid (timestamp).
    - Forzado explícito de tipos enteros y flotantes.
    - Elimina lecturas duplicadas por estación y marca temporal.
    """
    if df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()

    # 1. Estandarización de nombres de columna
    rename_dict = {
        "id": "station_id",
        "stationId": "station_id",
        "lat": "latitude",
        "lon": "longitude",
        "bikesAvailable": "bikes_available",
        "anchorsFree": "anchors_free",
    }
    df.rename(columns=rename_dict, inplace=True)

    # 2. Homogeneización de la marca temporal (recordedAt o timestamp)
    time_col = "recordedAt" if "recordedAt" in df.columns else "timestamp"
    if time_col in df.columns:
        df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        df["timestamp"] = df["timestamp"].dt.tz_convert("Europe/Madrid")
        if time_col != "timestamp":
            df.drop(columns=[time_col], inplace=True)

    # 3. Casteo de tipos numéricos
    int_cols = ["station_id", "capacity", "bikes_available", "anchors_free"]
    for col in int_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype("int64")

    float_cols = ["latitude", "longitude"]
    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # 4. Eliminación de lecturas duplicadas por estación e instante
    if "station_id" in df.columns and "timestamp" in df.columns:
        df.sort_values(by=["station_id", "timestamp"], inplace=True)
        df.drop_duplicates(subset=["station_id", "timestamp"], inplace=True)

    df.reset_index(drop=True, inplace=True)
    return df