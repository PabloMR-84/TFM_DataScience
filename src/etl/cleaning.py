"""
src/etl/cleaning.py
-------------------
Módulo de limpieza, estandarización de esquemas y transformaciones (Capa Silver).
Contiene funciones puras que reciben DataFrames y objetos en crudo (Bronze) y
devuelven DataFrames estandarizados y tipados para la Capa Silver, aplicando
validaciones de rango, conservando valores nulos reales, estandarizando todas
las marcas temporales a Europe/Madrid y vinculando las estaciones a la red viaria.
"""

from typing import Any
import numpy as np
import osmnx as ox
import pandas as pd

from etl.utils import parse_dict_field, safe_local_timestamp


def _get_largest_strongly_connected_graph(graph: Any) -> Any:
    """
    Extrae el mayor componente fuertemente conexo del grafo viario dirigido
    (compatible tanto con OSMnx 2.x como con OSMnx 1.x), asegurando que exista
    camino de ida y vuelta entre todos los nodos mapeados.
    """
    if hasattr(ox, "truncate") and hasattr(ox.truncate, "largest_component"):
        return ox.truncate.largest_component(graph, strongly=True)
    return ox.utils_graph.get_largest_component(graph, strongly=True)


def clean_stations(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y estandariza el maestro de estaciones (stations_raw.json).
    - Renombra atributos a snake_case.
    - Castea tipos numéricos y timestamps con zona horaria Europe/Madrid.
    - Conserva nulos en variables de estado mediante Int64.
    - Aplica validaciones de rangos (capacity > 0 y 0 <= bikes/anchors <= capacity).
    """
    if df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()

    rename_dict = {
        "id": "station_id",
        "lat": "latitude",
        "lon": "longitude",
        "bikesAvailable": "bikes_available",
        "anchorsFree": "anchors_free",
    }
    df.rename(columns=rename_dict, inplace=True)

    if "recordedAt" in df.columns:
        df["recordedAt"] = safe_local_timestamp(df["recordedAt"])

    if "name" in df.columns:
        df["name"] = df["name"].astype(str).str.strip()

    if "station_id" in df.columns:
        df["station_id"] = pd.to_numeric(df["station_id"], errors="coerce").astype("Int64")

    for col in ["capacity", "bikes_available", "anchors_free"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    # Validaciones de rangos físicos
    if "capacity" in df.columns:
        df.loc[df["capacity"] <= 0, "capacity"] = pd.NA

    if "bikes_available" in df.columns and "capacity" in df.columns:
        invalid_bikes = (df["bikes_available"] < 0) | (df["bikes_available"] > df["capacity"])
        df.loc[invalid_bikes, "bikes_available"] = pd.NA

    if "anchors_free" in df.columns and "capacity" in df.columns:
        invalid_anchors = (df["anchors_free"] < 0) | (df["anchors_free"] > df["capacity"])
        df.loc[invalid_anchors, "anchors_free"] = pd.NA

    return df


def clean_rebalancing(df_raw: pd.DataFrame) -> pd.DataFrame:
    """Limpia el reporte de transferencias logísticas (rebalancing_raw.json)."""
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

    if "timeWindow" in df.columns:
        tw_parsed = df["timeWindow"].apply(parse_dict_field)
        df["time_window_start"] = safe_local_timestamp(
            tw_parsed.apply(lambda x: x.get("start", None))
        )
        df["time_window_end"] = safe_local_timestamp(
            tw_parsed.apply(lambda x: x.get("end", None))
        )
        df.drop(columns=["timeWindow"], errors="ignore", inplace=True)

    if "expectedImpact" in df.columns:
        ei_parsed = df["expectedImpact"].apply(parse_dict_field)
        df["empties_avoided"] = pd.to_numeric(
            ei_parsed.apply(lambda x: x.get("emptiesAvoided", None)), errors="coerce"
        ).astype("Int64")
        df["fulls_avoided"] = pd.to_numeric(
            ei_parsed.apply(lambda x: x.get("fullsAvoided", None)), errors="coerce"
        ).astype("Int64")
        df.drop(columns=["expectedImpact"], errors="ignore", inplace=True)

    if "reasons" in df.columns:
        df["reasons_clean"] = (
            df["reasons"].astype(str).str.replace(r"[\[\]']", "", regex=True).str.strip()
        )
        df.drop(columns=["reasons"], errors="ignore", inplace=True)

    for col in ["origin_station_id", "destination_station_id", "bikes_to_move"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    if "bikes_to_move" in df.columns:
        df.loc[df["bikes_to_move"] <= 0, "bikes_to_move"] = pd.NA

    return df


def clean_rankings(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia métricas operativas y horas de estrés (rankings_raw.json).
    - Conserva window_start, window_end y generated_at en Europe/Madrid.
    - Preserva valores nulos en métricas desconocidas (NO rellena con 0).
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
        "fullHourShare": "full_hour_share",
        "windowStart": "window_start",
        "windowEnd": "window_end",
        "generatedAt": "generated_at",
        "demandVsStressedHint": "demand_vs_stressed_hint",
    }
    df.rename(columns=rename_dict, inplace=True)

    if "station_id" in df.columns:
        df["station_id"] = pd.to_numeric(df["station_id"], errors="coerce").astype("Int64")

    metric_cols = [
        "turnover_score",
        "empty_hours",
        "full_hours",
        "total_hours",
        "problem_hours",
        "empty_hour_share",
        "full_hour_share",
    ]
    for col in metric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            df.loc[df[col] < 0, col] = np.nan

    for date_col in ["window_start", "window_end", "generated_at"]:
        if date_col in df.columns:
            df[date_col] = safe_local_timestamp(df[date_col])

    if "full_hour_share" not in df.columns and {"full_hours", "total_hours"}.issubset(df.columns):
        df["full_hour_share"] = np.where(
            df["total_hours"] > 0, (df["full_hours"] / df["total_hours"]).round(4), np.nan
        )
    if "problem_hour_share" not in df.columns and {"problem_hours", "total_hours"}.issubset(df.columns):
        df["problem_hour_share"] = np.where(
            df["total_hours"] > 0, (df["problem_hours"] / df["total_hours"]).round(4), np.nan
        )

    return df


def clean_patterns(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia los perfiles basales de ocupación horaria (patterns_raw.json).
    - Conserva generated_at y window_end en Europe/Madrid.
    - Valida rango de occupancy_avg entre 0.0 y 1.0 y limpia valores -0.0.
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
        "generatedAt": "generated_at",
        "windowEnd": "window_end",
    }
    df.rename(columns=rename_dict, inplace=True)

    for col in ["station_id", "hour", "sample_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in ["occupancy_avg", "bikes_avg", "anchors_avg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")
            df[col] = df[col].replace(-0.0, 0.0)

    if "occupancy_avg" in df.columns:
        df.loc[(df["occupancy_avg"] < 0.0) | (df["occupancy_avg"] > 1.0), "occupancy_avg"] = np.nan

    if "day_type" in df.columns:
        df["day_type"] = df["day_type"].astype(str).str.strip().str.upper()

    for date_col in ["generated_at", "window_end"]:
        if date_col in df.columns:
            df[date_col] = safe_local_timestamp(df[date_col])

    return df


def clean_history(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia el histórico macro diario de la ciudad (bizi_history_raw.csv).
    - Homogeneiza la columna date a la zona horaria Europe/Madrid.
    - Define window_end (cierre de la jornada diaria = date + 1 día si no viene
      explícito) y conserva generated_at si existe en el CSV.
    - Conserva nulos reales sin rellenar con cero.
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
        "windowStart": "window_start",
        "windowEnd": "window_end",
        "generatedAt": "generated_at",
        "updatedAt": "generated_at",
    }
    df.rename(columns=rename_dict, inplace=True)

    time_col = "date_str" if "date_str" in df.columns else df.columns[0]
    parsed_dates = pd.to_datetime(df[time_col], errors="coerce")
    if parsed_dates.dt.tz is None:
        df["date"] = parsed_dates.dt.tz_localize("Europe/Madrid").dt.floor("D")
    else:
        df["date"] = parsed_dates.dt.tz_convert("Europe/Madrid").dt.floor("D")

    df["date_key"] = df["date"].dt.strftime("%Y%m%d").astype("Int64")

    if "window_end" in df.columns:
        df["window_end"] = safe_local_timestamp(df["window_end"]).fillna(
            df["date"] + pd.Timedelta(days=1)
        )
    else:
        df["window_end"] = df["date"] + pd.Timedelta(days=1)

    if "generated_at" in df.columns:
        df["generated_at"] = safe_local_timestamp(df["generated_at"])

    for col in ["city_avg_occupancy", "balance_index"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").round(4).astype("float64")

    if "city_avg_occupancy" in df.columns:
        df.loc[
            (df["city_avg_occupancy"] < 0.0) | (df["city_avg_occupancy"] > 1.0),
            "city_avg_occupancy",
        ] = np.nan

    for col in ["demand_score", "sample_count"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    df.sort_values(by="date", inplace=True)
    df.drop_duplicates(subset=["date_key"], keep="last", inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def clean_historical_snapshots(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Limpia y estandariza los snapshots históricos de estaciones (.parquet).
    - Homogeneiza timestamp a Europe/Madrid.
    - Conserva nulos reales en variables de estado mediante Int64.
    - Aplica controles de rango físico y deduplicación por (station_id, timestamp).
    """
    if df_raw.empty:
        return pd.DataFrame()

    df = df_raw.copy()

    rename_dict = {
        "id": "station_id",
        "stationId": "station_id",
        "lat": "latitude",
        "lon": "longitude",
        "bikesAvailable": "bikes_available",
        "anchorsFree": "anchors_free",
    }
    df.rename(columns=rename_dict, inplace=True)

    time_col = "recordedAt" if "recordedAt" in df.columns else "timestamp"
    if time_col in df.columns:
        df["timestamp"] = safe_local_timestamp(df[time_col])
        if time_col != "timestamp":
            df.drop(columns=[time_col], inplace=True)

    if "station_id" in df.columns:
        df["station_id"] = pd.to_numeric(df["station_id"], errors="coerce").astype("Int64")

    for col in ["capacity", "bikes_available", "anchors_free"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    for col in ["latitude", "longitude"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("float64")

    if "capacity" in df.columns:
        df.loc[df["capacity"] <= 0, "capacity"] = pd.NA

    if "bikes_available" in df.columns and "capacity" in df.columns:
        invalid_bikes = (df["bikes_available"] < 0) | (df["bikes_available"] > df["capacity"])
        df.loc[invalid_bikes, "bikes_available"] = pd.NA

    if "anchors_free" in df.columns and "capacity" in df.columns:
        invalid_anchors = (df["anchors_free"] < 0) | (df["anchors_free"] > df["capacity"])
        df.loc[invalid_anchors, "anchors_free"] = pd.NA

    if "station_id" in df.columns and "timestamp" in df.columns:
        df.sort_values(by=["station_id", "timestamp"], inplace=True)
        df.drop_duplicates(subset=["station_id", "timestamp"], inplace=True)

    df.reset_index(drop=True, inplace=True)
    return df


def clean_road_network_nodes(df_stations_silver: pd.DataFrame, road_graph: Any) -> pd.DataFrame:
    """
    Construye la tabla intermedia Silver (road_network_nodes_silver.parquet)
    vinculando cada estación de Bizi con su nodo viario más cercano dentro del
    mayor componente fuertemente conexo de la red viaria de Zaragoza.
    """
    if df_stations_silver.empty or road_graph is None:
        return pd.DataFrame()

    required_cols = ["station_id", "latitude", "longitude"]
    if not set(required_cols).issubset(df_stations_silver.columns):
        raise ValueError(f"df_stations_silver debe contener las columnas {required_cols}")

    base_cols = ["station_id", "name", "latitude", "longitude"]
    existing_cols = [c for c in base_cols if c in df_stations_silver.columns]

    df_nodes = (
        df_stations_silver[existing_cols]
        .dropna(subset=["station_id", "latitude", "longitude"])
        .drop_duplicates(subset=["station_id"], keep="last")
        .sort_values("station_id")
        .reset_index(drop=True)
    )
    df_nodes["station_id"] = df_nodes["station_id"].astype("Int64")

    # Filtrar al mayor componente fuertemente conexo para evitar nodos aislados sin salida
    strong_graph = _get_largest_strongly_connected_graph(road_graph)

    # Búsqueda del nodo viario más cercano (X = longitud, Y = latitud)
    nearest_node_ids, snap_dists_deg = ox.distance.nearest_nodes(
        strong_graph,
        X=df_nodes["longitude"].to_numpy(),
        Y=df_nodes["latitude"].to_numpy(),
        return_dist=True,
    )

    df_nodes["osm_node_id"] = pd.Series(nearest_node_ids, index=df_nodes.index, dtype="int64")
    df_nodes["node_latitude"] = [
        float(strong_graph.nodes[nid]["y"]) for nid in df_nodes["osm_node_id"]
    ]
    df_nodes["node_longitude"] = [
        float(strong_graph.nodes[nid]["x"]) for nid in df_nodes["osm_node_id"]
    ]

    # Distancia aproximada en metros entre el tótem de la estación y la calzada asignada
    lat_rad = np.radians(df_nodes["latitude"].to_numpy())
    dlat_m = (df_nodes["node_latitude"].to_numpy() - df_nodes["latitude"].to_numpy()) * 111_320.0
    dlon_m = (
        (df_nodes["node_longitude"].to_numpy() - df_nodes["longitude"].to_numpy())
        * 111_320.0
        * np.cos(lat_rad)
    )
    df_nodes["snap_distance_m"] = np.hypot(dlat_m, dlon_m).round(2)

    return df_nodes