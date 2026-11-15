"""
src/utils.py
-------------
Módulo de funciones auxiliares genéricas para inspección de datos,
parseo de cadenas/JSON, formateo de marcas temporales y auditorías
de calidad, cardinalidad y cobertura de datos.
"""

from typing import Any, Dict, Optional
import json
import pandas as pd


def obtener_tipos_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """Identifica y devuelve el tipo de dato y un valor de ejemplo por columna."""
    if df.empty:
        return pd.DataFrame(columns=["Columna", "Tipo_Dato", "Ejemplo"])

    return pd.DataFrame({
        "Columna": df.columns,
        "Tipo_Dato": [str(dtype) for dtype in df.dtypes],
        "Ejemplo": [df[col].iloc[0] if len(df) > 0 else None for col in df.columns],
    })


def parse_dict_field(val: Any) -> Dict:
    """Parsea de forma segura un campo que contiene un diccionario o una cadena JSON."""
    if pd.isna(val):
        return {}
    if isinstance(val, dict):
        return val
    try:
        valid_json = str(val).replace("'", '"')
        return json.loads(valid_json)
    except Exception:
        return {}


def safe_to_numeric(series: pd.Series, default_val: float = 0.0) -> pd.Series:
    """Convierte una serie a numérico reemplazando errores/nulos por un valor por defecto."""
    return pd.to_numeric(series, errors="coerce").fillna(default_val)


def safe_local_timestamp(series: pd.Series, timezone: str = "Europe/Madrid") -> pd.Series:
    """
    Convierte una serie a datetime garantizando la zona horaria local de forma segura.
    Elimina nombres de zona horaria entre paréntesis antes del parseo y unifica en UTC
    antes de convertir a Europe/Madrid.
    """
    if series.empty:
        return pd.Series(dtype=f"datetime64[ns, {timezone}]")

    if pd.api.types.is_datetime64_any_dtype(series):
        dt_series = pd.to_datetime(series, errors="coerce", utc=True)
    else:
        cleaned = (
            series.astype(str)
            .replace({"None": pd.NA, "nan": pd.NA, "NaT": pd.NA, "": pd.NA})
            .str.replace(r"\s*\([^)]*\)", "", regex=True)
        )
        dt_series = pd.to_datetime(cleaned, errors="coerce", utc=True)

    return dt_series.dt.tz_convert(timezone)


# ==============================================================================
# AUDITORÍA DE INTEGRIDAD Y COBERTURA DE DATOS
# ==============================================================================

def auditar_merge(
    df_izq: pd.DataFrame,
    df_der: pd.DataFrame,
    df_resultado: pd.DataFrame,
    on: Optional[Any] = None,
    nombre_union: str = "Merge",
    strict: bool = True,
) -> Dict[str, Any]:
    """
    Comprueba la cardinalidad tras un cruce (merge) de DataFrames.
    Si strict=True, lanza ValueError e impide guardar datos si se multiplicaron filas.
    """
    filas_izq = len(df_izq)
    filas_der = len(df_der)
    filas_res = len(df_resultado)

    cambio_filas = filas_res - filas_izq
    es_multiplicado = filas_res > filas_izq

    print(f"\n--- Auditoría de Integridad: [{nombre_union}] ---")
    print(f"  - Filas DataFrame izquierdo: {filas_izq:,}")
    print(f"  - Filas DataFrame derecho  : {filas_der:,}")
    print(f"  - Filas DataFrame resultado: {filas_res:,}")

    if es_multiplicado:
        msg = (
            f"[ERROR CARDINALIDAD] La unión '{nombre_union}' multiplicó "
            f"{cambio_filas:,} filas adicionales ({filas_izq:,} -> {filas_res:,})."
        )
        print(f"  - {msg}")
        if strict:
            raise ValueError(msg)
    else:
        print(f"  - [OK] Cardinalidad conservada exactamente ({filas_res:,} filas).")

    return {
        "nombre_union": nombre_union,
        "filas_izq": filas_izq,
        "filas_der": filas_der,
        "filas_resultado": filas_res,
        "se_multiplico": es_multiplicado,
    }


def evaluar_cobertura_scheduler(
    df_snapshots: pd.DataFrame,
    df_stations: Optional[pd.DataFrame] = None,
    df_calendar: Optional[pd.DataFrame] = None,
    frecuencia_minutos: int = 10,
    timestamp_col: str = "timestamp",
    station_id_col: str = "station_id",
) -> pd.DataFrame:
    """
    Evalúa la cobertura real de capturas tomadas por el scheduler frente a las
    capturas esperadas teóricas (144 capturas/día para frecuencia de 10 min)
    acotada al periodo operativo real del collector y deduplicando por ventana
    de 10 minutos para evitar porcentajes > 100%.
    """
    if df_snapshots.empty or timestamp_col not in df_snapshots.columns:
        return pd.DataFrame()

    df = df_snapshots.copy()
    df[timestamp_col] = safe_local_timestamp(df[timestamp_col])
    df = df.dropna(subset=[timestamp_col, station_id_col])

    # Redondear a franja de frecuencia para contar slots únicos (máx 144/día)
    df["slot"] = df[timestamp_col].dt.floor(f"{frecuencia_minutos}min")
    df["fecha"] = df["slot"].dt.floor("D")

    capturas_esperadas_dia = int((24 * 60) / frecuencia_minutos)

    conteos = (
        df.groupby(["fecha", station_id_col], as_index=False)["slot"]
        .nunique()
        .rename(columns={"slot": "capturas_reales"})
    )

    if df_stations is not None and not df_stations.empty and station_id_col in df_stations.columns:
        estaciones_df = df_stations[[station_id_col]].dropna().drop_duplicates()
    else:
        estaciones_df = df[[station_id_col]].dropna().drop_duplicates()

    # Acotar al periodo operativo real del scheduler (evita arrastrar meses previos del CSV macro)
    min_f, max_f = df["fecha"].min(), df["fecha"].max()
    if df_calendar is not None and not df_calendar.empty and "date" in df_calendar.columns:
        cal_dates = safe_local_timestamp(df_calendar["date"]).dt.floor("D").dropna().unique()
        cal_active = [d for d in cal_dates if min_f <= d <= max_f]
        fechas_df = pd.DataFrame({"fecha": sorted(cal_active)})
    else:
        fechas_df = pd.DataFrame({"fecha": pd.date_range(min_f, max_f, freq="D", tz="Europe/Madrid")})

    rejilla = fechas_df.merge(estaciones_df, how="cross")
    resumen = rejilla.merge(
        conteos,
        on=["fecha", station_id_col],
        how="left",
        validate="1:1",
    )

    resumen["capturas_reales"] = resumen["capturas_reales"].fillna(0).astype("Int64")
    resumen["capturas_esperadas"] = capturas_esperadas_dia
    resumen["pct_cobertura"] = (
        (resumen["capturas_reales"] / capturas_esperadas_dia) * 100
    ).clip(0.0, 100.0).round(2)
    resumen["sin_capturas"] = (resumen["capturas_reales"] == 0).astype("Int64")

    cobertura_media = resumen["pct_cobertura"].mean()
    dias_estacion_vacios = int(resumen["sin_capturas"].sum())

    print(f"\n--- Auditoría de Cobertura del Scheduler ({frecuencia_minutos} min) ---")
    print(f"  - Periodo operativo evaluado                : {min_f.strftime('%Y-%m-%d')} a {max_f.strftime('%Y-%m-%d')} ({len(fechas_df)} días)")
    print(f"  - Total combinaciones estación-día evaluadas: {len(resumen):,}")
    print(f"  - Combinaciones estación-día sin capturas   : {dias_estacion_vacios:,}")
    print(f"  - Capturas esperadas por día/estación       : {capturas_esperadas_dia}")
    print(f"  - Cobertura media operativa                 : {cobertura_media:.2f}%")

    return resumen.sort_values(["fecha", station_id_col]).reset_index(drop=True)