"""
src/utils.py
-------------
Módulo de funciones auxiliares genéricas para inspección de datos,
parseo de cadenas/JSON, formateo de marcas temporales y auditorías
de calidad e integridad de datos.
"""

from typing import Any, Dict, List, Optional
import json
import pandas as pd


def obtener_tipos_columnas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identifica y devuelve el tipo de dato y un valor de ejemplo por columna.
    """
    if df.empty:
        return pd.DataFrame(columns=["Columna", "Tipo_Dato", "Ejemplo"])

    return pd.DataFrame({
        "Columna": df.columns,
        "Tipo_Dato": [str(dtype) for dtype in df.dtypes],
        "Ejemplo": [df[col].iloc[0] if len(df) > 0 else None for col in df.columns],
    })


def parse_dict_field(val: Any) -> Dict:
    """
    Parsea de forma segura un campo que contiene un diccionario o una cadena JSON.
    """
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
    """
    Convierte una serie a numérico reemplazando errores/nulos por un valor por defecto.
    """
    return pd.to_numeric(series, errors="coerce").fillna(default_val)


def safe_local_timestamp(series: pd.Series, timezone: str = "Europe/Madrid") -> pd.Series:
    """
    Convierte una serie a datetime garantizando la zona horaria local de forma segura.
    """
    dt_series = pd.to_datetime(series, errors="coerce")
    if dt_series.dt.tz is None:
        return dt_series.dt.tz_localize("UTC").dt.tz_convert(timezone)
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
) -> Dict[str, Any]:
    """
    Comprueba la cardinalidad tras un cruce (merge) de DataFrames para detectar
    duplicaciones no deseadas o multiplicaciones inesperadas de filas (1:N o N:M).

    Returns:
        Dict con el reporte de cardinalidad (filas origen, filas destino, filas resultado).
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
        print(f"  - [ALERTA CARDINALIDAD] El merge generó {cambio_filas:,} filas adicionales.")
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
    frecuencia_minutos: int = 10,
    timestamp_col: str = "timestamp",
    station_id_col: str = "station_id",
) -> pd.DataFrame:
    """
    Evalúa la cobertura real de capturas tomadas por el scheduler frente
    a las capturas esperadas teóricas (ej. 6 capturas/hora para 10 min de frecuencia).

    Returns:
        DataFrame con métricas por día/estación de porcentaje de cobertura real.
    """
    if df_snapshots.empty or timestamp_col not in df_snapshots.columns:
        return pd.DataFrame()

    df = df_snapshots.copy()
    df[timestamp_col] = safe_local_timestamp(df[timestamp_col])
    df["fecha"] = df[timestamp_col].dt.date

    # Calcular capturas esperadas en 24h
    capturas_esperadas_dia = int((24 * 60) / frecuencia_minutos)

    resumen = (
        df.groupby(["fecha", station_id_col])[timestamp_col]
        .count()
        .reset_index()
        .rename(columns={timestamp_col: "capturas_reales"})
    )

    resumen["capturas_esperadas"] = capturas_esperadas_dia
    resumen["pct_cobertura"] = (
        (resumen["capturas_reales"] / capturas_esperadas_dia) * 100
    ).round(2)

    cobertura_media = resumen["pct_cobertura"].mean()
    print(f"\n--- Auditoría de Cobertura del Scheduler ({frecuencia_minutos} min) ---")
    print(f"  - Capturas esperadas por día/estación: {capturas_esperadas_dia}")
    print(f"  - Cobertura media global conseguida: {cobertura_media:.2f}%")

    return resumen