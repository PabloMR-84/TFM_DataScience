"""
src/utils.py
-------------
Módulo de funciones auxiliares genéricas para inspección de datos,
parseo de cadenas/JSON y formateo de marcas temporales.
"""

from typing import Any, Dict
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
    """Convierte una serie a datetime garantizando la zona horaria local de forma segura."""
    dt_series = pd.to_datetime(series, errors="coerce")
    if dt_series.dt.tz is None:
        return dt_series.dt.tz_localize("UTC").dt.tz_convert(timezone)
    return dt_series.dt.tz_convert(timezone)