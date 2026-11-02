"""
src/io.py
----------
Módulo de entrada y salida de datos (I/O).
Gestiona la descarga e ingesta desde la API (Capa Bronze) con control interactivo
en bloque (reutilización vs. reemplazo), la carga de parquets históricos y la
persistencia de DataFrames en las capas Silver y Gold.
"""

from typing import Dict, List, Optional
from pathlib import Path
from io import StringIO
import json
import pandas as pd
import requests

from src.config import (
    HEADERS,
    BASE_URL,
    RAW_DIR,
    HISTORICAL_RAW_DIR,
    SILVER_DIR,
    GOLD_DIR,
)


RAW_FILES_MAP = {
    "stations": "stations_raw.json",
    "rebalancing": "rebalancing_raw.json",
    "rankings": "rankings_raw.json",
    "patterns": "patterns_raw.json",
    "history": "bizi_history_raw.csv",
}


# ==============================================================================
# 0. GESTOR INTERACTIVO DE DESCARGAS EN BLOQUE (CAPA BRONZE)
# ==============================================================================

def resolve_download_mode(force_download: Optional[bool] = None) -> bool:
    """
    Comprueba la existencia de archivos descargados en data/raw/.
    - Si force_download es pasado explícitamente (True/False), lo respeta.
    - Si ya existen archivos y force_download es None, pregunta UNA SOLA VEZ
      al usuario en bloque si desea utilizarlos (por defecto) o reemplazarlos.
    - Si no existen archivos, descarga directamente (retorna True).
    """
    if force_download is not None:
        return force_download

    existing_files = [
        RAW_DIR / filename for filename in RAW_FILES_MAP.values() if (RAW_DIR / filename).exists()
    ]

    if not existing_files:
        print("[INFO] No se detectaron archivos previos en data/raw/. Descargando desde la API...")
        return True

    print(f"\n[DETECTADOS] Se encontraron {len(existing_files)}/5 archivos previos en '{RAW_DIR}'.")
    user_choice = input(
        "¿Deseas UTILIZAR los archivos locales existentes (U) o REEMPLAZARLOS descargando de nuevo desde la API (R)? [U/r]: "
    ).strip().lower()

    if user_choice in ["r", "reemplazar", "replace"]:
        print("[ACCIONAL] Opción seleccionada: Descargando y reemplazando archivos desde la API...")
        return True
    else:
        print("[ACCIONAL] Opción seleccionada: Utilizando archivos locales existentes.")
        return False


# ==============================================================================
# 1. FUNCIONES DE EXTRACCIÓN Y LECTURA (CAPA BRONZE)
# ==============================================================================

def get_stations_raw(replace_file: bool = False) -> pd.DataFrame:
    """Obtiene el maestro de estaciones desde API o caché local (stations_raw.json)."""
    file_path = RAW_DIR / RAW_FILES_MAP["stations"]

    if file_path.exists() and not replace_file:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        res = requests.get(f"{BASE_URL}/stations", headers=HEADERS, timeout=20)
        res.raise_for_status()
        data = res.json()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    stations_list = data.get("stations", data) if isinstance(data, dict) else data
    return pd.DataFrame(stations_list)


def get_rebalancing_raw(replace_file: bool = False) -> pd.DataFrame:
    """Obtiene el reporte de rebalanceo desde API o caché local (rebalancing_raw.json)."""
    file_path = RAW_DIR / RAW_FILES_MAP["rebalancing"]

    if file_path.exists() and not replace_file:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        res = requests.get(
            f"{BASE_URL}/rebalancing-report",
            headers=HEADERS,
            params={"format": "json"},
            timeout=20,
        )
        res.raise_for_status()
        data = res.json()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    rebal_list = data.get("transfers", data) if isinstance(data, dict) else data
    return pd.DataFrame(rebal_list)


def get_rankings_raw(limit: int = 1000, replace_file: bool = False) -> pd.DataFrame:
    """Obtiene los rankings de rotación/fricción desde API o caché local (rankings_raw.json)."""
    file_path = RAW_DIR / RAW_FILES_MAP["rankings"]

    if file_path.exists() and not replace_file:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        res = requests.get(
            f"{BASE_URL}/rankings",
            headers=HEADERS,
            params={"type": "turnover", "limit": limit},
            timeout=20,
        )
        res.raise_for_status()
        data = res.json()
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    rankings_list = data.get("rankings", data) if isinstance(data, dict) else data
    return pd.DataFrame(rankings_list)


def get_patterns_raw(
    station_ids: List[int], limit_stations: Optional[int] = None, replace_file: bool = False
) -> pd.DataFrame:
    """
    Obtiene patrones horarios iterando sobre la lista de station_ids.
    Si ya existe y replace_file es False, carga directamente del JSON local.
    """
    file_path = RAW_DIR / RAW_FILES_MAP["patterns"]

    if file_path.exists() and not replace_file:
        with open(file_path, "r", encoding="utf-8") as f:
            all_patterns = json.load(f)
    else:
        all_patterns = []
        ids_to_fetch = station_ids[:limit_stations] if limit_stations else station_ids

        for s_id in ids_to_fetch:
            try:
                res = requests.get(
                    f"{BASE_URL}/patterns",
                    headers=HEADERS,
                    params={"stationId": str(s_id)},
                    timeout=15,
                )
                if res.status_code == 200:
                    data_pat = res.json()
                    patterns = data_pat.get("patterns", data_pat) if isinstance(data_pat, dict) else data_pat
                    all_patterns.extend(patterns)
            except Exception as e:
                print(f"[AVISO] Error al descargar patrón de estación {s_id}: {e}")

        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(all_patterns, f, ensure_ascii=False, indent=4)

    return pd.DataFrame(all_patterns)


def get_history_raw(replace_file: bool = False) -> pd.DataFrame:
    """Obtiene el histórico macro diario desde API o caché local (bizi_history_raw.csv)."""
    file_path = RAW_DIR / RAW_FILES_MAP["history"]

    if file_path.exists() and not replace_file:
        return pd.read_csv(file_path)
    else:
        res = requests.get(
            f"{BASE_URL}/history",
            headers=HEADERS,
            params={"format": "csv"},
            timeout=30,
        )
        res.raise_for_status()

        with open(file_path, "w", encoding="utf-8") as f:
            f.write(res.text)

        return pd.read_csv(StringIO(res.text))


def fetch_all_raw_data(
    limit_patterns_stations: Optional[int] = None,
    force_download: Optional[bool] = None,
) -> Dict[str, pd.DataFrame]:
    """
    Orquestador de extracción Bronze:
    1. Resuelve en bloque si se reutiliza o reemplaza la descarga.
    2. Descarga o lee los 5 DataFrames raw de la API.
    """
    should_replace = resolve_download_mode(force_download)

    print("\n--- Carga/Descarga de fuentes Raw (Bronze) ---")
    df_stations = get_stations_raw(replace_file=should_replace)
    print(f" [OK] Stations Raw: {len(df_stations)} registros")

    df_rebal = get_rebalancing_raw(replace_file=should_replace)
    print(f" [OK] Rebalancing Raw: {len(df_rebal)} registros")

    df_rankings = get_rankings_raw(replace_file=should_replace)
    print(f" [OK] Rankings Raw: {len(df_rankings)} registros")

    station_ids = df_stations["id"].unique().tolist() if "id" in df_stations.columns else []
    df_patterns = get_patterns_raw(
        station_ids=station_ids,
        limit_stations=limit_patterns_stations,
        replace_file=should_replace,
    )
    print(f" [OK] Patterns Raw: {len(df_patterns)} registros")

    df_history = get_history_raw(replace_file=should_replace)
    print(f" [OK] History Raw: {len(df_history)} registros")

    return {
        "stations": df_stations,
        "rebalancing": df_rebal,
        "rankings": df_rankings,
        "patterns": df_patterns,
        "history": df_history,
    }


def load_historical_parquets(custom_dir: Optional[Path] = None) -> pd.DataFrame:
    """
    Carga de archivos .parquet históricos locales.
    NO se ve afectado por la comprobación interactiva de la API.
    """
    target_dir = custom_dir or HISTORICAL_RAW_DIR
    parquet_files = sorted(list(target_dir.glob("*.parquet")))

    if not parquet_files:
        raise FileNotFoundError(f"No se encontraron archivos .parquet en: {target_dir.resolve()}")

    dfs = [pd.read_parquet(f) for f in parquet_files]
    return pd.concat(dfs, ignore_index=True)


# ==============================================================================
# 2. FUNCIONES DE GUARDADO (CAPAS SILVER Y GOLD)
# ==============================================================================

def save_silver_df(df: pd.DataFrame, filename: str) -> Path:
    """Guarda un DataFrame procesado en data/silver/."""
    out_path = SILVER_DIR / filename
    if filename.endswith(".parquet"):
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    return out_path


def save_gold_df(df: pd.DataFrame, filename: str) -> Path:
    """Guarda una entidad de la Capa Gold en data/gold/."""
    out_path = GOLD_DIR / filename
    if filename.endswith(".parquet"):
        df.to_parquet(out_path, index=False)
    else:
        df.to_csv(out_path, index=False)
    return out_path