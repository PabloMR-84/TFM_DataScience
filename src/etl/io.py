"""
src/etl/io.py
--------------
Módulo de entrada y salida de datos (I/O).
Gestiona la limpieza previa de carpetas de salida (Silver y Gold), la descarga e
ingesta desde la API (Capa Bronze) con control interactivo en bloque, la
sincronización independiente del histórico Parquet desde GitHub, preserva
metadatos temporales de raíz, reporta estaciones sin patrones en la API, carga
parquets históricos y guarda capas Silver y Gold.
"""

from typing import Dict, List, Optional
from pathlib import Path
from io import StringIO
import json
import shutil
import subprocess
import sys
import pandas as pd
import requests

from etl.config import (
    ROOT_DIR,
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
# 0. GESTIÓN DE LIMPIEZA, DESCARGAS API Y SINCRONIZACIÓN GITHUB (I/O)
# ==============================================================================

def confirm_and_reset_output_dirs(auto_confirm: bool = False) -> None:
    """
    Solicita confirmación al usuario antes de iniciar el pipeline, advirtiendo
    que se eliminarán todos los archivos existentes en data/silver y data/gold
    para volver a generarlos desde cero. Si la respuesta es negativa, cierra el programa.
    Si auto_confirm=True, omite la pregunta interactiva (útil para ejecución automatizada).
    Borra únicamente los archivos internos para evitar bloqueos de sincronización
    (WinError 5) en Windows o Google Drive.
    """
    if not auto_confirm:
        print(
            "\n[AVISO] Al ejecutar este pipeline se ELIMINARÁN todos los archivos "
            f"anteriores de '{SILVER_DIR}' y '{GOLD_DIR}' y se volverán a generar desde cero."
        )
        respuesta = input("¿Deseas continuar y ejecutar el código? [s/N]: ").strip().lower()

        if respuesta not in ["s", "si", "sí", "y", "yes"]:
            print("[CANCELADO] Ejecución cancelada por el usuario. No se ha modificado ningún archivo.")
            sys.exit(0)

    archivos_borrados = 0
    for folder in [SILVER_DIR, GOLD_DIR]:
        folder.mkdir(parents=True, exist_ok=True)
        for item in folder.iterdir():
            try:
                if item.is_file() or item.is_symlink():
                    item.unlink()
                    archivos_borrados += 1
                elif item.is_dir():
                    shutil.rmtree(item)
            except PermissionError as e:
                print(
                    f"[ERROR] No se pudo borrar '{item.name}' porque está abierto en otro programa "
                    "(Excel, VS Code, etc.). Ciérralo e inténtalo de nuevo."
                )
                raise e

    print(
        f" [OK] Limpieza completada ({archivos_borrados} archivos anteriores "
        f"eliminados en '{SILVER_DIR.name}/' y '{GOLD_DIR.name}/')."
    )


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


def sync_historical_from_github(
    sync_github: Optional[bool] = None,
    remote_branch: str = "origin/main",
) -> None:
    """
    Pregunta de forma independiente al menú de la API si se desea descargar
    el histórico más reciente de 'data/raw/historical/' desde GitHub sin alterar
    ni mezclar el resto de archivos locales del repositorio.
    - Si sync_github es True/False, ejecuta directamente sin preguntar.
    - Si sync_github es None, solicita confirmación interactiva [s/N].
    """
    if sync_github is None:
        respuesta = input(
            "\n¿Deseas actualizar 'data/raw/historical/' desde GitHub antes de cargar el histórico? [s/N]: "
        ).strip().lower()
        should_sync = respuesta in ["s", "si", "sí", "y", "yes"]
    else:
        should_sync = sync_github

    if not should_sync:
        print("[INFO] Se utilizarán los archivos históricos locales existentes en data/raw/historical/.")
        return

    print(f"[ACCIONAL] Sincronizando 'data/raw/historical/' desde {remote_branch}...")
    try:
        subprocess.run(
            ["git", "fetch", "origin"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
        subprocess.run(
            ["git", "checkout", remote_branch, "--", "data/raw/historical/"],
            cwd=ROOT_DIR,
            check=True,
            capture_output=True,
            text=True,
        )
        print(" [OK] Histórico Parquet sincronizado correctamente desde GitHub.")
    except subprocess.CalledProcessError as e:
        err_msg = e.stderr.strip() if e.stderr else str(e)
        print(
            f"[AVISO] No se pudo sincronizar 'data/raw/historical/' desde GitHub ({err_msg}). "
            "Se continuará con los archivos locales existentes."
        )


def _propagate_root_temporal_metadata(
    df: pd.DataFrame, payload: Dict, fallback_timestamp: Optional[str] = None
) -> pd.DataFrame:
    """
    Propaga metadatos temporales de la raíz de la respuesta JSON (generatedAt,
    windowStart, windowEnd o window.start / window.end) hacia el DataFrame
    si no vienen ya incluidos a nivel de fila.
    """
    if df.empty or not isinstance(payload, dict):
        return df

    window_obj = payload.get("window") if isinstance(payload.get("window"), dict) else {}

    meta_map = {
        "generatedAt": payload.get("generatedAt") or payload.get("timestamp") or fallback_timestamp,
        "windowStart": payload.get("windowStart") or window_obj.get("start"),
        "windowEnd": payload.get("windowEnd") or window_obj.get("end"),
    }

    for col_name, meta_val in meta_map.items():
        if meta_val is not None and col_name not in df.columns:
            df[col_name] = meta_val

    return df


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
    df = pd.DataFrame(stations_list)
    if isinstance(data, dict):
        df = _propagate_root_temporal_metadata(df, data)
    return df


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
    df = pd.DataFrame(rebal_list)
    if isinstance(data, dict):
        df = _propagate_root_temporal_metadata(df, data)
    return df


def get_rankings_raw(limit: int = 1000, replace_file: bool = False) -> pd.DataFrame:
    """
    Obtiene los rankings de rotación/fricción desde API o caché local (rankings_raw.json),
    conservando windowStart, windowEnd y generatedAt desde la respuesta de la API.
    """
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
    df = pd.DataFrame(rankings_list)
    if isinstance(data, dict):
        df = _propagate_root_temporal_metadata(df, data)
    return df


def get_patterns_raw(
    station_ids: List[int], limit_stations: Optional[int] = None, replace_file: bool = False
) -> pd.DataFrame:
    """
    Obtiene patrones horarios iterando sobre la lista de station_ids,
    conservando metadatos temporales y reportando qué estaciones no devuelven patrón.
    """
    file_path = RAW_DIR / RAW_FILES_MAP["patterns"]

    if file_path.exists() and not replace_file:
        with open(file_path, "r", encoding="utf-8") as f:
            raw_loaded = json.load(f)

        file_mtime_iso = pd.Timestamp(file_path.stat().st_mtime, unit="s", tz="Europe/Madrid").isoformat()
        if isinstance(raw_loaded, dict):
            patterns_list = raw_loaded.get("patterns", [])
            df = pd.DataFrame(patterns_list)
            return _propagate_root_temporal_metadata(df, raw_loaded, fallback_timestamp=file_mtime_iso)

        df = pd.DataFrame(raw_loaded)
        if not df.empty and "generatedAt" not in df.columns and "windowEnd" not in df.columns:
            df["generatedAt"] = file_mtime_iso
        return df

    all_patterns = []
    empty_pattern_stations: List[int] = []
    ids_to_fetch = station_ids[:limit_stations] if limit_stations else station_ids
    fetch_ts_iso = pd.Timestamp.now(tz="Europe/Madrid").isoformat()

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
                if isinstance(data_pat, dict):
                    patterns = data_pat.get("patterns", [])
                    gen_at = data_pat.get("generatedAt") or data_pat.get("timestamp") or fetch_ts_iso
                    window_obj = data_pat.get("window") if isinstance(data_pat.get("window"), dict) else {}
                    win_end = data_pat.get("windowEnd") or window_obj.get("end")

                    if not patterns:
                        empty_pattern_stations.append(int(s_id))
                    for item in patterns:
                        if isinstance(item, dict):
                            item.setdefault("generatedAt", gen_at)
                            if win_end is not None:
                                item.setdefault("windowEnd", win_end)
                    all_patterns.extend(patterns)
                else:
                    if not data_pat:
                        empty_pattern_stations.append(int(s_id))
                    for item in data_pat:
                        if isinstance(item, dict):
                            item.setdefault("generatedAt", fetch_ts_iso)
                    all_patterns.extend(data_pat)
            else:
                empty_pattern_stations.append(int(s_id))
        except Exception as e:
            empty_pattern_stations.append(int(s_id))
            print(f"[AVISO] Error al descargar patrón de estación {s_id}: {e}")

    if empty_pattern_stations:
        print(
            f"[AVISO API PATTERNS] {len(empty_pattern_stations)}/{len(ids_to_fetch)} estaciones "
            f"sin curva basal en /api/patterns: {sorted(empty_pattern_stations)}"
        )

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