from datetime import datetime, timezone
import os
import uuid
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://datosbizi.com/api/stations"
API_KEY = os.getenv("BIZI_API_KEY")
OUTPUT_DIR = os.path.join("data", "raw", "historical")


def _build_session() -> requests.Session:
    session = requests.Session()
    retry_strategy = Retry(
        total=4,
        backoff_factor=2,  # Espera 2s, 4s, 8s, 16s entre reintentos
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
        raise_on_status=False,
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def extract_data():
    headers = {
        "Accept": "application/json",
        "User-Agent": "SmartRebalance-ZGZ-ETL/1.0",
        "X-Request-Id": f"gh-actions-{uuid.uuid4()}",
    }
    if API_KEY:
        headers["x-api-key"] = API_KEY

    # 1. Petición resiliente a la API (4 reintentos + timeout de 25s)
    session = _build_session()
    try:
        response = session.get(API_URL, headers=headers, timeout=25)
        response.raise_for_status()
        payload = response.json()
    except requests.exceptions.RequestException as exc:
        print(f"[ERROR API] No se pudo conectar con {API_URL} tras reintentos: {exc}")
        raise SystemExit(1)

    stations = payload.get("stations", [])
    if not stations:
        print("[AVISO] La API respondió correctamente pero devolvió 0 estaciones.")
        return

    # 2. Convertir la nueva captura a DataFrame
    df_new = pd.DataFrame(stations)

    if "recordedAt" in df_new.columns:
        df_new["recordedAt"] = pd.to_datetime(df_new["recordedAt"])

    now_utc = datetime.now(timezone.utc)
    df_new["extractedAt"] = now_utc

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. Nombre del archivo diario (ej: bizi_20261120.parquet)
    today_str = now_utc.strftime("%Y%m%d")
    file_name = f"bizi_{today_str}.parquet"
    file_path = os.path.join(OUTPUT_DIR, file_name)
    tmp_path = f"{file_path}.tmp"

    # 4. Concatenar si ya existe el fichero de hoy
    if os.path.exists(file_path):
        df_existing = pd.read_parquet(file_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    # 5. Escritura atómica (evita corromper el parquet si hay corte a mitad)
    df_combined.to_parquet(
        tmp_path, engine="pyarrow", compression="snappy", index=False
    )
    os.replace(tmp_path, file_path)

    print(
        f"Éxito: Actualizado {file_name}. Filas acumuladas hoy: {len(df_combined)}."
    )


if __name__ == "__main__":
    extract_data()