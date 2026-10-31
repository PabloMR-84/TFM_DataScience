from datetime import datetime, timezone
import os
import pandas as pd
import requests
from dotenv import load_dotenv

# Cargar variables de entorno desde el archivo .env si existe (para ejecución local)
load_dotenv()

API_URL = "https://datosbizi.com/api/stations"
API_KEY = os.getenv("BIZI_API_KEY")
OUTPUT_DIR = os.path.join("data", "raw", "historical")


def extract_data():
    # Configuración de cabeceras si se dispone de clave de API
    headers = {}
    if API_KEY:
        headers["Authorization"] = f"Bearer {API_KEY}"

    # 1. Petición a la API
    response = requests.get(API_URL, headers=headers, timeout=15)
    response.raise_for_status()
    payload = response.json()

    stations = payload.get("stations", [])
    if not stations:
        print("No se obtuvieron registros de la API.")
        return

    # 2. Convertir a DataFrame
    df = pd.DataFrame(stations)

    # 3. Tipado de fechas y timestamp UTC de extracción
    if "recordedAt" in df.columns:
        df["recordedAt"] = pd.to_datetime(df["recordedAt"])

    now_utc = datetime.now(timezone.utc)
    df["extractedAt"] = now_utc

    # 4. Crear carpeta de destino si no existe
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 5. Generar archivo Parquet comprimido (ej: bizi_20261031_120000.parquet)
    file_name = f"bizi_{now_utc.strftime('%Y%m%d_%H%M%S')}.parquet"
    file_path = os.path.join(OUTPUT_DIR, file_name)

    df.to_parquet(file_path, engine="pyarrow", compression="snappy", index=False)
    print(f"Éxito: Guardado {file_name} ({len(df)} estaciones).")


if __name__ == "__main__":
    extract_data()