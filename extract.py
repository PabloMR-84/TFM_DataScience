from datetime import datetime, timezone
import os
import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

API_URL = "https://datosbizi.com/api/stations"
API_KEY = os.getenv("BIZI_API_KEY")
OUTPUT_DIR = os.path.join("data", "raw", "historical")


def extract_data():
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

    # 2. Convertir la nueva captura a DataFrame
    df_new = pd.DataFrame(stations)

    if "recordedAt" in df_new.columns:
        df_new["recordedAt"] = pd.to_datetime(df_new["recordedAt"])

    now_utc = datetime.now(timezone.utc)
    df_new["extractedAt"] = now_utc

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. Nombre del archivo basado SOLO en la fecha (ej: bizi_20261031.parquet)
    today_str = now_utc.strftime("%Y%m%d")
    file_name = f"bizi_{today_str}.parquet"
    file_path = os.path.join(OUTPUT_DIR, file_name)

    # 4. Si el archivo del día ya existe, le concatenamos los datos nuevos
    if os.path.exists(file_path):
        df_existing = pd.read_parquet(file_path)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_new

    # 5. Guardar el archivo diario actualizado
    df_combined.to_parquet(
        file_path, engine="pyarrow", compression="snappy", index=False
    )
    print(
        f"Éxito: Actualizado {file_name}. Filas acumuladas hoy: {len(df_combined)}."
    )


if __name__ == "__main__":
    extract_data()