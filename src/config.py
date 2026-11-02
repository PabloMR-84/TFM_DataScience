"""
src/config.py
--------------
Módulo de configuración centralizada del proyecto SmartRebalance ZGZ.
Define rutas base de directorios (data/raw, data/silver, data/gold), credenciales 
de la API de Bizi Zaragoza y parámetros generales de ejecución.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# 1. Cargar variables de entorno desde el archivo .env ubicado en la raíz
load_dotenv()

# ==============================================================================
# 2. CONFIGURACIÓN DE SEGURIDAD Y ACCESO A LA API
# ==============================================================================
API_KEY: str = os.getenv("BIZI_API_KEY", "")
HEADERS: dict = {"x-api-key": API_KEY} if API_KEY else {}
BASE_URL: str = "https://datosbizi.com/api"

# ==============================================================================
# 3. GESTIÓN DE RUTAS Y DIRECTORIOS (MEDALLION ARCHITECTURE)
# ==============================================================================
# config.py ubicado en TFM/src/config.py -> ROOT_DIR apunta a TFM/
ROOT_DIR: Path = Path(__file__).resolve().parents[1]

# Rutas de almacenamiento de datos
DATA_DIR: Path = ROOT_DIR / "data"
RAW_DIR: Path = DATA_DIR / "raw"
HISTORICAL_RAW_DIR: Path = RAW_DIR / "historical"
SILVER_DIR: Path = DATA_DIR / "silver"
GOLD_DIR: Path = DATA_DIR / "gold"

# Asegurar la existencia física de los directorios locales en el sistema
for directory in [RAW_DIR, HISTORICAL_RAW_DIR, SILVER_DIR, GOLD_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 4. CONFIGURACIÓN DE ENDPOINTS PARA LA INGESTA (CAPA BRONZE)
# ==============================================================================
ENDPOINTS_CONFIG = [
    {
        "name": "stations",
        "url": f"{BASE_URL}/stations",
        "params": {},
        "filename": "stations_raw.json",
        "type": "json",
    },
    {
        "name": "rankings",
        "url": f"{BASE_URL}/rankings",
        "params": {"type": "turnover", "limit": 1000},
        "filename": "rankings_raw.json",
        "type": "json",
    },
    {
        "name": "rebalancing-report",
        "url": f"{BASE_URL}/rebalancing-report",
        "params": {"format": "json"},
        "filename": "rebalancing_raw.json",
        "type": "json",
    },
    {
        "name": "patterns",
        "url": f"{BASE_URL}/patterns",
        "params": {},  # Se pasa dinámicamente con stationId en io.py para el bucle
        "filename": "patterns_raw.json",
        "type": "json",
    },
    {
        "name": "history",
        "url": f"{BASE_URL}/history",
        "params": {"format": "csv"},
        "filename": "bizi_history_raw.csv",
        "type": "stream",
    },
]