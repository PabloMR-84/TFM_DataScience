"""
src/__init__.py
---------------
Paquete modular del proyecto SmartRebalance ZGZ.
"""

from src.config import ROOT_DIR, RAW_DIR, SILVER_DIR, GOLD_DIR
from src.io import (
    confirm_and_reset_output_dirs,
    fetch_all_raw_data,
    load_historical_parquets,
)
from src.cleaning import clean_stations, clean_historical_snapshots
from src.models_gold import build_gold_dataset
from src.utils import obtener_tipos_columnas, parse_dict_field

__all__ = [
    "ROOT_DIR",
    "RAW_DIR",
    "SILVER_DIR",
    "GOLD_DIR",
    "confirm_and_reset_output_dirs",
    "fetch_all_raw_data",
    "load_historical_parquets",
    "clean_stations",
    "clean_historical_snapshots",
    "build_gold_dataset",
    "obtener_tipos_columnas",
    "parse_dict_field",
]