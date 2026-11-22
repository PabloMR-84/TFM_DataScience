"""
src/modeling/predict.py

Módulo de inferencia batch desacoplado para SmartRebalance ZGZ.
Carga los modelos LightGBM serializados y metadata.json desde models/,
genera las características sobre data/gold/ y exporta las predicciones
multi-horizonte (+1h, +2h, +3h) para el motor de optimización de rutas.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
import time

import joblib
import numpy as np
import pandas as pd

from modeling.features import (
    align_tree_features,
    build_feature_matrix,
    get_features_for_horizon,
    prepare_observed_base,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def run_batch_inference(
    gold_dir: Path,
    model_dir: Path,
    output_path: Path,
    latest_only: bool = False,
    last_hours: int | None = 72,
) -> pd.DataFrame:
    """
    Ejecuta inferencia batch multi-horizonte usando los artefactos entrenados.

    Args:
        gold_dir: Carpeta con df_gold.parquet, dim_stations.parquet y dim_calendar.parquet.
        model_dir: Carpeta con lgbm_{h}h.joblib y metadata.json.
        output_path: Ruta destino del fichero .parquet con las predicciones.
        latest_only: Si es True, predice únicamente el timestamp más reciente disponible.
        last_hours: Si latest_only es False y se indica un entero, exporta las últimas N horas.
                    Si es None, exporta todo el rango observado.
    """
    t_start = time.time()
    metadata_path = model_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(
            f"No se encontró metadata.json en {model_dir}. Ejecuta primero el entrenamiento."
        )

    with metadata_path.open("r", encoding="utf-8") as f:
        metadata = json.load(f)

    horizons: list[int] = [int(h) for h in metadata["horizons_hours"]]
    station_categories: list[int] = [int(c) for c in metadata["station_categories"]]

    path_df_gold = gold_dir / "df_gold.parquet"
    path_dim_stations = gold_dir / "dim_stations.parquet"
    path_dim_calendar = gold_dir / "dim_calendar.parquet"

    for path in (path_df_gold, path_dim_stations, path_dim_calendar):
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el fichero requerido: {path}")

    logger.info("Cargando capa Gold desde %s...", gold_dir)
    df_gold = pd.read_parquet(path_df_gold)
    df_stations = pd.read_parquet(path_dim_stations)
    df_calendar = pd.read_parquet(path_dim_calendar)

    # Construimos features sobre toda la historia disponible para no perder lags (24h / 7d / perfil causal)
    logger.info("Construyendo matriz de características para inferencia...")
    df_obs = prepare_observed_base(df_gold, df_stations)
    df_features = build_feature_matrix(df_obs, horizons=horizons, df_cal=df_calendar)

    # Filtramos la ventana de salida tras haber calculado todos los retardos históricos
    max_ts = df_features["timestamp"].max()
    if latest_only:
        df_infer = df_features[df_features["timestamp"] == max_ts].copy()
        logger.info("Modo --latest-only: inferencia sobre timestamp %s (%d filas)", max_ts, len(df_infer))
    elif last_hours is not None:
        cutoff_ts = max_ts - pd.Timedelta(hours=last_hours)
        df_infer = df_features[df_features["timestamp"] >= cutoff_ts].copy()
        logger.info(
            "Ventana de salida: últimas %dh (%s -> %s | %d filas)",
            last_hours,
            df_infer["timestamp"].min(),
            max_ts,
            len(df_infer),
        )
    else:
        df_infer = df_features.copy()
        logger.info("Ventana de salida: histórico completo (%d filas)", len(df_infer))

    df_infer = df_infer.sort_values(["timestamp", "station_id"]).reset_index(drop=True)

    # Contrato de salida idéntico al consumido por el optimizador de rutas
    base_output_cols = [
        "station_id",
        "timestamp",
        "hour",
        "day_type_effective",
        "occupancy_pct",
        "bikes_available",
        "anchors_free",
    ]
    df_out = df_infer[base_output_cols].copy()

    for h in horizons:
        model_filename = metadata["artifacts"][f"{h}h"]
        model_path = model_dir / model_filename
        if not model_path.exists():
            raise FileNotFoundError(f"No se encontró el artefacto del modelo: {model_path}")

        logger.info("Cargando %s y prediciendo horizonte +%dh...", model_filename, h)
        model = joblib.load(model_path)
        feat_cols = get_features_for_horizon(h)
        X_infer, _ = align_tree_features(df_infer, feat_cols, station_categories)

        preds = np.clip(model.predict(X_infer), 0.0, 1.0)
        df_out[f"pred_lgbm_{h}h"] = preds.astype(np.float64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_parquet(output_path, index=False)

    logger.info(
        "[OK] Inferencia completada en %.2fs | Archivo: %s | Filas: %d | Estaciones: %d",
        time.time() - t_start,
        output_path,
        len(df_out),
        df_out["station_id"].nunique(),
    )
    return df_out


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Genera predicciones multi-horizonte de ocupación para SmartRebalance ZGZ."
    )
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=root_dir / "data" / "gold",
        help="Directorio de entrada con la capa Gold (por defecto: data/gold).",
    )
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=root_dir / "models" / "lgbm_v1",
        help="Directorio con los modelos entrenados y metadata.json (por defecto: models/lgbm_v1).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=root_dir / "data" / "predictions" / "predictions_lgbm.parquet",
        help="Ruta de salida del parquet de predicciones.",
    )
    parser.add_argument(
        "--latest-only",
        action="store_true",
        help="Si se activa, genera predicciones solo para el instante más reciente.",
    )
    parser.add_argument(
        "--last-hours",
        type=int,
        default=72,
        help="Número de horas recientes a exportar (ignorado si se usa --latest-only; 0 para todo).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    hours_window = None if args.last_hours <= 0 else args.last_hours
    run_batch_inference(
        gold_dir=args.gold_dir,
        model_dir=args.model_dir,
        output_path=args.output_path,
        latest_only=args.latest_only,
        last_hours=hours_window,
    )


if __name__ == "__main__":
    main()