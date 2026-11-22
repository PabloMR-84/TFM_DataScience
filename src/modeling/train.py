"""
src/modeling/train.py

Módulo de entrenamiento del modelo campeón (LightGBM multi-horizonte) para
SmartRebalance ZGZ. Lee la capa Gold de producción (data/gold/), calcula el
hash SHA-256 para trazabilidad de datos, entrena los modelos (+1h, +2h, +3h),
evalúa en Validación y Test y persiste los artefactos y metadatos en models/.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Any

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    mean_absolute_error,
    r2_score,
    recall_score,
    root_mean_squared_error,
)

from modeling.features import (
    HORIZONS_HOURS,
    OCCUPANCY_HIGH_THRESHOLD,
    OCCUPANCY_LOW_THRESHOLD,
    add_multihorizon_targets,
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

RANDOM_STATE: int = 42

# Hiperparámetros ganadores obtenidos mediante Optuna en la fase experimental
CHAMPION_LGBM_PARAMS: dict[int, dict[str, Any]] = {
    1: {
        "learning_rate": 0.032580045769112206,
        "num_leaves": 105,
        "max_depth": 12,
        "min_child_samples": 95,
        "subsample": 0.6883385751926184,
        "colsample_bytree": 0.6611367492367999,
        "reg_alpha": 0.6145982202528154,
        "reg_lambda": 0.006573856567096192,
    },
    2: {
        "learning_rate": 0.03361441955478077,
        "num_leaves": 123,
        "max_depth": 10,
        "min_child_samples": 80,
        "subsample": 0.7046065241548528,
        "colsample_bytree": 0.7045980821176709,
        "reg_alpha": 0.0017073967431528124,
        "reg_lambda": 2.9154431891537547,
    },
    3: {
        "learning_rate": 0.02015369312797273,
        "num_leaves": 110,
        "max_depth": 10,
        "min_child_samples": 93,
        "subsample": 0.919944621340081,
        "colsample_bytree": 0.6759156281069316,
        "reg_alpha": 0.02715581955282941,
        "reg_lambda": 0.0029072088906598446,
    },
}


def compute_file_sha256(file_path: Path, chunk_size: int = 8192) -> str:
    """Calcula la huella digital SHA-256 de un fichero para trazabilidad de datos."""
    sha256 = hashlib.sha256()
    with file_path.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            sha256.update(chunk)
    return sha256.hexdigest()


def evaluate_predictions(
    y_real: np.ndarray | pd.Series,
    y_pred: np.ndarray,
) -> dict[str, float]:
    """Calcula métricas de regresión y métricas operativas de vaciado/saturación."""
    y_real_arr = np.asarray(y_real, dtype=float)
    y_pred_clip = np.clip(np.asarray(y_pred, dtype=float), 0.0, 1.0)

    real_empty = (y_real_arr < OCCUPANCY_LOW_THRESHOLD).astype(int)
    pred_empty = (y_pred_clip < OCCUPANCY_LOW_THRESHOLD).astype(int)

    real_full = (y_real_arr > OCCUPANCY_HIGH_THRESHOLD).astype(int)
    pred_full = (y_pred_clip > OCCUPANCY_HIGH_THRESHOLD).astype(int)

    return {
        "mae": round(float(mean_absolute_error(y_real_arr, y_pred_clip)), 4),
        "rmse": round(float(root_mean_squared_error(y_real_arr, y_pred_clip)), 4),
        "r2": round(float(r2_score(y_real_arr, y_pred_clip)), 4),
        "f1_empty": round(float(f1_score(real_empty, pred_empty, zero_division=0)), 4),
        "recall_empty": round(float(recall_score(real_empty, pred_empty, zero_division=0)), 4),
        "f1_full": round(float(f1_score(real_full, pred_full, zero_division=0)), 4),
    }


def temporal_split_with_embargo(
    df_features: pd.DataFrame,
    horizons: list[int] = HORIZONS_HOURS,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Aplica una partición cronológica estricta (Train / Val / Test) sobre marcas
    temporales completas con ventana de embargo igual al horizonte máximo.
    """
    target_cols = [f"target_occ_{h}h" for h in horizons]
    df_clean = (
        df_features.dropna(subset=target_cols)
        .sort_values(["timestamp", "station_id"])
        .reset_index(drop=True)
    )

    unique_ts = np.sort(df_clean["timestamp"].unique())
    n_ts = len(unique_ts)

    embargo = pd.Timedelta(hours=max(horizons))
    train_cutoff_ts = pd.Timestamp(unique_ts[int(n_ts * train_ratio)])
    val_cutoff_ts = pd.Timestamp(unique_ts[int(n_ts * (train_ratio + val_ratio))])

    mask_train = df_clean["timestamp"] <= (train_cutoff_ts - embargo)
    mask_val = (df_clean["timestamp"] > train_cutoff_ts) & (
        df_clean["timestamp"] <= (val_cutoff_ts - embargo)
    )
    mask_test = df_clean["timestamp"] > val_cutoff_ts

    df_train = df_clean[mask_train].reset_index(drop=True)
    df_val = df_clean[mask_val].reset_index(drop=True)
    df_test = df_clean[mask_test].reset_index(drop=True)

    return df_train, df_val, df_test


def train_and_save_models(
    gold_dir: Path,
    output_dir: Path,
    horizons: list[int] = HORIZONS_HOURS,
) -> dict[str, Any]:
    """
    Ejecuta el flujo completo de entrenamiento, evaluación y guardado de modelos
    LightGBM multi-horizonte y sus metadatos de trazabilidad.
    """
    t_start = time.time()
    path_df_gold = gold_dir / "df_gold.parquet"
    path_dim_stations = gold_dir / "dim_stations.parquet"
    path_dim_calendar = gold_dir / "dim_calendar.parquet"

    for path in (path_df_gold, path_dim_stations, path_dim_calendar):
        if not path.exists():
            raise FileNotFoundError(f"No se encontró el fichero requerido: {path}")

    logger.info("Calculando huellas SHA-256 de los datos de entrada en %s...", gold_dir)
    data_hashes = {
        "df_gold.parquet": compute_file_sha256(path_df_gold),
        "dim_stations.parquet": compute_file_sha256(path_dim_stations),
        "dim_calendar.parquet": compute_file_sha256(path_dim_calendar),
    }

    logger.info("Cargando capa Gold y construyendo matriz de características...")
    df_gold = pd.read_parquet(path_df_gold)
    df_stations = pd.read_parquet(path_dim_stations)
    df_calendar = pd.read_parquet(path_dim_calendar)

    df_obs = prepare_observed_base(df_gold, df_stations)
    df_targets = add_multihorizon_targets(df_obs, horizons=horizons)
    df_features = build_feature_matrix(df_targets, horizons=horizons, df_cal=df_calendar)

    logger.info("Aplicando partición cronológica estricta con embargo de %dh...", max(horizons))
    df_train, df_val, df_test = temporal_split_with_embargo(df_features, horizons=horizons)
    logger.info(
        "Particiones -> Train: %d | Val: %d | Test: %d",
        len(df_train),
        len(df_val),
        len(df_test),
    )

    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_horizon: dict[str, Any] = {}
    artifacts_by_horizon: dict[str, str] = {}
    station_categories: list[int] | None = None

    for h in horizons:
        t_h = time.time()
        target_col = f"target_occ_{h}h"
        feat_cols = get_features_for_horizon(h)

        X_tr, station_categories = align_tree_features(df_train, feat_cols, station_categories)
        X_va, _ = align_tree_features(df_val, feat_cols, station_categories)
        X_te, _ = align_tree_features(df_test, feat_cols, station_categories)

        y_tr = df_train[target_col]
        y_va = df_val[target_col]
        y_te = df_test[target_col]

        params = {
            "objective": "regression",
            "metric": "mae",
            "n_estimators": 1000,
            "subsample_freq": 1,
            "subsample_for_bin": 50000,
            "random_state": RANDOM_STATE,
            "n_jobs": -1,
            "verbose": -1,
            **CHAMPION_LGBM_PARAMS[h],
        }

        logger.info("Entrenando LightGBM para horizonte +%dh...", h)
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_va, y_va)],
            callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)],
        )

        pred_va = model.predict(X_va)
        pred_te = model.predict(X_te)
        pred_naive_te = df_test["occupancy_pct"].to_numpy()

        val_metrics = evaluate_predictions(y_va, pred_va)
        test_metrics = evaluate_predictions(y_te, pred_te)
        naive_test_metrics = evaluate_predictions(y_te, pred_naive_te)

        model_filename = f"lgbm_{h}h.joblib"
        model_path = output_dir / model_filename
        joblib.dump(model, model_path)

        artifacts_by_horizon[f"{h}h"] = model_filename
        metrics_by_horizon[f"{h}h"] = {
            "best_iteration": int(model.best_iteration_),
            "train_time_sec": round(time.time() - t_h, 2),
            "validation": val_metrics,
            "test": test_metrics,
            "naive_baseline_test": naive_test_metrics,
        }

        logger.info(
            "[OK] +%dh | Val MAE=%.4f | Test MAE=%.4f (Naive=%.4f) | Test R2=%.4f",
            h,
            val_metrics["mae"],
            test_metrics["mae"],
            naive_test_metrics["mae"],
            test_metrics["r2"],
        )

    metadata = {
        "model_family": "LightGBMRegressor",
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "total_duration_sec": round(time.time() - t_start, 2),
        "random_state": RANDOM_STATE,
        "horizons_hours": horizons,
        "gold_dir": str(gold_dir),
        "data_sha256": data_hashes,
        "dataset_summary": {
            "total_rows_gold": int(len(df_gold)),
            "min_timestamp": str(df_obs["timestamp"].min()),
            "max_timestamp": str(df_obs["timestamp"].max()),
            "n_stations": int(df_obs["station_id"].nunique()),
            "train_rows": int(len(df_train)),
            "val_rows": int(len(df_val)),
            "test_rows": int(len(df_test)),
        },
        "station_categories": station_categories,
        "artifacts": artifacts_by_horizon,
        "hyperparameters": {f"{h}h": CHAMPION_LGBM_PARAMS[h] for h in horizons},
        "metrics": metrics_by_horizon,
    }

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info("Entrenamiento completado. Artefactos y metadatos guardados en %s", output_dir)
    return metadata


def parse_args() -> argparse.Namespace:
    root_dir = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(
        description="Entrena los modelos LightGBM multi-horizonte de SmartRebalance ZGZ."
    )
    parser.add_argument(
        "--gold-dir",
        type=Path,
        default=root_dir / "data" / "gold",
        help="Directorio de entrada con la capa Gold (por defecto: data/gold).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=root_dir / "models" / "lgbm_v1",
        help="Directorio de salida para modelos y metadata.json (por defecto: models/lgbm_v1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_and_save_models(gold_dir=args.gold_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()