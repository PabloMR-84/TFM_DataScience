"""
src/modeling/train.py

Módulo de entrenamiento del modelo campeón (LightGBM multi-horizonte) para
SmartRebalance ZGZ. Lee la capa Gold de producción (data/gold/), calcula el
hash SHA-256 para trazabilidad de datos, entrena los modelos (+1h, +2h, +3h),
calibra automáticamente los umbrales de alerta de saturación sobre Validación,
evalúa en Validación y Test y persiste los artefactos, metadata.json y el
historial acumulativo (training_history.jsonl) en models/.
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
    precision_score,
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
    get_saturation_alert_threshold,
    prepare_observed_base,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)

RANDOM_STATE: int = 42
CALIBRATION_MIN_THRESHOLD: float = 0.50
CALIBRATION_STEP: float = 0.01
HISTORY_FILENAME: str = "training_history.jsonl"

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


def calibrate_saturation_threshold(
    y_val: np.ndarray | pd.Series,
    y_pred_val: np.ndarray,
    horizon_h: int,
    min_threshold: float = CALIBRATION_MIN_THRESHOLD,
    max_threshold: float = OCCUPANCY_HIGH_THRESHOLD,
    step: float = CALIBRATION_STEP,
) -> float:
    """
    Calibra dinámicamente el umbral de alerta de saturación sobre Validación (df_val)
    maximizando el F1-Score del evento real (> 80 %) para evitar data leakage.
    Si Validación no contiene registros reales saturados, devuelve el umbral de respaldo.
    """
    y_val_bin = (np.asarray(y_val, dtype=float) > OCCUPANCY_HIGH_THRESHOLD).astype(int)
    if int(y_val_bin.sum()) == 0:
        fallback = get_saturation_alert_threshold(horizon_h)
        logger.warning(
            "Sin casos saturados (> %.2f) en Validación para +%dh; usando umbral de respaldo %.2f",
            OCCUPANCY_HIGH_THRESHOLD,
            horizon_h,
            fallback,
        )
        return fallback

    y_pred_clip = np.clip(np.asarray(y_pred_val, dtype=float), 0.0, 1.0)
    grid = np.round(np.arange(min_threshold, max_threshold + 1e-9, step), 3)

    best_threshold = max(
        grid,
        key=lambda th: f1_score(y_val_bin, (y_pred_clip > th).astype(int), zero_division=0),
    )
    return round(float(best_threshold), 4)


def evaluate_predictions(
    y_real: np.ndarray | pd.Series,
    y_pred: np.ndarray,
    pred_high_threshold: float = OCCUPANCY_HIGH_THRESHOLD,
) -> dict[str, float]:
    """
    Calcula métricas de regresión y métricas operativas de vaciado/saturación.

    La saturación real siempre se evalúa frente a OCCUPANCY_HIGH_THRESHOLD (0.80),
    mientras que la alerta predicha aplica pred_high_threshold (calibrado en df_val).
    También reporta f1_full_base_80 para auditar la ganancia frente al umbral fijo 0.80.
    """
    y_real_arr = np.asarray(y_real, dtype=float)
    y_pred_clip = np.clip(np.asarray(y_pred, dtype=float), 0.0, 1.0)

    real_empty = (y_real_arr < OCCUPANCY_LOW_THRESHOLD).astype(int)
    pred_empty = (y_pred_clip < OCCUPANCY_LOW_THRESHOLD).astype(int)

    real_full = (y_real_arr > OCCUPANCY_HIGH_THRESHOLD).astype(int)
    pred_full = (y_pred_clip > pred_high_threshold).astype(int)
    pred_full_base_80 = (y_pred_clip > OCCUPANCY_HIGH_THRESHOLD).astype(int)

    return {
        "mae": round(float(mean_absolute_error(y_real_arr, y_pred_clip)), 4),
        "rmse": round(float(root_mean_squared_error(y_real_arr, y_pred_clip)), 4),
        "r2": round(float(r2_score(y_real_arr, y_pred_clip)), 4),
        "f1_empty": round(float(f1_score(real_empty, pred_empty, zero_division=0)), 4),
        "recall_empty": round(float(recall_score(real_empty, pred_empty, zero_division=0)), 4),
        "precision_full": round(float(precision_score(real_full, pred_full, zero_division=0)), 4),
        "recall_full": round(float(recall_score(real_full, pred_full, zero_division=0)), 4),
        "f1_full": round(float(f1_score(real_full, pred_full, zero_division=0)), 4),
        "f1_full_base_80": round(float(f1_score(real_full, pred_full_base_80, zero_division=0)), 4),
        "saturation_alert_threshold": round(float(pred_high_threshold), 4),
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


def _build_history_entry(metadata: dict[str, Any]) -> dict[str, Any]:
    """Extrae un registro compacto de trazabilidad (SHA-256, dataset, umbrales y métricas)."""
    return {
        "run_id": metadata.get("run_id") or metadata.get("trained_at_utc"),
        "trained_at_utc": metadata.get("trained_at_utc"),
        "model_family": metadata.get("model_family"),
        "total_duration_sec": metadata.get("total_duration_sec"),
        "gold_dir": metadata.get("gold_dir"),
        "data_sha256": metadata.get("data_sha256"),
        "dataset_summary": metadata.get("dataset_summary"),
        "occupancy_thresholds": metadata.get("occupancy_thresholds"),
        "metrics": metadata.get("metrics"),
    }


def append_training_history(output_dir: Path, metadata: dict[str, Any]) -> Path:
    """
    Añade el entrenamiento actual al registro acumulativo training_history.jsonl.
    Si el historial aún no existe pero hay un metadata.json previo en output_dir,
    lo incorpora primero para conservar su SHA-256 y sus métricas antes de sobrescribirlo.
    """
    history_path = output_dir / HISTORY_FILENAME
    metadata_path = output_dir / "metadata.json"

    entries_to_append: list[dict[str, Any]] = []

    if not history_path.exists() and metadata_path.exists():
        try:
            with metadata_path.open("r", encoding="utf-8") as f:
                previous_metadata = json.load(f)
            if previous_metadata.get("trained_at_utc") != metadata.get("trained_at_utc"):
                entries_to_append.append(_build_history_entry(previous_metadata))
                logger.info(
                    "Rescatado entrenamiento previo (%s) en %s antes de sobrescribir metadata.json",
                    previous_metadata.get("trained_at_utc", "desconocido"),
                    HISTORY_FILENAME,
                )
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("No se pudo leer el metadata.json previo para el historial: %s", exc)

    entries_to_append.append(_build_history_entry(metadata))

    with history_path.open("a", encoding="utf-8") as f:
        for entry in entries_to_append:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return history_path


def train_and_save_models(
    gold_dir: Path,
    output_dir: Path,
    horizons: list[int] = HORIZONS_HOURS,
) -> dict[str, Any]:
    """
    Ejecuta el flujo completo de entrenamiento, calibración en Validación,
    evaluación y guardado de modelos LightGBM multi-horizonte, metadata.json
    y el historial acumulativo training_history.jsonl.
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
    alert_thresholds_by_horizon: dict[str, float] = {}
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

        # Calibración estricta en Validación (anti-leakage) del umbral de alerta de saturación
        alert_th = calibrate_saturation_threshold(y_va, pred_va, horizon_h=h)
        alert_thresholds_by_horizon[f"{h}h"] = alert_th

        val_metrics = evaluate_predictions(y_va, pred_va, pred_high_threshold=alert_th)
        test_metrics = evaluate_predictions(y_te, pred_te, pred_high_threshold=alert_th)
        naive_test_metrics = evaluate_predictions(
            y_te, pred_naive_te, pred_high_threshold=OCCUPANCY_HIGH_THRESHOLD
        )

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
            "[OK] +%dh | Val MAE=%.4f | Test MAE=%.4f (Naive=%.4f) | Test R2=%.4f | "
            "Test F1 Sat (th=%.2f)=%.4f [Base 0.80=%.4f, Prec=%.4f, Rec=%.4f]",
            h,
            val_metrics["mae"],
            test_metrics["mae"],
            naive_test_metrics["mae"],
            test_metrics["r2"],
            alert_th,
            test_metrics["f1_full"],
            test_metrics["f1_full_base_80"],
            test_metrics["precision_full"],
            test_metrics["recall_full"],
        )

    trained_at = datetime.now(timezone.utc)
    metadata = {
        "run_id": trained_at.strftime("%Y%m%dT%H%M%SZ"),
        "model_family": "LightGBMRegressor",
        "trained_at_utc": trained_at.isoformat(),
        "total_duration_sec": round(time.time() - t_start, 2),
        "random_state": RANDOM_STATE,
        "horizons_hours": horizons,
        "gold_dir": str(gold_dir),
        "history_file": HISTORY_FILENAME,
        "data_sha256": data_hashes,
        "occupancy_thresholds": {
            "low_real": OCCUPANCY_LOW_THRESHOLD,
            "high_real": OCCUPANCY_HIGH_THRESHOLD,
            "calibration_method": (
                f"val_f1_grid_search_{CALIBRATION_MIN_THRESHOLD:.2f}_"
                f"{OCCUPANCY_HIGH_THRESHOLD:.2f}_step_{CALIBRATION_STEP:.2f}"
            ),
            "saturation_alert_by_horizon": alert_thresholds_by_horizon,
        },
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

    # Primero añadimos al historial (rescatando el metadata.json previo si aún no existía historial)
    history_path = append_training_history(output_dir, metadata)

    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(
        "Entrenamiento completado. Artefactos en %s | Historial actualizado en %s",
        output_dir,
        history_path,
    )
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
        help="Directorio de salida para modelos, metadata.json e historial (por defecto: models/lgbm_v1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    train_and_save_models(gold_dir=args.gold_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()