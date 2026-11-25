# 🤖 Artefactos de Modelos y Model Card (`models/`)

Este directorio almacena los modelos entrenados en producción, sus metadatos de trazabilidad y el historial acumulativo de entrenamientos para **SmartRebalance ZGZ**.

## 🗂️ Estructura de Directorios

```text
models/
├── lgbm_v1/                      # Versión actual del modelo campeón (LightGBM multi-horizonte)
│   ├── lgbm_1h.joblib            # [Ignorado en Git] Modelo serializado para horizonte +1h
│   ├── lgbm_2h.joblib            # [Ignorado en Git] Modelo serializado para horizonte +2h
│   ├── lgbm_3h.joblib            # [Ignorado en Git] Modelo serializado para horizonte +3h
│   ├── metadata.json             # [Versionado en Git] Último entrenamiento (SHA-256, umbrales y métricas)
│   └── training_history.jsonl    # [Versionado en Git] Historial acumulativo de entrenamientos (SHA-256 y métricas)
└── README.md                     # Documentación de artefactos y Model Card
```

> **Política de control de versiones:** Los ficheros binarios (`*.joblib`, `*.pkl`) están excluidos en `.gitignore` para mantener ligero el repositorio. Tanto `metadata.json` (estado activo del modelo) como `training_history.jsonl` (registro acumulativo de ejecuciones previas) **sí se versionan en Git**, garantizando la trazabilidad exacta entre los datos de entrenamiento (`data_sha256`), los umbrales calibrados, los hiperparámetros y las métricas obtenidas en cada reentrenamiento.

---

## 📋 Model Card — `lgbm_v1` (LightGBM Multi-Horizonte)

### 1. Descripción y Uso Previsto
* **Arquitectura:** Tres regresores independientes `LGBMRegressor` (LightGBM) con hiperparámetros optimizados mediante Optuna para los horizontes tácticos $h \in \{1\text{ h}, 2\text{ h}, 3\text{ h}\}$, complementados con **calibración automática *anti-leakage* de umbrales de alerta de saturación** sobre el conjunto de Validación (`df_val`).
* **Variable objetivo:** Tasa de ocupación física real futura por estación ($\hat{y}_{s, t+h} \in [0, 1]$).
* **Uso previsto en producto:**
  * **Horizontes $+2\text{ h}$ y $+3\text{ h}$ (planificación táctica):** Alimentar el motor de optimización de rutas de la flota de rebalanceo (6 furgonetas de 22 plazas en operación 24 h) combinando las predicciones continuas (`pred_lgbm_{h}h`) con las banderas de alerta de vaciado (`pred_empty_{h}h`) y saturación calibrada (`pred_sat_{h}h`).
  * **Horizonte $+1\text{ h}$ (referencia a corto plazo):** Complementar el estado real observado. Para intervenciones reactivas inmediatas, la ocupación actual ya concentra la mayor parte de la señal.
* **Tipo de inferencia:** *Batch* (ejecutada bajo demanda o por ciclo mediante `scripts/predict_model.py` o `smartrebalance-predict`).

### 2. Datos de Entrenamiento y Validación
* **Fuente:** Capa Gold de producción (`data/gold/df_gold.parquet`, `dim_stations.parquet`, `dim_calendar.parquet`), filtrando exclusivamente registros físicos reales (`is_estimated == 0`).
* **Cobertura de referencia (`lgbm_v1`):** 276 estaciones activas de Zaragoza durante 23 días naturales (`31 oct – 22 nov`), con partición cronológica estricta (`70 % Train / 15 % Validación / 15 % Test`) y ventanas de embargo de $3\text{ horas}$ entre bloques para evitar fuga de datos (*data leakage*).
* **Predictores:** 30 variables por horizonte (27 base de estado físico, retardos de $10\text{ min}$ a $7\text{ días}$, perfil basal causal, saltos logísticos y calendario cíclico, más 3 variables cíclicas de la hora objetivo $t + h$).

### 3. Métricas de Evaluación en Test (*Out-of-Time*)

Resultados consolidados sobre el *snapshot* de 23 días (`gold_snapshot_ml_20261122`), incluyendo el impacto de la calibración de umbrales de saturación documentada en `notebooks/EDA_prediccion_saturacion.ipynb`:

| Horizonte | MAE (LightGBM) | MAE (Baseline Naive) | RMSE | $R^2$ | F1 Vacía ($< 20\%$) | F1 Sat. Base ($> 80\%$) | Umbral Calibrado (`df_val`) | F1 Sat. Calibrada (`df_test`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$+1\text{ h}$** | **0,0938** | 0,0999 | 0,1335 | 0,7333 | 0,8464 | 0,3888 | `$74,0\%$` | **0,4985** |
| **$+2\text{ h}$** | **0,1190** | 0,1411 | 0,1624 | 0,6058 | 0,7921 | 0,2348 | `$70,0\%$` | **0,4018** |
| **$+3\text{ h}$** | **0,1318** | 0,1702 | 0,1755 | 0,5383 | 0,7573 | 0,1702 | `$65,0\%$` | **0,3694** |

### 4. Limitaciones Conocidas
* **Suavizado de picos en estaciones saturadas ($> 80\%$):** Debido al fuerte desbalanceo de la red (~4,8 % de registros saturados frente a ~50 % vacíos), el regresor entrenado bajo MAE tiende a subestimar los picos extremos de llenado. Esta limitación se mitiga operativamente calibrando automáticamente en `df_val` los umbrales de alerta por horizonte (sin alterar la predicción continua `pred_lgbm_{h}h`), tal como se valida en `notebooks/EDA_prediccion_saturacion.ipynb`.
* **Estacionalidad limitada:** El entrenamiento de referencia cubre 23 días de otoño; cambios estacionales (invierno, verano, festividades atípicas o meteorología extrema) requieren reentrenamiento periódico sobre la capa Gold actualizada.
* **Ruido por intervenciones manuales y lecturas *offline*:** Las cargas y descargas de las furgonetas generan saltos abruptos no atribuibles a la demanda de usuarios (parcialmente mitigados mediante variables *proxy* de saltos logísticos). Asimismo, existen episodios puntuales de lecturas *offline* (`bikes_available == 0` y `anchors_free == 0`), auditados en `notebooks/modelado_predictivo.ipynb` (Sección 1.3) y documentados como mejora futura del ETL en `docs/entregas/02_datos_y_preparacion.md`.

---

## 🔍 Trazabilidad, Historial Acumulativo y Regeneración

### 1. Metadatos del modelo activo (`metadata.json`)
Cada ejecución de entrenamiento guarda en `models/lgbm_v1/metadata.json`:
* `run_id` y `trained_at_utc`: Identificador único y marca temporal UTC del entrenamiento.
* `data_sha256`: Huella digital SHA-256 exacta de `df_gold.parquet`, `dim_stations.parquet` y `dim_calendar.parquet`.
* `occupancy_thresholds`: Umbrales reales de negocio (`low_real = 0.20`, `high_real = 0.80`), método de calibración y umbrales de alerta de saturación calibrados por horizonte (`saturation_alert_by_horizon`), consumidos directamente por `src/modeling/predict.py`.
* `dataset_summary`: Rango temporal, volumen de filas y número de estaciones.
* `station_categories`: Niveles categóricos exactos de `station_id_cat` para garantizar consistencia entre entrenamiento e inferencia.
* `hyperparameters` y `metrics`: Configuración exacta y métricas en Validación y Test frente al *baseline* Naive (`mae`, `rmse`, `r2`, `f1_empty`, `recall_empty`, `precision_full`, `recall_full`, `f1_full`, `f1_full_base_80` y `saturation_alert_threshold`).

### 2. Historial acumulativo (`training_history.jsonl`)
Al reentrenar sobre la misma carpeta (`models/lgbm_v1/`), los archivos `.joblib` y `metadata.json` se actualizan con la última ejecución, pero `src/modeling/train.py` **añade automáticamente una nueva línea** a `training_history.jsonl` (rescatando también el `metadata.json` anterior si el historial aún no existía). Así se conserva la evolución histórica de todos los `data_sha256`, umbrales calibrados y métricas.

Consulta rápida en Python:
```python
import pandas as pd

df_hist = pd.read_json("models/lgbm_v1/training_history.jsonl", lines=True)
```

### 3. Regeneración y versionado de artefactos
Para regenerar los binarios `.joblib` localmente a partir de `data/gold/` (actualizando `models/lgbm_v1/` y su historial):

```bash
python scripts/train_model.py
```

Si deseas aislar una nueva versión sin sobrescribir `lgbm_v1/`:

```bash
python scripts/train_model.py --output-dir models/lgbm_v2
```