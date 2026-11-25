# 🧩 Código Fuente Empaquetado (`src/`)

Este directorio aloja la lógica reutilizable del proyecto **SmartRebalance ZGZ**, organizada bajo el estándar ***src-layout*** y empaquetada mediante `pyproject.toml`.

El código de esta carpeta no está pensado para ejecutarse directamente como scripts sueltos, sino para ser importado de forma limpia desde `scripts/`, `notebooks/` o los *entry points* CLI una vez instalado el paquete en modo editable (`pip install -e .`).

## 🗂️ Estructura de Paquetes

```text
src/
├── etl/                    # Pipeline de ingeniería de datos (Bronze -> Silver -> Gold)
│   ├── __init__.py         # Exportaciones públicas del subpaquete etl
│   ├── config.py           # Rutas del proyecto, variables de entorno y endpoints API
│   ├── utils.py            # Utilidades de parseo, fechas (Europe/Madrid) y auditorías
│   ├── io.py               # Descarga resiliente/caché de la API y persistencia en disco
│   ├── cleaning.py         # Saneamiento, tipado y validaciones físicas (Capa Silver)
│   └── models_gold.py      # Modelo en estrella, festivos y tabla maestra anti-leakage (Capa Gold)
│
├── modeling/               # Pipeline de Machine Learning multi-horizonte (+1h, +2h, +3h)
│   ├── __init__.py         # Exportaciones públicas del subpaquete modeling
│   ├── features.py         # Ingeniería de características compartida y umbrales de ocupación
│   ├── train.py            # Partición con embargo, SHA-256, calibración en Val y métricas
│   └── predict.py          # Inferencia batch desacoplada y alertas binarias calibradas
│
└── README.md               # Documentación de la arquitectura de código fuente
```

---

## 📦 Detalle de los Subpaquetes

### 1. `etl` (Ingeniería de Datos)
Encargado de transformar las lecturas crudas de DatosBizi y los festivos oficiales en un modelo dimensional limpio y libre de fuga temporal:
* **`config.py`:** Centraliza la resolución de rutas (`ROOT_DIR`, `RAW_DIR`, `SILVER_DIR`, `GOLD_DIR`), la lectura de `BIZI_API_KEY` desde `.env` y la definición de los *endpoints* de la API.
* **`utils.py`:** Contiene utilidades de soporte para parseo seguro de JSON, conversión estricta de marcas temporales a `Europe/Madrid`, auditoría de cardinalidad en cruces (`auditar_merge`) y cálculo de cobertura del *scheduler* (`evaluar_cobertura_scheduler`).
* **`io.py`:** Gestiona la ingesta desde la API de DatosBizi (con respaldo automático en modo *offline* si no hay clave o conexión) y la lectura/escritura en CSV y Parquet.
* **`cleaning.py`:** Construye la Capa Silver aplicando estandarización a `snake_case`, preservación de nulos con `Int64`, validación de rangos físicos y deduplicación por `(station_id, timestamp)`.
* **`models_gold.py`:** Construye la Capa Gold (`dim_stations`, `dim_calendar`, `dim_patterns`, `fact_station_snapshots` y `df_gold`), integra los festivos de Nager.Date (`ES-AR`) y Zaragoza, aplica cruces `merge_asof` hacia atrás (`available_at <= timestamp`) y aísla los registros de *Cold Start*.

### 2. `modeling` (Machine Learning en Producción)
Encargado de la preparación de *features*, el entrenamiento trazable, la calibración de alertas y la inferencia *batch* del modelo campeón (**LightGBM**):
* **`features.py`:** Pipeline determinista compartido entre entrenamiento e inferencia para evitar *training-serving skew*. Define los umbrales reales de negocio (`OCCUPANCY_LOW_THRESHOLD = 0.20`, `OCCUPANCY_HIGH_THRESHOLD = 0.80`) y los umbrales de respaldo por horizonte (`SATURATION_ALERT_THRESHOLDS` y `get_saturation_alert_threshold`). Incluye preparación de observaciones reales (`prepare_observed_base`), construcción de *targets* multi-horizonte (`add_multihorizon_targets`), retardos estrictos y *proxy* de furgonetas (`add_lag_and_rolling_features`), perfil basal causal (`add_causal_historical_profile`), calendario cíclico (`add_calendar_and_horizon_features`) y alineación de categorías (`align_tree_features`).
* **`train.py`:** Calcula las huellas **SHA-256** de los datos de entrada (`compute_file_sha256`), aplica partición cronológica con embargo de $3\text{ horas}$ (`temporal_split_with_embargo`), entrena los tres modelos `LGBMRegressor` con los hiperparámetros optimizados, calibra dinámicamente sobre `df_val` los umbrales de alerta de saturación (`calibrate_saturation_threshold`), evalúa en Validación y Test frente al *baseline* Naive (`evaluate_predictions`, incluyendo `precision_full`, `recall_full`, `f1_full` y `f1_full_base_80`) y guarda los artefactos `.joblib` y `metadata.json` en `models/`.
* **`predict.py`:** Carga los modelos serializados, las categorías y los umbrales calibrados (`occupancy_thresholds`) guardados en `metadata.json`, reconstruye la matriz de características sobre `data/gold/` y exporta las predicciones continuas recortadas a $[0, 1]$ (`pred_lgbm_{h}h`) junto con las banderas binarias de alerta (`pred_empty_{h}h`, `pred_sat_{h}h`) en `data/predictions/predictions_lgbm.parquet`.

---

## 💻 Uso como Paquete Python

Tras ejecutar `pip install -e .` desde la raíz del proyecto (`TFM/`), cualquier módulo puede importarse directamente sin manipular `sys.path`:

```python
from etl.config import GOLD_DIR
from modeling.features import build_feature_matrix
from modeling.train import train_and_save_models
from modeling.predict import run_batch_inference
```