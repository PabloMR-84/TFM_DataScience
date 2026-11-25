# Changelog

Todos los cambios notables en este proyecto serán documentados en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/),
y este proyecto adhiere a [Semantic Versioning](https://semver.org/lang/es/).


## [Unreleased]

## [0.3.3] - 2026-11-25

### Added
- **Estudio monográfico de saturación (`notebooks/EDA_prediccion_saturacion.ipynb`):** Análisis detallado del comportamiento del regresor campeón (`LightGBM Optuna`) ante episodios de llenado extremo ($> 80\%$), incluyendo rankings Top 10 Real vs. Predicho, auditoría visual y en minutos sobre Top 5 Real y Top 5 Predicho exclusivo, cuantificación del sesgo de subestimación (`-20,24 pp` / `-28,61 pp` / `-33,31 pp`), calibración *anti-leakage* de umbrales en Validación (*Threshold Tuning*) y contraste frente a un clasificador binario auxiliar (`LGBMClassifier`).
- **Calibración dinámica de umbrales de saturación y métricas ampliadas (`src/modeling/train.py`, `src/modeling/features.py`):** Incorporación de `calibrate_saturation_threshold()` para optimizar sobre `df_val` el umbral de alerta por horizonte (`0,74` en $+1\text{ h}$, `0,70` en $+2\text{ h}$, `0,65` en $+3\text{ h}$), constantes de respaldo `SATURATION_ALERT_THRESHOLDS` y registro de `precision_full`, `recall_full`, `f1_full` y `f1_full_base_80` en `models/lgbm_v1/metadata.json` (elevando `F1 Saturada` en Test a `0,4985` / `0,4018` / `0,3694`).
- **Historial acumulativo de entrenamientos (`models/lgbm_v1/training_history.jsonl` y `src/modeling/train.py`):** Registro automático en formato JSON Lines (`append_training_history()`) que conserva las huellas `data_sha256`, umbrales calibrados y métricas de ejecuciones previas sin perder trazabilidad al reentrenar sobre `models/lgbm_v1/`.
- **Banderas binarias de alerta en inferencia *batch* (`src/modeling/predict.py`):** Lectura automática de los umbrales calibrados desde `metadata.json` y exportación de las columnas `pred_empty_{h}h` y `pred_sat_{h}h` junto con las predicciones continuas `pred_lgbm_{h}h` en `data/predictions/predictions_lgbm.parquet`.
- **Auditoría global de lecturas *offline* (`notebooks/modelado_predictivo.ipynb` y `docs/entregas/02_datos_y_preparacion.md`):** Incorporación de la celda `1.3` para auditar episodios con `bikes_available == 0` y `anchors_free == 0`, dejando documentada su gestión futura en ETL/modelado sin alterar el *snapshot* congelado.

### Changed
- **Renombrado del orquestador de inferencia (`scripts/predict.py` → `scripts/predict_model.py`):** Estandarización de nomenclatura junto a `scripts/train_model.py` y actualización de las exportaciones públicas en `src/modeling/__init__.py`.
- **Actualización de entregables y documentación modular (`docs/entregas/03_machine_learning.md`, `README.md`, `models/README.md`, `notebooks/README.md`, `scripts/README.md`, `src/README.md`, `notebooks/modelado_predictivo.ipynb`):** Sincronización de la Entrega 03 al *snapshot* de 23 días (`gold_snapshot_ml_20261122`), referencias cruzadas al análisis de saturación, *Model Card* actualizada con métricas calibradas y documentación de `training_history.jsonl` y `scripts/predict_model.py`.
- **Empaquetado y control de versiones (`pyproject.toml`, `.gitignore`):** Actualización de versión a `0.3.3` y ajuste de comentario sobre el versionado de `metadata.json` y `training_history.jsonl`.


## [0.3.2] - 2026-11-22

### Added
- **Pipeline de modelado predictivo en producción (`src/modeling/`):** Implementación modular y desacoplada de ingeniería de características (`features.py`), entrenamiento multi-horizonte LightGBM con hiperparámetros optimizados (`train.py`) e inferencia *batch* (`predict.py`), con trazabilidad de datos mediante huella **SHA-256** registrada en `models/lgbm_v1/metadata.json`.
- **Orquestadores y *entry points* CLI (`scripts/train_model.py`, `scripts/predict.py`, `pyproject.toml`):** Scripts ejecutables desacoplados para entrenamiento e inferencia y comandos CLI registrados (`smartrebalance-train`, `smartrebalance-predict`).
- **Documentación modular y *Model Card* (`README.md`, `data/README.md`, `notebooks/README.md`, `models/README.md`, `scripts/README.md`, `src/README.md`):** Guía general simplificada en la raíz y documentación técnica detallada por carpeta, incluyendo la *Model Card* del modelo campeón.
- **Instantánea congelada extendida (`data/gold_snapshot_ml_20261122/`):** Nueva instantánea de 23 días (`31/10/2026` – `22/11/2026`, `782.736` registros reales) rastreada mediante `.gitkeep` para aislar la reproducibilidad de `notebooks/` frente a `data/gold/`.
- **Sincronización independiente del histórico Parquet desde GitHub (`src/etl/io.py` y `scripts/main_BBDD.py`):** Nueva función `sync_historical_from_github()` con confirmación separada (`[s/N]`) respecto a la descarga de la API (`[U/r]`) y soporte para ejecución desatendida (`auto_confirm`, `force_download`, `sync_github`).
- **Precomputación de red viaria urbana (`src/etl/config.py`, `src/etl/models_gold.py` y `pyproject.toml`):** Integración de grafo viario de Zaragoza mediante `osmnx` para generar `dim_station_distances.parquet` con distancias y tiempos reales entre estaciones.
- **Documentación de cobertura y cierre de Entrega 02 (`docs/entregas/02_datos_y_preparacion.md` y `data/README.md`):** Incorporación de la auditoría temporal de capturas (`coverage_report`), diagnóstico de tramos operativos, criterios de selección de jornadas e instrucciones de acceso local sin clave de API.

### Changed
- **Reevaluación experimental con 23 días (`notebooks/modelado_predictivo.ipynb`):** Reentrenamiento sobre `gold_snapshot_ml_20261122`, promoción de **LightGBM (Optuna)** como modelo campeón (`Test MAE = 0,0938 / 0,1190 / 0,1318`), incorporación de la decisión práctica de producto (`+1h` vs. `+2h/+3h`) y caso de estudio visual de estación saturada (`10.4`).
- **Empaquetado y control de versiones (`pyproject.toml`, `requirements.txt`, `.gitignore`):** Adición de dependencias de producción (`scikit-learn`, `lightgbm`, `joblib`), grupo opcional `[notebooks]` y exclusión de binarios de modelos (`*.joblib`, `*.pkl`) y `data/predictions/`, manteniendo versionado `metadata.json`.
- **Aligeramiento del esquema de `df_gold.parquet` (`src/etl/models_gold.py`):** Eliminación de atributos estáticos redundantes de estación (`name`, `latitude`, `longitude` y `capacity` tras calcular `occupancy_pct`), reduciendo el dataset analítico de 25 a 22 columnas.
- **Refinamiento de limpieza y modelado Gold (`src/etl/cleaning.py` y `src/etl/models_gold.py`):** Ajustes menores de consistencia tras la cuarta revisión de la Capa Gold.

### Removed
- **Instantánea obsoleta (`data/gold_snapshot_ml_20261115/.gitkeep`):** Retirada tras su sustitución por el *snapshot* consolidado de 23 días (`gold_snapshot_ml_20261122/`).

### Fixed
- **Pipeline de extracción automática (`extract.py` y `.github/workflows/extract.yml`):** Incorporación de política de reintentos con *backoff* exponencial (`Retry`), actualización de cabecera `x-api-key`, escritura atómica de ficheros Parquet y sincronización segura de `git push`.
- **Rastreo de históricos en Git (`.gitignore`):** Adición de excepción explícita (`!data/raw/historical/*.parquet`) para permitir los *commits* automáticos del *collector* tras `v0.3.1`.


## [0.3.1] - 2026-11-19

### Added
- **Modelado predictivo multi-horizonte (`notebooks/modelo_predictivo.ipynb`):** Cuaderno experimental completo con ingeniería de características *anti-leakage*, partición cronológica con embargo de $3\text{ horas}$, comparativa frente a *baselines* (*Naive*, Perfil Histórico Causal y Ridge), selección de `LightGBMRegressor` ($+1\text{ h}, +2\text{ h}, +3\text{ h}$), ajuste bayesiano con Optuna, evaluación ciega en *Test* y explicabilidad global mediante SHAP.
- **Documentación de la Entrega 03 (`docs/entregas/entrega_03_machine_learning.md`):** Memoria técnica completa con planteamiento del problema, arquitectura de datos, métricas de validación/test, auditoría de errores y contrato de salida para el optimizador de rutas.
- **Reproducibilidad experimental (`data/gold_snapshot_ml_20261115/` y `notebooks/README.md`):** Estructura rastreada mediante `.gitkeep` para aislar la instantánea congelada de la Capa Gold (`31/10/2026` – `15/11/2026`) junto con la guía de descarga desde Google Drive.
- **Dependencias de Machine Learning (`requirements.txt`):** Incorporación de las librerías de entrenamiento, optimización y explicabilidad (`scikit-learn`, `lightgbm`, `xgboost`, `optuna`, `shap`, entre otras).
- **Empaquetado estándar MLOps (`pyproject.toml`):** Configuración del proyecto como paquete instalable en modo editable (`pip install -e .`) bajo el layout `src/`, permitiendo importaciones limpias (`from etl...`, `from modeling...`) desde scripts y notebooks.
- **Separación de dominios en `src/`:** Creación de los subpaquetes `src/etl/` (ingeniería de datos) y `src/modeling/` (preparado para la modularización futura de producción), junto con el directorio `scripts/` para los orquestadores.

### Changed
- **Documentación principal y control de versiones (`README.md` y `.gitignore`):** Actualización del esquema de directorios e instrucciones de instalación con `requirements.txt` en `README.md`, y blindaje de `.gitignore` frente a ficheros `.parquet`, `.csv` y archivos de sincronización (`desktop.ini`), manteniendo rastreada la carpeta del *snapshot* de ML.
- **Reorganización del pipeline ETL (`src/etl/` y `scripts/main_BBDD.py`):** Traslado de `config.py`, `io.py`, `cleaning.py`, `models_gold.py` y `utils.py` a `src/etl/`, reubicación de `main_BBDD.py` en `scripts/` y actualización de `ROOT_DIR` en `src/etl/config.py` (`parents[2]`).


## [0.2.1] - 2026-11-16

### Added
- **Confirmación y limpieza previa de salidas (`src/io.py`):** Nueva función `confirm_and_reset_output_dirs()` que solicita confirmación explícita al usuario (`[s/N]`) antes de ejecutar el pipeline y elimina los archivos residuales previos en `data/silver/` y `data/gold/` para garantizar regeneraciones limpias desde cero.
- **Cálculo unificado de disponibilidad temporal (`src/models_gold.py`):** Incorporación de `_compute_available_at()` para exigir que tanto el cierre de ventana (`window_end`) como la fecha de publicación (`generated_at`) sean anteriores o iguales al instante de captura (`timestamp`).
- **Metadatos de disponibilidad en Capa Gold:** Inclusión de `available_at` en `dim_patterns.parquet` y `history_available_at` en `dim_calendar.parquet` estandarizados en zona horaria `Europe/Madrid`.

### Changed
- **Prevención estricta de fuga temporal (*anti-data leakage*) en `src/models_gold.py`:**
  - **Rankings:** Eliminada la rama residual que cruzaba métricas por `station_id` sin filtro de fecha; ahora se aplica exclusivamente `merge_asof` hacia atrás sobre `available_at <= timestamp`, dejando en `NaN` las capturas previas a la descarga sin imputar ceros.
  - **Patrones horarios (`Dim_Patterns`):** Sustitución del cruce directo por `merge_asof` estricto por `(station_id, day_type, hour)` exigiendo `available_at <= timestamp` antes de calcular `occupancy_estimated`.
  - **Contexto diario de la ciudad (`Dim_Calendar` y `df_gold`):** Cruce temporal contra `available_at <= timestamp` para impedir que los agregados de una jornada $D$ estén disponibles antes del cierre completo del día.
- **Cierre de ventana diaria (`src/cleaning.py`):** `clean_history()` calcula explícitamente `window_end = date + 1 día` cuando no viene informado en el CSV original y conserva `generated_at`.
- **Orquestador limpio (`main_BBDD.py`):** Delegación de la gestión de borrado de archivos a `src/io.py` (expuesta en `src/__init__.py`) e integración de `df_history_silver` en `build_gold_dataset()`.

### Fixed
- **Compatibilidad de borrado en Windows / Google Drive (`src/io.py`):** Eliminación individual de ficheros internos (`Path.unlink()`) en lugar de destruir los directorios raíz con `shutil.rmtree()`, evitando bloqueos `PermissionError: [WinError 5]` en carpetas sincronizadas.
- **Parseo de marcas temporales (`src/utils.py`):** Ajuste en `safe_local_timestamp()` con formato `ISO8601` y *fallback* controlado para evitar advertencias de inferencia en consola.