# Changelog

Todos los cambios notables en este proyecto serán documentados en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/),
y este proyecto adhiere a [Semantic Versioning](https://semver.org/lang/es/).


## [Unreleased]


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