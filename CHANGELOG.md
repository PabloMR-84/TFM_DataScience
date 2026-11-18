# Changelog

Todos los cambios notables en este proyecto serán documentados en este archivo.

El formato está basado en [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/),
y este proyecto adhiere a [Semantic Versioning](https://semver.org/lang/es/).


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


## [Unreleased]

### Added
- **Empaquetado estándar MLOps (`pyproject.toml`):** Configuración del proyecto como paquete instalable en modo editable (`pip install -e .`) bajo el layout `src/`, permitiendo importaciones limpias (`from etl...`, `from modeling...`) desde scripts, notebooks y tests sin alterar `sys.path`.
- **Separación de dominios en `src/`:** Creación de los subpaquetes `src/etl/` (ingeniería de datos) y `src/modeling/` (preparado para *feature engineering* y modelos predictivos), junto con el directorio `scripts/` para los orquestadores de ejecución.

### Changed
- **Reorganización del pipeline ETL (`src/etl/` y `scripts/main_BBDD.py`):** Traslado de `config.py`, `io.py`, `cleaning.py`, `models_gold.py` y `utils.py` a `src/etl/`, y reubicación de `main_BBDD.py` en `scripts/`, actualizando todas las importaciones internas al nuevo espacio de nombres.
- **Ajuste de ruta raíz (`src/etl/config.py`):** Actualización de `ROOT_DIR` a `Path(__file__).resolve().parents[2]` tras bajar un nivel en la jerarquía de directorios.