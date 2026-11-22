# 📓 Cuadernos de Experimentación (`notebooks/`)

Este directorio reúne los cuadernos de Jupyter utilizados para el análisis exploratorio de datos (EDA), la auditoría de calidad de la Capa Gold, la validación experimental de los modelos predictivos de ocupación y el futuro prototipado de optimización de rutas para **SmartRebalance ZGZ**.

> **Separación entre experimentación y producción:** Los cuadernos de esta carpeta trabajan sobre una **instantánea congelada** en `data/gold_snapshot_ml_20261122/` para garantizar que todas las métricas, gráficos y conclusiones de la memoria sean 100 % reproducibles. El código de producción desacoplado que entrena e infiere sobre los datos vivos (`data/gold/`) se encuentra en `src/modeling/` y `scripts/`.

---

## 📥 Descarga del Snapshot de Datos (`gold_snapshot_ml_20261122`)

Dado que el repositorio de GitHub no aloja ficheros binarios pesados (`*.parquet`) y la carpeta de producción `data/gold/` se actualiza continuamente con nuevas capturas cada 10 minutos, los cuadernos leen una **instantánea congelada e inmutable** correspondiente al periodo auditado: **`31 de octubre de 2026` – `22 de noviembre de 2026`** (**23 días naturales**, **782.736 registros 100 % reales** en **276 estaciones**).

### 1. Enlace de descarga

Descarga los cuatro archivos `.parquet` desde la carpeta compartida de Google Drive:

* 🔗 [**Descargar carpeta `gold_snapshot_ml_20261122` (Google Drive)**](https://drive.google.com/drive/folders/1CCxv2z4uk2fDO9MtaDbFdER9UDy2kmbX?usp=drive_link)

### 2. Ubicación en el repositorio

Guarda los archivos descargados dentro de `data/gold_snapshot_ml_20261122/` (carpeta versionada mediante `.gitkeep`), respetando exactamente esta estructura:

```text
TFM/
└── data/
    └── gold_snapshot_ml_20261122/
        ├── .gitkeep
        ├── df_gold.parquet          # Tabla consolidada principal (782.736 filas, 276 estaciones)
        ├── dim_calendar.parquet     # Dimensión temporal y festivos locales/nacionales (ES-AR)
        ├── dim_patterns.parquet     # Dimensión de perfiles basales históricos por estación
        ├── dim_stations.parquet     # Maestro geográfico y capacidad física de las 276 estaciones
        └── predictions/             # (Se genera automáticamente al ejecutar modelado_predictivo.ipynb)
            └── test_predictions_lgbm.parquet
```

> **Nota:** Todos los archivos `.parquet` están excluidos en `.gitignore` para evitar subir datos pesados al control de versiones.

---

## 🗂️ Contenido de la Carpeta

### 1. `modelado_predictivo.ipynb` *(Disponible)*
Cuaderno principal de experimentación y validación del modelo predictivo de ocupación a horizontes tácticos ($+1\text{ h}, +2\text{ h}, +3\text{ h}$):
* **Auditoría *anti-leakage* y EDA espacio-temporal:** Validación física de las 276 estaciones, contraste entre días laborables y festivos, flujos pendulares matinales y detección de saltos logísticos bruscos.
* **Ingeniería de características y partición temporal:** Construcción de retardos exactos ($10\text{ min}$ a $7\text{ días}$), perfil basal causal, calendario cíclico y partición cronológica estricta (`70 % Train / 15 % Validación / 15 % Test`) con ventanas de embargo de $3\text{ horas}$.
* **Benchmarking y optimización:** Comparativa frente a *baselines* (*Naive*, Perfil Histórico Causal y Regresión Ridge), evaluación de ensambles (*Random Forest*, *XGBoost* y *LightGBM*) y ajuste fino de hiperparámetros con **Optuna** seleccionando **LightGBM (Optuna)** como modelo campeón.
* **Evaluación ciega, caso de estudio y explicabilidad:** Auditoría de errores horarios y espaciales sobre *Test*, análisis específico de una estación en saturación ($> 80\%$), explicabilidad global mediante **SHAP** y exportación determinista de `test_predictions_lgbm.parquet`.

### 2. `rebalancing_routing_engine.ipynb` *(En desarrollo — Próximamente)*
Cuaderno reservado para el motor prescriptivo de optimización de rutas de la flota de rebalanceo (6 furgonetas de 22 plazas), que consumirá las predicciones *out-of-sample* generadas por `modelado_predictivo.ipynb`. *(Pendiente de incorporación al repositorio en la siguiente fase del proyecto).*

---

## ▶️ Requisitos para Ejecutar los Cuadernos

Asegúrate de tener activado el entorno virtual `.venv` e instaladas las dependencias opcionales de experimentación (`optuna`, `shap`, `xgboost`, `matplotlib`, `seaborn`) definidas en `pyproject.toml`:

```bash
pip install -e ".[notebooks]"
```