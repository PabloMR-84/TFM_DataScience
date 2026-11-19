# 📓 Cuadernos de Experimentación (`notebooks/`)

Este directorio reúne los cuadernos de Jupyter utilizados para el análisis exploratorio de datos (EDA), la auditoría de calidad de la Capa Gold y la validación experimental de los modelos predictivos de ocupación para **SmartRebalance ZGZ**.

## 📥 Descarga del Snapshot de Datos (`gold_snapshot_ml_20261115`)

Dado que el repositorio de GitHub no aloja ficheros binarios pesados (`*.parquet`) y la carpeta de producción `data/gold/` se actualiza continuamente con nuevas capturas cada 10 minutos, los cuadernos leen una **instantánea congelada e inmutable** correspondiente al periodo auditado (`31 de octubre de 2026` – `15 de noviembre de 2026`, `540.960` registros reales en `276` estaciones).

De este modo, al ejecutar los cuadernos se reproducen exactamente las mismas métricas, gráficos y conclusiones documentadas en sus celdas sin verse alterados por futuras ejecuciones del recolector.

### 1. Enlace de descarga

Descarga los cuatro archivos `.parquet` desde la carpeta compartida de Google Drive:

* 🔗 [**Descargar carpeta `gold_snapshot_ml_20261115` (Google Drive)**](https://drive.google.com/drive/folders/1I8Wwiss2vubQlXl92_SIGDLN5zQAImFd?usp=drive_link)

### 2. Ubicación en el repositorio

Guarda los archivos descargados dentro de `data/gold_snapshot_ml_20261115/` (carpeta ya creada y versionada mediante `.gitkeep`), respetando exactamente esta estructura:

```text
TFM/
└── data/
    └── gold_snapshot_ml_20261115/
        ├── .gitkeep
        ├── df_gold.parquet          # Tabla consolidada principal (540.960 filas, 276 estaciones)
        ├── dim_calendar.parquet     # Dimensión temporal y festivos locales/nacionales (ES-AR)
        ├── dim_patterns.parquet     # Dimensión de perfiles basales históricos por estación
        └── dim_stations.parquet     # Maestro geográfico y capacidad física de las 276 estaciones
```

> **Nota:** Los archivos `.parquet` están ignorados en `.gitignore` para evitar subir datos pesados al control de versiones.

## 🗂️ Contenido de la Carpeta

* **Cuaderno de Modelado Predictivo ($+1\text{ h}, +2\text{ h}, +3\text{ h}$):** Ingeniería de características sin fuga de datos (*anti-leakage*), partición cronológica con embargo de $3\text{ horas}$, comparativa frente a *baselines* (*Naive*, Perfil Histórico Causal y Regresión Ridge), selección de **LightGBM**, ajuste bayesiano con **Optuna**, evaluación ciega en *Test* y explicabilidad global mediante **SHAP**.