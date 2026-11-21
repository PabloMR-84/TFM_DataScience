# 📂 Arquitectura y Gestión de Datos (`data/`)

Este directorio almacena todas las capas de datos del proyecto **SmartRebalance ZGZ**, organizadas siguiendo una **Arquitectura Medallón** (`raw` $\rightarrow$ `silver` $\rightarrow$ `gold`) diseñada para garantizar la trazabilidad, la calidad física de las observaciones y la ausencia de fuga de información temporal (*anti-data leakage*).

> **Nota sobre credenciales:** El procedimiento para obtener y configurar la variable de entorno `BIZI_API_KEY` se encuentra detallado en el [`README.md` principal](../README.md) del repositorio. No obstante, el pipeline ETL permite ejecutarse en **modo 100 % offline** reutilizando los ficheros cacheados en `data/raw/`.

---

## 🌐 1. Procedencia de los Datos

Los datos del sistema de bicicletas públicas de Zaragoza provienen de dos fuentes abiertas principales:

1. **API de DatosBizi Zaragoza (`https://datosbizi.com/api`):**
   * **Fuente primaria:** Feed oficial **GBFS v2** del servicio Bizi Zaragoza (`https://zaragoza.publicbikesystem.net/customer/gbfs/v2/gbfs.json`), agregado y enriquecido mediante endpoints históricos y analíticos.
   * **Cobertura:** Las **276 estaciones** activas de la red urbana de Zaragoza.
   * **Endpoints consumidos en la ingesta:**
     * `/api/stations`: Estado instantáneo y metadatos geográficos/capacidad de las estaciones.
     * `/api/rebalancing`: Sugerencias operativas externas de redistribución.
     * `/api/rankings`: Métricas de rotación (`turnover_score`) y horas en estado crítico (`empty_hours`, `full_hours`, `problem_hours`).
     * `/api/patterns`: Curvas basales históricas de ocupación por estación, tipo de día y hora.
     * `/api/history`: Indicadores macro diarios agregados a nivel de ciudad (`demand_score`, `city_avg_occupancy`, `balance_index`).
2. **API Pública Nager.Date (`https://date.nager.at`):**
   * Utilizada durante la construcción de la dimensión calendario en la capa Gold para identificar automáticamente festivos nacionales y autonómicos (`ES-AR`), complementada con reglas locales recurrentes de Zaragoza (Cincomarzada, Fiestas del Pilar y vísperas críticas de movilidad).

---

## 🗂️ 2. Estructura de Directorios

```text
data/
├── raw/                              # Capa Bronze: Datos en crudo tal como llegan de la fuente
│   ├── historical/                   # [Versionado en Git] Capturas automáticas cada 10 min
│   │   └── bizi_YYYYMMDD.parquet     # Histórico particionado por día natural (UTC)
│   ├── stations_raw.json             # Maestro de estaciones (caché API)
│   ├── rebalancing_raw.json          # Recomendaciones de rebalanceo (caché API)
│   ├── rankings_raw.json             # Rankings de rotación y fricción (caché API)
│   ├── patterns_raw.json             # Perfiles basales horarios (caché API)
│   └── bizi_history_raw.csv          # Serie macro diaria de ciudad (caché API)
│
├── silver/                           # Capa Silver: Datos limpios, tipados y validados (regenerable)
│   ├── stations_silver.csv
│   ├── rebalancing_silver.csv
│   ├── rankings_silver.csv
│   ├── patterns_silver.csv
│   ├── bizi_history_silver.csv
│   └── historical_snapshots_silver.parquet
│
├── gold/                             # Capa Gold: Modelo Estrella y tabla maestra ML (regenerable)
│   ├── dim_stations.parquet          # Dimensión maestra de estaciones (276 filas)
│   ├── dim_calendar.parquet          # Dimensión temporal, festivos y métricas macro diarias
│   ├── dim_patterns.parquet          # Dimensión de perfiles basales horarios con available_at
│   ├── fact_station_snapshots.parquet# Tabla de hechos granular (cada 10 minutos)
│   ├── coverage_report.parquet       # Auditoría de completitud del scheduler (también en .csv)
│   └── df_gold.parquet               # Tabla maestra consolidada (One Big Table) para ML
│
└── gold_snapshot_ml_20261115/        # Instantánea congelada para reproducibilidad de notebooks
    └── .gitkeep                      # (Ver notebooks/README.md para enlace de descarga)
```

---

## 🥉 3. Detalle de las Capas del Flujo Medallón

### Capa Bronze / Raw (`data/raw/`)
Almacena las respuestas originales de la API sin alterar su estructura ni imputar valores, preservando los metadatos temporales de raíz (`generatedAt`, `windowStart`, `windowEnd`) para garantizar la trazabilidad.
* **Subcarpeta `data/raw/historical/` (Ingesta Automatizada):**
  * Contiene la serie temporal real de estado de las 276 estaciones capturada con una frecuencia de **10 minutos** ($144\text{ capturas/día}$ por estación).
  * Se alimenta de forma desatendida mediante un flujo de **GitHub Actions** (`extract.py`) que realiza peticiones resilientes a `/api/stations` y consolida las lecturas en archivos diarios en formato columnar **Apache Parquet** (`bizi_YYYYMMDD.parquet`) mediante escritura atómica.
  * A diferencia del resto de ficheros de datos, **esta subcarpeta sí está versionada en Git** para garantizar un histórico continuo, compartido e incremental en el repositorio.

### Capa Silver (`data/silver/`)
Generada automáticamente por el módulo `src/etl/cleaning.py`. Aplica reglas deterministas de saneamiento sobre la capa Raw:
* Estandarización de nombres de columnas a `snake_case`.
* Homogeneización estricta de todas las marcas temporales a la zona horaria local **`Europe/Madrid`**.
* Preservación de valores ausentes reales mediante tipos anulables de Pandas (`Int64`), evitando distorsionar promedios con ceros artificiales.
* Validación de integridad física ($capacity > 0$, $0 \le \text{bicicletas / anclajes} \le capacity$, ocupación $\in [0.0, 1.0]$) y deduplicación por `(station_id, timestamp)`.

### Capa Gold (`data/gold/`)
Generada automáticamente por `src/etl/models_gold.py` en formato Parquet optimizado para analítica y Machine Learning:
* **Modelo Dimensional en Estrella:** Separa entidades descriptivas (`dim_stations`, `dim_calendar`, `dim_patterns`) de los eventos granulares (`fact_station_snapshots`).
* **Tabla Maestra Consolidada (`df_gold.parquet`):** Dataset unificado listo para ingeniería de características (`src/modeling/`), donde cada fila representa el estado de una estación en un instante $t$ (cada 10 minutos).
* **Reglas Anti-Data Leakage:** Los cruces con rankings, patrones horarios y agregados diarios de ciudad se resuelven mediante `merge_asof` hacia atrás exigiendo disponibilidad temporal estricta ($\text{available\_at} \le \text{timestamp}$). Asimismo, los registros reconstruidos por arranque en frío (*Cold Start*) se aíslan en `occupancy_estimated` y se etiquetan con `is_estimated = 1`, manteniendo intacta la verdad observada en `occupancy_pct`.

---

## 🔄 4. Reproducibilidad y Control de Versiones

* **Regeneración completa:** Las carpetas `data/silver/` y `data/gold/` están excluidas en `.gitignore` y se reconstruyen limpiamente desde cero ejecutando el orquestador desde la raíz del proyecto:
  ```bash
  python scripts/main_BBDD.py
  ```
* **Sincronización de capturas históricas:** Para incorporar los últimos archivos diarios acumulados por el *scheduler* automático en GitHub, basta con actualizar la rama local:
  ```bash
  git pull origin main
  ```
* **Instantánea congelada (`gold_snapshot_ml_20261115/`):** Reservada para la ejecución reproducible de los cuadernos exploratorios y de modelado; las instrucciones de descarga de sus ficheros `.parquet` se detallan en [`notebooks/README.md`](../notebooks/README.md).