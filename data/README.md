# 📂 Arquitectura y Gestión de Datos (`data/`)

Este directorio almacena todas las capas de datos del proyecto **SmartRebalance ZGZ**, organizadas siguiendo una **Arquitectura Medallón** (`raw` $\rightarrow$ `silver` $\rightarrow$ `gold`), junto con las salidas de inferencia en producción (`predictions/`) y la instantánea congelada para experimentación (`gold_snapshot_ml_20261122/`).

> **Nota sobre credenciales:** El procedimiento para obtener y configurar la variable de entorno `BIZI_API_KEY` se encuentra detallado en el [`README.md` principal](../README.md). No obstante, el pipeline ETL permite ejecutarse en **modo 100 % offline** reutilizando los ficheros cacheados en `data/raw/`.

---

## 🌐 1. Procedencia de los Datos

Los datos del sistema de bicicletas públicas de Zaragoza provienen de dos fuentes abiertas principales:

1. **API de DatosBizi Zaragoza (`https://datosbizi.com/api`):**
   * **Fuente primaria:** Feed oficial **GBFS v2** del servicio Bizi Zaragoza (`https://zaragoza.publicbikesystem.net/customer/gbfs/v2/gbfs.json`), agregado y enriquecido mediante *endpoints* históricos y analíticos.
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
├── raw/                        # Capa Bronze: datos en crudo y caché local de la API
│   └── historical/             # [Versionado en Git] Capturas automáticas cada 10 min (Parquet diario)
├── silver/                     # Capa Silver: datos limpios, tipados y validados (regenerable)
├── gold/                       # Capa Gold: modelo en estrella y tabla maestra de producción (regenerable)
├── predictions/                # Salida de inferencia batch generada por scripts/predict.py (regenerable)
├── gold_snapshot_ml_20261122/  # Instantánea congelada exclusiva para reproducibilidad en notebooks/
└── README.md                   # Documentación de la arquitectura de datos
```

---

## 🥉 3. Detalle de las Capas y Directorios

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
* Validación de integridad física ($\text{capacity} > 0$, $0 \le \text{bicicletas / anclajes} \le \text{capacity}$, ocupación $\in [0.0, 1.0]$) y deduplicación por `(station_id, timestamp)`.

### Capa Gold (`data/gold/`)
Generada automáticamente por `src/etl/models_gold.py` en formato Parquet optimizado para analítica y Machine Learning en producción:
* **Modelo Dimensional en Estrella:** Separa entidades descriptivas (`dim_stations`, `dim_calendar`, `dim_patterns`) de los eventos granulares (`fact_station_snapshots`), junto con el informe de completitud del recolector (`coverage_report`).
* **Tabla Maestra Consolidada (`df_gold.parquet`):** Dataset unificado consumido directamente por `scripts/train_model.py` y `scripts/predict.py`, donde cada fila representa el estado de una estación en un instante $t$ (cada 10 minutos).
* **Reglas *Anti-Data Leakage*:** Los cruces con rankings, patrones horarios y agregados diarios de ciudad se resuelven mediante `merge_asof` hacia atrás exigiendo disponibilidad temporal estricta ($\text{available\\_at} \le \text{timestamp}$). Los registros reconstruidos por arranque en frío (*Cold Start*) se aíslan en `occupancy_estimated` y se etiquetan con `is_estimated = 1`, manteniendo intacta la verdad observada en `occupancy_pct`.
* **Trazabilidad SHA-256:** Cada ejecución de entrenamiento calcula la huella digital SHA-256 de los ficheros de `data/gold/` y la registra en `models/lgbm_v1/metadata.json`.

### Predicciones de Producción (`data/predictions/`)
Generada de forma desacoplada por `scripts/predict.py` sin necesidad de reentrenar los modelos:
* Almacena la salida de inferencia multi-horizonte (`predictions_lgbm.parquet`) con el estado físico actual y las previsiones de ocupación a $+1\text{ h}$, $+2\text{ h}$ y $+3\text{ h}$ para las 276 estaciones.
* Constituye el contrato de entrada para la futura fase de optimización de rutas de la flota de rebalanceo.

### Instantánea Congelada (`data/gold_snapshot_ml_20261122/`)
* Carpeta reservada **exclusivamente para los cuadernos de `notebooks/`** (periodo auditado de 23 días: `31 oct – 22 nov`).
* Al mantenerse aislada de la carpeta viva `data/gold/`, evita que nuevas capturas alteren las métricas, gráficos y conclusiones documentadas en la memoria del TFM. Las instrucciones de descarga de sus ficheros se detallan en [`notebooks/README.md`](../notebooks/README.md).

---

## 🔄 4. Reproducibilidad y Control de Versiones

* **Regeneración de Silver y Gold:** Las carpetas `data/silver/` y `data/gold/` están excluidas en `.gitignore` y se reconstruyen limpiamente desde cero ejecutando el pipeline ETL desde la raíz del proyecto:
  ```bash
  python scripts/main_BBDD.py
  ```
* **Regeneración de Predicciones:** La carpeta `data/predictions/` también está excluida en `.gitignore` y se genera bajo demanda con:
  ```bash
  python scripts/predict.py
  ```
* **Sincronización de capturas históricas:** Para incorporar los últimos archivos diarios acumulados por el *scheduler* automático en GitHub, basta con actualizar la rama local:
  ```bash
  git pull origin main
  ```
