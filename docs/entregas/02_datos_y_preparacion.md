# Entrega 02: Datos y Preparación — SmartRebalance ZGZ

## 1. Qué datos necesita vuestro producto

* **Observación analítica:** Representa el estado instantáneo y la disponibilidad de una estación Bizi Zaragoza en una franja horaria/instantánea concreta.

* **Datos imprescindibles:** Identificador de estación (`station_id`), coordenadas (`latitude`, `longitude`), capacidad total (`capacity`), bicicletas disponibles (`bikes_available`), anclajes libres (`anchors_free`) y marca temporal (`timestamp`).

* **Datos deseables y contexto operacional:** Indicadores de rotación (`turnover_score`), horas de estrés operativo (`empty_hours`, `full_hours`, `problem_hours`), perfiles basales esperados por estación y hora (`occupancy_avg`, `bikes_avg`), calendario de festivos (`is_holiday`, `day_type_effective`) e indicadores macro diarios a nivel de ciudad (`demand_score`, `city_avg_occupancy`, `balance_index`).

* **Granularidad e histórico:** Capturas cada 10 minutos para la serie temporal histórica en tiempo real y agregaciones horarias ($t$) y diarias.

* **Horizonte de anticipación (**$h$**):**

  * *Corto plazo (*$h = 1\text{ hora}$*):* Detección inminente de desabastecimiento o saturación.

  * *Táctico (*$h = 2\text{ a } 3\text{ horas}$*):* Planificación previa de rutas de rebalanceo antes de las horas punta.

* **Diferenciación metodológica clave:** Las transferencias sugeridas por la API (`bikes_to_move`) no representan la realidad observada, sino una recomendación externa. La verdad de terreno (*ground truth*) para entrenar y evaluar los modelos predictivos es exclusivamente la ocupación real física observada en la estación (`occupancy_pct`).

## 2. Fuentes reales y viabilidad

* **Fuente principal:** API pública de DatosBizi Zaragoza (`https://datosbizi.com/api`) para maestros, patrones, rankings y contexto histórico macro, complementada con capturas periódicas cada 10 minutos almacenadas en formato Apache Parquet.

* **Fuente auxiliar de calendario:** API pública Nager.Date (`https://date.nager.at/api/v3/PublicHolidays`) para la obtención automatizada de festivos nacionales y autonómicos de Aragón (`ES-AR`), combinada con el calendario de festivos locales de Zaragoza y vísperas.

* **Acceso y arquitectura de ingesta:** Descarga HTTP mediante peticiones REST (JSON) y streaming en bloque para CSVs, gestionada a través de una arquitectura modular en Python (`src/config.py`, `src/io.py`, `src/utils.py`, `src/cleaning.py`, `src/models_gold.py`) ejecutada desde el orquestador `main_BBDD.py`.

* **Estabilidad y cobertura:** Alta estabilidad. El pipeline incorpora auditorías de completitud mediante reportes de cobertura (`coverage_report_8.csv` en formato Parquet), evaluando `capturas_reales` frente a `capturas_esperadas`, `pct_cobertura` y los tramos `sin_capturas` por estación y día.

* **Plan de contingencia (Fallback):** Si la API no responde o se ejecuta en modo offline, `src/io.py` detecta los archivos cacheados en `data/raw/` y permite reutilizarlos en bloque sin conexión, apoyándose en la extrapolación de patrones de la Capa Gold para cubrir lagunas temporales.

## 3. Uso, privacidad y datos personales

* **Licencia:** Datos abiertos urbanos de la red Bizi Zaragoza bajo citación y normativa pública.

* **Privacidad:** No se recopila ni almacena **ningún dato personal**, nombre de usuario, trazabilidad de trayectos individuales ni datos sensibles de ciudadanos. Cumple con el principio de minimización al 100%.

## 4. Del dato bruto al dato utilizable (Flujo Medallion)

El procesamiento se estructura en un pipeline reproducible organizado en tres capas:

* **Bronze (Raw — `data/raw/`):**

  * Ingesta idempotente de fuentes estáticas y agregadas (`stations_raw.json`, `rankings_raw.json`, `rebalancing_raw.json`, `patterns_raw.json`, `bizi_history_raw.csv`) y lectura dinámica de las capturas instantáneas en `data/raw/historical/*.parquet`.

  * *(Nota de alcance: La fuente `mobility_raw.json` se mantiene aislada en Bronze y excluida temporalmente de las capas posteriores para garantizar la estabilidad del esquema).*

* **Silver (`data/silver/`):**

  * Limpieza pura y estandarización de esquemas (`src/cleaning.py` y `src/utils.py`): renombrado a *snake_case*, eliminación de duplicados por `(station_id, timestamp)`, desanidado seguro de estructuras JSON (`timeWindow`, `expectedImpact`), tipado explícito (`Int64` entero anulable de Pandas para preservar nulos reales sin distorsionar promedios, `float64`) y normalización temporal estandarizada a la zona horaria `Europe/Madrid`.

  * Salidas generadas: `stations_silver.csv`, `rebalancing_silver.csv`, `rankings_silver.csv`, `patterns_silver.csv`, `bizi_history_silver.csv` y `historical_snapshots_silver.parquet`.

* **Gold (`data/gold/`):**

  * Modelado Dimensional en Estrella (`src/models_gold.py`) almacenado íntegramente en formato columnar **Apache Parquet**:

    * `dim_stations.parquet`: Catálogo maestro de estaciones únicas (`station_id`, `name`, `latitude`, `longitude`, `capacity`).

    * `dim_calendar.parquet`: Dimensión temporal enriquecida con festivos vía API (`is_holiday`, `day_type_effective`) e indicadores macro diarios (`demand_score`, `city_avg_occupancy`, `balance_index`).

    * `dim_patterns.parquet`: Perfiles esperados por estación, tipo de día (`WEEKDAY` / `WEEKEND`) y hora (`occupancy_avg`, `bikes_avg`).

    * `fact_station_snapshots.parquet`: Tabla de hechos granular con las lecturas reales por estación e instante.

    * `df_gold.parquet`: Dataset analítico consolidado con controles relacionales estrictos (`validate="m:1"`) para evitar duplicaciones y *data leakage*.

  * **Resolución del *Cold Start*:** Cuando existen lagunas de capturas reales en una franja horaria, el pipeline reconstruye la ocupación esperada (`occupancy_estimated`) modulando el perfil basal horario de la estación (`occupancy_avg`) con el comportamiento macro de la ciudad ese día, trazando cada registro mediante la bandera `is_estimated` (`1` para extrapolaciones, `0` para lecturas reales) y el estado `cold_start_status`.

### Columnas explícitas del Dataset Gold (`df_gold.parquet`):

* **Claves e Identificación:** `station_id` (Int64), `date_key` (Int64, YYYYMMDD), `timestamp` (datetime64\[us, Europe/Madrid\]), `hour` (Int64).

* **Métricas Observadas y Estimadas (Target & Features):** `bikes_available` (Int64), `anchors_free` (Int64), `occupancy_pct` (float64, ratio real de ocupación), `occupancy_estimated` (float64, ocupación reconstruida).

* **Calidad y Trazabilidad:** `is_estimated` (Int64, flag binario de extrapolación), `cold_start_status` (string).

* **Contexto Temporal y Macro:** `day_type_effective` (string), `is_holiday` (Int64), `demand_score` (float64), `city_avg_occupancy` (float64), `balance_index` (float64).

* **Basales y Estrés Operativo:** `occupancy_avg` (float64), `turnover_score` (float64), `empty_hours` (float64), `full_hours` (float64), `problem_hours` (float64).

## 5. Riesgos y limitaciones

* **Falta de datos:** Ausencia de micro-datos de rutas individuales origen-destino exactas (se trabaja con estados de estación, agregaciones y recomendaciones) y aislamiento temporal de las señales de movilidad externa (`mobility_raw.json`).

* **Sesgos y lagunas de captura:** Posibles interrupciones puntuales en la recolección cada 10 minutos (monitorizadas mediante `coverage_report.csv`) y estacionalidad por condiciones meteorológicas extremas o eventos festivos atípicos.

* **Estrategia ante caídas de fuente:** Ejecución en modo caché local mediante `src/io.py` y uso de los basales horarios de `dim_patterns.parquet` junto con la imputación escalada de `occupancy_estimated` si la API quedase inaccesible.