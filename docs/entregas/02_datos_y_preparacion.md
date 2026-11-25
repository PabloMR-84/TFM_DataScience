# Entrega 02: Datos y Preparación — SmartRebalance ZGZ

## 1. Qué datos necesita vuestro producto

* **Observación analítica:** Representa el estado instantáneo y la disponibilidad de una estación Bizi Zaragoza en una franja horaria/instantánea concreta.

* **Datos imprescindibles:** Identificador de estación (`station_id`), coordenadas (`latitude`, `longitude`), capacidad total (`capacity`), bicicletas disponibles (`bikes_available`), anclajes libres (`anchors_free`) y marca temporal (`timestamp`).

* **Datos deseables y contexto operacional:** Indicadores de rotación (`turnover_score`), horas de estrés operativo (`empty_hours`, `full_hours`, `problem_hours`), perfiles basales esperados por estación y hora (`occupancy_avg`, `bikes_avg`), calendario de festivos (`is_holiday`, `day_type_effective`) e indicadores macro diarios a nivel de ciudad (`demand_score`, `city_avg_occupancy`, `balance_index`).

* **Granularidad e histórico:** Capturas cada 10 minutos para la serie temporal histórica en tiempo real y agregaciones horarias ($t$) y diarias.

* **Horizonte de anticipación ($h$):**
  * *Corto plazo ($h = 1\text{ hora}$):* Detección inminente de desabastecimiento o saturación.
  * *Táctico ($h = 2\text{ a } 3\text{ horas}$):* Planificación previa de rutas de rebalanceo antes de las horas punta.

* **Diferenciación metodológica clave:** Las transferencias sugeridas por la API (`bikes_to_move`) no representan la realidad observada, sino una recomendación externa. La verdad de terreno (*ground truth*) para entrenar y evaluar los modelos predictivos es exclusivamente la ocupación real física observada en la estación (`occupancy_pct`).

---

## 2. Fuentes reales, viabilidad y auditoría de cobertura

* **Fuente principal:** API pública de DatosBizi Zaragoza (`https://datosbizi.com/api`) para maestros, patrones, rankings y contexto histórico macro, complementada con capturas periódicas cada 10 minutos almacenadas en formato columnar Apache Parquet (`data/raw/historical/bizi_YYYYMMDD.parquet`) mediante un *workflow* automatizado en GitHub Actions.

* **Fuente auxiliar de calendario:** API pública Nager.Date (`https://date.nager.at/api/v3/PublicHolidays`) para la obtención automatizada de festivos nacionales y autonómicos de Aragón (`ES-AR`), combinada con el calendario recurrente de festivos locales de Zaragoza (San Valero, Cincomarzada, El Pilar) y vísperas críticas de movilidad.

* **Acceso y arquitectura de ingesta:** Descarga HTTP mediante peticiones REST (JSON) y streaming en bloque para CSV, gestionada a través de una arquitectura modular en Python (`src/etl/config.py`, `src/etl/io.py`, `src/etl/utils.py`, `src/etl/cleaning.py`, `src/etl/models_gold.py`) ejecutada desde el orquestador `scripts/main_BBDD.py`.

* **Plan de contingencia (Fallback):** Si la API no responde o se ejecuta en modo offline, `src/etl/io.py` detecta los archivos cacheados en `data/raw/` (`resolve_download_mode`) y permite reutilizarlos en bloque sin conexión (`[U/r]`), apoyándose en la extrapolación de patrones de la Capa Gold para cubrir lagunas temporales.

### Resumen de fechas reunidas y auditoría de cobertura (`coverage_report`)

El repositorio acumula ya **22 jornadas diarias** en `data/raw/historical/` (del **31 de octubre de 2026** al **21 de noviembre de 2026**). Sobre la ventana consolidada en el reporte de cobertura de la Capa Gold (`coverage_report.csv` / `.parquet`, del **31/10/2026 al 19/11/2026**, **20 días naturales** y **276 estaciones activas** a frecuencia nominal de 10 minutos —144 franjas/día por estación—), se obtienen los siguientes resultados:

* **Balance global:** **679.512 capturas reales** frente a **794.880 esperadas** (**85,49 % de cobertura media**; **97,57 % de cobertura mediana**), con **0 días o estaciones completamente vacíos** (`sin_capturas = 0`).
* **Patrón de ausencias:** En cada jornada auditada, el mínimo y el máximo de capturas por estación coinciden exactamente en las 276 estaciones. Esto demuestra que los huecos no se deben a fallos individuales de sensores, sino a **retrasos o micro-cortes síncronos en la ejecución programada del *collector* de GitHub Actions**.

| Tramo operativo | Fechas | Días | Capturas reales / esperadas | Cobertura media | Diagnóstico de datos reunidos y ausencias |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **1. Arranque del *scheduler*** | `31/10/2026` – `02/11/2026` | 3 | 24.564 / 119.232 | **20,60 %** | Puesta en marcha progresiva del *cron*: 3 capturas/estación el `31/10` (`2,08 %`), 5 el `01/11` (`3,47 %`) y 81 el `02/11` (`56,25 %`). |
| **2. Operación estable** | `03/11/2026` – `18/11/2026` | 16 | 619.620 / 635.904 | **97,44 %** | Serie continua de alta calidad (entre `93,06 %` el `16/11` y `100,00 %` completo el `08/11`). Micro-huecos aislados de 1 a 10 franjas de 10 min por latencia del *runner*. |
| **3. Corte de descarga** | `19/11/2026` | 1 | 35.328 / 39.744 | **88,89 %** | Jornada parcial hasta la hora de generación del *snapshot* (128/144 franjas capturadas; completada posteriormente junto con el `20/11` y `21/11` en `data/raw/historical/`). |

---

## 3. Uso, privacidad y acceso a las entradas locales

* **Licencia:** Datos abiertos urbanos de la red Bizi Zaragoza (Ayuntamiento de Zaragoza / DatosBizi) reutilizados bajo normativa pública de datos abiertos y atribución de fuente; código del proyecto publicado bajo licencia MIT en el repositorio.
* **Privacidad:** No se recopila ni almacena **ningún dato personal**, identificador de usuario, trazabilidad de trayectos individuales ni datos sensibles de ciudadanos. Cumple al 100 % con el principio de minimización de datos.
* **Disponibilidad de las cinco entradas locales:** Las muestras de entrada de la API (`stations_raw.json`, `rankings_raw.json`, `rebalancing_raw.json`, `patterns_raw.json`, `bizi_history_raw.csv`) junto con los snapshots `data/raw/historical/*.parquet` están disponibles en el repositorio para ejecutar `python scripts/main_BBDD.py` en modo local (`[U]`) sin necesidad de clave de API (`BIZI_API_KEY`). Si se desea refrescar desde la API en vivo (`[R]`), basta con configurar `BIZI_API_KEY` en el archivo `.env` a partir de `.env.example`.

---

## 4. Del dato bruto al dato utilizable (Flujo Medallion)

El procesamiento se estructura en un pipeline reproducible organizado en tres capas:

* **Bronze (Raw — `data/raw/`):**
  * Ingesta idempotente de fuentes estáticas y agregadas (`stations_raw.json`, `rankings_raw.json`, `rebalancing_raw.json`, `patterns_raw.json`, `bizi_history_raw.csv`), preservando los metadatos temporales de raíz (`generatedAt`, `windowStart`, `windowEnd`), y lectura dinámica de las capturas instantáneas en `data/raw/historical/*.parquet`.
  * *(Nota de alcance: La fuente `mobility_raw.json` se mantiene aislada en Bronze y excluida temporalmente de las capas posteriores para garantizar la estabilidad del esquema).*

* **Silver (`data/silver/`):**
  * Limpieza pura y estandarización de esquemas (`src/etl/cleaning.py` y `src/etl/utils.py`): renombrado a *snake_case*, eliminación de duplicados por `(station_id, timestamp)`, desanidado seguro de estructuras JSON (`timeWindow`, `expectedImpact`), tipado explícito (`Int64` entero anulable de Pandas para preservar nulos reales sin distorsionar promedios, `float64`), validación de rangos físicos (`capacity > 0`, `0 <= bikes/anchors <= capacity`, ocupación en `[0.0, 1.0]`) y normalización temporal estandarizada a `Europe/Madrid`.
  * Salidas generadas: `stations_silver.csv`, `rebalancing_silver.csv`, `rankings_silver.csv`, `patterns_silver.csv`, `bizi_history_silver.csv` y `historical_snapshots_silver.parquet`.

* **Gold (`data/gold/`):**
  * Modelado Dimensional en Estrella (`src/etl/models_gold.py`) almacenado íntegramente en formato columnar **Apache Parquet**:
    * `dim_stations.parquet`: Catálogo maestro de 276 estaciones únicas (`station_id`, `name`, `latitude`, `longitude`, `capacity`).
    * `dim_calendar.parquet`: Dimensión temporal enriquecida con festivos vía API (`is_holiday`, `day_type_effective`) e indicadores macro diarios (`demand_score`, `city_avg_occupancy`, `balance_index`).
    * `dim_patterns.parquet`: Perfiles esperados por estación, tipo de día (`WEEKDAY` / `WEEKEND`) y hora (`occupancy_avg`, `bikes_avg`).
    * `fact_station_snapshots.parquet`: Tabla de hechos granular con las lecturas reales por estación e instante.
    * `coverage_report.parquet` (y `.csv`): Auditoría de completitud diaria por estación.
    * `df_gold.parquet`: Dataset analítico consolidado con controles relacionales estrictos (`validate="m:1"`).

  * **Prevención estricta de fuga temporal (*Anti-Leakage*):** Todos los cruces con contexto agregado (`rankings_silver`, `dim_patterns` y métricas macro diarias de `bizi_history_silver`) exigen mediante `pd.merge_asof(..., direction="backward")` que haya finalizado tanto su periodo de observación como su publicación (`available_at = max(window_end, generated_at) <= timestamp`), manteniendo `NaN` reales (sin *fallbacks* ni rellenos con ceros) en cualquier instante anterior a su disponibilidad.

  * **Separación de verdad de terreno y extrapolación (*Cold Start*):** `occupancy_pct` conserva exclusivamente la ocupación real observada (`is_estimated = 0`), mientras que `occupancy_estimated` almacena en columna independiente la ocupación reconstruida a partir del perfil basal horario modulada por el contexto diario (`is_estimated = 1`). Las estaciones para las que `/api/patterns` no devuelve curva basal quedan trazadas explícitamente como `UNAVAILABLE_NO_PATTERN`.

### Columnas explícitas del Dataset Gold (`df_gold.parquet`)

* **Claves e Identificación:** `station_id` (Int64), `name` (string), `latitude` (float64), `longitude` (float64), `date_key` (Int64, YYYYMMDD), `timestamp` (datetime64[us, Europe/Madrid]), `hour` (Int64).
* **Métricas Observadas y Estimadas (Target & Features):** `bikes_available` (Int64), `anchors_free` (Int64), `occupancy_pct` (float64, *ground truth* real observado), `occupancy_estimated` (float64, ocupación reconstruida en columna separada).
* **Calidad y Trazabilidad:** `is_estimated` (Int64, flag binario `0` real / `1` extrapolado), `cold_start_status` (string: `OBSERVED`, `COLD_START_ESTIMATED`, `UNAVAILABLE_NO_PATTERN`).
* **Contexto Temporal y Macro:** `day_type_effective` (string), `is_holiday` (Int64), `demand_score` (float64), `city_avg_occupancy` (float64), `balance_index` (float64).
* **Basales y Estrés Operativo:** `occupancy_avg` (float64), `bikes_avg` (float64), `turnover_score` (float64), `empty_hours` (float64), `full_hours` (float64), `problem_hours` (float64).

---

## 5. Riesgos, limitaciones e implicaciones para el modelado y la aplicación

* **Interpretación de nulos en retardos largos (`occ_lag_24h` y `occ_lag_7d`):** Debido al arranque gradual del *scheduler* (`31/10` – `02/11`), los retardos de $-24\text{ h}$ y $-7\text{ días}$ presentan valores ausentes en la primera semana de histórico, alcanzando cobertura superior al **97 %** a partir del **10 de noviembre**. Por ello, en la fase predictiva se emplean alineaciones temporales con tolerancia estricta (`merge_asof`) para no saltar huecos y modelos basados en árboles (`LightGBM`) que gestionan los `NaN` de forma nativa.
* **Disponibilidad temporal de variables agregadas:** Al aplicar `available_at <= timestamp`, las métricas de `rankings` y `patterns` descargadas al final del periodo quedan nulas en las capturas previas de ese lote; en el modelado se compensa calculando perfiles históricos causales propios sobre la serie observada (`is_estimated == 0`).
* **Anomalía puntual de lecturas *offline* en origen (`bikes_available == 0` y `anchors_free == 0`) — Mejora futura documentada:** Durante la auditoría de la Capa Gold (`notebooks/modelado_predictivo.ipynb`, Sección 1.3) y el análisis monográfico de saturación (`notebooks/EDA_prediccion_saturacion.ipynb`), se detectaron episodios minoritarios en los que la API reporta simultáneamente cero bicicletas disponibles y cero anclajes libres (estación temporalmente desconectada o fuera de servicio). En el esquema actual, estos registros producen una caída artificial de `occupancy_pct` al `0 %` y un pico de unidades bloqueadas (`disabled_units = capacity`). Por el momento se mantienen sin filtrar ni reprocesar para preservar la paridad exacta con el *snapshot* auditado, quedando documentada como mejora futura la incorporación de un indicador explícito (`is_offline_anomaly`) en la Capa Silver (`src/etl/cleaning.py`) o su filtrado opcional en `prepare_observed_base()` (`src/modeling/features.py`) antes de entrenar e inferir, con el fin de evitar que estaciones desconectadas generen señales falsas de vaciado para el optimizador de rutas.
* **Criterios de selección de escenarios para demostración en la aplicación:**
  * **Jornada de referencia completa (`100 %` cobertura, 144/144 franjas):** **Domingo `08/11/2026`**, idónea para visualizar un ciclo completo de 24 horas de fin de semana sin interrupciones.
  * **Jornadas laborables de referencia (`98,61 %` – `99,31 %` cobertura):** **Martes `03/11/2026`** (143/144 franjas), **martes `10/11/2026`**, **jueves `12/11/2026`** o **martes `17/11/2026`** (142/144 franjas), recomendadas para demostrar las horas punta pendulares (07:00–09:30 y 17:30–20:00) y la simulación de rutas de rebalanceo.
  * **Tramo descartado para demos:** El bloque inicial (`31/10` – `02/11`), al concentrar las interrupciones de puesta en marcha del *collector*.
* **Ausencia de micro-datos origen-destino:** Se trabaja con estados de estación y agregaciones de flujo neto, manteniendo `mobility_raw.json` aislado en Bronze hasta validar su estabilidad.