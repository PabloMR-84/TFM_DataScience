# Entrega 02: Datos y Preparación — SmartRebalance ZGZ

## 1. Qué datos necesita vuestro producto
- **Observación analítica:** Representa el estado instantáneo y la disponibilidad de una estación Bizi Zaragoza en una franja horaria/instantánea concreta.
- **Datos imprescindibles:** Identificador de estación, coordenadas (lat, lon), capacidad total, bicicletas disponibles (`bikes_available`), anclajes libres (`anchors_free`) y marca temporal (`timestamp`).
- **Granularidad e histórico:** Capturas cada 10 minutos para la serie temporal histórica y agregaciones horarias ($t$).
- **Horizonte de anticipación ($h$):** 
  - *Corto plazo ($h = 1\text{ hora}$):* Detección inminente de desabastecimiento o saturación.
  - *Táctico ($h = 2\text{ a } 3\text{ horas}$):* Planificación previa de rutas de rebalanceo antes de las horas punta.
- **Diferenciación metodológica clave:** Las transferencias sugeridas por la API (`bikesToMove`) no representan la realidad observada, sino una recomendación externa. La verdad de terreno (*ground truth*) para entrenar y evaluar los modelos predictivos es exclusivamente la ocupación real física observada en la estación.

## 2. Fuentes reales y viabilidad
- **Fuente principal:** API pública de DatosBizi Zaragoza (`https://datosbizi.com/api`).
- **Acceso:** Descarga HTTP mediante peticiones REST (JSON) y streaming en bloque para CSVs pesados.
- **Estabilidad y cobertura:** Alta estabilidad (datos actualizados frecuentemente).
- **Plan de contingencia (Fallback):** Si la API no responde, el pipeline cuenta con un modo de ejecución offline que utiliza las muestras locales acumuladas en `data/raw/` y la extrapolación de patrones en la Capa Gold.

## 3. Uso, privacidad y datos personales
- **Licencia:** Datos abiertos urbanos de la red Bizi Zaragoza bajo citación y normativa pública.
- **Privacidad:** No se recopila ni almacena **ningún dato personal**, nombre de usuario, trazabilidad de trayectos individuales ni datos sensibles de ciudadanos. Cumple con el principio de minimización al 100%.

## 4. Del dato bruto al dato utilizable (Flujo Medallion)
- **Bronze (Raw):** Extracción idempotente almacenada en `data/raw/` (`stations_raw.json`, `rankings_raw.json`, `rebalancing_raw.json`, `patterns_raw.json`, `bizi_history_raw.csv` y parquets históricos en `/historical`).
- **Silver:** Limpieza de duplicados, tipado explícito (`Int64` entero anulable de Pandas para preservar nulos reales sin distorsionar promedios, `float64`), desanidado de JSONs internos, validación de rangos de ocupación y conversión estandarizada a la zona horaria `Europe/Madrid` guardados en `data/silver/`.
- **Gold (`df_gold.parquet`):** Modelo Dimensional en Estrella (`Dim_Stations`, `Dim_Patterns`, `Dim_Calendar`, `Fact_StationSnapshots`). Resuelve el *cold start* combinando basales horarios con escalados macro de la ciudad, incorporando la bandera `is_estimated` (`1` para extrapolaciones, `0` para lecturas reales) y controles relacionales estrictos (`validate="m:1"`) para evitar duplicaciones y *data leakage*.

### Columnas explícitas del Dataset Gold (`df_gold.parquet`):
- **Claves e Identificación:** `station_id` (Int64), `date_key` (Int64, YYYYMMDD), `timestamp` (datetime64), `hour` (Int64).
- **Métricas Observadas (Target & Features):** `bikes_available` (Int64), `anchors_free` (Int64), `occupancy_pct` (float64, ratio real de ocupación).
- **Calidad y Trazabilidad:** `is_estimated` (Int64, flag binario de extrapolación).
- **Contexto y Basales:** `day_type_effective` (string), `is_holiday` (Int64), `occupancy_avg` (float64), `demand_score` (float64), `turnover_score` (float64), `problem_hours` (float64).

## 5. Riesgos y limitaciones
- **Falta de datos:** Ausencia de micro-datos de rutas individuales origen-destino exactas (se trabaja con agregaciones y recomendaciones).
- **Sesgos:** Estacionalidad por condiciones meteorológicas extremas o eventos festivos atípicos.
- **Estrategia ante caídas de fuente:** Uso de los basales de `Dim_Patterns` e imputación sobre medias históricas si la API quedase inaccesible indefinidamente.