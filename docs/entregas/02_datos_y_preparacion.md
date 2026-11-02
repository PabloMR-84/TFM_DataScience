# Entrega 02: Datos y Preparación — SmartRebalance ZGZ

## 1. Qué datos necesita vuestro producto
- **Observación analítica:** Representa el estado instantáneo y la disponibilidad de una estación Bizi Zaragoza en una franja horaria/instantánea concreta.
- **Datos imprescindibles:** Identificador de estación, coordenadas (lat, lon), capacidad total, bicicletas disponibles, anclajes libres y marca temporal (timestamp).
- **Datos deseables:** Indicadores de rotación (`turnoverScore`), horas de estrés (`emptyHours`, `fullHours`), transferencias sugeridas (`bikesToMove`) y contexto macro diario (`city_avg_occupancy`).
- **Granularidad e histórico:** Capturas cada 10 minutos para la serie temporal histórica y agregaciones horarias/diarias.

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
- **Silver:** Limpieza de duplicados, tipado explícito (`int64`, `float64`), manejo de nulos, desanidado de JSONs internos y conversión estandarizada a la zona horaria `Europe/Madrid` guardados en `data/silver/`.
- **Gold:** Construcción de un Modelo Dimensional en Estrella (`Dim_Stations`, `Dim_Patterns`, `Dim_Calendar`, `Fact_StationSnapshots`) y generación del dataset unificado `df_gold.parquet`, solucionando el problema del *cold start* combinando basales horarios con escalados macro de la ciudad.

## 5. Riesgos y limitaciones
- **Falta de datos:** Ausencia de micro-datos de rutas individuales origen-destino exactas (se trabaja con agregaciones y recomendaciones).
- **Sesgos:** Estacionalidad por condiciones meteorológicas extremas o eventos festivos atípicos.
- **Estrategia ante caídas de fuente:** Uso de los basales de `Dim_Patterns` e imputación sobre medias históricas si la API quedase inaccesible indefinidamente.