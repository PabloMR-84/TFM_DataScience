# TFM_DataScience


## Estructura:

```text
TFM/
├── data/
│   ├── raw/                  # Archivos en bruto descargados de la API de DatosBizi
│   │   ├── historical/       # Subcarpeta para snapshots históricos en formato .parquet
│   │   ├── stations_raw.json # Maestro estático y estado instantáneo
│   │   ├── rankings_raw.json # Indicadores acumulados de estrés
│   │   ├── rebalancing_raw.json # Recomendaciones logísticas operativas
│   │   ├── patterns_raw.json # Basales de ocupación horaria
│   │   └── bizi_history_raw.csv # Contexto macro temporal a nivel de ciudad
│   ├── silver/               # DataFrames estandarizados y limpios (csv/parquet)
│   └── gold/                 # Tablas dimensionales, hechos y df_gold final (parquet)
│
├── src/                      # Código modular de ingeniería de datos
│   ├── __init__.py           # Inicializador del paquete src/
│   ├── config.py             # Configuración central (endpoints, variables de entorno, ROOT_DIR)
│   ├── utils.py              # Herramientas de soporte (parseo de JSON, conversión de tipos, timestamps)
│   ├── io.py                 # Orquestador I/O (descarga interactiva o caché, persistencia)
│   ├── cleaning.py           # Estandarización pura de esquemas (Capa Silver)
│   ├── models_gold.py        # Modelo dimensional, API de festivos y extrapoliación de Cold Start
│   └── features.py           # (Reservado para Feature Engineering en fase ML)
│
├── main_BBDD.py              # Script ejecutable principal del pipeline ETL
├── .env                      # Variables de entorno seguras (BIZI_API_KEY)
└── README.md                 # Documentación y alcance del proyecto
```