# ⚙️ Orquestadores y Ejecución de Pipelines (`scripts/`)

Este directorio contiene los puntos de entrada ejecutables para lanzar de forma repetible y desacoplada los tres flujos principales de **SmartRebalance ZGZ**: la preparación de datos (ETL), el entrenamiento del modelo predictivo y la inferencia *batch*.

> **Requisito previo:** Todos los comandos deben ejecutarse desde la **raíz del repositorio (`TFM/`)** con el entorno virtual `.venv` activado y el paquete local registrado (`pip install -e .`).

## 🗂️ Scripts Disponibles

```text
scripts/
├── main_BBDD.py      # Pipeline ETL completo (Bronze -> Silver -> Gold)
├── train_model.py    # Entrenamiento LightGBM multi-horizonte + trazabilidad SHA-256
├── predict.py        # Inferencia batch desacoplada sobre data/gold/
└── README.md         # Guía de ejecución y parámetros CLI
```

---

## ▶️ 1. Pipeline ETL (`main_BBDD.py`)

Procesa los ficheros de `data/raw/` (incluyendo el histórico en `data/raw/historical/`) y regenera desde cero las capas `data/silver/` y `data/gold/`.

```bash
python scripts/main_BBDD.py
```

* **Entrada:** `data/raw/` y API de DatosBizi (o modo *offline* si se reutiliza caché local).
* **Salida:** Tablas limpias en `data/silver/` y modelo en estrella + `df_gold.parquet` + `coverage_report` en `data/gold/`.

---

## ▶️ 2. Entrenamiento del Modelo (`train_model.py`)

Entrena los tres modelos **LightGBM** (`+1h`, `+2h`, `+3h`) sobre `data/gold/`, calcula el hash **SHA-256** de los ficheros de entrada, evalúa en Validación y Test frente al *baseline* Naive y guarda los artefactos y metadatos.

```bash
# Ejecución estándar (lee data/gold/ y guarda en models/lgbm_v1/)
python scripts/train_model.py

# Equivalente mediante entry point CLI definido en pyproject.toml
smartrebalance-train
```

### Argumentos opcionales
| Argumento | Descripción | Valor por defecto |
| :--- | :--- | :--- |
| `--gold-dir` | Directorio con la capa Gold de entrada | `data/gold` |
| `--output-dir` | Directorio destino para modelos `.joblib` y `metadata.json` | `models/lgbm_v1` |

**Ejemplo con rutas personalizadas:**
```bash
python scripts/train_model.py --gold-dir data/gold --output-dir models/lgbm_v1
```

---

## ▶️ 3. Inferencia Batch (`predict.py`)

Carga los modelos entrenados desde `models/lgbm_v1/` (sin reentrenar), reconstruye las características sobre `data/gold/` y exporta las predicciones multi-horizonte de ocupación.

* **Tipo de inferencia:** *Batch* (sin servidor HTTP/online).

```bash
# Ejecución estándar (últimas 72 horas observadas -> data/predictions/predictions_lgbm.parquet)
python scripts/predict.py

# Modo última captura (solo el instante más reciente disponible para las 276 estaciones)
python scripts/predict.py --latest-only

# Equivalente mediante entry point CLI
smartrebalance-predict --last-hours 168
```

### Argumentos opcionales
| Argumento | Descripción | Valor por defecto |
| :--- | :--- | :--- |
| `--gold-dir` | Directorio con la capa Gold de entrada | `data/gold` |
| `--model-dir` | Directorio con `lgbm_{h}h.joblib` y `metadata.json` | `models/lgbm_v1` |
| `--output-path` | Ruta del fichero `.parquet` de salida | `data/predictions/predictions_lgbm.parquet` |
| `--latest-only` | Si se activa, predice únicamente el `timestamp` más reciente | Desactivado |
| `--last-hours` | Ventana de horas recientes a exportar (`0` exporta todo el histórico) | `72` |