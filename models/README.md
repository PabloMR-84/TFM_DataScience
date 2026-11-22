# 🤖 Artefactos de Modelos y Model Card (`models/`)

Este directorio almacena los modelos entrenados en producción y sus metadatos de trazabilidad para **SmartRebalance ZGZ**.

## 🗂️ Estructura de Directorios

```text
models/
├── lgbm_v1/                  # Versión actual del modelo campeón (LightGBM multi-horizonte)
│   ├── lgbm_1h.joblib        # [Ignorado en Git] Modelo serializado para horizonte +1h
│   ├── lgbm_2h.joblib        # [Ignorado en Git] Modelo serializado para horizonte +2h
│   ├── lgbm_3h.joblib        # [Ignorado en Git] Modelo serializado para horizonte +3h
│   └── metadata.json         # [Versionado en Git] Huellas SHA-256, hiperparámetros y métricas
└── README.md                 # Documentación de artefactos y Model Card
```

> **Política de control de versiones:** Los ficheros binarios (`*.joblib`, `*.pkl`) están excluidos en `.gitignore` para mantener ligero el repositorio. El archivo `metadata.json` **sí se versiona en Git**, garantizando la trazabilidad exacta entre los datos de entrenamiento (`data_sha256`), los hiperparámetros y las métricas obtenidas en cada ejecución.

---

## 📋 Model Card — `lgbm_v1` (LightGBM Multi-Horizonte)

### 1. Descripción y Uso Previsto
* **Arquitectura:** Tres regresores independientes `LGBMRegressor` (LightGBM) con hiperparámetros optimizados mediante Optuna para los horizontes tácticos $h \in \{1\text{ h}, 2\text{ h}, 3\text{ h}\}$.
* **Variable objetivo:** Tasa de ocupación física real futura por estación ($\hat{y}_{s, t+h} \in [0, 1]$).
* **Uso previsto en producto:**
  * **Horizontes $+2\text{ h}$ y $+3\text{ h}$ (planificación táctica):** Alimentar el motor de optimización de rutas de la flota de rebalanceo (6 furgonetas de 22 plazas en operación 24 h) para anticipar vaciados antes de las horas punta.
  * **Horizonte $+1\text{ h}$ (referencia a corto plazo):** Complementar el estado real observado. Para intervenciones reactivas inmediatas, la ocupación actual ya concentra la mayor parte de la señal.
* **Tipo de inferencia:** *Batch* (ejecutada bajo demanda o por ciclo mediante `scripts/predict.py`).

### 2. Datos de Entrenamiento y Validación
* **Fuente:** Capa Gold de producción (`data/gold/df_gold.parquet`, `dim_stations.parquet`, `dim_calendar.parquet`), filtrando exclusivamente registros físicos reales (`is_estimated == 0`).
* **Cobertura de referencia (`lgbm_v1`):** 276 estaciones activas de Zaragoza durante 23 días naturales (`31 oct – 22 nov`), con partición cronológica estricta (`70 % Train / 15 % Validación / 15 % Test`) y ventanas de embargo de $3\text{ horas}$ entre bloques para evitar fuga de datos (*data leakage*).
* **Predictores:** 30 variables por horizonte (27 base de estado físico, retardos de $10\text{ min}$ a $7\text{ días}$, perfil basal causal, saltos logísticos y calendario cíclico, más 3 variables cíclicas de la hora objetivo $t + h$).

### 3. Métricas de Evaluación en Test (*Out-of-Time*)

| Horizonte | MAE (LightGBM) | MAE (Baseline Naive) | RMSE | $R^2$ | F1 Vacía ($< 20\%$) | F1 Saturada ($> 80\%$) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **$+1\text{ h}$** | **0,0938** | 0,0999 | 0,1335 | 0,7333 | 0,8464 | 0,3888 |
| **$+2\text{ h}$** | **0,1190** | 0,1411 | 0,1624 | 0,6058 | 0,7921 | 0,2348 |
| **$+3\text{ h}$** | **0,1318** | 0,1702 | 0,1755 | 0,5383 | 0,7573 | 0,1702 |

### 4. Limitaciones Conocidas
* **Subestimación en estaciones saturadas ($> 80\%$):** Debido al fuerte desbalanceo de la red (~4,8 % de registros saturados frente a ~50 % vacíos), el regresor suaviza los picos bruscos de llenado y presenta menor F1 en saturación que la persistencia (*Naive*).
* **Estacionalidad limitada:** El entrenamiento actual cubre 23 días de otoño; cambios estacionales (invierno, verano, festividades atípicas o meteorología extrema) requieren reentrenamiento periódico sobre la capa Gold actualizada.
* **Ruido por intervenciones manuales:** Las cargas y descargas de las furgonetas generan saltos abruptos no atribuibles a la demanda de usuarios, parcialmente mitigados mediante variables *proxy* de saltos logísticos.

---

## 🔍 Trazabilidad y Regeneración (`metadata.json`)

Cada entrenamiento guarda en `models/lgbm_v1/metadata.json`:
* `data_sha256`: Huella digital SHA-256 exacta de `df_gold.parquet`, `dim_stations.parquet` y `dim_calendar.parquet`.
* `dataset_summary`: Rango temporal, volumen de filas y número de estaciones.
* `station_categories`: Niveles categóricos exactos de `station_id_cat` para garantizar consistencia entre entrenamiento e inferencia.
* `hyperparameters` y `metrics`: Configuración exacta y resultados en Validación y Test frente al *baseline* Naive.

Para regenerar los binarios `.joblib` localmente a partir de `data/gold/`:

```bash
python scripts/train_model.py
```