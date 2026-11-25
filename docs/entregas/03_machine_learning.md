# Entrega 03 — Machine Learning: Predicción Multi-Horizonte de Ocupación (`SmartRebalance ZGZ`)

**Proyecto:** SmartRebalance ZGZ (Sistema de Redistribución Inteligente para Bizi Zaragoza)

**Archivo de entregable:** `docs/entregas/03_machine_learning.md`

**Cuadernos de experimentación asociados:** `notebooks/modelado_predictivo.ipynb` y `notebooks/EDA_prediccion_saturacion.ipynb` (sobre el snapshot reproducible `data/gold_snapshot_ml_20261122/`)

**Módulos de producción asociados:** `src/modeling/` (`features.py`, `train.py`, `predict.py`) y orquestadores `scripts/train_model.py` / `scripts/predict_model.py`

## 1. Qué problema resuelve el modelo

### 1.1. Necesidad del producto y usuarios finales

El sistema público de bicicleta compartida **Bizi Zaragoza** cuenta con **276 estaciones** distribuidas por toda la ciudad, pero dispone de una flota logística altamente restringida: únicamente **6 furgonetas** operando en turnos de **24 horas** con una capacidad física máxima de **22 bicicletas por vehículo**.

Si los operarios actúan únicamente de forma reactiva (acudiendo cuando una estación ya marca $0\%$ o $100\%$ de ocupación), llegan tarde al pico de demanda. El módulo de Machine Learning resuelve un problema de **predicción supervisada de series temporales tabulares multi-horizonte** que anticipa la ocupación futura de cada estación a $+1\text{ hora}$ (alerta inminente), $+2\text{ horas}$ y $+3\text{ horas}$ (planificación táctica de rutas).

* **Quién utiliza el resultado:**

  1. **Motor prescriptivo de optimización de rutas (Google OR-Tools):** Consume las predicciones continuas $\hat{y}_{s, t+h}$ y las banderas de alerta calibradas combinadas con la ocupación actual $y_{s, t}$ mediante un parámetro de régimen $\tau_{\text{mode}} \in [0, 1]$ para calcular la demanda neta de intervención por estación y asignar paradas óptimas a las 6 furgonetas.

  2. **Jefes de turno y operarios de logística:** Visualizan en el panel de control qué estaciones entrarán en riesgo de vaciado ($< 20\%$) o saturación ($> 80\%$) antes de que ocurra el colapso.

## 2. Datos de entrada y salida

Partiendo de la **Capa Gold** consolidada en la Entrega 02 (`df_gold.parquet`, con **782.736 registros 100 % reales** filtrados mediante `is_estimated == 0` a granularidad de 10 minutos durante **23 días naturales**, entre el `31/10/2026` y el `22/11/2026`, sobre **276 estaciones** y **2.836 instantes de captura**), se construye una matriz supervisada de **766.610 muestras válidas** multi-horizonte tras alinear los retardos y los horizontes futuros.

### 2.1. Variable objetivo (Salida esperada)

Para cada estación $s$ e instante de consulta $t$, se entrenan tres modelos independientes correspondientes a los horizontes tácticos $h \in \{1\text{ h}, 2\text{ h}, 3\text{ h}\}$:

* **`target_occ_1h`, `target_occ_2h`, `target_occ_3h`:** Porcentaje de ocupación física real observada (`occupancy_pct = bikes_available / capacity`, con valores en $[0, 1]$) alineado mediante `merge_asof` con tolerancia estricta de $10\text{ minutos}$ en $+1\text{ h}$, $+2\text{ h}$ y $+3\text{ h}$.

* **Desacoplamiento de recomendaciones externas:** Los valores sugeridos por la API de origen (`bikesToMove`) nunca se emplean como verdad observada (*ground truth*).

### 2.2. Variables de entrada disponibles en el momento de uso ($\le t$)

La ingeniería de características está centralizada en `src/modeling/features.py` (30 predictores por horizonte compartidos entre entrenamiento e inferencia para evitar *training-serving skew*) bajo **causalidad estricta**:

* **Estado físico actual y capacidad ($t$):** `occupancy_pct`, `bikes_available`, `anchors_free`, `capacity` y `disabled_units` (unidades bloqueadas por avería o mantenimiento: `capacity - bikes_available - anchors_free`).

* **Ubicación e identidad espacial:** `station_id_cat` (variable categórica nativa de las 276 estaciones), `latitude` y `longitude`.

* **Retardos, velocidad y volatilidad ($\le t$):** Retardos exactos mediante `merge_asof` (`occ_lag_10m`, `occ_lag_30m`, `occ_lag_1h`, `occ_lag_24h`, `occ_lag_7d`), velocidad de cambio reciente (`delta_occ_30m`, `delta_occ_1h`) y volatilidad móvil de la última hora (`rolling_std_1h`).

* **Perfil basal causal ($\le t$):** `causal_hist_occ_mean` (media histórica acumulada usando exclusivamente días previos del mismo `(station_id, day_type_effective, hour)`) y desviación frente al patrón esperado (`occ_diff_vs_hist_mean`).

* **Huella logística de furgonetas ($\le t$):** Variación brusca en 10 minutos (`delta_bikes_10m`), indicador de salto logístico (`is_logistic_jump` cuando $|\Delta \text{bicis}_{10\text{m}}| \ge 8$) y acumulado en la última hora (`logistic_jumps_last_1h`).

* **Codificación cíclica y calendario determinista ($t$ y $t+h$):** Componentes trigonométricas de hora y día de la semana (`hour_sin`, `hour_cos`, `dow_sin`, `dow_cos`), indicadores de fin de semana y festivo (`is_weekend_flag`, `is_holiday`) y variables conocidas de la hora objetivo (`target_hour_sin_{h}h`, `target_hour_cos_{h}h`, `target_is_weekend_{h}h`).

### 2.3. Control de riesgos, *Data Leakage* y calidad

1. **Prevención de fuga temporal (*Anti-Leakage*):** Ninguna variable agregada de `dim_calendar` ni perfiles globales estáticos de `dim_patterns` se incorporan vulnerando su marca de disponibilidad temporal (`available_at <= timestamp`). El perfil basal `causal_hist_occ_mean` se calcula de forma estrictamente expansiva (*expanding window*), y la calibración de umbrales de saturación se ejecuta exclusivamente sobre `df_val`.

2. **Trazabilidad criptográfica (SHA-256):** El módulo de producción (`src/modeling/train.py`) calcula y persiste el hash SHA-256 de `df_gold.parquet`, `dim_stations.parquet` y `dim_calendar.parquet` dentro de `models/lgbm_v1/metadata.json`.

3. **Auditoría de lecturas *offline*:** La Sección 1.3 de `notebooks/modelado_predictivo.ipynb` audita las capturas puntuales donde la API reporta simultáneamente `bikes_available == 0` y `anchors_free == 0`, dejándolas trazadas como mejora futura del filtro de calidad en ETL (documentado en la Entrega 02).

## 3. Baseline y selección de modelos

Para justificar rigurosamente la complejidad algorítmica, se contrastan tres niveles de modelado sobre exactamente las mismas particiones:

1. **Baselines de referencia:**

   * **Baseline 1 — Persistencia (*Naive*):** Asume que la ocupación futura será idéntica a la actual ($\hat{y}_{s, t+h} = y_{s, t}$). Es una referencia exigente a $+1\text{ h}$ por la autocorrelación de la serie, pero degrada rápidamente a $+2\text{ h}$ y $+3\text{ h}$.

   * **Baseline 2 — Perfil Histórico Causal:** Predice la media histórica acumulada hasta $t$ para esa estación, tipo de día y hora objetivo.

   * **Baseline 3 — Regresión Lineal Regularizada (`Ridge`):** Modelo lineal regularizado con estandarización numérica y codificación de estaciones.

2. **Alternativas de Machine Learning no lineal:**

   * **Random Forest Regressor:** Ensamble de árboles de decisión con control de profundidad y tamaño mínimo de hoja.

   * **XGBoost Regressor:** *Gradient Boosting* basado en histogramas con *early stopping*.

   * **LightGBM Regressor (Base y optimizado con Optuna):** *Gradient Boosting* orientado a hojas con soporte nativo para `station_id_cat`, función objetivo `regression` evaluada por `MAE`, *early stopping* sobre `df_val` y optimización bayesiana de hiperparámetros con **Optuna** específica para cada horizonte ($+1\text{ h}, +2\text{ h}, +3\text{ h}$).

## 4. Validación y evaluación

### 4.1. Estrategia de partición temporal con embargo

Para evitar cualquier contaminación entre conjuntos por el solapamiento de los horizontes futuros ($+3\text{ h}$), se aplica una **partición cronológica estricta (`70 % / 15 % / 15 %`)** sobre marcas temporales completas, separada por **ventanas de embargo de $3\text{ horas}$** (`10.488` registros descartados en fronteras):

* **Entrenamiento (`df_train`):** `531.458` registros.

* **Validación (`df_val`):** `109.848` registros, utilizado para *early stopping*, selección de arquitectura, optimización de hiperparámetros con **Optuna** y **calibración *anti-leakage* de los umbrales de alerta de saturación**.

* **Test ciego (`df_test`):** `114.816` registros (`44,23 %` laborables / `55,77 %` fin de semana), desbloqueado únicamente al final para evaluar la generalización real fuera de muestra.

### 4.2. Métricas orientadas al negocio

* **Métricas continuas:** **MAE** (error absoluto medio en puntos de ocupación, directamente traducible a bicicletas mediante $\text{MAE} \times \text{capacidad}$), **RMSE** y $R^2$.

* **Métricas operativas de eventos críticos:** **F1-Score**, **Precisión** y **Recall** en la detección temprana de **estación vacía ($< 20\%$)** y **estación saturada ($> 80\%$)**, auditando tanto el umbral base fijo (`0,80`) como el umbral calibrado en `df_val` para cada horizonte.

### 4.3. Resultados en Validación y Selección del Modelo Campeón

Sobre el histórico ampliado a 23 días, **LightGBM optimizado con Optuna** supera de forma consistente a todos los *baselines*, Random Forest, XGBoost y la configuración base en `df_val`, alcanzando un MAE de validación de **`0,0903`** en $+1\text{ h}$, **`0,1120`** en $+2\text{ h}$ y **`0,1231`** en $+3\text{ h}$.

### 4.4. Resultados en Evaluación Ciega (`df_test`)

La siguiente tabla resume el rendimiento fuera de muestra del modelo campeón **LightGBM (Optuna)** frente a la Persistencia (*Naive*), incluyendo el impacto de la calibración de umbrales de saturación validada en `notebooks/EDA_prediccion_saturacion.ipynb`:

| Horizonte | Val MAE | Test MAE (LightGBM) | Test MAE (Naive) | Test RMSE | Test $R^2$ | F1 Vacía ($< 20\%$) | Recall Vacía ($< 20\%$) | F1 Sat. Base ($> 80\%$) | Umbral Calibrado (`df_val`) | F1 Sat. Calibrada (`df_test`) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$+1\text{ h}$** | `0,0903` | **`0,0938`** | `0,0999` | `0,1335` | `0,7333` | `0,8464` | `0,8490` | `0,3888` | `$74,0\%$` | **`0,4985`** |
| **$+2\text{ h}$** | `0,1120` | **`0,1190`** | `0,1411` | `0,1624` | `0,6058` | `0,7921` | `0,7675` | `0,2348` | `$70,0\%$` | **`0,4018`** |
| **$+3\text{ h}$** | `0,1231` | **`0,1318`** | `0,1702` | `0,1755` | `0,5383` | `0,7573` | `0,7115` | `0,1702` | `$65,0\%$` | **`0,3694`** |

* **Contraste frente a los *baselines*:**
  * En **$+1\text{ h}$**, LightGBM supera a *Naive* en MAE (`0,0938` vs. `0,0999`), RMSE (`0,1335` vs. `0,1592`), $R^2$ (`0,7333` vs. `0,6212`) y F1 Vacía (`0,8464` vs. `0,8339`), si bien la ganancia en MAE es moderada (`0,61 pp`) debido a la fuerte inercia a muy corto plazo.
  * En los **horizontes tácticos $+2\text{ h}$ y $+3\text{ h}$**, el aprendizaje automático aporta el mayor valor diferencial: en $+2\text{ h}$ reduce el MAE a `0,1190` con $R^2 = 0,6058$ frente a *Naive* (`MAE = 0,1411`, $R^2 = 0,3230$) y *Ridge* (`MAE = 0,1430`, $R^2 = 0,4527$); en $+3\text{ h}$ mantiene `MAE = 0,1318` y $R^2 = 0,5383$ frente al colapso de *Naive* (`MAE = 0,1702`, $R^2 = 0,0713$).

### 4.5. Auditoría de errores, explicabilidad (SHAP) y análisis monográfico de saturación

* **Explicabilidad global (SHAP):** El análisis `TreeExplainer` confirma que el modelo aprende relaciones físicas coherentes, combinando la inercia reciente (`occupancy_pct` y retardos cortos), la identidad espacial de cada estación (`station_id_cat`), la velocidad de cambio reciente y las componentes cíclicas horarias tanto del instante actual ($t$) como del horizonte objetivo ($t + h$).

* **Patrón horario del error:** El MAE alcanza su mínimo de madrugada (`02:00–04:00`: ~`5,4 %` en $+1\text{ h}$, ~`8,4 %` en $+2\text{ h}$ y ~`10,2 %` en $+3\text{ h}$) y crece progresivamente hasta su máximo en la franja vespertina (`18:00–20:00`: ~`12,2 %` en $+1\text{ h}$, ~`14,5 %` en $+2\text{ h}$ y ~`15,8 %` en $+3\text{ h}$), coincidiendo con los picos de retorno y ocio.

* **Contraste espacial en $+2\text{ h}$:** Los mayores errores se concentran en nodos de altísima rotación (`75-Salvador Allende` con MAE = `0,2239`, `109-H. Miguel Servet` con `0,2063`, `91-S. Francisco: Fdo. Católico` con `0,2046`, `60-Bielsa: Cnos. Norte` con `0,1970` y `86-Almozara: Batalla Arapiles` con `0,1958`), mientras que las estaciones periféricas de baja variabilidad registran errores inferiores al `5 %` (`231-Heroínas Sitios: Gómara` con `0,0344`, `218-Desayuno Diamantes` con `0,0390`, `232-Cañones Zaragoza: Tella` con `0,0445`, `233-21 de Junio: Miguelete` con `0,0456` y `226-Camping Rosales` con `0,0466`).

* **Diagnóstico monográfico de estaciones saturadas (`notebooks/EDA_prediccion_saturacion.ipynb`):**
  * **Causa raíz:** Dado que los episodios saturados ($> 80\%$) representan ~`4,8 %` de la red, el regresor entrenado bajo MAE suaviza los picos extremos y subestima sistemáticamente la ocupación en episodios de llenado real (media real del `91,0 %`) en **`-20,24 pp` ($+1\text{ h}$)**, **`-28,61 pp` ($+2\text{ h}$)** y **`-33,31 pp` ($+3\text{ h}$)**.
  * **Solución adoptada (*Threshold Tuning* dinámico sobre `df_val`):** Mantener intacta la predicción continua `pred_lgbm_{h}h` (para no degradar el MAE ni distorsionar el volumen de bicicletas a mover) y calibrar automáticamente en `df_val` el umbral de alerta de saturación por horizonte (`74,0 %`, `70,0 %` y `65,0 %` en el *snapshot* auditado). Esto eleva el **F1 Saturada** en `Test` hasta **`0,4985` ($+1\text{ h}$)**, **`0,4018` ($+2\text{ h}$)** y **`0,3694` ($+3\text{ h}$)**, manteniendo ocupaciones reales medias elevadas (`60,9 %`–`65,5 %`) en los falsos positivos.
  * **Descarte de arquitectura dual en producción:** El contraste experimental frente a un clasificador binario auxiliar (`LGBMClassifier`) mostró mejoras marginales en los horizontes tácticos (`+0,0195` en $+2\text{ h}$ y `+0,0095` en $+3\text{ h}$) que no justifican duplicar la arquitectura de 3 a 6 modelos.

* **Otras limitaciones identificadas:** Ausencia de telemetría GPS directa de las furgonetas (mitigada parcialmente mediante las variables *proxy* de saltos logísticos) y presencia minoritaria de lecturas *offline* (`bikes_available == 0` y `anchors_free == 0`), cuya gestión futura queda documentada en la Entrega 02.

## 5. Decisión y uso en el producto

### 5.1. Solución elegida y puesta en producción (`src/modeling/`)

Se selecciona y modulariza en producción **`LightGBMRegressor` multi-horizonte optimizado con Optuna ($+1\text{ h}, +2\text{ h}, +3\text{ h}$)** junto con la **calibración automática en Validación de umbrales de alerta de saturación**:

* **Arquitectura empaquetada:**
  * Entrenamiento trazable: `src/modeling/train.py` / `scripts/train_model.py` (`smartrebalance-train`), que genera `models/lgbm_v1/lgbm_{1,2,3}h.joblib` y `models/lgbm_v1/metadata.json` (con hashes SHA-256, umbrales calibrados y métricas `mae`, `rmse`, `r2`, `f1_empty`, `recall_empty`, `precision_full`, `recall_full`, `f1_full` y `f1_full_base_80`).
  * Inferencia *batch* desacoplada: `src/modeling/predict.py` / `scripts/predict_model.py` (`smartrebalance-predict`), que exporta `data/predictions/predictions_lgbm.parquet` sin reentrenar.
* **Suficiencia para el MVP:** El error medio de ~`1,8–2,4` bicicletas en los horizontes $+1\text{ h}$ y $+2\text{ h}$, unido a una detección sólida de vaciados (`F1 Vacía` entre `0,7573` y `0,8464`) y a las alertas de saturación calibradas por horizonte, ofrece una base operativa robusta para priorizar las paradas de las 6 furgonetas aplicando el umbral de rentabilidad de intervención ($|q_s| \ge 4\text{ bicicletas}$).

### 5.2. Contrato de salida en inferencia (`data/predictions/predictions_lgbm.parquet`)

El archivo de salida generado por `scripts/predict_model.py` proporciona directamente las predicciones continuas y las alertas binarias calibradas para el motor prescriptivo de optimización de rutas y el panel de control:

| Campo | Tipo | Significado y uso en el producto |
| :--- | :--- | :--- |
| `station_id` | `int` | Identificador único de la estación (cruce con `dim_stations` para coordenadas GPS y capacidad). |
| `timestamp` | `datetime` | Marca temporal de emisión de la predicción (`Europe/Madrid`). |
| `hour` / `day_type_effective` | `int` / `str` | Contexto horario y tipo de día efectivo (`WEEKDAY` / `WEEKEND`). |
| `occupancy_pct` | `float [0, 1]` | Ocupación física real observada en el instante actual $t$. |
| `bikes_available` / `anchors_free` | `int` | Disponibilidad física real observada en $t$. |
| `pred_lgbm_1h`, `pred_lgbm_2h`, `pred_lgbm_3h` | `float [0, 1]` | Predicción continua de ocupación a $+1\text{ h}$, $+2\text{ h}$ y $+3\text{ h}$ (insumo para calcular $y_{s}^{\text{eff}}(\tau_{\text{mode}})$ y bicicletas esperadas). |
| `pred_empty_1h`, `pred_empty_2h`, `pred_empty_3h` | `int8 {0, 1}` | Bandera de alerta de vaciado (`pred_lgbm_{h}h < 0.20`). |
| `pred_sat_1h`, `pred_sat_2h`, `pred_sat_3h` | `int8 {0, 1}` | Bandera de alerta de saturación evaluada con el umbral calibrado en `df_val` y guardado en `metadata.json` (ej. `> 0.74`, `> 0.70`, `> 0.65`). |

* **Cautelas de uso en el producto:** El optimizador de rutas y el frontal deberán combinar siempre la predicción continua $\hat{y}_{s, t+h}$ con el estado físico actual $y_{s, t}$ mediante el selector de régimen $\tau_{\text{mode}}$ (reduciendo $\tau_{\text{mode}}$ en festivos extraordinarios como Pilares o incidencias imprevistas), utilizar las banderas `pred_sat_{h}h` para priorizar alertas preventivas de llenado y acotar cualquier orden de retirada/depósito a la capacidad física real disponible de la estación en el momento de llegada de la furgoneta.