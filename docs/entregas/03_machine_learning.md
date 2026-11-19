# Entrega 03 — Machine Learning: Predicción Multi-Horizonte de Ocupación (`SmartRebalance ZGZ`)

**Proyecto:** SmartRebalance ZGZ (Sistema de Redistribución Inteligente para Bizi Zaragoza)

**Archivo de entregable:** `docs/entregas/03_machine_learning.md`

**Cuaderno de experimentación asociado:** `notebooks/modelo_predictivo.ipynb` (sobre snapshot reproducible `data/gold_snapshot_ml_20261115/`)

## 1. Qué problema resuelve el modelo

### 1.1. Necesidad del producto y usuarios finales

El sistema público de bicicleta compartida **Bizi Zaragoza** cuenta con **276 estaciones** distribuidas por toda la ciudad, pero dispone de una flota logística altamente restringida: únicamente **6 furgonetas** operando en turnos de **24 horas** con una capacidad física máxima de **22 bicicletas por vehículo**.

Si los operarios actúan únicamente de forma reactiva (acudiendo cuando una estación ya marca $0\%$ o $100\%$ de ocupación), llegan tarde al pico de demanda. El módulo de Machine Learning resuelve un problema de **predicción supervisada de series temporales tabulares multi-horizonte** que anticipa la ocupación futura de cada estación a $+1\text{ hora}$ (alerta inminente), $+2\text{ horas}$ y $+3\text{ horas}$ (planificación táctica de rutas).

* **Quién utiliza el resultado:**

  1. **Motor prescriptivo de optimización de rutas (Google OR-Tools):** Consume las predicciones $\hat{y}_{s, t+h}$ combinadas con la ocupación actual $y_{s, t}$ mediante un parámetro de régimen $\tau_{\text{mode}} \in [0, 1]$ para calcular la demanda neta de intervención por estación y asignar paradas óptimas a las 6 furgonetas.

  2. **Jefes de turno y operarios de logística:** Visualizan en el panel de control qué estaciones entrarán en riesgo de vaciado ($<20\%$) o saturación ($>80\%$) antes de que ocurra el colapso.

## 2. Datos de entrada y salida

Partiendo de la **Capa Gold** consolidada en la Entrega 02 (`df_gold.parquet`, con **540.960 registros 100 % reales** filtrados mediante `is_estimated == 0` a granularidad de 10 minutos entre el `31/10/2026` y el `15/11/2026`), se construye en el cuaderno `notebooks/modelo_predictivo.ipynb` una matriz supervisada de **530.748 muestras válidas** tras alinear los retardos y horizontes futuros.

### 2.1. Variable objetivo (Salida esperada)

Para cada estación $s$ e instante de consulta $t$, se entrenan tres modelos independientes correspondientes a los horizontes tácticos $h \in \{1\text{ h}, 2\text{ h}, 3\text{ h}\}$:

* **`target_occ_1h`, `target_occ_2h`, `target_occ_3h`:** Porcentaje de ocupación física real observada (`occupancy_pct = bikes_available / capacity`, con valores en $[0, 1]$) exactamente $6$, $12$ y $18$ pasos de 10 minutos en el futuro.

* **Desacoplamiento de recomendaciones externas:** Los valores sugeridos por la API de origen (`bikesToMove`) nunca se emplean como verdad observada (*ground truth*).

### 2.2. Variables de entrada disponibles en el momento de uso ($\le t$)

Todas las *features* están diseñadas bajo **causalidad estricta** (solo utilizan información observada hasta el instante $t$ o determinista del calendario futuro $t+h$):

* **Estado físico actual ($t$):** `occupancy_pct`, `bikes_available`, `anchors_free` y `capacity`.

* **Inercia y dinámica reciente ($\le t$):** Retardos intradía y semanales (`occ_lag_10m`, `occ_lag_30m`, `occ_lag_1h`, `occ_lag_2h`, `occ_lag_24h`, `occ_lag_7d`) y velocidad de cambio reciente (`delta_occ_1h = occupancy_pct - occ_lag_1h`).

* **Contexto espacial e histórico causal:** Identificador categórico de estación (`station_id_cat`, con las 276 estaciones) y perfil basal causal (`causal_hist_occ_mean`), calculado dinámicamente en cada instante $t$ utilizando exclusivamente observaciones pasadas del mismo `(station_id, day_type_effective, hour)`.

* **Codificación cíclica y calendario determinista ($t$ y $t+h$):** Transformaciones trigonométricas de la hora actual (`hour_sin`, `hour_cos`), día de la semana (`dow_sin`, `dow_cos`), indicadores de fin de semana/festivo (`is_weekend`, `is_holiday`) y las variables deterministas conocidas de la hora objetivo (`target_hour_sin_{h}h`, `target_hour_cos_{h}h`, `target_is_weekend_{h}h`).

### 2.3. Control de riesgos, *Data Leakage* y calidad

1. **Prevención de fuga temporal (*Anti-Leakage*):** Ninguna variable agregada diaria de `dim_calendar` (`city_avg_occupancy`, `balance_index`, `demand_score`) ni perfiles globales estáticos de `dim_patterns` se incorporan sin respetar su marca de disponibilidad temporal (`available_at <= timestamp`). El perfil basal `causal_hist_occ_mean` se recalcula de forma estrictamente expansiva (*expanding window*) sin mirar al futuro.

2. **Sesgo de estacionalidad corta (16 días):** El *snapshot* inicial contiene 16 días completos donde el tramo final (`df_test`) coincide mayoritariamente con fin de semana (`85,42 % WEEKEND` frente a `20,00 %` en `df_train`), lo que supone una prueba de estrés exigente ante cambios de régimen semanal.

## 3. Baseline y selección de modelos

Para justificar rigurosamente la complejidad algorítmica, se contrastan tres niveles de modelado sobre exactamente las mismas particiones:

1. **Baselines de referencia:**

   * **Baseline 1 — Persistencia (*Naive*):** Asume que la ocupación futura será idéntica a la actual ($\hat{y}_{s, t+h} = y_{s, t}$). Es una referencia muy exigente a $+1\text{ h}$ por la alta autocorrelación de la serie, pero degrada rápidamente a $+2\text{ h}$ y $+3\text{ h}$.

   * **Baseline 2 — Perfil Histórico Causal:** Predice la media histórica acumulada hasta $t$ para esa estación, tipo de día y hora objetivo. Captura el ciclo diario pero ignora el estado actual de la estación.

   * **Baseline 3 — Regresión Lineal Regularizada (`Ridge`, $\alpha = 10.0$):** Modelo lineal con estandarización numérica y *One-Hot Encoding* de las 276 estaciones.

2. **Alternativas de Machine Learning no lineal:**

   * **Random Forest Regressor:** Ensamble de 120 árboles con profundidad máxima controlada (`max_depth=14`, `min_samples_leaf=25`).

   * **XGBoost Regressor:** *Gradient Boosting* basado en histogramas (`hist`, `learning_rate=0.05`, `max_depth=7`) con *early stopping*.

   * **LightGBM Regressor:** *Gradient Boosting* orientado a hojas (`num_leaves=63`, `learning_rate=0.05`, `min_child_samples=30`, `subsample=0.8`, `colsample_bytree=0.8`) con soporte nativo para la variable categórica `station_id_cat` y *early stopping* a 40 rondas.

## 4. Validación y evaluación

### 4.1. Estrategia de partición temporal con embargo

Para evitar cualquier contaminación entre conjuntos por el solapamiento de los horizontes futuros ($+3\text{ h}$), se aplica una **partición cronológica estricta (`70 % / 15 % / 15 %`)** separada por **ventanas de embargo de $3\text{ horas}$** en cada frontera:

* **Entrenamiento (`df_train`):** `370.944` registros (`70 %`).

* **Validación (`df_val`):** `79.212` registros (`15 %`), utilizado para selección de arquitectura, *early stopping* y ajuste de hiperparámetros con **Optuna** (20 *trials* TPE por horizonte).

* **Test ciego (`df_test`):** `79.488` registros (`15 %`), desbloqueado únicamente al final para medir la capacidad de generalización real fuera de muestra.

### 4.2. Métricas orientadas al negocio

* **Métricas continuas:** **MAE** (error absoluto medio en puntos de ocupación, directamente traducible a número de bicicletas mediante $\text{MAE} \times \text{capacidad}$), **RMSE** y $R^2$.

* **Métricas operativas de eventos críticos:** **F1-Score** y **Recall** en la detección temprana de **estación vacía ($<20\%$)** y **estación saturada ($>80\%$)**, donde el *Recall* de vaciado es crítico para que la furgoneta acuda antes de que el usuario se quede sin bicicletas.

### 4.3. Resultados en Validación y Selección del Modelo Campeón

En `df_val`, **LightGBM** supera consistentemente a todos los *baselines*, Random Forest y XGBoost en los tres horizontes (reduciendo el MAE frente a Naive en un **38,1 %** a $+2\text{ h}$ y un **45,5 %** a $+3\text{ h}$). Tras ejecutar una búsqueda bayesiana con **Optuna** (20 *trials* por horizonte), se comprobó que la configuración base de LightGBM (`num_leaves=63`, `min_child_samples=30`) mantenía un MAE ligeramente mejor en validación (**0,0909** vs. **0,0920** en $+1\text{ h}$; **0,1148** vs. **0,1156** en $+2\text{ h}$; **0,1264** vs. **0,1294** en $+3\text{ h}$), por lo que bajo criterio empírico estricto se seleccionó el **LightGBM base** como modelo campeón.

### 4.4. Resultados en Evaluación Ciega (`df_test`)

| Horizonte | Modelo | MAE | RMSE | $R^2$ | F1 Vacía ($<20\%$) | Recall Vacía ($<20\%$) | F1 Saturada ($>80\%$) | 
| ----- | ----- | ----- | ----- | ----- | ----- | ----- | ----- | 
| $+1\text{ h}$ | Baseline 1: Persistencia (Naive) | **0,0916** | 0,1402 | 0,6792 | **0,8157** | 0,8167 | **0,5789** | 
| $+1\text{ h}$ | Baseline 3: Regresión Ridge | 0,0970 | 0,1335 | 0,7090 | 0,8045 | 0,7905 | 0,3040 | 
| $+1\text{ h}$ | **Modelo Campeón: LightGBM** | 0,0938 | **0,1296** | **0,7259** | 0,8108 | **0,8277** | 0,4048 | 
| $+2\text{ h}$ | Baseline 1: Persistencia (Naive) | 0,1313 | 0,1915 | 0,4021 | 0,7311 | **0,7322** | **0,3897** | 
| $+2\text{ h}$ | Baseline 3: Regresión Ridge | 0,1336 | 0,1753 | 0,4988 | 0,7030 | 0,6548 | 0,0048 | 
| $+2\text{ h}$ | **Modelo Campeón: LightGBM** | **0,1230** | **0,1644** | **0,5594** | **0,7357** | 0,7219 | 0,1832 | 
| $+3\text{ h}$ | Baseline 1: Persistencia (Naive) | 0,1598 | 0,2270 | 0,1612 | 0,6690 | **0,6710** | **0,2847** | 
| $+3\text{ h}$ | Baseline 3: Regresión Ridge | 0,1563 | 0,2010 | 0,3424 | 0,6130 | 0,5354 | 0,0000 | 
| $+3\text{ h}$ | **Modelo Campeón: LightGBM** | **0,1402** | **0,1844** | **0,4464** | **0,6845** | 0,6510 | 0,1542 | 

### 4.5. Auditoría de errores, explicabilidad (SHAP) y limitaciones

* **Explicabilidad global (SHAP en $+2\text{ h}$):** El análisis `TreeExplainer` confirma que el modelo combina de forma coherente la inercia actual (`occupancy_pct`, $|\text{SHAP}| = 0,0998$) y la identidad espacial de la estación (`station_id_cat`, $0,0441$) con las componentes cíclicas horarias (`hour_cos = 0,0131`, `target_hour_cos_2h = 0,0107`), la disponibilidad física (`bikes_available = 0,0122`, `anchors_free = 0,0060`) y la velocidad de cambio (`delta_occ_1h = 0,0053`).

* **Patrón horario y espacial del error:** El MAE alcanza su mínimo de madrugada (`03:00–05:00`, $\approx 5,6\text{ \%}$ a $+1\text{ h}$ y $\approx 8,4\text{ \%}$ a $+2\text{ h}$) y su máximo por la tarde (`19:00–20:00`, $\approx 12,4\text{ \%}$ a $+1\text{ h}$ y $\approx 15,7\text{ \%}$ a $+2\text{ h}$). Espacialmente, los errores más altos se concentran en nodos turísticos y de altísima rotación (`108-Emperador Carlos V` con MAE $= 0,2355$, `102-Teatro Romano` con $0,2329$, `33-Echegaray: El Pilar` con $0,2305$), mientras que las estaciones residenciales periféricas presentan errores inferiores al $5,5\text{ \%}$ (`231-Heroínas Sitios: Gómara` con MAE $= 0,0487$).

* **Limitaciones identificadas:**

  1. **Ausencia de telemetría GPS de las furgonetas:** El modelo aprende sobre ocupaciones observadas que incluyen ocasionalmente intervenciones logísticas externas ($\ge 8$ bicicletas en 10 min), lo que introduce ruido puntual en las estaciones más intervenidas.

  2. **Subestimación de saturación extrema ($>80\%$) en fin de semana:** Al entrenar sobre un periodo con predominio de días laborables (`80 % WEEKDAY`) y evaluar sobre un `df_test` dominado por fin de semana (`85,4 % WEEKEND`), la regresión hacia la media suaviza los picos extremos de llenado ($>80\%$), problema que se mitigará directamente al reentrenar el flujo con más de 30 días de histórico acumulado en etapas posteriores.

## 5. Decisión y uso en el producto

### 5.1. Solución elegida y suficiencia para el MVP

Se selecciona **`LightGBMRegressor` multi-horizonte ($+1\text{ h}, +2\text{ h}, +3\text{ h}$)**, validado experimentalmente en `notebooks/modelo_predictivo.ipynb`, como motor predictivo oficial del MVP de **SmartRebalance ZGZ**:

* **Justificación técnica y operativa:** Ofrece el mejor equilibrio entre precisión continua (**MAE de 9,4 % a $+1\text{ h}$ y 12,3 % a $+2\text{ h}$**, equivalente a $\approx 1,8\text{ - }2,4\text{ bicicletas}$ en una estación estándar de 19–20 anclajes), sensibilidad en alertas de vaciado (**Recall de 82,8 %** a $+1\text{ h}$), explicabilidad auditada mediante SHAP y velocidad de inferencia ($<1\text{ segundo}$ para las 276 estaciones simultáneamente).

* **Suficiencia para el MVP:** El error medio de $\approx 2$ bicicletas es suficientemente acotado para discriminar qué estaciones requieren visitas prioritarias de las 6 furgonetas, especialmente al aplicar el umbral mínimo de rentabilidad de intervención ($|q_s| \ge 4\text{ bicicletas}$).

### 5.2. Esquema de salida validado y hoja de ruta hacia integración

En esta fase experimental, las predicciones generadas y auditadas en `notebooks/modelo_predictivo.ipynb` (`test_predictions_df`) establecen el contrato de datos que consumirá el futuro motor prescriptivo de optimización de rutas y el panel de control cuando se modularice en `src/modeling/` en la siguiente fase del proyecto:

| Campo | Tipo | Significado y uso previsto en el producto | 
| ----- | ----- | ----- | 
| `station_id` | `int` | Identificador único de la estación (cruce con `dim_stations` para coordenadas GPS y capacidad). | 
| `timestamp` | `datetime` | Marca temporal de emisión de la predicción (`Europe/Madrid`). | 
| `occupancy_pct` | `float [0, 1]` | Ocupación física real observada en el instante actual $t$. | 
| `pred_lgbm_1h` | `float [0, 1]` | Predicción de ocupación a $+1\text{ h}$ (usada en alertas de emergencia / régimen reactivo). | 
| `pred_lgbm_2h` | `float [0, 1]` | Predicción táctica principal a $+2\text{ h}$ (insumo base para planificación de turno de las 6 furgonetas). | 
| `pred_lgbm_3h` | `float [0, 1]` | Proyección extendida a $+3\text{ h}$ para anticipar tendencias sostenidas de vaciado o saturación. | 
| `expected_bikes_2h` | `int` | Estimación entera derivada ($\text{round}(\hat{y}_{s, t+2} \cdot \text{capacity})$) para calcular transferencias. | 
| `risk_status_2h` | `str` | Categoría operativa derivada (`CRITICAL_EMPTY` $<20\%$, `OPTIMAL` $20\%\text{ - }80\%$, `CRITICAL_FULL` $>80\%$). | 

* **Cautelas de uso en el producto:** El optimizador de rutas y el frontal deberán combinar siempre la predicción $\hat{y}_{s, t+h}$ con el estado físico actual $y_{s, t}$ mediante el selector de régimen $\tau_{\text{mode}}$ (reduciendo $\tau_{\text{mode}}$ en días festivos extraordinarios como Pilares o jornadas con incidencias operativas) y acotar cualquier orden de retirada/depósito a la capacidad física real disponible de la estación en el momento de llegada de la furgoneta.
