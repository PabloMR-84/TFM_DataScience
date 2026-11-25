# 📓 Cuadernos de Experimentación (`notebooks/`)

Este directorio reúne los cuadernos de Jupyter utilizados para la auditoría de calidad de la Capa Gold, la experimentación y validación de los modelos predictivos de ocupación, el análisis monográfico de saturación y el prototipado del motor de optimización de rutas para **SmartRebalance ZGZ**.

---

## 🗂️ Catálogo de Cuadernos y Orden de Ejecución

Para reproducir de forma íntegra los experimentos y artefactos del repositorio, los cuadernos deben ejecutarse siguiendo este orden:

### 1. [`modelado_predictivo.ipynb`](modelado_predictivo.ipynb) — *Modelado Predictivo Multi-Horizonte ($+1\text{ h}, +2\text{ h}, +3\text{ h}$)*
Cuaderno principal de entrenamiento, validación *anti-leakage* y evaluación ciega de los modelos de predicción de ocupación de estaciones:
* **Auditoría de la Capa Gold:** Validación de integridad física, comprobación de disponibilidad temporal estricta (`available_at <= timestamp`) y auditoría global de lecturas *offline* de la API (`bikes_available == 0` y `anchors_free == 0`).
* **Ingeniería de variables y partición cronológica:** Construcción de **766.610 muestras multi-horizonte** divididas en **Train (`531.458`)**, **Validación (`109.848`)** y **Test (`114.816`)** con ventanas de embargo de $3\text{ horas}$ (`10.488` registros aislados en fronteras).
* **Comparativa de modelos y optimización:** Evaluación frente a *baselines* (*Naive*, Perfil Histórico Causal y Ridge) y modelos de aprendizaje automático (Random Forest, XGBoost y LightGBM), optimización bayesiana con **Optuna** y análisis de explicabilidad con **SHAP**.
* **Artefactos generados:**
  * Predicciones *out-of-sample* sobre `Test`: `data/gold_snapshot_ml_20261122/predictions/test_predictions_lgbm.parquet`.
  * Modelos campeones y metadatos con trazabilidad SHA-256: `models/lgbm_v1/` (`lgbm_1h.joblib`, `lgbm_2h.joblib`, `lgbm_3h.joblib`, `metadata.json`).

### 2. [`EDA_prediccion_saturacion.ipynb`](EDA_prediccion_saturacion.ipynb) — *Análisis Monográfico de Predicción de Saturaciones ($> 80\%$)*
Cuaderno complementario que profundiza en el comportamiento del regresor campeón (`LightGBM Optuna`) ante episodios de llenado extremo detectados en la Sección 10 de `modelado_predictivo.ipynb`:
* **Rankings y auditoría por estación:** Contraste **Top 10 Real vs. Top 10 Predicho** en los tres horizontes ($+1\text{ h}, +2\text{ h}, +3\text{ h}$) y auditoría visual y métrica (en minutos) sobre el **Top 5 Real** y el **Top 5 Predicho exclusivo** (falsos negativos, falsos positivos en estados de casi-saturación y anomalías *offline*).
* **Caracterización del sesgo en saturación:** Cuantificación de la subestimación sistemática en picos de llenado (**$-20,24\text{ pp}$** en $+1\text{ h}$, **$-28,61\text{ pp}$** en $+2\text{ h}$ y **$-33,31\text{ pp}$** en $+3\text{ h}$ frente a una ocupación real media del **$91,0\%$**).
* **Calibración *anti-leakage* de umbrales (*Threshold Tuning*):** Ajuste estricto sobre `df_val` de los umbrales de alerta de saturación por horizonte (**$74,0\%$** en $+1\text{ h}$, **$70,0\%$** en $+2\text{ h}$ y **$65,0\%$** en $+3\text{ h}$), elevando sustancialmente el `F1 Saturada` en `Test` sin degradar la utilidad operativa de las alertas.
* **Decisión arquitectónica:** Contraste experimental frente a un clasificador binario auxiliar (`LGBMClassifier`), demostrando que sus ganancias marginales en horizontes tácticos (**$+0,0195$** en $+2\text{ h}$ y **$+0,0095$** en $+3\text{ h}$) no justifican duplicar la arquitectura de 3 a 6 modelos en producción.
* **Prerrequisito:** Ejecutar tras `modelado_predictivo.ipynb` (o disponer previamente de `test_predictions_lgbm.parquet` y el *snapshot* Gold).

### 3. `rebalancing_routing_engine.ipynb` — *Motor Prescriptivo de Optimización de Rutas (Fase Prescriptiva)*
Cuaderno orientado al diseño y validación del motor de ruteo para la flota real de Zaragoza (**6 furgonetas**, **22 plazas** por vehículo, operación continua **24 h**), consumiendo las predicciones *out-of-sample* y los umbrales calibrados de los cuadernos anteriores.

---

## 📊 Resumen de Métricas en Evaluación Ciega (`df_test`)

Resultados consolidados del modelo campeón **LightGBM (Optuna)** sobre el *snapshot* `gold_snapshot_ml_20261122`, incorporando la mejora obtenida mediante *Threshold Tuning* documentada en `EDA_prediccion_saturacion.ipynb`:

| Horizonte | MAE | RMSE | $R^2$ | F1 Vacía ($< 20%$) | F1 Saturada Base ($> 80%$) | Umbral Calibrado (`df_val`) | F1 Saturada Calibrada (`df_test`) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **$+1\text{ h}$** | `0,0938` | `0,1335` | `0,7333` | `0,8464` | `0,3888` | `74,0%` | **`0,4985`** |
| **$+2\text{ h}$** | `0,1190` | `0,1624` | `0,6058` | `0,7921` | `0,2348` | `70,0%` | **`0,4018`** |
| **$+3\text{ h}$** | `0,1318` | `0,1755` | `0,5383` | `0,7573` | `0,1702` | `65,0%` | **`0,3694`** |

---

## 📥 Descarga del Snapshot de Datos (`gold_snapshot_ml_20261122`)

Dado que el repositorio de GitHub no aloja ficheros binarios pesados (`*.parquet`) y la carpeta de producción `data/gold/` se actualiza continuamente con nuevas capturas cada 10 minutos, los cuadernos leen una **instantánea congelada e inmutable** correspondiente al periodo auditado (**31 de octubre de 2026 – 22 de noviembre de 2026**: **782.736 registros 100 % reales** en **276 estaciones**, **23 días naturales** y **2.836 instantes de captura** en zona horaria `Europe/Madrid`).

De este modo, al ejecutar los cuadernos se reproducen exactamente las mismas métricas, gráficos y conclusiones documentadas en sus celdas sin verse alterados por futuras ejecuciones del recolector.

### 1. Enlace de descarga
Descarga los cuatro archivos `.parquet` de la instantánea desde la carpeta compartida de Google Drive:
* 🔗 **[Descargar snapshot Gold (Google Drive)](https://drive.google.com/drive/folders/1I8Wwiss2vubQlXl92_SIGDLN5zQAImFd?usp=drive_link)** *(asegúrate de descargar la carpeta correspondiente a `gold_snapshot_ml_20261122`)*.

### 2. Ubicación en el repositorio
Guarda los archivos descargados dentro de `data/gold_snapshot_ml_20261122/`, respetando exactamente esta estructura:

```text
data/
└── gold_snapshot_ml_20261122/
    ├── df_gold.parquet                         # Tabla maestra consolidada (782.736 filas)
    ├── dim_stations.parquet                    # Catálogo geográfico y capacidad de las 276 estaciones
    ├── dim_calendar.parquet                    # Calendario local, festivos y contexto diario
    ├── dim_patterns.parquet                    # Perfiles históricos horarios por estación
    └── predictions/
        └── test_predictions_lgbm.parquet       # Generado al ejecutar modelado_predictivo.ipynb
```

---

## ⚙️ Requisitos de Ejecución

1. Activar el entorno virtual del proyecto configurado desde la raíz (`TFM/`) con las dependencias de `requirements.txt` y el paquete instalado en modo editable (`pip install -e .`).
2. Confirmar que los cuatro ficheros `.parquet` del *snapshot* están ubicados en `data/gold_snapshot_ml_20261122/`.
3. Ejecutar las celdas de **`modelado_predictivo.ipynb`** de principio a fin para entrenar/validar los modelos y generar `test_predictions_lgbm.parquet`.
4. Ejecutar **`EDA_prediccion_saturacion.ipynb`** para reproducir el diagnóstico completo de estaciones saturadas, la calibración de umbrales y la comparativa frente al clasificador binario.
