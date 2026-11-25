# TFM_DataScience — SmartRebalance ZGZ

**SmartRebalance ZGZ** es un sistema analítico y predictivo para la gestión proactiva de la red de bicicletas públicas **Bizi Zaragoza** (276 estaciones). El proyecto integra un pipeline de datos con arquitectura Medallón (*Bronze → Silver → Gold*), modelos de Machine Learning multi-horizonte (`+1h`, `+2h` y `+3h`) con trazabilidad SHA-256 y calibración dinámica de umbrales de saturación para anticipar vaciados y llenados, y la preparación de entradas para la futura optimización logística de la flota de rebalanceo (6 furgonetas de 22 plazas en operación 24 horas).

---

## Estructura del Proyecto

El repositorio sigue una arquitectura modular estándar (*src-layout*) que separa la ingesta, la lógica empaquetada, los scripts ejecutables, los artefactos entrenados y la experimentación. Cada carpeta principal incluye su propio `README.md` con el detalle técnico de su contenido:

```text
TFM/
├── .github/workflows/    # Automatización de captura periódica cada 10 min (GitHub Actions)
├── assets/               # Imágenes y recursos visuales para la documentación
├── data/                 # Capas Medallón (raw, silver, gold), predicciones y snapshots congelados
├── docs/                 # Documentación técnica y memoria de entregas del TFM
├── models/               # Modelos entrenados (locales) y metadatos versionados (metadata.json + SHA-256)
├── notebooks/            # Cuadernos de experimentación, evaluación, SHAP y análisis de saturación
├── scripts/              # Orquestadores ejecutables (ETL, entrenamiento e inferencia batch)
├── src/                  # Código fuente empaquetado del proyecto (etl/ y modeling/)
├── .env                  # Variables de entorno locales (BIZI_API_KEY, ignorado en Git)
├── .gitignore            # Exclusiones de control de versiones (binarios, datos vivos, caché)
├── CHANGELOG.md          # Registro cronológico de versiones y cambios
├── CONTRIBUTING.md       # Guía y convenciones de contribución al repositorio
├── extract.py            # Script ligero de captura automatizada (GitHub Actions)
├── LICENSE               # Licencia del repositorio
├── pyproject.toml        # Empaquetado, dependencias y entry points CLI
├── README.md             # Documentación general, instalación y puesta en marcha
└── requirements.txt      # Listado de dependencias del entorno
```

> **Documentación detallada por módulo:** Consulta los `README.md` específicos dentro de `data/`, `notebooks/`, `scripts/`, `src/` y `models/` para ver esquemas de datos, contratos de entrada/salida y detalles de implementación.

---

## Requisitos e Instalación

### 1. Requisitos previos

* **Python 3.10** o superior.
* **Git** para la gestión del repositorio.

### 2. Creación del entorno virtual e instalación de dependencias

Desde la raíz del proyecto (`TFM/`), crea y activa un entorno virtual e instala el proyecto en **modo editable** (`-e .`) para registrar los paquetes internos (`etl` y `modeling`) y los comandos CLI definidos en `pyproject.toml`:

```bash
# 1. Crear el entorno virtual
python -m venv .venv

# 2. Activar el entorno virtual
# En Windows (Git Bash / MINGW64):
source .venv/Scripts/activate
# En Windows (PowerShell):
# .venv\Scripts\Activate.ps1
# En Linux / macOS:
# source .venv/bin/activate

# 3. Instalar el paquete local y sus dependencias de producción en modo editable
pip install -e .

# (Opcional) Instalar dependencias adicionales para ejecutar los notebooks (Optuna, SHAP, XGBoost, gráficos)
pip install -e ".[notebooks]"
```

*(Como alternativa, también puedes instalar el entorno completo mediante `pip install -r requirements.txt` seguido de `pip install -e .`).*

---

## Ejecución Rápida (Pipelines)

Los flujos de datos, entrenamiento e inferencia están desacoplados y se ejecutan desde la raíz del proyecto (`TFM/`):

### 1. Pipeline ETL (*Bronze → Silver → Gold*)
Regenera las capas `data/silver/` y `data/gold/` a partir del histórico local y la API de DatosBizi:
```bash
python scripts/main_BBDD.py
```

### 2. Entrenamiento del Modelo Predictivo (LightGBM `+1h`, `+2h`, `+3h`)
Lee `data/gold/`, calcula el hash **SHA-256** de los datos de entrada, entrena los modelos multi-horizonte, calibra automáticamente los umbrales de alerta de saturación sobre Validación (`df_val`) y guarda los artefactos y `metadata.json` en `models/lgbm_v1/`:
```bash
python scripts/train_model.py
# o mediante el entry point CLI:
smartrebalance-train
```

### 3. Inferencia Batch (Predicciones y Alertas)
Carga los modelos serializados y los umbrales calibrados de `metadata.json` sin reentrenar y exporta las predicciones continuas junto con las banderas de alerta de vaciado/saturación a `data/predictions/predictions_lgbm.parquet`:
```bash
# Ventana reciente por defecto (últimas 72 horas)
python scripts/predict_model.py

# Solo el instante más reciente capturado (modo tiempo real)
smartrebalance-predict --latest-only
```

---

## Configuración de Credenciales y Entorno (`.env`)

Para realizar descargas en vivo de datasets extensos (como el histórico macro en CSV de `/api/history`) o ejecutar peticiones masivas, la API de DatosBizi requiere autenticación mediante una cabecera `x-api-key`.

### Pasos para obtener la clave de API (`BIZI_API_KEY`):

1. **Acceder a la sección de desarrolladores:**  
   Abre el navegador y dirígete al apartado de la API: [datosbizi.com/developers#register-api](https://datosbizi.com/developers#register-api).

2. **Ir al bloque de generación de claves:**  
   Localiza este bloque en la web y haz clic en el botón **"Iniciar sesión para crear tu clave"**.

   ![Consigue tu clave de API](assets/get_api_key.png)

3. **Autenticación en la plataforma:**  
   Inicia sesión con tu cuenta de usuario o regístrate directamente utilizando tu correo electrónico o mediante inicio de sesión rápido con **Google** o **GitHub**.

4. **Crear la clave personal:**  
   Una vez iniciada la sesión, escribe el motivo o identificación del proyecto en el campo *"¿Para qué la vas a usar?"* (por ejemplo: `Mi integración` o `TFM SmartRebalance`) y presiona el botón **"Crear clave"**.

5. **Copiar la clave:**  
   Copia la API Key alfanumérica resultante que se mostrará en pantalla.

### Configuración del archivo local `.env`:

Crea o edita un archivo llamado `.env` ubicado en la raíz del proyecto (`TFM/.env`) y añade tu clave con la siguiente sintaxis:

```env
BIZI_API_KEY=tu_clave_api_aqui
```

> **Nota de Seguridad e Integridad:**
> * El archivo `.env` se encuentra ignorado en `.gitignore` para evitar que las credenciales personales se suban al repositorio remoto.
> * Si ejecutas el proyecto sin una clave configurada, el pipeline utilizará el **modo de ejecución offline**, cargando los archivos locales previamente descargados en la carpeta `data/raw/` sin interrumpir el funcionamiento.