# TFM_DataScience — SmartRebalance ZGZ

## Estructura del Proyecto

```text
TFM/
├── .github/
│   └── workflows/
│       └── extract.yml             # Automatización de captura periódica cada 10 min (GitHub Actions)
│
├── assets/                         # Imágenes, capturas y recursos visuales para la documentación (README)
│
├── data/
│   ├── raw/                        # Archivos en bruto descargados de la API de DatosBizi
│   ├── silver/                     # DataFrames estandarizados y limpios (csv/parquet)
│   ├── gold/                       # Tablas dimensionales, hechos y df_gold final (parquet)
│   └── gold_snapshot_ml_20261115/  # Snapshot congelado de datos para reproducibilidad en notebooks/
│
├── docs/                           # Documentación técnica y entregables del proyecto
│
├── notebooks/                      # Cuadernos de modelado predictivo, evaluación y explicabilidad SHAP
│   ├── modelo_predictivo.ipynb     # Cuaderno principal de experimentación y validación ML (+1h, +2h, +3h)
│   └── README.md                   # Instrucciones de descarga del snapshot de datos desde Drive
│
├── scripts/                        # Scripts ejecutables y orquestadores de pipelines
│   └── main_BBDD.py                # Script principal del pipeline ETL (Bronze -> Silver -> Gold)
│
├── src/                            # Código fuente empaquetado del proyecto
│   ├── __init__.py                 # Inicializador del paquete principal
│   ├── etl/                        # Módulo de ingeniería de datos y construcción de Capa Gold
│   │   ├── __init__.py             # Exportaciones públicas del subpaquete etl
│   │   ├── config.py               # Configuración central (endpoints, variables de entorno, ROOT_DIR)
│   │   ├── utils.py                # Herramientas de soporte (parseo de JSON, tipos, timestamps)
│   │   ├── io.py                   # Orquestador I/O (descarga interactiva o caché, persistencia)
│   │   ├── cleaning.py             # Estandarización pura de esquemas (Capa Silver)
│   │   └── models_gold.py          # Modelo dimensional, API de festivos y extrapolación Cold Start
│   └── modeling/                   # Módulo de Machine Learning (en desarrollo)
│       └── __init__.py             # Inicializador del subpaquete modeling
│
├── .env                            # Variables de entorno seguras (BIZI_API_KEY)
├── .gitignore                      # Exclusiones de control de versiones (datos, entornos, temporales)
├── CHANGELOG.md                    # Registro cronológico de versiones y cambios del proyecto
├── CONTRIBUTING.md                 # Guía y convenciones de contribución al repositorio
├── extract.py                      # Script ligero de captura periódica automatizada (GitHub Actions)
├── LICENSE                         # Licencia del repositorio
├── pyproject.toml                  # Configuración de empaquetado y metadatos del proyecto
├── README.md                       # Documentación general, instalación y configuración
└── requirements.txt                # Dependencias y librerías necesarias para ejecutar el proyecto
```

---

## Requisitos e Instalación

### 1. Requisitos previos

* **Python 3.10** o superior.
* **Git** para la gestión del repositorio.

### 2. Creación del entorno virtual e instalación de dependencias

Desde la raíz del proyecto (`TFM/`), crea y activa un entorno virtual, instala las dependencias definidas en `requirements.txt` y registra el proyecto en **modo editable** (`-e .`) para que Python reconozca los paquetes internos (`etl` y `modeling`) desde cualquier script o cuaderno:

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

# 3. Instalar las dependencias del proyecto
pip install -r requirements.txt

# 4. Instalar el paquete local en modo editable
pip install -e .
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