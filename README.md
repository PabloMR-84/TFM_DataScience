# TFM_DataScience — SmartRebalance ZGZ

## Estructura del Proyecto

```text
TFM/
├── assets/                   # Imágenes, capturas y recursos visuales para la documentación (README)
│
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
│   ├── models_gold.py        # Modelo dimensional, API de festivos y extrapolación de Cold Start
│   └── features.py           # (Reservado para Feature Engineering en fase ML)
│
├── main_BBDD.py              # Script ejecutable principal del pipeline ETL
├── .env                      # Variables de entorno seguras (BIZI_API_KEY)
└── README.md                 # Documentación y alcance del proyecto
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

---

### Configuración del archivo local `.env`:

Crea o edita un archivo llamado `.env` ubicado en la raíz del proyecto (`TFM/.env`) y añade tu clave con la siguiente sintaxis:

```env
BIZI_API_KEY=tu_clave_api_aqui
```

> **Nota de Seguridad e Integridad:**
> * El archivo `.env` se encuentra ignorado en `.gitignore` para evitar que las credenciales personales se suban al repositorio remoto.
> * Si ejecutas el proyecto sin una clave configurada, el pipeline utilizará el **modo de ejecución offline**, cargando los archivos locales previamente descargados en la carpeta `data/raw/` sin interrumpir el funcionamiento.