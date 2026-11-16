# Guía de Contribución — SmartRebalance ZGZ

Gracias por contribuir al desarrollo de **SmartRebalance ZGZ**. Para mantener la integridad del pipeline ETL, la trazabilidad de datos y la reproducibilidad del proyecto, sigue estas pautas antes de enviar cambios.

## 1. Flujo de Trabajo con Git

- **`main`:** Rama estable de producción. Contiene las entregas validadas y el *workflow* automático de captura cada 10 minutos (`data/raw/historical/*.parquet`). No hagas *commits* directos sobre `main`.
- **`develop`:** Rama de integración para nuevas funcionalidades y correcciones.
- Crea siempre una rama específica a partir de `develop` (`feature/nombre-funcion` o `fix/descripcion-breve`) y abre un **Pull Request** hacia `develop` una vez verificada la ejecución completa.
- Utiliza mensajes de *commit* descriptivos siguiendo [Conventional Commits](https://www.conventionalcommits.org/es/) (`feat:`, `fix:`, `docs:`, `refactor:`).

## 2. Arquitectura Modular (`src/`)

Mantén el orquestador `main_BBDD.py` libre de lógica auxiliar. Toda función nueva debe ubicarse en su módulo correspondiente:

- `src/config.py`: Rutas del proyecto (`ROOT_DIR`), credenciales desde `.env` y configuración de *endpoints*.
- `src/io.py`: Lectura/escritura en disco, limpieza de carpetas de salida y control de descargas desde la API.
- `src/cleaning.py`: Transformaciones puras de la **Capa Silver** (validación de rangos, tipado y normalización).
- `src/models_gold.py`: Construcción del **Modelo Estrella** (`Dim_*`, `Fact_*`) y consolidación de `df_gold.parquet`.
- `src/utils.py`: Funciones transversales de soporte, manejo de zonas horarias y auditorías de cardinalidad/cobertura.
- `src/features.py`: Ingeniería de variables (*lags*, ventanas móviles y codificación cíclica) para modelos predictivos.

## 3. Reglas Técnicas y Calidad de Datos

1. **Cero fuga temporal (*No Data Leakage*):** Ninguna variable agregada o futura puede cruzarse con capturas pasadas sin verificar `available_at <= timestamp`.
2. **Conservación de nulos reales:** Usa tipos anulables de Pandas (`Int64`, `float64` con `NaN`) para valores desconocidos o anteriores a la ventana de disponibilidad; nunca imputes `0` de forma silenciosa en métricas observadas.
3. **Estandarización horaria:** Todas las marcas temporales deben procesarse mediante `safe_local_timestamp()` en zona horaria `Europe/Madrid`.
4. **Verificación previa al Pull Request:** Ejecuta `python main_BBDD.py` en modo local (`[s]` para regenerar Silver/Gold y `[U]` para usar caché de Bronze) y comprueba que las auditorías de cardinalidad (`auditar_merge`) y cobertura finalizan sin errores.
5. **Seguridad y archivos ignorados:** Nunca subas archivos `.env` con claves privadas (`BIZI_API_KEY`) ni artefactos temporales de `data/silver/` o `data/gold/`.