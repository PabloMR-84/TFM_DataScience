# Entrega 1 - Propuesta inicial de ideas de producto
**Proyecto de Data Science / IA**

Este documento contiene la propuesta de tres ideas iniciales para el proyecto final, enfocadas en la identificación de problemas reales y el valor potencial de las soluciones basadas en datos.

---

# Idea 1 - OptiCesta: Optimizador inteligente de la cesta de la compra

## Problema que resuelve
*   **Qué ocurre actualmente:** En el contexto económico actual de España (inflación y encarecimiento de la vivienda), el ahorro en productos de primera necesidad es crítico. Existen comparadores de precios, pero funcionan como buscadores de artículos individuales. El consumidor que realiza una compra completa no dispone de una herramienta para calcular el coste total de su lista en diferentes establecimientos de forma simultánea.
*   **Por qué supone un problema:** El ahorro real reside en la optimización del ticket total, no en productos aislados. Comparar manualmente una lista de 20-30 productos en varios supermercados es una tarea inasumible por tiempo. Las herramientas actuales son rígidas y no consideran la "conveniencia" (tiempo vs. ahorro).
*   **Qué decisión, proceso o situación podría mejorar con datos:** La planificación de la compra doméstica. La herramienta permite decidir si realizar toda la compra en un único establecimiento o dividirla (multi-súper) basándose en el coste total proyectado y preferencias de movilidad.
*   **Por qué no es simplemente un ejercicio técnico, sino un caso de uso con sentido:** Transforma la frustración por la subida de precios en una estrategia de ahorro ejecutable y personalizada, devolviendo al ciudadano el control sobre su gasto mensual.

## Motivación para realizar el proyecto
*   **Relevancia social y económica:** Impacto directo en la economía doméstica de las familias en un escenario de pérdida de poder adquisitivo.
*   **Interés profesional o técnico:** Reto analítico en la integración de flujos de datos heterogéneos y la necesidad de aplicar técnicas de *matching* de entidades (productos similares con distintos nombres).
*   **Utilidad potencial para un colectivo concreto:** Familias numerosas, estudiantes con presupuesto ajustado y jóvenes emancipados con gastos fijos altos.
*   **Posibilidad de aprender o aplicar contenidos relevantes del máster:** Aplicación de algoritmos de optimización, limpieza de datos y modelos de IA para categorización.
*   **Existencia de datos que podrían permitir trabajar la idea:** Los precios son públicos en las plataformas de e-commerce de los principales supermercados de España.

## A quién impacta
*   **Familias:** Hogares que necesitan optimizar grandes volúmenes de compra.
*   **Estudiantes:** Usuarios con ingresos bajos que requieren maximizar su presupuesto.
*   **Jóvenes emancipados:** Personas que buscan eficiencia ante la presión de alquileres altos.

## Por qué tiene valor
*   **Ahorro de costes:** Reducción directa del gasto en la cesta de la compra.
*   **Mejor toma de decisiones:** Flexibilidad para elegir estrategias de ahorro basadas en datos reales.
*   **Mejora de la planificación:** Optimización logística según ubicación y disponibilidad del usuario.

---

# Idea 2 - SmartRebalance ZGZ: Optimización de logística urbana para Bizi Zaragoza

## Problema que resuelve
*   **Qué ocurre actualmente:** El servicio de bicicletas públicas de Zaragoza sufre asimetrías de ocupación (barrios vacíos vs. centro saturado). La redistribución mediante camionetas opera de forma reactiva y a menudo ineficaz, sin anticiparse a los flujos de demanda.
*   **Por qué supone un problema:** Genera falta de fiabilidad (no hay bicis para salir o no hay anclajes para llegar), lo que desincentiva el transporte sostenible. Para la empresa, implica una logística costosa en combustible y tiempo de operarios.
*   **Qué decisión, proceso o situación podría mejorar con datos:** La planificación de rutas de redistribución. El sistema puede predecir qué estaciones se saturarán, permitiendo decidir qué estaciones visitar y cuántas bicis mover para minimizar tiempos y consumo.
*   **Por qué no es simplemente un ejercicio técnico, sino un caso de uso con sentido:** Aborda un problema crítico de movilidad en una *Smart City* real, garantizando que el sistema sea una alternativa fiable frente al coche privado.

## Motivación para realizar el proyecto
*   **Relevancia social y ambiental:** Fomento de la movilidad eléctrica y reducción de emisiones de CO2.
*   **Interés profesional o técnico:** Uso de Series Temporales para predicción de demanda y algoritmos de optimización de rutas (VRP - *Vehicle Routing Problem*).
*   **Utilidad potencial para un colectivo concreto:** Usuarios diarios del servicio Bizi Zaragoza, especialmente en barrios periféricos.
*   **Posibilidad de aprender o aplicar contenidos relevantes del máster:** Integración de APIs en tiempo real y visualización de datos geoespaciales.
*   **Existencia de datos que podrían permitir trabajar la idea:** El portal de Open Data del Ayuntamiento de Zaragoza ofrece información histórica y en tiempo real de las estaciones.

## A quién impacta
*   **Usuarios de Bizi:** Ganan fiabilidad y ahorro de tiempo en sus desplazamientos diarios.
*   **Empresa gestora:** Mejora su eficiencia operativa y reduce costes logísticos.
*   **Ayuntamiento:** Facilita el cumplimiento de los planes de movilidad sostenible.

## Por qué tiene valor
*   **Ahorro de tiempo y reducción de costes:** Optimización del uso de vehículos de reparto y combustible.
*   **Mejor toma de decisiones:** Sustitución de la intuición por rutas basadas en predicciones de demanda real.
*   **Mejor experiencia de usuario:** Incrementa la confianza y fidelización en el sistema de transporte público.

---

# Idea 3 - LocalFest: Agregador inteligente de eventos y fiestas locales

## Problema que resuelve
*   **Qué ocurre actualmente:** La información sobre fiestas locales y eventos en pequeños municipios está fragmentada en redes sociales, carteles físicos o bandos municipales. No existe una plataforma centralizada accesible para gente de fuera de la localidad.
*   **Por qué supone un problema:** El usuario pierde oportunidades de ocio cultural por falta de información. Los municipios pequeños pierden visibilidad y potencial económico al no poder atraer visitantes externos.
*   **Qué decisión, proceso o situación podría mejorar con datos:** La planificación del ocio de proximidad. Al geolocalizar al usuario, el sistema responde a "¿qué hay cerca de mí ahora?", unificando la oferta dispersa en una agenda estructurada.
*   **Por qué no es simplemente un ejercicio técnico, sino un caso de uso con sentido:** Requiere IA para capturar datos de fuentes no estructuradas (redes sociales, programas en PDF), resolviendo un problema de comunicación que afecta a la economía rural.

## Motivación para realizar el proyecto
*   **Relevancia social y económica:** Visibilidad para la "España vaciada" y dinamización de la hostelería local a través del turismo de cercanía.
*   **Interés profesional o técnico:** Reto en recolección de datos mediante Web Scraping y Procesamiento de Lenguaje Natural (NLP) para normalizar información de fuentes heterogéneas.
*   **Utilidad potencial para un colectivo concreto:** Jóvenes en busca de ocio económico y turistas interesados en experiencias locales auténticas.
*   **Posibilidad de aprender o aplicar contenidos relevantes del máster:** Limpieza de datos textuales, scraping avanzado y geolocalización.
*   **Existencia de datos que podrían permitir trabajar la idea:** Datos públicos en perfiles de ayuntamientos, diarios provinciales y agendas culturales regionales.

## A quién impacta
*   **Jóvenes y turistas:** Descubren eventos tradicionales de forma sencilla basada en su ubicación.
*   **Ayuntamientos:** Obtienen una plataforma de difusión masiva para sus tradiciones.
*   **Comercio local:** Se beneficia del flujo de visitantes atraídos por el evento.

## Por qué tiene valor
*   **Generación de conocimiento:** Unifica una oferta cultural que actualmente es invisible para los buscadores convencionales.
*   **Dinamización económica:** Fomenta el gasto en zonas rurales o barrios periféricos fuera de los circuitos comerciales.
*   **Mayor accesibilidad:** Democratiza la información de eventos tradicionales independientemente del tamaño del municipio.