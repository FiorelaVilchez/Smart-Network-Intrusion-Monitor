# Manual de Usuario: Smart Network Intrusion Monitor (SNIM)

Bienvenido a **SNIM**, la plataforma inteligente de monitoreo de intrusiones en la red. Este manual te guiará a través de la interfaz de usuario y te explicará cómo interpretar los datos mostrados por el modelo de Inteligencia Artificial.

---

## 1. Acceso a la Plataforma

Para ingresar a la plataforma, simplemente accede a la URL proporcionada por el equipo de TI (por ejemplo, el enlace público de Render). No requiere instalación de software adicional, solo un navegador web moderno (Chrome, Firefox, Edge, Safari).

> **[INSERTAR IMAGEN AQUÍ: Captura de pantalla de la pantalla completa inicial de la aplicación, antes de iniciar la simulación. Debe verse el panel lateral izquierdo y los contadores en 0.]**

---

## 2. El Panel de Control (Barra Lateral)

Ubicado a la izquierda de la pantalla, el panel de control te permite administrar la inyección de tráfico simulado a la aplicación.

### Controles Disponibles:
- **Botón Iniciar / Detener Simulación:** Activa el motor de inyección de paquetes de red. Una vez iniciado, el modelo comenzará a evaluar el tráfico en vivo.
- **Control de Velocidad (Registros por segundo):** Un deslizador (slider) que te permite definir cuántos paquetes de red por segundo analizará la aplicación. (Útil para pruebas de estrés o visualización calmada).
- **Control de Refresco UI:** Ajusta la frecuencia con la que las gráficas y las tablas se actualizan visualmente.

> **[INSERTAR IMAGEN AQUÍ: Captura de pantalla recortada mostrando únicamente la barra lateral izquierda (Sidebar) con los controles de simulación (botones y sliders).]**

---

## 3. Panel de Indicadores Clave (KPIs)

En la parte superior central de la pantalla principal, encontrarás tarjetas de resumen que ofrecen un vistazo instantáneo a la salud de la red.

1. **Tráfico Total:** Cantidad total de paquetes de red procesados desde que se inició la plataforma.
2. **Alertas / Anomalías:** Número absoluto de paquetes clasificados como ataques o intrusiones por la Inteligencia Artificial.
3. **Tasa de Anomalías:** El porcentaje de tráfico que representa un riesgo (Alertas / Tráfico Total).
4. **Metadatos del Modelo:** Información sobre el modelo actual en producción (Versión, F1-Score base, Número de características).

> **[INSERTAR IMAGEN AQUÍ: Captura de pantalla centrada en las "tarjetas de métricas" (Metrics Cards) ubicadas en la parte superior del dashboard, mostrando algunos números procesados.]**

---

## 4. Visualización en Tiempo Real

Debajo de los indicadores, SNIM proporciona herramientas gráficas y tabulares para el análisis detallado.

### Gráfica de Tráfico Reciente
Un gráfico de líneas en vivo que muestra el volumen de tráfico normal frente al tráfico anómalo en el tiempo. Si ves picos abruptos en la línea de color rojo, significa que la red está bajo un posible ataque (ej. escaneo de puertos, ataque DoS).

> **[INSERTAR IMAGEN AQUÍ: Captura de pantalla del gráfico de líneas de Altair, mostrando la evolución temporal del tráfico normal (azul) y anómalo (rojo).]**

### Registro de Anomalías (Tabla de Eventos)
Muestra un registro tabular (Dataframe) únicamente de las alertas generadas por el modelo, ordenadas de las más recientes a las más antiguas.

**Columnas importantes a observar:**
- **Timestamp:** Momento exacto de la detección.
- **Protocol & Service:** (Ej. `tcp` / `http`). Identifica qué área de la red está siendo atacada.
- **Anomaly Score:** El grado de rareza calculado por la IA. Mientras **más alto** sea este número, **más atípico** y peligroso es el evento.
- **True Label:** Etiqueta real del paquete (información extraída del dataset para propósitos académicos y de validación).

> **[INSERTAR IMAGEN AQUÍ: Captura de pantalla de la tabla "Registro de Anomalías Recientes", mostrando varias filas con datos resaltados y puntajes de anomalía altos.]**

---

## 5. Mantenimiento del Sistema y Reentrenamiento

El cerebro de SNIM no es estático; aprende y mejora continuamente utilizando un modelo dual de reentrenamiento. Todo esto sucede automáticamente sin que necesites ser un experto en programación.

### 5.1. Reentrenamiento Diario (De Fondo)
La aplicación almacena automáticamente cada registro procesado en su base de datos interna. Diariamente (generalmente en la madrugada), un robot de mantenimiento extrae estos datos, entrena una nueva Inteligencia Artificial, compara su rendimiento y, si resulta ser mejor, actualiza la página automáticamente. 

* **Si la app tarda en cargar la primera vez en el día:** Esto se conoce como un "Cold Start" en Render. Simplemente dale de 30 segundos a 1 minuto para que despierte y tendrás la versión más reciente lista.

### 5.2. Reentrenamiento Reactivo (En Vivo)
Para responder a emergencias de forma inmediata, la app te vigila en vivo. Puedes consultar la pestaña **"ℹ️ Acerca del Modelo"** para ver cuándo fue el último reentrenamiento y un historial de las adaptaciones de emergencia.

Durante la simulación en la pestaña de **"📡 Tráfico en Vivo"**, el sistema podría detenerse durante 2 a 5 segundos y mostrar un aviso brillante que dice: **"🔄 Reentrenamiento automático disparado — Motivo: [...]"**.

Esto es completamente normal e indica que el modelo sintió la necesidad urgente de adaptarse al tráfico actual basándose en 1 de 4 motivos principales:
1. **"Ya pasó el tiempo programado"**: El sistema se re-calibra cada 3 minutos automáticamente para adaptarse a pequeños cambios incluso cuando no hay ataques.
2. **"Se acumularon varias alertas"**: La red ha sido bombardeada con múltiples intrusiones (15 o más) y el sistema necesita aprender el patrón general rápido para detenerlas y no generar "falsas alarmas".
3. **"Apareció un tipo de ataque que el sistema no conocía"**: Se ha detectado un Zero-Day, un malware o método de intrusión del cual no había registro previo.
4. **"Se detectó un ataque muy grave"**: Algún paquete o conexión tuvo un grado de atipicidad estadísticamente masivo, requiriendo atención urgente.

En la pestaña "Acerca del Modelo" podrás auditar si la IA decidió "✅ Sí" promoverse y asimilar ese aprendizaje o si determinó que la mejora no valía la pena "❌ No" y prefirió retener la versión anterior.
