# Informe Técnico: Smart Network Intrusion Monitor (SNIM)

## 1. Resumen Ejecutivo
**Smart Network Intrusion Monitor (SNIM)** es una aplicación web de producción orientada a la ciberseguridad, diseñada para la detección de anomalías en el tráfico de red en tiempo real. Fue construida como parte de un proyecto de Aprendizaje de Máquina, integrando un modelo predictivo previamente entrenado sin necesidad de alterar su concepción original, envolviéndolo en un ecosistema completo de MLOps (Operaciones de Machine Learning).

El sistema consta de:
1. Un **Dashboard Interactivo** construido en Streamlit para el monitoreo en tiempo real.
2. Un **Simulador de Tráfico** que inyecta datos emulando un entorno de producción.
3. Una **Capa de Persistencia (SQLite)** para registrar predicciones y variables críticas.
4. Un **Pipeline de Mantenimiento (CI/CD)** para el reentrenamiento y despliegue automatizado del modelo.

## 2. Arquitectura del Sistema

La arquitectura de SNIM sigue un patrón monolítico modular, diseñado para un despliegue ágil en plataformas de nube como Render.

- **Frontend / Interfaz (UI):** `app.py` y `src/ui_components.py`. Manejan la renderización del dashboard en el navegador del usuario utilizando Streamlit, soportando actualización asíncrona de datos en vivo.
- **Lógica de Simulación:** `src/simulator.py`. Simula el flujo continuo de paquetes de red leyendo del dataset NSL-KDD (Test/Train), permitiendo inyectar datos a diferentes velocidades.
- **Lógica de Preprocesamiento:** `src/preprocessing.py`. Aplica la ingeniería de características idéntica a la fase de entrenamiento (One-Hot Encoding, reindexación a 121 columnas, escalado con StandardScaler).
- **Capa de Almacenamiento:** `src/storage.py`. Base de datos SQLite (`data/traffic_log.db`) que actúa como _Data Lake_ transaccional para auditar decisiones del modelo y proveer nuevos datos para el reentrenamiento.
- **Orquestación de Machine Learning:** `src/retrain.py` y `src/model_io.py`. Se encargan del ciclo de vida del modelo (carga, evaluación, reentrenamiento, y promoción).

## 3. Especificaciones del Modelo de Machine Learning

El modelo central del proyecto es un algoritmo de aprendizaje no supervisado diseñado para detectar valores atípicos (outliers).

- **Algoritmo:** `IsolationForest` (de `scikit-learn==1.8.0`).
- **Características de Entrada:** 121 características (features) después de aplicar One-Hot Encoding sobre las 43 columnas originales del dataset NSL-KDD.
- **Artefactos del Modelo:**
  - `models/model.pkl`: Objeto del modelo IsolationForest.
  - `models/scaler.pkl`: Objeto StandardScaler pre-ajustado.
  - `models/feature_names.json`: Esquema estricto de las 121 columnas requeridas.
  - `models/model_registry.json`: Historial inmutable de las versiones del modelo.
- **Métricas de Desempeño Base (F1-Score):** ~0.82 (según `metrics.json`).

El modelo clasifica internamente el tráfico como `1` (Normal) o `-1` (Anomalía), y genera un `anomaly_score`. Un mayor score indica una mayor probabilidad de intrusión.

## 4. Pipeline de Mantenimiento e Integración Continua (CI/CD)

Para garantizar la fiabilidad del software y combatir el "concept drift" (degradación del modelo en el tiempo), se implementaron pipelines en GitHub Actions:

### 4.1. Integración Continua (`ci.yml`)
Se activa ante cualquier Push o Pull Request a la rama `main`.
- **Testing Unitario:** Ejecuta la suite de pruebas en el directorio `tests/` con `pytest`. Valida que las transformaciones tensoriales sean correctas `(1, 121)` y que el flujo de reentrenamiento cumpla sus invariantes.
- **Healthcheck:** Lanza la aplicación Streamlit temporalmente para confirmar que responde con HTTP 200 en su endpoint de estado, bloqueando despliegues de código defectuoso.

### 4.2. Despliegue Continuo y Reentrenamiento Automatizado (Batch / Programado)
Se ejecuta diariamente de forma programada mediante `retrain.yml` (CRON).
1. **Extracción:** Obtiene el tráfico reciente guardado en `traffic_log.db`.
2. **Entrenamiento:** Combina los datos de producción con una muestra del entrenamiento original y ajusta un nuevo `IsolationForest`.
3. **Validación:** Evalúa el nuevo modelo contra un conjunto de validación.
4. **Decisión Automática:** Si el nuevo `f1_score` es mayor o igual al F1 actual menos una tolerancia de `0.01`, el modelo es **promovido**.
5. **GitOps & Render Auto-Deploy:** Si es promovido, el bot realiza un commit directo a `main` con los nuevos archivos `.pkl` y `.json`. La plataforma de hosting (Render) detecta este cambio e inicia un redespliegue sin tiempo de inactividad.

### 4.3. Estrategia de Reentrenamiento Dual (Batch vs Reactivo)

El sistema emplea un **mecanismo dual** para combatir la degradación del modelo:
1. **Mecanismo Programado (Background/Cron):** Garantiza que el modelo se actualice periódicamente (diario) asimilando patrones paulatinos, independientemente de si hay alertas graves. Es asíncrono y transparente gracias a GitHub Actions.
2. **Mecanismo Reactivo (Event-Driven / In-App):** Responde a eventos críticos en cuestión de segundos. Se ejecuta *síncronamente* dentro del ciclo principal del simulador en `app.py` utilizando el módulo `trigger_monitor.py`.

#### Disparadores Reactivos (Triggers)
En la evaluación en vivo, existen 4 condiciones (evaluadas mediante operación lógica OR) que disparan inmediatamente un reentrenamiento de emergencia:

1. **Intervalo de tiempo (`time_interval`):** $\geq$ 180 segundos (3 minutos) desde el último reentrenamiento exitoso o fallido.
   *Justificación:* Asegura que incluso en escenarios de calma total, el modelo intente adaptarse frecuentemente al entorno cambiante de la simulación.
2. **Conteo de alertas (`alert_threshold`):** $\geq$ 15 anomalías detectadas acumuladas.
   *Justificación:* Una ráfaga de detecciones anómalas puede indicar un ataque coordinado, requiriendo un modelo que calibre rápidamente la nueva normalidad y frene falsos positivos.
3. **Ataque nuevo (`new_attack_type`):** El `ground_truth` del paquete entrante no pertenece a la lista de etiquetas presentes en el dataset de entrenamiento base.
   *Justificación:* Si la red enfrenta un Zero-Day (ataque nunca visto por el modelo original), el reentrenamiento inmediato es vital para asimilar las distribuciones matemáticas de esta nueva amenaza.
4. **Severidad extrema (`extreme_severity`):** El `anomaly_score` del paquete supera el percentil 95 calculado sobre las anomalías del entrenamiento base.
   *Justificación:* Una anomalía con desviación estadística extrema indica un comportamiento de red altísimamente inusual que amerita respuesta inmediata.

#### Interpretación del Registro (model_registry.json)
El campo `trigger_reason` en el historial del modelo diferenciará quién solicitó el reentrenamiento:
- `"scheduled_cron"` (Batch vía GitHub Actions)
- `"time_interval"`, `"alert_threshold"`, `"new_attack_type"`, `"extreme_severity"` (Reactivos vía aplicación en vivo).

> **Nota / Limitación Conocida:** El reentrenamiento reactivo se ejecuta de forma *síncrona*, por lo que detiene temporalmente la inyección y simulación de tráfico (típicamente durante ~2-5 segundos). En un entorno ultra-crítico de baja latencia se delegaría a un worker secundario (Ej. Celery o Kafka), pero para la complejidad académica de este proyecto es un *trade-off* aceptable que favorece la simplicidad arquitectónica.

```text
Flujo de Reentrenamiento Dual:

 [Tráfico Real/Simulado]  -->  (app.py - Inferencia)  --+
                                        |               |
                                        v               v
            ¿Se cumple un trigger Reactivo en vivo?   (Persistencia en SQLite)
                  | (Sí)                                |
                  v                                     v
       [run_retraining() síncrono]            [CRON Diario (GitHub Actions)]
                  |                                     |
                  +---------> [Evaluación de Métricas] <+
                                        |
                            ¿Nuevo modelo es mejor? --> (Sí) -> Promocionar & Guardar en model_registry.json
```

## 5. Tecnologías y Dependencias
- **Lenguaje:** Python 3.11 / 3.12
- **Framework Web:** Streamlit
- **Machine Learning:** Scikit-Learn 1.8.0, Pandas, Numpy, Joblib
- **Testing:** Pytest
- **Base de Datos:** SQLite3
- **CI/CD:** GitHub Actions
- **Despliegue:** Render (Web Service)
