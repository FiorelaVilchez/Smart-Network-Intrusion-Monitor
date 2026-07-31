---
title: SNIM Smart Network Intrusion Monitor
emoji: 🛡️
colorFrom: blue
colorTo: red
sdk: streamlit
sdk_version: "1.38.0"
app_file: app.py
pinned: false
---

# 🛡️ SNIM — Smart Network Intrusion Monitor

**Real-time network intrusion detection dashboard** built on top of a pre-trained **Isolation Forest** model trained on the NSL-KDD dataset.

> Universidad project — Machine Learning, 2nd Unit · 2026

---

## ✨ Features

| Feature | Detail |
|---|---|
| **Live traffic simulation** | Streams `KDDTest+.txt` row-by-row with configurable speed (0.1–2.0 s/record) |
| **Real-time anomaly detection** | Every record passes through the full preprocessing pipeline and Isolation Forest inference |
| **Persistent log** | All processed records stored in `data/traffic_log.db` (SQLite) — survives app restarts |
| **Interactive dashboard** | KPI metrics, color-coded traffic table, anomaly trend chart, score distribution |
| **Model info panel** | Version registry, hyperparameters, and reference metrics visible at all times |

---

## 🏗️ Architecture

```
KDDTest+.txt (raw row)
      ↓
COLUMN_SCHEMA.json — 43-column parse & type coercion
      ↓
Drop: label, difficulty_level, num_outbound_cmds
      ↓
pd.get_dummies(protocol_type, service, flag)
      ↓
.reindex(feature_names.json, fill_value=0)  →  121 features
      ↓
StandardScaler.transform()      (scaler.pkl — NEVER fit_transform)
      ↓
IsolationForest.predict()       →  1 (normal) / -1 (anomaly)
IsolationForest.score_samples() →  negated  →  anomaly score
      ↓
Dashboard + SQLite (data/traffic_log.db)
```

---

## 🚀 Running locally

### Prerequisites
- Python 3.12+
- The `data/KDDTest+.txt` file (NSL-KDD test dataset, no header, CSV format)

### Install & run

```bash
# 1. Clone / copy the repository
# 2. Install dependencies (exact versions required to match the training env)
pip install -r requirements.txt

# 3. Launch the app
streamlit run app.py
```

The app will open at [http://localhost:8501](http://localhost:8501).

### First run checklist
- `models/model.pkl` ✅
- `models/scaler.pkl` ✅
- `models/feature_names.json` ✅
- `data/KDDTest+.txt` ✅ (downloaded automatically if missing, or place manually)
- `data/traffic_log.db` — created automatically on first run

---

## 🐳 Docker

```bash
# Build
docker build -t snim .

# Run (port 7860)
docker run -p 7860:7860 snim
```

Open [http://localhost:7860](http://localhost:7860).

---

## 📁 Project structure

```
├── app.py                  # Streamlit entrypoint
├── COLUMN_SCHEMA.json      # 43-column raw schema definition
├── requirements.txt        # Pinned Python dependencies
├── Dockerfile              # Container definition (port 7860)
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── preprocessing.py    # preprocess_record, predict_record
│   ├── simulator.py        # Sequential CSV reader with persistent pointer
│   ├── storage.py          # SQLite: init_db, insert_record, get_last_n
│   └── ui_components.py    # Reusable dashboard rendering functions
├── models/                 # Pre-trained artifacts (DO NOT MODIFY)
│   ├── model.pkl
│   ├── scaler.pkl
│   ├── feature_names.json
│   ├── best_params.json
│   ├── metrics.json
│   └── model_registry.json
└── data/
    ├── KDDTest+.txt        # NSL-KDD test set (22,544 rows)
    └── traffic_log.db      # Persistent SQLite log (auto-created)
```

---

## 🔄 Mantenimiento e Integración Continua (CI/CD)

El proyecto cuenta con pipelines automatizados configurados a través de GitHub Actions para asegurar la calidad y mejorar el modelo de forma continua.

### 1. Integración Continua (CI)
Cada vez que se realiza un `push` o un `pull_request` a la rama `main`, se ejecuta el workflow `ci.yml`:
- **Pruebas Automatizadas:** Se ejecutan tests unitarios (con `pytest`) para validar que la lógica de preprocesamiento, carga de modelos y reglas de reentrenamiento operen correctamente.
- **Healthcheck:** La aplicación Streamlit se inicializa en background y se realiza un request de verificación, garantizando que el dashboard no arroje errores fatales al arrancar.

### 2. Mantenimiento y Reentrenamiento (CD)
El mantenimiento del modelo está orquestado en `retrain.yml`, el cual corre diariamente a las 3:00 AM UTC o de forma manual. Este proceso llama al script `src/retrain.py`:
- **Extracción de datos:** Obtiene los registros históricos desde `data/traffic_log.db` (tráfico de producción simulado).
- **Entrenamiento Híbrido:** Combina estos registros con una muestra del dataset original (`KDDTrain+.txt`) y reentrena un nuevo Isolation Forest.
- **Validación Automática:** Compara el `f1_score` del nuevo modelo con el del modelo actual (`models/metrics.json`). Si el nuevo modelo es mejor o no empeora más de `0.01`, se promociona.
- **Registro:** Se anota la decisión en `models/model_registry.json`. Puedes revisar este archivo para auditar todo el historial y las decisiones (rechazos y promociones).
- **Despliegue Continuo a Render:** Si el modelo es promovido, el workflow hace commit y push de los nuevos artefactos de modelo a la rama `main`. Dado que el proyecto está conectado a Render con Auto-Deploy, Render detectará el push automáticamente y redesplegará el servicio en producción usando los nuevos artefactos **sin necesidad de intervención manual o secretos de despliegue extra**.

---

## 📊 Model reference metrics (NSL-KDD test set)

| Metric | Value |
|---|---|
| Accuracy | 81.74% |
| Precision | 91.40% |
| Recall | 74.98% |
| F1-Score | 82.38% |
| ROC-AUC | 95.12% |
| Avg. Precision | 95.77% |

**Model:** `IsolationForest`
**Hyperparameters:** `n_estimators=100, contamination=0.05, max_samples=0.5, max_features=0.7`
**Training samples:** 67,343

---

## ⚠️ Important notes

- `scaler.transform()` is **always** used — never `fit_transform()`. The scaler is pre-fitted on training data.
- The `label` column is **only** used to display ground-truth in the dashboard. It is **never** passed to the model.
- `data/traffic_log.db` is intentionally **not** in `.gitignore` so it persists across deployments in this demo.
- Package versions in `requirements.txt` are pinned to exactly match the training environment (`scikit-learn==1.8.0`, `joblib==1.5.3`) to prevent `joblib.load()` failures.

---

*Built with [Streamlit](https://streamlit.io) · Model trained with [scikit-learn](https://scikit-learn.org)*
