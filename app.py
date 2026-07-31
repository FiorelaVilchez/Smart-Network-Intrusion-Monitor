"""
app.py — SNIM: Smart Network Intrusion Monitor
================================================
Streamlit entrypoint for the real-time anomaly detection dashboard.

Architecture overview:
  - src/preprocessing.py  → feature pipeline + model inference
  - src/simulator.py      → sequential CSV file reader (no full-memory load)
  - src/storage.py        → SQLite persistence (data/traffic_log.db)
  - src/ui_components.py  → reusable rendering functions

The simulation loop is driven by Streamlit's native st.rerun() mechanism:
after each record is processed the page reruns, creating the illusion of
a continuous real-time feed. The delay is implemented with time.sleep().
"""

import time
import datetime
import os
import streamlit as st

# Must be the FIRST Streamlit call
st.set_page_config(
    page_title="SNIM — Smart Network Intrusion Monitor",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS — dark theme, Inter font, polished look
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* App background */
.stApp {
    background: linear-gradient(135deg, #0D0D1A 0%, #0A0A16 50%, #0D1117 100%);
    color: #E0E0E0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0F0F24 0%, #0A0A1A 100%);
    border-right: 1px solid rgba(124, 77, 255, 0.2);
}

/* Metric tiles */
[data-testid="stMetric"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(124,77,255,0.2);
    border-radius: 12px;
    padding: 16px !important;
    transition: border-color 0.2s;
}
[data-testid="stMetric"]:hover {
    border-color: rgba(124,77,255,0.5);
}
[data-testid="stMetricValue"] {
    font-size: 1.8rem !important;
    font-weight: 600 !important;
    color: #FFFFFF !important;
}
[data-testid="stMetricLabel"] {
    color: #9E9E9E !important;
    font-size: 0.8rem !important;
}

/* Tab styling */
[data-testid="stTabs"] button {
    color: #9E9E9E;
    font-weight: 500;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #7C4DFF;
    border-bottom-color: #7C4DFF !important;
}

/* Buttons */
.stButton > button {
    border-radius: 8px;
    font-weight: 600;
    transition: all 0.2s;
}

/* Dividers */
hr {
    border-color: rgba(255,255,255,0.08) !important;
}

/* Alert/info boxes */
.stAlert {
    border-radius: 10px;
}

/* Dataframe */
[data-testid="stDataFrame"] {
    border-radius: 10px;
    overflow: hidden;
}

/* Sidebar headers */
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #7C4DFF;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Local imports (after st.set_page_config)
# ---------------------------------------------------------------------------
from src.simulator    import init_simulator, reset_simulator, read_next_record
from src.preprocessing import predict_record
from src.storage      import init_db, insert_record, get_last_n, get_total_count
from src.ui_components import (
    render_model_info_sidebar,
    render_kpi_row,
    render_traffic_table,
    render_anomaly_chart,
    render_score_distribution,
)
from src.trigger_monitor import evaluate_triggers
from src.retrain import run_retraining

# ---------------------------------------------------------------------------
# Initialisation (runs every rerun, but idempotent)
# ---------------------------------------------------------------------------
init_db()
init_simulator()

if "last_retrain_timestamp" not in st.session_state:
    st.session_state.last_retrain_timestamp = time.time()
if "alerts_since_last_retrain" not in st.session_state:
    st.session_state.alerts_since_last_retrain = 0
if "retrain_history" not in st.session_state:
    st.session_state.retrain_history = []

# ---------------------------------------------------------------------------
# ─────────────────────────────── SIDEBAR ──────────────────────────────────
# ---------------------------------------------------------------------------
with st.sidebar:
    # Logo / brand
    st.markdown("""
        <div style='text-align:center; padding: 10px 0 4px 0;'>
            <span style='font-size:2.8rem;'>🛡️</span>
            <h1 style='margin:0; font-size:1.3rem; font-weight:700;
                       background: linear-gradient(90deg,#7C4DFF,#E040FB);
                       -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
                SNIM
            </h1>
            <p style='color:#6B6B8A; font-size:0.72rem; margin:2px 0 0 0;'>
                Smart Network Intrusion Monitor
            </p>
        </div>
    """, unsafe_allow_html=True)

    st.markdown("## ⚙️ Controles de simulación")

    # Speed slider
    speed = st.slider(
        "Velocidad (delay entre registros)",
        min_value=0.1,
        max_value=2.0,
        value=0.5,
        step=0.1,
        format="%.1f s",
        help="Tiempo de espera entre la lectura de cada fila del dataset.",
    )

    col_start, col_reset = st.columns(2)

    # Start / Pause button
    if st.session_state.sim_running:
        if col_start.button("⏸ Pausar", use_container_width=True, type="primary"):
            st.session_state.sim_running = False
            st.rerun()
    else:
        if col_start.button("▶ Iniciar", use_container_width=True, type="primary"):
            if st.session_state.sim_start_time is None:
                st.session_state.sim_start_time = datetime.datetime.now()
            st.session_state.sim_running = True
            st.rerun()

    # Reset button
    if col_reset.button("🔄 Reset", use_container_width=True):
        reset_simulator()
        st.rerun()

    # Alerts-only toggle (stored in session_state for use in main panel)
    if "alerts_only" not in st.session_state:
        st.session_state.alerts_only = False

    st.session_state.alerts_only = st.toggle(
        "🚨 Mostrar solo alertas",
        value=st.session_state.alerts_only,
    )

    # Simulation status badge
    if st.session_state.sim_running:
        st.markdown(
            '<div style="text-align:center; margin-top:8px;">'
            '<span style="background:#1B5E20; color:#69F0AE; padding:4px 14px; '
            'border-radius:20px; font-size:0.78rem; font-weight:600;">● EN VIVO</span>'
            '</div>',
            unsafe_allow_html=True,
        )
    else:
        status_label = "⏸ PAUSADO" if st.session_state.sim_row_count > 0 else "⏹ DETENIDO"
        st.markdown(
            f'<div style="text-align:center; margin-top:8px;">'
            f'<span style="background:#1A1A2E; color:#9E9E9E; padding:4px 14px; '
            f'border-radius:20px; font-size:0.78rem; font-weight:600;">{status_label}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    # Loop notice
    if st.session_state.sim_looped:
        st.caption(
            f"♻️ El dataset ha completado {st.session_state.sim_loop_count} vuelta(s). "
            "Reiniciando desde el principio automáticamente."
        )
        st.session_state.sim_looped = False  # reset flag so it only shows once

    # Model info panel
    render_model_info_sidebar()

# ---------------------------------------------------------------------------
# ─────────────────────────────── MAIN AREA ────────────────────────────────
# ---------------------------------------------------------------------------

# Page header
st.markdown("""
    <h1 style='margin-bottom:0; background: linear-gradient(90deg,#7C4DFF,#E040FB,#FF5252);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;
               font-size:2rem; font-weight:700;'>
        🛡️ SNIM — Smart Network Intrusion Monitor
    </h1>
    <p style='color:#6B6B8A; font-size:0.88rem; margin-top:2px;'>
        Detección de intrusiones en red en tiempo real · Modelo: Isolation Forest · Dataset: NSL-KDD
    </p>
""", unsafe_allow_html=True)

st.divider()

# ---------------------------------------------------------------------------
# KPI metrics row
# ---------------------------------------------------------------------------
total_session = st.session_state.sim_row_count
alerts_session = st.session_state.sim_alert_count

# Compute uptime string
if st.session_state.sim_start_time is not None:
    elapsed = datetime.datetime.now() - st.session_state.sim_start_time
    h, rem   = divmod(int(elapsed.total_seconds()), 3600)
    m, s     = divmod(rem, 60)
    uptime_str = f"{h:02d}:{m:02d}:{s:02d}"
else:
    uptime_str = "00:00:00"

try:
    db_total = get_total_count()
except Exception:
    db_total = 0

render_kpi_row(total_session, alerts_session, uptime_str, db_total)

st.divider()

# ---------------------------------------------------------------------------
# Tabs: Tráfico en Vivo | Análisis | Acerca del modelo
# ---------------------------------------------------------------------------
tab_traffic, tab_analysis, tab_about = st.tabs([
    "📡 Tráfico en Vivo",
    "📊 Análisis",
    "ℹ️ Acerca del Modelo",
])

with tab_traffic:
    col_table, col_chart = st.columns([3, 2])

    with col_table:
        st.markdown("#### Últimos registros procesados")
        render_traffic_table(
            st.session_state.sim_history,
            alerts_only=st.session_state.alerts_only,
        )

    with col_chart:
        st.markdown("#### Tendencia de anomalías")
        render_anomaly_chart(st.session_state.sim_history)

with tab_analysis:
    st.markdown("#### Distribución de scores de anomalía")
    render_score_distribution(st.session_state.sim_history)

    # Session stats
    if total_session > 0:
        st.divider()
        st.markdown("#### 📋 Estadísticas de la sesión")
        col_a, col_b, col_c = st.columns(3)
        col_a.metric("Registros procesados", f"{total_session:,}")
        col_b.metric("Anomalías detectadas", f"{alerts_session:,}")
        col_c.metric("Registros en BD total", f"{db_total:,}")

        # Protocol breakdown from in-memory history
        if st.session_state.sim_history:
            import pandas as pd
            hdf = pd.DataFrame(st.session_state.sim_history)
            if "protocol_type" in hdf.columns:
                st.markdown("**Distribución por protocolo (sesión)**")
                proto_counts = hdf["protocol_type"].value_counts().reset_index()
                proto_counts.columns = ["Protocolo", "Conteo"]
                st.bar_chart(proto_counts.set_index("Protocolo"), color="#7C4DFF")
    else:
        st.info("Inicia la simulación para ver estadísticas de análisis.")

with tab_about:
    import json

    _base = os.path.dirname(os.path.abspath(__file__))

    st.markdown("### 🔬 Información técnica del modelo")

    col_m, col_p = st.columns(2)

    with col_m:
        st.markdown("#### 📊 Métricas de referencia (test set NSL-KDD)")
        with open(os.path.join(_base, "models", "metrics.json")) as f:
            metrics = json.load(f)
        metric_names = {
            "accuracy":          "Accuracy",
            "precision":         "Precision",
            "recall":            "Recall",
            "f1_score":          "F1-Score",
            "roc_auc":           "ROC-AUC",
            "average_precision": "Avg. Precision",
        }
        import pandas as pd
        mdf = pd.DataFrame([
            {"Métrica": metric_names.get(k, k), "Valor": f"{v:.4f}", "Porcentaje": f"{v:.1%}"}
            for k, v in metrics.items()
        ])
        st.dataframe(mdf, use_container_width=True, hide_index=True)

    with col_p:
        st.markdown("#### ⚙️ Hiperparámetros del modelo")
        with open(os.path.join(_base, "models", "best_params.json")) as f:
            params = json.load(f)
        param_desc = {
            "n_estimators":  "Número de árboles en el ensemble",
            "contamination": "Proporción esperada de anomalías",
            "max_samples":   "Fracción de muestras por árbol",
            "max_features":  "Fracción de features por árbol",
        }
        pdf = pd.DataFrame([
            {"Parámetro": k, "Valor": str(v), "Descripción": param_desc.get(k, "")}
            for k, v in params.items()
        ])
        st.dataframe(pdf, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("#### 📜 Registro de versiones del modelo")
    with open(os.path.join(_base, "models", "model_registry.json")) as f:
        registry = json.load(f)
    reg_df = pd.DataFrame([
        {
            "Versión": r["version"],
            "Estado": r["status"].upper(),
            "Timestamp": r["timestamp"][:19].replace("T", " "),
            "Estimadores": r["hyperparameters"]["n_estimators"],
            "Accuracy": f"{r['metrics']['accuracy']:.4f}",
            "F1-Score": f"{r['metrics']['f1_score']:.4f}",
            "ROC-AUC": f"{r['metrics']['roc_auc']:.4f}",
            "Muestras": f"{r['training_samples']:,}",
        }
        for r in registry
    ])
    st.dataframe(reg_df, use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### ⚡ Reentrenamiento Reactivo (En Vivo)")
    st.markdown("El sistema cuenta con un mecanismo reactivo que evalúa cada paquete en vivo para decidir si se necesita reentrenar inmediatamente, además del reentrenamiento diario programado.")
    
    col_rt1, col_rt2 = st.columns(2)
    with col_rt1:
        elapsed_retrain = int(time.time() - st.session_state.last_retrain_timestamp)
        st.metric("Tiempo desde último reentrenamiento", f"{elapsed_retrain} s", help="Se dispara a los 180s")
    with col_rt2:
        st.metric("Alertas desde último reentrenamiento", st.session_state.alerts_since_last_retrain, help="Se dispara a las 15 alertas")
        
    st.markdown("#### Historial de Reentrenamientos en Sesión")
    if st.session_state.retrain_history:
        rh_df = pd.DataFrame(st.session_state.retrain_history)
        st.dataframe(rh_df, use_container_width=True, hide_index=True)
    else:
        st.info("No se ha disparado ningún reentrenamiento reactivo en esta sesión.")

    st.divider()
    st.markdown("""
        #### 🏗️ Arquitectura del pipeline

        ```
        KDDTest+.txt (fila cruda)
              ↓
        [ COLUMN_SCHEMA.json ] — 43 columnas → parse y tipado
              ↓
        Descartar: label, difficulty_level, num_outbound_cmds
              ↓
        pd.get_dummies(protocol_type, service, flag)
              ↓
        .reindex(feature_names.json, fill_value=0)  — alinear a 121 features
              ↓
        StandardScaler.transform()   (scaler.pkl — NUNCA fit_transform)
              ↓
        IsolationForest.predict()    → 1 (normal) / -1 (anomalía)
        IsolationForest.score_samples() → negado → score de anomalía
              ↓
        Dashboard + SQLite (traffic_log.db)
        ```
    """)

# ---------------------------------------------------------------------------
# ────────────────── SIMULATION LOOP (bottom of script) ───────────────────
# ---------------------------------------------------------------------------
# When sim_running is True, we process ONE record per rerun and then
# schedule another rerun after `speed` seconds. This keeps Streamlit
# responsive (no blocking loop) while creating a smooth real-time feed.

if st.session_state.sim_running:
    try:
        raw_row = read_next_record()

        if raw_row is not None:
            # Run model inference
            prediction = predict_record(raw_row)

            # Build the enriched record for in-memory history
            ts = datetime.datetime.now().isoformat(timespec="seconds")
            record = {
                "timestamp":      ts,
                "ground_truth":   str(raw_row.get("label", "unknown")),
                "is_anomaly":     prediction["is_anomaly"],
                "anomaly_score":  round(prediction["anomaly_score"], 6),
                "prediction_raw": prediction["prediction_raw"],
                "protocol_type":  raw_row.get("protocol_type", ""),
                "service":        raw_row.get("service", ""),
                "flag":           raw_row.get("flag", ""),
                "src_bytes":      raw_row.get("src_bytes", 0),
                "dst_bytes":      raw_row.get("dst_bytes", 0),
                "duration":       raw_row.get("duration", 0),
                "count":          raw_row.get("count", 0),
                "srv_count":      raw_row.get("srv_count", 0),
                "serror_rate":    raw_row.get("serror_rate", 0),
                "rerror_rate":    raw_row.get("rerror_rate", 0),
            }

            # Update session-state counters
            st.session_state.sim_row_count   += 1
            if prediction["is_anomaly"]:
                st.session_state.sim_alert_count += 1

            # Append to in-memory ring buffer (cap at 500 to avoid memory creep)
            st.session_state.sim_history.append(record)
            if len(st.session_state.sim_history) > 500:
                st.session_state.sim_history = st.session_state.sim_history[-500:]

            # Persist to SQLite
            insert_record(ts, record["ground_truth"], prediction, raw_row)
            
            # Evaluate Reactive Retraining Triggers
            if prediction["is_anomaly"]:
                st.session_state.alerts_since_last_retrain += 1
                
            should_trigger, reason_code, reason_msg = evaluate_triggers(
                is_anomaly=prediction["is_anomaly"],
                anomaly_score=record["anomaly_score"],
                ground_truth_label=record["ground_truth"],
                last_retrain_time=st.session_state.last_retrain_timestamp,
                alerts_since_last_retrain=st.session_state.alerts_since_last_retrain
            )
            
            if should_trigger:
                st.toast(f"🔄 Reentrenamiento automático disparado — Motivo: {reason_msg}", icon="🔄")
                
                # Sincronous retraining
                with st.spinner(f"Reentrenando modelo... ({reason_msg})"):
                    res = run_retraining(trigger_reason=reason_code, force_promote=False, sample_size=10000)
                
                # Reset counters unconditionally
                st.session_state.last_retrain_timestamp = time.time()
                st.session_state.alerts_since_last_retrain = 0
                
                st.session_state.retrain_history.insert(0, {
                    "Timestamp": datetime.datetime.now().isoformat(timespec="seconds").replace("T", " "),
                    "Motivo": reason_msg,
                    "Promovido": "✅ Sí" if res["promoted"] else "❌ No",
                    "F1 Nuevo": f"{res['new_f1']:.4f}",
                    "F1 Anterior": f"{res['current_f1']:.4f}"
                })
                if len(st.session_state.retrain_history) > 5:
                    st.session_state.retrain_history = st.session_state.retrain_history[:5]

        # Sleep for the configured delay, then trigger a rerun
        time.sleep(speed)
        st.rerun()

    except FileNotFoundError as exc:
        st.session_state.sim_running = False
        st.error(str(exc))
    except Exception as exc:
        st.session_state.sim_running = False
        st.error(f"Error durante la simulación: {exc}")
        raise
