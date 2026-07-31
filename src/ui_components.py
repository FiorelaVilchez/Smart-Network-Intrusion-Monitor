"""
src/ui_components.py
--------------------
Reusable Streamlit rendering functions that keep app.py clean.

All UI text is in Spanish (for the university audience).
Comments and docstrings are in English (for the technical report).
"""

import json
import os
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ---------------------------------------------------------------------------
# Paths to model metadata files
# ---------------------------------------------------------------------------
_BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_METRICS_PATH  = os.path.join(_BASE_DIR, "models", "metrics.json")
_PARAMS_PATH   = os.path.join(_BASE_DIR, "models", "best_params.json")
_REGISTRY_PATH = os.path.join(_BASE_DIR, "models", "model_registry.json")


# ---------------------------------------------------------------------------
# Helper: load JSON with cache
# ---------------------------------------------------------------------------
@st.cache_data
def _load_json(path: str) -> dict | list:
    with open(path, "r") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Model info sidebar panel
# ---------------------------------------------------------------------------
def render_model_info_sidebar() -> None:
    """Render the 'Información del modelo' panel in the Streamlit sidebar.

    Displays: active version from model_registry.json, hyperparameters from
    best_params.json, and reference metrics from metrics.json.
    """
    st.sidebar.markdown("---")
    st.sidebar.markdown("## 🔬 Información del modelo")

    # Active version from registry
    registry: list = _load_json(_REGISTRY_PATH)
    active = next((r for r in registry if r.get("status") == "production"), registry[-1])

    st.sidebar.markdown(f"**Versión activa:** `v{active['version']}`")
    st.sidebar.markdown(f"**Estado:** `{active['status'].upper()}`")
    st.sidebar.markdown(f"**Muestras de entrenamiento:** `{active['training_samples']:,}`")
    ts = active.get("timestamp", "N/A")[:19].replace("T", " ")
    st.sidebar.markdown(f"**Entrenado el:** `{ts}`")

    # Hyperparameters
    st.sidebar.markdown("#### ⚙️ Hiperparámetros")
    params: dict = _load_json(_PARAMS_PATH)
    param_labels = {
        "n_estimators":  "Estimadores",
        "contamination": "Contaminación",
        "max_samples":   "Max samples",
        "max_features":  "Max features",
    }
    for k, v in params.items():
        label = param_labels.get(k, k)
        st.sidebar.markdown(f"- **{label}:** `{v}`")

    # Reference metrics
    st.sidebar.markdown("#### 📊 Métricas de referencia")
    metrics: dict = _load_json(_METRICS_PATH)
    metric_labels = {
        "accuracy":           ("Accuracy",           "🎯"),
        "precision":          ("Precision",          "🔍"),
        "recall":             ("Recall",             "📡"),
        "f1_score":           ("F1-Score",           "⚖️"),
        "roc_auc":            ("ROC-AUC",            "📈"),
        "average_precision":  ("Avg. Precision",     "🏆"),
    }
    for k, (label, emoji) in metric_labels.items():
        v = metrics.get(k, "N/A")
        pct = f"{v:.1%}" if isinstance(v, float) else str(v)
        st.sidebar.markdown(f"{emoji} **{label}:** `{pct}`")

    st.sidebar.markdown("---")


# ---------------------------------------------------------------------------
# Top KPI metrics row
# ---------------------------------------------------------------------------
def render_kpi_row(
    total: int,
    alerts: int,
    uptime_str: str,
    db_total: int,
) -> None:
    """Render the four top-level KPI metric tiles.

    Args:
        total:      Records processed in the current session.
        alerts:     Anomalies detected in the current session.
        uptime_str: Human-readable uptime string (e.g. "00:02:34").
        db_total:   Total records ever stored in traffic_log.db.
    """
    alert_rate = (alerts / total * 100) if total > 0 else 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📦 Total procesados", f"{total:,}", help="Registros procesados en la sesión actual")
    c2.metric("🚨 Total alertas",    f"{alerts:,}", help="Anomalías detectadas en la sesión actual")
    c3.metric("⚠️ Tasa de alertas",  f"{alert_rate:.1f}%", help="Porcentaje de tráfico marcado como anómalo")
    c4.metric("⏱️ Uptime",           uptime_str,   help="Tiempo transcurrido desde el inicio de la simulación")


# ---------------------------------------------------------------------------
# Traffic table
# ---------------------------------------------------------------------------
def render_traffic_table(history: list, alerts_only: bool = False) -> None:
    """Render the live traffic table from the in-memory history buffer.

    Anomalous rows are highlighted in red; normal rows in green.

    Args:
        history:     List of record dicts (most-recent first, up to 500).
        alerts_only: If True, display only anomalous records.
    """
    if not history:
        st.info("⏳ Esperando datos de la simulación…")
        return

    # Show last 20 from history (history is appended chronologically)
    recent = history[-20:][::-1]  # reverse to show newest first

    display_cols = [
        "timestamp", "ground_truth", "is_anomaly", "anomaly_score",
        "protocol_type", "service", "flag",
        "src_bytes", "dst_bytes", "duration",
    ]

    df = pd.DataFrame(recent)
    # Ensure all display columns exist
    for col in display_cols:
        if col not in df.columns:
            df[col] = ""

    df = df[display_cols].copy()

    if alerts_only:
        df = df[df["is_anomaly"] == True]
        if df.empty:
            st.info("✅ No se han detectado alertas aún.")
            return

    # Pretty-format
    df["is_anomaly"]    = df["is_anomaly"].map({True: "🚨 ALERTA", False: "✅ Normal"})
    df["anomaly_score"] = df["anomaly_score"].round(4)
    df.columns = [
        "Timestamp", "Ground Truth", "Estado", "Score Anomalía",
        "Protocolo", "Servicio", "Flag",
        "Src Bytes", "Dst Bytes", "Duración",
    ]

    def _row_style(row):
        """Apply background colour based on anomaly status."""
        colour = "rgba(255,80,80,0.18)" if row["Estado"] == "🚨 ALERTA" else "rgba(80,200,120,0.12)"
        return [f"background-color: {colour}"] * len(row)

    styled = df.style.apply(_row_style, axis=1)
    st.dataframe(styled, use_container_width=True, height=420)


# ---------------------------------------------------------------------------
# Anomaly trend chart
# ---------------------------------------------------------------------------
def render_anomaly_chart(history: list) -> None:
    """Render a Plotly area chart showing the cumulative anomaly count over time.

    Uses a sliding window of the last 200 records from history.

    Args:
        history: List of processed record dicts (chronological order).
    """
    if len(history) < 2:
        st.info("📈 El gráfico aparecerá una vez haya suficientes datos.")
        return

    window = history[-200:]
    timestamps   = [r["timestamp"][11:19] for r in window]  # HH:MM:SS
    anomaly_flags = [1 if r.get("is_anomaly") else 0 for r in window]

    # Cumulative sum within the window
    cumulative = []
    total = 0
    for flag in anomaly_flags:
        total += flag
        cumulative.append(total)

    fig = go.Figure()

    # Area trace for cumulative anomalies
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=cumulative,
        mode="lines",
        fill="tozeroy",
        name="Anomalías acumuladas",
        line=dict(color="#FF5252", width=2),
        fillcolor="rgba(255,82,82,0.15)",
    ))

    # Normal traffic count (total - anomalies)
    normal_count = [i + 1 - c for i, c in enumerate(cumulative)]
    fig.add_trace(go.Scatter(
        x=timestamps,
        y=normal_count,
        mode="lines",
        fill="tozeroy",
        name="Tráfico normal acumulado",
        line=dict(color="#00E676", width=2),
        fillcolor="rgba(0,230,118,0.10)",
        visible="legendonly",
    ))

    fig.update_layout(
        title=dict(text="Anomalías detectadas (ventana deslizante – últimos 200 registros)",
                   font=dict(size=14, color="#E0E0E0")),
        xaxis=dict(title="Hora", showgrid=False, color="#9E9E9E",
                   tickangle=-45, nticks=10),
        yaxis=dict(title="Conteo acumulado", showgrid=True,
                   gridcolor="rgba(255,255,255,0.05)", color="#9E9E9E"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1, font=dict(color="#E0E0E0")),
        margin=dict(l=10, r=10, t=60, b=40),
        font=dict(family="Inter, sans-serif"),
    )

    st.plotly_chart(fig, use_container_width=True)


# ---------------------------------------------------------------------------
# Score distribution mini-chart
# ---------------------------------------------------------------------------
def render_score_distribution(history: list) -> None:
    """Render a histogram of anomaly scores for the current session window.

    Helps the operator understand the model's confidence distribution.
    """
    if len(history) < 10:
        return

    window = history[-300:]
    scores = [r.get("anomaly_score", 0) for r in window]

    fig = go.Figure()
    fig.add_trace(go.Histogram(
        x=scores,
        nbinsx=40,
        marker_color="#7C4DFF",
        opacity=0.8,
        name="Score",
    ))
    fig.update_layout(
        title=dict(text="Distribución de scores de anomalía (últimos 300 registros)",
                   font=dict(size=13, color="#E0E0E0")),
        xaxis=dict(title="Anomaly Score", color="#9E9E9E", showgrid=False),
        yaxis=dict(title="Frecuencia", color="#9E9E9E",
                   gridcolor="rgba(255,255,255,0.05)"),
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=10, r=10, t=50, b=30),
        font=dict(family="Inter, sans-serif"),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)
