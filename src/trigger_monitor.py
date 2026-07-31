"""
src/trigger_monitor.py
----------------------
Módulo responsable de monitorizar en tiempo real el tráfico procesado por la
aplicación para decidir si se debe disparar un reentrenamiento reactivo.

Existen 4 disparadores (triggers):
a) Intervalo de tiempo (>= 180s desde el último reentrenamiento).
b) Conteo de alertas (>= 15 anomalías detectadas).
c) Ataque nuevo no visto en el conjunto original de entrenamiento.
d) Severidad extrema (anomaly_score superior al percentil 95 del conjunto de entrenamiento).
"""

import os
import time
import numpy as np
import pandas as pd
import streamlit as st
from src.preprocessing import COLUMN_NAMES, DROP_COLS, CATEGORICAL_COLS, FEATURE_NAMES
from src.retrain import prepare_features
from src.model_io import load_model_bare, load_scaler_bare

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DATA_PATH = os.path.join(_BASE_DIR, "data", "KDDTrain+.txt")

@st.cache_resource(show_spinner=False)
def load_known_attack_types() -> set:
    """Carga y cachea la lista de etiquetas conocidas (normal + tipos de ataque)
    presentes en el dataset de entrenamiento original."""
    if not os.path.exists(TRAIN_DATA_PATH):
        # Fallback si no existe el archivo aún
        return {"normal", "neptune", "smurf", "guess_passwd", "pod", "teardrop", "portsweep", "ipsweep", "land", "ftp_write", "back", "imap", "satan", "phf", "nmap", "multihop", "warezmaster", "warezclient", "spy", "rootkit"}
    
    # Leer solo la columna 'label' para mayor eficiencia
    # En NSL-KDD, la columna label es la número 41 (índice 41 si 0-indexed), penúltima.
    # Pero para estar seguros cargamos usando los nombres.
    df = pd.read_csv(TRAIN_DATA_PATH, header=None, names=COLUMN_NAMES, usecols=["label"])
    return set(df["label"].unique())

@st.cache_resource(show_spinner=False)
def load_extreme_severity_threshold() -> float:
    """Calcula y cachea el percentil 95 de los anomaly_scores en el dataset de entrenamiento."""
    if not os.path.exists(TRAIN_DATA_PATH):
        return 0.7  # Umbral por defecto razonable
        
    df = pd.read_csv(TRAIN_DATA_PATH, header=None, names=COLUMN_NAMES)
    # Tomar una muestra para no procesar los 125k registros
    df_sample = df.sample(n=min(10000, len(df)), random_state=42).copy()
    
    # Reutilizar código de preprocesamiento, pero para ser independientes, lo importamos de preprocessing
    # Wait, prepare_features no está en preprocessing, está en retrain.py. Lo extraeré o lo reimplementaré corto aquí.
    
    X = df_sample.drop(columns=DROP_COLS, errors="ignore")
    for col in COLUMN_NAMES:
        if col not in X.columns and col not in DROP_COLS:
            if col in CATEGORICAL_COLS:
                X[col] = "unknown"
            else:
                X[col] = 0.0
    X = pd.get_dummies(X, columns=CATEGORICAL_COLS)
    X = X.reindex(columns=FEATURE_NAMES, fill_value=0)
    
    scaler = load_scaler_bare()
    X_scaled = scaler.transform(X)
    
    model = load_model_bare()
    scores = -model.score_samples(X_scaled)
    
    return float(np.percentile(scores, 95))

def evaluate_triggers(
    is_anomaly: bool,
    anomaly_score: float,
    ground_truth_label: str,
    last_retrain_time: float,
    alerts_since_last_retrain: int
) -> tuple[bool, str, str]:
    """
    Evalúa los 4 disparadores. Retorna (should_trigger, reason_code, human_reason).
    """
    current_time = time.time()
    
    # a) Intervalo de tiempo
    if (current_time - last_retrain_time) >= 180:
        return True, "time_interval", "Ya pasó el tiempo programado (3 minutos)."
        
    # b) Conteo de alertas
    if alerts_since_last_retrain >= 15:
        return True, "alert_threshold", "Se acumularon varias alertas (>= 15)."
        
    # c) Ataque nuevo no visto
    known_types = load_known_attack_types()
    if ground_truth_label not in known_types:
        return True, "new_attack_type", f"Apareció un tipo de ataque que el sistema no conocía: '{ground_truth_label}'."
        
    # d) Severidad extrema
    extreme_threshold = load_extreme_severity_threshold()
    if anomaly_score > extreme_threshold:
        return True, "extreme_severity", f"Se detectó un ataque muy grave (score={anomaly_score:.2f}, umbral={extreme_threshold:.2f})."
        
    return False, "", ""
