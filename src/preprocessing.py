"""
src/preprocessing.py
--------------------
Preprocessing pipeline that replicates the exact feature engineering applied
during model training. This module must NOT be changed unless the model is
retrained with a different pipeline.

Key invariants:
  - Column order from COLUMN_SCHEMA.json must be respected.
  - One-Hot Encoding via pd.get_dummies on categorical columns only.
  - Reindex against feature_names.json with fill_value=0 (single-row inference
    never generates all dummy categories, so missing ones become 0).
  - scaler.transform (never fit_transform) — the scaler is already fitted.
  - IsolationForest: predict() returns 1 (normal) / -1 (anomaly).
  - score_samples() is negated so that higher score means more anomalous.
"""

import json
import os
import numpy as np
import pandas as pd
import joblib
import streamlit as st

# ---------------------------------------------------------------------------
# Paths (relative to project root, resolved at import time)
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SCHEMA_PATH        = os.path.join(_BASE_DIR, "COLUMN_SCHEMA.json")
_FEATURE_NAMES_PATH = os.path.join(_BASE_DIR, "models", "feature_names.json")
_MODEL_PATH         = os.path.join(_BASE_DIR, "models", "model.pkl")
_SCALER_PATH        = os.path.join(_BASE_DIR, "models", "scaler.pkl")

# ---------------------------------------------------------------------------
# Schema & feature-name constants (loaded once at module level)
# ---------------------------------------------------------------------------
with open(_SCHEMA_PATH, "r") as f:
    _SCHEMA = json.load(f)

COLUMN_NAMES      = _SCHEMA["columns"]          # 43 raw column names in order
CATEGORICAL_COLS  = _SCHEMA["categorical_columns"]  # protocol_type, service, flag
DROP_COLS         = _SCHEMA["drop_columns"]     # label, difficulty_level, num_outbound_cmds
LABEL_COL         = _SCHEMA["label_column"]     # "label" – kept for ground truth display only

with open(_FEATURE_NAMES_PATH, "r") as f:
    FEATURE_NAMES = json.load(f)                # 121 columns expected by the model


# ---------------------------------------------------------------------------
# Cached model & scaler loaders (Streamlit resource cache)
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="Cargando modelo de detección…")
def load_model():
    """Load the pre-trained IsolationForest from disk. Cached for the session."""
    return joblib.load(_MODEL_PATH)


@st.cache_resource(show_spinner="Cargando scaler…")
def load_scaler():
    """Load the pre-fitted StandardScaler from disk. Cached for the session."""
    return joblib.load(_SCALER_PATH)


# ---------------------------------------------------------------------------
# Core preprocessing function
# ---------------------------------------------------------------------------
def preprocess_record(raw_row: dict) -> np.ndarray:
    """Transform a single raw NSL-KDD record into the feature array the model
    expects.

    Steps (must mirror training pipeline exactly):
      1. Build a single-row DataFrame using the raw column names.
      2. Drop label, difficulty_level, num_outbound_cmds.
      3. Apply pd.get_dummies on categorical columns.
      4. Reindex to the 121 feature_names (fill missing with 0).
      5. Apply StandardScaler.transform.

    Args:
        raw_row: dict mapping each of the 43 column names to its value.
                 The 'label' key is allowed (and ignored by this function).

    Returns:
        np.ndarray of shape (1, 121), ready for model.predict / score_samples.
    """
    # Step 1: Single-row DataFrame
    df = pd.DataFrame([raw_row], columns=COLUMN_NAMES)

    # Step 2: Drop non-feature columns
    df = df.drop(columns=DROP_COLS, errors="ignore")

    # Step 3: One-Hot Encoding (same as pd.get_dummies used in training)
    df = pd.get_dummies(df, columns=CATEGORICAL_COLS)

    # Step 4: Align to the exact 121-feature schema; fill unseen dummies with 0
    df = df.reindex(columns=FEATURE_NAMES, fill_value=0)

    # Step 5: Scale — transform only, scaler is already fitted.
    # Pass the DataFrame (not .values) so sklearn doesn't warn about missing
    # feature names (the scaler was fitted on a named DataFrame).
    scaler = load_scaler()
    X = scaler.transform(df)

    return X  # shape (1, 121)


# ---------------------------------------------------------------------------
# Prediction function
# ---------------------------------------------------------------------------
def predict_record(raw_row: dict) -> dict:
    """Run the full preprocessing + inference pipeline on a single raw record.

    Args:
        raw_row: dict with 43 NSL-KDD fields (including 'label').

    Returns:
        dict with keys:
          - is_anomaly (bool): True if the model flagged this record.
          - anomaly_score (float): Higher = more anomalous (negated score_samples).
          - prediction_raw (int): Raw model output (1 = normal, -1 = anomaly).
    """
    model = load_model()
    X = preprocess_record(raw_row)

    prediction_raw: int  = int(model.predict(X)[0])          # 1 or -1
    anomaly_score: float = float(-model.score_samples(X)[0]) # negate → higher = worse

    return {
        "is_anomaly":     prediction_raw == -1,
        "anomaly_score":  anomaly_score,
        "prediction_raw": prediction_raw,
    }
