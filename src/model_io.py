"""
src/model_io.py
---------------
Thin wrappers around joblib.load/dump for model artifacts.
No Streamlit dependency — safe to import from CLI scripts (retrain.py)
and test suites without a Streamlit runtime context.

Design note: preprocessing.py keeps its @st.cache_resource loaders for the
live app; retrain.py calls these bare functions instead so it doesn't need
the Streamlit runtime.
"""

import os
import joblib

_BASE_DIR    = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH   = os.path.join(_BASE_DIR, "models", "model.pkl")
SCALER_PATH  = os.path.join(_BASE_DIR, "models", "scaler.pkl")


def load_model_bare():
    """Load model.pkl without any Streamlit cache. Use in CLI / test contexts."""
    return joblib.load(MODEL_PATH)


def load_scaler_bare():
    """Load scaler.pkl without any Streamlit cache. Use in CLI / test contexts."""
    return joblib.load(SCALER_PATH)


def save_model(model) -> None:
    """Persist a trained model to models/model.pkl (overwrites)."""
    joblib.dump(model, MODEL_PATH)


def save_scaler(scaler) -> None:
    """Persist a fitted scaler to models/scaler.pkl (overwrites)."""
    joblib.dump(scaler, SCALER_PATH)
