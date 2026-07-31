"""
scratch/test_pipeline.py
Quick smoke-test of the preprocessing pipeline (no Streamlit needed).
Run with: python scratch/test_pipeline.py
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import json
import joblib
import pandas as pd
import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load schema
with open(os.path.join(BASE, "COLUMN_SCHEMA.json")) as f:
    schema = json.load(f)

COLUMN_NAMES     = schema["columns"]
CATEGORICAL_COLS = schema["categorical_columns"]
DROP_COLS        = schema["drop_columns"]

with open(os.path.join(BASE, "models", "feature_names.json")) as f:
    FEATURE_NAMES = json.load(f)

print(f"Column schema loaded: {len(COLUMN_NAMES)} raw columns")
print(f"Feature names loaded: {len(FEATURE_NAMES)} expected features")

# Load model and scaler
model  = joblib.load(os.path.join(BASE, "models", "model.pkl"))
scaler = joblib.load(os.path.join(BASE, "models", "scaler.pkl"))
print(f"Model loaded: {type(model).__name__}")
print(f"Scaler loaded: {type(scaler).__name__}")

# Read first 5 rows of KDDTest+.txt
data_path = os.path.join(BASE, "data", "KDDTest+.txt")
df = pd.read_csv(data_path, header=None, names=COLUMN_NAMES, nrows=5)
print(f"\nFirst 5 rows loaded. Shape: {df.shape}")
print(f"Label column sample: {df['label'].tolist()}")

# Run pipeline on first row
row = df.iloc[0].to_dict()
print(f"\nProcessing row 0: protocol={row['protocol_type']}, service={row['service']}, label={row['label']}")

# Preprocessing
row_df = pd.DataFrame([row], columns=COLUMN_NAMES)
row_df = row_df.drop(columns=DROP_COLS, errors="ignore")
row_df = pd.get_dummies(row_df, columns=CATEGORICAL_COLS)
row_df = row_df.reindex(columns=FEATURE_NAMES, fill_value=0)

print(f"After preprocessing shape: {row_df.shape}")
assert row_df.shape == (1, 121), f"FAIL: expected (1, 121) got {row_df.shape}"

# Scale
X = scaler.transform(row_df.values)
print(f"After scaling shape: {X.shape}")

# Predict
pred = model.predict(X)[0]
score = -model.score_samples(X)[0]

print(f"\nPrediction: {pred} ({'normal' if pred == 1 else 'ANOMALY'})")
print(f"Anomaly score (higher=worse): {score:.6f}")
print("\n✅ Pipeline smoke-test PASSED!")
