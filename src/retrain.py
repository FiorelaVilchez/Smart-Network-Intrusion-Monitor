"""
src/retrain.py
--------------
Automated retraining pipeline for the Isolation Forest model.
Simulates reading "new production data" from traffic_log.db, combines it with
a sample of the original training data (KDDTrain+.txt), retrains the model,
and evaluates it.

If the new model achieves an F1-score >= (current_f1 - 0.01), it is promoted.
Outputs retrain_status.json for CI/CD workflows.
"""

import os
import json
import datetime
import urllib.request
import sys
import argparse
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score

# Ensure src module is in path for direct execution
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Paths and config
# ---------------------------------------------------------------------------
_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DATA_DIR = os.path.join(_BASE_DIR, "data")
_MODELS_DIR = os.path.join(_BASE_DIR, "models")

TRAIN_DATA_PATH = os.path.join(_DATA_DIR, "KDDTrain+.txt")
TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"

SCHEMA_PATH = os.path.join(_BASE_DIR, "COLUMN_SCHEMA.json")
FEATURE_NAMES_PATH = os.path.join(_MODELS_DIR, "feature_names.json")
PARAMS_PATH = os.path.join(_MODELS_DIR, "best_params.json")
METRICS_PATH = os.path.join(_MODELS_DIR, "metrics.json")
REGISTRY_PATH = os.path.join(_MODELS_DIR, "model_registry.json")
STATUS_PATH = os.path.join(_BASE_DIR, "retrain_status.json")

# Import storage here to get db path
from src.storage import DB_PATH, get_all_for_retraining

with open(SCHEMA_PATH, "r") as f:
    _SCHEMA = json.load(f)

COLUMN_NAMES = _SCHEMA["columns"]
CATEGORICAL_COLS = _SCHEMA["categorical_columns"]
DROP_COLS = _SCHEMA["drop_columns"]

with open(FEATURE_NAMES_PATH, "r") as f:
    FEATURE_NAMES = json.load(f)

def download_data_if_needed():
    if not os.path.exists(TRAIN_DATA_PATH):
        print(f"Downloading training data from {TRAIN_URL}...")
        os.makedirs(_DATA_DIR, exist_ok=True)
        urllib.request.urlretrieve(TRAIN_URL, TRAIN_DATA_PATH)
        print("Download complete.")

def prepare_features(df: pd.DataFrame, fit_scaler: bool = False, scaler: StandardScaler = None):
    """Aligns a dataframe to the exact model feature space and scales it."""
    # Drop non-feature columns
    X = df.drop(columns=DROP_COLS + ["is_anomaly", "ground_truth", "timestamp", "prediction_raw", "anomaly_score", "id"], errors="ignore")
    
    # Fill missing columns that were expected by raw schema with 0 or empty string
    for col in COLUMN_NAMES:
        if col not in X.columns and col not in DROP_COLS:
            if col in CATEGORICAL_COLS:
                X[col] = "unknown"
            else:
                X[col] = 0.0

    # One-hot encoding
    X = pd.get_dummies(X, columns=CATEGORICAL_COLS)
    
    # Align to exactly 121 features
    X = X.reindex(columns=FEATURE_NAMES, fill_value=0)
    
    if fit_scaler:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        return X_scaled, scaler
    else:
        X_scaled = scaler.transform(X)
        return X_scaled

def run_retraining(trigger_reason: str, force_promote: bool = False, sample_size: int = 10000) -> dict:
    print(f"Starting retraining pipeline... (Trigger: {trigger_reason})")
    download_data_if_needed()

    # 1. Load original training data (sample)
    df_train = pd.read_csv(TRAIN_DATA_PATH, header=None, names=COLUMN_NAMES)
    df_train_sample = df_train.sample(n=min(sample_size, len(df_train)), random_state=42).copy()
    
    # The IsolationForest considers -1 as anomaly. We define "normal" as 1, "anomaly" as -1 for training.
    # Actually, IsolationForest doesn't need labels for training, but we need them for evaluation.
    df_train_sample["is_anomaly"] = df_train_sample["label"].apply(lambda x: 0 if x == "normal" else 1)

    # 2. Load "new production data" from SQLite
    try:
        df_prod = get_all_for_retraining()
        print(f"Loaded {len(df_prod)} records from production database.")
    except Exception as e:
        print(f"Warning: Could not load production data ({e}). Using only training sample.")
        df_prod = pd.DataFrame()

    # 3. Combine for retraining
    if not df_prod.empty:
        # Align production data to have same is_anomaly meaning (1 for anomaly, 0 for normal)
        # df_prod has "is_anomaly" as 1 or 0 already
        df_combined = pd.concat([df_train_sample, df_prod], ignore_index=True)
    else:
        df_combined = df_train_sample

    print(f"Total training samples: {len(df_combined)}")

    # Prepare features and fit scaler
    X_train, new_scaler = prepare_features(df_combined, fit_scaler=True)

    # 4. Train new model
    with open(PARAMS_PATH, "r") as f:
        params = json.load(f)
    
    print(f"Training IsolationForest with params: {params}")
    model = IsolationForest(
        n_estimators=params.get("n_estimators", 100),
        max_samples=params.get("max_samples", 0.5),
        contamination=params.get("contamination", 0.05),
        max_features=params.get("max_features", 0.7),
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train)

    # 5. Evaluate against validation set (we use the production data as validation if available, else a slice of train)
    print("Evaluating model...")
    if not df_prod.empty:
        df_val = df_prod.copy()
    else:
        # fallback to a different sample from train data
        df_val = df_train.drop(df_train_sample.index).sample(n=min(5000, len(df_train) - len(df_train_sample)), random_state=42)
        df_val["is_anomaly"] = df_val["label"].apply(lambda x: 0 if x == "normal" else 1)

    X_val = prepare_features(df_val, fit_scaler=False, scaler=new_scaler)
    
    preds_raw = model.predict(X_val)
    # Convert IF output (-1 anomaly, 1 normal) to (1 anomaly, 0 normal)
    preds = [1 if p == -1 else 0 for p in preds_raw]
    scores = -model.score_samples(X_val)

    y_true = df_val["is_anomaly"].values

    new_metrics = {
        "accuracy": float(accuracy_score(y_true, preds)),
        "precision": float(precision_score(y_true, preds, zero_division=0)),
        "recall": float(recall_score(y_true, preds, zero_division=0)),
        "f1_score": float(f1_score(y_true, preds, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.0,
        "average_precision": float(average_precision_score(y_true, scores)) if len(np.unique(y_true)) > 1 else 0.0,
    }

    print(f"New Model Metrics: {json.dumps(new_metrics, indent=2)}")

    # 6. Compare with current model
    with open(METRICS_PATH, "r") as f:
        current_metrics = json.load(f)
    
    current_f1 = current_metrics.get("f1_score", 0.0)
    new_f1 = new_metrics["f1_score"]
    
    print(f"Current F1: {current_f1:.4f} | New F1: {new_f1:.4f}")
    
    tolerance = 0.01
    promoted = False
    reason = ""

    if new_f1 >= (current_f1 - tolerance) or force_promote:
        promoted = True
        reason = f"New F1 ({new_f1:.4f}) is within tolerance of current F1 ({current_f1:.4f})"
        if force_promote:
            reason = "Force promoted via CLI"
    else:
        reason = f"New F1 ({new_f1:.4f}) is worse than current F1 ({current_f1:.4f}) by more than {tolerance}"

    print(f"Decision: {'PROMOTED' if promoted else 'REJECTED'} - {reason}")

    # 7. Update registry
    with open(REGISTRY_PATH, "r") as f:
        registry = json.load(f)
    
    next_version = max([r.get("version", 0) for r in registry]) + 1 if registry else 1
    
    new_entry = {
        "version": next_version,
        "timestamp": datetime.datetime.utcnow().isoformat(),
        "hyperparameters": params,
        "metrics": new_metrics,
        "training_samples": len(df_combined),
        "status": "production" if promoted else "rejected",
        "reason": reason,
        "trigger_reason": trigger_reason
    }
    registry.append(new_entry)
    
    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    # 8. Save artifacts if promoted
    if promoted:
        import joblib
        with open(os.path.join(_MODELS_DIR, "model.pkl"), "wb") as f:
            joblib.dump(model, f)
        with open(os.path.join(_MODELS_DIR, "scaler.pkl"), "wb") as f:
            joblib.dump(new_scaler, f)
        with open(METRICS_PATH, "w") as f:
            json.dump(new_metrics, f, indent=2)
        print("Model artifacts and metrics updated.")

    # 9. Write status file
    status = {
        "promoted": promoted,
        "reason": reason,
        "new_f1": new_f1,
        "current_f1": current_f1,
        "trigger_reason": trigger_reason
    }
    with open(STATUS_PATH, "w") as f:
        json.dump(status, f, indent=2)
    print(f"Wrote status to {STATUS_PATH}")
    return status

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force-promote", action="store_true", help="Promote even if metrics are worse")
    parser.add_argument("--sample-size", type=int, default=10000, help="Number of rows to sample from original train data")
    args = parser.parse_args()
    
    run_retraining(trigger_reason="scheduled_cron", force_promote=args.force_promote, sample_size=args.sample_size)

if __name__ == "__main__":
    main()
