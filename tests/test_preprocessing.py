import pytest
import numpy as np
from src.preprocessing import preprocess_record, predict_record

def test_preprocess_record_shape():
    """Valida que preprocess_record devuelve un array con exactamente 121 columnas."""
    # Ejemplo de registro crudo
    raw_row = {
        "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "SF",
        "src_bytes": 181, "dst_bytes": 5450, "land": 0, "wrong_fragment": 0,
        "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 1,
        "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 8, "srv_count": 8, "serror_rate": 0.0, "srv_serror_rate": 0.0,
        "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0, "dst_host_count": 9,
        "dst_host_srv_count": 9, "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 0.11,
        "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0,
        "dst_host_srv_rerror_rate": 0.0, "label": "normal", "difficulty_level": 21
    }
    
    # Preprocesar
    X = preprocess_record(raw_row)
    
    # Verificar tipo y forma
    assert isinstance(X, np.ndarray), "El resultado debe ser un np.ndarray"
    assert X.shape == (1, 121), f"Se esperaba shape (1, 121), pero se obtuvo {X.shape}"

def test_predict_record():
    """Valida que predict_record devuelve el formato esperado."""
    raw_row = {
        "duration": 0, "protocol_type": "tcp", "service": "http", "flag": "SF",
        "src_bytes": 181, "dst_bytes": 5450, "land": 0, "wrong_fragment": 0,
        "urgent": 0, "hot": 0, "num_failed_logins": 0, "logged_in": 1,
        "num_compromised": 0, "root_shell": 0, "su_attempted": 0, "num_root": 0,
        "num_file_creations": 0, "num_shells": 0, "num_access_files": 0,
        "num_outbound_cmds": 0, "is_host_login": 0, "is_guest_login": 0,
        "count": 8, "srv_count": 8, "serror_rate": 0.0, "srv_serror_rate": 0.0,
        "rerror_rate": 0.0, "srv_rerror_rate": 0.0, "same_srv_rate": 1.0,
        "diff_srv_rate": 0.0, "srv_diff_host_rate": 0.0, "dst_host_count": 9,
        "dst_host_srv_count": 9, "dst_host_same_srv_rate": 1.0,
        "dst_host_diff_srv_rate": 0.0, "dst_host_same_src_port_rate": 0.11,
        "dst_host_srv_diff_host_rate": 0.0, "dst_host_serror_rate": 0.0,
        "dst_host_srv_serror_rate": 0.0, "dst_host_rerror_rate": 0.0,
        "dst_host_srv_rerror_rate": 0.0, "label": "normal", "difficulty_level": 21
    }
    
    pred = predict_record(raw_row)
    
    assert "is_anomaly" in pred
    assert "anomaly_score" in pred
    assert "prediction_raw" in pred
    
    assert isinstance(pred["is_anomaly"], bool)
    assert isinstance(pred["anomaly_score"], float)
    assert pred["prediction_raw"] in [1, -1]
