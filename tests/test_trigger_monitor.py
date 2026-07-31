import time
import pytest
from unittest.mock import patch
from src.trigger_monitor import evaluate_triggers

@pytest.fixture
def mock_thresholds():
    with patch("src.trigger_monitor.load_known_attack_types", return_value={"normal", "neptune", "smurf"}), \
         patch("src.trigger_monitor.load_extreme_severity_threshold", return_value=0.75):
        yield

def test_evaluate_triggers_no_trigger(mock_thresholds):
    """Prueba que si no se cumple ninguna condición, no se dispara."""
    now = time.time()
    trigger, code, msg = evaluate_triggers(
        is_anomaly=False,
        anomaly_score=0.2,
        ground_truth_label="normal",
        last_retrain_time=now - 100,  # pasaron 100s, no 180
        alerts_since_last_retrain=5
    )
    assert trigger is False
    assert code == ""

def test_evaluate_triggers_time_interval(mock_thresholds):
    """Prueba que dispara por tiempo."""
    now = time.time()
    trigger, code, msg = evaluate_triggers(
        is_anomaly=False,
        anomaly_score=0.2,
        ground_truth_label="normal",
        last_retrain_time=now - 200,  # pasaron 200s (>=180)
        alerts_since_last_retrain=5
    )
    assert trigger is True
    assert code == "time_interval"
    assert "tiempo programado" in msg

def test_evaluate_triggers_alert_threshold(mock_thresholds):
    """Prueba que dispara por conteo de alertas."""
    now = time.time()
    trigger, code, msg = evaluate_triggers(
        is_anomaly=True,
        anomaly_score=0.2,
        ground_truth_label="neptune",
        last_retrain_time=now - 10,
        alerts_since_last_retrain=15  # >= 15
    )
    assert trigger is True
    assert code == "alert_threshold"
    assert "varias alertas" in msg

def test_evaluate_triggers_new_attack_type(mock_thresholds):
    """Prueba que dispara por ataque desconocido."""
    now = time.time()
    trigger, code, msg = evaluate_triggers(
        is_anomaly=True,
        anomaly_score=0.2,
        ground_truth_label="processtable",  # No está en el mock {"normal", "neptune", "smurf"}
        last_retrain_time=now - 10,
        alerts_since_last_retrain=5
    )
    assert trigger is True
    assert code == "new_attack_type"
    assert "processtable" in msg

def test_evaluate_triggers_extreme_severity(mock_thresholds):
    """Prueba que dispara por anomalía extrema."""
    now = time.time()
    trigger, code, msg = evaluate_triggers(
        is_anomaly=True,
        anomaly_score=0.85,  # > 0.75
        ground_truth_label="neptune",
        last_retrain_time=now - 10,
        alerts_since_last_retrain=5
    )
    assert trigger is True
    assert code == "extreme_severity"
    assert "score=0.85" in msg
