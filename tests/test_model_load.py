import pytest
import numpy as np
from src.model_io import load_model_bare, load_scaler_bare
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

def test_model_load_and_predict():
    """Valida que model.pkl y scaler.pkl cargan y predicen correctamente."""
    model = load_model_bare()
    scaler = load_scaler_bare()
    
    assert isinstance(model, IsolationForest), "El modelo debe ser IsolationForest"
    assert isinstance(scaler, StandardScaler), "El scaler debe ser StandardScaler"
    
    # Probar predicción con un mock array del shape correcto (1, 121)
    mock_X = np.random.rand(1, 121)
    
    try:
        pred = model.predict(mock_X)
        score = model.score_samples(mock_X)
    except ValueError as e:
        pytest.fail(f"La predicción falló con el shape esperado: {e}")
        
    assert pred.shape == (1,)
    assert pred[0] in [1, -1]
    assert score.shape == (1,)
