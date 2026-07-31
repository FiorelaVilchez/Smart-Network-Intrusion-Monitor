import pytest
import json
import os
import subprocess
import shutil

@pytest.fixture
def temp_metrics_file(tmp_path):
    """Crea un archivo metrics.json temporal para las pruebas."""
    original_metrics_path = "models/metrics.json"
    temp_metrics_path = tmp_path / "metrics.json"
    
    # Leer el actual y guardarlo
    with open(original_metrics_path, "r") as f:
        original_data = json.load(f)
        
    yield original_metrics_path, temp_metrics_path, original_data
    
    # Restaurar original (el test podría haberlo modificado indirectamente si no mockeamos bien,
    # aunque src/retrain.py está hardcodeado para escribir en models/metrics.json.
    # Para ser seguros, escribimos los originales de vuelta)
    with open(original_metrics_path, "w") as f:
        json.dump(original_data, f, indent=2)

def test_retrain_pipeline_logic(monkeypatch, tmp_path):
    """
    Valida, usando la salida del script retrain.py, la lógica de promoción.
    Como el script hace descargas y entrenamientos que pueden tardar,
    podemos simular la llamada o simplemente invocar el script con una muestra pequeña.
    Aquí optamos por inyectar args (sample_size pequeño) para que sea rápido.
    """
    # Hacer backup de artifacts que podrían ser sobrescritos
    import joblib
    model_bkp = "models/model.pkl.bkp"
    scaler_bkp = "models/scaler.pkl.bkp"
    reg_bkp = "models/model_registry.json.bkp"
    
    shutil.copy("models/model.pkl", model_bkp)
    shutil.copy("models/scaler.pkl", scaler_bkp)
    shutil.copy("models/model_registry.json", reg_bkp)
    
    try:
        # Correr retrain.py con un tamaño de muestra muy pequeño (ej. 10) para que sea rápido y probablemente obtenga métricas pobres
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        result = subprocess.run(["python", "src/retrain.py", "--sample-size", "50"], capture_output=True, text=True, env=env)
        assert result.returncode == 0, f"retrain.py falló:\n{result.stderr}"
        
        # Leer retrain_status.json
        assert os.path.exists("retrain_status.json")
        with open("retrain_status.json", "r") as f:
            status = json.load(f)
            
        assert "promoted" in status
        assert "new_f1" in status
        assert "current_f1" in status
        
        # Validar la lógica de promoción del script
        tolerance = 0.01
        if status["new_f1"] >= (status["current_f1"] - tolerance):
            assert status["promoted"] == True
        else:
            assert status["promoted"] == False
            
    finally:
        # Restaurar artifacts
        shutil.copy(model_bkp, "models/model.pkl")
        shutil.copy(scaler_bkp, "models/scaler.pkl")
        shutil.copy(reg_bkp, "models/model_registry.json")
        
        os.remove(model_bkp)
        os.remove(scaler_bkp)
        os.remove(reg_bkp)
        if os.path.exists("retrain_status.json"):
            os.remove("retrain_status.json")

def test_force_promote(monkeypatch, tmp_path):
    """Valida la bandera --force-promote del script retrain.py."""
    # Hacer backup
    model_bkp = "models/model.pkl.bkp"
    scaler_bkp = "models/scaler.pkl.bkp"
    reg_bkp = "models/model_registry.json.bkp"
    metrics_bkp = "models/metrics.json.bkp"
    
    shutil.copy("models/model.pkl", model_bkp)
    shutil.copy("models/scaler.pkl", scaler_bkp)
    shutil.copy("models/model_registry.json", reg_bkp)
    shutil.copy("models/metrics.json", metrics_bkp)
    
    try:
        env = os.environ.copy()
        env["PYTHONPATH"] = "."
        result = subprocess.run(["python", "src/retrain.py", "--sample-size", "50", "--force-promote"], capture_output=True, text=True, env=env)
        assert result.returncode == 0, f"retrain.py falló con force-promote:\n{result.stderr}"
        
        with open("retrain_status.json", "r") as f:
            status = json.load(f)
            
        assert status["promoted"] == True
        assert "Force promoted via CLI" in status["reason"]
        
    finally:
        # Restaurar artifacts
        shutil.copy(model_bkp, "models/model.pkl")
        shutil.copy(scaler_bkp, "models/scaler.pkl")
        shutil.copy(reg_bkp, "models/model_registry.json")
        shutil.copy(metrics_bkp, "models/metrics.json")
        
        os.remove(model_bkp)
        os.remove(scaler_bkp)
        os.remove(reg_bkp)
        os.remove(metrics_bkp)
        if os.path.exists("retrain_status.json"):
            os.remove("retrain_status.json")
