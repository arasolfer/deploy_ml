from datetime import datetime
import joblib 
import xgboost as xgb

def save_model(model:xgb.sklearn.XGBClassifier, model_path:str) -> None:
    """
    Guarda el modelo en un archivo usando joblib
    
    Args:
        model (xgb.sklearn.XGBClassifier): Modelo a guardar
        model_path (str): Ruta donde guardar el modelo.
    """
    current_date = datetime.now().strftime("%Y-%m-%d")
    full_path = f"{model_path}_{current_date}.joblib"
    joblib.dump(model, full_path)  
    print(f"Modelo guardado en {full_path}")