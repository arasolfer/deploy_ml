import xgboost as xgb
import pandas as pd

def train_model(X_train:pd.DataFrame, y_train:pd.Series, params:dict=None) -> xgb.sklearn.XGBClassifier:
    """
    Entrena un modelo XGBoost con los datos proporcionados.

    Args:
        X_train (pd.DataFrame): Conjunto de entrenamiento (características).
        y_train (pd.Series): Etiquetas del conjunto de entrenamiento.
        params (dict, opcional): Parámetros del modelo XGBoost. 
        Si no se proporciona, se utilizarán valores por defecto.

    Returns:
        xgb.XGBClassifier: Modelo entrenado.
    """
    
    if params is None: 
        params = {'objective':'binary:logistic',
                'eval_metric':'logloss',
                'n_estimators':300, 'max_depth':2,
                'learning_rate':0.1,'random_state':42
                }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train, y_train)
    
    return model