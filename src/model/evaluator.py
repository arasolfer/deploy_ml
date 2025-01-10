from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
import xgboost 
import pandas as pd

def evaluate_model(model: xgboost.sklearn.XGBClassifier, test_data:pd.DataFrame, y_test:pd.Series
                ) -> tuple[float, float, float, float, float]:
    """
    Evalúa el modelo utilizando varias métricas.

    Args:
        model (xgb.XGBClassifier): Modelo XGBoost entrenado.
        test_data (pd.DataFrame): Conjunto de características de prueba.
        y_test (pd.Series): Etiquetas reales del conjunto de prueba.

    Returns:
        Tuple[float, float, float, float, float]: 
            - Precisión (accuracy)
            - Precisión (precision)
            - Recall
            - Puntaje F1
            - Área bajo la curva ROC (AUC)
    """
    y_pred = model.predict(test_data)
    y_prob = model.predict_proba(test_data)[:,1]
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    
    return accuracy, precision, recall, f1, auc