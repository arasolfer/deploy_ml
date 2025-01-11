import sys
import os
import joblib
# agregar raiz del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from src.data.data_loader import load_data
from src.data.data_processor import process_data
from src.data.data_splitter import split_data
from src.model.trainer import train_model
from src.model.evaluator import evaluate_model
from src.model.saver import save_model

def main():
    # cargar datos
    data = load_data(file_path = "data/raw/diabetes.csv")
    
    # procesar los datos
    processed_data, target, scaler = process_data(df=data, 
                                        target_column='Outcome')
    
    # Guardar el escalador
    joblib.dump(scaler, "models/scaler.pkl")
    print("Escalador guardado exitosamente en models/scaler.pkl")
    
    # split de datos en test y train
    X_train, X_test, y_train, y_test = split_data(processed_data, target_column='Outcome')
    
    # entrenar el modelo
    model = train_model(X_train=X_train, y_train=y_train)
    # print(type(model))
    
    # evaluar el modelo
    accuracy, precision, recall, f1, auc = evaluate_model(model, test_data=X_test, y_test=y_test)
    # print(f"Accuracy: {accuracy}")
    # print(f"Precision: {precision}")
    # print(f"Recall: {recall}")
    # print(f"F1: {f1}")
    # print(f"AUC: {auc}")
    
    # guardar el modelo
    save_model(model, model_path="models/trained_model")

if __name__ == "__main__":
    main()