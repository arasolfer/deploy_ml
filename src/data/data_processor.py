from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
import pandas as pd

def process_data(df: pd.DataFrame, target_column:str = None) -> tuple[pd.DataFrame, pd.Series, StandardScaler]:
    """
    Procesa los datos:
    - Imputa valores faltantes y reemplaza valores extremos.
    - Escala las variables numéricas.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos a procesar.
        target_column(string): columna target del proyecto
    
    Returns:
        pd.DataFrame: DataFrame con los datos procesados.
    """
    # imputacion y limpieza de valores
    # reemplazar valores en las columnas relevantes
    df['BloodPressure'] = df['BloodPressure'].replace(
        0, df[df['BloodPressure'] > 0]['BloodPressure'].mean()
    )
    df['Glucose'] = df['Glucose'].replace(
        0, df[df['Glucose'] > 0]['Glucose'].mean()
    )
    df['SkinThickness'] = df['SkinThickness'].replace(
        0, df[df['SkinThickness'] > 0]['SkinThickness'].mean()
    )
    # limpiar valores de skin thickness
    mean_skin_thickness = df[df['SkinThickness'] <= 90]['SkinThickness'].mean()
    df.loc[df['SkinThickness'] > 90, 'SkinThickness'] = mean_skin_thickness
    mean_skin_thickness = df[df['SkinThickness'] >= 10]['SkinThickness'].mean()
    df.loc[df['SkinThickness'] < 10, 'SkinThickness'] = mean_skin_thickness
    
    df['Insulin'] = df['Insulin'].replace(
        0, df[df['Insulin'] > 0]['Insulin'].median()
    )
    df['BMI'] = df['BMI'].replace(
        0, df[df['BMI'] > 0]['BMI'].mean()
    )
    # limpiar valores de BMI
    mean_bmi2 = df[df['BMI'] <= 60]['BMI'].mean()
    df.loc[df['BMI'] > 60, 'BMI'] = mean_bmi2
    
    # definir el target
    target = df[target_column] if target_column else None
    
    # escalamiento de las variables numericas
    scaler = StandardScaler()
    numeric_cols = ['BloodPressure', 'Glucose', 'SkinThickness', 'Insulin', 'BMI']
    df[numeric_cols] = scaler.fit_transform(df[numeric_cols])
    
    return df, target, scaler