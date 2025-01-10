import pandas as pd

def load_data(file_path: str) -> pd.DataFrame:
    """
    Carga los datos desde un archivo CSV.
    
    Returns:
        pd.DataFrame: Dataframe con los datos cargados.
    
    """
    return pd.read_csv(file_path)