# data_processing.py
import pandas as pd

def cargar_datos(ruta_csv, index_col=0):
    df = pd.read_csv(ruta_csv, index_col=index_col)
    df = df.drop_duplicates()
    df['Date received'] = pd.to_datetime(df['Date received'], errors='coerce')
    df['Date sent to company'] = pd.to_datetime(df['Date sent to company'], errors='coerce')
    return df

def limpiar_datos(df):
    df = df[~df['Consumer disputed?'].isna()].copy()

    columnas_a_limpieza = ['Sub-product', 'Sub-issue', 'State', 'ZIP code', 'Timely response?']
    df = estandarizar_columnas_texto(df, columnas_a_limpieza)

    return df

def estandarizar_columnas_texto(df, columnas):
   
    for col in columnas:
        if col in df.columns:
            df[col] = df[col].fillna("unknown").astype(str).str.strip().str.lower()
    return df
def guardar_datos(df, ruta_salida):
    
    df.to_csv(ruta_salida, index=False, encoding='utf-8')

from sklearn.impute import SimpleImputer

def preprocesar_para_modelo(df, columnas_objetivo=["Consumer disputed?", "Disputa"]):
    
    df = limpiar_datos(df)

    if "Consumer disputed?" in df.columns:
        df["Disputa"] = df["Consumer disputed?"].map({"Yes": 1, "No": 0})

    df_modelo = pd.get_dummies(df, drop_first=True)

    if "Disputa" not in df_modelo.columns:
        raise ValueError("No se pudo generar la variable objetivo 'Disputa'.")

    X = df_modelo.drop(columns=columnas_objetivo, errors="ignore")
    y = df_modelo["Disputa"]
    columnas_modelo = X.columns.tolist()

    # Validar que X solo contenga datos numéricos
    X = X.select_dtypes(include=["number"])

    # Imputar valores faltantes
    imputer = SimpleImputer(strategy="mean")
    X_imputado = imputer.fit_transform(X)

    return X_imputado, y, columnas_modelo, df_modelo