import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
from datetime import datetime
from data_processing import limpiar_datos, preprocesar_para_modelo
from training import entrenar_modelos, evaluar_modelo_streamlit
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from fastapi import FastAPI

app = FastAPI()

st.set_page_config(page_title="Predicción de Disputas", layout="centered")
st.title("Predicción y Entrenamiento de Disputas de Clientes")

tab1, tab2, tab3 = st.tabs(["Entrenamiento", "Predicción por archivo", "Predicción manual"])

# ENTRENAMIENTO
with tab1:
    archivo = st.file_uploader("Sube tu archivo CSV para entrenamiento", type=["csv"], key="train")

    if archivo is not None:
        df = pd.read_csv(archivo)
        df = limpiar_datos(df)

        try:
            X, y, columnas_modelo, df_modelo = preprocesar_para_modelo(df)
        except Exception as e:
            st.error(f"Error en el preprocesamiento: {e}")
            st.stop()

        X = df_modelo.drop(columns=["Consumer disputed?", "Disputa"], errors="ignore")
        X = X.select_dtypes(include=["number"])
        X = X.reindex(columns=columnas_modelo, fill_value=0)

        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)


        st.success("Datos procesados correctamente")

        if st.button("Entrenar modelo"):
            X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
            modelos = entrenar_modelos(X_train, y_train, columnas_modelo)

            joblib.dump(modelos["Random Forest"], "../models/modelo_random_forest.pkl")
            joblib.dump(columnas_modelo, "../models/columnas_modelo.pkl")
            joblib.dump(scaler, "../models/scaler.pkl")

            st.success("Modelo entrenado y guardado")

            st.subheader("Evaluación del modelo Random Forest")
            evaluar_modelo_streamlit(modelos["Random Forest"], X_test, y_test, "Random Forest")


# PREDICCIÓN
with tab2:
    archivo_pred = st.file_uploader("Sube tu archivo CSV para predicción", type=["csv"], key="predict")

    if archivo_pred is not None:
        df = pd.read_csv(archivo_pred)
        df = limpiar_datos(df)

        modelo = joblib.load("../models/modelo_random_forest.pkl")
        columnas_entrenadas = joblib.load("../models/columnas_modelo.pkl")
        scaler = joblib.load("../models/scaler.pkl")

        X_pred = pd.get_dummies(df.drop(columns=["Consumer disputed?", "Disputa"], errors="ignore"), drop_first=True)
        X_pred = X_pred.reindex(columns=columnas_entrenadas, fill_value=0)

        if not set(columnas_entrenadas).issubset(X_pred.columns):
            st.error("El archivo no contiene todas las columnas necesarias para el modelo.")
            st.stop()

        X_scaled_pred = scaler.transform(X_pred)

        predicciones = modelo.predict(X_scaled_pred)
        probabilidades = modelo.predict_proba(X_scaled_pred)[:, 1]

        df_resultados = df.copy()
        df_resultados["Predicción"] = predicciones
        df_resultados["Probabilidad"] = probabilidades

        st.dataframe(df_resultados)

        nombre_archivo = f"predicciones_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        csv = df_resultados.to_csv(index=False).encode("utf-8")
        st.download_button("Descargar resultados", data=csv, file_name=nombre_archivo, mime="text/csv")

        conteo = df_resultados["Predicción"].value_counts()
        fig, ax = plt.subplots()
        ax.bar(["Disputa", "No Disputa"], conteo.values, color=["skyblue", "salmon"])
        ax.set_title("Distribución de Predicciones")
        st.pyplot(fig)

        st.metric("Total registros", len(df_resultados))
        st.metric("Disputas predichas", int(df_resultados["Predicción"].sum()))
        st.metric("Porcentaje de disputas", f"{df_resultados['Predicción'].mean()*100:.2f}%")

# PREDICCIÓN MANUAL
with tab3:
    st.subheader("Formulario de Predicción Manual")

    modelo = joblib.load("../models/modelo_random_forest.pkl")
    scaler = joblib.load("../models/scaler.pkl")
    columnas_modelo = joblib.load("../models/columnas_modelo.pkl")

    producto = st.selectbox("Producto", ["credit card", "mortgage", "loan", "unknown"])
    estado = st.selectbox("Estado", ["ca", "tx", "ny", "unknown"])
    subproducto = st.selectbox("Sub-product", ["online banking", "checking account", "unknown"])
    respuesta = st.selectbox("¿Respuesta a tiempo?", ["yes", "no", "unknown"])

    if st.button("Predecir disputa"):
        datos = {
            "Product": producto,
            "State": estado,
            "Sub-product": subproducto,
            "Timely response?": respuesta
        }

        df_input = pd.DataFrame([datos])
        df_input = pd.get_dummies(df_input, drop_first=True)
        X = df_input.reindex(columns=columnas_modelo, fill_value=0)
        X_scaled = scaler.transform(X)

        pred = modelo.predict(X_scaled)[0]
        proba = modelo.predict_proba(X_scaled)[0][1]

        st.metric("Probabilidad de disputa", f"{proba:.2%}")
        if pred == 1:
            st.success("Es probable que haya una disputa")
        else:
            st.info("No se espera disputa")