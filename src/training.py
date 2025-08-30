# training.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import joblib
from data_processing import cargar_datos, limpiar_datos, preprocesar_para_modelo

 
# Entrenar modelos
def entrenar_modelos(X_train, y_train, columnas_modelo):
    modelos = {
        "Regresión Logística": LogisticRegression(max_iter=1000),
        "Árbol de Decisión": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(),
        "KNN": KNeighborsClassifier(),
        "SVM": SVC(probability=True)
    }

    for nombre, modelo in modelos.items():
        modelo.fit(X_train, y_train)
        modelos[nombre] = modelo

    joblib.dump(modelos["Random Forest"], "../models/modelo_random_forest.pkl")
    joblib.dump(columnas_modelo, "../models/columnas_modelo.pkl")
    return modelos


# Evaluar modelos
def evaluar_modelo_streamlit(modelo, X_test, y_test, nombre):
    import streamlit as st
    y_pred = modelo.predict(X_test)
    st.subheader(f"Evaluación de {nombre}")
    st.write("Accuracy:", accuracy_score(y_test, y_pred))
    st.write("Precision:", precision_score(y_test, y_pred, zero_division=0))
    st.write("Recall:", recall_score(y_test, y_pred, zero_division=0))
    st.write("F1 Score:", f1_score(y_test, y_pred, zero_division=0))


# Ajuste de umbral para Random Forest
def ajustar_umbral_rf(modelo_rf, X_test, y_test):
    y_proba = modelo_rf.predict_proba(X_test)[:, 1]
    umbrales = np.arange(0.3, 0.8, 0.05)
    for u in umbrales:
        y_pred = (y_proba >= u).astype(int)
        precision = precision_score(y_test, y_pred, zero_division=0)
        print(f"Umbral: {u:.2f} → Precisión: {precision:.3f}")

    umbral_final = 0.5
    y_pred_final = (y_proba >= umbral_final).astype(int)
    matriz = confusion_matrix(y_test, y_pred_final)
    print("\nMatriz de confusión con umbral ajustado:")
    print(matriz)
    print("Reporte de clasificación:")
    print(classification_report(y_test, y_pred_final, target_names=["No Disputa", "Disputa"]))
    disp = ConfusionMatrixDisplay(confusion_matrix=matriz, display_labels=["No Disputa", "Disputa"])
    disp.plot(cmap="Blues")
    plt.title("Matriz de Confusión - Random Forest")
    plt.tight_layout()
    plt.show()

#  Clustering con KMeans
def aplicar_clustering(X_scaled, df_modelo):
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(X_scaled)
    df_modelo['cluster'] = clusters
    score = silhouette_score(X_scaled, clusters)
    print(f"\nSilhouette Score del clustering: {score:.3f}")
    return df_modelo

def evaluar_modelo(modelo, X_test, y_test, nombre):
    
    y_pred = modelo.predict(X_test)
    print(f"\nEvaluación de {nombre}")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, zero_division=0))
    print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))
    print("-" * 40)

# Punto de entrada
if __name__ == "__main__":
    ruta = "../data/datos_procesados.csv"
    df = cargar_datos(ruta)
    df = limpiar_datos(df)
    X, y, columnas_modelo, df_modelo = preprocesar_para_modelo(df)

    # Asegurar que X tenga las mismas columnas que se usarán en predicción
    X = df_modelo.drop(columns=["Consumer disputed?", "Disputa"], errors="ignore")
    X = X.select_dtypes(include=["number"])
    X = X.reindex(columns=columnas_modelo, fill_value=0)

    # Escalado
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    joblib.dump(scaler, "../models/scaler.pkl")


    # División de datos
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Entrenamiento
    modelos = entrenar_modelos(X_train, y_train, columnas_modelo)

    # Evaluación
    for nombre, modelo in modelos.items():
        evaluar_modelo(modelo, X_test, y_test, nombre)

    # Ajuste de umbral
    ajustar_umbral_rf(modelos["Random Forest"], X_test, y_test)

    # Clustering
    df_modelo = aplicar_clustering(X_scaled, df_modelo)