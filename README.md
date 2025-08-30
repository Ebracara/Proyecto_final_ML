# Predicción de Disputas en Quejas de Clientes

Este proyecto utiliza técnicas de Machine Learning para anticipar si una queja presentada por un cliente será disputada. El objetivo es mejorar la eficiencia operativa, reducir costes y optimizar la atención al cliente.

---

## Objetivos del Proyecto

- Predecir la probabilidad de disputa en una queja.
- Identificar variables clave que influyen en el conflicto.
- Proporcionar visualizaciones para análisis estratégico.
- Integrar el modelo en procesos de toma de decisiones.

---
## Funcionalidades
- Entrenamiento de modelos (Random Forest, SVM, etc.)
- Evaluación con métricas estándar
- Ajuste de umbral para Random Forest
- Clustering con KMeans
- Interfaz interactiva con Streamlit

## Estructura del Proyecto
```plaintext
PROYECTO_FINAL_ML/
|── 📁 data/                         # Datos del proyecto
│   ├── clientes_quejas.csv           # Datos originales
│   ├── datos_procesados.csv          # Datos procesados
│                      
├── 📁 notebooks/                         # Jupyter Notebooks del proceso completo
│   ├── 01_Fuentes.ipynb                   # Adquisición y exploración de datos
│   ├── 02_LimpiezaEDA.ipynb               # Limpieza y análisis exploratorio
│   ├── 03_Entrenamiento_Evaluacion.ipynb  # Modelado y evaluación
│           │
├── 📁 src/                         # Código fuente modularizado
│   ├── data_processing.py           # Pipeline ETL
│   ├── training.py                  # Entrenamiento de modelos
│   ├── app.py                       # API de inferencia 
│   │
├── 📁 models/                      # Modelos entrenados y configuraciones
│   ├── modelo_random_forest.pkl     # modelo final
│   ├── escaler.pkl                  # Escalador features
│   ├── columnas_modelo.pkl          # Feature names
│
├── 📁 docs/                         # Documentación del proyecto
│   ├── negocio.pptx                  # Presentación ejecutiva
│   ├── Análisis_proyecto.pptx        # Presentación técnica
│  
├── requirements.txt                 # Dependencias del proyecto
└── README.md                        # Este archivo
```
## Tecnologías utilizadas
- Python 3.11
- pandas, scikit-learn, seaborn, matplotlib
- Jupyter Notebook
- GridSearchCV para optimización
- KMeans y PCA para análisis exploratorio


## Ejecución
1. Ejecuta `app.py` con Streamlit:
   ```bash
   streamlit run app.py


## Autora

**Esther Begoña**  
Data Scientist | Santander, España  
🔗 [GitHub](https://github.com)

## Licencia

Este proyecto está bajo la licencia MIT. Puedes usarlo, modificarlo y compartirlo libremente.





