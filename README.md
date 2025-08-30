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
ustomer_complaints_ml/
├── 📁 notebooks/                    # Jupyter Notebooks del proceso completo
│   ├── 01_Fuentes.ipynb            # Adquisición y exploración de datos
│   ├── 02_LimpiezaEDA.ipynb         # Limpieza y análisis exploratorio
│   ├── 03_Entrenamiento_Evaluacion.ipynb  # Modelado y evaluación
│   └── 04_Evaluacion.ipynb          # Validación final y métricas
│
├── 📁 src/                          # Código fuente modularizado
│   ├── preprocessing.py             # Pipeline de preprocesamiento
│   ├── training.py                  # Entrenamiento de modelos
│   ├── evaluation.py                # Evaluación e interpretabilidad
│   ├── model_storage.py             # Gestión de modelos entrenados
│   └── utils.py                     # Utilidades generales
│
├── 📁 models/                       # Modelos entrenados y configuraciones
│   ├── trained_model_1.pkl          # Modelos experimentales
│   ├── trained_model_2.pkl          
│   ├── final_model.pkl              # Modelo final para producción
│   ├── model_config.yaml            # Configuración del modelo
│   └── model_metadata.json          # Metadatos y métricas
│
├── 📁 app/                          # Aplicación web interactiva
│   ├── app.py                       # Aplicación principal Streamlit
│   ├── requirements.txt             # Dependencias de la app
│   └── pages/                       # Páginas adicionales
│
├── 📁 data/                         # Datos del proyecto
│   ├── raw/                         # Datos originales
│   ├── processed/                   # Datos procesados
│   └── external/                    # Datos externos
│
├── 📁 docs/                         # Documentación del proyecto
│   ├── negocio.ppt                  # Presentación ejecutiva
│   ├── ds.ppt                       # Presentación técnica
│   ├
├── requirements.txt                 # Dependencias del proyecto
└── README.md                        # Este archivo

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

