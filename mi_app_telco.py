# Contenido para mi_app_telco.py

import streamlit as st
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
import time
import pickle # Para cargar modelos .pkl y el pipeline
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
# Importar skorch y tf solo si los archivos de modelo existen
# para evitar errores si solo se quiere ver el EDA
try:
    from skorch import NeuralNetClassifier
except ImportError:
    st.warning("skorch no está instalado. La pestaña de Predicción (PyTorch) no funcionará.")

try:
    import tensorflow as tf
except ImportError:
    st.warning("tensorflow no está instalado. Las pestañas de Keras no funcionarán.")


# --- Definición de la Arquitectura PyTorch ---
# Debe estar en el scope global para que pickle/skorch pueda cargar el modelo

# Primero, intentamos obtener INPUT_DIM del pipeline guardado
INPUT_DIM = 26 # Valor por defecto si falla la carga
try:
    with open('telco_preprocessing_pipeline.pkl', 'rb') as f:
        pipeline_data_carga = pickle.load(f)
        INPUT_DIM = len(pipeline_data_carga['processed_columns'])
except FileNotFoundError:
    pass # Usará el valor por defecto

class MLP_PyTorch_Binary(nn.Module):
    def __init__(self, input_dim=INPUT_DIM, num_units1=128, num_units2=32, dropout_p=0.2):
        super(MLP_PyTorch_Binary, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, num_units1),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(num_units1, num_units2),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(num_units2, 1) # Salida binaria (logits)
        )

    def forward(self, x):
        return self.layers(x)

# --- Carga de Artefactos (Modelos y Pipeline) ---

output_dir = 'analysis_outputs_telco'
DATASET_FILE = 'WA_Fn-UseC_-Telco-Customer-Churn.csv'

@st.cache_resource # Almacena en caché los recursos cargados
def load_artifacts():
    """Carga el modelo PyTorch, Keras, el pipeline y el CSV original."""
    
    pipeline = None
    model_pytorch = None
    model_keras = None
    df_original = None

    # Cargar Pipeline de Preprocesamiento (de Fase 1)
    try:
        with open('telco_preprocessing_pipeline.pkl', 'rb') as f:
            pipeline = pickle.load(f)
    except FileNotFoundError:
        st.error("Error Crítico: No se encontró 'telco_preprocessing_pipeline.pkl'.")
        st.error("Asegúrate de haber ejecutado la Fase 1 ('01_eda_telco.py') correctamente.")
    except Exception as e:
        st.error(f"Error al cargar 'telco_preprocessing_pipeline.pkl': {e}")

    # Cargar Modelo PyTorch (de Fase 3)
    try:
        with open(os.path.join(output_dir, 'best_pytorch_model.pkl'), 'rb') as f:
            model_pytorch = pickle.load(f)
    except FileNotFoundError:
        st.warning("Advertencia: No se encontró 'best_pytorch_model.pkl'. La pestaña de predicción no funcionará.")
    except Exception as e:
        st.error(f"Error al cargar 'best_pytorch_model.pkl': {e}")

    # Cargar Modelo Keras (de Fase 2)
    try:
        model_keras = tf.keras.models.load_model(os.path.join(output_dir, 'best_keras_model.keras'))
    except (FileNotFoundError, IOError):
         st.warning("Advertencia: No se encontró 'best_keras_model.keras'. La pestaña Keras no mostrará la evaluación.")
    except Exception as e:
        st.error(f"Error al cargar 'best_keras_model.keras': {e}")
        
    # Cargar CSV original (para EDA y opciones de predicción)
    try:
        df_original = pd.read_csv(DATASET_FILE)
    except FileNotFoundError:
        st.error(f"Error Crítico: No se encontró el CSV original '{DATASET_FILE}'.")
    except Exception as e:
        st.error(f"Error al cargar '{DATASET_FILE}': {e}")

    return model_pytorch, model_keras, pipeline, df_original

# Cargar los artefactos
best_pytorch_model, best_keras_model, preprocessing_pipeline, df_original_eda = load_artifacts()

# --- UI de Streamlit ---

st.title("PARCIAL Ciencia de Datos 1 - Telco Customer Churn") # Título principal
st.markdown("""
**Integrantes:**
* Bermudez Huayhua, Yudenio
* Carrasco Castañeda, Edwar Frank 
* Micha Velasques, Margaret Pilar
* Sangay Terrones, Jhonatan Smith
""") # Nombres de los integrantes

st.header("Contexto") # Encabezado de Contexto
st.markdown(""" 
El objetivo es construir, ajustar y desplegar una red neuronal profunda (DNN) para predecir si un cliente abandonará (Churn) la compañía telefónica. 
Se implementan dos modelos (Keras y PyTorch) y se comparan.
""") # Descripción del proyecto

# --- Pestañas ---
tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "1. Descripción del Dataset",
    "2. Análisis Exploratorio (EDA)",
    "3. Transformación de Variables",
    "4. Modelado con Keras",
    "5. Modelado con PyTorch",
    "6. Evaluación y Selección",
    "7. Predicción (App)"
])

# --- Pestaña 1: Descripción del Dataset ---
with tab1:
    st.header("1. Descripción del Dataset (Telco Churn)")
    st.markdown("Fuente: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)")
    
    st.subheader("Variables (Features)")
    st.markdown("""
    El dataset incluye:
    * **Datos Demográficos:** `gender`, `SeniorCitizen`, `Partner`, `Dependents`.
    * **Datos de Cuenta:** `tenure` (antigüedad), `Contract`, `PaperlessBilling`, `PaymentMethod`, `MonthlyCharges`, `TotalCharges`.
    * **Servicios:** `PhoneService`, `MultipleLines`, `InternetService`, `OnlineSecurity`, `OnlineBackup`, `DeviceProtection`, `TechSupport`, `StreamingTV`, `StreamingMovies`.
    * **Variable Objetivo:** `Churn` (Yes/No).
    """)
    
    if df_original_eda is not None:
        st.subheader("Dataset Original (Primeras 5 filas)")
        st.dataframe(df_original_eda.head())
    
        st.subheader("Información de las columnas (Original)")
        import io
        buffer = io.StringIO()
        df_original_eda.info(buf=buffer)
        s = buffer.getvalue()
        st.text(s)
    else:
        st.warning("El archivo CSV no se cargó. No se puede mostrar la descripción.")

# --- Pestaña 2: Análisis Exploratorio (EDA) ---
with tab2:
    st.header("2. Análisis Exploratorio de Datos (EDA)")
    
    if df_original_eda is not None:
        st.subheader("Distribución de la Variable Objetivo (Churn)")
        fig_churn, ax_churn = plt.subplots()
        churn_counts = df_original_eda['Churn'].value_counts()
        sns.barplot(x=churn_counts.index, y=churn_counts.values, ax=ax_churn, palette='pastel')
        ax_churn.set_title('Distribución de Churn (Abandono)')
        ax_churn.set_ylabel('Frecuencia')
        st.pyplot(fig_churn)
        st.markdown(f"**Desbalance:** {churn_counts.get('No', 0)} (No) vs. {churn_counts.get('Yes', 0)} (Sí).")

        st.subheader("Churn por Tipo de Contrato")
        fig_contract, ax_contract = plt.subplots()
        sns.countplot(data=df_original_eda, x='Contract', hue='Churn', ax=ax_contract, palette='Set2')
        ax_contract.set_title('Churn vs. Tipo de Contrato')
        st.pyplot(fig_contract)
        st.markdown("Los clientes 'Month-to-month' tienen una tasa de Churn significativamente mayor.")

        st.subheader("Churn por Antigüedad (Tenure)")
        fig_tenure, ax_tenure = plt.subplots()
        df_eda_cleaned = df_original_eda.copy()
        df_eda_cleaned['TotalCharges'] = pd.to_numeric(df_eda_cleaned['TotalCharges'], errors='coerce').fillna(0)
        sns.histplot(data=df_eda_cleaned, x='tenure', hue='Churn', multiple='stack', kde=True, ax=ax_tenure)
        ax_tenure.set_title('Distribución de Antigüedad vs. Churn')
        st.pyplot(fig_tenure)
        st.markdown("El Churn es mucho más alto en los primeros meses (baja antigüedad).")
    else:
        st.warning("No se pudieron cargar los datos para el EDA.")

# --- Pestaña 3: Transformación de Variables ---
with tab3:
    st.header("3. Transformación de Variables")
    st.markdown("Se aplicó un pipeline de preprocesamiento robusto:")
    st.markdown("""
    1.  **Limpieza:** `TotalCharges` se convirtió a numérico y los NaNs se rellenaron con 0. `customerID` fue eliminado.
    2.  **Codificación (Target):** `Churn` (Yes/No) se codificó a (1/0).
    3.  **Codificación (Features):**
        * Variables binarias (ej. `Partner`, `Dependents`): Se usó `LabelEncoder` (0/1).
        * Variables multiclase (ej. `Contract`): Se usó **One-Hot Encoding** (`pd.get_dummies`).
    4.  **Escalado:** Las variables numéricas (`tenure`, `MonthlyCharges`, `TotalCharges`) se escalaron usando `MinMaxScaler`.
    """)
    
    if preprocessing_pipeline is not None:
        st.subheader("Columnas Finales (Features)")
        
        # *** INICIO DE LA CORRECCIÓN ***
        # Usar la clave correcta 'processed_columns' del pipeline guardado
        column_key = 'processed_columns' if 'processed_columns' in preprocessing_pipeline else 'ohe_columns'
        
        if column_key in preprocessing_pipeline:
            st.dataframe(pd.DataFrame(columns=preprocessing_pipeline[column_key]))
            st.markdown(f"El dataset final tiene **{len(preprocessing_pipeline[column_key])}** características después del preprocesamiento.")
        else:
            st.error("Error en el pipeline guardado: Faltan las claves 'processed_columns' o 'ohe_columns'.")
        # *** FIN DE LA CORRECCIÓN ***
            
    else:
        st.warning("No se pudo cargar el pipeline de preprocesamiento.")

# --- Funciones para cargar artefactos (gráficos, reportes) ---
def load_image(file_name):
    path = os.path.join(output_dir, file_name)
    if os.path.exists(path):
        return path
    return None

def load_text(file_name):
    path = os.path.join(output_dir, file_name)
    if os.path.exists(path):
        try:
            with open(path, 'r') as f:
                return f.read()
        except Exception as e:
            return f"Error al leer el archivo {file_name}: {e}"
    return None

def load_csv(file_name):
    path = os.path.join(output_dir, file_name)
    if os.path.exists(path):
        try:
            return pd.read_csv(path, index_col=0)
        except Exception as e:
            return f"Error al leer el archivo {file_name}: {e}"
    return None

# --- Pestaña 4: Modelado con Keras ---
with tab4:
    st.header("4. Modelado con Keras / TensorFlow")
    st.markdown("Se implementó un MLP binario.")
    
    st.subheader("Mejores hiperparámetros encontrados (KerasTuner)")
    keras_hps = load_text('keras_results_best_hyperparameters.txt')
    if keras_hps:
        st.code(keras_hps)
    else:
        st.warning("Archivo 'keras_results_best_hyperparameters.txt' no encontrado. Ejecuta la Fase 2.")
    
    st.subheader("Curvas de Aprendizaje (Keras) [Test Set]")
    keras_curves = load_image('keras_results_learning_curves_plot.png')
    if keras_curves:
        st.image(keras_curves)
    else:
        st.warning("Archivo 'keras_results_learning_curves_plot.png' no encontrado. Ejecuta la Fase 2.")

# --- Pestaña 5: Modelado con PyTorch ---
with tab5:
    st.header("5. Modelado con PyTorch")
    st.markdown("Se implementó un MLP binario equivalente usando Skorch.")
    
    st.subheader("Mejores hiperparámetros encontrados (GridSearchCV)")
    pytorch_hps = load_text('pytorch_results_best_hyperparameters.txt')
    if pytorch_hps:
        st.code(pytorch_hps)
    else:
        st.warning("Archivo 'pytorch_results_best_hyperparameters.txt' no encontrado. Ejecuta la Fase 3.")
    
    st.subheader("Curvas de Aprendizaje (PyTorch) [CV]")
    pytorch_curves = load_image('pytorch_results_learning_curves_plot.png')
    if pytorch_curves:
        st.image(pytorch_curves)
    else:
        st.warning("Archivo 'pytorch_results_learning_curves_plot.png' no encontrado. Ejecuta la Fase 3.")

# --- Pestaña 6: Evaluación y Selección ---
with tab6:
    st.header("6. Evaluación del modelo y selección [Test Set]")
    st.markdown("Se comparan los modelos según las métricas de la rúbrica (AUC, F1, Accuracy) usando los artefactos guardados.")

    # Cargar artefactos de evaluación
    df_metrics_comp = load_csv('comparisons_metrics_table.csv')
    keras_report = load_text('keras_results_classification_report.txt')
    pytorch_report = load_text('pytorch_results_classification_report.txt')
    keras_cm_img = load_image('keras_results_confusion_matrix_plot.png')
    pytorch_cm_img = load_image('pytorch_results_confusion_matrix_plot.png')
    metrics_bar_img = load_image('comparisons_metrics_bar_plot.png')
    time_bar_img = load_image('comparisons_tuning_time_bar_plot.png')
    recommendation = load_text('comparisons_recommendation_string.txt')

    if df_metrics_comp is not None and isinstance(df_metrics_comp, pd.DataFrame):
        st.subheader("Tabla Comparativa de Métricas (Test Set)")
        st.dataframe(df_metrics_comp.style.format("{:.4f}"))
    else:
        st.warning("Archivo 'comparisons_metrics_table.csv' no encontrado o corrupto. Ejecuta la Fase 4.")

    if metrics_bar_img:
        st.subheader("Visualización de Comparación de Métricas")
        st.image(metrics_bar_img)
    
    if time_bar_img:
        st.subheader("Visualización de Tiempos de Ajuste (Tuning)")
        st.image(time_bar_img)

    st.subheader("Matrices de Confusión (Test Set)")
    col1_cm, col2_cm = st.columns(2)
    with col1_cm:
        if keras_cm_img:
            st.image(keras_cm_img)
        else:
            st.warning("Matriz Keras no encontrada.")
    with col2_cm:
        if pytorch_cm_img:
            st.image(pytorch_cm_img)
        else:
            st.warning("Matriz PyTorch no encontrada.")
            
    st.subheader("Reportes de Clasificación (Test Set)")
    col1_rep, col2_rep = st.columns(2)
    with col1_rep:
        st.markdown("**Keras**")
        if keras_report:
            st.text(keras_report)
        else:
            st.warning("Reporte Keras no encontrado.")
    with col2_rep:
        st.markdown("**PyTorch**")
        if pytorch_report:
            st.text(pytorch_report)
        else:
            st.warning("Reporte PyTorch no encontrado.")

    st.subheader("Recomendación y Selección del Mejor Modelo")
    if recommendation:
        st.markdown(recommendation)
    else:
        st.warning("Archivo de recomendación no encontrado. Ejecuta la Fase 4.")

# --- Pestaña 7: Predicción (App) ---
with tab7:
    st.header("7. Predicción de Churn (Telco)")
    st.markdown("Utilice el modelo **PyTorch** (ganador de la evaluación) para predecir si un cliente hará Churn.")

    if best_pytorch_model is None or preprocessing_pipeline is None or df_original_eda is None:
        st.error("El modelo PyTorch, el pipeline de preprocesamiento o el CSV original no están disponibles. No se puede realizar la predicción.")
    else:
        st.subheader("Ingrese las características del cliente:")
        
        # --- Creación de Formularios de Entrada ---
        le_partner = preprocessing_pipeline['label_encoders']['Partner']
        le_dependents = preprocessing_pipeline['label_encoders']['Dependents']
        le_phone = preprocessing_pipeline['label_encoders']['PhoneService']
        le_paperless = preprocessing_pipeline['label_encoders']['PaperlessBilling']
        le_gender = preprocessing_pipeline['label_encoders']['gender']

        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### Información de la Cuenta")
            input_tenure = st.number_input("Antigüedad (meses):", min_value=0, max_value=100, value=12)
            input_contract = st.selectbox("Tipo de Contrato:", options=sorted(df_original_eda['Contract'].unique()))
            input_payment = st.selectbox("Método de Pago:", options=sorted(df_original_eda['PaymentMethod'].unique()))
            input_paperless = st.selectbox("Facturación Electrónica:", options=le_paperless.classes_)
            input_monthly = st.number_input("Cargos Mensuales ($):", min_value=0.0, max_value=200.0, value=70.0, format="%.2f")
            input_total = st.number_input("Cargos Totales ($):", min_value=0.0, max_value=10000.0, value=1500.0, format="%.2f")

        with col2:
            st.markdown("#### Información Demográfica y Servicios")
            input_gender = st.selectbox("Género:", options=le_gender.classes_)
            input_senior = st.selectbox("¿Es Adulto Mayor? (SeniorCitizen):", options=[0, 1], format_func=lambda x: 'Sí' if x == 1 else 'No')
            input_partner = st.selectbox("¿Tiene Pareja?:", options=le_partner.classes_)
            input_dependents = st.selectbox("¿Tiene Dependientes?:", options=le_dependents.classes_)
            
            st.markdown("--- Servicios ---")
            input_phone = st.selectbox("Servicio Telefónico:", options=le_phone.classes_)
            
            options_multilines = sorted(df_original_eda['MultipleLines'].unique())
            input_multilines = st.selectbox("Múltiples Líneas:", options=options_multilines, 
                                             index=options_multilines.index('No phone service') if input_phone == 'No' else 0,
                                             disabled=(input_phone == 'No'))

            input_internet = st.selectbox("Servicio de Internet:", options=sorted(df_original_eda['InternetService'].unique()))
            
            options_internet = sorted(df_original_eda['OnlineSecurity'].unique())
            disabled_internet = (input_internet == 'No')
            default_index_internet = options_internet.index('No internet service') if disabled_internet and 'No internet service' in options_internet else 0

            input_online_sec = st.selectbox("Seguridad Online:", options=options_internet, index=default_index_internet, disabled=disabled_internet)
            input_online_back = st.selectbox("Respaldo Online:", options=options_internet, index=default_index_internet, disabled=disabled_internet)
            input_dev_prot = st.selectbox("Protección de Dispositivo:", options=options_internet, index=default_index_internet, disabled=disabled_internet)
            input_tech_sup = st.selectbox("Soporte Técnico:", options=options_internet, index=default_index_internet, disabled=disabled_internet)
            input_stream_tv = st.selectbox("Streaming TV:", options=options_internet, index=default_index_internet, disabled=disabled_internet)
            input_stream_mov = st.selectbox("Streaming Movies:", options=options_internet, index=default_index_internet, disabled=disabled_internet)


        predict_button = st.button("Predecir Churn")

        if predict_button:
            try:
                # 1. Crear DataFrame de entrada (1 fila)
                input_data_dict = {
                    'gender': input_gender,
                    'SeniorCitizen': input_senior,
                    'Partner': input_partner,
                    'Dependents': input_dependents,
                    'tenure': input_tenure,
                    'PhoneService': input_phone,
                    'MultipleLines': input_multilines,
                    'InternetService': input_internet,
                    'OnlineSecurity': input_online_sec,
                    'OnlineBackup': input_online_back,
                    'DeviceProtection': input_dev_prot,
                    'TechSupport': input_tech_sup,
                    'StreamingTV': input_stream_tv,
                    'StreamingMovies': input_stream_mov,
                    'Contract': input_contract,
                    'PaperlessBilling': input_paperless,
                    'PaymentMethod': input_payment,
                    'MonthlyCharges': input_monthly,
                    'TotalCharges': input_total
                }
                
                input_df = pd.DataFrame([input_data_dict])
                
                # 2. Aplicar Preprocesamiento
                
                # Label Encoding binarias (excepto SeniorCitizen)
                for col in [c for c in preprocessing_pipeline['binary_cols'] if c != 'SeniorCitizen']:
                    le = preprocessing_pipeline['label_encoders'][col]
                    input_df[col] = le.transform(input_df[col])
                
                # One-Hot Encoding multiclase
                input_df_ohe = pd.get_dummies(input_df, columns=preprocessing_pipeline['multi_class_cols'])
                
                # *** INICIO DE LA CORRECCIÓN ***
                # Alinear columnas con las del entrenamiento (pipeline['processed_columns'])
                processed_columns = preprocessing_pipeline['processed_columns']
                input_df_aligned = input_df_ohe.reindex(columns=processed_columns, fill_value=0)
                
                # Escalar numéricas
                numerical_cols = preprocessing_pipeline['numerical_cols']
                scaler = preprocessing_pipeline['scaler']
                input_df_aligned[numerical_cols] = scaler.transform(input_df_aligned[numerical_cols])
                # *** FIN DE LA CORRECCIÓN ***
                
                # 3. Preparar Tensor
                input_tensor = torch.tensor(input_df_aligned.values.astype(np.float32))

                # 4. Predicción (PyTorch)
                with torch.no_grad(): 
                    logits = best_pytorch_model.module_(input_tensor)
                    probability = torch.sigmoid(logits).item() 
                
                prediction_class = 1 if probability > 0.5 else 0
                
                # 5. Decodificar Resultado
                predicted_result_label = preprocessing_pipeline['target_le'].inverse_transform([prediction_class])[0]

                st.subheader("Resultado de la Predicción")
                if predicted_result_label == 'Yes':
                    st.error(f"Predicción: **CHURN (Sí)** (Probabilidad de Churn: {probability:.2%})")
                    st.warning("Este cliente tiene una alta probabilidad de abandonar.")
                else:
                    st.success(f"Predicción: **CHURN (No)** (Probabilidad de Churn: {probability:.2%})")
                    st.info("Este cliente probablemente permanecerá en la compañía.")
            
            except Exception as e:
                st.error(f"Error durante la predicción: {e}")
                st.exception(e)