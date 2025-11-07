# Contenido para 02_modelo_keras.py

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
import keras_tuner as kt
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay, 
    roc_auc_score, f1_score, accuracy_score
)
import pickle

# Definir directorio de salida
output_dir = 'analysis_outputs_telco'
os.makedirs(output_dir, exist_ok=True)

# --- 1. Configuración de Semilla ---
tf.random.set_seed(42)
np.random.seed(42)

# --- 2. Carga de Datos Preprocesados (de Fase 1) ---
print("Cargando datos preprocesados de la Fase 1...")

INPUT_DIM = None # Inicializar

try:
    # Cargar los datos divididos
    X_train_full = np.load('X_train_full.npy', allow_pickle=True)
    y_train_full = np.load('y_train_full.npy', allow_pickle=True)
    X_test = np.load('X_test.npy', allow_pickle=True)
    y_test = np.load('y_test.npy', allow_pickle=True)
    
    X_train = np.load('X_train.npy', allow_pickle=True)
    y_train = np.load('y_train.npy', allow_pickle=True)
    X_val = np.load('X_val.npy', allow_pickle=True)
    y_val = np.load('y_val.npy', allow_pickle=True)

    # Convertir a float32 para Keras/TensorFlow
    X_train = X_train.astype(np.float32)
    y_train = y_train.astype(np.float32)
    X_val = X_val.astype(np.float32)
    y_val = y_val.astype(np.float32)
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)
    X_train_full = X_train_full.astype(np.float32)
    y_train_full = y_train_full.astype(np.float32)

    INPUT_DIM = X_train.shape[1] 
    print(f"Datos cargados. Dimensión de entrada (features): {INPUT_DIM}")

except FileNotFoundError:
    print("\nError: Archivos .npy no encontrados.")
    print("Por favor, asegúrate de ejecutar primero el script '01_eda_telco.ipynb'.")
except Exception as e:
    print(f"Ocurrió un error inesperado al cargar los datos: {e}")

# --- 3. Definición del Modelo de Búsqueda (KerasTuner) ---
if INPUT_DIM is not None:

    def build_keras_model(hp):
        """Función para construir el modelo Keras que KerasTuner utilizará."""
        model = keras.Sequential(name="MLP_Keras_Binary_Telco")

        hp_units1 = hp.Int('units1', min_value=32, max_value=256, step=32)
        hp_units2 = hp.Int('units2', min_value=16, max_value=128, step=16)
        hp_dropout = hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)
        hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
        hp_activation = hp.Choice('activation', ['relu', 'tanh'])

        model.add(layers.Dense(units=hp_units1, activation=hp_activation, input_shape=(INPUT_DIM,)))
        model.add(layers.Dropout(rate=hp_dropout))
        model.add(layers.Dense(units=hp_units2, activation=hp_activation))
        model.add(layers.Dropout(rate=hp_dropout))
        
        model.add(layers.Dense(1, activation='sigmoid'))

        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
            loss='binary_crossentropy',
            metrics=['accuracy', tf.keras.metrics.AUC(name='auc')]
        )
        return model

    # --- 4. Configuración del Tuner ---
    search_early_stopping = EarlyStopping(
        monitor='val_auc', 
        patience=10, 
        mode='max',
        restore_best_weights=True 
    )

    tuner_keras = kt.Hyperband(
        build_keras_model,
        objective=kt.Objective("val_auc", direction="max"),
        max_epochs=50, 
        factor=3,
        seed=42,
        directory='keras_tuner_dir',
        project_name='telco_churn_keras',
        overwrite=True
    )

    print("\n--- 5. Ejecución del Tuning (KerasTuner) ---")
    start_time = time.time()

    tuner_keras.search(
        X_train, y_train,
        epochs=50,
        validation_data=(X_val, y_val),
        callbacks=[search_early_stopping],
        verbose=1
    )

    tuning_time_keras = time.time() - start_time
    print(f"\nBúsqueda (KerasTuner) completada en {tuning_time_keras:.2f} segundos")

    # --- 6. Obtención y Guardado del Mejor Modelo ---
    best_hps_keras = tuner_keras.get_best_hyperparameters(num_trials=1)[0]

    print("\n--- Mejores Hiperparámetros (Keras) ---")
    keras_best_hps_string = "Keras (Tuner):\n"
    keras_best_hps_string += f"- units1: {best_hps_keras.get('units1')}\n"
    keras_best_hps_string += f"- units2: {best_hps_keras.get('units2')}\n"
    keras_best_hps_string += f"- dropout: {best_hps_keras.get('dropout'):.2f}\n"
    keras_best_hps_string += f"- activation: {best_hps_keras.get('activation')}\n"
    keras_best_hps_string += f"- learning_rate: {best_hps_keras.get('learning_rate')}\n"
    print(keras_best_hps_string)

    with open(os.path.join(output_dir, 'keras_results_best_hyperparameters.txt'), 'w') as f:
        f.write(keras_best_hps_string)

    print("\n--- 7. Re-entrenando el mejor modelo Keras en X_train_full ---")
    best_keras_model = build_keras_model(best_hps_keras)

    final_early_stopping = EarlyStopping(
        monitor='val_auc', 
        patience=10, 
        mode='max',
        restore_best_weights=True
    )

    # *** INICIO DE LA CORRECCIÓN ***
    # Corregimos la línea 160: Usamos un batch_size fijo (ej. 32)
    history_keras = best_keras_model.fit(
        X_train_full, y_train_full,
        epochs=100, 
        validation_data=(X_test, y_test),
        callbacks=[final_early_stopping],
        batch_size=32, # Usamos un valor fijo, ya que no se incluyó en el tuner
        verbose=1
    )
    # *** FIN DE LA CORRECCIÓN ***

    print("Modelo Keras final entrenado.")

    # --- 8. Guardado de Artefactos (Keras) ---
    model_path = os.path.join(output_dir, 'best_keras_model.keras')
    best_keras_model.save(model_path)
    print(f"Modelo Keras guardado en '{model_path}'")

    # Guardar gráficos de curvas de aprendizaje
    try:
        fig_keras_learn, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        ax1.plot(history_keras.history['loss'], label='Pérdida (Train)')
        ax1.plot(history_keras.history['val_loss'], label='Pérdida (Test/Val)')
        ax1.set_title('Keras: Curva de Pérdida')
        ax1.set_xlabel('Época'); ax1.set_ylabel('Loss'); ax1.legend()
        
        ax2.plot(history_keras.history['auc'], label='AUC (Train)')
        ax2.plot(history_keras.history['val_auc'], label='AUC (Test/Val)')
        ax2.set_title('Keras: Curva ROC-AUC')
        ax2.set_xlabel('Época'); ax2.set_ylabel('AUC'); ax2.legend()
        
        fig_keras_learn.tight_layout()
        plot_path = os.path.join(output_dir, 'keras_results_learning_curves_plot.png')
        plt.savefig(plot_path)
        print(f"Gráfico de curvas de aprendizaje Keras guardado en '{plot_path}'")
        plt.close(fig_keras_learn) # Cerrar la figura
        
    except Exception as e:
        print(f"Error al guardar gráficos Keras: {e}")

    # Guardar reporte de clasificación
    y_pred_keras_proba = best_keras_model.predict(X_test)
    y_pred_keras_class = (y_pred_keras_proba > 0.5).astype(int)

    keras_report_str = classification_report(y_test, y_pred_keras_class, target_names=['No Churn (0)', 'Churn (1)'])
    print("\n--- Reporte de Clasificación Keras (Test Set) ---")
    print(keras_report_str)
    report_path = os.path.join(output_dir, 'keras_results_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("Reporte de Clasificación Keras (Test Set):\n")
        f.write(keras_report_str)
    print(f"Reporte de clasificación Keras guardado en '{report_path}'")
        
    # Guardar Matriz de Confusión
    try:
        cm_keras = confusion_matrix(y_test, y_pred_keras_class)
        disp_keras = ConfusionMatrixDisplay(confusion_matrix=cm_keras, display_labels=['No Churn', 'Churn'])
        fig_cm, ax_cm = plt.subplots()
        disp_keras.plot(ax=ax_cm, cmap='Blues')
        ax_cm.set_title("Keras: Matriz de Confusión (Test Set)")
        cm_path = os.path.join(output_dir, 'keras_results_confusion_matrix_plot.png')
        plt.savefig(cm_path)
        print(f"Matriz de confusión Keras guardada en '{cm_path}'")
        plt.close(fig_cm) # Cerrar la figura
    except Exception as e:
        print(f"Error al guardar CM Keras: {e}")
        
    # Guardar métricas principales
    metrics_keras = {
        "AUC_Test": roc_auc_score(y_test, y_pred_keras_proba),
        "F1_Test": f1_score(y_test, y_pred_keras_class),
        "Accuracy_Test": accuracy_score(y_test, y_pred_keras_class),
        "Tuning_Time_sec": tuning_time_keras
    }
    metrics_path = os.path.join(output_dir, 'keras_metrics.csv')
    pd.DataFrame([metrics_keras], index=["Keras"]).to_csv(metrics_path)
    print(f"Métricas Keras guardadas en '{metrics_path}'")

    print("\nFase 2 (Keras) completada y artefactos guardados.")

else:
    print("\n--- EJECUCIÓN OMITIDA ---")
    print("Los datos no se cargaron (INPUT_DIM es None). Se omitió el modelado y tuning de Keras.")