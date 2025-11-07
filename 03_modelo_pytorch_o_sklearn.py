# Contenido para 03_modelo_pytorch_o_sklearn.py

import torch
import torch.nn as nn
import torch.optim as optim
from skorch import NeuralNetClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
import os
import time
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    classification_report, confusion_matrix, ConfusionMatrixDisplay, 
    roc_auc_score, f1_score, accuracy_score, roc_curve
)
import pickle

# Definir directorio de salida
output_dir = 'analysis_outputs_telco'
os.makedirs(output_dir, exist_ok=True)

# --- 1. Configuración de Semilla ---
torch.manual_seed(42)
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
    
    # Convertir a float32 para PyTorch
    X_train_full = X_train_full.astype(np.float32)
    y_train_full = y_train_full.astype(np.float32) # Target debe ser float para BCEWithLogitsLoss
    X_test = X_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    INPUT_DIM = X_train_full.shape[1]
    print(f"Datos cargados. Dimensión de entrada (features): {INPUT_DIM}")

except FileNotFoundError:
    print("\nError: Archivos .npy no encontrados.")
    print("Por favor, asegúrate de ejecutar primero el script '01_eda_telco.ipynb'.")
except Exception as e:
    print(f"Ocurrió un error inesperado al cargar los datos: {e}")

# --- 3. Definición del Modelo PyTorch (Binario) ---
# Solo definir y ejecutar si los datos se cargaron
if INPUT_DIM is not None:

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

    # --- 4. Configuración del Wrapper de Skorch ---
    
    # *** INICIO DE LA CORRECCIÓN ***
    # Se eliminaron 'iterator_train__target_dtype' y 'iterator_valid__target_dtype'
    net_pytorch = NeuralNetClassifier(
        module=MLP_PyTorch_Binary,
        module__input_dim=INPUT_DIM,
        criterion=nn.BCEWithLogitsLoss, # Pérdida binaria (espera logits)
        optimizer=optim.Adam,
        max_epochs=50, 
        batch_size=32,
        verbose=0,
        # Usar validación cruzada (CV) de GridSearchCV, no train_split
        train_split=None 
    )
    # *** FIN DE LA CORRECCIÓN ***

    # --- 5. Configuración y Ejecución del Tuning (GridSearchCV) ---
    params_pytorch = {
        'optimizer__lr': [1e-3, 1e-4], # Learning rate
        'module__num_units1': [64, 128], # Neuronas capa 1
        'module__num_units2': [32, 64], # Neuronas capa 2
        'module__dropout_p': [0.2, 0.4], # Dropout
        'batch_size': [32, 64] # Batch size
    }

    gs_pytorch = GridSearchCV(
        estimator=net_pytorch,
        param_grid=params_pytorch,
        cv=3, # Validación cruzada de 3 folds
        scoring='roc_auc', # Métrica principal 
        verbose=2,
        n_jobs=-1, # Usar todos los cores
        refit=True # Re-entrenar el mejor modelo en X_train_full
    )

    print("\n--- 6. Ejecución del Tuning (PyTorch + GridSearchCV) ---")
    start_time = time.time()

    # GridSearchCV (skorch) espera X_train_full (float32)
    # y_train_full (float32) y [N, 1] para BCEWithLogitsLoss
    y_train_full_reshaped = y_train_full.reshape(-1, 1)
    
    gs_pytorch.fit(X_train_full, y_train_full_reshaped)

    tuning_time_pytorch = time.time() - start_time
    print(f"\nBúsqueda (PyTorch/GridSearchCV) completada en {tuning_time_pytorch:.2f} segundos")

    # --- 7. Obtención y Guardado del Mejor Modelo ---
    best_pytorch_model = gs_pytorch.best_estimator_
    best_pytorch_params = gs_pytorch.best_params_

    print("\n--- Mejores Hiperparámetros (PyTorch) ---")
    pytorch_best_params_string = "PyTorch (GridSearchCV):\n"
    pytorch_best_params_string += f"- max_epochs: {best_pytorch_model.max_epochs}\n" 
    for key, val in best_pytorch_params.items():
        pytorch_best_params_string += f"- {key}: {val}\n"
    print(pytorch_best_params_string)

    with open(os.path.join(output_dir, 'pytorch_results_best_hyperparameters.txt'), 'w') as f:
        f.write(pytorch_best_params_string)

    print("Modelo PyTorch final (Skorch) ya entrenado por GridSearchCV (refit=True).")
    
    model_path = os.path.join(output_dir, 'best_pytorch_model.pkl')
    with open(model_path, 'wb') as f:
        pickle.dump(best_pytorch_model, f)
    print(f"Modelo Skorch (PyTorch) guardado en '{model_path}'")
    
    # --- 8. Guardado de Artefactos (PyTorch) ---
    
    # Guardar gráficos de curvas de aprendizaje
    try:
        history_pytorch = best_pytorch_model.history
        fig_pytorch_learn, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Pérdida
        ax1.plot(history_pytorch[:, 'train_loss'], label='Pérdida (Train)')
        ax1.plot(history_pytorch[:, 'valid_loss'], label='Pérdida (Val)')
        ax1.set_title('PyTorch: Curva de Pérdida')
        ax1.set_xlabel('Época'); ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Skorch no calcula AUC en validación por defecto, usamos 'valid_loss'
        ax2.plot(history_pytorch[:, 'valid_loss'], label='Pérdida (Val)')
        ax2.set_title('PyTorch: Pérdida de Validación')
        ax2.set_xlabel('Época'); ax2.set_ylabel('Loss')
        ax2.legend()
        
        fig_pytorch_learn.tight_layout()
        plot_path = os.path.join(output_dir, 'pytorch_results_learning_curves_plot.png')
        plt.savefig(plot_path)
        print(f"Gráfico de curvas de aprendizaje PyTorch guardado en '{plot_path}'")
        plt.close(fig_pytorch_learn) # Cerrar la figura

    except Exception as e:
        print(f"Error al guardar gráficos PyTorch (puede ser normal si CV no guardó historial): {e}")

    # Guardar reporte de clasificación
    y_pred_pytorch_class = best_pytorch_model.predict(X_test)
    y_pred_pytorch_proba = best_pytorch_model.predict_proba(X_test)[:, 1] # Probabilidad de clase 1

    pytorch_report_str = classification_report(y_test, y_pred_pytorch_class, target_names=['No Churn (0)', 'Churn (1)'])
    print("\n--- Reporte de Clasificación PyTorch (Test Set) ---")
    print(pytorch_report_str)
    report_path = os.path.join(output_dir, 'pytorch_results_classification_report.txt')
    with open(report_path, 'w') as f:
        f.write("Reporte de Clasificación PyTorch (Test Set):\n")
        f.write(pytorch_report_str)
    print(f"Reporte de clasificación PyTorch guardado en '{report_path}'")
        
    # Guardar Matriz de Confusión
    try:
        cm_pytorch = confusion_matrix(y_test, y_pred_pytorch_class)
        disp_pytorch = ConfusionMatrixDisplay(confusion_matrix=cm_pytorch, display_labels=['No Churn', 'Churn'])
        fig_cm, ax_cm = plt.subplots()
        disp_pytorch.plot(ax=ax_cm, cmap='Greens')
        ax_cm.set_title("PyTorch: Matriz de Confusión (Test Set)")
        cm_path = os.path.join(output_dir, 'pytorch_results_confusion_matrix_plot.png')
        plt.savefig(cm_path)
        print(f"Matriz de confusión PyTorch guardada en '{cm_path}'")
        plt.close(fig_cm) # Cerrar la figura
    except Exception as e:
        print(f"Error al guardar CM PyTorch: {e}")
        
    # Guardar métricas principales
    metrics_pytorch = {
        "AUC_Test": roc_auc_score(y_test, y_pred_pytorch_proba),
        "F1_Test": f1_score(y_test, y_pred_pytorch_class),
        "Accuracy_Test": accuracy_score(y_test, y_pred_pytorch_class),
        "Tuning_Time_sec": tuning_time_pytorch
    }
    metrics_path = os.path.join(output_dir, 'pytorch_metrics.csv')
    pd.DataFrame([metrics_pytorch], index=["PyTorch"]).to_csv(metrics_path)
    print(f"Métricas PyTorch guardadas en '{metrics_path}'")

    print("\nFase 3 (PyTorch) completada y artefactos guardados.")

else:
    print("\n--- EJECUCIÓN OMITIDA ---")
    print("Los datos no se cargaron (INPUT_DIM es None). Se omitió el modelado y tuning de PyTorch.")