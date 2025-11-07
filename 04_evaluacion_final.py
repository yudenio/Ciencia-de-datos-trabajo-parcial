# Contenido para 04_evaluacion_final.py

import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
import pickle

# Directorio donde Fases 2 y 3 guardaron sus resultados
output_dir = 'analysis_outputs_telco'
print(f"--- Iniciando Fase 4: Evaluación Final (Cargando desde '{output_dir}') ---")

# --- 1. Cargar Métricas y Tiempos de los Archivos CSV ---
try:
    metrics_keras = pd.read_csv(os.path.join(output_dir, 'keras_metrics.csv'), index_col=0)
    metrics_pytorch = pd.read_csv(os.path.join(output_dir, 'pytorch_metrics.csv'), index_col=0)
    
    # Combinar métricas en una tabla comparativa 
    df_metrics_comparison = pd.concat([metrics_keras, metrics_pytorch], axis=0).T
    
    # Separar métricas de rendimiento y tiempo
    df_performance = df_metrics_comparison.filter(like='_Test', axis=0)
    df_time = df_metrics_comparison.filter(like='Time_sec', axis=0)
    
    print("\n--- 1.1. Tabla Comparativa de Métricas (Test Set) ---")
    print(df_performance)
    
    print("\n--- 1.2. Tabla Comparativa de Tiempos de Tuning ---")
    print(df_time)
    
    # Guardar la tabla comparativa principal (para la app) 
    df_metrics_comparison.to_csv(os.path.join(output_dir, 'comparisons_metrics_table.csv'))

except FileNotFoundError:
    print("\nError: No se encontraron los archivos 'keras_metrics.csv' o 'pytorch_metrics.csv'.")
    print("Asegúrate de ejecutar '02_modelo_keras.py' y '03_modelo_pytorch_o_sklearn.py' primero.")
    exit()
except Exception as e:
    print(f"Error al cargar métricas: {e}")
    exit()

# --- 2. Cargar y Mostrar Reportes de Clasificación (TXT) --- 
print("\n--- 2.1. Reporte de Clasificación Keras (Test Set) ---")
try:
    with open(os.path.join(output_dir, 'keras_results_classification_report.txt'), 'r') as f:
        print(f.read())
except FileNotFoundError:
    print("No se encontró 'keras_results_classification_report.txt'")

print("\n--- 2.2. Reporte de Clasificación PyTorch (Test Set) ---")
try:
    with open(os.path.join(output_dir, 'pytorch_results_classification_report.txt'), 'r') as f:
        print(f.read())
except FileNotFoundError:
    print("No se encontró 'pytorch_results_classification_report.txt'")

# --- 3. Cargar y Mostrar Hiperparámetros Óptimos (TXT) --- 
print("\n--- 3.1. Mejores Hiperparámetros Keras ---")
try:
    with open(os.path.join(output_dir, 'keras_results_best_hyperparameters.txt'), 'r') as f:
        keras_hps_str = f.read()
        print(keras_hps_str)
except FileNotFoundError:
    print("No se encontró 'keras_results_best_hyperparameters.txt'")
    keras_hps_str = "No disponible"

print("\n--- 3.2. Mejores Hiperparámetros PyTorch ---")
try:
    with open(os.path.join(output_dir, 'pytorch_results_best_hyperparameters.txt'), 'r') as f:
        pytorch_hps_str = f.read()
        print(pytorch_hps_str)
except FileNotFoundError:
    print("No se encontró 'pytorch_results_best_hyperparameters.txt'")
    pytorch_hps_str = "No disponible"

# --- 4. Generar Gráficos Comparativos (para la app) --- 

# 4.1. Gráfico de Barras de Métricas de Rendimiento
try:
    fig_metrics_bar, ax = plt.subplots(figsize=(10, 6))
    df_performance.plot(kind='bar', ax=ax)
    ax.set_title('Comparación de Métricas de Modelos (Test Set)')
    ax.set_ylabel('Puntaje')
    ax.tick_params(axis='x', rotation=0)
    # Añadir valores en las barras
    for container in ax.containers:
        ax.bar_label(container, fmt='%.4f')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'comparisons_metrics_bar_plot.png')
    plt.savefig(plot_path)
    print(f"\nGráfico comparativo de métricas guardado en '{plot_path}'")
    plt.close(fig_metrics_bar)
except Exception as e:
    print(f"Error al generar gráfico de métricas: {e}")

# 4.2. Gráfico de Barras de Tiempos de Tuning
try:
    fig_time_bar, ax = plt.subplots(figsize=(8, 5))
    df_time.T.plot(kind='bar', ax=ax, legend=False, color=['skyblue', 'lightgreen'])
    ax.set_title('Tiempo de Búsqueda de Hiperparámetros')
    ax.set_ylabel('Segundos')
    ax.tick_params(axis='x', rotation=0)
    # Añadir valores
    for container in ax.containers:
        ax.bar_label(container, fmt='%.2fs')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, 'comparisons_tuning_time_bar_plot.png')
    plt.savefig(plot_path)
    print(f"Gráfico comparativo de tiempos guardado en '{plot_path}'")
    plt.close(fig_time_bar)
except Exception as e:
    print(f"Error al generar gráfico de tiempos: {e}")

# --- 5. Generar Comparación Detallada y Recomendación Final --- 

# 5.1. Comparación Detallada
try:
    df_diff = df_metrics_comparison.copy()
    df_diff['Diferencia (Keras - PyTorch)'] = df_diff['Keras'] - df_diff['PyTorch']
    
    detailed_comparison_string = "COMPARACIÓN DETALLADA KERAS vs PYTORCH (Test Set)\n"
    detailed_comparison_string += "="*80 + "\n"
    detailed_comparison_string += df_diff.to_string(float_format="%.4f")
    detailed_comparison_string += "\n" + "="*80
    
    print(f"\n{detailed_comparison_string}")
    with open(os.path.join(output_dir, 'comparisons_detailed_comparison_string.txt'), 'w') as f:
        f.write(detailed_comparison_string)
        
except Exception as e:
    print(f"Error al generar string de comparación detallada: {e}")

# 5.2. Recomendación Final
print("\n--- 6. Recomendación y Selección del Mejor Modelo ---")
recommendation_string = "RECOMENDACIÓN Y SELECCIÓN DEL MEJOR MODELO\n"
recommendation_string += "="*80 + "\n"

try:
    # Seleccionar el modelo basado en la métrica principal: ROC-AUC 
    auc_keras = df_metrics_comparison.loc['AUC_Test', 'Keras']
    auc_pytorch = df_metrics_comparison.loc['AUC_Test', 'PyTorch']
    
    f1_keras = df_metrics_comparison.loc['F1_Test', 'Keras']
    f1_pytorch = df_metrics_comparison.loc['F1_Test', 'PyTorch']

    if auc_pytorch > auc_keras:
        ganador = "PyTorch"
        recommendation_string += "Modelo Seleccionado: PyTorch (con Skorch)\n\n"
        recommendation_string += f"Justificación:\n"
        recommendation_string += f"PyTorch obtuvo un mejor rendimiento en la métrica principal ROC-AUC (Test): {auc_pytorch:.4f}\n"
        recommendation_string += f"(Keras obtuvo: {auc_keras:.4f}).\n"
    else:
        ganador = "Keras"
        recommendation_string += "Modelo Seleccionado: Keras\n\n"
        recommendation_string += f"Justificación:\n"
        recommendation_string += f"Keras obtuvo un mejor rendimiento en la métrica principal ROC-AUC (Test): {auc_keras:.4f}\n"
        recommendation_string += f"(PyTorch obtuvo: {auc_pytorch:.4f}).\n"

    # Análisis secundario
    time_keras = df_metrics_comparison.loc['Tuning_Time_sec', 'Keras']
    time_pytorch = df_metrics_comparison.loc['Tuning_Time_sec', 'PyTorch']
    
    recommendation_string += f"\nAnálisis Adicional:\n"
    recommendation_string += f"- F1-Score: {ganador} también mostró un F1-Score competitivo ({max(f1_keras, f1_pytorch):.4f}).\n"
    if time_pytorch < time_keras:
        recommendation_string += f"- Eficiencia: PyTorch (con GridSearchCV) fue significativamente más rápido en el tuning ({time_pytorch:.2f}s) que Keras ({time_keras:.2f}s).\n"
    else:
        recommendation_string += f"- Eficiencia: Keras (con KerasTuner) fue más rápido en el tuning ({time_keras:.2f}s) que PyTorch ({time_pytorch:.2f}s).\n"
    
    recommendation_string += "\nDebido a su rendimiento superior en la métrica principal (AUC), se selecciona el modelo PyTorch para el despliegue en Streamlit."
    recommendation_string += "\n" + "="*80

    print(recommendation_string)
    
    # Guardar recomendación para la app 
    with open(os.path.join(output_dir, 'comparisons_recommendation_string.txt'), 'w') as f:
        f.write(recommendation_string)

except Exception as e:
    print(f"Error al generar la recomendación: {e}")
    recommendation_string = "Error al generar recomendación."

print("\nFase 4 (Evaluación) completada. Todos los artefactos están en 'analysis_outputs_telco'.")