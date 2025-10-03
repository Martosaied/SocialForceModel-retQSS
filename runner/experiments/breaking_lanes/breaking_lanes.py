import json
import os
import subprocess
import warnings
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit

from src.runner import run_experiment, compile_c_code, compile_model
from src.utils import load_config, create_output_dir, copy_results_to_latest
from src.math.Density import Density
from src.constants import Constants

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Publication-quality plotting settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Experiment parameters - based on performance_n_pedestrians cell sizes
CELL_SIZES = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10, 12.5, 25, 50]  # metros por celda - diferentes tamaños de celda
WIDTH = 20  # ancho del corredor en metros
GRID_SIZE = 50  # tamaño total de la grilla
PEDESTRIAN_DENSITY = 0.3
PEDESTRIAN_COUNT = int(PEDESTRIAN_DENSITY * WIDTH * GRID_SIZE)
PEDESTRIANS_IMPLEMENTATION = {
    Constants.PEDESTRIAN_NEIGHBORHOOD: "retqss_opt",
    Constants.PEDESTRIAN_MMOC: "retqss_baseline",
}
RUN_EXPERIMENT = True

def breaking_lanes():
    """
    Ejecuta experimentos para medir la formación de lanes usando social_force_model
    con pedestrian_implementation=1 (PEDESTRIAN_NEIGHBORHOOD) y 
    border_implementation=1 (CORRIDOR_ONLY) para diferentes tamaños de celda.
    """
    print(f"Ejecutando experimentos para {PEDESTRIAN_COUNT} peatones con diferentes tamaños de celda")
    print(f"para medir la formación de lanes usando social_force_model...\n")
    print(f"Configuración:")
    print(f"  - Pedestrian Implementation: {Constants.PEDESTRIAN_NEIGHBORHOOD} (PEDESTRIAN_NEIGHBORHOOD)")
    print(f"  - Border Implementation: {Constants.CORRIDOR_ONLY} (CORRIDOR_ONLY)")
    print(f"  - Ancho del corredor: {WIDTH}m")
    print(f"  - Tamaños de celda: {CELL_SIZES}m\n")
    if RUN_EXPERIMENT:
        run(50.0, Constants.PEDESTRIAN_MMOC)
        for cell_size in CELL_SIZES:
            print(f"Ejecutando experimento para tamaño de celda {cell_size}m...")
            run(cell_size, Constants.PEDESTRIAN_NEIGHBORHOOD)
            print(f"Experimento para tamaño de celda {cell_size}m completado.\n")

    # Graficar los resultados
    print("Graficando resultados...")
    plot_results()

def run(cell_size, implementation):
    """
    Ejecuta el experimento para un tamaño de celda dado.
    """
    config = load_config('./experiments/breaking_lanes/config.json')

    # Calcular divisiones de grilla desde el tamaño de celda
    grid_divisions = int(GRID_SIZE / cell_size)
    
    # Crear directorio de salida con el nombre del experimento
    output_dir = create_output_dir(
        'experiments/breaking_lanes/results', 
        f'cell_size_{cell_size}_implementation_{PEDESTRIANS_IMPLEMENTATION[implementation]}'
    )
    print(f"Directorio de salida creado: {output_dir}")

    # Configurar parámetros del experimento
    config['iterations'] = 1
    config['parameters']['N']['value'] = PEDESTRIAN_COUNT
    config['parameters']['PEDESTRIAN_IMPLEMENTATION']['value'] = implementation  # 1
    config['parameters']['BORDER_IMPLEMENTATION']['value'] = Constants.CORRIDOR_ONLY  # 1

    # Configurar corredor de 20m de ancho
    config['parameters']['FROM_Y'] = {
        "name": "FROM_Y",
        "type": "value",
        "value": (GRID_SIZE / 2) - (WIDTH / 2)
    }
    config['parameters']['TO_Y'] = {
        "name": "TO_Y", 
        "type": "value",
        "value": (GRID_SIZE / 2) + (WIDTH / 2)
    }

    # Guardar copia de configuración en el directorio del experimento
    config_copy_path = os.path.join(output_dir, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Actualizar el modelo social_force_model.mo con los parámetros
    subprocess.run(['sed', '-i', r's/\bGRID_DIVISIONS\s*=\s*[0-9]\+/GRID_DIVISIONS = ' + str(grid_divisions) + '/', '../retqss/model/social_force_model.mo'])
    subprocess.run(['sed', '-i', r's/\bN\s*=\s*[0-9]\+/N = ' + str(PEDESTRIAN_COUNT) + '/', '../retqss/model/social_force_model.mo'])

    # Compilar el código C++ si se solicita
    compile_c_code()

    # Compilar el modelo si se solicita
    compile_model('social_force_model')

    # Ejecutar experimento
    run_experiment(
        config, 
        output_dir, 
        'social_force_model', 
        plot=False, 
        copy_results=True
    )

    # Copiar resultados del directorio de salida al directorio latest
    copy_results_to_latest(output_dir)

    print(f"\nExperimento completado. Resultados guardados en {output_dir}")

def plot_results():
    """
    Grafica los resultados del experimento de Breaking Lanes.
    Primera barra es la implementación de referencia (naive), resto son optimizadas.
    """
    # Obtener todos los directorios de resultados
    results_dirs = [d for d in os.listdir('experiments/breaking_lanes/results') if os.path.isdir(os.path.join('experiments/breaking_lanes/results', d))]

    # Inicializar estructuras de datos
    performance_data = {}
    groups_data = {}
    
    # Leer los directorios de resultados
    for result_dir in results_dirs:
        if '_implementation_' not in result_dir:
            continue
            
        parts = result_dir.split('_implementation_')
        if len(parts) != 2:
            continue
            
        cell_size_str = parts[0].split('cell_size_')[1]
        implementation = parts[1]
        
        try:
            cell_size = float(cell_size_str)
        except ValueError:
            continue
            
        result_path = os.path.join('experiments/breaking_lanes/results', result_dir, 'latest')
        
        # Leer métricas del archivo CSV
        metrics_file = os.path.join(result_path, 'metrics.csv')
        if os.path.exists(metrics_file):
            df_metrics = pd.read_csv(metrics_file)
            
            # Inicializar listas si no existen
            if cell_size not in performance_data:
                performance_data[cell_size] = {}
                groups_data[cell_size] = {}
            
            if implementation not in performance_data[cell_size]:
                performance_data[cell_size][implementation] = []
                groups_data[cell_size][implementation] = []
            
            for _, row in df_metrics.iterrows():
                performance_data[cell_size][implementation].append(float(row['time']))
                groups_data[cell_size][implementation].append(int(row['clustering_based_groups']))

    # Preparar datos para el gráfico
    # Primero las optimizadas, luego la referencia (naive) al final
    reference_implementation = PEDESTRIANS_IMPLEMENTATION[Constants.PEDESTRIAN_MMOC]
    optimized_implementation = PEDESTRIANS_IMPLEMENTATION[Constants.PEDESTRIAN_NEIGHBORHOOD]
    
    # Obtener todos los tamaños de celda únicos
    all_cell_sizes = sorted(set(performance_data.keys()))
    
    # Preparar datos para gráfico
    x_labels = []
    performance_means = []
    performance_stds = []
    groups_means = []
    groups_stds = []
    performance_colors = []
    groups_colors = []
    
    # Agregar datos optimizados para cada tamaño de celda primero
    for cell_size in all_cell_sizes:
        if cell_size in performance_data and optimized_implementation in performance_data[cell_size]:
            if performance_data[cell_size][optimized_implementation]:
                perf_mean = np.mean(performance_data[cell_size][optimized_implementation])
                perf_std = np.std(performance_data[cell_size][optimized_implementation])
                groups_mean = np.mean(groups_data[cell_size][optimized_implementation])
                groups_std = np.std(groups_data[cell_size][optimized_implementation])
                
                x_labels.append(f'{cell_size}m')
                performance_means.append(perf_mean)
                performance_stds.append(perf_std)
                groups_means.append(groups_mean)
                groups_stds.append(groups_std)
                performance_colors.append('skyblue')  # Color azul para optimizadas
                groups_colors.append('lightgreen')  # Color verde para optimizadas
    
    # Agregar referencia al final (lado derecho)
    x_labels.append('RETQSS Base')
    ref_perf_mean = 0
    ref_perf_std = 0
    ref_groups_mean = 0
    ref_groups_std = 0
    
    # Buscar datos de referencia (puede estar en cualquier tamaño de celda)
    for cell_size in all_cell_sizes:
        if cell_size in performance_data and reference_implementation in performance_data[cell_size]:
            if performance_data[cell_size][reference_implementation]:
                ref_perf_mean = np.mean(performance_data[cell_size][reference_implementation])
                ref_perf_std = np.std(performance_data[cell_size][reference_implementation])
                ref_groups_mean = np.mean(groups_data[cell_size][reference_implementation])
                ref_groups_std = np.std(groups_data[cell_size][reference_implementation])
                break
    
    performance_means.append(ref_perf_mean)
    performance_stds.append(ref_perf_std)
    groups_means.append(ref_groups_mean)
    groups_stds.append(ref_groups_std)
    performance_colors.append('lightcoral')  # Color rojo para referencia
    groups_colors.append('lightcoral')  # Color verde para referencia

    # Crear los gráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Análisis de tamaño de celda - Referencia vs Optimizada', fontsize=16, fontweight='bold')
    
    # Gráfico 1: Rendimiento
    x_pos = np.arange(len(x_labels))
    bars1 = ax1.bar(x_pos, performance_means, yerr=performance_stds, 
                    capsize=5, alpha=0.7, color=performance_colors, edgecolor='navy', width=0.6)
    ax1.set_title('Rendimiento: Referencia vs Optimizada', fontsize=14)
    ax1.set_xlabel('Implementación', fontsize=12)
    ax1.set_ylabel('Tiempo de Ejecución (ms)', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
    ax1.set_ylim(0, max(performance_means) * 1.3 if max(performance_means) > 0 else 1)
    ax1.grid(True, alpha=0.3)
    
    # Agregar valores en las barras
    for i, (bar, mean, std) in enumerate(zip(bars1, performance_means, performance_stds)):
        height = bar.get_height()
        if height + std > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + max(performance_means) * 0.02,
                    f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Gráfico 2: Formación de Carriles
    bars2 = ax2.bar(x_pos, groups_means, yerr=groups_stds, 
                    capsize=5, alpha=0.7, color=groups_colors, edgecolor='navy', width=0.6)
    ax2.set_title('Formación de Carriles: RETQSS Base vs RETQSS Opt.', fontsize=14)
    ax2.set_xlabel('Implementación', fontsize=12)
    ax2.set_ylabel('Número de Carriles', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=45, ha='right', fontsize=10)
    ax2.set_ylim(0, max(groups_means) * 1.3 if max(groups_means) > 0 else 1)
    ax2.grid(True, alpha=0.3)
    
    # Agregar valores en las barras
    for i, (bar, mean, std) in enumerate(zip(bars2, groups_means, groups_stds)):
        height = bar.get_height()
        if height + std > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + std + max(groups_means) * 0.02,
                    f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Agregar leyenda
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightcoral', label='RETQSS Base'),
        Patch(facecolor='skyblue', label='RETQSS Opt.')
    ]
    ax1.legend(handles=legend_elements, loc='upper right')
    ax2.legend(handles=legend_elements, loc='upper right')
    
    plt.subplots_adjust(bottom=0.15)  # Más espacio para las etiquetas rotadas
    plt.savefig('experiments/breaking_lanes/performance_by_cell_size.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Imprimir resumen
    print("\n" + "="*80)
    print("RESUMEN DE RESULTADOS BREAKING LANES")
    print("="*80)
    print(f"{'Implementación':<15} {'Rendimiento (ms)':<15} {'Carriles':<10}")
    print("-" * 80)
    
    for i, label in enumerate(x_labels):
        perf_mean = performance_means[i]
        perf_std = performance_stds[i]
        groups_mean = groups_means[i]
        groups_std = groups_stds[i]
        print(f"{label:<15} {perf_mean:.2f}±{perf_std:.2f}     {groups_mean:.1f}±{groups_std:.1f}")
    
    print("="*80)

if __name__ == '__main__':
    breaking_lanes()
