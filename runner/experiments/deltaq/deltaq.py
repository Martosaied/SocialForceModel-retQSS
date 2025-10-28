import json
import os
import subprocess
from src.runner import run_experiment, compile_c_code, compile_model
from src.utils import load_config, create_output_dir, copy_results_to_latest, generate_map
from src import utils
from src.constants import Constants
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.math.Density import Density
from src.math.Clustering import Clustering
from src.plots.DensityRowGraph import DensityRowGraph


DELTAQ = [-8, -7, -6, -5, -4,-3.5, -3, -2.5, -2, -1.5, -1, -0.5, 0, 0.5] # , 1, 1.5, 2]

WIDTH = 50
PEDESTRIAN_COUNT = int(50 * (50 * 0.4) * 0.3)
VOLUMES = 1


def deltaq():
    print(f"Ejecutando iteraciones para {PEDESTRIAN_COUNT} peatones reduciendo Tolerancia y graficando carriles...\n")
    for deltaq in DELTAQ:
        print(f"Ejecutando experimento para deltaq: {deltaq}")
        run(deltaq)

    # Graficar los resultados
    print("Graficando resultados...")
    plot_results()

def run(deltaq):
    """
    Ejecuta el experimento para un número dado de peatones.
    """
    config = load_config('experiments/deltaq/config.json')

    # Crear directorio de salida con el nombre del experimento si se proporciona
    output_dir = create_output_dir(f'experiments/deltaq/results/deltaq_{deltaq}')
    print(f"Directorio de salida creado: {output_dir}")

    config['iterations'] = 10
    config['parameters']['N']['value'] = PEDESTRIAN_COUNT
    config['parameters']['PEDESTRIAN_IMPLEMENTATION']['value'] = Constants.PEDESTRIAN_MMOC
    config['parameters']['BORDER_IMPLEMENTATION']['value'] = Constants.BORDER_NONE

    # Agregar desde dónde hasta dónde se generan los peatones
    config['parameters']['FROM_Y']['value'] = WIDTH * 0.3
    config['parameters']['TO_Y']['value'] = WIDTH * 0.7

    # Guardar copia de configuración en el directorio del experimento
    config_copy_path = os.path.join(output_dir, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)

    formatted_tolerance = np.format_float_positional(1 * 10 ** deltaq)
    formatted_abs_tolerance = np.format_float_positional(1 * 10 ** (deltaq - 3))

    print(f"Tolerance={formatted_tolerance}")
    print(f"AbsTolerance={formatted_abs_tolerance}")

    # Reemplazar las divisiones de la grilla en el modelo
    subprocess.run(['sed', '-i', r's/\bN\s*=\s*[0-9]\+/N = ' + str(PEDESTRIAN_COUNT) + '/', '../retqss/model/helbing_only_qss.mo'])
    subprocess.run(['sed', '-i', r's/\bGRID_DIVISIONS\s*=\s*[0-9]\+/GRID_DIVISIONS = ' + str(VOLUMES) + '/', '../retqss/model/helbing_only_qss.mo'])
    subprocess.run([
        'sed', '-i',
        f's/^[[:space:]]*Tolerance=[^,]*/       Tolerance={formatted_tolerance}/g',
        '../retqss/model/helbing_only_qss.mo'
    ])
    subprocess.run([
        'sed', '-i',
        f's/^[[:space:]]*AbsTolerance=[^,]*/       AbsTolerance={formatted_abs_tolerance}/g',
        '../retqss/model/helbing_only_qss.mo'
    ])

    # Compilar el código C++ si se solicita
    compile_c_code()

    # Compilar el modelo si se solicita
    compile_model('helbing_only_qss')

    # Ejecutar experimento
    run_experiment(
        config, 
        output_dir, 
        'helbing_only_qss', 
        plot=False, 
        copy_results=True
    )

    # Copiar resultados del directorio de salida al directorio latest
    copy_results_to_latest(output_dir)

    print(f"\nExperimento completado. Resultados guardados en {output_dir}")

def plot_results():
    """
    Grafica los resultados del experimento de DeltaQ.
    """
    # Obtener todos los directorios de resultados
    results_dirs = [d for d in os.listdir('experiments/deltaq/results') if os.path.isdir(os.path.join('experiments/deltaq/results', d))]

    # Inicializar estructuras de datos
    performance_data = {deltaq: [] for deltaq in DELTAQ}
    groups_data = {deltaq: [] for deltaq in DELTAQ}
    
    # Leer los directorios de resultados
    for result_dir in results_dirs:
        deltaq = float(result_dir.split('_')[1])
        result_path = os.path.join('experiments/deltaq/results', result_dir, 'latest')
        
        # Leer métricas del archivo CSV
        metrics_file = os.path.join(result_path, 'metrics.csv')
        if os.path.exists(metrics_file):
            df_metrics = pd.read_csv(metrics_file)
            for _, row in df_metrics.iterrows():
                performance_data[deltaq].append(float(row['time']))
                groups_data[deltaq].append(int(row['clustering_based_groups']))

       
        # groups_data_file = []
        # for result_file in os.listdir(os.path.join('experiments/deltaq/results', result_dir, 'latest'))[:2]:
        #     print(f"Using solution.csv for deltaq {deltaq}: {result_file}")
        #     if result_file.endswith('.csv') and result_file != 'metrics.csv':
        #         df = pd.read_csv(os.path.join('experiments/deltaq/results', result_dir, 'latest', result_file))
        #         particles = (len(df.columns) - 1) / 5
        #         groups = Clustering(df, int(particles)).calculate_groups(start_index=100, sample_rate=5)
        #         groups_data_file.append(groups)
        #         print(f"Using solution.csv for deltaq {deltaq}: {groups} data points")
        # groups_data[deltaq].extend(groups_data_file)

    # Calcular estadísticas para cada deltaq
    performance_stats = {}
    groups_stats = {}
    
    for deltaq in DELTAQ:
        if performance_data[deltaq]:
            performance_stats[deltaq] = {
                'mean': np.mean(performance_data[deltaq]),
                'std': np.std(performance_data[deltaq])
            }
        else:
            performance_stats[deltaq] = {'mean': 0, 'std': 0}
            
        if groups_data[deltaq]:
            groups_stats[deltaq] = {
                'mean': np.mean(groups_data[deltaq]),
                'std': np.std(groups_data[deltaq])
            }
        else:
            groups_stats[deltaq] = {'mean': 0, 'std': 0}

    # Ordenar por valores de deltaq
    sorted_deltaqs = sorted(DELTAQ)
    
    # Extraer datos para graficar
    performance_means = [performance_stats[dq]['mean'] for dq in sorted_deltaqs]
    performance_stds = [performance_stats[dq]['std'] for dq in sorted_deltaqs]
    groups_means = [groups_stats[dq]['mean'] for dq in sorted_deltaqs]
    groups_stds = [groups_stats[dq]['std'] for dq in sorted_deltaqs]

    # Crear los gráficos
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.suptitle('Análisis de DeltaQ - Rendimiento y Formación de Carriles', fontsize=16, fontweight='bold')
    
    # Crear etiquetas con notación científica
    deltaq_labels = [f'1e{dq}' for dq in sorted_deltaqs]
    
    # Gráfico 1: Rendimiento por DeltaQ
    x_pos = np.arange(len(sorted_deltaqs))
    bars1 = ax1.bar(x_pos, performance_means, yerr=performance_stds, 
                    capsize=5, alpha=0.7, color='skyblue', edgecolor='navy', width=0.6)
    ax1.set_title('Rendimiento por DeltaQ', fontsize=14)
    ax1.set_xlabel('DeltaQ (Tolerancia)', fontsize=12)
    ax1.set_ylabel('Tiempo de Ejecución (segundos)', fontsize=12)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(deltaq_labels, rotation=45, ha='right', fontsize=10)
    ax1.set_ylim(0, max(performance_means) * 1.3)
    ax1.grid(True, alpha=0.3)
    
    # Agregar valores en las barras (solo si hay espacio suficiente)
    for i, (bar, mean, std) in enumerate(zip(bars1, performance_means, performance_stds)):
        height = bar.get_height()
        if height + std > 0:  # Solo mostrar si la barra es visible
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + max(performance_means) * 0.02,
                    f'{mean:.2f}±{std:.2f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    # Gráfico 2: Cantidad de Carriles por DeltaQ
    bars2 = ax2.bar(x_pos, groups_means, yerr=groups_stds, 
                    capsize=5, alpha=0.7, color='lightgreen', edgecolor='darkgreen', width=0.6)
    ax2.set_title('Cantidad de Carriles Detectados por DeltaQ', fontsize=14)
    ax2.set_xlabel('DeltaQ (Tolerancia)', fontsize=12)
    ax2.set_ylabel('Número de Carriles', fontsize=12)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(deltaq_labels, rotation=45, ha='right', fontsize=10)
    ax2.set_ylim(0, max(groups_means) * 1.3)
    ax2.grid(True, alpha=0.3)
    
    # Agregar valores en las barras (solo si hay espacio suficiente)
    for i, (bar, mean, std) in enumerate(zip(bars2, groups_means, groups_stds)):
        height = bar.get_height()
        if height + std > 0:  # Solo mostrar si la barra es visible
            ax2.text(bar.get_x() + bar.get_width()/2., height + std + max(groups_means) * 0.02,
                    f'{mean:.1f}±{std:.1f}', ha='center', va='bottom', fontsize=8, rotation=90)
    
    plt.subplots_adjust(bottom=0.15)  # Más espacio para las etiquetas rotadas
    plt.savefig('experiments/deltaq/performance_by_deltaq.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Imprimir resumen
    print("\n" + "="*80)
    print("RESUMEN DE RESULTADOS DELTAQ")
    print("="*80)
    print(f"{'DeltaQ':<8} {'Tolerancia':<12} {'Rendimiento (s)':<15} {'Carriles':<10}")
    print("-" * 80)
    
    for dq in sorted_deltaqs:
        perf_mean = performance_stats[dq]['mean']
        perf_std = performance_stats[dq]['std']
        groups_mean = groups_stats[dq]['mean']
        groups_std = groups_stats[dq]['std']
        tolerance = f"1e{dq}"
        print(f"{dq:<8} {tolerance:<12} {perf_mean:.2f}±{perf_std:.2f}     {groups_mean:.1f}±{groups_std:.1f}")
    
    print("="*80)

if __name__ == '__main__':
    deltaq()
