import json
import os
import warnings

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit

from src.utils import load_config, create_output_dir
from src.math.Density import Density
from src.constants import Constants
from src.experiments import (
    ConfigBuilder, ModelUpdater, ExperimentRunner,
    apply_publication_style, DEFAULT_COLORS
)

warnings.filterwarnings('ignore')
apply_publication_style()

# Experiment parameters
CELL_SIZES = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10, 12.5, 25, 50]
WIDTH = 20
GRID_SIZE = 50
PEDESTRIAN_DENSITY = 0.3
PEDESTRIAN_COUNT = int(PEDESTRIAN_DENSITY * WIDTH * GRID_SIZE)
PEDESTRIANS_IMPLEMENTATION = {
    Constants.PEDESTRIAN_NEIGHBORHOOD: "retqss_opt",
    Constants.PEDESTRIAN_MMOC: "retqss_baseline",
}
RUN_EXPERIMENT = False


def get_simulated_time():
    """Get simulated time from config.json."""
    config = load_config('./experiments/breaking_lanes/config.json')
    return config['parameters']['FORCE_TERMINATION_AT']['value']

def breaking_lanes():
    """Run experiments to measure lane formation with different cell sizes."""
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

    print("Graficando resultados...")
    plot_results()

def run(cell_size, implementation):
    """Execute experiment for a given cell size."""
    config = load_config('./experiments/breaking_lanes/config.json')
    grid_divisions = int(GRID_SIZE / cell_size)
    
    output_dir = create_output_dir(
        'experiments/breaking_lanes/results', 
        f'cell_size_{cell_size}_implementation_{PEDESTRIANS_IMPLEMENTATION[implementation]}'
    )
    print(f"Directorio de salida creado: {output_dir}")

    # Use ConfigBuilder for cleaner parameter setup
    ConfigBuilder(config) \
        .set_iterations(10) \
        .set_pedestrian_count(PEDESTRIAN_COUNT) \
        .set_pedestrian_implementation(implementation) \
        .set_border_implementation(Constants.CORRIDOR_ONLY) \
        .set_corridor(GRID_SIZE, WIDTH)

    # Save configuration
    config_copy_path = os.path.join(output_dir, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Update model parameters
    model = ModelUpdater('../retqss/model/social_force_model.mo')
    model.update_parameters({
        'GRID_DIVISIONS': grid_divisions,
        'N': PEDESTRIAN_COUNT
    })

    # Use ExperimentRunner for compilation and execution
    ExperimentRunner.run_standard_experiment(config, output_dir, 'social_force_model')

    print(f"\nExperimento completado. Resultados guardados en {output_dir}")

def plot_results():
    """Plot results from Breaking Lanes experiment."""
    # Get simulated time
    simulated_time = get_simulated_time()
    print(f"Tiempo simulado: {simulated_time} segundos")
    
    results_dirs = [d for d in os.listdir('experiments/breaking_lanes/results') if os.path.isdir(os.path.join('experiments/breaking_lanes/results', d))]

    performance_data = {}
    groups_data = {}
    execution_time_data = {}
    
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
        
        metrics_file = os.path.join(result_path, 'metrics.csv')
        if os.path.exists(metrics_file):
            df_metrics = pd.read_csv(metrics_file)
            
            if cell_size not in performance_data:
                performance_data[cell_size] = {}
                groups_data[cell_size] = {}
                execution_time_data[cell_size] = {}
            
            if implementation not in performance_data[cell_size]:
                performance_data[cell_size][implementation] = []
                groups_data[cell_size][implementation] = []
                execution_time_data[cell_size][implementation] = []
            
            for _, row in df_metrics.iterrows():
                execution_time = float(row['time'])
                # Calculate simulation speed ratio: simulated_time / execution_time
                speed_ratio = simulated_time / execution_time
                performance_data[cell_size][implementation].append(speed_ratio)
                groups_data[cell_size][implementation].append(int(row['clustering_based_groups']))
                execution_time_data[cell_size][implementation].append(execution_time)

    # Prepare data for plot
    reference_implementation = PEDESTRIANS_IMPLEMENTATION[Constants.PEDESTRIAN_MMOC]
    optimized_implementation = PEDESTRIANS_IMPLEMENTATION[Constants.PEDESTRIAN_NEIGHBORHOOD]
    
    all_cell_sizes = sorted(set(performance_data.keys()))
    
    x_labels = []
    performance_means = []
    performance_stds = []
    groups_means = []
    groups_stds = []
    time_means = []
    time_stds = []
    performance_colors = []
    groups_colors = []
    
    for cell_size in all_cell_sizes:
        if cell_size in performance_data and optimized_implementation in performance_data[cell_size]:
            if performance_data[cell_size][optimized_implementation]:
                perf_mean = np.mean(performance_data[cell_size][optimized_implementation])
                perf_std = np.std(performance_data[cell_size][optimized_implementation])
                groups_mean = np.mean(groups_data[cell_size][optimized_implementation])
                groups_std = np.std(groups_data[cell_size][optimized_implementation])
                time_mean = np.mean(execution_time_data[cell_size][optimized_implementation])
                time_std = np.std(execution_time_data[cell_size][optimized_implementation])
                
                x_labels.append(f'{cell_size}m')
                performance_means.append(perf_mean)
                performance_stds.append(perf_std)
                groups_means.append(groups_mean)
                groups_stds.append(groups_std)
                time_means.append(time_mean)
                time_stds.append(time_std)
                performance_colors.append(DEFAULT_COLORS['optimized'])
                groups_colors.append('lightgreen')
    
    x_labels.append('RETQSS Base')
    ref_perf_mean = 0
    ref_perf_std = 0
    ref_groups_mean = 0
    ref_groups_std = 0
    ref_time_mean = 0
    ref_time_std = 0
    
    for cell_size in all_cell_sizes:
        if cell_size in performance_data and reference_implementation in performance_data[cell_size]:
            if performance_data[cell_size][reference_implementation]:
                ref_perf_mean = np.mean(performance_data[cell_size][reference_implementation])
                ref_perf_std = np.std(performance_data[cell_size][reference_implementation])
                ref_groups_mean = np.mean(groups_data[cell_size][reference_implementation])
                ref_groups_std = np.std(groups_data[cell_size][reference_implementation])
                ref_time_mean = np.mean(execution_time_data[cell_size][reference_implementation])
                ref_time_std = np.std(execution_time_data[cell_size][reference_implementation])
                break
    
    performance_means.append(ref_perf_mean)
    performance_stds.append(ref_perf_std)
    groups_means.append(ref_groups_mean)
    groups_stds.append(ref_groups_std)
    time_means.append(ref_time_mean)
    time_stds.append(ref_time_std)
    performance_colors.append(DEFAULT_COLORS['reference'])
    groups_colors.append(DEFAULT_COLORS['reference'])

    fig, ax1 = plt.subplots(1, 1, figsize=(16, 12))
    x_pos = np.arange(len(x_labels))
    bars1 = ax1.bar(x_pos, performance_means, yerr=performance_stds, 
                    color=performance_colors, width=0.6)
    ax1.set_xlabel('Tamaño de Celda (m)')
    ax1.set_ylabel('RTF - tiempo simulado(s) / tiempo real(s)')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(x_labels, rotation=45, ha='right')
    
    # Set y-axis limits to accommodate values below 1.0
    min_val = min(performance_means) if len(performance_means) > 0 else 0
    max_val = max(performance_means) if len(performance_means) > 0 else 1
    max_std = max(performance_stds) if len(performance_stds) > 0 else 0
    y_min = max(0, min_val - max_std * 1.5)
    y_max = max_val * 1.4
    ax1.set_ylim(y_min, y_max)
    
    # Add horizontal line at 1.0 to show real-time threshold
    ax1.axhline(y=1.0, color='red', linestyle='--', label='Tiempo Real (1.0x)')
    ax1.grid(True)
    
    max_perf = max(performance_means) if len(performance_means) > 0 else 0
    for i, (bar, mean, std, time_mean, time_std) in enumerate(zip(bars1, performance_means, performance_stds, time_means, time_stds)):
        height = bar.get_height()
        if height + std > 0:
            ax1.text(bar.get_x() + bar.get_width()/2., height + std + max_perf * 0.03,
                    f'{mean:.2f}x\n({time_mean:.1f}s)', ha='center', va='bottom')
    
    from matplotlib.patches import Patch
    legend_elements_perf = [
        Patch(facecolor='lightcoral', label='RETQSS Base'),
        Patch(facecolor='skyblue', label='RETQSS Opt.'),
        Line2D([0], [0], color='red', linestyle='--', label='Tiempo Real (1.0x)'),
        Line2D([], [], linestyle='None', label=f'Tiempo simulado: {simulated_time:.0f}s'),
    ]
    ax1.legend(handles=legend_elements_perf, loc='upper right')

    plt.tight_layout()
    plt.subplots_adjust(left=0.08, bottom=0.15, top=0.94, right=0.98)
    plt.savefig('experiments/breaking_lanes/performance_by_cell_size.png')
    plt.close()
    
    # Update groups_colors to use lightgreen for optimized
    groups_colors_updated = []
    for i, color in enumerate(groups_colors):
        if color == DEFAULT_COLORS['reference']:
            groups_colors_updated.append(DEFAULT_COLORS['reference'])
        else:
            groups_colors_updated.append('lightgreen')

    fig, ax2 = plt.subplots(1, 1, figsize=(16, 12))
    bars2 = ax2.bar(x_pos, groups_means, yerr=groups_stds, 
                    color=groups_colors_updated, width=0.6)
    ax2.set_xlabel('Tamaño de Celda (m)')
    ax2.set_ylabel('Número de Carriles')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(x_labels, rotation=45, ha='right')
    max_groups = max(groups_means) if len(groups_means) > 0 else 0
    if max_groups > 0:
        ax2.set_ylim(0, max_groups * 1.4)
    ax2.grid(True)
    
    for i, (bar, mean, std) in enumerate(zip(bars2, groups_means, groups_stds)):
        height = bar.get_height()
        if height + std > 0:
            ax2.text(bar.get_x() + bar.get_width()/2., height + std + max_groups * 0.03 if max_groups > 0 else 0.03,
                    f'{mean:.1f}', ha='center', va='bottom')
    
    legend_elements_groups = [
        Patch(facecolor='lightcoral', label='RETQSS Base'),
        Patch(facecolor='lightgreen', edgecolor='darkgreen', label='RETQSS Opt.')
    ]
    ax2.legend(handles=legend_elements_groups, loc='upper right')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.08, bottom=0.15, top=0.94, right=0.98)
    plt.savefig('experiments/breaking_lanes/lane_groups_by_cell_size.png')
    plt.close()
    
    print("\n" + "="*80)
    print("RESUMEN DE RESULTADOS BREAKING LANES")
    print("="*80)
    print(f"{'Implementación':<15} {'Velocidad (ratio)':<20} {'Carriles':<10}")
    print("-" * 80)
    
    for i, label in enumerate(x_labels):
        perf_mean = performance_means[i]
        perf_std = performance_stds[i]
        groups_mean = groups_means[i]
        groups_std = groups_stds[i]
        print(f"{label:<15} {perf_mean:.2f}x±{perf_std:.2f}       {groups_mean:.1f}±{groups_std:.1f}")
    
    print("="*80)
    print(f"\nNota: Velocidad ratio = tiempo simulado ({simulated_time}s) / tiempo de ejecución")
    print("      Un ratio de 5.0x significa que la simulación corre 5 veces más rápido que tiempo real")

if __name__ == '__main__':
    breaking_lanes()
