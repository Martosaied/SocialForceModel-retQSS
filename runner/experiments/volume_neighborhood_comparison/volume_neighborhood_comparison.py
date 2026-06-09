import json
import os
import warnings

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

from src.utils import load_config, create_output_dir
from src.constants import Constants
from src.experiments import (
    ConfigBuilder, ModelUpdater, ExperimentRunner,
    apply_publication_style, DEFAULT_COLORS
)

warnings.filterwarnings('ignore')
apply_publication_style()

# Experiment parameters
CELL_SIZES = [2.0, 3.0, 4.0, 5.0, 7.5, 10, 12.5, 25, 50]
VOLUME_NEIGHBORHOOD_TYPES = [0, 1]  # 0: face sharing, 1: vertex sharing
WIDTH = 20
GRID_SIZE = 50
PEDESTRIAN_DENSITY = 0.3
PEDESTRIAN_COUNT = int(PEDESTRIAN_DENSITY * WIDTH * GRID_SIZE)
RUN_EXPERIMENT = True

NEIGHBORHOOD_NAMES = {
    0: "Face Sharing",
    1: "Vertex Sharing"
}

def volume_neighborhood_comparison():
    """Run experiments comparing volume neighborhood types."""
    print(f"Running volume neighborhood comparison experiments...")
    print(f"Configuration:")
    print(f"  - Pedestrian count: {PEDESTRIAN_COUNT}")
    print(f"  - Corridor width: {WIDTH}m")
    print(f"  - Grid size: {GRID_SIZE}m")
    print(f"  - Cell sizes: {CELL_SIZES}m")
    print(f"  - Neighborhood types: {[NEIGHBORHOOD_NAMES[t] for t in VOLUME_NEIGHBORHOOD_TYPES]}\n")
    
    if RUN_EXPERIMENT:
        total_experiments = len(CELL_SIZES) * len(VOLUME_NEIGHBORHOOD_TYPES)
        current = 0
        
        for cell_size in CELL_SIZES:
            for neighborhood_type in VOLUME_NEIGHBORHOOD_TYPES:
                if cell_size == 0.5 and neighborhood_type == 0:
                    print(f"\n{'='*80}")
                    print(f"Skipping: cell_size={cell_size}m, neighborhood_type={neighborhood_type}")
                    print(f"{'='*80}")
                    continue
                
                current += 1
                print(f"\n{'='*80}")
                print(f"[{current}/{total_experiments}] Running experiment:")
                print(f"  Cell size: {cell_size}m")
                print(f"  Neighborhood type: {NEIGHBORHOOD_NAMES[neighborhood_type]}")
                print(f"{'='*80}")
                
                run(cell_size, neighborhood_type)
                
                print(f"Experiment completed: cell_size={cell_size}m, type={NEIGHBORHOOD_NAMES[neighborhood_type]}")
    
    print("\n" + "="*80)
    print("Generating comparison visualizations...")
    print("="*80)
    plot_results()
    
    print("\n" + "="*80)
    print("All experiments completed successfully!")
    print("="*80)


def run(cell_size, neighborhood_type):
    """Execute experiment for a given cell size and neighborhood type."""
    config = load_config('./experiments/volume_neighborhood_comparison/config.json')
    grid_divisions = int(GRID_SIZE / cell_size)
    
    output_dir = create_output_dir(
        'experiments/volume_neighborhood_comparison/results',
        f'cell_size_{cell_size}_neighborhood_{neighborhood_type}'
    )
    print(f"Output directory created: {output_dir}")
    
    # Use ConfigBuilder for cleaner parameter setup
    ConfigBuilder(config) \
        .set_iterations(10) \
        .set_pedestrian_count(PEDESTRIAN_COUNT) \
        .set_pedestrian_implementation(Constants.PEDESTRIAN_NEIGHBORHOOD) \
        .set_border_implementation(Constants.CORRIDOR_ONLY) \
        .set_volume_neighborhood_type(neighborhood_type) \
        .set_grid_size(GRID_SIZE) \
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
    
    print(f"Experiment completed. Results saved in {output_dir}")


def plot_results():
    """Generate comparison plots for volume neighborhood types."""
    results_dirs = [d for d in os.listdir('experiments/volume_neighborhood_comparison/results') 
                   if os.path.isdir(os.path.join('experiments/volume_neighborhood_comparison/results', d))]
    
    data_by_config = {}
    
    for result_dir in results_dirs:
        if 'cell_size_' not in result_dir or '_neighborhood_' not in result_dir:
            continue
        
        try:
            # Parse directory name
            parts = result_dir.split('_neighborhood_')
            if len(parts) != 2:
                continue
            
            cell_size_str = parts[0].split('cell_size_')[1]
            neighborhood_type = int(parts[1])
            cell_size = float(cell_size_str)
            
            result_path = os.path.join('experiments/volume_neighborhood_comparison/results', result_dir, 'latest')
            
            metrics_file = os.path.join(result_path, 'metrics.csv')
            if os.path.exists(metrics_file):
                df_metrics = pd.read_csv(metrics_file)
                
                if cell_size not in data_by_config:
                    data_by_config[cell_size] = {}
                
                if neighborhood_type not in data_by_config[cell_size]:
                    data_by_config[cell_size][neighborhood_type] = {
                        'time': [],
                        'memory': [],
                        'lanes': [],
                        'collisions': []
                    }
                
                for _, row in df_metrics.iterrows():
                    data_by_config[cell_size][neighborhood_type]['time'].append(float(row['time']))
                    data_by_config[cell_size][neighborhood_type]['memory'].append(float(row['memory_usage']) / 1024)
                    data_by_config[cell_size][neighborhood_type]['lanes'].append(float(row['clustering_based_groups']))
                    data_by_config[cell_size][neighborhood_type]['collisions'].append(float(row['total_collisions']))
        
        except Exception as e:
            print(f"Error processing {result_dir}: {e}")
            continue
    
    if not data_by_config:
        print("No data found to plot!")
        return
    
    cell_sizes_sorted = sorted(data_by_config.keys())
    
    metrics = ['time', 'memory', 'lanes', 'collisions']
    metric_labels = ['Execution Time (s)', 'Memory Usage (MB)', 'Lanes Formed', 'Collisions']
    metric_titles = ['Performance Comparison', 'Memory Usage Comparison', 
                     'Lane Formation Comparison', 'Collision Comparison']
    
    plot_data = {metric: {nt: {'means': [], 'stds': []} for nt in VOLUME_NEIGHBORHOOD_TYPES} 
                 for metric in metrics}
    
    for cell_size in cell_sizes_sorted:
        for metric in metrics:
            for neighborhood_type in VOLUME_NEIGHBORHOOD_TYPES:
                if neighborhood_type in data_by_config[cell_size]:
                    values = data_by_config[cell_size][neighborhood_type][metric]
                    if values:
                        plot_data[metric][neighborhood_type]['means'].append(np.mean(values))
                        plot_data[metric][neighborhood_type]['stds'].append(np.std(values))
                    else:
                        plot_data[metric][neighborhood_type]['means'].append(0)
                        plot_data[metric][neighborhood_type]['stds'].append(0)
                else:
                    plot_data[metric][neighborhood_type]['means'].append(0)
                    plot_data[metric][neighborhood_type]['stds'].append(0)
    
    fig, axes = plt.subplots(2, 2, figsize=(20, 16))
    fig.suptitle('Volume Neighborhood Type Comparison: Face Sharing vs Vertex Sharing', 
                 y=0.995)
    
    colors = {0: DEFAULT_COLORS['face_sharing'], 1: DEFAULT_COLORS['vertex_sharing']}
    for idx, (metric, label, title) in enumerate(zip(metrics, metric_labels, metric_titles)):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        x_pos = np.arange(len(cell_sizes_sorted))
        width = 0.35
        
        for i, neighborhood_type in enumerate(VOLUME_NEIGHBORHOOD_TYPES):
            means = plot_data[metric][neighborhood_type]['means']
            stds = plot_data[metric][neighborhood_type]['stds']
            
            offset = width * (i - 0.5)
            ax.bar(x_pos + offset, means, width, yerr=stds,
                   color=colors[neighborhood_type],
                   label=NEIGHBORHOOD_NAMES[neighborhood_type])
        
        ax.set_xlabel('Cell Size (m)')
        ax.set_ylabel(label)
        ax.set_title(title)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([f'{cs}m' for cs in cell_sizes_sorted], rotation=45, ha='right')
        ax.legend(loc='upper left')
        ax.grid(True, axis='y')
    
    plt.tight_layout()
    plt.savefig('experiments/volume_neighborhood_comparison/volume_neighborhood_performance_comparison.png')
    plt.close()
    
    print("\n" + "="*80)
    print("SUMMARY STATISTICS")
    print("="*80)
    
    for metric, label in zip(metrics, metric_labels):
        print(f"\n{label}:")
        print(f"{'Cell Size':<12} {'Face Sharing':<25} {'Vertex Sharing':<25}")
        print("-" * 80)
        
        for i, cell_size in enumerate(cell_sizes_sorted):
            face_mean = plot_data[metric][0]['means'][i]
            face_std = plot_data[metric][0]['stds'][i]
            vertex_mean = plot_data[metric][1]['means'][i]
            vertex_std = plot_data[metric][1]['stds'][i]
            
            print(f"{cell_size:<12.1f} {face_mean:.2f} ± {face_std:.2f}{'':<12} {vertex_mean:.2f} ± {vertex_std:.2f}")
    
    print("\n" + "="*80)
    print("Visualization saved: volume_neighborhood_performance_comparison.png")
    print("="*80)

    # Generate LaTeX table with exact values
    results_dir = 'experiments/volume_neighborhood_comparison'
    latex_path = os.path.join(results_dir, 'volume_neighborhood_results.tex')
    with open(latex_path, 'w') as f:
        f.write(r'\begin{table}[htbp]' + '\n')
        f.write(r'\centering' + '\n')
        f.write(r'\caption{Comparación Volume Neighborhood: Face Sharing vs Vertex Sharing.}' + '\n')
        f.write(r'\label{tab:volume_neighborhood}' + '\n')
        f.write(r'\begin{tabular}{ccccccccc}' + '\n')
        f.write(r'\hline' + '\n')
        f.write(r'Celda (m) & Tiempo (s) FS & Tiempo (s) VS & Memoria (MB) FS & Memoria (MB) VS & Carriles FS & Carriles VS & Colisiones FS & Colisiones VS \\' + '\n')
        f.write(r'\hline' + '\n')
        for i, cell_size in enumerate(cell_sizes_sorted):
            row_parts = [f'{cell_size}']
            for metric in metrics:
                for nt in VOLUME_NEIGHBORHOOD_TYPES:
                    m = plot_data[metric][nt]['means'][i]
                    s = plot_data[metric][nt]['stds'][i]
                    row_parts.append(f'{m:.2f} $\\pm$ {s:.2f}')
            f.write(' & '.join(row_parts) + r' \\' + '\n')
        f.write(r'\hline' + '\n')
        f.write(r'\end{tabular}' + '\n')
        f.write(r'\end{table}' + '\n')
    print(f"LaTeX table saved to: {latex_path}")


if __name__ == '__main__':
    volume_neighborhood_comparison()
