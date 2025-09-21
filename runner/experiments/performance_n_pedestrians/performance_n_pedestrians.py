import json
import os
import subprocess
import numpy as np
from src.runner import run_experiment, compile_c_code, compile_model
from src.utils import load_config, create_output_dir, copy_results_to_latest
from src.config_manager import config_manager
import matplotlib.pyplot as plt
import pandas as pd
from itertools import product
from src.constants import Constants


execute_experiment = False

# Test configurations
N_PEDESTRIANS = [10, 50, 100, 200, 300, 500, 1000, 2000, 3000, 5000] #, 10000]
TARGET_DENSITY = 0.3  # peatones/m² - densidad constante
CELL_SIZES = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0, 7.5, 10]  # metros por celda - diferentes tamaños de celda

# Calculate grid size and divisions for each N and cell size
def calculate_grid_params(n_pedestrians, density=TARGET_DENSITY, cell_size=1.0):
    """Calculate grid size and divisions for given number of pedestrians and cell size"""
    grid_size = np.sqrt(n_pedestrians / density)
    grid_divisions = max(1, int(grid_size / cell_size))
    return grid_size, grid_divisions

# Generate all combinations for optimization experiments
OPTIMIZATION_COMBINATIONS = []
for n in N_PEDESTRIANS:
    for cell_size in CELL_SIZES:
        grid_size, grid_divisions = calculate_grid_params(n, TARGET_DENSITY, cell_size)
        OPTIMIZATION_COMBINATIONS.append((n, cell_size, grid_size, grid_divisions))

PEDESTRIANS_IMPLEMENTATION = {
    0: "qss",  # QSS solo, sin RETQSS
    1: "retqss",    # RETQSS sin optimizaciones  
    2: "retqss_opt", # RETQSS con optimizaciones de Helbing
}

def performance_n_pedestrians():
    """
    Enhanced performance testing for three model implementations:
    1. QSS solo, sin RETQSS (baseline)
    2. RETQSS sin optimizaciones 
    3. RETQSS con optimizaciones de Helbing (mejor tamaño de grilla)
    """
    print("Running comprehensive performance experiments...")
    print(f"Testing {len(N_PEDESTRIANS)} different N values: {N_PEDESTRIANS}")
    print(f"Target density: {TARGET_DENSITY} peatones/m²")
    print(f"Cell sizes: {CELL_SIZES} metros")
    
    # Calculate grid parameters for each N and cell size
    print(f"Grid parameters for different cell sizes:")
    for cell_size in CELL_SIZES:
        print(f"  Cell size {cell_size}m:")
        for n in N_PEDESTRIANS[:5]:  # Show first 5 N values
            grid_size, grid_divisions = calculate_grid_params(n, TARGET_DENSITY, cell_size)
            print(f"    N={n}: Grid size={grid_size:.1f}m, Divisions={grid_divisions}")
    
    # Calculate total experiments
    qss_experiments = len(N_PEDESTRIANS)  # QSS only varies N
    retqss_experiments = len(N_PEDESTRIANS)  # RETQSS only varies N
    
    # Calculate RETQSS optimization experiments (all N × cell_size combinations)
    retqss_opt_experiments = len(OPTIMIZATION_COMBINATIONS)
    total_experiments = qss_experiments + retqss_experiments + retqss_opt_experiments
    
    print(f"\nTotal experiments: {total_experiments}")
    print(f"  - QSS (solo): {qss_experiments} experiments")
    print(f"  - RETQSS (sin opt): {retqss_experiments} experiments")
    print(f"  - RETQSS (optimizado): {retqss_opt_experiments} experiments")
    print("="*60)
    
    results = []

    # Update configuration from command line arguments
    config_manager.update_from_dict({
        'skip_metrics': True
    })
    
    # Phase 1: Test QSS solo (baseline)
    print("\n1. Testing QSS solo (baseline)...")
    for i, n in enumerate(N_PEDESTRIANS, 1):
        print(f"   [{i}/{qss_experiments}] Running QSS solo with N={n}...")
        result = run_experiment_with_params(n, 0, 1)  # QSS doesn't use grid divisions
        results.append(result)
        print(f"   Completed")
    
    # Phase 2: Test RETQSS without optimizations
    print("\n2. Testing RETQSS without optimizations...")
    for i, n in enumerate(N_PEDESTRIANS, 1):
        print(f"   [{i}/{retqss_experiments}] Running RETQSS (sin opt) with N={n}...")
        result = run_experiment_with_params(n, 1, 1)  # Fixed grid divisions = 1
        results.append(result)
        print(f"   Completed")
    
    # Phase 3: Test RETQSS with optimizations (all N × cell_size combinations)
    print(f"\n3. Testing RETQSS with optimizations (all N × cell_size combinations)...")
    experiment_count = 0
    for n, cell_size, grid_size, grid_divisions in OPTIMIZATION_COMBINATIONS:
        experiment_count += 1
        print(f"   [{experiment_count}/{retqss_opt_experiments}] Running RETQSS (optimizado) with N={n}, cell_size={cell_size}m, M={grid_divisions}...")
        result = run_experiment_with_params(n, 2, grid_divisions, cell_size)
        results.append(result)
        print(f"   Completed")
    
    # Phase 4: Find optimal grid size for each N
    print("\n4. Finding optimal grid size for each N...")
    optimal_configs = find_optimal_grid_sizes(results)
    
    # Phase 5: Generate comprehensive plots
    print("\n5. Generating comprehensive plots...")
    plot_comprehensive_results(results, optimal_configs)
    
    # Phase 6: Generate RETQSS Opt cell sizes comparison
    print("\n6. Generating RETQSS Opt cell sizes comparison...")
    plot_retqss_opt_cell_sizes(results)
    
    # Phase 7: Generate RETQSS Opt best cell sizes comparison (excluding worst performers)
    print("\n7. Generating RETQSS Opt best cell sizes comparison...")
    plot_retqss_opt_best_cell_sizes(results)
    
    print("\n" + "="*60)
    print("All experiments completed successfully!")
    print("Results saved to CSV and plots generated.")

def run_experiment_with_params(n, implementation, grid_divisions, cell_size=1.0):
    """
    Run a single experiment with specified parameters and return timing results.
    """
    config = load_config('./experiments/performance_n_pedestrians/config.json')

    # Calculate grid parameters for this N and cell size
    grid_size, calculated_divisions = calculate_grid_params(n, TARGET_DENSITY, cell_size)
    
    # Use provided grid_divisions for optimization experiments, calculated for others
    if implementation == 2:  # RETQSS with optimizations
        actual_divisions = grid_divisions
        actual_grid_size = actual_divisions * cell_size
    else:
        actual_divisions = calculated_divisions
        actual_grid_size = grid_size

    # Create descriptive experiment name
    impl_name = PEDESTRIANS_IMPLEMENTATION[implementation]
    if impl_name == 'qss':  # QSS solo
        exp_name = f'n_{n}_qss'
        model_name = 'helbing_not_qss'
        pedestrian_implementation = Constants.PEDESTRIAN_MMOC
    elif impl_name == 'retqss':  # RETQSS without optimizations
        exp_name = f'n_{n}_retqss'
        model_name = 'social_force_model'
        pedestrian_implementation = Constants.PEDESTRIAN_MMOC
    elif impl_name == 'retqss_opt':  # RETQSS with optimizations
        exp_name = f'n_{n}_retqss_opt_m_{cell_size}'
        model_name = 'social_force_model'
        pedestrian_implementation = Constants.PEDESTRIAN_NEIGHBORHOOD

    print(f"Exp name: {exp_name}")
    
    # Create output directory
    output_dir = create_output_dir(
        'experiments/performance_n_pedestrians/results', 
        exp_name
    )

    if execute_experiment:
        # Update config parameters
        config['iterations'] = 10
        config['parameters']['N']['value'] = n
        config['parameters']['PEDESTRIAN_IMPLEMENTATION']['value'] = pedestrian_implementation
        config['parameters']['BORDER_IMPLEMENTATION']['value'] = Constants.BORDER_NONE
        config['parameters']['GRID_SIZE']['value'] = actual_grid_size  # Update grid size
        config['parameters']['FROM_Y']['value'] = actual_grid_size * 0.1
        config['parameters']['TO_Y']['value'] = actual_grid_size * 0.9

        # Save config copy in experiment directory
        config_copy_path = os.path.join(output_dir, 'config.json')
        with open(config_copy_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Update model file parameters based on implementation
        if implementation == 0:  # QSS solo
            model_path = '../retqss/model/helbing_not_qss.mo'
            subprocess.run(['sed', '-i', r's/\bN\s*=\s*[0-9]\+/N = ' + str(n) + '/', model_path])
            subprocess.run(['sed', '-i', r's/\bGRID_SIZE\s*=\s*[0-9.]\+/GRID_SIZE = ' + str(actual_grid_size) + '/', model_path])
        else:  # RETQSS implementations
            model_path = '../retqss/model/social_force_model.mo'
            subprocess.run(['sed', '-i', r's/\bN\s*=\s*[0-9]\+/N = ' + str(n) + '/', model_path])
            subprocess.run(['sed', '-i', r's/\bGRID_DIVISIONS\s*=\s*[0-9]\+/GRID_DIVISIONS = ' + str(actual_divisions) + '/', model_path])
            subprocess.run(['sed', '-i', r's/\bGRID_SIZE\s*=\s*[0-9.]\+/GRID_SIZE = ' + str(actual_grid_size) + '/', model_path])


    if execute_experiment:
        # Compile the C++ code and model
        compile_c_code()
        compile_model(model_name)

        # Run experiment
        run_experiment(
            config, 
            output_dir, 
            model_name, 
            plot=False, 
            copy_results=False
        )

        # Copy results from output directory to latest directory
        copy_results_to_latest(output_dir)

    # Read timing results from metrics.csv
    metrics_file = os.path.join( 
        'experiments/performance_n_pedestrians/results', 
        exp_name, 
        'latest',
        'metrics.csv'
    )

    # Read detailed metrics if available
    detailed_metrics = None
    if os.path.exists(metrics_file):
        try:
            metrics_df = pd.read_csv(metrics_file)
            if not metrics_df.empty and 'time' in metrics_df.columns:
                detailed_metrics = {
                    'total_iterations': len(metrics_df),
                    'avg_iteration_time': metrics_df['time'].mean(),
                    'min_iteration_time': metrics_df['time'].min(),
                    'max_iteration_time': metrics_df['time'].max(),
                    'std_iteration_time': metrics_df['time'].std(),
                    'avg_memory_usage': metrics_df['memory_usage'].mean() if 'memory_usage' in metrics_df.columns else None,
                }
        except (pd.errors.EmptyDataError, KeyError):
            detailed_metrics = None
    
    return {
        'n_pedestrians': n,
        'implementation': impl_name,
        'grid_divisions': actual_divisions,
        'grid_size': actual_grid_size,
        'cell_size': cell_size,
        'density': n / (actual_grid_size ** 2),
        'output_dir': output_dir,
        'detailed_metrics': detailed_metrics
    }

def find_optimal_grid_sizes(results):
    """
    Find the optimal cell size and grid configuration for each number of pedestrians based on RETQSS optimized results.
    """
    df = pd.DataFrame(results)
    
    # Filter only RETQSS optimized results
    retqss_opt_data = df[df['implementation'] == 'retqss_opt'].copy()
    
    if retqss_opt_data.empty:
        print("Warning: No RETQSS optimized results found for optimization analysis")
        return {}
    
    # Add average time column for easier analysis
    retqss_opt_data['avg_time'] = retqss_opt_data['detailed_metrics'].apply(
        lambda x: x['avg_iteration_time'] if x else float('inf')
    )
    
    # Find optimal cell size for each N (minimum average time)
    optimal_configs = {}
    for n in N_PEDESTRIANS:
        n_data = retqss_opt_data[retqss_opt_data['n_pedestrians'] == n]
        if not n_data.empty:
            best_idx = n_data['avg_time'].idxmin()
            best_config = n_data.loc[best_idx]
            optimal_configs[n] = {
                'cell_size': best_config['cell_size'],
                'grid_divisions': best_config['grid_divisions'],
                'grid_size': best_config['grid_size'],
                'avg_time': best_config['avg_time'],
                'std_time': best_config['detailed_metrics']['std_iteration_time'] if best_config['detailed_metrics'] else 0
            }
            print(f"  N={n}: Optimal cell_size={best_config['cell_size']}m, M={best_config['grid_divisions']} (avg_time={best_config['avg_time']:.4f}s)")
    
    return optimal_configs

def plot_comprehensive_results(results, optimal_configs=None):
    """
    Generate comprehensive plots comparing three model implementations:
    1. QSS solo, sin RETQSS (baseline)
    2. RETQSS sin optimizaciones
    3. RETQSS con optimizaciones de Helbing (mejor tamaño de grilla)
    """
    df = pd.DataFrame(results)
    
    # Create results directory if it doesn't exist
    results_dir = 'experiments/performance_n_pedestrians/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Filter data by implementation
    qss_data = df[df['implementation'] == 'qss'].sort_values('n_pedestrians')
    retqss_data = df[df['implementation'] == 'retqss'].sort_values('n_pedestrians')
    retqss_opt_data = df[df['implementation'] == 'retqss_opt']
    
    # Create results directory if it doesn't exist
    results_dir = 'experiments/performance_n_pedestrians/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Plot 1: Main comparison - All three implementations
    plt.figure(figsize=(12, 8))
    
    # Plot QSS baseline
    qss_times = qss_data['detailed_metrics'].apply(lambda x: x['avg_iteration_time'] if x else 0)
    qss_stds = qss_data['detailed_metrics'].apply(lambda x: x['std_iteration_time'] if x else 0)
    plt.errorbar(qss_data['n_pedestrians'], qss_times, yerr=qss_stds,
                fmt='o-', label='QSS', linewidth=4, markersize=10, color='skyblue', 
                capsize=6, capthick=3, elinewidth=3, alpha=0.8, markeredgecolor='navy', markeredgewidth=2)
    
    # Plot RETQSS without optimizations
    retqss_times = retqss_data['detailed_metrics'].apply(lambda x: x['avg_iteration_time'] if x else 0)
    retqss_stds = retqss_data['detailed_metrics'].apply(lambda x: x['std_iteration_time'] if x else 0)
    plt.errorbar(retqss_data['n_pedestrians'], retqss_times, yerr=retqss_stds,
                fmt='s-', label='RETQSS', linewidth=4, markersize=10, color='lightgreen', 
                capsize=6, capthick=3, elinewidth=3, alpha=0.8, markeredgecolor='darkgreen', markeredgewidth=2)
    
    # Plot RETQSS optimized (best configuration for each N)
    if optimal_configs:
        retqss_opt_best = []
        for n in N_PEDESTRIANS:
            if n in optimal_configs:
                n_data = retqss_opt_data[retqss_opt_data['n_pedestrians'] == n]
                n_data = n_data[n_data['grid_divisions'] == optimal_configs[n]['grid_divisions']]
                if not n_data.empty:
                    retqss_opt_best.append(n_data.iloc[0])
        
        if retqss_opt_best:
            retqss_opt_best_df = pd.DataFrame(retqss_opt_best).sort_values('n_pedestrians')
            retqss_opt_times = retqss_opt_best_df['detailed_metrics'].apply(lambda x: x['avg_iteration_time'] if x else 0)
            retqss_opt_stds = retqss_opt_best_df['detailed_metrics'].apply(lambda x: x['std_iteration_time'] if x else 0)
            plt.errorbar(retqss_opt_best_df['n_pedestrians'], retqss_opt_times, yerr=retqss_opt_stds,
                        fmt='^-', label='RETQSS Opt', linewidth=4, markersize=10, color='lightcoral', 
                        capsize=6, capthick=3, elinewidth=3, alpha=0.8, markeredgecolor='darkred', markeredgewidth=2)
    
    plt.xlabel('Número de Peatones (N)', fontsize=14)
    plt.ylabel('Tiempo Promedio de Ejecución (s)', fontsize=14)
    plt.title(f'Comparación de Rendimiento: Densidad Constante ({TARGET_DENSITY} peatones/m²)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)
    # plt.yscale('log')  # Changed to linear scale
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '01_performance_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 2: Speedup comparison - RETQSS vs QSS
    plt.figure(figsize=(12, 8))
    
    # Calculate speedup for RETQSS vs QSS
    merged_retqss = pd.merge(qss_data, retqss_data, on='n_pedestrians', suffixes=('_qss', '_retqss'))
    qss_times_merged = merged_retqss['detailed_metrics_qss'].apply(lambda x: x['avg_iteration_time'] if x else 0)
    retqss_times_merged = merged_retqss['detailed_metrics_retqss'].apply(lambda x: x['avg_iteration_time'] if x else 0)
    merged_retqss['speedup'] = qss_times_merged / retqss_times_merged
    
    plt.plot(merged_retqss['n_pedestrians'], merged_retqss['speedup'], 
            's-', label='RETQSS vs QSS', linewidth=4, markersize=10, color='lightgreen', alpha=0.8, markeredgecolor='darkgreen', markeredgewidth=2)
    
    # Calculate speedup for RETQSS optimized vs QSS
    if optimal_configs:
        retqss_opt_best = []
        for n in N_PEDESTRIANS:
            if n in optimal_configs:
                n_data = retqss_opt_data[retqss_opt_data['n_pedestrians'] == n]
                n_data = n_data[n_data['grid_divisions'] == optimal_configs[n]['grid_divisions']]
                if not n_data.empty:
                    retqss_opt_best.append(n_data.iloc[0])
        
        if retqss_opt_best:
            retqss_opt_best_df = pd.DataFrame(retqss_opt_best).sort_values('n_pedestrians')
            merged_opt = pd.merge(qss_data, retqss_opt_best_df, on='n_pedestrians', suffixes=('_qss', '_retqss_opt'))
            qss_times_opt = merged_opt['detailed_metrics_qss'].apply(lambda x: x['avg_iteration_time'] if x else 0)
            retqss_opt_times_opt = merged_opt['detailed_metrics_retqss_opt'].apply(lambda x: x['avg_iteration_time'] if x else 0)
            merged_opt['speedup'] = qss_times_opt / retqss_opt_times_opt
            
            plt.plot(merged_opt['n_pedestrians'], merged_opt['speedup'], 
                    '^-', label='RETQSS Opt vs QSS', linewidth=4, markersize=10, color='lightcoral', alpha=0.8, markeredgecolor='darkred', markeredgewidth=2)
    
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Rendimiento Igual', linewidth=2)
    plt.xlabel('Número de Peatones (N)', fontsize=14)
    plt.ylabel('Aceleración (Tiempo QSS / Tiempo RETQSS)', fontsize=14)
    plt.title('Aceleración: RETQSS vs QSS', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)
    # plt.yscale('log')  # Log scale for speedup visualization
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '02_speedup_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 3: Optimization impact - RETQSS optimized vs RETQSS no opt
    plt.figure(figsize=(12, 8))
    
    if optimal_configs:
        retqss_opt_best = []
        for n in N_PEDESTRIANS:
            if n in optimal_configs:
                n_data = retqss_opt_data[retqss_opt_data['n_pedestrians'] == n]
                n_data = n_data[n_data['grid_divisions'] == optimal_configs[n]['grid_divisions']]
                if not n_data.empty:
                    retqss_opt_best.append(n_data.iloc[0])
        
        if retqss_opt_best:
            retqss_opt_best_df = pd.DataFrame(retqss_opt_best).sort_values('n_pedestrians')
            merged_opt_vs_no_opt = pd.merge(retqss_data, retqss_opt_best_df, on='n_pedestrians', suffixes=('_retqss', '_opt'))
            retqss_times_comp = merged_opt_vs_no_opt['detailed_metrics_retqss'].apply(lambda x: x['avg_iteration_time'] if x else 0)
            retqss_opt_times_comp = merged_opt_vs_no_opt['detailed_metrics_opt'].apply(lambda x: x['avg_iteration_time'] if x else 0)
            merged_opt_vs_no_opt['speedup'] = retqss_times_comp / retqss_opt_times_comp
            
            plt.plot(merged_opt_vs_no_opt['n_pedestrians'], merged_opt_vs_no_opt['speedup'], 
                    'o-', label='Impacto de Optimización', linewidth=4, markersize=10, color='green', alpha=0.8)
    
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Sin Mejora', linewidth=2)
    plt.xlabel('Número de Peatones (N)', fontsize=14)
    plt.ylabel('Mejora (Tiempo RETQSS Opt / Tiempo RETQSS)', fontsize=14)
    plt.title('Impacto de Optimización: RETQSS Opt vs RETQSS', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '03_optimization_impact.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 4: Optimal cell size for each N
    plt.figure(figsize=(12, 8))
    
    if optimal_configs:
        n_values = list(optimal_configs.keys())
        cell_sizes = [optimal_configs[n]['cell_size'] for n in n_values]
        plt.plot(n_values, cell_sizes, 'o-', color='purple', linewidth=4, markersize=10, alpha=0.8)
        plt.xlabel('Número de Peatones (N)', fontsize=14)
        plt.ylabel('Tamaño Óptimo de Celda (m)', fontsize=14)
        plt.title('Tamaño Óptimo de Celda para Cada N', fontsize=16)
        plt.grid(True, alpha=0.3)
        plt.tick_params(axis='both', which='major', labelsize=12)
    else:
        plt.text(0.5, 0.5, 'No optimization data available', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
        plt.title('Tamaño Óptimo de Celda para Cada N', fontsize=16)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '04_optimal_cell_size.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 5: Performance vs Cell Size (for selected N values)
    plt.figure(figsize=(12, 8))
    
    selected_n_values = [500, 1000, 2000, 5000]  # Show a few representative N values
    # Use deltaq-style color palette
    colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsteelblue']
    
    for i, n in enumerate(selected_n_values):
        n_data = retqss_opt_data[retqss_opt_data['n_pedestrians'] == n].sort_values('cell_size')
        if not n_data.empty:
            times = n_data['detailed_metrics'].apply(lambda x: x['avg_iteration_time'] if x else 0)
            stds = n_data['detailed_metrics'].apply(lambda x: x['std_iteration_time'] if x else 0)
            plt.errorbar(n_data['cell_size'], times, yerr=stds,
                        fmt='o-', label=f'N={n}', linewidth=4, markersize=10, 
                        color=colors[i], capsize=6, capthick=3, elinewidth=3, alpha=0.8)
    
    plt.xlabel('Tamaño de Celda (m)', fontsize=14)
    plt.ylabel('Tiempo Promedio de Ejecución (s)', fontsize=14)
    plt.title('Rendimiento vs Tamaño de Celda', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '05_performance_vs_cell_size.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 6: Density vs Performance for different cell sizes
    plt.figure(figsize=(12, 8))
    
    # Show performance vs density for different cell sizes
    # Use deltaq-style color palette
    cell_size_colors = ['skyblue', 'lightgreen', 'lightcoral', 'lightsteelblue', 'lightpink', 'lightyellow', 'lightcyan', 'lightgray']
    
    for i, cell_size in enumerate(CELL_SIZES):
        cell_data = retqss_opt_data[retqss_opt_data['cell_size'] == cell_size]
        if not cell_data.empty:
            times = cell_data['detailed_metrics'].apply(lambda x: x['avg_iteration_time'] if x else 0)
            densities = cell_data['density']
            plt.scatter(densities, times, 
                       label=f'Cell size {cell_size}m', 
                       color=cell_size_colors[i], 
                       s=100, alpha=0.8, edgecolors='black', linewidth=1)
    
    plt.xlabel('Densidad (peatones/m²)', fontsize=14)
    plt.ylabel('Tiempo Promedio de Ejecución (s)', fontsize=14)
    plt.title('Densidad vs Rendimiento por Tamaño de Celda', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '06_density_vs_performance.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Plot 7: QSS vs RETQSS Performance and Memory Comparison
    plt.figure(figsize=(16, 8))
    
    # Subplot 1: Performance comparison
    plt.subplot(1, 2, 1)
    
    # Plot QSS performance
    qss_times = qss_data['detailed_metrics'].apply(lambda x: x['avg_iteration_time'] if x else 0)
    qss_stds = qss_data['detailed_metrics'].apply(lambda x: x['std_iteration_time'] if x else 0)
    plt.errorbar(qss_data['n_pedestrians'], qss_times, yerr=qss_stds,
                fmt='o-', label='QSS', linewidth=4, markersize=10, color='skyblue', 
                capsize=6, capthick=3, elinewidth=3, alpha=0.8, markeredgecolor='navy', markeredgewidth=2)
    
    # Plot RETQSS performance
    retqss_times = retqss_data['detailed_metrics'].apply(lambda x: x['avg_iteration_time'] if x else 0)
    retqss_stds = retqss_data['detailed_metrics'].apply(lambda x: x['std_iteration_time'] if x else 0)
    plt.errorbar(retqss_data['n_pedestrians'], retqss_times, yerr=retqss_stds,
                fmt='s-', label='RETQSS', linewidth=4, markersize=10, color='lightgreen', 
                capsize=6, capthick=3, elinewidth=3, alpha=0.8, markeredgecolor='darkgreen', markeredgewidth=2)
    
    plt.xlabel('Número de Peatones (N)', fontsize=14)
    plt.ylabel('Tiempo Promedio de Ejecución (s)', fontsize=14)
    plt.title('Comparación de Rendimiento: QSS vs RETQSS', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # Subplot 2: Memory usage comparison
    plt.subplot(1, 2, 2)
    
    # Plot QSS memory usage
    qss_memory = qss_data['detailed_metrics'].apply(lambda x: x['avg_memory_usage'] if x and x['avg_memory_usage'] else 0)
    qss_memory_stds = qss_data['detailed_metrics'].apply(lambda x: x['std_iteration_time'] if x else 0)  # Using std_iteration_time as proxy for memory std
    plt.errorbar(qss_data['n_pedestrians'], qss_memory, yerr=qss_memory_stds,
                fmt='o-', label='QSS', linewidth=4, markersize=10, color='skyblue', 
                capsize=6, capthick=3, elinewidth=3, alpha=0.8, markeredgecolor='navy', markeredgewidth=2)
    
    # Plot RETQSS memory usage
    retqss_memory = retqss_data['detailed_metrics'].apply(lambda x: x['avg_memory_usage'] if x and x['avg_memory_usage'] else 0)
    retqss_memory_stds = retqss_data['detailed_metrics'].apply(lambda x: x['std_iteration_time'] if x else 0)  # Using std_iteration_time as proxy for memory std
    plt.errorbar(retqss_data['n_pedestrians'], retqss_memory, yerr=retqss_memory_stds,
                fmt='s-', label='RETQSS', linewidth=4, markersize=10, color='lightgreen', 
                capsize=6, capthick=3, elinewidth=3, alpha=0.8, markeredgecolor='darkgreen', markeredgewidth=2)
    
    plt.xlabel('Número de Peatones (N)', fontsize=14)
    plt.ylabel('Uso Promedio de Memoria (MB)', fontsize=14)
    plt.title('Comparación de Uso de Memoria: QSS vs RETQSS', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '07_qss_vs_retqss_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_retqss_opt_cell_sizes(results):
    """
    Generate a focused plot comparing RETQSS Opt performance across all cell sizes.
    This plot shows how different cell sizes affect RETQSS Opt performance for different N values.
    """
    df = pd.DataFrame(results)
    
    # Create results directory if it doesn't exist
    results_dir = 'experiments/performance_n_pedestrians/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Filter only RETQSS optimized results
    retqss_opt_data = df[df['implementation'] == 'retqss_opt'].copy()
    
    if retqss_opt_data.empty:
        print("Warning: No RETQSS optimized results found for cell size comparison")
        return
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Define colors for different cell sizes
    cell_size_colors = {
        0.5: 'red',
        1.0: 'blue', 
        2.0: 'green',
        3.0: 'orange',
        4.0: 'purple',
        5.0: 'brown',
        7.5: 'pink',
        10.0: 'gray'
    }
    
    # Plot each cell size as a separate line
    for cell_size in sorted(CELL_SIZES):
        cell_data = retqss_opt_data[retqss_opt_data['cell_size'] == cell_size].sort_values('n_pedestrians')
        
        if not cell_data.empty:
            times = cell_data['detailed_metrics'].apply(lambda x: x['avg_iteration_time'] if x else 0)
            stds = cell_data['detailed_metrics'].apply(lambda x: x['std_iteration_time'] if x else 0)
            
            plt.errorbar(cell_data['n_pedestrians'], times, yerr=stds,
                        fmt='o-', label=f'Cell size {cell_size}m', 
                        linewidth=3, markersize=8, 
                        color=cell_size_colors[cell_size], 
                        capsize=4, capthick=2, elinewidth=2, 
                        alpha=0.8, markeredgecolor='black', markeredgewidth=1)
    
    plt.xlabel('Número de Peatones (N)', fontsize=16)
    plt.ylabel('Tiempo Promedio de Ejecución (s)', fontsize=16)
    plt.title('RETQSS Opt: Rendimiento vs Número de Peatones por Tamaño de Celda', fontsize=18)
    plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(0.02, 0.98))
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tick_params(axis='both', which='major', labelsize=14)
    
    # Add some styling
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '08_retqss_opt_cell_sizes_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a second plot showing the same data but with linear scale for better visibility of trends
    plt.figure(figsize=(14, 10))
    
    for cell_size in sorted(CELL_SIZES):
        cell_data = retqss_opt_data[retqss_opt_data['cell_size'] == cell_size].sort_values('n_pedestrians')
        
        if not cell_data.empty:
            times = cell_data['detailed_metrics'].apply(lambda x: x['avg_iteration_time'] if x else 0)
            stds = cell_data['detailed_metrics'].apply(lambda x: x['std_iteration_time'] if x else 0)
            
            plt.errorbar(cell_data['n_pedestrians'], times, yerr=stds,
                        fmt='o-', label=f'Cell size {cell_size}m', 
                        linewidth=3, markersize=8, 
                        color=cell_size_colors[cell_size], 
                        capsize=4, capthick=2, elinewidth=2, 
                        alpha=0.8, markeredgecolor='black', markeredgewidth=1)
    
    plt.xlabel('Número de Peatones (N)', fontsize=16)
    plt.ylabel('Tiempo Promedio de Ejecución (s)', fontsize=16)
    plt.title('RETQSS Opt: Rendimiento vs Número de Peatones por Tamaño de Celda (Escala Lineal)', fontsize=18)
    plt.legend(fontsize=12, loc='upper left', bbox_to_anchor=(0.02, 0.98))
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=14)
    
    # Add some styling
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '09_retqss_opt_cell_sizes_linear.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_retqss_opt_best_cell_sizes(results):
    """
    Generate a focused plot comparing RETQSS Opt performance for the best cell sizes only.
    Excludes the two worst performing cell sizes (10m and 7.5m) for better visualization.
    """
    df = pd.DataFrame(results)
    
    # Create results directory if it doesn't exist
    results_dir = 'experiments/performance_n_pedestrians/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Filter only RETQSS optimized results
    retqss_opt_data = df[df['implementation'] == 'retqss_opt'].copy()
    
    if retqss_opt_data.empty:
        print("Warning: No RETQSS optimized results found for cell size comparison")
        return
    
    # Define the best cell sizes (excluding 10m and 7.5m)
    best_cell_sizes = [0.5, 1.0, 2.0, 3.0, 4.0, 5.0]
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    
    # Define colors for the best cell sizes
    cell_size_colors = {
        0.5: 'red',
        1.0: 'blue', 
        2.0: 'green',
        3.0: 'orange',
        4.0: 'purple',
        5.0: 'brown'
    }
    
    # Plot each best cell size as a separate line
    for cell_size in sorted(best_cell_sizes):
        cell_data = retqss_opt_data[retqss_opt_data['cell_size'] == cell_size].sort_values('n_pedestrians')
        
        if not cell_data.empty:
            times = cell_data['detailed_metrics'].apply(lambda x: x['avg_iteration_time'] if x else 0)
            stds = cell_data['detailed_metrics'].apply(lambda x: x['std_iteration_time'] if x else 0)
            
            plt.errorbar(cell_data['n_pedestrians'], times, yerr=stds,
                        fmt='o-', label=f'Cell size {cell_size}m', 
                        linewidth=4, markersize=10, 
                        color=cell_size_colors[cell_size], 
                        capsize=5, capthick=3, elinewidth=3, 
                        alpha=0.8, markeredgecolor='black', markeredgewidth=2)
    
    plt.xlabel('Número de Peatones (N)', fontsize=16)
    plt.ylabel('Tiempo Promedio de Ejecución (s)', fontsize=16)
    plt.title('RETQSS Opt: Mejores Tamaños de Celda (Excluyendo 7.5m y 10m)', fontsize=18)
    plt.legend(fontsize=14, loc='upper left', bbox_to_anchor=(0.02, 0.98))
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tick_params(axis='both', which='major', labelsize=14)
    
    # Add some styling
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '10_retqss_opt_best_cell_sizes.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a second plot with linear scale for better trend visibility
    plt.figure(figsize=(14, 10))
    
    for cell_size in sorted(best_cell_sizes):
        cell_data = retqss_opt_data[retqss_opt_data['cell_size'] == cell_size].sort_values('n_pedestrians')
        
        if not cell_data.empty:
            times = cell_data['detailed_metrics'].apply(lambda x: x['avg_iteration_time'] if x else 0)
            stds = cell_data['detailed_metrics'].apply(lambda x: x['std_iteration_time'] if x else 0)
            
            plt.errorbar(cell_data['n_pedestrians'], times, yerr=stds,
                        fmt='o-', label=f'Cell size {cell_size}m', 
                        linewidth=4, markersize=10, 
                        color=cell_size_colors[cell_size], 
                        capsize=5, capthick=3, elinewidth=3, 
                        alpha=0.8, markeredgecolor='black', markeredgewidth=2)
    
    plt.xlabel('Número de Peatones (N)', fontsize=16)
    plt.ylabel('Tiempo Promedio de Ejecución (s)', fontsize=16)
    plt.title('RETQSS Opt: Mejores Tamaños de Celda - Escala Lineal (Excluyendo 7.5m y 10m)', fontsize=18)
    plt.legend(fontsize=14, loc='upper left', bbox_to_anchor=(0.02, 0.98))
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=14)
    
    # Add some styling
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '11_retqss_opt_best_cell_sizes_linear.png'), dpi=300, bbox_inches='tight')
    plt.show()

if __name__ == '__main__':
    performance_n_pedestrians()
