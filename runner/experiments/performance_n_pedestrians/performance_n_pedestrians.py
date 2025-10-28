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


execute_experiment = True

# Test configurations
N_PEDESTRIANS = [300, 500, 1000, 2000, 3000, 5000] #, 10000]
TARGET_DENSITY = 0.3  # peatones/m² - densidad constante
CELL_SIZES = [5.0]  # metros por celda - diferentes tamaños de celda

# Calculate grid size and divisions for each N and cell size
def calculate_grid_params(n_pedestrians, density=TARGET_DENSITY, cell_size=1.0):
    """Calculate grid size and divisions for given number of pedestrians and cell size"""    
    grid_size = np.sqrt(n_pedestrians / density)
    from_y = grid_size * 0.2
    to_y = grid_size * 0.8
    grid_divisions = max(1, int(grid_size / cell_size))
    return grid_size, grid_divisions, from_y, to_y

# Generate all combinations for optimization experiments
OPTIMIZATION_COMBINATIONS = []
for n in N_PEDESTRIANS:
    for cell_size in CELL_SIZES:
        grid_size, grid_divisions, from_y, to_y = calculate_grid_params(n, TARGET_DENSITY, cell_size)
        OPTIMIZATION_COMBINATIONS.append((n, cell_size, grid_size, grid_divisions, from_y, to_y))

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
            grid_size, grid_divisions, from_y, to_y = calculate_grid_params(n, TARGET_DENSITY, cell_size)
            print(f"    N={n}: Grid size={grid_size:.1f}m, Divisions={grid_divisions}, From y={from_y:.1f}m, To y={to_y:.1f}m")
    
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

    # # Update configuration from command line arguments
    # config_manager.update_from_dict({
    #     'skip_metrics': True
    # })
    
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
    for n, cell_size, grid_size, grid_divisions, from_y, to_y in OPTIMIZATION_COMBINATIONS:
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
    
    # Phase 6: Generate single bar chart comparison (replaces plots 2-6)
    print("\n6. Generating single bar chart comparison...")
    plot_performance_bar_chart(results, optimal_configs)
    
    # Phase 7: Generate RETQSS Opt cell sizes comparison
    print("\n7. Generating RETQSS Opt cell sizes comparison...")
    plot_retqss_opt_cell_sizes(results)
    
    # Phase 8: Generate RETQSS Opt best cell sizes comparison (excluding worst performers)
    print("\n8. Generating RETQSS Opt best cell sizes comparison...")
    plot_retqss_opt_best_cell_sizes(results)
    
    # Phase 9: Generate QSS vs RETQSS bar chart comparison
    print("\n9. Generating QSS vs RETQSS bar chart comparison...")
    plot_qss_vs_retqss_bar_chart(results)
    
    # Phase 10: Generate memory usage comparison
    print("\n10. Generating memory usage comparison...")
    plot_memory_usage_comparison(results, optimal_configs)

    # Phase 11: Generate breaking lanes comparison
    print("\n11. Generating breaking lanes comparison...")
    plot_enhanced_cell_size_comparison(results)
    
    print("\n" + "="*60)
    print("All experiments completed successfully!")
    print("Results saved to CSV and plots generated.")

def run_experiment_with_params(n, implementation, grid_divisions, cell_size=1.0):
    """
    Run a single experiment with specified parameters and return timing results.
    """
    config = load_config('./experiments/performance_n_pedestrians/config.json')

    # Calculate grid parameters for this N and cell size
    grid_size, calculated_divisions, from_y, to_y = calculate_grid_params(n, TARGET_DENSITY, cell_size)
    
    # Use provided grid_divisions for optimization experiments, calculated for others
    if implementation == 2:  # RETQSS with optimizations
        actual_divisions = grid_divisions
        actual_grid_size = actual_divisions * cell_size
    else:
        actual_divisions = 1
        actual_grid_size = grid_size

    # Create descriptive experiment name
    impl_name = PEDESTRIANS_IMPLEMENTATION[implementation]
    if impl_name == 'qss':  # QSS solo
        exp_name = f'n_{n}_qss'
        model_name = 'helbing_only_qss'
        pedestrian_implementation = Constants.PEDESTRIAN_MMOC
    elif impl_name == 'retqss':  # RETQSS without optimizations
        exp_name = f'n_{n}_retqss'
        model_name = 'social_force_model_naive'
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
        config['parameters']['BORDER_IMPLEMENTATION']['value'] = Constants.CORRIDOR_ONLY
        config['parameters']['GRID_SIZE']['value'] = actual_grid_size  # Update grid size
        config['parameters']['FROM_Y']['value'] = from_y
        config['parameters']['TO_Y']['value'] = to_y

        # Save config copy in experiment directory
        config_copy_path = os.path.join(output_dir, 'config.json')
        with open(config_copy_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Update model file parameters based on implementation
        if implementation == 0:  # QSS solo
            model_path = '../retqss/model/helbing_only_qss.mo'
            subprocess.run(['sed', '-i', r's/\bN\s*=\s*[0-9]\+/N = ' + str(n) + '/', model_path])
            subprocess.run(['sed', '-i', r's/\bGRID_SIZE\s*=\s*[0-9.]\+/GRID_SIZE = ' + str(actual_grid_size) + '/', model_path])
        elif implementation == 1:  # RETQSS without optimizations
            model_path = '../retqss/model/social_force_model_naive.mo'
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
                    'avg_memory_usage': (metrics_df['memory_usage'].mean() / 1024) if 'memory_usage' in metrics_df.columns else None,  # Convert KB to MB
                    'std_memory_usage': (metrics_df['memory_usage'].std() / 1024) if 'memory_usage' in metrics_df.columns else None,  # Convert KB to MB
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
    plt.suptitle('Comparación de Rendimiento: QSS vs RETQSS vs RETQSS Opt', fontsize=16, fontweight='bold')
    
    # Plot QSS baseline
    qss_times = qss_data['detailed_metrics'].apply(lambda x: x['avg_iteration_time'] if x else 0)
    qss_stds = qss_data['detailed_metrics'].apply(lambda x: x['std_iteration_time'] if x else 0)
    plt.errorbar(qss_data['n_pedestrians'], qss_times, yerr=qss_stds,
                fmt='o-', label='QSS', linewidth=4, markersize=10, color='#FF6B6B', 
                capsize=6, capthick=3, elinewidth=3, alpha=0.8, markeredgecolor='#D63031', markeredgewidth=2)
    
    # Plot RETQSS without optimizations
    retqss_times = retqss_data['detailed_metrics'].apply(lambda x: x['avg_iteration_time'] if x else 0)
    retqss_stds = retqss_data['detailed_metrics'].apply(lambda x: x['std_iteration_time'] if x else 0)
    plt.errorbar(retqss_data['n_pedestrians'], retqss_times, yerr=retqss_stds,
                fmt='s-', label='RETQSS', linewidth=4, markersize=10, color='#4ECDC4', 
                capsize=6, capthick=3, elinewidth=3, alpha=0.8, markeredgecolor='#00B894', markeredgewidth=2)
    
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
                        fmt='^-', label='RETQSS Opt', linewidth=4, markersize=10, color='#6C5CE7', 
                        capsize=6, capthick=3, elinewidth=3, alpha=0.8, markeredgecolor='#5A4FCF', markeredgewidth=2)
    
    plt.xlabel('Número de Peatones (N)', fontsize=14)
    plt.ylabel('Tiempo Promedio de Ejecución (s)', fontsize=14)
    plt.title(f'Comparación de Rendimiento: Densidad Constante ({TARGET_DENSITY} peatones/m²)', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)
    # plt.yscale('log')  # Changed to linear scale
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '01_performance_comparison.png'), dpi=300, bbox_inches='tight')

    plt.yscale('log')  # Changed to linear scale
    plt.savefig(os.path.join(results_dir, '01_performance_comparison_log.png'), dpi=300, bbox_inches='tight')
    
    # Plot 2: QSS vs RETQSS Performance and Memory Comparison
    plt.figure(figsize=(16, 8))
    plt.suptitle('Comparación Completa: QSS vs RETQSS (Rendimiento y Memoria)', fontsize=16, fontweight='bold')
    
    # Subplot 1: Performance comparison
    plt.subplot(1, 2, 1)
    
    # Plot QSS performance
    qss_times = qss_data['detailed_metrics'].apply(lambda x: x['avg_iteration_time'] if x else 0)
    qss_stds = qss_data['detailed_metrics'].apply(lambda x: x['std_iteration_time'] if x else 0)
    plt.errorbar(qss_data['n_pedestrians'], qss_times, yerr=qss_stds,
                fmt='o-', label='QSS', linewidth=4, markersize=10, color='red', 
                capsize=6, capthick=3, elinewidth=3, alpha=0.8, markeredgecolor='darkgreen', markeredgewidth=2)
    
    # Plot RETQSS performance
    retqss_times = retqss_data['detailed_metrics'].apply(lambda x: x['avg_iteration_time'] if x else 0)
    retqss_stds = retqss_data['detailed_metrics'].apply(lambda x: x['std_iteration_time'] if x else 0)
    plt.errorbar(retqss_data['n_pedestrians'], retqss_times, yerr=retqss_stds,
                fmt='s-', label='RETQSS', linewidth=4, markersize=10, color='orange', 
                capsize=6, capthick=3, elinewidth=3, alpha=0.8, markeredgecolor='darkorange', markeredgewidth=2)
    
    plt.xlabel('Número de Peatones (N)', fontsize=14)
    plt.ylabel('Tiempo Promedio de Ejecución (s)', fontsize=14)
    plt.title('Comparación de Rendimiento: QSS vs RETQSS', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # Subplot 2: Memory usage comparison
    plt.subplot(1, 2, 2)
    
    # Plot QSS memory usage (already converted to MB in metrics)
    qss_memory = qss_data['detailed_metrics'].apply(lambda x: x['avg_memory_usage'] if x and x['avg_memory_usage'] else 0)
    qss_memory_stds = qss_data['detailed_metrics'].apply(lambda x: x['std_memory_usage'] if x and x['std_memory_usage'] else 0)  # Using actual memory std
    plt.errorbar(qss_data['n_pedestrians'], qss_memory, yerr=qss_memory_stds,
                fmt='o-', label='QSS', linewidth=4, markersize=10, color='red', 
                capsize=6, capthick=3, elinewidth=3, alpha=0.8, markeredgecolor='darkgreen', markeredgewidth=2)
    
    # Plot RETQSS memory usage (already converted to MB in metrics)
    retqss_memory = retqss_data['detailed_metrics'].apply(lambda x: x['avg_memory_usage'] if x and x['avg_memory_usage'] else 0)
    retqss_memory_stds = retqss_data['detailed_metrics'].apply(lambda x: x['std_memory_usage'] if x and x['std_memory_usage'] else 0)  # Using actual memory std
    plt.errorbar(retqss_data['n_pedestrians'], retqss_memory, yerr=retqss_memory_stds,
                fmt='s-', label='RETQSS', linewidth=4, markersize=10, color='orange', 
                capsize=6, capthick=3, elinewidth=3, alpha=0.8, markeredgecolor='darkorange', markeredgewidth=2)
    
    plt.xlabel('Número de Peatones (N)', fontsize=14)
    plt.ylabel('Uso Promedio de Memoria (MB)', fontsize=14)
    plt.title('Comparación de Uso de Memoria: QSS vs RETQSS', fontsize=16)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '03_qss_vs_retqss_comparison.png'), dpi=300, bbox_inches='tight')
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
    plt.suptitle('RETQSS Opt: Rendimiento vs Número de Peatones por Tamaño de Celda', fontsize=16, fontweight='bold')
    
    # Define colors for different cell sizes
    cell_size_colors = {
        0.5: 'red',
        1.0: 'blue', 
        2.0: 'green',
        3.0: 'orange',
        4.0: 'purple',
        5.0: 'brown',
        7.5: 'pink',
        10.0: 'gray',
        12.5: 'cyan'
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
    plt.savefig(os.path.join(results_dir, '04_retqss_opt_cell_sizes_comparison.png'), dpi=300, bbox_inches='tight')
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
    plt.savefig(os.path.join(results_dir, '05_retqss_opt_cell_sizes_linear.png'), dpi=300, bbox_inches='tight')
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
    best_cell_sizes = [1.0, 2.0, 3.0, 4.0, 5.0]
    
    # Create the plot
    plt.figure(figsize=(14, 10))
    plt.suptitle('RETQSS Opt: Mejores Tamaños de Celda (Excluyendo 7.5m y 10m)', fontsize=16, fontweight='bold')
    
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
    plt.savefig(os.path.join(results_dir, '06_retqss_opt_best_cell_sizes.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a second plot with linear scale for better trend visibility
    plt.figure(figsize=(14, 10))
    plt.suptitle('RETQSS Opt: Mejores Tamaños de Celda - Escala Lineal (Excluyendo 7.5m y 10m)', fontsize=16, fontweight='bold')
    
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
    plt.savefig(os.path.join(results_dir, '07_retqss_opt_best_cell_sizes_linear.png'), dpi=300, bbox_inches='tight')
    plt.show()

def plot_performance_bar_chart(results, optimal_configs=None):
    """
    Generate a single bar chart showing performance comparison similar to plot 01 but in bar format.
    Similar to the style used in deltaq.py and breaking_lanes.py experiments.
    """
    df = pd.DataFrame(results)
    
    # Create results directory if it doesn't exist
    results_dir = 'experiments/performance_n_pedestrians/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Filter data by implementation
    qss_data = df[df['implementation'] == 'qss'].sort_values('n_pedestrians')
    retqss_data = df[df['implementation'] == 'retqss'].sort_values('n_pedestrians')
    retqss_opt_data = df[df['implementation'] == 'retqss_opt']
    
    # Prepare data for bar chart
    n_values = sorted(N_PEDESTRIANS)
    qss_means = []
    qss_stds = []
    retqss_means = []
    retqss_stds = []
    retqss_opt_means = []
    retqss_opt_stds = []
    
    # Extract performance data for each N value
    for n in n_values:
        # QSS data
        qss_n = qss_data[qss_data['n_pedestrians'] == n]
        if not qss_n.empty and qss_n['detailed_metrics'].iloc[0]:
            qss_means.append(qss_n['detailed_metrics'].iloc[0]['avg_iteration_time'])
            qss_stds.append(qss_n['detailed_metrics'].iloc[0]['std_iteration_time'])
        else:
            qss_means.append(0)
            qss_stds.append(0)
        
        # RETQSS data
        retqss_n = retqss_data[retqss_data['n_pedestrians'] == n]
        if not retqss_n.empty and retqss_n['detailed_metrics'].iloc[0]:
            retqss_means.append(retqss_n['detailed_metrics'].iloc[0]['avg_iteration_time'])
            retqss_stds.append(retqss_n['detailed_metrics'].iloc[0]['std_iteration_time'])
        else:
            retqss_means.append(0)
            retqss_stds.append(0)
        
        # RETQSS Opt data (best configuration for each N)
        if optimal_configs and n in optimal_configs:
            retqss_opt_n = retqss_opt_data[retqss_opt_data['n_pedestrians'] == n]
            retqss_opt_n = retqss_opt_n[retqss_opt_n['grid_divisions'] == optimal_configs[n]['grid_divisions']]
            if not retqss_opt_n.empty and retqss_opt_n['detailed_metrics'].iloc[0]:
                retqss_opt_means.append(retqss_opt_n['detailed_metrics'].iloc[0]['avg_iteration_time'])
                retqss_opt_stds.append(retqss_opt_n['detailed_metrics'].iloc[0]['std_iteration_time'])
            else:
                retqss_opt_means.append(0)
                retqss_opt_stds.append(0)
        else:
            retqss_opt_means.append(0)
            retqss_opt_stds.append(0)
    
    # Create the bar chart
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.suptitle('Comparación de Rendimiento: QSS vs RETQSS vs RETQSS Opt', fontsize=18, fontweight='bold')
    
    # Set up bar positions
    x = np.arange(len(n_values))
    width = 0.25
    
    # Create bars
    bars1 = ax.bar(x - width, qss_means, width, yerr=qss_stds, 
                   capsize=5, alpha=0.8, color='#FF6B6B', edgecolor='#D63031', 
                   linewidth=1.5, label='QSS')
    
    bars2 = ax.bar(x, retqss_means, width, yerr=retqss_stds, 
                   capsize=5, alpha=0.8, color='#4ECDC4', edgecolor='#00B894', 
                   linewidth=1.5, label='RETQSS')
    
    bars3 = ax.bar(x + width, retqss_opt_means, width, yerr=retqss_opt_stds, 
                   capsize=5, alpha=0.8, color='#6C5CE7', edgecolor='#5A4FCF', 
                   linewidth=1.5, label='RETQSS Opt')
    
    # Add value labels on top of bars
    for bars, means, stds in [(bars1, qss_means, qss_stds), 
                              (bars2, retqss_means, retqss_stds), 
                              (bars3, retqss_opt_means, retqss_opt_stds)]:
        for bar, mean, std in zip(bars, means, stds):
            if mean > 0:  # Only show label if there's data
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + std + height*0.01,
                       f'{mean:.2f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Customize the plot
    ax.set_xlabel('Número de Peatones (N)', fontsize=14)
    ax.set_ylabel('Tiempo Promedio de Ejecución (s)', fontsize=14)
    ax.set_title(f'Comparación de Rendimiento: Densidad Constante ({TARGET_DENSITY} peatones/m²)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in n_values])
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '02_performance_comparison_bar_chart.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Performance comparison bar chart generated successfully!")
    print("Generated file: 02_performance_comparison_bar_chart.png")

def plot_enhanced_cell_size_comparison(results):
    """
    Create a comprehensive bar chart comparison for all cell sizes (0.5m to 10m).
    Shows performance differences clearly with enhanced styling and value labels.
    """
    df = pd.DataFrame(results)
    
    # Create results directory if it doesn't exist
    results_dir = 'experiments/performance_n_pedestrians/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Filter only RETQSS optimized results
    retqss_opt_data = df[df['implementation'] == 'retqss_opt'].copy()
    
    if retqss_opt_data.empty:
        print("Warning: No RETQSS optimized results found for enhanced cell size comparison")
        return
    
    # Use all cell sizes
    all_cell_sizes = sorted(retqss_opt_data['cell_size'].unique())
    
    # Add performance metrics
    retqss_opt_data['avg_time'] = retqss_opt_data['detailed_metrics'].apply(
        lambda x: x['avg_iteration_time'] if x else 0
    )
    retqss_opt_data['std_time'] = retqss_opt_data['detailed_metrics'].apply(
        lambda x: x['std_iteration_time'] if x else 0
    )
    
    # Create comprehensive bar chart comparison for all cell sizes
    plt.figure(figsize=(20, 12))
    plt.suptitle('Comparación Completa: Rendimiento por Tamaño de Celda (Todos los Valores)', fontsize=20, fontweight='bold')
    
    # Create subplots for different N values
    n_values = sorted(retqss_opt_data['n_pedestrians'].unique())
    n_cols = 3
    n_rows = (len(n_values) + n_cols - 1) // n_cols
    
    # Define colors for all cell sizes
    cell_size_colors = {
        0.5: '#FF6B6B',   # Red
        1.0: '#4ECDC4',   # Teal
        2.0: '#45B7D1',   # Blue
        3.0: '#96CEB4',   # Green
        4.0: '#FFEAA7',   # Yellow
        5.0: '#DDA0DD',   # Plum
        7.5: '#FFB347',   # Orange
        10.0: '#98D8C8'   # Mint
    }
    
    for i, n in enumerate(n_values):
        plt.subplot(n_rows, n_cols, i + 1)
        
        n_data = retqss_opt_data[retqss_opt_data['n_pedestrians'] == n].sort_values('cell_size')
        
        if not n_data.empty:
            cell_sizes = n_data['cell_size']
            times = n_data['avg_time']
            stds = n_data['std_time']
            
            # Create bars with colors for each cell size
            colors = [cell_size_colors.get(cs, '#CCCCCC') for cs in cell_sizes]
            bars = plt.bar(range(len(cell_sizes)), times, yerr=stds, 
                          color=colors, alpha=0.8, capsize=5, 
                          edgecolor='black', linewidth=1.5)
            
            # Add value labels on top of bars
            for j, (bar, time, std) in enumerate(zip(bars, times, stds)):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + std + height*0.01,
                        f'{time:.1f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
            
            plt.xlabel('Tamaño de Celda (m)', fontsize=12)
            plt.ylabel('Tiempo (s)', fontsize=12)
            plt.title(f'N = {n}', fontsize=14, fontweight='bold')
            plt.xticks(range(len(cell_sizes)), [f'{cs}m' for cs in cell_sizes], rotation=45)
            plt.grid(True, alpha=0.3, axis='y')
            
            # Highlight the best performing cell size
            best_idx = times.idxmin()
            best_bar = bars[times.index.get_loc(best_idx)]
            best_bar.set_edgecolor('gold')
            best_bar.set_linewidth(4)
            
            # Add performance ranking text
            sorted_times = sorted(zip(cell_sizes, times), key=lambda x: x[1])
            ranking_text = "Ranking: " + " > ".join([f"{cs}m" for cs, _ in sorted_times[:3]])
            plt.text(0.02, 0.98, ranking_text, transform=plt.gca().transAxes, 
                    fontsize=8, verticalalignment='top', 
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '08_comprehensive_cell_size_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Comprehensive cell size comparison plot generated successfully!")
    print("Generated file: 08_comprehensive_cell_size_comparison.png")

def plot_qss_vs_retqss_bar_chart(results):
    """
    Generate a comprehensive comparison between QSS and RETQSS implementations.
    Left subplot: Performance comparison (bar chart)
    Right subplot: Memory usage comparison (line chart)
    """
    df = pd.DataFrame(results)
    
    # Create results directory if it doesn't exist
    results_dir = 'experiments/performance_n_pedestrians/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Filter data by implementation
    qss_data = df[df['implementation'] == 'qss'].sort_values('n_pedestrians')
    retqss_data = df[df['implementation'] == 'retqss'].sort_values('n_pedestrians')
    
    if qss_data.empty or retqss_data.empty:
        print("Warning: Missing QSS or RETQSS data for comparison")
        return
    
    # Prepare data for both charts
    n_values = sorted(N_PEDESTRIANS)
    qss_means = []
    qss_stds = []
    retqss_means = []
    retqss_stds = []
    qss_memory = []
    qss_memory_stds = []
    retqss_memory = []
    retqss_memory_stds = []
    
    # Extract performance and memory data for each N value
    for n in n_values:
        # QSS data
        qss_n = qss_data[qss_data['n_pedestrians'] == n]
        if not qss_n.empty and qss_n['detailed_metrics'].iloc[0]:
            qss_means.append(qss_n['detailed_metrics'].iloc[0]['avg_iteration_time'])
            qss_stds.append(qss_n['detailed_metrics'].iloc[0]['std_iteration_time'])
            # Memory data (already converted to MB in metrics)
            if qss_n['detailed_metrics'].iloc[0]['avg_memory_usage']:
                qss_memory.append(qss_n['detailed_metrics'].iloc[0]['avg_memory_usage'])
                qss_memory_stds.append(qss_n['detailed_metrics'].iloc[0]['std_memory_usage'] if qss_n['detailed_metrics'].iloc[0]['std_memory_usage'] else 0)
            else:
                qss_memory.append(0)
                qss_memory_stds.append(0)
        else:
            qss_means.append(0)
            qss_stds.append(0)
            qss_memory.append(0)
            qss_memory_stds.append(0)
        
        # RETQSS data
        retqss_n = retqss_data[retqss_data['n_pedestrians'] == n]
        if not retqss_n.empty and retqss_n['detailed_metrics'].iloc[0]:
            retqss_means.append(retqss_n['detailed_metrics'].iloc[0]['avg_iteration_time'])
            retqss_stds.append(retqss_n['detailed_metrics'].iloc[0]['std_iteration_time'])
            # Memory data (already converted to MB in metrics)
            if retqss_n['detailed_metrics'].iloc[0]['avg_memory_usage']:
                retqss_memory.append(retqss_n['detailed_metrics'].iloc[0]['avg_memory_usage'])
                retqss_memory_stds.append(retqss_n['detailed_metrics'].iloc[0]['std_memory_usage'] if retqss_n['detailed_metrics'].iloc[0]['std_memory_usage'] else 0)
            else:
                retqss_memory.append(0)
                retqss_memory_stds.append(0)
        else:
            retqss_means.append(0)
            retqss_stds.append(0)
            retqss_memory.append(0)
            retqss_memory_stds.append(0)
    
    # Create the combined chart with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    fig.suptitle('Comparación QSS vs RETQSS: Rendimiento y Uso de Memoria', fontsize=18, fontweight='bold')
    
    # LEFT SUBPLOT: Performance comparison (bar chart)
    x = np.arange(len(n_values))
    width = 0.35
    
    # Create performance bars
    bars1 = ax1.bar(x - width/2, qss_means, width, yerr=qss_stds, 
                    capsize=6, alpha=0.8, color='#FF6B6B', edgecolor='#D63031', 
                    linewidth=2, label='QSS', hatch='///')
    
    bars2 = ax1.bar(x + width/2, retqss_means, width, yerr=retqss_stds, 
                    capsize=6, alpha=0.8, color='#4ECDC4', edgecolor='#00B894', 
                    linewidth=2, label='RETQSS', hatch='\\\\\\')
    
    # Add value labels on performance bars
    for bars, means, stds in [(bars1, qss_means, qss_stds), 
                              (bars2, retqss_means, retqss_stds)]:
        for bar, mean, std in zip(bars, means, stds):
            if mean > 0:  # Only show label if there's data
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + std + height*0.02,
                        f'{mean:.2f}s', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # Customize performance subplot
    ax1.set_xlabel('Número de Peatones (N)', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Tiempo Promedio de Ejecución (s)', fontsize=14, fontweight='bold')
    ax1.set_title('Rendimiento: QSS vs RETQSS', fontsize=16, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(n) for n in n_values], fontsize=12)
    ax1.legend(fontsize=12, loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    ax1.tick_params(axis='both', which='major', labelsize=12)
    
    # Add performance improvement annotations
    for i, (qss_mean, retqss_mean) in enumerate(zip(qss_means, retqss_means)):
        if qss_mean > 0 and retqss_mean > 0:
            improvement = ((qss_mean - retqss_mean) / qss_mean) * 100
            if improvement > 0:
                ax1.annotate(f'+{improvement:.1f}%', 
                            xy=(i, max(qss_mean, retqss_mean)), 
                            xytext=(i, max(qss_mean, retqss_mean) + max(qss_stds[i], retqss_stds[i]) * 0.5),
                            ha='center', va='bottom', fontsize=8, fontweight='bold',
                            color='green', arrowprops=dict(arrowstyle='->', color='green', lw=1.2))
            else:
                ax1.annotate(f'{improvement:.1f}%', 
                            xy=(i, max(qss_mean, retqss_mean)), 
                            xytext=(i, max(qss_mean, retqss_mean) + max(qss_stds[i], retqss_stds[i]) * 0.5),
                            ha='center', va='bottom', fontsize=8, fontweight='bold',
                            color='red', arrowprops=dict(arrowstyle='->', color='red', lw=1.2))
    
    # RIGHT SUBPLOT: Memory usage comparison (line chart)
    # Filter out zero values for line chart
    valid_indices = [i for i, (qss_mem, retqss_mem) in enumerate(zip(qss_memory, retqss_memory)) 
                     if qss_mem > 0 and retqss_mem > 0]
    
    if valid_indices:
        valid_n_values = [n_values[i] for i in valid_indices]
        valid_qss_memory = [qss_memory[i] for i in valid_indices]
        valid_retqss_memory = [retqss_memory[i] for i in valid_indices]
        valid_qss_memory_stds = [qss_memory_stds[i] for i in valid_indices]
        valid_retqss_memory_stds = [retqss_memory_stds[i] for i in valid_indices]
        
        # Create memory line chart
        ax2.errorbar(valid_n_values, valid_qss_memory, yerr=valid_qss_memory_stds,
                    fmt='o-', label='QSS', linewidth=3, markersize=8, 
                    color='#FF6B6B', capsize=5, capthick=2, elinewidth=2, 
                    alpha=0.8, markeredgecolor='#D63031', markeredgewidth=2)
        
        ax2.errorbar(valid_n_values, valid_retqss_memory, yerr=valid_retqss_memory_stds,
                    fmt='s-', label='RETQSS', linewidth=3, markersize=8, 
                    color='#4ECDC4', capsize=5, capthick=2, elinewidth=2, 
                    alpha=0.8, markeredgecolor='#00B894', markeredgewidth=2)
        
        # Add value labels on memory points
        for i, (n, qss_mem, retqss_mem) in enumerate(zip(valid_n_values, valid_qss_memory, valid_retqss_memory)):
            ax2.annotate(f'{qss_mem:.1f}MB', 
                        xy=(n, qss_mem), xytext=(n, qss_mem + max(valid_qss_memory) * 0.05),
                        ha='center', va='bottom', fontsize=8, fontweight='bold', color='#D63031')
            ax2.annotate(f'{retqss_mem:.1f}MB', 
                        xy=(n, retqss_mem), xytext=(n, retqss_mem - max(valid_retqss_memory) * 0.05),
                        ha='center', va='top', fontsize=8, fontweight='bold', color='#00B894')
    else:
        # If no memory data available, show empty plot with message
        ax2.text(0.5, 0.5, 'No hay datos de memoria disponibles', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=14)
    
    # Customize memory subplot
    ax2.set_xlabel('Número de Peatones (N)', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Uso Promedio de Memoria (MB)', fontsize=14, fontweight='bold')
    ax2.set_title('Uso de Memoria: QSS vs RETQSS', fontsize=16, fontweight='bold')
    ax2.legend(fontsize=12, loc='upper left')
    ax2.grid(True, alpha=0.3)
    ax2.tick_params(axis='both', which='major', labelsize=12)
    
    # Add some styling to both subplots
    for ax in [ax1, ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
    
    # Add summary statistics text box for performance
    qss_avg = np.mean([m for m in qss_means if m > 0])
    retqss_avg = np.mean([m for m in retqss_means if m > 0])
    overall_improvement = ((qss_avg - retqss_avg) / qss_avg) * 100 if qss_avg > 0 else 0
    
    perf_stats_text = f'Mejora Promedio: {overall_improvement:.1f}%\nQSS: {qss_avg:.2f}s\nRETQSS: {retqss_avg:.2f}s'
    ax1.text(0.02, 0.98, perf_stats_text, transform=ax1.transAxes, 
             fontsize=9, verticalalignment='top', 
             bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='gray'))
    
    # Add summary statistics text box for memory
    if valid_indices:
        qss_avg_mem = np.mean(valid_qss_memory)
        retqss_avg_mem = np.mean(valid_retqss_memory)
        mem_improvement = ((qss_avg_mem - retqss_avg_mem) / qss_avg_mem) * 100 if qss_avg_mem > 0 else 0
        
        mem_stats_text = f'Memoria Promedio:\nQSS: {qss_avg_mem:.1f}MB\nRETQSS: {retqss_avg_mem:.1f}MB\nMejora: {mem_improvement:.1f}%'
        ax2.text(0.02, 0.98, mem_stats_text, transform=ax2.transAxes, 
                 fontsize=9, verticalalignment='top', 
                 bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '03_qss_vs_retqss_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("QSS vs RETQSS comprehensive comparison generated successfully!")
    print("Generated file: 03_qss_vs_retqss_comparison.png")

def plot_memory_usage_comparison(results, optimal_configs=None):
    """
    Generate a comprehensive memory usage comparison between QSS, RETQSS, and RETQSS Opt implementations.
    This chart shows memory consumption patterns across different N values for all three implementations.
    """
    df = pd.DataFrame(results)
    
    # Create results directory if it doesn't exist
    results_dir = 'experiments/performance_n_pedestrians/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Filter data by implementation
    qss_data = df[df['implementation'] == 'qss'].sort_values('n_pedestrians')
    retqss_data = df[df['implementation'] == 'retqss'].sort_values('n_pedestrians')
    retqss_opt_data = df[df['implementation'] == 'retqss_opt']
    
    # Prepare data for bar chart
    n_values = sorted(N_PEDESTRIANS)
    qss_memory = []
    qss_memory_stds = []
    retqss_memory = []
    retqss_memory_stds = []
    retqss_opt_memory = []
    retqss_opt_memory_stds = []
    
    # Extract memory data for each N value
    for n in n_values:
        # QSS memory data (already converted to MB in metrics)
        qss_n = qss_data[qss_data['n_pedestrians'] == n]
        if not qss_n.empty and qss_n['detailed_metrics'].iloc[0] and qss_n['detailed_metrics'].iloc[0]['avg_memory_usage']:
            qss_memory.append(qss_n['detailed_metrics'].iloc[0]['avg_memory_usage'])
            qss_memory_stds.append(qss_n['detailed_metrics'].iloc[0]['std_memory_usage'] if qss_n['detailed_metrics'].iloc[0]['std_memory_usage'] else 0)  # Using actual memory std
        else:
            qss_memory.append(0)
            qss_memory_stds.append(0)
        
        # RETQSS memory data (already converted to MB in metrics)
        retqss_n = retqss_data[retqss_data['n_pedestrians'] == n]
        if not retqss_n.empty and retqss_n['detailed_metrics'].iloc[0] and retqss_n['detailed_metrics'].iloc[0]['avg_memory_usage']:
            retqss_memory.append(retqss_n['detailed_metrics'].iloc[0]['avg_memory_usage'])
            retqss_memory_stds.append(retqss_n['detailed_metrics'].iloc[0]['std_memory_usage'] if retqss_n['detailed_metrics'].iloc[0]['std_memory_usage'] else 0)  # Using actual memory std
        else:
            retqss_memory.append(0)
            retqss_memory_stds.append(0)
        
        # RETQSS Opt memory data (best configuration for each N) (already converted to MB in metrics)
        if optimal_configs and n in optimal_configs:
            retqss_opt_n = retqss_opt_data[retqss_opt_data['n_pedestrians'] == n]
            retqss_opt_n = retqss_opt_n[retqss_opt_n['grid_divisions'] == optimal_configs[n]['grid_divisions']]
            if not retqss_opt_n.empty and retqss_opt_n['detailed_metrics'].iloc[0] and retqss_opt_n['detailed_metrics'].iloc[0]['avg_memory_usage']:
                retqss_opt_memory.append(retqss_opt_n['detailed_metrics'].iloc[0]['avg_memory_usage'])
                retqss_opt_memory_stds.append(retqss_opt_n['detailed_metrics'].iloc[0]['std_memory_usage'] if retqss_opt_n['detailed_metrics'].iloc[0]['std_memory_usage'] else 0)  # Using actual memory std
            else:
                retqss_opt_memory.append(0)
                retqss_opt_memory_stds.append(0)
        else:
            retqss_opt_memory.append(0)
            retqss_opt_memory_stds.append(0)
    
    # Create the memory usage line chart
    fig, ax = plt.subplots(figsize=(16, 10))
    fig.suptitle('Comparación de Uso de Memoria: QSS vs RETQSS vs RETQSS Opt', fontsize=18, fontweight='bold')
    
    # Filter out zero values for line chart
    valid_indices = [i for i, (qss_mem, retqss_mem, retqss_opt_mem) in enumerate(zip(qss_memory, retqss_memory, retqss_opt_memory)) 
                     if qss_mem > 0 and retqss_mem > 0 and retqss_opt_mem > 0]
    
    if valid_indices:
        valid_n_values = [n_values[i] for i in valid_indices]
        valid_qss_memory = [qss_memory[i] for i in valid_indices]
        valid_retqss_memory = [retqss_memory[i] for i in valid_indices]
        valid_retqss_opt_memory = [retqss_opt_memory[i] for i in valid_indices]
        valid_qss_memory_stds = [qss_memory_stds[i] for i in valid_indices]
        valid_retqss_memory_stds = [retqss_memory_stds[i] for i in valid_indices]
        valid_retqss_opt_memory_stds = [retqss_opt_memory_stds[i] for i in valid_indices]
        
        # Create memory line chart
        ax.errorbar(valid_n_values, valid_qss_memory, yerr=valid_qss_memory_stds,
                    fmt='o-', label='QSS', linewidth=4, markersize=10, 
                    color='#FF6B6B', capsize=6, capthick=3, elinewidth=3, 
                    alpha=0.8, markeredgecolor='#D63031', markeredgewidth=2)
        
        ax.errorbar(valid_n_values, valid_retqss_memory, yerr=valid_retqss_memory_stds,
                    fmt='s-', label='RETQSS', linewidth=4, markersize=10, 
                    color='#4ECDC4', capsize=6, capthick=3, elinewidth=3, 
                    alpha=0.8, markeredgecolor='#00B894', markeredgewidth=2)
        
        ax.errorbar(valid_n_values, valid_retqss_opt_memory, yerr=valid_retqss_opt_memory_stds,
                    fmt='^-', label='RETQSS Opt', linewidth=4, markersize=10, 
                    color='#6C5CE7', capsize=6, capthick=3, elinewidth=3, 
                    alpha=0.8, markeredgecolor='#5A4FCF', markeredgewidth=2)
        
        # Add value labels on memory points
        for i, (n, qss_mem, retqss_mem, retqss_opt_mem) in enumerate(zip(valid_n_values, valid_qss_memory, valid_retqss_memory, valid_retqss_opt_memory)):
            ax.annotate(f'{qss_mem:.1f}MB', 
                        xy=(n, qss_mem), xytext=(n, qss_mem + max(valid_qss_memory) * 0.05),
                        ha='center', va='bottom', fontsize=8, fontweight='bold', color='#D63031')
            ax.annotate(f'{retqss_mem:.1f}MB', 
                        xy=(n, retqss_mem), xytext=(n, retqss_mem - max(valid_retqss_memory) * 0.05),
                        ha='center', va='top', fontsize=8, fontweight='bold', color='#00B894')
            ax.annotate(f'{retqss_opt_mem:.1f}MB', 
                        xy=(n, retqss_opt_mem), xytext=(n, retqss_opt_mem + max(valid_retqss_opt_memory) * 0.05),
                        ha='center', va='bottom', fontsize=8, fontweight='bold', color='#5A4FCF')
    else:
        # If no memory data available, show empty plot with message
        ax.text(0.5, 0.5, 'No hay datos de memoria disponibles', 
                ha='center', va='center', transform=ax.transAxes, fontsize=14)
    
    # Customize the plot
    ax.set_xlabel('Número de Peatones (N)', fontsize=14, fontweight='bold')
    ax.set_ylabel('Uso Promedio de Memoria (MB)', fontsize=14, fontweight='bold')
    ax.set_title(f'Comparación de Uso de Memoria: Densidad Constante ({TARGET_DENSITY} peatones/m²)', fontsize=16)
    ax.legend(fontsize=12, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    # Add some styling
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_linewidth(1.5)
    ax.spines['bottom'].set_linewidth(1.5)
    
    # Add summary statistics text box
    if valid_indices:
        qss_avg_mem = np.mean(valid_qss_memory)
        retqss_avg_mem = np.mean(valid_retqss_memory)
        retqss_opt_avg_mem = np.mean(valid_retqss_opt_memory)
        
        stats_text = f'Memoria Promedio:\nQSS: {qss_avg_mem:.1f}MB\nRETQSS: {retqss_avg_mem:.1f}MB\nRETQSS Opt: {retqss_opt_avg_mem:.1f}MB'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '04_memory_usage_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # Create a second plot showing memory efficiency (memory per pedestrian)
    plt.figure(figsize=(16, 10))
    plt.suptitle('Eficiencia de Memoria: MB por Peatón', fontsize=18, fontweight='bold')
    
    # Calculate memory per pedestrian
    qss_mem_per_ped = [mem/n if n > 0 and mem > 0 else 0 for mem, n in zip(qss_memory, n_values)]
    retqss_mem_per_ped = [mem/n if n > 0 and mem > 0 else 0 for mem, n in zip(retqss_memory, n_values)]
    retqss_opt_mem_per_ped = [mem/n if n > 0 and mem > 0 else 0 for mem, n in zip(retqss_opt_memory, n_values)]
    
    # Filter out zero values for efficiency line chart
    valid_eff_indices = [i for i, (qss_eff, retqss_eff, retqss_opt_eff) in enumerate(zip(qss_mem_per_ped, retqss_mem_per_ped, retqss_opt_mem_per_ped)) 
                         if qss_eff > 0 and retqss_eff > 0 and retqss_opt_eff > 0]
    
    if valid_eff_indices:
        valid_eff_n_values = [n_values[i] for i in valid_eff_indices]
        valid_qss_eff = [qss_mem_per_ped[i] for i in valid_eff_indices]
        valid_retqss_eff = [retqss_mem_per_ped[i] for i in valid_eff_indices]
        valid_retqss_opt_eff = [retqss_opt_mem_per_ped[i] for i in valid_eff_indices]
        
        # Create efficiency line chart
        plt.errorbar(valid_eff_n_values, valid_qss_eff,
                    fmt='o-', label='QSS', linewidth=4, markersize=10, 
                    color='#FF6B6B', alpha=0.8, markeredgecolor='#D63031', markeredgewidth=2)
        
        plt.errorbar(valid_eff_n_values, valid_retqss_eff,
                    fmt='s-', label='RETQSS', linewidth=4, markersize=10, 
                    color='#4ECDC4', alpha=0.8, markeredgecolor='#00B894', markeredgewidth=2)
        
        plt.errorbar(valid_eff_n_values, valid_retqss_opt_eff,
                    fmt='^-', label='RETQSS Opt', linewidth=4, markersize=10, 
                    color='#6C5CE7', alpha=0.8, markeredgecolor='#5A4FCF', markeredgewidth=2)
        
        # Add value labels for efficiency
        for i, (n, qss_eff, retqss_eff, retqss_opt_eff) in enumerate(zip(valid_eff_n_values, valid_qss_eff, valid_retqss_eff, valid_retqss_opt_eff)):
            plt.annotate(f'{qss_eff:.3f}MB', 
                        xy=(n, qss_eff), xytext=(n, qss_eff + max(valid_qss_eff) * 0.05),
                        ha='center', va='bottom', fontsize=8, fontweight='bold', color='#D63031')
            plt.annotate(f'{retqss_eff:.3f}MB', 
                        xy=(n, retqss_eff), xytext=(n, retqss_eff - max(valid_retqss_eff) * 0.05),
                        ha='center', va='top', fontsize=8, fontweight='bold', color='#00B894')
            plt.annotate(f'{retqss_opt_eff:.3f}MB', 
                        xy=(n, retqss_opt_eff), xytext=(n, retqss_opt_eff + max(valid_retqss_opt_eff) * 0.05),
                        ha='center', va='bottom', fontsize=8, fontweight='bold', color='#5A4FCF')
    else:
        # If no efficiency data available, show empty plot with message
        plt.text(0.5, 0.5, 'No hay datos de eficiencia disponibles', 
                ha='center', va='center', transform=plt.gca().transAxes, fontsize=14)
    
    plt.xlabel('Número de Peatones (N)', fontsize=14, fontweight='bold')
    plt.ylabel('Memoria por Peatón (MB)', fontsize=14, fontweight='bold')
    plt.title('Eficiencia de Memoria: MB por Peatón', fontsize=16)
    plt.legend(fontsize=12, loc='upper left')
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)
    
    # Add some styling
    ax2 = plt.gca()
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.spines['left'].set_linewidth(1.5)
    ax2.spines['bottom'].set_linewidth(1.5)
    
    # Add efficiency statistics
    if valid_eff_indices:
        qss_avg_eff = np.mean(valid_qss_eff)
        retqss_avg_eff = np.mean(valid_retqss_eff)
        retqss_opt_avg_eff = np.mean(valid_retqss_opt_eff)
        
        eff_stats_text = f'Eficiencia Promedio:\nQSS: {qss_avg_eff:.3f}MB/p\nRETQSS: {retqss_avg_eff:.3f}MB/p\nRETQSS Opt: {retqss_opt_avg_eff:.3f}MB/p'
        ax2.text(0.02, 0.98, eff_stats_text, transform=ax2.transAxes, 
                 fontsize=10, verticalalignment='top', 
                 bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.9, edgecolor='gray'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '05_memory_efficiency_comparison.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    print("Memory usage comparison plots generated successfully!")
    print("Generated files: 04_memory_usage_comparison.png, 05_memory_efficiency_comparison.png")

if __name__ == '__main__':
    performance_n_pedestrians()
