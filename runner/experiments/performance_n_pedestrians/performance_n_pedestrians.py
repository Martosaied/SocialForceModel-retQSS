import json
import os
import subprocess
import numpy as np
from src.runner import run_experiment, compile_c_code, compile_model
from src.utils import load_config, create_output_dir, copy_results_to_latest
from src.config_manager import config_manager
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import pandas as pd
from itertools import product
from src.constants import Constants


execute_experiment = False

# Test configurations
N_PEDESTRIANS = [1000, 3000, 5000] #, 10000]
TARGET_DENSITY = 0.3  # peatones/m² - densidad constante
CELL_SIZES = [3.0, 4.0, 5.0, 7.5, 10.0]  # metros por celda - diferentes tamaños de celda


def get_simulated_time():
    """Get simulated time from config.json."""
    config = load_config('experiments/performance_n_pedestrians/config.json')
    return config['parameters']['FORCE_TERMINATION_AT']['value']

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

    # Phase 12: Generate cell size comparison with exec time
    print("\n12. Generating cell size comparison (exec time)...")
    plot_cell_size_comparison_exec_time(results)

    # Phase 13: Generate LaTeX table (memory per pedestrian + RTF)
    print("\n13. Generating LaTeX table (memory per pedestrian + RTF)...")
    generate_memory_rtf_latex_table(results, optimal_configs)

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
    
    # Only create a new output directory when actually running experiments.
    # When plotting/re-reading existing results, we point to the canonical results path.
    if execute_experiment:
        output_dir = create_output_dir(
            'experiments/performance_n_pedestrians/results',
            exp_name
        )
    else:
        output_dir = os.path.join('experiments/performance_n_pedestrians/results', exp_name)

    if execute_experiment:
        # Update config parameters
        config['iterations'] = 10
        config['parameters']['N']['value'] = n
        config['parameters']['PEDESTRIAN_IMPLEMENTATION']['value'] = pedestrian_implementation
        config['parameters']['BORDER_IMPLEMENTATION']['value'] = 1
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
                # Get simulated time from config
                simulated_time = get_simulated_time()
                
                # Calculate simulation speed ratio: simulated_time / execution_time
                # Higher is better (simulation runs faster than real-time)
                speed_ratios = simulated_time / metrics_df['time']
                
                lanes_mean = metrics_df['clustering_based_groups'].mean() if 'clustering_based_groups' in metrics_df.columns else None
                lanes_std = metrics_df['clustering_based_groups'].std() if 'clustering_based_groups' in metrics_df.columns else None

                detailed_metrics = {
                    'total_iterations': len(metrics_df),
                    'execution_time': metrics_df['time'].mean(),  # Execution time in seconds
                    'execution_time_std': metrics_df['time'].std(),  # Std dev of execution time in seconds
                    'avg_iteration_time': speed_ratios.mean(),  # Now contains speed ratio instead of time
                    'min_iteration_time': speed_ratios.min(),  # Minimum speed ratio
                    'max_iteration_time': speed_ratios.max(),  # Maximum speed ratio
                    'std_iteration_time': speed_ratios.std(),  # Std of speed ratio
                    'avg_memory_usage': (metrics_df['memory_usage'].mean() / 1024) if 'memory_usage' in metrics_df.columns else None,  # Convert KB to MB
                    'std_memory_usage': (metrics_df['memory_usage'].std() / 1024) if 'memory_usage' in metrics_df.columns else None,  # Convert KB to MB
                    'lanes_detected_mean': lanes_mean,
                    'lanes_detected_std': lanes_std,
                    'simulated_time': simulated_time,
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
    
    # Find optimal cell size for each N (maximum speed ratio / RTF)
    optimal_configs = {}
    for n in N_PEDESTRIANS:
        n_data = retqss_opt_data[retqss_opt_data['n_pedestrians'] == n]
        if not n_data.empty:
            best_idx = n_data['avg_time'].idxmax()
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
    retqss_opt_exec_data = retqss_opt_data[retqss_opt_data['detailed_metrics'].notnull()].copy()
    retqss_opt_exec_data['exec_time'] = retqss_opt_exec_data['detailed_metrics'].apply(
        lambda x: x.get('execution_time') if x else None
    )
    retqss_opt_exec_data = retqss_opt_exec_data[retqss_opt_exec_data['exec_time'].notnull()]
    
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
    plt.ylabel('Ratio de Velocidad (tiempo simulado / tiempo real)', fontsize=14)
    plt.title(f'Velocidad de Simulación: Densidad Constante ({TARGET_DENSITY} peatones/m²)', fontsize=16)
    
    # Add horizontal line at 1.0 to show real-time threshold
    plt.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Tiempo Real (1.0x)')
    
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=12)
    # plt.yscale('log')  # Changed to linear scale
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '01_performance_comparison.png'),  bbox_inches='tight')

    plt.yscale('log')  # Changed to linear scale
    plt.savefig(os.path.join(results_dir, '01_performance_comparison_log.png'),  bbox_inches='tight')
    
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
    
    plt.xlabel('Número de Peatones (N)', fontsize=20)
    plt.ylabel('Uso Promedio de Memoria (MB)', fontsize=20)
    plt.legend(fontsize=20)
    plt.grid(True, alpha=0.3)
    plt.tick_params(axis='both', which='major', labelsize=20)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '03_qss_vs_retqss_comparison.png'),  bbox_inches='tight')
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
    plt.savefig(os.path.join(results_dir, '04_retqss_opt_cell_sizes_comparison.png'),  bbox_inches='tight')
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
    plt.savefig(os.path.join(results_dir, '05_retqss_opt_cell_sizes_linear.png'),  bbox_inches='tight')
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
    plt.savefig(os.path.join(results_dir, '06_retqss_opt_best_cell_sizes.png'),  bbox_inches='tight')
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
    plt.savefig(os.path.join(results_dir, '07_retqss_opt_best_cell_sizes_linear.png'),  bbox_inches='tight')
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
    retqss_opt_data = df[df['implementation'] == 'retqss_opt'].copy()

    # Add exec_time column for finding best configuration by execution time
    retqss_opt_data['exec_time'] = retqss_opt_data['detailed_metrics'].apply(
        lambda x: x.get('execution_time') if x else None
    )

    # Prepare data for bar charts
    n_values = sorted(N_PEDESTRIANS)
    qss_means = []
    qss_stds = []
    retqss_means = []
    retqss_stds = []
    retqss_opt_means = []
    retqss_opt_stds = []
    qss_exec_means = []
    qss_exec_stds = []
    retqss_exec_means = []
    retqss_exec_stds = []
    retqss_opt_exec_means = []
    retqss_opt_exec_stds = []
    
    # Extract performance data for each N value
    for n in n_values:
        # QSS data
        qss_n = qss_data[qss_data['n_pedestrians'] == n]
        if not qss_n.empty and qss_n['detailed_metrics'].iloc[0]:
            qss_means.append(qss_n['detailed_metrics'].iloc[0]['avg_iteration_time'])
            qss_stds.append(qss_n['detailed_metrics'].iloc[0]['std_iteration_time'])
            qss_exec_means.append(qss_n['detailed_metrics'].iloc[0]['execution_time'])
            qss_exec_stds.append(qss_n['detailed_metrics'].iloc[0].get('execution_time_std') or 0)
        else:
            qss_means.append(0)
            qss_stds.append(0)
            qss_exec_means.append(0)
            qss_exec_stds.append(0)
        
        # RETQSS data
        retqss_n = retqss_data[retqss_data['n_pedestrians'] == n]
        if not retqss_n.empty and retqss_n['detailed_metrics'].iloc[0]:
            retqss_means.append(retqss_n['detailed_metrics'].iloc[0]['avg_iteration_time'])
            retqss_stds.append(retqss_n['detailed_metrics'].iloc[0]['std_iteration_time'])
            retqss_exec_means.append(retqss_n['detailed_metrics'].iloc[0]['execution_time'])
            retqss_exec_stds.append(retqss_n['detailed_metrics'].iloc[0].get('execution_time_std') or 0)
        else:
            retqss_means.append(0)
            retqss_stds.append(0)
            retqss_exec_means.append(0)
            retqss_exec_stds.append(0)
        
        # RETQSS Opt data (best configuration for each N by avg iteration time)
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

        # RETQSS Opt execution time (best configuration for each N by execution time)
        retqss_opt_exec_n = retqss_opt_data[retqss_opt_data['n_pedestrians'] == n]
        if not retqss_opt_exec_n.empty:
            best_idx = retqss_opt_exec_n['exec_time'].idxmin()
            best_exec = retqss_opt_exec_n.loc[best_idx]['detailed_metrics']
            retqss_opt_exec_means.append(best_exec['execution_time'])
            retqss_opt_exec_stds.append(best_exec.get('execution_time_std') or 0)
        else:
            retqss_opt_exec_means.append(0)
            retqss_opt_exec_stds.append(0)
    
    # Create the bar chart
    fig, ax = plt.subplots(figsize=(16, 10))
    
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
    
    # Customize the plot
    ax.set_xlabel('Número de Peatones (N)')
    ax.set_ylabel('Ratio de velocidad (tiempo simulado / tiempo real)')
    ax.set_title(f'Velocidad de Simulación: Densidad Constante ({TARGET_DENSITY} peatones/m²)', fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in n_values])
    
    # Add horizontal line at 1.0 to show real-time threshold
    ax.axhline(y=1.0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='Tiempo Real (1.0x)')
    
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', which='major', labelsize=12)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '02_performance_comparison_bar_chart.png'),  bbox_inches='tight')
    plt.show()
    
    print("Performance comparison bar chart generated successfully!")
    print("Generated file: 02_performance_comparison_bar_chart.png")

    # Create the execution time bar chart
    fig, ax = plt.subplots(figsize=(18, 14))

    # Create bars
    bars1 = ax.bar(x - width, qss_exec_means, width, yerr=qss_exec_stds,
                   capsize=5, alpha=0.8, color='#FF6B6B', edgecolor='#D63031',
                   linewidth=1.5, label='QSS')

    bars2 = ax.bar(x, retqss_exec_means, width, yerr=retqss_exec_stds,
                   capsize=5, alpha=0.8, color='#4ECDC4', edgecolor='#00B894',
                   linewidth=1.5, label='RETQSS')

    bars3 = ax.bar(x + width, retqss_opt_exec_means, width, yerr=retqss_opt_exec_stds,
                   capsize=5, alpha=0.8, color='#6C5CE7', edgecolor='#5A4FCF',
                   linewidth=1.5, label='RETQSS Opt')

    # Customize the plot
    ax.set_xlabel('Número de Peatones (N)', fontsize=35)
    ax.set_ylabel('Tiempo promedio de ejecución (segundos)', fontsize=35)
    ax.set_ylim(0, max(qss_exec_means) + max(qss_exec_means) * 0.3)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in n_values])

    ax.legend(fontsize=30)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', which='major', labelsize=30)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '02_performance_comparison_bar_chart_exec_time.png'),  bbox_inches='tight')
    plt.show()

    print("Execution time bar chart generated successfully!")
    print("Generated file: 02_performance_comparison_bar_chart_exec_time.png")

    # Create a second execution time bar chart without RETQSS (only QSS vs RETQSS Opt)
    fig, ax = plt.subplots(figsize=(18, 14))

    width_2 = 0.35
    bars1 = ax.bar(x - width_2 / 2, qss_exec_means, width_2, yerr=qss_exec_stds,
                   capsize=5, alpha=0.8, color='#FF6B6B', edgecolor='#D63031',
                   linewidth=1.5, label='RETQSS Base')

    bars2 = ax.bar(x + width_2 / 2, retqss_opt_exec_means, width_2, yerr=retqss_opt_exec_stds,
                   capsize=5, alpha=0.8, color='#6C5CE7', edgecolor='#5A4FCF',
                   linewidth=1.5, label='RETQSS Opt')

    ax.set_xlabel('Número de Peatones (N)', fontsize=35)
    ax.set_ylabel('Tiempo promedio de ejecución (segundos)', fontsize=35)
    max_exec_time = max(qss_exec_means + retqss_opt_exec_means) if (qss_exec_means and retqss_opt_exec_means) else 0
    ax.set_ylim(0, max_exec_time + max_exec_time * 0.3 if max_exec_time > 0 else 1)
    ax.set_xticks(x)
    ax.set_xticklabels([str(n) for n in n_values])

    ax.legend(fontsize=30)
    ax.grid(True, alpha=0.3, axis='y')
    ax.tick_params(axis='both', which='major', labelsize=30)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '02b_performance_comparison_bar_chart_exec_time_no_retqss.png'),  bbox_inches='tight')
    plt.show()

    print("Execution time bar chart (no RETQSS) generated successfully!")
    print("Generated file: 02b_performance_comparison_bar_chart_exec_time_no_retqss.png")

    # Generate LaTeX table with exact values
    latex_path = os.path.join(results_dir, 'performance_n_pedestrians_results.tex')
    with open(latex_path, 'w') as f:
        f.write(r'\begin{table}[htbp]' + '\n')
        f.write(r'\centering' + '\n')
        f.write(r'\caption{Resultados del experimento Performance N Peatones: RTF y tiempo de ejecución (s) por implementación.}' + '\n')
        f.write(r'\label{tab:performance_n_pedestrians}' + '\n')
        f.write(r'\begin{tabular}{ccccccc}' + '\n')
        f.write(r'\hline' + '\n')
        f.write(r'N & QSS RTF & QSS Tiempo (s) & RETQSS RTF & RETQSS Tiempo (s) & RETQSS Opt RTF & RETQSS Opt Tiempo (s) \\' + '\n')
        f.write(r'\hline' + '\n')
        for i, n in enumerate(n_values):
            f.write(f'{n} & {qss_means[i]:.4f} $\\pm$ {qss_stds[i]:.4f} & {qss_exec_means[i]:.2f} $\\pm$ {qss_exec_stds[i]:.2f} & ')
            f.write(f'{retqss_means[i]:.4f} $\\pm$ {retqss_stds[i]:.4f} & {retqss_exec_means[i]:.2f} $\\pm$ {retqss_exec_stds[i]:.2f} & ')
            f.write(f'{retqss_opt_means[i]:.4f} $\\pm$ {retqss_opt_stds[i]:.4f} & {retqss_opt_exec_means[i]:.2f} $\\pm$ {retqss_opt_exec_stds[i]:.2f} \\\\\n')
        f.write(r'\hline' + '\n')
        f.write(r'\end{tabular}' + '\n')
        f.write(r'\end{table}' + '\n')
    print(f"LaTeX table saved to: {latex_path}")

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

    simulated_time = get_simulated_time()

    def load_execution_time_stats(output_dir):
        metrics_path = os.path.join(output_dir, 'latest', 'metrics.csv')
        if os.path.exists(metrics_path):
            df_metrics = pd.read_csv(metrics_path)
            if not df_metrics.empty and 'time' in df_metrics.columns:
                return df_metrics['time'].mean(), df_metrics['time'].std()
        return 0, 0
    
    # Use all cell sizes
    all_cell_sizes = sorted(retqss_opt_data['cell_size'].unique())
    
    # Add performance metrics
    retqss_opt_data['avg_time'] = retqss_opt_data['detailed_metrics'].apply(
        lambda x: x['avg_iteration_time'] if x else 0
    )
    retqss_opt_data['std_time'] = retqss_opt_data['detailed_metrics'].apply(
        lambda x: x['std_iteration_time'] if x else 0
    )
    execution_stats = retqss_opt_data['output_dir'].apply(load_execution_time_stats)
    retqss_opt_data['execution_time_mean'] = execution_stats.apply(lambda x: x[0])
    retqss_opt_data['execution_time_std'] = execution_stats.apply(lambda x: x[1])
    
    # Create comprehensive bar chart comparison for all cell sizes
    plt.figure(figsize=(20, 10))
    
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
            exec_times = n_data['execution_time_mean']
            
            # Create bars with colors for each cell size
            colors = [cell_size_colors.get(cs, '#CCCCCC') for cs in cell_sizes]
            bars = plt.bar(range(len(cell_sizes)), times, yerr=stds, 
                          color=colors, alpha=0.8, capsize=5, 
                          edgecolor='black', linewidth=1.5)
            
            max_rtf = max(times) if len(times) > 0 else 0
            max_std = max(stds) if len(stds) > 0 else 0
            if max_rtf > 0:
                plt.ylim(0, max_rtf + max_std + max_rtf * 0.7)
            plt.xlabel('Tamaño de Celda (m)', fontsize=25)
            plt.ylabel('RTF', fontsize=25)
            plt.title(f'N = {n}', fontsize=25, fontweight='bold')
            plt.xticks(range(len(cell_sizes)), [f'{cs}m' for cs in cell_sizes], rotation=45)
            
            # Add horizontal line at 1.0 to show real-time threshold
            plt.axhline(y=1.0, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
            
            plt.grid(True, alpha=0.3, axis='y')
            legend_handles = [
                Line2D([0], [0], color='red', linestyle='--', label='Tiempo Real (1.0x)'),
                Line2D([], [], linestyle='None', label=f'Tiempo simulado: {simulated_time:.0f}s'),
            ]
            plt.legend(handles=legend_handles, fontsize=20, loc='upper right')
            
            # Highlight the best performing cell size
            best_idx = times.idxmax()
            best_bar = bars[times.index.get_loc(best_idx)]
            best_bar.set_edgecolor('gold')
            best_bar.set_linewidth(4)
            
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '08_comprehensive_cell_size_comparison.png'),  bbox_inches='tight')
    plt.show()
    
    print("Comprehensive cell size comparison plot generated successfully!")
    print("Generated file: 08_comprehensive_cell_size_comparison.png")

    # Generate LaTeX table for cell size comparison (RTF)
    latex_path = os.path.join(results_dir, 'cell_size_comparison_rtf_results.tex')
    with open(latex_path, 'w') as f:
        f.write(r'\begin{table}[htbp]' + '\n')
        f.write(r'\centering' + '\n')
        f.write(r'\caption{RTF por tamaño de celda y número de peatones (RETQSS Opt).}' + '\n')
        f.write(r'\label{tab:cell_size_rtf}' + '\n')
        f.write(r'\begin{tabular}{ccc}' + '\n')
        f.write(r'\hline' + '\n')
        f.write(r'N & Celda (m) & RTF (mean $\pm$ std) \\' + '\n')
        f.write(r'\hline' + '\n')
        for n in n_values:
            n_data = retqss_opt_data[retqss_opt_data['n_pedestrians'] == n].sort_values('cell_size')
            for _, row in n_data.iterrows():
                cs = row['cell_size']
                rtf_m, rtf_s = row['avg_time'], row['std_time']
                f.write(f'{n} & {cs} & {rtf_m:.4f} $\\pm$ {rtf_s:.4f} \\\\\n')
        f.write(r'\hline' + '\n')
        f.write(r'\end{tabular}' + '\n')
        f.write(r'\end{table}' + '\n')
    print(f"LaTeX table saved to: {latex_path}")

def plot_cell_size_comparison_exec_time(results):
    """
    Create a bar chart comparison for all cell sizes showing execution time on Y axis.
    RTF is displayed as a label on each bar instead of being the Y axis variable.
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

    simulated_time = get_simulated_time()

    # Use all cell sizes
    all_cell_sizes = sorted(retqss_opt_data['cell_size'].unique())

    # Add performance metrics from detailed_metrics
    retqss_opt_data['avg_time'] = retqss_opt_data['detailed_metrics'].apply(
        lambda x: x['avg_iteration_time'] if x else 0
    )
    retqss_opt_data['std_time'] = retqss_opt_data['detailed_metrics'].apply(
        lambda x: x['std_iteration_time'] if x else 0
    )
    retqss_opt_data['execution_time_mean'] = retqss_opt_data['detailed_metrics'].apply(
        lambda x: x.get('execution_time', 0) if x else 0
    )
    retqss_opt_data['execution_time_std'] = retqss_opt_data['detailed_metrics'].apply(
        lambda x: x.get('execution_time_std', 0) or 0 if x else 0
    )

    # Create comprehensive bar chart comparison for all cell sizes
    plt.figure(figsize=(20, 10))

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
            rtf_values = n_data['avg_time']
            rtf_stds = n_data['std_time']
            exec_times = n_data['execution_time_mean']
            exec_stds = n_data['execution_time_std']

            # Create bars with execution time on Y axis
            colors = [cell_size_colors.get(cs, '#CCCCCC') for cs in cell_sizes]
            bars = plt.bar(range(len(cell_sizes)), exec_times, yerr=exec_stds,
                          color=colors, alpha=0.8, capsize=5,
                          edgecolor='black', linewidth=1.5)

            max_exec = max(exec_times) if len(exec_times) > 0 else 0
            max_std = max(exec_stds) if len(exec_stds) > 0 else 0
            if max_exec > 0:
                plt.ylim(0, max_exec + max_std + max_exec * 0.7)

            # Only show X axis label on middle figure (index 1)
            if i == 1:
                plt.xlabel('Tamaño de Celda (m)', fontsize=25)
            # Only show Y axis label on first figure (index 0)
            if i == 0:
                plt.ylabel('Tiempo de Ejecución (s)', fontsize=25)
            plt.xticks(range(len(cell_sizes)), [f'{cs}m' for cs in cell_sizes], rotation=45)

            plt.grid(True, alpha=0.3, axis='y')
            legend_handles = [
                Line2D([], [], linestyle='None', label=f'N = {n}'),
                Line2D([], [], linestyle='None', label=f'Tiempo simulado: {simulated_time:.0f}s'),
            ]
            plt.legend(handles=legend_handles, fontsize=20, loc='upper right')

            # Highlight the best performing cell size (lowest execution time)
            best_idx = exec_times.idxmin()
            best_bar = bars[exec_times.index.get_loc(best_idx)]
            best_bar.set_edgecolor('gold')
            best_bar.set_linewidth(4)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '08b_cell_size_comparison_exec_time.png'),  bbox_inches='tight')
    plt.show()

    print("Cell size comparison (execution time) plot generated successfully!")
    print("Generated file: 08b_cell_size_comparison_exec_time.png")

    # Generate LaTeX table for cell size comparison (execution time)
    latex_path = os.path.join(results_dir, 'cell_size_comparison_exec_time_results.tex')
    with open(latex_path, 'w') as f:
        f.write(r'\begin{table}[htbp]' + '\n')
        f.write(r'\centering' + '\n')
        f.write(r'\caption{Tiempo de ejecución (s) por tamaño de celda y número de peatones (RETQSS Opt).}' + '\n')
        f.write(r'\label{tab:cell_size_exec_time}' + '\n')
        f.write(r'\begin{tabular}{ccc}' + '\n')
        f.write(r'\hline' + '\n')
        f.write(r'N & Celda (m) & Tiempo Ejec (s) (mean $\pm$ std) \\' + '\n')
        f.write(r'\hline' + '\n')
        for n in n_values:
            n_data = retqss_opt_data[retqss_opt_data['n_pedestrians'] == n].sort_values('cell_size')
            for _, row in n_data.iterrows():
                cs = row['cell_size']
                exec_m = row['execution_time_mean']
                exec_s = row['execution_time_std']
                f.write(f'{n} & {cs} & {exec_m:.2f} $\\pm$ {exec_s:.2f} \\\\\n')
        f.write(r'\hline' + '\n')
        f.write(r'\end{tabular}' + '\n')
        f.write(r'\end{table}' + '\n')
    print(f"LaTeX table saved to: {latex_path}")

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
            qss_means.append(qss_n['detailed_metrics'].iloc[0]['execution_time'])
            qss_stds.append(qss_n['detailed_metrics'].iloc[0]['execution_time_std'])
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
            retqss_means.append(retqss_n['detailed_metrics'].iloc[0]['execution_time'])
            retqss_stds.append(retqss_n['detailed_metrics'].iloc[0]['execution_time_std'])
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
    
    # Figure 1: Performance comparison (bar chart)
    fig_perf, ax1 = plt.subplots(figsize=(16, 10))
    x = np.arange(len(n_values))
    width = 0.35
    
    # Create performance bars
    bars1 = ax1.bar(x - width/2, qss_means, width, yerr=qss_stds, 
                    capsize=6, alpha=0.8, color='#FF6B6B', edgecolor='#D63031', 
                    linewidth=2, label='QSS', hatch='///')
    
    bars2 = ax1.bar(x + width/2, retqss_means, width, yerr=retqss_stds, 
                    capsize=6, alpha=0.8, color='#4ECDC4', edgecolor='#00B894', 
                    linewidth=2, label='RETQSS', hatch='\\\\\\')
    
    # Customize performance subplot
    ax1.set_xlabel('Número de Peatones (N)', fontsize=20)
    ax1.set_ylabel('Tiempo Promedio de ejecución (segundos)', fontsize=20)
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(n) for n in n_values])
    ax1.set_ylim(0, max(qss_means) + max(qss_stds) + max(qss_means) * 0.7)

    # Add horizontal line at simulated_time for reference
    simulated_time = get_simulated_time()
    ax1.axhline(y=simulated_time, color='green', linestyle='--', linewidth=2, alpha=0.5, label=f'Tiempo simulado: {simulated_time:.0f}s')

    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Save performance figure
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '03a_qss_vs_retqss_performance.png'),  bbox_inches='tight')
    plt.show()

    # Figure 2: Memory usage comparison (line chart)
    fig_mem, ax2 = plt.subplots(figsize=(16, 10))

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
                        ha='center', va='bottom', fontsize=18, fontweight='bold', color='#D63031')
            ax2.annotate(f'{retqss_mem:.1f}MB',
                        xy=(n, retqss_mem), xytext=(n, retqss_mem - max(valid_retqss_memory) * 0.05),
                        ha='center', va='top', fontsize=18, fontweight='bold', color='#00B894')
    else:
        # If no memory data available, show empty plot with message
        ax2.text(0.5, 0.5, 'No hay datos de memoria disponibles',
                ha='center', va='center', transform=ax2.transAxes)

    # Customize memory plot
    ax2.set_xlabel('Número de Peatones (N)', fontsize=20)
    ax2.set_ylabel('Uso Promedio de Memoria (MB)', fontsize=20)
    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)
    
    # Add some styling to both figures
    for ax in [ax1]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
    for ax in [ax2]:
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['bottom'].set_linewidth(1.5)
    
    # Add summary statistics text box for performance
    qss_avg = np.mean([m for m in qss_means if m > 0])
    retqss_avg = np.mean([m for m in retqss_means if m > 0])
    overall_improvement = ((qss_avg - retqss_avg) / qss_avg) * 100 if qss_avg > 0 else 0

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '03b_qss_vs_retqss_memory.png'),  bbox_inches='tight')
    plt.show()
    
    print("QSS vs RETQSS comparison generated successfully!")
    print("Generated files: 03a_qss_vs_retqss_performance.png, 03b_qss_vs_retqss_memory.png")

    # Generate LaTeX table for QSS vs RETQSS comparison
    latex_path = os.path.join(results_dir, 'qss_vs_retqss_results.tex')
    with open(latex_path, 'w') as f:
        f.write(r'\begin{table}[htbp]' + '\n')
        f.write(r'\centering' + '\n')
        f.write(r'\caption{Comparación QSS vs RETQSS: Tiempo de ejecución (s) y memoria (MB).}' + '\n')
        f.write(r'\label{tab:qss_vs_retqss}' + '\n')
        f.write(r'\begin{tabular}{ccccc}' + '\n')
        f.write(r'\hline' + '\n')
        f.write(r'N & QSS Tiempo (s) & RETQSS Tiempo (s) & QSS Memoria (MB) & RETQSS Memoria (MB) \\' + '\n')
        f.write(r'\hline' + '\n')
        for i, n in enumerate(n_values):
            qss_std = qss_stds[i] if i < len(qss_stds) and qss_stds[i] is not None else 0
            retqss_std = retqss_stds[i] if i < len(retqss_stds) and retqss_stds[i] is not None else 0
            qss_m_std = qss_memory_stds[i] if i < len(qss_memory_stds) and qss_memory_stds[i] is not None else 0
            retqss_m_std = retqss_memory_stds[i] if i < len(retqss_memory_stds) and retqss_memory_stds[i] is not None else 0
            qss_m_val = qss_memory[i] if i < len(qss_memory) else 0
            retqss_m_val = retqss_memory[i] if i < len(retqss_memory) else 0
            f.write(f'{n} & {qss_means[i]:.2f} $\\pm$ {qss_std:.2f} & {retqss_means[i]:.2f} $\\pm$ {retqss_std:.2f} & {qss_m_val:.2f} $\\pm$ {qss_m_std:.2f} & {retqss_m_val:.2f} $\\pm$ {retqss_m_std:.2f} \\\\\n')
        f.write(r'\hline' + '\n')
        f.write(r'\end{tabular}' + '\n')
        f.write(r'\end{table}' + '\n')
    print(f"LaTeX table saved to: {latex_path}")

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
    fig, ax = plt.subplots(figsize=(18, 14))
    
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
                        ha='center', va='bottom', fontsize=25, fontweight='bold', color='#D63031')
            ax.annotate(f'{retqss_mem:.1f}MB', 
                        xy=(n, retqss_mem), xytext=(n, retqss_mem - max(valid_retqss_memory) * 0.05),
                        ha='center', va='top', fontsize=25, fontweight='bold', color='#00B894')
            ax.annotate(f'{retqss_opt_mem:.1f}MB', 
                        xy=(n, retqss_opt_mem), xytext=(n, retqss_opt_mem + max(valid_retqss_opt_memory) * 0.05),
                        ha='center', va='bottom', fontsize=25, fontweight='bold', color='#5A4FCF')
    else:
        # If no memory data available, show empty plot with message
        ax.text(0.5, 0.5, 'No hay datos de memoria disponibles', 
                ha='center', va='center', transform=ax.transAxes, fontsize=20)
    
    # Customize the plot
    ax.set_xlabel('Número de Peatones (N)', fontsize=35, fontweight='bold')
    ax.set_ylabel('Uso Promedio de Memoria (MB)', fontsize=35, fontweight='bold')
    ax.legend(fontsize=30, loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', which='major', labelsize=30)
    
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
            
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '04_memory_usage_comparison.png'),  bbox_inches='tight')
    plt.show()

    # Create a second memory usage chart without RETQSS (only QSS vs RETQSS Opt)
    fig2, ax_no_retqss = plt.subplots(figsize=(18, 14))

    valid_indices_2 = [i for i, (qss_mem, retqss_opt_mem) in enumerate(zip(qss_memory, retqss_opt_memory))
                       if qss_mem > 0 and retqss_opt_mem > 0]

    if valid_indices_2:
        valid_n_values_2 = [n_values[i] for i in valid_indices_2]
        valid_qss_memory_2 = [qss_memory[i] for i in valid_indices_2]
        valid_retqss_opt_memory_2 = [retqss_opt_memory[i] for i in valid_indices_2]
        valid_qss_memory_stds_2 = [qss_memory_stds[i] for i in valid_indices_2]
        valid_retqss_opt_memory_stds_2 = [retqss_opt_memory_stds[i] for i in valid_indices_2]

        ax_no_retqss.errorbar(valid_n_values_2, valid_qss_memory_2, yerr=valid_qss_memory_stds_2,
                              fmt='o-', label='RETQSS Base', linewidth=4, markersize=10,
                              color='#FF6B6B', capsize=6, capthick=3, elinewidth=3,
                              alpha=0.8, markeredgecolor='#D63031', markeredgewidth=2)

        ax_no_retqss.errorbar(valid_n_values_2, valid_retqss_opt_memory_2, yerr=valid_retqss_opt_memory_stds_2,
                              fmt='^-', label='RETQSS Opt', linewidth=4, markersize=10,
                              color='#6C5CE7', capsize=6, capthick=3, elinewidth=3,
                              alpha=0.8, markeredgecolor='#5A4FCF', markeredgewidth=2)

        for n, qss_mem, retqss_opt_mem in zip(valid_n_values_2, valid_qss_memory_2, valid_retqss_opt_memory_2):
            ax_no_retqss.annotate(f'{qss_mem:.1f}MB',
                                  xy=(n, qss_mem), xytext=(n, qss_mem + max(valid_qss_memory_2) * 0.05),
                                  ha='center', va='bottom', fontsize=25, fontweight='bold', color='#D63031')
            ax_no_retqss.annotate(f'{retqss_opt_mem:.1f}MB',
                                  xy=(n, retqss_opt_mem), xytext=(n, retqss_opt_mem + max(valid_retqss_opt_memory_2) * 0.05),
                                  ha='center', va='bottom', fontsize=25, fontweight='bold', color='#5A4FCF')
    else:
        ax_no_retqss.text(0.5, 0.5, 'No hay datos de memoria disponibles',
                          ha='center', va='center', transform=ax_no_retqss.transAxes, fontsize=20)

    ax_no_retqss.set_xlabel('Número de Peatones (N)', fontsize=35, fontweight='bold')
    ax_no_retqss.set_ylabel('Uso Promedio de Memoria (MB)', fontsize=35, fontweight='bold')
    ax_no_retqss.legend(fontsize=30, loc='upper left')
    ax_no_retqss.grid(True, alpha=0.3)
    ax_no_retqss.tick_params(axis='both', which='major', labelsize=30)

    ax_no_retqss.spines['top'].set_visible(False)
    ax_no_retqss.spines['right'].set_visible(False)
    ax_no_retqss.spines['left'].set_linewidth(1.5)
    ax_no_retqss.spines['bottom'].set_linewidth(1.5)

    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, '04b_memory_usage_comparison_no_retqss.png'), bbox_inches='tight')
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
    plt.savefig(os.path.join(results_dir, '05_memory_efficiency_comparison.png'),  bbox_inches='tight')
    plt.show()
    
    print("Memory usage comparison plots generated successfully!")
    print("Generated files: 04_memory_usage_comparison.png, 05_memory_efficiency_comparison.png")

def generate_memory_rtf_latex_table(results, optimal_configs=None):
    """
    Generate a LaTeX table with memory per pedestrian and RTF for each N.
    Includes QSS, RETQSS, and RETQSS Opt (best config per N when available).
    """
    df = pd.DataFrame(results)

    # Create results directory if it doesn't exist
    results_dir = 'experiments/performance_n_pedestrians/results'
    os.makedirs(results_dir, exist_ok=True)

    qss_data = df[df['implementation'] == 'qss'].sort_values('n_pedestrians')
    retqss_data = df[df['implementation'] == 'retqss'].sort_values('n_pedestrians')
    retqss_opt_data = df[df['implementation'] == 'retqss_opt'].copy()

    def extract_metrics(series, n):
        if series is None:
            return None, None, None, None, None, None, None
        metrics = series['detailed_metrics']
        if not metrics:
            return None, None, None, None, None, None, None
        rtf = metrics.get('avg_iteration_time')
        rtf_std = metrics.get('std_iteration_time')
        avg_mem = metrics.get('avg_memory_usage')
        mem_std = metrics.get('std_memory_usage')
        mem_per_ped = (avg_mem / n) if avg_mem is not None and n > 0 else None
        mem_per_ped_std = (mem_std / n) if mem_std is not None and n > 0 else None
        abs_perf = metrics.get('execution_time')
        abs_perf_std = metrics.get('execution_time_std')
        sim_time = metrics.get('simulated_time')
        return mem_per_ped, mem_per_ped_std, rtf, rtf_std, abs_perf, abs_perf_std, sim_time

    def format_value(value, digits):
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return "--"
        return f"{value:.{digits}f}"

    def format_mean_std(mean, std, digits):
        if mean is None or (isinstance(mean, float) and np.isnan(mean)):
            return "--"
        if std is None or (isinstance(std, float) and np.isnan(std)):
            return f"{mean:.{digits}f}"
        return f"{mean:.{digits}f}±{std:.{digits}f}"

    n_values = sorted(N_PEDESTRIANS)
    def build_table(title, label, rows):
        table_lines = []
        table_lines.append("\\begin{table}[ht]")
        table_lines.append("\\centering")
        table_lines.append(f"\\caption{{{title}}}")
        table_lines.append(f"\\label{{{label}}}")
        table_lines.append("\\begin{tabular}{r r r r r}")
        table_lines.append("\\hline")
        table_lines.append("N & Tiempo simulado (s) & Mem/peat\\'on (MB) & RTF & Rend. abs (s) \\\\")
        table_lines.append("\\hline")
        table_lines.extend(rows)
        table_lines.append("\\hline")
        table_lines.append("\\end{tabular}")
        table_lines.append("\\end{table}")
        return table_lines

    qss_rows = []
    retqss_rows = []
    retqss_opt_rows = []

    for n in n_values:
        qss_row = qss_data[qss_data['n_pedestrians'] == n]
        retqss_row = retqss_data[retqss_data['n_pedestrians'] == n]

        qss_series = qss_row.iloc[0] if not qss_row.empty else None
        retqss_series = retqss_row.iloc[0] if not retqss_row.empty else None

        qss_mem, qss_mem_std, qss_rtf, qss_rtf_std, qss_abs_perf, qss_abs_perf_std, qss_sim_time = extract_metrics(qss_series, n)
        retqss_mem, retqss_mem_std, retqss_rtf, retqss_rtf_std, retqss_abs_perf, retqss_abs_perf_std, retqss_sim_time = extract_metrics(retqss_series, n)

        # RETQSS Opt: use optimal config if provided, otherwise pick best RTF
        retqss_opt_series = None
        retqss_opt_n = retqss_opt_data[retqss_opt_data['n_pedestrians'] == n]
        if not retqss_opt_n.empty:
            if optimal_configs and n in optimal_configs:
                retqss_opt_n = retqss_opt_n[retqss_opt_n['grid_divisions'] == optimal_configs[n]['grid_divisions']]
                if not retqss_opt_n.empty:
                    retqss_opt_series = retqss_opt_n.iloc[0]
            else:
                retqss_opt_n = retqss_opt_n[retqss_opt_n['detailed_metrics'].notnull()].copy()
                if not retqss_opt_n.empty:
                    retqss_opt_n['rtf'] = retqss_opt_n['detailed_metrics'].apply(
                        lambda x: x['avg_iteration_time'] if x else 0
                    )
                    retqss_opt_series = retqss_opt_n.sort_values('rtf', ascending=False).iloc[0]

        retqss_opt_mem, retqss_opt_mem_std, retqss_opt_rtf, retqss_opt_rtf_std, retqss_opt_abs_perf, retqss_opt_abs_perf_std, retqss_opt_sim_time = extract_metrics(retqss_opt_series, n)

        qss_rows.append(" & ".join([
            str(n),
            format_value(qss_sim_time, 0),
            format_mean_std(qss_mem, qss_mem_std, 6),
            format_mean_std(qss_rtf, qss_rtf_std, 2),
            format_mean_std(qss_abs_perf, qss_abs_perf_std, 2),
        ]) + " \\\\")

        retqss_rows.append(" & ".join([
            str(n),
            format_value(retqss_sim_time, 0),
            format_mean_std(retqss_mem, retqss_mem_std, 6),
            format_mean_std(retqss_rtf, retqss_rtf_std, 2),
            format_mean_std(retqss_abs_perf, retqss_abs_perf_std, 2),
        ]) + " \\\\")

        retqss_opt_rows.append(" & ".join([
            str(n),
            format_value(retqss_opt_sim_time, 0),
            format_mean_std(retqss_opt_mem, retqss_opt_mem_std, 6),
            format_mean_std(retqss_opt_rtf, retqss_opt_rtf_std, 2),
            format_mean_std(retqss_opt_abs_perf, retqss_opt_abs_perf_std, 2),
        ]) + " \\\\")

    lines = []
    lines.extend(build_table(
        "QSS: Memoria por peat\\'on y RTF seg\\'un n\\'umero de peatones.",
        "tab:memoria-rtf-qss",
        qss_rows
    ))
    lines.append("")
    lines.extend(build_table(
        "RETQSS: Memoria por peat\\'on y RTF seg\\'un n\\'umero de peatones.",
        "tab:memoria-rtf-retqss",
        retqss_rows
    ))
    lines.append("")
    lines.extend(build_table(
        "RETQSS Opt: Memoria por peat\\'on y RTF seg\\'un n\\'umero de peatones.",
        "tab:memoria-rtf-retqss-opt",
        retqss_opt_rows
    ))

    output_path = os.path.join(results_dir, '06_memory_rtf_table.tex')
    with open(output_path, 'w') as f:
        f.write("\n".join(lines))

    print("LaTeX table generated successfully!")
    print(f"Generated file: {output_path}")

if __name__ == '__main__':
    performance_n_pedestrians()
