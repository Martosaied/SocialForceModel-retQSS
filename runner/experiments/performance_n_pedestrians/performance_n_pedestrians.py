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

# Test configurations
N_PEDESTRIANS = [10, 50, 100, 200, 300, 500, 1000, 2000, 3000]
GRID_DIVISIONS = [1, 2, 3, 5, 10, 15, 20, 25, 30, 40, 50]  # Different M values for RETQSS

PEDESTRIANS_IMPLEMENTATION = {
    0: "helbing_not_qss",  # Helbing sin QSS
    1: "mmoc",    # RETQSS sin optimizaciones  
    2: "retqss", # RETQSS con optimizaciones
}

def performance_n_pedestrians():
    """
    Enhanced performance testing for three model implementations:
    1. Helbing without QSS (baseline)
    2. RETQSS without optimizations 
    3. RETQSS with optimizations (best grid size)
    """
    print("Running comprehensive performance experiments...")
    print(f"Testing {len(N_PEDESTRIANS)} different N values: {N_PEDESTRIANS}")
    print(f"Testing {len(GRID_DIVISIONS)} different M values: {GRID_DIVISIONS}")
    
    # Calculate total experiments
    helbing_experiments = len(N_PEDESTRIANS)  # Helbing only varies N
    retqss_no_opt_experiments = len(N_PEDESTRIANS)  # RETQSS no opt only varies N
    retqss_opt_experiments = len(N_PEDESTRIANS) * len(GRID_DIVISIONS)  # RETQSS opt varies both N and M
    total_experiments = helbing_experiments + retqss_no_opt_experiments + retqss_opt_experiments
    
    print(f"Total experiments: {total_experiments}")
    print(f"  - Helbing (no QSS): {helbing_experiments} experiments")
    print(f"  - RETQSS (no opt): {retqss_no_opt_experiments} experiments")
    print(f"  - RETQSS (optimized): {retqss_opt_experiments} experiments")
    print("="*60)
    
    results = []

    # Update configuration from command line arguments
    config_manager.update_from_dict({
        'skip_metrics': True
    })
    
    # Phase 1: Test Helbing without QSS (baseline)
    print("\n1. Testing Helbing without QSS (baseline)...")
    for i, n in enumerate(N_PEDESTRIANS, 1):
        print(f"   [{i}/{helbing_experiments}] Running Helbing (no QSS) with N={n}...")
        result = run_experiment_with_params(n, 0, 1)  # Helbing doesn't use grid divisions
        results.append(result)
        print(f"   Completed")
    
    # Phase 2: Test RETQSS without optimizations
    print("\n2. Testing RETQSS without optimizations...")
    for i, n in enumerate(N_PEDESTRIANS, 1):
        print(f"   [{i}/{retqss_no_opt_experiments}] Running RETQSS (no opt) with N={n}...")
        result = run_experiment_with_params(n, 1, 1)  # Fixed grid divisions = 1
        results.append(result)
        print(f"   Completed")
    
    # Phase 3: Test RETQSS with optimizations (all grid sizes)
    print(f"\n3. Testing RETQSS with optimizations (all grid sizes)...")
    experiment_count = 0
    for n in N_PEDESTRIANS:
        for m in GRID_DIVISIONS:
            experiment_count += 1
            print(f"   [{experiment_count}/{retqss_opt_experiments}] Running RETQSS (optimized) with N={n}, M={m}...")
            result = run_experiment_with_params(n, 2, m)
            results.append(result)
            print(f"   Completed")
    
    # Phase 4: Find optimal grid size for each N
    print("\n4. Finding optimal grid size for each N...")
    optimal_configs = find_optimal_grid_sizes(results)
    
    # Phase 5: Generate comprehensive plots
    print("\n5. Generating comprehensive plots...")
    plot_comprehensive_results(results, optimal_configs)
    
    print("\n" + "="*60)
    print("All experiments completed successfully!")
    print("Results saved to CSV and plots generated.")

def run_experiment_with_params(n, implementation, grid_divisions):
    """
    Run a single experiment with specified parameters and return timing results.
    """
    config = load_config('./experiments/performance_n_pedestrians/config.json')

    # Create descriptive experiment name
    impl_name = PEDESTRIANS_IMPLEMENTATION[implementation]
    if implementation == 0:  # Helbing without QSS
        exp_name = f'n_{n}_helbing_not_qss'
        model_name = 'helbing_not_qss'
    elif implementation == 1:  # RETQSS without optimizations
        exp_name = f'n_{n}_mmoc'
        model_name = 'social_force_model'
    else:  # RETQSS with optimizations
        exp_name = f'n_{n}_retqss_m_{grid_divisions}'
        model_name = 'social_force_model'
    
    # Create output directory
    output_dir = create_output_dir(
        'experiments/performance_n_pedestrians/results', 
        exp_name
    )
    
    # Update config parameters
    config['iterations'] = 1
    config['parameters']['N']['value'] = n
    config['parameters']['PEDESTRIAN_IMPLEMENTATION']['value'] = implementation
    config['parameters']['BORDER_IMPLEMENTATION']['value'] = -1 # no border

    # Save config copy in experiment directory
    config_copy_path = os.path.join(output_dir, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Update model file parameters based on implementation
    if implementation == 0:  # Helbing without QSS
        model_path = '../retqss/model/helbing_not_qss.mo'
        subprocess.run(['sed', '-i', r's/\bN\s*=\s*[0-9]\+/N = ' + str(n) + '/', model_path])
    else:  # RETQSS implementations
        model_path = '../retqss/model/social_force_model.mo'
        subprocess.run(['sed', '-i', r's/\bN\s*=\s*[0-9]\+/N = ' + str(n) + '/', model_path])
        subprocess.run(['sed', '-i', r's/\bGRID_DIVISIONS\s*=\s*[0-9]\+/GRID_DIVISIONS = ' + str(grid_divisions) + '/', model_path])

    # # Compile the C++ code and model
    # compile_c_code()
    # compile_model(model_name)

    # # Run experiment
    # run_experiment(
    #     config, 
    #     output_dir, 
    #     model_name, 
    #     plot=False, 
    #     copy_results=False
    # )

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
        'grid_divisions': grid_divisions,
        'output_dir': output_dir,
        'detailed_metrics': detailed_metrics
    }

def find_optimal_grid_sizes(results):
    """
    Find the optimal grid size for each number of pedestrians based on RETQSS optimized results.
    """
    df = pd.DataFrame(results)
    
    # Filter only RETQSS optimized results
    retqss_opt_data = df[df['implementation'] == 'retqss'].copy()
    
    if retqss_opt_data.empty:
        print("Warning: No RETQSS optimized results found for optimization analysis")
        return {}
    
    # Add average time column for easier analysis
    retqss_opt_data['avg_time'] = retqss_opt_data['detailed_metrics'].apply(
        lambda x: x['avg_iteration_time'] if x else float('inf')
    )
    
    # Find optimal grid size for each N (minimum average time)
    optimal_configs = {}
    for n in N_PEDESTRIANS:
        n_data = retqss_opt_data[retqss_opt_data['n_pedestrians'] == n]
        if not n_data.empty:
            best_idx = n_data['avg_time'].idxmin()
            best_config = n_data.loc[best_idx]
            optimal_configs[n] = {
                'grid_divisions': best_config['grid_divisions'],
                'avg_time': best_config['avg_time'],
                'std_time': best_config['detailed_metrics']['std_iteration_time'] if best_config['detailed_metrics'] else 0
            }
            print(f"  N={n}: Optimal M={best_config['grid_divisions']} (avg_time={best_config['avg_time']:.4f}s)")
    
    return optimal_configs

def plot_comprehensive_results(results, optimal_configs=None):
    """
    Generate comprehensive plots comparing three model implementations:
    1. Helbing without QSS (baseline)
    2. RETQSS without optimizations
    3. RETQSS with optimizations (best grid size)
    """
    df = pd.DataFrame(results)
    
    # Create results directory if it doesn't exist
    results_dir = 'experiments/performance_n_pedestrians/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Filter data by implementation
    helbing_data = df[df['implementation'] == 'helbing_not_qss'].sort_values('n_pedestrians')
    retqss_no_opt_data = df[df['implementation'] == 'mmoc'].sort_values('n_pedestrians')
    retqss_opt_data = df[df['implementation'] == 'retqss']
    
    # Create a single comprehensive plot
    plt.figure(figsize=(16, 12))
    
    # Plot 1: Main comparison - All three implementations
    plt.subplot(2, 3, 1)
    
    # Plot Helbing baseline
    helbing_times = helbing_data['detailed_metrics'].apply(lambda x: x['avg_iteration_time'] if x else 0)
    helbing_stds = helbing_data['detailed_metrics'].apply(lambda x: x['std_iteration_time'] if x else 0)
    plt.errorbar(helbing_data['n_pedestrians'], helbing_times, yerr=helbing_stds,
                fmt='o-', label='Helbing (sin QSS)', linewidth=3, markersize=8, color='blue', capsize=4)
    
    # Plot RETQSS without optimizations
    retqss_no_opt_times = retqss_no_opt_data['detailed_metrics'].apply(lambda x: x['avg_iteration_time'] if x else 0)
    retqss_no_opt_stds = retqss_no_opt_data['detailed_metrics'].apply(lambda x: x['std_iteration_time'] if x else 0)
    plt.errorbar(retqss_no_opt_data['n_pedestrians'], retqss_no_opt_times, yerr=retqss_no_opt_stds,
                fmt='s-', label='RETQSS (sin opt)', linewidth=3, markersize=8, color='orange', capsize=4)
    
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
                        fmt='^-', label='RETQSS (optimizado)', linewidth=3, markersize=8, color='red', capsize=4)
    
    plt.xlabel('Número de Peatones (N)')
    plt.ylabel('Tiempo Promedio de Ejecución (s)')
    plt.title('Comparación de Rendimiento: Tres Implementaciones de Modelos')
    plt.legend()
    plt.grid(True, alpha=0.3)
    # plt.yscale('log')  # Changed to linear scale
    
    # Plot 2: Speedup comparison - RETQSS vs Helbing
    plt.subplot(2, 3, 2)
    
    # Calculate speedup for RETQSS no opt vs Helbing
    merged_no_opt = pd.merge(helbing_data, retqss_no_opt_data, on='n_pedestrians', suffixes=('_helbing', '_mmoc'))
    helbing_times_merged = merged_no_opt['detailed_metrics_helbing'].apply(lambda x: x['avg_iteration_time'] if x else 0)
    retqss_no_opt_times_merged = merged_no_opt['detailed_metrics_mmoc'].apply(lambda x: x['avg_iteration_time'] if x else 0)
    merged_no_opt['speedup'] = helbing_times_merged / retqss_no_opt_times_merged
    
    plt.plot(merged_no_opt['n_pedestrians'], merged_no_opt['speedup'], 
            's-', label='RETQSS (sin opt) vs Helbing', linewidth=2, markersize=6, color='orange')
    
    # Calculate speedup for RETQSS optimized vs Helbing
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
            merged_opt = pd.merge(helbing_data, retqss_opt_best_df, on='n_pedestrians', suffixes=('_helbing', '_retqss_opt'))
            helbing_times_opt = merged_opt['detailed_metrics_helbing'].apply(lambda x: x['avg_iteration_time'] if x else 0)
            retqss_opt_times_opt = merged_opt['detailed_metrics_retqss_opt'].apply(lambda x: x['avg_iteration_time'] if x else 0)
            merged_opt['speedup'] = helbing_times_opt / retqss_opt_times_opt
            
            plt.plot(merged_opt['n_pedestrians'], merged_opt['speedup'], 
                    '^-', label='RETQSS (optimizado) vs Helbing', linewidth=2, markersize=6, color='red')
    
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Rendimiento Igual')
    plt.xlabel('Número de Peatones (N)')
    plt.ylabel('Aceleración (Tiempo Helbing / Tiempo RETQSS)')
    plt.title('Aceleración: RETQSS vs Helbing')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Optimization impact - RETQSS optimized vs RETQSS no opt
    plt.subplot(2, 3, 3)
    
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
            merged_opt_vs_no_opt = pd.merge(retqss_no_opt_data, retqss_opt_best_df, on='n_pedestrians', suffixes=('_no_opt', '_opt'))
            retqss_no_opt_times_comp = merged_opt_vs_no_opt['detailed_metrics_no_opt'].apply(lambda x: x['avg_iteration_time'] if x else 0)
            retqss_opt_times_comp = merged_opt_vs_no_opt['detailed_metrics_opt'].apply(lambda x: x['avg_iteration_time'] if x else 0)
            merged_opt_vs_no_opt['speedup'] = retqss_no_opt_times_comp / retqss_opt_times_comp
            
            plt.plot(merged_opt_vs_no_opt['n_pedestrians'], merged_opt_vs_no_opt['speedup'], 
                    'o-', label='Impacto de Optimización', linewidth=2, markersize=6, color='green')
    
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Sin Mejora')
    plt.xlabel('Número de Peatones (N)')
    plt.ylabel('Aceleración (Tiempo Sin Opt / Tiempo Optimizado)')
    plt.title('Impacto de Optimización: RETQSS Optimizado vs Sin Opt')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 4: Optimal grid divisions for each N
    plt.subplot(2, 3, 4)
    
    if optimal_configs:
        n_values = list(optimal_configs.keys())
        m_values = [optimal_configs[n]['grid_divisions'] for n in n_values]
        plt.plot(n_values, m_values, 'o-', color='purple', linewidth=3, markersize=8)
        plt.xlabel('Número de Peatones (N)')
        plt.ylabel('Divisiones Óptimas de Grilla (M)')
        plt.title('Divisiones Óptimas de Grilla para Cada N')
        plt.grid(True, alpha=0.3)
    
    # Plot 5: Performance vs Grid Divisions (for selected N values)
    plt.subplot(2, 3, 5)
    
    selected_n_values = [100, 500, 1000, 2000]  # Show a few representative N values
    colors = plt.cm.viridis(np.linspace(0, 1, len(selected_n_values)))
    
    for i, n in enumerate(selected_n_values):
        n_data = retqss_opt_data[retqss_opt_data['n_pedestrians'] == n].sort_values('grid_divisions')
        if not n_data.empty:
            times = n_data['detailed_metrics'].apply(lambda x: x['avg_iteration_time'] if x else 0)
            stds = n_data['detailed_metrics'].apply(lambda x: x['std_iteration_time'] if x else 0)
            plt.errorbar(n_data['grid_divisions'], times, yerr=stds,
                        fmt='o-', label=f'N={n}', linewidth=2, markersize=6, 
                        color=colors[i], capsize=3, alpha=0.8)
    
    plt.xlabel('Divisiones de Grilla (M)')
    plt.ylabel('Tiempo Promedio de Ejecución (s)')
    plt.title('Rendimiento vs Divisiones de Grilla')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    
    # Plot 6: Summary statistics
    plt.subplot(2, 3, 6)
    
    # Calculate summary statistics
    summary_data = []
    for impl in ['helbing_not_qss', 'mmoc', 'retqss']:
        impl_data = df[df['implementation'] == impl]
        if not impl_data.empty:
            times = impl_data['detailed_metrics'].apply(lambda x: x['avg_iteration_time'] if x else 0)
            summary_data.append({
                'Implementation': impl.replace('_', ' ').title(),
                'Avg Time': times.mean(),
                'Min Time': times.min(),
                'Max Time': times.max()
            })
    
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        x_pos = range(len(summary_df))
        plt.bar(x_pos, summary_df['Avg Time'], alpha=0.7, color=['blue', 'orange', 'red'])
        plt.xlabel('Implementación')
        plt.ylabel('Tiempo Promedio de Ejecución (s)')
        plt.title('Resumen: Rendimiento Promedio')
        plt.xticks(x_pos, summary_df['Implementation'], rotation=45)
        plt.grid(True, alpha=0.3)
        plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'comprehensive_three_model_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()




if __name__ == '__main__':
    performance_n_pedestrians()
