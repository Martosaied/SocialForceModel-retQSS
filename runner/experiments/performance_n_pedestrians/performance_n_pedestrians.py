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
    0: "mmoc",
    1: "retqss",
}

def performance_n_pedestrians():
    """
    Enhanced performance testing for both MMOC and RETQSS implementations.
    Tests different numbers of pedestrians (N) and grid divisions (M).
    For RETQSS: tests all combinations of N and M.
    """
    print("Running comprehensive performance experiments...")
    print(f"Testing {len(N_PEDESTRIANS)} different N values: {N_PEDESTRIANS}")
    print(f"Testing {len(GRID_DIVISIONS)} different M values: {GRID_DIVISIONS}")
    
    # Calculate total experiments
    mmoc_experiments = len(N_PEDESTRIANS)  # MMOC only varies N
    retqss_experiments = len(N_PEDESTRIANS) * len(GRID_DIVISIONS)  # RETQSS varies both N and M
    total_experiments = mmoc_experiments + retqss_experiments
    
    print(f"Total experiments: {total_experiments}")
    print(f"  - MMOC: {mmoc_experiments} experiments")
    print(f"  - RETQSS: {retqss_experiments} experiments")
    print("="*60)
    
    results = []

    # Update configuration from command line arguments
    config_manager.update_from_dict({
        'skip_metrics': True
    })
    
    # Test MMOC implementation with different N values
    print("\n1. Testing MMOC implementation with different N values...")
    for i, n in enumerate(N_PEDESTRIANS, 1):
        print(f"   [{i}/{mmoc_experiments}] Running MMOC with N={n}...")
        result = run_experiment_with_params(n, 0, 10)  # MMOC doesn't use grid divisions
        results.append(result)
        print(f"   Completed")
    
    # Test RETQSS implementation with all combinations of N and M
    print(f"\n2. Testing RETQSS implementation with all combinations of N and M...")
    experiment_count = 0
    for n in N_PEDESTRIANS:
        for m in GRID_DIVISIONS:
            experiment_count += 1
            print(f"   [{experiment_count}/{retqss_experiments}] Running RETQSS with N={n}, M={m}...")
            result = run_experiment_with_params(n, 1, m)
            results.append(result)
            print(f"   Completed")
    
    # Plot comprehensive results
    print("\n3. Generating comprehensive plots...")
    plot_comprehensive_results(results)
    
    print("\n" + "="*60)
    print("All experiments completed successfully!")
    print("Results saved to CSV and plots generated.")

def run_experiment_with_params(n, implementation, grid_divisions):
    """
    Run a single experiment with specified parameters and return timing results.
    """
    # config = load_config('./experiments/performance_n_pedestrians/config.json')

    # # Create descriptive experiment name
    impl_name = PEDESTRIANS_IMPLEMENTATION[implementation]
    if implementation == 0:  # MMOC
        exp_name = f'n_{n}_mmoc'
    else:  # RETQSS
        exp_name = f'n_{n}_retqss_m_{grid_divisions}'
    
    # # Create output directory
    output_dir = create_output_dir(
        'experiments/performance_n_pedestrians/results', 
        exp_name
    )
    
    # Update config parameters
    config['iterations'] = 10
    config['parameters']['N']['value'] = n
    config['parameters']['PEDESTRIAN_IMPLEMENTATION']['value'] = implementation
    config['parameters']['BORDER_IMPLEMENTATION']['value'] = -1 # no border

    # Save config copy in experiment directory
    config_copy_path = os.path.join(output_dir, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Update model file parameters
    model_path = '../retqss/model/social_force_model.mo'
    subprocess.run(['sed', '-i', r's/\bN\s*=\s*[0-9]\+/N = ' + str(n) + '/', model_path])
    subprocess.run(['sed', '-i', r's/\bGRID_DIVISIONS\s*=\s*[0-9]\+/GRID_DIVISIONS = ' + str(grid_divisions) + '/', model_path])

    # Compile the C++ code and model
    compile_c_code()
    compile_model('social_force_model')

    # Run experiment
    run_experiment(
        config, 
        output_dir, 
        'social_force_model', 
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

    print(detailed_metrics)
    
    return {
        'n_pedestrians': n,
        'implementation': impl_name,
        'grid_divisions': grid_divisions,
        'output_dir': output_dir,
        'detailed_metrics': detailed_metrics
    }

def plot_comprehensive_results(results):
    """
    Generate comprehensive plots comparing MMOC vs RETQSS performance.
    """
    df = pd.DataFrame(results)
    
    # Create results directory if it doesn't exist
    results_dir = 'experiments/performance_n_pedestrians/results'
    os.makedirs(results_dir, exist_ok=True)
    
    # Filter data
    mmoc_data = df[(df['implementation'] == 'mmoc')].sort_values('n_pedestrians')
    retqss_data = df[df['implementation'] == 'retqss']
    
    # Create a single comprehensive plot
    plt.figure(figsize=(14, 10))
    
    # Plot 1: Main comparison - MMOC vs RETQSS with different M values
    plt.subplot(2, 2, 1)
    
    # Plot MMOC baseline
    mmoc_times = mmoc_data['detailed_metrics'].apply(lambda x: x['avg_iteration_time'] if x else 0)
    mmoc_stds = mmoc_data['detailed_metrics'].apply(lambda x: x['std_iteration_time'] if x else 0)
    plt.errorbar(mmoc_data['n_pedestrians'], mmoc_times, yerr=mmoc_stds,
                fmt='o-', label='MMOC', linewidth=3, markersize=8, color='blue', capsize=4)
    
    # Plot RETQSS for different M values, showing improvement with more divisions
    colors = plt.cm.viridis(np.linspace(0, 1, len(GRID_DIVISIONS)))
    for i, m in enumerate(sorted(GRID_DIVISIONS)):
        retqss_m_data = retqss_data[retqss_data['grid_divisions'] == m].sort_values('n_pedestrians')
        if not retqss_m_data.empty:
            retqss_times = retqss_m_data['detailed_metrics'].apply(lambda x: x['avg_iteration_time'] if x else 0)
            retqss_stds = retqss_m_data['detailed_metrics'].apply(lambda x: x['std_iteration_time'] if x else 0)
            plt.errorbar(retqss_m_data['n_pedestrians'], retqss_times, yerr=retqss_stds,
                        fmt='s-', label=f'RETQSS M={m}', linewidth=2, markersize=6, 
                        color=colors[i], capsize=3, alpha=0.8)
    
    plt.xlabel('Number of Pedestrians (N)')
    plt.ylabel('Average Execution Time (s)')
    plt.title('Performance Comparison: MMOC vs RETQSS with Different Grid Divisions')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot 2: Speedup comparison - How much better RETQSS is vs MMOC
    plt.subplot(2, 2, 2)
    
    for i, m in enumerate(sorted(GRID_DIVISIONS)):
        retqss_m_data = retqss_data[retqss_data['grid_divisions'] == m].sort_values('n_pedestrians')
        if not retqss_m_data.empty:
            # Merge with MMOC data to calculate speedup
            merged_data = pd.merge(mmoc_data, retqss_m_data, on='n_pedestrians', suffixes=('_mmoc', '_retqss'))
            mmoc_times_merged = merged_data['detailed_metrics_mmoc'].apply(lambda x: x['avg_iteration_time'] if x else 0)
            retqss_times_merged = merged_data['detailed_metrics_retqss'].apply(lambda x: x['avg_iteration_time'] if x else 0)
            merged_data['speedup'] = mmoc_times_merged / retqss_times_merged
            
            plt.plot(merged_data['n_pedestrians'], merged_data['speedup'], 
                    's-', label=f'RETQSS M={m}', linewidth=2, markersize=6, 
                    color=colors[i], alpha=0.8)
    
    plt.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Equal Performance')
    plt.xlabel('Number of Pedestrians (N)')
    plt.ylabel('Speedup (MMOC Time / RETQSS Time)')
    plt.title('Speedup: How Much Better RETQSS is vs MMOC')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Plot 3: Best RETQSS performance for each N
    plt.subplot(2, 2, 3)
    
    # Find best RETQSS configuration for each N
    retqss_data['avg_time'] = retqss_data['detailed_metrics'].apply(lambda x: x['avg_iteration_time'] if x else float('inf'))
    best_retqss_per_n = retqss_data.loc[retqss_data.groupby('n_pedestrians')['avg_time'].idxmin()]
    best_retqss_per_n = best_retqss_per_n.sort_values('n_pedestrians')
    
    if not best_retqss_per_n.empty:
        # Plot best RETQSS vs MMOC
        best_retqss_times = best_retqss_per_n['detailed_metrics'].apply(lambda x: x['avg_iteration_time'] if x else 0)
        best_retqss_stds = best_retqss_per_n['detailed_metrics'].apply(lambda x: x['std_iteration_time'] if x else 0)
        plt.errorbar(best_retqss_per_n['n_pedestrians'], best_retqss_times, yerr=best_retqss_stds,
                    fmt='o-', label='Best RETQSS', linewidth=3, markersize=8, color='red', capsize=4)
        
        plt.errorbar(mmoc_data['n_pedestrians'], mmoc_times, yerr=mmoc_stds,
                    fmt='s-', label='MMOC', linewidth=3, markersize=8, color='blue', capsize=4)
        
        plt.xlabel('Number of Pedestrians (N)')
        plt.ylabel('Average Execution Time (s)')
        plt.title('Best RETQSS Configuration vs MMOC')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Plot 4: Optimal M for each N
    plt.subplot(2, 2, 4)
    
    if not best_retqss_per_n.empty:
        plt.plot(best_retqss_per_n['n_pedestrians'], best_retqss_per_n['grid_divisions'], 
                'o-', color='green', linewidth=3, markersize=8)
        plt.xlabel('Number of Pedestrians (N)')
        plt.ylabel('Optimal Grid Divisions (M)')
        plt.title('Optimal Grid Divisions for Each N')
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'comprehensive_performance_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()




if __name__ == '__main__':
    performance_n_pedestrians()
