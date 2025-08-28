import json
import os
import subprocess
from src.runner import run_experiment, compile_c_code, compile_model
from src.utils import load_config, create_output_dir, copy_results_to_latest
from src.config_manager import print_config_status, get_performance_mode
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob

def check_experiment_flags():
    """Check and display the current experiment flags."""
    print_config_status()

def get_experiment_flags():
    """Get the current experiment flags as a dictionary."""
    from src.config_manager import get_config
    return get_config().to_dict()

def subway_attack_rate():
    """
    Run the subway_hub model experiment comparing social force model with/without
    across different population sizes (N: 300, 500, 700).
    Keeps OBJECTIVE_SUBWAY_HUB_DT at 400 for consistent station changes.
    """
    print("Running subway hub model with social force comparison across different durations...\n")
    
    # Check and display experiment flags
    check_experiment_flags()

    # # Define the different termination times to test
    # termination_times = [1800, 1200, 900]  # 30min, 20min, 15min
    
    # output_dirs = {}
    
    # # Run experiments for each duration and social force configuration
    # for termination_time in termination_times:
    #     print(f"\n=== Testing FORCE_TERMINATION_AT = {termination_time}s ({termination_time/60:.1f} min) ===")
        
    #     # Run without social force model
    #     output_dirs[f'{termination_time}s_no_sf'] = run_single_experiment(termination_time, social_force=0)
        
    #     # Run with social force model
    #     output_dirs[f'{termination_time}s_with_sf'] = run_single_experiment(termination_time, social_force=1)
    
    # # Analyze and plot comparative results
    # print("\nAnalyzing comparative results...")
    # plot_comparative_results(output_dirs, termination_times, "duration")
    
    # Run population size experiment
    print("\n" + "="*80)
    print("RUNNING POPULATION SIZE EXPERIMENT")
    print("="*80)
    run_population_size_experiment()
    
    print(f"\nComparative experiment completed. Results saved in experiments/subway_attack_rate/results")


def run_population_size_experiment():
    """
    Run experiment comparing social force model with/without across different population sizes.
    Tests N values: 300, 500, 700 with PEDESTRIANS_COUNT = N - 20.
    Keeps FORCE_TERMINATION_AT fixed at 1800s.
    """
    print("Running population size experiment...\n")
    
    # Define the different population sizes to test
    population_sizes = [300, 500, 700]  # N values
    termination_time = 900  # Fixed at 15 minutes
    
    output_dirs = {}
    
    # Run experiments for each population size and social force configuration
    for n in population_sizes:
        pedestrians_count = n - 20  # Always difference of 20
        print(f"\n=== Testing N = {n}, PEDESTRIANS_COUNT = {pedestrians_count} ===")
        
        # Run without social force model
        output_dirs[f'N{n}_no_sf'] = run_population_experiment(n, pedestrians_count, termination_time, social_force=0)
        
        # Run with social force model
        output_dirs[f'N{n}_with_sf'] = run_population_experiment(n, pedestrians_count, termination_time, social_force=1)
    
    # Analyze and plot comparative results
    print("\nAnalyzing population size comparative results...")
    plot_comparative_results(output_dirs, population_sizes, "population")


def run_population_experiment(n, pedestrians_count, termination_time, social_force):
    """
    Run the subway_hub model experiment with specific population parameters.
    
    Args:
        n: Total population size (N)
        pedestrians_count: Number of pedestrians (PEDESTRIANS_COUNT)
        termination_time: FORCE_TERMINATION_AT value in seconds
        social_force: 0 for disabled, 1 for enabled
    
    Returns:
        Output directory path
    """
    sf_label = "with_sf" if social_force else "no_sf"
    print(f"Running experiment: N={n}, PEDESTRIANS_COUNT={pedestrians_count}, social force {'ENABLED' if social_force else 'DISABLED'}...")
    
    # Load the base configuration
    base_config = load_config('./experiments/subway_attack_rate/subway_hub.json')
    
    # Modify configuration
    config = base_config.copy()
    config['parameters']['FORCE_TERMINATION_AT']['value'] = termination_time
    config['parameters']['SOCIAL_FORCE_MODEL']['value'] = social_force
    config['parameters']['N']['value'] = n
    
    # Ensure OBJECTIVE_SUBWAY_HUB_DT is set to 600 (10 minutes)
    if 'OBJECTIVE_SUBWAY_HUB_DT' not in config['parameters']:
        config['parameters']['OBJECTIVE_SUBWAY_HUB_DT'] = {
            "name": "OBJECTIVE_SUBWAY_HUB_DT",
            "type": "value",
            "value": 400.0
        }
    else:
        config['parameters']['OBJECTIVE_SUBWAY_HUB_DT']['value'] = 400.0
    
    # Create output directory
    output_dir = create_output_dir(
        'experiments/subway_attack_rate/results',
        f'subway_N{n}_ped{pedestrians_count}_{sf_label}'
    )
    
    # Save config copy
    config_copy_path = os.path.join(output_dir, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Modify the model file using sed commands
    print(f"Modifying model file: N={n}, PEDESTRIANS_COUNT={pedestrians_count}")
    subprocess.run(['sed', '-i', r's/\bN\s*=\s*[0-9]\+/N = ' + str(n) + '/', '../retqss/model/subway_hub.mo'])
    subprocess.run(['sed', '-i', r's/\bPEDESTRIANS_COUNT\s*=\s*[0-9]\+/PEDESTRIANS_COUNT = ' + str(pedestrians_count) + '/', '../retqss/model/subway_hub.mo'])
    
    # Compile and run
    compile_c_code()
    compile_model('subway_hub')
    
    run_experiment(
        config,
        output_dir,
        'subway_hub',
        plot=False,
        copy_results=True
    )

    # remove all other columns from the result_*.csv files.
    for result_file in glob.glob(os.path.join(output_dir, 'result_*.csv')):
        df = pd.read_csv(result_file)
        df = df[['PS[1]']]
        df.to_csv(result_file, index=False)
    
    return output_dir


def run_single_experiment(termination_time, social_force):
    """
    Run the subway_hub model experiment with specific parameters.
    
    Args:
        termination_time: FORCE_TERMINATION_AT value in seconds
        social_force: 0 for disabled, 1 for enabled
    
    Returns:
        Output directory path
    """
    sf_label = "with_sf" if social_force else "no_sf"
    print(f"Running experiment: {termination_time}s, social force {'ENABLED' if social_force else 'DISABLED'}...")
    
    # Load the base configuration
    base_config = load_config('./experiments/subway_attack_rate/subway_hub.json')
    
    # Modify configuration
    config = base_config.copy()
    config['parameters']['FORCE_TERMINATION_AT']['value'] = termination_time
    config['parameters']['SOCIAL_FORCE_MODEL']['value'] = social_force
    
    # Ensure OBJECTIVE_SUBWAY_HUB_DT is set to 600 (10 minutes)
    if 'OBJECTIVE_SUBWAY_HUB_DT' not in config['parameters']:
        config['parameters']['OBJECTIVE_SUBWAY_HUB_DT'] = {
            "name": "OBJECTIVE_SUBWAY_HUB_DT",
            "type": "value",
            "value": 400.0
        }
    else:
        config['parameters']['OBJECTIVE_SUBWAY_HUB_DT']['value'] = 400.0
    
    # Create output directory
    output_dir = create_output_dir(
        'experiments/subway_attack_rate/results',
        f'subway_{termination_time}s_{sf_label}'
    )
    
    # Save config copy
    config_copy_path = os.path.join(output_dir, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    # Compile and run
    compile_c_code()
    compile_model('subway_hub')
    
    run_experiment(
        config,
        output_dir,
        'subway_hub',
        plot=False,
        copy_results=True
    )

    # remove all other columns from the result_*.csv files.
    for result_file in glob.glob(os.path.join(output_dir, 'result_*.csv')):
        df = pd.read_csv(result_file)
        df = df[['PS[1]']]
        df.to_csv(result_file, index=False)
    
    return output_dir


def analyze_particle_1_exposure_rate(results_dir):
    """
    Analyze the exposure rate of particle 1 across all iterations.
    
    Args:
        results_dir: Directory containing result_*.csv files
    
    Returns:
        Dictionary with particle 1 exposure statistics
    """
    result_files = glob.glob(os.path.join(results_dir, 'result_*.csv'))

    if not result_files:
        raise FileNotFoundError(f"No result files found in {results_dir}")
    
    exposed_count = 0
    total_iterations = len(result_files)
    
    for result_file in result_files:
        try:
            df = pd.read_csv(result_file)
            
            if 'PS[1]' in df.columns and len(df) > 0:
                final_state = df['PS[1]'].iloc[-1]
                if pd.notna(final_state) and final_state == 1:  # EXPOSED state
                    exposed_count += 1
        except Exception as e:
            print(f"Warning: Could not process {result_file}: {e}")
            continue
    
    exposure_rate = exposed_count / total_iterations if total_iterations > 0 else 0
    exposure_rate_pct = exposure_rate * 100
    
    return {
        'total_iterations': total_iterations,
        'exposed_count': exposed_count,
        'exposure_rate': exposure_rate,
        'exposure_rate_pct': exposure_rate_pct
    }


def plot_comparative_results(output_dirs, test_values, experiment_type):
    """
    Create simple comparative plots showing the differences.
    
    Args:
        output_dirs: Dictionary containing paths to experiment results
        test_values: List of test values (durations or population sizes)
        experiment_type: "duration" or "population"
    """
    print(f"Creating {experiment_type} comparative plots...")
    
    # Collect data for all experiments
    data = {}
    
    if experiment_type == "duration":
        for test_value in test_values:
            try:
                # Get results for both social force configurations
                no_sf_results_dir = os.path.join(os.path.dirname(output_dirs[f'{test_value}s_no_sf']), 'latest')
                with_sf_results_dir = os.path.join(os.path.dirname(output_dirs[f'{test_value}s_with_sf']), 'latest')
                
                data[f'{test_value}s'] = {
                    'no_sf': analyze_particle_1_exposure_rate(no_sf_results_dir),
                    'with_sf': analyze_particle_1_exposure_rate(with_sf_results_dir)
                }
            except Exception as e:
                print(f"Warning: Could not analyze data for {test_value}s: {e}")
                continue
    else:  # population
        for test_value in test_values:
            try:
                # Get results for both social force configurations
                no_sf_results_dir = os.path.join(os.path.dirname(output_dirs[f'N{test_value}_no_sf']), 'latest')
                with_sf_results_dir = os.path.join(os.path.dirname(output_dirs[f'N{test_value}_with_sf']), 'latest')
                
                data[f'N{test_value}'] = {
                    'no_sf': analyze_particle_1_exposure_rate(no_sf_results_dir),
                    'with_sf': analyze_particle_1_exposure_rate(with_sf_results_dir)
                }
            except Exception as e:
                print(f"Warning: Could not analyze data for N{test_value}: {e}")
                continue
    
    if not data:
        print("Error: No data available for plotting")
        return
    
    # Create simple comparison plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Prepare data for plotting
    if experiment_type == "duration":
        labels = [f'{t}s\n({t/60:.1f}min)' for t in test_values if f'{t}s' in data]
        no_sf_rates = [data[f'{t}s']['no_sf']['exposure_rate_pct'] for t in test_values if f'{t}s' in data]
        with_sf_rates = [data[f'{t}s']['with_sf']['exposure_rate_pct'] for t in test_values if f'{t}s' in data]
        title_suffix = "by Duration and Social Force Model"
        xlabel = "Simulation Duration"
    else:  # population
        labels = [f'N={t}\n(ped={t-20})' for t in test_values if f'N{t}' in data]
        no_sf_rates = [data[f'N{t}']['no_sf']['exposure_rate_pct'] for t in test_values if f'N{t}' in data]
        with_sf_rates = [data[f'N{t}']['with_sf']['exposure_rate_pct'] for t in test_values if f'N{t}' in data]
        title_suffix = "by Population Size and Social Force Model"
        xlabel = "Population Size (N)"
    
    if not labels:
        print("Error: No valid data for plotting")
        return
    
    # Plot 1: Bar chart comparing exposure rates
    x = np.arange(len(labels))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, no_sf_rates, width, label='Without Social Force', 
                    color='lightblue', alpha=0.8, edgecolor='black', linewidth=1)
    bars2 = ax1.bar(x + width/2, with_sf_rates, width, label='With Social Force', 
                    color='lightcoral', alpha=0.8, edgecolor='black', linewidth=1)
    
    ax1.set_xlabel(xlabel, fontsize=12)
    ax1.set_ylabel('Particle 1 Exposure Rate (%)', fontsize=12)
    ax1.set_title(f'Exposure Rate Comparison {title_suffix}', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:  # Only add labels for non-zero values
                ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                        f'{height:.1f}%', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # Plot 2: Difference between with/without social force
    differences = [with_sf - no_sf for with_sf, no_sf in zip(with_sf_rates, no_sf_rates)]
    colors = ['red' if d > 0 else 'green' for d in differences]
    
    bars_diff = ax2.bar(labels, differences, color=colors, alpha=0.8, 
                        edgecolor='black', linewidth=1)
    ax2.set_xlabel(xlabel, fontsize=12)
    ax2.set_ylabel('Difference in Exposure Rate (%)', fontsize=12)
    ax2.set_title('Social Force Model Impact\n(With SF - Without SF)', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    
    # Add value labels on bars
    for bar, diff in zip(bars_diff, differences):
        height = bar.get_height()
        if abs(height) > 0.1:  # Only add labels for significant differences
            ax2.text(bar.get_x() + bar.get_width()/2., height + (0.1 if diff > 0 else -0.3),
                    f'{diff:+.1f}%', ha='center', va='bottom' if diff > 0 else 'top', 
                    fontweight='bold', fontsize=10)
    
    plt.tight_layout()
    
    # Save the plot
    base_results_dir = 'experiments/subway_attack_rate/results'
    os.makedirs(base_results_dir, exist_ok=True)
    comparison_plot_path = os.path.join(base_results_dir, f'social_force_{experiment_type}_comparison.png')
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparative plot saved to: {comparison_plot_path}")
    
    plt.close()
    
    # Print summary results
    print(f"\n" + "="*80)
    print(f"SUMMARY RESULTS - {experiment_type.upper()} EXPERIMENT")
    print(f"="*80)
    
    if experiment_type == "duration":
        for test_value in test_values:
            if f'{test_value}s' in data:
                no_sf_data = data[f'{test_value}s']['no_sf']
                with_sf_data = data[f'{test_value}s']['with_sf']
                diff = with_sf_data['exposure_rate_pct'] - no_sf_data['exposure_rate_pct']
                
                print(f"{test_value}s ({test_value/60:.1f} min):")
                print(f"  Without Social Force: {no_sf_data['exposure_rate_pct']:.1f}% ({no_sf_data['exposed_count']}/{no_sf_data['total_iterations']})")
                print(f"  With Social Force:    {with_sf_data['exposure_rate_pct']:.1f}% ({with_sf_data['exposed_count']}/{with_sf_data['total_iterations']})")
                print(f"  Difference:          {diff:+.1f} percentage points")
                print()
    else:  # population
        for test_value in test_values:
            if f'N{test_value}' in data:
                no_sf_data = data[f'N{test_value}']['no_sf']
                with_sf_data = data[f'N{test_value}']['with_sf']
                diff = with_sf_data['exposure_rate_pct'] - no_sf_data['exposure_rate_pct']
                
                print(f"N={test_value} (pedestrians={test_value-20}):")
                print(f"  Without Social Force: {no_sf_data['exposure_rate_pct']:.1f}% ({no_sf_data['exposed_count']}/{no_sf_data['total_iterations']})")
                print(f"  With Social Force:    {with_sf_data['exposure_rate_pct']:.1f}% ({with_sf_data['exposed_count']}/{with_sf_data['total_iterations']})")
                print(f"  Difference:          {diff:+.1f} percentage points")
                print()


if __name__ == '__main__':
    subway_attack_rate()
