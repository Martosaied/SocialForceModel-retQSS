import json
import os
import subprocess
from src.runner import run_experiment, compile_c_code, compile_model
from src.utils import load_config, create_output_dir, copy_results_to_latest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np


def subway_attack_rate():
    """
    Run the subway_hub model experiment comparing attack rate with and without social force model.
    """
    print("Running subway hub model with social force comparison...\n")

    output_dirs = {
        'no_social_force': create_output_dir(
            'experiments/subway_attack_rate/results',
            'subway_no_social_force'
        ),
        'with_social_force': create_output_dir(
            'experiments/subway_attack_rate/results',
            'subway_with_social_force'
        )
    }

    # Run experiments for both configurations
    # run_social_force_comparison()

    # Analyze and plot comparative attack rate evolution
    print("Analyzing comparative attack rate evolution...")
    # plot_comparative_attack_rate_evolution(output_dirs)
    
    # Analyze and compare particle 1 exposure rates
    print("\nAnalyzing particle 1 exposure rate comparison...")
    particle_1_comparison = compare_particle_1_exposure_rates(output_dirs)

    print(f"\nComparative experiment completed. Results saved in {output_dirs}")


def run_social_force_comparison():
    """
    Run the subway_hub model experiment comparing social force model enabled vs disabled.
    """
    print("Running comparative social force model experiments...")
    
    output_dirs = {}
    
    # Load the base configuration
    base_config = load_config('subway_hub.json')
    
    # Configuration 1: Social Force Model DISABLED (0)
    print("\n1. Running experiment with Social Force Model DISABLED...")
    config_no_sf = base_config.copy()
    config_no_sf['parameters']['SOCIAL_FORCE_MODEL']['value'] = 0
    
    output_dir_no_sf = create_output_dir(
        'experiments/subway_attack_rate/results',
        'subway_no_social_force'
    )
    print(f"Created output directory: {output_dir_no_sf}")
    
    # Save config copy
    config_copy_path = os.path.join(output_dir_no_sf, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config_no_sf, f, indent=2)
    
    # Compile and run
    compile_c_code()
    compile_model('subway_hub')
    
    run_experiment(
        config_no_sf,
        output_dir_no_sf,
        'subway_hub',
        plot=False,
        copy_results=True
    )
    copy_results_to_latest(output_dir_no_sf)
    output_dirs['no_social_force'] = output_dir_no_sf
    
    # Configuration 2: Social Force Model ENABLED (1)
    print("\n2. Running experiment with Social Force Model ENABLED...")
    config_with_sf = base_config.copy()
    config_with_sf['parameters']['SOCIAL_FORCE_MODEL']['value'] = 1
    
    output_dir_with_sf = create_output_dir(
        'experiments/subway_attack_rate/results',
        'subway_with_social_force'
    )
    print(f"Created output directory: {output_dir_with_sf}")
    
    # Save config copy
    config_copy_path = os.path.join(output_dir_with_sf, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config_with_sf, f, indent=2)
    
    run_experiment(
        config_with_sf,
        output_dir_with_sf,
        'subway_hub',
        plot=False,
        copy_results=True
    )
    copy_results_to_latest(output_dir_with_sf)
    output_dirs['with_social_force'] = output_dir_with_sf
    
    return output_dirs

def analyze_particle_1_exposure_rate(results_dir):
    """
    Analyze the exposure rate of particle 1 across all iterations.
    
    Args:
        results_dir: Directory containing result_0.csv, result_1.csv, etc.
    
    Returns:
        Dictionary with particle 1 exposure statistics
    """
    # Find all result files
    result_files = []
    for i in range(100):  # Check up to 100 iterations
        result_file = os.path.join(results_dir, f'result_{i}.csv')
        if os.path.exists(result_file):
            result_files.append(result_file)
        else:
            break
    
    if not result_files:
        raise FileNotFoundError(f"No result files found in {results_dir}")
    
    print(f"Analyzing particle 1 exposure across {len(result_files)} iterations...")
    
    # Track particle 1's final state in each iteration
    particle_1_final_states = []
    exposed_count = 0
    total_iterations = len(result_files)
    
    for result_file in result_files:
        df = pd.read_csv(result_file)
        
        # Get particle 1's final state (last row, PS[1] column)
        if 'PS[1]' in df.columns:
            final_state = df['PS[1]'].iloc[-1]  # Last row of the simulation
            
            # Check if particle 1 ended up exposed (state 1)
            if pd.notna(final_state) and final_state == 1:  # EXPOSED state
                exposed_count += 1
                particle_1_final_states.append(1)  # Exposed
            else:
                particle_1_final_states.append(0)  # Not exposed
        else:
            print(f"Warning: PS[1] column not found in {result_file}")
            particle_1_final_states.append(0)  # Assume not exposed if column missing
    
    # Calculate exposure rate
    exposure_rate = exposed_count / total_iterations if total_iterations > 0 else 0
    exposure_rate_pct = exposure_rate * 100
    
    # Calculate additional statistics
    not_exposed_count = total_iterations - exposed_count
    
    # Create detailed analysis dictionary
    analysis = {
        'total_iterations': total_iterations,
        'exposed_count': exposed_count,
        'not_exposed_count': not_exposed_count,
        'exposure_rate': exposure_rate,
        'exposure_rate_pct': exposure_rate_pct,
        'particle_1_final_states': particle_1_final_states,
        'results_dir': results_dir
    }
    
    print(f"Particle 1 exposure analysis:")
    print(f"  Total iterations: {total_iterations}")
    print(f"  Exposed in {exposed_count} iterations")
    print(f"  Not exposed in {not_exposed_count} iterations")
    print(f"  Exposure rate: {exposure_rate:.4f} ({exposure_rate_pct:.2f}%)")
    
    return analysis


def compare_particle_1_exposure_rates(output_dirs):
    """
    Compare particle 1 exposure rates between experiments with and without social force model.
    
    Args:
        output_dirs: Dictionary containing paths to both experiment results
    
    Returns:
        Dictionary with comparative analysis
    """
    print("Comparing particle 1 exposure rates between social force model configurations...")
    
    # Get the latest results directories for both experiments
    no_sf_results_dir = os.path.join(os.path.dirname(output_dirs['no_social_force']), 'latest')
    with_sf_results_dir = os.path.join(os.path.dirname(output_dirs['with_social_force']), 'latest')
    
    # Analyze particle 1 exposure for both configurations
    print("\nAnalyzing WITHOUT social force model...")
    no_sf_analysis = analyze_particle_1_exposure_rate(no_sf_results_dir)
    
    print("\nAnalyzing WITH social force model...")
    with_sf_analysis = analyze_particle_1_exposure_rate(with_sf_results_dir)
    
    # Calculate comparative statistics
    exposure_rate_diff = with_sf_analysis['exposure_rate'] - no_sf_analysis['exposure_rate']
    exposure_rate_diff_pct = with_sf_analysis['exposure_rate_pct'] - no_sf_analysis['exposure_rate_pct']
    
    # Calculate relative change
    if no_sf_analysis['exposure_rate'] > 0:
        relative_change = (exposure_rate_diff / no_sf_analysis['exposure_rate']) * 100
    else:
        relative_change = 0
    
    # Create comparative analysis
    comparison = {
        'without_social_force': no_sf_analysis,
        'with_social_force': with_sf_analysis,
        'difference': {
            'absolute': exposure_rate_diff,
            'absolute_pct': exposure_rate_diff_pct,
            'relative_pct': relative_change
        },
        'summary': {
            'no_sf_exposure_rate': no_sf_analysis['exposure_rate_pct'],
            'with_sf_exposure_rate': with_sf_analysis['exposure_rate_pct'],
            'difference_pct': exposure_rate_diff_pct,
            'relative_change_pct': relative_change
        }
    }
    
    # Print comparative results
    print(f"\n" + "="*60)
    print(f"PARTICLE 1 EXPOSURE RATE COMPARISON")
    print(f"="*60)
    print(f"Without Social Force Model:")
    print(f"  Exposure Rate: {no_sf_analysis['exposure_rate_pct']:.2f}% ({no_sf_analysis['exposed_count']}/{no_sf_analysis['total_iterations']})")
    print(f"")
    print(f"With Social Force Model:")
    print(f"  Exposure Rate: {with_sf_analysis['exposure_rate_pct']:.2f}% ({with_sf_analysis['exposed_count']}/{with_sf_analysis['total_iterations']})")
    print(f"")
    print(f"Difference:")
    print(f"  Absolute: {exposure_rate_diff_pct:+.2f} percentage points")
    print(f"  Relative: {relative_change:+.1f}% change")
    print(f"="*60)
    
    # Create visualization
    plot_particle_1_exposure_comparison(comparison)
    
    return comparison


def plot_particle_1_exposure_comparison(comparison):
    """
    Create visualization comparing particle 1 exposure rates.
    
    Args:
        comparison: Dictionary with comparative analysis data
    """
    print("Creating particle 1 exposure rate comparison plot...")
    
    # Extract data
    no_sf_data = comparison['without_social_force']
    with_sf_data = comparison['with_social_force']
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Bar chart comparing exposure rates
    configurations = ['Without SF', 'With SF']
    exposure_rates = [no_sf_data['exposure_rate_pct'], with_sf_data['exposure_rate_pct']]
    colors = ['lightblue', 'lightcoral']
    
    bars = ax1.bar(configurations, exposure_rates, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
    ax1.set_ylabel('Exposure Rate (%)', fontsize=12)
    ax1.set_title('Particle 1 Exposure Rate Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, rate in zip(bars, exposure_rates):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{rate:.1f}%', ha='center', va='bottom', fontweight='bold')
    
    # Plot 2: Pie charts showing exposed vs not exposed
    # Without SF
    ax2.pie([no_sf_data['exposed_count'], no_sf_data['not_exposed_count']], 
            labels=['Exposed', 'Not Exposed'], 
            colors=['red', 'lightgreen'], 
            autopct='%1.1f%%', startangle=90)
    ax2.set_title('Without Social Force Model', fontsize=12, fontweight='bold')
    
    # With SF
    ax3.pie([with_sf_data['exposed_count'], with_sf_data['not_exposed_count']], 
            labels=['Exposed', 'Not Exposed'], 
            colors=['red', 'lightgreen'], 
            autopct='%1.1f%%', startangle=90)
    ax3.set_title('With Social Force Model', fontsize=12, fontweight='bold')
    
    # Plot 4: Summary statistics table
    ax4.axis('tight')
    ax4.axis('off')
    
    # Create summary table
    table_data = [
        ['Metric', 'Without SF', 'With SF', 'Difference'],
        ['Exposure Rate (%)', f"{no_sf_data['exposure_rate_pct']:.2f}%", 
         f"{with_sf_data['exposure_rate_pct']:.2f}%", 
         f"{comparison['summary']['difference_pct']:+.2f} pp"],
        ['Exposed Count', str(no_sf_data['exposed_count']), 
         str(with_sf_data['exposed_count']), 
         f"{with_sf_data['exposed_count'] - no_sf_data['exposed_count']:+d}"],
        ['Total Iterations', str(no_sf_data['total_iterations']), 
         str(with_sf_data['total_iterations']), '-'],
        ['Relative Change', '-', '-', f"{comparison['summary']['relative_change_pct']:+.1f}%"]
    ]
    
    table = ax4.table(cellText=table_data[1:], colLabels=table_data[0], 
                     cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    # Style the table
    for i in range(len(table_data)):
        for j in range(len(table_data[0])):
            if i == 0:  # Header row
                table[(i, j)].set_facecolor('#4CAF50')
                table[(i, j)].set_text_props(weight='bold', color='white')
            else:
                table[(i, j)].set_facecolor('#f0f0f0')
    
    ax4.set_title('Summary Statistics', fontsize=14, fontweight='bold', pad=20)
    
    plt.tight_layout()
    
    # Save the plot
    base_results_dir = 'experiments/subway_attack_rate/results'
    comparison_plot_path = os.path.join(base_results_dir, 'particle_1_exposure_comparison.png')
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    print(f"Particle 1 exposure comparison plot saved to: {comparison_plot_path}")
    
    plt.close()


def analyze_multiple_iterations(results_dir, config_path=None):
    """
    Analyze multiple iterations and calculate averaged attack rate data.
    
    Args:
        results_dir: Directory containing result_0.csv, result_1.csv, etc.
        config_path: Path to config file for initial conditions
    
    Returns:
        Dictionary with averaged data across all iterations
    """
    # Find all result files
    result_files = []
    for i in range(100):  # Check up to 100 iterations
        result_file = os.path.join(results_dir, f'result_{i}.csv')
        if os.path.exists(result_file):
            result_files.append(result_file)
        else:
            break
    
    if not result_files:
        raise FileNotFoundError(f"No result files found in {results_dir}")
    
    print(f"Found {len(result_files)} iterations to analyze")
    
    # Analyze each iteration
    all_iterations_data = []
    for result_file in result_files:
        data = analyze_attack_rate_data(result_file, config_path)
        all_iterations_data.append(data)
    
    # Calculate averages across iterations
    return calculate_averaged_data(all_iterations_data)


def calculate_averaged_data(all_iterations_data):
    """
    Calculate averaged data across multiple iterations.
    
    Args:
        all_iterations_data: List of data dictionaries from each iteration
    
    Returns:
        Dictionary with averaged data
    """
    if not all_iterations_data:
        return {}
    
    # Get the number of iterations
    num_iterations = len(all_iterations_data)
    
    # Find the common time points across all iterations
    # We'll use the shortest time series to ensure all iterations have data
    min_length = min(len(data['times']) for data in all_iterations_data)
    
    # Initialize averaged data
    averaged_data = {
        'times': all_iterations_data[0]['times'][:min_length],
        'attack_rates': [],
        'attack_rates_pct': [],
        'susceptible_counts': [],
        'exposed_counts': [],
        'infected_counts': [],
        'recovered_counts': [],
        'final_attack_rates': [],
        'final_attack_rates_pct': [],
        'initial_infected': all_iterations_data[0]['initial_infected'],
        'initial_susceptible': all_iterations_data[0]['initial_susceptible'],
        'total_population': all_iterations_data[0]['total_population'],
        'num_iterations': num_iterations
    }
    
    # Calculate averages for each time point
    for t in range(min_length):
        attack_rates_at_t = [data['attack_rates'][t] for data in all_iterations_data]
        attack_rates_pct_at_t = [data['attack_rates_pct'][t] for data in all_iterations_data]
        susceptible_at_t = [data['susceptible_counts'][t] for data in all_iterations_data]
        exposed_at_t = [data['exposed_counts'][t] for data in all_iterations_data]
        infected_at_t = [data['infected_counts'][t] for data in all_iterations_data]
        recovered_at_t = [data['recovered_counts'][t] for data in all_iterations_data]
        
        averaged_data['attack_rates'].append(np.mean(attack_rates_at_t))
        averaged_data['attack_rates_pct'].append(np.mean(attack_rates_pct_at_t))
        averaged_data['susceptible_counts'].append(np.mean(susceptible_at_t))
        averaged_data['exposed_counts'].append(np.mean(exposed_at_t))
        averaged_data['infected_counts'].append(np.mean(infected_at_t))
        averaged_data['recovered_counts'].append(np.mean(recovered_at_t))
    
    # Calculate averages for final attack rates
    final_attack_rates = [data['final_attack_rate'] for data in all_iterations_data]
    final_attack_rates_pct = [data['final_attack_rate_pct'] for data in all_iterations_data]
    
    averaged_data['final_attack_rate'] = np.mean(final_attack_rates)
    averaged_data['final_attack_rate_pct'] = np.mean(final_attack_rates_pct)
    averaged_data['final_attack_rate_std'] = np.std(final_attack_rates)
    averaged_data['final_attack_rate_pct_std'] = np.std(final_attack_rates_pct)
    
    # Also calculate standard deviations for time series (for error bars)
    averaged_data['attack_rates_std'] = []
    averaged_data['attack_rates_pct_std'] = []
    
    for t in range(min_length):
        attack_rates_at_t = [data['attack_rates'][t] for data in all_iterations_data]
        attack_rates_pct_at_t = [data['attack_rates_pct'][t] for data in all_iterations_data]
        
        averaged_data['attack_rates_std'].append(np.std(attack_rates_at_t))
        averaged_data['attack_rates_pct_std'].append(np.std(attack_rates_pct_at_t))
    
    return averaged_data


def analyze_attack_rate_data(result_csv_path, config_path=None):
    """
    Helper function to analyze attack rate data from a result CSV file.
    Returns times, attack_rates, and population counts.
    
    Attack Rate Formula: AR = (E/Nsus) / I₀
    Where:
    - E = final number of agents in the Exposed state at the end of the simulation
    - Nsus = initial number of Susceptible agents (calculated as N - I₀)
    - I₀ = initial number of Infected agents
    - N = total number of agents
    """
    df = pd.read_csv(result_csv_path)
    
    # Get initial conditions from config file if provided
    initial_infected = 9  # default value from model
    total_population = 150  # default value from model
    
    if config_path and os.path.exists(config_path):
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            initial_infected = config.get('parameters', {}).get('INITIAL_INFECTED', {}).get('value', 9)
            total_population = config.get('parameters', {}).get('N', {}).get('value', 150)
        except:
            pass  # use defaults if config can't be read
    
    # Calculate initial susceptible population
    initial_susceptible = total_population - initial_infected
    
    times = []
    attack_rates = []
    attack_rates_pct = []
    susceptible_counts = []
    exposed_counts = []
    infected_counts = []
    recovered_counts = []

    for index, row in df.iterrows():
        time = row['time']

        # Count particles in each state
        susceptible = 0
        exposed = 0
        infected = 0
        recovered = 0

        # Count particles by state (PS[i] columns)
        for col in df.columns:
            if col.startswith('PS[') and col.endswith(']'):
                state = row[col]
                if pd.notna(state):
                    if state == 0:  # SUSCEPTIBLE
                        susceptible += 1
                    elif state == 1:  # EXPOSED
                        exposed += 1
                    elif state in [2, 3, 4]:  # PRE_SYMPTOMATIC, SYMPTOMATIC, ASYMPTOMATIC
                        infected += 1
                    elif state == 5:  # RECOVERED
                        recovered += 1

        times.append(time)
        susceptible_counts.append(susceptible)
        exposed_counts.append(exposed)
        infected_counts.append(infected)
        recovered_counts.append(recovered)

    # Calculate attack rate using AR = (E_final/Nsus) / I₀
    # We use the final exposed count from the last time step
    final_exposed = exposed_counts[-1] if exposed_counts else 0
    
    if initial_susceptible > 0 and initial_infected > 0:
        attack_rate = (final_exposed / initial_susceptible) / initial_infected
    else:
        attack_rate = 0
    
    # For time series, we calculate incremental attack rates based on exposed at each time
    for i, exposed in enumerate(exposed_counts):
        if initial_susceptible > 0 and initial_infected > 0:
            time_attack_rate = (exposed / initial_susceptible) / initial_infected
        else:
            time_attack_rate = 0
        
        attack_rates.append(time_attack_rate)
        attack_rates_pct.append(time_attack_rate)
    
    return {
        'times': times,
        'attack_rates': attack_rates,
        'attack_rates_pct': attack_rates_pct,
        'susceptible_counts': susceptible_counts,
        'exposed_counts': exposed_counts,
        'infected_counts': infected_counts,
        'recovered_counts': recovered_counts,
        'final_attack_rate': attack_rate,
        'final_attack_rate_pct': attack_rate,
        'initial_infected': initial_infected,
        'initial_susceptible': initial_susceptible,
        'total_population': total_population
    }


def plot_comparative_attack_rate_evolution(output_dirs):
    """
    Plot comparative analysis of attack rate evolution with and without social force model.
    """
    print("Creating comparative attack rate analysis...")
    
    # Analyze data from both experiments using multiple iterations
    # The output_dirs contain the run directory path, we need to get the latest directory
    no_sf_results_dir = os.path.join(os.path.dirname(output_dirs['no_social_force']), 'latest')
    with_sf_results_dir = os.path.join(os.path.dirname(output_dirs['with_social_force']), 'latest')
    
    # Config paths for reading initial conditions
    no_sf_config_path = os.path.join(no_sf_results_dir, 'config.json')
    with_sf_config_path = os.path.join(with_sf_results_dir, 'config.json')
    
    data_no_sf = analyze_multiple_iterations(no_sf_results_dir, no_sf_config_path)
    data_with_sf = analyze_multiple_iterations(with_sf_results_dir, with_sf_config_path)
    
    # Create comprehensive comparative plot
    fig = plt.figure(figsize=(20, 16))
    
    # Plot 1: Direct comparison of attack rates with error bars
    ax1 = plt.subplot(3, 2, 1)
    ax1.plot(data_no_sf['times'], data_no_sf['attack_rates_pct'], 
             'b-', linewidth=3, label=f'Without SF ({data_no_sf["num_iterations"]} iter)', alpha=0.8)
    ax1.plot(data_with_sf['times'], data_with_sf['attack_rates_pct'], 
             'r-', linewidth=3, label=f'With SF ({data_with_sf["num_iterations"]} iter)', alpha=0.8)
    
    # Add error bars
    ax1.fill_between(data_no_sf['times'], 
                     np.array(data_no_sf['attack_rates_pct']) - np.array(data_no_sf['attack_rates_pct_std']),
                     np.array(data_no_sf['attack_rates_pct']) + np.array(data_no_sf['attack_rates_pct_std']),
                     alpha=0.2, color='blue')
    ax1.fill_between(data_with_sf['times'], 
                     np.array(data_with_sf['attack_rates_pct']) - np.array(data_with_sf['attack_rates_pct_std']),
                     np.array(data_with_sf['attack_rates_pct']) + np.array(data_with_sf['attack_rates_pct_std']),
                     alpha=0.2, color='red')
    
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel('Attack Rate', fontsize=12)
    ax1.set_title('Attack Rate Comparison: Social Force Model Impact\n(AR = (E/Nsus) / I₀)', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=11)
    
    # Add max value annotations
    max_no_sf = max(data_no_sf['attack_rates_pct'])
    max_with_sf = max(data_with_sf['attack_rates_pct'])
    max_idx_no_sf = data_no_sf['attack_rates_pct'].index(max_no_sf)
    max_idx_with_sf = data_with_sf['attack_rates_pct'].index(max_with_sf)
    
    ax1.annotate(f'Max Without SF: {max_no_sf:.2f}%', 
                xy=(data_no_sf['times'][max_idx_no_sf], max_no_sf),
                xytext=(10, 10), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightblue', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    ax1.annotate(f'Max With SF: {max_with_sf:.2f}%', 
                xy=(data_with_sf['times'][max_idx_with_sf], max_with_sf),
                xytext=(10, -30), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='lightcoral', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    
    # Plot 2: Active cases comparison (Exposed + Infected)
    ax2 = plt.subplot(3, 2, 2)
    active_no_sf = [e + i for e, i in zip(data_no_sf['exposed_counts'], data_no_sf['infected_counts'])]
    active_with_sf = [e + i for e, i in zip(data_with_sf['exposed_counts'], data_with_sf['infected_counts'])]
    
    ax2.plot(data_no_sf['times'], active_no_sf, 'b-', linewidth=3, label='Without Social Force Model', alpha=0.8)
    ax2.plot(data_with_sf['times'], active_with_sf, 'r-', linewidth=3, label='With Social Force Model', alpha=0.8)
    ax2.fill_between(data_no_sf['times'], active_no_sf, alpha=0.2, color='blue')
    ax2.fill_between(data_with_sf['times'], active_with_sf, alpha=0.2, color='red')
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Active Cases (Exposed + Infected)', fontsize=12)
    ax2.set_title('Active Cases Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=11)
    
    # Plot 3: Population distribution without social force
    ax3 = plt.subplot(3, 2, 3)
    ax3.stackplot(data_no_sf['times'], 
                  data_no_sf['susceptible_counts'], 
                  data_no_sf['exposed_counts'], 
                  data_no_sf['infected_counts'], 
                  data_no_sf['recovered_counts'],
                  labels=['Susceptible', 'Exposed', 'Infected', 'Recovered'],
                  colors=['lightblue', 'yellow', 'red', 'lightgreen'],
                  alpha=0.8)
    ax3.set_xlabel('Time (s)', fontsize=12)
    ax3.set_ylabel('Number of Individuals', fontsize=12)
    ax3.set_title('Population Distribution - WITHOUT Social Force Model', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(loc='upper right', fontsize=10)
    
    # Plot 4: Population distribution with social force
    ax4 = plt.subplot(3, 2, 4)
    ax4.stackplot(data_with_sf['times'], 
                  data_with_sf['susceptible_counts'], 
                  data_with_sf['exposed_counts'], 
                  data_with_sf['infected_counts'], 
                  data_with_sf['recovered_counts'],
                  labels=['Susceptible', 'Exposed', 'Infected', 'Recovered'],
                  colors=['lightblue', 'yellow', 'red', 'lightgreen'],
                  alpha=0.8)
    ax4.set_xlabel('Time (s)', fontsize=12)
    ax4.set_ylabel('Number of Individuals', fontsize=12)
    ax4.set_title('Population Distribution - WITH Social Force Model', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    
    # Save the comparative plot
    base_results_dir = 'experiments/subway_attack_rate/results'
    comparative_plot_path = os.path.join(base_results_dir, 'social_force_comparison.png')
    plt.savefig(comparative_plot_path, dpi=300, bbox_inches='tight')
    print(f"Comparative analysis plot saved to: {comparative_plot_path}")
    
    plt.close()
    
if __name__ == '__main__':
    subway_attack_rate()
