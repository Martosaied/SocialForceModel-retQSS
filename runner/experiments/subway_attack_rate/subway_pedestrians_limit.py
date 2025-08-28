import json
import os
import subprocess
import time
import psutil
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import glob
from src.runner import run_experiment, compile_c_code, compile_model
from src.utils import load_config, create_output_dir, copy_results_to_latest
from src.config_manager import print_config_status, get_performance_mode

def check_experiment_flags():
    """Check and display the current experiment flags."""
    print_config_status()

def subway_pedestrians_limit_experiment():
    """
    Comprehensive experiment to test the limits of retqss optimizations by increasing
    pedestrian counts and measuring performance degradation, memory usage, and simulation stability.
    
    This experiment will:
    1. Test pedestrian counts from 100 to 10000 in increasing steps
    2. Test different pedestrian implementations (0=mmoc, 1=retqss, 2=volume-based)
    3. Test with and without social force model
    4. Measure execution time, memory usage, and simulation stability
    5. Identify breaking points where optimizations fail
    """
    print("="*80)
    print("RETQSS PEDESTRIAN LIMITS EXPERIMENT")
    print("="*80)
    print("Testing the limits of retqss optimizations by increasing pedestrian counts...\n")
    
    # Check and display experiment flags
    check_experiment_flags()
    
    # Define test parameters
    pedestrian_counts = [100, 200, 500, 1000, 2000, 3000, 5000, 7000, 10000]
    implementations = [0, 1, 2]  # 0=mmoc, 1=retqss, 2=volume-based
    social_force_modes = [0, 1]  # 0=disabled, 1=enabled
    termination_time = 600  # 10 minutes to keep experiments manageable
    
    # Store results
    results = {
        'execution_times': {},
        'memory_usage': {},
        'success_rates': {},
        'error_messages': {},
        'breaking_points': {}
    }
    
    # Run experiments
    for implementation in implementations:
        impl_name = {0: 'mmoc', 1: 'retqss', 2: 'volume_based'}[implementation]
        print(f"\n{'='*60}")
        print(f"TESTING IMPLEMENTATION: {impl_name.upper()}")
        print(f"{'='*60}")
        
        for social_force in social_force_modes:
            sf_label = "with_sf" if social_force else "no_sf"
            print(f"\n--- Testing {impl_name} with social force {'ENABLED' if social_force else 'DISABLED'} ---")
            
            results['execution_times'][f'{impl_name}_{sf_label}'] = []
            results['memory_usage'][f'{impl_name}_{sf_label}'] = []
            results['success_rates'][f'{impl_name}_{sf_label}'] = []
            results['error_messages'][f'{impl_name}_{sf_label}'] = []
            
            breaking_point_found = False
            
            for ped_count in pedestrian_counts:
                if breaking_point_found:
                    print(f"Skipping {ped_count} pedestrians (breaking point reached)")
                    continue
                
                print(f"\nTesting {ped_count} pedestrians...")
                
                try:
                    # Run single experiment
                    start_time = time.time()
                    start_memory = psutil.virtual_memory().used
                    
                    output_dir = run_pedestrian_limit_experiment(
                        ped_count, implementation, social_force, termination_time
                    )
                    
                    end_time = time.time()
                    end_memory = psutil.virtual_memory().used
                    
                    execution_time = end_time - start_time
                    memory_usage = (end_memory - start_memory) / (1024 * 1024)  # MB
                    
                    results['execution_times'][f'{impl_name}_{sf_label}'].append(execution_time)
                    results['memory_usage'][f'{impl_name}_{sf_label}'].append(memory_usage)
                    results['success_rates'][f'{impl_name}_{sf_label}'].append(1.0)
                    results['error_messages'][f'{impl_name}_{sf_label}'].append(None)
                    
                    print(f"✓ Success: {execution_time:.2f}s, {memory_usage:.1f}MB")
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"✗ Failed: {error_msg}")
                    
                    results['execution_times'][f'{impl_name}_{sf_label}'].append(None)
                    results['memory_usage'][f'{impl_name}_{sf_label}'].append(None)
                    results['success_rates'][f'{impl_name}_{sf_label}'].append(0.0)
                    results['error_messages'][f'{impl_name}_{sf_label}'].append(error_msg)
                    
                    # Mark breaking point
                    results['breaking_points'][f'{impl_name}_{sf_label}'] = ped_count
                    breaking_point_found = True
    
    # Analyze and plot results
    print("\n" + "="*80)
    print("ANALYZING RESULTS")
    print("="*80)
    
    analyze_and_plot_results(results, pedestrian_counts)
    
    # Generate comprehensive report
    generate_limit_report(results, pedestrian_counts)
    
    print(f"\nExperiment completed. Results saved in experiments/subway_attack_rate/results")


def run_pedestrian_limit_experiment(pedestrian_count, implementation, social_force, termination_time):
    """
    Run a single pedestrian limit experiment.
    
    Args:
        pedestrian_count: Number of pedestrians to test
        implementation: Pedestrian implementation (0=mmoc, 1=retqss, 2=volume-based)
        social_force: Social force model (0=disabled, 1=enabled)
        termination_time: Simulation termination time in seconds
    
    Returns:
        Output directory path
    """
    # Load the base configuration
    base_config = load_config('./experiments/subway_attack_rate/subway_hub.json')
    
    # Modify configuration
    config = base_config.copy()
    config['parameters']['FORCE_TERMINATION_AT']['value'] = termination_time
    config['parameters']['SOCIAL_FORCE_MODEL']['value'] = social_force
    config['parameters']['PEDESTRIAN_IMPLEMENTATION']['value'] = implementation
    
    # Set pedestrian counts
    n = pedestrian_count + 20  # Add some passengers
    pedestrians_count = pedestrian_count
    
    config['parameters']['N']['value'] = n
    config['parameters']['PEDESTRIANS_COUNT']['value'] = pedestrians_count
    
    # Ensure OBJECTIVE_SUBWAY_HUB_DT is set to 400
    if 'OBJECTIVE_SUBWAY_HUB_DT' not in config['parameters']:
        config['parameters']['OBJECTIVE_SUBWAY_HUB_DT'] = {
            "name": "OBJECTIVE_SUBWAY_HUB_DT",
            "type": "value",
            "value": 400.0
        }
    else:
        config['parameters']['OBJECTIVE_SUBWAY_HUB_DT']['value'] = 400.0
    
    # Create output directory
    impl_name = {0: 'mmoc', 1: 'retqss', 2: 'volume_based'}[implementation]
    sf_label = "with_sf" if social_force else "no_sf"
    
    output_dir = create_output_dir(
        'experiments/subway_attack_rate/results',
        f'limit_test_N{n}_ped{pedestrians_count}_{impl_name}_{sf_label}'
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
    
    return output_dir


def analyze_and_plot_results(results, pedestrian_counts):
    """
    Analyze and plot the results of the pedestrian limit experiments.
    """
    # Create comprehensive plots
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Plot 1: Execution time vs pedestrian count
    ax1 = axes[0, 0]
    for key, times in results['execution_times'].items():
        valid_indices = [i for i, t in enumerate(times) if t is not None]
        if valid_indices:
            valid_times = [times[i] for i in valid_indices]
            valid_counts = [pedestrian_counts[i] for i in valid_indices]
            ax1.plot(valid_counts, valid_times, 'o-', label=key, linewidth=2, markersize=6)
    
    ax1.set_xlabel('Number of Pedestrians', fontsize=12)
    ax1.set_ylabel('Execution Time (seconds)', fontsize=12)
    ax1.set_title('Execution Time vs Pedestrian Count', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Memory usage vs pedestrian count
    ax2 = axes[0, 1]
    for key, memory in results['memory_usage'].items():
        valid_indices = [i for i, m in enumerate(memory) if m is not None]
        if valid_indices:
            valid_memory = [memory[i] for i in valid_indices]
            valid_counts = [pedestrian_counts[i] for i in valid_indices]
            ax2.plot(valid_counts, valid_memory, 'o-', label=key, linewidth=2, markersize=6)
    
    ax2.set_xlabel('Number of Pedestrians', fontsize=12)
    ax2.set_ylabel('Memory Usage (MB)', fontsize=12)
    ax2.set_title('Memory Usage vs Pedestrian Count', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Success rate vs pedestrian count
    ax3 = axes[1, 0]
    for key, success_rates in results['success_rates'].items():
        ax3.plot(pedestrian_counts[:len(success_rates)], success_rates, 'o-', label=key, linewidth=2, markersize=6)
    
    ax3.set_xlabel('Number of Pedestrians', fontsize=12)
    ax3.set_ylabel('Success Rate', fontsize=12)
    ax3.set_title('Success Rate vs Pedestrian Count', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim(-0.1, 1.1)
    
    # Plot 4: Performance comparison (execution time per pedestrian)
    ax4 = axes[1, 1]
    for key, times in results['execution_times'].items():
        valid_indices = [i for i, t in enumerate(times) if t is not None]
        if valid_indices:
            valid_times = [times[i] for i in valid_indices]
            valid_counts = [pedestrian_counts[i] for i in valid_indices]
            time_per_pedestrian = [t/c for t, c in zip(valid_times, valid_counts)]
            ax4.plot(valid_counts, time_per_pedestrian, 'o-', label=key, linewidth=2, markersize=6)
    
    ax4.set_xlabel('Number of Pedestrians', fontsize=12)
    ax4.set_ylabel('Execution Time per Pedestrian (seconds)', fontsize=12)
    ax4.set_title('Scalability Analysis', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    base_results_dir = 'experiments/subway_attack_rate/results'
    os.makedirs(base_results_dir, exist_ok=True)
    comparison_plot_path = os.path.join(base_results_dir, 'pedestrian_limits_analysis.png')
    plt.savefig(comparison_plot_path, dpi=300, bbox_inches='tight')
    print(f"Analysis plot saved to: {comparison_plot_path}")
    
    plt.close()


def generate_limit_report(results, pedestrian_counts):
    """
    Generate a comprehensive report of the pedestrian limit experiments.
    """
    report_path = 'experiments/subway_attack_rate/results/pedestrian_limits_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RETQSS PEDESTRIAN LIMITS EXPERIMENT REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write("EXPERIMENT SUMMARY\n")
        f.write("-" * 40 + "\n")
        f.write(f"Tested pedestrian counts: {pedestrian_counts}\n")
        f.write(f"Implementations tested: mmoc, retqss, volume_based\n")
        f.write(f"Social force modes: enabled, disabled\n")
        f.write(f"Total experiments: {len(pedestrian_counts) * 3 * 2}\n\n")
        
        f.write("BREAKING POINTS ANALYSIS\n")
        f.write("-" * 40 + "\n")
        for key, breaking_point in results['breaking_points'].items():
            f.write(f"{key}: {breaking_point} pedestrians\n")
        f.write("\n")
        
        f.write("PERFORMANCE ANALYSIS\n")
        f.write("-" * 40 + "\n")
        for key in results['execution_times'].keys():
            times = results['execution_times'][key]
            memory = results['memory_usage'][key]
            success = results['success_rates'][key]
            
            valid_times = [t for t in times if t is not None]
            valid_memory = [m for m in memory if m is not None]
            success_rate = sum(success) / len(success) if success else 0
            
            if valid_times:
                f.write(f"\n{key}:\n")
                f.write(f"  Success rate: {success_rate:.1%}\n")
                f.write(f"  Max execution time: {max(valid_times):.2f}s\n")
                f.write(f"  Max memory usage: {max(valid_memory):.1f}MB\n")
                f.write(f"  Last successful count: {pedestrian_counts[len(valid_times)-1]}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("RECOMMENDATIONS\n")
        f.write("="*80 + "\n")
        
        # Find best performing implementation
        best_impl = None
        best_count = 0
        for key, breaking_point in results['breaking_points'].items():
            if breaking_point > best_count:
                best_count = breaking_point
                best_impl = key
        
        if best_impl:
            f.write(f"Best performing implementation: {best_impl} (up to {best_count} pedestrians)\n")
        
        f.write("\nOptimization recommendations:\n")
        f.write("1. Use volume-based implementation for large crowds (>1000 pedestrians)\n")
        f.write("2. Disable social force model for maximum performance\n")
        f.write("3. Consider memory constraints for very large simulations\n")
        f.write("4. Monitor execution time scaling for different implementations\n")
    
    print(f"Comprehensive report saved to: {report_path}")


if __name__ == '__main__':
    subway_pedestrians_limit_experiment()

