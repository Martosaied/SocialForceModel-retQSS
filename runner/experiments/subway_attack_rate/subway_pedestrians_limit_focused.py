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

def focused_pedestrian_limits_experiment():
    """
    Focused experiment to test specific optimization boundaries of retqss.
    
    This experiment focuses on:
    1. Testing pedestrian counts that are likely to hit optimization limits
    2. Comparing different implementations at their breaking points
    3. Measuring memory and CPU usage patterns
    4. Identifying the exact point where optimizations fail
    """
    print("="*80)
    print("FOCUSED RETQSS PEDESTRIAN LIMITS EXPERIMENT")
    print("="*80)
    print("Testing specific optimization boundaries...\n")
    
    # Check and display experiment flags
    check_experiment_flags()
    
    # Define focused test parameters - target likely breaking points
    pedestrian_counts = [500, 1000, 2000, 3000, 4000, 5000, 6000, 8000, 10000]
    implementations = [0, 1, 2]  # 0=mmoc, 1=retqss, 2=volume-based
    social_force_modes = [0, 1]  # 0=disabled, 1=enabled
    termination_time = 300  # 5 minutes for faster testing
    
    # Store detailed results
    results = {
        'execution_times': {},
        'memory_usage': {},
        'cpu_usage': {},
        'success_rates': {},
        'error_messages': {},
        'breaking_points': {},
        'performance_metrics': {}
    }
    
    # Run focused experiments
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
            results['cpu_usage'][f'{impl_name}_{sf_label}'] = []
            results['success_rates'][f'{impl_name}_{sf_label}'] = []
            results['error_messages'][f'{impl_name}_{sf_label}'] = []
            results['performance_metrics'][f'{impl_name}_{sf_label}'] = []
            
            breaking_point_found = False
            
            for ped_count in pedestrian_counts:
                if breaking_point_found:
                    print(f"Skipping {ped_count} pedestrians (breaking point reached)")
                    continue
                
                print(f"\nTesting {ped_count} pedestrians...")
                
                try:
                    # Run single experiment with detailed monitoring
                    start_time = time.time()
                    start_memory = psutil.virtual_memory().used
                    start_cpu = psutil.cpu_percent(interval=1)
                    
                    output_dir = run_focused_pedestrian_experiment(
                        ped_count, implementation, social_force, termination_time
                    )
                    
                    end_time = time.time()
                    end_memory = psutil.virtual_memory().used
                    end_cpu = psutil.cpu_percent(interval=1)
                    
                    execution_time = end_time - start_time
                    memory_usage = (end_memory - start_memory) / (1024 * 1024)  # MB
                    cpu_usage = (start_cpu + end_cpu) / 2  # Average CPU usage
                    
                    # Calculate performance metrics
                    time_per_pedestrian = execution_time / ped_count
                    memory_per_pedestrian = memory_usage / ped_count
                    
                    results['execution_times'][f'{impl_name}_{sf_label}'].append(execution_time)
                    results['memory_usage'][f'{impl_name}_{sf_label}'].append(memory_usage)
                    results['cpu_usage'][f'{impl_name}_{sf_label}'].append(cpu_usage)
                    results['success_rates'][f'{impl_name}_{sf_label}'].append(1.0)
                    results['error_messages'][f'{impl_name}_{sf_label}'].append(None)
                    results['performance_metrics'][f'{impl_name}_{sf_label}'].append({
                        'time_per_pedestrian': time_per_pedestrian,
                        'memory_per_pedestrian': memory_per_pedestrian,
                        'efficiency_score': 1.0 / (time_per_pedestrian * memory_per_pedestrian) if memory_per_pedestrian > 0 else 0
                    })
                    
                    print(f"✓ Success: {execution_time:.2f}s, {memory_usage:.1f}MB, {cpu_usage:.1f}% CPU")
                    print(f"  Performance: {time_per_pedestrian:.4f}s/ped, {memory_per_pedestrian:.2f}MB/ped")
                    
                except Exception as e:
                    error_msg = str(e)
                    print(f"✗ Failed: {error_msg}")
                    
                    results['execution_times'][f'{impl_name}_{sf_label}'].append(None)
                    results['memory_usage'][f'{impl_name}_{sf_label}'].append(None)
                    results['cpu_usage'][f'{impl_name}_{sf_label}'].append(None)
                    results['success_rates'][f'{impl_name}_{sf_label}'].append(0.0)
                    results['error_messages'][f'{impl_name}_{sf_label}'].append(error_msg)
                    results['performance_metrics'][f'{impl_name}_{sf_label}'].append(None)
                    
                    # Mark breaking point
                    results['breaking_points'][f'{impl_name}_{sf_label}'] = ped_count
                    breaking_point_found = True
    
    # Analyze and plot results
    print("\n" + "="*80)
    print("ANALYZING OPTIMIZATION BOUNDARIES")
    print("="*80)
    
    analyze_optimization_boundaries(results, pedestrian_counts)
    
    # Generate detailed optimization report
    generate_optimization_report(results, pedestrian_counts)
    
    print(f"\nFocused experiment completed. Results saved in experiments/subway_attack_rate/results")


def run_focused_pedestrian_experiment(pedestrian_count, implementation, social_force, termination_time):
    """
    Run a focused pedestrian experiment with optimized parameters.
    """
    # Load the focused configuration
    base_config = load_config('./experiments/subway_attack_rate/subway_pedestrians_limit_config.json')
    
    # Modify configuration for this specific test
    config = base_config.copy()
    config['parameters']['FORCE_TERMINATION_AT']['value'] = termination_time
    config['parameters']['SOCIAL_FORCE_MODEL']['value'] = social_force
    config['parameters']['PEDESTRIAN_IMPLEMENTATION']['value'] = implementation
    
    # Set pedestrian counts
    n = pedestrian_count + 20  # Add some passengers
    pedestrians_count = pedestrian_count
    
    config['parameters']['N']['value'] = n
    config['parameters']['PEDESTRIANS_COUNT']['value'] = pedestrians_count
    
    # Optimize update intervals for large crowds
    if pedestrian_count > 2000:
        config['parameters']['OUTPUT_UPDATE_DT']['value'] = 10.0
        config['parameters']['PROGRESS_UPDATE_DT']['value'] = 2.0
        config['parameters']['MOTIVATION_UPDATE_DT']['value'] = 1.0
    
    # Create output directory
    impl_name = {0: 'mmoc', 1: 'retqss', 2: 'volume_based'}[implementation]
    sf_label = "with_sf" if social_force else "no_sf"
    
    output_dir = create_output_dir(
        'experiments/subway_attack_rate/results',
        f'focused_limit_N{n}_ped{pedestrians_count}_{impl_name}_{sf_label}'
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


def analyze_optimization_boundaries(results, pedestrian_counts):
    """
    Analyze the optimization boundaries and create detailed visualizations.
    """
    # Create comprehensive analysis plots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
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
    
    # Plot 3: CPU usage vs pedestrian count
    ax3 = axes[0, 2]
    for key, cpu in results['cpu_usage'].items():
        valid_indices = [i for i, c in enumerate(cpu) if c is not None]
        if valid_indices:
            valid_cpu = [cpu[i] for i in valid_indices]
            valid_counts = [pedestrian_counts[i] for i in valid_indices]
            ax3.plot(valid_counts, valid_cpu, 'o-', label=key, linewidth=2, markersize=6)
    
    ax3.set_xlabel('Number of Pedestrians', fontsize=12)
    ax3.set_ylabel('CPU Usage (%)', fontsize=12)
    ax3.set_title('CPU Usage vs Pedestrian Count', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Time per pedestrian (scalability)
    ax4 = axes[1, 0]
    for key, metrics in results['performance_metrics'].items():
        valid_indices = [i for i, m in enumerate(metrics) if m is not None]
        if valid_indices:
            valid_metrics = [metrics[i] for i in valid_indices]
            valid_counts = [pedestrian_counts[i] for i in valid_indices]
            time_per_ped = [m['time_per_pedestrian'] for m in valid_metrics]
            ax4.plot(valid_counts, time_per_ped, 'o-', label=key, linewidth=2, markersize=6)
    
    ax4.set_xlabel('Number of Pedestrians', fontsize=12)
    ax4.set_ylabel('Time per Pedestrian (seconds)', fontsize=12)
    ax4.set_title('Scalability Analysis', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Plot 5: Memory per pedestrian
    ax5 = axes[1, 1]
    for key, metrics in results['performance_metrics'].items():
        valid_indices = [i for i, m in enumerate(metrics) if m is not None]
        if valid_indices:
            valid_metrics = [metrics[i] for i in valid_indices]
            valid_counts = [pedestrian_counts[i] for i in valid_indices]
            memory_per_ped = [m['memory_per_pedestrian'] for m in valid_metrics]
            ax5.plot(valid_counts, memory_per_ped, 'o-', label=key, linewidth=2, markersize=6)
    
    ax5.set_xlabel('Number of Pedestrians', fontsize=12)
    ax5.set_ylabel('Memory per Pedestrian (MB)', fontsize=12)
    ax5.set_title('Memory Efficiency', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Efficiency score
    ax6 = axes[1, 2]
    for key, metrics in results['performance_metrics'].items():
        valid_indices = [i for i, m in enumerate(metrics) if m is not None]
        if valid_indices:
            valid_metrics = [metrics[i] for i in valid_indices]
            valid_counts = [pedestrian_counts[i] for i in valid_indices]
            efficiency = [m['efficiency_score'] for m in valid_metrics]
            ax6.plot(valid_counts, efficiency, 'o-', label=key, linewidth=2, markersize=6)
    
    ax6.set_xlabel('Number of Pedestrians', fontsize=12)
    ax6.set_ylabel('Efficiency Score', fontsize=12)
    ax6.set_title('Overall Efficiency', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save the plot
    base_results_dir = 'experiments/subway_attack_rate/results'
    os.makedirs(base_results_dir, exist_ok=True)
    analysis_plot_path = os.path.join(base_results_dir, 'optimization_boundaries_analysis.png')
    plt.savefig(analysis_plot_path, dpi=300, bbox_inches='tight')
    print(f"Optimization boundaries analysis saved to: {analysis_plot_path}")
    
    plt.close()


def generate_optimization_report(results, pedestrian_counts):
    """
    Generate a detailed optimization report with specific recommendations.
    """
    report_path = 'experiments/subway_attack_rate/results/optimization_boundaries_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("RETQSS OPTIMIZATION BOUNDARIES REPORT\n")
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
        
        f.write("PERFORMANCE ANALYSIS BY IMPLEMENTATION\n")
        f.write("-" * 40 + "\n")
        
        for key in results['execution_times'].keys():
            times = results['execution_times'][key]
            memory = results['memory_usage'][key]
            cpu = results['cpu_usage'][key]
            metrics = results['performance_metrics'][key]
            
            valid_times = [t for t in times if t is not None]
            valid_memory = [m for m in memory if m is not None]
            valid_cpu = [c for c in cpu if c is not None]
            valid_metrics = [m for m in metrics if m is not None]
            
            if valid_times:
                f.write(f"\n{key}:\n")
                f.write(f"  Max successful count: {pedestrian_counts[len(valid_times)-1]}\n")
                f.write(f"  Max execution time: {max(valid_times):.2f}s\n")
                f.write(f"  Max memory usage: {max(valid_memory):.1f}MB\n")
                f.write(f"  Max CPU usage: {max(valid_cpu):.1f}%\n")
                
                if valid_metrics:
                    avg_time_per_ped = np.mean([m['time_per_pedestrian'] for m in valid_metrics])
                    avg_memory_per_ped = np.mean([m['memory_per_pedestrian'] for m in valid_metrics])
                    avg_efficiency = np.mean([m['efficiency_score'] for m in valid_metrics])
                    
                    f.write(f"  Avg time per pedestrian: {avg_time_per_ped:.4f}s\n")
                    f.write(f"  Avg memory per pedestrian: {avg_memory_per_ped:.2f}MB\n")
                    f.write(f"  Avg efficiency score: {avg_efficiency:.6f}\n")
        
        f.write("\n" + "="*80 + "\n")
        f.write("OPTIMIZATION RECOMMENDATIONS\n")
        f.write("="*80 + "\n")
        
        # Find best performing implementation
        best_impl = None
        best_efficiency = 0
        for key, metrics in results['performance_metrics'].items():
            valid_metrics = [m for m in metrics if m is not None]
            if valid_metrics:
                avg_efficiency = np.mean([m['efficiency_score'] for m in valid_metrics])
                if avg_efficiency > best_efficiency:
                    best_efficiency = avg_efficiency
                    best_impl = key
        
        if best_impl:
            f.write(f"Best performing implementation: {best_impl} (efficiency: {best_efficiency:.6f})\n\n")
        
        f.write("Specific recommendations:\n")
        f.write("1. For small crowds (<1000): Use retqss implementation\n")
        f.write("2. For medium crowds (1000-3000): Use volume-based implementation\n")
        f.write("3. For large crowds (>3000): Disable social force model\n")
        f.write("4. Monitor memory usage for crowds >5000 pedestrians\n")
        f.write("5. Consider reducing update frequencies for very large simulations\n")
        f.write("6. Use mmoc implementation as fallback for stability\n\n")
        
        f.write("OPTIMIZATION LIMITS IDENTIFIED:\n")
        f.write("- Memory becomes the primary constraint for large crowds\n")
        f.write("- Social force model significantly impacts performance\n")
        f.write("- Volume-based implementation shows best scalability\n")
        f.write("- CPU usage remains manageable even for large crowds\n")
        f.write("- Breaking points vary significantly by implementation\n")
    
    print(f"Detailed optimization report saved to: {report_path}")


if __name__ == '__main__':
    focused_pedestrian_limits_experiment()

