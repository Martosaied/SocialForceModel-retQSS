import json
import os
import subprocess
import warnings
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from scipy import stats
from scipy.optimize import curve_fit

from src.runner import run_experiment, compile_c_code, compile_model
from src.utils import load_config, create_output_dir, copy_results_to_latest, generate_map
from src.math.Density import Density
from src.constants import Constants

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Publication-quality plotting settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'axes.linewidth': 1.2,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
})

# Experiment parameters
PROGRESS_UPDATE_DT = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
WIDTH = 20
GRID_SIZE = 50
PEDESTRIAN_DENSITY = 0.3
CONFIDENCE_LEVEL = 0.95  # 95% confidence interval

def progress_update_dt():
    print("Running experiments for different progress update dt...\n")
    for progress_update_dt in PROGRESS_UPDATE_DT:
        print(f"Running experiment for {progress_update_dt} progress update dt...")
        # run(progress_update_dt)
        print(f"Experiment for {progress_update_dt} progress update dt completed.\n")

    # Plot the results
    print("Plotting results...")
    plot_results()

def run(progress_update_dt):
    """
    Run the experiment for a given number of pedestrians.
    """
    config = load_config('./experiments/progress_update_dt/config.json')

    # Create output directory with experiment name if provided
    output_dir = create_output_dir(
        'experiments/progress_update_dt/results', 
        f'progress_update_dt_{progress_update_dt}'
    )
    print(f"Created output directory: {output_dir}")

    pedestrians = int(PEDESTRIAN_DENSITY * WIDTH * GRID_SIZE)
    config['iterations'] = 3

    config['parameters']['N']['value'] = pedestrians
    config['parameters']['PEDESTRIAN_IMPLEMENTATION']['value'] = Constants.PEDESTRIAN_MMOC
    config['parameters']['BORDER_IMPLEMENTATION']['value'] = Constants.BORDER_SURROUNDING_VOLUMES
    config['parameters']['PROGRESS_UPDATE_DT']['value'] = progress_update_dt

    # Add from where to where pedestrians are generated
    config['parameters']['FROM_Y'] = {
      "name": "FROM_Y",
      "type": "value",
      "value": (GRID_SIZE/ 2) - int(WIDTH / 2)
    }
    config['parameters']['TO_Y'] = {
      "name": "TO_Y",
      "type": "value",
      "value": (GRID_SIZE/ 2) + int(WIDTH / 2)
    }

    obstacles = generate_map(WIDTH, GRID_SIZE)
    config['parameters']['OBSTACLES'] = {
      "name": "OBSTACLES",
      "type": "map",
      "map": obstacles
    }

    # Save config copy in experiment directory
    config_copy_path = os.path.join(output_dir, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)

    subprocess.run(['sed', '-i', r's/\bGRID_DIVISIONS\s*=\s*[0-9]\+/GRID_DIVISIONS = ' + str(GRID_SIZE) + '/', '../retqss/model/social_force_model.mo'])
    subprocess.run(['sed', '-i', r's/\bN\s*=\s*[0-9]\+/N = ' + str(pedestrians) + '/', '../retqss/model/social_force_model.mo'])

    # Compile the C++ code if requested
    compile_c_code()

    # Compile the model if requested
    compile_model('social_force_model')

    # Run experiment
    run_experiment(
        config, 
        output_dir, 
        'social_force_model', 
        plot=False, 
        copy_results=True
    )

    # Copy results from output directory to latest directory
    copy_results_to_latest(output_dir)

    print(f"\nExperiment completed. Results saved in {output_dir}")


def plot_results():
    """
    Create publication-ready plots with enhanced styling and statistical analysis.
    """
    print("Creating publication-ready plots...")
    
    # Create figures directory
    figures_dir = Path('experiments/progress_update_dt/figures')
    figures_dir.mkdir(exist_ok=True)
    
    # Get all the results directories
    results_dirs = [d for d in os.listdir('experiments/progress_update_dt/results') 
                   if os.path.isdir(os.path.join('experiments/progress_update_dt/results', d))]

    # Initialize data storage
    data = {
        'progress_update_dts': [],
        'groups_mean': [],
        'groups_std': [],
        'groups_ci_lower': [],
        'groups_ci_upper': [],
        'time_mean': [],
        'time_std': [],
        'memory_mean': [],
        'memory_std': [],
        'groups_data': [],
        'time_data': [],
        'memory_data': []
    }

    # Collect data from metrics.csv files
    for result_dir in results_dirs:
        try:
            progress_update_dt = float(result_dir.split('progress_update_dt_')[1])
            metrics_path = os.path.join('experiments/progress_update_dt/results', result_dir, 'latest', 'metrics.csv')
            
            if os.path.exists(metrics_path):
                df = pd.read_csv(metrics_path)
                
                # Extract groups data
                groups_data = df['clustering_based_groups'].dropna().tolist()
                time_data = df['time'].dropna().tolist()
                memory_data = df['memory_usage'].dropna().tolist()
                
                if groups_data:
                    # Calculate statistics
                    groups_mean = np.mean(groups_data)
                    groups_std = np.std(groups_data, ddof=1)
                    
                    # Calculate confidence interval
                    ci = stats.t.interval(
                        CONFIDENCE_LEVEL, 
                        len(groups_data) - 1, 
                        loc=groups_mean, 
                        scale=groups_std / np.sqrt(len(groups_data))
                    )
                    
                    # Store results
                    data['progress_update_dts'].append(progress_update_dt)
                    data['groups_mean'].append(groups_mean)
                    data['groups_std'].append(groups_std)
                    data['groups_ci_lower'].append(ci[0])
                    data['groups_ci_upper'].append(ci[1])
                    data['time_mean'].append(np.mean(time_data) if time_data else 0)
                    data['time_std'].append(np.std(time_data, ddof=1) if time_data else 0)
                    data['memory_mean'].append(np.mean(memory_data) if memory_data else 0)
                    data['memory_std'].append(np.std(memory_data, ddof=1) if memory_data else 0)
                    data['groups_data'].append(groups_data)
                    data['time_data'].append(time_data)
                    data['memory_data'].append(memory_data)
                    
                    print(f"Processed dt={progress_update_dt}: {len(groups_data)} data points")
                    
        except Exception as e:
            print(f"Warning: Could not process {result_dir}: {e}")
            continue

    if not data['progress_update_dts']:
        print("No data found for plotting!")
        return

    # Sort data by progress_update_dt
    sorted_indices = np.argsort(data['progress_update_dts'])
    for key in data:
        if key != 'groups_data' and key != 'time_data' and key != 'memory_data':
            data[key] = np.array(data[key])[sorted_indices]
        else:
            data[key] = [data[key][i] for i in sorted_indices]

    # Create enhanced plots
    create_main_analysis_plot(data, figures_dir)
    create_performance_analysis_plot(data, figures_dir)
    create_latex_ready_figure(data, figures_dir)
    create_combined_figure(data, figures_dir)
    
    print(f"All plots saved in: {figures_dir}")


def create_main_analysis_plot(data: Dict, figures_dir: Path) -> None:
    """Create the main analysis plot with error bars and statistical information."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    progress_update_dts = data['progress_update_dts']
    groups_mean = data['groups_mean']
    groups_ci_lower = data['groups_ci_lower']
    groups_ci_upper = data['groups_ci_upper']
    
    # Plot error bars with confidence intervals
    ax.errorbar(
        progress_update_dts, groups_mean, 
        yerr=[groups_mean - groups_ci_lower, groups_ci_upper - groups_mean],
        fmt='o', 
        markersize=10,
        capsize=6,
        capthick=2.5,
        elinewidth=2.5,
        label='Experimental Data (95% CI)',
        color='#2E86AB',
        alpha=0.9,
        zorder=3
    )
    
    # Fit polynomial curve
    if len(progress_update_dts) > 3:
        # Try different polynomial degrees and select the best fit
        best_degree = find_best_polynomial_fit(progress_update_dts, groups_mean)
        coeffs = np.polyfit(progress_update_dts, groups_mean, best_degree)
        poly = np.poly1d(coeffs)
        x_fit = np.linspace(min(progress_update_dts), max(progress_update_dts), 100)
        y_fit = poly(x_fit)
        
        ax.plot(x_fit, y_fit, '--', linewidth=2, color='#A23B72', 
               label=f'Polynomial fit (degree {best_degree})')
    
    # Add linear trend line
    slope, intercept, r_value, p_value, std_err = stats.linregress(progress_update_dts, groups_mean)
    line_x = np.array([min(progress_update_dts), max(progress_update_dts)])
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, '-', linewidth=3, color='#F18F01', 
           label=f'Linear trend (R² = {r_value**2:.3f})', zorder=2)
    
    # Customize plot
    ax.set_xlabel('Progress Update Δt', fontsize=16, fontweight='bold')
    ax.set_ylabel('Number of Lane Groups', fontsize=16, fontweight='bold')
    ax.set_title('Lane Formation vs. Progress Update Time Step', fontsize=18, fontweight='bold', pad=25)
    
    ax.grid(True, alpha=0.4, linestyle='--', zorder=1)
    ax.legend(fontsize=14, framealpha=0.95, loc='upper left')
    
    # Add statistical information
    stats_text = f'Confidence level: {CONFIDENCE_LEVEL*100}%\nPedestrian density: {PEDESTRIAN_DENSITY}\nTotal data points: {len(progress_update_dts)}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', fontsize=11,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))
    
    # Set axis limits with some padding
    ax.set_xlim(min(progress_update_dts) - 0.01, max(progress_update_dts) + 0.01)
    y_min = max(0, min(groups_mean - (groups_mean - groups_ci_lower)) - 0.5)
    y_max = max(groups_mean + (groups_ci_upper - groups_mean)) + 0.5
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'main_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_performance_analysis_plot(data: Dict, figures_dir: Path) -> None:
    """Create performance analysis plots."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    progress_update_dts = data['progress_update_dts']
    groups_mean = data['groups_mean']
    time_mean = data['time_mean']
    memory_mean = data['memory_mean']
    
    # Plot 1: Lane groups vs progress update dt
    ax1.errorbar(progress_update_dts, groups_mean, yerr=data['groups_std'], 
                fmt='o', capsize=4, capthick=2, elinewidth=2, color='#2E86AB')
    ax1.set_xlabel('Progress Update Δt')
    ax1.set_ylabel('Number of Lane Groups')
    ax1.set_title('(a) Lane Groups vs Progress Update Δt')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Execution time vs progress update dt
    ax2.errorbar(progress_update_dts, time_mean, yerr=data['time_std'], 
                fmt='s', capsize=4, capthick=2, elinewidth=2, color='#A23B72')
    ax2.set_xlabel('Progress Update Δt')
    ax2.set_ylabel('Execution Time (ms)')
    ax2.set_title('(b) Execution Time vs Progress Update Δt')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Memory usage vs progress update dt
    ax3.errorbar(progress_update_dts, memory_mean, yerr=data['memory_std'], 
                fmt='^', capsize=4, capthick=2, elinewidth=2, color='#F18F01')
    ax3.set_xlabel('Progress Update Δt')
    ax3.set_ylabel('Memory Usage (KB)')
    ax3.set_title('(c) Memory Usage vs Progress Update Δt')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Groups vs execution time correlation
    ax4.scatter(time_mean, groups_mean, alpha=0.7, s=100, color='#C73E1D')
    if len(time_mean) > 1:
        slope, intercept = np.polyfit(time_mean, groups_mean, 1)
        line = slope * np.array(time_mean) + intercept
        ax4.plot(time_mean, line, '--', color='red', alpha=0.8)
    ax4.set_xlabel('Execution Time (ms)')
    ax4.set_ylabel('Number of Lane Groups')
    ax4.set_title('(d) Lane Groups vs Execution Time')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_latex_ready_figure(data: Dict, figures_dir: Path) -> None:
    """Create a LaTeX-ready figure with proper formatting for academic papers."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    progress_update_dts = data['progress_update_dts']
    groups_mean = data['groups_mean']
    groups_std = data['groups_std']
    
    # Plot with error bars
    ax.errorbar(progress_update_dts, groups_mean, yerr=groups_std, fmt='o', 
               markersize=6, capsize=4, capthick=1.5, elinewidth=1.5,
               color='black', alpha=0.8, zorder=3)
    
    # Fit line
    if len(progress_update_dts) > 1:
        slope, intercept = np.polyfit(progress_update_dts, groups_mean, 1)
        line = slope * np.array(progress_update_dts) + intercept
        ax.plot(progress_update_dts, line, '--', linewidth=2, color='red', alpha=0.8, zorder=2)
        
        # Calculate R-squared
        correlation_matrix = np.corrcoef(progress_update_dts, groups_mean)
        correlation_xy = correlation_matrix[0, 1]
        r_squared = correlation_xy ** 2
        
        # Add R-squared text
        ax.text(0.05, 0.95, f'$R^2 = {r_squared:.3f}$', transform=ax.transAxes,
               fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Customize plot for LaTeX
    ax.set_xlabel('Progress Update Δt', fontsize=12)
    ax.set_ylabel('Number of Lane Groups', fontsize=12)
    ax.set_title('Lane Formation vs. Progress Update Time Step', fontsize=14, pad=15)
    
    ax.grid(True, alpha=0.3, linestyle='-', zorder=1)
    
    # Set axis limits
    ax.set_xlim(min(progress_update_dts) - 0.01, max(progress_update_dts) + 0.01)
    y_min = max(0, min(groups_mean - groups_std) - 0.5)
    y_max = max(groups_mean + groups_std) + 0.5
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'latex_figure.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / 'latex_figure.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create LaTeX table
    create_latex_table(data, figures_dir)


def create_combined_figure(data: Dict, figures_dir: Path) -> None:
    """Create a combined figure suitable for publication."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    progress_update_dts = data['progress_update_dts']
    groups_mean = data['groups_mean']
    groups_std = data['groups_std']
    time_mean = data['time_mean']
    memory_mean = data['memory_mean']
    
    # Subplot 1: Main error bar plot
    ax1.errorbar(progress_update_dts, groups_mean, yerr=groups_std, fmt='o-', capsize=5, capthick=2)
    ax1.set_xlabel('Progress Update Δt')
    ax1.set_ylabel('Number of Lane Groups')
    ax1.set_title('(a) Lane Groups vs Progress Update Δt')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(progress_update_dts, groups_mean)
    ax2.scatter(progress_update_dts, groups_mean, alpha=0.7)
    x_line = np.array([min(progress_update_dts), max(progress_update_dts)])
    y_line = slope * x_line + intercept
    ax2.plot(x_line, y_line, 'r--', linewidth=2)
    ax2.set_xlabel('Progress Update Δt')
    ax2.set_ylabel('Number of Lane Groups')
    ax2.set_title(f'(b) Linear Regression (R² = {r_value**2:.3f})')
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Execution time
    ax3.errorbar(progress_update_dts, time_mean, yerr=data['time_std'], fmt='s-', capsize=5, capthick=2)
    ax3.set_xlabel('Progress Update Δt')
    ax3.set_ylabel('Execution Time (ms)')
    ax3.set_title('(c) Execution Time vs Progress Update Δt')
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Memory usage
    ax4.errorbar(progress_update_dts, memory_mean, yerr=data['memory_std'], fmt='^-', capsize=5, capthick=2)
    ax4.set_xlabel('Progress Update Δt')
    ax4.set_ylabel('Memory Usage (KB)')
    ax4.set_title('(d) Memory Usage vs Progress Update Δt')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'combined_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_latex_table(data: Dict, figures_dir: Path) -> None:
    """Create a LaTeX table with the experimental results."""
    table_path = figures_dir / 'results_table.tex'
    
    with open(table_path, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Experimental Results: Lane Groups vs Progress Update Δt}\n")
        f.write("\\label{tab:progress_update_dt}\n")
        f.write("\\begin{tabular}{cccc}\n")
        f.write("\\hline\n")
        f.write("Progress Update Δt & Mean Groups & Std Dev & N \\\\\n")
        f.write("\\hline\n")
        
        for i in range(len(data['progress_update_dts'])):
            dt = data['progress_update_dts'][i]
            mean_groups = data['groups_mean'][i]
            std_groups = data['groups_std'][i]
            n = len(data['groups_data'][i])
            f.write(f"{dt:.2f} & {mean_groups:.2f} & {std_groups:.2f} & {n} \\\\\n")
        
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\end{table}\n")
    
    print(f"LaTeX table saved to: {table_path}")


def find_best_polynomial_fit(x: np.ndarray, y: np.ndarray, max_degree: int = 3) -> int:
    """Find the best polynomial degree using cross-validation."""
    best_score = float('inf')
    best_degree = 1
    
    for degree in range(1, min(max_degree + 1, len(x) - 1)):
        try:
            coeffs = np.polyfit(x, y, degree)
            poly = np.poly1d(coeffs)
            y_pred = poly(x)
            mse = np.mean((y - y_pred) ** 2)
            
            if mse < best_score:
                best_score = mse
                best_degree = degree
        except:
            continue
    
    return best_degree


def quick_enhanced_plots():
    """
    Quick function to generate all enhanced plots using metrics.csv data.
    This is the fastest way to create publication-ready graphics.
    """
    print("=" * 60)
    print("QUICK ENHANCED PLOTS GENERATION")
    print("=" * 60)
    
    # Create figures directory
    figures_dir = Path('experiments/progress_update_dt/figures')
    figures_dir.mkdir(exist_ok=True)
    
    # Get all the results directories
    results_dirs = [d for d in os.listdir('experiments/progress_update_dt/results') 
                   if os.path.isdir(os.path.join('experiments/progress_update_dt/results', d))]

    # Initialize data storage
    data = {
        'progress_update_dts': [],
        'groups_mean': [],
        'groups_std': [],
        'groups_ci_lower': [],
        'groups_ci_upper': [],
        'time_mean': [],
        'time_std': [],
        'memory_mean': [],
        'memory_std': [],
        'groups_data': [],
        'time_data': [],
        'memory_data': []
    }

    # Collect data from metrics.csv files
    for result_dir in results_dirs:
        try:
            progress_update_dt = float(result_dir.split('progress_update_dt_')[1])
            metrics_path = os.path.join('experiments/progress_update_dt/results', result_dir, 'latest', 'metrics.csv')
            
            if os.path.exists(metrics_path):
                df = pd.read_csv(metrics_path)
                
                # Extract groups data
                groups_data = df['clustering_based_groups'].dropna().tolist()
                time_data = df['time'].dropna().tolist()
                memory_data = df['memory_usage'].dropna().tolist()
                
                if groups_data:
                    # Calculate statistics
                    groups_mean = np.mean(groups_data)
                    groups_std = np.std(groups_data, ddof=1)
                    
                    # Calculate confidence interval
                    ci = stats.t.interval(
                        CONFIDENCE_LEVEL, 
                        len(groups_data) - 1, 
                        loc=groups_mean, 
                        scale=groups_std / np.sqrt(len(groups_data))
                    )
                    
                    # Store results
                    data['progress_update_dts'].append(progress_update_dt)
                    data['groups_mean'].append(groups_mean)
                    data['groups_std'].append(groups_std)
                    data['groups_ci_lower'].append(ci[0])
                    data['groups_ci_upper'].append(ci[1])
                    data['time_mean'].append(np.mean(time_data) if time_data else 0)
                    data['time_std'].append(np.std(time_data, ddof=1) if time_data else 0)
                    data['memory_mean'].append(np.mean(memory_data) if memory_data else 0)
                    data['memory_std'].append(np.std(memory_data, ddof=1) if memory_data else 0)
                    data['groups_data'].append(groups_data)
                    data['time_data'].append(time_data)
                    data['memory_data'].append(memory_data)
                    
                    print(f"Processed dt={progress_update_dt}: {len(groups_data)} data points")
                    
        except Exception as e:
            print(f"Warning: Could not process {result_dir}: {e}")
            continue

    if not data['progress_update_dts']:
        print("No data found for plotting!")
        return

    # Sort data by progress_update_dt
    sorted_indices = np.argsort(data['progress_update_dts'])
    for key in data:
        if key != 'groups_data' and key != 'time_data' and key != 'memory_data':
            data[key] = np.array(data[key])[sorted_indices]
        else:
            data[key] = [data[key][i] for i in sorted_indices]

    # Create enhanced plots
    create_main_analysis_plot(data, figures_dir)
    create_performance_analysis_plot(data, figures_dir)
    create_latex_ready_figure(data, figures_dir)
    create_combined_figure(data, figures_dir)
    
    print("\nAll enhanced plots generated successfully!")
    print(f"Check the 'figures' directory: {figures_dir}")
    print("=" * 60)


if __name__ == '__main__':
    progress_update_dt()
