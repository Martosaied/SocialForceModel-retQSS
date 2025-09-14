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
from src.utils import load_config, create_output_dir, copy_results_to_latest
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
CELL_SIZES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]  # Cell sizes in meters
PEDESTRIANS_IMPLEMENTATION = {
    Constants.NO_PEDESTRIANS: "no_helbing",
    Constants.PEDESTRIAN_NEIGHBORHOOD: "retqss",
}
WIDTH = 20
GRID_SIZE = 50
PEDESTRIAN_DENSITY = 0.3
CONFIDENCE_LEVEL = 0.95  # 95% confidence interval

def breaking_lanes():
    print("Ejecutando experimentos para 300 peatones con diferentes tamaños de celda para ver si los carriles se rompen...\n")
    # run(1.0, Constants.NO_PEDESTRIANS)
    for cell_size in CELL_SIZES:
        for implementation in PEDESTRIANS_IMPLEMENTATION:
            print(f"Ejecutando experimento para tamaño de celda {cell_size}m con implementación {implementation}...")
            # run(cell_size, implementation)
            print(f"Experimento para tamaño de celda {cell_size}m con implementación {implementation} completado.\n")

    # Graficar los resultados
    print("Graficando resultados...")
    plot_results()

def run(cell_size, implementation):
    """
    Run the experiment for a given cell size in meters.
    """
    config = load_config('./experiments/breaking_lanes/config.json')

    # Calculate grid divisions from cell size
    grid_divisions = int(GRID_SIZE / cell_size)
    
    # Create output directory with experiment name if provided
    output_dir = create_output_dir(
        'experiments/breaking_lanes/results', 
        f'cell_size_{cell_size}_implementation_{PEDESTRIANS_IMPLEMENTATION[implementation]}'
    )
    print(f"Created output directory: {output_dir}")

    pedestrians = int(PEDESTRIAN_DENSITY * WIDTH * GRID_SIZE)
    config['iterations'] = 20
    if implementation != Constants.NO_PEDESTRIANS:
        config['parameters']['N']['value'] = pedestrians
        config['parameters']['PEDESTRIAN_IMPLEMENTATION']['value'] = Constants.PEDESTRIAN_NEIGHBORHOOD
        config['parameters']['BORDER_IMPLEMENTATION']['value'] = Constants.BORDER_NONE
    else:
        config['parameters']['N']['value'] = pedestrians
        config['parameters']['PEDESTRIAN_IMPLEMENTATION']['value'] = Constants.PEDESTRIAN_NEIGHBORHOOD
        config['parameters']['BORDER_IMPLEMENTATION']['value'] = Constants.BORDER_NONE
        config['parameters']['PEDESTRIAN_A_1']['value'] = 0 # Turn off the pedestrian effect
        config['parameters']['PEDESTRIAN_A_2']['value'] = 0

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


    # Save config copy in experiment directory
    config_copy_path = os.path.join(output_dir, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)

    subprocess.run(['sed', '-i', r's/\bGRID_DIVISIONS\s*=\s*[0-9]\+/GRID_DIVISIONS = ' + str(grid_divisions) + '/', '../retqss/model/social_force_model.mo'])
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
    figures_dir = Path('experiments/breaking_lanes/figures')
    figures_dir.mkdir(exist_ok=True)
    
    # Get all the results directories
    results_dirs = [d for d in os.listdir('experiments/breaking_lanes/results') 
                   if os.path.isdir(os.path.join('experiments/breaking_lanes/results', d))]

    # Separate the results directories by implementation
    results_dirs_by_implementation = {
        "retqss": [],
        "no_helbing": []
    }
    for results_dir in results_dirs:
        try:
            implementation = results_dir.split('_implementation_')[1]
            results_dirs_by_implementation[implementation].append(results_dir)
        except:
            continue

    # Initialize data storage
    data = {
        'cell_sizes': [],
        'clustering_groups_mean': [],
        'clustering_groups_std': [],
        'memory_mean': [],
        'memory_std': [],
        'time_mean': [],
        'time_std': [],
        'velocity_y_mean': [],
        'velocity_y_std': [],
        'clustering_groups_data': [],
        'memory_data': [],
        'time_data': [],
        'velocity_y_data': []
    }

    # Process no_helbing implementation (baseline)
    baseline_data = {
        'clustering_groups': []
    }
    
    for result_dir in results_dirs_by_implementation['no_helbing']:
        try:
            metrics_path = os.path.join('experiments/breaking_lanes/results', result_dir, 'latest', 'metrics.csv')
            if os.path.exists(metrics_path):
                df = pd.read_csv(metrics_path)
                clustering_data = df['clustering_based_groups'].dropna().tolist()
                
                if clustering_data:
                    baseline_data['clustering_groups'].extend(clustering_data)
        except Exception as e:
            print(f"Warning: Could not process no_helbing {result_dir}: {e}")
            continue

    # Process retqss implementation
    for result_dir in results_dirs_by_implementation['retqss']:
        try:
            cell_size = float(result_dir.split('cell_size_')[1].split('_')[0])
            metrics_path = os.path.join('experiments/breaking_lanes/results', result_dir, 'latest', 'metrics.csv')
            
            if os.path.exists(metrics_path):
                df = pd.read_csv(metrics_path)
                
                # Extract metrics data
                clustering_data = df['clustering_based_groups'].dropna().tolist()
                memory_data = df['memory_usage'].dropna().tolist()
                time_data = df['time'].dropna().tolist()
                
                # Calculate velocity data from individual CSV files
                velocity_data = []
                results_path = os.path.join('experiments/breaking_lanes/results', result_dir, 'latest')
                for result_file in os.listdir(results_path):
                    if result_file.endswith('.csv') and result_file != 'metrics.csv':
                        try:
                            result_df = pd.read_csv(os.path.join(results_path, result_file))
                            prev_mean_velocity_y = None
                            for index, row in result_df.iterrows():
                                vy_key = f'VY[{index + 1}]'
                                if vy_key in row and pd.notna(row[vy_key]):
                                    velocity_y = row[vy_key]
                                    if prev_mean_velocity_y is not None:
                                        velocity_data.append(abs(velocity_y - prev_mean_velocity_y))
                                    prev_mean_velocity_y = velocity_y
                        except Exception as e:
                            continue
                
                if clustering_data:  # Only add if we have valid data
                    # Calculate statistics
                    clustering_mean = np.mean(clustering_data)
                    clustering_std = np.std(clustering_data, ddof=1)
                    memory_mean = np.mean(memory_data) if memory_data else 0
                    memory_std = np.std(memory_data, ddof=1) if memory_data else 0
                    time_mean = np.mean(time_data) if time_data else 0
                    time_std = np.std(time_data, ddof=1) if time_data else 0
                    velocity_mean = np.mean(velocity_data) if velocity_data else 0
                    velocity_std = np.std(velocity_data, ddof=1) if velocity_data else 0
                    
                    # Store results
                    data['cell_sizes'].append(cell_size)
                    data['clustering_groups_mean'].append(clustering_mean)
                    data['clustering_groups_std'].append(clustering_std)
                    data['memory_mean'].append(memory_mean)
                    data['memory_std'].append(memory_std)
                    data['time_mean'].append(time_mean)
                    data['time_std'].append(time_std)
                    data['velocity_y_mean'].append(velocity_mean)
                    data['velocity_y_std'].append(velocity_std)
                    data['clustering_groups_data'].append(clustering_data)
                    data['memory_data'].append(memory_data)
                    data['time_data'].append(time_data)
                    data['velocity_y_data'].append(velocity_data)
                    
                    print(f"Processed cell_size={cell_size}: {len(clustering_data)} data points")
                    
        except Exception as e:
            print(f"Warning: Could not process {result_dir}: {e}")
            continue

    if not data['cell_sizes']:
        print("No data found for plotting!")
        return

    # Sort data by cell sizes
    sorted_indices = np.argsort(data['cell_sizes'])
    for key in data:
        if key != 'clustering_groups_data' and key != 'memory_data' and key != 'time_data' and key != 'velocity_y_data':
            data[key] = np.array(data[key])[sorted_indices]
        else:
            data[key] = [data[key][i] for i in sorted_indices]

    # Create enhanced plots
    create_main_analysis_plot(data, baseline_data, figures_dir)
    create_performance_analysis_plot(data, figures_dir)
    create_latex_ready_figure(data, baseline_data, figures_dir)
    create_combined_figure(data, baseline_data, figures_dir)
    
    print(f"All plots saved in: {figures_dir}")


def create_main_analysis_plot(data: Dict, baseline_data: Dict, figures_dir: Path) -> None:
    """Create the main analysis plot with error bars and statistical information."""
    fig, ax = plt.subplots(figsize=(12, 8))
    
    cell_sizes = data['cell_sizes']
    clustering_mean = data['clustering_groups_mean']
    clustering_std = data['clustering_groups_std']
    
    # Plot error bars with standard deviation
    ax.errorbar(
        cell_sizes, clustering_mean, 
        yerr=clustering_std,
        fmt='o', 
        markersize=10,
        capsize=6,
        capthick=2.5,
        elinewidth=2.5,
        label='RETQSS Implementation',
        color='#2E86AB',
        alpha=0.9,
        zorder=3
    )
    
    # Add baseline (no_helbing) if available
    if baseline_data['clustering_groups']:
        baseline_mean = np.mean(baseline_data['clustering_groups'])
        baseline_std = np.std(baseline_data['clustering_groups'], ddof=1)
        ax.axhline(y=baseline_mean, color='red', linestyle='--', linewidth=2, 
                  label=f'No Helbing Baseline ({baseline_mean:.2f} ± {baseline_std:.2f})', zorder=2)
        ax.fill_between([min(cell_sizes), max(cell_sizes)], 
                       baseline_mean - baseline_std, baseline_mean + baseline_std, 
                       alpha=0.3, color='red', zorder=1)
    
    # Fit polynomial curve
    if len(cell_sizes) > 3:
        best_degree = find_best_polynomial_fit(cell_sizes, clustering_mean)
        coeffs = np.polyfit(cell_sizes, clustering_mean, best_degree)
        poly = np.poly1d(coeffs)
        x_fit = np.linspace(min(cell_sizes), max(cell_sizes), 100)
        y_fit = poly(x_fit)
        
        ax.plot(x_fit, y_fit, '--', linewidth=2, color='#A23B72', 
               label=f'Polynomial fit (degree {best_degree})')
    
    # Add linear trend line
    slope, intercept, r_value, p_value, std_err = stats.linregress(cell_sizes, clustering_mean)
    line_x = np.array([min(cell_sizes), max(cell_sizes)])
    line_y = slope * line_x + intercept
    ax.plot(line_x, line_y, '-', linewidth=3, color='#F18F01', 
           label=f'Linear trend (R² = {r_value**2:.3f})', zorder=2)
    
    # Customize plot
    ax.set_xlabel('Cell Size (meters)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Number of Lane Groups', fontsize=16, fontweight='bold')
    ax.set_title('Lane Formation vs. Cell Size', fontsize=18, fontweight='bold', pad=25)
    
    ax.grid(True, alpha=0.4, linestyle='--', zorder=1)
    ax.legend(fontsize=14, framealpha=0.95, loc='upper left')
    
    # Add statistical information
    stats_text = f'Confidence level: {CONFIDENCE_LEVEL*100}%\nPedestrian density: {PEDESTRIAN_DENSITY}\nTotal data points: {len(cell_sizes)}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
           verticalalignment='top', fontsize=11,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))
    
    # Set axis limits with some padding
    ax.set_xlim(min(cell_sizes) - 0.1, max(cell_sizes) + 0.1)
    y_min = max(0, min(clustering_mean - clustering_std) - 0.5)
    y_max = max(clustering_mean + clustering_std) + 0.5
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'main_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_performance_analysis_plot(data: Dict, figures_dir: Path) -> None:
    """Create performance analysis plots."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    
    cell_sizes = data['cell_sizes']
    clustering_mean = data['clustering_groups_mean']
    time_mean = data['time_mean']
    memory_mean = data['memory_mean']
    velocity_mean = data['velocity_y_mean']
    
    # Plot 1: Lane groups vs cell sizes
    ax1.errorbar(cell_sizes, clustering_mean, yerr=data['clustering_groups_std'], 
                fmt='o', capsize=4, capthick=2, elinewidth=2, color='#2E86AB')
    ax1.set_xlabel('Cell Size (meters)')
    ax1.set_ylabel('Number of Lane Groups')
    ax1.set_title('(a) Lane Groups vs Cell Size')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Execution time vs cell sizes
    ax2.errorbar(cell_sizes, time_mean, yerr=data['time_std'], 
                fmt='s', capsize=4, capthick=2, elinewidth=2, color='#A23B72')
    ax2.set_xlabel('Cell Size (meters)')
    ax2.set_ylabel('Execution Time (ms)')
    ax2.set_title('(b) Execution Time vs Cell Size')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Memory usage vs cell sizes
    ax3.errorbar(cell_sizes, memory_mean, yerr=data['memory_std'], 
                fmt='^', capsize=4, capthick=2, elinewidth=2, color='#F18F01')
    ax3.set_xlabel('Cell Size (meters)')
    ax3.set_ylabel('Memory Usage (KB)')
    ax3.set_title('(c) Memory Usage vs Cell Size')
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Velocity variation vs cell sizes
    ax4.errorbar(cell_sizes, velocity_mean, yerr=data['velocity_y_std'], 
                fmt='d', capsize=4, capthick=2, elinewidth=2, color='#C73E1D')
    ax4.set_xlabel('Cell Size (meters)')
    ax4.set_ylabel('Velocity Variation (m/s)')
    ax4.set_title('(d) Velocity Variation vs Cell Size')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'performance_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_latex_ready_figure(data: Dict, baseline_data: Dict, figures_dir: Path) -> None:
    """Create a LaTeX-ready figure with proper formatting for academic papers."""
    fig, ax = plt.subplots(figsize=(8, 6))
    
    cell_sizes = data['cell_sizes']
    clustering_mean = data['clustering_groups_mean']
    clustering_std = data['clustering_groups_std']
    
    # Plot with error bars
    ax.errorbar(cell_sizes, clustering_mean, yerr=clustering_std, fmt='o', 
               markersize=6, capsize=4, capthick=1.5, elinewidth=1.5,
               color='black', alpha=0.8, zorder=3)
    
    # Add baseline if available
    if baseline_data['clustering_groups']:
        baseline_mean = np.mean(baseline_data['clustering_groups'])
        ax.axhline(y=baseline_mean, color='red', linestyle='--', linewidth=2, 
                  label=f'Baseline ({baseline_mean:.2f})', zorder=2)
    
    # Fit line
    if len(cell_sizes) > 1:
        slope, intercept = np.polyfit(cell_sizes, clustering_mean, 1)
        line = slope * np.array(cell_sizes) + intercept
        ax.plot(cell_sizes, line, '--', linewidth=2, color='blue', alpha=0.8, zorder=2)
        
        # Calculate R-squared
        correlation_matrix = np.corrcoef(cell_sizes, clustering_mean)
        correlation_xy = correlation_matrix[0, 1]
        r_squared = correlation_xy ** 2
        
        # Add R-squared text
        ax.text(0.05, 0.95, f'$R^2 = {r_squared:.3f}$', transform=ax.transAxes,
               fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Customize plot for LaTeX
    ax.set_xlabel('Cell Size (meters)', fontsize=12)
    ax.set_ylabel('Number of Lane Groups', fontsize=12)
    ax.set_title('Lane Formation vs. Cell Size', fontsize=14, pad=15)
    
    ax.grid(True, alpha=0.3, linestyle='-', zorder=1)
    ax.legend(fontsize=10)
    
    # Set axis limits
    ax.set_xlim(min(cell_sizes) - 0.1, max(cell_sizes) + 0.1)
    y_min = max(0, min(clustering_mean - clustering_std) - 0.5)
    y_max = max(clustering_mean + clustering_std) + 0.5
    ax.set_ylim(y_min, y_max)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'latex_figure.pdf', dpi=300, bbox_inches='tight')
    plt.savefig(figures_dir / 'latex_figure.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Create LaTeX table
    create_latex_table(data, baseline_data, figures_dir)


def create_combined_figure(data: Dict, baseline_data: Dict, figures_dir: Path) -> None:
    """Create a combined figure suitable for publication."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    cell_sizes = data['cell_sizes']
    clustering_mean = data['clustering_groups_mean']
    clustering_std = data['clustering_groups_std']
    time_mean = data['time_mean']
    memory_mean = data['memory_mean']
    
    # Subplot 1: Main error bar plot
    ax1.errorbar(cell_sizes, clustering_mean, yerr=clustering_std, fmt='o-', capsize=5, capthick=2)
    if baseline_data['clustering_groups']:
        baseline_mean = np.mean(baseline_data['clustering_groups'])
        ax1.axhline(y=baseline_mean, color='red', linestyle='--', linewidth=2)
    ax1.set_xlabel('Cell Size (meters)')
    ax1.set_ylabel('Number of Lane Groups')
    ax1.set_title('(a) Lane Groups vs Cell Size')
    ax1.grid(True, alpha=0.3)
    
    # Subplot 2: Linear regression
    slope, intercept, r_value, p_value, std_err = stats.linregress(cell_sizes, clustering_mean)
    ax2.scatter(cell_sizes, clustering_mean, alpha=0.7)
    x_line = np.array([min(cell_sizes), max(cell_sizes)])
    y_line = slope * x_line + intercept
    ax2.plot(x_line, y_line, 'r--', linewidth=2)
    ax2.set_xlabel('Cell Size (meters)')
    ax2.set_ylabel('Number of Lane Groups')
    ax2.set_title(f'(b) Linear Regression (R² = {r_value**2:.3f})')
    ax2.grid(True, alpha=0.3)
    
    # Subplot 3: Execution time
    ax3.errorbar(cell_sizes, time_mean, yerr=data['time_std'], fmt='s-', capsize=5, capthick=2)
    ax3.set_xlabel('Cell Size (meters)')
    ax3.set_ylabel('Execution Time (ms)')
    ax3.set_title('(c) Execution Time vs Cell Size')
    ax3.grid(True, alpha=0.3)
    
    # Subplot 4: Memory usage
    ax4.errorbar(cell_sizes, memory_mean, yerr=data['memory_std'], fmt='^-', capsize=5, capthick=2)
    ax4.set_xlabel('Cell Size (meters)')
    ax4.set_ylabel('Memory Usage (KB)')
    ax4.set_title('(d) Memory Usage vs Cell Size')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(figures_dir / 'combined_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()


def create_latex_table(data: Dict, baseline_data: Dict, figures_dir: Path) -> None:
    """Create a LaTeX table with the experimental results."""
    table_path = figures_dir / 'results_table.tex'
    
    with open(table_path, 'w') as f:
        f.write("\\begin{table}[h]\n")
        f.write("\\centering\n")
        f.write("\\caption{Experimental Results: Lane Groups vs Cell Size}\n")
        f.write("\\label{tab:breaking_lanes}\n")
        f.write("\\begin{tabular}{cccc}\n")
        f.write("\\hline\n")
        f.write("Cell Size (m) & Mean Groups & Std Dev & N \\\\\n")
        f.write("\\hline\n")
        
        for i in range(len(data['cell_sizes'])):
            cell_size = data['cell_sizes'][i]
            mean_groups = data['clustering_groups_mean'][i]
            std_groups = data['clustering_groups_std'][i]
            n = len(data['clustering_groups_data'][i])
            f.write(f"{cell_size:.1f} & {mean_groups:.2f} & {std_groups:.2f} & {n} \\\\\n")
        
        # Add baseline row if available
        if baseline_data['clustering_groups']:
            baseline_mean = np.mean(baseline_data['clustering_groups'])
            baseline_std = np.std(baseline_data['clustering_groups'], ddof=1)
            baseline_n = len(baseline_data['clustering_groups'])
            f.write("\\hline\n")
            f.write(f"Baseline (No Helbing) & {baseline_mean:.2f} & {baseline_std:.2f} & {baseline_n} \\\\\n")
        
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
    figures_dir = Path('experiments/breaking_lanes/figures')
    figures_dir.mkdir(exist_ok=True)
    
    # Get all the results directories
    results_dirs = [d for d in os.listdir('experiments/breaking_lanes/results') 
                   if os.path.isdir(os.path.join('experiments/breaking_lanes/results', d))]

    # Separate the results directories by implementation
    results_dirs_by_implementation = {
        "retqss": [],
        "no_helbing": []
    }
    for results_dir in results_dirs:
        try:
            implementation = results_dir.split('_implementation_')[1]
            results_dirs_by_implementation[implementation].append(results_dir)
        except:
            continue

    # Initialize data storage
    data = {
        'cell_sizes': [],
        'clustering_groups_mean': [],
        'clustering_groups_std': [],
        'memory_mean': [],
        'memory_std': [],
        'time_mean': [],
        'time_std': [],
        'velocity_y_mean': [],
        'velocity_y_std': [],
        'clustering_groups_data': [],
        'memory_data': [],
        'time_data': [],
        'velocity_y_data': []
    }

    # Process no_helbing implementation (baseline)
    baseline_data = {
        'clustering_groups': []
    }
    
    for result_dir in results_dirs_by_implementation['no_helbing']:
        try:
            metrics_path = os.path.join('experiments/breaking_lanes/results', result_dir, 'latest', 'metrics.csv')
            if os.path.exists(metrics_path):
                df = pd.read_csv(metrics_path)
                clustering_data = df['clustering_based_groups'].dropna().tolist()
                
                if clustering_data:
                    baseline_data['clustering_groups'].extend(clustering_data)
        except Exception as e:
            print(f"Warning: Could not process no_helbing {result_dir}: {e}")
            continue

    # Process retqss implementation
    for result_dir in results_dirs_by_implementation['retqss']:
        try:
            cell_size = float(result_dir.split('cell_size_')[1].split('_')[0])
            metrics_path = os.path.join('experiments/breaking_lanes/results', result_dir, 'latest', 'metrics.csv')
            
            if os.path.exists(metrics_path):
                df = pd.read_csv(metrics_path)
                
                # Extract metrics data
                clustering_data = df['clustering_based_groups'].dropna().tolist()
                memory_data = df['memory_usage'].dropna().tolist()
                time_data = df['time'].dropna().tolist()
                
                # Calculate velocity data from individual CSV files
                velocity_data = []
                results_path = os.path.join('experiments/breaking_lanes/results', result_dir, 'latest')
                for result_file in os.listdir(results_path):
                    if result_file.endswith('.csv') and result_file != 'metrics.csv':
                        try:
                            result_df = pd.read_csv(os.path.join(results_path, result_file))
                            prev_mean_velocity_y = None
                            for index, row in result_df.iterrows():
                                vy_key = f'VY[{index + 1}]'
                                if vy_key in row and pd.notna(row[vy_key]):
                                    velocity_y = row[vy_key]
                                    if prev_mean_velocity_y is not None:
                                        velocity_data.append(abs(velocity_y - prev_mean_velocity_y))
                                    prev_mean_velocity_y = velocity_y
                        except Exception as e:
                            continue
                
                if clustering_data:  # Only add if we have valid data
                    # Calculate statistics
                    clustering_mean = np.mean(clustering_data)
                    clustering_std = np.std(clustering_data, ddof=1)
                    memory_mean = np.mean(memory_data) if memory_data else 0
                    memory_std = np.std(memory_data, ddof=1) if memory_data else 0
                    time_mean = np.mean(time_data) if time_data else 0
                    time_std = np.std(time_data, ddof=1) if time_data else 0
                    velocity_mean = np.mean(velocity_data) if velocity_data else 0
                    velocity_std = np.std(velocity_data, ddof=1) if velocity_data else 0
                    
                    # Store results
                    data['cell_sizes'].append(cell_size)
                    data['clustering_groups_mean'].append(clustering_mean)
                    data['clustering_groups_std'].append(clustering_std)
                    data['memory_mean'].append(memory_mean)
                    data['memory_std'].append(memory_std)
                    data['time_mean'].append(time_mean)
                    data['time_std'].append(time_std)
                    data['velocity_y_mean'].append(velocity_mean)
                    data['velocity_y_std'].append(velocity_std)
                    data['clustering_groups_data'].append(clustering_data)
                    data['memory_data'].append(memory_data)
                    data['time_data'].append(time_data)
                    data['velocity_y_data'].append(velocity_data)
                    
                    print(f"Processed cell_size={cell_size}: {len(clustering_data)} data points")
                    
        except Exception as e:
            print(f"Warning: Could not process {result_dir}: {e}")
            continue

    if not data['cell_sizes']:
        print("No data found for plotting!")
        return

    # Sort data by cell sizes
    sorted_indices = np.argsort(data['cell_sizes'])
    for key in data:
        if key != 'clustering_groups_data' and key != 'memory_data' and key != 'time_data' and key != 'velocity_y_data':
            data[key] = np.array(data[key])[sorted_indices]
        else:
            data[key] = [data[key][i] for i in sorted_indices]

    # Create enhanced plots
    create_main_analysis_plot(data, baseline_data, figures_dir)
    create_performance_analysis_plot(data, figures_dir)
    create_latex_ready_figure(data, baseline_data, figures_dir)
    create_combined_figure(data, baseline_data, figures_dir)
    
    print("\nAll enhanced plots generated successfully!")
    print(f"Check the 'figures' directory: {figures_dir}")
    print("=" * 60)


if __name__ == '__main__':
    breaking_lanes()
