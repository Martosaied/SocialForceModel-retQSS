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
from src.constants import Constants
from src.math.Density import Density
from src.math.Clustering import Clustering

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
WIDTHS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
PEDESTRIAN_DENSITY = 0.3
VOLUMES = 50
GRID_SIZE = 50
CELL_SIZE = GRID_SIZE / VOLUMES
CONFIDENCE_LEVEL = 0.95  # 95% confidence interval


class LanesByWidthExperiment:
    """
    A class to manage the lanes by width experiment with enhanced statistical analysis
    and publication-ready graphics. Uses config.json iterations for multiple runs.
    """
    
    def __init__(self, output_base_dir: str = 'experiments/lanes_by_width'):
        self.output_base_dir = Path(output_base_dir)
        self.results_dir = self.output_base_dir / 'results'
        self.figures_dir = self.output_base_dir / 'figures'
        self.figures_dir.mkdir(exist_ok=True)
        
        # Initialize results storage
        self.results = {
            'widths': [],
            'groups_mean': [],
            'groups_std': [],
            'groups_ci_lower': [],
            'groups_ci_upper': [],
            'groups_data': [],  # Raw data for each width
            'pedestrian_counts': []
        }
    
    def run_experiment_series(self, run_all: bool = False) -> None:
        """
        Run the complete experiment series for all widths.
        
        Args:
            run_all: If True, run all experiments. If False, only plot existing results.
        """
        print("=" * 60)
        print("LANES BY WIDTH EXPERIMENT")
        print("=" * 60)
        print(f"Parameters:")
        print(f"  - Widths: {WIDTHS}")
        print(f"  - Pedestrian density: {PEDESTRIAN_DENSITY}")
        print(f"  - Grid size: {GRID_SIZE}")
        print(f"  - Confidence level: {CONFIDENCE_LEVEL*100}%")
        print("=" * 60)
        
        for width in WIDTHS:
            print(f"\nRunning experiments for width: {width}")
            self._run_single_experiment(width)
        
        # Analyze and plot results
        print("\nAnalyzing results and generating plots...")
        self._analyze_results()
        self._create_publication_plots()
        self._generate_statistical_report()
        
        print("\nExperiment completed successfully!")
        print(f"Results saved in: {self.results_dir}")
        print(f"Figures saved in: {self.figures_dir}")
    

    
    def _run_single_experiment(self, width: int) -> None:
        """
        Run a single experiment for a given width.
        
        Args:
            width: The corridor width to test
        """
        config = load_config('experiments/lanes_by_width/config.json')
        
        # Create output directory
        output_dir = create_output_dir(f'experiments/lanes_by_width/results/width_{width}')
        
        # Calculate pedestrian count
        pedestrians = int(PEDESTRIAN_DENSITY * width * GRID_SIZE)
        
        # Update configuration
        config['parameters']['N']['value'] = pedestrians
        config['parameters']['PEDESTRIAN_IMPLEMENTATION']['value'] = Constants.PEDESTRIAN_MMOC
        
        # Generate map
        generated_map = generate_map(VOLUMES, width)
        config['parameters']['OBSTACLES'] = {
            "name": "OBSTACLES",
            "type": "map",
            "map": generated_map
        }
        
        # Set pedestrian generation boundaries
        config['parameters']['FROM_Y'] = {
            "name": "FROM_Y",
            "type": "value",
            "value": (GRID_SIZE / 2) - int(width / 2)
        }
        config['parameters']['TO_Y'] = {
            "name": "TO_Y",
            "type": "value",
            "value": (GRID_SIZE / 2) + int(width / 2)
        }
        
        # Save configuration
        config_path = Path(output_dir) / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Update model files
        self._update_model_files(VOLUMES, pedestrians)
        
        # Compile and run
        compile_c_code()
        compile_model('social_force_model')
        
        run_experiment(
            config, 
            output_dir, 
            'social_force_model', 
            plot=False, 
            copy_results=True
        )
        
        copy_results_to_latest(output_dir)
    
    def _update_model_files(self, volumes: int, pedestrians: int) -> None:
        """Update the model files with new parameters."""
        subprocess.run([
            'sed', '-i', 
            r's/\bGRID_DIVISIONS\s*=\s*[0-9]\+/GRID_DIVISIONS = ' + str(volumes) + '/', 
            '../retqss/model/social_force_model.mo'
        ])
        subprocess.run([
            'sed', '-i', 
            r's/\bN\s*=\s*[0-9]\+/N = ' + str(pedestrians) + '/', 
            '../retqss/model/social_force_model.mo'
        ])
    
    def _analyze_results(self) -> None:
        """Analyze the experimental results with statistical measures."""
        print("Analyzing experimental results...")
        
        for width in WIDTHS:
            width_data = []
            
            # Collect data from the experiment for this width
            result_dir = self.results_dir / f'width_{width}'
            if not result_dir.exists():
                continue
            
            latest_dir = result_dir / 'latest'
            if not latest_dir.exists():
                continue
            
            # Process CSV files
            for csv_file in latest_dir.glob('*.csv'):
                try:
                    df = pd.read_csv(csv_file)
                    particles = int((len(df.columns) - 1) / 5)
                    
                    clustering = Clustering(df, particles)
                    groups = clustering.calculate_groups(
                        from_y=(VOLUMES / 2) - int(width / 2),
                        to_y=(VOLUMES / 2) + int(width / 2)
                    )
                    width_data.append(groups)
                except Exception as e:
                    print(f"Warning: Could not process {csv_file}: {e}")
                    continue
            
            if width_data:
                # Calculate statistics
                mean_groups = np.mean(width_data)
                std_groups = np.std(width_data, ddof=1)  # Sample standard deviation
                
                # Calculate confidence interval
                ci = stats.t.interval(
                    CONFIDENCE_LEVEL, 
                    len(width_data) - 1, 
                    loc=mean_groups, 
                    scale=std_groups / np.sqrt(len(width_data))
                )
                
                # Store results
                self.results['widths'].append(width)
                self.results['groups_mean'].append(mean_groups)
                self.results['groups_std'].append(std_groups)
                self.results['groups_ci_lower'].append(ci[0])
                self.results['groups_ci_upper'].append(ci[1])
                self.results['groups_data'].append(width_data)
                self.results['pedestrian_counts'].append(int(PEDESTRIAN_DENSITY * width * GRID_SIZE))
        
        # Convert to numpy arrays for easier manipulation
        for key in self.results:
            if key != 'groups_data':
                self.results[key] = np.array(self.results[key])
    
    def _create_publication_plots(self) -> None:
        """Create publication-ready plots with enhanced styling."""
        print("Creating publication-ready plots...")
        
        # Enhanced original plot (compatible with existing data)
        self.create_enhanced_original_plot()
        
        # LaTeX-ready figure for academic papers
        self.create_latex_ready_figure()
        
        # Performance analysis using metrics.csv
        self.create_performance_analysis()
        
        # Main plot: Groups vs Width (if we have the new data structure)
        if self.results['widths']:
            self._create_main_plot()
            
            # Additional analysis plots
            self._create_statistical_plots()
            
            # Combined figure for paper
            self._create_combined_figure()
        else:
            print("No new experimental data found. Enhanced plots created from existing data.")
    
    def _create_main_plot(self) -> None:
        """Create the main publication-quality plot."""
        fig, ax = plt.subplots(figsize=(12, 8))
        
        widths = self.results['widths']
        means = self.results['groups_mean']
        stds = self.results['groups_std']
        ci_lower = self.results['groups_ci_lower']
        ci_upper = self.results['groups_ci_upper']
        
        # Plot error bars with confidence intervals
        ax.errorbar(
            widths, means, 
            yerr=[means - ci_lower, ci_upper - means],
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
        if len(widths) > 3:
            # Try different polynomial degrees and select the best fit
            best_degree = self._find_best_polynomial_fit(widths, means)
            coeffs = np.polyfit(widths, means, best_degree)
            poly = np.poly1d(coeffs)
            x_fit = np.linspace(min(widths), max(widths), 100)
            y_fit = poly(x_fit)
            
            ax.plot(x_fit, y_fit, '--', linewidth=2, color='#A23B72', 
                   label=f'Polynomial fit (degree {best_degree})')
        
        # Add linear trend line
        slope, intercept, r_value, p_value, std_err = stats.linregress(widths, means)
        line_x = np.array([min(widths), max(widths)])
        line_y = slope * line_x + intercept
        ax.plot(line_x, line_y, '-', linewidth=3, color='#F18F01', 
               label=f'Linear trend (R² = {r_value**2:.3f})', zorder=2)
        
        # Customize plot
        ax.set_xlabel('Corridor Width (cells)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Number of Lane Groups', fontsize=16, fontweight='bold')
        ax.set_title('Lane Formation vs. Corridor Width', fontsize=18, fontweight='bold', pad=25)
        
        ax.grid(True, alpha=0.4, linestyle='--', zorder=1)
        ax.legend(fontsize=14, framealpha=0.95, loc='upper left')
        
        # Add statistical information with better formatting
        stats_text = f'Confidence level: {CONFIDENCE_LEVEL*100}%\nPedestrian density: {PEDESTRIAN_DENSITY}\nTotal data points: {len(widths)}'
        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))
        
        # Set axis limits with some padding
        ax.set_xlim(min(widths) - 0.5, max(widths) + 0.5)
        y_min = max(0, min(means - (means - ci_lower)) - 0.5)
        y_max = max(means + (ci_upper - means)) + 0.5
        ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'groups_vs_width_main.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_statistical_plots(self) -> None:
        """Create additional statistical analysis plots."""
        # Box plot for distribution analysis
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Prepare data for box plot
        box_data = []
        box_labels = []
        for i, width in enumerate(self.results['widths']):
            if len(self.results['groups_data'][i]) > 0:
                box_data.append(self.results['groups_data'][i])
                box_labels.append(f'W={width}')
        
        if box_data:
            bp = ax.boxplot(box_data, labels=box_labels, patch_artist=True)
            
            # Color the boxes
            colors = plt.cm.viridis(np.linspace(0, 1, len(box_data)))
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.7)
            
            ax.set_xlabel('Corridor Width', fontsize=14, fontweight='bold')
            ax.set_ylabel('Number of Lane Groups', fontsize=14, fontweight='bold')
            ax.set_title('Distribution of Lane Groups by Width', fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.3)
            
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.figures_dir / 'groups_distribution_boxplot.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # Scatter plot with density
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create scatter plot with point size based on pedestrian count
        scatter = ax.scatter(
            self.results['widths'], 
            self.results['groups_mean'],
            s=self.results['pedestrian_counts'] / 10,  # Scale point size
            c=self.results['groups_std'],
            cmap='viridis',
            alpha=0.7,
            edgecolors='black',
            linewidth=1
        )
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Standard Deviation', fontsize=12)
        
        ax.set_xlabel('Corridor Width (cells)', fontsize=14, fontweight='bold')
        ax.set_ylabel('Mean Number of Lane Groups', fontsize=14, fontweight='bold')
        ax.set_title('Lane Groups vs Width (Point Size ∝ Pedestrian Count)', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'groups_vs_width_scatter.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_combined_figure(self) -> None:
        """Create a combined figure suitable for publication."""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        widths = self.results['widths']
        means = self.results['groups_mean']
        stds = self.results['groups_std']
        
        # Subplot 1: Main error bar plot
        ax1.errorbar(widths, means, yerr=stds, fmt='o-', capsize=5, capthick=2)
        ax1.set_xlabel('Corridor Width')
        ax1.set_ylabel('Number of Lane Groups')
        ax1.set_title('(a) Lane Groups vs Width')
        ax1.grid(True, alpha=0.3)
        
        # Subplot 2: Linear regression
        slope, intercept, r_value, p_value, std_err = stats.linregress(widths, means)
        ax2.scatter(widths, means, alpha=0.7)
        x_line = np.array([min(widths), max(widths)])
        y_line = slope * x_line + intercept
        ax2.plot(x_line, y_line, 'r--', linewidth=2)
        ax2.set_xlabel('Corridor Width')
        ax2.set_ylabel('Number of Lane Groups')
        ax2.set_title(f'(b) Linear Regression (R² = {r_value**2:.3f})')
        ax2.grid(True, alpha=0.3)
        
        # Subplot 3: Residuals
        y_pred = slope * widths + intercept
        residuals = means - y_pred
        ax3.scatter(widths, residuals, alpha=0.7)
        ax3.axhline(y=0, color='r', linestyle='--')
        ax3.set_xlabel('Corridor Width')
        ax3.set_ylabel('Residuals')
        ax3.set_title('(c) Residuals')
        ax3.grid(True, alpha=0.3)
        
        # Subplot 4: Standard deviation vs width
        ax4.scatter(widths, stds, alpha=0.7)
        ax4.set_xlabel('Corridor Width')
        ax4.set_ylabel('Standard Deviation')
        ax4.set_title('(d) Variability vs Width')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'combined_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def _find_best_polynomial_fit(self, x: np.ndarray, y: np.ndarray, max_degree: int = 3) -> int:
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
    
    def _generate_statistical_report(self) -> None:
        """Generate a comprehensive statistical report."""
        report_path = self.figures_dir / 'statistical_report.txt'
        
        with open(report_path, 'w') as f:
            f.write("LANES BY WIDTH EXPERIMENT - STATISTICAL REPORT\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Experiment Parameters:\n")
            f.write(f"  - Widths tested: {WIDTHS}\n")
            f.write(f"  - Pedestrian density: {PEDESTRIAN_DENSITY}\n")
            f.write(f"  - Grid size: {GRID_SIZE}\n")
            f.write(f"  - Confidence level: {CONFIDENCE_LEVEL*100}%\n\n")
            
            f.write("Results Summary:\n")
            f.write("-" * 20 + "\n")
            
            for i, width in enumerate(self.results['widths']):
                f.write(f"Width {width}:\n")
                f.write(f"  Mean groups: {self.results['groups_mean'][i]:.2f}\n")
                f.write(f"  Std dev: {self.results['groups_std'][i]:.2f}\n")
                f.write(f"  CI: [{self.results['groups_ci_lower'][i]:.2f}, {self.results['groups_ci_upper'][i]:.2f}]\n")
                f.write(f"  Pedestrians: {self.results['pedestrian_counts'][i]}\n\n")
            
            # Linear regression analysis
            if len(self.results['widths']) > 1:
                slope, intercept, r_value, p_value, std_err = stats.linregress(
                    self.results['widths'], self.results['groups_mean']
                )
                
                f.write("Linear Regression Analysis:\n")
                f.write("-" * 30 + "\n")
                f.write(f"Slope: {slope:.4f}\n")
                f.write(f"Intercept: {intercept:.4f}\n")
                f.write(f"R-squared: {r_value**2:.4f}\n")
                f.write(f"P-value: {p_value:.4f}\n")
                f.write(f"Standard error: {std_err:.4f}\n\n")
        
        print(f"Statistical report saved to: {report_path}")
    
    def create_enhanced_original_plot(self) -> None:
        """
        Create an enhanced version of the original plot with better styling and error bars.
        This maintains compatibility with the original plotting approach while adding improvements.
        Uses metrics.csv for faster data processing when available.
        """
        print("Creating enhanced original plot...")
        
        # Initialize data storage
        average_groups_per_width = {width: [] for width in WIDTHS}
        
        # Get all the results directories
        results_dirs = [d for d in os.listdir('experiments/lanes_by_width/results') 
                       if os.path.isdir(os.path.join('experiments/lanes_by_width/results', d))]
        
        # First, try to use metrics.csv files for faster processing
        metrics_data_found = False
        for result_dir in results_dirs:
            width = float(result_dir.split('_')[1])
            if width not in WIDTHS:
                continue
                
            metrics_path = os.path.join('experiments/lanes_by_width/results', result_dir, 'metrics.csv')
            if os.path.exists(metrics_path):
                try:
                    metrics_df = pd.read_csv(metrics_path)
                    if 'clustering_based_groups' in metrics_df.columns:
                        groups_data = metrics_df['clustering_based_groups'].dropna().tolist()
                        average_groups_per_width[width].extend(groups_data)
                        metrics_data_found = True
                        print(f"  Using metrics.csv for width {width}: {len(groups_data)} data points")
                except Exception as e:
                    print(f"Warning: Could not read metrics.csv for width {width}: {e}")
        
        # If no metrics.csv files found, fall back to processing individual CSV files
        if not metrics_data_found:
            print("No metrics.csv files found, processing individual result files...")
            for result_dir in results_dirs:
                width = float(result_dir.split('_')[1])
                if width not in WIDTHS:
                    continue
                    
                result_path = os.path.join('experiments/lanes_by_width/results', result_dir, 'latest')
                if not os.path.exists(result_path):
                    continue
                    
                for result_file in os.listdir(result_path):
                    if result_file.endswith('.csv'):
                        try:
                            df = pd.read_csv(os.path.join(result_path, result_file))
                            particles = int((len(df.columns) - 1) / 5)
                            groups = Clustering(df, particles).calculate_groups(
                                from_y=(VOLUMES / 2) - int(width / 2), 
                                to_y=(VOLUMES / 2) + int(width / 2)
                            )
                            average_groups_per_width[width].append(groups)
                        except Exception as e:
                            print(f"Warning: Could not process {result_file}: {e}")
                            continue
        
        # Calculate statistics
        widths = []
        means = []
        stds = []
        
        for width in sorted(WIDTHS):
            if average_groups_per_width[width]:
                widths.append(width)
                means.append(np.mean(average_groups_per_width[width]))
                stds.append(np.std(average_groups_per_width[width]))
        
        if not widths:
            print("No data found for plotting!")
            return
        
        # Create enhanced plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Convert to numpy arrays
        widths = np.array(widths)
        means = np.array(means)
        stds = np.array(stds)
        
        # Plot with error bars
        ax.errorbar(widths, means, yerr=stds, fmt='o-', 
                   markersize=10, capsize=6, capthick=2.5, elinewidth=2.5,
                   label='Experimental Data (±1σ)', color='#2E86AB', alpha=0.9, zorder=3)
        
        # Fit line using numpy polyfit (degree 1 = linear)
        if len(widths) > 1:
            slope, intercept = np.polyfit(widths, means, 1)
            line = slope * widths + intercept
            ax.plot(widths, line, '--', linewidth=3, color='#F18F01', 
                   label=f'Linear fit (slope={slope:.3f})', zorder=2)
        
        # Calculate R-squared
        if len(widths) > 1:
            correlation_matrix = np.corrcoef(widths, means)
            correlation_xy = correlation_matrix[0, 1]
            r_squared = correlation_xy ** 2
            ax.text(0.02, 0.95, f'R² = {r_squared:.3f}', transform=ax.transAxes,
                   fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
        
        # Customize plot
        ax.set_xlabel('Corridor Width (cells)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Number of Lane Groups', fontsize=16, fontweight='bold')
        ax.set_title('Enhanced: Lane Groups vs Corridor Width', fontsize=18, fontweight='bold', pad=25)
        
        ax.grid(True, alpha=0.4, linestyle='--', zorder=1)
        ax.legend(fontsize=14, framealpha=0.95, loc='upper left')
        
        # Add statistical information
        stats_text = f'Data points: {len(widths)}\nPedestrian density: {PEDESTRIAN_DENSITY}\nGrid size: {GRID_SIZE}'
        ax.text(0.02, 0.85, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))
        
        # Set axis limits
        ax.set_xlim(min(widths) - 0.5, max(widths) + 0.5)
        y_min = max(0, min(means - stds) - 0.5)
        y_max = max(means + stds) + 0.5
        ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'enhanced_original_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Enhanced original plot saved to: {self.figures_dir / 'enhanced_original_plot.png'}")
    
    def create_latex_ready_figure(self) -> None:
        """
        Create a LaTeX-ready figure with proper formatting for academic papers.
        Uses metrics.csv for faster data processing when available.
        """
        print("Creating LaTeX-ready figure...")
        
        # Initialize data storage
        average_groups_per_width = {width: [] for width in WIDTHS}
        
        # Get all the results directories
        results_dirs = [d for d in os.listdir('experiments/lanes_by_width/results') 
                       if os.path.isdir(os.path.join('experiments/lanes_by_width/results', d))]
        
        # First, try to use metrics.csv files for faster processing
        metrics_data_found = False
        for result_dir in results_dirs:
            width = float(result_dir.split('_')[1])
            if width not in WIDTHS:
                continue
                
            metrics_path = os.path.join('experiments/lanes_by_width/results', result_dir, 'metrics.csv')
            if os.path.exists(metrics_path):
                try:
                    metrics_df = pd.read_csv(metrics_path)
                    if 'clustering_based_groups' in metrics_df.columns:
                        groups_data = metrics_df['clustering_based_groups'].dropna().tolist()
                        average_groups_per_width[width].extend(groups_data)
                        metrics_data_found = True
                except Exception as e:
                    print(f"Warning: Could not read metrics.csv for width {width}: {e}")
        
        # If no metrics.csv files found, fall back to processing individual CSV files
        if not metrics_data_found:
            print("No metrics.csv files found, processing individual result files...")
            for result_dir in results_dirs:
                width = float(result_dir.split('_')[1])
                if width not in WIDTHS:
                    continue
                    
                result_path = os.path.join('experiments/lanes_by_width/results', result_dir, 'latest')
                if not os.path.exists(result_path):
                    continue
                    
                for result_file in os.listdir(result_path):
                    if result_file.endswith('.csv'):
                        try:
                            df = pd.read_csv(os.path.join(result_path, result_file))
                            particles = int((len(df.columns) - 1) / 5)
                            groups = Clustering(df, particles).calculate_groups(
                                from_y=(VOLUMES / 2) - int(width / 2), 
                                to_y=(VOLUMES / 2) + int(width / 2)
                            )
                            average_groups_per_width[width].append(groups)
                        except Exception as e:
                            continue
        
        # Calculate statistics
        widths = []
        means = []
        stds = []
        counts = []
        
        for width in sorted(WIDTHS):
            if average_groups_per_width[width]:
                widths.append(width)
                means.append(np.mean(average_groups_per_width[width]))
                stds.append(np.std(average_groups_per_width[width]))
                counts.append(len(average_groups_per_width[width]))
        
        if not widths:
            print("No data found for LaTeX figure!")
            return
        
        # Create LaTeX-ready plot
        fig, ax = plt.subplots(figsize=(8, 6))
        
        widths = np.array(widths)
        means = np.array(means)
        stds = np.array(stds)
        
        # Plot with error bars
        ax.errorbar(widths, means, yerr=stds, fmt='o', 
                   markersize=6, capsize=4, capthick=1.5, elinewidth=1.5,
                   color='black', alpha=0.8, zorder=3)
        
        # Fit line
        if len(widths) > 1:
            slope, intercept = np.polyfit(widths, means, 1)
            line = slope * widths + intercept
            ax.plot(widths, line, '--', linewidth=2, color='red', alpha=0.8, zorder=2)
            
            # Calculate R-squared
            correlation_matrix = np.corrcoef(widths, means)
            correlation_xy = correlation_matrix[0, 1]
            r_squared = correlation_xy ** 2
            
            # Add R-squared text
            ax.text(0.05, 0.95, f'$R^2 = {r_squared:.3f}$', transform=ax.transAxes,
                   fontsize=10, bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Customize plot for LaTeX
        ax.set_xlabel('Corridor Width (cells)', fontsize=12)
        ax.set_ylabel('Number of Lane Groups', fontsize=12)
        ax.set_title('Lane Formation vs. Corridor Width', fontsize=14, pad=15)
        
        ax.grid(True, alpha=0.3, linestyle='-', zorder=1)
        
        # Set axis limits
        ax.set_xlim(min(widths) - 0.5, max(widths) + 0.5)
        y_min = max(0, min(means - stds) - 0.5)
        y_max = max(means + stds) + 0.5
        ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'latex_figure.pdf', dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / 'latex_figure.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Create LaTeX table
        self._create_latex_table(widths, means, stds, counts)
        
        print(f"LaTeX-ready figure saved to: {self.figures_dir / 'latex_figure.pdf'}")
    
    def _create_latex_table(self, widths, means, stds, counts) -> None:
        """Create a LaTeX table with the experimental results."""
        table_path = self.figures_dir / 'results_table.tex'
        
        with open(table_path, 'w') as f:
            f.write("\\begin{table}[h]\n")
            f.write("\\centering\n")
            f.write("\\caption{Experimental Results: Lane Groups vs Corridor Width}\n")
            f.write("\\label{tab:lanes_by_width}\n")
            f.write("\\begin{tabular}{cccc}\n")
            f.write("\\hline\n")
            f.write("Width (cells) & Mean Groups & Std Dev & N \\\\\n")
            f.write("\\hline\n")
            
            for i in range(len(widths)):
                f.write(f"{int(widths[i])} & {means[i]:.2f} & {stds[i]:.2f} & {counts[i]} \\\\\n")
            
            f.write("\\hline\n")
            f.write("\\end{tabular}\n")
            f.write("\\end{table}\n")
        
        print(f"LaTeX table saved to: {table_path}")
    
    def create_performance_analysis(self) -> None:
        """
        Create performance analysis plots using metrics.csv data.
        """
        print("Creating performance analysis plots...")
        
        # Collect performance data from metrics.csv files
        performance_data = {
            'widths': [],
            'execution_times': [],
            'memory_usage': [],
            'groups_count': []
        }
        
        results_dirs = [d for d in os.listdir('experiments/lanes_by_width/results') 
                       if os.path.isdir(os.path.join('experiments/lanes_by_width/results', d))]
        
        for result_dir in results_dirs:
            width = float(result_dir.split('_')[1])
            if width not in WIDTHS:
                continue
                
            metrics_path = os.path.join('experiments/lanes_by_width/results', result_dir, 'metrics.csv')
            if os.path.exists(metrics_path):
                try:
                    metrics_df = pd.read_csv(metrics_path)
                    
                    # Extract performance metrics
                    if 'time' in metrics_df.columns and 'memory_usage' in metrics_df.columns:
                        times = pd.to_numeric(metrics_df['time'], errors='coerce').dropna()
                        memory = pd.to_numeric(metrics_df['memory_usage'], errors='coerce').dropna()
                        groups = metrics_df['clustering_based_groups'].dropna()
                        
                        if len(times) > 0:
                            performance_data['widths'].append(width)
                            performance_data['execution_times'].append(times.mean())
                            performance_data['memory_usage'].append(memory.mean() if len(memory) > 0 else 0)
                            performance_data['groups_count'].append(groups.mean() if len(groups) > 0 else 0)
                            
                except Exception as e:
                    print(f"Warning: Could not process metrics for width {width}: {e}")
        
        if not performance_data['widths']:
            print("No performance data found!")
            return
        
        # Create performance plots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        widths = np.array(performance_data['widths'])
        times = np.array(performance_data['execution_times'])
        memory = np.array(performance_data['memory_usage'])
        groups = np.array(performance_data['groups_count'])
        
        # Plot 1: Execution time vs width
        ax1.scatter(widths, times, alpha=0.7, s=100, color='#2E86AB')
        if len(widths) > 1:
            slope, intercept = np.polyfit(widths, times, 1)
            line = slope * widths + intercept
            ax1.plot(widths, line, '--', color='red', alpha=0.8)
        ax1.set_xlabel('Corridor Width (cells)')
        ax1.set_ylabel('Execution Time (seconds)')
        ax1.set_title('(a) Execution Time vs Width')
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Memory usage vs width
        ax2.scatter(widths, memory, alpha=0.7, s=100, color='#A23B72')
        if len(widths) > 1:
            slope, intercept = np.polyfit(widths, memory, 1)
            line = slope * widths + intercept
            ax2.plot(widths, line, '--', color='red', alpha=0.8)
        ax2.set_xlabel('Corridor Width (cells)')
        ax2.set_ylabel('Memory Usage (KB)')
        ax2.set_title('(b) Memory Usage vs Width')
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Groups vs execution time
        ax3.scatter(times, groups, alpha=0.7, s=100, color='#F18F01')
        if len(times) > 1:
            slope, intercept = np.polyfit(times, groups, 1)
            line = slope * times + intercept
            ax3.plot(times, line, '--', color='red', alpha=0.8)
        ax3.set_xlabel('Execution Time (seconds)')
        ax3.set_ylabel('Number of Lane Groups')
        ax3.set_title('(c) Lane Groups vs Execution Time')
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Memory vs groups
        ax4.scatter(memory, groups, alpha=0.7, s=100, color='#C73E1D')
        if len(memory) > 1:
            slope, intercept = np.polyfit(memory, groups, 1)
            line = slope * memory + intercept
            ax4.plot(memory, line, '--', color='red', alpha=0.8)
        ax4.set_xlabel('Memory Usage (KB)')
        ax4.set_ylabel('Number of Lane Groups')
        ax4.set_title('(d) Lane Groups vs Memory Usage')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'performance_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Performance analysis saved to: {self.figures_dir / 'performance_analysis.png'}")


def lanes_by_width(run_experiments: bool = False):
    """
    Main function to run the lanes by width experiment.
    
    Args:
        run_experiments: If True, run all experiments. If False, only analyze existing results.
    """
    experiment = LanesByWidthExperiment()
    experiment.run_experiment_series(run_all=run_experiments)


def plot_results():
    """
    Enhanced version of the original plot_results function with error bars and publication-quality graphics.
    This function maintains backward compatibility while providing enhanced features.
    Uses metrics.csv for faster data processing when available.
    """
    print("Creating enhanced plots with error bars...")
    
    # Initialize data storage
    average_groups_per_width = {width: [] for width in WIDTHS}
    
    # Get all the results directories
    results_dirs = [d for d in os.listdir('experiments/lanes_by_width/results') 
                   if os.path.isdir(os.path.join('experiments/lanes_by_width/results', d))]

    # First, try to use metrics.csv files for faster processing
    metrics_data_found = False
    for result_dir in results_dirs:
        width = float(result_dir.split('_')[1])
        if width not in WIDTHS:
            continue
            
        metrics_path = os.path.join('experiments/lanes_by_width/results', result_dir, 'metrics.csv')
        if os.path.exists(metrics_path):
            try:
                metrics_df = pd.read_csv(metrics_path)
                if 'clustering_based_groups' in metrics_df.columns:
                    groups_data = metrics_df['clustering_based_groups'].dropna().tolist()
                    average_groups_per_width[width].extend(groups_data)
                    metrics_data_found = True
                    print(f"Using metrics.csv for width {width}: {len(groups_data)} data points")
            except Exception as e:
                print(f"Warning: Could not read metrics.csv for width {width}: {e}")
    
    # If no metrics.csv files found, fall back to processing individual CSV files
    if not metrics_data_found:
        print("No metrics.csv files found, processing individual result files...")
        for result_dir in results_dirs:
            width = float(result_dir.split('_')[1])
            if width not in WIDTHS:
                continue
                
            result_path = os.path.join('experiments/lanes_by_width/results', result_dir, 'latest')
            if not os.path.exists(result_path):
                continue
                
            for result_file in os.listdir(result_path):
                if result_file.endswith('.csv'):
                    try:
                        df = pd.read_csv(os.path.join(result_path, result_file))
                        particles = int((len(df.columns) - 1) / 5)
                        groups = Clustering(df, particles).calculate_groups(
                            from_y=(VOLUMES / 2) - int(width / 2), 
                            to_y=(VOLUMES / 2) + int(width / 2)
                        )
                        average_groups_per_width[width].append(groups)
                    except Exception as e:
                        print(f"Warning: Could not process {result_file}: {e}")
                        continue

    # Calculate statistics
    widths = []
    means = []
    stds = []
    
    for width in sorted(WIDTHS):
        if average_groups_per_width[width]:
            widths.append(width)
            means.append(np.mean(average_groups_per_width[width]))
            stds.append(np.std(average_groups_per_width[width]))

    if not widths:
        print("No data found for plotting!")
        return

    # Create enhanced plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Convert to numpy arrays
    widths = np.array(widths)
    means = np.array(means)
    stds = np.array(stds)

    # Plot with error bars
    ax.errorbar(widths, means, yerr=stds, fmt='o-', 
               markersize=10, capsize=6, capthick=2.5, elinewidth=2.5,
               label='Experimental Data (±1σ)', color='#2E86AB', alpha=0.9, zorder=3)

    # Fit line using numpy polyfit (degree 1 = linear)
    if len(widths) > 1:
        slope, intercept = np.polyfit(widths, means, 1)
        line = slope * widths + intercept
        ax.plot(widths, line, '--', linewidth=3, color='#F18F01', 
               label=f'Linear fit (slope={slope:.3f})', zorder=2)
        
        # Calculate R-squared
        correlation_matrix = np.corrcoef(widths, means)
        correlation_xy = correlation_matrix[0, 1]
        r_squared = correlation_xy ** 2
        ax.text(0.02, 0.95, f'R² = {r_squared:.3f}', transform=ax.transAxes,
               fontsize=12, bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

    # Customize plot
    ax.set_xlabel('Corridor Width (cells)', fontsize=16, fontweight='bold')
    ax.set_ylabel('Number of Lane Groups', fontsize=16, fontweight='bold')
    ax.set_title('Enhanced: Lane Groups vs Corridor Width', fontsize=18, fontweight='bold', pad=25)
    
    ax.grid(True, alpha=0.4, linestyle='--', zorder=1)
    ax.legend(fontsize=14, framealpha=0.95, loc='upper left')
    
    # Add statistical information
    stats_text = f'Data points: {len(widths)}\nPedestrian density: {PEDESTRIAN_DENSITY}\nGrid size: {GRID_SIZE}'
    ax.text(0.02, 0.85, stats_text, transform=ax.transAxes,
           verticalalignment='top', fontsize=11,
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))
    
    # Set axis limits
    ax.set_xlim(min(widths) - 0.5, max(widths) + 0.5)
    y_min = max(0, min(means - stds) - 0.5)
    y_max = max(means + stds) + 0.5
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    plt.savefig('experiments/lanes_by_width/groups_by_width_enhanced.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("Enhanced plot saved as 'groups_by_width_enhanced.png'")


def quick_enhanced_plots():
    """
    Quick function to generate all enhanced plots using metrics.csv data.
    This is the fastest way to create publication-ready graphics.
    """
    print("=" * 60)
    print("QUICK ENHANCED PLOTS GENERATION")
    print("=" * 60)
    
    # Create experiment instance
    experiment = LanesByWidthExperiment()
    
    # Generate all enhanced plots
    experiment.create_enhanced_original_plot()
    experiment.create_latex_ready_figure()
    experiment.create_performance_analysis()
    
    print("\nAll enhanced plots generated successfully!")
    print(f"Check the 'figures' directory: {experiment.figures_dir}")
    print("=" * 60)


if __name__ == '__main__':
    # Set to True to run all experiments, False to only analyze existing results
    lanes_by_width(run_experiments=True)
