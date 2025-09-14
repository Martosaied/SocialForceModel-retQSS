import json
import os
import subprocess
import warnings
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

from src.runner import run_experiment, compile_c_code, compile_model
from src.utils import load_config, create_output_dir, copy_results_to_latest, generate_map
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
WIDTHS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
PEDESTRIAN_DENSITY = 0.3
VOLUMES = 50
GRID_SIZE = 50
CELL_SIZE = GRID_SIZE / VOLUMES


class LanesByWidthExperiment:
    """
    A simplified class to manage the lanes by width experiment.
    Only reads from metrics.csv files and creates a single plot.
    """
    
    def __init__(self, output_base_dir: str = 'experiments/lanes_by_width'):
        self.output_base_dir = Path(output_base_dir)
        self.results_dir = self.output_base_dir / 'results'
        self.figures_dir = self.output_base_dir / 'figures'
        self.figures_dir.mkdir(exist_ok=True)
    
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
        print("=" * 60)
        
        if run_all:
            for width in WIDTHS:
                print(f"\nRunning experiments for width: {width}")
                self._run_single_experiment(width)
        
        # Analyze and plot results
        print("\nAnalyzing results and generating plot...")
        self._create_lanes_plot()
        
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
    
    def _create_lanes_plot(self) -> None:
        """
        Create a single plot showing lanes by width with standard deviation and linear fit.
        Reads data from metrics.csv files.
        """
        print("Creating lanes by width plot...")
        
        # Initialize data storage
        lanes_data = {width: [] for width in WIDTHS}
        
        # Get all the results directories
        results_dirs = [d for d in os.listdir('experiments/lanes_by_width/results') 
                       if os.path.isdir(os.path.join('experiments/lanes_by_width/results', d))]
        
        # Read data from metrics.csv files
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
                        lanes_data[width].extend(groups_data)
                        print(f"  Using metrics.csv for width {width}: {len(groups_data)} data points")
                except Exception as e:
                    print(f"Warning: Could not read metrics.csv for width {width}: {e}")
        
        # Calculate statistics
        widths = []
        means = []
        stds = []
        
        for width in sorted(WIDTHS):
            if lanes_data[width]:
                widths.append(width)
                means.append(np.mean(lanes_data[width]))
                stds.append(np.std(lanes_data[width]))
        
        if not widths:
            print("No data found for plotting!")
            return
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Convert to numpy arrays
        widths = np.array(widths)
        means = np.array(means)
        stds = np.array(stds)
        
        # Plot with error bars
        ax.errorbar(widths, means, yerr=stds, fmt='o-', 
                   markersize=10, capsize=6, capthick=2.5, elinewidth=2.5,
                   label='Experimental Data (±1σ)', color='#2E86AB', alpha=0.9, zorder=3)
        
        # Fit linear line
        if len(widths) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(widths, means)
            line_x = np.array([min(widths), max(widths)])
            line_y = slope * line_x + intercept
            ax.plot(line_x, line_y, '--', linewidth=3, color='#F18F01', 
                   label=f'Linear fit (R² = {r_value**2:.3f})', zorder=2)
        
        # Customize plot
        ax.set_xlabel('Corridor Width (cells)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Number of Lane Groups', fontsize=16, fontweight='bold')
        ax.set_title('Lane Formation vs. Corridor Width', fontsize=18, fontweight='bold', pad=25)
        
        ax.grid(True, alpha=0.4, linestyle='--', zorder=1)
        ax.legend(fontsize=14, framealpha=0.95, loc='upper left')
        
        # Add statistical information
        stats_text = f'Data points: {len(widths)}\nPedestrian density: {PEDESTRIAN_DENSITY}\nGrid size: {GRID_SIZE}'
        ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
               verticalalignment='top', fontsize=11,
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.9))
        
        # Set axis limits
        ax.set_xlim(min(widths) - 0.5, max(widths) + 0.5)
        y_min = max(0, min(means - stds) - 0.5)
        y_max = max(means + stds) + 0.5
        ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'lanes_by_width.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"Plot saved to: {self.figures_dir / 'lanes_by_width.png'}")


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
    Simple function to create the lanes by width plot using existing results.
    """
    print("Creating lanes by width plot from existing results...")
    
    experiment = LanesByWidthExperiment()
    experiment._create_lanes_plot()


if __name__ == '__main__':
    # Set to True to run all experiments, False to only analyze existing results
    lanes_by_width(run_experiments=False)