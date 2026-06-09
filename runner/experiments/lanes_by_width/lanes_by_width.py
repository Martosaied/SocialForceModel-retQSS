import json
import os
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy import stats

from src.math.Clustering import Clustering
from src.utils import load_config, create_output_dir
from src.constants import Constants
from src.experiments import (
    ConfigBuilder, ModelUpdater, ExperimentRunner,
    apply_publication_style, DEFAULT_COLORS
)

# Suppress warnings
warnings.filterwarnings('ignore')

# Apply consistent styling
apply_publication_style()

# Use metrics.csv file to get the data
use_metrics_csv = True
run_experiments = True

# Experiment parameters
WIDTHS = [10]
PEDESTRIAN_DENSITY = 0.3
VOLUMES = 50
GRID_SIZE = 50
CELL_SIZE = GRID_SIZE / VOLUMES


class LanesByWidthExperiment:
    """Manage the lanes by width experiment."""
    
    def __init__(self, output_base_dir: str = 'experiments/lanes_by_width'):
        self.output_base_dir = Path(output_base_dir)
        self.results_dir = self.output_base_dir / 'results'
        self.figures_dir = self.output_base_dir / 'figures'
        self.figures_dir.mkdir(exist_ok=True)
    
    def run_experiment_series(self, run_all: bool = False) -> None:
        """Run the complete experiment series for all widths."""
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
        
        print("\nAnalyzing results and generating plot...")
        self._create_lanes_plot()
        
        print("\nExperiment completed successfully!")
        print(f"Results saved in: {self.results_dir}")
        print(f"Figures saved in: {self.figures_dir}")
    
    def _run_single_experiment(self, width: int) -> None:
        """Run a single experiment for a given width."""
        config = load_config('experiments/lanes_by_width/config.json')
        output_dir = create_output_dir(f'experiments/lanes_by_width/results/width_{width}')
        
        pedestrians = int(PEDESTRIAN_DENSITY * width * GRID_SIZE)
        
        # Use ConfigBuilder for cleaner parameter setup
        ConfigBuilder(config) \
            .set_pedestrian_count(pedestrians) \
            .set_pedestrian_implementation(Constants.PEDESTRIAN_MMOC) \
            .set_corridor(GRID_SIZE, width)
        
        # Save configuration
        config_path = Path(output_dir) / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        # Update model parameters
        model = ModelUpdater('../retqss/model/helbing_only_qss.mo')
        model.update_parameters({
            'GRID_DIVISIONS': VOLUMES,
            'N': pedestrians
        })
        
        # Use ExperimentRunner for compilation and execution
        ExperimentRunner.run_standard_experiment(
            config,
            output_dir,
            'helbing_only_qss',
            copy_results=True,
        )
    
    def _create_lanes_plot(self) -> None:
        """Create plot showing lanes by width with standard deviation and linear fit."""
        print("Creating lanes by width plot...")
        
        lanes_data = {width: [] for width in WIDTHS}
        
        results_root = Path('experiments/lanes_by_width/results')
        width_dirs = [d for d in results_root.iterdir() if d.is_dir()]
        
        for width_dir in width_dirs:
            if not width_dir.name.startswith('width_'):
                continue
            try:
                width = float(width_dir.name.split('_')[1])
            except (IndexError, ValueError):
                continue
            if width not in WIDTHS:
                continue

            latest_dir = width_dir / 'latest'
            if latest_dir.is_dir():
                run_dirs = [latest_dir]
            else:
                run_dirs = sorted([d for d in width_dir.glob('experiment_*') if d.is_dir()])
            
            if use_metrics_csv:
                for run_dir in run_dirs:
                    metrics_path = run_dir / 'metrics.csv'
                    if not metrics_path.exists():
                        continue
                    try:
                        metrics_df = pd.read_csv(metrics_path)
                        if 'clustering_based_groups' in metrics_df.columns:
                            groups_data = metrics_df['clustering_based_groups'].dropna().tolist()
                            lanes_data[width].extend(groups_data)
                            print(f"  Using metrics.csv for width {width}: {len(groups_data)} data points")
                    except Exception as e:
                        print(f"Warning: Could not read metrics.csv for width {width}: {e}")
            else:
                groups_data = []
                for run_dir in run_dirs:
                    for result_file in os.listdir(run_dir):
                        if result_file.endswith('.csv') and result_file != 'metrics.csv':
                            df = pd.read_csv(run_dir / result_file)
                            particles = (len(df.columns) - 1) / 5
                            groups = Clustering(df, int(particles)).calculate_groups(start_index=100, sample_rate=5)
                            groups_data.append(groups)
                lanes_data[width].extend(groups_data)
                print(f"Using solution.csv for width {width}: {len(groups_data)} data points")
        
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
        
        fig, ax = plt.subplots(figsize=(12, 8))
        label_size = 18
        tick_size = 16
        legend_size = 16
        
        widths = np.array(widths)
        means = np.array(means)
        stds = np.array(stds)
        
        ax.errorbar(
            widths,
            means,
            yerr=stds,
            fmt='o',
            color='steelblue',
            zorder=3,
            markersize=8,
            elinewidth=2,
            capsize=4,
        )
        
        if len(widths) > 1:
            slope, intercept, r_value, p_value, std_err = stats.linregress(widths, means)
            line_x = np.array([min(widths), max(widths)])
            line_y = slope * line_x + intercept
            ax.plot(
                line_x,
                line_y,
                '--',
                color='lightcoral',
                label=f'Ajuste Lineal (R² = {r_value**2:.3f})',
                zorder=2,
                linewidth=2,
            )
        
        ax.set_xlabel('Ancho del Corredor (Metros)', fontsize=label_size)
        ax.set_ylabel('Número de Carriles (Promedio)', fontsize=label_size)
        
        ax.grid(True, zorder=1)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.legend(loc='upper left', fontsize=legend_size)
        
        ax.set_xlim(min(widths) - 0.5, max(widths) + 0.5)
        y_min = max(0, min(means - stds) - 0.5)
        y_max = max(means + stds) + 0.5
        ax.set_ylim(y_min, y_max)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'lanes_by_width.png')
        plt.close()
        
        print(f"Plot saved to: {self.figures_dir / 'lanes_by_width.png'}")


def lanes_by_width():
    """Main function to run the lanes by width experiment."""
    experiment = LanesByWidthExperiment()
    experiment.run_experiment_series(run_all=run_experiments)


def plot_results():
    """Create the lanes by width plot using existing results."""
    print("Creating lanes by width plot from existing results...")
    experiment = LanesByWidthExperiment()
    experiment._create_lanes_plot()


if __name__ == '__main__':
    lanes_by_width()
