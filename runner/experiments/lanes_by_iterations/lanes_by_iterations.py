import json
import os
import subprocess
from src.runner import run_experiment, compile_c_code, compile_model
from src.utils import load_config, create_output_dir, copy_results_to_latest, generate_map
from src.constants import Constants
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.math.Clustering import Clustering
import seaborn as sns
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
from src.config_manager import config_manager

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
warnings.filterwarnings('ignore')   

PEDESTRIAN_COUNT = int(20 * 50 * 0.3)
WIDTH = 20
VOLUMES = 50
RUN_SIMULATION = False

class LanesByIterationsExperiment:
    """
    Enhanced experiment class for analyzing lane formation evolution across iterations.
    """
    
    def __init__(self):
        self.results_dir = Path('experiments/lanes_by_iterations/results')
        self.figures_dir = Path('experiments/lanes_by_iterations/figures')
        self.figures_dir.mkdir(exist_ok=True)
        
    def run(self):
        """Run the complete experiment."""
        print(f"Running iterations for {PEDESTRIAN_COUNT} pedestrians and plotting lanes by iteration...\n")
        if RUN_SIMULATION:
            self._run_simulation()

        self._plot_results()
        
    def _run_simulation(self):
        """Run the simulation experiment."""
        config = load_config('experiments/lanes_by_iterations/config.json')

        # Create output directory with experiment name if provided
        output_dir = create_output_dir('experiments/lanes_by_iterations/results')
        print(f"Created output directory: {output_dir}")

        config['parameters']['N']['value'] = PEDESTRIAN_COUNT
        config['parameters']['PEDESTRIAN_IMPLEMENTATION']['value'] = Constants.PEDESTRIAN_MMOC

        # Replace the map in the config
        generated_map = generate_map(VOLUMES, WIDTH)
        # config['parameters']['OBSTACLES'] = {
        #   "name": "OBSTACLES",
        #   "type": "map",
        #   "map": generated_map
        # }

        # Add from where to where pedestrians are generated
        config['parameters']['FROM_Y'] = {
          "name": "FROM_Y",
          "type": "value",
          "value": (VOLUMES/ 2) - int(WIDTH / 2)
        }
        config['parameters']['TO_Y'] = {
          "name": "TO_Y",
          "type": "value",
          "value": (VOLUMES/ 2) + int(WIDTH / 2)
        }

        # Save config copy in experiment directory
        config_copy_path = os.path.join(output_dir, 'config.json')
        with open(config_copy_path, 'w') as f:
            json.dump(config, f, indent=2)

        subprocess.run(['sed', '-i', r's/\bGRID_DIVISIONS\s*=\s*[0-9]\+/GRID_DIVISIONS = ' + str(VOLUMES) + '/', '../retqss/model/helbing_not_qss.mo'])
        subprocess.run(['sed', '-i', r's/\bN\s*=\s*[0-9]\+/N = ' + str(PEDESTRIAN_COUNT) + '/', '../retqss/model/helbing_not_qss.mo'])

        # Compile the C++ code if requested
        compile_c_code()

        # Compile the model if requested
        compile_model('helbing_not_qss')

        # Deactivate metrics calculation
        config_manager.update_from_dict({
            'skip_metrics': True
        })

        # Run experiment
        run_experiment(
            config, 
            output_dir, 
            'helbing_not_qss', 
            plot=False, 
            copy_results=True
        )

        # Copy results from output directory to latest directory
        copy_results_to_latest(output_dir)

        print(f"\nExperiment completed. Results saved in {output_dir}")

    def _plot_results(self):
        """
        Creates the original simple plots.
        """
        print("Creating plots...")
        
        # Create the enhanced plot with better styling
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 12))
        
        # Left plot - All iterations
        ax1.set_title('Número de Grupos por Tiempo (Todas las Iteraciones)', 
                     fontsize=20, fontweight='bold', pad=20)
        ax1.set_xlabel('Tiempo (segundos)', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Número de Grupos', fontsize=16, fontweight='bold')
        ax1.tick_params(axis='both', which='major', labelsize=12)
        ax1.grid(True, alpha=0.3)

        # Right plot - Averaged
        ax2.set_title('Número de Grupos por Tiempo (Promediado)', 
                     fontsize=20, fontweight='bold', pad=20)
        ax2.set_xlabel('Tiempo (segundos)', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Número de Grupos', fontsize=16, fontweight='bold')
        ax2.tick_params(axis='both', which='major', labelsize=12)
        ax2.grid(True, alpha=0.3)

        # Get the results files
        results_files = [f for f in os.listdir(os.path.join(self.results_dir, 'latest')) if f.endswith('.csv')]

        # Define a color palette for different iterations
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', 
                 '#DDA0DD', '#98D8C8', '#F7DC6F', '#BB8FCE', '#85C1E9']

        # Read the results files
        groups_per_time = {}
        groups_per_time_averaged = {}
        for i, result_file in enumerate(results_files):
            df = pd.read_csv(os.path.join(self.results_dir, 'latest', result_file))
            particles = (len(df.columns) - 1) / 5
            iteration_groups = {}
            
            for _, row in df.iterrows():
                if row['time'] % 10 != 0:
                    continue

                groups = Clustering(df, int(particles)).calculate_groups_by_time(row)
                iteration_groups[row['time']] = len(groups)

                if row['time'] not in groups_per_time_averaged:
                    groups_per_time_averaged[row['time']] = [len(groups)]
                else:
                    groups_per_time_averaged[row['time']].append(len(groups))

            # Plot each iteration with a different color
            color = colors[i % len(colors)]
            ax1.plot(iteration_groups.keys(), iteration_groups.values(), 
                    linewidth=2, alpha=0.8, color=color, 
                    label=f'Iteración {i+1}')
        
        # Add legend to the first plot
        ax1.legend(fontsize=12, frameon=True, fancybox=True, shadow=True, loc='upper right')
        
        mean_groups_per_time = {k: np.mean(v) for k, v in groups_per_time_averaged.items()}
        std_groups_per_time = {k: np.std(v) for k, v in groups_per_time_averaged.items()}
        
        # Plot mean line with enhanced styling and friendly color
        ax2.plot(mean_groups_per_time.keys(), mean_groups_per_time.values(), 
                linewidth=3, color='#2ECC71', label='Promedio de Grupos')
        
        # Plot standard deviation fill with enhanced styling and friendly color
        ax2.fill_between(
            list(mean_groups_per_time.keys()), 
            (np.array(list(mean_groups_per_time.values())) - np.array(list(std_groups_per_time.values()))), 
            (np.array(list(mean_groups_per_time.values())) + np.array(list(std_groups_per_time.values()))), 
            alpha=0.3, color='#2ECC71', label='Desviación Estándar'
        )
        
        # Add enhanced legend
        ax2.legend(fontsize=14, frameon=True, fancybox=True, shadow=True, loc='upper right')

        # Add main title for the entire figure
        fig.suptitle('Análisis de Formación de Carriles por Iteraciones', 
                    fontsize=24, fontweight='bold', y=0.95)
        
        # Adjust layout and save with high quality
        plt.tight_layout()
        fig.savefig(os.path.join(self.figures_dir, 'groups_by_iterations.png'), 
                   dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close()
        
        print("Plots saved!")
        


def lanes_by_iterations():
    """Main function to run the enhanced lanes by iterations experiment."""
    experiment = LanesByIterationsExperiment()
    experiment.run()



if __name__ == '__main__':
    lanes_by_iterations()
