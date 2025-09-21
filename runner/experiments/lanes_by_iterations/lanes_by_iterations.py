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
RUN_SIMULATION = True

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
        
        # Create the original simple plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        ax1.set_title('Number of groups per time(all iterations)')
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Number of groups')

        ax2.set_title('Number of groups per time(averaged)')
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Number of groups')

        # Get the results files
        results_files = [f for f in os.listdir(os.path.join(self.results_dir, 'latest')) if f.endswith('.csv')]

        # Read the results files
        groups_per_time = {}
        groups_per_time_averaged = {}
        for result_file in results_files:
            df = pd.read_csv(os.path.join(self.results_dir, 'latest', result_file))
            particles = (len(df.columns) - 1) / 5
            for _, row in df.iterrows():
                if row['time'] % 10 != 0:
                    continue

                groups = Clustering(df, int(particles)).calculate_groups_by_time(row)
                groups_per_time[row['time']] = len(groups)

                if row['time'] not in groups_per_time_averaged:
                    groups_per_time_averaged[row['time']] = [len(groups)]
                else:
                    groups_per_time_averaged[row['time']].append(len(groups))

            ax1.plot(groups_per_time.keys(), groups_per_time.values())
        
        mean_groups_per_time = {k: np.mean(v) for k, v in groups_per_time_averaged.items()}
        std_groups_per_time = {k: np.std(v) for k, v in groups_per_time_averaged.items()}
        ax2.plot(mean_groups_per_time.keys(), mean_groups_per_time.values(), label='all iterations')
        ax2.fill_between(
            list(mean_groups_per_time.keys()), 
            (np.array(list(mean_groups_per_time.values())) - np.array(list(std_groups_per_time.values()))), 
            (np.array(list(mean_groups_per_time.values())) + np.array(list(std_groups_per_time.values()))), 
            alpha=0.2
        )

        fig.savefig(os.path.join(self.figures_dir, 'groups_by_iterations.png'))
        plt.close()
        
        print("Plots saved!")
        


def lanes_by_iterations():
    """Main function to run the enhanced lanes by iterations experiment."""
    experiment = LanesByIterationsExperiment()
    experiment.run()



if __name__ == '__main__':
    lanes_by_iterations()
