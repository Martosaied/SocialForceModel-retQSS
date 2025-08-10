import json
import os
import subprocess
from src.runner import run_experiment, compile_c_code, compile_model
from src.utils import load_config, create_output_dir, copy_results_to_latest, generate_map
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.math.Density import Density
from src.constants import Constants

PROGRESS_UPDATE_DT = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
WIDTH = 20
GRID_SIZE = 50
PEDESTRIAN_DENSITY = 0.3

def progress_update_dt():
    print("Running experiments for different progress update dt...\n")
    for progress_update_dt in PROGRESS_UPDATE_DT:
        print(f"Running experiment for {progress_update_dt} progress update dt...")
        run(progress_update_dt)
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
    Plot the results of the experiments.
    """
    # Get all the results directories
    results_dirs = [d for d in os.listdir('experiments/progress_update_dt/results') if os.path.isdir(os.path.join('experiments/progress_update_dt/results', d))]

    progress_update_dts = PROGRESS_UPDATE_DT

    # Make 4 subplots
    fig, axs = plt.subplots(3, 1, figsize=(20, 20))
    ax1, ax2, ax3 = axs.flatten()

    ax1.set_title('Progress update dt')
    ax1.set_xlabel('Progress update dt')
    ax1.set_ylabel('Number of lanes formed')

    ax2.set_title('Time per progress update dt')
    ax2.set_xlabel('Progress update dt')
    ax2.set_ylabel('Time(ms)')

    ax3.set_title('Memory usage per progress update dt')
    ax3.set_xlabel('Progress update dt')
    ax3.set_ylabel('Memory usage(KB)')

    clustering_based_groups_per_volume = {}
    memory_per_volume = {}
    time_per_volume = {}
    for result_dir in results_dirs:
        progress_update_dt = float(result_dir.split('progress_update_dt_')[1])
        print(progress_update_dt)

        df = pd.read_csv(os.path.join('experiments/progress_update_dt/results', result_dir, 'latest', 'metrics.csv'))
        clustering_based_groups_per_volume[progress_update_dt] = np.mean(df['clustering_based_groups'].tolist())
        memory_per_volume[progress_update_dt] = np.mean(df['memory_usage'].tolist())
        time_per_volume[progress_update_dt] = np.mean(df['time'].tolist())



    clustering_based_groups_per_volume = dict(sorted(clustering_based_groups_per_volume.items()))
    memory_per_volume = dict(sorted(memory_per_volume.items()))
    time_per_volume = dict(sorted(time_per_volume.items()))

    # Plot a boxplot per amount of volumes
    ax1.plot(progress_update_dts, clustering_based_groups_per_volume.values())
    ax2.plot(progress_update_dts, time_per_volume.values())
    ax3.plot(progress_update_dts, memory_per_volume.values())
    plt.legend()
    plt.savefig(f'experiments/progress_update_dt/progress_update_dt.png')
    plt.close()


if __name__ == '__main__':
    breaking_lanes()
