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

PEDESTRIAN_COUNT = int(20 * 50 * 0.3)
WIDTH = 20
VOLUMES = 50

def lanes_by_iterations():
    print(f"Running iterations for {PEDESTRIAN_COUNT} pedestrians and plotting lanes by iteration...\n")
    run()

    # Plot the results
    print("Plotting results...")
    plot_results()

def run():
    """
    Run the experiment for a given number of pedestrians.
    """
    config = load_config('experiments/lanes_by_iterations/config.json')

    # Create output directory with experiment name if provided
    output_dir = create_output_dir('experiments/lanes_by_iterations/results')
    print(f"Created output directory: {output_dir}")

    config['parameters'][0]['value'] = PEDESTRIAN_COUNT
    config['parameters'][1]['value'] = Constants.PEDESTRIAN_MMOC

        # Replace the map in the config
    generated_map = generate_map(VOLUMES,WIDTH)
    config['parameters'].append({
      "name": "OBSTACLES",
      "type": "map",
      "map": generated_map
    })

    # Add from where to where pedestrians are generated
    config['parameters'].append({
      "name": "FROM_Y",
      "type": "value",
      "value": (VOLUMES/ 2) - int(WIDTH / 2)
    })
    config['parameters'].append({
      "name": "TO_Y",
      "type": "value",
      "value": (VOLUMES/ 2) + int(WIDTH / 2)
    })

    # Save config copy in experiment directory
    config_copy_path = os.path.join(output_dir, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)

    subprocess.run(['sed', '-i', r's/\bGRID_DIVISIONS\s*=\s*[0-9]\+/GRID_DIVISIONS = ' + str(VOLUMES) + '/', '../retqss/model/social_force_model.mo'])
    subprocess.run(['sed', '-i', r's/\bN\s*=\s*[0-9]\+/N = ' + str(PEDESTRIAN_COUNT) + '/', '../retqss/model/social_force_model.mo'])

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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.set_title('Number of groups per time(all iterations)')
    ax1.set_xlabel('Time')
    ax1.set_ylabel('Number of groups')

    ax2.set_title('Number of groups per time(averaged)')
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Number of groups')

    # Get the results files
    results_files = [f for f in os.listdir(os.path.join('experiments/lanes_by_iterations/results', 'latest')) if f.endswith('.csv')]

    # Read the results files
    groups_per_time = {}
    groups_per_time_averaged = {}
    for result_file in results_files:
        df = pd.read_csv(os.path.join('experiments/lanes_by_iterations/results/latest', result_file))
        particles = (len(df.columns) - 1) / 5
        for index, row in df.iterrows():
            if index % 5 != 0:
                continue

            groups = Clustering(row, int(particles)).calculate_groups()
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

    fig.savefig(f'experiments/lanes_by_iterations/groups_by_iterations.png')
    plt.close()

if __name__ == '__main__':
    lanes_by_iterations()
