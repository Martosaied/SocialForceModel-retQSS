import json
import os
import subprocess
from src.runner import run_experiment, compile_c_code, compile_model
from src.utils import load_config, create_output_dir, copy_results_to_latest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.math.Density import Density
from src.constants import Constants

VOLUMES = [1,3,5,10,20,30,40,50,60,70,80,90,100,110,120,130,140,150,160,170,180,190,200]
PEDESTRIANS_IMPLEMENTATION = {
    1: "retqss",
}
WIDTH = 20
GRID_SIZE = 50
PEDESTRIAN_DENSITY = 0.3

def breaking_lanes():
    print("Running experiments for 300 pedestrians in different volumes to see if the lanes break...\n")
    for volume in VOLUMES:
        for implementation in PEDESTRIANS_IMPLEMENTATION:
            print(f"Running experiment for {volume} volumes with implementation {implementation}...")
            # run(volume, implementation)
            print(f"Experiment for {volume} volumes with implementation {implementation} completed.\n")

    # Plot the results
    print("Plotting results...")
    plot_results()

def run(volume, implementation):
    """
    Run the experiment for a given number of pedestrians.
    """
    config = load_config('config.json')

    # Create output directory with experiment name if provided
    output_dir = create_output_dir(
        'experiments/lanes_by_volumes/results', 
        f'volume_{volume}_implementation_{PEDESTRIANS_IMPLEMENTATION[implementation]}'
    )
    print(f"Created output directory: {output_dir}")

    pedestrians = int(PEDESTRIAN_DENSITY * WIDTH * GRID_SIZE)
    config['parameters'][0]['value'] = pedestrians
    config['parameters'][1]['value'] = Constants.PEDESTRIAN_NEIGHBORHOOD
    config['parameters'][2]['value'] = Constants.BORDER_NONE

    # Add from where to where pedestrians are generated
    config['parameters'].append({
      "name": "FROM_Y",
      "type": "value",
      "value": (GRID_SIZE/ 2) - int(WIDTH / 2)
    })
    config['parameters'].append({
      "name": "TO_Y",
      "type": "value",
      "value": (GRID_SIZE/ 2) + int(WIDTH / 2)
    })


    # Save config copy in experiment directory
    config_copy_path = os.path.join(output_dir, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)

    subprocess.run(['sed', '-i', r's/\bGRID_DIVISIONS\s*=\s*[0-9]\+/GRID_DIVISIONS = ' + str(volume) + '/', '../retqss/model/social_force_model.mo'])
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
    results_dirs = [d for d in os.listdir('experiments/lanes_by_volumes/results') if os.path.isdir(os.path.join('experiments/lanes_by_volumes/results', d))]

    # Separate the results directories by implementation
    results_dirs_by_implementation = {
        "retqss": []
    }
    for results_dir in results_dirs:
        implementation = results_dir.split('_implementation_')[1]
        results_dirs_by_implementation[implementation].append(results_dir)

    plt.figure(figsize=(20, 20))
    plt.title('Number of groups per amount of volumes')
    plt.xlabel('Amount of volumes per side')
    plt.ylabel('Number of lanes formed')
    groups_per_volume = {}
    for result_dir in results_dirs_by_implementation['retqss']:
        # Get the results files
        results_files = [f for f in os.listdir(os.path.join('experiments/lanes_by_volumes/results', result_dir, 'latest')) if f.endswith('.csv')]

        # Read the results files
        all_groups = []
        amount_of_volumes = int(result_dir.split('volume_')[1].split('_')[0])
        for result_file in results_files:
            df = pd.read_csv(os.path.join('experiments/lanes_by_volumes/results', result_dir, 'latest', result_file))
            particles = int((len(df.columns) - 1) / 5)
            groups = Density().calculate_lanes_by_density(df, particles)
            all_groups.append(groups)

        
        groups_per_volume[amount_of_volumes] = np.mean(all_groups)

    groups_per_volume = dict(sorted(groups_per_volume.items()))
    # Plot a boxplot per amount of volumes
    plt.bar(list(map(str, groups_per_volume.keys())), groups_per_volume.values(), width=0.5, align='center')

    plt.legend()
    plt.savefig(f'experiments/lanes_by_volumes/breaking_lanes.png')
    plt.close()


if __name__ == '__main__':
    breaking_lanes()
