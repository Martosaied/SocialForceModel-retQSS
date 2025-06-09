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

Rs = [0.01, 0.03, 0.05, 0.07, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5]
Bs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0]
As = [2.1, 2.5, 3, 4]

WIDTH = 10
PEDESTRIAN_DENSITY = 0.3
VOLUMES = 50
GRID_SIZE = 50
CELL_SIZE = GRID_SIZE / VOLUMES


def lanes_heatmap():
    print("Running iterations for pedestrians with fixed width and plotting lanes by R, B, A...\n")
    for r in Rs:
        for b in Bs:
            for a in As:
                print(f"Running experiment for R: {r}, B: {b}, A: {a}")
                # run(r, b, a)

    # Plot the results
    print("Plotting results...")
    plot_results()

def run(r, b, a):
    """
    Run the experiment for a given number of pedestrians.
    """
    config = load_config('experiments/lanes_heatmap/config.json')

    # Create output directory with experiment name if provided
    output_dir = create_output_dir(f'experiments/lanes_heatmap/results/R_{r}_B_{b}_A_{a}')
    print(f"Created output directory: {output_dir}")

    pedestrians = int(PEDESTRIAN_DENSITY * WIDTH * VOLUMES)

    config['parameters'][0]['value'] = pedestrians
    config['parameters'][1]['value'] = Constants.MMOC


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
    config['parameters'].append({
      "name": "PEDESTRIAN_R",
      "type": "value",
      "value": r
    })
    config['parameters'].append({
      "name": "PEDESTRIAN_B_2",
      "type": "value",
      "value": b
    })
    config['parameters'].append({
      "name": "PEDESTRIAN_A_2",
      "type": "value",
      "value": a
    })

    # Save config copy in experiment directory
    config_copy_path = os.path.join(output_dir, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Replace the grid divisions in the model
    subprocess.run(['sed', '-i', r's/\bGRID_DIVISIONS\s*=\s*[0-9]\+/GRID_DIVISIONS = ' + str(VOLUMES) + '/', '../retqss/model/social_force_model.mo'])
    # Replace the pedestrians in the model
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
    results_dirs = [d for d in os.listdir('experiments/lanes_heatmap/results') if os.path.isdir(os.path.join('experiments/lanes_heatmap/results', d))]

    # Plot the results in 4 heatmaps where X is R, Y is B and A is fixed
    grouped_results = {}
    for a in As:
        # Get the results for the current A
        results = [d for d in results_dirs if f'A_{a}' in d]
        for b in Bs:
            for r in Rs:
                for result_file in os.listdir(os.path.join('experiments/lanes_heatmap/results', f'R_{r}_B_{b}_A_{a}', 'latest')):
                    if result_file.endswith('.csv'):
                        df = pd.read_csv(os.path.join('experiments/lanes_heatmap/results', f'R_{r}_B_{b}_A_{a}', 'latest', result_file))
                        particles = (len(df.columns) - 1) / 5
                        groups_per_A = []
                        for index, row in df.iterrows():
                            if index < 100 and index % 5 != 0:
                                continue
                            groups = Clustering(
                                row, 
                                int(particles), 
                            ).calculate_groups(
                                from_y=(VOLUMES/ 2) - int(WIDTH / 2), 
                                to_y=(VOLUMES/ 2) + int(WIDTH / 2)
                            )
                            if f'R_{r}_B_{b}_A_{a}' not in grouped_results:
                                grouped_results[f'R_{r}_B_{b}_A_{a}'] = []
                            grouped_results[f'R_{r}_B_{b}_A_{a}'].append(len(groups))

                        # Mean and std of the groups
                        mean_groups = np.mean(grouped_results[f'R_{r}_B_{b}_A_{a}'])
                        std_groups = np.std(grouped_results[f'R_{r}_B_{b}_A_{a}'])
                        print(f'Average groups for R:{r}, B:{b}, A:{a}: {mean_groups}, Std groups: {std_groups}')



            

if __name__ == '__main__':
    lanes_heatmap()
