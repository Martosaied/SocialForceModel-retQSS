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

B = [0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.4,1.5,1.6,1.7,1.8,2.0]
WIDTH = 15
PEDESTRIAN_DENSITY = 0.3
VOLUMES = 50
GRID_SIZE = 50
CELL_SIZE = GRID_SIZE / VOLUMES


def lanes_by_B():
    print("Running iterations for pedestrians with fixed width and plotting lanes by B...\n")
    for b in B:
        print(f"Running experiment for B: {b}")
        # run(b)

    # Plot the results
    print("Plotting results...")
    plot_results()

def run(b):
    """
    Run the experiment for a given number of pedestrians.
    """
    config = load_config('experiments/lanes_by_B/config.json')

    # Create output directory with experiment name if provided
    output_dir = create_output_dir(f'experiments/lanes_by_B/results/B_{b}')
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
      "name": "PEDESTRIAN_B_2",
      "type": "value",
      "value": b
    })
    config['parameters'].append({
      "name": "PEDESTRIAN_B_1",
      "type": "value",
      "value": b
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
    results_dirs = [d for d in os.listdir('experiments/lanes_by_B/results') if os.path.isdir(os.path.join('experiments/lanes_by_B/results', d))]

    plt.figure(figsize=(10, 10))
    plt.title('Average number of groups per B')
    plt.xlabel('B')
    plt.ylabel('Number of groups')

    # Read the results directories
    average_groups_per_B = {
        b: []
        for b in B
    }
    std_groups_per_B = {
        b: []
        for b in B
    }
    for result_dir in results_dirs:
        b = float(result_dir.split('_')[1])
        for result_file in os.listdir(os.path.join('experiments/lanes_by_B/results', result_dir, 'latest')):
            if result_file.endswith('.csv'):
                df = pd.read_csv(os.path.join('experiments/lanes_by_B/results', result_dir, 'latest', result_file))
                particles = (len(df.columns) - 1) / 5
                groups_per_B = []
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
                    groups_per_B.append(len(groups))

                average_groups_per_B[b].append(np.mean(groups_per_B))

    # Mean the groups per width
    for b in B:
        std_groups_per_B[b] = np.std(average_groups_per_B[b])
        average_groups_per_B[b] = np.mean(average_groups_per_B[b])

    # Sort the groups per width_
    average_groups_per_B = dict(sorted(average_groups_per_B.items(), key=lambda item: item[0]))
    std_groups_per_B = dict(sorted(std_groups_per_B.items(), key=lambda item: item[0]))

    n_groups = np.array(list(average_groups_per_B.values()))
    std_n_groups = np.array(list(std_groups_per_B.values()))
    Bs = np.array(list(average_groups_per_B.keys()))


    plt.errorbar(Bs, n_groups, yerr=std_n_groups, fmt='o', label='Data Points')
    # # Fit line using numpy polyfit (degree 1 = linear)
    # slope, intercept = np.polyfit(Bs, n_groups, 1)
    # line = slope * Bs + intercept
    # plt.plot(Bs, line, label='Fitted Line', color='red')

    plt.legend()
    plt.xlabel('B')
    plt.ylabel('Number of groups')
    plt.title('Scatter Plot with Line of Best Fit')
    plt.grid(True)
    plt.savefig(f'experiments/lanes_by_B/groups_by_B.png')
    plt.close()


if __name__ == '__main__':
    lanes_by_B()
