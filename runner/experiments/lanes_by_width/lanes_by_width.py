import json
import os
import subprocess
from src.runner import run_experiment, compile_c_code, compile_model
from src.utils import load_config, create_output_dir, copy_results_to_latest, generate_map
from src.constants import Constants
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.math.Density import Density
from src.math.Clustering import Clustering


WIDTHS = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
PEDESTRIAN_DENSITY = 0.3
VOLUMES = 50
GRID_SIZE = 50
CELL_SIZE = GRID_SIZE / VOLUMES


def lanes_by_width():
    print("Running iterations for 300 pedestrians reducing width and plotting lanes by width...\n")
    for width in WIDTHS:
        print(f"Running experiment for width: {width}")
        # run(width)

    # Plot the results
    print("Plotting results...")
    plot_results()

def run(width):
    """
    Run the experiment for a given number of pedestrians.
    """
    config = load_config('experiments/lanes_by_width/config.json')

    # Create output directory with experiment name if provided
    output_dir = create_output_dir(f'experiments/lanes_by_width/results/width_{width}')
    print(f"Created output directory: {output_dir}")

    pedestrians = int(PEDESTRIAN_DENSITY * width * GRID_SIZE)

    config['parameters'][0]['value'] = pedestrians
    config['parameters'][1]['value'] = Constants.PEDESTRIAN_MMOC


    # # Replace the map in the config
    generated_map = generate_map(VOLUMES, width)
    config['parameters'].append({
      "name": "OBSTACLES",
      "type": "map",
      "map": generated_map
    })

    # Add from where to where pedestrians are generated
    config['parameters'].append({
      "name": "FROM_Y",
      "type": "value",
      "value": (GRID_SIZE/ 2) - int(width / 2)
    })
    config['parameters'].append({
      "name": "TO_Y",
      "type": "value",
      "value": (GRID_SIZE/ 2) + int(width / 2)
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
    results_dirs = [d for d in os.listdir('experiments/lanes_by_width/results') if os.path.isdir(os.path.join('experiments/lanes_by_width/results', d))]

    plt.figure(figsize=(10, 10))
    plt.title('Number of groups per width at 15sec')
    plt.xlabel('Width')
    plt.ylabel('Number of groups')

    # Read the results directories
    average_groups_per_width = {
        width: []
        for width in WIDTHS
    }
    std_groups_per_width = {
        width: []
        for width in WIDTHS
    }
    for result_dir in results_dirs:
        width = float(result_dir.split('_')[1])
        for result_file in os.listdir(os.path.join('experiments/lanes_by_width/results', result_dir, 'latest')):
            if result_file.endswith('.csv'):
                df = pd.read_csv(os.path.join('experiments/lanes_by_width/results', result_dir, 'latest', result_file))
                particles = int((len(df.columns) - 1) / 5)
                groups_per_width = []
                groups = Clustering(df, particles).calculate_groups(
                    from_y=(VOLUMES/ 2) - int(width / 2), 
                    to_y=(VOLUMES/ 2) + int(width / 2)
                )

            average_groups_per_width[width].append(groups)

    # Mean the groups per width
    for width in WIDTHS:
        std_groups_per_width[width] = np.std(average_groups_per_width[width])
        average_groups_per_width[width] = np.mean(average_groups_per_width[width])


    # Sort the groups per width_
    average_groups_per_width = dict(sorted(average_groups_per_width.items(), key=lambda item: item[0]))
    std_groups_per_width = dict(sorted(std_groups_per_width.items(), key=lambda item: item[0]))

    n_groups = np.array(list(average_groups_per_width.values()))
    std_n_groups = np.array(list(std_groups_per_width.values()))
    widths = np.array(list(average_groups_per_width.keys()))


    plt.errorbar(widths, n_groups, yerr=std_n_groups, fmt='o', label='Data Points')
    # Fit line using numpy polyfit (degree 1 = linear)
    slope, intercept = np.polyfit(widths, n_groups, 1)
    line = slope * widths + intercept
    plt.plot(widths, line, label='Fitted Line', color='red')

    plt.legend()
    plt.xlabel('Width')
    plt.ylabel('Number of groups')
    plt.title('Scatter Plot with Line of Best Fit')
    plt.grid(True)
    plt.savefig(f'experiments/lanes_by_width/groups_by_width.png')
    plt.close()


if __name__ == '__main__':
    lanes_by_width()
