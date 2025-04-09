import json
import os
import subprocess
from src.runner import run_experiment, compile_c_code, compile_model
from src.utils import load_config, create_output_dir, copy_results_to_latest
from src.constants import Constants
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from src.plotter import calculate_groups


WIDTHS = [16, 14, 12, 10, 8]

def generate_map(width):
    """
    Generate a map with the given width.
    """
    if width < 2:
        raise ValueError("Width must be at least 2 meters.")

    matrix = [[0] * 20 for _ in range(20)]
    
    # Define how many top/bottom rows to set as obstacle
    rows_as_obstacle = int((20 - width) / 2)

    for i in range(rows_as_obstacle):
        matrix[i] = [1] * 20
        matrix[20 - i - 1] = [1] * 20

    return matrix

def lanes_by_width():
    print("Running iterations for 300 pedestrians reducing width and plotting lanes by iteration...\n")
    for width in WIDTHS:
        print(f"Running experiment for width: {width}")
        run(width)

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

    config['parameters'][0]['value'] = 300
    config['parameters'][1]['value'] = Constants.MMOC


    # Replace the map in the config
    generated_map = generate_map(width)
    config['parameters'].append({
      "name": "OBSTACLES",
      "type": "map",
      "map": generated_map
    })

    # Add from where to where pedestrians are generated
    config['parameters'].append({
      "name": "FROM_Y",
      "type": "value",
      "value": 10 - int(width / 2)
    })
    config['parameters'].append({
      "name": "TO_Y",
      "type": "value",
      "value": 10 + int(width / 2)
    })

    # Save config copy in experiment directory
    config_copy_path = os.path.join(output_dir, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Replace the grid divisions in the model
    subprocess.run(['sed', '-i', r's/\bGRID_DIVISIONS\s*=\s*[0-9]\+/GRID_DIVISIONS = ' + str(20) + '/', '../retqss/model/social_force_model.mo'])

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
    groups_per_width = {}
    for result_dir in results_dirs:
        df = pd.read_csv(os.path.join('experiments/lanes_by_width/results', result_dir, 'latest', 'result_0.csv'))
        particles = (len(df.columns) - 1) / 5
        groups = calculate_groups(df.iloc[150], int(particles))
        groups_per_width[int(result_dir.split('_')[1])] = len(groups)

    # Sort the groups per width_
    groups_per_width = dict(sorted(groups_per_width.items(), key=lambda item: item[0]))
    n_groups = np.array(list(groups_per_width.values()))
    widths = np.array(list(groups_per_width.keys()))

    plt.scatter(widths, n_groups, label='Data Points')
    # Fit line using numpy polyfit (degree 1 = linear)
    slope, intercept = np.polyfit(widths, n_groups, 1)
    line = slope * widths + intercept
    plt.plot(widths, line, label='Fitted Line')

    plt.legend()
    plt.xlabel('Width')
    plt.ylabel('Number of groups')
    plt.title('Scatter Plot with Line of Best Fit')
    plt.grid(True)
    plt.savefig(f'experiments/lanes_by_width/groups_by_width.png')
    plt.close()


if __name__ == '__main__':
    lanes_by_iterations()
