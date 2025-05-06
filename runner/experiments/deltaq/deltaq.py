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


DELTAQ = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]


def deltaq():
    print("Running iterations for 300 pedestrians reducing Tolerance and plotting lanes...\n")
    for deltaq in DELTAQ:
        print(f"Running experiment for deltaq: {deltaq}")
        run(deltaq)

    # Plot the results
    print("Plotting results...")
    plot_results()

def run(deltaq):
    """
    Run the experiment for a given number of pedestrians.
    """
    config = load_config('experiments/deltaq/config.json')

    # Create output directory with experiment name if provided
    output_dir = create_output_dir(f'experiments/deltaq/results/deltaq_{deltaq}')
    print(f"Created output directory: {output_dir}")

    config['parameters'][0]['value'] = 300
    config['parameters'][1]['value'] = Constants.MMOC

    # Save config copy in experiment directory
    config_copy_path = os.path.join(output_dir, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Replace the grid divisions in the model
    subprocess.run(['sed', '-i', r's/\bGRID_DIVISIONS\s*=\s*[0-9]\+/GRID_DIVISIONS = ' + str(10) + '/', '../retqss/model/social_force_model.mo'])
    subprocess.run(['sed', '-i', r's/\Tolerance={1e-[0-9]\+}\+/Tolerance={1e-' + str(deltaq) + '}/', '../retqss/model/social_force_model.mo'])
    subprocess.run(['sed', '-i', r's/\AbsTolerance={1e-[0-9]\+}\+/AbsTolerance={1e-' + str(deltaq+3) + '}/', '../retqss/model/social_force_model.mo'])

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
    results_dirs = [d for d in os.listdir('experiments/deltaq/results') if os.path.isdir(os.path.join('experiments/deltaq/results', d))]

    # Read the results directories
    groups_per_deltaq = {
        deltaq: []
        for deltaq in DELTAQ
    }
    performance_per_deltaq = {
        deltaq: []
        for deltaq in DELTAQ
    }
    for result_dir in results_dirs:
        deltaq = int(result_dir.split('_')[1])
        for result_file in os.listdir(os.path.join('experiments/deltaq/results', result_dir, 'latest')):
            if result_file.endswith('.txt'):
                with open(os.path.join('experiments/deltaq/results', result_dir, 'latest', result_file), 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        performance_per_deltaq[deltaq].append(float(line))
                    

            if result_file.endswith('.csv'):
                df = pd.read_csv(os.path.join('experiments/deltaq/results', result_dir, 'latest', result_file))
                particles = (len(df.columns) - 1) / 5
                groups = calculate_groups(df.iloc[150], int(particles))
                groups_per_deltaq[deltaq].append(len(groups))

    # Mean the groups per width
    for deltaq in DELTAQ:
        groups_per_deltaq[deltaq] = np.mean(groups_per_deltaq[deltaq])
    
    for deltaq in DELTAQ:
        performance_per_deltaq[deltaq] = np.mean(performance_per_deltaq[deltaq])

    # Sort the groups per width_
    groups_per_deltaq = dict(sorted(groups_per_deltaq.items(), key=lambda item: item[0]))
    n_groups = np.array(list(groups_per_deltaq.values()))
    performance_per_deltaq = dict(sorted(performance_per_deltaq.items(), key=lambda item: item[0]))
    performance = np.array(list(performance_per_deltaq.values()))
    deltas = np.array(list(map(str, groups_per_deltaq.keys())))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
    ax1.set_title('Performance per deltaq at 15sec')
    ax1.set_xlabel('DeltaQ')
    ax1.set_ylabel('Performance')

    ax2.set_title('Number of groups per deltaq at 15sec')
    ax2.set_xlabel('DeltaQ')
    ax2.set_ylabel('Number of groups')

    # X axis is the deltaq, historigram with only the deltaq values
    ax1.bar(deltas, performance, label='Performance(ms)')
    ax2.bar(deltas, n_groups, label='Number of groups')

    plt.legend()
    plt.savefig(f'experiments/deltaq/performance_by_deltaq.png')
    plt.close()


if __name__ == '__main__':
    lanes_by_iterations()
