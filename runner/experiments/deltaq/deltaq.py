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


DELTAQ = [12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
PEDESTRIAN_COUNT = int(20 * 50 * 0.3)
WIDTH = 20
VOLUMES = 50


def deltaq():
    print("Running iterations for 300 pedestrians reducing Tolerance and plotting lanes...\n")
    for deltaq in DELTAQ:
        print(f"Running experiment for deltaq: {deltaq}")
        # run(deltaq)

    # Plot the results
    print("Plotting results...")
    # plot_results()
    plot_iterations_by_deltaq()

def run(deltaq):
    """
    Run the experiment for a given number of pedestrians.
    """
    config = load_config('experiments/deltaq/config.json')

    # Create output directory with experiment name if provided
    output_dir = create_output_dir(f'experiments/deltaq/results/deltaq_{deltaq}')
    print(f"Created output directory: {output_dir}")

    config['parameters'][0]['value'] = PEDESTRIAN_COUNT
    config['parameters'][1]['value'] = Constants.MMOC

    generated_map = generate_map(VOLUMES, WIDTH)
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

    # Replace the grid divisions in the model
    subprocess.run(['sed', '-i', r's/\bN\s*=\s*[0-9]\+/N = ' + str(PEDESTRIAN_COUNT) + '/', '../retqss/model/social_force_model.mo'])
    subprocess.run(['sed', '-i', r's/\bGRID_DIVISIONS\s*=\s*[0-9]\+/GRID_DIVISIONS = ' + str(VOLUMES) + '/', '../retqss/model/social_force_model.mo'])
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

def plot_iterations_by_deltaq():
    # Get all the results directories
    fig, ax = plt.subplots(figsize=(20, 10))
    ax.set_title('Number of groups per time(averaged)')
    ax.set_xlabel('Time')
    ax.set_ylabel('Number of groups')

    for deltaq in DELTAQ:

        # Get the results files
        results_files = [f for f in os.listdir(os.path.join('experiments/deltaq/results', f'deltaq_{deltaq}', 'latest')) if f.endswith('.csv')]

        # Read the results files
        groups_per_time = {}
        groups_per_time_averaged = {}
        for result_file in results_files:
            df = pd.read_csv(os.path.join('experiments/deltaq/results', f'deltaq_{deltaq}', 'latest', result_file))
            particles = (len(df.columns) - 1) / 5
            for index, row in df.iterrows():
                groups = Clustering(row, int(particles)).calculate_groups()
                groups_per_time[row['time']] = len(groups)

                if row['time'] not in groups_per_time_averaged:
                    groups_per_time_averaged[row['time']] = [len(groups)]
                else:
                    groups_per_time_averaged[row['time']].append(len(groups))
        
        mean_groups_per_time = {k: np.mean(v) for k, v in groups_per_time_averaged.items()}
        std_groups_per_time = {k: np.std(v) for k, v in groups_per_time_averaged.items()}
        ax.plot(mean_groups_per_time.keys(), mean_groups_per_time.values(), label=f'deltaq: {deltaq}', color=f'C{deltaq}')
        ax.fill_between(
            list(mean_groups_per_time.keys()), 
            (np.array(list(mean_groups_per_time.values())) - np.array(list(std_groups_per_time.values()))), 
            (np.array(list(mean_groups_per_time.values())) + np.array(list(std_groups_per_time.values()))), 
            alpha=0.2
        )

    fig.savefig(f'experiments/deltaq/iterations_by_deltaq.png')
    plt.close()


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
