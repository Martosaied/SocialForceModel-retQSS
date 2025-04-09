import json
import os
import subprocess
from src.runner import run_experiment, compile_c_code, compile_model
from src.utils import load_config, create_output_dir, copy_results_to_latest
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

"""
    This experiment is used to plot the average velocity of the pedestrians in different scenarios.
    We want to see if the average velocity is affected by the formation of lanes.

    We will calculate the avg velocity in 6 different moments of the simulation for different number of volumes.
    Also we want to generate the frame of the simulation for each moment to see the formation of lanes.
"""

GRID_DIVISIONS = [1, 2, 3, 5, 10, 20, 50, 100]

def average_velocity():
    print("Running experiments for different number of volumes...\n")
    for n in GRID_DIVISIONS:
        print(f"Running experiment for {n} volumes...")
        # run(n)
        print(f"Experiment for {n} volumes completed.\n")

    # Plot the results
    print("Plotting results...")
    plot_results()

def run(n):
    """
    Run the experiment for a given number of volumes.
    """
    config = load_config('config.json')

    # Create output directory with experiment name if provided
    output_dir = create_output_dir(
        'experiments/average_velocity/results', 
        f'n_{n}'
    )
    print(f"Created output directory: {output_dir}")

    config['parameters'][0]['value'] = 1000 # N
    config['parameters'][1]['value'] = 1 # PEDESTRIAN_IMPLEMENTATION

    # Save config copy in experiment directory
    config_copy_path = os.path.join(output_dir, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)

    subprocess.run(['sed', '-i', r's/\bGRID_DIVISIONS\s*=\s*[0-9]\+/GRID_DIVISIONS = ' + str(n) + '/', '../retqss/model/social_force_model.mo'])
    subprocess.run(['sed', '-i', r's/\bN\s*=\s*[0-9]\+/N = 1000/', '../retqss/model/social_force_model.mo'])

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
    )

    # Copy results from output directory to latest directory
    copy_results_to_latest(output_dir)

    print(f"\nExperiment completed. Results saved in {output_dir}")


def plot_results():
    """
    Plot the results of the experiments.
    """
    # Get all the results directories
    results_dirs = [d for d in os.listdir('experiments/average_velocity/results') if os.path.isdir(os.path.join('experiments/average_velocity/results', d))]
    
    # Sort the results directories by N
    results_dirs = sorted(results_dirs, key=lambda x: int(x.split('n_')[1]))

    times = [5, 10, 15, 20, 25, 29]
    average_velocities = {
        1: [],
        2: [],
        3: [],
        5: [],
        10: [],
        20: [],
        50: [],
        100: []
    }

    # Get the average velocity from the results
    for results_dir in results_dirs:
        data = pd.read_csv(os.path.join('experiments/average_velocity/results', results_dir, 'latest', 'result_0.csv'))
        volumes = int(results_dir.split('n_')[1])
        for time in times:
            velocities = data[data['time'] == float(time)]

            vx_columns = [col for col in velocities.columns if col.startswith("VX")]
            vy_columns = [col for col in velocities.columns if col.startswith("VY")]
            velocities_x = velocities[vx_columns]
            velocities_y = velocities[vy_columns]
                        
            # Calculate the average velocity
            avg_velocities_x = velocities_x.mean(axis=1).values[0]
            avg_velocities_y = velocities_y.mean(axis=1).values[0]

            # Calculate the length of the velocity vector
            avg_velocities = np.sqrt(avg_velocities_x**2 + avg_velocities_y**2)
            average_velocities[volumes].append(avg_velocities)

    # Generate one plot for each number of volumes
    # Put every histogram in comparison with the one with 1 volume
    # The y axis is the avg velocity and the x axis is the time
    plt.figure(figsize=(10, 5))
    plt.plot(times, average_velocities[1], label=f'1 volume')
    for volumes in GRID_DIVISIONS[1:]: 
        plt.plot(times, average_velocities[volumes], label=f'{volumes} volumes')
    plt.xlabel('Time')
    plt.ylabel('Average velocity')
    plt.title('Average velocity of pedestrians in different scenarios')
    
    plt.legend()
    plt.savefig(f'experiments/average_velocity/results/average_velocity.png')
    plt.close()

if __name__ == '__main__':
    average_velocity()
