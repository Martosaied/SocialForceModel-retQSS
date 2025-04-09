import json
import os
import subprocess
from src.runner import run_experiment, compile_c_code, compile_model
from src.utils import load_config, create_output_dir, copy_results_to_latest
import matplotlib.pyplot as plt

N = [10, 50, 100, 200, 300, 500, 1000, 2000, 3000]
PEDESTRIANS_IMPLEMENTATION = {
    0: "mmoc",
    1: "retqss",
}

def performance_n_pedestrians(GRID_DIVISIONS=1):
    print("Running experiments for different number of pedestrians...\n")
    for n in N:
        for implementation in PEDESTRIANS_IMPLEMENTATION:
            print(f"Running experiment for {n} pedestrians with implementation {implementation}...")
            run(n, implementation, GRID_DIVISIONS)
            print(f"Experiment for {n} pedestrians with implementation {implementation} completed.\n")

    # Plot the results
    print("Plotting results...")
    plot_results()

def run(n, implementation, GRID_DIVISIONS):
    """
    Run the experiment for a given number of pedestrians.
    """
    config = load_config('config.json')

    # Create output directory with experiment name if provided
    output_dir = create_output_dir(
        'experiments/performance_n_pedestrians/results', 
        f'n_{n}_implementation_{PEDESTRIANS_IMPLEMENTATION[implementation]}'
    )
    print(f"Created output directory: {output_dir}")

    config['parameters'][0]['value'] = n
    config['parameters'][1]['value'] = implementation
    # Save config copy in experiment directory
    config_copy_path = os.path.join(output_dir, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)

    subprocess.run(['sed', '-i', r's/\bN\s*=\s*[0-9]\+/N = ' + str(n) + '/', '../retqss/model/social_force_model.mo'])
    subprocess.run(['sed', '-i', r's/\bGRID_DIVISIONS\s*=\s*[0-9]\+/GRID_DIVISIONS = ' + str(GRID_DIVISIONS) + '/', '../retqss/model/social_force_model.mo'])

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
        copy_results=False
    )

    # Copy results from output directory to latest directory
    copy_results_to_latest(output_dir)

    print(f"\nExperiment completed. Results saved in {output_dir}")


def plot_results():
    """
    Plot the results of the experiments.
    """
    # Get all the results directories
    results_dirs = [d for d in os.listdir('experiments/performance_n_pedestrians/results') if os.path.isdir(os.path.join('experiments/performance_n_pedestrians/results', d))]

    # Separate the results directories by implementation
    results_dirs_by_implementation = {
        "mmoc": [],
        "retqss": []
    }
    for results_dir in results_dirs:
        implementation = results_dir.split('_implementation_')[1]
        results_dirs_by_implementation[implementation].append(results_dir)

    plt.figure(figsize=(10, 5))
    for implementation, results_dirs in results_dirs_by_implementation.items():
        times = []
        n_pedestrians = []
        for results_dir in results_dirs:
            benchmark_file = os.path.join('experiments/performance_n_pedestrians/results', results_dir, 'latest/benchmark.txt')
            with open(benchmark_file, 'r') as f:
                times.append(float(f.read()))
            n_pedestrians.append(int(results_dir.split('n_')[1].split('_implementatio')[0]))

        # Sort the times by N
        times = [x for _, x in sorted(zip(n_pedestrians, times))]
        n_pedestrians = sorted(n_pedestrians)

        plt.plot(n_pedestrians, times, label=implementation)
        plt.legend(loc='upper left')


    # Plot the results
    plt.xlabel('Number of Pedestrians')
    plt.ylabel('Time (s)')
    plt.title('Performance of Social Force Model for Different Number of Pedestrians')
    plt.savefig('experiments/performance_n_pedestrians/results/performance_n_pedestrians.png')



if __name__ == '__main__':
    performance_n_pedestrians()
