import json
import os
import subprocess
from src.runner import run_experiment, compile_c_code, compile_model
from src.utils import load_config, create_output_dir, copy_results_to_latest
import matplotlib.pyplot as plt

GRID_DIVISIONS = [2, 3, 5, 10, 20]
PEDESTRIANS_IMPLEMENTATION = {
    1: "retqss",
}

def performance_n_volumes():
    print("Running experiments for different number of volumes...\n")
    for n in GRID_DIVISIONS:
        for implementation in PEDESTRIANS_IMPLEMENTATION:
            print(f"Running experiment for {n} volumes with implementation {implementation}...")
            # run(n, implementation)
            print(f"Experiment for {n} volumes with implementation {implementation} completed.\n")

    # Plot the results
    print("Plotting results...")
    plot_results()

def run(n, implementation):
    """
    Run the experiment for a given number of volumes.
    """
    config = load_config('config.json')

    # Create output directory with experiment name if provided
    output_dir = create_output_dir(
        'experiments/performance_n_volumes/results', 
        f'n_{n}_implementation_{PEDESTRIANS_IMPLEMENTATION[implementation]}'
    )
    print(f"Created output directory: {output_dir}")

    config['parameters'][0]['value'] = 1000 # N
    config['parameters'][1]['value'] = implementation

    # Save config copy in experiment directory
    config_copy_path = os.path.join(output_dir, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)

    subprocess.run(['sed', '-i', r's/\bN\s*=\s*[0-9]\+/N = 1000/', '../retqss/model/social_force_model.mo'])
    subprocess.run(['sed', '-i', r's/\bGRID_DIVISIONS\s*=\s*[0-9]\+/GRID_DIVISIONS = ' + str(n) + '/', '../retqss/model/social_force_model.mo'])

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
    results_dirs = [d for d in os.listdir('experiments/performance_n_volumes/results') if os.path.isdir(os.path.join('experiments/performance_n_volumes/results', d))]
    mmoc_results_dirs = [d for d in results_dirs if 'mmoc' in d]
    retqss_results_dirs = [d for d in results_dirs if 'retqss' in d]

    # Sort the results directories by N
    retqss_results_dirs = sorted(retqss_results_dirs, key=lambda x: int(x.split('n_')[1].split('_impl')[0]))

    # Separate the results directories by implementation
    results_dirs_by_implementation = {
        "retqss": retqss_results_dirs,
        "mmoc": mmoc_results_dirs
    }

    plt.figure(figsize=(10, 5))
    retqss_times = []
    mmoc_times = []
    for results_dir in retqss_results_dirs:
        benchmark_file = os.path.join('experiments/performance_n_volumes/results', results_dir, 'latest/benchmark.txt')
        with open(benchmark_file, 'r') as f:
            retqss_times.append(float(f.read()))

    for _ in retqss_results_dirs:
        benchmark_file = os.path.join('experiments/performance_n_volumes/results', mmoc_results_dirs[0], 'latest/benchmark.txt')
        with open(benchmark_file, 'r') as f:
            mmoc_times.append(float(f.read()))


    # Sort the times by N
    retqss_times = [x for _, x in sorted(zip(GRID_DIVISIONS, retqss_times))]
    mmoc_times = [x for _, x in sorted(zip(GRID_DIVISIONS, mmoc_times))]
    
    plt.plot(GRID_DIVISIONS, retqss_times, label='retqss')
    plt.plot(GRID_DIVISIONS, mmoc_times, label='mmoc')
    plt.legend(loc='upper left')


    # Plot the results
    plt.xlabel('Number of Volumes')
    plt.ylabel('Time (s)')
    plt.title('Performance of Social Force Model for Different Number of Volumes')
    plt.savefig('experiments/performance_n_volumes/results/performance_n_volumes.png')



if __name__ == '__main__':
    performance_n_volumes()
