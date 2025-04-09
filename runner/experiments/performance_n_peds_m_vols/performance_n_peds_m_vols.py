import json
import os
import subprocess
from src.runner import run_experiment, compile_c_code, compile_model
from src.utils import load_config, create_output_dir, copy_results_to_latest
import matplotlib.pyplot as plt

VOLUMES = [1, 2, 3, 5, 10, 20]
N = [10, 50, 100, 200, 300, 500, 1000, 2000, 3000]

PEDESTRIANS_IMPLEMENTATION = {
    1: "retqss",
    0: "mmoc",
}

def performance_n_peds_m_vols():
    print("Running experiment for MMOC...")
    for n in N:
        # run(n, 1, 0)
        print(f"Experiment for {n} pedestrians and 1 volume with implementation MMOC completed.\n")
    print("Experiment for MMOC completed.\n")

    print("Running experiments for different number of volumes...\n")

    print("Running experiments for different number of pedestrians...\n")

    for n in N:
        for volumes in VOLUMES:
            print(f"Running experiment for {n} pedestrians and {volumes} volumes with implementation retqss...")
            # run(n, volumes, 1)
            print(f"Experiment for {n} pedestrians and {volumes} volumes with implementation retqss completed.\n")

    # Plot the results
    print("Plotting results...")
    plot_results()

def run(n, volumes,implementation):
    """
    Run the experiment for a given number of volumes.
    """
    config = load_config('config.json')

    # Create output directory with experiment name if provided
    output_dir = create_output_dir(
        'experiments/performance_n_peds_m_vols/results', 
        f'n_{n}_volumes_{volumes}_{PEDESTRIANS_IMPLEMENTATION[implementation]}'
    )
    print(f"Created output directory: {output_dir}")

    config['parameters'][0]['value'] = n # N
    config['parameters'][1]['value'] = implementation

    # Save config copy in experiment directory
    config_copy_path = os.path.join(output_dir, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)

    subprocess.run(['sed', '-i', r's/\bGRID_DIVISIONS\s*=\s*[0-9]\+/GRID_DIVISIONS = ' + str(volumes) + '/', '../retqss/model/social_force_model.mo'])
    subprocess.run(['sed', '-i', r's/\bN\s*=\s*[0-9]\+/N = ' + str(n) + '/', '../retqss/model/social_force_model.mo'])

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
    results_dirs = [d for d in os.listdir('experiments/performance_n_peds_m_vols/results') if os.path.isdir(os.path.join('experiments/performance_n_peds_m_vols/results', d))]
    mmoc_results_dirs = [d for d in results_dirs if 'mmoc' in d]
    retqss_results_dirs = [d for d in results_dirs if 'retqss' in d]

    # Sort the results directories by N
    retqss_results_dirs = sorted(retqss_results_dirs, key=lambda x: int(x.split('n_')[1].split('_volumes_')[0]))
    mmoc_results_dirs = sorted(mmoc_results_dirs, key=lambda x: int(x.split('n_')[1].split('_volumes_')[0]))

    # Separate the results directories by implementation
    results_dirs_by_implementation = {
        "retqss": retqss_results_dirs,
        "mmoc": mmoc_results_dirs
    }

    # Generate a plot for each amount of volumes 
    plt.figure(figsize=(10, 5))
    for volumes in VOLUMES:
        retqss_times = []
        for results_dir in results_dirs_by_implementation['retqss']:
            if int(results_dir.split('volumes_')[1].split('_')[0]) == volumes:
                benchmark_file = os.path.join('experiments/performance_n_peds_m_vols/results', results_dir, 'latest/benchmark.txt')
                with open(benchmark_file, 'r') as f:
                    retqss_times.append(float(f.read()))

        retqss_times = [x for _, x in sorted(zip(N, retqss_times))]
        plt.plot(N, retqss_times, label=f'{volumes} volumes')


    mmoc_times = []
    for results_dir in results_dirs_by_implementation['mmoc']:
        benchmark_file = os.path.join('experiments/performance_n_peds_m_vols/results', results_dir, 'latest/benchmark.txt')
        with open(benchmark_file, 'r') as f:
            mmoc_times.append(float(f.read()))
        
    mmoc_times = [x for _, x in sorted(zip(N, mmoc_times))]
    plt.plot(N, mmoc_times, label='mmoc')
    plt.legend(loc='upper left')
    plt.xlabel('Number of Pedestrians')
    plt.ylabel('Time (s)')
    plt.title(f'Performance of Social Force Model by number of volumes')
    plt.savefig(f'experiments/performance_n_peds_m_vols/results/performance_n_peds_m_vols.png')

if __name__ == '__main__':
    performance_n_peds_m_vols()
