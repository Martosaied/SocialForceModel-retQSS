import distutils.dir_util
import subprocess
import shutil
import random
import time
import os
import multiprocessing
import string
import pandas as pd
import numpy as np

from src.plotter import Plotter
from src.math.Density import Density
from src.math.Clustering import Clustering
from src.utils import process_parameters, get_parameter_combinations
from src.config_manager import get_config


def run_model(model_name: str, directory: str, parameters: dict):
    """Run the specified model command and handle the solution file."""
    cmd = f"{directory}/{model_name}.sh"
    cmd_dir = os.path.dirname(cmd)
    solution_path = os.path.join(cmd_dir, "solution.csv")

    # Check if the command exists and is executable
    if not os.path.isfile(cmd) or not os.access(cmd, os.X_OK):
        raise FileNotFoundError(f"Model command not found or not executable: {cmd}")

    try:
        config = get_config()
        # Run the command
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            text=True,
            capture_output=not config.verbose
        )

        metrics = {
            'time': 0,
            'memory_usage': 0,
            'density_based_groups': 0,
            'clustering_based_groups': 0,
        }
        if not config.verbose:
            # Get the time from the output
            metrics['time'] = result.stderr.split('User time (seconds): ')[1].split('\n')[0].strip()
            metrics['memory_usage'] = result.stderr.split('Maximum resident set size (kbytes): ')[1].split('\n')[0].strip()

            # Use config properties to determine what to calculate
            if config.should_calculate_metrics:
                df = pd.read_csv(solution_path)
                particles = parameters.get('N', 300)
                
                # Calculate density-based groups if not skipped (disabled for now, no need to calculate)
                # metrics['density_based_groups'] = Density(df, particles).calculate_groups()
                metrics['density_based_groups'] = 0
                
                # Calculate clustering-based groups if not skipped
                metrics['clustering_based_groups'] = Clustering(df, particles).calculate_groups()
            else:
                # Skip all calculations
                print("Skipping density and clustering calculations")

        # Check if solution.csv was created
        if not os.path.exists(solution_path):
            raise FileNotFoundError(f"Solution file not found at: {solution_path}")

        return solution_path, metrics

    except subprocess.CalledProcessError as e:
        print(f"Error running {model_name} model: {e}")
        print(f"Error output: {e.stderr}")
        raise


def setup_parameters(model_name: str, parameters: dict, iteration: int, directory: str):
    """Setup parameters for the model."""

    random.seed(iteration)
    seed = random.randint(0, 1000000)
    # Create a parameters.config file
    f = open(f"{directory}/parameters.config", "w")
    for param, value in parameters.items():
        f.write(f"{param}={value}\n")

    f.write(f"RANDOM_SEED={seed}\n")
    f.close()

def change_sh_file(model_name: str, random_str: str):
    """Change the sh file to run the model with the specified parameters."""
    # Using sed to change the sh file change /build/social_force_model/ for /build/tmp_<random_string>/<model_name>/
    # Get the directory of the sh file
    sh_file = f"../retqss/build/{model_name}_{random_str}/{model_name}.sh"
    subprocess.run(['sed', '-i', r's/\/build\/' + model_name + '\//\/build\/' + model_name + '_' + random_str + '\//', sh_file])

    # Add time to the sh file
    subprocess.run(['sed', '-i', r's/\.\/' + model_name + '/\/usr\/bin\/time -v \.\/' + model_name + '/', sh_file])


def run_parallel_model(
    model_name: str, 
    parameters: dict, 
    iteration: int, 
    directory: str, 
    metrics_file: str, 
    output_dir: str, 
    plot: bool, 
    copy_results: bool,
    results: list):
    try:
        print("Setting up parameters...")
        setup_parameters(model_name, parameters, iteration, directory)

        # Run the model and get path to solution file
        solution_path, metrics = run_model(model_name, directory, parameters)

        # Write metrics to file
        metrics_file.write(f"{metrics['time']},{metrics['memory_usage']},{metrics['density_based_groups']},{metrics['clustering_based_groups']}\n")

        print(f"Flushing metrics file")
        metrics_file.flush()

        # Define the destination path for this iteration
        result_file = os.path.join(output_dir, f'result_{iteration}.csv')

        # Move and rename the solution file
        if copy_results:
            shutil.move(solution_path, result_file)
            print(f"Saved results for iteration {iteration} to {result_file}")

        # Remove the tmp directory 
        print(f"Removing {directory}")
        shutil.rmtree(directory)

    except Exception as e:
        print(f"Error in iteration {iteration}: {str(e)}")
        # Create error log file
        error_file = os.path.join(output_dir, f'error_iteration_{iteration}.txt')
        with open(error_file, 'w') as f:
            f.write(f"Error during iteration {iteration}:\n{str(e)}")
        raise



def run_iterations(num_iterations: int, model_name: str, output_dir: str = "output", parameters: dict = {}, plot: bool = True, copy_results: bool = True, max_concurrent_processes: int = None):
    """Run experiment iterations using the specified model."""
    
    # Set default max concurrent processes to CPU count if not specified
    if max_concurrent_processes is None:
        max_concurrent_processes = multiprocessing.cpu_count()
    
    print(f"Running {num_iterations} iterations with max {max_concurrent_processes} concurrent processes")

    metrics_file = os.path.join(output_dir, f'metrics.csv')

    metrics_file = open(metrics_file, 'w')
    metrics_file.write('time,memory_usage,density_based_groups,clustering_based_groups\n')
    metrics_file.flush()
    results = []
    processes = []
    
    for iteration in range(num_iterations):
        print(f"\nStarting iteration {iteration + 1}/{num_iterations}")

        # Wait if we've reached the maximum number of concurrent processes
        while len([p for p in processes if p.is_alive()]) >= max_concurrent_processes:
            time.sleep(0.1)  # Small delay to avoid busy waiting
            # Clean up completed processes
            processes = [p for p in processes if p.is_alive()]

        random_str = ''.join(random.choices(string.ascii_uppercase + string.digits, k=3))
        tmp_dir = f"../retqss/build/{model_name}_{random_str}"
        print(f"Copying from ../retqss/build/{model_name} to {tmp_dir}")
        compile_c_code()
        compile_model(model_name)
        try:
            distutils.dir_util.copy_tree(f"../retqss/build/{model_name}", tmp_dir)
            subprocess.run( f"cp ../retqss/build/{model_name}/{model_name} {tmp_dir}", shell=True, check=True, capture_output=False)
            change_sh_file(model_name, random_str)

            # Run the model in parallel
            p = multiprocessing.Process(target=run_parallel_model, args=(model_name, parameters, iteration, tmp_dir, metrics_file, output_dir, plot, copy_results, results))
            p.start()
            processes.append(p)

        except Exception as e:
            print(f"Error copying ../retqss/build/{model_name} to {tmp_dir}: {e}")

    # Wait for all remaining processes to complete
    for p in processes:
        p.join()
        print(f"Process {p.name} completed")

    metrics_file.close()


def run_experiment(config: dict, output_dir: str, model_name: str, plot: bool = True, copy_results: bool = True):
    """Run experiment iterations using the specified model."""
    num_iterations = config.get('iterations', 1)
    max_concurrent_processes = config.get('max_concurrent_processes', 10)
    print(f"Running {num_iterations} iterations for {model_name}...")

    parameters = process_parameters(config.get('parameters', {}))

    grouped_parameters = get_parameter_combinations(parameters)
    for params in grouped_parameters:
        print(f"Running with parameters: {params}")
        run_iterations(num_iterations, model_name, output_dir, params, plot, copy_results, max_concurrent_processes)


def compile_c_code():
    """Compile the C++ code for the specified model."""
    cmd = f"cd ../retqss/src && make"
    subprocess.run(cmd, shell=True, check=True, capture_output=True)


def compile_model(model_name: str):
    """Compile the model for the specified model."""
    cmd = f"cd ../retqss/model/scripts && ./build.sh {model_name}"
    subprocess.run(cmd, shell=True, check=True, capture_output=True)
