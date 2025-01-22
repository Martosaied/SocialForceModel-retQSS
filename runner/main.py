import time
import argparse
import json
import os
import itertools
from datetime import datetime
import pandas as pd
from pathlib import Path
import subprocess
import shutil
import random
import multiprocessing

from plotter import generate_gif
from utils import load_config, create_output_dir, process_parameters, get_parameter_combinations

def run_model(model_name: str):
    """Run the specified model command and handle the solution file."""
    cmd = f"../retqss/build/{model_name}/{model_name}.sh"
    cmd_dir = os.path.dirname(cmd)
    solution_path = os.path.join(cmd_dir, "solution.csv")
    
    # Check if the command exists and is executable
    if not os.path.isfile(cmd) or not os.access(cmd, os.X_OK):
        raise FileNotFoundError(f"Model command not found or not executable: {cmd}")
    
    try:
        # Run the command
        result = subprocess.run(
            cmd,
            shell=True,
            check=True,
            text=True,
            # capture_output=True
        )
        
        # Check if solution.csv was created
        if not os.path.exists(solution_path):
            raise FileNotFoundError(f"Solution file not found at: {solution_path}")
        
        return solution_path
        
    except subprocess.CalledProcessError as e:
        print(f"Error running {model_name} model: {e}")
        print(f"Error output: {e.stderr}")
        raise

def setup_parameters(model_name: str, parameters: dict, iteration: int):
    """Setup parameters for the model."""

    random.seed(iteration)
    seed = random.randint(0, 1000000)

    # Create a parameters.config file
    f = open(f"../retqss/build/{model_name}/parameters.config", "w")
    for param, value in parameters.items():
        f.write(f"{param}={value}\n")

    f.write(f"RANDOM_SEED={seed}\n")
    f.close()

def run_iterations(num_iterations: int, model_name: str, output_dir: str, parameters: dict):
    """Run experiment iterations using the specified model."""

    time_file = os.path.join(output_dir, f'benchmark.txt')
    time_file = open(time_file, 'w')

    for iteration in range(num_iterations):
        print(f"\nStarting iteration {iteration + 1}/{num_iterations}")

        try:
            print("Setting up parameters...")
            setup_parameters(model_name, parameters, iteration)

            # Measure time
            start_time = time.time()

            # Run the model and get path to solution file
            solution_path = run_model(model_name)

            # Measure time
            end_time = time.time()
            time_file.write(f"{end_time - start_time}\n")

            # Define the destination path for this iteration
            result_file = os.path.join(output_dir, f'result_{iteration}.csv')
            
            # Move and rename the solution file
            shutil.move(solution_path, result_file)
            print(f"Saved results for iteration {iteration} to {result_file}")

            # Generate GIF
            if iteration == 0:
                generate_gif(result_file, output_dir)
                print(f"Generated GIF for first iteration only")
            
        except Exception as e:
            print(f"Error in iteration {iteration}: {str(e)}")
            # Create error log file
            error_file = os.path.join(output_dir, f'error_iteration_{iteration}.txt')
            with open(error_file, 'w') as f:
                f.write(f"Error during iteration {iteration}:\n{str(e)}")
            raise

    time_file.close()


def run_experiment(config: dict, output_dir: str, model_name: str):
    """Run experiment iterations using the specified model."""
    num_iterations = config.get('iterations', 1)
    print(f"Running {num_iterations} iterations for {model_name}...")

    parameters = process_parameters(config.get('parameters', []))

    grouped_parameters = get_parameter_combinations(parameters)
    for params in grouped_parameters:
        print(f"Running with parameters: {params}")
        run_iterations(num_iterations, model_name, output_dir, params)

def compile_c_code():
    """Compile the C++ code for the specified model."""
    cmd = f"cd ../retqss/src && make"
    subprocess.run(cmd, shell=True, check=True)

def compile_model(model_name: str):
    """Compile the model for the specified model."""
    cmd = f"cd ../retqss/model/scripts && ./build.sh {model_name}"
    subprocess.run(cmd, shell=True, check=True)

def main():
    parser = argparse.ArgumentParser(description='Run experiments with JSON configuration')
    parser.add_argument('config', type=str, help='Path to JSON configuration file')
    parser.add_argument('model', type=str, help='Name of the model to run (e.g., social_force_model)')
    parser.add_argument('--experiment-name', type=str, help='Name of the experiment (creates a subdirectory)')
    parser.add_argument('--compile', action='store_true', help='Compile the model before running')
    parser.add_argument('--compile-c', action='store_true', help='Compile the C++ code before running')
    parser.add_argument('--output-dir', type=str, default='experiments',
                       help='Base directory for experiment outputs (default: experiments)')
    
    args = parser.parse_args()
    
    # Validate config file exists
    if not os.path.exists(args.config):
        raise FileNotFoundError(f"Configuration file not found: {args.config}")
    
    # Load configuration
    config = load_config(args.config)
    
    # Create output directory with experiment name if provided
    output_dir = create_output_dir(args.output_dir, args.experiment_name)
    print(f"Created output directory: {output_dir}")
    
    # Save config copy in experiment directory
    config_copy_path = os.path.join(output_dir, 'config.json')
    with open(config_copy_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Compile the C++ code if requested
    if args.compile_c:
        compile_c_code()

    # Compile the model if requested
    if args.compile:
        compile_model(args.model)
    
    # Run experiment
    run_experiment(config, output_dir, args.model)
    print(f"\nExperiment completed. Results saved in {output_dir}")

if __name__ == '__main__':
    main()
