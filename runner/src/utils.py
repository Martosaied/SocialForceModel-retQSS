import argparse
import json
import os
from datetime import datetime
import pandas as pd
from pathlib import Path
import subprocess
import shutil
import numpy as np
import itertools
from typing import Dict, List, Any, Iterator
import psutil
import time

def elapsed_since(start: float) -> float:
    return time.time() - start

def load_config(config_path: str) -> dict:
    """Load experiment configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def copy_results_to_latest(output_dir: str) -> str:
    """Copy results from output directory to latest directory."""
    latest_dir = os.path.join(output_dir, '../latest')
    os.makedirs(latest_dir, exist_ok=True)
    shutil.copytree(output_dir, latest_dir, dirs_exist_ok=True)
    return latest_dir

def create_output_dir(base_dir: str = "results", experiment_name: str = None) -> str:
    """Create and return path to timestamped directory within experiment folder.
    
    Args:
        base_dir: Base directory for all experiments
        experiment_name: Name of the experiment (optional)
    
    Returns:
        Path to the created directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if experiment_name:
        # Create experiment directory if it doesn't exist
        experiment_dir = os.path.join(base_dir, experiment_name)
        os.makedirs(experiment_dir, exist_ok=True)
        # Create timestamped directory within experiment directory
        output_dir = os.path.join(experiment_dir, f"run_{timestamp}")
    else:
        # If no experiment name, use old format
        output_dir = os.path.join(base_dir, f"experiment_{timestamp}")
    
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def process_map(mapping: List[List[int]]) -> List[int]:
    obstacle_indices = []
    for i in range(len(mapping)):
        for j in range(len(mapping[i])):
            if mapping[i][j] == 1:
                obstacle_indices.append(i % len(mapping) + len(mapping[i]) * j + 2)
    
    return obstacle_indices

def process_walls(walls: List[List[Dict[str, int]]]) -> List[str]:
    processed_walls = []
    for wall in walls:
        processed_walls.append(f'{wall["from_x"]}/{wall["from_y"]}/{wall["to_x"]}/{wall["to_y"]}')

    return [f'{",".join(processed_walls)}']

def parse_walls(walls: str) -> List[Dict[str, int]]:
    if walls is '' or walls is None or walls == []:
        return []

    walls = walls.split(',')
    processed_walls = []
    for wall in walls:
        coordinates = wall.split('/')
        processed_walls.append({
            'from_x': float(coordinates[0]),
            'from_y': float(coordinates[1]),
            'to_x': float(coordinates[2]),
            'to_y': float(coordinates[3])
        })
    return processed_walls

def custom_process_by_name(name: str, value: Any) -> Any:
    if name == 'WALLS':
        return process_walls(value)
    else:
        raise ValueError(f"Invalid parameter name: {name}")

def process_parameter(param: dict) -> dict:
    """Process parameter configuration and return a dictionary."""

    if param['type'] == 'range':
        return np.arange(param['min'], param['max'], param['step'])
    elif param['type'] == 'list':
        return param['list']
    elif param['type'] == 'value':
        return [param['value']]
    elif param['type'] == 'map':
        return [f'{process_map(param['map'])}'.replace(' ', '').replace('[', '').replace(']', '')]
    elif param['type'] == 'custom':
        return custom_process_by_name(param['name'], param['value'])
    else:
        raise ValueError(f"Invalid parameter type: {param['type']}")

def process_parameters(parameters: list) -> dict:
    """Process parameter configuration and return a dictionary."""
    return {param['name']: process_parameter(param) for param in parameters.values()}

def get_parameter_combinations(params_dict: Dict[str, List[Any]]) -> Iterator[Dict[str, Any]]:
    """
    Generate all possible combinations of parameters while maintaining dictionary structure.
    
    Args:
        params_dict: Dictionary where each key has an array of possible values
            Example: {'param1': [1, 2], 'param2': ['a', 'b']}
    
    Returns:
        Iterator of dictionaries, each containing one combination of parameters
            Example: [
                {'param1': 1, 'param2': 'a'},
                {'param1': 1, 'param2': 'b'},
                {'param1': 2, 'param2': 'a'},
                {'param1': 2, 'param2': 'b'}
            ]
    """
    # Get the parameter names and their possible values
    param_names = list(params_dict.keys())
    param_values = [params_dict[name] for name in param_names]
    
    # Generate all possible combinations
    for combination in itertools.product(*param_values):
        # Create a dictionary for this combination
        yield dict(zip(param_names, combination))

def get_process_memory(pid: int) -> int:
    process = psutil.Process(pid)
    return process.memory_info().rss


def track_and_run(cmd: str):
   # Start the subprocess
    process = subprocess.Popen(
        cmd, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        stdin=subprocess.PIPE,
        shell=True
    )


    while process.poll() is None:
        time.sleep(1)

    return process.communicate()

def generate_map(volumes, width):
    """
    Generate a map with the given width.
    """
    if width < 2:
        raise ValueError("Width must be at least 2 meters.")

    matrix = [[0] * volumes for _ in range(volumes)]
    
    # Define how many top/bottom rows to set as obstacle
    rows_as_obstacle = int((volumes - width) / 2)

    for i in range(rows_as_obstacle):
        matrix[i] = [1] * volumes
        matrix[volumes - i - 1] = [1] * volumes

    return matrix

def set_fixed_variable(variable: str, value: Any, model_path: str = '../retqss/model/social_force_model.mo'):
    """
    Set a fixed variable in the config file.
    """
    subprocess.run(['sed', '-i', r's/\b' + variable + '\s*=\s*[0-9]\+/' + variable + ' = ' + str(value) + '/', model_path])