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


def load_config(config_path: str) -> dict:
    """Load experiment configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def create_output_dir(base_dir: str = "experiments", experiment_name: str = None) -> str:
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

def process_map(map):
    obstacle_indices = []
    for i in range(len(map)):
        for j in range(len(map[i])):
            if map[i][j] == 1:
                obstacle_indices.append(i % len(map) + len(map[i]) * j + 1)
    
    return obstacle_indices

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
    else:
        raise ValueError(f"Invalid parameter type: {param['type']}")

def process_parameters(parameters: list) -> dict:
    """Process parameter configuration and return a dictionary."""
    return {param['name']: process_parameter(param) for param in parameters}

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