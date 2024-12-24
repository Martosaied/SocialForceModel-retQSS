import argparse
import json
import os
from datetime import datetime
import pandas as pd
from pathlib import Path
import subprocess
import shutil


def load_config(config_path: str) -> dict:
    """Load experiment configuration from JSON file."""
    with open(config_path, 'r') as f:
        return json.load(f)

def create_output_dir(base_dir: str = "experiments") -> str:
    """Create and return path to timestamped output directory."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"experiment_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir

def process_parameters(parameters: list) -> dict:
    """Process parameter configuration and return a dictionary."""
    return {param['name']: {**param, 'value': param['min']} for param in parameters}

def update_parameter(parameter: str, param_info: dict, parameters_state: dict) -> bool:
    """Update parameter value and return True if moved."""
    if param_info['type'] == 'range':
        if param_info['value'] + param_info['step'] > param_info['max']:
            return False
        parameters_state[parameter]['value'] = param_info['value'] + param_info['step']
        return True
    # elif param_info['type'] == 'list':
    #     if param_info['value'] + 1 > len(param_info['list']):
    #         return False
    #     parameters_state[parameter]['value'] = param_info['list'][param_info['value'] + 1]
    #     return True
    return False
