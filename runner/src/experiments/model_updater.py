"""Model file updater utilities for experiment configuration."""

import subprocess
from typing import Optional, Dict


class ModelUpdater:
    """Utility for updating .mo model files using sed commands."""
    
    def __init__(self, model_path: str):
        """Initialize with model file path."""
        self.model_path = model_path
    
    def update_parameter(self, param_name: str, value) -> None:
        """Update a single parameter in the model file."""
        subprocess.run([
            'sed', '-i', 
            rf's/\b{param_name}\s*=\s*[0-9.]\+/{param_name} = {value}/', 
            self.model_path
        ])
    
    def update_parameters(self, params: Dict[str, any]) -> None:
        """Update multiple parameters at once."""
        for param_name, value in params.items():
            if value is not None:
                self.update_parameter(param_name, value)


# Standalone function for simple use cases
def update_model_parameter(model_path: str, param_name: str, value) -> None:
    """Update a single parameter in a model file."""
    subprocess.run([
        'sed', '-i', 
        rf's/\b{param_name}\s*=\s*[0-9.]\+/{param_name} = {value}/', 
        model_path
    ])
