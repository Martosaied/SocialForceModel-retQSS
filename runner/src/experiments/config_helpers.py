"""
Configuration helper utilities for experiment setup.

This module provides a fluent API for building experiment configurations
and helper functions for common calculations like corridor boundaries
and grid parameters.
"""

import numpy as np
from typing import Any, Dict, Tuple


class ConfigBuilder:
    """Fluent interface for building experiment configurations.
    
    Provides a clean, chainable API for setting experiment parameters
    instead of manual dictionary manipulation.
    
    Example:
        ConfigBuilder(config) \\
            .set_iterations(10) \\
            .set_pedestrian_count(300) \\
            .set_pedestrian_implementation(Constants.PEDESTRIAN_NEIGHBORHOOD) \\
            .set_corridor(50.0, 20.0)
    """
    
    def __init__(self, config: Dict):
        """Initialize with existing config dictionary.
        
        Args:
            config: Existing configuration dictionary to modify
        """
        self.config = config
    
    def set_iterations(self, iterations: int):
        """Set number of iterations.
        
        Args:
            iterations: Number of experiment iterations to run
            
        Returns:
            self for method chaining
        """
        self.config['iterations'] = iterations
        return self
    
    def set_parameter(self, name: str, value: Any):
        """Set a parameter value.
        
        Args:
            name: Parameter name
            value: Parameter value
            
        Returns:
            self for method chaining
        """
        if name not in self.config['parameters']:
            self.config['parameters'][name] = {"name": name, "type": "value"}
        self.config['parameters'][name]['value'] = value
        return self
    
    def set_pedestrian_count(self, count: int):
        """Set N (pedestrian count).
        
        Args:
            count: Number of pedestrians
            
        Returns:
            self for method chaining
        """
        return self.set_parameter('N', count)
    
    def set_pedestrian_implementation(self, implementation: int):
        """Set PEDESTRIAN_IMPLEMENTATION.
        
        Args:
            implementation: Implementation type (e.g., Constants.PEDESTRIAN_NEIGHBORHOOD)
            
        Returns:
            self for method chaining
        """
        return self.set_parameter('PEDESTRIAN_IMPLEMENTATION', implementation)
    
    def set_border_implementation(self, implementation: int):
        """Set BORDER_IMPLEMENTATION.
        
        Args:
            implementation: Border implementation type (e.g., Constants.CORRIDOR_ONLY)
            
        Returns:
            self for method chaining
        """
        return self.set_parameter('BORDER_IMPLEMENTATION', implementation)
    
    def set_grid_size(self, size: float):
        """Set GRID_SIZE.
        
        Args:
            size: Grid size in meters
            
        Returns:
            self for method chaining
        """
        return self.set_parameter('GRID_SIZE', size)
    
    def set_volume_neighborhood_type(self, neighborhood_type: int):
        """Set VOLUME_NEIGHBORHOOD_TYPE.
        
        Args:
            neighborhood_type: 0 for face sharing, 1 for vertex sharing
            
        Returns:
            self for method chaining
        """
        return self.set_parameter('VOLUME_NEIGHBORHOOD_TYPE', neighborhood_type)
    
    def set_corridor(self, grid_size: float, width: float):
        """Set corridor boundaries (FROM_Y and TO_Y).
        
        Calculates centered corridor boundaries based on grid size and width.
        
        Args:
            grid_size: Total grid size in meters
            width: Corridor width in meters
            
        Returns:
            self for method chaining
        """
        from_y = (grid_size / 2) - (width / 2)
        to_y = (grid_size / 2) + (width / 2)
        self.set_parameter('FROM_Y', from_y)
        self.set_parameter('TO_Y', to_y)
        return self
    
    def build(self) -> Dict:
        """Return the configured config.
        
        Returns:
            The modified configuration dictionary
        """
        return self.config


# Standalone helper functions

def calculate_corridor_boundaries(grid_size: float, width: float) -> Tuple[float, float]:
    """Calculate FROM_Y and TO_Y for corridor configuration.
    
    Args:
        grid_size: Total grid size in meters
        width: Corridor width in meters
        
    Returns:
        Tuple of (from_y, to_y) boundaries
    """
    from_y = (grid_size / 2) - (width / 2)
    to_y = (grid_size / 2) + (width / 2)
    return from_y, to_y


def calculate_grid_params(n_pedestrians: int, density: float, 
                         cell_size: float) -> Tuple[float, int, float, float]:
    """Calculate grid size and divisions from parameters.
    
    Args:
        n_pedestrians: Number of pedestrians
        density: Target pedestrian density (pedestrians/m²)
        cell_size: Size of each cell in meters
        
    Returns:
        Tuple of (grid_size, grid_divisions, from_y, to_y)
    """
    grid_size = np.sqrt(n_pedestrians / density)
    grid_divisions = max(1, int(grid_size / cell_size))
    from_y, to_y = calculate_corridor_boundaries(grid_size, grid_size * 0.6)
    return grid_size, grid_divisions, from_y, to_y

