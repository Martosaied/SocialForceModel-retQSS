"""
Configuration Manager for experiment flags and settings.
This module provides a centralized way to manage experiment configuration.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import argparse


@dataclass
class ExperimentConfig:
    """Configuration class for experiment settings."""
    
    # Performance flags
    verbose: bool = False
    skip_metrics: bool = False
    
    # Output flags
    plot: bool = True
    copy_results: bool = True
    
    # Additional settings
    output_dir: str = "results"
    experiment_name: Optional[str] = None
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)
    

    
    @property
    def should_calculate_metrics(self) -> bool:
        """Check if any metrics should be calculated."""
        return not self.skip_metrics
    
    @property
    def performance_mode(self) -> str:
        """Get the current performance mode as a string."""
        if self.skip_metrics:
            return "no_metrics"
        else:
            return "full"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'verbose': self.verbose,
            'skip_metrics': self.skip_metrics,
            'plot': self.plot,
            'copy_results': self.copy_results,
            'output_dir': self.output_dir,
            'experiment_name': self.experiment_name,
            'custom_params': self.custom_params,
            'performance_mode': self.performance_mode
        }
    
    def print_status(self):
        """Print current configuration status."""
        print("Experiment Configuration:")
        print(f"  Verbose: {self.verbose}")
        print(f"  Skip metrics: {self.skip_metrics}")
        print(f"  Plot: {self.plot}")
        print(f"  Copy results: {self.copy_results}")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Performance mode: {self.performance_mode}")
        if self.custom_params:
            print(f"  Custom parameters: {self.custom_params}")
        print()


class ConfigManager:
    """Centralized configuration manager."""
    
    def __init__(self):
        self._config = ExperimentConfig()
    
    @property
    def config(self) -> ExperimentConfig:
        """Get the current configuration."""
        return self._config
    
    def update_from_args(self, args: argparse.Namespace):
        """Update configuration from command line arguments."""
        if hasattr(args, 'verbose'):
            self._config.verbose = args.verbose
        if hasattr(args, 'skip_metrics'):
            self._config.skip_metrics = args.skip_metrics
        if hasattr(args, 'plot'):
            self._config.plot = args.plot
        if hasattr(args, 'copy_results'):
            self._config.copy_results = args.copy_results
        if hasattr(args, 'output_dir'):
            self._config.output_dir = args.output_dir
        if hasattr(args, 'experiment_name'):
            self._config.experiment_name = args.experiment_name
    
    def set_custom_param(self, key: str, value: Any):
        """Set a custom parameter."""
        self._config.custom_params[key] = value
    
    def get_custom_param(self, key: str, default: Any = None) -> Any:
        """Get a custom parameter."""
        return self._config.custom_params.get(key, default)
    
    def reset(self):
        """Reset configuration to defaults."""
        self._config = ExperimentConfig()
    
    def print_status(self):
        """Print current configuration status."""
        self._config.print_status()


# Global instance
config_manager = ConfigManager()


# Convenience functions for backward compatibility
def get_config() -> ExperimentConfig:
    """Get the current configuration."""
    return config_manager.config


def update_config_from_args(args: argparse.Namespace):
    """Update configuration from command line arguments."""
    config_manager.update_from_args(args)


def print_config_status():
    """Print current configuration status."""
    config_manager.print_status()


def is_verbose() -> bool:
    """Check if verbose mode is enabled."""
    return config_manager.config.verbose


def should_skip_metrics() -> bool:
    """Check if metric calculations should be skipped."""
    return config_manager.config.skip_metrics


def get_performance_mode() -> str:
    """Get the current performance mode as a string."""
    return config_manager.config.performance_mode
