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
    fast_mode: bool = False
    
    # Output flags
    plot: bool = True
    copy_results: bool = True
    
    # Additional settings
    output_dir: str = "results"
    experiment_name: Optional[str] = None
    
    # Custom parameters
    custom_params: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization to handle fast mode logic."""
        if self.fast_mode:
            self.skip_metrics = True
    
    @property
    def should_calculate_clustering(self) -> bool:
        """Check if clustering should be calculated."""
        return not (self.fast_mode or self.skip_metrics)
    
    @property
    def should_calculate_density(self) -> bool:
        """Check if density should be calculated."""
        return not (self.fast_mode or self.skip_metrics)
    
    @property
    def should_calculate_metrics(self) -> bool:
        """Check if any metrics should be calculated."""
        return not (self.fast_mode or self.skip_metrics)
    
    @property
    def performance_mode(self) -> str:
        """Get the current performance mode as a string."""
        if self.fast_mode:
            return "fast"
        elif self.skip_metrics:
            return "no_metrics"
        else:
            return "full"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        return {
            'verbose': self.verbose,
            'skip_metrics': self.skip_metrics,
            'fast_mode': self.fast_mode,
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
        print(f"  Fast mode: {self.fast_mode}")
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
        if hasattr(args, 'fast_mode'):
            self._config.fast_mode = args.fast_mode
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


def is_fast_mode() -> bool:
    """Check if fast mode is enabled."""
    return config_manager.config.fast_mode


def should_calculate_metrics() -> bool:
    """Check if any metrics should be calculated."""
    return config_manager.config.should_calculate_metrics


def should_calculate_clustering() -> bool:
    """Check if clustering should be calculated."""
    return config_manager.config.should_calculate_clustering


def should_calculate_density() -> bool:
    """Check if density should be calculated."""
    return config_manager.config.should_calculate_density


def get_performance_mode() -> str:
    """Get the current performance mode as a string."""
    return config_manager.config.performance_mode
