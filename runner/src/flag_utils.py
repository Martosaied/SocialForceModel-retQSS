"""
Utility functions for the enhanced flag system.
This module provides helper functions to easily check and use experiment flags.
"""

from .config_manager import get_config, print_config_status, get_performance_mode

def is_verbose():
    """Check if verbose mode is enabled."""
    return get_config().verbose

def should_skip_metrics():
    """Check if metric calculations should be skipped."""
    return get_config().skip_metrics

def is_fast_mode():
    """Check if fast mode is enabled."""
    return get_config().fast_mode

def get_all_flags():
    """Get all current flags as a dictionary."""
    config = get_config()
    return {
        'verbose': config.verbose,
        'skip_metrics': config.skip_metrics,
        'fast_mode': config.fast_mode
    }

def print_flags():
    """Print the current flag status."""
    print_config_status()

def should_calculate_metrics():
    """Check if any metrics should be calculated (not in fast mode)."""
    return get_config().should_calculate_metrics

def should_calculate_clustering():
    """Check if clustering should be calculated."""
    return get_config().should_calculate_clustering

def should_calculate_density():
    """Check if density should be calculated."""
    return get_config().should_calculate_density
