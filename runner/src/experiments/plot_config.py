"""
Shared plot styling configuration for experiments.

This module provides consistent styling across all experiment visualizations,
including publication-quality matplotlib settings and color schemes.
"""

import matplotlib.pyplot as plt

# Publication-quality plot styling configuration
PUBLICATION_STYLE = {
    'font.size': 18,
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica', 'Liberation Sans', 'Tahoma', 'Verdana'],
    'axes.titlesize': 35,
    'axes.titleweight': 'bold',
    'axes.titlepad': 10,
    'axes.labelsize': 35,
    'axes.labelweight': 'bold',
    'axes.labelpad': 6,
    'axes.linewidth': 1.2,
    'axes.edgecolor': '#333333',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'axes.axisbelow': True,
    'xtick.labelsize': 30,
    'ytick.labelsize': 30,
    'xtick.direction': 'out',
    'ytick.direction': 'out',
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'grid.color': '#b0b0b0',
    'grid.linestyle': '--',
    'grid.linewidth': 0.8,
    'grid.alpha': 0.3,
    'legend.fontsize': 30,
    'legend.frameon': True,
    'legend.framealpha': 0.9,
    'legend.edgecolor': '#cccccc',
    'lines.linewidth': 2.0,
    'lines.markersize': 6,
    'errorbar.capsize': 4,
    'figure.titlesize': 35,
    'figure.titleweight': 'bold',
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.3,
    'patch.edgecolor': '#333333',
    'patch.force_edgecolor': True,
    'patch.linewidth': 0.8,
}

# Default color scheme for common experiment categories
DEFAULT_COLORS = {
    'qss': '#FF6B6B',
    'retqss': '#4ECDC4',
    'retqss_opt': '#6C5CE7',
    'baseline': 'lightcoral',
    'optimized': 'skyblue',
    'reference': 'lightcoral',
    'face_sharing': 'skyblue',
    'vertex_sharing': 'coral',
}


def apply_publication_style():
    """Apply publication-quality styling to all plots.
    
    This should be called at the beginning of each experiment script
    to ensure consistent styling across all visualizations.
    """
    plt.rcParams.update(PUBLICATION_STYLE)


def reset_style():
    """Reset to default matplotlib style.
    
    Useful for restoring default settings after applying custom styles.
    """
    plt.rcdefaults()
