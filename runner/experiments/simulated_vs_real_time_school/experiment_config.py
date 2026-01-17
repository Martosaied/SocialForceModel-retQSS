"""
Configuration file for the simulated vs real time experiment.
Modify these parameters to customize the experiment without editing the main script.
"""

# ============================================================================
# SCENARIO CONFIGURATION
# ============================================================================

# School scenario parameters
CLASSROOM_EDGE = 7.0  # meters per classroom edge (49m² per classroom)
PEDESTRIANS_PER_CLASSROOM = 25  # students per classroom
GRID_DIVISIONS = 15  # 15x15 grid for the school layout

# Derived parameters (calculated automatically)
GRID_SIZE = GRID_DIVISIONS * CLASSROOM_EDGE  # Total grid size in meters
CLASSROOM_COUNT = 56  # Number of classrooms in 15x15 hallways configuration
PEDESTRIAN_COUNT = CLASSROOM_COUNT * PEDESTRIANS_PER_CLASSROOM  # Total pedestrians

# ============================================================================
# EXPERIMENT PARAMETERS
# ============================================================================

# Simulation durations to test (in seconds)
# From tens of seconds to tens of minutes
SIMULATION_DURATIONS = [
    10,      # 10 seconds
    30,      # 30 seconds
    60,      # 1 minute
    120,     # 2 minutes
    300,     # 5 minutes
    600,     # 10 minutes
    900,     # 15 minutes
    1200,    # 20 minutes
]

# Alternative shorter test suite (for quick testing)
SIMULATION_DURATIONS_SHORT = [
    10,      # 10 seconds
    30,      # 30 seconds
    60,      # 1 minute
    120,     # 2 minutes
]

# Alternative extended test suite (for comprehensive analysis)
SIMULATION_DURATIONS_EXTENDED = [
    10,      # 10 seconds
    30,      # 30 seconds
    60,      # 1 minute
    120,     # 2 minutes
    300,     # 5 minutes
    600,     # 10 minutes
    900,     # 15 minutes
    1200,    # 20 minutes
    1800,    # 30 minutes
    3600,    # 1 hour
]

# ============================================================================
# MODEL IMPLEMENTATIONS
# ============================================================================

# Implementation configurations
# Key: implementation ID
# Value: dictionary with name, description, color, and marker for plotting
IMPLEMENTATIONS = {
    1: {
        'name': 'Sin Optimizaciones',
        'description': 'RETQSS sin optimizaciones',
        'color': '#e74c3c',  # Red
        'marker': 'o'
    },
    2: {
        'name': 'Con Optimizaciones',
        'description': 'RETQSS optimizado',
        'color': '#2ecc71',  # Green
        'marker': 's'
    }
}

# ============================================================================
# EXECUTION CONTROL
# ============================================================================

# Set to True to run experiments, False to only generate plots from existing data
RUN_EXPERIMENT = True

# Number of iterations per experiment (for statistical significance)
# Each iteration runs the same simulation and timing is averaged
NUM_ITERATIONS = 10

# Model file to use
MODEL_NAME = 'helbing_school_hallway'

# Volume neighborhood implementation
VOLUME_NEIGHBORHOOD_TYPE = 0  # Face sharing

# Border implementations (different for each model)
BORDER_IMPLEMENTATION_UNOPTIMIZED = 4  # For implementation without optimizations
BORDER_IMPLEMENTATION_OPTIMIZED = 3    # For implementation with optimizations (BORDER_SURROUNDING_VOLUMES)

# ============================================================================
# PLOTTING CONFIGURATION
# ============================================================================

# Figure size for main plot
MAIN_FIGURE_SIZE = (14, 8)

# Figure size for secondary plot
SECONDARY_FIGURE_SIZE = (14, 8)

# DPI for saved figures
FIGURE_DPI = 300

# Whether to show value labels on plot points
SHOW_VALUE_LABELS = True

# Font sizes
TITLE_FONTSIZE = 16
AXIS_LABEL_FONTSIZE = 14
LEGEND_FONTSIZE = 11
ANNOTATION_FONTSIZE = 9

# ============================================================================
# OUTPUT CONFIGURATION
# ============================================================================

# Base directory for results
RESULTS_DIR = 'experiments/simulated_vs_real_time_school/results'

# Directory for test results
TEST_RESULTS_DIR = 'experiments/simulated_vs_real_time_school/test_results'

# Directory for custom results
CUSTOM_RESULTS_DIR = 'experiments/simulated_vs_real_time_school/custom_results'

# Output file names
RESULTS_CSV = 'experiments/simulated_vs_real_time_school/results_summary.csv'
MAIN_PLOT = 'experiments/simulated_vs_real_time_school/simulated_vs_real_time_comparison.png'
SECONDARY_PLOT = 'experiments/simulated_vs_real_time_school/execution_time_comparison.png'

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_experiment_info():
    """Return a dictionary with experiment information."""
    return {
        'scenario': 'Escuela (pasillos)',
        'grid_divisions': GRID_DIVISIONS,
        'grid_size': GRID_SIZE,
        'classroom_count': CLASSROOM_COUNT,
        'pedestrian_count': PEDESTRIAN_COUNT,
        'simulation_durations': SIMULATION_DURATIONS,
        'implementations': list(IMPLEMENTATIONS.keys()),
        'num_iterations': NUM_ITERATIONS,
        'total_experiments': len(SIMULATION_DURATIONS) * len(IMPLEMENTATIONS),
        'total_runs': len(SIMULATION_DURATIONS) * len(IMPLEMENTATIONS) * NUM_ITERATIONS
    }


def print_experiment_info():
    """Print experiment configuration information."""
    info = get_experiment_info()
    
    print("="*80)
    print("CONFIGURACIÓN DEL EXPERIMENTO")
    print("="*80)
    print(f"\nEscenario: {info['scenario']}")
    print(f"Grilla: {info['grid_divisions']}x{info['grid_divisions']} ({info['grid_size']}m x {info['grid_size']}m)")
    print(f"Aulas: {info['classroom_count']}")
    print(f"Peatones: {info['pedestrian_count']}")
    print(f"\nDuraciones a simular: {info['simulation_durations']} segundos")
    print(f"Implementaciones: {info['implementations']}")
    print(f"Iteraciones por experimento: {info['num_iterations']}")
    print(f"Total de experimentos: {info['total_experiments']}")
    print(f"Total de corridas: {info['total_runs']}")
    print("="*80)


# ============================================================================
# VALIDATION
# ============================================================================

def validate_configuration():
    """Validate experiment configuration."""
    errors = []
    
    # Check that durations are positive
    if any(d <= 0 for d in SIMULATION_DURATIONS):
        errors.append("All simulation durations must be positive")
    
    # Check that implementations are valid
    valid_implementations = [1, 2]
    if not all(impl in valid_implementations for impl in IMPLEMENTATIONS.keys()):
        errors.append(f"Invalid implementation IDs. Must be in {valid_implementations}")
    
    # Check that grid divisions is positive
    if GRID_DIVISIONS <= 0:
        errors.append("GRID_DIVISIONS must be positive")
    
    # Check that pedestrian count is positive
    if PEDESTRIAN_COUNT <= 0:
        errors.append("PEDESTRIAN_COUNT must be positive")
    
    if errors:
        print("Configuration errors found:")
        for error in errors:
            print(f"  - {error}")
        return False
    
    return True


if __name__ == '__main__':
    # When run directly, print configuration and validate
    if validate_configuration():
        print_experiment_info()
        print("\n✓ Configuration is valid")
    else:
        print("\n✗ Configuration has errors")

