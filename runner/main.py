import argparse
import json
import os

from src.runner import run_experiment, compile_c_code, compile_model
from src.config_manager import update_config_from_args, print_config_status
from src.utils import load_config, create_output_dir, copy_results_to_latest
from experiments.performance_n_pedestrians.performance_n_pedestrians import performance_n_pedestrians
from experiments.performance_n_volumes.performance_n_volumes import performance_n_volumes
from experiments.performance_n_peds_m_vols.performance_n_peds_m_vols import performance_n_peds_m_vols
from experiments.average_velocity.average_velocity import average_velocity
from experiments.breaking_lanes.breaking_lanes import breaking_lanes
from experiments.breaking_obstacles.breaking_obstacles import breaking_obstacles
from experiments.lanes_by_iterations.lanes_by_iterations import lanes_by_iterations
from experiments.lanes_by_width.lanes_by_width import lanes_by_width
from experiments.deltaq.deltaq import deltaq
from experiments.lanes_by_B.lanes_by_B import lanes_by_B
from experiments.lanes_by_R.lanes_by_R import lanes_by_R
from experiments.lanes_by_A.lanes_by_A import lanes_by_A
from experiments.lanes_heatmap.lanes_heatmap import lanes_heatmap
from experiments.subway_attack_rate.subway_attack_rate import subway_attack_rate
from experiments.progress_update_dt.progress_update_dt import progress_update_dt
from src.plotter import Plotter


# Experiment registry to avoid long if-elif chains
EXPERIMENT_REGISTRY = {
    'performance_n_pedestrians': performance_n_pedestrians,
    'performance_n_volumes': performance_n_volumes,
    'performance_n_peds_m_vols': performance_n_peds_m_vols,
    'average_velocity': average_velocity,
    'breaking_lanes': breaking_lanes,
    'lanes_by_iterations': lanes_by_iterations,
    'lanes_by_width': lanes_by_width,
    'deltaq': deltaq,
    'lanes_by_B': lanes_by_B,
    'lanes_by_R': lanes_by_R,
    'lanes_by_A': lanes_by_A,
    'lanes_heatmap': lanes_heatmap,
    'breaking_obstacles': breaking_obstacles,
    'subway_attack_rate': subway_attack_rate,
    'progress_update_dt': progress_update_dt,
}


def validate_config_file(config_path):
    """Validate that the config file exists."""
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Configuration file not found: {config_path}")


def load_and_save_config(config_path, output_dir=None):
    """Load configuration and optionally save a copy to output directory."""
    validate_config_file(config_path)
    config = load_config(config_path)
    
    if output_dir:
        config_copy_path = os.path.join(output_dir, 'config.json')
        with open(config_copy_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    return config


def setup_run_parser(subparsers):
    """Setup the run command parser."""
    run_parser = subparsers.add_parser('run', description='Run experiments with JSON configuration')
    run_parser.add_argument('model', type=str, help='Name of the model to run (e.g., social_force_model)')
    run_parser.add_argument('--config', type=str, default='config.json', help='Path to JSON configuration file')
    run_parser.add_argument('--experiment-name', type=str, help='Name of the experiment (creates a subdirectory)')
    run_parser.add_argument('--compile', action='store_true', help='Compile the model before running')
    run_parser.add_argument('--compile-c', action='store_true', help='Compile the C++ code before running')
    run_parser.add_argument('--output-dir', type=str, default='results',
                            help='Base directory for experiment outputs (default: results)')
    run_parser.add_argument('--plot', action='store_true', help='Plot the results')
    run_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    run_parser.add_argument('--skip-metrics', action='store_true', help='Skip metric calculations (clustering and density) to save time')
    run_parser.add_argument('--fast-mode', action='store_true', help='Enable fast mode (skip all calculations)')


def setup_plot_parser(subparsers):
    """Setup the plot command parser."""
    plot_parser = subparsers.add_parser('plot', description='Plot results')
    plot_parser.add_argument('plot', type=str, help='plot type (gif, grouped_lanes)')
    plot_parser.add_argument('solution_file', type=str, help='Path to solution file')
    plot_parser.add_argument('--output-dir', type=str, default='results',
                             help='Base directory for experiment outputs (default: results)')
    plot_parser.add_argument('--config', type=str, default='config.json', help='Path to JSON configuration file')


def setup_experiments_parser(subparsers):
    """Setup the experiments command parser."""
    experiments_parser = subparsers.add_parser('experiments', description='Run experiments')
    experiments_parser.add_argument('experiment', type=str, help='Name of the experiment to run')
    experiments_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    experiments_parser.add_argument('--skip-metrics', action='store_true', help='Skip metric calculations (clustering and density) to save time')
    experiments_parser.add_argument('--fast-mode', action='store_true', help='Enable fast mode (skip all calculations)')


def handle_run_command(args):
    """Handle the run command."""
    # Create output directory with experiment name if provided
    output_dir = create_output_dir(args.output_dir, args.experiment_name)
    print(f"Created output directory: {output_dir}")

    # Load configuration and save copy
    config = load_and_save_config(args.config, output_dir)

    # Compile if requested
    if args.compile_c:
        compile_c_code()
    if args.compile:
        compile_model(args.model)

    # Update configuration from command line arguments
    update_config_from_args(args)

    # Run experiment
    run_experiment(config, output_dir, args.model, plot=args.plot)

    # Copy results from output directory to latest directory
    copy_results_to_latest(output_dir)
    print(f"\nExperiment completed. Results saved in {output_dir}")


def handle_plot_command(args):
    """Handle the plot command."""
    config = load_and_save_config(args.config)
    plotter = Plotter()
    
    plot_handlers = {
        'gif': lambda: plotter.flow_graph(args.solution_file, args.output_dir, config),
        'grouped_lanes': lambda: plotter.grouped_lanes_graph([args.solution_file], args.output_dir),
        'pedestrian_heatmap': lambda: (
            plotter.density_heatmap(args.solution_file, args.output_dir),
            plotter.density_row_graph(args.solution_file, args.output_dir)
        ),
        'flow_graph_infections': lambda: plotter.flow_graph_infections(args.solution_file, args.output_dir, config)
    }
    
    if args.plot in plot_handlers:
        plot_handlers[args.plot]()
    else:
        print(f"Unknown plot type: {args.plot}")


def handle_experiments_command(args):
    """Handle the experiments command."""
    # Update configuration from command line arguments
    update_config_from_args(args)
    
    if args.experiment in EXPERIMENT_REGISTRY:
        EXPERIMENT_REGISTRY[args.experiment]()
    else:
        print(f"Experiment {args.experiment} not found")
        print(f"Available experiments: {', '.join(EXPERIMENT_REGISTRY.keys())}")


def main():
    parser = argparse.ArgumentParser(description='Run experiments with JSON configuration')
    subparsers = parser.add_subparsers(dest='command', title='Tools', help='sub-command help')

    # Setup all command parsers
    setup_run_parser(subparsers)
    setup_plot_parser(subparsers)
    setup_experiments_parser(subparsers)

    args = parser.parse_args()

    # Handle commands
    if args.command == 'run':
        handle_run_command(args)
    elif args.command == 'plot':
        handle_plot_command(args)
    elif args.command == 'experiments':
        handle_experiments_command(args)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
