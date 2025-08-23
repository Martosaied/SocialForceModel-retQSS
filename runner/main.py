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


def main():
    parser = argparse.ArgumentParser(description='Run experiments with JSON configuration')
    subparsers = parser.add_subparsers(dest='command', title='Tools', help='sub-command help')

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

    plot_parser = subparsers.add_parser('plot', description='Plot results')
    plot_parser.add_argument('plot', type=str, help='plot type (gif, grouped_lanes)')
    plot_parser.add_argument('solution_file', type=str, help='Path to solution file')
    plot_parser.add_argument('--output-dir', type=str, default='results',
                             help='Base directory for experiment outputs (default: results)')
    plot_parser.add_argument('--config', type=str, default='config.json', help='Path to JSON configuration file')
    experiments_parser = subparsers.add_parser('experiments', description='Run experiments')
    experiments_parser.add_argument('experiment', type=str, help='Name of the experiment to run')
    experiments_parser.add_argument('--verbose', action='store_true', help='Verbose output')
    experiments_parser.add_argument('--skip-metrics', action='store_true', help='Skip metric calculations (clustering and density) to save time')
    experiments_parser.add_argument('--fast-mode', action='store_true', help='Enable fast mode (skip all calculations)')

    args = parser.parse_args()

    if args.command == 'run':
        # Validate config file exists
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Configuration file not found: {args.config}")

        # Load configuration
        config = load_config(args.config)

        # Create output directory with experiment name if provided
        output_dir = create_output_dir(args.output_dir, args.experiment_name)
        print(f"Created output directory: {output_dir}")

        # Save config copy in experiment directory
        config_copy_path = os.path.join(output_dir, 'config.json')
        with open(config_copy_path, 'w') as f:
            json.dump(config, f, indent=2)

        # Compile the C++ code if requested
        if args.compile_c:
            compile_c_code()

        # Compile the model if requested
        if args.compile:
            compile_model(args.model)

        # Update configuration from command line arguments
        update_config_from_args(args)

        # Run experiment
        run_experiment(config, output_dir, args.model, plot=args.plot)

        # Copy results from output directory to latest directory
        copy_results_to_latest(output_dir)
        print(f"\nExperiment completed. Results saved in {output_dir}")
    elif args.command == 'plot':
        # Validate config file exists
        if not os.path.exists(args.config):
            raise FileNotFoundError(f"Configuration file not found: {args.config}")

        # Load configuration
        config = load_config(args.config)

        plotter = Plotter()
        if args.plot == 'gif':
            plotter.flow_graph(args.solution_file, args.output_dir, config)
        elif args.plot == 'grouped_lanes':
            plotter.grouped_lanes_graph([args.solution_file], args.output_dir)
        elif args.plot == 'pedestrian_heatmap':
            plotter.density_heatmap(args.solution_file, args.output_dir)
            plotter.density_row_graph(args.solution_file, args.output_dir)
        elif args.plot == 'flow_graph_infections':
            plotter.flow_graph_infections(args.solution_file, args.output_dir, config)
    elif args.command == 'experiments':
        # Update configuration from command line arguments
        update_config_from_args(args)
        
        if args.experiment == 'performance_n_pedestrians':
            performance_n_pedestrians()
        elif args.experiment == 'performance_n_volumes':
            performance_n_volumes()
        elif args.experiment == 'performance_n_peds_m_vols':
            performance_n_peds_m_vols()
        elif args.experiment == 'average_velocity':
            average_velocity()
        elif args.experiment == 'breaking_lanes':
            breaking_lanes()
        elif args.experiment == 'lanes_by_iterations':
            lanes_by_iterations()
        elif args.experiment == 'lanes_by_width':
            lanes_by_width()
        elif args.experiment == 'deltaq':
            deltaq()
        elif args.experiment == 'lanes_by_B':
            lanes_by_B()
        elif args.experiment == 'lanes_by_R':
            lanes_by_R()
        elif args.experiment == 'lanes_by_A':
            lanes_by_A()
        elif args.experiment == 'lanes_heatmap':
            lanes_heatmap()
        elif args.experiment == 'breaking_obstacles':
            breaking_obstacles()
        elif args.experiment == 'subway_attack_rate':
            subway_attack_rate()
        elif args.experiment == 'progress_update_dt':
            progress_update_dt()
        elif args.experiment == 'flag_example':
            flag_example()
        else:
            print(f"Experiment {args.experiment} not found")
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
