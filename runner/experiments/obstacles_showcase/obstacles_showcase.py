"""
Obstacles Showcase Experiment

This experiment demonstrates how to use volumes as obstacles in the simulation.
It runs two simulations:
1. A 100x100m scenario with a wall in the middle that pedestrians must avoid
2. A 100x100m scenario with random obstacles scattered throughout the geometry

The output is a single frame from each simulation showing the obstacle configuration
and pedestrian positions.
"""

import json
import os
import subprocess
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import sqrt

from src.runner import run_experiment, compile_c_code, compile_model
from src.utils import load_config, create_output_dir, copy_results_to_latest, process_parameters, get_parameter_combinations
from src.constants import Constants

# Configuration
GRID_SIZE = 100  # 100x100 meters
GRID_DIVISIONS = 20  # 20x20 grid = 5m per cell
CELL_SIZE = GRID_SIZE / GRID_DIVISIONS
WIDTH = 40  # Corridor width (20m centered in the grid)
PEDESTRIAN_DENSITY = 0.2  # Lower density to better visualize obstacles
MODEL_NAME = 'social_force_model'


def generate_wall_obstacles(divisions):
    """
    Generate a wall obstacle in the middle of the grid.
    The wall is placed vertically in the center of the grid, partially blocking the corridor.

    The obstacle matrix uses:
    - 0 = free space
    - 1 = obstacle

    Volume indexing: volume_id = y + divisions * x + 1
    - Volume 1 is at (y=0, x=0) bottom-left
    - Volume 2 is at (y=1, x=0) directly above Volume 1
    - obstacles[y][x] marks the volume at position (x, y)
    """
    obstacles = np.zeros((divisions, divisions))

    # Place a vertical wall in the middle of the grid
    # Middle column is at index divisions // 2
    mid_col = divisions // 2
    mid_col2 = divisions // 2 - 1

    # For 100x100m grid with 20 divisions (5m cells) and 20m corridor centered at Y=50:
    # The corridor goes from FROM_Y=40 to TO_Y=60, which corresponds to rows 8-11
    # (row 8 = Y 40-45, row 9 = Y 45-50, row 10 = Y 50-55, row 11 = Y 55-60)
    # Place wall in the middle of the corridor

    for row in range(9, 11):  # Rows 9-10 (middle of the corridor)
        obstacles[row, mid_col] = 1  # obstacles[y, x] - vertical wall at x=mid_col
        obstacles[row, mid_col2] = 1  # obstacles[y, x] - vertical wall at x=mid_col2

    return obstacles.tolist()


def generate_random_obstacles(divisions, num_obstacles=12, seed=42):
    """
    Generate random obstacles scattered throughout the geometry.
    Obstacles are placed randomly, avoiding the edges of the corridor.

    Args:
        divisions: Number of grid divisions per side
        num_obstacles: Number of random obstacle cells to place
        seed: Random seed for reproducibility
    """
    np.random.seed(seed)
    obstacles = np.zeros((divisions, divisions))

    # For 100x100m grid with 20 divisions (5m cells) and 20m corridor centered at Y=50:
    # The corridor goes from FROM_Y=40 to TO_Y=60 (rows 8-11)
    # Place obstacles within and around the corridor area
    corridor_rows = list(range(7, 13))  # Rows 7-12 (including some buffer around corridor)

    placed = 0
    attempts = 0
    max_attempts = 150

    while placed < num_obstacles and attempts < max_attempts:
        row = np.random.choice(corridor_rows)
        col = np.random.randint(10, divisions - 10)  # Avoid edges

        # Don't place obstacles at the very edges of the corridor
        if obstacles[row, col] == 0:
            obstacles[row, col] = 1
            placed += 1

        attempts += 1

    # obstacles[y][x] marks volume at (x, y) - no transpose needed
    return obstacles.tolist()


def plot_single_frame_with_obstacles(solution_file, output_path, parameters, obstacles_map, title, frame_time=None):
    """
    Plot a single frame of the simulation showing obstacles and pedestrian positions.

    Args:
        solution_file: Path to the solution CSV file
        output_path: Path to save the output image
        parameters: Simulation parameters dictionary
        obstacles_map: 2D list representing obstacle positions (1=obstacle, 0=free)
        title: Title for the plot
        frame_time: Optional specific time to capture (uses middle of simulation if None)
    """
    # Get grid parameters
    grid_size = parameters.get('GRID_SIZE', 100)
    volumes_count = len(obstacles_map)
    cell_size = grid_size / volumes_count
    n_pedestrians = parameters.get('N', 200)
    from_y = parameters.get('FROM_Y', 15)
    to_y = parameters.get('TO_Y', 35)

    # Read the solution file
    df = pd.read_csv(solution_file)

    # Determine the frame to plot
    if frame_time is not None:
        time_diff = abs(df['time'] - frame_time)
        frame_index = time_diff.idxmin()
    else:
        # Use a frame near the middle-end of the simulation
        frame_index = min(len(df) - 1, int(len(df) * 0.7))

    row = df.iloc[frame_index]
    prev_row = df.iloc[frame_index - 1] if frame_index > 0 else None

    # Create the figure
    fig, ax = plt.subplots(figsize=(14, 14))
    ax.set_xlim(0, grid_size)
    ax.set_ylim(0, grid_size)

    # Draw corridor area in white
    corridor_main = plt.Rectangle(
        (0, from_y), grid_size, to_y - from_y,
        facecolor='#FFFFFF', alpha=1, edgecolor='none'
    )
    ax.add_patch(corridor_main)

    # Draw grid lines
    for i in range(volumes_count + 1):
        ax.axvline(x=i * cell_size, color='lightgray', linestyle='-', alpha=0.5, linewidth=0.8)
        ax.axhline(y=i * cell_size, color='lightgray', linestyle='-', alpha=0.5, linewidth=0.8)

    # Draw obstacles - obstacles_array[y][x] marks volume at (x, y)
    obstacles_array = np.array(obstacles_map)
    for y in range(volumes_count):
        for x in range(volumes_count):
            if obstacles_array[y, x] == 1:
                x_pos = x * cell_size
                y_pos = y * cell_size

                rect = plt.Rectangle(
                    (x_pos, y_pos), cell_size, cell_size,
                    facecolor='#404040', alpha=0.9, edgecolor='black', linewidth=1
                )
                ax.add_patch(rect)

    # Plot pedestrians
    positions_x = []
    positions_y = []
    colors = []
    velocities_x = []
    velocities_y = []

    for i in range(1, n_pedestrians + 1):
        px_col = f'PX[{i}]'
        py_col = f'PY[{i}]'
        ps_col = f'PS[{i}]'

        if px_col not in row or pd.isna(row[px_col]):
            continue

        x = row[px_col]
        y = row[py_col]
        state = row.get(ps_col, 0)

        # Calculate velocities
        vx, vy = 0, 0
        if prev_row is not None:
            dt = row['time'] - prev_row['time']
            if dt > 0:
                prev_x = prev_row.get(px_col, x)
                prev_y = prev_row.get(py_col, y)
                vx = (x - prev_x) / dt
                vy = (y - prev_y) / dt

        positions_x.append(x)
        positions_y.append(y)
        colors.append('#FF4444' if state == 1 else '#4444FF')
        velocities_x.append(vx)
        velocities_y.append(vy)

    # Plot scatter points
    if positions_x:
        ax.scatter(positions_x, positions_y, c=colors, s=150, alpha=0.8, edgecolors='black', linewidth=0.5)

        # Add velocity vectors
        if prev_row is not None:
            lengths = [sqrt(vx**2 + vy**2) for vx, vy in zip(velocities_x, velocities_y)]
            norm_vx = [vx/l if l > 0 else 0 for vx, l in zip(velocities_x, lengths)]
            norm_vy = [vy/l if l > 0 else 0 for vy, l in zip(velocities_y, lengths)]
            ax.quiver(positions_x, positions_y, norm_vx, norm_vy,
                     color='black', alpha=0.5, width=0.003, scale=25)

    # Create legend
    legend_elements = [
        plt.Rectangle((0, 0), 1, 1, facecolor='#404040', alpha=0.9, edgecolor='black', label='Obstacles'),
        plt.scatter([], [], c='#FF4444', s=100, label='Moving Right'),
        plt.scatter([], [], c='#4444FF', s=100, label='Moving Left'),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=12)

    # Labels and title
    fig.suptitle(title, fontsize=18, fontweight='bold')
    ax.set_title(f'Time: {row["time"]:.2f}s | Active Pedestrians: {len(positions_x)}', fontsize=14)
    ax.set_xlabel('X Position (meters)', fontsize=14)
    ax.set_ylabel('Y Position (meters)', fontsize=14)
    ax.set_aspect('equal')

    # Save the frame
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

    print(f"Frame saved to: {output_path}")
    return output_path


def run_simulation(config, output_dir, experiment_name, obstacles_map=None):
    """
    Run a single simulation with the given obstacle configuration.

    Args:
        config: Base configuration dictionary
        output_dir: Output directory for results
        experiment_name: Name for this experiment run
    """
    # Calculate pedestrian count
    pedestrians = int(PEDESTRIAN_DENSITY * WIDTH * GRID_SIZE)
    config['parameters']['N']['value'] = pedestrians
    config['parameters']['GRID_SIZE']['value'] = GRID_SIZE

    # Set corridor boundaries
    config['parameters']['FROM_Y']['value'] = (GRID_SIZE / 2) - (WIDTH / 2)
    config['parameters']['TO_Y']['value'] = (GRID_SIZE / 2) + (WIDTH / 2)

    if obstacles_map is not None:
        config['parameters']['OBSTACLES']['map'] = obstacles_map

    # Save config copy
    config_path = os.path.join(output_dir, 'config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    # Update model parameters
    subprocess.run([
        'sed', '-i',
        f's/\\bGRID_DIVISIONS\\s*=\\s*[0-9]\\+/GRID_DIVISIONS = {GRID_DIVISIONS}/',
        '../retqss/model/social_force_model.mo'
    ])
    subprocess.run([
        'sed', '-i',
        f's/\\bN\\s*=\\s*[0-9]\\+/N = {pedestrians}/',
        '../retqss/model/social_force_model.mo'
    ])

    # Compile and run
    compile_c_code()
    compile_model(MODEL_NAME)

    run_experiment(
        config,
        output_dir,
        MODEL_NAME,
        plot=False,
        copy_results=True
    )

    return os.path.join(output_dir, 'result_0.csv')


def obstacles_showcase():
    """
    Main function to run the obstacles showcase experiment.
    Runs two simulations demonstrating different obstacle configurations.
    """
    print("=" * 60)
    print("OBSTACLES SHOWCASE EXPERIMENT")
    print("=" * 60)
    print(f"\nGrid: {GRID_SIZE}x{GRID_SIZE}m with {GRID_DIVISIONS}x{GRID_DIVISIONS} divisions")
    print(f"Cell size: {CELL_SIZE}m")
    print(f"Corridor width: {WIDTH}m")
    print()

    # Load base config
    base_config = load_config('./experiments/obstacles_showcase/config.json')

    # ===== SIMULATION 1: Wall Obstacle =====
    print("-" * 60)
    print("SIMULATION 1: Wall Obstacle in the Middle")
    print("-" * 60)

    wall_output_dir = create_output_dir(
        'experiments/obstacles_showcase/results',
        'wall_obstacle'
    )

    wall_obstacles = generate_wall_obstacles(GRID_DIVISIONS)

    wall_config = json.loads(json.dumps(base_config))  # Deep copy
    wall_solution = run_simulation(wall_config, wall_output_dir, 'wall_obstacle', wall_obstacles)

    # Get parameters for plotting
    wall_params = list(get_parameter_combinations(process_parameters(wall_config.get('parameters', {}))))[0]

    # Plot frame
    wall_frame_path = os.path.join(wall_output_dir, 'wall_obstacle_frame.png')
    plot_single_frame_with_obstacles(
        wall_solution,
        wall_frame_path,
        wall_params,
        wall_obstacles,
        'Obstacle Showcase: Wall in the Middle (100x100m scenario)'
    )

    copy_results_to_latest(wall_output_dir)
    print(f"\nWall simulation completed. Results in: {wall_output_dir}")

    # ===== SIMULATION 2: Random Obstacles =====
    print("\n" + "-" * 60)
    print("SIMULATION 2: Random Obstacles")
    print("-" * 60)

    random_output_dir = create_output_dir(
        'experiments/obstacles_showcase/results',
        'random_obstacles'
    )

    random_obstacles = generate_random_obstacles(GRID_DIVISIONS, num_obstacles=4)

    random_config = json.loads(json.dumps(base_config))  # Deep copy
    random_solution = run_simulation(random_config, random_output_dir, 'random_obstacles', random_obstacles)

    # Get parameters for plotting
    random_params = list(get_parameter_combinations(process_parameters(random_config.get('parameters', {}))))[0]

    # Plot frame
    random_frame_path = os.path.join(random_output_dir, 'random_obstacles_frame.png')
    plot_single_frame_with_obstacles(
        random_solution,
        random_frame_path,
        random_params,
        random_obstacles,
        'Obstacle Showcase: Random Obstacles (100x100m scenario)'
    )

    copy_results_to_latest(random_output_dir)
    print(f"\nRandom obstacles simulation completed. Results in: {random_output_dir}")

    # # Summary
    print("\n" + "=" * 60)
    print("EXPERIMENT COMPLETED")
    print("=" * 60)
    print(f"\nOutput frames:")
    print(f"  1. Wall obstacle:    {wall_frame_path}")
    print()


if __name__ == '__main__':
    obstacles_showcase()
