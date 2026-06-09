import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np
from math import sqrt
from src.utils import parse_walls, process_parameters, get_parameter_combinations

class FlowGraphSingleFrame:
    """
    Plot a single frame of the flow graph at the middle of the simulation.

    Args:
        solution_file: The path to the solution file.
        output_dir: The path to the output directory. It will be created if it doesn't exist.
        parameters: The parameters of the simulation.
        frame_time: The time at which to capture the frame. If None, uses the middle of the simulation.

    Returns:
        None
    """

    def __init__(self, solution_file, output_dir, parameters, frame_time=None):
        self.solution_file = solution_file
        self.output_dir = output_dir
        self.parameters = list(get_parameter_combinations(process_parameters(parameters.get('parameters', {}))))[0]
        self.frame_time = [5, 30, 70, 160, 230, 290]

    def _get_volume_type_map(self, volumes_count):
        """
        Create a map of volume IDs to their types (obstacle, hallway, classroom).
        Returns a dictionary mapping volume_id -> volume_type
        """
        volume_type_map = {}

        # Get processed volume IDs (comma-separated strings)
        obstacles_str = self.parameters.get('OBSTACLES', '')
        hallways_str = self.parameters.get('HALLWAYS', '')
        classrooms_str = self.parameters.get('CLASSROOMS', '')

        # Parse obstacle volume IDs
        if obstacles_str:
            obstacle_ids = [int(x.strip()) for x in obstacles_str.split(',') if x.strip()]
            for volume_id in obstacle_ids:
                volume_type_map[volume_id] = 'obstacle'

        # Parse hallway volume IDs
        if hallways_str:
            hallway_ids = [int(x.strip()) for x in hallways_str.split(',') if x.strip()]
            for volume_id in hallway_ids:
                volume_type_map[volume_id] = 'hallway'

        # Parse classroom volume IDs
        if classrooms_str:
            classroom_ids = [int(x.strip()) for x in classrooms_str.split(',') if x.strip()]
            for volume_id in classroom_ids:
                volume_type_map[volume_id] = 'classroom'

        return volume_type_map

    def plot_single_frame(self, frame_time):
        walls = self.parameters.get('WALLS', [])
        walls = parse_walls(walls)

        # Get grid parameters
        GRID_SIZE = self.parameters.get('GRID_SIZE', 50)
        VOLUMES_COUNT = 100
        CELL_SIZE = GRID_SIZE / 100
        N = self.parameters.get('N', 300)

        # Corridor parameters define the visible Y window
        FROM_Y = self.parameters.get('FROM_Y', 0)
        TO_Y = self.parameters.get('TO_Y', GRID_SIZE)

        # Get volume type mapping
        volume_type_map = self._get_volume_type_map(VOLUMES_COUNT)

        # Define colors for different volume types
        volume_colors = {
            'obstacle': 'black',
            'hallway': 'lightgray',
            'classroom': 'lightblue'
        }

        # Read the solution file
        df = pd.read_csv(self.solution_file)
        
        # Find the closest time to the specified frame_time
        time_diff = abs(df['time'] - frame_time)
        frame_index = time_diff.idxmin()
        
        row = df.iloc[frame_index]
        
        # Get previous row for velocity calculation
        prev_row = None
        if frame_index > 0:
            prev_row = df.iloc[frame_index - 1]

        # Collect particle positions and velocities first (needed for axis bounds)
        frame_positions_x = []
        frame_positions_y = []
        frame_positions_color = []
        frame_velocities_x = []
        frame_velocities_y = []

        for i in range(1, N):
            if row.get(f'PX[{i}]') is None:
                continue

            x = row[f'PX[{i}]']
            y = row[f'PY[{i}]']
            state = row[f'PS[{i}]']

            vx = 0
            vy = 0
            if prev_row is not None:
                dt = row['time'] - prev_row['time']
                if dt > 0:
                    prev_x = prev_row[f'PX[{i}]']
                    prev_y = prev_row[f'PY[{i}]']
                    vx = (x - prev_x) / dt
                    vy = (y - prev_y) / dt

            if state == 1:
                color = '#FF4444'
            else:
                color = '#4444FF'

            frame_positions_x.append(x)
            frame_positions_y.append(y)
            frame_positions_color.append(color)
            frame_velocities_x.append(vx)
            frame_velocities_y.append(vy)

        # Plot in corridor-relative coordinates: Y axis goes 0 -> (TO_Y - FROM_Y)
        y_offset = FROM_Y
        x_min, x_max = 0, GRID_SIZE
        y_min, y_max = 0, TO_Y - FROM_Y

        def ny(y_world):
            return y_world - y_offset

        frame_positions_y_plot = [ny(y) for y in frame_positions_y]

        x_extent = x_max - x_min
        y_extent = y_max - y_min
        BASE_W = 16.0
        if x_extent > 0 and y_extent > 0:
            fig_h = BASE_W * (y_extent / x_extent)
            fig_h = max(min(fig_h, 24.0), 2.0)
        else:
            fig_h = 12.0

        fig, ax = plt.subplots(figsize=(BASE_W, fig_h))
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect('equal')

        # Pre-calculate cell positions and IDs for efficiency
        cell_positions = []
        cell_ids = []
        for i in range(VOLUMES_COUNT):
            for j in range(VOLUMES_COUNT):
                cell_id = i % VOLUMES_COUNT + VOLUMES_COUNT * j + 1
                cell_positions.append((j * CELL_SIZE, i * CELL_SIZE))
                cell_ids.append(cell_id)

        # Add grid lines aligned with cell boundaries; skip rows below the
        # relative origin so the grid starts at y = 0 on the plotted axis
        grid_lines_x = [CELL_SIZE * i for i in range(VOLUMES_COUNT + 1)]
        grid_lines_y = [CELL_SIZE * i for i in range(VOLUMES_COUNT + 1)]
        for gx in grid_lines_x:
            ax.axvline(x=gx, color='lightgray', linestyle='-', alpha=0.4, linewidth=1.2)
        for gy in grid_lines_y:
            if ny(gy) >= 0:
                ax.axhline(y=ny(gy), color='lightgray', linestyle='-', alpha=0.4, linewidth=1.2)

        # Color each cell based on volume type
        for pos, cell_id in zip(cell_positions, cell_ids):
            if cell_id in volume_type_map:
                volume_type = volume_type_map[cell_id]
                color = volume_colors[volume_type]
                alpha = 0.7 if volume_type == 'obstacle' else 0.5

                rect = plt.Rectangle(
                    (pos[0], ny(pos[1])),
                    CELL_SIZE,
                    CELL_SIZE,
                    facecolor=color,
                    alpha=alpha,
                    edgecolor='none'
                )
                ax.add_patch(rect)

        # Plot wall segments
        for wall in walls:
            ax.plot(
                [wall['from_x'], wall['to_x']],
                [ny(wall['from_y']), ny(wall['to_y'])],
                'k-', linewidth=6, label='_nolegend_'
            )

        # Plot scatter points
        if len(frame_positions_x) > 0:
            scatter = ax.scatter(frame_positions_x, frame_positions_y_plot, c=frame_positions_color, s=200)
        
        # Add velocity vectors with quiver only if we have velocity data
        if prev_row is not None and len(frame_velocities_x) > 0:
            # Scale factor for velocity vectors (adjust this value to make arrows more visible)
            length_velocities = [
                sqrt(frame_velocities_x[i] ** 2 + frame_velocities_y[i] ** 2) for i in range(len(frame_velocities_x))
            ]
            normalize_velocities_x = [
                frame_velocities_x[i] / length_velocities[i] if length_velocities[i] > 0 else 0 for i in range(len(frame_velocities_x))
            ]
            normalize_velocities_y = [
                frame_velocities_y[i] / length_velocities[i] if length_velocities[i] > 0 else 0 for i in range(len(frame_velocities_y))
            ]
            ax.quiver(frame_positions_x, frame_positions_y_plot,
                    np.array(normalize_velocities_x), 
                    np.array(normalize_velocities_y),
                    color='gray', alpha=0.5, width=0.003, scale=30)

        # Add axis labels with units
        ax.set_xlabel('Posición X (metros)', fontsize=25)
        ax.set_ylabel('Posición Y relativa (metros)', fontsize=25)

        # Improve tick labels
        ax.tick_params(axis='both', which='major', labelsize=20)

        # Hide Y tick labels in the bottom padding (y < 0); keep margin without negative numbers
        ax.yaxis.set_major_formatter(
            FuncFormatter(lambda y, pos: '' if y < 0 else f'{y:g}')
        )

        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save the single frame
        output_path = os.path.join(self.output_dir, f'flowgraph_single_frame_{frame_time}.png')
        plt.savefig(output_path, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)
        
        print(f"Frame único guardado en: {output_path}")
        return output_path

    def plot(self):
        for frame_time in self.frame_time:
            self.plot_single_frame(frame_time)