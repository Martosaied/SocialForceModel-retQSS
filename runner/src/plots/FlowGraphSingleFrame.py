import os
import pandas as pd
import matplotlib.pyplot as plt
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
        self.frame_time = frame_time


    def plot(self):
        walls = self.parameters.get('WALLS', [])
        walls = parse_walls(walls)
        
        # Get grid parameters
        GRID_SIZE = self.parameters.get('GRID_SIZE', 50)
        VOLUMES_COUNT = 10
        CELL_SIZE = GRID_SIZE / VOLUMES_COUNT
        N = self.parameters.get('N', 300)

        # Read the solution file
        df = pd.read_csv(self.solution_file)
        
        # Determine the frame to plot
        if self.frame_time is not None:
            # Find the closest time to the specified frame_time
            time_diff = abs(df['time'] - self.frame_time)
            frame_index = time_diff.idxmin()
        else:
            # Use the middle of the simulation
            frame_index = 150
        
        row = df.iloc[frame_index]
        
        # Get previous row for velocity calculation
        prev_row = None
        if frame_index > 0:
            prev_row = df.iloc[frame_index - 1]
        
        # Create the figure
        fig, ax = plt.subplots(figsize=(12, 12))

        # Set up the plot area
        ax.set_xlim(0, GRID_SIZE)
        ax.set_ylim(0, GRID_SIZE)

        # Pre-calculate cell positions and IDs for efficiency
        cell_positions = []
        cell_ids = []
        for i in range(VOLUMES_COUNT):
            for j in range(VOLUMES_COUNT):
                cell_id = i % VOLUMES_COUNT + VOLUMES_COUNT * j + 1
                cell_positions.append((j * CELL_SIZE, i * CELL_SIZE))
                cell_ids.append(cell_id)
        
        # Pre-calculate grid lines
        grid_lines_x = [CELL_SIZE * i for i in range(VOLUMES_COUNT + 1)]
        grid_lines_y = [CELL_SIZE * i for i in range(VOLUMES_COUNT + 1)]

        # Add grid lines efficiently
        for x in grid_lines_x:
            ax.axvline(x=x, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
        for y in grid_lines_y:
            ax.axhline(y=y, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)

        # Plot wall segments
        for wall in walls:
            ax.plot(
                [wall['from_x'], wall['to_x']], 
                [wall['from_y'], wall['to_y']], 
                'k-', linewidth=3, label='_nolegend_'
            )

        # Plot each particle
        frame_positions_x = []
        frame_positions_y = []
        frame_positions_color = []
        frame_velocities_x = []
        frame_velocities_y = []
        
        for i in range(1, N):  # N particles
            if row.get(f'PX[{i}]') is None:
                continue

            x = row[f'PX[{i}]']
            y = row[f'PY[{i}]']
            state = row[f'PS[{i}]']

            # Calculate velocities from position changes if we have a previous frame
            vx = 0
            vy = 0
            if prev_row is not None:
                dt = row['time'] - prev_row['time']
                if dt > 0:  # Avoid division by zero
                    prev_x = prev_row[f'PX[{i}]']
                    prev_y = prev_row[f'PY[{i}]']
                    vx = (x - prev_x) / dt
                    vy = (y - prev_y) / dt

            if state == 1:  # Right
                color = 'red'
            else:  # Left
                color = 'blue'

            frame_positions_x.append(x)
            frame_positions_y.append(y)
            frame_positions_color.append(color)
            frame_velocities_x.append(vx)
            frame_velocities_y.append(vy)

        # Plot scatter points
        if len(frame_positions_x) > 0:
            scatter = ax.scatter(frame_positions_x, frame_positions_y, c=frame_positions_color, s=100, alpha=0.8)
        
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
            ax.quiver(frame_positions_x, frame_positions_y, 
                    np.array(normalize_velocities_x), 
                    np.array(normalize_velocities_y),
                    color='black', alpha=0.7, width=0.003, scale=20)
        
        # Create legend elements
        legend_elements = [
            plt.scatter([], [], c='blue', s=100, label='Izquierda'),
            plt.scatter([], [], c='red', s=100, label='Derecha'),
        ]
        
        # Add velocity vector to legend if we have velocity data
        if prev_row is not None and len(frame_velocities_x) > 0:
            # Add a simple line element for velocity vectors in legend
            legend_elements.append(plt.Line2D([0], [0], color='black', alpha=0.7, linewidth=2, label='Vectores de Velocidad'))
        
        # Add legend
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
                title='Leyenda', fontsize=12, title_fontsize=14)
        
        # Add main title and timestamp
        fig.suptitle('Simulación de Movimiento de Peatones - Frame Único', fontsize=16)
        ax.set_title(f'Tiempo: {row["time"]:.2f} (Frame {frame_index})', fontsize=14)
        
        # Add axis labels
        ax.set_xlabel('Posición X', fontsize=12)
        ax.set_ylabel('Posición Y', fontsize=12)
        
        # Ensure the output directory exists
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save the single frame
        output_path = os.path.join(self.output_dir, 'flowgraph_single_frame.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"Frame único guardado en: {output_path}")
        return output_path
