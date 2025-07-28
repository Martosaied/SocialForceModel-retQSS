import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import imageio
from math import sqrt
from src.utils import parse_walls, process_parameters, get_parameter_combinations

# SEIR-like model states 
state_id = {
    0: "SUSCEPTIBLE",
    1: "EXPOSED",
    2: "PRE_SYMPTOMATIC",
    3: "SYMPTOMATIC",
    4: "ASYMPTOMATIC",
    5: "RECOVERED",
    6: "DEAD"
}

state_color = {
    "SUSCEPTIBLE": "blue",
    "EXPOSED": "yellow",
    "PRE_SYMPTOMATIC": "purple",
    "SYMPTOMATIC": "red",
    "ASYMPTOMATIC": "orange",
    "RECOVERED": "green",
    "DEAD": "black"
}


class FlowGraphInfections:
    """
    Plot a flow graph of the simulation.

    The flow graph is a GIF that shows the flow of pedestrians in the simulation.

    Args:
        solution_file: The path to the solution file.
        output_dir: The path to the output directory. It will be created if it doesn't exist.
        parameters: The parameters of the simulation.

    Returns:
        None
    """

    def __init__(self, solution_file, output_dir, parameters):
        self.solution_file = solution_file
        self.output_dir = output_dir
        self.parameters = list(get_parameter_combinations(process_parameters(parameters.get('parameters', {}))))[0]

    def plot(self):
        walls = self.parameters.get('WALLS', [])
        GRID_SIZE = self.parameters.get('GRID_SIZE', 1)
        VOLUMES_COUNT = self.parameters.get('VOLUMES_COUNT', 20)
        CELL_SIZE = GRID_SIZE / VOLUMES_COUNT
        walls = parse_walls(walls)

        frames_dir = os.path.join(self.output_dir, 'frames')
        df = pd.read_csv(self.solution_file)

        # Create output directory for frames if it doesn't exist
        os.makedirs(frames_dir, exist_ok=True)

        # Create frames
        prev_row = None
        for index, row in df.iterrows():
            if index % 3 != 0:
                continue

            # Create a new figure for each frame
            plt.figure(figsize=(20, 20))

            # Set up the plot area
            plt.xlim(0, GRID_SIZE)
            plt.ylim(0, GRID_SIZE)

            # Add grid lines and color cells based on VC[ID] values
            for i in range(VOLUMES_COUNT + 1):  # 21 lines to create 20 divisions
                plt.axhline(y=CELL_SIZE * i, color='gray', linestyle='-', alpha=0.3)
                plt.axvline(x=CELL_SIZE * i, color='gray', linestyle='-', alpha=0.3)
            
            # Color each cell based on VC[ID] values
            for i in range(VOLUMES_COUNT):
                for j in range(VOLUMES_COUNT):
                    cell_id = i % VOLUMES_COUNT + VOLUMES_COUNT * j + 1
                    vc_value = row.get(f'VC[{cell_id}]', 0)
                    
                    # Color the cell based on VC value (red intensity)
                    if vc_value > 0:
                        # Create a rectangle for the cell
                        rect = plt.Rectangle(
                            (j * CELL_SIZE, i * CELL_SIZE), 
                            CELL_SIZE, 
                            CELL_SIZE, 
                            facecolor='red', 
                            alpha=min(1, vc_value*10),  # Normalize and cap alpha
                            edgecolor='none'
                        )
                        plt.gca().add_patch(rect)

            # Plot wall segments
            for wall in walls:
                plt.plot(
                    [wall['from_x'], wall['to_x']], 
                    [wall['from_y'], wall['to_y']], 
                    'k-', linewidth=3, label='_nolegend_'
                )  # 'k-' means black solid line

            # Plot each particle
            frame_positions_x = []
            frame_positions_y = []
            frame_positions_color = []
            frame_velocities_x = []
            frame_velocities_y = []
            
            # For legend
            left_scatter = None
            right_scatter = None
            
            for i in range(1, self.parameters.get('N', 300)):  # 300 particles
                if row.get(f'PX[{i}]') is None:
                    continue

                x = row[f'PX[{i}]']
                y = row[f'PY[{i}]']
                state = row[f'PS[{i}]']
                infected = row[f'infectionsCount']
                recovered = row[f'recoveredCount']

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

                color = state_color[state_id[state]]

                frame_positions_x.append(x)
                frame_positions_y.append(y)
                frame_positions_color.append(color)
                frame_velocities_x.append(vx)
                frame_velocities_y.append(vy)

            # Plot scatter points
            scatter = plt.scatter(frame_positions_x, frame_positions_y, c=frame_positions_color, s=100)
            
            # Create legend elements
            legend_elements = [
                plt.scatter([], [], c=state_color[state_id[i]], s=300, label=state_id[i]) for i in range(7)
            ]
            
            # Add legend
            plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
                    title='Particles', fontsize=14, title_fontsize=16)
            
            # Add velocity vectors with quiver only if we have velocity data
            if prev_row is not None:
                # Scale factor for velocity vectors (adjust this value to make arrows more visible)
                length_velocities = [
                    sqrt(frame_velocities_x[i] ** 2 + frame_velocities_y[i] ** 2) for i in range(len(frame_velocities_x))
                ]
                normalize_velocities_x = [
                    frame_velocities_x[i] / length_velocities[i] for i in range(len(frame_velocities_x))
                ]
                normalize_velocities_y = [
                    frame_velocities_y[i] / length_velocities[i] for i in range(len(frame_velocities_y))
                ]
                plt.quiver(frame_positions_x, frame_positions_y, 
                        np.array(normalize_velocities_x), 
                        np.array(normalize_velocities_y),
                        color='black', alpha=0.5, width=0.003)

            # Add main title and timestamp
            plt.suptitle('Subway Simulation', fontsize=20)
            plt.title(f'Time: {row["time"]:.2f} Infected: {infected} Recovered: {recovered}', fontsize=16)
            
            # Save the frame
            plt.savefig(os.path.join(frames_dir, f'frame_{index:04d}.png'), dpi=50)
            plt.close()
            
            # Update previous row for next iteration
            prev_row = row.copy()

        # Create GIF from frames
        frames = []
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])

        for frame_file in frame_files:
            frames.append(imageio.imread(os.path.join(frames_dir, frame_file)))

        # Save as GIF
        imageio.mimsave(os.path.join(self.output_dir, 'particle_simulation.gif'), frames, fps=15)

        # Clean up frames directory
        for frame_file in frame_files:
            os.remove(os.path.join(frames_dir, frame_file))
        os.rmdir(frames_dir)