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
        N = self.parameters.get('N', 300)

        frames_dir = os.path.join(self.output_dir, 'frames')
        df = pd.read_csv(self.solution_file)

        # Create output directory for frames if it doesn't exist
        os.makedirs(frames_dir, exist_ok=True)

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

        # Pre-calculate particle column names for vectorized access
        px_cols = [f'PX[{i}]' for i in range(1, N + 1)]
        py_cols = [f'PY[{i}]' for i in range(1, N + 1)]
        ps_cols = [f'PS[{i}]' for i in range(1, N + 1)]
        vc_cols = [f'VC[{i}]' for i in range(1, VOLUMES_COUNT * VOLUMES_COUNT + 1)]
        
        # Pre-calculate state colors for faster lookup
        state_colors_array = [state_color[state_id[i]] for i in range(7)]

        # Filter rows to process (every 3rd row)
        rows_to_process = df.iloc[::3].copy()
        
        # Pre-calculate all particle data for vectorized operations
        particle_data = {}
        for idx, row_idx in enumerate(rows_to_process.index):
            row = rows_to_process.loc[row_idx]
            
            # Get all particle positions and states at once
            positions_x = [row.get(col, np.nan) for col in px_cols]
            positions_y = [row.get(col, np.nan) for col in py_cols]
            states = [row.get(col, 0) for col in ps_cols]
            
            # Filter out particles with NaN positions
            valid_mask = ~(np.isnan(positions_x) | np.isnan(positions_y))
            
            particle_data[idx] = {
                'positions_x': np.array(positions_x)[valid_mask],
                'positions_y': np.array(positions_y)[valid_mask],
                'states': np.array(states)[valid_mask],
                'colors': [state_colors_array[int(s)] for s in np.array(states)[valid_mask]],
                'vc_values': [row.get(col, 0) for col in vc_cols],
                'time': row['time'],
                'infected': row['infectionsCount'],
                'recovered': row['recoveredCount'],
                'velocities': None  # Initialize velocities as None for all frames
            }

        # Pre-calculate velocities for all frames
        for idx in range(1, len(particle_data)):
            prev_data = particle_data[idx - 1]
            curr_data = particle_data[idx]
            
            dt = curr_data['time'] - prev_data['time']
            if dt > 0:
                # Calculate velocities for particles that exist in both frames
                min_particles = min(len(prev_data['positions_x']), len(curr_data['positions_x']))
                if min_particles > 0:
                    vx = (curr_data['positions_x'][:min_particles] - prev_data['positions_x'][:min_particles]) / dt
                    vy = (curr_data['positions_y'][:min_particles] - prev_data['positions_y'][:min_particles]) / dt
                    
                    # Calculate normalized velocities efficiently
                    lengths = np.sqrt(vx**2 + vy**2)
                    valid_velocities = lengths > 0
                    
                    normalized_vx = np.zeros_like(vx)
                    normalized_vy = np.zeros_like(vy)
                    if valid_velocities.any():
                        # Use boolean indexing correctly
                        valid_indices = np.where(valid_velocities)[0]
                        normalized_vx[valid_indices] = vx[valid_indices] / lengths[valid_indices]
                        normalized_vy[valid_indices] = vy[valid_indices] / lengths[valid_indices]
                    
                    particle_data[idx]['velocities'] = {
                        'vx': vx,
                        'vy': vy,
                        'normalized_vx': normalized_vx,
                        'normalized_vy': normalized_vy,
                        'valid': valid_velocities
                    }
                else:
                    particle_data[idx]['velocities'] = None
            else:
                particle_data[idx]['velocities'] = None

        # Create frames with optimized plotting
        for idx, (frame_idx, data) in enumerate(particle_data.items()):
            # Create a new figure for each frame
            fig, ax = plt.subplots(figsize=(20, 20))

            # Set up the plot area
            ax.set_xlim(0, GRID_SIZE)
            ax.set_ylim(0, GRID_SIZE)

            # Add grid lines efficiently
            for x in grid_lines_x:
                ax.axvline(x=x, color='gray', linestyle='-', alpha=0.3)
            for y in grid_lines_y:
                ax.axhline(y=y, color='gray', linestyle='-', alpha=0.3)
            
            # Color each cell based on VC[ID] values efficiently
            for i, (pos, cell_id) in enumerate(zip(cell_positions, cell_ids)):
                vc_value = data['vc_values'][cell_id - 1]  # cell_id is 1-indexed
                
                if vc_value > 0:
                    rect = plt.Rectangle(
                        pos, 
                        CELL_SIZE, 
                        CELL_SIZE, 
                        facecolor='red', 
                        alpha=min(1, vc_value),
                        edgecolor='none'
                    )
                    ax.add_patch(rect)

            # Plot wall segments
            for wall in walls:
                ax.plot(
                    [wall['from_x'], wall['to_x']], 
                    [wall['from_y'], wall['to_y']], 
                    'k-', linewidth=3, label='_nolegend_'
                )

            # Plot particles efficiently
            if len(data['positions_x']) > 0:
                scatter = ax.scatter(data['positions_x'], data['positions_y'], 
                                   c=data['colors'], s=100)
                
                # Add velocity vectors if available
                if data['velocities'] is not None and data['velocities']['valid'].any():
                    valid_mask = data['velocities']['valid']
                    valid_indices = np.where(valid_mask)[0]
                    ax.quiver(data['positions_x'][valid_indices], 
                            data['positions_y'][valid_indices],
                            data['velocities']['normalized_vx'][valid_indices],
                            data['velocities']['normalized_vy'][valid_indices],
                            color='black', alpha=0.5, width=0.003)

            # Create legend elements
            legend_elements = [
                plt.scatter([], [], c=state_color[state_id[i]], s=300, label=state_id[i]) for i in range(7)
            ]
            
            # Add legend
            plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
                    title='Particles', fontsize=14, title_fontsize=16)

            # Add main title and timestamp
            fig.suptitle('Subway Simulation', fontsize=20)
            ax.set_title(f'Time: {data["time"]:.2f} Infected: {data["infected"]} Recovered: {data["recovered"]}', fontsize=16)
            
            # Save the frame
            plt.savefig(os.path.join(frames_dir, f'frame_{frame_idx:04d}.png'), dpi=50)
            plt.close(fig)

        # Create GIF from frames efficiently
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
        frames = [imageio.imread(os.path.join(frames_dir, frame_file)) for frame_file in frame_files]

        # Save as GIF
        imageio.mimsave(os.path.join(self.output_dir, 'particle_simulation.gif'), frames, fps=15)

        # Clean up frames directory
        for frame_file in frame_files:
            os.remove(os.path.join(frames_dir, frame_file))
        os.rmdir(frames_dir)