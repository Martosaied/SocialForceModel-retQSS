import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import imageio
from math import sqrt
from src.utils import parse_walls, process_parameters, get_parameter_combinations

class FlowGraph:
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

    def plot(self):
        walls = self.parameters.get('WALLS', [])
        walls = parse_walls(walls)
        
        # Get grid parameters - use the same approach as FlowGraphInfections
        GRID_SIZE = self.parameters.get('GRID_SIZE', 50)
        # Determine VOLUMES_COUNT from GRID_DIVISIONS or default to 20
        VOLUMES_COUNT = 10
        CELL_SIZE = GRID_SIZE / VOLUMES_COUNT
        N = self.parameters.get('N', 300)
        
        # Get volume type mapping
        volume_type_map = self._get_volume_type_map(VOLUMES_COUNT)
        
        # Define colors for different volume types
        volume_colors = {
            'obstacle': 'black',
            'hallway': 'lightgray', 
            'classroom': 'lightblue'
        }

        frames_dir = os.path.join(self.output_dir, 'frames')
        df = pd.read_csv(self.solution_file)

        # Create output directory for frames if it doesn't exist
        os.makedirs(frames_dir, exist_ok=True)
        
        # Pre-calculate cell positions and IDs for efficiency (same as FlowGraphInfections)
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

        # Create frames
        prev_row = None
        for index, row in df.iterrows():
            if index % 10 != 0:
                continue

            # Create a new figure for each frame with better proportions
            fig, ax = plt.subplots(figsize=(16, 12))

            # Set up the plot area
            ax.set_xlim(0, GRID_SIZE)
            ax.set_ylim(0, GRID_SIZE)

            # Add grid lines efficiently - ensure they align with cell boundaries
            for x in grid_lines_x:
                ax.axvline(x=x, color='lightgray', linestyle='-', alpha=0.4, linewidth=0.8)
            for y in grid_lines_y:
                ax.axhline(y=y, color='lightgray', linestyle='-', alpha=0.4, linewidth=0.8)
            
            # Color each cell based on volume type (similar to FlowGraphInfections VC coloring)
            for i, (pos, cell_id) in enumerate(zip(cell_positions, cell_ids)):
                if cell_id in volume_type_map:
                    volume_type = volume_type_map[cell_id]
                    color = volume_colors[volume_type]
                    alpha = 0.7 if volume_type == 'obstacle' else 0.5
                    
                    rect = plt.Rectangle(
                        pos, 
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
                    [wall['from_y'], wall['to_y']], 
                    'k-', linewidth=4, label='_nolegend_'
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
                    color = '#FF4444'  # Brighter red
                else:  # Left
                    color = '#4444FF'  # Brighter blue

                frame_positions_x.append(x)
                frame_positions_y.append(y)
                frame_positions_color.append(color)
                frame_velocities_x.append(vx)
                frame_velocities_y.append(vy)

            # Plot scatter points
            if len(frame_positions_x) > 0:
                scatter = ax.scatter(frame_positions_x, frame_positions_y, c=frame_positions_color, s=200)
            
            # Create legend elements with updated colors
            legend_elements = [
                plt.scatter([], [], c='#4444FF', s=200, label='Left'),
                plt.scatter([], [], c='#FF4444', s=200, label='Right'),
                plt.Rectangle((0, 0), 1, 1, facecolor=volume_colors['obstacle'], alpha=0.7, label='Obstacles') if self.parameters.get('OBSTACLES', '') else None,
                plt.Rectangle((0, 0), 1, 1, facecolor=volume_colors['hallway'], alpha=0.5, label='Hallways') if self.parameters.get('HALLWAYS', '') else None,
                plt.Rectangle((0, 0), 1, 1, facecolor=volume_colors['classroom'], alpha=0.5, label='Classrooms') if self.parameters.get('CLASSROOMS', '') else None,
            ]
            
            # Add legend with better styling
            plt.legend(handles=[elem for elem in legend_elements if elem is not None], 
                      loc='center left', bbox_to_anchor=(1, 0.5),
                      title='Legend', fontsize=16, title_fontsize=18,
                      frameon=True, fancybox=True, shadow=True)
            
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
                        color='black', alpha=0.5, width=0.003)

            # Add main title and timestamp with better styling
            fig.suptitle('Pedestrian Movement Simulation', fontsize=24, fontweight='bold', y=0.95)
            ax.set_title(f'Time: {row["time"]:.2f} seconds', fontsize=18, fontweight='bold', pad=20)
            
            # Add axis labels
            ax.set_xlabel('X Position', fontsize=16, fontweight='bold')
            ax.set_ylabel('Y Position', fontsize=16, fontweight='bold')
            
            # Improve tick labels
            ax.tick_params(axis='both', which='major', labelsize=12)
            
            # Save the frame with optimized size
            plt.savefig(os.path.join(frames_dir, f'frame_{index:04d}.png'), 
                       dpi=60, bbox_inches='tight', facecolor='white', edgecolor='none')
            plt.close(fig)
            
            # Update previous row for next iteration
            prev_row = row.copy()

        # Create GIF from frames
        frames = []
        frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])

        for frame_file in frame_files:
            frames.append(imageio.imread(os.path.join(frames_dir, frame_file)))

        # Save as GIF with optimized size
        imageio.mimsave(os.path.join(self.output_dir, 'particle_simulation.gif'), frames, fps=15)

        # Clean up frames directory
        for frame_file in frame_files:
            os.remove(os.path.join(frames_dir, frame_file))
        os.rmdir(frames_dir)