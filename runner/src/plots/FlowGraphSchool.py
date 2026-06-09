import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import imageio
from math import sqrt
from src.utils import parse_walls, process_parameters, get_parameter_combinations

class FlowGraphSchool:
    """
    Plot a flow graph of the school simulation.

    The flow graph is a GIF that shows the flow of pedestrians in the school simulation.

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
        self.raw_parameters = parameters['parameters']
        # Number of witness particles to track (default: 1)
        self.num_witness_particles = parameters.get('num_witness_particles', 3)

    def _get_school_config_matrices(self):
        """
        Get school configuration matrices for obstacles, hallways, and classrooms.
        Returns dictionaries with the matrix data.
        """
        # Get the matrices from parameters - they are stored in the 'map' field
        obstacles_param = self.raw_parameters.get('OBSTACLES', {})['map']
        hallways_param = self.raw_parameters.get('HALLWAYS', {})['map']
        classrooms_param = self.raw_parameters.get('CLASSROOMS', {})['map']
        
        # Extract the map data
        obstacles = obstacles_param
        hallways = hallways_param
        classrooms = classrooms_param
        
        return {
            'obstacles': obstacles,
            'hallways': hallways, 
            'classrooms': classrooms
        }

    def plot(self):
        walls = self.parameters.get('WALLS', [])
        walls = parse_walls(walls)
        
        # Get grid parameters - use school scenario approach
        GRID_SIZE = self.parameters.get('GRID_SIZE', 50)
        # Get grid divisions from parameters
        GRID_DIVISIONS = len(self.raw_parameters.get('OBSTACLES', {})['map'])
        CELL_SIZE = GRID_SIZE / GRID_DIVISIONS
        N = self.parameters.get('N', 300)
        
        # Get pedestrian radius and calculate appropriate circle size
        PEDESTRIAN_R = self.parameters.get('PEDESTRIAN_R', 0.3)
        # Calculate circle size based on pedestrian radius relative to grid size
        # Scale factor to make circles visible but proportional to actual size
        circle_size = max(15, min(150, (PEDESTRIAN_R / GRID_SIZE) * 10000))
        # Testigos: clearly larger than regular pedestrians; legend matches
        witness_marker_scale = 2.45
        circle_size_witness = circle_size * witness_marker_scale
        witness_edge_lw = 3.0
        # Trajectory: thicker line + slightly higher alpha than before
        trace_lw_max = 7.0
        trace_lw_min = 3.0
        trace_lw_taper = 0.14  # newer segments stay closer to trace_lw_max
        trace_alpha_floor = 0.45
        trace_alpha_cap = 0.95

        # Get corridor parameters
        FROM_Y = self.parameters.get('FROM_Y', 0)
        TO_Y = self.parameters.get('TO_Y', GRID_SIZE)
        
        # Get school configuration matrices
        school_config = self._get_school_config_matrices()
        
        # Define colors for different cell types (same as generate_single_flowgraph)
        cell_colors = {
            'obstacle': 'black',
            'hallway': 'lightgray', 
            'classroom': 'lightblue'
        }

        frames_dir = os.path.join(self.output_dir, 'frames')
        df = pd.read_csv(self.solution_file)

        # Create output directory for frames if it doesn't exist
        os.makedirs(frames_dir, exist_ok=True)
        
        # Convert matrices to numpy arrays for easier processing (same as generate_single_flowgraph)
        obstacles = np.array(school_config['obstacles'])
        hallways = np.array(school_config['hallways'])
        classrooms = np.array(school_config['classrooms'])
        
        # Pre-calculate grid lines
        grid_lines_x = [CELL_SIZE * i for i in range(GRID_DIVISIONS + 1)]
        grid_lines_y = [CELL_SIZE * i for i in range(GRID_DIVISIONS + 1)]

        # Find active witness particles from the first frame
        witness_particle_ids = []
        witness_trajectories = {}  # Dictionary to store trajectories for each witness
        
        # Look for the first frame to find active particles
        for index, row in df.iterrows():
            if index % 10 != 0:
                continue
            # Find active particles
            active_particles = []
            for i in range(1, N):
                if row.get(f'PX[{i}]') is not None:
                    active_particles.append(i)
            
            # Select up to num_witness_particles from active particles
            import random
            if len(active_particles) > 0:
                num_to_select = min(self.num_witness_particles, len(active_particles))
                witness_particle_ids = random.sample(active_particles, num_to_select)
                break
        
        # Initialize trajectories for each witness
        for witness_id in witness_particle_ids:
            witness_trajectories[witness_id] = {'x': [], 'y': []}

        # Distinct colors per testigo (stable by witness id order)
        witness_palette = ['#E63946', '#2A9D8F', '#F4A261', '#8338EC', '#06D6A0', '#FF006E']
        witness_color_by_id = {
            wid: witness_palette[i % len(witness_palette)]
            for i, wid in enumerate(witness_particle_ids)
        }
        
        # Debug: Print witness particle info
        if len(witness_particle_ids) > 0:
            print(f"Selected {len(witness_particle_ids)} witness particles: {witness_particle_ids}")
        else:
            print("No active particles found for witness selection")
        
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
            
            if FROM_Y is not None and TO_Y is not None:
                # Highlight the corridor area out of FROM_Y and TO_Y
                corridor_rect = plt.Rectangle(
                    (0, 0), 
                    GRID_SIZE, 
                    GRID_SIZE, 
                    facecolor='#808080', 
                    alpha=0.3,
                    edgecolor='none',
                    label='_nolegend_'
                )
                corridor_rect_2 = plt.Rectangle(
                    (0, FROM_Y), 
                    GRID_SIZE, 
                    TO_Y - FROM_Y, 
                    facecolor='#FFFFFF', 
                    alpha=1,
                    edgecolor='none',
                    label='_nolegend_'
                )
                ax.add_patch(corridor_rect)
                ax.add_patch(corridor_rect_2)

            # Add grid lines efficiently - ensure they align with cell boundaries
            for x in grid_lines_x:
                ax.axvline(x=x, color='lightgray', linestyle='-', alpha=0.4, linewidth=1.2)
            for y in grid_lines_y:
                ax.axhline(y=y, color='lightgray', linestyle='-', alpha=0.4, linewidth=1.2)
            
            # Color each cell based on its type (same as generate_single_flowgraph)
            for i in range(GRID_DIVISIONS):
                for j in range(GRID_DIVISIONS):
                    x = j * CELL_SIZE
                    y = (GRID_DIVISIONS - 1 - i) * CELL_SIZE  # Flip Y axis to match matrix indexing
                    
                    # Determine cell type (priority: obstacle > classroom > hallway)
                    if obstacles[i, j] == 1:
                        cell_type = 'obstacle'
                        alpha = 0.8
                    elif classrooms[i, j] == 1:
                        cell_type = 'classroom'
                        alpha = 0.6
                    elif hallways[i, j] == 1:
                        cell_type = 'hallway'
                        alpha = 0.4
                    else:
                        cell_type = 'hallway'  # Default to hallway
                        alpha = 0.4
                    
                    if cell_type != 'empty':
                        color = cell_colors[cell_type]
                        rect = plt.Rectangle(
                            (x, y), 
                            CELL_SIZE, 
                            CELL_SIZE, 
                            facecolor=color, 
                            alpha=alpha,
                            edgecolor='black',
                            linewidth=0.5
                        )
                        ax.add_patch(rect)

            # Plot wall segments
            for wall in walls:
                ax.plot(
                    [wall['from_x'], wall['to_x']], 
                    [wall['from_y'], wall['to_y']], 
                    'k-', linewidth=6, label='_nolegend_'
                )

            # Plot each particle
            frame_positions_x = []
            frame_positions_y = []
            frame_positions_color = []
            frame_velocities_x = []
            frame_velocities_y = []
            
            # Store witness particles for this frame
            witness_particles = {}
            
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

                # Check if this is a witness particle
                if i in witness_particle_ids:
                    witness_particles[i] = {'x': x, 'y': y}
                    # Add to trajectory
                    witness_trajectories[i]['x'].append(x)
                    witness_trajectories[i]['y'].append(y)
                    # Skip adding to regular particles - witness is plotted separately
                    continue
                else:
                    if state == 1:  # Right
                        color = '#FF4444'  # Brighter red
                    else:  # Left
                        color = '#4444FF'  # Brighter blue

                frame_positions_x.append(x)
                frame_positions_y.append(y)
                frame_positions_color.append(color)
                frame_velocities_x.append(vx)
                frame_velocities_y.append(vy)

            # Plot all scatter points (including witness)
            if len(frame_positions_x) > 0:
                scatter = ax.scatter(frame_positions_x, frame_positions_y, c=frame_positions_color, s=circle_size)
            
            # Plot witness particles separately with larger size and border for emphasis
            for witness_id, witness_data in witness_particles.items():
                color = witness_color_by_id[witness_id]
                ax.scatter(witness_data['x'], witness_data['y'], c=color, s=circle_size_witness, 
                          edgecolors='black', linewidth=witness_edge_lw, zorder=10)
                # Debug: Print witness position
                if index % 100 == 0:  # Print every 10th frame
                    print(f"Frame {index}: Witness {witness_id} at ({witness_data['x']:.2f}, {witness_data['y']:.2f})")
            
            # Draw witness particle trajectories (only up to current position)
            for witness_id, trajectory in witness_trajectories.items():
                if len(trajectory['x']) > 1:
                    color = witness_color_by_id[witness_id]
                    nseg = len(trajectory['x'])
                    # Gradient: older segments thinner/more transparent, recent part bold
                    for j in range(1, nseg):
                        t = j / max(1, nseg - 1)
                        alpha = min(trace_alpha_cap, trace_alpha_floor + t * (trace_alpha_cap - trace_alpha_floor))
                        # Recent segments (large j) get linewidth near trace_lw_max
                        linewidth = max(
                            trace_lw_min,
                            trace_lw_max - (nseg - j) * trace_lw_taper,
                        )
                        ax.plot(trajectory['x'][j-1:j+1], trajectory['y'][j-1:j+1], 
                               color=color, linewidth=linewidth, alpha=alpha, zorder=5,
                               label='_nolegend_', solid_capstyle='round', solid_joinstyle='round')
            
            # Create legend elements with updated colors (same as generate_single_flowgraph)
            legend_elements = [
                plt.scatter([], [], c='#4444FF', s=circle_size, label='Personas'),
            ]
            
            # Add witness particles to legend
            for i, wid in enumerate(witness_particle_ids):
                color = witness_color_by_id[wid]
                label = f'Testigo {i+1}' if len(witness_particle_ids) > 1 else 'Testigo'
                legend_elements.append(plt.scatter([], [], c=color, s=circle_size_witness, label=label))
            
            # Add other legend elements
            legend_elements.extend([
                plt.Rectangle((0, 0), 1, 1, facecolor=cell_colors['obstacle'], alpha=0.8, label='Obstáculos'),
                plt.Rectangle((0, 0), 1, 1, facecolor=cell_colors['classroom'], alpha=0.6, label='Aulas'),
                plt.Rectangle((0, 0), 1, 1, facecolor=cell_colors['hallway'], alpha=0.4, label='Pasillos'),
            ])
            
            # Add legend with better styling
            plt.legend(handles=[elem for elem in legend_elements if elem is not None], 
                      loc='center left', bbox_to_anchor=(1, 0.5),
                      title='Leyenda', fontsize=16, title_fontsize=18,
                      frameon=True, fancybox=True, shadow=True)
            
            # Add axis labels with units
            ax.set_xlabel('Posición X (metros)', fontsize=14, fontweight='bold')
            ax.set_ylabel('Posición Y (metros)', fontsize=14, fontweight='bold')
            
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