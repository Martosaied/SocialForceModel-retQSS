import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import imageio
import os
import pandas as pd
from math import sqrt
from itertools import groupby, product
from utils import parse_walls

def generate_gif(solution_file, output_dir, parameters):
    walls = parameters['WALLS']
    walls = parse_walls(walls)

    frames_dir = os.path.join(output_dir, 'frames')
    # Read the CSV file
    df = pd.read_csv(solution_file)

    # Create output directory for frames if it doesn't exist
    os.makedirs(frames_dir, exist_ok=True)

    # Create frames
    prev_row = None
    for index, row in df.iterrows():
        # Create a new figure for each frame
        plt.figure(figsize=(20, 20))

        # Set up the plot area
        plt.xlim(0, parameters['GRID_SIZE'])
        plt.ylim(0, parameters['GRID_SIZE'])

        # Add grid lines
        for i in range(21):  # 21 lines to create 20 divisions
            plt.axhline(y=i, color='gray', linestyle='-', alpha=0.3)
            plt.axvline(x=i, color='gray', linestyle='-', alpha=0.3)

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
        
        for i in range(1, 1000):  # 300 particles
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
                color = 'green'
            else:  # Left
                color = 'blue'

            frame_positions_x.append(x)
            frame_positions_y.append(y)
            frame_positions_color.append(color)
            frame_velocities_x.append(vx)
            frame_velocities_y.append(vy)

        # Plot scatter points
        scatter = plt.scatter(frame_positions_x, frame_positions_y, c=frame_positions_color, s=300)
        
        # Create legend elements
        legend_elements = [
            plt.scatter([], [], c='blue', s=300, label='Left'),
            plt.scatter([], [], c='green', s=300, label='Right')
        ]
        
        # Add legend
        plt.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
                  title='Particles', fontsize=14, title_fontsize=16)
        
        # Add velocity vectors with quiver only if we have velocity data
        if prev_row is not None:
            # Scale factor for velocity vectors (adjust this value to make arrows more visible)
            scale = 0.5  # Reduced scale since velocities might be larger with position differences
            plt.quiver(frame_positions_x, frame_positions_y, 
                      np.array(frame_velocities_x) * scale, 
                      np.array(frame_velocities_y) * scale,
                      color='black', alpha=0.5, width=0.003)

        # Add main title and timestamp
        plt.suptitle('Pedestrian Movement Simulation', fontsize=20)
        plt.title(f'Time: {row["time"]:.2f}', fontsize=16)
        
        # Save the frame
        plt.savefig(os.path.join(frames_dir, f'frame_{index:04d}.png'))
        plt.close()
        
        # Update previous row for next iteration
        prev_row = row.copy()

    # Create GIF from frames
    frames = []
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])

    for frame_file in frame_files:
        frames.append(imageio.imread(os.path.join(frames_dir, frame_file)))

    # Save as GIF
    imageio.mimsave(os.path.join(output_dir, 'particle_simulation.gif'), frames, fps=15)

    # Clean up frames directory
    for frame_file in frame_files:
        os.remove(os.path.join(frames_dir, frame_file))
    os.rmdir(frames_dir)

MIN_X_DISTANCE = 3
MIN_Y_DISTANCE = 0.5

def get_related(pair, row):
    return sqrt(
        pow(row[f'PX[{pair[0]}]'] - row[f'PX[{pair[1]}]'], 2) + pow(row[f'PY[{pair[0]}]'] - row[f'PY[{pair[1]}]'], 2)
    ) < MIN_DISTANCE

def calculate_groups(row, particles):    
    def distance(tup1, tup2):
        return abs(tup1[1] - tup2[1]) < MIN_X_DISTANCE and abs(tup1[2] - tup2[2]) < MIN_Y_DISTANCE
    
    # initializing list
    particles_list = [(i, row[f'PX[{i}]'], row[f'PY[{i}]'], row[f'PS[{i}]']) for i in range(1, particles)]
            
    # Group Adjacent Coordinates
    # Using product() + groupby() + list comprehension
    man_tups = [sorted(sub) for sub in product(particles_list, repeat = 2) if distance(*sub) and sub[0][3] == sub[1][3]]
    
    res_dict = {ele: {ele} for ele in particles_list}
    for tup1, tup2 in man_tups:
        res_dict[tup1] |= res_dict[tup2]
        res_dict[tup2] = res_dict[tup1]
    
    res = [[*next(val)] for key, val in groupby(
            sorted(res_dict.values(), key = id), id)]

    particles_groups = map(lambda x: set([y[0] for y in x]), res)

    # Remove duplicates
    unique_groups = []
    for group in particles_groups:
        if group not in unique_groups:
            unique_groups.append(group)
    
    return unique_groups


def generate_grouped_directioned_graph(solution_file, output_dir):
    df = pd.read_csv(solution_file)

    particles = (len(df.columns) - 1) / 5
    groups_per_time = {}
    for index, row in df.iterrows():
        if index % 20 != 0:
            continue

        groups = calculate_groups(row, int(particles))
        groups_per_time[row['time']] = len(groups)

        # Graphic representation of the group position
        # Create a new figure for each frame
        plt.figure(figsize=(20, 20))

        # Set up the plot area
        plt.xlim(0, 20)
        plt.ylim(0, 20)

        # Add grid lines
        for i in range(21):  # 21 lines to create 20 divisions
            plt.axhline(y=i, color='gray', linestyle='-', alpha=0.3)
            plt.axvline(x=i, color='gray', linestyle='-', alpha=0.3)

        for group in groups:
            # Create a random color for the group
            color = np.random.rand(3,)
            
            # Arrays to store positions and velocities for quiver
            positions_x = []
            positions_y = []
            velocities_x = []
            velocities_y = []
            
            for particle in group:
                x = row[f'PX[{particle}]']
                y = row[f'PY[{particle}]']
                
                # Calculate velocities if we have a previous frame
                vx = 0
                vy = 0
                if df.iloc[index - 1] is not None:
                    dt = row['time'] - df.iloc[index - 1]['time']
                    if dt > 0:  # Avoid division by zero
                        prev_x = df.iloc[index - 1][f'PX[{particle}]']
                        prev_y = df.iloc[index - 1][f'PY[{particle}]']
                        vx = (x - prev_x) / dt
                        vy = (y - prev_y) / dt
                
                positions_x.append(x)
                positions_y.append(y)
                velocities_x.append(vx)
                velocities_y.append(vy)
                
                plt.scatter(x, y, color=color, s=300)
            
            # Add velocity vectors with quiver if we have velocity data
            if df.iloc[index - 1] is not None and positions_x:
                # Scale factor for velocity vectors
                scale = 0.5
                plt.quiver(positions_x, positions_y, 
                          np.array(velocities_x) * scale, 
                          np.array(velocities_y) * scale,
                          color='black', alpha=0.5, width=0.003)
        
        plt.title(f'Time: {row["time"]:.2f}')
        plt.savefig(f'{output_dir}/group_{index}.png')
        plt.close()

    # Create a new figure for the linear graph with title and x and y labels
    plt.figure(figsize=(20, 20))
    plt.plot(list(groups_per_time.keys()), list(groups_per_time.values()))
    plt.title('Number of groups per time')
    plt.xlabel('Time')
    plt.ylabel('Number of groups')
    plt.savefig(f'{output_dir}/linear_graph.png')
    plt.close()

