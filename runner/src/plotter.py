import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import imageio
import os
import pandas as pd
from math import sqrt
from itertools import groupby, product
from src.utils import parse_walls

def generate_gif(solution_file, output_dir, parameters):
    walls = parameters.get('WALLS', '')
    walls = parse_walls(walls)

    # obstacles = parameters.get('OBSTACLES', '')
    # obstacles = parse_obstacles(obstacles)

    frames_dir = os.path.join(output_dir, 'frames')
    # Read the CSV file
    df = pd.read_csv(solution_file)

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
        plt.xlim(0, parameters.get('GRID_SIZE', 20))
        plt.ylim(0, parameters.get('GRID_SIZE', 20))

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
        
        for i in range(1, parameters.get('N', 300)):  # 300 particles
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

MIN_X_DISTANCE = 25
MIN_Y_DISTANCE = 0.5

def distance(tup1, tup2):
    return abs(tup1[1] - tup2[1]) < MIN_X_DISTANCE and abs(tup1[2] - tup2[2]) < MIN_Y_DISTANCE

def unify_groups(particles_groups):
    # Remove duplicates
    unique_groups = {}
    for group in particles_groups:
        sorted_group = tuple(sorted(group))
        hash_id = str(hash(sorted_group))
        unique_groups[hash_id] = set(sorted_group)

    return list(unique_groups.values())

def join_groups(particles_groups):
    for group in particles_groups:
        for other_group in particles_groups:
            if len(set(group) & set(other_group)) > 0:
                group.update(other_group)

    return particles_groups

def direct_group(particle, particles_list):
    particle_group = set([particle[0]])
    for other_particle in particles_list:
        if distance(particle, other_particle) and particle[3] == other_particle[3]:
            particle_group.add(other_particle[0])

    return particle_group

def calculate_groups(row, particles):    
    # initializing list
    particles_list = [(i, row[f'PX[{i}]'], row[f'PY[{i}]'], row[f'PS[{i}]']) for i in range(1, particles)]

    particles_groups = []
    for particle in particles_list:
        particle_group = direct_group(particle, particles_list)
        particles_groups.append(particle_group)


    previous_group_quantity = 0
    while len(particles_groups) != previous_group_quantity:
        previous_group_quantity = len(particles_groups)
        particles_groups = join_groups(particles_groups)
        particles_groups = unify_groups(particles_groups)

    return particles_groups

def generate_grouped_directioned_graph(results, output_dir):
    groups_per_time = {}
    for result_file in results:
        df = pd.read_csv(result_file)

        particles = (len(df.columns) - 1) / 5
        for index, row in df.iterrows():
            if index % 5 != 0:
                continue

            groups = calculate_groups(row, int(particles))
            if row['time'] not in groups_per_time:
                groups_per_time[row['time']] = [len(groups)]
            else:
                groups_per_time[row['time']].append(len(groups))

            # Graphic representation of the group position
            # Create a new figure for each frame
            plt.figure(figsize=(20, 20))

            # Set up the plot area
            plt.xlim(0, 50)
            plt.ylim(0, 50)

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
            
            plt.suptitle('Grouped Lanes', fontsize=20)
            plt.title(f'Time: {row["time"]:.2f}', fontsize=16)
            plt.legend()
            plt.xlabel('X', fontsize=16)
            plt.ylabel('Y', fontsize=16)

            plt.savefig(f'{output_dir}/group_{index}.png')
            plt.close()

    # Create a new figure for the linear graph with title and x and y labels
    # Show the mean of the groups per time and its standard deviation
    plt.figure(figsize=(20, 20))
    mean_groups_per_time = {k: np.mean(v) for k, v in groups_per_time.items()}
    std_groups_per_time = {k: np.std(v) for k, v in groups_per_time.items()}
    plt.plot(list(mean_groups_per_time.keys()), list(mean_groups_per_time.values()))
    plt.fill_between(
        list(mean_groups_per_time.keys()), 
        (np.array(list(mean_groups_per_time.values())) - np.array(list(std_groups_per_time.values()))), 
        (np.array(list(mean_groups_per_time.values())) + np.array(list(std_groups_per_time.values()))), 
        alpha=0.2
    )
    plt.title('Number of groups per time')
    plt.xlabel('Time')
    plt.ylabel('Number of groups')
    plt.savefig(f'{output_dir}/linear_graph.png')
    plt.close()

def benchmark_graph():
    results = {
        'N': [100, 200, 400, 800, 1600],
        'mmoc': [7.09, 17.46, 48.89, 156.24, 551.06],
        'retqss': [2.77, 5.06, 10.98, 25.70, 67.70]
    }

    description = """
    The simulations were made using the same parameters for both models. 
    Same ones used by Helbing and Molnar 1995. No obstacles were used.
    The number of volumes for the retqss implementation was 100.
    """

    plt.figure(figsize=(10, 10))
    plt.plot(results['N'], results['mmoc'], label='mmoc only')
    plt.plot(results['N'], results['retqss'], label='mmoc+retqss')
    plt.title('Comparacion de tiempos de ejecucion*')
    plt.xlabel('Numero de peatones en la simulacion')
    plt.ylabel('Tiempo(s)')
    plt.legend()
    plt.savefig(f'benchmark_graph.png')
    plt.close()

benchmark_graph()