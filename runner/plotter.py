import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import imageio
import os
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
        plt.xlim(0, 20)
        plt.ylim(0, 20)

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
        
        for i in range(1, 300):  # 300 particles
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
                  title='Particles', fontsize=12, title_fontsize=14)
        
        # Add velocity vectors with quiver only if we have velocity data
        if prev_row is not None:
            # Scale factor for velocity vectors (adjust this value to make arrows more visible)
            scale = 0.5  # Reduced scale since velocities might be larger with position differences
            plt.quiver(frame_positions_x, frame_positions_y, 
                      np.array(frame_velocities_x) * scale, 
                      np.array(frame_velocities_y) * scale,
                      color='black', alpha=0.5, width=0.003)

        # Add main title and timestamp
        plt.title('Pedestrian Movement Simulation', fontsize=16)
        plt.subtitle(f'Time: {row["time"]:.2f}')
        
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


