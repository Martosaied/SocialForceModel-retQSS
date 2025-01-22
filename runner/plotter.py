import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os

def generate_gif(solution_file, output_dir):
    frames_dir = os.path.join(output_dir, 'frames')
    # Read the CSV file
    df = pd.read_csv(solution_file)

    # Create output directory for frames if it doesn't exist
    os.makedirs(frames_dir, exist_ok=True)

    # Create frames
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

        # Plot each particle
        frame_positions_x = []
        frame_positions_y = []
        frame_positions_color = []
        for i in range(1, 300):  # 300 particles

            x = row[f'PX[{i}]']
            y = row[f'PY[{i}]']
            state = row[f'PS[{i}]']

            # Different colors based on particle state
            if state == 4:  # Infected
                color = 'red'
            elif state == 1:  # Recovered
                color = 'green'
            else:  # Susceptible
                color = 'blue'

            frame_positions_x.append(x)
            frame_positions_y.append(y)
            frame_positions_color.append(color)

        plt.scatter(frame_positions_x, frame_positions_y, c=frame_positions_color, s=300)
        # Add title with time

        plt.title(f'Time: {row["time"]:.2f}')

        # Save the frame
        plt.savefig(os.path.join(frames_dir, f'frame_{index:04d}.png'))
        plt.close()

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


