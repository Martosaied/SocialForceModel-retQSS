import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import imageio
import os

# Read the CSV file
df = pd.read_csv('./solution.csv')

print(df.head())

# Create output directory for frames if it doesn't exist
os.makedirs('frames', exist_ok=True)


# Create frames
for index, row in df.iterrows():
    # Create a new figure for each frame
    plt.figure(figsize=(10, 10))

    # Set up the plot area
    plt.xlim(0, 1)
    plt.ylim(0, 1)

    # Add grid lines
    for i in range(11):  # 11 lines to create 10 divisions
        plt.axhline(y=i/7, color='gray', linestyle='-', alpha=0.3)
        plt.axvline(x=i/7, color='gray', linestyle='-', alpha=0.3)

    # Plot each particle
    for i in range(1, 40):  # 10 particles
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

        plt.scatter(x, y, c=color, s=100)

    # Add title with time
    plt.title(f'Time: {row["time"]:.2f}')

    # Save the frame
    plt.savefig(f'frames/frame_{index:04d}.png')
    plt.close()

# Create GIF from frames
frames = []
frame_files = sorted([f for f in os.listdir('frames') if f.endswith('.png')])

for frame_file in frame_files:
    frames.append(imageio.imread(f'frames/{frame_file}'))

# Save as GIF
imageio.mimsave('particle_simulation.gif', frames, fps=4)

# Clean up frames directory
for frame_file in frame_files:
    os.remove(f'frames/{frame_file}')
os.rmdir('frames')

print("Animation saved as particle_simulation.gif")
