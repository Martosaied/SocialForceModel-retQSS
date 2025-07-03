import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
from src.math.Density import Density

class DensityRowGraph:
    """
    Plot a density row graph of the simulation.
    """
    def __init__(self, solution_file, output_dir):
        self.solution_file = solution_file
        self.output_dir = output_dir

    def plot(self, title):
        df = pd.read_csv(self.solution_file)
        particles = (len(df.columns) - 1) / 5
        right_pedestrian_counts, left_pedestrian_counts = Density(grid_size=100).calculate_pedestrian_counts(df, particles)
        
        # Compute average density per row for right-moving pedestrians
        right_row_avg = np.mean(right_pedestrian_counts, axis=1)
        # Compute average density per row for left-moving pedestrians
        left_row_avg = np.mean(left_pedestrian_counts, axis=1)
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))
        
        # Plot average density per row for right-moving pedestrians
        ax1.plot(range(len(right_row_avg)), right_row_avg, marker='o')
        ax1.set_title('Average Density per Row (Right-Moving)', fontsize=16)
        ax1.set_xlabel('Row', fontsize=14)
        ax1.set_ylabel('Average Density', fontsize=14)
        ax1.grid(True)
        
        # Plot average density per row for left-moving pedestrians
        ax2.plot(range(len(left_row_avg)), left_row_avg, marker='o')
        ax2.set_title('Average Density per Row (Left-Moving)', fontsize=16)
        ax2.set_xlabel('Row', fontsize=14)
        ax2.set_ylabel('Average Density', fontsize=14)
        ax2.grid(True)
        
        # # Detect local maximums for right-moving pedestrians
        # right_local_max = []
        # right_mean = np.mean(right_row_avg)
        # right_std = np.std(right_row_avg)
        # right_threshold = right_mean + 2 * right_std  # Threshold for outliers
        # for i in range(1, len(right_row_avg) - 1):
        #     if right_row_avg[i] > right_row_avg[i - 1] and right_row_avg[i] > right_row_avg[i + 1] and right_row_avg[i] > right_threshold:
        #         right_local_max.append(i)
        
        # # Detect local maximums for left-moving pedestrians
        # left_local_max = []
        # left_mean = np.mean(left_row_avg)
        # left_std = np.std(left_row_avg)
        # left_threshold = left_mean + 2 * left_std  # Threshold for outliers
        # for i in range(1, len(left_row_avg) - 1):
        #     if left_row_avg[i] > left_row_avg[i - 1] and left_row_avg[i] > left_row_avg[i + 1] and left_row_avg[i] > left_threshold:
        #         left_local_max.append(i)
        
        # # Print the rows corresponding to local maximums
        # print("Local maximums for right-moving pedestrians:", right_local_max)
        # print("Local maximums for left-moving pedestrians:", left_local_max)

        lanes = Density(grid_size=100).calculate_lanes_by_density(df, particles)
        
        # Adjust layout and save
        plt.suptitle(f'{title} - Average Density per Row by Direction', fontsize=20, y=1.05)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f'{title}_row_density_graphs.png'), dpi=300, bbox_inches='tight')
        plt.close()
        