import numpy as np
import scipy.stats as stats

class Density:
    """
    This class contains the functions to calculate the density of the pedestrians.
    """
    def __init__(self, grid_size=20, map_size=50):
        self.grid_size = grid_size
        self.map_size = map_size

    def normalize_density(self, density_array):
        """
        Normalize the density array to the range [0, 1].
        """
        if np.max(density_array) == 0:
            return density_array
        return density_array / np.max(density_array)

    def calculate_pedestrian_counts(self, df, particles):
        # Create 50x50 grids to store pedestrian counts for each state
        right_pedestrian_counts = np.zeros((self.grid_size, self.grid_size))
        left_pedestrian_counts = np.zeros((self.grid_size, self.grid_size))
        
        # Process each time step
        for index, row in df.iterrows():

            # Reset grids for this time step
            current_right_counts = np.zeros((self.grid_size, self.grid_size))
            current_left_counts = np.zeros((self.grid_size, self.grid_size))
            
            # Count pedestrians in each cell
            for i in range(1, int(particles)):
                if row.get(f'PX[{i}]') is None:
                    continue
                    
                x = row[f'PX[{i}]']
                y = row[f'PY[{i}]']
                state = row[f'PS[{i}]']
                
                # Convert the coordinates to grid indices using the map size
                grid_x = min(int(x * self.grid_size / self.map_size), self.grid_size - 1)
                grid_y = min(int(y * self.grid_size / self.map_size), self.grid_size - 1)

                if grid_x < 0 or grid_y < 0 or grid_x >= self.grid_size or grid_y >= self.grid_size:
                    continue
                
                # Add to appropriate grid based on state
                if state == 1:  # Right
                    current_right_counts[grid_y, grid_x] += 1
                else:  # Left
                    current_left_counts[grid_y, grid_x] += 1
            
            # Add to total counts
            right_pedestrian_counts += current_right_counts
            left_pedestrian_counts += current_left_counts

        # Normalize the density arrays
        right_pedestrian_counts = self.normalize_density(right_pedestrian_counts)
        left_pedestrian_counts = self.normalize_density(left_pedestrian_counts)

        return right_pedestrian_counts, left_pedestrian_counts

    def calculate_zscore(self, density_array):
        """
        Calculate the z-score for each element in the density array.
        Z-score = (x - mean) / standard_deviation
        """
        mean = np.mean(density_array)
        std = np.std(density_array)
        if std == 0:
            return np.zeros_like(density_array)
        return (density_array - mean) / std

    def outlier_cimbala(self, density_array):
        def iteration(density_array):
            mean = np.mean(density_array)
            std = np.std(density_array)

            # Calculate the absolute value of deviation from the mean
            deviation = np.abs(density_array - mean)
            
            # We focus on the row with the max deviation
            max_deviation = np.max(deviation)
            max_deviation_index = np.argmax(deviation)

            # Calculate the t-student distribution with alpha = 0.05 and degrees of freedom = len(density_array) - 2
            t_student = stats.t.ppf(0.99, len(density_array) - 2)

            # Calculate tau
            tau = t_student * (len(density_array) - 1) / (np.sqrt(len(density_array)) * np.sqrt(len(density_array) - 2 + t_student**2))

            if max_deviation > tau * std:
                return max_deviation_index
            else:
                return None
        
        outliers = []
        while True:
            index = iteration(density_array)
            if index is None:
                break
            density_array = np.delete(density_array, index)
            outliers.append(index)

        return outliers


    def calculate_lanes_by_density(self, df, particles):
        right_pedestrian_counts, left_pedestrian_counts = self.calculate_pedestrian_counts(df, particles)
        
        # Compute average density per row for right-moving pedestrians
        right_row_avg = np.mean(right_pedestrian_counts, axis=1)
        # Compute average density per row for left-moving pedestrians
        left_row_avg = np.mean(left_pedestrian_counts, axis=1)

        right_local_max = []
        left_local_max = []
        outliers = []
        
        # Detect local maximums for right-moving pedestrians using z-score
        for i in range(1, len(right_row_avg) - 1):
            if (right_row_avg[i] > right_row_avg[i - 1] and right_row_avg[i] > right_row_avg[i + 1]):
                right_local_max.append(right_row_avg[i])
                right_row_avg[i] = -1

        # Detect local maximums for left-moving pedestrians using z-score
        for i in range(1, len(left_row_avg) - 1):
            if (left_row_avg[i] > left_row_avg[i - 1] and left_row_avg[i] > left_row_avg[i + 1]):
                left_local_max.append(left_row_avg[i])
                left_row_avg[i] = -1

        # Remove -1 from the arrays
        right_row_avg = [x for x in right_row_avg if x > 0]
        left_row_avg = [x for x in left_row_avg if x > 0]

        # Detect local maximums for left-moving pedestrians
        left_mean = np.mean(left_row_avg)
        left_std = np.std(left_row_avg)
        left_threshold = left_mean + 2 * left_std  # Threshold for outliers

        # Detect outliers for right-moving pedestrians
        right_mean = np.mean(right_row_avg)
        right_std = np.std(right_row_avg)
        right_threshold = right_mean + 2 * right_std  # Threshold for outliers

        for i in range(len(right_local_max)):
            if right_local_max[i] > right_threshold:
                outliers.append(right_local_max[i])

        for i in range(len(left_local_max)):
            if left_local_max[i] > left_threshold:
                outliers.append(left_local_max[i])


        return len(outliers)
        # return len(right_local_max) + len(left_local_max) + len(outliers)
