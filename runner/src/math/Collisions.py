import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform

"""
This module contains functions to detect collisions/personal space invasions 
between pedestrians in crowd simulations.
"""

class Collisions:
    """
    Class for detecting collisions and personal space invasions in pedestrian simulations.
    A collision is defined as two pedestrians being within a threshold distance (default: 2 * PEDESTRIAN_R).
    """
    
    def __init__(self, df, particles, collision_threshold=0.3, start_index=0):
        """
        Initialize the Collisions detector.
        
        Args:
            df: DataFrame with simulation results (columns: time, PX[i], PY[i], VX[i], VY[i], PS[i])
            particles: Number of particles in the simulation
            collision_threshold: Distance threshold for collision detection (default: 2 * 0.3 = 0.6m)
            start_index: Index to start analyzing (to skip initial transient behavior)
        """
        self.df = df
        self.particles = particles
        self.collision_threshold = collision_threshold
        self.start_index = start_index
    
    def calculate_collisions_at_timestep(self, row):
        """
        Calculate collisions for a single timestep.
        
        Args:
            row: DataFrame row with particle positions
            
        Returns:
            dict: Collision statistics for this timestep
        """
        # Extract positions for all active particles
        positions = []
        pedestrian_ids = []
        
        for i in range(1, self.particles + 1):
            px_col = f'PX[{i}]'
            py_col = f'PY[{i}]'
            
            if px_col in row and py_col in row:
                px = row[px_col]
                py = row[py_col]
                
                # Only include active pedestrians (non-zero positions)
                if not pd.isna(px) and not pd.isna(py) and (px != 0 or py != 0):
                    positions.append([px, py])
                    pedestrian_ids.append(i)
        
        # Need at least 2 pedestrians for collisions
        if len(positions) < 2:
            return {
                'collision_count': 0,
                'collision_rate': 0,
                'collision_pairs': []
            }
        
        # Calculate pairwise distances
        positions_array = np.array(positions)
        distances = pdist(positions_array)
        distance_matrix = squareform(distances)
        
        # Count collisions (pairs within threshold)
        collision_count = 0
        collision_pairs = []
        
        n_pedestrians = len(positions)
        for i in range(n_pedestrians):
            for j in range(i + 1, n_pedestrians):
                if distance_matrix[i, j] < self.collision_threshold:
                    collision_count += 1
                    collision_pairs.append((pedestrian_ids[i], pedestrian_ids[j], distance_matrix[i, j]))
        
        # Calculate collision rate (collisions per possible pair)
        max_possible_pairs = n_pedestrians * (n_pedestrians - 1) / 2
        collision_rate = collision_count / max_possible_pairs if max_possible_pairs > 0 else 0
        
        return {
            'collision_count': collision_count,
            'collision_rate': collision_rate,
            'collision_pairs': collision_pairs,
            'n_pedestrians': n_pedestrians
        }
    
    def calculate_total_collisions(self, sample_rate=1):
        """
        Calculate total collisions across all timesteps in the simulation.
        
        Args:
            sample_rate: Process every Nth row (default: 1 = all rows)
            
        Returns:
            dict: Overall collision statistics
        """
        total_collisions = 0
        total_timesteps = 0
        collision_rates = []
        
        for index, row in self.df.iterrows():
            # Skip initial transient period
            if index < self.start_index:
                continue
            
            # Apply sample rate
            if index % sample_rate != 0:
                continue
            
            # Calculate collisions for this timestep
            timestep_stats = self.calculate_collisions_at_timestep(row)
            
            total_collisions += timestep_stats['collision_count']
            collision_rates.append(timestep_stats['collision_rate'])
            total_timesteps += 1
        
        return {
            'total_collisions': total_collisions,
            'total_timesteps': total_timesteps,
            'avg_collisions_per_timestep': total_collisions / total_timesteps if total_timesteps > 0 else 0,
            'avg_collision_rate': np.mean(collision_rates) if collision_rates else 0,
            'max_collision_rate': np.max(collision_rates) if collision_rates else 0,
            'std_collision_rate': np.std(collision_rates) if collision_rates else 0
        }


def calculate_collisions_from_csv(csv_path, particles, collision_threshold=0.6, start_index=0):
    """
    Convenience function to calculate collisions directly from a CSV file.
    
    Args:
        csv_path: Path to the solution CSV file
        particles: Number of particles in the simulation
        collision_threshold: Distance threshold for collision detection
        start_index: Index to start analyzing
        
    Returns:
        dict: Collision statistics
    """
    try:
        df = pd.read_csv(csv_path)
        collisions = Collisions(df, particles, collision_threshold, start_index)
        return collisions.calculate_total_collisions()
    except Exception as e:
        print(f"Error calculating collisions from {csv_path}: {e}")
        return {
            'total_collisions': 0,
            'total_timesteps': 0,
            'avg_collisions_per_timestep': 0,
            'avg_collision_rate': 0,
            'max_collision_rate': 0,
            'std_collision_rate': 0
        }


