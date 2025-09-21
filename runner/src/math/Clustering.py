import numpy as np
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster

"""
This module contains the functions to cluster the pedestrians.
"""

MIN_X_DISTANCE = 5
MIN_Y_DISTANCE = 0.8
MIN_LANE_SIZE = 4
MIN_DIRECTION_CONSISTENCY = 0.7  # Cosine similarity threshold for lane direction
MAX_LANE_WIDTH = 2.0  # Maximum width for a lane

X_INDEX = 1
Y_INDEX = 2
VX_INDEX = 3
VY_INDEX = 4
PS_INDEX = 5

class Clustering:
    """
    This class contains the functions to cluster the pedestrians.
    """
    def __init__(self, df, particles, min_x_distance=MIN_X_DISTANCE, min_y_distance=MIN_Y_DISTANCE):
        self.df = df
        self.particles = particles
        self.min_x_distance = min_x_distance
        self.min_y_distance = min_y_distance
    
    def distance(self, tup1, tup2):
        return abs(tup1[1] - tup2[1]) < self.min_x_distance and abs(tup1[2] - tup2[2]) < self.min_y_distance

    def unify_groups(self, particles_groups):
        # Remove duplicates
        unique_groups = {}
        for group in particles_groups:
            sorted_group = tuple(sorted(group))
            hash_id = str(hash(sorted_group))
            unique_groups[hash_id] = set(sorted_group)

        return list(unique_groups.values())

    def join_groups(self, particles_groups):
        for group in particles_groups:
            for other_group in particles_groups:
                if len(set(group) & set(other_group)) > 0:
                    group.update(other_group)

        return particles_groups

    def similar_velocity(self, particle, other_particle):
        return True
        v1 = np.array([particle[VX_INDEX], particle[VY_INDEX]])
        v2 = np.array([other_particle[VX_INDEX], other_particle[VY_INDEX]])

        dot = np.dot(v1, v2)
        mag1 = np.linalg.norm(v1)
        mag2 = np.linalg.norm(v2)
        
        if mag1 == 0 or mag2 == 0:
            return True
        
        cos_theta = dot / (mag1 * mag2)
        cos_theta = np.clip(cos_theta, -1.0, 1.0)  # numerical safety
        
        return np.degrees(np.arccos(cos_theta)) < 30

    def calculate_direction_consistency(self, group, particles_list):
        """Calculate how consistent the direction is within a group"""
        if len(group) < 2:
            return 1.0
        
        velocities = []
        for particle_id in group:
            particle = next((p for p in particles_list if p[0] == particle_id), None)
            if particle:
                v = np.array([particle[VX_INDEX], particle[VY_INDEX]])
                if np.linalg.norm(v) > 0:
                    velocities.append(v / np.linalg.norm(v))  # normalize
        
        if len(velocities) < 2:
            return 1.0
        
        # Calculate average direction consistency
        velocities = np.array(velocities)
        mean_direction = np.mean(velocities, axis=0)
        mean_direction = mean_direction / np.linalg.norm(mean_direction)
        
        consistencies = [np.dot(v, mean_direction) for v in velocities]
        return np.mean(consistencies)

    def calculate_lane_width(self, group, particles_list):
        """Calculate the width of a potential lane"""
        if len(group) < 2:
            return 0.0
        
        positions = []
        for particle_id in group:
            particle = next((p for p in particles_list if p[0] == particle_id), None)
            if particle:
                positions.append([particle[X_INDEX], particle[Y_INDEX]])
        
        if len(positions) < 2:
            return 0.0
        
        positions = np.array(positions)
        
        # Calculate the width perpendicular to the main direction
        if len(positions) >= 2:
            # Get the main direction
            velocities = []
            for particle_id in group:
                particle = next((p for p in particles_list if p[0] == particle_id), None)
                if particle:
                    v = np.array([particle[VX_INDEX], particle[VY_INDEX]])
                    if np.linalg.norm(v) > 0:
                        velocities.append(v / np.linalg.norm(v))
            
            if velocities:
                mean_direction = np.mean(velocities, axis=0)
                mean_direction = mean_direction / np.linalg.norm(mean_direction)
                
                # Project positions onto perpendicular direction
                perpendicular = np.array([-mean_direction[1], mean_direction[0]])
                projections = np.dot(positions, perpendicular)
                width = np.max(projections) - np.min(projections)
                return width
        
        return 0.0

    def direct_group(self, particle, particles_list):
        particle_group = set([particle[0]])
        for other_particle in particles_list:
            if self.distance(particle, other_particle) and particle[PS_INDEX] == other_particle[PS_INDEX] and self.similar_velocity(particle, other_particle):
                particle_group.add(other_particle[0])

        return particle_group

    def calculate_groups(self, from_y=0, to_y=50, start_index=0, sample_rate=5):
        groups = []
        for index, row in self.df.iterrows():
            if start_index > index or index % sample_rate != 0:
                continue

            # initializing list
            particles_list = [(i, row[f'PX[{i}]'], row[f'PY[{i}]'], row[f'VX[{i}]'], row[f'VY[{i}]'], row[f'PS[{i}]']) for i in range(1, self.particles) if row[f'PY[{i}]'] >= from_y and row[f'PY[{i}]'] <= to_y]

            particles_groups = []
            for particle in particles_list:
                particle_group = self.direct_group(particle, particles_list)
                particles_groups.append(particle_group)

            previous_group_quantity = 0
            while len(particles_groups) != previous_group_quantity:
                previous_group_quantity = len(particles_groups)
                particles_groups = self.join_groups(particles_groups)
                particles_groups = self.unify_groups(particles_groups)

            groups.append(len([group for group in particles_groups if len(group) > 3]))

        return np.mean(groups)

    def calculate_groups_by_time(self, row, from_y=0, to_y=50):
        particles_list = [(i, row[f'PX[{i}]'], row[f'PY[{i}]'], row[f'VX[{i}]'], row[f'VY[{i}]'], row[f'PS[{i}]']) for i in range(1, self.particles) if row[f'PY[{i}]'] >= from_y and row[f'PY[{i}]'] <= to_y]

        particles_groups = []
        for particle in particles_list:
            particle_group = self.direct_group(particle, particles_list)
            particles_groups.append(particle_group)

        previous_group_quantity = 0
        while len(particles_groups) != previous_group_quantity:
            previous_group_quantity = len(particles_groups)
            particles_groups = self.join_groups(particles_groups)
            particles_groups = self.unify_groups(particles_groups)

        return [group for group in particles_groups if len(group) > 3]