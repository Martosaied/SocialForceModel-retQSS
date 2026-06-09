import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from src.math.Clustering import Clustering, MIN_X_DISTANCE, MIN_Y_DISTANCE
from src.utils import get_parameter_combinations, process_parameters

class GroupedLanesGraph:
    """
    Plot a grouped lanes graph of the simulation.

    The grouped lanes graph is a graph that shows the grouped lanes of the simulation.
    """

    def __init__(self, results, output_dir, parameters):
        self.results = results
        self.output_dir = output_dir
        self.parameters = list(get_parameter_combinations(process_parameters(parameters.get('parameters', {}))))[0]

    def plot(self):
        groups_per_time = {}
        for result_file in self.results:
            df = pd.read_csv(result_file)

            particles = int((len(df.columns) - 1) / 5)
            for index, row in df.iterrows():
                print(f"Time: {row['time']}")

                if row['time'] < 89.0 or row['time'] % 5 != 0:
                    continue

                groups = Clustering(df, int(particles)).calculate_groups_by_time(row)
                print(f"Groups at time {row['time']}: {len(groups)}")
                if row['time'] not in groups_per_time:
                    groups_per_time[row['time']] = [len(groups)]
                else:
                    groups_per_time[row['time']].append(len(groups))

                # Get corridor parameters (assuming they exist in the data or can be inferred)
                FROM_Y = self.parameters.get('FROM_Y', 0)
                TO_Y = self.parameters.get('TO_Y', 50)

                # Graphic representation of the group position
                # Size the figure so its aspect matches the data extents
                x_extent = 50
                y_extent = TO_Y - FROM_Y
                BASE_W = 16.0
                fig_h = max(min(BASE_W * (y_extent / x_extent), 24.0), 2.0) if x_extent > 0 and y_extent > 0 else 12.0
                fig, ax = plt.subplots(figsize=(BASE_W, fig_h))

                # Set up the plot area — only the corridor width, drawn from 0
                # regardless of its world-frame position
                ax.set_xlim(0, x_extent)
                ax.set_ylim(0, y_extent)
                ax.set_aspect('equal')

                # Add grid lines efficiently
                GRID_SIZE = 50
                VOLUMES_COUNT = 1
                CELL_SIZE = GRID_SIZE / VOLUMES_COUNT
                
                # Pre-calculate grid lines
                grid_lines_x = [CELL_SIZE * i for i in range(VOLUMES_COUNT + 1)]
                grid_lines_y = [CELL_SIZE * i for i in range(VOLUMES_COUNT + 1)]
                
                for x in grid_lines_x:
                    ax.axvline(x=x, color='lightgray', linestyle='-', alpha=0.4, linewidth=0.8)
                for y in grid_lines_y:
                    ax.axhline(y=y, color='lightgray', linestyle='-', alpha=0.4, linewidth=0.8)

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
                        y = row[f'PY[{particle}]'] - FROM_Y

                        # Calculate velocities if we have a previous frame
                        vx = 0
                        vy = 0
                        if df.iloc[index - 1] is not None:
                            dt = row['time'] - df.iloc[index - 1]['time']
                            if dt > 0:  # Avoid division by zero
                                prev_x = df.iloc[index - 1][f'PX[{particle}]']
                                prev_y = df.iloc[index - 1][f'PY[{particle}]'] - FROM_Y
                                vx = (x - prev_x) / dt
                                vy = (y - prev_y) / dt

                        positions_x.append(x)
                        positions_y.append(y)
                        velocities_x.append(vx)
                        velocities_y.append(vy)

                        ax.scatter(x, y, color=color, s=200, alpha=0.8, edgecolors='black', linewidths=1.5)
                    
                    # Add velocity vectors with quiver if we have velocity data
                    if positions_x and any(v != 0 for v in velocities_x + velocities_y):
                        # Normalize velocities for better visualization
                        length_velocities = [
                            sqrt(velocities_x[i] ** 2 + velocities_y[i] ** 2) for i in range(len(velocities_x))
                        ]
                        normalize_velocities_x = [
                            velocities_x[i] / length_velocities[i] if length_velocities[i] > 0 else 0 for i in range(len(velocities_x))
                        ]
                        normalize_velocities_y = [
                            velocities_y[i] / length_velocities[i] if length_velocities[i] > 0 else 0 for i in range(len(velocities_y))
                        ]
                        ax.quiver(positions_x, positions_y, 
                                np.array(normalize_velocities_x), 
                                np.array(normalize_velocities_y),
                                color='gray', alpha=0.5, width=0.002, scale=20)
                
                # Add axis labels with units
                ax.set_xlabel('Posición X (metros)', fontsize=16, fontweight='bold')
                ax.set_ylabel('Posición Y (metros)', fontsize=16, fontweight='bold')
                
                # Improve tick labels
                ax.tick_params(axis='both', which='major', labelsize=12)
                
                # Create legend elements
                legend_elements = [
                    plt.Line2D([0], [0], color='gray', alpha=0.5, linewidth=2, label='Vectores de Velocidad'),
                ]
                
                # Add legend with better styling
                ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
                        title='Leyenda', fontsize=16, title_fontsize=18,
                        frameon=True, fancybox=True, shadow=True)

                plt.savefig(f'{self.output_dir}/group_{index}.png', dpi=60, bbox_inches='tight', facecolor='white', edgecolor='none')
                plt.close(fig)

        # Create a new figure for the linear graph with title and x and y labels
        # Show the mean of the groups per time and its standard deviation
        fig, ax = plt.subplots(figsize=(16, 10))
        mean_groups_per_time = {k: np.mean(v) for k, v in groups_per_time.items()}
        std_groups_per_time = {k: np.std(v) for k, v in groups_per_time.items()}
        
        times = list(mean_groups_per_time.keys())
        means = list(mean_groups_per_time.values())
        stds = list(std_groups_per_time.values())
        
        ax.plot(times, means, linewidth=3, color='#4444FF', label='Promedio de Grupos')
        ax.fill_between(times, 
                       np.array(means) - np.array(stds), 
                       np.array(means) + np.array(stds), 
                       alpha=0.3, color='#4444FF', label='Desviación Estándar')

        ax.set_title('Número de Grupos por Tiempo', fontsize=20, fontweight='bold', pad=20)
        ax.set_xlabel('Tiempo (segundos)', fontsize=16, fontweight='bold')
        ax.set_ylabel('Número de Grupos', fontsize=16, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=14, frameon=True, fancybox=True, shadow=True)
        ax.tick_params(axis='both', which='major', labelsize=12)


        # Revisar tambien el mean de tamaño de los grupos y su std
        print(f"Mean groups per time: {np.mean(list(mean_groups_per_time.values()))}")
        print(f"Std groups per time: {np.std(list(std_groups_per_time.values()))}")
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/linear_graph.png', dpi=100, bbox_inches='tight', facecolor='white', edgecolor='none')
        plt.close(fig)



        