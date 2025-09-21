import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt
from src.math.Clustering import Clustering

class GroupedLanesGraph:
    """
    Plot a grouped lanes graph of the simulation.

    The grouped lanes graph is a graph that shows the grouped lanes of the simulation.
    """

    def __init__(self, results, output_dir):
        self.results = results
        self.output_dir = output_dir

    def plot(self):
        groups_per_time = {}
        for result_file in self.results:
            df = pd.read_csv(result_file)

            particles = int((len(df.columns) - 1) / 5)
            for index, row in df.iterrows():
                if  index % 10 != 0:
                    continue

                groups = Clustering(df, int(particles)).calculate_groups_by_time(row)
                print(f"Groups at time {row['time']}: {len(groups)}")
                if row['time'] not in groups_per_time:
                    groups_per_time[row['time']] = [len(groups)]
                else:
                    groups_per_time[row['time']].append(len(groups))

                # Graphic representation of the group position
                # Create a new figure for each frame
                fig, ax = plt.subplots(figsize=(12, 12))

                # Set up the plot area
                ax.set_xlim(0, 50)
                ax.set_ylim(0, 50)

                # Add grid lines efficiently
                GRID_SIZE = 50
                VOLUMES_COUNT = 10
                CELL_SIZE = GRID_SIZE / VOLUMES_COUNT
                
                # Pre-calculate grid lines
                grid_lines_x = [CELL_SIZE * i for i in range(VOLUMES_COUNT + 1)]
                grid_lines_y = [CELL_SIZE * i for i in range(VOLUMES_COUNT + 1)]
                
                for x in grid_lines_x:
                    ax.axvline(x=x, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)
                for y in grid_lines_y:
                    ax.axhline(y=y, color='gray', linestyle='-', alpha=0.3, linewidth=0.5)

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
                        
                        ax.scatter(x, y, color=color, s=100, alpha=0.8)
                    
                    # Add velocity vectors with quiver if we have velocity data
                    if df.iloc[index - 1] is not None and positions_x:
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
                                color='black', alpha=0.7, width=0.003, scale=20)
                
                # Add main title and timestamp
                fig.suptitle('Carriles Agrupados - Simulación de Peatones', fontsize=16)
                ax.set_title(f'Tiempo: {row["time"]:.2f} (Frame {index})', fontsize=14)
                
                # Add axis labels
                ax.set_xlabel('Posición X', fontsize=12)
                ax.set_ylabel('Posición Y', fontsize=12)
                
                # Create legend elements
                legend_elements = [
                    plt.Line2D([0], [0], color='black', alpha=0.7, linewidth=2, label='Vectores de Velocidad')
                ]
                
                # Add legend
                ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1, 0.5),
                        title='Leyenda', fontsize=12, title_fontsize=14)

                plt.savefig(f'{self.output_dir}/group_{index}.png', dpi=300, bbox_inches='tight')
                plt.close(fig)

        # Create a new figure for the linear graph with title and x and y labels
        # Show the mean of the groups per time and its standard deviation
        fig, ax = plt.subplots(figsize=(12, 8))
        mean_groups_per_time = {k: np.mean(v) for k, v in groups_per_time.items()}
        std_groups_per_time = {k: np.std(v) for k, v in groups_per_time.items()}
        
        times = list(mean_groups_per_time.keys())
        means = list(mean_groups_per_time.values())
        stds = list(std_groups_per_time.values())
        
        ax.plot(times, means, linewidth=2, color='blue', label='Promedio de Grupos')
        ax.fill_between(times, 
                       np.array(means) - np.array(stds), 
                       np.array(means) + np.array(stds), 
                       alpha=0.3, color='blue', label='Desviación Estándar')

        ax.set_title('Número de Grupos por Tiempo', fontsize=16)
        ax.set_xlabel('Tiempo', fontsize=12)
        ax.set_ylabel('Número de Grupos', fontsize=12)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=12)


        # Revisar tambien el mean de tamaño de los grupos y su std
        print(f"Mean groups per time: {np.mean(list(mean_groups_per_time.values()))}")
        print(f"Std groups per time: {np.std(list(std_groups_per_time.values()))}")
        
        plt.tight_layout()
        plt.savefig(f'{self.output_dir}/linear_graph.png', dpi=300, bbox_inches='tight')
        plt.close(fig)



        