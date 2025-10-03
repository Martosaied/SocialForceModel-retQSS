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
                if index < 900 or index % 10 != 0:
                    continue

                groups = Clustering(df, int(particles)).calculate_groups_by_time(row)
                print(f"Groups at time {row['time']}: {len(groups)}")
                if row['time'] not in groups_per_time:
                    groups_per_time[row['time']] = [len(groups)]
                else:
                    groups_per_time[row['time']].append(len(groups))

                # Graphic representation of the group position
                # Create a new figure for each frame with better proportions
                fig, ax = plt.subplots(figsize=(16, 12))

                # Set up the plot area
                ax.set_xlim(0, 50)
                ax.set_ylim(0, 50)
                
                # Get corridor parameters (assuming they exist in the data or can be inferred)
                FROM_Y = 20  # Default values, should be passed as parameters
                TO_Y = 30
                
                # Highlight the corridor area
                corridor_rect = plt.Rectangle(
                    (0, 0), 
                    50, 
                    50, 
                    facecolor='#808080', 
                    alpha=0.3,
                    edgecolor='none',
                    label='_nolegend_'
                )
                corridor_rect_2 = plt.Rectangle(
                    (0, FROM_Y), 
                    50, 
                    TO_Y - FROM_Y, 
                    facecolor='#FFFFFF', 
                    alpha=1,
                    edgecolor='none',
                    label='_nolegend_'
                )
                ax.add_patch(corridor_rect)
                ax.add_patch(corridor_rect_2)

                # Add grid lines efficiently
                GRID_SIZE = 50
                VOLUMES_COUNT = 10
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
                
                # Count active pedestrians
                active_pedestrians = len([i for i in range(1, particles + 1) if row.get(f'PX[{i}]') is not None])
                
                # Add main title and timestamp with better styling
                fig.suptitle('Carriles Agrupados - Simulación de Peatones', fontsize=24, fontweight='bold', y=0.95)
                ax.set_title(f'Tiempo: {row["time"]:.2f} segundos | Peatones: {active_pedestrians} | Grupos: {len(groups)} | Corredor: {abs(FROM_Y-TO_Y):.1f}m', 
                            fontsize=16, fontweight='bold', pad=20)
                
                # Add axis labels with units
                ax.set_xlabel('Posición X (metros)', fontsize=16, fontweight='bold')
                ax.set_ylabel('Posición Y (metros)', fontsize=16, fontweight='bold')
                
                # Improve tick labels
                ax.tick_params(axis='both', which='major', labelsize=12)
                
                # Create legend elements
                legend_elements = [
                    plt.Line2D([0], [0], color='gray', alpha=0.5, linewidth=2, label='Vectores de Velocidad'),
                    plt.Rectangle((0, 0), 1, 1, facecolor='#FFFFFF', alpha=1, label='Corredor'),
                    plt.Rectangle((0, 0), 1, 1, facecolor='#808080', alpha=0.3, label='Área Externa')
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



        