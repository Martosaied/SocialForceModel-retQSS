import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
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
                if index % 5 != 0:
                    continue

                groups = Clustering(df, int(particles)).calculate_groups_by_time(row)
                if row['time'] not in groups_per_time:
                    groups_per_time[row['time']] = [len(groups)]
                else:
                    groups_per_time[row['time']].append(len(groups))

                # Graphic representation of the group position
                # Create a new figure for each frame
                plt.figure(figsize=(20, 20))

                # Set up the plot area
                plt.xlim(0, 50)
                plt.ylim(0, 50)

                # Add grid lines
                for i in range(21):  # 21 lines to create 20 divisions
                    plt.axhline(y=i, color='gray', linestyle='-', alpha=0.3)
                    plt.axvline(x=i, color='gray', linestyle='-', alpha=0.3)

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
                        
                        plt.scatter(x, y, color=color, s=300)
                    
                    # Add velocity vectors with quiver if we have velocity data
                    if df.iloc[index - 1] is not None and positions_x:
                        # Scale factor for velocity vectors
                        scale = 0.5
                        plt.quiver(positions_x, positions_y, 
                                np.array(velocities_x) * scale, 
                                np.array(velocities_y) * scale,
                                color='black', alpha=0.5, width=0.003)
                
                plt.suptitle('Grouped Lanes', fontsize=20)
                plt.title(f'Time: {row["time"]:.2f}', fontsize=16)
                plt.legend()
                plt.xlabel('X', fontsize=16)
                plt.ylabel('Y', fontsize=16)

                plt.savefig(f'{self.output_dir}/group_{index}.png')
                plt.close()

        # Create a new figure for the linear graph with title and x and y labels
        # Show the mean of the groups per time and its standard deviation
        plt.figure(figsize=(20, 20))
        mean_groups_per_time = {k: np.mean(v) for k, v in groups_per_time.items()}
        std_groups_per_time = {k: np.std(v) for k, v in groups_per_time.items()}
        plt.plot(list(mean_groups_per_time.keys()), list(mean_groups_per_time.values()))
        plt.fill_between(
            list(mean_groups_per_time.keys()), 
            (np.array(list(mean_groups_per_time.values())) - np.array(list(std_groups_per_time.values()))), 
            (np.array(list(mean_groups_per_time.values())) + np.array(list(std_groups_per_time.values()))), 
            alpha=0.2
        )

        plt.title('Number of groups per time')
        plt.xlabel('Time')
        plt.ylabel('Number of groups')
        plt.savefig(f'{output_dir}/linear_graph.png')
        plt.close()



        