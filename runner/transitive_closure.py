import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from math import sqrt

MIN_DISTANCE = 2

def get_related(pair, row):
    return sqrt(
        pow(row[f'PX[{pair[0]}]'] - row[f'PX[{pair[1]}]'], 2) + pow(row[f'PY[{pair[0]}]'] - row[f'PY[{pair[1]}]'], 2)
    ) < MIN_DISTANCE

def get_pairs(row, particles):
    pairs = []
    for i in range(1, particles):
        for j in range(i+1, particles):
            if i == j or row[f'PS[{i}]'] != row[f'PS[{j}]']:
                continue
            pairs.append((i, j))
    return pairs


def calculate_closures(row, pairs):
    closures = []
    for pair in pairs:
        related = get_related(pair, row)
        if related:
            for closure in closures:
                if pair[0] in closure or pair[1] in closure:
                    closure.add(pair[0])
                    closure.add(pair[1])
                    break
            else:
                closures.append(set(pair))

    return closures
    


def get_transitive_closure(solution_file, output_dir):
    df = pd.read_csv(solution_file)

    particles = (len(df.columns) - 1) / 5
    pairs = get_pairs(df.iloc[0], int(particles))
    for index, row in df.iterrows():
        closures = calculate_closures(row, pairs)
        
        # Graphic representation of the closure position
        # Create a new figure for each frame
        plt.figure(figsize=(20, 20))

        # Set up the plot area
        plt.xlim(0, 20)
        plt.ylim(0, 20)

        # Add grid lines
        for i in range(21):  # 21 lines to create 20 divisions
            plt.axhline(y=i, color='gray', linestyle='-', alpha=0.3)
            plt.axvline(x=i, color='gray', linestyle='-', alpha=0.3)

        for closure in closures:
            # Create a random color for the closure
            color = np.random.rand(3,)
            for particle in closure:
                plt.scatter(row[f'PX[{particle}]'], row[f'PY[{particle}]'], color=color, s=300)
        
        plt.savefig(f'{output_dir}/closure_{index}.png')
        plt.close()


get_transitive_closure('./experiments/retqss-implementation/latest/result_0.csv', './experiments/retqss-implementation/latest')