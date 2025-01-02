import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read the CSV file
df = pd.read_csv('./experiments/experiment_20241229_194422/result_0.csv')

print(df.head())

# Create frames
for index, row in df.iterrows():
    diff = []
    for i in range(1, 20):
        blue_dots = [row[f'PY[{j}]'] < i and row[f'PS[{j}]'] == 1 for j in range(1, 300)]
        red_dots = [row[f'PY[{j}]'] < i and row[f'PS[{j}]'] == 0 for j in range(1, 300)]

        diff.append(np.sum(blue_dots) - np.sum(red_dots))

    
    der_diff = []
    for i in range(len(diff)-1):
        der_diff.append(diff[i]-diff[i+1])
   
    plt.plot(der_diff)
    plt.title(f'Time: {row["time"]:.2f}')
    
plt.show()

