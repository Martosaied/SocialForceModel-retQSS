import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# Read the CSV file
# for i in range(0, 99):
df = pd.read_csv(f'./experiments/helbing-lanes-20m/run_20250104_123432/result_50.csv')
df1 = pd.read_csv(f'./experiments/helbing-lanes-12m/run_20250104_135925/result_6.csv')

# get the 150th row
row = df.iloc[150]
row1 = df1.iloc[150]
# skew line by the Y axis
diff = []
diff1 = []
for i in range(1, 20):
    blue_dots = [row[f'PY[{j}]'] < i and row[f'PS[{j}]'] == 1 for j in range(1, 120)]
    red_dots = [row[f'PY[{j}]'] < i and row[f'PS[{j}]'] == 0 for j in range(1, 120)]
    blue_dots1 = [row1[f'PY[{j}]'] < i and row1[f'PS[{j}]'] == 1 for j in range(1, 120)]
    red_dots1 = [row1[f'PY[{j}]'] < i and row1[f'PS[{j}]'] == 0 for j in range(1, 120)]
    diff.append(np.sum(blue_dots) - np.sum(red_dots))
    diff1.append(np.sum(blue_dots1) - np.sum(red_dots1))

der_diff = []
der_diff1 = []
for i in range(len(diff)-1):
    der_diff.append(diff[i]-diff[i+1])
    der_diff1.append(diff1[i]-diff1[i+1])

plt.plot(der_diff)
plt.plot(der_diff1)
plt.title(f'Time: {row["time"]:.2f}')

# save the plot
plt.savefig(f'./experiments/helbing-lanes-20m/skew.png')

