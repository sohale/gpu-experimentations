# plot_cuda_perf.py, graph_run_times.py


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Data extracted from the provided output
data = {
    256: [0.000642728, 0.000345782, 0.000327035, 0.000323144, 0.000323786],
    512: [0.00137551, 0.00109886, 0.00108816, 0.00108282, 0.0010732],
    1024: [0.00412079, 0.00406615, 0.00404431, 0.00403746, 0.0040325],
    2048: [0.0159238, 0.0150116, 0.0149968, 0.0149611, 0.0149117]
}

# Convert data into a Pandas DataFrame
df = pd.DataFrame.from_dict(data, orient='index').T

# Compute mean and standard deviation for each N
mean_times = df.mean()
std_times = df.std()

# Plotting the mean execution time with error bars
plt.figure(figsize=(8, 5))
plt.errorbar(mean_times.index, mean_times.values, yerr=std_times.values, fmt='o-', capsize=5, label="Execution Time")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Matrix Size (N)")
plt.ylabel("Execution Time (s)")
plt.title("CUDA Matrix Multiplication Performance")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)
plt.show()

