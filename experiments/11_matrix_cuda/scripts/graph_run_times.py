# plot_cuda_perf.py, graph_run_times.py

import pandas as pd
import matplotlib.pyplot as plt

# Load results, skipping comment lines
df = pd.read_csv("runtime_results.csv", comment='#')

print(df.columns)
ΞN = "N"
Ξtime = "Time"

# Compute mean and standard deviation for each N
mean_times = df.groupby(ΞN)[Ξtime].mean()
std_times = df.groupby(ΞN)[Ξtime].std()

# Plot mean execution time with error bars
plt.figure(figsize=(8, 5))
plt.errorbar(mean_times.index, mean_times.values, yerr=std_times.values, fmt='o-', capsize=5, label="Execution Time")
plt.xscale('log')
plt.yscale('log')
plt.xlabel("Matrix Size (N)")
plt.ylabel("Execution Time (s)")
plt.title("CUDA Matrix Multiplication Performance")
plt.legend()
plt.grid(True, which="both", linestyle="--", linewidth=0.5)

# Save plot
plt.savefig("cuda_runtime_plot.png")
plt.show()
