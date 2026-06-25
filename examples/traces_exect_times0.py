"""tracé log_log des temps d execution en fonction de nbe d'oscillateurs avec courbe de N et N**2"""

import csv
import matplotlib.pyplot as plt
import numpy as np
import os
from collections import defaultdict 


def read_csv_data(filename="execution_times.csv"):
    """Read CSV file without pandas."""
    
    if not os.path.isfile(filename):
        return None
    
    data = defaultdict(list)
    with open(filename, 'r') as f:
        reader = csv.reader(f)
        headers = next(reader)
        
        for row in reader:
            if not row or len(row) < 3:
                continue
            if not row[0] or not row[1] or not row[2]:
                continue
            
            try:
                N = int(row[0])
                backend = row[1]
                mean_time = float(row[2])
                data[(N, backend)].append(mean_time)
            except ValueError:
                continue
    
    if not data:
        return None
    
    return data, headers
          


def plot_from_csv(filename="execution_times.csv"):
    """Plot log-log scaling from CSV file."""
    
    result = read_csv_data(filename)
    if result is None:
        return
    
    data, headers = result
    
    if not data:
        return
    
    avg ={}
    for (N, backend), times in data.items():
        avg_mean = np.mean(times)
        avg_std = np.std(times) if len(times) > 1 else 0.0
        avg[(N, backend)] = (avg_mean, avg_std)

    N_values = sorted({N for (N, _) in data.keys()})
    backends = sorted({backend for (_, backend) in data.keys()})
  
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    colors = {"python": "blue", "cython": "orange", "c": "green", "cpp": "red"}
    markers = {"python": "o", "cython": "s", "c": "^", "cpp": "D"}
    labels = {"python": "Python", "cython": "Cython", "c": "C", "cpp": "C++"}
    
  
    
    first_mean_time = None
    
    for backend in backends:
        if backend == "python":
            
            continue
        N_vals = []
        mean_times = []
        std_times = []
        
        for N in N_values:
            if (N, backend) in avg:
                mean, std = avg[(N, backend)]
                N_vals.append(N)
                mean_times.append(mean)
                std_times.append(std)
    
        
        if backend == backends[0] and len(mean_times) > 0:
            first_mean_time = mean_times[0]

        
        ax.errorbar(
            N_vals, mean_times,
            yerr = std_times,
            marker=markers.get(backend, "o"),
            color=colors.get(backend, "black"),
            label=labels.get(backend, backend),
            linewidth=2,
            markersize=8,
            capsize=4,
            elinewidth=1,
            markeredgewidth = 1,
            alpha = 0.8,
        )
        print(f" {labels.get(backend, backend)} :")
        print(f" N= {N_vals} ")
        print(f" temps = {mean_times}")
        print(f" ecart_type ={std_times}")
        print()
    
    if first_mean_time is not None and len(N_values) > 0:
        N_ref = np.array(N_values)
        t_ref_n2 = N_ref**2 / (N_ref[0]**2 / first_mean_time)
        t_ref_n = N_ref / N_ref[0] * first_mean_time
        
        ax.loglog(N_ref, t_ref_n2, 'k--', label=r'$N^2$', alpha=0.7)
        ax.loglog(N_ref, t_ref_n, 'k:', label=r'$N$', alpha=0.7)
    
    ax.set_xlabel("n_nodes")
    ax.set_ylabel("Temps d'execution (s)")
    
    ax.legend(loc="upper left")
    ax.grid(True, alpha=0.3)
    
  
  
    
    fig.tight_layout()
    plt.show()
    
    # Calcul des pentes
    if len(N_values) >= 3:
        for backend in backends:
            N_vals = []
            mean_times = []
            
            for N in N_values:
                if (N, backend) in avg:
                    mean, _ = avg[(N, backend)]
                    N_vals.append(N)
                    mean_times.append(mean)
            
            if len(N_vals) >= 3:
                logN = np.log(N_vals)
                logT = np.log(mean_times)
                slope, intercept = np.polyfit(logN, logT, 1)
                print(f"  {labels.get(backend, backend):<10} : pente = {slope:.2f} ->  O(N^{slope:.2f})")
    
    """for backend in backends:
            N_vals = []
            mean_times = []
            
            for N in N_values:
                if (N, backend) in avg:
                    mean, _ = avg[(N, backend)]
                    N_vals.append(N)
                    mean_times.append(mean)
            if len(N_vals)>= 3:
                N_array = np.array(N_vals)
                T_array = np.array(mean_times)

                coefs = np.polyfit(N_array, T_array)
                a,b,c = coefs

    for N in N_values:
        if (N, "python") in avg and (N, "cpp") in avg:
            t_py, _ = avg[(N, "python")]
            t_cpp, _ = avg[(N, "cpp")]
            if t_cpp > 0:
                speedup = t_py / t_cpp
                print(f"{N:>6} {speedup:>16.2f}x")"""


if __name__ == "__main__":
    plot_from_csv("execution_times.csv")