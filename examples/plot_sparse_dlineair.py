import csv
import matplotlib.pyplot as plt
 
CSV_FILE = "gpu_memory_usage_sparse.csv"
 
 
def load_memory_csv(filename=CSV_FILE):
    N_values, mem_values = [], []
    with open(filename, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row["mem_delta_mb"] not in ("", "None"):
                N_values.append(int(row["N"]))
                mem_values.append(float(row["mem_delta_mb"]))
    return N_values, mem_values
 
 
def plot_gpu_memory(N_values, mem_values, out_file="gpu_memory_vs_N.png"):
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(N_values, mem_values, marker="o", color="#4C72B0", label="Mémoire GPU totale")
 
    for N, mem in zip(N_values, mem_values):
        ax.annotate(f"{mem:.1f} MB", (N, mem),
                    textcoords="offset points", xytext=(5, -12),
                    fontsize=9, color="#4C72B0")
 
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Nombre de nœuds N (échelle log)")
    ax.set_ylabel("Mémoire GPU nécessaire (MB, échelle log)")
    ax.set_title("Mémoire GPU nécessaire en fonction de N")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
 
    plt.tight_layout()
    plt.savefig(out_file, dpi=150)
    print(f"Figure sauvegardée : {out_file}")
 
 
if __name__ == "__main__":
    N_values, mem_values = load_memory_csv()
    plot_gpu_memory(N_values, mem_values)
 