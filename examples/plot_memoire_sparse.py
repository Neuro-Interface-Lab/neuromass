import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("08_memoire_gpu_sparse_par_N.csv")

fig, ax = plt.subplots(figsize=(10, 6))

# --- Mémoire totale sparse ---
ax.plot(df["N"], df["total_GPU_memory_MB"],
        marker="o", linewidth=2, color="#4C72B0",
        label="Mémoire GPU totale (sparse)")

# Annotations sur la courbe sparse
for x, y in zip(df["N"], df["total_GPU_memory_MB"]):
    ax.annotate(f"{y:.1f} MB", (x, y),
                xytext=(0, -18), textcoords="offset points",
                ha="center", va="top", fontsize=9, color="#1F4E79")

# Échelles log-log
ax.set_xscale("log")
ax.set_yscale("log")

ax.set_xlabel("Nombre de nœuds N (échelle log)", fontsize=12)
ax.set_ylabel("Mémoire GPU nécessaire (Mo, échelle log)", fontsize=12)
ax.set_title("Mémoire GPU nécessaire en fonction de N — Cas sparse (CSR, d=10)",
             fontsize=13, pad=15)

ax.grid(True, which="both", alpha=0.3)
ax.legend(loc="upper left", fontsize=11)

plt.tight_layout()
plt.savefig("memoire_GPU_sparse_vs_N.png", dpi=150, bbox_inches="tight")
plt.show()
print("Figure enregistrée : memoire_GPU_sparse_vs_N.png")