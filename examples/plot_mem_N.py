import pandas as pd
import matplotlib.pyplot as plt

# Charger le fichier CSV
fichier = "08_memoire_gpu_par_N.csv"
df = pd.read_csv(fichier)

fig, ax = plt.subplots(figsize=(10, 6))

# Tracer la mémoire totale en fonction de N
ax.plot(
    df["N"],
    df["total_GPU_memory_MB"],
    marker="o",
    linewidth=2,
    color="#4C72B0",
    label="Mémoire GPU totale"
)

# Ajouter les valeurs SOUS chaque point
for x, y in zip(df["N"], df["total_GPU_memory_MB"]):
    ax.annotate(
        f"{y:.1f} MB",
        (x, y),
        xytext=(0, -12),            # ← texte 12 pts SOUS le point
        textcoords="offset points",
        ha="center",
        va="top",                   # ← ancrage en haut du texte
        fontsize=9,
        color="#1F4E79"
    )

# Échelle log-log
ax.set_xscale("log")
ax.set_yscale("log")

# Axes
ax.set_xlabel("Nombre de nœuds N (échelle log)")
ax.set_ylabel("Mémoire GPU nécessaire (MB, échelle log)")
ax.set_title("Mémoire GPU nécessaire en fonction de N")

# Grille
ax.grid(True, which="both", alpha=0.3)

# Marges pour laisser respirer
ax.set_ylim(bottom=df["total_GPU_memory_MB"].min() * 0.5,
            top=df["total_GPU_memory_MB"].max() * 3)
ax.set_xlim(left=df["N"].min() * 0.6,
            right=df["N"].max() * 1.5)

# Légende
ax.legend(loc="upper left")

plt.tight_layout()
plt.savefig("memoire_GPU_vs_N.png", dpi=300, bbox_inches="tight")
plt.show()

print("Figure enregistrée : memoire_GPU_vs_N.png")