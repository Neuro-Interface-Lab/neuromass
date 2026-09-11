import pandas as pd
import matplotlib.pyplot as plt

# Charger le fichier CSV
fichier = "cuda_api_summary_detailed.csv"
df = pd.read_csv(fichier)

# Garder uniquement les principales interactions CUDA
interactions_principales = [
    "cudaDeviceSynchronize",
    "cudaMalloc",
    "cudaFree",
    "cudaMemcpy",
    "cudaLaunchKernel"
]

df_plot = df[df["interaction"].isin(interactions_principales)].copy()

# Trier du plus grand au plus petit
df_plot = df_plot.sort_values("temps_total_ms", ascending=False)

# ============================================================
# 1. Temps total par interaction
# ============================================================

plt.figure(figsize=(11, 6))

bars = plt.bar(
    df_plot["interaction"],
    df_plot["temps_total_ms"]
)

plt.ylabel("Temps total (ms)")
plt.xlabel("Interaction CUDA")
plt.title("Temps total des interactions CUDA")

plt.xticks(rotation=25)
plt.grid(axis="y", alpha=0.3)

# Afficher les valeurs au-dessus des barres
for bar, valeur in zip(bars, df_plot["temps_total_ms"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{valeur:.1f} ms",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    "temps_interactions_CUDA.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()


# ============================================================
# 2. Répartition en pourcentage
# ============================================================

temps_total = df_plot["temps_total_ms"].sum()

df_plot["pourcentage"] = (
    df_plot["temps_total_ms"] / temps_total * 100
)

plt.figure(figsize=(11, 6))

bars = plt.bar(
    df_plot["interaction"],
    df_plot["pourcentage"]
)

plt.ylabel("Part du temps total (%)")
plt.xlabel("Interaction CUDA")
plt.title("Répartition du temps d'exécution des interactions CUDA")

plt.xticks(rotation=25)
plt.grid(axis="y", alpha=0.3)

# Valeurs en %
for bar, valeur in zip(bars, df_plot["pourcentage"]):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height(),
        f"{valeur:.2f} %",
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()

plt.savefig(
    "repartition_temps_interactions_CUDA.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()