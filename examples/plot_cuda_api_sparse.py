import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("04_cuda_api_summary_sparse.csv")

# ✅ Filtrer uniquement les interactions importantes
interactions_voulues = [
    "cudaDeviceSynchronize",
    "cudaMalloc",
    "cudaFree",
    "cudaMemcpy",
    "cudaLaunchKernel",
]
df = df[df["interaction"].isin(interactions_voulues)]

# Trier par temps décroissant
df = df.sort_values("temps_total_ms", ascending=False)

fig, ax = plt.subplots(figsize=(11, 6))

bars = ax.bar(df["interaction"], df["temps_total_ms"],
              color="#4C72B0", width=0.6)

# Annotations au-dessus des barres
for b, v, n in zip(bars, df["temps_total_ms"], df["nombre_appels"]):
    ax.text(b.get_x() + b.get_width()/2, v * 1.05,
            f"{v:.2f} ms\n({n} appels)",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

ax.set_xlabel("Interaction CUDA", fontsize=12)
ax.set_ylabel("Temps total (ms)", fontsize=12)
ax.set_title("Temps total des interactions CUDA — Cas sparse (CSR)",
             fontsize=13, pad=15)

plt.xticks(rotation=20, ha="right", fontsize=10)

ax.set_ylim(top=df["temps_total_ms"].max() * 1.25)
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("cuda_api_sparse.png", dpi=150, bbox_inches="tight")
plt.show()
print("Figure enregistrée : cuda_api_sparse.png")