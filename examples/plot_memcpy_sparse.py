import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("06_memcpy_summary_sparse.csv")

fig, axes = plt.subplots(1, 2, figsize=(6, 5))
colors = ["#4C72B0", "#DD8452"]

# --- 1. Volume total (avec %) ---
ax1 = axes[0]
bars1 = ax1.bar(df["operation"], df["taille_totale_MB"], color=colors)
ax1.set_ylabel("Volume total transféré (Mo, échelle log)")
ax1.set_title("Volume total par direction", pad=15)
ax1.set_yscale("log")
ax1.grid(True, which="both", axis="y", alpha=0.3)
# MARGE RÉDUITE : ×1.3 au lieu de ×5
ax1.set_ylim(top=df["taille_totale_MB"].max() * 1.3,
             bottom=df["taille_totale_MB"].min() * 0.7)
for b, v, gb, p, n in zip(bars1, df["taille_totale_MB"],
                          df["taille_totale_GB"], df["taille_pourcent"],
                          df["nombre_transferts"]):
    ax1.text(b.get_x() + b.get_width()/2, v * 1.05,
             f"{v:,.0f} Mo\n({gb:.2f} Go, {p}%)\n{n} transferts",
             ha="center", fontsize=9, fontweight="bold")

# Temps total (avec %) ---
ax2 = axes[1]
bars2 = ax2.bar(df["operation"], df["temps_total_ms"], color=colors)
ax2.set_ylabel("Temps total (ms, échelle log)")
ax2.set_title("Temps total par direction", pad=15)
ax2.set_yscale("log")
ax2.grid(True, which="both", axis="y", alpha=0.3)
# ⬇ MARGE RÉDUITE : ×1.3 au lieu de ×5
ax2.set_ylim(top=df["temps_total_ms"].max() * 1.3,
             bottom=df["temps_total_ms"].min() * 0.7)
for b, v, p in zip(bars2, df["temps_total_ms"], df["temps_pourcent"]):
    ax2.text(b.get_x() + b.get_width()/2, v * 1.05,
             f"{v:.2f} ms\n({p}%)",
             ha="center", fontsize=10, fontweight="bold")

plt.suptitle("Transferts mémoire CPU ↔ GPU — Cas sparse (CSR)",
             fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig("memory_transfers_sparse.png", dpi=150, bbox_inches="tight")
plt.show()
print("Figure enregistrée : memory_transfers_sparse.png")