import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("03_configuration_gpu_sparse.csv")

fig, ax1 = plt.subplots(figsize=(14, 7))

# =========================================================
# Axe gauche : Threads lancés
# =========================================================
color1 = "#4C72B0"

ax1.set_xlabel(
    "N (nombre d'oscillateurs, échelle logarithmique)",
    fontsize=12
)

ax1.set_ylabel(
    "Threads lancés",
    color=color1,
    fontsize=12
)

line1 = ax1.plot(
    df["N_nodes"],
    df["threads_lances"],
    "o-",
    color=color1,
    linewidth=2,
    markersize=7,
    label="Threads lancés"
)

ax1.tick_params(axis="y", labelcolor=color1)


# =========================================================
# Étiquettes Threads
# =========================================================
for x, y in zip(df["N_nodes"], df["threads_lances"]):

    ax1.annotate(
        f"{y:,}".replace(",", " "),
        (x, y),
        xytext=(0, 18),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=9,
        color=color1,
        rotation=45,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor=color1,
            alpha=0.9
        )
    )


# =========================================================
# Axe droit : Nombre de blocs
# =========================================================
color2 = "#DD8452"

ax2 = ax1.twinx()

ax2.set_ylabel(
    "Nombre de blocs",
    color=color2,
    fontsize=12
)

line2 = ax2.plot(
    df["N_nodes"],
    df["nombre_blocs"],
    "s-",
    color=color2,
    linewidth=2,
    markersize=7,
    label="Nombre de blocs"
)

ax2.tick_params(axis="y", labelcolor=color2)


# =========================================================
# Étiquettes Blocs
# =========================================================
for x, y in zip(df["N_nodes"], df["nombre_blocs"]):

    ax2.annotate(
        f"{y}",
        (x, y),
        xytext=(0, -22),
        textcoords="offset points",
        ha="center",
        va="top",
        fontsize=9,
        color=color2,
        rotation=45,
        bbox=dict(
            boxstyle="round,pad=0.25",
            facecolor="white",
            edgecolor=color2,
            alpha=0.9
        )
    )


# =========================================================
# Échelle logarithmique
# =========================================================
ax1.set_xscale("log")
ax1.set_yscale("log")
ax2.set_yscale("log")


# =========================================================
# Grille
# =========================================================
ax1.grid(
    True,
    which="both",
    alpha=0.3
)


# =========================================================
# Marges
# =========================================================
ax1.set_xlim(
    left=df["N_nodes"].min() * 0.5,
    right=df["N_nodes"].max() * 2
)

ax1.set_ylim(
    bottom=df["threads_lances"].min() * 0.5,
    top=df["threads_lances"].max() * 3
)


# =========================================================
# Légende
# =========================================================
lines = line1 + line2
labels = [line.get_label() for line in lines]

ax1.legend(
    lines,
    labels,
    loc="upper left",
    fontsize=11
)


# =========================================================
# Titre
# =========================================================
plt.title(
    "Threads et blocs en fonction de N — Cas sparse (CSR)\n"
    "(256 threads par bloc)",
    fontsize=14,
    pad=20
)


# =========================================================
# Mise en page
# =========================================================
plt.tight_layout()

plt.savefig(
    "threads_blocs_sparse.png",
    dpi=200,
    bbox_inches="tight"
)

plt.show()

print("Figure enregistrée : threads_blocs_sparse.png")