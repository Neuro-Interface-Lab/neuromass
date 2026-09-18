import pandas as pd
import matplotlib.pyplot as plt

# Lecture du fichier CSV
fichier = "03_configuration_gpu_par_N.csv"
df = pd.read_csv(fichier)

df = df.dropna(subset=["N_nodes"])

# Graphe


fig, ax1 = plt.subplots(figsize=(6, 5))


# Nombre de blocs

ligne1, = ax1.plot(
    df["N_nodes"],
    df["nombre_blocs"],
    marker="o",
    linewidth=2,
    label="Nombre de blocs"
)

ax1.set_xlabel("Nombre de nœuds N")
ax1.set_ylabel("Nombre de blocs")
ax1.grid(True, alpha=0.3)

# Valeurs des blocs
for x, y in zip(df["N_nodes"], df["nombre_blocs"]):
    ax1.annotate(
        f"{int(y)}",
        (x, y),
        xytext=(0, 10),
        textcoords="offset points",
        ha="center",
        fontsize=9
    )


# Nombre de threads

ax2 = ax1.twinx()

ligne2, = ax2.plot(
    df["N_nodes"],
    df["threads_lances"],
    marker="s",
    linewidth=2,
    label="Threads lancés"
)

ax2.set_ylabel("Nombre de threads lancés")

# Valeurs des threads
for x, y in zip(df["N_nodes"], df["threads_lances"]):
    ax2.annotate(
        f"{int(y)}",
        (x, y),
        xytext=(0, -18),
        textcoords="offset points",
        ha="center",
        fontsize=9
    )


plt.title("Configuration du lancement CUDA en fonction de N")

ax1.legend(
    [ligne1, ligne2],
    ["Nombre de blocs", "Threads lancés"],
    loc="upper left"
)

plt.tight_layout()

# Sauvegarde
plt.savefig(
    "configuration_GPU_vs_N.png",
    dpi=300,
    bbox_inches="tight"
)

plt.show()