import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# Chargement des données
# ============================================================

df = pd.read_csv("execution_times_naive_omp_threads.csv")

# Vérification
print(df.head())
print("\nBackends :", df["backend"].unique())
print("Threads :", df["n_threads"].unique())
print("N :", df["N"].unique())


# ============================================================
# Paramètres d'affichage
# ============================================================

threads = sorted(df["n_threads"].unique())
backends = df["backend"].unique()

markers = ["o", "s", "^"]


# ============================================================
# Un graphique par backend
# ============================================================

for backend in backends:

    plt.figure(figsize=(10, 6))

    data_backend = df[df["backend"] == backend]

    for i, n_threads in enumerate(threads):

        data = data_backend[
            data_backend["n_threads"] == n_threads
        ].sort_values("N")

        plt.plot(
            data["N"],
            data["mean_s"],
            marker=markers[i],
            linewidth=2,
            markersize=7,
            label=f"{n_threads} threads"
        )

    # --------------------------------------------------------
    # Échelle logarithmique
    # --------------------------------------------------------

    plt.xscale("log")
    plt.yscale("log")

    # --------------------------------------------------------
    # Axes
    # --------------------------------------------------------

    plt.xlabel(
        "N (nombre d'oscillateurs)",
        fontsize=12
    )

    plt.ylabel(
        "Temps d'exécution moyen (s)",
        fontsize=12
    )

    # --------------------------------------------------------
    # Grille
    # --------------------------------------------------------

    plt.grid(
        True,
        which="both",
        alpha=0.3
    )

    # --------------------------------------------------------
    # Légende
    # --------------------------------------------------------

    plt.legend(
        fontsize=11,
        title="Nombre de threads"
    )

    # --------------------------------------------------------
    # Titre
    # --------------------------------------------------------

    plt.title(
        f"Performance du backend {backend.upper()} "
        f"selon le nombre de threads\n"
        "Kuramoto Naive Dense",
        fontsize=14,
        pad=15
    )

    plt.tight_layout()

    # --------------------------------------------------------
    # Sauvegarde
    # --------------------------------------------------------

    filename = f"performance_{backend}_threads.png"

    plt.savefig(
        filename,
        dpi=200,
        bbox_inches="tight"
    )

    plt.show()

    print(f"Figure enregistrée : {filename}")