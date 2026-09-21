import pandas as pd
import matplotlib.pyplot as plt

# ============================================================
# PARAMÈTRES
# ============================================================

backend = "cpp"

file_1 = "execution_times_naive_1threads.csv"
file_multi = "execution_times_naive_omp_threads.csv"

# ============================================================
# LECTURE
# ============================================================

df1 = pd.read_csv(file_1)
df_multi = pd.read_csv(file_multi)

# Données C++ avec 1 thread
t1 = df1[
    (df1["backend"] == backend) &
    (df1["n_threads"] == 1)
][["N", "mean_s"]].copy()

t1 = t1.rename(columns={"mean_s": "T1"})

# Données C++ avec 8 threads
t8 = df_multi[
    (df_multi["backend"] == backend) &
    (df_multi["n_threads"] == 8)
][["N", "mean_s"]].copy()

t8 = t8.rename(columns={"mean_s": "T8"})

# ============================================================
# ASSOCIATION DES DONNÉES
# ============================================================

df = pd.merge(t1, t8, on="N")

# Gain = temps 1 thread / temps 8 threads
df["gain"] = df["T1"] / df["T8"]

df = df.sort_values("N")

print(df)

# ============================================================
# TRACÉ
# ============================================================

plt.figure(figsize=(9, 6))

# Gain mesuré
plt.plot(
    df["N"],
    df["gain"],
    marker="o",
    linewidth=2,
    label="Gain mesuré (1 thread / 8 threads)"
)

# Gain idéal avec 8 threads
plt.axhline(
    y=8,
    linestyle="--",
    linewidth=2,
    label="Gain idéal = 8"
)

plt.xlabel("Nombre d'oscillateurs N")
plt.ylabel("Gain en temps d'exécution (T₁ / T₈)")

plt.title("Gain OpenMP — Naive Dense — C++")

plt.grid(True, alpha=0.3)
plt.legend()

plt.tight_layout()

# Sauvegarde
plt.savefig("gain_naive_cpp_1_vs_8.png", dpi=300)

plt.show()