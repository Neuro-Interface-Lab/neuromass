import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import csv


# Lire les données du fichier CSV mean-field

N = []
means = []
stds = []

with open("execution_times_meanfield_stats.csv", "r") as f:
    reader = csv.reader(f)
    next(reader)  # Sauter l'en-tête
    for row in reader:
        N.append(int(row[0]))          # Colonne 0 : N
        means.append(float(row[1]))    # Colonne 1 : mean_time
        stds.append(float(row[2]))     # Colonne 2 : std_time


plt.figure(figsize=(12, 8))

# Points expérimentaux avec barres d'erreur
plt.errorbar(N, means, yerr=stds, fmt='o-', capsize=5,
             linewidth=2, markersize=8, color='blue',
             label='GPU Mean-Field')


N_ref = np.array(N)

# O(N) référence
T_ref_O_N = (N_ref / N_ref[0]) * means[0]
plt.loglog(N_ref, T_ref_O_N, 'r--', linewidth=2,
           label='O(N) référence', alpha=0.7)

# O(N²) référence
T_ref_O_N2 = (N_ref / N_ref[0])**2 * means[0]
plt.loglog(N_ref, T_ref_O_N2, 'g--', linewidth=2,
           label='O(N²) référence', alpha=0.5)


plt.xlabel('N (nombre d\'oscillateurs)', fontsize=14)
plt.ylabel('Temps d\'exécution (s)', fontsize=14)
plt.title('Performances GPU Mean-Field - Échelle log-log', fontsize=16)
plt.grid(True, alpha=0.3)
plt.legend(fontsize=12)

plt.savefig('meanfield_performance.png', dpi=300)
print(" Graphique sauvegardé : meanfield_performance.png")