import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# Loi d'Amdahl
# ============================================================
def amdahl(p, n):
    return 1 / ((1 - p) + p / n)

# ============================================================
# Configuration
# ============================================================
THREADS_MAX = 8
N_VALUES = 200

# ============================================================
# Générer les courbes théoriques
# ============================================================
n_vals = np.linspace(1, THREADS_MAX, N_VALUES)

amdahl_50 = [amdahl(0.50, n) for n in n_vals]
amdahl_75 = [amdahl(0.75, n) for n in n_vals]
amdahl_90 = [amdahl(0.90, n) for n in n_vals]
amdahl_95 = [amdahl(0.95, n) for n in n_vals]
amdahl_99 = [amdahl(0.99, n) for n in n_vals]

# ============================================================
# PALETTE HAUT CONTRASTE
# ============================================================
COLORS = {
    '50': '#E74C3C',    # Rouge vif
    '75': '#E67E22',    # Orange
    '90': '#27AE60',    # Vert
    '95': '#2980B9',    # Bleu
    '99': '#8E44AD'     # Violet
}

# ============================================================
# Tracer les courbes
# ============================================================
plt.figure(figsize=(12, 8))

plt.plot(n_vals, amdahl_50, '--', label='50% parallélisable', 
         color=COLORS['50'], linewidth=2.5)
plt.plot(n_vals, amdahl_75, '--', label='75% parallélisable', 
         color=COLORS['75'], linewidth=2.5)
plt.plot(n_vals, amdahl_90, '--', label='90% parallélisable', 
         color=COLORS['90'], linewidth=2.5)
plt.plot(n_vals, amdahl_95, '--', label='95% parallélisable', 
         color=COLORS['95'], linewidth=2.5)
plt.plot(n_vals, amdahl_99, '--', label='99% parallélisable', 
         color=COLORS['99'], linewidth=2.5)

# Configuration
plt.xlabel('Nombre de processeurs / threads', fontsize=14, fontweight='bold')
plt.ylabel('Speedup (gain)', fontsize=14, fontweight='bold')
plt.title("Loi d'Amdahl", fontsize=16, fontweight='bold')
plt.grid(True, alpha=0.3, linestyle='--')
plt.legend(loc='upper left', fontsize=12)
plt.xlim(0, THREADS_MAX + 1)
plt.ylim(0, THREADS_MAX + 1)

# ============================================================
# Annotations
# ============================================================
for n in [2, 4, 8]:
    s = amdahl(0.95, n)
    plt.annotate(f'{s:.2f}x', xy=(n, s), xytext=(n+0.2, s-0.3),
                 fontsize=10, fontweight='bold', color=COLORS['95'])

for n in [2, 4, 8]:
    s = amdahl(0.99, n)
    plt.annotate(f'{s:.2f}x', xy=(n, s), xytext=(n+0.2, s+0.2),
                 fontsize=10, fontweight='bold', color=COLORS['99'])

for n in [4, 8]:
    s = amdahl(0.90, n)
    plt.annotate(f'{s:.2f}x', xy=(n, s), xytext=(n-0.8, s-0.2),
                 fontsize=10, fontweight='bold', color=COLORS['90'])

# ============================================================
# Sauvegarde
# ============================================================
plt.tight_layout()
plt.savefig('amdahl_theorique.png', dpi=300, bbox_inches='tight')
print("✅ Graphique sauvegardé : amdahl_theorique.png")
plt.show()

# ============================================================
# Résumé
# ============================================================
print("\n" + "=" * 60)
print("📊 SPEEDUP THÉORIQUE POUR 8 THREADS")
print("=" * 60)

for p, label, color in [(0.50, '50%', COLORS['50']), 
                        (0.75, '75%', COLORS['75']), 
                        (0.90, '90%', COLORS['90']), 
                        (0.95, '95%', COLORS['95']), 
                        (0.99, '99%', COLORS['99'])]:
    s = amdahl(p, 8)
    print(f"{label:>5} parallélisable : {s:.2f}x")