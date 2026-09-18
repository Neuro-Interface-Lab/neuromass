"""Extrait la mémoire GPU totale allouée par appel (fenêtre glissante)."""

import sqlite3
import csv


SQLITE_FILE = "profil_memoire_sparse_linear_d_quadratique.sqlite"
RESULTS_CSV = "gpu_memory_sparse_quadratic.csv"

N_PER_BLOCKS = {
    2: 500, 4: 1000, 20: 5000, 40: 10000, 79: 20000,
    118: 30000, 157: 40000, 391: 100000, 1172: 300000,
}


def extract_memory_by_kernel_window(sqlite_file):
    """Associe chaque allocation au kernel le plus proche (par correlationId)."""
    con = sqlite3.connect(sqlite_file)
    cur = con.cursor()

    # 1. Récupérer les kernels
    cur.execute("""
        SELECT k.start, k.end, k.gridX, k.correlationId, s.value AS name
        FROM CUPTI_ACTIVITY_KIND_KERNEL k
        JOIN StringIds s ON k.demangledName = s.id
        WHERE s.value LIKE '%sparse_kuramoto%'
        ORDER BY k.start
    """)
    kernels = cur.fetchall()
    print(f"Nombre d'appels kernel : {len(kernels)}")

    # 2. Somme TOTALE de toutes les allocations (malloc uniquement)
    cur.execute("""
        SELECT start, bytes, memoryOperationType
        FROM CUDA_GPU_MEMORY_USAGE_EVENTS
        WHERE memoryOperationType = 1   -- malloc
        ORDER BY start
    """)
    allocations = cur.fetchall()
    print(f"Nombre d'allocations (malloc) : {len(allocations)}")

    # 3. Total alloué sur toute la session
    total_bytes = sum(a[1] for a in allocations)
    print(f"Mémoire totale allouée (cumul) : {total_bytes / 1e6:.2f} Mo")

    # 4. Calcul : mémoire allouée entre le début du 1er kernel et la fin du dernier
    # (approximation de la mémoire "live" au pic)
    if kernels:
        t_start = min(k[0] for k in kernels)
        t_end = max(k[1] for k in kernels)

        # Somme des allocations qui sont encore "vivantes" (malloc - free) à t_end
        # Approximation : somme de toutes les allocations faites avant t_end
        # moins celles libérées avant t_end
        cur.execute("""
            SELECT
                (SELECT COALESCE(SUM(bytes), 0)
                 FROM CUDA_GPU_MEMORY_USAGE_EVENTS
                 WHERE memoryOperationType = 1 AND start <= ?) -
                (SELECT COALESCE(SUM(bytes), 0)
                 FROM CUDA_GPU_MEMORY_USAGE_EVENTS
                 WHERE memoryOperationType = 2 AND start <= ?)
        """, (t_end, t_end))
        live_bytes = cur.fetchone()[0]
        print(f"Mémoire 'live' au pic : {live_bytes / 1e6:.2f} Mo")

    # 5. Grouper par N (via gridX)
    grouped = {}
    for k_start, k_end, grid_x, corr_id, name in kernels:
        N = N_PER_BLOCKS.get(grid_x, grid_x * 256)

        # Pour ce kernel, calculer la mémoire allouée depuis le kernel précédent
        # (approximation : toutes les allocations sur [k_start - 100ms, k_end])
        window_start = k_start - 100_000_000   # 100 ms avant
        mem_bytes = 0
        for a_start, a_bytes, a_type in allocations:
            if window_start <= a_start <= k_end:
                mem_bytes += a_bytes

        grouped.setdefault(N, []).append(mem_bytes / 1e6)  # Mo

    con.close()
    return grouped


def main():
   
    print("  Extraction mémoire GPU par N (fenêtre glissante)")
   

    grouped = extract_memory_by_kernel_window(SQLITE_FILE)

    csv_rows = []
    print(f"\n{'N':<10}{'mémoire moy (Mo)':<20}{'n mesures':<12}")
    for N in sorted(grouped.keys()):
        mems = grouped[N]
        mean_mem = sum(mems) / len(mems)
        csv_rows.append([N, mean_mem])
        print(f"{N:<10}{mean_mem:<20.3f}{len(mems):<12}")

    with open(RESULTS_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["N", "mem_gpu_MB"])
        w.writerows(csv_rows)
    print(f"\n {RESULTS_CSV}")


if __name__ == "__main__":
    main()