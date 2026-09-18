# exemples d utilisation des scripts de calcul Kuramoto

1. exec_time_GPU_csr.py
 calcule le temps d'exécution, pour le cas sparse de format CSR 
 renvoie les fichiers "execution_times_GPU_sparse_stats.csv" et "gpu_memory_usage_sparse.csv"

Traçage :
    -plot_cuda_api_sparse.py
    -plot_memcpy_sparse.py
    -plot_memoire_sparse.py
    -plot_tegrastats_sparse.py
    -plot_threads_blocs_sparse.py

2. exec_time_GPU_meanfield.py
calcule le temps d'exécution, pour la version qui utilise le paramètre d'ordre 
renvoie les fichiers "execution_times_meanfield_stats.csv"  

3. exec_time_GPU_naive.py 
calcule d'exécution, pour le cas naive (matrice dense)
renvoie les fichiers "execution_times_GPU_dense_stats.csv"

Traçage :
    -plot_config_GPU_naive.py
    -plot_mem_N_naive.py
    -plot_memcpy_naive.py
    -plot_tegrastats_naive.py
    -plot_transferts_naive.py

4. exec_time_GPU_sparse_linear.py, exec_time_GPU_sparse_quadratic.py

Calculent le temps d'exécution pour le cas sparse avec un degré d qui croît :
- proportionnellement à N (cas linéaire)
- proportionnellement à N² (cas quadratique)
Renvoient les fichiers :
- execution_times_GPU_sparse_linear.csv
- execution_times_GPU_sparse_quadratic.csv

Traçage :
    -plot_cas_d.py

5. extraire_les_pics_memoire.py
Extrait la mémoire GPU allouée par appel de kernel depuis un fichier SQLite
généré par nsys. Utilisé pour les cas sparse avec d proportionnel à N
et d proportionnel à N².
Renvoie les fichiers :
- gpu_memory_sparse_linear.csv
- gpu_memory_sparse_quadratic.csv




