# exemples d'utilisation des scripts de benchmark Kuramoto

## avant_omp

Résultats de référence obtenus avant parallélisation OpenMP (version séquentielle).

- execution_times_naive_seq.csv
  temps d'exécution du cas naïf (matrice dense), séquentiel
  Traçage : benchmark_naive_seq.png, exec_time_naive65.png

- execution_times_meanfield_seq.csv
  temps d'exécution du cas mean-field (paramètre d'ordre), séquentiel
  Traçage : exec_time_meanfield_sparal_.png, exec_time_meanfield65.png

- execution_times_sparse_seq.csv
  temps d'exécution du cas sparse (CSR), séquentiel
  Traçage : exec_time_sparse_sparalf.png, exec_time_sparse65.png

## courbes_GPU

Courbes de performance obtenues sur GPU, utilisées comme référence de comparaison avec les versions CPU/OpenMP.

- gpu_performance_CSR.png, gpu_performance_CSR_linear.png, gpu_csr_reference.png
  performance GPU pour le cas sparse (format CSR)

- meanfield_performance.png, meanfield_performance_avec_2000.png, meanfield_performance_5x10e6.png
  performance GPU pour le cas mean-field, à différentes tailles de réseau (jusqu'à 2000 nœuds, puis 5x10^6)

- donnes_csv_avec_2000.png
  visualisation des données brutes pour N = 2000

## omp_8threads

Benchmarks OpenMP à nombre de threads fixe (8 threads).

1. exec_time_naive_8threads.py
   calcule le temps d'exécution du cas naïf (matrice dense) avec 8 threads OpenMP
   renvoie le fichier "execution_times_naive_8threads.csv"

2. exec_time_meanfield_8threads.py
   calcule le temps d'exécution du cas mean-field avec 8 threads OpenMP
   renvoie le fichier "execution_times_meanfield_8threads.csv"

3. exec_time_sparse_8threads.py
   calcule le temps d'exécution du cas sparse (CSR) avec 8 threads OpenMP
   renvoie le fichier "execution_times_sparse_8threads.csv"

Traçage :
    -benchmark_8threads_comparison.png (comparaison naïf / mean-field / sparse à 8 threads)

## omp_threads358

Benchmarks OpenMP avec un nombre de threads variable (3, 5, 8 threads), pour étudier la scalabilité de chaque modèle.

1. exec_time_meanfield_omp.py
   calcule le temps d'exécution du cas mean-field pour plusieurs nombres de threads OpenMP
   renvoie le fichier "execution_times_meanfield_omp_threads.csv"

   Traçage :
       -diagramme_meanfieldomp.png
       -exec_time_meanfieldomp65.png

2. plot_naive_omp.py
   trace le temps d'exécution du cas naïf en fonction du nombre de threads OpenMP
   à partir du fichier "execution_times_naive_omp_threads.csv"

   Traçage :
       -diagramme_naiveomp.png
       -exec_time_ompnaive65.png

3. exec_time_sparse_omp.py
   calcule le temps d'exécution du cas sparse (CSR) pour plusieurs nombres de threads OpenMP
   renvoie le fichier "execution_times_sparse_omp_threads.csv"

   Traçage :
       -diagramme_sparseomp.png
       -exec_time_sparseomp.png
       -exec_time_sprarseomp65.png
