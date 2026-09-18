# Native Kuramoto kernels

This directory is reserved for compiled low-level implementations specific to
Kuramoto-family models.

Recommended usage:

- `*.pyx` for Cython kernels;
- `*.c` for C sources;
- `*.cpp` for C++ sources;
- thin Python wrappers should stay in `model.py` or adjacent Python modules.

Keeping native sources inside each model family makes it easier to:

- isolate model-specific kernels;
- experiment with several implementations side by side;
- evolve the build configuration incrementally in `pyproject.toml`.

# Kuramoto Model

Implémentation du modèle d'oscillateurs de Kuramoto avec plusieurs backends de calcul (Python, Cython, C, C++).

## Modèles disponibles

- **Naïf (dense)** : couplage complet via une matrice d'adjacence dense
- **Mean-field** : formulation optimisée utilisant le paramètre d'ordre (r, psi)
- **Sparse (CSR)** : pour les réseaux creux, couplage stocké au format CSR

## Backends disponibles

Chaque modèle peut être exécuté avec :

- `python` : implémentation pure Python
- `cython` : wrapper Cython appelant le noyau C
- `c` : noyau C (séquentiel ou OpenMP)
- `cpp` : noyau C++ (séquentiel ou OpenMP)

Les versions C et C++ ont chacune une variante `_omp` parallélisée avec OpenMP.

## Fichiers principaux

```text
kuramoto_c_kernel.h / .c        # noyau C (séquentiel + OpenMP)
kuramoto_cpp_kernel.hpp / .cpp  # noyau C++ (séquentiel + OpenMP)
c_backend_wrapper.pyx           # wrapper Cython vers le noyau C
cpp_backend_wrapper.pyx         # wrapper Cython vers le noyau C++
cython_backend.pyx              # implémentation Cython native
```

Chaque noyau (C ou C++) expose 6 fonctions :

| Fonction |               Modèle |           Exécution |

 `simulate_naive_kuramoto_*` | Naïf (dense) | séquentiel |
`simu_para_complexe_*` | Mean-field | séquentiel |
 `simu_sparse_*` | Sparse (CSR) | séquentiel |
 `simulate_naive_kuramoto_*_omp` | Naïf (dense) | OpenMP |
 `simu_para_complexe_*_omp` | Mean-field | OpenMP |
  `simu_sparse_*_omp` | Sparse (CSR) | OpenMP |

(`*` = `c` ou `cpp` selon le backend)

## Compilation

```bash
pip install numpy scipy cython setuptools
python setup.py build_ext --inplace
```

## Exemple d'utilisation

```python
import numpy as np
from neuromass.models.kuramoto.model import MeanFieldKuramotoModel

n = 100
omega = np.random.normal(size=n)
theta0 = np.random.uniform(0, 2 * np.pi, size=n)

model = MeanFieldKuramotoModel(n_nodes=n, omega=omega, epsilon=1.0)

time, theta = model.solve(theta0=theta0, T=10.0, dt=0.01, backend="cpp")
```

Backends disponibles pour `backend=` : `"python"`, `"cython"`, `"c"`, `"cpp"`.

## Sortie

Le solveur renvoie `(time, theta)` :

- `time` : instants de simulation
- `theta` : phase de chaque oscillateur à chaque instant (shape `(n_nodes, n_steps + 1)`)
