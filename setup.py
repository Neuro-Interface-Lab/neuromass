import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup
import os


NATIVE_DIR = "src/neuromass/models/kuramoto/_native"
NUMPY_INCLUDE = np.get_include()

OMP_COMPILE_ARGS = ["-fopenmp", "-O3"]
OMP_LINK_ARGS = ["-fopenmp"]

#Détection CUDA

CUDA_AVAILABLE = os.system("which nvcc > /dev/null 2>&1") == 0

c_backend_sources = [
    f"{NATIVE_DIR}/c_backend_wrapper.pyx",
    f"{NATIVE_DIR}/kuramoto_c_kernel.c",
]

cuda_compile_args = []
cuda_link_args = []
cuda_include_dirs = []
cuda_library_dirs = []

if CUDA_AVAILABLE:
    print("CUDA détecté")
    
    # Vérifier si la bibliothèque dynamique existe
    cuda_lib = f"{NATIVE_DIR}/libkuramoto_cuda.so"
    if os.path.exists(cuda_lib):
        print("Bibliothèque CUDA trouvée")
        cuda_link_args = ["-L" + NATIVE_DIR, "-lkuramoto_cuda", "-lcudart"]
        cuda_include_dirs = ["/usr/local/cuda/include"]
        cuda_library_dirs = ["/usr/local/cuda/lib64"]
        print(" Backend C compilé avec CUDA")
    else:
        print("Bibliothèque CUDA manquante")
else:
    print("NVCC non trouvé, backend C sans GPU")


extensions = [
    Extension(
        name="neuromass.models.kuramoto._native.cython_backend",
        sources=[f"{NATIVE_DIR}/cython_backend.pyx"],
        include_dirs=[NUMPY_INCLUDE],
        extra_compile_args=OMP_COMPILE_ARGS,
        extra_link_args=OMP_LINK_ARGS,
    ),
    Extension(
        name="neuromass.models.kuramoto._native.c_backend",
        sources=[
            f"{NATIVE_DIR}/c_backend_wrapper.pyx",
            f"{NATIVE_DIR}/kuramoto_c_kernel.c",
        ],
        include_dirs=[NUMPY_INCLUDE, NATIVE_DIR] + cuda_include_dirs,
        extra_compile_args=OMP_COMPILE_ARGS + cuda_compile_args,
        extra_link_args=OMP_LINK_ARGS + [
            "-Wl,--no-as-needed",           # ← Force le lien même si pas utilisé
            "-L" + NATIVE_DIR,
            "-l:libkuramoto_cuda.so",       # ← Lien direct avec la bibliothèque
            "-lcudart",
            "-Wl,--undefined=simulate_naive_kuramoto_cuda"  # ← Force la résolution du symbole
        ],
        library_dirs=[NATIVE_DIR] + cuda_library_dirs,
        runtime_library_dirs=[NATIVE_DIR] + cuda_library_dirs,
    ),
   Extension(
        name="neuromass.models.kuramoto._native.cpp_backend",
        sources=[
            f"{NATIVE_DIR}/cpp_backend_wrapper.pyx",
            f"{NATIVE_DIR}/kuramoto_cpp_kernel.cpp",
        ],
        include_dirs=[NUMPY_INCLUDE, NATIVE_DIR],
        language="c++",
        extra_compile_args=OMP_COMPILE_ARGS,
        extra_link_args=OMP_LINK_ARGS,
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    )
)
