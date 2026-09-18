import sys
import subprocess

import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

NATIVE_DIR = "src/neuromass/models/kuramoto/_native"
NUMPY_INCLUDE = np.get_include()

if sys.platform == "darwin":
    # Apple clang has no native OpenMP; use Homebrew's libomp.
    libomp_prefix = subprocess.check_output(
        ["brew", "--prefix", "libomp"]
    ).decode().strip()

    OMP_COMPILE_ARGS = [
        "-Xpreprocessor", "-fopenmp",
        f"-I{libomp_prefix}/include",
        "-O3",
    ]
    OMP_LINK_ARGS = [
        "-Xpreprocessor", "-fopenmp",
        f"-L{libomp_prefix}/lib",
        "-lomp",
    ]
else:
    OMP_COMPILE_ARGS = ["-fopenmp", "-O3"]
    OMP_LINK_ARGS = ["-fopenmp"]

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
        include_dirs=[NUMPY_INCLUDE, NATIVE_DIR],
        extra_compile_args=OMP_COMPILE_ARGS,
        extra_link_args=OMP_LINK_ARGS,
    ),
    Extension(
        name="neuromass.models.kuramoto._native.cpp_backend",
        sources=[
            f"{NATIVE_DIR}/cpp_backend_wrapper.pyx",
            f"{NATIVE_DIR}/kuramoto_cpp_kernel.cpp",
        ],
        include_dirs=[NUMPY_INCLUDE, NATIVE_DIR],
        language="c++",
        extra_compile_args=OMP_COMPILE_ARGS + ["-std=c++11"],
        extra_link_args=OMP_LINK_ARGS,
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={
            "language_level": "3",
            "boundscheck": False,
            "wraparound": False,
        },
    )
)