import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup

NATIVE_DIR = "src/neuromass/models/kuramoto/_native"
NUMPY_INCLUDE = np.get_include()

extensions = [
    Extension(
        name="neuromass.models.kuramoto._native.cython_backend",
        sources=[f"{NATIVE_DIR}/cython_backend.pyx"],
        include_dirs=[NUMPY_INCLUDE],
    ),
    Extension(
        name="neuromass.models.kuramoto._native.c_backend",
        sources=[
            f"{NATIVE_DIR}/c_backend_wrapper.pyx",
            f"{NATIVE_DIR}/kuramoto_c_kernel.c",
        ],
        include_dirs=[NUMPY_INCLUDE, NATIVE_DIR],
    ),
    Extension(
        name="neuromass.models.kuramoto._native.cpp_backend",
        sources=[
            f"{NATIVE_DIR}/cpp_backend_wrapper.pyx",
            f"{NATIVE_DIR}/kuramoto_cpp_kernel.cpp",
        ],
        include_dirs=[NUMPY_INCLUDE, NATIVE_DIR],
        language="c++",
    ),
]

setup(
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3"},
    )
)
