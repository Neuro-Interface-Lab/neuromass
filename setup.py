import numpy as np
from Cython.Build import cythonize
from setuptools import Extension, setup
import os
import glob
import subprocess


NATIVE_DIR = "src/neuromass/models/kuramoto/_native"
NUMPY_INCLUDE = np.get_include()

OMP_COMPILE_ARGS = ["-fopenmp", "-O3"]
OMP_LINK_ARGS = ["-fopenmp"]


CUDA_AVAILABLE = os.system("which nvcc > /dev/null 2>&1") == 0

cuda_objects = []
cuda_link_args = []
cuda_library_dirs = []
cuda_link_args = []

if CUDA_AVAILABLE:
    print("CUDA detecte, compilation des fichiers .cu...")
    nvcc_path = subprocess.check_output(["which", "nvcc"]).decode().strip()
    CUDA_HOME = os.path.dirname(os.path.dirname(nvcc_path))
    print(f"CUDA_HOME detecte : {CUDA_HOME}")

    cu_files = glob.glob(f"{NATIVE_DIR}/*.cu")

    for cu_file in cu_files:
        obj_file = cu_file.replace(".cu", ".o")
        subprocess.run([
            "nvcc", "-c", cu_file, "-o", obj_file, "-rdc=true", "-Xcompiler", "-fPIC",
            "-O3",
        ], check=True)
        cuda_objects.append(obj_file)
    dlink_file = f"{NATIVE_DIR}/dlink.o"
    subprocess.run(
        ["nvcc", "-dlink"] + cuda_objects + ["-o", dlink_file, "-Xcompiler", "-fPIC"],
        check=True,
    )
    cuda_objects.append(dlink_file)

    cuda_include_dirs = [f"{CUDA_HOME}/include"]
    cuda_library_dirs = [f"{CUDA_HOME}/lib64"]
    cuda_link_args = ["-lcudart"]
 
else:
    print("nvcc non trouve : compilation sans le backend GPU")


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
        library_dirs=cuda_library_dirs,  
        extra_compile_args=OMP_COMPILE_ARGS + ["-std=c99"],
        extra_link_args=OMP_LINK_ARGS + cuda_link_args + ["-lstdc++"],
        extra_objects=cuda_objects,
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