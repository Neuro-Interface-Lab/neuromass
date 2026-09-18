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

---
- Files in this directory

-Source files (written manually)

**C kernels**
- `kuramoto_c_kernel.c` — CPU kernels (OpenMP) for dense, sparse, mean-field
- `kuramoto_c_kernel.h` — header

**C++ kernels**
- `kuramoto_cpp_kernel.cpp` — C++ kernels (OpenMP)
- `kuramoto_cpp_kernel.hpp` — header

**CUDA kernels**
- `kuramoto_cuda_kernel.cu` — GPU kernels (dense, sparse, mean-field)
- `kuramoto_cuda_kernel.h` — header
- `kuramoto_cuda_kernel.hpp` — additional C++ header
- `kuramoto_cuda_wrapper.cu` — CUDA wrapper (alloc, copy, launch, retrieve)

**Cython wrappers**
- `c_backend_wrapper.pyx` — wrapper around the C kernel
- `cpp_backend_wrapper.pyx` — wrapper around the C++ kernel
- `cython_backend.pyx` — pure Cython implementation

### Generated files (by `setup.py`)

**From Cython (`.pyx` → `.c` / `.cpp`)**
- `c_backend_wrapper.cpp`
- `cpp_backend_wrapper.cpp`
- `cython_backend.c`

**From CUDA (`.cu` → `.o`)**
- `kuramoto_cuda_kernel.o`
- `kuramoto_cuda_wrapper.o`
- `dlink.o` (device link)

**Compiled Python modules (`.so`)**
- `c_backend.cpython-XXX.so`
- `cpp_backend.cpython-XXX.so`
- `cython_backend.cpython-XXX.so`

### Build flow