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

