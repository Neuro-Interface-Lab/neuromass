Developer's Corner
==================

This page gathers practical information for contributors working on
``neuromass`` internals.

The overall spirit is inspired by the `NRV developer documentation <https://nrv.readthedocs.io/en/latest/>`_
and the `NRV repository structure <https://github.com/nrv-framework/NRV>`_:
keep a clear separation between user-facing Python APIs, scientific model
implementations, compiled kernels, examples, tests, and documentation.

Development Philosophy
----------------------

``neuromass`` is meant to support two complementary layers:

- a high-level Python interface for end users;
- low-level numerical kernels that may be written in Python, Cython, C, or C++.

This separation is especially important for scientific development because we
want to:

- prototype models quickly in pure Python;
- maintain several interchangeable implementations of the same model;
- benchmark numerical accuracy and execution speed;
- progressively replace naive kernels with optimized backends.

Project Layout
--------------

The repository currently follows a ``src/`` layout:

.. code-block:: text

   neuromass/
   ├── docs/
   ├── examples/
   ├── src/
   │   └── neuromass/
   │       ├── models/
   │       │   └── kuramoto/
   │       │       └── _native/
   │       └── utils/
   ├── tests/
   ├── environment.yml
   ├── pyproject.toml
   └── setup.py

The intended responsibilities are:

- ``src/neuromass/``: public package code;
- ``src/neuromass/models/``: scientific models and their Python APIs;
- ``src/neuromass/models/<family>/_native/``: compiled sources specific to one model family;
- ``src/neuromass/utils/``: reusable generators, helpers, and data utilities;
- ``examples/``: runnable demonstrations, sanity checks, and benchmarks;
- ``tests/``: unit tests and regression tests;
- ``docs/``: Sphinx documentation sources.

Compiled Backends
-----------------

The current Kuramoto prototype exposes one Python API and several backends:

- ``python``
- ``cython``
- ``c``
- ``cpp``

This pattern should be preserved for future models whenever possible:

1. define a stable high-level model class in Python;
2. expose a reference implementation for validation;
3. add compiled kernels behind the same interface;
4. compare all implementations in examples and tests.

When a compiled source is modified, the editable package should be rebuilt:

.. code-block:: bash

   pip install -e . --no-build-isolation

Typical cases requiring a rebuild include:

- editing ``.pyx`` files;
- editing ``.c`` or ``.cpp`` kernels;
- editing C/C++ headers;
- changing extension declarations in ``setup.py`` or packaging metadata.

Working Environment
-------------------

The repository includes an ``environment.yml`` file intended for Conda or
Mamba-based environments.

A typical local workflow is:

.. code-block:: bash

   mamba env create -f environment.yml
   mamba activate neuromass
   pip install -e . --no-build-isolation

For pure Python edits, an editable installation is usually enough and no rebuild
is required.

Documentation Workflow
----------------------

Documentation is built with Sphinx and the Furo theme.

To work on the documentation locally:

.. code-block:: bash

   pip install -e .[docs]
   sphinx-build -b html docs docs/_build

When adding a new model or utility:

- document the user-facing API;
- add at least one runnable example when possible;
- keep docstrings in NumPy style for scientific code.

Testing Strategy
----------------

At this stage, tests should focus on:

- import and packaging sanity;
- shape and type consistency;
- agreement between reference and compiled implementations;
- reproducible helper utilities.

As the project grows, additional test categories should be introduced:

- numerical regression tests;
- performance-oriented smoke tests;
- scientific validation scripts for known analytical limits.

Scientific Sanity Checks
------------------------

Examples are not only tutorials. They also serve as lightweight scientific
checks.

For instance, the current Kuramoto examples cover:

- backend-to-backend consistency;
- trajectory visualization;
- order parameter evolution;
- comparison between finite-size simulations and the Lorentzian critical
  coupling prediction.

Contributor Guidelines
----------------------

When adding new functionality, prefer the following sequence:

1. start with the simplest correct implementation;
2. add a clear example;
3. add a minimal test;
4. optimize only after the reference behavior is verified.

For new model families, the recommended pattern is:

.. code-block:: text

   src/neuromass/models/<family>/
   ├── __init__.py
   ├── model.py
   └── _native/

This keeps the Python interface close to the model-specific compiled kernels and
helps maintain clean scientific boundaries between implementations.

