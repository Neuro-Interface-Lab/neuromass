# neuromass

`neuromass` is an in-progress Python package for neural mass, mean-field, and
related computational neuroscience models.

## Project layout

```text
neuromass/
├── docs/
├── examples/
├── src/
│   └── neuromass/
│       └── models/
│           └── kuramoto/
│               └── _native/
├── tests/
├── environment.yml
└── pyproject.toml
```

The public API lives in `src/neuromass/`, while model-specific compiled sources
can be grouped inside each model family under `_native/`.
