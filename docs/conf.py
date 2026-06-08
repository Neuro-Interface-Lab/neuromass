import os
import sys

sys.path.insert(0, os.path.abspath("../src"))

project = "neuromass"
author = "neuromass contributors"
release = "0.1.0"
version = release
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
]
templates_path = ["_templates"]
exclude_patterns = ["_build"]
html_theme = "furo"
autosummary_generate = True
