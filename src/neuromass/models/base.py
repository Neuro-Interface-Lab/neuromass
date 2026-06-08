"""Base abstractions for neuromass models."""

from dataclasses import dataclass


@dataclass(slots=True)
class BaseModel:
    """Base class for model definitions exposed through the Python API."""

    name: str

