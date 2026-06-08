"""High-level Python interface for Kuramoto models."""

from dataclasses import dataclass

from ..base import BaseModel


@dataclass(slots=True)
class NaiveKuramotoModel(BaseModel):
    """Initial pure-Python placeholder for a Kuramoto model."""

    name: str = "kuramoto-naive"
    coupling: float = 1.0
