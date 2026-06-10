"""Multilingual AI-Human Text Detection Package.

This package provides tools for detecting AI-generated vs human-written text
across multiple languages using both statistical and transformer-based models.
"""

from importlib import import_module

__version__ = "0.1.0"
__author__ = "Tuong Vy Vu"
__email__ = "35751592+vutuongvy@users.noreply.github.com"

__all__ = ["data", "models", "training", "evaluation", "utils"]


def __getattr__(name: str):
    """Lazily import subpackages to avoid heavy optional dependencies at import time."""
    if name in __all__:
        module = import_module(f".{name}", __name__)
        globals()[name] = module
        return module
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")