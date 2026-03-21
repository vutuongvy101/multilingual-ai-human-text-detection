"""Multilingual AI-Human Text Detection Package.

This package provides tools for detecting AI-generated vs human-written text
across multiple languages using both statistical and transformer-based models.
"""

__version__ = "0.1.0"
__author__ = "Tuong Vy Vu"
__email__ = "your.email@example.com"

from . import data
from . import models
from . import training
from . import evaluation
from . import utils

__all__ = ["data", "models", "training", "evaluation", "utils"]