"""Model definitions and loading utilities."""

from pathlib import Path
from typing import Union

from .statistical import StatisticalClassifier
from .transformer import TransformerClassifier


def load_model(
    model_path: str,
    model_type: str,
) -> Union[StatisticalClassifier, TransformerClassifier]:
    """Load a trained model from disk.

    Args:
        model_path: Path to model directory or file
        model_type: 'statistical' or 'transformer'

    Returns:
        Loaded model instance
    """
    path = Path(model_path)

    if model_type == "statistical":
        if path.is_dir():
            joblib_files = list(path.glob("*_model.joblib"))
            if not joblib_files:
                joblib_files = list(path.glob("*.joblib"))
            if not joblib_files:
                raise FileNotFoundError(f"No joblib model found in {model_path}")
            path = joblib_files[0]
        return StatisticalClassifier.load_model(str(path))

    if model_type == "transformer":
        return TransformerClassifier.load_model(model_path)

    raise ValueError(f"Unknown model type: {model_type}. Use 'statistical' or 'transformer'.")


__all__ = ["StatisticalClassifier", "TransformerClassifier", "load_model"]
