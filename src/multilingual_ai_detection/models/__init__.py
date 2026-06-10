"""Model definitions and loading utilities."""

from pathlib import Path

from .statistical import StatisticalClassifier


def load_model(
    model_path: str,
    model_type: str,
):
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
        from .transformer import TransformerClassifier

        return TransformerClassifier.load_model(model_path)

    raise ValueError(f"Unknown model type: {model_type}. Use 'statistical' or 'transformer'.")

def __getattr__(name: str):
    """Lazily expose transformer classes to avoid importing transformers unnecessarily."""
    if name == "TransformerClassifier":
        from .transformer import TransformerClassifier

        globals()[name] = TransformerClassifier
        return TransformerClassifier
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["StatisticalClassifier", "TransformerClassifier", "load_model"]
