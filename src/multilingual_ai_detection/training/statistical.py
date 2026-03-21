"""Statistical model training utilities."""

import os
from pathlib import Path
from typing import Optional

from ..models import StatisticalClassifier
from .common import compute_metrics


def train_statistical_model(
    train_texts: list,
    train_labels: list,
    val_texts: Optional[list] = None,
    val_labels: Optional[list] = None,
    classifier: str = "logistic",
    max_features: int = 10000,
    output_dir: str = "models/statistical",
    **kwargs
) -> StatisticalClassifier:
    """Train a statistical classifier.

    Args:
        train_texts: Training text data
        train_labels: Training labels
        val_texts: Validation text data (optional)
        val_labels: Validation labels (optional)
        classifier: Type of classifier ('logistic' or 'naive_bayes')
        max_features: Maximum number of TF-IDF features
        output_dir: Directory to save the model
        **kwargs: Additional arguments for StatisticalClassifier

    Returns:
        Trained StatisticalClassifier
    """
    # Initialize model
    model = StatisticalClassifier(
        classifier=classifier,
        max_features=max_features,
        **kwargs
    )

    # Fit the model
    model.fit(train_texts, train_labels)

    # Evaluate on validation set if provided
    if val_texts is not None and val_labels is not None:
        val_predictions = model.predict(val_texts)
        val_metrics = compute_metrics((model.predict_proba(val_texts), val_labels))
        print("Validation Metrics:")
        for metric, value in val_metrics.items():
            print(f"  {metric}: {value:.4f}")

    # Save the model
    os.makedirs(output_dir, exist_ok=True)
    model_path = Path(output_dir) / f"{classifier}_model.joblib"
    model.save_model(model, str(model_path))

    return model