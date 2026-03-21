"""Model evaluation functions."""

from typing import Dict, List, Optional, Union

import pandas as pd

from ..models import StatisticalClassifier, TransformerClassifier
from .metrics import (
    compute_classification_metrics,
    compute_per_class_metrics,
    compute_per_language_metrics,
    generate_confusion_matrix,
)


def evaluate_model(
    model: Union[StatisticalClassifier, TransformerClassifier],
    test_data: Union[list, pd.DataFrame],
    test_labels: Optional[list] = None,
    class_names: Optional[List[str]] = None,
    output_dir: Optional[str] = None
) -> Dict:
    """Comprehensive model evaluation.

    Args:
        model: Trained model
        test_data: Test data (texts for statistical, DataFrame for transformer)
        test_labels: Test labels (if not in test_data)
        class_names: Names for classes
        output_dir: Directory to save evaluation results

    Returns:
        Dictionary with evaluation results
    """
    if class_names is None:
        class_names = ["Human", "AI"]

    # Get predictions
    if isinstance(model, StatisticalClassifier):
        if not isinstance(test_data, list):
            raise ValueError("For statistical models, test_data must be a list of texts")
        predictions = model.predict(test_data)
        probabilities = model.predict_proba(test_data)
    elif isinstance(model, TransformerClassifier):
        if isinstance(test_data, pd.DataFrame):
            texts = test_data["text"].tolist()
            test_labels = test_data["label"].tolist()
        else:
            texts = test_data
        predictions = model.predict(texts)
        probabilities = model.predict_proba(texts)
    else:
        raise ValueError(f"Unknown model type: {type(model)}")

    # Compute overall metrics
    overall_metrics = compute_classification_metrics(
        test_labels, predictions, probabilities
    )

    # Per-class metrics
    per_class_df = compute_per_class_metrics(
        test_labels, predictions, class_names
    )

    # Confusion matrix
    cm_df = generate_confusion_matrix(
        test_labels, predictions, class_names
    )

    # Per-language metrics if language info is available
    per_lang_df = None
    if isinstance(test_data, pd.DataFrame) and "lang" in test_data.columns:
        per_lang_df = compute_per_language_metrics(
            test_data, predictions, probabilities
        )

    results = {
        "overall_metrics": overall_metrics,
        "per_class_metrics": per_class_df,
        "confusion_matrix": cm_df,
        "per_language_metrics": per_lang_df,
        "predictions": predictions,
        "probabilities": probabilities,
    }

    # Save results if output directory is provided
    if output_dir:
        import os
        os.makedirs(output_dir, exist_ok=True)

        # Save metrics
        with open(f"{output_dir}/metrics.txt", "w") as f:
            f.write("Overall Metrics:\n")
            for k, v in overall_metrics.items():
                f.write(f"{k}: {v:.4f}\n")

        # Save detailed results
        per_class_df.to_csv(f"{output_dir}/per_class_metrics.csv", index=False)
        cm_df.to_csv(f"{output_dir}/confusion_matrix.csv")

        if per_lang_df is not None:
            per_lang_df.to_csv(f"{output_dir}/per_language_metrics.csv", index=False)

        # Save predictions
        pred_df = pd.DataFrame({
            "true_label": test_labels,
            "predicted_label": predictions,
        })
        if probabilities:
            prob_df = pd.DataFrame(probabilities, columns=[f"prob_{i}" for i in range(len(probabilities[0]))])
            pred_df = pd.concat([pred_df, prob_df], axis=1)

        pred_df.to_csv(f"{output_dir}/predictions.csv", index=False)

    return results


def print_evaluation_report(results: Dict):
    """Print a formatted evaluation report.

    Args:
        results: Results from evaluate_model
    """
    print("=" * 50)
    print("MODEL EVALUATION REPORT")
    print("=" * 50)

    print("\nOVERALL METRICS:")
    for metric, value in results["overall_metrics"].items():
        print(f"  {metric.capitalize()}: {value:.4f}")

    print("\nPER-CLASS METRICS:")
    print(results["per_class_metrics"].to_string(index=False))

    print("\nCONFUSION MATRIX:")
    print(results["confusion_matrix"])

    if results["per_language_metrics"] is not None:
        print("\nPER-LANGUAGE METRICS:")
        print(results["per_language_metrics"].to_string(index=False))

    print("=" * 50)