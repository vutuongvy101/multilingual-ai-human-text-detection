"""Common training utilities and metrics."""

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


def compute_metrics(eval_pred):
    """Compute metrics for evaluation.

    Args:
        eval_pred: Tuple of (predictions, labels) from transformers trainer

    Returns:
        Dictionary of metrics
    """
    predictions, labels = eval_pred
    predictions = np.argmax(predictions, axis=1)

    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
        "precision": precision_score(labels, predictions, average="weighted"),
        "recall": recall_score(labels, predictions, average="weighted"),
    }


def tokenize_function(tokenizer, max_length: int = 512):
    """Create a tokenize function for transformers.

    Args:
        tokenizer: HuggingFace tokenizer
        max_length: Maximum sequence length

    Returns:
        Tokenization function
    """
    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            padding=False,  # Will be handled by data collator
            max_length=max_length,
        )
    return tokenize